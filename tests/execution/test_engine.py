"""Tests for the execution engine — plan builder + dispatch."""

from __future__ import annotations

from datetime import datetime, timezone

from eo.execution.engine import build_execution_plan, execute_plan
from eo.planner.decision_matrix import LoadDecision
from eo.planner.iterative import Plan, PlanCell


def _utc() -> datetime:
    return datetime(2026, 5, 20, 12, tzinfo=timezone.utc)


def _cell(load, slot, action) -> PlanCell:
    return PlanCell(
        load=load, slot_index=slot, timestamp=_utc(),
        decision=LoadDecision(action=action, reason="t",
                              rule_id=7, utility_score=10),
    )


# ── build_execution_plan ───────────────────────────────────────────────────
class TestBuildPlan:
    def test_emits_turn_on_when_off_in_world(self):
        plan = Plan(cells=(_cell("boiler", 0, "on"),))
        ep = build_execution_plan(
            plan,
            load_entities={"boiler": ("switch.boiler", "switch")},
            loads_currently_on=set(),
            cycle_ts=_utc(),
        )
        assert len(ep.commands) == 1
        assert ep.commands[0].service == "turn_on"
        assert ep.commands[0].entity_id == "switch.boiler"

    def test_emits_turn_off_when_on_in_world(self):
        plan = Plan(cells=(_cell("boiler", 0, "off"),))
        ep = build_execution_plan(
            plan,
            load_entities={"boiler": ("switch.boiler", "switch")},
            loads_currently_on={"boiler"},
            cycle_ts=_utc(),
        )
        assert ep.commands[0].service == "turn_off"

    def test_no_command_when_already_in_desired_state(self):
        plan = Plan(cells=(
            _cell("a", 0, "on"),
            _cell("b", 0, "off"),
        ))
        ep = build_execution_plan(
            plan,
            load_entities={
                "a": ("switch.a", "switch"),
                "b": ("switch.b", "switch"),
            },
            loads_currently_on={"a"},
            cycle_ts=_utc(),
        )
        assert ep.commands == ()

    def test_only_slot_0_considered(self):
        plan = Plan(cells=(
            _cell("a", 0, "on"),
            _cell("a", 1, "off"),  # later slot, should be ignored
        ))
        ep = build_execution_plan(
            plan,
            load_entities={"a": ("switch.a", "switch")},
            loads_currently_on=set(),
            cycle_ts=_utc(),
        )
        assert len(ep.commands) == 1
        assert ep.commands[0].target_action == "on"

    def test_unbound_loads_silently_skipped(self):
        plan = Plan(cells=(_cell("ghost", 0, "on"),))
        ep = build_execution_plan(
            plan, load_entities={}, loads_currently_on=set(), cycle_ts=_utc(),
        )
        assert ep.commands == ()


# ── execute_plan ───────────────────────────────────────────────────────────
class TestExecute:
    def test_acks_recorded_per_command(self):
        from eo.execution.types import CommandRequest, ExecutionPlan

        plan = ExecutionPlan(cycle_ts=_utc(), commands=(
            CommandRequest("a", "on", "switch", "turn_on", "switch.a"),
            CommandRequest("b", "off", "switch", "turn_off", "switch.b"),
        ))

        calls = []
        def send(domain, service, data):
            calls.append((domain, service, data))
            return True

        result = execute_plan(plan, send)
        assert result.all_acknowledged
        assert len(calls) == 2
        assert len(result.results) == 2

    def test_failed_callback_recorded_as_not_acknowledged(self):
        from eo.execution.types import CommandRequest, ExecutionPlan

        plan = ExecutionPlan(cycle_ts=_utc(), commands=(
            CommandRequest("a", "on", "switch", "turn_on", "switch.a"),
        ))

        def send(*_a, **_kw):
            return False

        result = execute_plan(plan, send)
        assert not result.all_acknowledged
        assert result.results[0].acknowledged is False

    def test_exceptions_in_callback_caught(self):
        from eo.execution.types import CommandRequest, ExecutionPlan

        plan = ExecutionPlan(cycle_ts=_utc(), commands=(
            CommandRequest("a", "on", "switch", "turn_on", "switch.a"),
        ))

        def send(*_a, **_kw):
            raise RuntimeError("HA unreachable")

        result = execute_plan(plan, send)
        assert result.results[0].acknowledged is False
        assert result.results[0].error is not None
        assert "RuntimeError" in result.results[0].error

    def test_loads_now_on_lists_acked_ons(self):
        from eo.execution.types import CommandRequest, ExecutionPlan

        plan = ExecutionPlan(cycle_ts=_utc(), commands=(
            CommandRequest("a", "on", "switch", "turn_on", "switch.a"),
            CommandRequest("b", "on", "switch", "turn_on", "switch.b"),
            CommandRequest("c", "off", "switch", "turn_off", "switch.c"),
        ))

        # b's ACK fails.
        responses = iter([True, False, True])
        def send(*_a, **_kw):
            return next(responses)

        result = execute_plan(plan, send)
        assert result.loads_now_on == ("a",)
