"""Tests for the policy pipeline orchestrator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from eo.planner.decision_matrix import LoadDecision
from eo.planner.iterative import Plan, PlanCell
from eo.policy.antiflap import AntiflapState
from eo.policy.degraded_mode import (
    DegradedLevel,
    DegradedModeInputs,
)
from eo.policy.pipeline import (
    PolicyPipelineInputs,
    run_policy_pipeline,
)


def _utc(min_offset: int = 0) -> datetime:
    return datetime(2026, 5, 20, 12, tzinfo=timezone.utc) + timedelta(minutes=min_offset)


def _cell(load, slot, action="on", rule_id=7, utility=10) -> PlanCell:
    return PlanCell(
        load=load, slot_index=slot, timestamp=_utc(),
        decision=LoadDecision(action=action, reason="planner",
                              rule_id=rule_id, utility_score=utility),
    )


# ── End-to-end happy path ──────────────────────────────────────────────────
class TestHappyPath:
    def test_pipeline_passes_clean_plan_unchanged(self):
        plan = Plan(cells=(_cell("boiler", 0),))
        inputs = PolicyPipelineInputs(
            load_watts={"boiler": 1000},
            available_w=5000,
            slot_periods={0: "valley"},
            antiflap_state={},
            now=_utc(0),
            degraded_inputs=DegradedModeInputs(),
        )
        summary = run_policy_pipeline(plan, inputs)
        assert summary.degraded_level == DegradedLevel.NORMAL
        assert summary.overrides == ()
        assert summary.final_plan.cells[0].decision.action == "on"


# ── Layers compose ─────────────────────────────────────────────────────────
class TestComposition:
    def test_capacity_then_peak_then_antiflap_then_degraded(self):
        plan = Plan(cells=(
            _cell("a", 0, utility=30, rule_id=7),
            _cell("b", 0, utility=10, rule_id=7),
            _cell("c", 0, utility=20, rule_id=8),  # min_runtime_only would be set by matrix; here rule_id 8 alone is enough to track in degraded L1 below if we set it explicitly
        ))

        inputs = PolicyPipelineInputs(
            load_watts={"a": 1500, "b": 1500, "c": 1500},
            available_w=2000,            # only one load fits
            slot_periods={0: "valley"},
            antiflap_state={},
            now=_utc(0),
            degraded_inputs=DegradedModeInputs(),
        )

        summary = run_policy_pipeline(plan, inputs)
        # 'a' has highest utility → kept. b, c → off due to capacity_budget.
        actions = {c.load: c.decision.action for c in summary.final_plan.cells}
        assert actions["a"] == "on"
        assert actions["b"] == "off"
        assert actions["c"] == "off"
        # At least 2 overrides from capacity_budget.
        cap_overrides = [o for o in summary.overrides if o.layer == "capacity_budget"]
        assert len(cap_overrides) >= 2

    def test_degraded_l3_overrides_everything(self):
        plan = Plan(cells=(
            _cell("a", 0, utility=50, rule_id=3),
            _cell("b", 0, utility=40, rule_id=7),
        ))

        inputs = PolicyPipelineInputs(
            load_watts={"a": 500, "b": 500},
            available_w=5000,                # ample
            slot_periods={0: "valley"},
            antiflap_state={},
            now=_utc(0),
            degraded_inputs=DegradedModeInputs(sensor_age_max_minutes=120),
        )
        summary = run_policy_pipeline(plan, inputs)
        assert summary.degraded_level == DegradedLevel.L3_SENSORS_STALE
        for cell in summary.final_plan.cells:
            assert cell.decision.action == "off"

    def test_peak_prohibition_after_capacity_budget(self):
        # Capacity budget passes both, but peak ban kicks them out.
        plan = Plan(cells=(_cell("a", 0, utility=50, rule_id=7),))
        inputs = PolicyPipelineInputs(
            load_watts={"a": 500},
            available_w=5000,
            slot_periods={0: "peak"},
            antiflap_state={},
            now=_utc(0),
            degraded_inputs=DegradedModeInputs(),
        )
        summary = run_policy_pipeline(plan, inputs)
        assert summary.final_plan.cells[0].decision.action == "off"
        peak_overrides = [o for o in summary.overrides
                          if o.layer == "peak_prohibition"]
        assert len(peak_overrides) == 1

    def test_antiflap_blocks_recent_change(self):
        plan = Plan(cells=(_cell("a", 0, action="off"),))  # planner wants OFF
        state = {"a": AntiflapState(load="a", last_action="on",
                                    last_change_ts=_utc(-10))}
        inputs = PolicyPipelineInputs(
            load_watts={"a": 500},
            available_w=5000,
            slot_periods={0: "valley"},
            antiflap_state=state,
            now=_utc(0),
            degraded_inputs=DegradedModeInputs(),
        )
        summary = run_policy_pipeline(plan, inputs)
        assert summary.final_plan.cells[0].decision.action == "on"
        flap_overrides = [o for o in summary.overrides if o.layer == "antiflap"]
        assert len(flap_overrides) == 1


# ── Summary serialisation ──────────────────────────────────────────────────
class TestSummary:
    def test_to_dict_serialisable(self):
        import json
        plan = Plan(cells=(_cell("a", 0),))
        summary = run_policy_pipeline(
            plan,
            PolicyPipelineInputs(
                load_watts={"a": 500},
                available_w=5000,
                slot_periods={0: "valley"},
                antiflap_state={},
                now=_utc(0),
                degraded_inputs=DegradedModeInputs(),
            ),
        )
        json.dumps(summary.to_dict())
