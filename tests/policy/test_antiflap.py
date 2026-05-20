"""Tests for the antiflap Schmitt trigger."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eo.planner.decision_matrix import LoadDecision
from eo.planner.iterative import Plan, PlanCell
from eo.policy.antiflap import (
    AntiflapConfig,
    AntiflapState,
    apply_antiflap,
)


def _utc(min_offset: int = 0) -> datetime:
    return datetime(2026, 5, 20, 12, tzinfo=timezone.utc) + timedelta(minutes=min_offset)


def _cell(load, slot, action) -> PlanCell:
    return PlanCell(
        load=load, slot_index=slot, timestamp=_utc(),
        decision=LoadDecision(action=action, reason="planner",
                              rule_id=2, utility_score=10),
    )


# ── Validation ─────────────────────────────────────────────────────────────
class TestValidation:
    def test_naive_now_raises(self):
        plan = Plan(cells=(_cell("a", 0, "on"),))
        with pytest.raises(ValueError):
            apply_antiflap(plan, {}, now=datetime(2026, 5, 20, 12))

    def test_negative_hold_rejected(self):
        with pytest.raises(ValueError):
            AntiflapConfig(decision_hold_minutes=-1)


# ── Behaviour ──────────────────────────────────────────────────────────────
class TestBehaviour:
    def test_first_run_no_prior_state_passes_through(self):
        plan = Plan(cells=(_cell("a", 0, "on"),))
        result = apply_antiflap(plan, state_by_load={}, now=_utc(0))
        assert result.adjusted_plan.cells[0].decision.action == "on"
        assert result.overrides == ()

    def test_same_action_as_prior_passes_through(self):
        plan = Plan(cells=(_cell("a", 0, "on"),))
        state = {"a": AntiflapState(load="a", last_action="on",
                                    last_change_ts=_utc(-5))}
        result = apply_antiflap(plan, state_by_load=state, now=_utc(0))
        assert result.overrides == ()

    def test_change_within_hold_blocked(self):
        plan = Plan(cells=(_cell("boiler", 0, "off"),))  # planner wants OFF
        state = {"boiler": AntiflapState(
            load="boiler", last_action="on",
            last_change_ts=_utc(-10),  # 10 min ago
        )}
        config = AntiflapConfig(decision_hold_minutes=30)
        result = apply_antiflap(plan, state, now=_utc(0), config=config)
        # Hold active → revert to "on"
        cell = result.adjusted_plan.cells[0]
        assert cell.decision.action == "on"
        assert "antiflap" in cell.decision.reason
        assert len(result.overrides) == 1

    def test_change_after_hold_allowed(self):
        plan = Plan(cells=(_cell("boiler", 0, "off"),))
        state = {"boiler": AntiflapState(
            load="boiler", last_action="on",
            last_change_ts=_utc(-45),  # 45 min ago
        )}
        config = AntiflapConfig(decision_hold_minutes=30)
        result = apply_antiflap(plan, state, now=_utc(0), config=config)
        assert result.adjusted_plan.cells[0].decision.action == "off"
        assert result.overrides == ()

    def test_only_slot_0_affected(self):
        plan = Plan(cells=(
            _cell("a", 0, "off"),
            _cell("a", 1, "off"),
            _cell("a", 2, "off"),
        ))
        state = {"a": AntiflapState(load="a", last_action="on",
                                    last_change_ts=_utc(-5))}
        config = AntiflapConfig(decision_hold_minutes=30)
        result = apply_antiflap(plan, state, now=_utc(0), config=config)
        # Slot 0 reverted to "on"; slots 1 and 2 untouched.
        actions_by_slot = {c.slot_index: c.decision.action
                           for c in result.adjusted_plan.cells}
        assert actions_by_slot[0] == "on"
        assert actions_by_slot[1] == "off"
        assert actions_by_slot[2] == "off"

    def test_naive_last_change_ts_normalised(self):
        # If the persisted state slipped a naive datetime past us, do not crash.
        plan = Plan(cells=(_cell("a", 0, "off"),))
        state = {"a": AntiflapState(
            load="a", last_action="on",
            last_change_ts=datetime(2026, 5, 20, 11, 50),  # naive
        )}
        config = AntiflapConfig(decision_hold_minutes=30)
        result = apply_antiflap(plan, state, now=_utc(0), config=config)
        # 10 min ago by naive → still under hold → reverted.
        assert result.adjusted_plan.cells[0].decision.action == "on"
