"""Tests for the peak prohibition layer."""

from __future__ import annotations

from datetime import datetime, timezone

from eo.planner.decision_matrix import LoadDecision
from eo.planner.iterative import Plan, PlanCell
from eo.policy.peak_prohibition import apply_peak_prohibition


def _cell(load, slot, action="on", rule_id=7) -> PlanCell:
    return PlanCell(
        load=load, slot_index=slot,
        timestamp=datetime(2026, 5, 20, 12, tzinfo=timezone.utc),
        decision=LoadDecision(
            action=action, reason="test",
            rule_id=rule_id, utility_score=10,
        ),
    )


def test_peak_on_load_forced_off():
    plan = Plan(cells=(_cell("boiler", 0, action="on"),))
    result = apply_peak_prohibition(plan, slot_periods={0: "peak"})
    cell = result.adjusted_plan.cells[0]
    assert cell.decision.action == "off"
    assert len(result.overrides) == 1
    assert "peak" in result.overrides[0].reason.lower()


def test_off_loads_unchanged():
    plan = Plan(cells=(_cell("boiler", 0, action="off"),))
    result = apply_peak_prohibition(plan, slot_periods={0: "peak"})
    assert result.overrides == ()


def test_valley_and_mid_untouched():
    plan = Plan(cells=(
        _cell("a", 0, action="on"),
        _cell("b", 1, action="on"),
    ))
    result = apply_peak_prohibition(plan, slot_periods={0: "valley", 1: "mid"})
    assert result.overrides == ()


def test_rule_4_exception_passes_through():
    """allow_peak_on_critical → matrix emits rule_id=4 with action=on.
    Defence-in-depth must respect that exception."""
    plan = Plan(cells=(_cell("emergency", 0, action="on", rule_id=4),))
    result = apply_peak_prohibition(plan, slot_periods={0: "peak"})
    assert result.adjusted_plan.cells[0].decision.action == "on"
    assert result.overrides == ()


def test_unknown_slot_period_treated_as_non_peak():
    plan = Plan(cells=(_cell("a", 5, action="on"),))
    result = apply_peak_prohibition(plan, slot_periods={})
    # No slot_periods[5] → not peak → no override.
    assert result.overrides == ()
