"""Tests for the greedy capacity budget."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from eo.planner.decision_matrix import LoadDecision
from eo.planner.iterative import Plan, PlanCell
from eo.policy.capacity_budget import apply_capacity_budget


def _cell(load, slot, action="on", utility=10, rule_id=2) -> PlanCell:
    return PlanCell(
        load=load, slot_index=slot,
        timestamp=datetime(2026, 5, 20, 12, tzinfo=timezone.utc),
        decision=LoadDecision(
            action=action, reason="test",
            rule_id=rule_id, utility_score=utility,
        ),
    )


# ── Basic ───────────────────────────────────────────────────────────────────
class TestBasic:
    def test_no_overrides_when_under_budget(self):
        plan = Plan(cells=(
            _cell("a", 0, utility=10),
            _cell("b", 0, utility=5),
        ))
        result = apply_capacity_budget(
            plan, load_watts={"a": 1000, "b": 500}, available_w=2000,
        )
        assert result.overrides == ()
        # All "on" cells preserved.
        on_in_result = [c for c in result.adjusted_plan.cells if c.decision.action == "on"]
        assert len(on_in_result) == 2

    def test_empty_plan_returns_empty(self):
        result = apply_capacity_budget(
            Plan(cells=()), load_watts={}, available_w=1000,
        )
        assert result.adjusted_plan.cells == ()
        assert result.overrides == ()

    def test_no_on_cells_no_overrides(self):
        plan = Plan(cells=(_cell("a", 0, action="off"),))
        result = apply_capacity_budget(
            plan, load_watts={"a": 1000}, available_w=500,
        )
        assert result.overrides == ()


# ── Overload behaviour ─────────────────────────────────────────────────────
class TestOverload:
    def test_lowest_utility_dropped_first(self):
        # Budget 2000 W. Three loads with watts (1500, 1500, 1500) and utility
        # 30, 20, 10. Sum = 4500. Keeps loads with utility 30 and drops the
        # rest (one fits at 1500 → 1500; second drops because total would be 3000).
        # Sort: 30, 20, 10. Allocate: 30 → 1500, 20 → 1500 would be 3000 → drop, 10 → drop.
        plan = Plan(cells=(
            _cell("a", 0, utility=30),
            _cell("b", 0, utility=20),
            _cell("c", 0, utility=10),
        ))
        result = apply_capacity_budget(
            plan, load_watts={"a": 1500, "b": 1500, "c": 1500},
            available_w=2000,
        )
        kept = {c.load for c in result.adjusted_plan.cells if c.decision.action == "on"}
        dropped = {c.load for c in result.adjusted_plan.cells if c.decision.action == "off"}
        assert kept == {"a"}
        assert dropped == {"b", "c"}
        assert {o.load for o in result.overrides} == {"b", "c"}

    def test_tie_breaks_by_higher_load_watts(self):
        # Two loads with same utility; larger one filled first.
        plan = Plan(cells=(
            _cell("small", 0, utility=10),
            _cell("big", 0, utility=10),
        ))
        result = apply_capacity_budget(
            plan, load_watts={"small": 500, "big": 1500},
            available_w=1500,
        )
        kept = {c.load for c in result.adjusted_plan.cells if c.decision.action == "on"}
        # big fills first; small fits in zero remaining → dropped.
        assert kept == {"big"}

    def test_each_slot_budgeted_independently(self):
        plan = Plan(cells=(
            _cell("a", 0, utility=30),
            _cell("b", 0, utility=10),
            _cell("a", 1, utility=30),
            _cell("b", 1, utility=10),
        ))
        # Budget 1000 W per slot; each load is 800 W → only one fits per slot.
        result = apply_capacity_budget(
            plan, load_watts={"a": 800, "b": 800}, available_w=1000,
        )
        slot0_on = [c for c in result.adjusted_plan.cells
                    if c.slot_index == 0 and c.decision.action == "on"]
        slot1_on = [c for c in result.adjusted_plan.cells
                    if c.slot_index == 1 and c.decision.action == "on"]
        assert {c.load for c in slot0_on} == {"a"}
        assert {c.load for c in slot1_on} == {"a"}


# ── "Sábado de Gloria" — Gemini R1 Test 2 ─────────────────────────────────
class TestSabadoDeGloria:
    """Pool (1 kW) + Termo (2 kW) compete; useful inverter budget 2.5 kW."""

    def test_pool_and_termo_collision_serialised(self):
        # Both ON in slot 0 with high utility — but sum = 3000 W > 2500.
        # Termo (higher watts) is preferred under tie-break.
        plan = Plan(cells=(
            _cell("pool", 0, utility=50, rule_id=7),
            _cell("termo", 0, utility=50, rule_id=7),
        ))
        result = apply_capacity_budget(
            plan, load_watts={"pool": 1000, "termo": 2000},
            available_w=2500,
        )
        kept_in_slot0 = {c.load for c in result.adjusted_plan.cells
                        if c.slot_index == 0 and c.decision.action == "on"}
        # Tie on utility → larger watts wins.
        assert "termo" in kept_in_slot0
        assert "pool" not in kept_in_slot0


# ── Validation ─────────────────────────────────────────────────────────────
class TestValidation:
    def test_negative_budget_raises(self):
        with pytest.raises(ValueError):
            apply_capacity_budget(Plan(cells=()), {}, available_w=-100)

    def test_missing_load_watts_treated_as_zero(self):
        plan = Plan(cells=(_cell("unknown", 0, utility=10),))
        result = apply_capacity_budget(plan, load_watts={}, available_w=0)
        # 0 watts fits in 0 budget → kept.
        kept = {c.load for c in result.adjusted_plan.cells if c.decision.action == "on"}
        assert kept == {"unknown"}
