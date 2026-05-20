"""Tests for the three-level degraded mode policy."""

from __future__ import annotations

from datetime import datetime, timezone

from eo.planner.decision_matrix import LoadDecision
from eo.planner.iterative import Plan, PlanCell
from eo.policy.degraded_mode import (
    DegradedLevel,
    DegradedModeConfig,
    DegradedModeInputs,
    apply_degraded_mode,
    classify_degraded_level,
)


def _cell(load, slot, action="on", rule_id=7, min_runtime_only=False) -> PlanCell:
    return PlanCell(
        load=load, slot_index=slot,
        timestamp=datetime(2026, 5, 20, 12, tzinfo=timezone.utc),
        decision=LoadDecision(
            action=action, reason="test",
            rule_id=rule_id, utility_score=10,
            min_runtime_only=min_runtime_only,
        ),
    )


# ── Classification ──────────────────────────────────────────────────────────
class TestClassification:
    def test_all_normal_returns_normal(self):
        assert classify_degraded_level(DegradedModeInputs()) == DegradedLevel.NORMAL

    def test_sensor_age_dominant(self):
        # Even with bad forecast and stale AEMET, sensor stale wins.
        inputs = DegradedModeInputs(
            forecast_mae=5.0, aemet_age_hours=48,
            sensor_age_max_minutes=60,
        )
        assert classify_degraded_level(inputs) == DegradedLevel.L3_SENSORS_STALE

    def test_aemet_dominant_over_forecast(self):
        inputs = DegradedModeInputs(forecast_mae=5.0, aemet_age_hours=48)
        assert classify_degraded_level(inputs) == DegradedLevel.L2_AEMET_STALE

    def test_forecast_alone(self):
        inputs = DegradedModeInputs(forecast_mae=1.0)
        assert classify_degraded_level(inputs) == DegradedLevel.L1_FORECAST_DEGRADED

    def test_at_threshold_not_triggered(self):
        # MAE exactly at threshold is still NORMAL (strict >).
        inputs = DegradedModeInputs(forecast_mae=0.5)
        assert classify_degraded_level(inputs) == DegradedLevel.NORMAL


# ── Config validation ──────────────────────────────────────────────────────
class TestConfigValidation:
    def test_invalid_thresholds_rejected(self):
        import pytest
        with pytest.raises(ValueError):
            DegradedModeConfig(forecast_mae_threshold=0)
        with pytest.raises(ValueError):
            DegradedModeConfig(aemet_stale_hours_threshold=-1)


# ── Level 1 ────────────────────────────────────────────────────────────────
class TestLevel1:
    def test_min_runtime_only_decisions_dropped(self):
        plan = Plan(cells=(
            _cell("a", 0, rule_id=8, min_runtime_only=True),
            _cell("b", 0, rule_id=7, min_runtime_only=False),
        ))
        result, level = apply_degraded_mode(
            plan, DegradedModeInputs(forecast_mae=1.0),
        )
        assert level == DegradedLevel.L1_FORECAST_DEGRADED
        actions = {c.load: c.decision.action for c in result.adjusted_plan.cells}
        assert actions["a"] == "off"
        assert actions["b"] == "on"

    def test_off_cells_unchanged_in_l1(self):
        plan = Plan(cells=(_cell("a", 0, action="off"),))
        result, _ = apply_degraded_mode(plan, DegradedModeInputs(forecast_mae=1.0))
        assert result.adjusted_plan.cells[0].decision.action == "off"


# ── Level 2 ────────────────────────────────────────────────────────────────
class TestLevel2:
    def test_all_non_critical_dropped(self):
        plan = Plan(cells=(
            _cell("a", 0, rule_id=3),   # critical-driven
            _cell("b", 0, rule_id=7),   # optimistic
            _cell("c", 0, rule_id=8, min_runtime_only=True),
        ))
        result, level = apply_degraded_mode(
            plan, DegradedModeInputs(aemet_age_hours=48),
        )
        assert level == DegradedLevel.L2_AEMET_STALE
        actions = {c.load: c.decision.action for c in result.adjusted_plan.cells}
        assert actions["a"] == "on"   # rule_id=3 kept
        assert actions["b"] == "off"
        assert actions["c"] == "off"


# ── Level 3 ────────────────────────────────────────────────────────────────
class TestLevel3:
    def test_all_loads_forced_off(self):
        plan = Plan(cells=(
            _cell("a", 0, rule_id=3),
            _cell("b", 0, rule_id=7),
            _cell("c", 0, rule_id=8),
        ))
        result, level = apply_degraded_mode(
            plan, DegradedModeInputs(sensor_age_max_minutes=60),
        )
        assert level == DegradedLevel.L3_SENSORS_STALE
        # Every on becomes off — even rule_id=3.
        actions = {c.load: c.decision.action for c in result.adjusted_plan.cells}
        assert all(a == "off" for a in actions.values())
        # Three overrides.
        assert len(result.overrides) == 3


# ── No degradation pathway ──────────────────────────────────────────────────
class TestNormal:
    def test_no_overrides_when_normal(self):
        plan = Plan(cells=(_cell("a", 0, rule_id=7),))
        result, level = apply_degraded_mode(plan, DegradedModeInputs())
        assert level == DegradedLevel.NORMAL
        assert result.overrides == ()
        assert result.adjusted_plan.cells[0].decision.action == "on"
