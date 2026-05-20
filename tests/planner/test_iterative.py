"""Tests for the iterative planner convergence loop.

Includes Gemini R1's "Monstruo del Bucle" reproduction and the
forced_states injection guarantee (SPEC §1.4 P5).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eo.planner.decision_matrix import SlotContext
from eo.planner.iterative import (
    MAX_PLANNER_ITERATIONS,
    ForcedState,
    LoadInputs,
    PlannerResult,
    SlotMeta,
    iterate,
)
from eo.planner.load_quota import (
    DebtState,
    LoadQuotaConfig,
    LoadQuotaState,
)


def _config(**kw) -> LoadQuotaConfig:
    base = dict(target_hours_per_window=1.0, window_days=7,
                min_runtime_minutes=30.0, daily_physical_max_hours=8.0,
                required_confidence_pct=60.0, allow_peak_on_critical=False)
    base.update(kw)
    return LoadQuotaConfig(**base)


def _state(debt: DebtState) -> LoadQuotaState:
    return LoadQuotaState(
        debt_state=debt, accumulated_h=0, target_scaled_h=1.0,
        remaining_h=1.0, remaining_executable_h=8.0,
        days_left_in_window=3, days_with_telemetry=7,
        data_quality_factor=1.0,
    )


def _slot(i: int, period="valley") -> SlotMeta:
    return SlotMeta(
        timestamp=datetime(2026, 5, 20, tzinfo=timezone.utc) + timedelta(minutes=15 * i),
        period=period,
        prob_surplus_tomorrow=0.0,
        prob_surplus_next_valley=0.0,
    )


def _stable_context(load, slot, iteration):
    return SlotContext(
        period=slot.period,
        solar_surplus_now=False,
        prob_surplus_tomorrow=slot.prob_surplus_tomorrow,
        prob_surplus_next_valley=slot.prob_surplus_next_valley,
        no_valley_before_deadline=False,
    )


# ── Convergence ─────────────────────────────────────────────────────────────
class TestConvergence:
    def test_stable_context_converges_in_2_iterations(self):
        loads = [LoadInputs(name="boiler", config=_config(), quota_state=_state(DebtState.HIGH))]
        slots = [_slot(i) for i in range(4)]
        result = iterate(loads, slots, _stable_context)
        # Iteration 1 = first plan; iteration 2 = same plan → convergence.
        assert result.converged is True
        assert result.iterations == 2
        assert result.oscillation_detected is False

    def test_empty_loads_returns_empty_plan(self):
        slots = [_slot(i) for i in range(2)]
        result = iterate([], slots, _stable_context)
        assert result.plan.cells == ()
        # Empty plan converges trivially after 2 iterations.

    def test_max_iter_validation(self):
        loads = [LoadInputs(name="x", config=_config(), quota_state=_state(DebtState.OK))]
        slots = [_slot(0)]
        with pytest.raises(ValueError):
            iterate(loads, slots, _stable_context, max_iter=0)


# ── Oscillation detection ─────────────────────────────────────────────────
class TestOscillation:
    def test_oscillating_context_detected(self):
        """Build a context_builder that flips action each iteration → 2-cycle."""
        loads = [LoadInputs(name="x", config=_config(), quota_state=_state(DebtState.HIGH))]
        slots = [_slot(0)]

        # Use prob_surplus_tomorrow to flip the decision (rows 6 vs 7).
        # On even iterations: prob = 0.9 → row 6 (OFF).
        # On odd iterations:  prob = 0.0 → row 7 (ON).
        def flipping(load, slot, iteration):
            prob = 0.9 if iteration % 2 == 0 else 0.0
            return SlotContext(
                period="valley", solar_surplus_now=False,
                prob_surplus_tomorrow=prob,
                prob_surplus_next_valley=0.0,
                no_valley_before_deadline=False,
            )

        result = iterate(loads, slots, flipping)
        assert result.oscillation_detected is True
        assert result.converged is False


# ── Max iterations guardrail ──────────────────────────────────────────────
class TestMaxIterGuardrail:
    def test_monstruo_del_bucle_caps_iterations(self):
        """Gemini R1 Test 1: a load whose every iteration flips → planner
        must not loop forever.

        We engineer a 3-cycle by cycling the period through the three
        tariff bands, which makes the decision matrix fire a different rule
        each step (7 → 5 → 11). Without max_iter, this loops forever; with
        max_iter=5 we get capped (and oscillation_detected stays False
        because the hash period is 3, not 2).
        """
        loads = [LoadInputs(name="termo", config=_config(),
                            quota_state=_state(DebtState.HIGH))]
        slots = [_slot(0)]

        def three_cycle(load, slot, iteration):
            phase = iteration % 3
            period = {0: "valley", 1: "peak", 2: "mid"}[phase]
            return SlotContext(
                period=period, solar_surplus_now=False,
                prob_surplus_tomorrow=0.0,
                prob_surplus_next_valley=0.0,
                no_valley_before_deadline=True,  # so mid + high debt fires row 11
            )

        result = iterate(loads, slots, three_cycle, max_iter=5)
        assert result.converged is False
        assert result.oscillation_detected is False
        assert result.iterations == 5


# ── Forced states ─────────────────────────────────────────────────────────
class TestForcedStates:
    def test_forced_off_bypasses_decision_matrix(self):
        """SPEC §1.4 P5: forced_states (e.g. anti-flap block) override the
        planner's natural choice."""
        loads = [LoadInputs(name="x", config=_config(),
                            quota_state=_state(DebtState.HIGH))]
        slots = [_slot(0, period="valley")]  # would naturally fire ON
        forced = [ForcedState(load="x", slot_index=0, action="off",
                              reason="anti_flap_active")]
        result = iterate(loads, slots, _stable_context, forced_states=forced)
        cell = result.plan.cells_for_load("x")[0]
        assert cell.decision.action == "off"
        assert "forced by policy" in cell.decision.reason
        assert "anti_flap_active" in cell.decision.reason

    def test_forced_on_bypasses_off_decision(self):
        loads = [LoadInputs(name="x", config=_config(),
                            quota_state=_state(DebtState.OK))]  # row 1 → off
        slots = [_slot(0)]
        forced = [ForcedState(load="x", slot_index=0, action="on",
                              reason="manual_override")]
        result = iterate(loads, slots, _stable_context, forced_states=forced)
        cell = result.plan.cells_for_load("x")[0]
        assert cell.decision.action == "on"

    def test_only_targeted_slot_forced(self):
        loads = [LoadInputs(name="x", config=_config(),
                            quota_state=_state(DebtState.HIGH))]
        slots = [_slot(0), _slot(1), _slot(2)]
        forced = [ForcedState(load="x", slot_index=0, action="off", reason="slot0_only")]
        result = iterate(loads, slots, _stable_context, forced_states=forced)
        cells = result.plan.cells_for_load("x")
        assert cells[0].decision.action == "off"
        # Slot 1 and 2 follow normal logic (HIGH debt, valley, low prob → row 7 ON).
        assert cells[1].decision.action == "on"
        assert cells[2].decision.action == "on"


# ── Alerts collection ─────────────────────────────────────────────────────
class TestAlerts:
    def test_critical_peak_alert_collected(self):
        loads = [LoadInputs(name="boiler", config=_config(),
                            quota_state=_state(DebtState.CRITICAL))]
        slots = [_slot(0, period="peak")]
        result = iterate(loads, slots, _stable_context)
        assert len(result.alerts) > 0
        assert any("boiler" in a for a in result.alerts)


# ── Result helpers ─────────────────────────────────────────────────────────
class TestResultHelpers:
    def test_to_dict_serialises_summary(self):
        import json
        loads = [LoadInputs(name="x", config=_config(),
                            quota_state=_state(DebtState.OK))]
        slots = [_slot(i) for i in range(3)]
        result = iterate(loads, slots, _stable_context)
        json.dumps(result.to_dict())

    def test_cells_for_load_filters(self):
        loads = [
            LoadInputs(name="a", config=_config(), quota_state=_state(DebtState.OK)),
            LoadInputs(name="b", config=_config(), quota_state=_state(DebtState.OK)),
        ]
        slots = [_slot(0), _slot(1)]
        result = iterate(loads, slots, _stable_context)
        assert len(result.plan.cells_for_load("a")) == 2
        assert len(result.plan.cells_for_load("b")) == 2


# ── Spec constant ─────────────────────────────────────────────────────────
class TestSpecConstants:
    def test_max_iterations_matches_spec(self):
        assert MAX_PLANNER_ITERATIONS == 5
