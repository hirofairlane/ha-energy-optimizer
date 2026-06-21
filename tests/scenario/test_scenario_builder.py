"""Tests for the ScenarioBuilder."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from eo.planner.load_quota import DebtState
from eo.scenario.scenario_builder import (
    QuantileHourForecast,
    RiskTolerance,
    build_scenario,
    risk_from_debt_state,
)


def _utc(h: int) -> datetime:
    return datetime(2026, 5, 20, h, tzinfo=timezone.utc)


def _qhf(h, p10, p50, p90) -> QuantileHourForecast:
    return QuantileHourForecast(hour_start=_utc(h), p10=p10, p50=p50, p90=p90)


# ── Debt → risk mapping (SPEC §1.2 S4) ─────────────────────────────────────
class TestDebtMapping:
    def test_ok_low_medium_map_to_median(self):
        for d in (DebtState.OK, DebtState.LOW, DebtState.MEDIUM):
            assert risk_from_debt_state(d) == RiskTolerance.MEDIAN

    def test_high_critical_irreachable_map_to_conservative(self):
        for d in (DebtState.HIGH, DebtState.CRITICAL, DebtState.IRREACHABLE):
            assert risk_from_debt_state(d) == RiskTolerance.CONSERVATIVE


# ── Quantile picking ───────────────────────────────────────────────────────
class TestQuantilePicking:
    def test_median_uses_p50(self):
        solar = [_qhf(0, 1, 5, 9)]
        house = [_qhf(0, 0.5, 1.0, 2.0)]
        s = build_scenario(solar, house, RiskTolerance.MEDIAN)
        # Each hour split into 4 slots: p50 / 4
        assert all(v == 5 / 4 for v in s.solar_kwh)
        assert all(v == 1.0 / 4 for v in s.house_kwh)

    def test_conservative_uses_p10_solar_p90_house(self):
        solar = [_qhf(0, 1, 5, 9)]
        house = [_qhf(0, 0.5, 1.0, 2.0)]
        s = build_scenario(solar, house, RiskTolerance.CONSERVATIVE)
        assert all(v == 1 / 4 for v in s.solar_kwh)
        assert all(v == 2.0 / 4 for v in s.house_kwh)

    def test_optimistic_uses_p90_solar_p10_house(self):
        solar = [_qhf(0, 1, 5, 9)]
        house = [_qhf(0, 0.5, 1.0, 2.0)]
        s = build_scenario(solar, house, RiskTolerance.OPTIMISTIC)
        assert all(v == 9 / 4 for v in s.solar_kwh)
        assert all(v == 0.5 / 4 for v in s.house_kwh)

    def test_stress_matches_conservative_quantile_choice(self):
        solar = [_qhf(0, 1, 5, 9)]
        house = [_qhf(0, 0.5, 1.0, 2.0)]
        s = build_scenario(solar, house, RiskTolerance.STRESS)
        assert all(v == 1 / 4 for v in s.solar_kwh)
        assert all(v == 2.0 / 4 for v in s.house_kwh)


# ── Slot interpolation ─────────────────────────────────────────────────────
class TestInterpolation:
    def test_two_hours_eight_slots(self):
        solar = [_qhf(0, 1, 4, 8), _qhf(1, 1, 2, 4)]
        house = [_qhf(0, 0, 1, 2), _qhf(1, 0, 1, 2)]
        s = build_scenario(solar, house, RiskTolerance.MEDIAN)
        assert len(s.slot_starts) == 8
        # First 4 entries: hour 0, p50=4 → 1.0 each
        assert s.solar_kwh[:4] == (1.0, 1.0, 1.0, 1.0)
        # Next 4: hour 1, p50=2 → 0.5 each
        assert s.solar_kwh[4:] == (0.5, 0.5, 0.5, 0.5)


# ── Validation ─────────────────────────────────────────────────────────────
class TestValidation:
    def test_empty_input_yields_empty_scenario(self):
        s = build_scenario([], [], RiskTolerance.MEDIAN)
        assert s.slot_starts == ()
        assert s.solar_kwh == ()
        assert s.house_kwh == ()

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            build_scenario(
                [_qhf(0, 1, 5, 9)],
                [_qhf(0, 0.5, 1.0, 2.0), _qhf(1, 0, 1, 2)],
                RiskTolerance.MEDIAN,
            )

    def test_mismatched_hour_starts_raise(self):
        with pytest.raises(ValueError, match="hour_start"):
            build_scenario(
                [_qhf(0, 1, 5, 9)],
                [_qhf(5, 0.5, 1.0, 2.0)],
                RiskTolerance.MEDIAN,
            )


# ── Metadata ───────────────────────────────────────────────────────────────
class TestMetadata:
    def test_confidence_flag_propagates(self):
        solar = [_qhf(0, 1, 5, 9)]
        house = [_qhf(0, 0.5, 1.0, 2.0)]
        s = build_scenario(solar, house, RiskTolerance.MEDIAN,
                            confidence_is_heuristic=True)
        assert s.confidence_is_heuristic is True
        s2 = build_scenario(solar, house, RiskTolerance.MEDIAN,
                             confidence_is_heuristic=False)
        assert s2.confidence_is_heuristic is False

    def test_debt_state_propagates(self):
        s = build_scenario([], [], RiskTolerance.MEDIAN, debt_state=DebtState.HIGH)
        assert s.debt_state == DebtState.HIGH

    def test_to_dict_serialises(self):
        import json
        solar = [_qhf(0, 1, 5, 9)]
        house = [_qhf(0, 0.5, 1.0, 2.0)]
        s = build_scenario(solar, house, RiskTolerance.CONSERVATIVE,
                           debt_state=DebtState.HIGH)
        json.dumps(s.to_dict())
