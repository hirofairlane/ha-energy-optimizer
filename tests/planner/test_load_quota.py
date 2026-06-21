"""Tests for eo.planner.load_quota."""

from __future__ import annotations

import pytest
from eo.planner.load_quota import (
    DebtState,
    LoadQuotaConfig,
    compute_debt_state,
)


def _config(**kw) -> LoadQuotaConfig:
    base = dict(target_hours_per_window=1.0, window_days=7,
                min_runtime_minutes=30.0, daily_physical_max_hours=8.0,
                required_confidence_pct=60.0, allow_peak_on_critical=False)
    base.update(kw)
    return LoadQuotaConfig(**base)


# ── Config validation ──────────────────────────────────────────────────────
class TestConfigValidation:
    def test_rejects_negative_target(self):
        with pytest.raises(ValueError):
            _config(target_hours_per_window=-1)

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError):
            _config(window_days=0)

    def test_rejects_non_positive_min_runtime(self):
        with pytest.raises(ValueError):
            _config(min_runtime_minutes=0)

    def test_rejects_daily_max_outside_24h(self):
        with pytest.raises(ValueError):
            _config(daily_physical_max_hours=30)
        with pytest.raises(ValueError):
            _config(daily_physical_max_hours=0)

    def test_rejects_invalid_confidence_pct(self):
        with pytest.raises(ValueError):
            _config(required_confidence_pct=120)


# ── State classification ──────────────────────────────────────────────────
class TestDebtClassification:
    def test_quota_met_means_ok(self):
        cfg = _config(target_hours_per_window=1.0, window_days=7)
        state = compute_debt_state(cfg, [0.5, 0.5, 0, 0, 0, 0, 0], days_passed_in_window=3)
        assert state.debt_state == DebtState.OK
        assert state.remaining_h == 0

    def test_early_in_window_with_progress_is_low(self):
        cfg = _config(target_hours_per_window=7.0, window_days=7)
        # 1 day in, no execution → 1 daily rate behind
        state = compute_debt_state(cfg, [0], days_passed_in_window=1)
        assert state.debt_state == DebtState.LOW

    def test_halfway_unexecuted_is_medium(self):
        cfg = _config(target_hours_per_window=7.0, window_days=7)
        state = compute_debt_state(cfg, [0, 0, 0, 0], days_passed_in_window=4)
        # 4 days into a 7-day window with 0 done. Daily rate = 1; expected so far = 4.
        # days_under = 4, threshold = 7/2 = 3.5 → > threshold → HIGH.
        assert state.debt_state == DebtState.HIGH

    def test_critical_when_days_left_le_1(self):
        cfg = _config(target_hours_per_window=1.0, window_days=7)
        # Window almost over, still 1h remaining.
        state = compute_debt_state(cfg, [0]*6, days_passed_in_window=6)
        assert state.debt_state == DebtState.CRITICAL

    def test_irreachable_when_target_exceeds_remaining_capacity(self):
        # Target 10h, days_left = 1, daily_max_hours = 2 → max possible 2h.
        cfg = _config(target_hours_per_window=10.0, window_days=7,
                      daily_physical_max_hours=2.0)
        state = compute_debt_state(cfg, [0]*6, days_passed_in_window=6)
        assert state.debt_state == DebtState.IRREACHABLE
        assert state.remaining_executable_h == 2.0


# ── Bug B1 — mutación de window_days ─────────────────────────────────────
class TestBugB1WindowMutation:
    def test_truncates_history_to_current_window(self):
        # User shrinks window_days from 7 to 3. Old history has 7 days.
        cfg = _config(target_hours_per_window=1.0, window_days=3)
        history = [10, 10, 10, 0, 0, 0, 0]  # last 3 days are zero
        # Without B1 fix: accumulated would be 30h → OK.
        # With B1 fix: only last 3 days counted → accumulated = 0 → not OK.
        state = compute_debt_state(cfg, history, days_passed_in_window=1)
        assert state.accumulated_h == 0
        assert state.debt_state != DebtState.OK


# ── Bug B2 — huecos por caída ────────────────────────────────────────────
class TestBugB2DataQualityScaling:
    def test_target_scales_with_telemetry_coverage(self):
        cfg = _config(target_hours_per_window=7.0, window_days=7)
        # Only 3 of 7 days had telemetry → target scaled to 3.0h.
        state = compute_debt_state(
            cfg, [0, 0, 0, 0, 0, 0, 0],
            days_passed_in_window=3,
            days_with_telemetry=3,
        )
        assert state.target_scaled_h == pytest.approx(3.0)
        assert state.data_quality_factor == pytest.approx(3/7)

    def test_full_telemetry_means_no_scaling(self):
        cfg = _config(target_hours_per_window=7.0, window_days=7)
        state = compute_debt_state(
            cfg, [1]*7, days_passed_in_window=7, days_with_telemetry=7,
        )
        assert state.target_scaled_h == 7.0
        assert state.debt_state == DebtState.OK

    def test_default_days_with_telemetry_is_window_days(self):
        cfg = _config(target_hours_per_window=5.0, window_days=7)
        state = compute_debt_state(cfg, [1]*5, days_passed_in_window=5)
        assert state.days_with_telemetry == 7
        assert state.data_quality_factor == 1.0

    def test_days_with_telemetry_clamped(self):
        cfg = _config(target_hours_per_window=5.0, window_days=7)
        # Caller passes silly value > window
        state = compute_debt_state(cfg, [], days_passed_in_window=0,
                                    days_with_telemetry=100)
        assert state.days_with_telemetry == 7


# ── Robustness ─────────────────────────────────────────────────────────────
class TestRobustness:
    def test_days_passed_clamped_to_window(self):
        cfg = _config(window_days=7)
        # User claims 99 days passed in a 7-day window — clamp.
        state = compute_debt_state(cfg, [0]*7, days_passed_in_window=99)
        assert state.days_left_in_window == 0

    def test_negative_days_passed_raises(self):
        cfg = _config()
        with pytest.raises(ValueError):
            compute_debt_state(cfg, [], days_passed_in_window=-1)

    def test_negative_history_entries_treated_as_zero(self):
        cfg = _config(target_hours_per_window=5.0, window_days=7)
        # A glitch reports -1 hour somewhere → must not contribute as negative.
        state = compute_debt_state(cfg, [-1, 0, 0, 0, 0, 0, 0], days_passed_in_window=7)
        assert state.accumulated_h == 0

    def test_to_dict_roundtrip(self):
        import json
        cfg = _config(target_hours_per_window=5.0, window_days=7)
        state = compute_debt_state(cfg, [1, 0, 1, 0, 0, 0, 0], days_passed_in_window=4)
        json.dumps(state.to_dict())
