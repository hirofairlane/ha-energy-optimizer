"""Tests for the 11-row decision matrix (SPEC §1.4 P6)."""

from __future__ import annotations

import pytest

from eo.planner.decision_matrix import (
    LoadDecision,
    SlotContext,
    decide_load_for_slot,
)
from eo.planner.load_quota import (
    DebtState,
    LoadQuotaConfig,
    LoadQuotaState,
    compute_debt_state,
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


def _ctx(**kw) -> SlotContext:
    base = dict(period="valley", solar_surplus_now=False,
                prob_surplus_tomorrow=0.0,
                prob_surplus_next_valley=0.0,
                no_valley_before_deadline=False)
    base.update(kw)
    return SlotContext(**base)


# ── Row 1: OK ───────────────────────────────────────────────────────────────
def test_row1_debt_ok_returns_off():
    d = decide_load_for_slot(_config(), _state(DebtState.OK), _ctx())
    assert d.action == "off"
    assert d.rule_id == 1


# ── Row 2: Surplus NOW ─────────────────────────────────────────────────────
def test_row2_surplus_now_turns_on_even_when_mid():
    d = decide_load_for_slot(
        _config(), _state(DebtState.LOW),
        _ctx(period="mid", solar_surplus_now=True),
    )
    assert d.action == "on"
    assert d.rule_id == 2


def test_row2_surplus_in_peak_is_overridden_by_peak_ban():
    # SPEC: surplus → ON only if period != peak.
    d = decide_load_for_slot(
        _config(), _state(DebtState.LOW),
        _ctx(period="peak", solar_surplus_now=True),
    )
    assert d.action == "off"
    assert d.rule_id == 5


# ── Rows 3 / 4: Critical + tariff ───────────────────────────────────────────
def test_row3_critical_in_valley_forces_on():
    d = decide_load_for_slot(
        _config(), _state(DebtState.CRITICAL), _ctx(period="valley"),
    )
    assert d.action == "on"
    assert d.rule_id == 3


def test_row3_critical_in_mid_forces_on():
    d = decide_load_for_slot(
        _config(), _state(DebtState.CRITICAL), _ctx(period="mid"),
    )
    assert d.action == "on"


def test_row4_critical_in_peak_blocks_with_alert():
    d = decide_load_for_slot(
        _config(allow_peak_on_critical=False),
        _state(DebtState.CRITICAL), _ctx(period="peak"),
    )
    assert d.action == "off"
    assert d.rule_id == 4
    assert d.alert is True


def test_row4_critical_in_peak_with_override_runs_with_alert():
    d = decide_load_for_slot(
        _config(allow_peak_on_critical=True),
        _state(DebtState.CRITICAL), _ctx(period="peak"),
    )
    assert d.action == "on"
    assert d.rule_id == 4
    assert d.alert is True


def test_row4_irreachable_in_peak_also_blocks_with_alert():
    d = decide_load_for_slot(
        _config(), _state(DebtState.IRREACHABLE), _ctx(period="peak"),
    )
    assert d.action == "off"
    assert d.alert is True


# ── Row 5: peak without critical ────────────────────────────────────────────
def test_row5_peak_non_critical_off():
    for debt in (DebtState.LOW, DebtState.MEDIUM, DebtState.HIGH):
        d = decide_load_for_slot(
            _config(), _state(debt), _ctx(period="peak"),
        )
        assert d.action == "off"
        assert d.rule_id == 5


# ── Row 6: valley + high prob_surplus → defer ──────────────────────────────
def test_row6_valley_high_prob_tomorrow_defers():
    d = decide_load_for_slot(
        _config(required_confidence_pct=60),
        _state(DebtState.HIGH),
        _ctx(period="valley", prob_surplus_tomorrow=0.8),
    )
    assert d.action == "off"
    assert d.rule_id == 6


# ── Rows 7 / 8 / 9: valley + low prob_surplus ──────────────────────────────
def test_row7_valley_low_prob_high_debt_on():
    d = decide_load_for_slot(
        _config(),
        _state(DebtState.HIGH),
        _ctx(period="valley", prob_surplus_tomorrow=0.2),
    )
    assert d.action == "on"
    assert d.rule_id == 7
    assert d.min_runtime_only is False


def test_row8_valley_low_prob_medium_debt_min_runtime():
    d = decide_load_for_slot(
        _config(),
        _state(DebtState.MEDIUM),
        _ctx(period="valley", prob_surplus_tomorrow=0.2),
    )
    assert d.action == "on"
    assert d.rule_id == 8
    assert d.min_runtime_only is True


def test_row9_valley_low_prob_low_debt_off():
    d = decide_load_for_slot(
        _config(),
        _state(DebtState.LOW),
        _ctx(period="valley", prob_surplus_tomorrow=0.2),
    )
    assert d.action == "off"
    assert d.rule_id == 9


# ── Rows 10 / 11: mid period ───────────────────────────────────────────────
def test_row10_mid_high_prob_next_valley_defers():
    d = decide_load_for_slot(
        _config(required_confidence_pct=60),
        _state(DebtState.HIGH),
        _ctx(period="mid", prob_surplus_next_valley=0.85),
    )
    assert d.action == "off"
    assert d.rule_id == 10


def test_row11_mid_no_valley_high_debt_on():
    d = decide_load_for_slot(
        _config(),
        _state(DebtState.HIGH),
        _ctx(period="mid", no_valley_before_deadline=True),
    )
    assert d.action == "on"
    assert d.rule_id == 11


# ── Robustness ─────────────────────────────────────────────────────────────
def test_unknown_period_defaults_to_off():
    d = decide_load_for_slot(
        _config(), _state(DebtState.HIGH),
        _ctx(period="mars-tariff"),
    )
    assert d.action == "off"
    assert d.rule_id == 0
