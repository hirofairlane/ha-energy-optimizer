"""Tests for invariant helpers."""

from __future__ import annotations

import pytest

from eo.simulator.invariants import (
    ENERGY_TOLERANCE_KWH,
    InvariantViolation,
    check_charge_discharge_mutex,
    check_debt_monotonic_without_execution,
    check_energy_conservation,
    check_inverter_capacity,
    check_min_runtime,
    check_no_contradictory_action_in_slot,
    check_soc_bounded,
)


class TestSocBounded:
    def test_in_bounds_returns_none(self):
        assert check_soc_bounded(50, 20, 80) is None

    def test_at_boundary_ok(self):
        assert check_soc_bounded(20, 20, 80) is None
        assert check_soc_bounded(80, 20, 80) is None

    def test_below_returns_message(self):
        msg = check_soc_bounded(10, 20, 80)
        assert msg is not None
        assert "outside" in msg

    def test_strict_raises(self):
        with pytest.raises(InvariantViolation):
            check_soc_bounded(10, 20, 80, strict=True)


class TestChargeDischargeMutex:
    def test_no_overlap_ok(self):
        assert check_charge_discharge_mutex(1.0, 0.0) is None
        assert check_charge_discharge_mutex(0.0, 1.0) is None
        assert check_charge_discharge_mutex(0.0, 0.0) is None

    def test_both_positive_violation(self):
        msg = check_charge_discharge_mutex(0.5, 0.5)
        assert msg is not None
        assert "simultaneously" in msg

    def test_strict_raises(self):
        with pytest.raises(InvariantViolation):
            check_charge_discharge_mutex(0.5, 0.5, strict=True)


class TestInverterCapacity:
    def test_under_cap_ok(self):
        assert check_inverter_capacity(1.0, 0.0, 0.5, 5.0) is None

    def test_at_cap_ok(self):
        assert check_inverter_capacity(2.0, 1.0, 2.0, 5.0) is None

    def test_over_cap_violation(self):
        msg = check_inverter_capacity(3.0, 2.0, 1.0, 5.0)
        assert msg is not None
        assert "exceeded" in msg


class TestEnergyConservation:
    def test_balanced_step(self):
        # Solar 4, grid_imp 0, bat_dis 0 = 4 sources
        # House 3, planned 0.5, bat_chg 0.5, exp 0, curt 0 = 4 sinks
        assert check_energy_conservation(
            solar_kwh=4.0, grid_import_kwh=0, battery_discharge_kwh=0,
            house_kwh=3.0, planned_loads_kwh=0.5,
            battery_charge_kwh=0.5, grid_export_kwh=0,
            pv_curtailed_kwh=0, unmet_load_kwh=0,
        ) is None

    def test_violation_outside_tolerance(self):
        # 4 sources, but sinks total 3 → 1 kWh missing
        msg = check_energy_conservation(
            solar_kwh=4.0, grid_import_kwh=0, battery_discharge_kwh=0,
            house_kwh=3.0, planned_loads_kwh=0,
            battery_charge_kwh=0, grid_export_kwh=0,
            pv_curtailed_kwh=0, unmet_load_kwh=0,
        )
        assert msg is not None
        assert "conservation" in msg

    def test_unmet_load_balances_demand(self):
        # House asked for 5 kWh, only 2 served (imported), 3 unmet.
        # sources = 0 + 2 + 0 = 2
        # sinks = 5 + 0 + 0 + 0 + 0 - 3 = 2  ✓
        assert check_energy_conservation(
            solar_kwh=0, grid_import_kwh=2.0, battery_discharge_kwh=0,
            house_kwh=5.0, planned_loads_kwh=0,
            battery_charge_kwh=0, grid_export_kwh=0,
            pv_curtailed_kwh=0, unmet_load_kwh=3.0,
        ) is None

    def test_curtailment_balances_surplus(self):
        # Solar 5, house 1, curtailed 4.
        # sources = 5; sinks = 1 + 0 + 0 + 0 + 4 - 0 = 5  ✓
        assert check_energy_conservation(
            solar_kwh=5.0, grid_import_kwh=0, battery_discharge_kwh=0,
            house_kwh=1.0, planned_loads_kwh=0,
            battery_charge_kwh=0, grid_export_kwh=0,
            pv_curtailed_kwh=4.0, unmet_load_kwh=0,
        ) is None

    def test_strict_raises(self):
        with pytest.raises(InvariantViolation):
            check_energy_conservation(
                solar_kwh=4.0, grid_import_kwh=0, battery_discharge_kwh=0,
                house_kwh=3.0, planned_loads_kwh=0,
                battery_charge_kwh=0, grid_export_kwh=0,
                pv_curtailed_kwh=0, unmet_load_kwh=0,
                strict=True,
            )


class TestMinRuntime:
    def test_all_segments_long_enough_ok(self):
        assert check_min_runtime("boiler", [30, 60, 45], 30) is None

    def test_short_segment_violation(self):
        msg = check_min_runtime("boiler", [30, 10, 40], 30)
        assert msg is not None
        assert "min_runtime" in msg

    def test_zero_duration_ignored(self):
        # A zero-length segment is "didn't run", not a violation.
        assert check_min_runtime("boiler", [0, 30], 30) is None

    def test_strict_raises(self):
        with pytest.raises(InvariantViolation):
            check_min_runtime("pool", [5], 30, strict=True)


class TestNoContradictoryActions:
    def test_consistent_actions_ok(self):
        assert check_no_contradictory_action_in_slot(
            0, ["turn_on:switch.a", "turn_on:switch.b"]
        ) is None

    def test_contradictory_actions_flagged(self):
        msg = check_no_contradictory_action_in_slot(
            5, ["turn_on:switch.x", "turn_off:switch.x"]
        )
        assert msg is not None
        assert "contradictory" in msg
        assert "switch.x" in msg

    def test_ignores_unknown_format(self):
        assert check_no_contradictory_action_in_slot(
            0, ["no_colon", "turn_on:switch.b"]
        ) is None


class TestDebtMonotonic:
    def test_execution_allows_drop(self):
        assert check_debt_monotonic_without_execution(
            "boiler", debt_before=2.0, debt_after=1.0, executed_hours=1.0
        ) is None

    def test_drop_without_execution_flagged(self):
        msg = check_debt_monotonic_without_execution(
            "boiler", debt_before=2.0, debt_after=1.0, executed_hours=0.0
        )
        assert msg is not None
        assert "decreased" in msg

    def test_constant_or_growing_debt_ok(self):
        assert check_debt_monotonic_without_execution(
            "boiler", 2.0, 2.0, 0.0
        ) is None
        assert check_debt_monotonic_without_execution(
            "boiler", 2.0, 2.5, 0.0
        ) is None


class TestConstants:
    def test_tolerance_value_sane(self):
        assert 0 < ENERGY_TOLERANCE_KWH < 0.1
