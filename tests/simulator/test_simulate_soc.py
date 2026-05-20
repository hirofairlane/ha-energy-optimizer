"""Tests for the simulate_soc orchestrator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eo.simulator.invariants import InvariantViolation
from eo.simulator.physics_model import (
    BatteryConfig,
    HouseSystemConfig,
)
from eo.simulator.simulate_soc import (
    SimulationResult,
    SlotInput,
    simulate_soc,
)


def _utc(min_offset: int = 0) -> datetime:
    return datetime(2026, 5, 20, 12, tzinfo=timezone.utc) + timedelta(minutes=min_offset)


def _battery(**kw) -> BatteryConfig:
    base = dict(capacity_kwh=10.0, health_min_pct=20.0, health_max_pct=90.0,
                charge_efficiency=1.0, discharge_efficiency=1.0,
                max_charge_w=5000.0, max_discharge_w=5000.0)
    base.update(kw)
    return BatteryConfig(**base)


def _system(**kw) -> HouseSystemConfig:
    base = dict(inverter_max_w=5000.0,
                grid_max_import_w=15000.0, grid_max_export_w=10000.0)
    base.update(kw)
    return HouseSystemConfig(**base)


def _slot(min_offset, **kw):
    base = dict(timestamp=_utc(min_offset), solar_kwh=0.0,
                house_kwh=0.0, planned_loads_kwh=0.0,
                forced_charge_w=0.0, forced_discharge_w=0.0,
                dt_hours=0.25)
    base.update(kw)
    return SlotInput(**base)


# ── Empty / smoke ──────────────────────────────────────────────────────────
class TestEmpty:
    def test_empty_input_returns_initial_soc(self):
        r = simulate_soc([], initial_soc_pct=50.0,
                         battery=_battery(), system=_system())
        assert r.final_soc_pct == 50.0
        assert r.slots == []
        assert r.all_violations == []

    def test_single_slot_zero_flows_keeps_soc(self):
        slots = [_slot(0)]
        r = simulate_soc(slots, initial_soc_pct=50.0,
                         battery=_battery(), system=_system())
        assert r.final_soc_pct == pytest.approx(50.0)
        assert r.total_grid_import_kwh == 0
        assert r.total_grid_export_kwh == 0


# ── Time ordering ──────────────────────────────────────────────────────────
class TestTimeOrdering:
    def test_unsorted_input_raises(self):
        slots = [_slot(15), _slot(0)]  # out of order
        with pytest.raises(ValueError, match="ascending"):
            simulate_soc(slots, initial_soc_pct=50.0,
                         battery=_battery(), system=_system())

    def test_duplicate_timestamps_raise(self):
        slots = [_slot(0), _slot(0)]
        with pytest.raises(ValueError):
            simulate_soc(slots, initial_soc_pct=50.0,
                         battery=_battery(), system=_system())


# ── Initial SOC out of bounds ──────────────────────────────────────────────
class TestInitialSoc:
    def test_initial_soc_below_min_clamps_and_records(self):
        slots = [_slot(0)]
        r = simulate_soc(slots, initial_soc_pct=10.0,  # < health_min 20
                         battery=_battery(), system=_system())
        assert any("initial:" in v for v in r.all_violations)
        # And it clamped before the step.
        assert r.final_soc_pct >= 20 - 1e-6

    def test_strict_mode_raises_on_out_of_bounds_initial(self):
        slots = [_slot(0)]
        with pytest.raises(InvariantViolation):
            simulate_soc(slots, initial_soc_pct=10.0,
                         battery=_battery(), system=_system(),
                         strict_invariants=True)


# ── Energy conservation across full runs ───────────────────────────────────
class TestEnergyConservation:
    def test_24h_pure_self_consumption_balances(self):
        # 96 slots × 15 min = 24 h. Solar curve + constant house draw.
        battery = _battery()
        system = _system()
        slots: list[SlotInput] = []
        for i in range(96):
            hour = i // 4
            # Triangular solar curve centred at hour 12, peak 4 kWh/slot.
            solar_kwh = max(0.0, 1.0 * (1 - abs(hour - 12) / 8))
            slots.append(_slot(15 * i, solar_kwh=solar_kwh, house_kwh=0.3))
        r = simulate_soc(slots, initial_soc_pct=50.0,
                         battery=battery, system=system)
        # No invariant violations expected.
        assert r.all_violations == []
        # Daily totals: sources ≈ sinks.
        sources = sum(s.solar_kwh for s in slots) + r.total_grid_import_kwh
        sinks = sum(s.house_kwh for s in slots) + r.total_grid_export_kwh + r.total_pv_curtailed_kwh
        battery_delta_dc = (battery.capacity_kwh
                            * (r.final_soc_pct - 50.0) / 100.0)
        # Net battery contribution from AC side (using efficiencies=1.0):
        # battery_delta_dc = charge - discharge
        net_bat = r.total_battery_charge_kwh - r.total_battery_discharge_kwh
        assert abs(net_bat - battery_delta_dc) < 0.1  # bounded float drift
        # Overall: sources within tolerance of sinks once we account for
        # battery storage delta.
        assert abs((sources - sinks) - net_bat) < 0.1

    def test_strict_mode_raises_on_bad_initial(self):
        # Smoke: strict initialisation gate works.
        slots = [_slot(0)]
        with pytest.raises(InvariantViolation):
            simulate_soc(slots, initial_soc_pct=5.0,
                         battery=_battery(), system=_system(),
                         strict_invariants=True)


# ── Trajectory shape ───────────────────────────────────────────────────────
class TestTrajectory:
    def test_charging_increases_soc_over_slots(self):
        # 4 slots × 15 min of pure surplus
        slots = [_slot(15 * i, solar_kwh=1.0) for i in range(4)]
        r = simulate_soc(slots, initial_soc_pct=50.0,
                         battery=_battery(), system=_system())
        socs = [s.soc_pct for s in r.slots]
        assert socs == sorted(socs)  # monotonic non-decreasing
        assert r.final_soc_pct > 50

    def test_discharging_decreases_soc(self):
        slots = [_slot(15 * i, house_kwh=1.0) for i in range(4)]
        r = simulate_soc(slots, initial_soc_pct=80.0,
                         battery=_battery(), system=_system())
        socs = [s.soc_pct for s in r.slots]
        assert socs == sorted(socs, reverse=True)
        assert r.final_soc_pct < 80

    def test_forced_charge_overrides_self_consumption(self):
        # Even with no surplus, forced_charge must charge from grid.
        slots = [_slot(0, forced_charge_w=2000.0, dt_hours=0.25)]
        r = simulate_soc(slots, initial_soc_pct=50.0,
                         battery=_battery(), system=_system())
        slot = r.slots[0]
        assert slot.battery_charge_kwh > 0
        assert slot.grid_import_kwh == pytest.approx(0.5, rel=1e-3)


# ── planned_loads accounted ─────────────────────────────────────────────────
class TestPlannedLoads:
    def test_planned_loads_increase_house_demand(self):
        # 4 slots × 15 min: solar=1 kWh, house=0.2 kWh, planned=0.5 kWh.
        # Net = 1 - 0.2 - 0.5 = 0.3 kWh surplus.
        slots = [_slot(15 * i, solar_kwh=1.0, house_kwh=0.2, planned_loads_kwh=0.5)
                 for i in range(4)]
        r = simulate_soc(slots, initial_soc_pct=50.0,
                         battery=_battery(), system=_system())
        # Total kWh delivered = 4 × 0.3 = 1.2 kWh into battery (no losses).
        assert r.total_battery_charge_kwh == pytest.approx(1.2, rel=1e-3)
        assert r.total_battery_discharge_kwh == 0
        assert r.all_violations == []


# ── Result helpers ─────────────────────────────────────────────────────────
class TestResult:
    def test_has_violations_flag(self):
        slots = [_slot(0)]
        r_ok = simulate_soc(slots, initial_soc_pct=50.0,
                            battery=_battery(), system=_system())
        assert r_ok.has_violations is False

        r_bad = simulate_soc(slots, initial_soc_pct=5.0,
                             battery=_battery(), system=_system())
        assert r_bad.has_violations is True
