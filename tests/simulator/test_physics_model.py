"""Tests for SingleBatteryPhysicsModel."""

from __future__ import annotations

import pytest

from eo.simulator.physics_model import (
    BatteryConfig,
    HouseSystemConfig,
    SingleBatteryPhysicsModel,
)


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


# ── Config validation ──────────────────────────────────────────────────────
class TestConfigs:
    def test_battery_rejects_non_positive_capacity(self):
        with pytest.raises(ValueError):
            BatteryConfig(capacity_kwh=0)
        with pytest.raises(ValueError):
            BatteryConfig(capacity_kwh=-1)

    def test_battery_rejects_inverted_health_bounds(self):
        with pytest.raises(ValueError):
            _battery(health_min_pct=80, health_max_pct=20)

    def test_battery_rejects_out_of_range_efficiency(self):
        with pytest.raises(ValueError):
            _battery(charge_efficiency=0)
        with pytest.raises(ValueError):
            _battery(charge_efficiency=1.5)

    def test_system_rejects_non_positive_inverter(self):
        with pytest.raises(ValueError):
            HouseSystemConfig(inverter_max_w=0)


# ── Surplus path ───────────────────────────────────────────────────────────
class TestSurplusFlow:
    def test_surplus_charges_battery_until_full(self):
        m = SingleBatteryPhysicsModel()
        # 1 kWh surplus, SOC 50 %, 1.0 hour
        r = m.step(soc_pct=50.0, net_kwh=1.0, dt_hours=1.0,
                   battery=_battery(), system=_system())
        assert r.battery_charge_kwh == pytest.approx(1.0, rel=1e-6)
        assert r.battery_discharge_kwh == 0
        assert r.grid_export_kwh == 0
        assert r.soc_pct_after > 50

    def test_overflow_surplus_exports_to_grid(self):
        # Battery essentially full, but lots of surplus → goes to grid.
        m = SingleBatteryPhysicsModel()
        r = m.step(soc_pct=89.99, net_kwh=5.0, dt_hours=1.0,
                   battery=_battery(), system=_system())
        # tiny charge into the last bit of headroom, rest exports.
        assert r.grid_export_kwh > 0
        assert r.battery_charge_kwh < 0.1

    def test_surplus_curtailed_when_grid_export_capped(self):
        m = SingleBatteryPhysicsModel()
        # Battery full, no export allowed → all surplus is curtailed.
        r = m.step(soc_pct=90.0, net_kwh=3.0, dt_hours=1.0,
                   battery=_battery(), system=_system(grid_max_export_w=0))
        assert r.battery_charge_kwh == 0
        assert r.grid_export_kwh == 0
        assert r.pv_curtailed_kwh == pytest.approx(3.0, rel=1e-6)

    def test_inverter_caps_charge_power(self):
        m = SingleBatteryPhysicsModel()
        # 10 kWh surplus in 1h → 10 kW; but inverter is 5 kW.
        r = m.step(soc_pct=50.0, net_kwh=10.0, dt_hours=1.0,
                   battery=_battery(), system=_system(inverter_max_w=5000.0))
        assert r.battery_charge_kwh <= 5.0 + 1e-9
        # The remainder exports.
        assert r.grid_export_kwh > 0


# ── Deficit path ───────────────────────────────────────────────────────────
class TestDeficitFlow:
    def test_deficit_discharges_battery(self):
        m = SingleBatteryPhysicsModel()
        r = m.step(soc_pct=80.0, net_kwh=-2.0, dt_hours=1.0,
                   battery=_battery(), system=_system())
        assert r.battery_discharge_kwh == pytest.approx(2.0, rel=1e-6)
        assert r.battery_charge_kwh == 0
        assert r.grid_import_kwh == 0
        assert r.soc_pct_after < 80

    def test_deficit_imports_from_grid_when_battery_empty(self):
        m = SingleBatteryPhysicsModel()
        # SOC at health_min, can't discharge → import.
        r = m.step(soc_pct=20.0, net_kwh=-2.0, dt_hours=1.0,
                   battery=_battery(), system=_system())
        assert r.battery_discharge_kwh == 0
        assert r.grid_import_kwh == pytest.approx(2.0, rel=1e-6)

    def test_unmet_load_when_grid_capped(self):
        m = SingleBatteryPhysicsModel()
        r = m.step(soc_pct=20.0, net_kwh=-5.0, dt_hours=1.0,
                   battery=_battery(), system=_system(grid_max_import_w=2000.0))
        assert r.grid_import_kwh == pytest.approx(2.0, rel=1e-6)
        assert r.unmet_load_kwh == pytest.approx(3.0, rel=1e-6)


# ── Forced actions (planner-driven) ─────────────────────────────────────────
class TestForcedActions:
    def test_forced_charge_overrides_natural_self_consumption(self):
        m = SingleBatteryPhysicsModel()
        # Surplus is 0 but planner forces a charge from grid.
        r = m.step(soc_pct=50.0, net_kwh=0.0, dt_hours=1.0,
                   battery=_battery(), system=_system(),
                   forced_charge_w=3000.0)
        assert r.battery_charge_kwh == pytest.approx(3.0, rel=1e-6)
        # The 3 kWh comes from grid.
        assert r.grid_import_kwh == pytest.approx(3.0, rel=1e-6)

    def test_forced_discharge_clipped_at_floor(self):
        m = SingleBatteryPhysicsModel()
        # SOC at min, asked to discharge a lot.
        r = m.step(soc_pct=20.0, net_kwh=0.0, dt_hours=1.0,
                   battery=_battery(), system=_system(),
                   forced_discharge_w=4000.0)
        assert r.battery_discharge_kwh == 0
        # Should record a clip violation.
        assert any("forced_discharge clipped" in v for v in r.violations)

    def test_cannot_force_both_directions(self):
        m = SingleBatteryPhysicsModel()
        with pytest.raises(ValueError):
            m.step(soc_pct=50.0, net_kwh=0.0, dt_hours=1.0,
                   battery=_battery(), system=_system(),
                   forced_charge_w=2000.0, forced_discharge_w=2000.0)

    def test_negative_forced_powers_rejected(self):
        m = SingleBatteryPhysicsModel()
        with pytest.raises(ValueError):
            m.step(soc_pct=50.0, net_kwh=0.0, dt_hours=1.0,
                   battery=_battery(), system=_system(),
                   forced_charge_w=-100.0)


# ── SOC bounds ─────────────────────────────────────────────────────────────
class TestSocBounds:
    def test_soc_never_exceeds_health_max(self):
        m = SingleBatteryPhysicsModel()
        # Try to push SOC past max with massive forced charge.
        r = m.step(soc_pct=89.0, net_kwh=0.0, dt_hours=1.0,
                   battery=_battery(), system=_system(),
                   forced_charge_w=10000.0)
        assert r.soc_pct_after <= 90 + 1e-6

    def test_soc_never_drops_below_health_min(self):
        m = SingleBatteryPhysicsModel()
        r = m.step(soc_pct=21.0, net_kwh=-10.0, dt_hours=1.0,
                   battery=_battery(), system=_system())
        assert r.soc_pct_after >= 20 - 1e-6


# ── Efficiency ────────────────────────────────────────────────────────────
class TestEfficiency:
    def test_charge_efficiency_reduces_dc_delta(self):
        m = SingleBatteryPhysicsModel()
        # 1 kWh AC into a 10 kWh battery at 90 % charge efficiency →
        # 0.9 kWh delivered to DC, SOC rises by 9 pp.
        r = m.step(soc_pct=50.0, net_kwh=1.0, dt_hours=1.0,
                   battery=_battery(charge_efficiency=0.9), system=_system())
        delta_soc = r.soc_pct_after - 50.0
        # 0.9 kWh / 10 kWh × 100 = 9 pp
        assert delta_soc == pytest.approx(9.0, rel=1e-3)

    def test_discharge_efficiency_drains_more_dc(self):
        m = SingleBatteryPhysicsModel()
        # 1 kWh delivered to AC at 90 % discharge efficiency →
        # 1 / 0.9 ≈ 1.111 kWh out of DC, SOC drops by 11.11 pp.
        r = m.step(soc_pct=80.0, net_kwh=-1.0, dt_hours=1.0,
                   battery=_battery(discharge_efficiency=0.9), system=_system())
        delta = 80.0 - r.soc_pct_after
        assert delta == pytest.approx(11.111, rel=1e-3)


# ── Validation ─────────────────────────────────────────────────────────────
class TestValidation:
    def test_non_positive_dt_raises(self):
        m = SingleBatteryPhysicsModel()
        with pytest.raises(ValueError):
            m.step(soc_pct=50.0, net_kwh=0.0, dt_hours=0,
                   battery=_battery(), system=_system())
