"""Unit tests for eo.checks.setup_integrity.

The check is a pure function, so the tests are lightweight: build a wizard
dict, run check(), assert on the report.
"""

from __future__ import annotations

from eo.checks.setup_integrity import check


def _wizard(**overrides) -> dict:
    """Build a minimal valid wizard config, override pieces per test."""
    base = {
        "sensors": {
            "battery_soc": "sensor.battery_soc",
            "battery_power": "sensor.battery_power",
            "grid_power": "sensor.grid_power",
            "solar_power": "sensor.solar_power",
            "pool_switch": "switch.pool_pump",
            "dishwasher_switch": "switch.dishwasher",
            "dishwasher_state": "sensor.dishwasher_state",
        },
        "custom_loads": [],
        "hvac_zones": [],
        "grid_submeters": [],
    }
    base.update(overrides)
    return base


# ── Happy path ───────────────────────────────────────────────────────────────
class TestNoConflicts:
    def test_empty_wizard(self):
        report = check({})
        assert report.ok is True
        assert report.conflicts == []

    def test_none_input(self):
        report = check(None)  # type: ignore[arg-type]
        assert report.ok is True
        assert report.conflicts == []

    def test_minimal_valid_setup(self):
        report = check(_wizard())
        assert report.ok is True
        assert report.conflicts == []

    def test_full_setup_no_collisions(self):
        wiz = _wizard(
            custom_loads=[
                {"name": "Boiler", "switch": "switch.boiler", "watts": 2000},
                {"name": "EV", "switch": "switch.ev_charger", "watts": 7400},
            ],
            hvac_zones=[
                {
                    "name": "Aerotermia",
                    "climate": "climate.aerotermia",
                    "temp_sensor": "sensor.indoor_aero",
                    "temp_heat": "number.ebusd_heat",
                    "temp_cool": "number.ebusd_cool",
                },
            ],
            grid_submeters=[
                {"name": "ACS", "entity": "sensor.acs_power"},
            ],
        )
        report = check(wiz)
        assert report.ok is True
        assert report.conflicts == []


# ── Critical conflicts (the Karplyak bug) ───────────────────────────────────
class TestCriticalConflicts:
    def test_pool_switch_also_custom_load(self):
        """Karplyak #4 exact reproduction: pool_switch == custom_loads[0].switch."""
        wiz = _wizard(
            custom_loads=[
                {"name": "Boiler", "switch": "switch.pool_pump", "watts": 2000},
            ],
        )
        report = check(wiz)
        assert report.ok is False
        assert len(report.critical_conflicts) == 1
        conflict = report.critical_conflicts[0]
        assert conflict.entity_id == "switch.pool_pump"
        assert conflict.actuable_count == 2
        roles = {o.role for o in conflict.occurrences}
        assert "sensors.pool_switch" in roles
        assert "custom_loads[Boiler].switch" in roles

    def test_two_custom_loads_same_switch(self):
        wiz = _wizard(
            custom_loads=[
                {"name": "A", "switch": "switch.shared", "watts": 1000},
                {"name": "B", "switch": "switch.shared", "watts": 500},
            ],
        )
        report = check(wiz)
        assert report.ok is False
        assert len(report.critical_conflicts) == 1
        assert report.critical_conflicts[0].entity_id == "switch.shared"

    def test_actuable_collides_with_sensor_role(self):
        """A switch entity used as both pool_switch and battery_soc — clearly wrong."""
        wiz = _wizard(
            sensors={
                "battery_soc": "switch.bad",
                "pool_switch": "switch.bad",
            },
        )
        report = check(wiz)
        assert report.ok is False
        assert len(report.critical_conflicts) == 1

    def test_hvac_climate_collides_with_custom_load(self):
        wiz = _wizard(
            custom_loads=[
                {"name": "Aero", "switch": "climate.aerotermia", "watts": 3000},
            ],
            hvac_zones=[
                {"name": "Aero", "climate": "climate.aerotermia"},
            ],
        )
        report = check(wiz)
        assert report.ok is False
        critical = report.critical_conflicts
        assert len(critical) == 1
        assert critical[0].entity_id == "climate.aerotermia"


# ── Warnings (non-actuable duplicates) ──────────────────────────────────────
class TestWarnings:
    def test_same_sensor_in_two_sensor_roles(self):
        wiz = _wizard(
            sensors={
                "battery_soc": "sensor.x",
                "temp_outdoor": "sensor.x",
            },
        )
        report = check(wiz)
        assert report.ok is True
        assert len(report.warnings) == 1
        assert report.warnings[0].entity_id == "sensor.x"
        assert report.warnings[0].severity == "warning"

    def test_submeter_also_used_as_grid_power(self):
        wiz = _wizard(
            sensors={"grid_power": "sensor.acs_power"},
            grid_submeters=[{"name": "ACS", "entity": "sensor.acs_power"}],
        )
        report = check(wiz)
        assert report.ok is True
        assert len(report.warnings) == 1

    def test_warnings_dont_fail_setup(self):
        """An OK setup with cosmetic dup is still OK overall."""
        wiz = _wizard(
            sensors={
                "battery_soc": "sensor.dup",
                "temp_outdoor": "sensor.dup",
            },
        )
        report = check(wiz)
        assert report.ok is True


# ── Mixed scenarios ─────────────────────────────────────────────────────────
class TestMixed:
    def test_one_critical_plus_one_warning(self):
        wiz = _wizard(
            sensors={
                "battery_soc": "sensor.x",
                "temp_outdoor": "sensor.x",        # warning
                "pool_switch": "switch.flap",
            },
            custom_loads=[
                {"name": "B", "switch": "switch.flap"},  # critical
            ],
        )
        report = check(wiz)
        assert report.ok is False
        assert len(report.critical_conflicts) == 1
        assert len(report.warnings) == 1
        # Critical first in sort order
        assert report.conflicts[0].severity == "critical"
        assert report.conflicts[1].severity == "warning"


# ── Robustness ──────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_empty_strings_ignored(self):
        wiz = _wizard(sensors={"pool_switch": "", "dishwasher_switch": ""})
        report = check(wiz)
        assert report.ok is True
        assert report.conflicts == []

    def test_none_values_ignored(self):
        wiz = _wizard(sensors={"pool_switch": None, "battery_soc": "sensor.x"})
        report = check(wiz)
        assert report.ok is True

    def test_non_string_values_ignored(self):
        wiz = _wizard(sensors={"pool_switch": 123, "battery_soc": "sensor.x"})
        report = check(wiz)
        assert report.ok is True

    def test_malformed_custom_loads_skipped(self):
        wiz = _wizard(custom_loads=["not a dict", None, {"switch": "switch.ok"}])
        report = check(wiz)
        assert report.ok is True

    def test_summary_includes_all_conflicts(self):
        wiz = _wizard(
            sensors={"pool_switch": "switch.dup"},
            custom_loads=[{"name": "X", "switch": "switch.dup"}],
        )
        report = check(wiz)
        summary = report.to_summary()
        assert "CRITICAL" in summary
        assert "switch.dup" in summary
        assert "pool_switch" in summary
        assert "custom_loads[X]" in summary

    def test_to_dict_serialisable(self):
        import json
        wiz = _wizard(
            custom_loads=[{"name": "Boiler", "switch": "switch.pool_pump"}],
        )
        report = check(wiz)
        d = report.to_dict()
        # Round-trip JSON: catches any non-serialisable types
        json.dumps(d)
        assert d["critical_count"] == 1
        assert d["ok"] is False
