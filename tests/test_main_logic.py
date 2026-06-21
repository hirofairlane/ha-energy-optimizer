"""Regression / smoke tests for the pure decision logic in the v4 entrypoint
(`rootfs/usr/bin/energy_optimizer.py`).

The `eo/` package (v5 engine) already has its own extensive suite; this file
covers the still-shipping v4 reactive engine's derived-value and decision
functions — the ones reached every cycle when `v5_engine_enabled` is false
(the default). They are exercised without MQTT/HA/InfluxDB: the module is
imported with `ENERGY_OPT_TESTING=1` so it has no import-time side effects
(no `/data` mkdir), and `ENERGY_OPT_DATA` points at a temp dir.

Run: `pytest tests/test_main_logic.py`
"""
from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest

_SRC = (
    pathlib.Path(__file__).parent.parent
    / "rootfs" / "usr" / "bin" / "energy_optimizer.py"
)


@pytest.fixture(scope="session")
def mod(tmp_path_factory):
    os.environ["ENERGY_OPT_TESTING"] = "1"
    os.environ["ENERGY_OPT_DATA"] = str(tmp_path_factory.mktemp("data"))
    spec = importlib.util.spec_from_file_location("energy_optimizer_app", _SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── import-time hygiene ──────────────────────────────────────────────────────

def test_module_imports_without_side_effects(mod):
    # The whole point: importing the shipped entrypoint must not require /data
    # to be writable nor talk to HA. If we got here the import succeeded.
    assert mod.ADD_ON_VERSION
    assert callable(mod.calculate_optimal_soc)


def test_version_matches_config_yaml(mod):
    import yaml
    cfg = yaml.safe_load((pathlib.Path(_SRC).parents[3] / "config.yaml").read_text())
    assert mod.ADD_ON_VERSION == cfg["version"]


# ── tariff period classification ─────────────────────────────────────────────

def test_current_tariff_periods(mod, monkeypatch):
    from datetime import datetime

    class _DT(datetime):
        _now = datetime(2026, 6, 17, 11, 0, 0)  # Wednesday 11:00 -> peak

        @classmethod
        def now(cls, tz=None):
            return cls._now

    monkeypatch.setattr(mod, "datetime", _DT)

    _DT._now = datetime(2026, 6, 17, 11, 0, 0)   # Wed 11h -> peak
    assert mod.current_tariff()["period"] == "peak"

    _DT._now = datetime(2026, 6, 17, 3, 0, 0)    # Wed 03h -> valley
    assert mod.current_tariff()["period"] == "valley"

    _DT._now = datetime(2026, 6, 17, 16, 0, 0)   # Wed 16h -> mid
    assert mod.current_tariff()["period"] == "mid"

    _DT._now = datetime(2026, 6, 20, 11, 0, 0)   # Saturday -> valley all day
    assert mod.current_tariff()["period"] == "valley"


# ── entity auto-discovery scoring ────────────────────────────────────────────

def test_score_entity_domain_gate(mod):
    # A switch can never score for a sensor-only role.
    st = {"entity_id": "switch.battery_soc",
          "attributes": {"friendly_name": "battery soc"}}
    assert mod._score_entity(st, "battery_soc") == 0


def test_score_entity_device_class_unit_and_keywords(mod):
    st = {
        "entity_id": "sensor.battery_state_of_capacity",
        "attributes": {
            "friendly_name": "Battery SOC",
            "device_class": "battery",
            "unit_of_measurement": "%",
        },
    }
    # device_class(+50) + unit(+25/40) + keyword hits -> clearly positive,
    # and higher than a same-domain sensor with no matching hints.
    score = mod._score_entity(st, "battery_soc")
    other = mod._score_entity(
        {"entity_id": "sensor.kitchen_humidity",
         "attributes": {"friendly_name": "kitchen humidity"}},
        "battery_soc",
    )
    assert score > other
    assert score >= 50


def test_get_entity_candidates_ranks_best_first(mod):
    states = [
        {"entity_id": "sensor.kitchen_humidity",
         "attributes": {"friendly_name": "Kitchen humidity"}},
        {"entity_id": "sensor.battery_state_of_capacity",
         "attributes": {"friendly_name": "Battery SOC",
                        "device_class": "battery",
                        "unit_of_measurement": "%"}},
    ]
    cands = mod.get_entity_candidates("battery_soc", states)
    assert cands[0]["entity_id"] == "sensor.battery_state_of_capacity"


# ── solar geometry proxy (pure) ──────────────────────────────────────────────

def test_solar_proxy_is_zero_at_night(mod):
    assert mod._solar_proxy_for_hour(2, 6) == 0.0   # 02:00 -> sun below horizon


def test_solar_proxy_peaks_at_summer_noon(mod):
    noon_summer = mod._solar_proxy_for_hour(13, 6)
    noon_winter = mod._solar_proxy_for_hour(13, 12)
    assert 0.0 < noon_winter < noon_summer <= 1.0


# ── storm detection ──────────────────────────────────────────────────────────

def test_is_storm_forecast_current_condition(mod, monkeypatch):
    monkeypatch.setattr(mod, "cfg", lambda *a, **k: "weather.aemet")
    monkeypatch.setattr(mod, "ha_state",
                        lambda e: {"state": "lightning", "attributes": {}})
    assert mod.is_storm_forecast() is True


def test_is_storm_forecast_from_future_entry(mod, monkeypatch):
    monkeypatch.setattr(mod, "cfg", lambda *a, **k: "weather.aemet")
    monkeypatch.setattr(mod, "ha_state", lambda e: {
        "state": "sunny",
        "attributes": {"forecast": [{"condition": "sunny"},
                                    {"condition": "pouring"}]},
    })
    assert mod.is_storm_forecast() is True


def test_is_storm_forecast_clear(mod, monkeypatch):
    monkeypatch.setattr(mod, "cfg", lambda *a, **k: "weather.aemet")
    monkeypatch.setattr(mod, "ha_state", lambda e: {
        "state": "sunny", "attributes": {"forecast": [{"condition": "cloudy"}]},
    })
    assert mod.is_storm_forecast() is False


# ── optimal-SOC valley charging target (issue #5 / #8 regressions) ───────────

def _patch_soc_inputs(mod, monkeypatch, *, base_kw=0.4, correction=1.0,
                      health=("bill_reducer",)):
    """Make calculate_optimal_soc deterministic: no InfluxDB, no history."""
    monkeypatch.setattr(mod, "_compute_solar_correction_factor", lambda: correction)
    monkeypatch.setattr(mod, "_get_avg_night_consumption_kw", lambda: base_kw)
    monkeypatch.setattr(mod, "_solar_historical_baseline", lambda d: None)
    monkeypatch.setattr(mod, "_daylight_hours_today", lambda: None)

    def _cfg(key, default=None):
        return {"battery_capacity_kwh": 10.0,
                "battery_health_mode": health[0]}.get(key, default)
    monkeypatch.setattr(mod, "cfg", _cfg)


def test_optimal_soc_scales_with_peak_hours(mod, monkeypatch):
    # Issue #5: a 16h-peak tariff must demand more stored energy than the
    # Spanish 8h-peak default for the same base load and no solar.
    _patch_soc_inputs(mod, monkeypatch)
    sensors = {"solar_tomorrow": 0.0, "temp_outdoor": 15.0}

    spain = mod.calculate_optimal_soc(sensors, {"peak_hours": list(range(8))})
    ukraine = mod.calculate_optimal_soc(sensors, {"peak_hours": list(range(16))})

    assert ukraine["peak_h_count"] == 16
    assert spain["peak_h_count"] == 8
    assert ukraine["target_soc"] > spain["target_soc"]


def test_optimal_soc_respects_health_mode_floor(mod, monkeypatch):
    # Issue #8: with bill_reducer the floor is 10 %, not a hardcoded 30 %.
    _patch_soc_inputs(mod, monkeypatch, health=("bill_reducer",))
    # Lots of solar, tiny base load -> battery_needed ~0 -> clamps to floor.
    sensors = {"solar_tomorrow": 50.0, "temp_outdoor": 20.0}
    res = mod.calculate_optimal_soc(sensors, {"peak_hours": list(range(8))})
    assert res["health_mode_min"] == 10
    assert res["target_soc"] == 10


def test_optimal_soc_cold_weather_raises_target(mod, monkeypatch):
    _patch_soc_inputs(mod, monkeypatch)
    warm = mod.calculate_optimal_soc(
        {"solar_tomorrow": 0.0, "temp_outdoor": 20.0},
        {"peak_hours": list(range(8))},
    )
    cold = mod.calculate_optimal_soc(
        {"solar_tomorrow": 0.0, "temp_outdoor": 0.0},
        {"peak_hours": list(range(8))},
    )
    assert cold["temp_adj_kwh"] > warm["temp_adj_kwh"]
    assert cold["target_soc"] >= warm["target_soc"]
