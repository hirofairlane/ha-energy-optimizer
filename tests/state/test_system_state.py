"""Tests for SystemState dataclass + persistence."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from eo.policy.antiflap import AntiflapState
from eo.state.system_state import (
    BatteryState,
    ExecutionWorldState,
    SystemState,
)


def _utc(min_offset: int = 0) -> datetime:
    return datetime(2026, 5, 20, 12, tzinfo=timezone.utc) + timedelta(minutes=min_offset)


def _state(**kw) -> SystemState:
    base = dict(
        cycle_ts=_utc(0),
        battery=BatteryState(soc_pct=55.0, power_w=-200, last_updated=_utc(-2)),
        forecast_quality={"solar.p50": {"mae": 0.4}},
        planner_hash_history=("abc", "def"),
        load_debt={"boiler": "low"},
        antiflap_state={
            "boiler": AntiflapState(load="boiler", last_action="off",
                                    last_change_ts=_utc(-60)),
        },
        execution_world_state=ExecutionWorldState(
            loads_on=("pool",), last_reconciled_at=_utc(-1),
        ),
    )
    base.update(kw)
    return SystemState(**base)


# ── Construction ───────────────────────────────────────────────────────────
class TestConstruction:
    def test_naive_cycle_ts_rejected(self):
        with pytest.raises(ValueError):
            SystemState(
                cycle_ts=datetime(2026, 5, 20, 12),
                battery=BatteryState(50, 0, None),
                forecast_quality={},
                planner_hash_history=(),
                load_debt={},
                antiflap_state={},
                execution_world_state=ExecutionWorldState(loads_on=(), last_reconciled_at=None),
            )


# ── Immutability + replacers ────────────────────────────────────────────────
class TestImmutable:
    def test_with_battery_returns_new_instance(self):
        s = _state()
        new = BatteryState(soc_pct=70, power_w=500, last_updated=_utc(0))
        s2 = s.with_battery(new)
        assert s2.battery is new
        assert s.battery.soc_pct == 55.0   # original unchanged

    def test_with_world_state(self):
        s = _state()
        w = ExecutionWorldState(loads_on=("boiler",), last_reconciled_at=_utc(0))
        s2 = s.with_world_state(w)
        assert s2.execution_world_state is w
        assert s.execution_world_state.loads_on == ("pool",)

    def test_with_planner_history(self):
        s = _state()
        s2 = s.with_planner_history(("x", "y", "z"))
        assert s2.planner_hash_history == ("x", "y", "z")
        assert s.planner_hash_history == ("abc", "def")


# ── Serialisation round-trip ───────────────────────────────────────────────
class TestSerialisation:
    def test_to_dict_then_from_dict_round_trips(self):
        s = _state()
        d = s.to_dict()
        # Make sure it survives JSON.
        re = SystemState.from_dict(json.loads(json.dumps(d)))
        assert re.battery.soc_pct == s.battery.soc_pct
        assert re.battery.power_w == s.battery.power_w
        assert re.planner_hash_history == s.planner_hash_history
        assert re.load_debt == s.load_debt
        # antiflap_state preserved
        assert set(re.antiflap_state.keys()) == set(s.antiflap_state.keys())
        for name in s.antiflap_state:
            assert re.antiflap_state[name].last_action == s.antiflap_state[name].last_action

    def test_battery_state_serialisation(self):
        b = BatteryState(soc_pct=50, power_w=100, last_updated=_utc(0))
        d = b.to_dict()
        re = BatteryState.from_dict(d)
        assert re == b

    def test_battery_state_none_last_updated(self):
        b = BatteryState(soc_pct=50, power_w=0, last_updated=None)
        d = b.to_dict()
        assert d["last_updated"] is None
        re = BatteryState.from_dict(d)
        assert re.last_updated is None

    def test_execution_world_state_is_on(self):
        w = ExecutionWorldState(loads_on=("boiler", "pool"), last_reconciled_at=None)
        assert w.is_on("boiler") is True
        assert w.is_on("dishwasher") is False

    def test_from_dict_normalises_naive_iso(self):
        d = _state().to_dict()
        # Strip the tz suffix from cycle_ts to simulate a stale file.
        d["cycle_ts"] = "2026-05-20T12:00:00"
        re = SystemState.from_dict(d)
        assert re.cycle_ts.tzinfo is not None


# ── Persistence ────────────────────────────────────────────────────────────
class TestPersistence:
    def test_save_load_round_trip(self, tmp_path: Path):
        s = _state()
        path = tmp_path / "state.json"
        s.save(path)
        re = SystemState.load(path)
        assert re is not None
        assert re.battery.soc_pct == s.battery.soc_pct
        assert re.planner_hash_history == s.planner_hash_history

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        assert SystemState.load(tmp_path / "missing.json") is None

    def test_atomic_write_no_tmp_leftovers(self, tmp_path: Path):
        s = _state()
        path = tmp_path / "state.json"
        s.save(path)
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_save_creates_parent_dir(self, tmp_path: Path):
        s = _state()
        deep_path = tmp_path / "deep" / "dirs" / "state.json"
        s.save(deep_path)
        assert deep_path.exists()
