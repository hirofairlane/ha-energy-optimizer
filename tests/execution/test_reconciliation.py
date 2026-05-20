"""Tests for the reconciliation helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from eo.execution.reconciliation import (
    reconcile_load_debt,
    reconcile_world_state,
)


def _utc() -> datetime:
    return datetime(2026, 5, 20, 12, tzinfo=timezone.utc)


# ── reconcile_world_state ──────────────────────────────────────────────────
class TestWorldState:
    def test_naive_now_raises(self):
        with pytest.raises(ValueError):
            reconcile_world_state([], lambda _e: False, datetime(2026, 5, 20))

    def test_empty_returns_empty(self):
        w = reconcile_world_state([], lambda _e: False, _utc())
        assert w.loads_on == ()
        assert w.last_reconciled_at == _utc()

    def test_loads_on_filtered(self):
        states = {"switch.a": True, "switch.b": False, "switch.c": True}
        w = reconcile_world_state(
            [("a", "switch.a"), ("b", "switch.b"), ("c", "switch.c")],
            is_on=lambda e: states.get(e, False),
            now=_utc(),
        )
        assert w.loads_on == ("a", "c")

    def test_callback_exception_treated_as_off(self):
        def is_on(_e):
            raise RuntimeError("entity gone")

        w = reconcile_world_state(
            [("a", "switch.a")], is_on=is_on, now=_utc(),
        )
        assert w.loads_on == ()


# ── reconcile_load_debt ────────────────────────────────────────────────────
class TestLoadDebt:
    def test_rebuilds_history_per_load(self):
        history = {
            "a": [1.0, 0.5, 0.0, 0.0, 1.5, 2.0, 1.0],
            "b": [0.5] * 7,
        }
        out = reconcile_load_debt(
            ["a", "b", "c_unknown"],
            window_days=7,
            fetch_hours_on_per_day=lambda name, days: history.get(name, []),
        )
        assert out["a"] == history["a"]
        assert out["b"] == history["b"]
        assert out["c_unknown"] == []

    def test_clips_to_window(self):
        out = reconcile_load_debt(
            ["a"],
            window_days=3,
            fetch_hours_on_per_day=lambda _n, _d: [1, 2, 3, 4, 5, 6, 7],
        )
        assert out["a"] == [5, 6, 7]

    def test_negative_entries_clamped_to_zero(self):
        out = reconcile_load_debt(
            ["a"],
            window_days=5,
            fetch_hours_on_per_day=lambda _n, _d: [-1, 1.5, -0.5, 2, 0],
        )
        assert out["a"] == [0.0, 1.5, 0.0, 2.0, 0.0]

    def test_callback_exception_yields_empty(self):
        def bad(name, days):
            raise RuntimeError("DB down")

        out = reconcile_load_debt(["a"], window_days=7, fetch_hours_on_per_day=bad)
        assert out["a"] == []

    def test_zero_window_raises(self):
        with pytest.raises(ValueError):
            reconcile_load_debt(["a"], window_days=0,
                                fetch_hours_on_per_day=lambda _n, _d: [])
