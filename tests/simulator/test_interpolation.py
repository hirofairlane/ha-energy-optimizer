"""Tests for hourly → slot interpolation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eo.simulator.interpolation import (
    SLOT_MINUTES,
    SLOTS_PER_HOUR,
    interpolate_hourly_to_slots,
    split_hour_into_slots,
)


def _utc(h: int = 0) -> datetime:
    return datetime(2026, 5, 20, h, tzinfo=timezone.utc)


class TestSplitHour:
    def test_default_4_slots_per_hour(self):
        slots = split_hour_into_slots(_utc(10), 2.0)
        assert len(slots) == 4
        # Each slot is 15 min apart.
        for (t, _), expected_min in zip(slots, [0, 15, 30, 45]):
            assert t.minute == expected_min
        # Each slot contains 0.5 kWh
        for _, kwh in slots:
            assert kwh == pytest.approx(0.5)

    def test_zero_kwh_input_yields_zeros(self):
        slots = split_hour_into_slots(_utc(2), 0.0)
        assert all(kwh == 0 for _, kwh in slots)

    def test_custom_slots_per_hour(self):
        slots = split_hour_into_slots(_utc(0), 6.0, slots_per_hour=6)
        assert len(slots) == 6
        for _, kwh in slots:
            assert kwh == pytest.approx(1.0)

    def test_naive_ts_raises(self):
        with pytest.raises(ValueError):
            split_hour_into_slots(datetime(2026, 5, 20, 12), 1.0)

    def test_invalid_slots_per_hour_raises(self):
        with pytest.raises(ValueError):
            split_hour_into_slots(_utc(10), 1.0, slots_per_hour=0)


class TestInterpolateSeries:
    def test_two_consecutive_hours(self):
        series = [(_utc(10), 4.0), (_utc(11), 2.0)]
        out = interpolate_hourly_to_slots(series)
        # 2 hours × 4 slots = 8 entries
        assert len(out) == 8
        # First 4 slots are at hour 10, each 1.0 kWh
        assert all(t.hour == 10 for t, _ in out[:4])
        assert all(kwh == 1.0 for _, kwh in out[:4])
        # Next 4 at hour 11, each 0.5 kWh
        assert all(t.hour == 11 for t, _ in out[4:])
        assert all(kwh == 0.5 for _, kwh in out[4:])

    def test_empty_input_yields_empty(self):
        assert interpolate_hourly_to_slots([]) == []


class TestConstants:
    def test_constants_consistent(self):
        assert SLOTS_PER_HOUR == 4
        assert SLOT_MINUTES == 15
