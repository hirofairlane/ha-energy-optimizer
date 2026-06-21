"""Helpers to refine hourly forecasts into 15-min slots.

The forecasters emit per-hour kWh (4 slots worth of energy per hour). The
simulator needs per-slot kWh, so we evenly split each hour into 4 slots by
default — equal-quartile distribution.

Why not piecewise-linear? Hour-to-hour PV ramps are smooth on clear days,
but on overcast days the per-hour kWh is already an averaged quantity.
Splitting equally avoids over-fitting to forecast structure that may not
exist; the planner's 15-min slot is a control resolution, not a forecast
resolution improvement.

The functions here are pure and operate on plain Python lists/tuples — no
pandas / numpy dependency so they can be reused in lightweight contexts
and tested without ML pulled in.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

SLOTS_PER_HOUR: int = 4
SLOT_MINUTES: int = 60 // SLOTS_PER_HOUR  # 15 min


def split_hour_into_slots(
    hour_start: datetime,
    hourly_kwh: float,
    slots_per_hour: int = SLOTS_PER_HOUR,
) -> list[tuple[datetime, float]]:
    """Split one hour's kWh into N equal slots.

    Returns a list of ``(slot_start, slot_kwh)`` tuples.
    """
    if slots_per_hour < 1:
        raise ValueError("slots_per_hour must be ≥ 1")
    if hour_start.tzinfo is None:
        raise ValueError("hour_start must be tz-aware")
    per_slot = hourly_kwh / slots_per_hour
    step = timedelta(minutes=60 / slots_per_hour)
    return [
        (hour_start + step * i, per_slot)
        for i in range(slots_per_hour)
    ]


def interpolate_hourly_to_slots(
    hourly_series: Sequence[tuple[datetime, float]],
    slots_per_hour: int = SLOTS_PER_HOUR,
) -> list[tuple[datetime, float]]:
    """Refine a series of hourly ``(hour_start, kwh)`` into per-slot kWh.

    Hours are assumed to be contiguous and ascending. The function does not
    fill gaps — if your input has hour 8 then hour 10, the output skips hour 9.
    """
    out: list[tuple[datetime, float]] = []
    for hour_start, kwh in hourly_series:
        out.extend(split_hour_into_slots(hour_start, kwh, slots_per_hour))
    return out
