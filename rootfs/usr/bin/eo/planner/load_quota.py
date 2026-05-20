"""Load quota tracking and debt-state classification.

Per-load configuration:
    target_hours_per_window    e.g. 1 h every 7 days for a winter pool pump
    window_days                e.g. 7 days
    min_runtime_minutes        e.g. 30 (shortest viable ON segment)
    daily_physical_max_hours   physical ceiling (e.g. pool pump can't run > 6h/day)
    required_confidence_pct    threshold on prob_surplus (SPEC §1.2 S4)
    allow_peak_on_critical     escape hatch for emergencies (default false)

Debt classification (SPEC §1.4 P6 + Gemini R1 "irreachable"):
    ok          accumulated ≥ target → no action needed
    low         1 day under daily rate
    medium      ≥ window_days/2 days under daily rate
    high        > window_days/2 days under daily rate
    critical    days_left ≤ 1 AND remaining > 0
    irreachable remaining > daily_physical_max × days_left → quota physically impossible

Bug fixes (SPEC §1.10):
    B1 — if the user shrinks window_days mid-flight, we truncate the
         in-memory history with ``hours_on_per_day_last_N[-window_days:]``
         instead of summing stale entries.
    B2 — if days_with_telemetry < window_days (add-on was down), we scale
         the target by ``days_with_telemetry / window_days``. Otherwise the
         system invents "phantom debt" because zero-telemetry days look like
         zero-execution days.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Sequence


class DebtState(str, enum.Enum):
    OK = "ok"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    IRREACHABLE = "irreachable"


@dataclass(frozen=True)
class LoadQuotaConfig:
    target_hours_per_window: float
    window_days: int
    min_runtime_minutes: float = 30.0
    daily_physical_max_hours: float = 24.0
    required_confidence_pct: float = 60.0
    allow_peak_on_critical: bool = False

    def __post_init__(self):
        if self.target_hours_per_window < 0:
            raise ValueError("target_hours_per_window must be ≥ 0")
        if self.window_days < 1:
            raise ValueError("window_days must be ≥ 1")
        if self.min_runtime_minutes <= 0:
            raise ValueError("min_runtime_minutes must be > 0")
        if not 0 < self.daily_physical_max_hours <= 24:
            raise ValueError("daily_physical_max_hours must be in (0, 24]")
        if not 0 <= self.required_confidence_pct <= 100:
            raise ValueError("required_confidence_pct must be in [0, 100]")


@dataclass(frozen=True)
class LoadQuotaState:
    """Computed quota status for one load at one point in time."""
    debt_state: DebtState
    accumulated_h: float          # hours executed within the window so far
    target_scaled_h: float        # target after data-quality scaling
    remaining_h: float            # max(0, target_scaled - accumulated)
    remaining_executable_h: float # ceiling under daily_physical_max
    days_left_in_window: int
    days_with_telemetry: int      # echoed from input for transparency
    data_quality_factor: float    # in [0, 1]

    def to_dict(self) -> dict:
        return {
            "debt_state": self.debt_state.value,
            "accumulated_h": round(self.accumulated_h, 3),
            "target_scaled_h": round(self.target_scaled_h, 3),
            "remaining_h": round(self.remaining_h, 3),
            "remaining_executable_h": round(self.remaining_executable_h, 3),
            "days_left_in_window": self.days_left_in_window,
            "days_with_telemetry": self.days_with_telemetry,
            "data_quality_factor": round(self.data_quality_factor, 3),
        }


def compute_debt_state(
    config: LoadQuotaConfig,
    hours_on_per_day_last_N: Sequence[float],
    days_passed_in_window: int,
    days_with_telemetry: int | None = None,
) -> LoadQuotaState:
    """Classify the debt state of one load.

    Parameters
    ----------
    config
        Static load quota config.
    hours_on_per_day_last_N
        Per-day execution history, oldest first. Length may exceed
        ``window_days`` (we slice the tail — fix B1).
    days_passed_in_window
        How many full days have elapsed since the current window started.
        Range [0, window_days].
    days_with_telemetry
        Of the ``window_days`` days, how many were genuinely observed (i.e.
        the add-on was up and recording). If None, assumed equal to
        ``window_days``. Used for the data-quality scale (fix B2).
    """
    if days_passed_in_window < 0:
        raise ValueError("days_passed_in_window must be ≥ 0")
    if days_passed_in_window > config.window_days:
        # Clamp rather than raise — a stale state object should not crash
        # the planner. Just treat as end-of-window.
        days_passed_in_window = config.window_days

    # B1: truncate to the configured window. Older entries are stale config.
    truncated = list(hours_on_per_day_last_N[-config.window_days:])
    accumulated = sum(max(0.0, h) for h in truncated)

    # B2: scale target by data-quality factor.
    if days_with_telemetry is None:
        days_with_telemetry = config.window_days
    days_with_telemetry = max(0, min(config.window_days, days_with_telemetry))
    if config.window_days > 0:
        data_quality_factor = days_with_telemetry / config.window_days
    else:
        data_quality_factor = 1.0
    target_scaled = config.target_hours_per_window * data_quality_factor

    days_left = max(0, config.window_days - days_passed_in_window)
    remaining = max(0.0, target_scaled - accumulated)
    remaining_executable = days_left * config.daily_physical_max_hours

    # ── Classify ────────────────────────────────────────────────────────
    if remaining <= 1e-9:
        state = DebtState.OK
    elif remaining > remaining_executable + 1e-9:
        state = DebtState.IRREACHABLE
    elif days_left <= 1:
        state = DebtState.CRITICAL
    else:
        # Compute days_under_target_rate: how many full daily-rate units
        # of the executed window have we missed?
        if config.window_days > 0:
            daily_rate = target_scaled / config.window_days
        else:
            daily_rate = 0.0
        days_elapsed = days_passed_in_window
        expected_so_far = daily_rate * days_elapsed
        deficit = max(0.0, expected_so_far - accumulated)
        days_under = deficit / daily_rate if daily_rate > 0 else 0.0

        threshold = config.window_days / 2
        if days_under <= 1:
            state = DebtState.LOW
        elif days_under <= threshold:
            state = DebtState.MEDIUM
        else:
            state = DebtState.HIGH

    return LoadQuotaState(
        debt_state=state,
        accumulated_h=accumulated,
        target_scaled_h=target_scaled,
        remaining_h=remaining,
        remaining_executable_h=remaining_executable,
        days_left_in_window=days_left,
        days_with_telemetry=days_with_telemetry,
        data_quality_factor=data_quality_factor,
    )
