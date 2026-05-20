"""Scenario builder — debt-aware quantile collapse and slot interpolation."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from eo.planner.load_quota import DebtState
from eo.simulator.interpolation import (
    SLOTS_PER_HOUR,
    interpolate_hourly_to_slots,
)


# ── Risk tolerance and debt mapping ─────────────────────────────────────────
class RiskTolerance(str, enum.Enum):
    OPTIMISTIC = "optimistic"      # solar P90, house P10
    MEDIAN = "median"              # solar P50, house P50
    CONSERVATIVE = "conservative"  # solar P10, house P90
    STRESS = "stress"              # like CONSERVATIVE but for alerting only


# SPEC §1.2 S4 — canonical mapping from debt to risk.
_DEBT_TO_RISK: dict[DebtState, RiskTolerance] = {
    DebtState.OK:          RiskTolerance.MEDIAN,
    DebtState.LOW:         RiskTolerance.MEDIAN,
    DebtState.MEDIUM:      RiskTolerance.MEDIAN,
    DebtState.HIGH:        RiskTolerance.CONSERVATIVE,
    DebtState.CRITICAL:    RiskTolerance.CONSERVATIVE,
    DebtState.IRREACHABLE: RiskTolerance.CONSERVATIVE,
}


def risk_from_debt_state(debt: DebtState) -> RiskTolerance:
    """Return the canonical risk tolerance for a debt state.

    SPEC §1.2 S4:
        low / medium → median
        high / critical / irreachable → conservative
        stress is reserved for alerting / infeasibility — never normal
        planning.
    """
    return _DEBT_TO_RISK.get(debt, RiskTolerance.MEDIAN)


# ── Input quantile forecast ─────────────────────────────────────────────────
@dataclass(frozen=True)
class QuantileHourForecast:
    """Generic per-hour quantile prediction.

    The Solar/House forecasters emit slightly different shapes (the solar
    one ships per-hour ``HourForecast``s with clear_sky_kwh; house only
    has ``predict()`` returning numpy arrays). The pipeline wires both into
    this common shape before passing them to :func:`build_scenario`.
    """
    hour_start: datetime
    p10: float
    p50: float
    p90: float


# ── Scenario output ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Scenario:
    slot_starts: tuple[datetime, ...]
    solar_kwh: tuple[float, ...]
    house_kwh: tuple[float, ...]
    confidence_is_heuristic: bool
    risk_tolerance: RiskTolerance
    debt_state: DebtState | None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not (len(self.slot_starts) == len(self.solar_kwh) == len(self.house_kwh)):
            raise ValueError("slot_starts, solar_kwh and house_kwh must be same length")

    def to_dict(self) -> dict:
        return {
            "risk_tolerance": self.risk_tolerance.value,
            "debt_state": self.debt_state.value if self.debt_state else None,
            "confidence_is_heuristic": self.confidence_is_heuristic,
            "slots": len(self.slot_starts),
            "metadata": dict(self.metadata),
            "solar_kwh_total": round(sum(self.solar_kwh), 3),
            "house_kwh_total": round(sum(self.house_kwh), 3),
        }


# ── Builder ─────────────────────────────────────────────────────────────────
def _pick_solar_quantile(qh: QuantileHourForecast, risk: RiskTolerance) -> float:
    if risk == RiskTolerance.OPTIMISTIC:
        return qh.p90
    if risk == RiskTolerance.CONSERVATIVE or risk == RiskTolerance.STRESS:
        return qh.p10
    return qh.p50


def _pick_house_quantile(qh: QuantileHourForecast, risk: RiskTolerance) -> float:
    if risk == RiskTolerance.OPTIMISTIC:
        return qh.p10
    if risk == RiskTolerance.CONSERVATIVE or risk == RiskTolerance.STRESS:
        return qh.p90
    return qh.p50


def build_scenario(
    solar_forecasts: Sequence[QuantileHourForecast],
    house_forecasts: Sequence[QuantileHourForecast],
    risk_tolerance: RiskTolerance,
    debt_state: DebtState | None = None,
    slots_per_hour: int = SLOTS_PER_HOUR,
    confidence_is_heuristic: bool = True,
) -> Scenario:
    """Collapse per-hour quantiles into a per-slot Scenario.

    Both inputs must share the same set of ``hour_start`` timestamps and
    be sorted ascending. The output series has ``slots_per_hour ×
    len(forecasts)`` entries.
    """
    if not solar_forecasts:
        return Scenario(
            slot_starts=(), solar_kwh=(), house_kwh=(),
            confidence_is_heuristic=confidence_is_heuristic,
            risk_tolerance=risk_tolerance, debt_state=debt_state,
        )
    if len(solar_forecasts) != len(house_forecasts):
        raise ValueError(
            "solar_forecasts and house_forecasts must have the same length"
        )

    for sf, hf in zip(solar_forecasts, house_forecasts):
        if sf.hour_start != hf.hour_start:
            raise ValueError(
                "solar/house hour_start mismatch — caller must align"
            )

    solar_hourly = [
        (qh.hour_start, _pick_solar_quantile(qh, risk_tolerance))
        for qh in solar_forecasts
    ]
    house_hourly = [
        (qh.hour_start, _pick_house_quantile(qh, risk_tolerance))
        for qh in house_forecasts
    ]

    solar_slots = interpolate_hourly_to_slots(solar_hourly, slots_per_hour)
    house_slots = interpolate_hourly_to_slots(house_hourly, slots_per_hour)

    # Both interpolations share the same time axis — assert equality.
    assert [t for t, _ in solar_slots] == [t for t, _ in house_slots]

    return Scenario(
        slot_starts=tuple(t for t, _ in solar_slots),
        solar_kwh=tuple(v for _, v in solar_slots),
        house_kwh=tuple(v for _, v in house_slots),
        confidence_is_heuristic=confidence_is_heuristic,
        risk_tolerance=risk_tolerance,
        debt_state=debt_state,
        metadata={"slots_per_hour": slots_per_hour},
    )
