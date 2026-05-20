"""Three-level degraded-mode policy (SPEC §1.5 POL6).

Inputs:
    forecast_mae         current forecast quality (typically solar MAE_24h)
    aemet_age_hours      how long since the last fresh AEMET observation
    sensor_age_max_min   most stale of the planner's critical sensors
    ...

Levels (escalating):
    Nivel 1  forecast_mae crosses threshold
        → mark loads that were ``min_runtime_only`` (matrix rule 8 / 11) as
          OFF; keep the more confident decisions (rule 7 still fires).
        → tag the cycle with mode=degraded_1 for downstream logging.

    Nivel 2  AEMET stale > 24h
        → force surplus_expected = False globally (handled by the
          ScenarioBuilder upstream) and OFF every deferred load that
          relied on tomorrow's surplus expectation (decisions with
          rule_id in {6, 10} that would have been ON were already OFF,
          but rule_id in {7, 8, 11} were ON despite the risk — keep them
          but degrade min_runtime ones).

    Nivel 3  sensor_age_max_min > 30 min
        → all deferred loads OFF, no exceptions. The system is blind;
          we don't gamble.

Each level subsumes the lower one — passing level 3 thresholds implies
level 1 and 2 actions too. The function returns the most severe level
triggered.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace

from eo.planner.decision_matrix import LoadDecision
from eo.planner.iterative import Plan, PlanCell

from eo.policy.types import PolicyOverride, PolicyPipelineResult


LAYER_NAME = "degraded_mode"


class DegradedLevel(int, enum.Enum):
    NORMAL = 0
    L1_FORECAST_DEGRADED = 1
    L2_AEMET_STALE = 2
    L3_SENSORS_STALE = 3


@dataclass(frozen=True)
class DegradedModeConfig:
    forecast_mae_threshold: float = 0.5          # kWh — beyond this, forecast is unreliable
    aemet_stale_hours_threshold: float = 24.0
    sensor_stale_minutes_threshold: float = 30.0

    def __post_init__(self):
        if self.forecast_mae_threshold <= 0:
            raise ValueError("forecast_mae_threshold must be > 0")
        if self.aemet_stale_hours_threshold <= 0:
            raise ValueError("aemet_stale_hours_threshold must be > 0")
        if self.sensor_stale_minutes_threshold <= 0:
            raise ValueError("sensor_stale_minutes_threshold must be > 0")


@dataclass(frozen=True)
class DegradedModeInputs:
    forecast_mae: float | None = None
    aemet_age_hours: float | None = None
    sensor_age_max_minutes: float | None = None


def classify_degraded_level(
    inputs: DegradedModeInputs, config: DegradedModeConfig = DegradedModeConfig(),
) -> DegradedLevel:
    """Return the most severe level triggered by ``inputs``."""
    if (
        inputs.sensor_age_max_minutes is not None
        and inputs.sensor_age_max_minutes > config.sensor_stale_minutes_threshold
    ):
        return DegradedLevel.L3_SENSORS_STALE
    if (
        inputs.aemet_age_hours is not None
        and inputs.aemet_age_hours > config.aemet_stale_hours_threshold
    ):
        return DegradedLevel.L2_AEMET_STALE
    if (
        inputs.forecast_mae is not None
        and inputs.forecast_mae > config.forecast_mae_threshold
    ):
        return DegradedLevel.L1_FORECAST_DEGRADED
    return DegradedLevel.NORMAL


def apply_degraded_mode(
    plan: Plan,
    inputs: DegradedModeInputs,
    config: DegradedModeConfig = DegradedModeConfig(),
) -> tuple[PolicyPipelineResult, DegradedLevel]:
    """Apply degradation overrides and return ``(result, triggered_level)``."""
    level = classify_degraded_level(inputs, config)
    if level == DegradedLevel.NORMAL:
        return PolicyPipelineResult(adjusted_plan=plan, overrides=()), level

    new_cells: list[PlanCell] = []
    overrides: list[PolicyOverride] = []

    for cell in plan.cells:
        if cell.decision.action != "on":
            new_cells.append(cell)
            continue

        # Force OFF rules per level.
        should_off = False
        reason = ""

        if level == DegradedLevel.L3_SENSORS_STALE:
            # All deferred loads OFF regardless of debt.
            should_off = True
            reason = (
                "degraded_mode L3: sensor age exceeds threshold — "
                "auto_horizon disabled, deferred loads OFF"
            )
        elif level == DegradedLevel.L2_AEMET_STALE:
            # Stop running on min_runtime/optimistic decisions. Keep CRITICAL
            # forced ones (rule 3) — they are not predictions, they are
            # deadline-driven obligations.
            if cell.decision.rule_id != 3:
                should_off = True
                reason = (
                    "degraded_mode L2: AEMET stale > threshold — "
                    "surplus_expected forced to False"
                )
        elif level == DegradedLevel.L1_FORECAST_DEGRADED:
            # Only drop min_runtime_only decisions (rules 8 and 11). The
            # higher-confidence rules (2, 3, 7) stay.
            if cell.decision.min_runtime_only:
                should_off = True
                reason = (
                    "degraded_mode L1: forecast MAE above threshold — "
                    "min_runtime decisions retired"
                )

        if should_off:
            new_decision = LoadDecision(
                action="off",
                reason=reason,
                rule_id=cell.decision.rule_id,
                utility_score=cell.decision.utility_score,
                alert=cell.decision.alert,
                min_runtime_only=cell.decision.min_runtime_only,
            )
            new_cells.append(replace(cell, decision=new_decision))
            overrides.append(PolicyOverride(
                layer=LAYER_NAME, load=cell.load, slot_index=cell.slot_index,
                original_action="on", new_action="off", reason=reason,
            ))
        else:
            new_cells.append(cell)

    return (
        PolicyPipelineResult(
            adjusted_plan=Plan(cells=tuple(new_cells)),
            overrides=tuple(overrides),
        ),
        level,
    )
