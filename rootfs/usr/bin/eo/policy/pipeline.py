"""Policy pipeline orchestrator.

Threads the four policy layers in their canonical order
(SPEC §1.5 POL1) and aggregates overrides for the transparency layer:

    capacity_budget → peak_prohibition → antiflap → degraded_mode

Each layer takes the plan emitted by the previous one and produces a new
plan plus a list of overrides. No layer mutates state in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from eo.planner.iterative import Plan
from eo.policy.antiflap import (
    AntiflapConfig,
    AntiflapState,
    apply_antiflap,
)
from eo.policy.capacity_budget import apply_capacity_budget
from eo.policy.degraded_mode import (
    DegradedLevel,
    DegradedModeConfig,
    DegradedModeInputs,
    apply_degraded_mode,
)
from eo.policy.peak_prohibition import apply_peak_prohibition
from eo.policy.types import PolicyOverride


@dataclass(frozen=True)
class PolicyPipelineInputs:
    """Static inputs the pipeline needs from the orchestrator."""
    load_watts: Mapping[str, float]
    available_w: float
    slot_periods: Mapping[int, str]
    antiflap_state: dict[str, AntiflapState]
    now: datetime
    degraded_inputs: DegradedModeInputs


@dataclass(frozen=True)
class PolicyPipelineSummary:
    final_plan: Plan
    overrides: tuple[PolicyOverride, ...]
    degraded_level: DegradedLevel

    def to_dict(self) -> dict:
        return {
            "cells_count": len(self.final_plan.cells),
            "override_count": len(self.overrides),
            "degraded_level": int(self.degraded_level),
            "overrides_by_layer": {
                layer: len(overrides)
                for layer, overrides in _group(self.overrides).items()
            },
        }


def _group(overrides):
    out: dict[str, list[PolicyOverride]] = {}
    for o in overrides:
        out.setdefault(o.layer, []).append(o)
    return out


def run_policy_pipeline(
    raw_plan: Plan,
    inputs: PolicyPipelineInputs,
    antiflap_config: AntiflapConfig = AntiflapConfig(),
    degraded_config: DegradedModeConfig = DegradedModeConfig(),
) -> PolicyPipelineSummary:
    overrides: list[PolicyOverride] = []

    step1 = apply_capacity_budget(
        raw_plan, inputs.load_watts, inputs.available_w,
    )
    overrides.extend(step1.overrides)

    step2 = apply_peak_prohibition(step1.adjusted_plan, inputs.slot_periods)
    overrides.extend(step2.overrides)

    step3 = apply_antiflap(
        step2.adjusted_plan, inputs.antiflap_state,
        now=inputs.now, config=antiflap_config,
    )
    overrides.extend(step3.overrides)

    step4_result, degraded_level = apply_degraded_mode(
        step3.adjusted_plan, inputs.degraded_inputs, degraded_config,
    )
    overrides.extend(step4_result.overrides)

    return PolicyPipelineSummary(
        final_plan=step4_result.adjusted_plan,
        overrides=tuple(overrides),
        degraded_level=degraded_level,
    )
