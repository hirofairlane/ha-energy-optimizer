"""Shared types for the policy pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from eo.planner.iterative import Plan


@dataclass(frozen=True)
class PolicyOverride:
    """A single rewrite from raw_plan → policy_adjusted_plan.

    The list of overrides emitted by the pipeline is what populates the
    transparency layer's "Planner wanted X. Policy applied Y because Z."
    pattern (SPEC §1.8 T3).
    """
    layer: str                       # "capacity_budget" / "peak_prohibition" / ...
    load: str
    slot_index: int
    original_action: str
    new_action: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "load": self.load,
            "slot_index": self.slot_index,
            "original_action": self.original_action,
            "new_action": self.new_action,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PolicyPipelineResult:
    adjusted_plan: Plan
    overrides: tuple[PolicyOverride, ...] = ()

    def overrides_by_layer(self) -> dict[str, list[PolicyOverride]]:
        out: dict[str, list[PolicyOverride]] = {}
        for o in self.overrides:
            out.setdefault(o.layer, []).append(o)
        return out
