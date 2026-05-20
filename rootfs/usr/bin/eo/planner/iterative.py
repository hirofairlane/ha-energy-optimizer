"""Iterative planner — converges on a stable per-load schedule.

The planner alternates:
    1. Compute a candidate plan: apply ``decide_load_for_slot()`` per load
       per slot, honoring ``forced_states`` (slot-0 policy mask).
    2. Simulate the resulting trajectory with ``simulate_soc`` to see how the
       candidate plan affects SOC, grid flow, and surplus signals.
    3. If the new plan equals the previous one (by hash) → converged. Else
       re-decide using updated signals from the simulation, up to
       ``max_iter`` times.

Convergence guard:
    * Hash compare against the previous plan → stop on equality.
    * 2-cycle detection: if hash[n] == hash[n−2] (oscillation) → stop.
    * Hard cap ``MAX_PLANNER_ITERATIONS = 5`` (SPEC §1.4 P3 + Gemini R1
      Test 1 "Monstruo del Bucle").

This module is pure: it does not touch the policy layer, the execution
engine, or any I/O. It takes deterministic inputs and emits a ``PlannerResult``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Sequence

from eo.planner.decision_matrix import (
    LoadDecision,
    SlotContext,
    decide_load_for_slot,
)
from eo.planner.load_quota import LoadQuotaConfig, LoadQuotaState


MAX_PLANNER_ITERATIONS: int = 5


# ── Inputs ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LoadInputs:
    """Static info + current quota state for one load."""
    name: str
    config: LoadQuotaConfig
    quota_state: LoadQuotaState


@dataclass(frozen=True)
class SlotMeta:
    """Per-slot metadata that does not depend on the load."""
    timestamp: datetime
    period: str
    # Optional summaries from the scenario, reused by all loads in this slot.
    prob_surplus_tomorrow: float = 0.0
    prob_surplus_next_valley: float = 0.0


# Type alias for the per-load × per-slot context builder. The planner calls
# this once per iteration to build SlotContext objects. This is the seam where
# the iteration loop incorporates ``forced_states`` and any signal from the
# previous simulation (e.g. updated solar_surplus_now estimate).
ContextBuilder = Callable[[LoadInputs, SlotMeta, int], SlotContext]


# ── Plan ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PlanCell:
    """One decision: a single (load, slot) cell of the plan grid."""
    load: str
    slot_index: int
    timestamp: datetime
    decision: LoadDecision

    def hash_key(self) -> tuple:
        # We hash the *action* plus the rule_id (which captures intent),
        # ignoring scores so that score noise doesn't break convergence.
        return (self.load, self.slot_index, self.decision.action, self.decision.rule_id)


@dataclass(frozen=True)
class Plan:
    cells: tuple[PlanCell, ...]

    def hash(self) -> str:
        h = hashlib.sha256()
        for c in self.cells:
            h.update(repr(c.hash_key()).encode())
        return h.hexdigest()

    def cells_for_load(self, load: str) -> tuple[PlanCell, ...]:
        return tuple(c for c in self.cells if c.load == load)

    def cells_for_slot(self, slot_index: int) -> tuple[PlanCell, ...]:
        return tuple(c for c in self.cells if c.slot_index == slot_index)


# ── Forced states ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ForcedState:
    """A hard override for one (load, slot) cell. Bypasses the decision matrix.

    ``reason`` is propagated into the LoadDecision so the explanation layer
    can show "Forced to OFF by anti-flap" instead of an arbitrary OFF.
    """
    load: str
    slot_index: int
    action: str       # "on" / "off"
    reason: str


# ── Result ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PlannerResult:
    plan: Plan
    iterations: int
    converged: bool
    plan_hashes: tuple[str, ...]
    oscillation_detected: bool
    alerts: tuple[str, ...]                   # rule-4 emissions, etc.

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "converged": self.converged,
            "oscillation_detected": self.oscillation_detected,
            "plan_hashes": list(self.plan_hashes),
            "cells_count": len(self.plan.cells),
            "alerts": list(self.alerts),
        }


# ── Build a candidate plan ─────────────────────────────────────────────────
def _build_plan(
    loads: Sequence[LoadInputs],
    slots: Sequence[SlotMeta],
    iteration: int,
    context_builder: ContextBuilder,
    forced_lookup: Mapping[tuple[str, int], ForcedState],
) -> tuple[Plan, list[str]]:
    cells: list[PlanCell] = []
    alerts: list[str] = []
    for load in loads:
        for slot_index, slot in enumerate(slots):
            forced = forced_lookup.get((load.name, slot_index))
            if forced is not None:
                decision = LoadDecision(
                    action=forced.action,
                    reason=f"forced by policy: {forced.reason}",
                    rule_id=0,
                    utility_score=0,
                )
            else:
                ctx = context_builder(load, slot, iteration)
                decision = decide_load_for_slot(load.config, load.quota_state, ctx)
            cells.append(PlanCell(
                load=load.name,
                slot_index=slot_index,
                timestamp=slot.timestamp,
                decision=decision,
            ))
            if decision.alert:
                alerts.append(
                    f"{load.name} @ slot {slot_index} ({slot.timestamp.isoformat()}): "
                    f"{decision.reason}"
                )
    return Plan(cells=tuple(cells)), alerts


# ── Iterate ────────────────────────────────────────────────────────────────
def iterate(
    loads: Sequence[LoadInputs],
    slots: Sequence[SlotMeta],
    context_builder: ContextBuilder,
    forced_states: Sequence[ForcedState] = (),
    max_iter: int = MAX_PLANNER_ITERATIONS,
) -> PlannerResult:
    """Run the convergence loop.

    The ``context_builder`` callback is the seam where the caller injects
    per-iteration signals. On the first iteration it typically returns the
    pristine scenario; on subsequent iterations it can update
    ``solar_surplus_now`` and ``prob_surplus_*`` based on the previous plan's
    simulated trajectory.

    The planner does NOT call into ``simulate_soc`` itself — that orchestration
    happens one level up (so the planner stays a pure function over the
    context_builder it is given).
    """
    if max_iter < 1:
        raise ValueError("max_iter must be ≥ 1")

    forced_lookup = {(f.load, f.slot_index): f for f in forced_states}

    plan_hashes: list[str] = []
    plans: list[Plan] = []
    alerts_all: list[str] = []

    converged = False
    oscillation_detected = False

    for iteration in range(1, max_iter + 1):
        plan, alerts = _build_plan(
            loads, slots, iteration, context_builder, forced_lookup
        )
        h = plan.hash()

        plans.append(plan)
        plan_hashes.append(h)
        alerts_all.extend(alerts)

        # Convergence: identical to previous plan.
        if len(plan_hashes) >= 2 and plan_hashes[-1] == plan_hashes[-2]:
            converged = True
            break

        # 2-cycle oscillation: plan[N] == plan[N-2], same as previous-previous.
        if len(plan_hashes) >= 3 and plan_hashes[-1] == plan_hashes[-3]:
            oscillation_detected = True
            break

    return PlannerResult(
        plan=plans[-1],
        iterations=len(plans),
        converged=converged,
        plan_hashes=tuple(plan_hashes),
        oscillation_detected=oscillation_detected,
        alerts=tuple(alerts_all),
    )
