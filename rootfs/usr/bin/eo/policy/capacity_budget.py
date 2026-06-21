"""Greedy capacity-budget allocator.

Within each slot, the loads' total demand cannot exceed the available
power budget (typically ``inverter_max_w − reserved_house_load_w``). When
the planner's raw_plan would saturate it, this layer turns OFF the
lowest-utility loads until the schedule fits.

Algorithm (O(N log N) per slot, where N is the number of loads with action=on):
    1. Collect ON cells for the slot.
    2. Sort descending by utility_score (ties: higher load_watts first —
       larger loads filled first, so we don't trickle in a sea of tiny
       ones).
    3. Allocate greedily until ``available_w`` is exhausted; the rest get
       overridden to OFF with a reason that includes the budget arithmetic.

Validated by the SPEC's "Sábado de Gloria" test case (Gemini R1 Test 2):
two concurrent valley loads that exceed inverter capacity must be
serialised across slots, not fired in parallel.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from eo.planner.decision_matrix import LoadDecision
from eo.planner.iterative import Plan, PlanCell
from eo.policy.types import PolicyOverride, PolicyPipelineResult

LAYER_NAME = "capacity_budget"


def apply_capacity_budget(
    plan: Plan,
    load_watts: Mapping[str, float],
    available_w: float,
) -> PolicyPipelineResult:
    """Trim loads in saturated slots, biggest-utility-first.

    Parameters
    ----------
    plan
        Raw plan from the iterative planner.
    load_watts
        Static map ``{load_name: watts}`` of nominal consumption per load.
        Loads missing from this map are assumed to draw 0 watts (e.g. a
        sensor-only entry).
    available_w
        Per-slot AC power budget. Typically
        ``inverter_max_w − reserved_house_load_w``.

    Returns
    -------
    PolicyPipelineResult with the adjusted plan and per-cell overrides.
    """
    if available_w < 0:
        raise ValueError("available_w must be ≥ 0")

    # Group cells by slot for processing.
    by_slot: dict[int, list[PlanCell]] = {}
    for cell in plan.cells:
        by_slot.setdefault(cell.slot_index, []).append(cell)

    new_cells: list[PlanCell] = []
    overrides: list[PolicyOverride] = []

    for slot_index in sorted(by_slot.keys()):
        slot_cells = by_slot[slot_index]
        on_cells = [c for c in slot_cells if c.decision.action == "on"]
        off_cells = [c for c in slot_cells if c.decision.action != "on"]

        if not on_cells:
            new_cells.extend(slot_cells)
            continue

        # Sort ON cells: highest utility first, larger load wins ties.
        on_cells_sorted = sorted(
            on_cells,
            key=lambda c: (
                -c.decision.utility_score,
                -load_watts.get(c.load, 0.0),
                c.load,  # stable ordering for tests
            ),
        )

        allocated_w = 0.0
        kept: list[PlanCell] = []
        for cell in on_cells_sorted:
            w = float(load_watts.get(cell.load, 0.0))
            if allocated_w + w <= available_w + 1e-9:
                allocated_w += w
                kept.append(cell)
            else:
                # Override to OFF.
                new_decision = LoadDecision(
                    action="off",
                    reason=(
                        f"capacity_budget: would exceed available "
                        f"{available_w:.0f} W "
                        f"({allocated_w:.0f} W already allocated, "
                        f"this load needs {w:.0f} W)"
                    ),
                    rule_id=cell.decision.rule_id,
                    utility_score=cell.decision.utility_score,
                    alert=cell.decision.alert,
                    min_runtime_only=cell.decision.min_runtime_only,
                )
                new_cells.append(replace(cell, decision=new_decision))
                overrides.append(PolicyOverride(
                    layer=LAYER_NAME, load=cell.load, slot_index=slot_index,
                    original_action="on", new_action="off",
                    reason=new_decision.reason,
                ))

        # Sort kept back into deterministic order (by load name).
        kept.sort(key=lambda c: c.load)
        new_cells.extend(kept)
        new_cells.extend(off_cells)

    # Final ordering: by (slot_index, load) — keeps tests deterministic.
    new_cells.sort(key=lambda c: (c.slot_index, c.load))
    return PolicyPipelineResult(
        adjusted_plan=Plan(cells=tuple(new_cells)),
        overrides=tuple(overrides),
    )
