"""Peak-tariff prohibition for deferrable loads.

Defence in depth on top of the decision matrix's row 5: any deferred load
that ended up scheduled ON during a peak slot is forced OFF here, *unless*
the matrix specifically allowed it via row 4 (``allow_peak_on_critical``
with debt = CRITICAL / IRREACHABLE — those decisions carry ``alert=True``
and ``rule_id=4``).

The intent is to keep "deferred loads never run in peak" as an explicit
runtime guarantee even if a future code change loosens the matrix.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from eo.planner.decision_matrix import LoadDecision
from eo.planner.iterative import Plan, PlanCell
from eo.policy.types import PolicyOverride, PolicyPipelineResult

LAYER_NAME = "peak_prohibition"


def apply_peak_prohibition(
    plan: Plan,
    slot_periods: Mapping[int, str],
) -> PolicyPipelineResult:
    """Force OFF any ON load in a peak slot, except matrix rule 4 exceptions.

    Parameters
    ----------
    plan
        Plan from the previous policy stage (or the raw planner output).
    slot_periods
        Map ``{slot_index: period}`` with ``period`` in ``{"peak", "mid", "valley"}``.
    """
    new_cells: list[PlanCell] = []
    overrides: list[PolicyOverride] = []

    for cell in plan.cells:
        period = slot_periods.get(cell.slot_index)
        if (
            cell.decision.action == "on"
            and period == "peak"
            and cell.decision.rule_id != 4
        ):
            new_decision = LoadDecision(
                action="off",
                reason=(
                    f"peak_prohibition: load was scheduled ON in a peak slot "
                    f"(rule {cell.decision.rule_id}). "
                    "Deferred loads do not run during peak by policy."
                ),
                rule_id=cell.decision.rule_id,
                utility_score=cell.decision.utility_score,
                alert=cell.decision.alert,
                min_runtime_only=cell.decision.min_runtime_only,
            )
            new_cells.append(replace(cell, decision=new_decision))
            overrides.append(PolicyOverride(
                layer=LAYER_NAME, load=cell.load, slot_index=cell.slot_index,
                original_action="on", new_action="off",
                reason=new_decision.reason,
            ))
        else:
            new_cells.append(cell)

    return PolicyPipelineResult(
        adjusted_plan=Plan(cells=tuple(new_cells)),
        overrides=tuple(overrides),
    )
