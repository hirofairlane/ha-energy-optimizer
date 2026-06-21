"""Schmitt-trigger anti-flap (SPEC §1.5 POL5).

The policy refuses to change a load's state — turn ON if it's currently
OFF, or turn OFF if it's currently ON — within ``decision_hold_minutes``
of the last actual change. The motivation is mechanical: termo / pool
pump relays wear out fast under repeated cycling, and a planner that
oscillates on the boundary of an indicator (e.g. SOC ± hysteresis_margin)
can otherwise toggle every cycle.

State carried across cycles (per load):
    last_action               — "on" or "off"
    last_change_ts            — absolute time the last change happened

Anti-flap only applies to slot 0 (the immediate present); future slots
are speculative and don't bear executable commands yet. The downstream
forced_states injection (SPEC §1.4 P5) feeds slot-0 blocks back to the
planner so the simulated trajectory stays consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from eo.planner.decision_matrix import LoadDecision
from eo.planner.iterative import Plan, PlanCell
from eo.policy.types import PolicyOverride, PolicyPipelineResult

LAYER_NAME = "antiflap"


@dataclass(frozen=True)
class AntiflapState:
    """Per-load last-change record."""
    load: str
    last_action: str         # "on" or "off"
    last_change_ts: datetime


@dataclass(frozen=True)
class AntiflapConfig:
    decision_hold_minutes: float = 30.0

    def __post_init__(self):
        if self.decision_hold_minutes < 0:
            raise ValueError("decision_hold_minutes must be ≥ 0")


def apply_antiflap(
    plan: Plan,
    state_by_load: dict[str, AntiflapState],
    now: datetime,
    config: AntiflapConfig = AntiflapConfig(),
) -> PolicyPipelineResult:
    """Block slot-0 transitions that happen within decision_hold_minutes.

    Parameters
    ----------
    plan
        Plan from previous policy stage.
    state_by_load
        Per-load AntiflapState recorded after the previous cycle. Loads not
        present in the dict are treated as if last_change happened far in
        the past (no hold active).
    now
        The cycle's "now" timestamp. Used to measure age of last_change.
    config
        Hold duration.
    """
    if now.tzinfo is None:
        raise ValueError("now must be tz-aware")

    new_cells: list[PlanCell] = []
    overrides: list[PolicyOverride] = []
    hold = timedelta(minutes=config.decision_hold_minutes)

    for cell in plan.cells:
        if cell.slot_index != 0:
            new_cells.append(cell)
            continue

        prior = state_by_load.get(cell.load)
        if prior is None:
            new_cells.append(cell)
            continue

        # If the plan agrees with the last action, nothing to enforce.
        if cell.decision.action == prior.last_action:
            new_cells.append(cell)
            continue

        # Action would change → check the hold.
        prior_ts = prior.last_change_ts
        if prior_ts.tzinfo is None:
            prior_ts = prior_ts.replace(tzinfo=now.tzinfo)
        elapsed = now - prior_ts
        if elapsed >= hold:
            new_cells.append(cell)
            continue

        # Within hold → freeze to prior action.
        new_decision = LoadDecision(
            action=prior.last_action,
            reason=(
                f"antiflap: change blocked — last action was "
                f"'{prior.last_action}' {elapsed.total_seconds() / 60:.1f} min "
                f"ago (< {config.decision_hold_minutes:.0f} min hold)"
            ),
            rule_id=cell.decision.rule_id,
            utility_score=cell.decision.utility_score,
            alert=cell.decision.alert,
            min_runtime_only=cell.decision.min_runtime_only,
        )
        new_cells.append(replace(cell, decision=new_decision))
        overrides.append(PolicyOverride(
            layer=LAYER_NAME, load=cell.load, slot_index=0,
            original_action=cell.decision.action,
            new_action=prior.last_action,
            reason=new_decision.reason,
        ))

    return PolicyPipelineResult(
        adjusted_plan=Plan(cells=tuple(new_cells)),
        overrides=tuple(overrides),
    )
