"""Execution engine — pure plan-to-commands mapper + HA dispatcher.

Inputs:
  * A policy-adjusted Plan (slot 0 only — future slots stay speculative
    in v5.0.0; SPEC §1.6 E2).
  * A static map ``{load_name: (entity_id, domain)}`` from the wizard.
  * A current world state snapshot (which loads are ON right now), so we
    only emit commands when the desired state differs.
  * A callable ``send_command(domain, service, data) -> bool`` — the seam
    where the addon's existing ``ha_service`` wires in.

The engine does not poll for ACK; ``send_command`` returns success and we
trust the next cycle's reconciliation pass (SPEC §1.6 E3) to catch any
divergence. This is the "optimistic hybrid" pattern from Gemini R2 §5.1.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Mapping

from eo.execution.types import (
    CommandRequest,
    CommandResult,
    ExecutionPlan,
    ExecutionResult,
)
from eo.planner.iterative import Plan


# Type alias for the HA dispatch callback. Returns True on a successful
# service call (status 2xx, no exception). False means the command failed
# to dispatch — the engine records ``acknowledged=False`` and moves on.
SendCommandFn = Callable[[str, str, dict], bool]


def build_execution_plan(
    policy_plan: Plan,
    load_entities: Mapping[str, tuple[str, str]],   # {load: (entity_id, domain)}
    loads_currently_on: set[str],
    cycle_ts: datetime,
    only_slot_index: int = 0,
) -> ExecutionPlan:
    """Filter the policy-adjusted plan down to a single slot and emit
    one CommandRequest per (load, desired transition).

    A command is only emitted if the desired state differs from the
    current world state — no spurious turn_on when the switch is already
    on.
    """
    commands: list[CommandRequest] = []
    for cell in policy_plan.cells:
        if cell.slot_index != only_slot_index:
            continue
        binding = load_entities.get(cell.load)
        if binding is None:
            # No physical entity bound to this load name — skip silently.
            continue
        entity_id, domain = binding
        desired = cell.decision.action
        currently_on = cell.load in loads_currently_on

        if desired == "on" and not currently_on:
            commands.append(CommandRequest(
                load=cell.load, target_action="on",
                domain=domain, service="turn_on",
                entity_id=entity_id,
                data={"entity_id": entity_id},
            ))
        elif desired == "off" and currently_on:
            commands.append(CommandRequest(
                load=cell.load, target_action="off",
                domain=domain, service="turn_off",
                entity_id=entity_id,
                data={"entity_id": entity_id},
            ))
        # If desired matches the current state, no command needed.

    return ExecutionPlan(cycle_ts=cycle_ts, commands=tuple(commands))


def execute_plan(
    plan: ExecutionPlan,
    send_command: SendCommandFn,
) -> ExecutionResult:
    """Dispatch every command in the plan and collect per-command results.

    ``send_command(domain, service, data)`` is the addon's HA helper.
    Exceptions from the callback are caught and reported as failed ACKs —
    the engine never raises during dispatch.
    """
    results: list[CommandResult] = []
    for cmd in plan.commands:
        ok = False
        err: str | None = None
        try:
            ok = bool(send_command(cmd.domain, cmd.service, cmd.data))
        except Exception as e:  # noqa: BLE001 — defensive on a callback boundary
            err = f"{type(e).__name__}: {e}"
        results.append(CommandResult(
            request=cmd,
            issued=True,
            acknowledged=ok,
            error=err,
        ))
    return ExecutionResult(plan=plan, results=tuple(results))
