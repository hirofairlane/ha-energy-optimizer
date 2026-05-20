"""Execution engine and cycle orchestrator (SPEC §1.6 + §2).

Submodules:
  types.py         — dataclasses for commands, ack results, execution plans.
  engine.py        — pure execute_plan() that maps load decisions to HA
                     commands and reports per-command outcomes.
  reconciliation.py — post-restart reconstruction of SystemState from
                     telemetry (SPEC §1.6 E3: "telemetry wins").
  cycle.py         — top-level v5 cycle orchestrator that threads sensors,
                     forecasts, scenario, planner, policy, execution.
"""

from eo.execution.types import (  # noqa: F401
    CommandRequest,
    CommandResult,
    ExecutionPlan,
    ExecutionResult,
)
