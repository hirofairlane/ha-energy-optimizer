"""Aggregated system state — SPEC §1.7 ST1-ST3.

The true state of the engine is no longer just the SOC: it's a tuple of
SOC + forecast quality + planner history + load debt + antiflap + last
confirmed world state. ``SystemState`` collects them into a single
immutable dataclass that is built once per cycle and passed downstream.

ChatGPT R2's POMDP framing: future moves toward Bayesian belief updates
would build on top of this dataclass, not replace it. For v5.0.0 we keep
it as plain organisation of state.
"""

from eo.state.system_state import (  # noqa: F401
    BatteryState,
    ExecutionWorldState,
    SystemState,
)
