"""Deterministic SOC + power-flow simulator.

The simulator is the engine's "physics oracle" — given:
    * a schedule of solar / house / planned-load forecasts in 15-min slots,
    * the battery's current state,
    * a configurable PhysicsModel,

it produces a per-slot trace of SOC, battery flow, grid flow, and any
invariant violations along the way.

It is a **pure function** (no I/O, no global state, no logging on the hot
path) so the planner can call it many times during convergence iteration
(SPEC §1.4 P3-P5) without side effects.

Submodules:
    physics_model.py — interface contract + SingleBatteryPhysicsModel.
    invariants.py    — physical / energetic / temporal assertions.
    interpolation.py — hourly → 15-min slot helpers.
    simulate_soc.py  — the public simulate_soc() entry point.
"""

from eo.simulator.physics_model import (  # noqa: F401
    BatteryConfig,
    HouseSystemConfig,
    PhysicsModel,
    PhysicsStepResult,
    SingleBatteryPhysicsModel,
)
