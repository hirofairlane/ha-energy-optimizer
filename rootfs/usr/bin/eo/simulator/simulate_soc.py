"""Pure deterministic SOC simulator.

The orchestrator: walks 15-min slots, calls the injected ``PhysicsModel``
at each step, checks invariants, accumulates results.

The function is genuinely pure — no I/O, no logging, no global state. The
planner calls it many times during convergence iteration (SPEC §1.4 P3),
so it must be cheap and side-effect-free.

Typical usage:

    inputs = [
        SlotInput(timestamp=..., solar_kwh=..., house_kwh=..., planned_loads_kwh=...),
        ...
    ]
    result = simulate_soc(inputs, initial_soc_pct=50.0,
                         battery=battery_cfg, system=system_cfg)
    print(result.final_soc_pct, result.total_grid_import_kwh)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from eo.simulator.invariants import (
    check_charge_discharge_mutex,
    check_energy_conservation,
    check_inverter_capacity,
    check_soc_bounded,
)
from eo.simulator.physics_model import (
    BatteryConfig,
    HouseSystemConfig,
    PhysicsModel,
    SingleBatteryPhysicsModel,
)


# ── Inputs and outputs ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class SlotInput:
    """Per-slot inputs to the simulator.

    All kWh values are for the *duration of the slot* (15 min by default).
    ``forced_charge_w`` / ``forced_discharge_w`` let the planner pin the
    battery action; if both are 0 the physics model chooses self-consumption.
    """
    timestamp: datetime
    solar_kwh: float
    house_kwh: float
    planned_loads_kwh: float = 0.0
    forced_charge_w: float = 0.0
    forced_discharge_w: float = 0.0
    dt_hours: float = 0.25                # 15 min default

    @property
    def net_kwh(self) -> float:
        return self.solar_kwh - self.house_kwh - self.planned_loads_kwh


@dataclass(frozen=True)
class SlotResult:
    timestamp: datetime
    soc_pct: float                         # SOC at end of slot
    battery_charge_kwh: float
    battery_discharge_kwh: float
    grid_import_kwh: float
    grid_export_kwh: float
    pv_curtailed_kwh: float
    unmet_load_kwh: float
    violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimulationResult:
    slots: list[SlotResult]
    final_soc_pct: float
    total_grid_import_kwh: float
    total_grid_export_kwh: float
    total_pv_curtailed_kwh: float
    total_unmet_load_kwh: float
    total_battery_charge_kwh: float
    total_battery_discharge_kwh: float
    all_violations: list[str]

    @property
    def has_violations(self) -> bool:
        return bool(self.all_violations)


# ── Simulation orchestrator ────────────────────────────────────────────────
def simulate_soc(
    slot_inputs: Sequence[SlotInput],
    initial_soc_pct: float,
    battery: BatteryConfig,
    system: HouseSystemConfig,
    physics_model: PhysicsModel | None = None,
    *,
    strict_invariants: bool = False,
) -> SimulationResult:
    """Walk through ``slot_inputs`` and produce a per-slot trace.

    Parameters
    ----------
    slot_inputs
        Ordered sequence of slot inputs. Slots must be sorted by ``timestamp``;
        we trust the caller (no resort) but check for monotonic time.
    initial_soc_pct
        SOC at the start of the first slot.
    battery
        Static battery config (capacity, health limits, efficiencies).
    system
        Power-flow envelope (inverter cap, grid limits).
    physics_model
        Optional override. Defaults to :class:`SingleBatteryPhysicsModel`.
    strict_invariants
        If True, the first violation raises :class:`InvariantViolation`.
        Used by the unit tests; production sets this to False and accumulates.
    """
    if not slot_inputs:
        return SimulationResult(
            slots=[],
            final_soc_pct=initial_soc_pct,
            total_grid_import_kwh=0.0,
            total_grid_export_kwh=0.0,
            total_pv_curtailed_kwh=0.0,
            total_unmet_load_kwh=0.0,
            total_battery_charge_kwh=0.0,
            total_battery_discharge_kwh=0.0,
            all_violations=[],
        )

    if physics_model is None:
        physics_model = SingleBatteryPhysicsModel()

    # Monotonic-time check (cheap, catches misuse early).
    prev_ts: datetime | None = None
    for s in slot_inputs:
        if prev_ts is not None and s.timestamp <= prev_ts:
            raise ValueError(
                f"slot inputs not strictly ascending: {prev_ts} >= {s.timestamp}"
            )
        prev_ts = s.timestamp

    soc = float(initial_soc_pct)
    # If the caller starts outside health bounds, clamp once and record.
    accumulated_violations: list[str] = []
    pre_violation = check_soc_bounded(
        soc,
        battery.health_min_pct, battery.health_max_pct,
        strict=strict_invariants,
    )
    if pre_violation:
        accumulated_violations.append(f"initial: {pre_violation}")
        soc = max(battery.health_min_pct, min(battery.health_max_pct, soc))

    slots: list[SlotResult] = []
    tot_imp = tot_exp = tot_curt = tot_unmet = 0.0
    tot_chg = tot_dis = 0.0

    for inp in slot_inputs:
        step = physics_model.step(
            soc_pct=soc,
            net_kwh=inp.net_kwh,
            dt_hours=inp.dt_hours,
            battery=battery,
            system=system,
            forced_charge_w=inp.forced_charge_w,
            forced_discharge_w=inp.forced_discharge_w,
        )

        slot_violations: list[str] = list(step.violations)

        v = check_soc_bounded(
            step.soc_pct_after,
            battery.health_min_pct, battery.health_max_pct,
            strict=strict_invariants,
        )
        if v:
            slot_violations.append(v)

        v = check_charge_discharge_mutex(
            step.battery_charge_kwh, step.battery_discharge_kwh,
            strict=strict_invariants,
        )
        if v:
            slot_violations.append(v)

        v = check_inverter_capacity(
            charge_kwh=step.battery_charge_kwh,
            discharge_kwh=step.battery_discharge_kwh,
            grid_import_kwh=step.grid_import_kwh,
            inverter_max_kwh=(system.inverter_max_w * inp.dt_hours) / 1000.0,
            strict=strict_invariants,
        )
        if v:
            slot_violations.append(v)

        v = check_energy_conservation(
            solar_kwh=inp.solar_kwh,
            grid_import_kwh=step.grid_import_kwh,
            battery_discharge_kwh=step.battery_discharge_kwh,
            house_kwh=inp.house_kwh,
            planned_loads_kwh=inp.planned_loads_kwh,
            battery_charge_kwh=step.battery_charge_kwh,
            grid_export_kwh=step.grid_export_kwh,
            pv_curtailed_kwh=step.pv_curtailed_kwh,
            unmet_load_kwh=step.unmet_load_kwh,
            strict=strict_invariants,
        )
        if v:
            slot_violations.append(v)

        slot_result = SlotResult(
            timestamp=inp.timestamp,
            soc_pct=step.soc_pct_after,
            battery_charge_kwh=step.battery_charge_kwh,
            battery_discharge_kwh=step.battery_discharge_kwh,
            grid_import_kwh=step.grid_import_kwh,
            grid_export_kwh=step.grid_export_kwh,
            pv_curtailed_kwh=step.pv_curtailed_kwh,
            unmet_load_kwh=step.unmet_load_kwh,
            violations=tuple(slot_violations),
        )
        slots.append(slot_result)
        accumulated_violations.extend(
            f"slot {inp.timestamp.isoformat()}: {msg}" for msg in slot_violations
        )

        # Update running totals + soc carrying into next step.
        soc = step.soc_pct_after
        tot_imp += step.grid_import_kwh
        tot_exp += step.grid_export_kwh
        tot_curt += step.pv_curtailed_kwh
        tot_unmet += step.unmet_load_kwh
        tot_chg += step.battery_charge_kwh
        tot_dis += step.battery_discharge_kwh

    return SimulationResult(
        slots=slots,
        final_soc_pct=soc,
        total_grid_import_kwh=tot_imp,
        total_grid_export_kwh=tot_exp,
        total_pv_curtailed_kwh=tot_curt,
        total_unmet_load_kwh=tot_unmet,
        total_battery_charge_kwh=tot_chg,
        total_battery_discharge_kwh=tot_dis,
        all_violations=accumulated_violations,
    )
