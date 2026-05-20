"""Power-flow physics for one timestep.

The simulator is structured around a ``PhysicsModel`` interface (SPEC §1.3
SIM4, decision D2). Today's installation (single battery + hybrid inverter +
optional grid + PV string) is served by :class:`SingleBatteryPhysicsModel`.
Future deployments — second battery AC-coupled, EV charger, export-limited
inverters — will provide their own ``PhysicsModel`` implementations without
having to extend ``simulate_soc()`` with another ``if``.

Conventions used throughout the simulator:
    * Energy units in kWh per slot.
    * Power units in W (rates), kWh = W × dt_hours / 1000.
    * SOC in percentage points [0, 100].
    * Sign convention:
        net_kwh > 0  → surplus  (solar > house+loads) → charge or export
        net_kwh < 0  → deficit  (house+loads > solar) → discharge or import
        grid_kwh > 0 → import (drawing from the grid)
        grid_kwh < 0 → export (selling to the grid)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol


# ── Configuration dataclasses ───────────────────────────────────────────────
@dataclass(frozen=True)
class BatteryConfig:
    """Static battery specification.

    All percentages are absolute (0-100). All powers in watts.
    """
    capacity_kwh: float                  # usable energy capacity
    health_min_pct: float = 25.0         # lower SOC bound enforced by simulator
    health_max_pct: float = 85.0         # upper SOC bound enforced by simulator
    charge_efficiency: float = 0.95      # AC → battery DC, one-way
    discharge_efficiency: float = 0.95   # battery DC → AC, one-way
    max_charge_w: float = 5000.0         # power limit on charge
    max_discharge_w: float = 5000.0      # power limit on discharge

    def __post_init__(self):
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if not 0 <= self.health_min_pct < self.health_max_pct <= 100:
            raise ValueError(
                "health_min_pct must satisfy 0 ≤ min < max ≤ 100"
            )
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("charge_efficiency must be in (0, 1]")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("discharge_efficiency must be in (0, 1]")
        if self.max_charge_w <= 0 or self.max_discharge_w <= 0:
            raise ValueError("max_charge_w and max_discharge_w must be positive")


@dataclass(frozen=True)
class HouseSystemConfig:
    """Power-flow envelope for the whole installation."""
    inverter_max_w: float                # AC throughput cap (PV + bat combined)
    grid_max_import_w: float = 15000.0   # contracted / physical import cap
    grid_max_export_w: float = 10000.0   # contracted export cap (0 for off-grid)

    def __post_init__(self):
        if self.inverter_max_w <= 0:
            raise ValueError("inverter_max_w must be positive")
        if self.grid_max_import_w < 0 or self.grid_max_export_w < 0:
            raise ValueError("grid limits must be non-negative")


# ── Step result ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PhysicsStepResult:
    """Outcome of one physics step.

    Energy fields are kWh during the step (positive). ``grid_import_kwh``
    and ``grid_export_kwh`` are mutually exclusive in normal operation.
    ``violations`` is a list of human-readable strings — empty when the step
    was conflict-free. Populating ``violations`` does not raise (the caller
    decides what to do).
    """
    soc_pct_after: float
    battery_charge_kwh: float
    battery_discharge_kwh: float
    grid_import_kwh: float
    grid_export_kwh: float
    pv_curtailed_kwh: float
    unmet_load_kwh: float
    violations: tuple[str, ...] = ()

    @property
    def net_battery_kwh(self) -> float:
        return self.battery_charge_kwh - self.battery_discharge_kwh


# ── Interface ───────────────────────────────────────────────────────────────
class PhysicsModel(Protocol):
    """Protocol for one-timestep power-flow models.

    Future implementations may model multiple batteries, EV chargers, AC
    coupling, etc. Each step receives the current SOC and the net energy
    flow (solar − house − planned loads), and returns the resulting state.
    """

    def step(
        self,
        soc_pct: float,
        net_kwh: float,
        dt_hours: float,
        battery: BatteryConfig,
        system: HouseSystemConfig,
        forced_charge_w: float = 0.0,
        forced_discharge_w: float = 0.0,
    ) -> PhysicsStepResult:
        ...


# ── SingleBatteryPhysicsModel (v5.0.0 default) ─────────────────────────────
class SingleBatteryPhysicsModel:
    """One hybrid inverter + one battery + grid + PV.

    Decision policy in each step (matches ``decide_battery`` policy as a
    pure function of the inputs):

      1. Forced charge/discharge requests take precedence (planner-driven).
      2. Otherwise, surplus → charge battery to health_max, then export.
      3. Deficit → discharge battery to health_min, then import.

    All flows are bounded by the inverter and grid caps. Anything that
    cannot flow gets curtailed (PV) or marked as unmet load (deficit
    beyond imports).
    """

    @staticmethod
    def _soc_to_kwh(soc_pct: float, capacity_kwh: float) -> float:
        return capacity_kwh * (soc_pct / 100.0)

    @staticmethod
    def _kwh_to_soc(kwh: float, capacity_kwh: float) -> float:
        return 100.0 * kwh / capacity_kwh

    def step(
        self,
        soc_pct: float,
        net_kwh: float,
        dt_hours: float,
        battery: BatteryConfig,
        system: HouseSystemConfig,
        forced_charge_w: float = 0.0,
        forced_discharge_w: float = 0.0,
    ) -> PhysicsStepResult:
        if dt_hours <= 0:
            raise ValueError("dt_hours must be positive")
        if forced_charge_w < 0 or forced_discharge_w < 0:
            raise ValueError("forced powers must be ≥ 0")
        if forced_charge_w > 0 and forced_discharge_w > 0:
            raise ValueError(
                "cannot force charge and discharge in the same step"
            )

        # Convert powers to energy per step.
        max_charge_kwh = (battery.max_charge_w * dt_hours) / 1000.0
        max_discharge_kwh = (battery.max_discharge_w * dt_hours) / 1000.0
        inverter_kwh = (system.inverter_max_w * dt_hours) / 1000.0
        grid_import_kwh_cap = (system.grid_max_import_w * dt_hours) / 1000.0
        grid_export_kwh_cap = (system.grid_max_export_w * dt_hours) / 1000.0

        # Headroom in the battery, after applying charge/discharge efficiency.
        # We track energy *delivered into the cells* (charge) and *drawn from
        # cells* (discharge). The grid-side AC kWh is what we report.
        soc_kwh = self._soc_to_kwh(soc_pct, battery.capacity_kwh)
        min_kwh = self._soc_to_kwh(battery.health_min_pct, battery.capacity_kwh)
        max_kwh = self._soc_to_kwh(battery.health_max_pct, battery.capacity_kwh)
        room_to_full_dc = max(0.0, max_kwh - soc_kwh)
        room_to_empty_dc = max(0.0, soc_kwh - min_kwh)

        violations: list[str] = []

        charge_ac_kwh = 0.0
        discharge_ac_kwh = 0.0

        if forced_charge_w > 0:
            forced_kwh = (forced_charge_w * dt_hours) / 1000.0
            # AC → DC via charge_efficiency: ac_kwh → dc_kwh = ac × eff.
            allowed_dc = min(room_to_full_dc, forced_kwh * battery.charge_efficiency)
            # Back to AC after the physical cap.
            charge_ac_kwh = min(forced_kwh, allowed_dc / battery.charge_efficiency)
            charge_ac_kwh = min(charge_ac_kwh, max_charge_kwh, inverter_kwh)
            if charge_ac_kwh < forced_kwh - 1e-9:
                violations.append(
                    f"forced_charge clipped: requested {forced_kwh:.3f} kWh, "
                    f"applied {charge_ac_kwh:.3f}"
                )
        elif forced_discharge_w > 0:
            forced_kwh = (forced_discharge_w * dt_hours) / 1000.0
            # DC → AC: dc_kwh → ac_kwh = dc × eff. So to deliver ac, need dc = ac/eff.
            allowed_ac_from_dc = room_to_empty_dc * battery.discharge_efficiency
            discharge_ac_kwh = min(
                forced_kwh, allowed_ac_from_dc, max_discharge_kwh, inverter_kwh
            )
            if discharge_ac_kwh < forced_kwh - 1e-9:
                violations.append(
                    f"forced_discharge clipped: requested {forced_kwh:.3f} kWh, "
                    f"applied {discharge_ac_kwh:.3f}"
                )
        else:
            if net_kwh > 0:
                surplus = net_kwh
                # Charge to fill, AC-side capped by inverter / max_charge.
                max_charge_ac_step = min(
                    max_charge_kwh,
                    inverter_kwh,
                    room_to_full_dc / battery.charge_efficiency,
                )
                charge_ac_kwh = min(surplus, max_charge_ac_step)
            elif net_kwh < 0:
                deficit = -net_kwh
                max_discharge_ac_step = min(
                    max_discharge_kwh,
                    inverter_kwh,
                    room_to_empty_dc * battery.discharge_efficiency,
                )
                discharge_ac_kwh = min(deficit, max_discharge_ac_step)

        # Update SOC from DC-side energy.
        dc_into_battery = charge_ac_kwh * battery.charge_efficiency
        dc_out_of_battery = discharge_ac_kwh / battery.discharge_efficiency
        soc_kwh_after = soc_kwh + dc_into_battery - dc_out_of_battery
        # Float guard
        soc_kwh_after = max(min_kwh, min(max_kwh, soc_kwh_after))
        soc_pct_after = self._kwh_to_soc(soc_kwh_after, battery.capacity_kwh)

        # Reconcile grid flow.
        # Effective AC house draw vs available PV:
        #   net_kwh = solar - load   (positive = surplus, negative = deficit)
        # After battery action, residual that has to leave/enter via grid.
        if forced_charge_w > 0 or forced_discharge_w > 0:
            # When the planner forces a battery action, the residual is what
            # remains after the forced flow. Surplus past the battery goes to
            # export; deficit past the battery comes from import. PV that
            # cannot land anywhere (battery full, export saturated) becomes
            # curtailment.
            residual_after_battery = net_kwh + discharge_ac_kwh - charge_ac_kwh
        else:
            residual_after_battery = net_kwh + discharge_ac_kwh - charge_ac_kwh

        grid_import_kwh = 0.0
        grid_export_kwh = 0.0
        pv_curtailed_kwh = 0.0
        unmet_load_kwh = 0.0

        if residual_after_battery > 0:
            # surplus that did not go into the battery
            grid_export_kwh = min(residual_after_battery, grid_export_kwh_cap)
            pv_curtailed_kwh = residual_after_battery - grid_export_kwh
        elif residual_after_battery < 0:
            need = -residual_after_battery
            grid_import_kwh = min(need, grid_import_kwh_cap)
            unmet_load_kwh = need - grid_import_kwh

        # Final physical invariant: PV + battery_dis + grid_imp ≤ inverter
        # (AC bus throughput). Tolerate ε for float.
        ac_through_bus = max(0.0, charge_ac_kwh) + discharge_ac_kwh + grid_import_kwh
        if ac_through_bus > inverter_kwh * 1.0001 + 1e-6:
            violations.append(
                f"inverter throughput exceeded: {ac_through_bus:.3f} kWh > "
                f"{inverter_kwh:.3f} kWh cap"
            )

        return PhysicsStepResult(
            soc_pct_after=soc_pct_after,
            battery_charge_kwh=charge_ac_kwh,
            battery_discharge_kwh=discharge_ac_kwh,
            grid_import_kwh=grid_import_kwh,
            grid_export_kwh=grid_export_kwh,
            pv_curtailed_kwh=pv_curtailed_kwh,
            unmet_load_kwh=unmet_load_kwh,
            violations=tuple(violations),
        )
