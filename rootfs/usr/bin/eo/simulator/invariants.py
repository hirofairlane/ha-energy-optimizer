"""Invariant assertions for the simulator (SPEC §1.3 SIM6).

Two modes per invariant:

  * **Hard mode** (``strict=True``) — used in tests and offline experiments.
    Violation raises :class:`InvariantViolation`. The build/test should fail.
  * **Soft mode** (``strict=False``, the default) — used in production. The
    violation is returned as a string. The caller accumulates them on the
    ``SimulationResult`` and a counter decides if the engine flips to
    degraded mode after N consecutive violations (SPEC §1.3 SIM5).

Categories:
    physical   — SOC bounds, charge/discharge mutex, inverter cap.
    energetic  — energy conservation per step within tolerance ε.
    temporal   — min runtime, no contradictory in-same-slot actions, debt monotonicity.
"""

from __future__ import annotations


class InvariantViolation(AssertionError):
    """Raised in strict mode when an invariant fails."""


# Numerical tolerance for energy conservation checks (kWh). 1 Wh is generous
# enough for accumulated float error in a 15-min step.
ENERGY_TOLERANCE_KWH: float = 0.001


# ── Physical ────────────────────────────────────────────────────────────────
def check_soc_bounded(
    soc_pct: float, min_pct: float, max_pct: float, *, strict: bool = False
) -> str | None:
    if min_pct - 1e-6 <= soc_pct <= max_pct + 1e-6:
        return None
    msg = (
        f"SOC {soc_pct:.3f} % outside health bounds "
        f"[{min_pct:.3f}, {max_pct:.3f}]"
    )
    if strict:
        raise InvariantViolation(msg)
    return msg


def check_charge_discharge_mutex(
    charge_kwh: float, discharge_kwh: float, *, strict: bool = False
) -> str | None:
    if charge_kwh > 1e-9 and discharge_kwh > 1e-9:
        msg = (
            f"battery charging and discharging simultaneously: "
            f"charge={charge_kwh:.3f} kWh, discharge={discharge_kwh:.3f} kWh"
        )
        if strict:
            raise InvariantViolation(msg)
        return msg
    return None


def check_inverter_capacity(
    charge_kwh: float,
    discharge_kwh: float,
    grid_import_kwh: float,
    inverter_max_kwh: float,
    *,
    strict: bool = False,
) -> str | None:
    """The AC bus carries: battery_chg + battery_dis + grid_import.

    PV is delivered to the AC bus too but already accounts for the inverter
    on the DC→AC side; ``inverter_max_kwh`` is the AC throughput limit per
    step.
    """
    flow = charge_kwh + discharge_kwh + grid_import_kwh
    if flow > inverter_max_kwh + ENERGY_TOLERANCE_KWH:
        msg = (
            f"inverter capacity exceeded: AC throughput {flow:.3f} kWh > "
            f"{inverter_max_kwh:.3f} kWh"
        )
        if strict:
            raise InvariantViolation(msg)
        return msg
    return None


# ── Energetic ──────────────────────────────────────────────────────────────
def check_energy_conservation(
    solar_kwh: float,
    grid_import_kwh: float,
    battery_discharge_kwh: float,
    house_kwh: float,
    planned_loads_kwh: float,
    battery_charge_kwh: float,
    grid_export_kwh: float,
    pv_curtailed_kwh: float,
    unmet_load_kwh: float,
    *,
    tolerance_kwh: float = ENERGY_TOLERANCE_KWH,
    strict: bool = False,
) -> str | None:
    """Conservation per step:

        sources = solar + grid_import + battery_discharge
        sinks   = house + planned_loads + battery_charge + grid_export
                + pv_curtailed (PV that could not flow anywhere)
                − unmet_load   (load demanded but not supplied)

        sources ≈ sinks, within tolerance.

    The signs of ``pv_curtailed_kwh`` and ``unmet_load_kwh`` are positive in
    the result struct; here we put curtailment on the *sink* side (it removes
    energy from the surplus side without serving anything) and unmet load on
    the *source* side reduction (it never reached anyone — we subtract from
    the demand side).
    """
    sources = solar_kwh + grid_import_kwh + battery_discharge_kwh
    sinks = (
        house_kwh + planned_loads_kwh
        + battery_charge_kwh + grid_export_kwh
        + pv_curtailed_kwh
        - unmet_load_kwh
    )
    diff = abs(sources - sinks)
    if diff > tolerance_kwh:
        msg = (
            f"energy conservation violated: |sources − sinks| = "
            f"{diff:.4f} kWh > tolerance {tolerance_kwh} kWh "
            f"(sources={sources:.3f}, sinks={sinks:.3f})"
        )
        if strict:
            raise InvariantViolation(msg)
        return msg
    return None


# ── Temporal ───────────────────────────────────────────────────────────────
def check_min_runtime(
    load_name: str,
    on_durations_minutes: list[float],
    min_runtime_minutes: float,
    *,
    strict: bool = False,
) -> str | None:
    """Every ON segment must be at least ``min_runtime_minutes`` long.

    Pass the per-segment durations of a specific load over the simulation
    horizon. Emergency stops can be excluded by the caller (e.g. by
    appending them as a single >= min_runtime segment).
    """
    too_short = [d for d in on_durations_minutes if 0 < d < min_runtime_minutes - 1e-6]
    if too_short:
        msg = (
            f"load '{load_name}' violated min_runtime: "
            f"segments {too_short} < {min_runtime_minutes} min"
        )
        if strict:
            raise InvariantViolation(msg)
        return msg
    return None


def check_no_contradictory_action_in_slot(
    slot_index: int,
    actions: list[str],
    *,
    strict: bool = False,
) -> str | None:
    """Within one slot, the same load cannot have both an ON and an OFF action.

    ``actions`` is the list of decision actions taken in this slot
    (e.g. ``["turn_on:switch.boiler", "turn_off:switch.boiler"]``).
    """
    by_target: dict[str, set[str]] = {}
    for a in actions:
        if ":" not in a:
            continue
        verb, target = a.split(":", 1)
        by_target.setdefault(target, set()).add(verb)
    conflicts = [
        t for t, verbs in by_target.items()
        if "turn_on" in verbs and "turn_off" in verbs
    ]
    if conflicts:
        msg = (
            f"slot {slot_index}: contradictory actions on {conflicts} "
            f"(both turn_on and turn_off in the same slot)"
        )
        if strict:
            raise InvariantViolation(msg)
        return msg
    return None


def check_debt_monotonic_without_execution(
    load_name: str,
    debt_before: float,
    debt_after: float,
    executed_hours: float,
    *,
    strict: bool = False,
) -> str | None:
    """Debt (remaining hours to meet quota) must not *decrease* unless the
    load actually executed during the slot.

    This catches a class of planner bugs where procrastinating produces a
    favourable debt score by re-interpreting the window mid-flight.
    """
    if executed_hours > 1e-6:
        return None  # execution happened, debt may legitimately drop
    if debt_after < debt_before - 1e-6:
        msg = (
            f"load '{load_name}': debt decreased "
            f"({debt_before:.3f} → {debt_after:.3f}) "
            f"without execution"
        )
        if strict:
            raise InvariantViolation(msg)
        return msg
    return None
