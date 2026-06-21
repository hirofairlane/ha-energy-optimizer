"""Post-cycle and post-restart reconciliation (SPEC §1.6 E2-E3).

Two concerns:

  * **Per-cycle reconciliation** — at the start of every cycle we re-read
    the loads' current ON/OFF state from HA so the policy layer's
    capacity_budget and antiflap operate on world truth, not on the
    state we *think* we left things in.

  * **Post-restart reconciliation** — when the add-on starts fresh, the
    persisted SystemState.json may be stale or missing. We rebuild the
    per-load debt counters by querying the telemetry backends
    (InfluxDB / MariaDB / recorder) and override whatever the file said
    (Gemini R2 §5.4: "telemetry wins").

Both reconcile to the same `ExecutionWorldState` shape so the cycle code
does not branch on "first run vs steady state".

The functions are pure — they take callables for telemetry / state fetch
and return a new ExecutionWorldState, leaving I/O to the caller.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable

from eo.state.system_state import ExecutionWorldState

# Type alias for "is this load currently ON?" lookups. The addon wires this
# to ha_state(entity_id) checks. Returns True / False.
IsOnFn = Callable[[str], bool]


def reconcile_world_state(
    load_entities: Iterable[tuple[str, str]],   # (load_name, entity_id)
    is_on: IsOnFn,
    now: datetime,
) -> ExecutionWorldState:
    """Re-read every load's current state from HA and produce a fresh
    ExecutionWorldState.

    Exceptions from ``is_on`` are caught and treated as "unknown → OFF"
    so a single misbehaving entity does not break the cycle.
    """
    if now.tzinfo is None:
        raise ValueError("now must be tz-aware")

    on_set: list[str] = []
    for load_name, entity_id in load_entities:
        try:
            if is_on(entity_id):
                on_set.append(load_name)
        except Exception:
            # Sensor missing or HA hiccup — defensive default: assume OFF.
            continue
    return ExecutionWorldState(
        loads_on=tuple(sorted(on_set)),
        last_reconciled_at=now,
    )


# Telemetry-side reconstruction of execution history. The signature is
# kept generic so the addon can plug InfluxDB / MariaDB / recorder backends
# behind a single callback that returns the on-hours-per-day for the
# requested load over the requested window.
HoursOnFn = Callable[[str, int], list[float]]


def reconcile_load_debt(
    load_names: Iterable[str],
    window_days: int,
    fetch_hours_on_per_day: HoursOnFn,
) -> dict[str, list[float]]:
    """Rebuild ``hours_on_per_day_last_N`` for each load from telemetry.

    The result is fed to ``compute_debt_state(...)`` to refresh the
    SystemState.load_debt after a restart.
    """
    if window_days < 1:
        raise ValueError("window_days must be ≥ 1")
    out: dict[str, list[float]] = {}
    for load in load_names:
        try:
            hours = fetch_hours_on_per_day(load, window_days)
            if not isinstance(hours, list):
                hours = list(hours)
            # Defensive: clip to window_days entries.
            out[load] = [max(0.0, float(h)) for h in hours[-window_days:]]
        except Exception:
            out[load] = []
    return out
