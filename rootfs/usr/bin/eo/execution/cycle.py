"""Top-level v5 cycle orchestrator.

Pieces together every previous-phase module into a single ``run_v5_cycle``
function. The orchestrator is itself a pure-ish function: every external
dependency (sensors, telemetry, HA dispatch, notifications) is injected
as a callback in ``CycleContext``. The orchestrator never imports any of
the legacy monolith — strangler-fig stays intact.

Order of operations (SPEC §2.1):

    state_in → reconcile_world → debt → scenario(per-load risk)
            → iterate_planner(with forced_states from antiflap)
            → policy_pipeline → execution → state_out
            → persist + emit explanation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Sequence

from eo.execution.engine import (
    SendCommandFn,
    build_execution_plan,
    execute_plan,
)
from eo.execution.reconciliation import (
    IsOnFn,
    reconcile_world_state,
)
from eo.execution.types import ExecutionResult
from eo.planner.iterative import (
    ForcedState,
    LoadInputs,
    PlannerResult,
    SlotMeta,
    iterate,
)
from eo.planner.load_quota import LoadQuotaConfig, compute_debt_state
from eo.policy.antiflap import AntiflapConfig, AntiflapState
from eo.policy.degraded_mode import (
    DegradedLevel,
    DegradedModeConfig,
    DegradedModeInputs,
)
from eo.policy.pipeline import PolicyPipelineInputs, run_policy_pipeline
from eo.policy.types import PolicyOverride
from eo.scenario.scenario_builder import (
    QuantileHourForecast,
    RiskTolerance,
    Scenario,
    build_scenario,
    risk_from_debt_state,
)
from eo.state.system_state import BatteryState, ExecutionWorldState, SystemState


@dataclass(frozen=True)
class LoadDeclaration:
    """One configured deferrable load."""
    name: str
    entity_id: str
    domain: str
    nominal_watts: float
    quota_config: LoadQuotaConfig


@dataclass(frozen=True)
class CycleContext:
    """Callbacks + static config the cycle needs from the host (monolith).

    The orchestrator never touches HA, InfluxDB, files, or globals
    directly — everything goes through these callables.
    """
    now: datetime
    loads: tuple[LoadDeclaration, ...]
    inverter_max_w: float
    reserved_house_load_w: float
    slot_periods: Mapping[int, str]           # period per slot index
    horizon_slots: int                         # typically 48 h × 4 = 192
    # Forecasts and signals — passed in pre-built so this layer stays pure.
    solar_forecasts: Sequence[QuantileHourForecast]
    house_forecasts: Sequence[QuantileHourForecast]
    forecast_quality: dict
    aemet_age_hours: float | None
    sensor_age_max_minutes: float | None
    # Sensor reads — battery state for the planner's initial SOC.
    battery_state: BatteryState
    # Per-load history → debt rebuild seed (window_days entries per load).
    hours_on_per_day_last_window: Mapping[str, list[float]]
    # Callbacks — the strangler-fig seam.
    is_on: IsOnFn
    send_command: SendCommandFn
    notify_alert: Callable[[str], None] = field(default=lambda _msg: None)

    def __post_init__(self):
        if self.now.tzinfo is None:
            raise ValueError("now must be tz-aware")
        if self.horizon_slots < 1:
            raise ValueError("horizon_slots must be ≥ 1")


@dataclass(frozen=True)
class CycleResult:
    new_state: SystemState
    scenario: Scenario | None
    planner_result: PlannerResult | None
    policy_overrides: tuple[PolicyOverride, ...]
    degraded_level: DegradedLevel
    execution_result: ExecutionResult | None
    alerts: tuple[str, ...]


def _build_scenario_for_loads(
    loads: Sequence[LoadDeclaration],
    solar_forecasts: Sequence[QuantileHourForecast],
    house_forecasts: Sequence[QuantileHourForecast],
    load_debt_states: dict[str, "str"],
) -> Scenario | None:
    """Pick the worst-risk debt state among all loads so a single scenario
    serves the planner. Conservative-by-default if any single load is HIGH /
    CRITICAL / IRREACHABLE.
    """
    from eo.planner.load_quota import DebtState
    if not solar_forecasts or not house_forecasts:
        return None
    # Combine: take the most cautious risk across all loads.
    risks = []
    for load in loads:
        debt_str = load_debt_states.get(load.name, DebtState.OK.value)
        debt_enum = DebtState(debt_str) if debt_str in (s.value for s in DebtState) else DebtState.OK
        risks.append(risk_from_debt_state(debt_enum))
    if RiskTolerance.CONSERVATIVE in risks:
        chosen = RiskTolerance.CONSERVATIVE
    else:
        chosen = RiskTolerance.MEDIAN
    return build_scenario(
        solar_forecasts=solar_forecasts,
        house_forecasts=house_forecasts,
        risk_tolerance=chosen,
        debt_state=None,
        confidence_is_heuristic=True,
    )


def _build_planner_inputs(
    loads: Sequence[LoadDeclaration],
    hours_on_per_day: Mapping[str, list[float]],
    days_passed_in_window: int = 0,
    days_with_telemetry: int | None = None,
) -> tuple[list[LoadInputs], dict[str, str]]:
    """Return planner LoadInputs[] and the load_debt summary for SystemState."""
    inputs: list[LoadInputs] = []
    debt_summary: dict[str, str] = {}
    for load in loads:
        quota_state = compute_debt_state(
            load.quota_config,
            hours_on_per_day.get(load.name, []),
            days_passed_in_window=days_passed_in_window,
            days_with_telemetry=days_with_telemetry,
        )
        inputs.append(LoadInputs(
            name=load.name, config=load.quota_config, quota_state=quota_state,
        ))
        debt_summary[load.name] = quota_state.debt_state.value
    return inputs, debt_summary


def _forced_states_from_antiflap(
    antiflap_state: Mapping[str, AntiflapState],
    now: datetime,
    config: AntiflapConfig,
) -> list[ForcedState]:
    """Slot-0 hard masks for loads still within decision_hold."""
    from datetime import timedelta
    hold = timedelta(minutes=config.decision_hold_minutes)
    out: list[ForcedState] = []
    for load, st in antiflap_state.items():
        prior_ts = st.last_change_ts
        if prior_ts.tzinfo is None:
            from datetime import timezone
            prior_ts = prior_ts.replace(tzinfo=timezone.utc)
        if now - prior_ts < hold:
            out.append(ForcedState(
                load=load,
                slot_index=0,
                action=st.last_action,
                reason=(
                    f"antiflap hold active "
                    f"({(now - prior_ts).total_seconds() / 60:.1f} min "
                    f"since last change)"
                ),
            ))
    return out


# ── Top-level cycle ─────────────────────────────────────────────────────────
def run_v5_cycle(
    prior_state: SystemState | None,
    ctx: CycleContext,
    antiflap_config: AntiflapConfig = AntiflapConfig(),
    degraded_config: DegradedModeConfig = DegradedModeConfig(),
) -> CycleResult:
    """One full v5 decision cycle.

    Returns the new SystemState plus every intermediate artefact so the
    transparency layer can persist a complete trace.
    """
    # 1. Reconcile world state from HA.
    world = reconcile_world_state(
        load_entities=[(ld.name, ld.entity_id) for ld in ctx.loads],
        is_on=ctx.is_on,
        now=ctx.now,
    )

    # 2. Compute debt + planner inputs.
    planner_loads, debt_summary = _build_planner_inputs(
        ctx.loads, ctx.hours_on_per_day_last_window,
    )

    # 3. Build scenario.
    scenario = _build_scenario_for_loads(
        ctx.loads, ctx.solar_forecasts, ctx.house_forecasts, debt_summary,
    )

    # 4. Generate SlotMeta with periods + dummy probs (the iterative
    # planner uses the context_builder to consume scenario data per slot).
    slots = [
        SlotMeta(
            timestamp=(ctx.solar_forecasts[0].hour_start
                       if ctx.solar_forecasts else ctx.now),
            period=ctx.slot_periods.get(i, "mid"),
            prob_surplus_tomorrow=0.0,
            prob_surplus_next_valley=0.0,
        )
        for i in range(ctx.horizon_slots)
    ]

    # 5. forced_states from antiflap (slot-0 holds).
    antiflap_state = (
        dict(prior_state.antiflap_state) if prior_state else {}
    )
    forced_states = _forced_states_from_antiflap(
        antiflap_state, ctx.now, antiflap_config,
    )

    # 6. Iterate the planner — its context builder is a stable closure
    # over the slot meta (scenario data flows in via probabilities the
    # caller should plumb up from the simulator; v5.0.0 ships with a
    # neutral context — production tuning happens in v5.1).
    def context_builder(load, slot, iteration):
        from eo.planner.decision_matrix import SlotContext
        return SlotContext(
            period=slot.period,
            solar_surplus_now=False,
            prob_surplus_tomorrow=slot.prob_surplus_tomorrow,
            prob_surplus_next_valley=slot.prob_surplus_next_valley,
            no_valley_before_deadline=False,
        )

    planner_result = iterate(
        loads=planner_loads,
        slots=slots,
        context_builder=context_builder,
        forced_states=forced_states,
    )

    # 7. Policy pipeline.
    available_w = max(0.0, ctx.inverter_max_w - ctx.reserved_house_load_w)
    policy_summary = run_policy_pipeline(
        raw_plan=planner_result.plan,
        inputs=PolicyPipelineInputs(
            load_watts={ld.name: ld.nominal_watts for ld in ctx.loads},
            available_w=available_w,
            slot_periods=ctx.slot_periods,
            antiflap_state=antiflap_state,
            now=ctx.now,
            degraded_inputs=DegradedModeInputs(
                forecast_mae=ctx.forecast_quality.get("solar.p50", {}).get("mae"),
                aemet_age_hours=ctx.aemet_age_hours,
                sensor_age_max_minutes=ctx.sensor_age_max_minutes,
            ),
        ),
        antiflap_config=antiflap_config,
        degraded_config=degraded_config,
    )

    # 8. Execute slot 0 to the world.
    exec_plan = build_execution_plan(
        policy_plan=policy_summary.final_plan,
        load_entities={ld.name: (ld.entity_id, ld.domain) for ld in ctx.loads},
        loads_currently_on=set(world.loads_on),
        cycle_ts=ctx.now,
    )
    exec_result = execute_plan(exec_plan, ctx.send_command)

    # 9. Update antiflap_state for any actually-acknowledged transition.
    new_antiflap = dict(antiflap_state)
    for r in exec_result.results:
        if r.acknowledged:
            new_antiflap[r.request.load] = AntiflapState(
                load=r.request.load,
                last_action=r.request.target_action,
                last_change_ts=ctx.now,
            )

    # 10. Build new SystemState.
    history_tail = (
        list(prior_state.planner_hash_history)[-9:] if prior_state else []
    )
    if planner_result.plan_hashes:
        history_tail.append(planner_result.plan_hashes[-1])
    new_state = SystemState(
        cycle_ts=ctx.now,
        battery=ctx.battery_state,
        forecast_quality=dict(ctx.forecast_quality),
        planner_hash_history=tuple(history_tail),
        load_debt=debt_summary,
        antiflap_state=new_antiflap,
        execution_world_state=ExecutionWorldState(
            loads_on=tuple(sorted(
                set(world.loads_on) | set(exec_result.loads_now_on)
                - {r.request.load for r in exec_result.results
                   if r.request.target_action == "off" and r.acknowledged}
            )),
            last_reconciled_at=ctx.now,
        ),
    )

    # 11. Alerts: planner critical-peak + degraded mode transitions.
    alerts = list(planner_result.alerts)
    if policy_summary.degraded_level != DegradedLevel.NORMAL:
        alerts.append(
            f"degraded_mode active: level={policy_summary.degraded_level.name}"
        )

    return CycleResult(
        new_state=new_state,
        scenario=scenario,
        planner_result=planner_result,
        policy_overrides=policy_summary.overrides,
        degraded_level=policy_summary.degraded_level,
        execution_result=exec_result,
        alerts=tuple(alerts),
    )
