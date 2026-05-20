# Energy Optimizer v5.0.0 — Engine architecture

This document describes the **v5 predictive engine** shipped in v5.0.0.
For the user-facing changelog see [README.md](../README.md#changelog).
For the spec that drove the design see
[`audit/SPEC_v5.0.0_FROZEN.md`](../audit/SPEC_v5.0.0_FROZEN.md).

The engine is delivered behind `v5_engine_enabled` (default `false`). When
the flag is off the legacy v4 reactive cycle runs unchanged. The two
engines live side by side in the codebase (strangler-fig pattern); the v5
modules are imported lazily so the legacy install carries no runtime cost
from their presence.

---

## 1. Why a new engine

The v4 engine took a decision per cycle by inspecting the current sensors
and applying a set of heuristics ("if surplus and SOC has headroom →
charge", etc). It was fast and intuitive but suffered three structural
limits:

1. **No horizon awareness.** A load deferred during peak could not "see"
   that tomorrow's surplus would have made the cycle unnecessary; a load
   fired in valley couldn't anticipate the upcoming surplus either.
2. **Predictor inflation.** The legacy `predict_soc` autoregressive chain
   produced R²=0.998 — which sounded great but came from autocorrelation,
   not from genuine predictive power. It misled the engine into
   over-confident horizon plans.
3. **Implicit state.** Each module read sensors directly and decided. The
   simulator and the planner were essentially the same function. Tests
   were hard to write, and behaviour was hard to audit.

The v5 engine fixes all three: explicit horizon, separate forecasting
from simulation, explicit invariants and traceability.

---

## 2. Module map

The new engine lives in the `eo/` package under `rootfs/usr/bin/`. Each
sub-package is independently importable and unit-tested.

```
eo/
├── checks/            v4.0.2 hotfix; v5 reuses the same pattern
│   └── setup_integrity.py
├── forecasters/       quantile predictors with explicit quality tracking
│   ├── clear_sky.py
│   ├── atmospheric_factor.py
│   ├── solar_forecaster.py
│   ├── house_forecaster.py
│   ├── quality.py
│   └── training.py
├── simulator/         pure SOC + power-flow integrator with invariants
│   ├── physics_model.py
│   ├── interpolation.py
│   ├── invariants.py
│   └── simulate_soc.py
├── planner/           debt-aware horizon planner with convergence guard
│   ├── load_quota.py
│   ├── utility_score.py
│   ├── decision_matrix.py
│   └── iterative.py
├── policy/            operational constraints over the planner's output
│   ├── capacity_budget.py
│   ├── peak_prohibition.py
│   ├── antiflap.py
│   ├── degraded_mode.py
│   ├── pipeline.py
│   └── types.py
├── scenario/          quantile collapse + risk mapping
│   └── scenario_builder.py
├── state/             aggregated SystemState dataclass
│   └── system_state.py
└── execution/         HA dispatch + reconciliation + cycle orchestrator
    ├── engine.py
    ├── reconciliation.py
    ├── types.py
    └── cycle.py
```

Tests mirror this layout under `tests/`. Running `python3 -m pytest tests/`
exercises all 355 cases in ~22 s.

---

## 3. End-to-end cycle

```
   ┌─────────────────────────────────────────────────────────────────┐
   │ run_v5_cycle(prior_state, CycleContext)                         │
   └─────────────────────────────────────────────────────────────────┘
                                  │
            ┌─────────────────────┴─────────────────────┐
            ▼                                           ▼
   1. reconcile_world_state                  2. compute_debt_state
      (HA is_on per load)                       per load × LoadQuotaConfig
                                                  → DebtState (ok / low /
                                                     medium / high /
                                                     critical / irreachable)
                                  │
                                  ▼
   3. build_scenario(solar_qhf, house_qhf, risk_from_worst_debt)
      → Scenario {solar_kwh[slot], house_kwh[slot],
                  confidence_is_heuristic, risk_tolerance}
                                  │
                                  ▼
   4. forced_states_from_antiflap(prior_state.antiflap_state, now)
      slot-0 hard masks for loads still inside decision_hold
                                  │
                                  ▼
   5. iterate(loads, slots, context_builder, forced_states, max_iter=5)
      ──────────────────────────────────────────────────────────
      For each candidate plan:
        decide_load_for_slot(quota_state, SlotContext)
          → LoadDecision (action, rule_id, utility_score, alert,
                          min_runtime_only)
      Stop on:
        • plan_hash[N] == plan_hash[N-1]  (converged)
        • plan_hash[N] == plan_hash[N-2]  (2-cycle oscillation)
        • iterations == max_iter          (Monstruo del Bucle guardrail)
      → raw_plan
                                  │
                                  ▼
   6. run_policy_pipeline(raw_plan, ...)
        capacity_budget   greedy by utility_score per slot
        peak_prohibition  defence-in-depth; respects rule_id=4 exception
        antiflap          Schmitt trigger on slot-0 transitions
        degraded_mode     L1/L2/L3 escalation on forecast quality, AEMET
                          freshness, sensor staleness
      → policy_adjusted_plan + overrides[]
                                  │
                                  ▼
   7. build_execution_plan + execute_plan(send_command callback)
      Emits commands only when desired ≠ current world state.
      Optimistic-hybrid: no synchronous ACK wait. Failures are recorded
      and the next cycle's reconciliation pass corrects divergence.
                                  │
                                  ▼
   8. SystemState.from(reconciled, exec_result, ...)
      Atomic JSON persistence (tmp + fsync + os.replace).
                                  │
                                  ▼
                          CycleResult
      (scenario, planner_result, policy_overrides, exec_result,
       degraded_level, alerts, new_state)
```

---

## 4. Key design decisions

### 4.1 Forecasting layer (SPEC §1.1)

- **No `pvlib` dependency**. The clear-sky baseline is implemented from
  Spencer (1971) declination + Beer-Lambert attenuation through
  Kasten-Young air mass. Precision is sufficient: the ML residual
  (`AtmosphericFactorModel`) learns the deviation.
- **Quantile heads (P10 / P50 / P90)**. Three separate
  `GradientBoostingRegressor`s per forecaster, trained with
  `loss="quantile"`. The downstream `ScenarioBuilder` collapses the
  three into a single per-slot value based on risk tolerance.
- **House forecaster is NOT chained.** Features for predicting hour t
  come from observed past values and from already-decided planner
  schedule inputs — never from the model's own earlier predictions. This
  is the deliberate counterpoint to v3.x's autoregressive `predict_soc`.

### 4.2 Simulator (SPEC §1.3)

- **Pure function.** Same inputs → same outputs. No I/O, no logging on
  the hot path. The planner can iterate the simulator many times during
  convergence without side effects.
- **Injectable PhysicsModel.** Today only `SingleBatteryPhysicsModel`.
  When EV chargers or multi-battery arrive, they implement the same
  `PhysicsModel` protocol — no `if` cascades inside the simulator.
- **15-min internal timestep.** Forecasts are hourly; the simulator
  interpolates to 96 slots/day to match the planner's control granularity.
- **Hard invariants.** SOC bounds, charge/discharge mutex, inverter
  capacity, energy conservation per step (ε = 1 Wh), min-runtime,
  no-contradictory-actions, debt monotonicity. In strict mode (tests),
  they raise. In production they accumulate; three consecutive
  violations flip the policy layer into degraded mode.

### 4.3 Planner (SPEC §1.4)

- **Debt state with the two audit-round bug fixes:**
  - **B1.** If the user shrinks `window_days`, we truncate the in-memory
    history with `hours_on_per_day_last_N[-window_days:]` so we do not
    sum stale entries.
  - **B2.** If `days_with_telemetry < window_days` (add-on was down),
    we scale the target proportionally so zero-telemetry days don't look
    like zero-execution days and inflate phantom debt.
- **`irreachable` state.** Added during the audit when the remaining
  hours exceed the physical capacity (`daily_max × days_left`). The
  planner stops chasing it and fires a Telegram alert.
- **Convergence guard.** Hash every plan; stop on equality, detect
  2-cycle oscillation, hard-cap at `MAX_PLANNER_ITERATIONS = 5`. The
  three failure modes are distinguishable in the `PlannerResult` so the
  transparency layer can log the cause.
- **`forced_states` injection.** The policy layer's slot-0 holds
  (antiflap, degraded-mode forced-OFF) are passed back into the planner
  as hard masks so the simulated trajectory never diverges from what
  will execute.

### 4.4 Policy layer (SPEC §1.5)

- **Four pure transforms in a fixed order:**
  `capacity_budget → peak_prohibition → antiflap → degraded_mode`.
- **No mutation in place.** Each step emits a fresh `Plan`. The full
  triple `raw_plan / policy_adjusted_plan / execution_plan` is preserved
  for replay, debugging, and the transparency UI.
- **Greedy capacity budget.** O(N log N) per slot, sorting ON cells by
  `utility_score` then `nominal_watts`. Loads that don't fit are
  overridden to OFF with a reason that includes the budget arithmetic.
  Solves Gemini R1 Test 2 "Sábado de Gloria".
- **Three-level degraded mode.**
  - L1 (forecast MAE > threshold) → drop `min_runtime_only` decisions.
  - L2 (AEMET stale > 24 h) → drop everything except `rule_id=3`
    (critical-driven forced-ON).
  - L3 (sensor stale > 30 min) → all deferred loads OFF.

### 4.5 State (SPEC §1.7)

- **Aggregated `SystemState` dataclass.** Frozen, immutable per cycle.
  Aggregates `battery_state + forecast_quality + planner_history +
  load_debt + antiflap_state + execution_world_state`. ChatGPT's
  POMDP framing is realised here as state organisation — Bayesian belief
  updates are out of scope for v5.0.0.
- **Atomic write.** `tmp + fsync + os.replace`. No WAL. Reconciliation
  from telemetry at the start of the next cycle catches any state/world
  divergence.

### 4.6 Execution (SPEC §1.6)

- **Optimistic hybrid.** Dispatch the command, persist state assuming
  success, reconcile from HA telemetry next cycle. Avoids blocking the
  cycle on a synchronous ACK that may never arrive.
- **Reconciliation post-restart.** Re-build `hours_on_per_day_last_N`
  from InfluxDB / MariaDB / recorder at startup; the persisted
  `PlannerState.json` is treated as a history of intentions, not as
  ground truth.
- **No `if` inside `simulate_soc()`.** When new asset types arrive, they
  ship a `PhysicsModel` implementation. SPEC decision D2.

---

## 5. The audit trail

The architecture was iterated three times with Gemini and ChatGPT acting
as external auditors. The full trail is in `audit/`:

- `for-gemini.md` / `for-chatgpt.md` — initial confrontational primers.
- `quorum-round-2.md` — mirror document with all R1 disagreements.
- `quorum-round-3.md` — closing round; convergence on 16 design points,
  6 explicit cross-signatures requested.
- `Gemini - reply 1.odt` … `gemini reply 3.odt` — Gemini's responses.
- `Chatgpt - reply 1.pdf` / `chatgpt - reply 2.odt` — ChatGPT's responses.
- `SPEC_v5.0.0_FROZEN.md` — the design freeze that drove implementation.

Two decisions were owner-taken in the absence of ChatGPT's round-3
response and are flagged inside the spec:

- **D1 — POMDP framing.** Realised as `SystemState` dataclass (state
  organisation), not as Bayesian belief updates.
- **D2 — `simulate_soc()` guardrail.** Interface contract with
  injectable `PhysicsModel`. Today only `SingleBatteryPhysicsModel`; the
  seam stays open for multi-asset.

---

## 6. What v5.0.0 explicitly does not do

To keep the scope honest:

- ❌ Full utility function (cost + comfort + wear + risk + switching).
  The planner emits a small ordinal `utility_score` per cell; the global
  utility lands in v5.1+.
- ❌ Autocalibration of tilt / azimuth. Defaults are `azimuth=south`,
  `tilt=latitude`.
- ❌ `PowerFlowModel` split into a separate microgrid engine. Stays inside
  `SingleBatteryPhysicsModel`.
- ❌ HVAC thermal-mass model. Heat pump remains reactive.
- ❌ Joint quantile / Monte-Carlo forecasts. Quantiles are marginal and
  independent; `confidence_is_heuristic` is set to `True` in the
  `Scenario`.
- ❌ MILP / MPC. The greedy capacity budget covers the realistic load
  competition cases.
- ❌ Addon-side wiring to activate the engine. See
  [v5-wiring.md](v5-wiring.md).

---

## 7. Pointers for contributors

- **Adding a new forecaster head** — implement in `eo/forecasters/`, add a
  pytest module under `tests/forecasters/`, expose it via
  `eo/forecasters/__init__.py`. Follow the existing `feature_names`
  schema-check pattern for `joblib.load` safety.
- **Adding a new policy layer** — implement as a pure transform in
  `eo/policy/your_layer.py`, return `PolicyPipelineResult`. Wire it into
  `eo/policy/pipeline.py` between `peak_prohibition` and `antiflap`
  unless there is a reason to land elsewhere.
- **Changing the decision matrix** — never edit a row in place. Bump
  rule IDs and append. The transparency layer relies on stable IDs to
  build historical UI badges.
- **Adding a `PhysicsModel`** — implement the protocol in
  `eo/simulator/physics_model.py`. Do NOT add another `if` to
  `simulate_soc()`. The dispatch happens through dependency injection.
- **Running the tests** — `cd files && python3 -m pytest tests/`. The
  test suite is fast (~22 s) and exhaustive.
