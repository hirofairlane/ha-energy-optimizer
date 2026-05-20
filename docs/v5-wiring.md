# Energy Optimizer v5 — Wiring guide

> **Status:** the v5 engine is shipped in v5.0.0 but **dormant** by default.
> Setting `v5_engine_enabled: true` in the addon configuration is **not
> enough** to drive the cycle yet — the adapter that connects the
> monolithic addon to the new `run_v5_cycle()` orchestrator is not
> committed yet. This document captures what the adapter needs to do, so
> when it lands the work is bounded and reviewable.

If you only want to **install and use v5.0.0 today**, you don't need
this document. The legacy v4 cycle keeps running and is unchanged. You
do get the v4.0.2 setup integrity checker as a bonus. The flag warning
in the log is expected — it documents the gap below.

---

## 1. What "wiring" means

`eo.execution.cycle.run_v5_cycle(prior_state, ctx)` takes a single
[`CycleContext`](../rootfs/usr/bin/eo/execution/cycle.py) that bundles:

- **Sensor reads** — current `BatteryState` (SOC, power, last-update
  age).
- **Per-load history** — `hours_on_per_day_last_window` for every
  declared load, computed from telemetry (InfluxDB / MariaDB /
  recorder).
- **Forecasts** — two sequences of `QuantileHourForecast` (solar, house)
  built from the trained `SolarForecaster` and `HouseForecaster`.
- **Tariff geometry** — `slot_periods` mapping each slot index to
  `"peak"` / `"mid"` / `"valley"`.
- **HA callbacks** — `is_on(entity_id) → bool`, `send_command(domain,
  service, data) → bool`, `notify_alert(msg)`.

When all of those are wired, `run_v5_cycle()` produces a `CycleResult`
that drops into `decisions.json` and the `SystemState.json` persistence
file.

The legacy monolith already has helpers for each of those pieces; the
adapter's job is to translate the existing helpers into the
`CycleContext` shape **without re-implementing** the data fetch.

---

## 2. Concrete tasks for the adapter

### 2.1 Train the forecasters

The v5 models are not trained. They cannot predict without ML weights.
The plan:

1. **Run an offline backfill** that pulls 60-90 days of hourly observed
   solar production and household consumption from InfluxDB.
2. **Compute the target factors** for the atmospheric model with
   `eo.forecasters.training.compute_atmospheric_factor(observed_kwh,
   clear_sky_kwh)`.
3. **Build feature vectors** with
   `eo.forecasters.atmospheric_factor.make_features_for_hour(...)`
   and
   `eo.forecasters.house_forecaster.make_house_features_for_hour(...)`.
   Need at least `MIN_TRAIN_SAMPLES = 50` valid samples per model
   before fit will accept the data.
4. **Train + save** with
   `eo.forecasters.training.train_atmospheric_factor_model(samples)` and
   `eo.forecasters.training.train_house_forecaster(samples)`. Save the
   joblib artefacts under `/data/forecasters/atmospheric.joblib` and
   `/data/forecasters/house.joblib`.
5. **Nightly retrain** — register an APScheduler job at 03:00 that
   repeats steps 1-4. The legacy monolith already has the cron slot;
   the adapter just plugs in the v5 training routine.

### 2.2 Implement the per-cycle data fetch

In the legacy monolith, the cycle reads sensors directly via
`ha_float()` calls. The adapter should wrap those into a `CycleContext`:

```python
def build_v5_context_from_monolith() -> CycleContext:
    # Reuse the monolith's existing helpers, do NOT re-implement.
    return CycleContext(
        now=datetime.now(timezone.utc),
        loads=tuple(_v5_load_declarations_from_wizard()),
        inverter_max_w=float(_wiz("inverter_max_w", "inverter_max_w", 5000)),
        reserved_house_load_w=float(_wiz("reserved_house_w", default=500)),
        slot_periods=_build_slot_periods_for_horizon(),
        horizon_slots=48 * 4,
        solar_forecasts=_load_solar_forecasts(),
        house_forecasts=_load_house_forecasts(),
        forecast_quality=ForecastQualityTracker.load("/data/forecast_quality.json").stats_all(window_hours=72),
        aemet_age_hours=_aemet_age_hours_or_none(),
        sensor_age_max_minutes=_max_sensor_age_minutes(),
        battery_state=BatteryState(
            soc_pct=ha_float(_wiz("battery_soc", "sensor_battery_soc"), 50.0),
            power_w=ha_float(_wiz("battery_power", "sensor_battery_power"), 0.0),
            last_updated=_sensor_last_updated("battery_soc"),
        ),
        hours_on_per_day_last_window=reconcile_load_debt(
            load_names=[l.name for l in _wizard_loads()],
            window_days=7,
            fetch_hours_on_per_day=_fetch_hours_on_for_load,
        ),
        is_on=lambda entity_id: ha_state(entity_id) and ha_state(entity_id).get("state") == "on",
        send_command=ha_service,
        notify_alert=send_telegram_alert,
    )
```

### 2.3 Replace the cycle entry point

In `energy_optimizer.py::main()`, gate the scheduler:

```python
if v5_enabled and _v5_artifacts_ready():
    scheduler.add_job(_v5_cycle_wrapper, "interval", minutes=interval, id="cycle")
else:
    scheduler.add_job(run_cycle, "interval", minutes=interval, id="cycle")
```

Where `_v5_artifacts_ready()` checks that the forecaster joblib files
exist and the forecast_quality tracker has enough samples to drive the
degraded-mode thresholds. If not, fall back to legacy with a warning so
the user sees what's missing.

`_v5_cycle_wrapper()` calls `build_v5_context_from_monolith()`, then
`run_v5_cycle()`, then persists `SystemState.json` and writes the
explanation into `decisions.json` so the existing Activity tab keeps
working.

### 2.4 Wizard UI fields

The v5 quota system has new per-load knobs that the wizard doesn't
expose yet:

- `target_hours_per_window` (number)
- `window_days` (int 1-30)
- `min_runtime_minutes` (int)
- `required_confidence_pct` (slider 0-100)
- `allow_peak_on_critical` (toggle, default off)

Add them inside the existing custom-load / pool / dishwasher panels in
the embedded JS wizard. Reading legacy `wizard_config.json` v3.5+
without these fields must still work — the v5 `LoadQuotaConfig` carries
sensible defaults.

---

## 3. Stay-safe checklist before flipping the flag

Before setting `v5_engine_enabled: true` in production:

- [ ] Forecaster artefacts exist under `/data/forecasters/`.
- [ ] `forecast_quality.json` shows at least 14 days of accumulated
      observations.
- [ ] `SystemState.json` saves and reloads cleanly across an addon
      restart.
- [ ] A dry-run cycle (with `send_command` mocked to log-only) produces
      a sane `CycleResult` and no `InvariantViolation` strings.
- [ ] `binary_sensor.energy_optimizer_setup_conflict` is `off` (no
      configuration collisions).
- [ ] Soak-test at least 48 hours on a non-critical day window. Compare
      the `policy_overrides` log against the v4 cycle's prior decisions
      and confirm the differences are intended.

---

## 4. Rolling back

If anything misbehaves, set `v5_engine_enabled: false` and restart the
addon. The legacy v4 cycle is one config flip away — no migration, no
state loss. The v5 SystemState file is preserved for the next attempt.
