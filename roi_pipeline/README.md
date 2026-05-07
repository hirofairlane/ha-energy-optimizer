# ROI Projection Pipeline (work in progress — feat/roi-projection-ml)

Replaces the linear extrapolation in the in-panel ROI calculator (which
needs >50 years of payback for any sensible expansion) with a deterministic
battery simulator backed by a year of real InfluxDB history, plus an ML
forecast layer that enables a TOU arbitrage policy.

## Stage 1 — DONE (2026-05-06)

| Script | Purpose |
|---|---|
| `01_etl.py` | Pull 12 months × 6 canonical sensors from InfluxDB v1, hourly resample, semantic ffill (limit=1h for power flows, unbounded for SOC/temp), Kirchhoff-derived `house_kwh`. Writes `data/etl_12m_1h.parquet`. |
| `02_simulator.py` | Deterministic battery simulator (capacity, eff, p_max, soc bounds) under a self-consume-first policy. Spanish 2.0TD Nufri tariff with national holiday calendar. Sweeps capacities `[0, 5, 8, 10, 12, 14, 16, 20]`. |
| `03_validate.py` | Reconstructs SOC by integrating real `battery_w` and compares to the SOC sensor (validation MAE), and contrasts self-consume vs naive TOU pre-charge policy. |

### Stage 1 results (12 months ending 2026-05-06)

| Capacity | Net cost / yr | Saved vs 0 | Self-suff |
|---|---|---|---|
| 0 (no battery) | €1726 | — | 29.1% |
| 10 kWh (current) | €1459 | €267 | 38.8% |
| 16 kWh | €1406 | €319 | 41.3% |
| 20 kWh | €1390 | €335 | 42.1% |

Marginal value of expansion at €430 / kWh installed: payback 55–62 years
across the entire range. Self-consume policy gives **+€267 / yr** for the
existing 10 kWh battery; the naive valley-charge TOU variant is
**counterproductive (−€174 / yr)** because pre-charging from grid
cannibalises the surplus-solar capacity the next day's PV would have used.

## Stage 1 known limitation (track as deuda técnica)

`03_validate.py` reports MAE = 26 % between simulated and real SOC (target
8 %), with bias only −4.8 % over 8760 h. So the simulator does not drift —
it just doesn't match individual hours. Likely causes: usable capacity
< 10 kWh after DoD/degradation, real round-trip closer to 0.85 than 0.92,
unmodelled standby losses. **Stage 2 must calibrate before forecasting.**

## Stage 2 — TODO

1. **Calibrate `BatteryConfig`** by minimising SOC MAE against real
   history. Grid search or scipy.optimize over (`capacity_eff_kwh`,
   `eff_chg`, `eff_dis`, `idle_w`). Persist the fitted config so the panel
   uses Sergio's actual battery, not nameplate.

2. **Two `GradientBoostingRegressor` forecasters** with `TimeSeriesSplit`:
   - `solar_kwh(t+1h … t+24h)` — features: hour, dow, month, sun_elev,
     temp_outdoor, lag_24h. Target R² > 0.75.
   - `house_kwh(t+1h … t+24h)` — features: hour, dow, month, is_weekend,
     temp_outdoor, lag_1h, lag_24h. Target R² > 0.55.
   - Optionally fit quantile heads (α=0.1, α=0.9) for confidence bands.

3. **Optimal TOU policy** that uses the forecasts:
   - Pre-charge from grid during P3 _only_ when
     `forecast_solar(t+24h) - forecast_house(t+24h) < reserve_target`.
     This makes valley→peak arbitrage actually profitable instead of
     cannibalising next-day solar.

4. **Re-run capacity sweep** with the calibrated simulator + optimal
   policy. Compare against stage 1 numbers.

5. **Endpoint** `/api/wizard/roi-projection` in `energy_optimizer.py` that
   exposes the curve. Wizard panel renders it in the History tab,
   replacing the linear ROI calculator.

## How to run

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python 01_etl.py          # ~30s, requires LAN access to InfluxDB
python 02_simulator.py    # reads parquet from stage 1
python 03_validate.py     # validates simulator + sweeps policies
```

`01_etl.py` requires reaching `192.168.1.131:8086` — which is **only
accessible from Sergio's LAN**. Stage 2 work that doesn't touch real data
(refactoring, model code, tests against synthetic data) can run anywhere;
final validation needs a fresh ETL on Sergio's network.
