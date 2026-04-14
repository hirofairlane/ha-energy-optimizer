# Energy Optimizer — Home Assistant Add-on

Smart energy management add-on for Home Assistant OS. Combines a **scikit-learn ML model** with dynamic electricity tariff rules to automatically control a solar battery (Huawei Luna2000), heat pump (ebusd), pool pump, pool cleaner, and dishwasher.

> **Installation:** Settings → Add-ons → Add-on store → ⋮ → Repositories → add `https://github.com/hirofairlane/ha-energy-optimizer`

---

## Table of contents

1. [Features](#features)
2. [Web panel](#web-panel)
3. [Battery charging logic](#battery-charging-logic)
4. [Savings calculation](#savings-calculation)
5. [ML model](#ml-model)
6. [Solar terrain correction](#solar-terrain-correction)
7. [InfluxDB integration](#influxdb-integration)
8. [Configuration reference](#configuration-reference)
9. [Electricity tariff](#electricity-tariff)
10. [Persistent data](#persistent-data)
11. [Changelog](#changelog)

---

## Features

| Feature | Description |
|---|---|
| **Smart valley charging** | Calculates exactly how much battery to charge from the grid at night (valley rate) to cover tomorrow's peak demand — no more, no less |
| **Solar terrain correction** | Learns your real production vs HA forecast from InfluxDB history. Corrects for local shading (hills, buildings) automatically |
| **Temperature-aware target** | Cold days = more heat pump during peak hours → charges more battery. Hot days = more cooling → same adjustment |
| **Storm protection** | Reads AEMET weather data; pre-charges to a configurable reserve when adverse weather is imminent |
| **Heat pump control** | Adjusts setpoints based on season, indoor temperature, and free solar surplus (SOC ≥ 99%) |
| **Pool pump** | Runs during solar surplus or valley tariff to meet daily/weekly runtime targets |
| **Pool cleaner** | Auto-starts with pool pump, auto-stops after 15 min (~1.5 kWh) |
| **Dishwasher** | Monitored; can be controlled to run during surplus or valley |
| **ML SOC prediction** | GradientBoostingRegressor trained on up to 365 days of InfluxDB history (R²≈0.998) |
| **Prediction accuracy chart** | Live SOC: actual vs ML-predicted (24h) + 8h forward forecast. Shows MAE badge |
| **Solar history charts** | 7-day and 12-month Actual vs HA Forecast line charts — reveals systematic forecast deviations |
| **Daily savings chart** | 7-day bar chart with € value labels; uses counterfactual method vs no-battery baseline |
| **Telegram instant alerts** | Emergency charge, storm mode, forced grid charge |
| **Daily summary** | HTML email + Telegram report at configurable time |

---

## Web panel

Four tabs, accessible via HA ingress (port 8765, no external port needed):

| Tab | Contents |
|---|---|
| 📊 Dashboard | Live KPIs · battery card with manual charge buttons · smart target reasoning · recent decision log |
| 📈 Charts | SOC actual vs predicted (24h) + MAE · Solar 7d actual vs forecast · Solar 12m actual vs forecast · Daily savings 7d |
| ⚡ Tariff | Per-day weekend config · per-hour timeline · price editor · Reset to defaults |
| ⚙️ Setup | Notification toggles · battery threshold sliders · decision interval · Data Sources debug panel |

---

## Battery charging logic

### Core principle

At night (00:00–08:00) the tariff is **valley** (cheapest: €0.1147/kWh). Importing from the grid at night is cheap — there is no benefit in using the battery to cover night consumption. The battery's job is to store cheap valley electricity so the house can avoid importing expensive **peak** electricity (€0.2234/kWh) the next day.

> **The question the system answers:**  
> *"How full does the battery need to be at the start of tomorrow's peak hours so I never need to import from the grid at peak prices?"*

### Step-by-step calculation

**1. Solar forecast (terrain-corrected)**

```
solar_forecast_raw   = sensor.energy_production_tomorrow   (HA / PVforecast)
terrain_factor       = median(actual_daily / ha_forecast_daily)  over last 30 days
solar_tomorrow       = solar_forecast_raw × terrain_factor
```

The terrain factor is computed automatically from InfluxDB history. If a hill blocks your afternoon sun and HA consistently over-predicts by 15%, the factor will converge to ~0.85. See [Solar terrain correction](#solar-terrain-correction).

**2. Solar energy available during peak hours**

Peak tariff applies 10:00–14:00 and 18:00–22:00 (8 hours/day on weekdays).

- The window 10:00–15:00 overlaps with the morning peak. Roughly **45% of daily solar production** falls in this window.
- The evening peak (18:00–22:00) receives near-zero solar in mainland Spain regardless of season.

```
solar_during_peak = solar_tomorrow × 0.45
```

**3. Peak consumption estimate**

Base load is estimated from the average grid power during the last 14 nights (22:00–08:00), queried from InfluxDB. Night hours are used because:
- No solar interference
- No big appliance cycles
- Result reflects the true household base load

```
peak_base_kwh = base_load_kW × 8   (8 peak hours)
```

**4. Temperature correction**

The heat pump (ebusd) is the dominant variable load. Cold days mean it runs harder and longer during peak hours:

| Outdoor temperature | Correction |
|---|---|
| < 5 °C | +3.0 kWh |
| 5–10 °C | +2.0 kWh |
| 10–15 °C | +1.0 kWh |
| 15–25 °C | 0 kWh |
| 25–30 °C | +0.5 kWh |
| > 30 °C | +1.5 kWh |

```
peak_total_kwh = peak_base_kwh + temperature_correction_kwh
```

**5. Battery charge target**

```
battery_gap_kwh = max(0,  peak_total_kwh − solar_during_peak_kwh)
target_SOC      = (battery_gap_kwh / battery_capacity_kWh) × 100 + 5%  (safety buffer)
target_SOC      = clamp(target_SOC, 30%, 95%)
```

The **30% floor** ensures the battery always has a minimum reserve. The **5% safety buffer** absorbs forecast errors.

### Example

> Outdoor temp: 8 °C · Solar tomorrow (corrected): 9.0 kWh · Base load: 0.98 kW · Battery: 10 kWh

```
solar_during_peak = 9.0 × 0.45 = 4.05 kWh
peak_base_kwh     = 0.98 × 8   = 7.84 kWh
temp_correction   =              2.00 kWh  (8 °C → 5–10 °C bracket)
peak_total_kwh    =              9.84 kWh
battery_gap_kwh   = 9.84 − 4.05 = 5.79 kWh
target_SOC        = (5.79 / 10) × 100 + 5 = 63%
```

The Dashboard "Smart target" line shows the full breakdown in real time.

### Other charging rules

| Situation | Action |
|---|---|
| SOC < emergency threshold (default 10%) | Force-charge at any tariff, any time |
| Storm forecast (AEMET) | Pre-charge to storm threshold (default 80%) |
| Valley + SOC below smart target | Charge at configured power (default ~3 kW) |
| Peak tariff | No grid charging under any normal circumstance |
| SOC ≥ 99% (free solar surplus) | Heat pump boost / pool pump starts |

---

## Savings calculation

### Methodology: counterfactual baseline

The system calculates savings by comparing **what actually happened** against the counterfactual of **no smart battery management** — i.e., if the battery were absent, the house would have imported everything from the grid at spot price and exported surplus solar at export price.

For every 15-minute decision cycle:

```
# Sensor convention:
#   grid_power    > 0 → importing from grid
#   grid_power    < 0 → exporting to grid
#   battery_power > 0 → battery charging
#   battery_power < 0 → battery discharging

grid_without_battery = grid_power − battery_power

def energy_cost(g, import_price, export_price):
    if g > 0:
        return g × (interval_hours / 1000) × import_price    # paying for import
    else:
        return g × (interval_hours / 1000) × export_price    # receiving for export

saving_interval = energy_cost(grid_without_battery) − energy_cost(grid_power)
```

Daily savings are the sum of all positive (and occasional negative) interval savings.

### What this captures

| Scenario | Effect |
|---|---|
| Battery discharging at peak → house self-sufficient | `grid_without` > 0, `grid_actual` ≈ 0 → large positive saving |
| Battery charged from solar at midday, discharged at peak | Captured across the full cycle |
| Battery charged from grid at valley, discharged at peak | Saving = peak_price − valley_price per kWh ≈ €0.11/kWh |
| Battery management made no difference in a cycle | saving ≈ 0 |
| Model charged when unnecessary (rare) | Small negative saving — honest accounting |

### Savings reference prices (2.0TD Spain — all-in, taxes included)

| Period | Hours | Price |
|---|---|---|
| Peak (Punta) | Weekdays 10–14h, 18–22h | €0.2234/kWh |
| Shoulder (Llano) | Weekdays 08–10h, 14–18h, 22–00h | €0.1483/kWh |
| Valley (Valle) | 00–08h · All day weekends | €0.1147/kWh |
| Export (Excedentes) | — | €0.040/kWh |

Prices are stored in `/data/tariff.json`. Use "↩ Reset to defaults" in the Tariff tab to restore these values.

---

## ML model

### What it predicts

A **GradientBoostingRegressor** predicts the battery SOC value for the current moment, given recent sensor readings. It is not a multi-step planner — it predicts "what the SOC should be right now" from features, which is then used as a sanity check and to populate the predicted-SOC line in the charts.

### Features

| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23) — captures daily production patterns |
| `weekday` | Day of week (0–6) — captures pool/dishwasher schedules |
| `month` | Month (1–12) — captures seasonal solar variation |
| `lag1` | SOC 15 min ago |
| `lag4` | SOC 1 h ago |
| `roll4` | Rolling mean SOC over last 1 h |
| `solar_proxy` | **Continuous** sun elevation angle 0–1 (see below) |

### Solar proxy: continuous elevation angle

Before v2.6.1, `solar_proxy` was a binary flag (1 if 08:00–19:00, 0 otherwise). This treated dawn, noon, and dusk identically.

Since v2.6.1, it uses a geometric calculation for latitude 40.67°N (Guadarrama, Madrid):

```python
declination = 23.45° × sin(360/365 × (day_of_year − 81))
hour_angle  = 15° × (hour − 13.5)   # solar noon ≈ 13:30 local in Madrid
elevation   = arcsin(sin(lat) × sin(decl) + cos(lat) × cos(decl) × cos(h_angle))
solar_proxy = max(0, elevation) / 70°   # 70° ≈ peak summer elevation at this lat
```

Resulting values (June, 40.67°N):

| Hour | Elevation | Proxy |
|---|---|---|
| 07:00 | ~5° | 0.07 |
| 09:00 | ~35° | 0.50 |
| 13:30 | ~72° | 1.00 |
| 17:00 | ~35° | 0.50 |
| 19:00 | ~8° | 0.11 |
| 20:00 | ~0° | 0.00 |

This means the model can distinguish "afternoon with hill shadow" from "midday full sun" — the `hour` feature already captures the terrain effect implicitly through historical lag data, but the continuous proxy adds an explicit geometric signal.

### Training

- **Data source:** InfluxDB first (up to 365 days); falls back to HA recorder (progressive: 60→30→14→7 days)
- **Schedule:** nightly at 03:00 (configurable via `retrain_cron`)
- **Pipeline:** `StandardScaler → GradientBoostingRegressor(n_estimators=150, max_depth=4)`
- **Validation:** 3-fold cross-validation R² reported in Dashboard KPI
- **Auto-retrain on feature change:** `MODEL_FEATURE_VER` constant; if the saved model was trained with an older feature set, a background retrain is triggered automatically on the next cycle

### 8-hour forward forecast

The Charts tab shows a green dashed line extending 8 hours into the future. It is computed by chaining single-step predictions:

```
for h in 1..8:
    solar_proxy_h = elevation_angle(now + h hours)
    soc_h         = model.predict([hour, weekday, month, soc_{h-1}, soc_{h-1}, soc_{h-1}, solar_proxy_h])
```

Lag features are updated with each predicted value, so compounding errors grow with horizon. The line is indicative, not a precise forecast.

---

## Solar terrain correction

HA solar forecast sensors (e.g. Forecast.Solar) use panel orientation and installed capacity but have no knowledge of local terrain. A hill, building, or tree can block the sun during specific hours every day, causing a **systematic over-prediction** that no amount of cloud adjustment will fix.

### How the correction is computed

Every 6 hours the system queries InfluxDB for the last 30 days of:
- `energy_production_today` — actual production (MAX per UTC day = full-day total)
- `energy_production_tomorrow` — HA forecast (LAST value per day = the forecast for the next day)

```
ratios = [actual_day_D / forecast_day_{D-1} for each day D]
terrain_factor = median(ratios)   # median is robust against cloudy-day outliers
terrain_factor = clamp(terrain_factor, 0.30, 1.50)
```

At least 7 days of data are required before the factor deviates from 1.0.

### Where it is applied

1. `calculate_optimal_soc()` — corrected solar feeds the charge target calculation
2. Dashboard "Solar tomorrow" KPI — shows "Terrain factor: XX%" when ≠ 100%
3. Decision log — charge reason includes terrain-corrected solar value

### Example

> HA forecast: 9.5 kWh · Terrain factor: 0.87 (hill blocks 13% of afternoon production)  
> Corrected forecast used for decisions: 9.5 × 0.87 = **8.3 kWh**

Without correction the system would assume more solar than it gets, potentially under-charging at night.

---

## InfluxDB integration

InfluxDB (HAOS add-on v5.0.2, running InfluxDB 1.8.x) is the **primary data source** for ML training and for all multi-day chart data. HA recorder is only a fallback.

### Connection

| Parameter | Default | Notes |
|---|---|---|
| `influxdb_url` | `http://172.30.32.1:8086` | HA supervisor bridge IP |
| `influxdb_db` | `homeassistant` | |
| `influxdb_user` | _(empty)_ | Leave empty if auth is disabled |
| `influxdb_password` | _(empty)_ | |

Auth is auto-detected: the add-on tries with credentials first; if InfluxDB returns 401 (auth disabled on server), it retries without credentials.

### Data format (old HA integration)

The HA→InfluxDB integration (pre-2023) stores data as:
- **Measurement** = unit of the sensor (e.g. `%`, `W`, `kWh`, `°C`)
- **Tag** `entity_id` = sensor name **without domain prefix** (e.g. `battery_state_of_capacity`, not `sensor.battery_state_of_capacity`)

All InfluxDB queries in this add-on strip the domain prefix accordingly.

### What is queried

| Data | Query | Used for |
|---|---|---|
| Battery SOC history | `MAX("value")` per 15 min, last 365 days | ML training |
| Grid power history | `MAX("value")` per raw point, last 14 days | Night consumption estimate |
| Solar actual (daily) | `MAX("value") GROUP BY time(1d)` | Solar 7d/12m charts, terrain correction |
| Solar forecast (daily) | `LAST("value") GROUP BY time(1d)` shifted +1d | Solar charts, terrain correction |
| Solar yesterday | `MAX("value")` for previous UTC day | Charts |

### Setup tab — Data Sources debug panel

The **Setup** tab has a **Test connections** button that runs a live diagnostic:
- InfluxDB reachability and auth mode
- Available databases
- Measurements list
- Entity tag discovery for the battery SOC sensor
- Sample record count (7 days)

---

## Configuration reference

### Sensors

| Option | Description | Default |
|---|---|---|
| `sensor_battery_soc` | Battery state of charge (%) | `sensor.battery_state_of_capacity` |
| `sensor_battery_power` | Charge/discharge power (W, +ve=charge) | `sensor.battery_charge_discharge_power` |
| `sensor_grid_power` | Grid meter (W, +ve=import) | `sensor.acometida_general_power` |
| `sensor_solar_current_hour` | Solar production this hour (kWh) | `sensor.energy_current_hour` |
| `sensor_solar_next_hour` | Solar forecast next hour (kWh) | `sensor.energy_next_hour` |
| `sensor_solar_today` | Cumulative production today (kWh) | `sensor.energy_production_today` |
| `sensor_solar_tomorrow` | Forecast for tomorrow (kWh) | `sensor.energy_production_tomorrow` |
| `sensor_temp_outdoor` | Outdoor temperature (°C) | `sensor.ebusd_broadcast_outsidetemp_temp2` |
| `sensor_temp_salon` | Indoor temperature (°C) | `sensor.media_salon` |
| `sensor_weather` | AEMET weather entity | `weather.aemet` |

### Actuators

| Option | Description | Default |
|---|---|---|
| `switch_pool` | Pool pump switch | `switch.depuradora` |
| `switch_pool_cleaner` | Pool cleaner switch | `switch.limpiafondos` |
| `number_hvac_cool` | Heat pump cooling setpoint | `number.ebusd_ctls2_z1coolingtemp_tempv` |
| `number_hvac_heat` | Heat pump heating setpoint | `number.ebusd_ctls2_z1manualtemp_tempv` |
| `number_battery_charge_cutoff` | Battery grid charge cutoff SOC | `number.battery_grid_charge_cutoff_soc` |
| `select_battery_mode` | Battery working mode select | `select.battery_working_mode` |

### Battery thresholds

| Option | Default | Description |
|---|---|---|
| `battery_emergency_threshold` | 10% | Force-charge below this SOC at any tariff |
| `battery_low_threshold` | 30% | Low battery warning level |
| `battery_medium_threshold` | 50% | Medium battery level |
| `battery_storm_threshold` | 80% | Pre-charge target when storm is forecast |
| `battery_capacity_kwh` | 10.0 | Total usable battery capacity |

### Scheduling

| Option | Default | Description |
|---|---|---|
| `decision_interval_minutes` | 15 | How often the optimization cycle runs |
| `retrain_cron` | `0 3 * * *` | Nightly ML retrain schedule |
| `summer_start_month` | 6 | First month of summer mode |
| `summer_end_month` | 9 | Last month of summer mode |

### Notifications

| Option | Default | Description |
|---|---|---|
| `notify_email_service` | `correo_gmail_com` | HA notify service name for email |
| `notify_email_target` | _(empty)_ | Recipient email — set in HA add-on config UI, not in repo |
| `notify_telegram_service` | `telegramsergio` | HA notify service name for Telegram |
| `notify_daily_time` | `23:00` | Time to send daily summary |
| `notify_email_enabled` | true | Enable email daily summary |
| `notify_telegram_daily_enabled` | true | Enable Telegram daily summary |
| `notify_telegram_alerts_enabled` | true | Enable instant Telegram alerts |

---

## Electricity tariff

Default prices — **Spain 2.0TD (Energía Nufri), all costs prorated including taxes and IVA:**

| Period | Hours | Price |
|---|---|---|
| Peak (Punta) | Weekdays 10–14h, 18–22h | **€0.2234/kWh** |
| Shoulder (Llano) | Weekdays 08–10h, 14–18h, 22–00h | **€0.1483/kWh** |
| Valley (Valle) | 00–08h + all day weekends | **€0.1147/kWh** |
| Export (Excedentes) | — | **€0.040/kWh** |

Weekends can be configured per day-of-week in the **Tariff** tab. Prices are editable per hour slot. Use "↩ Reset to defaults" to restore the above values.

---

## Persistent data

All data lives in `/data/` inside the add-on container (persists across restarts and updates):

| File | Contents |
|---|---|
| `model.pkl` | Trained scikit-learn pipeline + metadata (R², n_samples, feature_version) |
| `decisions.json` | Last 500 decision cycles — full sensor snapshot, tariff, actions, prediction |
| `savings.json` | Cumulative kWh avoided at peak + EUR saved since first run |
| `tariff.json` | Custom tariff configuration (periods, prices, weekend days) |
| `setup.json` | GUI runtime overrides — notification toggles, threshold sliders |

---

## Changelog

### v2.6.3
- **Battery logic reoriented:** target SOC now based on covering tomorrow's PEAK demand (10-14h + 18-22h), not overnight consumption. Night is valley — cheap grid import is fine.
- **Temperature correction:** outdoor temperature adjusts peak consumption estimate for heat pump / cooling load (+0 to +3 kWh depending on bracket).
- Dashboard "Smart target" line shows full reasoning: peak demand · solar peak · battery gap · temp adjustment.

### v2.6.2
- Fixed night consumption estimate: `sensor.acometida_general_power` was absent from HA recorder. Now queries InfluxDB first (320k+ records, 14d). Logs sample count for transparency.

### v2.6.1
- Continuous solar proxy: replaced binary `solar_proxy` (0/1) with geometric sun elevation angle (0.0–1.0) for latitude 40.67°N. Model now distinguishes dawn/noon/dusk.
- Solar terrain correction factor: median actual/forecast ratio from InfluxDB (30d), cached 6h, applied to all solar-dependent decisions. Shown as "Terrain factor: XX%" in Dashboard.
- Auto-retrain on feature version mismatch (`MODEL_FEATURE_VER`).

### v2.6.0
- Solar charts redesigned: two line charts (7d daily + 12m monthly), both Actual vs HA Forecast. Data from InfluxDB.
- Savings bar chart: value labels (€X.XX) on each bar.
- Savings formula: counterfactual grid-without-battery method. Previously only counted peak discharge events; now captures the full charge/discharge cycle value.

### v2.5.8
- SOC predicted line was invisible: alignment now uses a 15-min bucket grid as common x-axis. Both actual (InfluxDB) and predicted (decisions.json) aligned by nearest timestamp (±7.5 min).
- MAE badge on SOC chart title.
- Solar "Today (forecast)" bar: yesterday's `solar_tomorrow` reading vs today's actual.

### v2.5.7
- `notify_email_target` default changed to `""` — email address removed from public repo. Set in HA add-on Configuration UI.

### v2.5.6
- InfluxDB as primary ML data source (365 days, R²=0.998 vs 0.066 with HA recorder).
- Auth auto-detection: retries without credentials if InfluxDB returns 401.
- Entity ID domain-prefix stripping for old HA integration format.
- Data Sources debug panel in Setup tab.
- Progressive history fallback: 60→30→14→7 days for HA recorder.
- Charts: InfluxDB-backed actual SOC + 8h chained ML forecast.

### v2.5 – v2.4
- Weather forecast widget (AEMET, 5-day, storm alert).
- Real all-in tariff prices (2.0TD Spain).
- Solar proxy using actual `sun.sun` elevation.
- HA ingress fix: `__BASE__` replaced at request time with `X-Ingress-Path` header.

### v2.3
- Ingress base-path fix for GUI buttons and charts.
- Tariff editor: per-day weekend configuration.

### v2.1 – v2.2
- Initial public release.
- Telegram instant alerts.
- 4-tab GUI: Dashboard, Charts, Tariff, Setup.
