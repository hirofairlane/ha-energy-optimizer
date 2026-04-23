# Energy Optimizer — Home Assistant Add-on

Smart energy management add-on for Home Assistant OS. Its primary goal is to **minimise your electricity bill** — whether you have a solar battery, a set of schedulable loads (heat pump, EV charger, pool pump, irrigation…), or both. A **scikit-learn ML model** combined with dynamic tariff rules decides in real time when to charge or discharge the battery, when to shift loads to cheap/solar windows, and how much to pre-charge at night — so you import as little peak-rate energy as possible and export as little solar as possible.

> 🔒 **100% local — no cloud, no telemetry.** Everything runs inside the Docker container on your own hardware. Your energy data, sensor readings, and ML model never leave your home network. The add-on only communicates with your local Home Assistant instance and, optionally, a local InfluxDB instance.

> **Installation:** Settings → Add-ons → Add-on store → ⋮ → Repositories → add `https://github.com/hirofairlane/ha-energy-optimizer`

### Python dependencies (bundled in Docker image)

| Library | Version | Purpose |
|---|---|---|
| **scikit-learn** | ≥ 1.3 | `GradientBoostingRegressor` + `Pipeline` + `StandardScaler` for SOC prediction |
| Flask | ≥ 3.0 | Internal API and web panel (port 8765) |
| APScheduler | ≥ 3.10 | Decision cycle and retrain scheduling |
| requests | — | Home Assistant REST API client |
| numpy / pandas | — | Feature engineering and time-series resampling |

---

## Table of contents

1. [Installation](#installation)
2. [Features](#features)
3. [Setup Wizard](#setup-wizard)
4. [Web panel](#web-panel)
5. [Energy flow diagram](#energy-flow-diagram)
6. [Battery charging logic](#battery-charging-logic)
7. [Savings calculation](#savings-calculation)
8. [ML model](#ml-model)
9. [Solar terrain correction](#solar-terrain-correction)
10. [InfluxDB integration](#influxdb-integration)
11. [Configuration reference](#configuration-reference)
12. [Electricity tariff](#electricity-tariff)
13. [Persistent data](#persistent-data)
14. [Changelog](#changelog)

---

## Installation

**Requirements:** Home Assistant OS or Supervised (any architecture: amd64, aarch64, armv7).

1. In HA go to **Settings → Add-ons → Add-on store → ⋮ menu → Repositories** and add:
   ```
   https://github.com/hirofairlane/ha-energy-optimizer
   ```
2. Find **Energy Optimizer** in the store and click **Install**.
3. Start the add-on and open the web panel — the **Setup Wizard** will guide you through the rest.

No YAML editing required. All configuration is done through the wizard and the web panel.

### What you need

| | Required | Optional but recommended |
|---|---|---|
| **Battery** | Any smart battery with HA entities for SOC + charge/discharge power | Working mode select, cutoff SOC, force-charge switch (Huawei Luna2000 natively supported) |
| **Solar** | Forecast.Solar or PVforecast integration | Real-time production sensor |
| **History** | HA Recorder (14-day window) | InfluxDB for 90-day ML training window |
| **Schedulable loads** | At least one (heat pump, pool pump, EV charger, or any `switch.*`) | Multiple loads — each adds a scheduling optimisation layer |

The add-on is useful even with **no battery** if you have loads you can shift: it will schedule them to coincide with solar surplus or valley-tariff windows.

---

## Features

| Feature | Description |
|---|---|
| **Setup Wizard** | 8-step guided configuration — auto-discovers your HA entities with ML-based scoring. No YAML editing required |
| **Smart valley charging** | Calculates exactly how much battery to charge from the grid at night to cover tomorrow's peak demand — no more, no less |
| **Energy flow diagram** | Animated SVG showing real-time power flows between Solar, Battery, Grid, and House nodes |
| **Live 7-day averages** | Each sensor card shows instantaneous value + Ø7d rolling average (solar: daylight hours only) |
| **Solar terrain correction** | Learns your real production vs HA forecast from InfluxDB history — corrects for local shading automatically |
| **Temperature-aware target** | Cold days = more heat pump during peak hours → charges more battery |
| **Storm protection** | Reads weather entity; pre-charges to a configurable reserve when adverse weather is imminent |
| **Heat pump control** | Adjusts setpoints based on season, indoor temperature, and free solar surplus (SOC ≥ 99%) |
| **Pool pump** | Runs during solar surplus or valley tariff to meet daily/weekly runtime targets |
| **Pool cleaner** | Auto-starts with pool pump, auto-stops after 15 min (~1.5 kWh) |
| **Dishwasher** | Monitored; recommendation to run during surplus or valley |
| **ML SOC prediction** | GradientBoostingRegressor trained on up to 90 days of history. Dynamic features from wizard config |
| **Prediction accuracy chart** | Live SOC: actual vs ML-predicted (24h) + 8h forward forecast. Shows MAE badge |
| **Solar history charts** | 7-day and 12-month Actual vs HA Forecast line charts |
| **Daily savings chart** | 7-day bar chart with € value labels; counterfactual method |
| **Telegram instant alerts** | Emergency charge, storm mode, forced grid charge |
| **Daily summary** | HTML email + Telegram report at configurable time |
| **Debug section** | Tweaks tab shows how each sensor role resolves (wizard → options → fallback) with live HA value |

---

## Setup Wizard

The wizard is the recommended way to configure the add-on. It runs through 8 steps and auto-discovers your HA entities using a keyword + device class + unit scoring algorithm.


<img width="969" height="267" alt="Data → Location → Grid → Solar → Battery → Loads → Tariff → Done" src="https://github.com/user-attachments/assets/ca98e827-4d50-4e98-99ca-a615ee265e30" />


### Steps

| Step | What it configures |
|---|---|
| **Data** | History source: InfluxDB v2 (recommended, 90 days) or HA Recorder (14 days) |
| **Location** | GPS coordinates and timezone — used for solar geometry and ML features |
| **Grid** | Grid meter sensor (import/export) |
| **Solar** | Production sensor + Forecast.Solar sensors (today/tomorrow/hourly) |
| **Battery** | SOC sensor, charge/discharge power, working mode select, charge cutoff, backup SOC, force charge switch |
| **Loads** | Per-device sub-wizards for each enabled appliance (see [Supported loads](#supported-loads)) |
| **Tariff** | Contracted power, tariff type, peak/shoulder/valley prices |
| **Done** | Data quality score summary and save |

### Data Quality Thermometer

The wizard header shows a live quality score (0–100%):

| Score | Meaning |
|---|---|
| ≥ 70% 🟢 | Enough data for reliable ML predictions |
| 40–70% 🟠 | Partial data — predictions will work but with higher uncertainty |
| < 40% 🔴 | Insufficient data — add InfluxDB or wait for recorder history to accumulate |

Score factors: history source (30 pts) + key sensor coverage (25 pts) + sample count bonus (30 pts) + optional sensors (15 pts).

### Entity auto-discovery

For each sensor role, the wizard queries all HA entities and scores them:

| Signal | Points |
|---|---|
| Matching `device_class` | +50 |
| Matching unit of measurement | +40 |
| Each matching keyword in entity ID | +25 |

Top candidates are shown as clickable cards. The selected entity is saved to `wizard_config.json` and takes precedence over `options.json` for all decisions.

### Supported loads

The Loads step shows a card for each appliance type. Select the ones present in your installation — the wizard then walks through a sub-wizard for each one. Each selected device also appears as a small emoji dot in the wizard navigation bar.

| Load | Icon | What you configure | What the engine does |
|---|---|---|---|
| **HVAC** | 🌡️ | Climate entity or heat/cool setpoint numbers per zone. 24h schedule with Comfort 🟢 / Surplus 🔵 / Minimum ⚫ temperature tiers | Raises/lowers setpoints based on tariff period, solar surplus (SOC ≥ 99%), and indoor temperature. Multi-zone supported |
| **Pool pump** | 🏊 | Pool switch + optional runtime sensors (daily/weekly hours) | Runs during solar surplus (SOC ≥ 99%) or valley tariff to meet runtime targets |
| **Pool cleaner** | 🤿 | Cleaner switch entity | Auto-starts with the pool pump, auto-stops after 15 min (~1.5 kWh) |
| **Dishwasher** | 🍽️ | State sensor + optional switch | Monitors cycle state; recommends (or triggers) start during solar surplus or valley |
| **Washing machine** | 👕 | State sensor + power meter | Monitors cycle; recommends start during cheapest/greenest window |
| **Dryer** | 🌀 | State sensor + power meter | Same as washing machine |
| **EV Charger** | 🚗 | Switch or number entity (Wallbox, OCPP, Zappi…) | Schedules charging during valley tariff or solar surplus |
| **Custom** | ⚙️ | Any `switch.*` entity + estimated watts + schedule preference | Schedules any switch-controlled load (irrigation pump, water heater, etc.) according to the selected window |

#### Custom load scheduling options

When adding a Custom load you choose one of four scheduling modes:

| Mode | When the switch is turned on |
|---|---|
| 🌙 **Valley tariff only** | During the cheapest grid tariff window (typically 00:00–08:00) |
| ☀️ **Solar surplus only** | When solar production exceeds house consumption (SOC ≥ 99%) |
| ☀️🌙 **Solar + Valley** | Either of the above |
| ⏱ **Custom hours** | A fixed time range you specify (e.g. `10-14,22-06`) |

Multiple Custom loads can be added (e.g. irrigation pump + water heater), each with its own entity, wattage, and schedule.

---

## Web panel

Five tabs, accessible via HA ingress (port 8765, no external port needed):

| Tab | Contents |
|---|---|
| 📊 **Dashboard** | Live KPIs · 4-card power panel · animated energy flow diagram · battery card with manual charge buttons · smart target reasoning · recent decision log |
| 📈 **Charts** | SOC actual vs predicted (24h) + MAE · Solar 7d actual vs forecast · Solar 12m actual vs forecast · Daily savings 7d · Power flow 24h |
| ⚡ **Tariff** | Per-day weekend config · per-hour timeline · price editor · Reset to defaults |
| ⚙️ **Setup (Tweaks)** | Notification toggles · battery threshold sliders · decision interval · Data Sources connectivity test · **Debug: sensor resolution table** |
| 🧙 **Wizard** | Full setup wizard (see above) |

### Setup (Tweaks) tab

The Tweaks tab is the runtime control panel — all changes take effect immediately without restarting the add-on.

| Section | What it does |
|---|---|
| **Notifications** | Enable/disable email daily summary, Telegram daily summary, and instant Telegram alerts individually. |
| **Battery thresholds** | Sliders for emergency, low, medium, and storm SOC thresholds. Adjust without editing `config.yaml`. |
| **Decision interval** | How often (in minutes) the optimization engine runs. Default 15 min. |
| **Data Sources** | Connectivity test for InfluxDB and HA Recorder — shows last-read timestamp, row count, and active/fallback status. |
| **Debug: sensor resolution** | Table that auto-loads when you open the tab and shows exactly how every sensor role was resolved. A **Refresh** button re-reads live HA values. |

#### Debug sensor resolution table

| Column | Meaning |
|---|---|
| **Role** | Internal name (e.g. `solar_power`, `battery_soc`) |
| **Entity ID** | The HA entity resolved for this role |
| **Source** | `wizard` (wizard_config.json) · `options` (config.yaml) · `fallback` (built-in default) |
| **Value** | Current state read from HA at refresh time |
| **Status** | ✓ valid reading · ⚠ entity exists but value is unexpected · ✗ not found or unavailable |

This is the first place to look when a Dashboard card shows 0 W or "unavailable" — it shows whether the problem is entity resolution or a real sensor issue.

---

### Live power panel

Four cards updated every 30 seconds, each showing instantaneous value + Ø7d rolling average:

| Card | Color | Average logic |
|---|---|---|
| ☀️ Solar | Yellow | Mean only when solar > 0 W (excludes night) |
| ⚡ Grid | Green (export) / Red (import) | Net mean (positive = selling) |
| 🔋 Battery | Green (charging) / Red (discharging) | Net mean |
| 🏠 House | Orange | Mean of consumption samples > 0 |

---

## Energy flow diagram

An animated SVG diagram shows real-time power flows between four nodes:

```
        ☀️ Solar
       /    \
   🔋 Bat  ⚡ Grid
       \    /
        🏠 Casa
```

### Sensor conventions

| Sensor | Convention |
|---|---|
| `solar` | Always ≥ 0 W |
| `grid` | Positive = exporting (selling), Negative = importing (buying) |
| `battery` | Positive = charging (energy IN to battery), Negative = discharging (energy OUT) |

**House balance:** `P_casa = solar − grid − battery`

### Flow priority order

```
A. Solar → Casa  (first priority)
   Solar → Battery  (surplus charging)
   Solar → Grid  (remaining export)

B. Battery → Casa  (discharge covers remaining house load)
   Battery → Grid  (excess discharge if any)

C. Grid → Battery  (emergency: grid covers what solar can't charge)
   Grid → Casa  (grid covers remaining house load)
```

Each line is animated only when its flow exceeds 10 W. Animation speed is proportional to power:
`speed = max(0.35, 2.4 − power/1800)` seconds per cycle.

---

## Battery charging logic

### Core principle

At night (00:00–08:00) the tariff is **valley** (cheapest). Importing from the grid at night is cheap — the battery's job is to store cheap valley electricity so the house can avoid importing expensive **peak** electricity the next day.

> **The question the system answers:**  
> *"How full does the battery need to be at the start of tomorrow's peak hours so I never need to import from the grid at peak prices?"*

### Step-by-step calculation

**1. Solar forecast (terrain-corrected)**

```
solar_forecast_raw = tomorrow's production forecast (kWh)
terrain_factor     = median(actual_day / forecast_day) over last 30 days
solar_tomorrow     = solar_forecast_raw × terrain_factor
```

**2. Solar during peak hours**

~45% of daily production falls in the 10:00–15:00 morning peak window:
```
solar_during_peak = solar_tomorrow × 0.45
```

**3. Peak consumption estimate**

From average grid power during the last 14 nights (22:00–08:00) via InfluxDB:
```
peak_base_kwh = base_load_kW × 8   (8 peak hours)
```

**4. Temperature correction**

| Outdoor temperature | Correction |
|---|---|
| < 5 °C | +3.0 kWh |
| 5–10 °C | +2.0 kWh |
| 10–15 °C | +1.0 kWh |
| 15–25 °C | 0 kWh |
| 25–30 °C | +0.5 kWh |
| > 30 °C | +1.5 kWh |

**5. Battery charge target**

```
battery_gap_kwh = max(0, peak_total_kwh − solar_during_peak_kwh)
target_SOC      = (battery_gap_kwh / battery_capacity_kWh) × 100 + 5%
target_SOC      = clamp(target_SOC, 30%, 95%)
```

The Dashboard "Smart target" line shows the full breakdown in real time.

### Other charging rules

| Situation | Action |
|---|---|
| SOC < emergency threshold (default 10%) | Force-charge at any tariff, any time |
| Storm forecast | Pre-charge to storm threshold (default 80%) |
| Valley + SOC below smart target | Charge at configured power |
| Peak tariff | No grid charging under any normal circumstance |
| SOC ≥ 99% (free solar surplus) | Heat pump boost / pool pump starts |

---

## Savings calculation

### Methodology: counterfactual baseline

For every 15-minute decision cycle:

```python
# Sensor conventions:
#   grid_power    < 0 → buying from grid (import)
#   grid_power    > 0 → selling to grid  (export)
#   battery_power > 0 → battery charging
#   battery_power < 0 → battery discharging

# Without battery: what would the grid meter read?
grid_without_battery = grid_power + battery_power

def energy_cost(g, import_price, export_price):
    if g < 0:
        return -g × interval_hours × import_price / 1000   # cost of import
    else:
        return -g × interval_hours × export_price / 1000   # income from export

saving = energy_cost(grid_without_battery) - energy_cost(grid_power)
```

Daily savings are the sum of all interval savings.

---

## ML model (scikit-learn)

**scikit-learn** is bundled inside the Docker image — no manual installation needed.

### What it predicts

A `GradientBoostingRegressor` wrapped in a scikit-learn `Pipeline(StandardScaler → GBR)` predicts the battery SOC for the current moment from recent sensor readings. Used as a sanity check and to populate the predicted-SOC line in Charts.

### Dynamic features

The feature set is built from the sensors configured in the wizard. If a sensor has no history, it is excluded. The feature list is saved alongside the model:

| Feature | Condition |
|---|---|
| `hour`, `weekday`, `month` | Always |
| `lag1`, `lag4`, `roll4` (SOC lags) | Always |
| `solar_proxy` (geometric sun elevation 0.0–1.0) | Always |
| `temp_out`, `temp_out_lag4` | If outdoor temp sensor has history |
| `solar_lag1`, `solar_roll4` | If solar sensor has history |
| `grid_lag1`, `grid_roll4`, `grid_abs_lag1` | If grid sensor has history |
| `sm_{name}_lag1` | For each configured sub-meter |

### Training

- **Source:** InfluxDB v2 (wizard config, 90 days) → InfluxDB v1 (options, 60 days) → HA Recorder (14 days)
- **Schedule:** nightly at 03:00 (`retrain_cron` option)
- **Pipeline:** `StandardScaler → GradientBoostingRegressor(n_estimators=150, max_depth=4)`
- **Validation:** 3-fold cross-validation R² shown in Dashboard
- **Auto-retrain:** triggers when feature version changes

### 8-hour forward forecast

Computed by chaining single-step predictions, updating lag features with each predicted value. Shown as a dashed line in Charts.

---

## Solar terrain correction

HA solar forecast sensors (Forecast.Solar, etc.) use panel orientation and capacity but have no knowledge of local terrain. Hills or buildings cause systematic over-prediction that the system learns and corrects automatically.

Every 6 hours, from 30 days of InfluxDB history:
```
ratios = [actual_day_D / forecast_day_{D-1} for each day D]
terrain_factor = median(ratios)
terrain_factor = clamp(terrain_factor, 0.30, 1.50)
```

The median is robust against cloudy-day outliers. Requires at least 7 days of data. Shown as "Terrain factor: XX%" in the Dashboard.

---

## InfluxDB integration

InfluxDB is the primary data source for ML training and multi-day charts. HA Recorder is the fallback.

### Connection

| Parameter | Default | Notes |
|---|---|---|
| `influxdb_url` | `http://172.30.32.1:8086` | HA supervisor bridge IP (standard for HAOS) |
| `influxdb_db` | `homeassistant` | |
| `influxdb_user` | _(empty)_ | Leave empty if auth disabled |
| `influxdb_password` | _(empty)_ | |

Auth is auto-detected: tries with credentials first, retries without if InfluxDB returns 401.

### Data format

The HA→InfluxDB integration (pre-2023) stores data as:
- **Measurement** = unit of the sensor (`%`, `W`, `kWh`, `°C`)
- **Tag** `entity_id` = sensor name **without domain prefix** (e.g. `battery_state_of_capacity`)

---

## Configuration reference

All options can be set in the HA add-on Configuration UI. The wizard saves its own config to `/data/wizard_config.json` which takes priority over these options for all sensor lookups.

### Sensors

| Option | Default | Description |
|---|---|---|
| `sensor_battery_soc` | `sensor.battery_state_of_capacity` | Battery state of charge (%) |
| `sensor_battery_power` | `sensor.battery_charge_discharge_power` | Charge/discharge power (W, **+ve = charging**) |
| `sensor_grid_power` | _(empty)_ | Grid meter (W, **+ve = export, −ve = import**) |
| `sensor_solar_power` | _(empty)_ | Panel output right now (W, always ≥ 0) |
| `sensor_solar_current_hour` | `sensor.energy_current_hour` | Solar production this hour (kWh) |
| `sensor_solar_next_hour` | `sensor.energy_next_hour` | Solar forecast next hour (kWh) |
| `sensor_solar_today` | `sensor.energy_production_today` | Cumulative production today (kWh) |
| `sensor_solar_tomorrow` | `sensor.energy_production_tomorrow` | Forecast for tomorrow (kWh) |
| `sensor_temp_outdoor` | _(empty)_ | Outdoor temperature (°C) |
| `sensor_temp_salon` | _(empty)_ | Indoor temperature (°C) |
| `sensor_weather` | _(empty)_ | Weather entity (for storm detection) |

### Actuators

| Option | Default | Description |
|---|---|---|
| `switch_pool` | _(empty)_ | Pool pump switch |
| `switch_pool_cleaner` | _(empty)_ | Pool cleaner switch |
| `number_hvac_cool` | _(empty)_ | Heat pump cooling setpoint |
| `number_hvac_heat` | _(empty)_ | Heat pump heating setpoint |
| `number_battery_charge_cutoff` | `number.battery_grid_charge_cutoff_soc` | Battery grid charge cutoff SOC |
| `number_battery_charge_power` | _(empty)_ | Battery charge power limit |
| `number_battery_backup_soc` | _(empty)_ | Battery backup SOC |
| `switch_battery_force_charge` | _(empty)_ | Battery force charge switch |
| `select_battery_mode` | `select.battery_working_mode` | Battery working mode select |
| `sensor_dishwasher_state` | _(empty)_ | Dishwasher state sensor |

### Battery thresholds

| Option | Default | Description |
|---|---|---|
| `battery_emergency_threshold` | 10% | Force-charge below this SOC at any tariff |
| `battery_low_threshold` | 30% | Low battery level |
| `battery_medium_threshold` | 50% | Medium battery level |
| `battery_storm_threshold` | 80% | Pre-charge target when storm is forecast |
| `battery_capacity_kwh` | 10.0 | Total usable battery capacity (kWh) |

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
| `notify_email_service` | _(empty)_ | HA notify service name for email |
| `notify_email_target` | _(empty)_ | Recipient email address |
| `notify_telegram_service` | _(empty)_ | HA notify service name for Telegram |
| `notify_daily_time` | `23:00` | Time to send daily summary |
| `notify_email_enabled` | true | Enable email daily summary |
| `notify_telegram_daily_enabled` | true | Enable Telegram daily summary |
| `notify_telegram_alerts_enabled` | true | Enable instant Telegram alerts |

---

## Electricity tariff

Default prices — **Spain 2.0TD, all costs prorated including taxes and IVA:**

| Period | Hours | Price |
|---|---|---|
| Peak (Punta) | Weekdays 10–14h, 18–22h | **€0.2234/kWh** |
| Shoulder (Llano) | Weekdays 08–10h, 14–18h, 22–00h | **€0.1483/kWh** |
| Valley (Valle) | 00–08h + all day weekends | **€0.1147/kWh** |
| Export (Excedentes) | — | **€0.040/kWh** |

All prices and weekend days are editable per-hour in the **Tariff** tab. Use "↩ Reset to defaults" to restore the above values.

---

## Persistent data

All data lives in `/data/` inside the add-on container (persists across restarts and updates):

| File | Contents |
|---|---|
| `model.pkl` | Trained scikit-learn pipeline + metadata (R², feature list, feature version) |
| `wizard_config.json` | Entity IDs, hardware selection, HVAC zones, tariff from the setup wizard |
| `decisions.json` | Last 500 decision cycles — full sensor snapshot, tariff, actions, prediction |
| `savings.json` | Cumulative kWh avoided at peak + EUR saved since first run |
| `tariff.json` | Custom tariff configuration (periods, prices, weekend days) |
| `setup.json` | GUI runtime overrides — notification toggles, threshold sliders |

---

## Changelog

### v3.3.1
- Removed all personal/installation-specific defaults from `config.yaml` and Python fallbacks. Add-on now ships clean for any installation.

### v3.3.0
- **Sensor convention corrected:** `battery_power` is `+ve = charging, −ve = discharging` throughout display, flow physics, and averages.
- House balance: `P_casa = solar − grid − battery`.
- Flow vectors rewritten with correct priority order including emergency-charge-from-grid path.
- Battery card: green = charging, red = discharging.

### v3.2.9
- **Debug section in Tweaks tab:** table showing sensor role resolution (wizard → options → fallback), raw HA state, parsed value, and ✓/⚠/✗ status. Auto-loads when tab opens.

### v3.2.6 – v3.2.8
- `_live_averages_7d()`: W-level averages from 15-min decision samples. Solar average excludes night samples.
- `/api/debug/sensors` endpoint added.
- Physics-based flow animation: 7 computed flow vectors, speed proportional to power.

### v3.0.0
- **Setup Wizard:** 8-step guided configuration with entity auto-discovery, data quality thermometer, HVAC multi-zone scheduling, device sub-dots in nav bar.
- **Dynamic ML features:** feature set built from wizard-configured sensors, saved alongside model.
- **Multi-source history:** InfluxDB v2 → InfluxDB v1 → HA Recorder cascade.
- **`_wiz()` resolution:** wizard_config.json → options.json → hardcoded fallback for every sensor role.

### v2.6.1 – v2.6.4
- Grid power sign convention fixed.
- Continuous solar proxy (geometric sun elevation 0.0–1.0).
- Solar terrain correction factor (median actual/forecast, 30 days, cached 6h).
- Battery logic reoriented to cover tomorrow's peak demand.
- Temperature correction for heat pump load.

### v2.5.6 – v2.6.0
- InfluxDB as primary ML source (365 days) with auth auto-detection.
- Data Sources debug panel in Setup.
- Solar charts: Actual vs HA Forecast (7d + 12m).
- Savings bar chart with € labels and counterfactual method.
- 8-hour chained ML forecast.

### v2.4 – v2.5
- Weather forecast widget (5-day, storm alert).
- HA ingress fix (`X-Ingress-Path` header).
- Tariff editor with per-day weekend configuration.
- Telegram instant alerts.
- 4-tab GUI: Dashboard, Charts, Tariff, Setup.
