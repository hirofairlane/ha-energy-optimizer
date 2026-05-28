# Energy Optimizer — Home Assistant Add-on

Smart energy management add-on for Home Assistant OS. Its primary goal is to **minimise your electricity bill** — whether you have a solar battery, a set of schedulable loads (heat pump, EV charger, pool pump, irrigation…), or both. A **scikit-learn ML model** combined with dynamic tariff rules decides in real time when to charge or discharge the battery, when to shift loads to cheap/solar windows, and how much to pre-charge at night — so you import as little peak-rate energy as possible and export as little solar as possible.

> 🔒 **100% local — no cloud, no telemetry.** Everything runs inside the Docker container on your own hardware. Your energy data, sensor readings, and ML model never leave your home network. The add-on only communicates with your local Home Assistant instance and, optionally, a local InfluxDB instance.
<img width="1157" height="1113" alt="image" src="https://github.com/user-attachments/assets/f2235103-6d81-4d58-b8ca-00148d9acd58" />

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

> **v5 deep dive:** the predictive engine introduced in v5.0.0 has its own
> architecture write-up at [docs/architecture-v5.md](docs/architecture-v5.md)
> and a wiring guide for activating the dormant `v5_engine_enabled` flag at
> [docs/v5-wiring.md](docs/v5-wiring.md).

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
<img width="952" height="862" alt="image" src="https://github.com/user-attachments/assets/92348e09-e4ba-4a80-b406-7add53b455a5" />
| **Smart valley charging** | Calculates exactly how much battery to charge from the grid at night to cover tomorrow's peak demand — no more, no less |
| **Energy flow diagram** | Animated SVG showing real-time power flows between Solar, Battery, Grid, and House nodes |
<img width="556" height="312" alt="image" src="https://github.com/user-attachments/assets/0d08be64-683c-4aac-a5c1-02898d194154" />

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
| **Historical averages** | Day view KPI cards show all-time Ø below each value. History tab adds a 5-card summary with all-time and last-12-months averages for solar, consumption, export, import, and self-sufficiency |
| **Battery ROI calculator** | Enter cost + capacity of additional storage — calculates payback period from your actual average daily savings |
| **Battery health mode** | Three operating modes in Tweaks: ⚡ Bill Reducer (10–95%), ⚖️ Optimized (20–90%), 🛡️ Battery Guard (25–85%). Controls the SOC range used for the nightly charge target |
| **Split battery sensors** | For inverters that report charge and discharge as two separate positive entities (Deye, Solarman, Growatt…), enable the "Split sensors" toggle in the Battery wizard step and pick the two entity IDs. The add-on combines them automatically (`charge − discharge`) |
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
| **Battery health mode** | Three-button selector controlling the SOC operating range for the nightly charge target (see below). |
| **Notifications** | Enable/disable email daily summary, Telegram daily summary, and instant Telegram alerts individually. |
| **Battery thresholds** | Sliders for emergency, low, medium, and storm SOC thresholds. Adjust without editing `config.yaml`. |
| **Decision interval** | How often (in minutes) the optimization engine runs. Default 15 min. |
| **Data Sources** | Connectivity test for InfluxDB and HA Recorder — shows last-read timestamp, row count, and active/fallback status. |
| **Debug: sensor resolution** | Table that auto-loads when you open the tab and shows exactly how every sensor role was resolved. A **Refresh** button re-reads live HA values. |

#### Battery health mode

| Mode | SOC range | Description |
|---|---|---|
| ⚡ **Bill Reducer** | 10% – 95% | Default. Uses full battery capacity every cycle — maximum daily savings. |
| ⚖️ **Optimized** | 20% – 90% | Sweet spot for most installations: ~95% of savings benefit with moderate cycle protection. |
| 🛡️ **Battery Guard** | 25% – 85% | Prioritises longevity — recommended for batteries older than 3 years or with visible capacity degradation. |

The selected mode clamps the nightly charge target: the engine will never set a target below the mode's minimum or above its maximum, regardless of the calculated optimal. Saved in `/data/setup.json` and applied immediately without restart.

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

### Day view

A date navigator (← Today →) plus five KPI cards: Solar / Consumed / Exported / Imported / Self-sufficiency. Each card shows the day's total and, below it, the **all-time daily average (Ø)** computed from all recorded days. Hover over a card for a tooltip explaining the calculation. Below the KPIs: a stacked hourly energy bar chart and a SOC line for the selected day.

### History view

A 5-card summary row at the top shows **all-time** and **last-12-months** daily averages for all five KPIs — useful for spotting seasonal patterns or tracking improvement over time.

Below the charts, the **Battery ROI calculator** lets you enter the cost (€) and extra capacity (kWh) of additional storage and calculates:
- Your current average daily savings (from `savings.json`)
- Estimated extra savings proportional to the added capacity
- Payback period in days or years

> The calculator uses a linear proportionality assumption. Real results depend on your tariff, usage patterns, and how often the battery is the limiting factor.

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

<img width="507" height="295" alt="image" src="https://github.com/user-attachments/assets/e7306a9c-eda7-4647-a7dd-02e24281e2d1" />


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

- **Source:** InfluxDB v2 (wizard config, 90 days) → InfluxDB v1 (options, 60 days) → MariaDB direct (60 days, recorder DB) → HA Recorder REST (14 days)
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

## MariaDB / MySQL recorder integration

If your HA recorder is backed by MariaDB or MySQL and the REST history endpoint returns empty results (a known issue for some installations — see GitHub issue #2), the add-on can read history directly from the recorder database, bypassing the REST API.

Configured from the Setup Wizard's first step. The schema is auto-detected:
- **Modern (HA ≥ 2023.4)**: `JOIN states_meta sm ON s.metadata_id = sm.metadata_id WHERE sm.entity_id = ?`
- **Legacy**: `WHERE entity_id = ?` directly on `states`

Connection parameters (kept in `wizard_config.json` only — no secrets in `options`):

| Field | Typical value |
|---|---|
| Host / IP | `core-mariadb` (HA add-on) or your MariaDB server |
| Port | `3306` |
| Database | `homeassistant` |
| Username / Password | as configured in the recorder `db_url` |

The Test connection button counts samples for every configured sensor over the last 60 days; if any returns rows, MariaDB is used as the active source on subsequent cycles. Falls back to HA Recorder REST otherwise.

---

## Configuration reference

> Since v3.0 the **Setup wizard** (Setup tab → "Configure entities") is the recommended way to fill these. The wizard scans your HA states, scores candidates by name/unit/device_class and proposes the best match for each role. All options can also be set in the HA add-on Configuration UI as a fallback, but the wizard's `/data/wizard_config.json` takes priority over them for all sensor lookups.

### Sensors

| Option | Default | Description |
|---|---|---|
| `sensor_battery_soc` | `sensor.battery_state_of_capacity` | Battery state of charge (%) — Huawei Modbus standard |
| `sensor_battery_power` | `sensor.battery_charge_discharge_power` | Charge/discharge power (W, **+ve = charging**) — Huawei Modbus standard |
| `sensor_grid_power` | _(empty)_ | Grid meter (W, **+ve = export, −ve = import**) |
| `sensor_solar_power` | _(empty)_ | Panel output right now (W, always ≥ 0) |
| `sensor_solar_current_hour` | `sensor.energy_current_hour` | Solar production this hour (kWh) — HA Energy dashboard |
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
| `number_battery_charge_cutoff` | `number.battery_grid_charge_cutoff_soc` | Battery grid charge cutoff SOC — Huawei Modbus standard |
| `number_battery_charge_power` | _(empty)_ | Battery charge power limit |
| `number_battery_backup_soc` | _(empty)_ | Battery backup SOC |
| `switch_battery_force_charge` | _(empty)_ | Battery force charge switch |
| `select_battery_mode` | `select.battery_working_mode` | Battery working mode select — Huawei Modbus standard |
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
| `notify_email_target` | _(empty)_ | Recipient email address — set in HA add-on config UI, not in repo |
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

### v5.0.2 — GoodWe support: opt-in mode select + `battery_backup_soc` reset

Fixes two bugs in the battery control layer that prevented Energy Optimizer from working on GoodWe inverters and could leave any installation with an inflated minimum discharge SOC after a valley charge. Reported and diagnosed by [@andredp](https://github.com/andredp) on a GoodWe GW3648-EM with the [mletenay HACS integration](https://github.com/mletenay/home-assistant-goodwe-inverter) (issue #7).

- **Bug A — mandatory mode select.** `set_battery_charge_target` and `set_battery_self_consumption` unconditionally called `select.select_option` against the configured `battery_mode_select` entity. The hardcoded fallback was a Huawei Luna2000-specific id (`select.battery_working_mode`), so a GoodWe install with the wizard field left blank still fired the call against a non-existent entity, got a 400 back, and tainted the whole battery operation as `ok=False`. GoodWe doesn't need a mode select at all — `goodwe_fast_charging_switch` + `goodwe_fast_charging_soc` are standalone registers that operate independently of the inverter's operation mode.
- **Bug B — `battery_backup_soc` never reset.** `set_battery_charge_target` raised `battery_backup_soc` to the valley charge target (e.g. 60 %) but `set_battery_self_consumption` never lowered it back when the engine flipped to self-consumption at peak start. On Huawei this entity is the minimum charging SOC and the silent inflation was tolerable; on GoodWe it maps to `goodwe_battery_soc_protection` and pinned the battery above the valley target all day, defeating the entire optimisation.
- **Fix.** The mode select is now opt-in: the hardcoded Huawei default is removed, and the call is guarded with `if mode_entity:` so it only fires when the user explicitly populates `select_battery_mode` in the wizard or `setup.json`. `set_battery_self_consumption` now also resets `battery_backup_soc` to `min_soc`, mirroring how `battery_cutoff_soc` is already reset.
- **Migration note for Huawei users.** If you were relying on the implicit `select.battery_working_mode` default and never set the entity in the wizard, you must add it explicitly under Setup Wizard → Battery → "Mode select entity" after upgrading. If you already configured it (the recommended path documented in the wizard since v3.x), nothing changes.

### v5.0.1 — Retrain fix for InfluxDB-backed installs

Hotfix. The nightly retrain (`retrain_cron`, default `0 3 * * *`) was silently failing on every cycle for any installation with `data_source: influxdb` in the wizard, leaving the model frozen at the SOC predictions it had at install time. Traceback was buried in the addon log — the engine kept producing decisions from a stale model and never surfaced the problem.

- **Root cause.** `_influx_query` requests `epoch=ms` so InfluxDB returns `time` as an integer (Unix milliseconds), but `_rows_to_15min_series` assumed HA-REST style ISO strings and called `.replace("Z", …)` on the value. The `except` clause caught `KeyError / ValueError / TypeError` but not `AttributeError`, so the first row aborted the whole training set.
- **Fix.** `_rows_to_15min_series` now detects the type of `last_changed` (`str` / `int` / `float`) and converts integer timestamps via magnitude thresholds (seconds / milliseconds / nanoseconds). `AttributeError` is also added to the `except` so a single malformed row no longer kills the batch.
- **Impact.** Confirmed on a real install: the post-fix retrain produced 8637 samples × 22 features (R² 0.997), versus the previous 14-feature model trained 18 days earlier. The eight extra features are the wizard's `grid_submeters` (per-appliance power readings) that had been silently dropped because they were added after the last successful training.
- Affects: HAOS installs with `data_source: influxdb` (the default for users who configure the InfluxDB step in the wizard). HA-REST-only installs were not affected because their `last_changed` is already a string.

### v5.0.0 — Predictive engine rewrite

Major release. The decision engine moves from reactive heuristics to a **predictive planner** with explicit forecasting, deterministic SOC simulation, an iterative planner with convergence guarantees, a four-layer policy pipeline, and full per-cell traceability. The new engine is delivered behind a feature flag (`v5_engine_enabled: false` by default) so existing v4 installations keep running unchanged until the flag is flipped.

**Why a major bump.** Although `wizard_config.json` v3.5+ remains backward-compatible, the cycle's observable behaviour changes radically when the v5 engine is active: forecasts come from quantile heads, decisions are computed from a horizon-aware planner, and every override (capacity budget, peak prohibition, antiflap, degraded mode) is recorded in a structured trace. Calling that anything less than a major bump would be dishonest.

**Architecture (under `rootfs/usr/bin/eo/`).** Built strangler-fig style alongside the existing monolithic `energy_optimizer.py`. The v4 engine is untouched.

- `eo/forecasters/` — `ClearSkyModel` (deterministic PV baseline, no `pvlib`), `AtmosphericFactorModel` (ML residual with P10/P50/P90 quantile heads), `SolarForecaster` (composition), `HouseForecaster` (standalone quantile model, **not chained** to avoid the v3.x autoregressive inflated R²), `ForecastQualityTracker` (rolling MAE / bias / calibration error per series), `training` helpers.
- `eo/simulator/` — pure, side-effect-free 15-min-timestep SOC simulator with an injectable `PhysicsModel` (today `SingleBatteryPhysicsModel`; future EV / multi-battery without touching the simulator). Hard invariants for SOC bounds, charge/discharge mutex, inverter capacity, energy conservation, min-runtime, contradictory actions, debt monotonicity. Soft mode in production, strict mode in tests.
- `eo/planner/` — debt-state classifier with the two bug fixes from review (window-mutation truncation + telemetry-coverage scaling), 11-row decision matrix from the spec, utility-score function, iterative convergence loop with plan-hash + 2-cycle detection + `MAX_PLANNER_ITERATIONS=5` guardrail. Implements the `forced_states` injection insight from the audit round so the simulated trajectory never diverges from execution.
- `eo/policy/` — four-layer pipeline (`capacity_budget` → `peak_prohibition` → `antiflap` → `degraded_mode`). Each layer is a pure transform; the triple `raw_plan / policy_adjusted_plan / execution_plan` is preserved end-to-end. Three-level degraded mode (forecast MAE high → drop `min_runtime_only` decisions; AEMET stale → drop everything except `rule_id=3`; sensors stale > 30 min → all deferred loads OFF).
- `eo/scenario/` — `ScenarioBuilder` collapses quantiles into a single coherent Scenario whose risk tolerance is derived from the worst debt state across declared loads.
- `eo/state/` — aggregated `SystemState` dataclass (battery + forecast quality + planner history + load debt + antiflap + execution world state) with atomic JSON persistence (`tmp + fsync + os.replace`).
- `eo/execution/` — pure execution engine and cycle orchestrator. Optimistic-hybrid dispatch (no synchronous ACK wait); telemetry-driven reconciliation after restart (telemetry beats persisted state if they diverge).

**Tests.** 355 pytest cases across the package. Includes the three high-pressure scenarios from the spec review: "Monstruo del Bucle" (oscillating load hits `max_iter` cap), "Sábado de Gloria" (concurrent loads on weekend valley serialised by capacity budget), "Cap del Fin del Mundo" (unreachable quota produces `irreachable` debt state and Telegram alert).

**Defaults.** `v5_engine_enabled: false`. The flag turns the engine on once the addon-specific wiring is complete (forecasters trained on the user's historical data, data-source bindings for hours-on-per-day rebuilds, entity registry for `is_on` / `send_command`). Wiring guide ships in a future point release.

### v4.0.2 — Setup integrity checker + first modular package

Hotfix release that lays the ground for the upcoming v5.0.0 rewrite while delivering one immediately useful fix.

- **Setup integrity checker**: detects when the same `entity_id` is configured in multiple actuable roles in the wizard (e.g. the same `switch.*` used as `pool_switch` and as a `custom_loads[].switch`). This was the root cause of the "blitzwolf switching by itself" symptom reported in #4 — the pool branch issued `turn_on` and the custom-load branch issued `turn_off` in the very same decision cycle. On startup the add-on now logs every conflict, publishes `binary_sensor.energy_optimizer_setup_conflict` to Home Assistant (with the full conflict list as attributes), and fires a Telegram alert if any CRITICAL collision is found.
- **First module of the new `eo` package**: `eo.checks.setup_integrity` is a pure, unit-tested module (18 pytest cases) introduced as the template for how v5.0.0 modules will be organised. The legacy monolithic `energy_optimizer.py` keeps running unchanged; new functionality grows alongside it (strangler-fig pattern).
- **`pytest.ini` and `tests/` directory**: development tests now live in the repo and run with `python3 -m pytest tests/` (not shipped inside the Docker image).

### v4.0.1 — Full traceability of side effects

Audit pass on top of v4.0.0 to close every "black box" — every switch toggle, every number write, every API call the add-on performs is now recorded in `decisions.json` and visible in the Activity tab. No silent actions.

- **Pool cleaner auto-stop**: the 15-min APScheduler-deferred turn-off was firing outside the cycle and never reaching `decisions.json`. Now wrapped through a new `_record_event()` helper that persists deferred / out-of-cycle actions to the same history. Tagged `scheduler_event` in the UI.
- **Pool cleaner start**: previously bundled implicitly with the pool pump's reason string. Now emitted as its own `pool_cleaner` entry with its own `explanation`.
- **Heat pump dual call** (`number.set_value` + optional `climate.set_temperature`): previously only the number write was logged. The climate mirror call now gets its own `climate_setpoint` entry, tracked independently with its own ok/explanation.
- **Manual API endpoints** (`/api/battery/charge`, `/api/battery/self-consumption`): when called from outside the engine (UI button, external automation, curl), the action is now recorded as a `manual_api` event with the requester IP, the target SOC, and the full reasoning. No more invisible overrides.
- **Activity tab visual coverage**: extended the `tagMap` and CSS to include `custom_load`, `pool_cleaner`, `climate_setpoint`, `dishwasher`, `manual_api`, `scheduler_event`. Previously unknown types rendered as undifferentiated gray pills; now each event family has a distinct colour so external interventions are visually distinguishable from cycle decisions.

### v4.0.0 — Decision engine rewrite + transparency layer

Major release. The internal decision engine has been rewritten to (a) respect every user-configurable setting that was previously silently overridden, and (b) explain what it did and why, in every cycle.

**The two visible changes for everyone:**

1. **Every decision now ships an `explanation` dict** with `what`, `why`, the `inputs` used, the `formula` applied, the actual `calculation`, and the `alternatives_rejected`. The Activity tab renders an expandable row per decision (click the ▶ in front of any reason) that shows the full reasoning. The legacy `reason` short string is kept on every decision for backwards compatibility with downstream consumers.
2. **The smart battery target SOC adapts to your actual tariff geometry**, not a hardcoded Spanish 2.0TD shape. Users with longer/shorter peak windows (Ukraine 16h peak, Australia 4h peak, Germany variable, …) now get targets that match their real consumption profile.

**Decision engine fixes**, by impact:

- **`calculate_optimal_soc` peak-hour scaling.** The previous `peak_base_kwh = base_kw × 8` hardcode caused systematic 2× errors on tariffs with non-Spanish peak length. Surfaced by @Karplyak in issue #5 (Ukrainian 16h peak ⇒ target stuck at ~36 % when ~70 % was right). Now scales with `len(tariff.peak_hours)`.
- **`calculate_optimal_soc` solar-overlap.** Previously `solar_during_peak = solar_tomorrow × 0.45` (Spanish heuristic). Now computed as the overlap between the user's actual peak window and a 7-19h "useful sun" window, divided by that window's length.
- **Temperature adjustment scales with peak length.** The HVAC-load bands (0/0.5/1/2/3 kWh) were tuned for an 8h peak. Now multiplied by `peak_h_ratio = len(peak_hours) / 8`.
- **`decide_battery` respects `battery_health_mode`.** Previously hardcoded `target_soc: 30/40` and `if soc >= 95` ignored the user's selected profile, meaning Battery Guard (25-85 %) users would "never reach full" according to the engine. Now resolves the floor as `max(30, health_mode_min)` and the ceiling as `health_mode_max`.
- **`decide_battery` uses configurable `battery_low_threshold`.** Previously hardcoded `if soc < 20` ignored the option even though it was configurable. Now reads `cfg("battery_low_threshold", 30)`.
- **`decide_battery` uses the configured `battery_charge_power` entity.** Previously hardcoded 1500/2000/3000 W. Now reads the wizard's configured charge-power number entity and caps emergency charges to it.
- **Mid-tariff opportunistic top-up.** Previously mid-period only reacted to dangerously low SOC. Now also tops up toward the smart target if mid is the cheapest period available before peak (for tariffs where mid-hours overlap with daytime sun, e.g. Spanish 2.0TD 14-18h).

### v3.5.5
- **Custom loads scheduler actually drives the switches now (issue #4, reported by @Karplyak).** The wizard's Loads step lets the user add `custom_loads` with a switch entity, watts estimate, and a scheduling mode (`valley`, `solar`, `both`, `hours`). The wizard was persisting them correctly to `wizard_config.json`, but **no Python code on the engine side was consuming them** — the switches were never being turned on/off by the optimizer. Implemented `decide_custom_loads()` and wired it into the decision cycle:
  - `valley` → on during P3 tariff only
  - `solar` → on when grid export ≥ load wattage and SOC > 30 %
  - `both` → on if either of the above
  - `hours` → on when current hour is inside any range in the user's spec (`10-14,22-06` style, ranges wrap midnight)
  Only emits a service call when the desired state differs from the current one, so quiet hours don't spam the bus.

### v3.5.4
- **Wizard "Flip sign" toggle on the Grid step actually does something now.** It was being saved to `wizard_config.json` as `grid_flip` but never read back — users whose meter reports `+ve = importing from grid` had a no-op toggle and the engine kept seeing inverted signs (battery decisions + savings counter all biased). Reported by @Karplyak in issue #2 after testing v3.5.3.
- **`_influx_wizard_history` query corrected.** Was using `FROM "<entity_short>"` while the HA→Influx integration stores `measurement = unit` + `entity_id` as TAG. The query returned an empty result for every entity and the loader silently fell back to `ha_history_influx`. Worked for setups with the legacy `influxdb_url` option populated; broken for users configuring InfluxDB only through the wizard. Switched to the same `FROM /.*/ WHERE entity_id = ...` pattern as the rest of the code.
- **`/api/wizard/data-quality` now counts `grid_submeters`.** The endpoint iterated only over the wizard's `sensors` dict — the score reported OK while sub-meter feeds (Meross, etc.) used as `sm_<name>` features in ML training were silently uncounted. Built a unified `all_entities` dict spanning both, used in the InfluxDB / MariaDB / HA-Recorder loops.

### v3.5.3
- **MariaDB direct query fix (issue #2, reported by @Karplyak):** the recorder query was filtering by `last_changed_ts`, which HA only populates when the state value actually changes. Power sensors that emit the same reading repeatedly, or rows where only attributes were updated, leave `last_changed_ts` NULL and got silently dropped → 0 rows returned. Switched to `last_updated_ts` (and the legacy `last_updated`), which is populated on every write. Hotfix on top of v3.5.2.

### v3.5.2
- **MariaDB / MySQL recorder support (issue #2):** wizard now offers a third data source besides InfluxDB and HA Recorder REST. When the HA recorder is backed by MariaDB and the REST endpoint returns nothing (a known issue for some installations), the add-on can read history directly from the recorder DB. New section in the Setup Wizard's first step: host, port, database, user, password, and a Test connection button. Schema is auto-detected (legacy `states.entity_id` vs post-2023.4 `states_meta` JOIN).
- **ML predict feature-name fix:** the chained 8 h SOC forecast was passing `temp_out` to a model trained with `temp_outdoor`, raising a sklearn warning every cycle. Renamed to match the trained feature; dropped two stale features (`temp_out_lag4`, `grid_abs_lag1`) that were never trained.
- **Version drift fix:** `ADD_ON_VERSION` was 3.5.0 while `config.yaml` was 3.5.1; aligned both to 3.5.2.

### v3.5.1
- **Savings counter rewritten with counterfactual method:** the previous filter (peak hours + battery discharging + grid import > 500 W) was under-counting by ~97 % because it ignored self-consumption in valley/off-peak and the cases where the battery covered 100 % of the load. New logic compares grid cost with battery against a hypothetical no-battery scenario per cycle (`grid_cf = grid + battery`). JSON is backwards-compatible: `total_eur_saved` stays primary, `total_kwh_avoided_peak` is kept as secondary, `total_kwh_throughput` added for diagnostics. Per-cycle delta is capped at ±0.30 € as sanity.

### v3.5.0
- **Single source of truth for version:** `ADD_ON_VERSION` in `energy_optimizer.py` is now canonical. Build copies `config.yaml` to `/addon-config.yaml` and a startup check logs a warning if the two drift, so panel and Supervisor never disagree.
- **Wizard data-quality fix:** the InfluxDB sample-count query was using `entity_id` as the measurement name. With the HA→InfluxDB integration the measurement is the unit (`%`, `W`, `kWh`, …) and `entity_id` is a tag — query corrected accordingly.
- **Branch consolidation:** merges parallel v3.4.1 work that had diverged on a separate machine. No user-facing regressions; per-version entries below remain authoritative for individual features.

### v3.4.1
- **Split battery sensors (Deye/Solarman/Growatt support):** The Battery step of the Setup Wizard now has a "Split sensors" toggle for inverters that report charge and discharge as two separate positive-valued entities instead of one signed sensor. Enable the toggle and select the two entities — the engine combines them as `charge − discharge` so the rest of the logic behaves identically. The debug table in Tweaks also adapts to show both entities when split mode is active.

### v3.4.0
- **Average consumption metrics in Charts:** Day view KPI cards show all-time daily averages (Ø) below each value with tooltip explaining the calculation. History tab adds a full 5-card summary row with all-time and last-12-months averages for solar, consumption, export, import, and self-sufficiency.
- **Battery ROI calculator:** New section in the History tab. Enter the cost and capacity of additional storage — the calculator uses your actual average daily savings to estimate payback time and projected annual gain.
- **Battery health mode:** Three-button selector in the Tweaks tab controls the SOC operating range the engine targets: ⚡ **Bill Reducer** (10–95%, default), ⚖️ **Optimized** (20–90%), 🛡️ **Battery Guard** (25–85%). Affects the nightly charge target clamp. Persisted in `setup.json`.

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
