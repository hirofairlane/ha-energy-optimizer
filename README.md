# Energy Optimizer — Home Assistant Add-on

Smart energy management add-on for Home Assistant OS. Combines dynamic electricity tariff rules with a scikit-learn ML model to automatically control a solar battery, heat pump, pool pump, pool cleaner, and dishwasher.

---

## Features

- **Battery management** — Avoids grid charging during peak tariff. Charges at valley rate using an adaptive SOC target calculated from rolling overnight consumption history and tomorrow's solar forecast.
- **Storm protection** — Reads AEMET weather data and pre-charges the battery to a configurable reserve level when adverse weather is forecast.
- **Heat pump control** — Adjusts setpoints based on season, indoor temperature, and free solar power (SOC ≥ 99%).
- **Pool pump** — Runs during solar surplus or valley tariff to meet daily/weekly hours.
- **Pool cleaner (limpiafondos)** — Automatically starts with the pool pump and stops after 15 minutes (~1.5 kWh device).
- **Dishwasher** — Monitored via `sensor.lavavajillas_operation_state`; included in consumption predictions; started during solar surplus or valley tariff if a control switch is configured.
- **ML model** — GradientBoostingRegressor trained nightly on up to 60 days of battery SOC history. Falls back to rules-only mode when data is insufficient.
- **Telegram instant alerts** — Sends an immediate notification when an emergency charge, storm mode, or forced grid charge is triggered.
- **Daily summary** — HTML email and/or Telegram report sent every day at a configurable time, with actions taken, savings, and solar production.
- **Web panel** — Built-in dashboard accessible via HA ingress on port 8765.

---

## Web Panel

Four tabs:

| Tab | Contents |
|---|---|
| 📊 Dashboard | Live KPIs, battery status, estimated savings, recent decision log |
| 📈 Charts | Actual vs predicted SOC (24h), solar production (3-day bar), daily savings (7-day bar) |
| ⚡ Tariff | Per-day weekend configuration, per-hour peak/valley/mid timeline, price editor |
| ⚙️ Setup | Notification toggles, battery threshold sliders, decision interval |

Setup changes are saved to `/data/setup.json` and take effect immediately (no restart needed, except for decision interval).

---

## Installation

### Option A — Local add-on
1. Copy the `ha-energy-optimizer/` folder to `/addons/` on your HAOS filesystem (via Samba or SSH).
2. In Home Assistant: **Settings → Add-ons → Add-on store → ⋮ → Check for updates**.
3. Install **Energy Optimizer** and configure via the add-on options.

### Option B — GitHub repository
1. In Home Assistant: **Settings → Add-ons → Add-on store → ⋮ → Repositories**.
2. Add `https://github.com/hirofairlane/ha-energy-optimizer`.
3. Install and configure.

---

## Configuration

All options can be set in the add-on configuration panel. Key parameters:

### Devices

| Option | Description | Default |
|---|---|---|
| `sensor_battery_soc` | Battery state of charge sensor | `sensor.battery_state_of_capacity` |
| `sensor_battery_power` | Battery charge/discharge power | `sensor.battery_charge_discharge_power` |
| `sensor_grid_power` | Grid import/export power | `sensor.acometida_general_power` |
| `switch_pool` | Pool pump switch | `switch.depuradora` |
| `switch_pool_cleaner` | Pool cleaner switch (limpiafondos) | `switch.limpiafondos` |
| `sensor_dishwasher_state` | Dishwasher operation state | `sensor.lavavajillas_operation_state` |
| `switch_dishwasher` | Dishwasher control switch (optional) | _(empty)_ |
| `sensor_weather` | AEMET weather entity | `weather.aemet` |

### Battery thresholds

| Option | Description | Default |
|---|---|---|
| `battery_emergency_threshold` | Force-charge at any tariff below this SOC (%) | `10` |
| `battery_storm_threshold` | Pre-charge to this SOC when storm is forecast (%) | `80` |
| `battery_capacity_kwh` | Total battery capacity | `10.0` |

### Scheduling

| Option | Description | Default |
|---|---|---|
| `decision_interval_minutes` | How often the optimization cycle runs | `15` |
| `retrain_cron` | Cron expression for nightly ML retraining | `0 3 * * *` |
| `summer_start_month` / `summer_end_month` | Summer season range | `6` / `9` |

### Notifications

| Option | Description | Default |
|---|---|---|
| `notify_email_service` | HA notify service for email | `correo_gmail_com` |
| `notify_email_target` | Recipient email address | — |
| `notify_telegram_service` | HA notify service for Telegram | `telegramsergio` |
| `notify_daily_time` | Time to send the daily summary | `23:00` |
| `notify_email_enabled` | Enable email daily summary | `true` |
| `notify_telegram_daily_enabled` | Enable Telegram daily summary | `true` |
| `notify_telegram_alerts_enabled` | Enable instant Telegram alerts | `true` |

All notification toggles can also be changed at runtime from the **Setup** tab in the web panel.

---

## Electricity tariff (default — Spain PVPC)

| Period | Hours (weekdays) | Price |
|---|---|---|
| Valley | 00:00–08:00 | €0.08/kWh |
| Mid | 08:00–10:00, 14:00–18:00, 22:00–00:00 | €0.18/kWh |
| Peak | 10:00–14:00, 18:00–22:00 | €0.30/kWh |
| Export | — | €0.06/kWh |

Weekends and public holidays can be configured as all-day valley directly in the **Tariff** tab.

---

## Persistent data (in `/data/`)

| File | Contents |
|---|---|
| `model.pkl` | Trained scikit-learn pipeline |
| `decisions.json` | Last 500 decision cycles with full sensor snapshot |
| `savings.json` | Cumulative kWh and EUR savings |
| `tariff.json` | Custom tariff configuration |
| `setup.json` | GUI runtime overrides (notifications, thresholds) |

---

## Changelog

### v2.3
- Fixed HA ingress base-path: all GUI buttons and charts now work correctly
- Tariff editor: per-day-of-week weekend configuration (click any day to toggle)
- README translated to English

### v2.2
- Real-time Telegram alerts for emergency charges, storm mode, forced grid charges
- 4-tab GUI: Dashboard, Charts, Tariff, Setup
- GUI Setup panel with notification toggles and threshold sliders (persisted to `/data/setup.json`)
- Enhanced charts: predicted vs actual SOC + 7-day daily savings timeline
- Limpiafondos auto-runs first 15 min with pool pump, then auto-stops
- Dishwasher monitored and included in ML predictions

### v2.1
- Initial public release
