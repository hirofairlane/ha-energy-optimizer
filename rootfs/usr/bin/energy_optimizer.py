#!/usr/bin/env python3
"""
Energy Optimizer — Home Assistant Add-on v2.4
Smart energy management: battery, heat pump, pool pump, pool cleaner, dishwasher
Logic: adaptive tariff rules + scikit-learn ML model + consumption history

Changes in v2.4:
  - History API fix: 60s timeout + no_attributes=true (resolves "0 samples" on retrain)
  - Minimum training samples lowered to 48 (3×15min cycles per day for 4 days)
  - Solar proxy now uses actual sun.sun elevation from HA (correct for Guadarrama, Madrid)
  - Real all-in tariff prices from 2.0TD bill (taxes + IVA included): P1=0.284, P2=0.189, P3=0.146
  - Export price updated to real excedentes rate: 0.040 €/kWh
  - Weather forecast widget in Dashboard (condition, temperature, 5-day forecast, storm alert)
  - New /api/weather endpoint using weather.aemet
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, request

# ── Persistent data paths ────────────────────────────────────────────────────
DATA_DIR       = Path("/data")
MODEL_FILE     = DATA_DIR / "model.pkl"
DECISIONS_FILE = DATA_DIR / "decisions.json"
SAVINGS_FILE   = DATA_DIR / "savings.json"
TARIFF_FILE    = DATA_DIR / "tariff.json"
SETUP_FILE     = DATA_DIR / "setup.json"
DATA_DIR.mkdir(exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("energy-optimizer")

# ── Options (from /data/options.json — HA add-on config) ────────────────────
def load_options() -> dict:
    path = Path("/data/options.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    log.warning("options.json not found, using defaults")
    return {}

OPT = load_options()

# ── Setup overrides (from GUI — stored in /data/setup.json) ──────────────────
_SETUP: dict = {}

def _load_setup_cache():
    global _SETUP
    if SETUP_FILE.exists():
        try:
            _SETUP = json.loads(SETUP_FILE.read_text())
        except Exception:
            _SETUP = {}
    else:
        _SETUP = {}

def save_setup(data: dict):
    global _SETUP
    _SETUP.update(data)
    SETUP_FILE.write_text(json.dumps(_SETUP, indent=2))
    log.info("Setup configuration saved")

def cfg(key: str, default=None):
    """Get config value: GUI setup overrides > options.json > default."""
    if key in _SETUP:
        return _SETUP[key]
    return OPT.get(key, default)

# ── HA Client ────────────────────────────────────────────────────────────────
HA_BASE  = "http://supervisor/core"
HA_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
HEADERS  = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}

def ha_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{HA_BASE}{path}", headers=HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        log.warning(f"GET {path} → {r.status_code}: {r.text[:120]}")
    except requests.RequestException as e:
        log.error(f"GET {path} error: {e}")
    return None

def ha_post(path: str, data: dict = None) -> bool:
    try:
        r = requests.post(f"{HA_BASE}{path}", headers=HEADERS, json=data or {}, timeout=10)
        ok = r.status_code in (200, 201)
        if not ok:
            log.warning(f"POST {path} → {r.status_code}: {r.text[:120]}")
        return ok
    except requests.RequestException as e:
        log.error(f"POST {path} error: {e}")
        return False

def ha_state(entity_id: str) -> dict | None:
    return ha_get(f"/api/states/{entity_id}")

def ha_str(entity_id: str, default: str = "") -> str:
    s = ha_state(entity_id)
    return s.get("state", default) if s else default

def ha_float(entity_id: str, default: float = 0.0) -> float:
    s = ha_state(entity_id)
    if s:
        try:
            return float(s["state"])
        except (ValueError, KeyError, TypeError):
            pass
    return default

def ha_service(domain: str, service: str, data: dict = None) -> bool:
    log.info(f"  → {domain}.{service} {data or ''}")
    return ha_post(f"/api/services/{domain}/{service}", data or {})

def ha_set_number(entity_id: str, value: float) -> bool:
    return ha_service("number", "set_value", {"entity_id": entity_id, "value": value})

def ha_set_select(entity_id: str, option: str) -> bool:
    return ha_service("select", "select_option", {"entity_id": entity_id, "option": option})

def ha_switch(entity_id: str, turn_on: bool) -> bool:
    return ha_service("switch", "turn_on" if turn_on else "turn_off", {"entity_id": entity_id})

def ha_history(entity_id: str, days: int = 14) -> list:
    """Fetch entity history from HA recorder.

    Uses no_attributes=true to cut payload size (critical for 60-day requests)
    and a 60-second timeout so long history calls don't fail silently.
    """
    from datetime import timezone
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    try:
        r = requests.get(
            f"{HA_BASE}/api/history/period/{start}",
            headers=HEADERS,
            params={
                "filter_entity_id": entity_id,
                "minimal_response": "true",
                "no_attributes":    "true",
            },
            timeout=60,
        )
        if r.status_code == 200:
            result = r.json()
            if result and isinstance(result, list) and len(result) > 0:
                rows = result[0]
                log.info(f"  History {entity_id}: {len(rows)} records ({days}d)")
                return rows
            log.warning(f"  History {entity_id}: empty response from HA recorder")
        else:
            log.warning(f"  History {entity_id} HTTP {r.status_code}: {r.text[:120]}")
    except requests.Timeout:
        log.error(f"  History {entity_id}: timed out requesting {days}d of data")
    except requests.RequestException as e:
        log.error(f"  History {entity_id}: {e}")
    return []

# ── InfluxDB history (primary source for ML training) ────────────────────────
def _influx_query(url: str, db: str, q: str, user: str, password: str) -> tuple:
    """Run an InfluxQL query. Auto-detects whether auth is needed.

    Returns (response_or_None, error_str, auth_mode_used).
    Tries with credentials first; if InfluxDB returns 401 (auth disabled and
    rejecting credentials, which happens in some 1.x configs), retries without.
    """
    endpoint = f"{url.rstrip('/')}/query"
    params   = {"db": db, "q": q, "epoch": "ms"}
    # Attempt 1: with credentials (if provided)
    try:
        auth = (user, password) if user else None
        r = requests.get(endpoint, params=params, auth=auth, timeout=30)
        if r.status_code == 200:
            return r, None, "auth" if auth else "no_auth"
        if r.status_code == 401 and auth:
            # InfluxDB 1.x returns 401 when auth is DISABLED and credentials
            # are sent — retry without credentials
            log.info("  InfluxDB: 401 with credentials — retrying without auth")
            r2 = requests.get(endpoint, params=params, auth=None, timeout=30)
            if r2.status_code == 200:
                log.info("  InfluxDB: connected without auth (auth disabled on server)")
                return r2, None, "no_auth"
            return None, f"http_{r2.status_code}", "no_auth"
        return None, f"http_{r.status_code}", "auth" if auth else "no_auth"
    except requests.Timeout:
        return None, "timeout", "unknown"
    except Exception as e:
        return None, str(e)[:80], "unknown"

def ha_history_influx(entity_id: str, days: int = 60) -> tuple:
    """Fetch entity history from InfluxDB.

    Uses InfluxQL with epoch=ms so timestamps come back as integers.
    Returns (rows, error_str) — rows is a list of
    {"last_changed": iso_str, "state": str} dicts, error_str is None on success.
    """
    from datetime import timezone as _tz
    url      = cfg("influxdb_url", "").strip()
    db       = cfg("influxdb_db", "homeassistant").strip()
    user     = cfg("influxdb_user", "").strip()
    password = cfg("influxdb_password", "")
    if not url:
        return [], "not_configured"
    # Old HA InfluxDB integration stores measurements as units (%, W, A…)
    # and entity_id as a tag WITHOUT the domain prefix.
    entity_short = entity_id.split(".")[-1] if "." in entity_id else entity_id
    q = (f'SELECT "value" FROM /.*/ WHERE "entity_id" = \'{entity_short}\' '
         f'AND time >= now() - {days}d ORDER BY time ASC')
    resp, err, _mode = _influx_query(url, db, q, user, password)
    if err:
        log.warning(f"  InfluxDB {entity_id}: {err}")
        return [], err
    data    = resp.json()
    results = data.get("results", [])
    if not results or "series" not in results[0]:
        log.warning(f"  InfluxDB {entity_id}: no series in response")
        return [], "no_series"
    series = results[0]["series"][0]
    cols   = series.get("columns", [])
    time_i = cols.index("time")  if "time"  in cols else 0
    val_i  = cols.index("value") if "value" in cols else 1
    rows = []
    for point in series.get("values", []):
        try:
            ms  = int(point[time_i])
            ts  = datetime.fromtimestamp(ms / 1000, tz=_tz.utc).isoformat()
            val = point[val_i]
            if val is None:
                continue
            rows.append({"last_changed": ts, "state": str(val)})
        except (IndexError, TypeError, ValueError):
            continue
    log.info(f"  InfluxDB {entity_id}: {len(rows)} records ({days}d)")
    return rows, None

# ── Tariff management ────────────────────────────────────────────────────────
DEFAULT_TARIFF = {
    # All-in prorated prices for 2.0TD (Spain) — all costs included (taxes, tolls, charges)
    # Source: Energia Nufri invoice, Apr-2026 update
    #   P1 (Punta)  = 0.2234 €/kWh  (peak weekdays 10-14h, 18-22h)
    #   P2 (Llano)  = 0.1483 €/kWh  (shoulder)
    #   P3 (Valle)  = 0.1147 €/kWh  (valley 00-08h / weekends)
    #   Export (excedentes) = 0.040 €/kWh (fixed)
    "prices":       {"peak": 0.2234, "mid": 0.1483, "valley": 0.1147, "export": 0.040},
    "peak_hours":   [10, 11, 12, 13, 18, 19, 20, 21],
    "valley_hours": [0, 1, 2, 3, 4, 5, 6, 7],
    "weekend_days": [5, 6],   # 0=Mon … 6=Sun; these days use valley tariff all day
}

def load_tariff() -> dict:
    if TARIFF_FILE.exists():
        try:
            data = json.loads(TARIFF_FILE.read_text())
            # Back-fill weekend_days if loading an older config
            data.setdefault("weekend_days", [5, 6])
            return data
        except Exception:
            pass
    return dict(DEFAULT_TARIFF)

def save_tariff(tariff_cfg: dict):
    TARIFF_FILE.write_text(json.dumps(tariff_cfg, indent=2))
    log.info("Tariff configuration saved")

def current_tariff() -> dict:
    tariff_cfg   = load_tariff()
    prices       = tariff_cfg.get("prices",       DEFAULT_TARIFF["prices"])
    peak_h       = tariff_cfg.get("peak_hours",   DEFAULT_TARIFF["peak_hours"])
    valley_h     = tariff_cfg.get("valley_hours", DEFAULT_TARIFF["valley_hours"])
    weekend_days = tariff_cfg.get("weekend_days", [5, 6])
    now          = datetime.now()
    hour         = now.hour
    weekend      = now.weekday() in weekend_days

    if weekend or hour in valley_h:
        period = "valley"
    elif hour in peak_h:
        period = "peak"
    else:
        period = "mid"

    return {
        "period":       period,
        "price_kwh":    prices[period],
        "export_kwh":   prices.get("export", 0.06),
        "prices":       prices,
        "hour":         hour,
        "weekend":      weekend,
        "weekend_days": weekend_days,
    }

# ── Sun status ───────────────────────────────────────────────────────────────
def get_sun_status() -> dict:
    s = ha_state("sun.sun")
    if s:
        attrs = s.get("attributes", {})
        return {
            "is_day":    s.get("state") == "above_horizon",
            "elevation": attrs.get("elevation"),
            "rising":    attrs.get("rising"),
            "source":    "sun.sun",
        }
    hour = datetime.now().hour
    return {"is_day": 8 <= hour <= 20, "elevation": None, "source": "fallback"}

# ── Sensor reading ───────────────────────────────────────────────────────────
def read_sensors() -> dict:
    return {
        "battery_soc":        ha_float(cfg("sensor_battery_soc",       "sensor.battery_state_of_capacity")),
        "battery_power":      ha_float(cfg("sensor_battery_power",      "sensor.battery_charge_discharge_power")),
        "grid_power":         ha_float(cfg("sensor_grid_power",         "sensor.acometida_general_power")),
        "solar_current_hour": ha_float(cfg("sensor_solar_current_hour", "sensor.energy_current_hour")),
        "solar_next_hour":    ha_float(cfg("sensor_solar_next_hour",    "sensor.energy_next_hour")),
        "solar_today":        ha_float(cfg("sensor_solar_today",        "sensor.energy_production_today")),
        "solar_tomorrow":     ha_float(cfg("sensor_solar_tomorrow",     "sensor.energy_production_tomorrow")),
        "temp_outdoor":       ha_float(cfg("sensor_temp_outdoor",       "sensor.ebusd_broadcast_outsidetemp_temp2")),
        "temp_indoor":        ha_float(cfg("sensor_temp_salon",         "sensor.media_salon")),
        "pool_hours_day":     ha_float(cfg("sensor_pool_hours_day",     "sensor.depuradora_encendida_24h")),
        "pool_hours_week":    ha_float(cfg("sensor_pool_hours_week",    "sensor.depuradora_encendida_semana")),
        "dishwasher_state":   ha_str(cfg("sensor_dishwasher_state",     "sensor.lavavajillas_operation_state")),
        "ts":                 datetime.now().isoformat(),
    }

# ── Battery direct control ───────────────────────────────────────────────────
def set_battery_charge_target(target_soc: int, charge_power_w: int = 3000) -> bool:
    ok = all([
        ha_set_number(cfg("number_battery_charge_cutoff", "number.battery_grid_charge_cutoff_soc"), target_soc),
        ha_set_number(cfg("number_battery_charge_power",  "a186f9599e9cad7127bca381f7a8bfb2"), charge_power_w),
        ha_set_number(cfg("number_battery_backup_soc",    "de4a2bf18222f354228cdb112b65e882"), target_soc),
        ha_switch(cfg("switch_battery_force_charge",      "02db00e10018b01211507db92819a25a"), True),
        ha_set_select(cfg("select_battery_mode",          "select.battery_working_mode"), "time_of_use_luna2000"),
    ])
    log.info(f"  Battery → charge to {target_soc}% @ {charge_power_w}W — {'OK' if ok else 'ERROR'}")
    return ok

def set_battery_self_consumption(min_soc: int = 20) -> bool:
    ok = all([
        ha_set_number(cfg("number_battery_charge_cutoff", "number.battery_grid_charge_cutoff_soc"), min_soc),
        ha_set_select(cfg("select_battery_mode",          "select.battery_working_mode"), "maximise_self_consumption"),
    ])
    log.info(f"  Battery → self-consumption (min {min_soc}%) — {'OK' if ok else 'ERROR'}")
    return ok

# ── Consumption estimation & optimal SOC ─────────────────────────────────────
_consumption_cache: dict = {"kw": 0.5, "updated": None}

def _refresh_consumption_cache():
    global _consumption_cache
    entity      = cfg("sensor_grid_power", "sensor.acometida_general_power")
    rows        = ha_history(entity, days=14)
    night_watts = []
    for row in rows:
        try:
            ts  = datetime.fromisoformat(row["last_changed"].replace("Z", "+00:00"))
            val = float(row["state"])
            if ts.hour >= 22 or ts.hour < 8:
                night_watts.append(abs(val))
        except (KeyError, ValueError, TypeError):
            continue
    kw = (sum(night_watts) / len(night_watts) / 1000) if len(night_watts) >= 20 else 0.5
    _consumption_cache = {"kw": round(kw, 3), "updated": datetime.now()}
    log.info(f"  Consumption cache refreshed: {kw:.3f} kW avg night")

def _get_avg_night_consumption_kw() -> float:
    now = datetime.now()
    if (_consumption_cache["updated"] is None or
            (now - _consumption_cache["updated"]).total_seconds() > 3600):
        threading.Thread(target=_refresh_consumption_cache, daemon=True).start()
    return _consumption_cache["kw"]

def calculate_optimal_soc(sensors: dict) -> dict:
    """Estimate valley charge target balancing overnight consumption and solar forecast."""
    now  = datetime.now()
    hour = now.hour
    hours_until_solar = (8 - hour) if hour < 8 else (24 - hour + 8)

    avg_kw       = _get_avg_night_consumption_kw()
    needed_kwh   = avg_kw * hours_until_solar
    battery_kwh  = float(cfg("battery_capacity_kwh", 10.0))
    soc_coverage = (needed_kwh / battery_kwh) * 100

    solar_tm = sensors.get("solar_tomorrow", 0)
    if solar_tm < 2:
        solar_adj = +25
    elif solar_tm < 5:
        solar_adj = +10
    elif solar_tm > 15:
        solar_adj = -15
    elif solar_tm > 10:
        solar_adj = -5
    else:
        solar_adj = 0

    target = max(30, min(95, round(soc_coverage + solar_adj + 10)))

    return {
        "target_soc":     target,
        "needed_kwh":     round(needed_kwh, 2),
        "avg_night_kw":   avg_kw,
        "solar_tomorrow": solar_tm,
        "hours_covered":  hours_until_solar,
        "solar_adj":      solar_adj,
    }

# ── Storm detection (AEMET OpenData) ─────────────────────────────────────────
STORM_CONDITIONS = {"lightning", "lightning-rainy", "exceptional", "hail", "pouring"}

def is_storm_forecast() -> bool:
    entity = cfg("sensor_weather", "weather.aemet")
    s = ha_state(entity)
    if not s:
        return False
    if s.get("state", "").lower() in STORM_CONDITIONS:
        return True
    for entry in s.get("attributes", {}).get("forecast", [])[:8]:
        if entry.get("condition", "").lower() in STORM_CONDITIONS:
            return True
    return False

# ── ML Model ─────────────────────────────────────────────────────────────────
def _history_to_df(rows: list):
    records = []
    for row in rows:
        try:
            ts  = datetime.fromisoformat(row["last_changed"].replace("Z", "+00:00"))
            val = float(row["state"])
            records.append({"ts": ts, "value": val})
        except (KeyError, ValueError, TypeError):
            continue
    if len(records) < 48:   # 48 = 4 readings/h × 12 hours — minimum viable dataset
        return None
    df = pd.DataFrame(records).set_index("ts").sort_index()
    df["hour"]        = df.index.hour
    df["weekday"]     = df.index.weekday
    df["month"]       = df.index.month
    df["lag1"]        = df["value"].shift(1)
    df["lag4"]        = df["value"].shift(4)
    df["roll4"]       = df["value"].rolling(4).mean()
    df["solar_proxy"] = df.index.hour.map(lambda h: 1 if 8 <= h <= 19 else 0)
    return df.dropna()

FEATURES = ["hour", "weekday", "month", "lag1", "lag4", "roll4", "solar_proxy"]

def train_model() -> bool:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    import numpy as np

    log.info("═══ ML Training started ═══")
    entity = cfg("sensor_battery_soc", "sensor.battery_state_of_capacity")
    rows = []

    # Primary: InfluxDB (years of retention — far better for ML)
    if cfg("influxdb_url", ""):
        rows, _err = ha_history_influx(entity, days=365)
        if rows:
            log.info(f"  Source: InfluxDB — {len(rows)} records / 365d")

    # Fallback: HA recorder with progressive window (default 10d retention)
    if not rows:
        log.info("  InfluxDB not configured or empty — using HA recorder")
        for days in [60, 30, 14, 7]:
            rows = ha_history(entity, days=days)
            if rows:
                log.info(f"  Source: HA recorder — {len(rows)} records / {days}d")
                break
            log.warning(f"  HA recorder {days}d empty — trying shorter window")

    df = _history_to_df(rows)
    if df is None:
        log.warning(f"Insufficient history ({len(rows)} samples). Rules-only mode.")
        return False

    X, y = df[FEATURES], df["value"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr",    GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                             learning_rate=0.08, subsample=0.8, random_state=42)),
    ])
    scores = cross_val_score(pipe, X, y, cv=3, scoring="r2")
    pipe.fit(X, y)
    joblib.dump({"pipeline": pipe, "features": FEATURES,
                 "trained_at": datetime.now().isoformat(),
                 "r2_cv_mean": float(np.mean(scores)), "n_samples": len(df)}, MODEL_FILE)
    log.info(f"Model trained: {len(df)} samples, R²={np.mean(scores):.3f}")
    return True

def predict_soc(sensors: dict) -> dict:
    if MODEL_FILE.exists():
        try:
            art = joblib.load(MODEL_FILE)
            now = datetime.now()
            # Use actual sun.sun elevation for solar proxy (correct for Guadarrama lat 40.67°N)
            sun_data   = get_sun_status()
            solar_live = 1 if sun_data.get("is_day") else 0
            X   = pd.DataFrame([{
                "hour":        now.hour,
                "weekday":     now.weekday(),
                "month":       now.month,
                "lag1":        sensors["battery_soc"],
                "lag4":        sensors["battery_soc"],
                "roll4":       sensors["battery_soc"],
                "solar_proxy": solar_live,
            }])
            pred = float(art["pipeline"].predict(X[FEATURES])[0])
            return {"predicted_soc": round(max(0.0, min(100.0, pred)), 1),
                    "method": "ml", "r2": art.get("r2_cv_mean"), "trained_at": art.get("trained_at")}
        except Exception as e:
            log.warning(f"ML prediction error: {e}")
    solar_tm = sensors.get("solar_tomorrow", 0)
    pred = 80.0 if solar_tm < cfg("solar_tomorrow_irrisoria_kwh", 2.0) else 50.0
    return {"predicted_soc": pred, "method": "rules_fallback"}

# ── Heat pump logic ──────────────────────────────────────────────────────────
def _is_summer() -> bool:
    m = datetime.now().month
    return cfg("summer_start_month", 6) <= m <= cfg("summer_end_month", 9)

def decide_heat_pump(sensors: dict) -> dict:
    soc        = sensors["battery_soc"]
    batt_power = sensors["battery_power"]
    temp_in    = sensors["temp_indoor"]
    sun        = get_sun_status()
    daytime    = sun["is_day"]
    free_power = (soc >= 99 and batt_power > 0)

    if _is_summer():
        entity = cfg("number_hvac_cool", "number.ebusd_ctls2_z1coolingtemp_tempv")
        if free_power:
            target, reason = 16.0, "Free solar power (SOC≥99%, charging from panels)"
        elif temp_in > 26 and daytime:
            target, reason = 20.0, f"Indoor too hot ({temp_in:.1f}°C > 26°C)"
        else:
            target, reason = 25.0, "Summer base setpoint"
    else:
        entity = cfg("number_hvac_heat", "number.ebusd_ctls2_z1manualtemp_tempv")
        if free_power:
            target, reason = 18.5, "Free solar power (SOC≥99%, charging from panels)"
        elif temp_in < 16 and daytime:
            target, reason = 17.0, f"Indoor too cold ({temp_in:.1f}°C < 16°C)"
        else:
            target, reason = 16.0, "Winter base setpoint"

    return {"entity": entity, "target": target, "reason": reason,
            "season": "summer" if _is_summer() else "winter"}

# ── Battery decision ─────────────────────────────────────────────────────────
def decide_battery(sensors: dict, tariff: dict, prediction: dict) -> dict:
    soc       = sensors["battery_soc"]
    period    = tariff["period"]
    prices    = tariff["prices"]
    thr_emerg = cfg("battery_emergency_threshold", 10)
    storm_thr = cfg("battery_storm_threshold", 80)

    storm = is_storm_forecast()
    if storm and soc < storm_thr and period != "peak":
        return {
            "action": "charge", "target_soc": storm_thr, "power_w": 2000,
            "reason": f"⚡ Storm forecast — maintaining {storm_thr}% reserve",
            "storm": True, "alert": True,
            "alert_msg": (f"⚡ *STORM MODE ACTIVATED*\n"
                          f"Charging battery to {storm_thr}% storm reserve\n"
                          f"Current SOC: {soc:.0f}%"),
        }

    if period == "peak":
        if soc < thr_emerg:
            return {
                "action": "charge", "target_soc": 30, "power_w": 1500,
                "reason": f"EMERGENCY during peak: SOC={soc:.0f}% < {thr_emerg}%",
                "alert": True,
                "alert_msg": (f"🚨 *EMERGENCY GRID CHARGE*\n"
                              f"SOC critical: {soc:.0f}% (threshold: {thr_emerg}%)\n"
                              f"Forced charge during PEAK tariff ({prices['peak']}€/kWh)"),
            }
        return {"action": "none", "reason": f"Peak ({prices['peak']}€/kWh): no grid charging"}

    if period == "valley":
        if soc >= 95:
            return {"action": "self_consumption", "reason": f"Valley, battery full ({soc:.0f}%)"}
        optimal = calculate_optimal_soc(sensors)
        target  = optimal["target_soc"]
        if soc < target:
            power = 3000 if (target - soc) > 20 else 2000
            return {
                "action": "charge", "target_soc": target, "power_w": power,
                "reason": (f"Valley — smart target {target}% "
                           f"(~{optimal['needed_kwh']:.1f} kWh overnight, "
                           f"{optimal['solar_tomorrow']:.1f} kWh solar tomorrow)"),
                "optimal": optimal,
            }
        return {"action": "self_consumption",
                "reason": f"Valley, at optimal level ({soc:.0f}% ≥ {target}%)"}

    if period == "mid":
        if soc < thr_emerg:
            return {
                "action": "charge", "target_soc": 30, "power_w": 1500,
                "reason": f"Mid + critical SOC ({soc:.0f}%) → emergency charge",
                "alert": True,
                "alert_msg": (f"🚨 *EMERGENCY GRID CHARGE*\n"
                              f"SOC critical: {soc:.0f}% during mid-peak\n"
                              f"Forced charge to 30%"),
            }
        if soc < 20:
            return {
                "action": "charge", "target_soc": 40, "power_w": 1500,
                "reason": f"Mid + very low SOC ({soc:.0f}%) → charge to 40%",
                "alert": True,
                "alert_msg": (f"⚠️ *GRID CHARGE (mid-peak)*\n"
                              f"Low battery: {soc:.0f}% — charging to 40%\n"
                              f"Tariff: {prices['mid']}€/kWh"),
            }
        return {"action": "self_consumption",
                "reason": f"Mid period, self-consumption (SOC={soc:.0f}%)"}

    return {"action": "none", "reason": "No battery action"}

# ── Instant Telegram alert ───────────────────────────────────────────────────
def send_telegram_alert(message: str):
    """Send an immediate Telegram notification for incidents and forced grid charges."""
    if not cfg("notify_telegram_alerts_enabled", True):
        return
    tg_svc = cfg("notify_telegram_service", "")
    if not tg_svc:
        log.warning("  Telegram alert skipped — no service configured")
        return
    ok = ha_service("notify", tg_svc, {
        "message": message,
        "data": {"parse_mode": "markdown"},
    })
    log.info(f"  Telegram instant alert: {'OK' if ok else 'ERROR'}")

# ── Pool pump + cleaner logic ────────────────────────────────────────────────
_scheduler_ref = None  # set in main()

def decide_pool(sensors: dict, tariff: dict) -> dict:
    month   = datetime.now().month
    summer  = cfg("summer_start_month", 6) <= month <= cfg("summer_end_month", 9)
    hrs_day = sensors.get("pool_hours_day", 0)
    hrs_wk  = sensors.get("pool_hours_week", 0)
    prices  = tariff["prices"]

    if summer and hrs_day >= 1.0:
        return {"action": False, "reason": f"Daily hours met ({hrs_day:.1f}h ≥ 1h/day)"}
    if not summer and hrs_wk >= 1.0:
        return {"action": False, "reason": f"Weekly hours met ({hrs_wk:.1f}h ≥ 1h/week)"}

    solar_now = sensors["solar_current_hour"]
    soc       = sensors["battery_soc"]
    period    = tariff["period"]

    if solar_now > 1.2 and soc > 50:
        return {"action": True, "reason": f"Solar surplus ({solar_now:.2f} kWh/h), SOC={soc:.0f}%"}
    if period == "valley":
        return {"action": True, "reason": f"Valley tariff ({prices['valley']}€/kWh), hours pending"}
    if summer and hrs_day < 0.1 and datetime.now().hour >= 20:
        return {"action": True, "reason": f"Urgent: only {hrs_day:.1f}h today, late hour"}

    return {"action": False, "reason": f"Waiting — period={period}, solar={solar_now:.2f} kWh/h"}

def _start_pool_with_cleaner(pool_sw: str, cleaner_sw: str):
    """Turn on pool pump; start limpiafondos (~1.5 kWh) and auto-stop after 15 min."""
    ha_switch(pool_sw, True)
    if cleaner_sw and _scheduler_ref:
        ha_switch(cleaner_sw, True)
        log.info("  [POOL] Limpiafondos ON — will auto-stop in 15 min")
        shutoff_time = datetime.now() + timedelta(minutes=15)
        _scheduler_ref.add_job(
            lambda: (ha_switch(cleaner_sw, False),
                     log.info("  [POOL] Limpiafondos OFF (15 min timer completed)")),
            "date", run_date=shutoff_time,
            id="cleaner_shutoff", replace_existing=True,
        )

# ── Dishwasher logic ─────────────────────────────────────────────────────────
def decide_dishwasher(sensors: dict, tariff: dict) -> dict:
    state  = sensors.get("dishwasher_state", "")
    period = tariff["period"]
    solar  = sensors["solar_current_hour"]
    soc    = sensors["battery_soc"]
    prices = tariff["prices"]

    if state.lower() == "running":
        return {"action": None, "reason": "Already running", "state": state}
    if state.lower() != "ready":
        return {"action": None, "reason": f"Not ready (state: {state or 'unknown'})", "state": state}

    if solar > 1.5 and soc > 60:
        return {"action": True,
                "reason": f"Solar surplus ({solar:.2f} kWh/h) + good SOC ({soc:.0f}%)",
                "state": state}
    if period == "valley":
        return {"action": True, "reason": f"Valley tariff ({prices['valley']}€/kWh)", "state": state}
    if period == "peak":
        return {"action": False,
                "reason": f"Peak tariff ({prices['peak']}€/kWh) — waiting", "state": state}
    return {"action": False, "reason": "Mid tariff — waiting for solar or valley", "state": state}

# ── Savings tracker ──────────────────────────────────────────────────────────
def _load_savings() -> dict:
    if SAVINGS_FILE.exists():
        try:
            return json.loads(SAVINGS_FILE.read_text())
        except Exception:
            pass
    return {"total_kwh_avoided_peak": 0.0, "total_eur_saved": 0.0,
            "decisions_count": 0, "since": datetime.now().date().isoformat()}

def _update_savings(sensors: dict, tariff: dict):
    if tariff["period"] != "peak" or sensors["battery_power"] >= 0 or sensors["grid_power"] > 500:
        return
    savings    = _load_savings()
    prices     = tariff["prices"]
    interval_h = cfg("decision_interval_minutes", 15) / 60
    kwh        = min(abs(sensors["battery_power"]) * interval_h / 1000, 2.0)
    eur        = kwh * (prices["peak"] - prices.get("export", 0.06))
    savings["total_kwh_avoided_peak"] = round(savings.get("total_kwh_avoided_peak", 0) + kwh, 3)
    savings["total_eur_saved"]        = round(savings.get("total_eur_saved", 0) + eur, 3)
    savings["decisions_count"]        = savings.get("decisions_count", 0) + 1
    SAVINGS_FILE.write_text(json.dumps(savings, indent=2))

# ── Decision cycle ───────────────────────────────────────────────────────────
def run_cycle() -> dict:
    log.info("━━━ Decision cycle ━━━━━━━━━━━━━━━━━━━━━━")
    sensors    = read_sensors()
    tariff     = current_tariff()
    prediction = predict_soc(sensors)

    decision = {"timestamp": datetime.now().isoformat(), "sensors": sensors,
                "tariff": tariff, "prediction": prediction, "actions": [], "skipped": []}

    # 1. Battery
    bat = decide_battery(sensors, tariff, prediction)
    if bat["action"] == "charge":
        ok = set_battery_charge_target(bat["target_soc"], bat.get("power_w", 3000))
        decision["actions"].append({"type": "battery", "action": "charge",
            "target_soc": bat["target_soc"], "reason": bat["reason"], "ok": ok})
        log.info(f"  [BATTERY] Charge → {bat['target_soc']}% — {bat['reason']}")
        if bat.get("alert") and bat.get("alert_msg"):
            send_telegram_alert(bat["alert_msg"])
    elif bat["action"] == "self_consumption":
        ok = set_battery_self_consumption()
        decision["actions"].append({"type": "battery", "action": "self_consumption",
            "reason": bat["reason"], "ok": ok})
        log.info(f"  [BATTERY] Self-consumption — {bat['reason']}")
    else:
        decision["skipped"].append({"type": "battery", "reason": bat["reason"]})
        log.info(f"  [BATTERY] No action — {bat['reason']}")

    # 2. Heat pump
    hp = decide_heat_pump(sensors)
    ok = ha_set_number(hp["entity"], hp["target"])
    decision["actions"].append({"type": "heat_pump", "entity": hp["entity"],
        "value": hp["target"], "reason": hp["reason"], "ok": ok})
    log.info(f"  [HEAT PUMP/{hp['season'].upper()}] {hp['reason']} → {hp['target']}°C")

    # 3. Pool pump + cleaner
    pool     = decide_pool(sensors, tariff)
    pool_sw  = cfg("switch_pool", "switch.depuradora")
    clean_sw = cfg("switch_pool_cleaner", "switch.limpiafondos")
    if pool["action"]:
        _start_pool_with_cleaner(pool_sw, clean_sw)
        decision["actions"].append({"type": "pool", "action": True,
            "reason": pool["reason"] + " (+ limpiafondos 15 min)", "ok": True})
        log.info(f"  [POOL] ON — {pool['reason']}")
    else:
        ha_switch(pool_sw, False)
        decision["skipped"].append({"type": "pool", "reason": pool["reason"]})
        log.info(f"  [POOL] OFF — {pool['reason']}")

    # 4. Dishwasher
    dw = decide_dishwasher(sensors, tariff)
    if dw["action"] is True:
        dw_sw = cfg("switch_dishwasher", "")
        if dw_sw:
            ok = ha_switch(dw_sw, True)
            decision["actions"].append({"type": "dishwasher", "action": "start",
                "reason": dw["reason"], "ok": ok})
            log.info(f"  [DISHWASHER] START — {dw['reason']}")
        else:
            decision["skipped"].append({"type": "dishwasher",
                "reason": f"Ready ({dw['reason']}) — no control switch configured"})
    elif dw["action"] is False:
        decision["skipped"].append({"type": "dishwasher", "reason": dw["reason"]})
        log.info(f"  [DISHWASHER] Waiting — {dw['reason']}")
    else:
        log.info(f"  [DISHWASHER] {dw['reason']}")

    _update_savings(sensors, tariff)
    _save_decision(decision)
    return decision

def _save_decision(d: dict):
    history = []
    if DECISIONS_FILE.exists():
        try:
            history = json.loads(DECISIONS_FILE.read_text())
        except Exception:
            history = []
    history.append(d)
    DECISIONS_FILE.write_text(json.dumps(history[-500:], indent=2, default=str))

# ── Daily summary notification ───────────────────────────────────────────────
def send_daily_summary():
    """Build daily HTML email + Telegram summary. Both channels independently togglable."""
    today = datetime.now().date().isoformat()
    log.info(f"═══ Daily summary for {today} ═══")

    today_decisions = []
    if DECISIONS_FILE.exists():
        try:
            all_dec = json.loads(DECISIONS_FILE.read_text())
            today_decisions = [d for d in all_dec if d.get("timestamp", "")[:10] == today]
        except Exception as e:
            log.warning(f"Could not load decisions for summary: {e}")

    savings     = _load_savings()
    sensors_now = read_sensors()

    action_counts: dict = {}
    action_details: list = []
    for dec in today_decisions:
        ts            = dec.get("timestamp", "")[:16].replace("T", " ")
        tariff_period = dec.get("tariff", {}).get("period", "?")
        soc           = dec.get("sensors", {}).get("battery_soc", 0)
        for a in dec.get("actions", []):
            key = a["type"]
            action_counts[key] = action_counts.get(key, 0) + 1
            action_details.append({"time": ts, "type": key, "reason": a.get("reason", ""),
                                    "ok": a.get("ok", True), "period": tariff_period, "soc": soc})

    n_cycles   = len(today_decisions)
    solar_peak = max((d.get("sensors", {}).get("solar_today", 0) for d in today_decisions), default=0)
    eur_saved  = savings.get("total_eur_saved", 0)
    kwh_peak   = savings.get("total_kwh_avoided_peak", 0)
    since      = savings.get("since", today)

    tg_lines = [
        "⚡ *Energy Optimizer — Daily Report*",
        f"📅 {today}", "",
        f"🔋 Battery SOC now: *{sensors_now['battery_soc']:.0f}%*",
        f"☀️ Solar production today: *{solar_peak:.1f} kWh*",
        f"🔄 Decision cycles: *{n_cycles}*", "",
        "*Actions today:*",
    ]
    if action_details:
        for atype, count in action_counts.items():
            tg_lines.append(f"  • {atype}: {count}x")
        tg_lines.append("")
        tg_lines.append("*Last 5 actions:*")
        for a in action_details[-5:]:
            icon = "✅" if a["ok"] else "❌"
            tg_lines.append(f"  {icon} `{a['time'][-5:]}` [{a['type']}] {a['reason'][:60]}")
    else:
        tg_lines.append("  — No actions taken today")
    tg_lines += ["", f"💰 Savings (since {since}): *€{eur_saved:.2f}* ({kwh_peak:.1f} kWh at peak avoided)"]
    telegram_msg = "\n".join(tg_lines)

    rows_html = ""
    for a in action_details:
        color   = "#4ade80" if a["ok"] else "#f87171"
        p_badge = {"peak": "#f87171", "valley": "#4ade80", "mid": "#fbbf24"}.get(a["period"], "#94a3b8")
        rows_html += (
            f"<tr>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #334155;color:#94a3b8'>{a['time'][-8:]}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #334155'>"
            f"<span style='background:rgba(56,189,248,.15);color:#38bdf8;padding:2px 7px;"
            f"border-radius:4px;font-size:12px'>{a['type']}</span></td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #334155'>"
            f"<span style='background:{p_badge}33;color:{p_badge};padding:1px 5px;"
            f"border-radius:3px;font-size:11px'>{a['period']}</span></td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #334155;font-size:12px;"
            f"color:#e2e8f0'>{a['reason']}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #334155;"
            f"color:{color};font-weight:700'>{'✓' if a['ok'] else '✗'}</td>"
            f"</tr>"
        )
    if not rows_html:
        rows_html = "<tr><td colspan='5' style='padding:12px;color:#94a3b8;text-align:center'>No actions today</td></tr>"

    ks = "display:inline-block;background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 20px;margin:6px;text-align:center;min-width:120px"
    html_body = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;padding:24px;margin:0">
  <h1 style="color:#38bdf8;font-size:20px;margin-bottom:4px">⚡ Energy Optimizer — Daily Report</h1>
  <p style="color:#94a3b8;margin-bottom:20px">{today}</p>
  <div style="margin-bottom:20px">
    <span style="{ks}"><div style="font-size:28px;font-weight:700;color:#38bdf8">{sensors_now['battery_soc']:.0f}%</div>
      <div style="font-size:11px;color:#94a3b8">Battery SOC</div></span>
    <span style="{ks}"><div style="font-size:28px;font-weight:700;color:#4ade80">{solar_peak:.1f}</div>
      <div style="font-size:11px;color:#94a3b8">Solar today (kWh)</div></span>
    <span style="{ks}"><div style="font-size:28px;font-weight:700;color:#fbbf24">{n_cycles}</div>
      <div style="font-size:11px;color:#94a3b8">Cycles run</div></span>
    <span style="{ks}"><div style="font-size:28px;font-weight:700;color:#4ade80">€{eur_saved:.2f}</div>
      <div style="font-size:11px;color:#94a3b8">Total savings</div></span>
  </div>
  <h2 style="color:#94a3b8;font-size:13px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">Actions taken today</h2>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead><tr style="background:#1e293b">
      <th style="padding:8px;text-align:left;color:#94a3b8;font-weight:500">Time</th>
      <th style="padding:8px;text-align:left;color:#94a3b8;font-weight:500">Type</th>
      <th style="padding:8px;text-align:left;color:#94a3b8;font-weight:500">Period</th>
      <th style="padding:8px;text-align:left;color:#94a3b8;font-weight:500">Reason</th>
      <th style="padding:8px;text-align:left;color:#94a3b8;font-weight:500">OK</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <p style="margin-top:20px;font-size:12px;color:#475569">
    Savings tracked since {since}: {kwh_peak:.1f} kWh covered at peak tariff → €{eur_saved:.2f} saved
  </p>
</body></html>"""

    email_svc   = cfg("notify_email_service", "")
    email_to    = cfg("notify_email_target", "")
    tg_svc      = cfg("notify_telegram_service", "")
    email_on    = cfg("notify_email_enabled", True)
    tg_daily_on = cfg("notify_telegram_daily_enabled", True)

    sent_any = False
    if email_on and email_svc and email_to:
        ok = ha_service("notify", email_svc, {
            "title": f"⚡ Energy Optimizer — {today}",
            "message": f"Daily report for {today}. See HTML for details.",
            "target": [email_to],
            "data": {"html": html_body},
        })
        log.info(f"  Email ({email_svc} → {email_to}): {'OK' if ok else 'ERROR'}")
        sent_any = True

    if tg_daily_on and tg_svc:
        ok = ha_service("notify", tg_svc, {
            "message": telegram_msg,
            "data": {"parse_mode": "markdown"},
        })
        log.info(f"  Telegram daily summary ({tg_svc}): {'OK' if ok else 'ERROR'}")
        sent_any = True

    if not sent_any:
        log.info("  No notification services configured or all disabled")

    return {"date": today, "cycles": n_cycles, "actions": len(action_details),
            "solar_kwh": solar_peak, "savings_eur": eur_saved}

# ── Flask web panel ──────────────────────────────────────────────────────────
app = Flask(__name__)

# __BASE__ is replaced at request time with the HA ingress path prefix
# so that fetch('__BASE__/api/...') resolves correctly behind the proxy.
PANEL = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Energy Optimizer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0f172a;--s:#1e293b;--b:#334155;--a:#38bdf8;--g:#4ade80;--y:#fbbf24;--r:#f87171;--o:#fb923c;--t:#e2e8f0;--m:#94a3b8;--p:#a78bfa}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t);font-family:system-ui,sans-serif;padding:1rem;font-size:14px}
h1{color:var(--a);font-size:1.3rem;margin-bottom:.8rem;display:flex;align-items:center;gap:.5rem}
h2{font-size:.7rem;color:var(--m);text-transform:uppercase;letter-spacing:.08em;margin:.8rem 0 .5rem}
.tabs{display:flex;gap:.25rem;margin-bottom:1rem;border-bottom:1px solid var(--b);padding-bottom:.5rem;flex-wrap:wrap}
.tab{background:transparent;border:none;color:var(--m);padding:.4rem .9rem;border-radius:.4rem;cursor:pointer;font-size:.8rem;font-weight:600;transition:.15s}
.tab:hover{color:var(--t);background:rgba(255,255,255,.05)}
.tab.active{color:var(--a);background:rgba(56,189,248,.1)}
.tab-content{display:none}
.tab-content.active{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.6rem;margin-bottom:1rem}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-bottom:1rem}
@media(max-width:700px){.grid2{grid-template-columns:1fr}}
.card{background:var(--s);border-radius:.75rem;padding:.9rem;border:1px solid var(--b)}
.metric{font-size:1.8rem;font-weight:700;color:var(--a);line-height:1}
.label{font-size:.72rem;color:var(--m);margin-top:.3rem}
.sub{font-size:.78rem;color:var(--m);margin-top:.15rem}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:.3rem;font-size:.72rem;font-weight:600}
.peak{color:var(--r);background:rgba(248,113,113,.12)}
.valley{color:var(--g);background:rgba(74,222,128,.12)}
.mid{color:var(--y);background:rgba(251,191,36,.12)}
.running{color:var(--g);background:rgba(74,222,128,.12)}
.ready{color:var(--y);background:rgba(251,191,36,.12)}
.btn{background:var(--a);color:#0f172a;border:none;padding:.45rem 1rem;border-radius:.5rem;font-weight:700;cursor:pointer;font-size:.8rem;transition:.15s opacity,.1s transform;display:inline-flex;align-items:center;gap:.35rem}
.btn:hover{opacity:.85}.btn:active{transform:scale(.97)}.btn:disabled{opacity:.4;cursor:default}
.btn-y{background:var(--y)}.btn-g{background:var(--g)}.btn-r{background:var(--r)}.btn-p{background:var(--p)}.btn-sm{padding:.3rem .7rem;font-size:.72rem}
.actions{display:flex;gap:.5rem;margin-bottom:1rem;flex-wrap:wrap}
.batt-card{background:var(--s);border-radius:.75rem;padding:.9rem;border:1px solid var(--b);margin-bottom:1rem}
.batt-header{display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;margin-bottom:.8rem}
.batt-soc{font-size:3rem;font-weight:700;color:var(--a);line-height:1}
.batt-meta{display:flex;flex-direction:column;gap:.3rem}
.batt-target{font-size:.78rem;color:var(--m)}
.batt-target span{color:var(--g);font-weight:700}
.charge-btns{display:flex;gap:.4rem;flex-wrap:wrap}
.storm-badge{background:rgba(248,113,113,.15);border:1px solid var(--r);color:var(--r);padding:.3rem .7rem;border-radius:.5rem;font-size:.75rem;font-weight:700;display:none}
.savings{background:linear-gradient(135deg,rgba(74,222,128,.07),rgba(56,189,248,.07));border:1px solid rgba(74,222,128,.25);border-radius:.75rem;padding:.9rem;display:flex;gap:2rem;align-items:center;flex-wrap:wrap;margin-bottom:1rem}
.savings-num{font-size:2rem;font-weight:700;color:var(--g)}
/* Tariff */
.tariff-note{font-size:.68rem;color:var(--m);margin-bottom:.5rem}
.day-row{display:flex;gap:.4rem;margin-bottom:.8rem;flex-wrap:wrap}
.day-chip{display:inline-flex;align-items:center;gap:.3rem;padding:.3rem .6rem;border-radius:.4rem;font-size:.75rem;font-weight:600;cursor:pointer;border:1px solid var(--b);background:var(--s);color:var(--m);user-select:none;transition:.15s}
.day-chip.weekend{background:rgba(74,222,128,.15);border-color:var(--g);color:var(--g)}
.timeline{display:grid;grid-template-columns:repeat(24,1fr);gap:2px;margin:.4rem 0 .8rem}
.t-hour{text-align:center;font-size:.58rem;padding:.3rem .1rem;border-radius:.2rem;cursor:pointer;transition:.1s}
.t-hour:hover{filter:brightness(1.3)}
.t-hour.peak{background:rgba(248,113,113,.35);color:var(--r)}
.t-hour.valley{background:rgba(74,222,128,.35);color:var(--g)}
.t-hour.mid{background:rgba(251,191,36,.35);color:var(--y)}
.price-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.5rem;margin-bottom:.7rem}
.price-field label{font-size:.68rem;color:var(--m);display:block;margin-bottom:.2rem}
.price-field input{width:100%;background:var(--b);border:1px solid #475569;border-radius:.35rem;padding:.35rem .5rem;color:var(--t);font-size:.85rem}
/* Setup */
.setup-section{background:var(--s);border-radius:.75rem;padding:1rem;border:1px solid var(--b);margin-bottom:1rem}
.setup-section h3{color:var(--a);font-size:.85rem;margin-bottom:.8rem;border-bottom:1px solid var(--b);padding-bottom:.4rem;font-weight:600}
.toggle-row{display:flex;align-items:center;justify-content:space-between;padding:.5rem 0;border-bottom:1px solid rgba(51,65,85,.5)}
.toggle-row:last-child{border-bottom:none}
.toggle-label{font-size:.82rem;color:var(--t)}
.toggle-desc{font-size:.68rem;color:var(--m);margin-top:.1rem}
.toggle{position:relative;display:inline-block;width:40px;height:22px}
.toggle input{opacity:0;width:0;height:0}
.slider-sw{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:var(--b);border-radius:22px;transition:.2s}
.slider-sw:before{position:absolute;content:"";height:16px;width:16px;left:3px;bottom:3px;background:var(--m);border-radius:50%;transition:.2s}
input:checked+.slider-sw{background:var(--a)}
input:checked+.slider-sw:before{transform:translateX(18px);background:#0f172a}
.range-row{padding:.6rem 0;border-bottom:1px solid rgba(51,65,85,.5)}
.range-row:last-child{border-bottom:none}
.range-header{display:flex;justify-content:space-between;margin-bottom:.3rem}
.range-name{font-size:.82rem;color:var(--t)}
.range-val{font-size:.82rem;color:var(--a);font-weight:700}
input[type=range]{width:100%;accent-color:var(--a);cursor:pointer}
.setup-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:600px){.setup-grid{grid-template-columns:1fr}}
.save-row{display:flex;gap:.5rem;margin-top:1rem;align-items:center;flex-wrap:wrap}
.save-hint{font-size:.7rem;color:var(--m)}
/* Toast */
.toast{position:fixed;bottom:1.2rem;right:1.2rem;padding:.6rem 1.2rem;border-radius:.5rem;font-weight:600;font-size:.85rem;opacity:0;transition:.25s opacity;pointer-events:none;z-index:999;max-width:320px}
.toast.show{opacity:1}
.toast.ok{background:rgba(74,222,128,.95);color:#0f172a}
.toast.err{background:rgba(248,113,113,.95);color:#0f172a}
.toast.info{background:rgba(56,189,248,.95);color:#0f172a}
/* Table */
table{width:100%;border-collapse:collapse;font-size:.78rem}
td,th{padding:.4rem .5rem;border-bottom:1px solid var(--b);text-align:left}
th{color:var(--m);font-weight:500}
.ok-c{color:var(--g)}.skip{color:var(--m)}.err-c{color:var(--r)}
.tag{display:inline-block;padding:.1rem .4rem;border-radius:.25rem;font-size:.68rem;background:var(--b);color:var(--m)}
.tag.bat{background:rgba(56,189,248,.12);color:var(--a)}
.tag.hp{background:rgba(74,222,128,.12);color:var(--g)}
.tag.pool{background:rgba(251,191,36,.12);color:var(--y)}
.tag.dw{background:rgba(251,146,60,.12);color:var(--o)}
/* Weather */
.weather-card{background:var(--s);border-radius:.75rem;padding:.9rem;border:1px solid var(--b);margin-bottom:1rem}
.weather-main{display:flex;gap:1rem;align-items:center;margin-bottom:.8rem;flex-wrap:wrap}
.weather-icon{font-size:2.4rem;line-height:1}
.weather-temp{font-size:2rem;font-weight:700;color:var(--a)}
.weather-cond{font-size:.82rem;color:var(--m);text-transform:capitalize;margin-top:.1rem}
.forecast-row{display:flex;gap:.4rem;overflow-x:auto;padding-bottom:.2rem}
.forecast-day{background:var(--bg);border-radius:.5rem;padding:.45rem .55rem;min-width:66px;text-align:center;font-size:.72rem;flex-shrink:0}
.forecast-day .fday{color:var(--m);margin-bottom:.15rem}
.forecast-day .ftemp{color:var(--t);font-weight:600}
.forecast-day .fmin{color:var(--m)}
.storm-alert-w{background:rgba(248,113,113,.12);border:1px solid var(--r);color:var(--r);padding:.4rem .8rem;border-radius:.5rem;font-size:.78rem;font-weight:700;margin-bottom:.7rem}
</style>
</head>
<body>
<h1>⚡ Energy Optimizer <span id="ver" style="font-size:.75rem;color:var(--m);font-weight:400">v2.5.7</span></h1>
<div id="notify" class="toast"></div>
<div class="tabs">
  <button class="tab active" onclick="showTab('dashboard')">📊 Dashboard</button>
  <button class="tab" onclick="showTab('charts')">📈 Charts</button>
  <button class="tab" onclick="showTab('tariff')">⚡ Tariff</button>
  <button class="tab" onclick="showTab('setup')">⚙️ Setup</button>
</div>

<!-- DASHBOARD -->
<div id="tab-dashboard" class="tab-content active">
  <div id="kpis" class="grid"></div>
  <div id="savings-box"></div>
  <div id="weather-box"></div>
  <div class="batt-card">
    <h2 style="margin-top:0">🔋 Battery</h2>
    <div class="batt-header">
      <div>
        <div class="batt-soc" id="batt-soc-big">--%</div>
        <div class="label" id="batt-dir">--</div>
      </div>
      <div class="batt-meta">
        <div class="batt-target" id="batt-target-line"></div>
        <div id="storm-badge" class="storm-badge">⚡ STORM MODE</div>
      </div>
    </div>
    <div class="charge-btns">
      <button class="btn btn-sm" onclick="chargeToTarget(30)">Charge 30%</button>
      <button class="btn btn-sm" onclick="chargeToTarget(50)">Charge 50%</button>
      <button class="btn btn-sm" onclick="chargeToTarget(80)">Charge 80%</button>
      <button class="btn btn-sm" onclick="chargeToTarget(99)">Charge 99%</button>
      <button class="btn btn-sm btn-g" onclick="selfConsumption()">☀ Self-consumption</button>
    </div>
  </div>
  <div class="actions">
    <button id="btn-run" class="btn" onclick="runCycle()">▶ Run cycle now</button>
    <button id="btn-retrain" class="btn btn-y" onclick="retrain()">🔁 Retrain model</button>
    <button id="btn-summary" class="btn btn-p" onclick="sendSummary()">📧 Send daily summary</button>
  </div>
  <h2>Recent decisions</h2>
  <div class="card" style="overflow-x:auto">
    <table><thead><tr><th>Time</th><th>Type</th><th>Reason</th><th>OK</th></tr></thead>
    <tbody id="log"></tbody></table>
  </div>
</div>

<!-- CHARTS -->
<div id="tab-charts" class="tab-content">
  <div class="grid2">
    <div class="card">
      <h2 style="margin-top:0">Battery SOC — actual vs predicted (24h)</h2>
      <canvas id="socChart"></canvas>
    </div>
    <div class="card">
      <h2 style="margin-top:0">Solar production (kWh)</h2>
      <canvas id="solarChart"></canvas>
    </div>
  </div>
  <div class="card" style="margin-bottom:1rem">
    <h2 style="margin-top:0">Daily savings — last 7 days (€)</h2>
    <canvas id="savingsChart" style="max-height:180px"></canvas>
  </div>
  <div class="actions"><button class="btn btn-sm" onclick="loadCharts()">🔄 Refresh</button></div>
</div>

<!-- TARIFF -->
<div id="tab-tariff" class="tab-content">
  <div class="card">
    <h2 style="margin-top:0">Weekend days (all-day valley)</h2>
    <div class="tariff-note">Click to toggle. Days marked in green use valley tariff the entire day.</div>
    <div class="day-row" id="tariff-days"></div>

    <h2>Weekday hour schedule</h2>
    <div class="tariff-note">Click to cycle: <span class="badge valley">valley</span> → <span class="badge mid">mid</span> → <span class="badge peak">peak</span>. Applies to non-weekend days.</div>
    <div class="timeline" id="tariff-timeline"></div>

    <h2>Prices</h2>
    <div class="price-grid">
      <div class="price-field"><label>Peak (€/kWh)</label><input id="p-peak" type="number" step="0.01" min="0"></div>
      <div class="price-field"><label>Mid (€/kWh)</label><input id="p-mid" type="number" step="0.01" min="0"></div>
      <div class="price-field"><label>Valley (€/kWh)</label><input id="p-valley" type="number" step="0.01" min="0"></div>
      <div class="price-field"><label>Export (€/kWh)</label><input id="p-export" type="number" step="0.01" min="0"></div>
    </div>
    <div style="display:flex;gap:.5rem;margin-top:.5rem;flex-wrap:wrap">
      <button class="btn btn-g btn-sm" onclick="saveTariff()">💾 Save tariff</button>
      <button class="btn btn-sm" style="background:var(--b);color:var(--m)" onclick="resetTariff()">↩ Reset to defaults</button>
    </div>
  </div>
</div>

<!-- SETUP -->
<div id="tab-setup" class="tab-content">
  <div class="setup-section">
    <h3>🔔 Notifications</h3>
    <div class="toggle-row">
      <div><div class="toggle-label">Email — daily summary</div>
        <div class="toggle-desc">Send HTML report every day at the configured time</div></div>
      <label class="toggle"><input type="checkbox" id="sw-email" onchange="markDirty()"><span class="slider-sw"></span></label>
    </div>
    <div class="toggle-row">
      <div><div class="toggle-label">Telegram — daily summary</div>
        <div class="toggle-desc">Send concise Telegram report every day</div></div>
      <label class="toggle"><input type="checkbox" id="sw-tg-daily" onchange="markDirty()"><span class="slider-sw"></span></label>
    </div>
    <div class="toggle-row">
      <div><div class="toggle-label">Telegram — instant alerts</div>
        <div class="toggle-desc">Emergency charges, storm mode, forced grid charges</div></div>
      <label class="toggle"><input type="checkbox" id="sw-tg-alerts" onchange="markDirty()"><span class="slider-sw"></span></label>
    </div>
    <div class="range-row" style="margin-top:.6rem">
      <div class="range-header"><span class="range-name">Daily summary time</span>
        <span class="range-val" id="val-notify-time">23:00</span></div>
      <input type="time" id="inp-notify-time" value="23:00"
        oninput="document.getElementById('val-notify-time').textContent=this.value;markDirty()"
        style="background:var(--b);border:1px solid #475569;border-radius:.35rem;padding:.3rem .5rem;color:var(--t);font-size:.85rem;width:140px">
    </div>
  </div>
  <div class="setup-section">
    <h3>🔋 Battery thresholds</h3>
    <div class="setup-grid">
      <div>
        <div class="range-row">
          <div class="range-header"><span class="range-name">Emergency threshold</span><span class="range-val" id="val-emerg">10%</span></div>
          <input type="range" id="rng-emerg" min="1" max="30" value="10" oninput="document.getElementById('val-emerg').textContent=this.value+'%';markDirty()">
          <div class="toggle-desc">Force charge at any tariff below this SOC</div>
        </div>
        <div class="range-row">
          <div class="range-header"><span class="range-name">Low threshold</span><span class="range-val" id="val-low">30%</span></div>
          <input type="range" id="rng-low" min="10" max="50" value="30" oninput="document.getElementById('val-low').textContent=this.value+'%';markDirty()">
        </div>
      </div>
      <div>
        <div class="range-row">
          <div class="range-header"><span class="range-name">Medium threshold</span><span class="range-val" id="val-med">50%</span></div>
          <input type="range" id="rng-med" min="30" max="80" value="50" oninput="document.getElementById('val-med').textContent=this.value+'%';markDirty()">
        </div>
        <div class="range-row">
          <div class="range-header"><span class="range-name">Storm reserve</span><span class="range-val" id="val-storm">80%</span></div>
          <input type="range" id="rng-storm" min="50" max="100" value="80" oninput="document.getElementById('val-storm').textContent=this.value+'%';markDirty()">
          <div class="toggle-desc">Charge to this level when storm is forecast</div>
        </div>
      </div>
    </div>
  </div>
  <div class="setup-section">
    <h3>⏱ Automation</h3>
    <div class="range-row">
      <div class="range-header"><span class="range-name">Decision interval</span><span class="range-val" id="val-interval">15 min</span></div>
      <input type="range" id="rng-interval" min="5" max="60" step="5" value="15" oninput="document.getElementById('val-interval').textContent=this.value+' min';markDirty()">
      <div class="toggle-desc">How often the optimization cycle runs (restart to apply)</div>
    </div>
    <div class="range-row">
      <div class="range-header"><span class="range-name">Battery capacity</span><span class="range-val" id="val-batt-cap">10 kWh</span></div>
      <input type="range" id="rng-batt-cap" min="2" max="30" step="0.5" value="10" oninput="document.getElementById('val-batt-cap').textContent=this.value+' kWh';markDirty()">
    </div>
  </div>
  <div class="save-row">
    <button id="btn-save-setup" class="btn btn-g" onclick="saveSetup()">💾 Save settings</button>
    <button class="btn btn-sm" style="background:var(--b);color:var(--t)" onclick="loadSetup()">↩ Reset</button>
    <span class="save-hint" id="setup-hint"></span>
  </div>
  <div class="setup-section" style="margin-top:1rem">
    <h3>🔌 Data sources</h3>
    <div style="font-size:.72rem;color:var(--m);margin-bottom:.7rem">
      InfluxDB URL and credentials are set in the add-on options (config). Use this to verify connectivity.
    </div>
    <button class="btn btn-sm" id="btn-ds-test" onclick="testDataSources()">🔍 Test connections</button>
    <div id="ds-result" style="margin-top:.7rem;font-size:.78rem;line-height:1.7"></div>
  </div>
</div>

<script>
const BASE = "__BASE__";

// ── Tab switching ─────────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('.tab[onclick="showTab(\''+name+'\')"]').classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
  if (name==='charts') loadCharts();
  if (name==='tariff') loadTariff();
  if (name==='setup')  loadSetup();
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let _tt=null;
function notify(msg,type='ok',ms=3500){
  const el=document.getElementById('notify');
  el.textContent=msg; el.className='toast '+type+' show';
  if(_tt) clearTimeout(_tt);
  _tt=setTimeout(()=>el.classList.remove('show'),ms);
}

// ── Tariff editor ─────────────────────────────────────────────────────────────
let tariffCfg=null;
const DAY_NAMES=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];

async function loadTariff(){
  try {
    tariffCfg=await fetch(BASE+'/api/tariff').then(r=>r.json());
    renderDayRow();
    renderTariffTimeline();
    document.getElementById('p-peak').value   = tariffCfg.prices.peak;
    document.getElementById('p-mid').value    = tariffCfg.prices.mid;
    document.getElementById('p-valley').value = tariffCfg.prices.valley;
    document.getElementById('p-export').value = tariffCfg.prices.export;
  } catch(e){ notify('Error loading tariff','err'); }
}

function renderDayRow(){
  const wd=tariffCfg.weekend_days||[5,6];
  document.getElementById('tariff-days').innerHTML=DAY_NAMES.map((d,i)=>
    '<div class="day-chip'+(wd.includes(i)?' weekend':'')+'" onclick="toggleDay('+i+')">'+
    (wd.includes(i)?'🌿 ':'')+ d+'</div>'
  ).join('');
}

function toggleDay(i){
  const wd=tariffCfg.weekend_days||[5,6];
  tariffCfg.weekend_days=wd.includes(i)?wd.filter(x=>x!==i):[...wd,i].sort((a,b)=>a-b);
  renderDayRow();
}

function renderTariffTimeline(){
  if(!tariffCfg) return;
  const {peak_hours,valley_hours}=tariffCfg;
  document.getElementById('tariff-timeline').innerHTML=
    Array.from({length:24},(_,h)=>{
      const p=peak_hours.includes(h)?'peak':valley_hours.includes(h)?'valley':'mid';
      return '<div class="t-hour '+p+'" onclick="toggleHour('+h+')" title="'+h+':00">'+h+'</div>';
    }).join('');
}

function toggleHour(h){
  const {peak_hours,valley_hours}=tariffCfg;
  const isPeak=peak_hours.includes(h), isValley=valley_hours.includes(h);
  if(isValley){
    tariffCfg.valley_hours=valley_hours.filter(x=>x!==h);
  } else if(!isPeak){
    tariffCfg.peak_hours=[...peak_hours,h].sort((a,b)=>a-b);
  } else {
    tariffCfg.peak_hours=peak_hours.filter(x=>x!==h);
    tariffCfg.valley_hours=[...valley_hours,h].sort((a,b)=>a-b);
  }
  renderTariffTimeline();
}

async function resetTariff(){
  if(!confirm('Reset tariff to built-in defaults? This will overwrite your current prices and schedule.')) return;
  const r=await fetch(BASE+'/api/tariff/reset',{method:'POST'}).then(r=>r.json());
  if(r.ok){ notify('✓ Tariff reset to defaults'); loadTariff(); }
  else notify('✗ Error resetting tariff','err');
}

async function saveTariff(){
  tariffCfg.prices={
    peak:   parseFloat(document.getElementById('p-peak').value)||0.30,
    mid:    parseFloat(document.getElementById('p-mid').value)||0.18,
    valley: parseFloat(document.getElementById('p-valley').value)||0.08,
    export: parseFloat(document.getElementById('p-export').value)||0.06,
  };
  const r=await fetch(BASE+'/api/tariff',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(tariffCfg)}).then(r=>r.json());
  notify(r.ok?'✓ Tariff saved':'✗ Error saving tariff',r.ok?'ok':'err');
}

// ── Setup panel ───────────────────────────────────────────────────────────────
let _dirty=false;
function markDirty(){_dirty=true;document.getElementById('setup-hint').textContent='Unsaved changes';}

async function loadSetup(){
  try {
    const s=await fetch(BASE+'/api/setup').then(r=>r.json());
    document.getElementById('sw-email').checked      =s.notify_email_enabled!==false;
    document.getElementById('sw-tg-daily').checked   =s.notify_telegram_daily_enabled!==false;
    document.getElementById('sw-tg-alerts').checked  =s.notify_telegram_alerts_enabled!==false;
    document.getElementById('inp-notify-time').value =s.notify_daily_time||'23:00';
    document.getElementById('val-notify-time').textContent=s.notify_daily_time||'23:00';
    const set=(id,valId,v,sfx)=>{document.getElementById(id).value=v;document.getElementById(valId).textContent=v+sfx;};
    set('rng-emerg','val-emerg',s.battery_emergency_threshold??10,'%');
    set('rng-low','val-low',s.battery_low_threshold??30,'%');
    set('rng-med','val-med',s.battery_medium_threshold??50,'%');
    set('rng-storm','val-storm',s.battery_storm_threshold??80,'%');
    set('rng-interval','val-interval',s.decision_interval_minutes??15,' min');
    set('rng-batt-cap','val-batt-cap',s.battery_capacity_kwh??10,' kWh');
    _dirty=false; document.getElementById('setup-hint').textContent='';
  } catch(e){ notify('Error loading setup','err'); }
}

async function saveSetup(){
  const btn=document.getElementById('btn-save-setup');
  btn.disabled=true; btn.textContent='⏳ Saving…';
  try {
    const r=await fetch(BASE+'/api/setup',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        notify_email_enabled:            document.getElementById('sw-email').checked,
        notify_telegram_daily_enabled:   document.getElementById('sw-tg-daily').checked,
        notify_telegram_alerts_enabled:  document.getElementById('sw-tg-alerts').checked,
        notify_daily_time:               document.getElementById('inp-notify-time').value,
        battery_emergency_threshold:     parseInt(document.getElementById('rng-emerg').value),
        battery_low_threshold:           parseInt(document.getElementById('rng-low').value),
        battery_medium_threshold:        parseInt(document.getElementById('rng-med').value),
        battery_storm_threshold:         parseInt(document.getElementById('rng-storm').value),
        decision_interval_minutes:       parseInt(document.getElementById('rng-interval').value),
        battery_capacity_kwh:            parseFloat(document.getElementById('rng-batt-cap').value),
      })
    }).then(r=>r.json());
    if(r.ok){notify('✓ Settings saved');_dirty=false;document.getElementById('setup-hint').textContent='Saved ✓';}
    else notify('✗ Error saving settings','err');
  } catch(e){ notify('✗ Error: '+e.message,'err'); }
  finally{btn.disabled=false;btn.textContent='💾 Save settings';}
}

// ── Charts ────────────────────────────────────────────────────────────────────
let socInst=null,solInst=null,savInst=null;
function mkChart(id,type,labels,datasets,yExtra={},maxH=null){
  const el=document.getElementById(id); if(!el) return null;
  if(maxH) el.style.maxHeight=maxH;
  return new Chart(el.getContext('2d'),{type,data:{labels,datasets},
    options:{responsive:true,maintainAspectRatio:true,
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:11}}}},
      scales:{x:{ticks:{color:'#94a3b8',maxTicksLimit:10,font:{size:10}},grid:{color:'#334155'}},
              y:{ticks:{color:'#94a3b8',font:{size:10}},grid:{color:'#334155'},...yExtra}}}});
}

async function loadCharts(){
  try {
    const cd=await fetch(BASE+'/api/chart-data').then(r=>r.json());
    // Combine past + future labels for x-axis
    const futureLabels=cd.soc.future_labels||[];
    const futurePred=cd.soc.future_predicted||[];
    const nPast=cd.soc.labels.length;
    const allLabels=[...cd.soc.labels,...futureLabels];
    // Pad arrays: actual and past-predicted are null in the future segment
    const actualData=[...cd.soc.actual,...Array(futureLabels.length).fill(null)];
    const pastPredData=[...(cd.soc.predicted||[]),...Array(futureLabels.length).fill(null)];
    // Future forecast: null for past segment, values for future
    const futureData=[...Array(nPast).fill(null),...futurePred];
    if(socInst) socInst.destroy();
    socInst=new Chart(document.getElementById('socChart').getContext('2d'),{
      type:'line',
      data:{labels:allLabels,datasets:[
        {label:'Actual SOC %',data:actualData,borderColor:'#38bdf8',backgroundColor:'rgba(56,189,248,.08)',fill:true,tension:.3,pointRadius:0},
        {label:'Predicted (past) %',data:pastPredData,borderColor:'#a78bfa',backgroundColor:'transparent',fill:false,tension:.3,pointRadius:0,borderDash:[4,3]},
        {label:'Forecast (next 8h) %',data:futureData,borderColor:'#4ade80',backgroundColor:'rgba(74,222,128,.06)',fill:true,tension:.3,pointRadius:0,borderDash:[6,3]},
      ]},
      options:{responsive:true,maintainAspectRatio:true,
        plugins:{legend:{labels:{color:'#94a3b8',font:{size:11}}}},
        scales:{
          x:{ticks:{color:'#94a3b8',maxTicksLimit:10,font:{size:10}},grid:{color:'#334155'}},
          y:{min:0,max:100,ticks:{color:'#94a3b8',font:{size:10}},grid:{color:'#334155'}}
        }
      }
    });
    if(solInst) solInst.destroy();
    solInst=mkChart('solarChart','bar',['Yesterday','Today','Tomorrow'],[
      {label:'kWh',data:[cd.solar.yesterday,cd.solar.today,cd.solar.tomorrow],
       backgroundColor:['rgba(251,191,36,.5)','rgba(74,222,128,.7)','rgba(56,189,248,.5)'],
       borderColor:['#fbbf24','#4ade80','#38bdf8'],borderWidth:1,borderRadius:4}]);
    if(savInst) savInst.destroy();
    if(cd.savings_daily&&cd.savings_daily.labels.length>0){
      savInst=mkChart('savingsChart','bar',cd.savings_daily.labels,[
        {label:'€ saved',data:cd.savings_daily.values,backgroundColor:'rgba(74,222,128,.6)',borderColor:'#4ade80',borderWidth:1,borderRadius:4}
      ],{},'180px');
    }
  } catch(e){ console.warn('Chart error',e); }
}

// ── Status & decisions ────────────────────────────────────────────────────────
const PC={peak:'peak',valley:'valley',mid:'mid'};
const tagMap={battery:'bat',heat_pump:'hp',pool:'pool',dishwasher:'dw'};

async function load(){
  try {
    const [s,d,sv]=await Promise.all([
      fetch(BASE+'/api/status').then(r=>r.json()),
      fetch(BASE+'/api/decisions?limit=30').then(r=>r.json()),
      fetch(BASE+'/api/savings').then(r=>r.json()),
    ]);
    const {sensors:ss,tariff:t,model:m,optimal:opt,storm}=s;
    const dw=ss.dishwasher_state||'--';
    const dwCls=dw.toLowerCase()==='running'?'running':dw.toLowerCase()==='ready'?'ready':'';
    document.getElementById('kpis').innerHTML=`
      <div class="card"><div class="metric">${(ss.solar_today??0).toFixed(1)}</div>
        <div class="label">Solar today (kWh)</div>
        <div class="sub">Next hr: ${(ss.solar_next_hour??0).toFixed(2)} kWh</div></div>
      <div class="card"><div class="metric">${(ss.solar_tomorrow??0).toFixed(1)}</div>
        <div class="label">Solar tomorrow (kWh)</div></div>
      <div class="card"><div class="metric ${PC[t.period]}">${(t.price_kwh??0).toFixed(2)}€</div>
        <div class="label">Price/kWh &nbsp;<span class="badge ${PC[t.period]}">${t.period}</span></div>
        <div class="sub">${t.weekend?'Weekend — valley all day':''}</div></div>
      <div class="card"><div class="metric">${(ss.temp_indoor??0).toFixed(1)}°C</div>
        <div class="label">Indoor temp</div>
        <div class="sub">Outdoor: ${(ss.temp_outdoor??0).toFixed(1)}°C</div></div>
      <div class="card"><div class="metric" style="font-size:1.1rem"><span class="badge ${dwCls}">${dw}</span></div>
        <div class="label">Dishwasher</div></div>
      <div class="card"><div class="metric" style="font-size:1rem">${m?.r2_cv_mean!=null?'R²='+m.r2_cv_mean.toFixed(2):'No model'}</div>
        <div class="label">ML model</div>
        <div class="sub">${m?.n_samples?m.n_samples+' samples':''}</div></div>`;

    const bpow=ss.battery_power??0;
    const bdir=bpow>50?'▲ Charging':bpow<-50?'▼ Discharging':'● Idle';
    document.getElementById('batt-soc-big').textContent=Number(ss.battery_soc??0).toFixed(0)+'%';
    document.getElementById('batt-dir').textContent=bdir+'  '+Math.abs(bpow).toFixed(0)+' W';
    document.getElementById('storm-badge').style.display=storm?'inline-block':'none';
    if(opt) document.getElementById('batt-target-line').innerHTML=
      'Smart target: <span>'+opt.target_soc+'%</span> — '+opt.needed_kwh+' kWh overnight, '+opt.solar_tomorrow+' kWh solar tomorrow';

    document.getElementById('savings-box').innerHTML=`
      <div class="savings">
        <div><div class="savings-num">€${(sv.total_eur_saved??0).toFixed(2)}</div>
          <div class="label">Estimated savings</div></div>
        <div><div class="metric" style="color:var(--g)">${(sv.total_kwh_avoided_peak??0).toFixed(1)}</div>
          <div class="label">kWh covered at peak</div></div>
        <div style="font-size:.72rem;color:var(--m)">Since ${sv.since??'--'}</div>
      </div>`;

    const rows=d.slice().reverse().flatMap(dec=>{
      const ts=new Date(dec.timestamp).toLocaleTimeString('en',{hour:'2-digit',minute:'2-digit'});
      return [
        ...(dec.actions||[]).map(a=>`<tr><td>${ts}</td><td><span class="tag ${tagMap[a.type]||''}">${a.type}</span></td><td class="ok-c">${a.reason}</td><td class="${a.ok?'ok-c':'err-c'}">${a.ok?'✓':'✗'}</td></tr>`),
        ...(dec.skipped||[]).map(a=>`<tr><td>${ts}</td><td><span class="tag">${a.type}</span></td><td class="skip">${a.reason}</td><td class="skip">–</td></tr>`)
      ];
    }).join('');
    document.getElementById('log').innerHTML=rows||'<tr><td colspan="4" style="color:var(--m)">No decisions yet</td></tr>';
  } catch(e){ console.error('Load error',e); }
}

// ── Actions ───────────────────────────────────────────────────────────────────
async function runCycle(){
  const btn=document.getElementById('btn-run');
  btn.disabled=true; btn.textContent='⏳ Running…';
  try{
    const d=await fetch(BASE+'/api/run',{method:'POST'}).then(r=>r.json());
    notify('✓ Cycle done — '+(d.actions?.length??0)+' actions, '+(d.skipped?.length??0)+' skipped');
    load();
  }catch(e){notify('✗ Error running cycle','err');}
  finally{btn.disabled=false;btn.textContent='▶ Run cycle now';}
}

async function retrain(){
  const btn=document.getElementById('btn-retrain');
  btn.disabled=true; btn.textContent='⏳ Training…';
  try{
    const d=await fetch(BASE+'/api/retrain',{method:'POST'}).then(r=>r.json());
    notify(d.message,d.ok?'ok':'info',5000); load();
  }catch(e){notify('✗ Error during training','err');}
  finally{btn.disabled=false;btn.textContent='🔁 Retrain model';}
}

async function chargeToTarget(soc){
  if(!confirm('Charge battery to '+soc+'%?')) return;
  notify('⏳ Sending command…','info',2000);
  const r=await fetch(BASE+'/api/battery/charge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target_soc:soc})}).then(r=>r.json());
  notify(r.ok?'✓ Charging to '+soc+'% started':'✗ Error setting charge',r.ok?'ok':'err'); load();
}

async function selfConsumption(){
  if(!confirm('Switch to self-consumption mode?')) return;
  const r=await fetch(BASE+'/api/battery/self-consumption',{method:'POST'}).then(r=>r.json());
  notify(r.ok?'✓ Self-consumption mode activated':'✗ Error',r.ok?'ok':'err'); load();
}

async function sendSummary(){
  const btn=document.getElementById('btn-summary');
  btn.disabled=true; btn.textContent='⏳ Sending…';
  try{
    const r=await fetch(BASE+'/api/send-summary',{method:'POST'}).then(r=>r.json());
    notify('✓ Summary sent — '+r.cycles+' cycles, €'+(r.savings_eur||0).toFixed(2)+' saved','ok',5000);
  }catch(e){notify('✗ Error sending summary','err');}
  finally{btn.disabled=false;btn.textContent='📧 Send daily summary';}
}

// ── Data sources debug ───────────────────────────────────────────────────────
async function testDataSources(){
  const btn=document.getElementById('btn-ds-test');
  const out=document.getElementById('ds-result');
  btn.disabled=true; btn.textContent='⏳ Testing…'; out.innerHTML='';
  try{
    const r=await fetch(BASE+'/api/influx-debug').then(r=>r.json());
    let influxLine;
    const errLabels={
      not_configured:'add influxdb_url in add-on options',
      no_series:'connected but no data for entity — check entity_id and DB name',
      timeout:'connection timed out — check influxdb_url',
    };
    if(!r.configured){
      influxLine=`<span style="color:var(--m)">InfluxDB — not configured</span> `
        +`<span style="color:var(--m);font-size:.68rem">(add influxdb_url in add-on options)</span>`;
    } else if(r.ok){
      const authNote=r.auth_mode==='no_auth'
        ?` <span style="color:var(--y);font-size:.68rem">(no auth)</span>`:'';
      influxLine=`<span style="color:var(--g)">✓ InfluxDB OK${authNote}</span>`
        +` — <b>${r.records_7d}</b> records (7d) · ${r.first_ts} → ${r.last_ts}`
        +` <span style="color:var(--m);font-size:.68rem">${r.url} / ${r.db}</span>`;
    } else if(r.ping_ok){
      // Can reach InfluxDB but can't get data
      influxLine=`<span style="color:var(--y)">⚠ InfluxDB reachable but no data</span>`
        +` <span style="color:var(--m);font-size:.68rem">`
        +(errLabels[r.error]||r.error||'')
        +` · ${r.url} / ${r.db}</span>`;
    } else {
      // Can't reach at all
      const errMsg=errLabels[r.error]||r.error||'unreachable';
      influxLine=`<span style="color:var(--r)">✗ InfluxDB — ${errMsg}</span>`
        +` <span style="color:var(--m);font-size:.68rem">${r.url} / ${r.db}</span>`;
    }
    let diag='';
    if(r.databases?.length) diag+=`<div style="color:var(--m);font-size:.68rem;margin-top:.2rem">Databases: ${r.databases.join(', ')}</div>`;
    if(r.measurements?.length) diag+=`<div style="color:var(--m);font-size:.68rem">Measurements (${r.db}): ${r.measurements.join(', ')}</div>`;
    if(r.sample_entities?.length) diag+=`<div style="color:var(--m);font-size:.68rem">Battery entity_id tags found: ${r.sample_entities.join(', ')}</div>`;
    else if(r.ping_ok&&!r.ok) diag+=`<div style="color:var(--y);font-size:.68rem">No battery entity_id tags found — entity may use different tag name or measurement</div>`;
    out.innerHTML=`<div>${influxLine}</div>${diag}`
      +`<div style="margin-top:.3rem"><span style="color:var(--a)">HA recorder</span>`
      +` — <b>${r.recorder_7d??'?'}</b> records (1d) · used as fallback when InfluxDB is empty</div>`;
  }catch(e){
    out.innerHTML=`<span style="color:var(--r)">✗ ${e.message}</span>`;
  }finally{
    btn.disabled=false; btn.textContent='🔍 Test connections';
  }
}

// ── Weather widget ────────────────────────────────────────────────────────────
const WEATHER_ICONS={sunny:'☀️',clear:'🌙',partlycloudy:'⛅',cloudy:'☁️',fog:'🌫️',
  rainy:'🌧️',snowy:'❄️','lightning-rainy':'⛈️',lightning:'🌩️',
  windy:'💨',hail:'🌨️',pouring:'🌊',exceptional:'⚠️'};
function weatherIcon(c){return WEATHER_ICONS[c]||'🌡️';}

async function loadWeather(){
  const box=document.getElementById('weather-box');
  try{
    const w=await fetch(BASE+'/api/weather').then(r=>r.json());
    if(!w.ok){box.innerHTML='';return;}
    const stormHtml=w.storm
      ?'<div class="storm-alert-w">⚡ STORM FORECAST — battery will be pre-charged to storm reserve</div>':'';
    const fcHtml=(w.forecast||[]).map(f=>{
      const d=new Date(f.datetime);
      const day=isNaN(d)?f.datetime.slice(0,10):d.toLocaleDateString('en',{weekday:'short',month:'short',day:'numeric'});
      const precip=f.precipitation?`<div style="color:#38bdf8;font-size:.62rem">${f.precipitation}mm</div>`:'';
      return `<div class="forecast-day">
        <div class="fday">${day}</div>
        <div style="font-size:1.1rem">${weatherIcon(f.condition)}</div>
        <div class="ftemp">${f.temperature??'--'}°</div>
        <div class="fmin">${f.templow??'--'}°</div>
        ${precip}
      </div>`;
    }).join('');
    const meta=w.humidity||w.wind_speed
      ?`<div style="font-size:.72rem;color:var(--m);margin-top:.2rem">`
        +(w.humidity?`💧 ${w.humidity}%`:'')
        +(w.humidity&&w.wind_speed?' &nbsp;·&nbsp; ':'')
        +(w.wind_speed?`💨 ${w.wind_speed} km/h`:'')
       +`</div>`:'';
    box.innerHTML=`<div class="weather-card">
      <h2 style="margin-top:0">🌤 Weather forecast</h2>
      ${stormHtml}
      <div class="weather-main">
        <div class="weather-icon">${weatherIcon(w.condition)}</div>
        <div>
          <div class="weather-temp">${w.temperature??'--'}°C</div>
          <div class="weather-cond">${(w.condition||'').replace(/-/g,' ')}</div>
          ${meta}
        </div>
      </div>
      ${fcHtml?`<div class="forecast-row">${fcHtml}</div>`:''}
    </div>`;
  }catch(e){box.innerHTML='';}
}

// ── Init ──────────────────────────────────────────────────────────────────────
load();
loadWeather();
loadCharts();
setInterval(load,30000);
setInterval(loadWeather,300000);
setInterval(loadCharts,120000);
</script>
</body></html>"""

# ── API endpoints ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # Inject HA ingress base path so JS fetch() calls reach this add-on, not HA core
    base = request.headers.get("X-Ingress-Path", "").rstrip("/")
    return PANEL.replace("__BASE__", base)

@app.route("/api/status")
def api_status():
    sensors = read_sensors()
    tariff  = current_tariff()
    model   = {}
    if MODEL_FILE.exists():
        try:
            art   = joblib.load(MODEL_FILE)
            model = {k: art.get(k) for k in ("trained_at", "r2_cv_mean", "n_samples")}
        except Exception:
            pass
    optimal = calculate_optimal_soc(sensors)
    storm   = is_storm_forecast()
    return jsonify({"sensors": sensors, "tariff": tariff, "model": model,
                    "optimal": optimal, "storm": storm})

@app.route("/api/decisions")
def api_decisions():
    limit = int(request.args.get("limit", 50))
    if DECISIONS_FILE.exists():
        return jsonify(json.loads(DECISIONS_FILE.read_text())[-limit:])
    return jsonify([])

@app.route("/api/savings")
def api_savings():
    return jsonify(_load_savings())

@app.route("/api/tariff", methods=["GET"])
def api_tariff_get():
    return jsonify(load_tariff())

@app.route("/api/tariff", methods=["POST"])
def api_tariff_post():
    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "error": "No data"}), 400
    save_tariff(data)
    return jsonify({"ok": True})

@app.route("/api/tariff/reset", methods=["POST"])
def api_tariff_reset():
    save_tariff(dict(DEFAULT_TARIFF))
    log.info("Tariff reset to built-in defaults")
    return jsonify({"ok": True})

@app.route("/api/setup", methods=["GET"])
def api_setup_get():
    return jsonify(dict(_SETUP))

@app.route("/api/setup", methods=["POST"])
def api_setup_post():
    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "error": "No data"}), 400
    clamp = lambda v, lo, hi: max(lo, min(hi, v))
    allowed = {
        "notify_email_enabled":           lambda v: bool(v),
        "notify_telegram_daily_enabled":  lambda v: bool(v),
        "notify_telegram_alerts_enabled": lambda v: bool(v),
        "notify_daily_time":              lambda v: str(v)[:5],
        "battery_emergency_threshold":    lambda v: clamp(int(v), 1, 30),
        "battery_low_threshold":          lambda v: clamp(int(v), 10, 50),
        "battery_medium_threshold":       lambda v: clamp(int(v), 30, 80),
        "battery_storm_threshold":        lambda v: clamp(int(v), 50, 100),
        "decision_interval_minutes":      lambda v: clamp(int(v), 5, 60),
        "battery_capacity_kwh":           lambda v: clamp(float(v), 2.0, 30.0),
    }
    validated = {}
    for key, coerce in allowed.items():
        if key in data:
            try:
                validated[key] = coerce(data[key])
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": f"Invalid value for {key}"}), 400
    save_setup(validated)
    return jsonify({"ok": True})

@app.route("/api/chart-data")
def api_chart_data():
    from datetime import timezone as _tz
    soc_entity = cfg("sensor_battery_soc", "sensor.battery_state_of_capacity")
    influx_u   = cfg("influxdb_url", "").strip()

    # ── SOC actual (past 24h) — prefer InfluxDB, fallback HA recorder ─────────
    if influx_u:
        rows, _ = ha_history_influx(soc_entity, days=1)
    if not influx_u or not rows:
        rows = ha_history(soc_entity, days=1)

    soc_labels, soc_actual = [], []
    for r in rows:
        try:
            ts  = datetime.fromisoformat(r["last_changed"].replace("Z", "+00:00"))
            val = float(r["state"])
            soc_labels.append(ts.strftime("%H:%M"))
            soc_actual.append(round(val, 1))
        except (KeyError, ValueError, TypeError):
            continue

    # ── SOC predicted (past) — from decisions.json ────────────────────────────
    soc_predicted = [None] * len(soc_labels)
    if DECISIONS_FILE.exists():
        try:
            cutoff  = (datetime.now() - timedelta(days=1)).isoformat()
            history = json.loads(DECISIONS_FILE.read_text())
            pred_by_time = {}
            for dec in history:
                if dec.get("timestamp", "") >= cutoff:
                    ts_str = datetime.fromisoformat(dec["timestamp"]).strftime("%H:%M")
                    pred   = dec.get("prediction", {}).get("predicted_soc")
                    if pred is not None:
                        pred_by_time[ts_str] = pred
            soc_predicted = [pred_by_time.get(lbl) for lbl in soc_labels]
        except Exception:
            pass

    # ── SOC forecast (future 8h) — chained ML predictions ────────────────────
    future_labels, future_predicted = [], []
    sensors = read_sensors()
    if MODEL_FILE.exists():
        try:
            art   = joblib.load(MODEL_FILE)
            now   = datetime.now()
            soc_f = sensors["battery_soc"]
            for h in range(1, 9):
                ft    = now + timedelta(hours=h)
                is_day_f = 7 <= ft.hour <= 20   # simple heuristic for future hours
                X = pd.DataFrame([{
                    "hour":        ft.hour,
                    "weekday":     ft.weekday(),
                    "month":       ft.month,
                    "lag1":        soc_f,
                    "lag4":        soc_f,
                    "roll4":       soc_f,
                    "solar_proxy": 1 if is_day_f else 0,
                }])
                pred  = float(art["pipeline"].predict(X[FEATURES])[0])
                soc_f = round(max(0.0, min(100.0, pred)), 1)
                future_labels.append(ft.strftime("%H:%M"))
                future_predicted.append(soc_f)
        except Exception as e:
            log.warning(f"Future SOC forecast: {e}")

    # ── Solar yesterday — InfluxDB MAX, fallback decisions.json ──────────────
    solar_today     = sensors.get("solar_today", 0)
    solar_tomorrow  = sensors.get("solar_tomorrow", 0)
    solar_yesterday = 0.0
    solar_entity_id = cfg("sensor_solar_today", "sensor.energy_production_today").split(".")[-1]

    if influx_u:
        now_local   = datetime.now()
        yest_date   = (now_local - timedelta(days=1)).date()
        yest_start  = datetime(yest_date.year, yest_date.month, yest_date.day,
                               tzinfo=_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        today_start = datetime(now_local.year, now_local.month, now_local.day,
                               tzinfo=_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        q_sol = (f'SELECT MAX("value") FROM /.*/ WHERE "entity_id" = \'{solar_entity_id}\' '
                 f'AND time >= \'{yest_start}\' AND time < \'{today_start}\'')
        r_sol, _, _ = _influx_query(
            influx_u, cfg("influxdb_db", "homeassistant").strip(), q_sol,
            cfg("influxdb_user", "").strip(), cfg("influxdb_password", ""))
        if r_sol:
            try:
                s = r_sol.json()["results"][0].get("series", [])
                if s and s[0].get("values"):
                    v = s[0]["values"][0][1]
                    if v is not None:
                        solar_yesterday = float(v)
            except Exception:
                pass

    if not solar_yesterday and DECISIONS_FILE.exists():
        try:
            history   = json.loads(DECISIONS_FILE.read_text())
            yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
            prev_vals = [d["sensors"].get("solar_today", 0)
                         for d in history if d["timestamp"][:10] == yesterday]
            if prev_vals:
                solar_yesterday = max(prev_vals)
        except Exception:
            pass

    # ── Daily savings ─────────────────────────────────────────────────────────
    savings_labels, savings_values = [], []
    if DECISIONS_FILE.exists():
        try:
            history    = json.loads(DECISIONS_FILE.read_text())
            daily: dict = {}
            interval_h = cfg("decision_interval_minutes", 15) / 60
            prices     = load_tariff().get("prices", DEFAULT_TARIFF["prices"])
            for dec in history:
                day    = dec.get("timestamp", "")[:10]
                tariff = dec.get("tariff", {})
                sens   = dec.get("sensors", {})
                if (tariff.get("period") == "peak"
                        and sens.get("battery_power", 0) < 0
                        and sens.get("grid_power", 500) <= 500):
                    kwh = min(abs(sens["battery_power"]) * interval_h / 1000, 2.0)
                    eur = kwh * (prices.get("peak", 0.30) - prices.get("export", 0.06))
                    daily[day] = round(daily.get(day, 0) + eur, 3)
            for i in range(6, -1, -1):
                day = (datetime.now() - timedelta(days=i)).date().isoformat()
                savings_labels.append(day[5:])
                savings_values.append(round(daily.get(day, 0), 2))
        except Exception:
            pass

    return jsonify({
        "soc": {
            "labels":           soc_labels,
            "actual":           soc_actual,
            "predicted":        soc_predicted,
            "future_labels":    future_labels,
            "future_predicted": future_predicted,
        },
        "solar":         {"yesterday": round(solar_yesterday, 1),
                          "today": round(solar_today, 1), "tomorrow": round(solar_tomorrow, 1)},
        "savings_daily": {"labels": savings_labels, "values": savings_values},
    })

@app.route("/api/run", methods=["POST"])
def api_run():
    return jsonify(run_cycle())

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    ok = train_model()
    return jsonify({"ok": ok,
                    "message": "Model retrained successfully ✓" if ok
                               else "Insufficient history — rules-only mode"})

@app.route("/api/battery/charge", methods=["POST"])
def api_battery_charge():
    data       = request.get_json() or {}
    target_soc = max(10, min(100, int(data.get("target_soc", 80))))
    ok         = set_battery_charge_target(target_soc)
    return jsonify({"ok": ok, "target_soc": target_soc})

@app.route("/api/battery/self-consumption", methods=["POST"])
def api_battery_self_consumption():
    return jsonify({"ok": set_battery_self_consumption()})

@app.route("/api/send-summary", methods=["POST"])
def api_send_summary():
    result = send_daily_summary()
    return jsonify({"ok": True, **result})

@app.route("/api/options")
def api_options():
    return jsonify({k: v for k, v in OPT.items() if "token" not in k.lower()})

@app.route("/api/influx-debug")
def api_influx_debug():
    entity    = cfg("sensor_battery_soc", "sensor.battery_state_of_capacity")
    influx_u  = cfg("influxdb_url", "").strip()   # strip accidental spaces
    influx_db = cfg("influxdb_db", "homeassistant").strip()
    user      = cfg("influxdb_user", "").strip()
    password  = cfg("influxdb_password", "")

    ping_ok, ping_err, auth_mode = False, None, "unknown"
    databases, measurements, sample_entities = [], [], []

    if influx_u:
        # 1. Ping — SHOW DATABASES
        r_ping, ping_err, auth_mode = _influx_query(
            influx_u, influx_db, "SHOW DATABASES", user, password)
        ping_ok = r_ping is not None
        if r_ping:
            try:
                s = r_ping.json()["results"][0]["series"][0]["values"]
                databases = [v[0] for v in s if not v[0].startswith("_")]
            except Exception:
                pass

        # 2. Show measurements in target db
        if ping_ok:
            r_m, _, _ = _influx_query(
                influx_u, influx_db, "SHOW MEASUREMENTS LIMIT 20", user, password)
            if r_m:
                try:
                    s = r_m.json()["results"][0]["series"][0]["values"]
                    measurements = [v[0] for v in s]
                except Exception:
                    pass

        # 3. Find entity_id tags in the % measurement (where SOC lives)
        # and do a direct probe for the configured entity
        if ping_ok:
            # All entity_ids in the % measurement (SOC unit)
            r_e, _, _ = _influx_query(
                influx_u, influx_db,
                'SHOW TAG VALUES FROM "%" WITH KEY = "entity_id"',
                user, password)
            if r_e:
                try:
                    for series in r_e.json()["results"][0].get("series", []):
                        for v in series.get("values", []):
                            tag_val = v[1] if len(v) > 1 else v[0]
                            if "battery" in tag_val.lower() or "soc" in tag_val.lower() or "capacit" in tag_val.lower():
                                sample_entities.append(tag_val)
                except Exception:
                    pass

            # Direct probe: try the entity_id with and without domain prefix
            entity_short = entity.split(".")[-1] if "." in entity else entity
            for probe_id in [entity_short, entity]:
                r_probe, _, _ = _influx_query(
                    influx_u, influx_db,
                    f'SELECT "value" FROM "%" WHERE "entity_id" = \'{probe_id}\' ORDER BY time DESC LIMIT 1',
                    user, password)
                if r_probe:
                    try:
                        s = r_probe.json()["results"][0].get("series", [])
                        if s:
                            sample_entities.insert(0, f"✓ FOUND as '{probe_id}'")
                            break
                    except Exception:
                        pass

    rows_i, err_i = (ha_history_influx(entity, days=7) if influx_u
                     else ([], "not_configured"))
    rows_r = ha_history(entity, days=1)

    return jsonify({
        "configured":       bool(influx_u),
        "url":              influx_u,
        "db":               influx_db,
        "ping_ok":          ping_ok,
        "auth_mode":        auth_mode,
        "databases":        databases,
        "measurements":     measurements,
        "sample_entities":  sample_entities,
        "records_7d":       len(rows_i),
        "first_ts":         rows_i[0]["last_changed"][:16]  if rows_i else None,
        "last_ts":          rows_i[-1]["last_changed"][:16] if rows_i else None,
        "ok":               len(rows_i) > 0,
        "error":            err_i or ping_err,
        "recorder_7d":      len(rows_r),
    })

@app.route("/api/weather")
def api_weather():
    entity = cfg("sensor_weather", "weather.aemet")
    s = ha_state(entity)
    if not s:
        return jsonify({"ok": False, "error": "weather entity unavailable"})
    attrs    = s.get("attributes", {})
    forecast = attrs.get("forecast", [])[:5]
    return jsonify({
        "ok":         True,
        "condition":  s.get("state", ""),
        "temperature": attrs.get("temperature"),
        "humidity":   attrs.get("humidity"),
        "wind_speed": attrs.get("wind_speed"),
        "forecast": [
            {
                "datetime":     f.get("datetime", ""),
                "condition":    f.get("condition", ""),
                "temperature":  f.get("temperature"),
                "templow":      f.get("templow"),
                "precipitation": f.get("precipitation"),
            }
            for f in forecast
        ],
        "storm": is_storm_forecast(),
    })

# ── Startup ──────────────────────────────────────────────────────────────────
def main():
    global _scheduler_ref
    _load_setup_cache()

    log.info("═══════════════════════════════════════")
    log.info("   Energy Optimizer v2.5.7 — HAOS")
    log.info("═══════════════════════════════════════")
    log.info(f"  Supervisor token:        {'OK' if HA_TOKEN else 'NOT FOUND'}")
    log.info(f"  Email enabled:           {cfg('notify_email_enabled', True)}")
    log.info(f"  Telegram daily:          {cfg('notify_telegram_daily_enabled', True)}")
    log.info(f"  Telegram instant alerts: {cfg('notify_telegram_alerts_enabled', True)}")

    if not MODEL_FILE.exists():
        log.info("No saved model — starting initial training...")
        threading.Thread(target=train_model, daemon=True).start()

    threading.Thread(target=_refresh_consumption_cache, daemon=True).start()

    scheduler = BackgroundScheduler(timezone="Europe/Madrid")
    _scheduler_ref = scheduler

    interval = cfg("decision_interval_minutes", 15)
    scheduler.add_job(run_cycle, "interval", minutes=interval, id="cycle",
                      next_run_time=datetime.now() + timedelta(seconds=15))

    cron = cfg("retrain_cron", "0 3 * * *").split()
    if len(cron) == 5:
        scheduler.add_job(train_model, "cron",
                          minute=cron[0], hour=cron[1], day=cron[2],
                          month=cron[3], day_of_week=cron[4], id="retrain")

    summary_time = cfg("notify_daily_time", "23:00").split(":")
    if len(summary_time) == 2:
        scheduler.add_job(send_daily_summary, "cron",
                          hour=int(summary_time[0]), minute=int(summary_time[1]),
                          id="daily_summary")
        log.info(f"  Daily summary: {cfg('notify_daily_time','23:00')} → "
                 f"email={cfg('notify_email_service','—')} "
                 f"telegram={cfg('notify_telegram_service','—')}")

    scheduler.start()
    log.info(f"  Cycle every {interval} min · Retrain: {cfg('retrain_cron','0 3 * * *')}")
    app.run(host="0.0.0.0", port=8765, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
