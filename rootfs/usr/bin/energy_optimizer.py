#!/usr/bin/env python3
"""
Energy Optimizer — Home Assistant Add-on v2.1
Smart energy management: battery, heat pump, pool pump, pool cleaner, dishwasher
Logic: adaptive tariff rules + scikit-learn ML model + consumption history
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
DATA_DIR.mkdir(exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("energy-optimizer")

# ── Options ──────────────────────────────────────────────────────────────────
def load_options() -> dict:
    path = Path("/data/options.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    log.warning("options.json not found, using defaults")
    return {}

OPT = load_options()

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
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    result = ha_get(
        f"/api/history/period/{start}",
        params={"filter_entity_id": entity_id, "minimal_response": "true"},
    )
    if result and isinstance(result, list) and len(result) > 0:
        return result[0]
    return []

# ── Tariff management ────────────────────────────────────────────────────────
DEFAULT_TARIFF = {
    "prices":      {"peak": 0.30, "mid": 0.18, "valley": 0.08, "export": 0.06},
    "peak_hours":  [10, 11, 12, 13, 18, 19, 20, 21],
    "valley_hours":[0, 1, 2, 3, 4, 5, 6, 7],
}

def load_tariff() -> dict:
    if TARIFF_FILE.exists():
        try:
            return json.loads(TARIFF_FILE.read_text())
        except Exception:
            pass
    return dict(DEFAULT_TARIFF)

def save_tariff(cfg: dict):
    TARIFF_FILE.write_text(json.dumps(cfg, indent=2))
    log.info("Tariff configuration saved")

def current_tariff() -> dict:
    cfg     = load_tariff()
    prices  = cfg.get("prices",       DEFAULT_TARIFF["prices"])
    peak_h  = cfg.get("peak_hours",   DEFAULT_TARIFF["peak_hours"])
    valley_h= cfg.get("valley_hours", DEFAULT_TARIFF["valley_hours"])
    now     = datetime.now()
    hour    = now.hour
    weekend = now.weekday() >= 5

    if weekend or hour in valley_h:
        period = "valley"
    elif hour in peak_h:
        period = "peak"
    else:
        period = "mid"

    return {
        "period":     period,
        "price_kwh":  prices[period],
        "export_kwh": prices.get("export", 0.06),
        "prices":     prices,
        "hour":       hour,
        "weekend":    weekend,
    }

# ── Sun status (uses HA sun.sun — respects actual house location) ─────────────
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
    o = OPT
    return {
        "battery_soc":        ha_float(o.get("sensor_battery_soc",       "sensor.battery_state_of_capacity")),
        "battery_power":      ha_float(o.get("sensor_battery_power",      "sensor.battery_charge_discharge_power")),
        "grid_power":         ha_float(o.get("sensor_grid_power",         "sensor.acometida_general_power")),
        "solar_current_hour": ha_float(o.get("sensor_solar_current_hour", "sensor.energy_current_hour")),
        "solar_next_hour":    ha_float(o.get("sensor_solar_next_hour",    "sensor.energy_next_hour")),
        "solar_today":        ha_float(o.get("sensor_solar_today",        "sensor.energy_production_today")),
        "solar_tomorrow":     ha_float(o.get("sensor_solar_tomorrow",     "sensor.energy_production_tomorrow")),
        "temp_outdoor":       ha_float(o.get("sensor_temp_outdoor",       "sensor.ebusd_broadcast_outsidetemp_temp2")),
        "temp_indoor":        ha_float(o.get("sensor_temp_salon",         "sensor.media_salon")),
        "pool_hours_day":     ha_float(o.get("sensor_pool_hours_day",     "sensor.depuradora_encendida_24h")),
        "pool_hours_week":    ha_float(o.get("sensor_pool_hours_week",    "sensor.depuradora_encendida_semana")),
        "dishwasher_state":   ha_str(o.get("sensor_dishwasher_state",     "sensor.lavavajillas_operation_state")),
        "ts": datetime.now().isoformat(),
    }

# ── Battery direct control ───────────────────────────────────────────────────
def set_battery_charge_target(target_soc: int, charge_power_w: int = 3000) -> bool:
    o = OPT
    ok = all([
        ha_set_number(o.get("number_battery_charge_cutoff", "number.battery_grid_charge_cutoff_soc"), target_soc),
        ha_set_number(o.get("number_battery_charge_power",  "a186f9599e9cad7127bca381f7a8bfb2"), charge_power_w),
        ha_set_number(o.get("number_battery_backup_soc",    "de4a2bf18222f354228cdb112b65e882"), target_soc),
        ha_switch(o.get("switch_battery_force_charge",  "02db00e10018b01211507db92819a25a"), True),
        ha_set_select(o.get("select_battery_mode",      "select.battery_working_mode"), "time_of_use_luna2000"),
    ])
    log.info(f"  Battery → charge to {target_soc}% @ {charge_power_w}W — {'OK' if ok else 'ERROR'}")
    return ok

def set_battery_self_consumption(min_soc: int = 20) -> bool:
    o = OPT
    ok = all([
        ha_set_number(o.get("number_battery_charge_cutoff", "number.battery_grid_charge_cutoff_soc"), min_soc),
        ha_set_select(o.get("select_battery_mode", "select.battery_working_mode"), "maximise_self_consumption"),
    ])
    log.info(f"  Battery → self-consumption (min {min_soc}%) — {'OK' if ok else 'ERROR'}")
    return ok

# ── Consumption estimation & optimal SOC ─────────────────────────────────────
_consumption_cache: dict = {"kw": 0.5, "updated": None}

def _refresh_consumption_cache():
    global _consumption_cache
    entity = OPT.get("sensor_grid_power", "sensor.acometida_general_power")
    rows   = ha_history(entity, days=14)
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
    """
    Estimate the optimal valley charge target by balancing:
    - Expected overnight consumption (from 14-day rolling history)
    - Solar forecast for tomorrow
    - Battery capacity
    - 10% safety buffer
    """
    now  = datetime.now()
    hour = now.hour
    hours_until_solar = (8 - hour) if hour < 8 else (24 - hour + 8)

    avg_kw       = _get_avg_night_consumption_kw()
    needed_kwh   = avg_kw * hours_until_solar
    battery_kwh  = float(OPT.get("battery_capacity_kwh", 10.0))
    soc_coverage = (needed_kwh / battery_kwh) * 100

    solar_tm = sensors.get("solar_tomorrow", 0)
    # Solar adjustment: less solar tomorrow → charge more tonight
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
    entity = OPT.get("sensor_weather", "weather.aemet")
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
def _history_to_df(rows: list) -> pd.DataFrame | None:
    records = []
    for row in rows:
        try:
            ts  = datetime.fromisoformat(row["last_changed"].replace("Z", "+00:00"))
            val = float(row["state"])
            records.append({"ts": ts, "value": val})
        except (KeyError, ValueError, TypeError):
            continue
    if len(records) < 100:
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
    rows = ha_history(OPT.get("sensor_battery_soc", "sensor.battery_state_of_capacity"), days=60)
    df   = _history_to_df(rows)
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
    import numpy as np
    joblib.dump({"pipeline": pipe, "features": FEATURES,
                 "trained_at": datetime.now().isoformat(),
                 "r2_cv_mean": float(np.mean(scores)), "n_samples": len(df)}, MODEL_FILE)
    log.info(f"Model trained: {len(df)} samples, R²={np.mean(scores):.3f}")
    return True

def predict_soc(sensors: dict) -> dict:
    if MODEL_FILE.exists():
        try:
            art  = joblib.load(MODEL_FILE)
            now  = datetime.now()
            X = pd.DataFrame([{"hour": now.hour, "weekday": now.weekday(), "month": now.month,
                                "lag1": sensors["battery_soc"], "lag4": sensors["battery_soc"],
                                "roll4": sensors["battery_soc"],
                                "solar_proxy": 1 if 8 <= now.hour <= 19 else 0}])
            pred = float(art["pipeline"].predict(X[FEATURES])[0])
            return {"predicted_soc": round(max(0.0, min(100.0, pred)), 1),
                    "method": "ml", "r2": art.get("r2_cv_mean"), "trained_at": art.get("trained_at")}
        except Exception as e:
            log.warning(f"ML prediction error: {e}")
    solar_tm = sensors.get("solar_tomorrow", 0)
    pred = 80.0 if solar_tm < OPT.get("solar_tomorrow_irrisoria_kwh", 2.0) else 50.0
    return {"predicted_soc": pred, "method": "rules_fallback"}

# ── Heat pump logic ──────────────────────────────────────────────────────────
def _is_summer() -> bool:
    m = datetime.now().month
    return OPT.get("summer_start_month", 6) <= m <= OPT.get("summer_end_month", 9)

def decide_heat_pump(sensors: dict) -> dict:
    soc        = sensors["battery_soc"]
    batt_power = sensors["battery_power"]
    temp_in    = sensors["temp_indoor"]
    sun        = get_sun_status()
    daytime    = sun["is_day"]
    free_power = (soc >= 99 and batt_power > 0)

    if _is_summer():
        entity = OPT.get("number_hvac_cool", "number.ebusd_ctls2_z1coolingtemp_tempv")
        if free_power:
            target, reason = 16.0, "Free solar power (SOC≥99%, charging from panels)"
        elif temp_in > 26 and daytime:
            target, reason = 20.0, f"Indoor too hot ({temp_in:.1f}°C > 26°C)"
        else:
            target, reason = 25.0, "Summer base setpoint"
    else:
        entity = OPT.get("number_hvac_heat", "number.ebusd_ctls2_z1manualtemp_tempv")
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
    thr_emerg = OPT.get("battery_emergency_threshold", 10)
    storm_thr = OPT.get("battery_storm_threshold", 80)

    # Storm protection (highest priority outside peak emergency)
    storm = is_storm_forecast()
    if storm and soc < storm_thr and period != "peak":
        return {"action": "charge", "target_soc": storm_thr, "power_w": 2000,
                "reason": f"⚡ Storm forecast — maintaining {storm_thr}% reserve",
                "storm": True}

    # Peak: no grid charging except emergency
    if period == "peak":
        if soc < thr_emerg:
            return {"action": "charge", "target_soc": 30, "power_w": 1500,
                    "reason": f"EMERGENCY during peak: SOC={soc:.0f}% < {thr_emerg}%"}
        return {"action": "none",
                "reason": f"Peak ({prices['peak']}€/kWh): no grid charging"}

    # Valley: smart optimal target
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

    # Mid: emergency only
    if period == "mid":
        if soc < thr_emerg:
            return {"action": "charge", "target_soc": 30, "power_w": 1500,
                    "reason": f"Mid + critical SOC ({soc:.0f}%) → emergency charge"}
        if soc < 20:
            return {"action": "charge", "target_soc": 40, "power_w": 1500,
                    "reason": f"Mid + very low SOC ({soc:.0f}%) → charge to 40%"}
        return {"action": "self_consumption",
                "reason": f"Mid period, self-consumption (SOC={soc:.0f}%)"}

    return {"action": "none", "reason": "No battery action"}

# ── Pool pump + cleaner logic ────────────────────────────────────────────────
_scheduler_ref: BackgroundScheduler | None = None  # set in main()

def decide_pool(sensors: dict, tariff: dict) -> dict:
    month   = datetime.now().month
    summer  = OPT.get("summer_start_month", 6) <= month <= OPT.get("summer_end_month", 9)
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
    """Turn on pool pump and schedule cleaner shutoff after 15 min."""
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
        return {"action": True,
                "reason": f"Valley tariff ({prices['valley']}€/kWh)",
                "state": state}
    if period == "peak":
        return {"action": False,
                "reason": f"Peak tariff ({prices['peak']}€/kWh) — waiting",
                "state": state}
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
    interval_h = OPT.get("decision_interval_minutes", 15) / 60
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
    pool_sw  = OPT.get("switch_pool", "switch.depuradora")
    clean_sw = OPT.get("switch_pool_cleaner", "switch.limpiafondos")
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
        dw_sw = OPT.get("switch_dishwasher", "")
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

# ── Flask web panel ──────────────────────────────────────────────────────────
app = Flask(__name__)

PANEL = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Energy Optimizer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0f172a;--s:#1e293b;--b:#334155;--a:#38bdf8;--g:#4ade80;--y:#fbbf24;--r:#f87171;--o:#fb923c;--t:#e2e8f0;--m:#94a3b8}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t);font-family:system-ui,sans-serif;padding:1rem;font-size:14px}
h1{color:var(--a);font-size:1.3rem;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem}
h2{font-size:.7rem;color:var(--m);text-transform:uppercase;letter-spacing:.08em;margin:.8rem 0 .5rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.6rem;margin-bottom:1rem}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-bottom:1rem}
@media(max-width:600px){.grid2{grid-template-columns:1fr}}
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
.btn-y{background:var(--y)}.btn-g{background:var(--g)}.btn-r{background:var(--r)}.btn-sm{padding:.3rem .7rem;font-size:.72rem}
.actions{display:flex;gap:.5rem;margin-bottom:1rem;flex-wrap:wrap}
/* Battery control card */
.batt-card{background:var(--s);border-radius:.75rem;padding:.9rem;border:1px solid var(--b);margin-bottom:1rem}
.batt-header{display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;margin-bottom:.8rem}
.batt-soc{font-size:3rem;font-weight:700;color:var(--a);line-height:1}
.batt-meta{display:flex;flex-direction:column;gap:.3rem}
.batt-target{font-size:.78rem;color:var(--m)}
.batt-target span{color:var(--g);font-weight:700}
.charge-btns{display:flex;gap:.4rem;flex-wrap:wrap}
.storm-badge{background:rgba(248,113,113,.15);border:1px solid var(--r);color:var(--r);padding:.3rem .7rem;border-radius:.5rem;font-size:.75rem;font-weight:700;display:none}
/* Savings */
.savings{background:linear-gradient(135deg,rgba(74,222,128,.07),rgba(56,189,248,.07));border:1px solid rgba(74,222,128,.25);border-radius:.75rem;padding:.9rem;display:flex;gap:2rem;align-items:center;flex-wrap:wrap;margin-bottom:1rem}
.savings-num{font-size:2rem;font-weight:700;color:var(--g)}
/* Tariff editor */
.tariff-section{margin-bottom:1rem}
.tariff-section summary{cursor:pointer;list-style:none;font-size:.7rem;color:var(--m);text-transform:uppercase;letter-spacing:.08em;padding:.1rem 0}
.tariff-section summary::-webkit-details-marker{display:none}
.tariff-section[open] summary{color:var(--a);margin-bottom:.8rem}
.timeline{display:grid;grid-template-columns:repeat(24,1fr);gap:2px;margin:.6rem 0 .9rem}
.t-hour{text-align:center;font-size:.58rem;padding:.3rem .1rem;border-radius:.2rem;cursor:pointer;transition:.1s}
.t-hour:hover{filter:brightness(1.3)}
.t-hour.peak{background:rgba(248,113,113,.35);color:var(--r)}
.t-hour.valley{background:rgba(74,222,128,.35);color:var(--g)}
.t-hour.mid{background:rgba(251,191,36,.35);color:var(--y)}
.price-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.5rem;margin-bottom:.7rem}
.price-field label{font-size:.68rem;color:var(--m);display:block;margin-bottom:.2rem}
.price-field input{width:100%;background:var(--b);border:1px solid #475569;border-radius:.35rem;padding:.35rem .5rem;color:var(--t);font-size:.85rem}
.tariff-note{font-size:.68rem;color:var(--m);margin-bottom:.6rem}
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
.ok{color:var(--g)}.skip{color:var(--m)}.err{color:var(--r)}
.tag{display:inline-block;padding:.1rem .4rem;border-radius:.25rem;font-size:.68rem;background:var(--b);color:var(--m)}
.tag.bat{background:rgba(56,189,248,.12);color:var(--a)}
.tag.hp{background:rgba(74,222,128,.12);color:var(--g)}
.tag.pool{background:rgba(251,191,36,.12);color:var(--y)}
.tag.dw{background:rgba(251,146,60,.12);color:var(--o)}
</style>
</head>
<body>
<h1>⚡ Energy Optimizer <span id="ver" style="font-size:.75rem;color:var(--m);font-weight:400">v2.1</span></h1>
<div id="notify" class="toast"></div>

<!-- KPI grid -->
<div id="kpis" class="grid"></div>

<!-- Savings -->
<div id="savings-box"></div>

<!-- Battery control -->
<div class="batt-card">
  <h2 style="margin-top:0">⚡ Battery</h2>
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

<!-- Action buttons -->
<div class="actions">
  <button id="btn-run" class="btn" onclick="runCycle()">▶ Run cycle now</button>
  <button id="btn-retrain" class="btn btn-y" onclick="retrain()">🔁 Retrain model</button>
</div>

<!-- Charts -->
<div class="grid2">
  <div class="card"><h2 style="margin-top:0">Battery SOC — last 24h</h2><canvas id="socChart"></canvas></div>
  <div class="card"><h2 style="margin-top:0">Solar production (kWh)</h2><canvas id="solarChart"></canvas></div>
</div>

<!-- Tariff editor -->
<details class="card tariff-section">
  <summary>⚡ Tariff configuration (click to expand)</summary>
  <div class="tariff-note">Click any hour to cycle: <span class="badge valley">valley</span> → <span class="badge mid">mid</span> → <span class="badge peak">peak</span>. Weekends are always valley.</div>
  <div class="timeline" id="tariff-timeline"></div>
  <div class="price-grid">
    <div class="price-field"><label>Peak (€/kWh)</label><input id="p-peak" type="number" step="0.01"></div>
    <div class="price-field"><label>Mid (€/kWh)</label><input id="p-mid" type="number" step="0.01"></div>
    <div class="price-field"><label>Valley (€/kWh)</label><input id="p-valley" type="number" step="0.01"></div>
    <div class="price-field"><label>Export (€/kWh)</label><input id="p-export" type="number" step="0.01"></div>
  </div>
  <button class="btn btn-g btn-sm" onclick="saveTariff()">💾 Save tariff</button>
</details>

<!-- Decisions log -->
<h2>Recent decisions</h2>
<div class="card" style="overflow-x:auto">
  <table><thead><tr><th>Time</th><th>Type</th><th>Reason</th><th>OK</th></tr></thead>
  <tbody id="log"></tbody></table>
</div>

<script>
// ── Toast ────────────────────────────────────────────────────────────────────
let _toastTimer = null;
function notify(msg, type='ok', ms=3500) {
  const el = document.getElementById('notify');
  el.textContent = msg;
  el.className = `toast ${type} show`;
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), ms);
}

// ── Tariff editor ────────────────────────────────────────────────────────────
let tariffCfg = null;

async function loadTariff() {
  tariffCfg = await fetch('/api/tariff').then(r => r.json());
  renderTariffTimeline();
  document.getElementById('p-peak').value   = tariffCfg.prices.peak;
  document.getElementById('p-mid').value    = tariffCfg.prices.mid;
  document.getElementById('p-valley').value = tariffCfg.prices.valley;
  document.getElementById('p-export').value = tariffCfg.prices.export;
}

function renderTariffTimeline() {
  if (!tariffCfg) return;
  const {peak_hours, valley_hours} = tariffCfg;
  document.getElementById('tariff-timeline').innerHTML =
    Array.from({length:24}, (_, h) => {
      const p = peak_hours.includes(h) ? 'peak' : valley_hours.includes(h) ? 'valley' : 'mid';
      return `<div class="t-hour ${p}" onclick="toggleHour(${h})" title="${h}:00">${h}</div>`;
    }).join('');
}

function toggleHour(h) {
  const {peak_hours, valley_hours} = tariffCfg;
  const isPeak   = peak_hours.includes(h);
  const isValley = valley_hours.includes(h);
  if (isValley) {
    tariffCfg.valley_hours = valley_hours.filter(x => x !== h);
  } else if (!isPeak) {
    tariffCfg.peak_hours = [...peak_hours, h].sort((a,b) => a-b);
  } else {
    tariffCfg.peak_hours   = peak_hours.filter(x => x !== h);
    tariffCfg.valley_hours = [...valley_hours, h].sort((a,b) => a-b);
  }
  renderTariffTimeline();
}

async function saveTariff() {
  tariffCfg.prices = {
    peak:   parseFloat(document.getElementById('p-peak').value)   || 0.30,
    mid:    parseFloat(document.getElementById('p-mid').value)    || 0.18,
    valley: parseFloat(document.getElementById('p-valley').value) || 0.08,
    export: parseFloat(document.getElementById('p-export').value) || 0.06,
  };
  const r = await fetch('/api/tariff', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(tariffCfg)
  }).then(r => r.json());
  notify(r.ok ? '✓ Tariff saved' : '✗ Error saving tariff', r.ok ? 'ok' : 'err');
}

// ── Charts ───────────────────────────────────────────────────────────────────
let socInst=null, solInst=null;
function mkChart(id,type,labels,datasets,yExtra={}) {
  return new Chart(document.getElementById(id).getContext('2d'), {
    type, data:{labels, datasets},
    options:{responsive:true, maintainAspectRatio:true,
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:11}}}},
      scales:{x:{ticks:{color:'#94a3b8',maxTicksLimit:8,font:{size:10}},grid:{color:'#334155'}},
              y:{ticks:{color:'#94a3b8',font:{size:10}},grid:{color:'#334155'},...yExtra}}}
  });
}

async function loadCharts() {
  try {
    const cd = await fetch('/api/chart-data').then(r => r.json());
    if(socInst) socInst.destroy();
    socInst = mkChart('socChart','line', cd.soc.labels,
      [{label:'SOC %', data:cd.soc.values, borderColor:'#38bdf8',
        backgroundColor:'rgba(56,189,248,.1)', fill:true, tension:.3, pointRadius:0}],
      {min:0, max:100});
    if(solInst) solInst.destroy();
    solInst = mkChart('solarChart','bar',
      ['Yesterday','Today','Tomorrow'],
      [{label:'kWh',
        data:[cd.solar.yesterday, cd.solar.today, cd.solar.tomorrow],
        backgroundColor:['rgba(251,191,36,.5)','rgba(74,222,128,.7)','rgba(56,189,248,.5)'],
        borderColor:['#fbbf24','#4ade80','#38bdf8'], borderWidth:1, borderRadius:4}]);
  } catch(e) { console.warn('Chart load error', e); }
}

// ── Main status load ─────────────────────────────────────────────────────────
const PC = {peak:'peak', valley:'valley', mid:'mid'};
const tagMap = {battery:'bat', heat_pump:'hp', pool:'pool', dishwasher:'dw'};

async function load() {
  try {
    const [s, d, sv] = await Promise.all([
      fetch('/api/status').then(r => r.json()),
      fetch('/api/decisions?limit=30').then(r => r.json()),
      fetch('/api/savings').then(r => r.json()),
    ]);
    const {sensors:ss, tariff:t, model:m, optimal:opt, storm} = s;
    const dw    = ss.dishwasher_state || '--';
    const dwCls = dw.toLowerCase()==='running'?'running':dw.toLowerCase()==='ready'?'ready':'';

    document.getElementById('kpis').innerHTML = `
      <div class="card">
        <div class="metric">${(ss.solar_today??0).toFixed(1)}</div>
        <div class="label">Solar today (kWh)</div>
        <div class="sub">Next hr: ${(ss.solar_next_hour??0).toFixed(2)} kWh</div>
      </div>
      <div class="card">
        <div class="metric">${(ss.solar_tomorrow??0).toFixed(1)}</div>
        <div class="label">Solar tomorrow (kWh)</div>
      </div>
      <div class="card">
        <div class="metric ${PC[t.period]}">${(t.price_kwh??0).toFixed(2)}€</div>
        <div class="label">Price/kWh &nbsp;<span class="badge ${PC[t.period]}">${t.period}</span></div>
        <div class="sub">${t.weekend?'Weekend (valley all day)':''}</div>
      </div>
      <div class="card">
        <div class="metric">${(ss.temp_indoor??0).toFixed(1)}°C</div>
        <div class="label">Indoor temp</div>
        <div class="sub">Outdoor: ${(ss.temp_outdoor??0).toFixed(1)}°C</div>
      </div>
      <div class="card">
        <div class="metric" style="font-size:1.1rem"><span class="badge ${dwCls}">${dw}</span></div>
        <div class="label">Dishwasher</div>
      </div>
      <div class="card">
        <div class="metric" style="font-size:1rem">${m?.r2_cv_mean!=null?'R²='+m.r2_cv_mean.toFixed(2):'No model'}</div>
        <div class="label">ML model</div>
        <div class="sub">${m?.n_samples?m.n_samples+' samples':''}</div>
      </div>`;

    // Battery card
    const bpow  = ss.battery_power ?? 0;
    const bdir  = bpow > 50 ? '▲ Charging' : bpow < -50 ? '▼ Discharging' : '● Idle';
    const bsoc  = Number(ss.battery_soc ?? 0).toFixed(0);
    document.getElementById('batt-soc-big').textContent = bsoc + '%';
    document.getElementById('batt-dir').textContent = `${bdir}  ${Math.abs(bpow).toFixed(0)} W`;
    const stormEl = document.getElementById('storm-badge');
    if (storm) { stormEl.style.display='inline-block'; }
    else        { stormEl.style.display='none'; }
    if (opt) {
      document.getElementById('batt-target-line').innerHTML =
        `Smart target: <span>${opt.target_soc}%</span>` +
        ` — ${opt.needed_kwh} kWh overnight, ` +
        `${opt.solar_tomorrow} kWh solar tomorrow`;
    }

    // Savings
    document.getElementById('savings-box').innerHTML = `
      <div class="savings">
        <div><div class="savings-num">€${(sv.total_eur_saved??0).toFixed(2)}</div>
          <div class="label">Estimated savings</div></div>
        <div><div class="metric" style="color:var(--g)">${(sv.total_kwh_avoided_peak??0).toFixed(1)}</div>
          <div class="label">kWh covered at peak</div></div>
        <div style="font-size:.72rem;color:var(--m)">Since ${sv.since??'--'}</div>
      </div>`;

    // Decisions log
    const rows = d.slice().reverse().flatMap(dec => {
      const ts = new Date(dec.timestamp).toLocaleTimeString('en', {hour:'2-digit',minute:'2-digit'});
      return [
        ...(dec.actions||[]).map(a =>
          `<tr><td>${ts}</td><td><span class="tag ${tagMap[a.type]||''}">${a.type}</span></td>
           <td class="ok">${a.reason}</td><td class="${a.ok?'ok':'err'}">${a.ok?'✓':'✗'}</td></tr>`),
        ...(dec.skipped||[]).map(a =>
          `<tr><td>${ts}</td><td><span class="tag">${a.type}</span></td>
           <td class="skip">${a.reason}</td><td class="skip">–</td></tr>`)
      ];
    }).join('');
    document.getElementById('log').innerHTML = rows ||
      '<tr><td colspan="4" style="color:var(--m)">No decisions yet</td></tr>';
  } catch(e) { console.error('Load error', e); }
}

// ── Button actions ───────────────────────────────────────────────────────────
async function runCycle() {
  const btn = document.getElementById('btn-run');
  btn.disabled = true; btn.textContent = '⏳ Running…';
  try {
    const d = await fetch('/api/run', {method:'POST'}).then(r => r.json());
    notify(`✓ Cycle done — ${d.actions?.length??0} actions, ${d.skipped?.length??0} skipped`);
    load(); loadCharts();
  } catch(e) { notify('✗ Error running cycle', 'err'); }
  finally { btn.disabled=false; btn.textContent='▶ Run cycle now'; }
}

async function retrain() {
  const btn = document.getElementById('btn-retrain');
  btn.disabled = true; btn.textContent = '⏳ Training…';
  try {
    const d = await fetch('/api/retrain', {method:'POST'}).then(r => r.json());
    notify(d.message, d.ok ? 'ok' : 'info', 5000);
    load();
  } catch(e) { notify('✗ Error during training', 'err'); }
  finally { btn.disabled=false; btn.textContent='🔁 Retrain model'; }
}

async function chargeToTarget(soc) {
  if (!confirm(`Charge battery to ${soc}%?`)) return;
  notify('⏳ Sending command…', 'info', 2000);
  const r = await fetch('/api/battery/charge', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({target_soc: soc})
  }).then(r => r.json());
  notify(r.ok ? `✓ Charging to ${soc}% started` : '✗ Error setting charge', r.ok ? 'ok' : 'err');
  load();
}

async function selfConsumption() {
  if (!confirm('Switch to self-consumption mode?')) return;
  const r = await fetch('/api/battery/self-consumption', {method:'POST'}).then(r => r.json());
  notify(r.ok ? '✓ Self-consumption mode activated' : '✗ Error', r.ok ? 'ok' : 'err');
  load();
}

// ── Init ─────────────────────────────────────────────────────────────────────
load();
loadCharts();
loadTariff();
setInterval(load, 30000);
setInterval(loadCharts, 120000);
</script>
</body></html>"""

# ── API endpoints ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return PANEL

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

@app.route("/api/chart-data")
def api_chart_data():
    soc_entity = OPT.get("sensor_battery_soc", "sensor.battery_state_of_capacity")
    rows = ha_history(soc_entity, days=1)
    soc_labels, soc_values = [], []
    for r in rows:
        try:
            ts  = datetime.fromisoformat(r["last_changed"].replace("Z", "+00:00"))
            val = float(r["state"])
            soc_labels.append(ts.strftime("%H:%M"))
            soc_values.append(round(val, 1))
        except (KeyError, ValueError, TypeError):
            continue

    sensors        = read_sensors()
    solar_today    = sensors.get("solar_today", 0)
    solar_tomorrow = sensors.get("solar_tomorrow", 0)
    solar_yesterday = 0.0
    if DECISIONS_FILE.exists():
        try:
            history   = json.loads(DECISIONS_FILE.read_text())
            yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
            prev_vals = [d["sensors"].get("solar_today", 0)
                         for d in history if d["timestamp"][:10] == yesterday]
            if prev_vals:
                solar_yesterday = max(prev_vals)
        except Exception:
            pass

    return jsonify({
        "soc":   {"labels": soc_labels, "values": soc_values},
        "solar": {"yesterday": round(solar_yesterday, 1),
                  "today": round(solar_today, 1), "tomorrow": round(solar_tomorrow, 1)},
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

@app.route("/api/options")
def api_options():
    return jsonify({k: v for k, v in OPT.items() if "token" not in k.lower()})

# ── Startup ──────────────────────────────────────────────────────────────────
def main():
    global _scheduler_ref
    log.info("═══════════════════════════════════════")
    log.info("   Energy Optimizer v2.1 — HAOS")
    log.info("═══════════════════════════════════════")
    log.info(f"  Supervisor token: {'OK' if HA_TOKEN else 'NOT FOUND'}")

    if not MODEL_FILE.exists():
        log.info("No saved model — starting initial training...")
        threading.Thread(target=train_model, daemon=True).start()

    # Warm up consumption cache in background
    threading.Thread(target=_refresh_consumption_cache, daemon=True).start()

    scheduler = BackgroundScheduler(timezone="Europe/Madrid")
    _scheduler_ref = scheduler

    interval = OPT.get("decision_interval_minutes", 15)
    scheduler.add_job(run_cycle, "interval", minutes=interval, id="cycle",
                      next_run_time=datetime.now() + timedelta(seconds=15))

    cron = OPT.get("retrain_cron", "0 3 * * *").split()
    if len(cron) == 5:
        scheduler.add_job(train_model, "cron",
                          minute=cron[0], hour=cron[1], day=cron[2],
                          month=cron[3], day_of_week=cron[4], id="retrain")

    scheduler.start()
    log.info(f"  Cycle every {interval} min · Retrain: {OPT.get('retrain_cron','0 3 * * *')}")
    app.run(host="0.0.0.0", port=8765, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
