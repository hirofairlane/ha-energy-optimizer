#!/usr/bin/env python3
"""
Energy Optimizer — Home Assistant Add-on v2.0
Smart energy management: battery, heat pump, pool pump, dishwasher
Logic: tariff-based rules + scikit-learn ML model
History: HA REST API /api/history (native recorder)
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
OPTIONS_FILE   = DATA_DIR / "options.json"
MODEL_FILE     = DATA_DIR / "model.pkl"
DECISIONS_FILE = DATA_DIR / "decisions.json"
SAVINGS_FILE   = DATA_DIR / "savings.json"
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
HEADERS  = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type":  "application/json",
}

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
    if s:
        return s.get("state", default)
    return default

def ha_float(entity_id: str, default: float = 0.0) -> float:
    s = ha_state(entity_id)
    if s:
        try:
            return float(s["state"])
        except (ValueError, KeyError, TypeError):
            pass
    log.debug(f"Could not read float from {entity_id}, using {default}")
    return default

def ha_service(domain: str, service: str, data: dict = None) -> bool:
    log.info(f"  → Service: {domain}.{service} {data or ''}")
    return ha_post(f"/api/services/{domain}/{service}", data or {})

def ha_set_number(entity_id: str, value: float) -> bool:
    return ha_service("number", "set_value", {"entity_id": entity_id, "value": value})

def ha_set_select(entity_id: str, option: str) -> bool:
    return ha_service("select", "select_option", {"entity_id": entity_id, "option": option})

def ha_switch(entity_id: str, turn_on: bool) -> bool:
    return ha_service("switch", "turn_on" if turn_on else "turn_off", {"entity_id": entity_id})

def ha_history(entity_id: str, days: int = 45) -> list:
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    result = ha_get(
        f"/api/history/period/{start}",
        params={"filter_entity_id": entity_id, "minimal_response": "true"},
    )
    if result and isinstance(result, list) and len(result) > 0:
        return result[0]
    return []

# ── Electricity tariff ───────────────────────────────────────────────────────
PRICE = {
    "peak":   0.30,
    "mid":    0.18,
    "valley": 0.08,
    "export": 0.06,
}

def current_tariff() -> dict:
    now     = datetime.now()
    hour    = now.hour
    weekend = now.weekday() >= 5

    if weekend or (0 <= hour < 8):
        period = "valley"
    elif (10 <= hour < 14) or (18 <= hour < 22):
        period = "peak"
    else:
        period = "mid"

    return {
        "period":     period,
        "price_kwh":  PRICE[period],
        "export_kwh": PRICE["export"],
        "hour":       hour,
        "weekend":    weekend,
    }

# ── Sensor reading ───────────────────────────────────────────────────────────
def read_sensors() -> dict:
    o = OPT
    return {
        "battery_soc":        ha_float(o.get("sensor_battery_soc",         "sensor.battery_state_of_capacity")),
        "battery_power":      ha_float(o.get("sensor_battery_power",        "sensor.battery_charge_discharge_power")),
        "grid_power":         ha_float(o.get("sensor_grid_power",           "sensor.acometida_general_power")),
        "solar_current_hour": ha_float(o.get("sensor_solar_current_hour",   "sensor.energy_current_hour")),
        "solar_next_hour":    ha_float(o.get("sensor_solar_next_hour",      "sensor.energy_next_hour")),
        "solar_today":        ha_float(o.get("sensor_solar_today",          "sensor.energy_production_today")),
        "solar_tomorrow":     ha_float(o.get("sensor_solar_tomorrow",       "sensor.energy_production_tomorrow")),
        "temp_outdoor":       ha_float(o.get("sensor_temp_outdoor",         "sensor.ebusd_broadcast_outsidetemp_temp2")),
        "temp_indoor":        ha_float(o.get("sensor_temp_salon",           "sensor.media_salon")),
        "pool_hours_day":     ha_float(o.get("sensor_pool_hours_day",       "sensor.depuradora_encendida_24h")),
        "pool_hours_week":    ha_float(o.get("sensor_pool_hours_week",      "sensor.depuradora_encendida_semana")),
        "dishwasher_state":   ha_str(o.get("sensor_dishwasher_state",       "sensor.lavavajillas_operation_state")),
        "ts": datetime.now().isoformat(),
    }

# ── Battery direct control ───────────────────────────────────────────────────
def set_battery_charge_target(target_soc: int, charge_power_w: int = 3000) -> bool:
    """
    Charge battery from grid up to target_soc%.
    Replicates the logic from the HA scripts using direct entity control.
    Mode: time_of_use_luna2000 (force grid charge to cutoff SOC).
    """
    o = OPT
    cutoff_ent = o.get("number_battery_charge_cutoff", "number.battery_grid_charge_cutoff_soc")
    power_ent  = o.get("number_battery_charge_power",  "a186f9599e9cad7127bca381f7a8bfb2")
    backup_ent = o.get("number_battery_backup_soc",    "de4a2bf18222f354228cdb112b65e882")
    switch_ent = o.get("switch_battery_force_charge",  "02db00e10018b01211507db92819a25a")
    mode_ent   = o.get("select_battery_mode",          "select.battery_working_mode")

    results = [
        ha_set_number(cutoff_ent, target_soc),
        ha_set_number(power_ent, charge_power_w),
        ha_set_number(backup_ent, target_soc),
        ha_switch(switch_ent, True),
        ha_set_select(mode_ent, "time_of_use_luna2000"),
    ]
    ok = all(results)
    log.info(f"  Battery → charge to {target_soc}% @ {charge_power_w}W — {'OK' if ok else 'ERROR'}")
    return ok

def set_battery_self_consumption(min_soc: int = 20) -> bool:
    """
    Switch to self-consumption mode (stop grid charging, use solar first).
    """
    o = OPT
    cutoff_ent = o.get("number_battery_charge_cutoff", "number.battery_grid_charge_cutoff_soc")
    mode_ent   = o.get("select_battery_mode",          "select.battery_working_mode")

    results = [
        ha_set_number(cutoff_ent, min_soc),
        ha_set_select(mode_ent, "maximise_self_consumption"),
    ]
    ok = all(results)
    log.info(f"  Battery → self-consumption (min {min_soc}%) — {'OK' if ok else 'ERROR'}")
    return ok

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
    entity = OPT.get("sensor_battery_soc", "sensor.battery_state_of_capacity")
    rows   = ha_history(entity, days=60)
    df     = _history_to_df(rows)

    if df is None:
        log.warning(f"Insufficient history ({len(rows)} samples). Rules-only mode.")
        return False

    X, y = df[FEATURES], df["value"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr",    GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )),
    ])
    scores = cross_val_score(pipe, X, y, cv=3, scoring="r2")
    pipe.fit(X, y)

    joblib.dump({
        "pipeline":   pipe,
        "features":   FEATURES,
        "trained_at": datetime.now().isoformat(),
        "r2_cv_mean": float(np.mean(scores)),
        "n_samples":  len(df),
    }, MODEL_FILE)
    log.info(f"Model trained: {len(df)} samples, R² CV={np.mean(scores):.3f}")
    return True

def predict_soc(sensors: dict) -> dict:
    if MODEL_FILE.exists():
        try:
            art  = joblib.load(MODEL_FILE)
            pipe = art["pipeline"]
            now  = datetime.now()
            X = pd.DataFrame([{
                "hour":        now.hour,
                "weekday":     now.weekday(),
                "month":       now.month,
                "lag1":        sensors["battery_soc"],
                "lag4":        sensors["battery_soc"],
                "roll4":       sensors["battery_soc"],
                "solar_proxy": 1 if 8 <= now.hour <= 19 else 0,
            }])
            pred = float(pipe.predict(X[FEATURES])[0])
            pred = max(0.0, min(100.0, pred))
            return {
                "predicted_soc": pred,
                "method":        "ml",
                "r2":            art.get("r2_cv_mean"),
                "trained_at":    art.get("trained_at"),
            }
        except Exception as e:
            log.warning(f"ML prediction error: {e}. Falling back to rules.")

    solar_tomorrow = sensors.get("solar_tomorrow", 0)
    pred = 80.0 if solar_tomorrow < OPT.get("solar_tomorrow_irrisoria_kwh", 2.0) else 50.0
    return {"predicted_soc": pred, "method": "rules_fallback"}

# ── Heat pump logic ──────────────────────────────────────────────────────────
def _is_summer() -> bool:
    m = datetime.now().month
    return OPT.get("summer_start_month", 6) <= m <= OPT.get("summer_end_month", 9)

def decide_heat_pump(sensors: dict) -> dict:
    soc        = sensors["battery_soc"]
    batt_power = sensors["battery_power"]
    temp_in    = sensors["temp_indoor"]
    hour       = datetime.now().hour
    daytime    = 8 <= hour <= 20
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

    return {
        "entity": entity,
        "target": target,
        "reason": reason,
        "season": "summer" if _is_summer() else "winter",
    }

# ── Battery decision ─────────────────────────────────────────────────────────
def decide_battery(sensors: dict, tariff: dict, prediction: dict) -> dict:
    soc          = sensors["battery_soc"]
    period       = tariff["period"]
    solar_tm     = sensors["solar_tomorrow"]
    low_solar_tm = solar_tm < OPT.get("solar_tomorrow_irrisoria_kwh", 2.0)
    thr_emerg    = OPT.get("battery_emergency_threshold", 10)
    thr_low      = OPT.get("battery_low_threshold",       30)
    thr_med      = OPT.get("battery_medium_threshold",    50)

    # Peak: no grid charging except emergency
    if period == "peak":
        if soc < thr_emerg:
            return {"action": "charge", "target_soc": 30, "power_w": 1500,
                    "reason": f"EMERGENCY during peak: SOC={soc:.0f}% < {thr_emerg}%"}
        return {"action": "none",
                "reason": f"Peak ({PRICE['peak']}€/kWh): no grid charging"}

    # Valley: cheapest energy — maximise charging opportunity
    if period == "valley":
        if soc >= 95:
            return {"action": "self_consumption",
                    "reason": f"Valley, battery already full ({soc:.0f}%)"}
        if low_solar_tm and soc < 80:
            return {"action": "charge", "target_soc": 80, "power_w": 3000,
                    "reason": f"Valley + low solar tomorrow ({solar_tm:.1f} kWh) → charge to 80%"}
        if soc < thr_low:
            return {"action": "charge", "target_soc": 50, "power_w": 3000,
                    "reason": f"Valley + low SOC ({soc:.0f}%) → charge to 50%"}
        if soc < thr_med:
            return {"action": "charge", "target_soc": thr_med, "power_w": 2000,
                    "reason": f"Valley + mid SOC ({soc:.0f}%) → charge to {thr_med}%"}
        return {"action": "self_consumption",
                "reason": f"Valley, battery sufficient ({soc:.0f}%)"}

    # Mid: only act if critical
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

# ── Pool pump logic ──────────────────────────────────────────────────────────
def decide_pool(sensors: dict, tariff: dict) -> dict:
    month   = datetime.now().month
    summer  = OPT.get("summer_start_month", 6) <= month <= OPT.get("summer_end_month", 9)
    hrs_day = sensors.get("pool_hours_day", 0)
    hrs_wk  = sensors.get("pool_hours_week", 0)

    if summer and hrs_day >= 1.0:
        return {"action": False, "reason": f"Daily hours met ({hrs_day:.1f}h ≥ 1h/day)"}
    if not summer and hrs_wk >= 1.0:
        return {"action": False, "reason": f"Weekly hours met ({hrs_wk:.1f}h ≥ 1h/week)"}

    solar_now = sensors["solar_current_hour"]
    soc       = sensors["battery_soc"]
    period    = tariff["period"]

    if solar_now > 1.2 and soc > 50:
        return {"action": True,
                "reason": f"Solar surplus ({solar_now:.2f} kWh/h), SOC={soc:.0f}%"}
    if period == "valley":
        return {"action": True,
                "reason": f"Valley tariff ({PRICE['valley']}€/kWh), hours pending"}
    if summer and hrs_day < 0.1 and datetime.now().hour >= 20:
        return {"action": True,
                "reason": f"Urgent: only {hrs_day:.1f}h today, late hour"}

    return {"action": False,
            "reason": f"Waiting — period={period}, solar={solar_now:.2f} kWh/h"}

# ── Dishwasher logic ─────────────────────────────────────────────────────────
def decide_dishwasher(sensors: dict, tariff: dict) -> dict:
    state  = sensors.get("dishwasher_state", "")
    period = tariff["period"]
    solar  = sensors["solar_current_hour"]
    soc    = sensors["battery_soc"]

    if state.lower() == "running":
        return {"action": None, "reason": "Already running", "state": state}
    if state.lower() != "ready":
        return {"action": None, "reason": f"Not ready (state: {state or 'unknown'})", "state": state}

    # Ready — decide best time to start
    if solar > 1.5 and soc > 60:
        return {"action": True,
                "reason": f"Solar surplus ({solar:.2f} kWh/h) + good SOC ({soc:.0f}%) — optimal time",
                "state": state}
    if period == "valley":
        return {"action": True,
                "reason": f"Valley tariff ({PRICE['valley']}€/kWh) — cheap electricity",
                "state": state}
    if period == "peak":
        return {"action": False,
                "reason": f"Peak tariff ({PRICE['peak']}€/kWh) — waiting for better conditions",
                "state": state}

    return {"action": False,
            "reason": "Mid tariff — waiting for solar surplus or valley",
            "state": state}

# ── Savings tracker ──────────────────────────────────────────────────────────
def _load_savings() -> dict:
    if SAVINGS_FILE.exists():
        try:
            return json.loads(SAVINGS_FILE.read_text())
        except Exception:
            pass
    return {
        "total_kwh_avoided_peak": 0.0,
        "total_eur_saved": 0.0,
        "decisions_count": 0,
        "since": datetime.now().date().isoformat(),
    }

def _update_savings(sensors: dict, tariff: dict):
    """
    Estimate savings: when battery discharges during peak and covers load,
    we avoid buying from the grid at peak price.
    """
    if tariff["period"] != "peak":
        return
    batt_power = sensors["battery_power"]
    grid_power = sensors["grid_power"]
    if batt_power >= 0 or grid_power > 500:
        return  # Not discharging, or still pulling from grid heavily

    savings    = _load_savings()
    interval_h = OPT.get("decision_interval_minutes", 15) / 60
    kwh        = min(abs(batt_power) * interval_h / 1000, 2.0)  # cap at 2 kWh per cycle
    eur        = kwh * (PRICE["peak"] - PRICE["export"])

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

    decision = {
        "timestamp":  datetime.now().isoformat(),
        "sensors":    sensors,
        "tariff":     tariff,
        "prediction": prediction,
        "actions":    [],
        "skipped":    [],
    }

    # 1. Battery ──────────────────────────────────────────────────────────────
    bat = decide_battery(sensors, tariff, prediction)
    if bat["action"] == "charge":
        ok = set_battery_charge_target(bat["target_soc"], bat.get("power_w", 3000))
        decision["actions"].append({
            "type": "battery", "action": "charge",
            "target_soc": bat["target_soc"], "reason": bat["reason"], "ok": ok,
        })
        log.info(f"  [BATTERY] Charge → {bat['target_soc']}% — {bat['reason']}")
    elif bat["action"] == "self_consumption":
        ok = set_battery_self_consumption()
        decision["actions"].append({
            "type": "battery", "action": "self_consumption",
            "reason": bat["reason"], "ok": ok,
        })
        log.info(f"  [BATTERY] Self-consumption — {bat['reason']}")
    else:
        decision["skipped"].append({"type": "battery", "reason": bat["reason"]})
        log.info(f"  [BATTERY] No action — {bat['reason']}")

    # 2. Heat pump ────────────────────────────────────────────────────────────
    hp = decide_heat_pump(sensors)
    ok = ha_set_number(hp["entity"], hp["target"])
    decision["actions"].append({
        "type": "heat_pump", "entity": hp["entity"],
        "value": hp["target"], "reason": hp["reason"], "ok": ok,
    })
    log.info(f"  [HEAT PUMP/{hp['season'].upper()}] {hp['reason']} → {hp['target']}°C")

    # 3. Pool pump ────────────────────────────────────────────────────────────
    pool      = decide_pool(sensors, tariff)
    pool_sw   = OPT.get("switch_pool", "switch.depuradora")
    ok        = ha_switch(pool_sw, pool["action"])
    key       = "actions" if pool["action"] else "skipped"
    decision[key].append({
        "type": "pool", "action": pool["action"],
        "reason": pool["reason"], "ok": ok,
    })
    log.info(f"  [POOL] {'ON' if pool['action'] else 'OFF'} — {pool['reason']}")

    # 4. Dishwasher ───────────────────────────────────────────────────────────
    dw = decide_dishwasher(sensors, tariff)
    if dw["action"] is True:
        dw_sw = OPT.get("switch_dishwasher", "")
        if dw_sw:
            ok = ha_switch(dw_sw, True)
            decision["actions"].append({
                "type": "dishwasher", "action": "start",
                "reason": dw["reason"], "ok": ok,
            })
            log.info(f"  [DISHWASHER] START — {dw['reason']}")
        else:
            decision["skipped"].append({
                "type": "dishwasher",
                "reason": f"Ready ({dw['reason']}) — no control switch configured",
            })
            log.info(f"  [DISHWASHER] Ready but no switch configured — {dw['reason']}")
    elif dw["action"] is False:
        decision["skipped"].append({"type": "dishwasher", "reason": dw["reason"]})
        log.info(f"  [DISHWASHER] Waiting — {dw['reason']}")
    else:
        log.info(f"  [DISHWASHER] {dw['reason']}")

    # Update savings
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
    history = history[-500:]
    DECISIONS_FILE.write_text(json.dumps(history, indent=2, default=str))

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
.sub{font-size:.78rem;color:var(--m);margin-top:.2rem}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:.3rem;font-size:.72rem;font-weight:600}
.peak{color:var(--r);background:rgba(248,113,113,.12)}
.valley{color:var(--g);background:rgba(74,222,128,.12)}
.mid{color:var(--y);background:rgba(251,191,36,.12)}
.running{color:var(--g);background:rgba(74,222,128,.12)}
.ready{color:var(--y);background:rgba(251,191,36,.12)}
.actions{display:flex;gap:.5rem;margin-bottom:1rem;flex-wrap:wrap}
button{background:var(--a);color:#0f172a;border:none;padding:.45rem 1rem;border-radius:.5rem;font-weight:700;cursor:pointer;font-size:.8rem;transition:.15s opacity,.1s transform}
button:hover{opacity:.85}button:active{transform:scale(.97)}
.btn-y{background:var(--y)}.btn-g{background:var(--g)}.btn-r{background:var(--r)}
.btn-sm{padding:.3rem .7rem;font-size:.72rem}
.charge-wrap{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.5rem}
.savings{background:linear-gradient(135deg,rgba(74,222,128,.08),rgba(56,189,248,.08));border:1px solid rgba(74,222,128,.3);border-radius:.75rem;padding:.9rem;display:flex;gap:2rem;align-items:center;flex-wrap:wrap;margin-bottom:1rem}
.savings-num{font-size:2rem;font-weight:700;color:var(--g)}
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
<h1>⚡ Energy Optimizer</h1>

<div id="kpis" class="grid"></div>

<div id="savings-box"></div>

<div class="actions">
  <button onclick="runCycle()">▶ Run cycle now</button>
  <button class="btn-y" onclick="retrain()">🔁 Retrain model</button>
</div>

<div class="card" style="margin-bottom:1rem">
  <h2 style="margin-top:0">⚡ Battery manual control</h2>
  <div class="charge-wrap">
    <button class="btn-sm" onclick="chargeToTarget(30)">Charge 30%</button>
    <button class="btn-sm" onclick="chargeToTarget(50)">Charge 50%</button>
    <button class="btn-sm" onclick="chargeToTarget(80)">Charge 80%</button>
    <button class="btn-sm" onclick="chargeToTarget(99)">Charge 99%</button>
    <button class="btn-sm btn-g" onclick="selfConsumption()">☀ Self-consumption</button>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <h2 style="margin-top:0">Battery SOC — last 24h</h2>
    <canvas id="socChart"></canvas>
  </div>
  <div class="card">
    <h2 style="margin-top:0">Solar — yesterday / today / tomorrow</h2>
    <canvas id="solarChart"></canvas>
  </div>
</div>

<h2>Recent decisions</h2>
<div class="card" style="overflow-x:auto">
  <table><thead><tr><th>Time</th><th>Type</th><th>Reason</th><th>OK</th></tr></thead>
  <tbody id="log"></tbody></table>
</div>

<script>
const PC={peak:'peak',valley:'valley',mid:'mid'};
let socInst=null,solInst=null;

function mkChart(id,type,labels,datasets,yOpts={}){
  const ctx=document.getElementById(id).getContext('2d');
  const base={responsive:true,maintainAspectRatio:true,
    plugins:{legend:{labels:{color:'#94a3b8',font:{size:11}}}},
    scales:{x:{ticks:{color:'#94a3b8',maxTicksLimit:8,font:{size:10}},grid:{color:'#334155'}},
            y:{ticks:{color:'#94a3b8',font:{size:10}},grid:{color:'#334155'},...yOpts}}};
  return new Chart(ctx,{type,data:{labels,datasets},options:base});
}

async function loadCharts(){
  try{
    const cd=await fetch('/api/chart-data').then(r=>r.json());
    if(socInst) socInst.destroy();
    socInst=mkChart('socChart','line',cd.soc.labels,[{
      label:'SOC %',data:cd.soc.values,
      borderColor:'#38bdf8',backgroundColor:'rgba(56,189,248,.1)',
      fill:true,tension:.3,pointRadius:0
    }],{min:0,max:100});
    if(solInst) solInst.destroy();
    solInst=mkChart('solarChart','bar',
      ['Yesterday','Today','Tomorrow'],
      [{label:'kWh',data:[cd.solar.yesterday,cd.solar.today,cd.solar.tomorrow],
        backgroundColor:['rgba(251,191,36,.5)','rgba(74,222,128,.7)','rgba(56,189,248,.5)'],
        borderColor:['#fbbf24','#4ade80','#38bdf8'],borderWidth:1,borderRadius:4}]);
  }catch(e){console.warn('Chart load error',e);}
}

async function load(){
  try{
    const [s,d,sv]=await Promise.all([
      fetch('/api/status').then(r=>r.json()),
      fetch('/api/decisions?limit=30').then(r=>r.json()),
      fetch('/api/savings').then(r=>r.json()),
    ]);
    const {sensors:ss,tariff:t,model:m}=s;
    const dw=ss.dishwasher_state||'--';
    const dwCls=dw.toLowerCase()==='running'?'running':dw.toLowerCase()==='ready'?'ready':'';
    const battDir=ss.battery_power>0?'▲ Charging':'▼ Discharging';

    document.getElementById('kpis').innerHTML=`
      <div class="card"><div class="metric">${Number(ss.battery_soc??0).toFixed(0)}%</div>
        <div class="label">Battery SOC</div>
        <div class="sub">${battDir} ${Math.abs(ss.battery_power??0).toFixed(0)}W</div></div>
      <div class="card"><div class="metric">${(ss.solar_today??0).toFixed(1)}</div>
        <div class="label">Solar today (kWh)</div>
        <div class="sub">Next hr: ${(ss.solar_next_hour??0).toFixed(2)} kWh</div></div>
      <div class="card"><div class="metric">${(ss.solar_tomorrow??0).toFixed(1)}</div>
        <div class="label">Solar tomorrow (kWh)</div></div>
      <div class="card"><div class="metric ${PC[t.period]}">${(t.price_kwh??0).toFixed(2)}€</div>
        <div class="label">Price/kWh &nbsp;<span class="badge ${PC[t.period]}">${t.period}</span></div></div>
      <div class="card"><div class="metric">${(ss.temp_indoor??0).toFixed(1)}°C</div>
        <div class="label">Indoor temp</div>
        <div class="sub">Outdoor: ${(ss.temp_outdoor??0).toFixed(1)}°C</div></div>
      <div class="card"><div class="metric" style="font-size:1.1rem"><span class="badge ${dwCls}">${dw}</span></div>
        <div class="label">Dishwasher</div></div>
      <div class="card"><div class="metric" style="font-size:1rem">${m?.r2_cv_mean!=null?'R²='+m.r2_cv_mean.toFixed(2):'No model'}</div>
        <div class="label">ML model</div>
        <div class="sub">${m?.n_samples?m.n_samples+' samples':''}</div></div>`;

    document.getElementById('savings-box').innerHTML=`
      <div class="savings">
        <div><div class="savings-num">€${(sv.total_eur_saved??0).toFixed(2)}</div>
          <div class="label">Estimated savings (peak avoided)</div></div>
        <div><div class="metric" style="color:var(--g)">${(sv.total_kwh_avoided_peak??0).toFixed(1)}</div>
          <div class="label">kWh covered by battery at peak</div></div>
        <div style="font-size:.72rem;color:var(--m)">Since ${sv.since??'--'}</div>
      </div>`;

    const tagMap={battery:'bat',heat_pump:'hp',pool:'pool',dishwasher:'dw'};
    const rows=d.slice().reverse().flatMap(dec=>{
      const ts=new Date(dec.timestamp).toLocaleTimeString('en',{hour:'2-digit',minute:'2-digit'});
      return [
        ...(dec.actions||[]).map(a=>`<tr><td>${ts}</td>
          <td><span class="tag ${tagMap[a.type]||''}">${a.type}</span></td>
          <td class="ok">${a.reason}</td>
          <td class="${a.ok?'ok':'err'}">${a.ok?'✓':'✗'}</td></tr>`),
        ...(dec.skipped||[]).map(a=>`<tr><td>${ts}</td>
          <td><span class="tag">${a.type}</span></td>
          <td class="skip">${a.reason}</td>
          <td class="skip">–</td></tr>`)
      ];
    }).join('');
    document.getElementById('log').innerHTML=rows||
      '<tr><td colspan="4" style="color:var(--m)">No decisions yet</td></tr>';
  }catch(e){console.error('Load error',e);}
}

async function runCycle(){
  const d=await fetch('/api/run',{method:'POST'}).then(r=>r.json());
  alert(`Cycle executed\n✓ ${d.actions?.length??0} actions\n– ${d.skipped?.length??0} skipped`);
  load();loadCharts();
}
async function retrain(){
  const d=await fetch('/api/retrain',{method:'POST'}).then(r=>r.json());
  alert(d.message);load();
}
async function chargeToTarget(soc){
  if(!confirm(`Charge battery to ${soc}%?`)) return;
  const r=await fetch('/api/battery/charge',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({target_soc:soc})
  }).then(r=>r.json());
  alert(r.ok?`✓ Charging to ${soc}% started`:'Error setting charge target');
  load();
}
async function selfConsumption(){
  if(!confirm('Switch to self-consumption mode?')) return;
  const r=await fetch('/api/battery/self-consumption',{method:'POST'}).then(r=>r.json());
  alert(r.ok?'✓ Self-consumption mode activated':'Error');
  load();
}

load();loadCharts();
setInterval(load,30000);
setInterval(loadCharts,120000);
</script>
</body></html>"""

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
    return jsonify({"sensors": sensors, "tariff": tariff, "model": model})

@app.route("/api/decisions")
def api_decisions():
    limit = int(request.args.get("limit", 50))
    if DECISIONS_FILE.exists():
        return jsonify(json.loads(DECISIONS_FILE.read_text())[-limit:])
    return jsonify([])

@app.route("/api/savings")
def api_savings():
    return jsonify(_load_savings())

@app.route("/api/chart-data")
def api_chart_data():
    # Battery SOC — last 24h from HA history
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

    # Solar data from sensors + decisions log for yesterday
    sensors        = read_sensors()
    solar_today    = sensors.get("solar_today", 0)
    solar_tomorrow = sensors.get("solar_tomorrow", 0)
    solar_yesterday = 0.0
    if DECISIONS_FILE.exists():
        try:
            history   = json.loads(DECISIONS_FILE.read_text())
            yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
            prev_vals = [
                d["sensors"].get("solar_today", 0)
                for d in history
                if d["timestamp"][:10] == yesterday
            ]
            if prev_vals:
                solar_yesterday = max(prev_vals)
        except Exception:
            pass

    return jsonify({
        "soc":   {"labels": soc_labels, "values": soc_values},
        "solar": {
            "yesterday": round(solar_yesterday, 1),
            "today":     round(solar_today, 1),
            "tomorrow":  round(solar_tomorrow, 1),
        },
    })

@app.route("/api/run", methods=["POST"])
def api_run():
    return jsonify(run_cycle())

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    ok = train_model()
    return jsonify({
        "ok": ok,
        "message": "Model retrained successfully ✓" if ok
                   else "Insufficient history — rules-only mode",
    })

@app.route("/api/battery/charge", methods=["POST"])
def api_battery_charge():
    data       = request.get_json() or {}
    target_soc = max(10, min(100, int(data.get("target_soc", 80))))
    ok         = set_battery_charge_target(target_soc)
    return jsonify({"ok": ok, "target_soc": target_soc})

@app.route("/api/battery/self-consumption", methods=["POST"])
def api_battery_self_consumption():
    ok = set_battery_self_consumption()
    return jsonify({"ok": ok})

@app.route("/api/options")
def api_options():
    safe = {k: v for k, v in OPT.items() if "token" not in k.lower()}
    return jsonify(safe)

# ── Startup ──────────────────────────────────────────────────────────────────
def main():
    log.info("═══════════════════════════════════════")
    log.info("   Energy Optimizer v2.0 — HAOS")
    log.info("═══════════════════════════════════════")
    log.info(f"  Supervisor token: {'OK' if HA_TOKEN else 'NOT FOUND'}")

    if not MODEL_FILE.exists():
        log.info("No saved model — starting initial training thread...")
        threading.Thread(target=train_model, daemon=True).start()

    scheduler = BackgroundScheduler(timezone="Europe/Madrid")
    interval  = OPT.get("decision_interval_minutes", 15)
    scheduler.add_job(
        run_cycle, "interval", minutes=interval, id="cycle",
        next_run_time=datetime.now() + timedelta(seconds=15),
    )

    cron = OPT.get("retrain_cron", "0 3 * * *").split()
    if len(cron) == 5:
        scheduler.add_job(
            train_model, "cron",
            minute=cron[0], hour=cron[1], day=cron[2],
            month=cron[3], day_of_week=cron[4],
            id="retrain",
        )

    scheduler.start()
    log.info(f"  Cycle every {interval} min · Retrain: {OPT.get('retrain_cron','0 3 * * *')}")
    app.run(host="0.0.0.0", port=8765, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
