#!/usr/bin/env python3
"""
Energy Optimizer — Home Assistant Add-on
Smart energy management: battery, heat pump, pool pump, pool cleaner, dishwasher
Logic: adaptive tariff rules + scikit-learn ML model + consumption history

Version is defined ONCE in ADD_ON_VERSION below and must be kept in sync with
the `version:` field in config.yaml (HA Supervisor reads that file directly).
A startup check logs a warning if they drift. See README for release notes.
"""

import math
import os
import re
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, request

# ── ML feature version — increment when feature engineering changes ───────────
MODEL_FEATURE_VER = 3   # v3: dynamic features from wizard (temp_outdoor, solar, grid, submeters)
ADD_ON_VERSION = "5.0.13"  # SOURCE OF TRUTH — bump here AND in config.yaml together

# ── Location (solar elevation formula) ───────────────────────────────────────
HOME_LAT = 40.67   # Guadarrama, Madrid — °N (fallback; wizard overrides)
HOME_LON = -4.00   # Guadarrama, Madrid — °E (fallback)

# ── Solar correction factor cache ─────────────────────────────────────────────
_solar_correction_cache: tuple | None = None   # (factor, datetime)

# ── Persistent data paths ────────────────────────────────────────────────────
DATA_DIR       = Path("/data")
MODEL_FILE     = DATA_DIR / "model.pkl"
DECISIONS_FILE = DATA_DIR / "decisions.json"
SAVINGS_FILE   = DATA_DIR / "savings.json"
TARIFF_FILE    = DATA_DIR / "tariff.json"
SETUP_FILE     = DATA_DIR / "setup.json"
WIZARD_FILE    = DATA_DIR / "wizard_config.json"
SURPLUS_STATE_FILE = DATA_DIR / "surplus_state.json"
COOLING_GATE_STATE_FILE = DATA_DIR / "cooling_gate_state.json"
POOL_MANUAL_STATE_FILE = DATA_DIR / "pool_manual_state.json"
HVAC_DECISION_STATE_FILE = DATA_DIR / "hvac_decision_state.json"
LOG_SUMMARY_STATE_FILE = DATA_DIR / "log_summary_state.json"
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

# ── Version sync check ───────────────────────────────────────────────────────
ADDON_CONFIG_FILE = Path("/addon-config.yaml")  # copied from config.yaml at build

def _check_version_sync() -> None:
    """Warn if ADD_ON_VERSION drifts from the version declared in config.yaml.

    config.yaml is what HA Supervisor reads; ADD_ON_VERSION is what the panel
    and the log banner show. Both must move together when releasing.
    """
    if not ADDON_CONFIG_FILE.exists():
        return  # not packaged in container (dev/test); nothing to compare
    try:
        m = re.search(r'^version:\s*"?([^"\s]+)"?\s*$',
                      ADDON_CONFIG_FILE.read_text(), flags=re.MULTILINE)
    except Exception:
        return
    if m and m.group(1) != ADD_ON_VERSION:
        log.warning(f"  ⚠ Version drift: ADD_ON_VERSION={ADD_ON_VERSION} "
                    f"but config.yaml says {m.group(1)} — bump them together.")

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

def _health_mode_limits() -> tuple:
    """Return (min_soc, max_soc) for charging based on battery_health_mode setting."""
    return {
        "bill_reducer":  (10, 95),
        "optimized":     (20, 90),
        "battery_guard": (25, 85),
    }.get(cfg("battery_health_mode", "bill_reducer"), (10, 95))

# ── Wizard config (from GUI wizard — stored in /data/wizard_config.json) ──────
_WIZARD: dict = {}

def _load_wizard_cache():
    global _WIZARD
    if WIZARD_FILE.exists():
        try:
            _WIZARD = json.loads(WIZARD_FILE.read_text())
        except Exception:
            _WIZARD = {}
    else:
        _WIZARD = {}

def save_wizard(data: dict):
    global _WIZARD
    _WIZARD.update(data)
    WIZARD_FILE.write_text(json.dumps(_WIZARD, indent=2))
    log.info("Wizard configuration saved")

def _wiz(role: str, legacy_cfg_key: str = None, legacy_default=None):
    """3-tier fallback: wizard_config.json > setup.json/options.json > hardcoded default."""
    sensors = _WIZARD.get("sensors", {})
    if role in sensors and sensors[role]:
        return sensors[role]
    if legacy_cfg_key:
        return cfg(legacy_cfg_key, legacy_default)
    return legacy_default

def _home_lat() -> float:
    loc = _WIZARD.get("location", {})
    return float(loc.get("latitude") or HOME_LAT)

def _home_lon() -> float:
    loc = _WIZARD.get("location", {})
    return float(loc.get("longitude") or HOME_LON)

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
        # Expose the configured hour-sets so downstream decision logic
        # (notably calculate_optimal_soc) can scale with the user's actual
        # tariff geometry instead of assuming the Spanish 2.0TD default of
        # 8 peak hours. Without this every non-Spanish installation under-
        # or over-estimates peak consumption proportionally — see fix for
        # Karplyak's 16h-peak Ukrainian tariff in issue #5.
        "peak_hours":   list(peak_h),
        "valley_hours": list(valley_h),
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

# ── Entity discovery ─────────────────────────────────────────────────────────
ROLE_HINTS = {
    "grid_power":          {"domains":["sensor"],"device_class":["power"],"units":["W","kW"],"keywords":["grid","acometida","net","meter","import","export","red"]},
    "solar_power":         {"domains":["sensor"],"device_class":["power"],"units":["W","kW"],"keywords":["solar","pv","placas","inverter","photovoltaic","panels","produccion"]},
    "battery_soc":         {"domains":["sensor"],"device_class":["battery"],"units":["%"],"keywords":["soc","battery","bateria","capacity","charge","estado"]},
    "battery_power":       {"domains":["sensor"],"device_class":["power"],"units":["W","kW"],"keywords":["battery","bateria","charge","discharge","luna","storage"]},
    "temp_outdoor":        {"domains":["sensor"],"device_class":["temperature"],"units":["°C","°F"],"keywords":["outdoor","outside","exterior","outside_temp","outsidetemp","ambient","exterior"]},
    "temp_indoor":         {"domains":["sensor"],"device_class":["temperature"],"units":["°C","°F"],"keywords":["indoor","interior","room","salon","living","home","temperatura"]},
    "solar_fc_today":      {"domains":["sensor"],"device_class":["energy"],"units":["kWh"],"keywords":["today","hoy","production_today","forecast_today","solar_today"]},
    "solar_fc_tomorrow":   {"domains":["sensor"],"device_class":["energy"],"units":["kWh"],"keywords":["tomorrow","manana","production_tomorrow","forecast_tomorrow"]},
    "solar_fc_h0":         {"domains":["sensor"],"device_class":["energy","power"],"units":["kWh","W"],"keywords":["current_hour","esta_hora","h0","energy_now","forecast_hour"]},
    "solar_fc_h1":         {"domains":["sensor"],"device_class":["energy","power"],"units":["kWh","W"],"keywords":["next_hour","proxima_hora","h1","energy_next"]},
    "weather":             {"domains":["weather"],"device_class":[],"units":[],"keywords":["aemet","openweather","weather","tiempo","forecast"]},
    "battery_mode_select": {"domains":["select"],"device_class":[],"units":[],"keywords":["battery","working_mode","bateria","luna","storage","mode"]},
    "battery_cutoff_soc":  {"domains":["number"],"device_class":[],"units":["%"],"keywords":["cutoff","corte","charge_cutoff","grid_charge","soc"]},
    "battery_backup_soc":  {"domains":["number"],"device_class":[],"units":["%"],"keywords":["backup","reserva","minimum","minimo","reserve"]},
    "battery_charge_power":{"domains":["number"],"device_class":["power"],"units":["W","kW"],"keywords":["charge_power","charging_power","potencia_carga"]},
    "battery_force_charge":{"domains":["switch","button","input_boolean"],"device_class":[],"units":[],"keywords":["force_charge","forzar","force","forced","solar","storage","battery","luna","sungrow","huawei"]},
    "pool_switch":         {"domains":["switch"],"device_class":[],"units":[],"keywords":["pool","piscina","pump","bomba","depuradora","filtro"]},
    "pool_cleaner":        {"domains":["switch"],"device_class":[],"units":[],"keywords":["cleaner","limpiafondos","limpia","robot","vacuum"]},
    "pool_hours_day":      {"domains":["sensor"],"device_class":["duration"],"units":["h","min"],"keywords":["pool","piscina","hours","horas","encendida","runtime"]},
    "hvac_mode":           {"domains":["climate","select","input_select"],"device_class":[],"units":[],"keywords":["hvac","clima","aerotermia","heat_pump","bomba_calor","ebus","viessmann","climate"]},
    "hvac_temp_heat":      {"domains":["number","input_number"],"device_class":["temperature"],"units":["°C"],"keywords":["heat","calor","heating","setpoint","consigna","target","manualtemp","z1manual"]},
    "hvac_temp_cool":      {"domains":["number","input_number"],"device_class":["temperature"],"units":["°C"],"keywords":["cool","frio","cooling","setpoint","consigna","target","coolingtemp","z1cooling"]},
    "dishwasher_state":    {"domains":["sensor"],"device_class":[],"units":[],"keywords":["dishwasher","lavavajillas","washer","machine","operation","estado"]},
    "washer_state":        {"domains":["sensor"],"device_class":[],"units":[],"keywords":["washer","washing","lavadora","wash","wm","laundry","operation"]},
    "washer_power":        {"domains":["sensor"],"device_class":["power"],"units":["W","kW"],"keywords":["washer","washing","lavadora","wash","laundry"]},
    "dryer_state":         {"domains":["sensor"],"device_class":[],"units":[],"keywords":["dryer","secadora","dry","tumble","dryer","secar"]},
    "dryer_power":         {"domains":["sensor"],"device_class":["power"],"units":["W","kW"],"keywords":["dryer","secadora","dry","tumble"]},
    "ev_charger":          {"domains":["switch","number"],"device_class":[],"units":["A","W"],"keywords":["ev","car","coche","charger","cargador","wallbox","zappi","ocpp"]},
}

def _score_entity(state: dict, role: str) -> int:
    hints = ROLE_HINTS.get(role, {})
    entity_id = state.get("entity_id", "")
    domain = entity_id.split(".")[0] if "." in entity_id else ""
    attrs = state.get("attributes", {})
    dc = attrs.get("device_class", "")
    unit = attrs.get("unit_of_measurement", "")
    name_lower = (attrs.get("friendly_name", "") + " " + entity_id).lower()
    score = 0
    if hints.get("domains") and domain not in hints["domains"]:
        return 0
    if dc and hints.get("device_class") and dc in hints["device_class"]:
        score += 50
    if unit and hints.get("units") and unit in hints["units"]:
        score += 40
    for kw in hints.get("keywords", []):
        if kw.lower() in name_lower:
            score += 25
    return score

def get_entity_candidates(role: str, all_states: list, top_n: int = 8) -> list:
    """Return top_n scored entity candidates for a given role."""
    scored = []
    for state in all_states:
        s = _score_entity(state, role)
        if s > 0:
            attrs = state.get("attributes", {})
            scored.append({
                "entity_id": state.get("entity_id"),
                "name": attrs.get("friendly_name", state.get("entity_id")),
                "state": state.get("state"),
                "unit": attrs.get("unit_of_measurement", ""),
                "device_class": attrs.get("device_class", ""),
                "score": s,
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]

def get_entities_by_domain(domain: str, all_states: list) -> list:
    return [
        {
            "entity_id": s.get("entity_id"),
            "name": s.get("attributes", {}).get("friendly_name", s.get("entity_id")),
            "state": s.get("state"),
        }
        for s in all_states
        if s.get("entity_id", "").startswith(domain + ".")
    ]

# ── Solar geometry & correction ──────────────────────────────────────────────
def _solar_proxy_for_hour(hour: int, month: int) -> float:
    """Continuous solar proxy 0–1 from geometric elevation angle.
    Replaces the old binary (8 ≤ h ≤ 19). Computed for HOME_LAT.
    0 = sun below horizon; 1 = summer noon peak (~72° elevation)."""
    doy   = month * 30.4 - 15   # approx day-of-year for mid-month
    decl  = math.radians(23.45 * math.sin(math.radians(360 / 365 * (doy - 81))))
    lat_r = math.radians(HOME_LAT)
    # Solar noon in Madrid ≈ 13:30 local (UTC+1/+2, solar time equation avg)
    h_ang = math.radians(15 * (hour - 13.5))
    sin_e = (math.sin(lat_r) * math.sin(decl) +
             math.cos(lat_r) * math.cos(decl) * math.cos(h_ang))
    elev  = max(0.0, math.degrees(math.asin(max(-1.0, min(1.0, sin_e)))))
    return round(min(1.0, elev / 70.0), 3)   # 70° ≈ peak summer noon elevation


def _compute_solar_correction_factor() -> float:
    """Median ratio of actual/forecast daily production from InfluxDB (last 30d).
    Captures systematic terrain effects (e.g. a hill blocking afternoon sun).
    Cached for 6h. Returns 1.0 if InfluxDB unavailable or insufficient data."""
    global _solar_correction_cache
    now = datetime.now()
    if (_solar_correction_cache and
            (now - _solar_correction_cache[1]).total_seconds() < 6 * 3600):
        return _solar_correction_cache[0]

    factor = 1.0
    try:
        influx_u = cfg("influxdb_url", "").strip()
        if not influx_u:
            return 1.0
        sol_id  = cfg("sensor_solar_today",    "sensor.energy_production_today").split(".")[-1]
        fc_id   = cfg("sensor_solar_tomorrow", "sensor.energy_production_tomorrow").split(".")[-1]
        db      = cfg("influxdb_db", "homeassistant").strip()
        usr     = cfg("influxdb_user",     "").strip()
        pwd     = cfg("influxdb_password", "")

        q_act = (f'SELECT MAX("value") FROM /.*/ WHERE "entity_id" = \'{sol_id}\' '
                 f'AND time >= now() - 32d GROUP BY time(1d) fill(null)')
        q_fc  = (f'SELECT LAST("value") FROM /.*/ WHERE "entity_id" = \'{fc_id}\' '
                 f'AND time >= now() - 33d GROUP BY time(1d) fill(null)')
        r_act, _, _ = _influx_query(influx_u, db, q_act, usr, pwd)
        r_fc,  _, _ = _influx_query(influx_u, db, q_fc,  usr, pwd)

        actual_d, forecast_d = {}, {}
        if r_act:
            s = r_act.json()["results"][0].get("series", [])
            if s:
                ci = {c: i for i, c in enumerate(s[0]["columns"])}
                for pt in s[0]["values"]:
                    if pt[ci["max"]] and float(pt[ci["max"]]) > 1.0:
                        d = datetime.fromtimestamp(pt[ci["time"]] / 1000).date()
                        actual_d[d] = float(pt[ci["max"]])
        if r_fc:
            s = r_fc.json()["results"][0].get("series", [])
            if s:
                ci = {c: i for i, c in enumerate(s[0]["columns"])}
                for pt in s[0]["values"]:
                    if pt[ci["last"]] and float(pt[ci["last"]]) > 1.0:
                        d = datetime.fromtimestamp(pt[ci["time"]] / 1000).date() + timedelta(days=1)
                        forecast_d[d] = float(pt[ci["last"]])

        ratios = [actual_d[d] / forecast_d[d] for d in actual_d
                  if d in forecast_d and forecast_d[d] > 1.0]
        if len(ratios) >= 7:
            ratios.sort()
            median = ratios[len(ratios) // 2]
            factor = max(0.3, min(1.5, round(median, 3)))
            log.info(f"Solar correction factor: {factor:.3f} (n={len(ratios)} days)")
    except Exception as e:
        log.debug(f"Solar correction factor error: {e}")

    _solar_correction_cache = (factor, now)
    return factor


# ── Sensor reading ───────────────────────────────────────────────────────────
def _read_battery_power() -> float:
    """Read battery power W. Supports single signed sensor or split charge/discharge.

    Single sensor (Huawei default): +ve = charging, -ve = discharging.
    Split sensors (Deye/Solarman): charge_entity ≥ 0, discharge_entity ≥ 0.
      Combined: charge − discharge → +ve = charging, -ve = discharging.
    """
    if _WIZARD.get("battery_power_split"):
        sc  = _WIZARD.get("sensors", {})
        chg = ha_float(sc.get("battery_charge_entity",    "")) or 0.0
        dis = ha_float(sc.get("battery_discharge_entity", "")) or 0.0
        return chg - dis
    return ha_float(_wiz("battery_power", "sensor_battery_power",
                         "sensor.battery_charge_discharge_power"))


def read_sensors() -> dict:
    # Wizard exposes a "Flip sign (multiply by −1)" toggle on the Grid step
    # for installations whose meter reports the opposite convention than the
    # add-on assumes (we want +ve = exporting, −ve = importing). Apply once
    # here so the rest of the engine sees the canonical sign.
    grid_raw = ha_float(_wiz("grid_power", "sensor_grid_power", ""))
    if _WIZARD.get("grid_flip") and grid_raw is not None:
        grid_raw = -grid_raw

    s: dict = {
        "battery_soc":        ha_float(_wiz("battery_soc",        "sensor_battery_soc",       "sensor.battery_state_of_capacity")),
        "battery_power":      _read_battery_power(),
        "grid_power":         grid_raw,
        "solar_power":        ha_float(_wiz("solar_power",        "sensor_solar_power",        "")),
        "solar_current_hour": ha_float(_wiz("solar_fc_h0",        "sensor_solar_current_hour", "sensor.energy_current_hour")),
        "solar_next_hour":    ha_float(_wiz("solar_fc_h1",        "sensor_solar_next_hour",    "sensor.energy_next_hour")),
        "solar_today":        ha_float(_wiz("solar_fc_today",     "sensor_solar_today",        "sensor.energy_production_today")),
        "solar_tomorrow":     ha_float(_wiz("solar_fc_tomorrow",  "sensor_solar_tomorrow",     "sensor.energy_production_tomorrow")),
        "temp_outdoor":       ha_float(_wiz("temp_outdoor",       "sensor_temp_outdoor",       "")),
        "aemet_night_low":    ha_float(_wiz("aemet_night_low",    "sensor_aemet_night_low",    "")),
        "temp_indoor":        ha_float(_wiz("temp_indoor",        "sensor_temp_salon",         "")),
        "pool_hours_day":     ha_float(cfg("sensor_pool_hours_day",  "sensor.depuradora_encendida_24h")),
        "pool_hours_week":    ha_float(cfg("sensor_pool_hours_week", "sensor.depuradora_encendida_semana")),
        "dishwasher_state":   ha_str(_wiz("dishwasher_state",     "sensor_dishwasher_state",   "")),
        "ts":                 datetime.now().isoformat(),
    }

    # ── Appliances from wizard (washer / dryer) ───────────────────────────────
    for role, key in [("washer_state", "washer_state"), ("washer_power", "washer_power"),
                      ("dryer_state",  "dryer_state"),  ("dryer_power",  "dryer_power")]:
        entity = _WIZARD.get("sensors", {}).get(role)
        if entity:
            s[key] = ha_float(entity) if "power" in role else ha_str(entity)

    # ── Grid sub-meters (wizard-configured extra power sensors) ───────────────
    for sm in _WIZARD.get("grid_submeters", []):
        entity = sm.get("entity", "")
        label  = sm.get("name", entity)
        if entity:
            s[f"submeter_{label}"] = ha_float(entity)

    # ── Per-zone indoor temps (read each zone's dedicated sensor) ────────────
    for i, zone in enumerate(_WIZARD.get("hvac_zones", [])):
        ts_entity = zone.get("temp_sensor")
        if ts_entity:
            s[f"temp_zone_{i}"] = ha_float(ts_entity)

    # ── Heat-pump hydraulic loop temps. These are the single most reliable
    #    ground-truth that the unit is actually doing physical work — way more
    #    trustworthy than the power-consumption sensor or the pump-state flag,
    #    both of which silently desync on many ebusd integrations.
    #
    #    `hp_flow_in_c`  = water leaving the heat pump (ida, towards the
    #                      hydraulic loop / underfloor circuit).
    #    `hp_flow_out_c` = water returning to the heat pump (vuelta).
    #    `hp_delta_t_c`  = ida − vuelta (matches the user's template
    #                      `sensor.aerotermia_delta_t_calefaccion` convention):
    #         heating: ΔT > 0  (water leaves hot, returns cooler)
    #         cooling: ΔT < 0  (water leaves cold, returns warmer)
    #         idle:    ΔT ≈ 0  (no circulation, both legs converge)
    hp_in  = _wiz("hp_flow_in_temp",  "sensor_hp_flow_in_temp",  "")
    hp_out = _wiz("hp_flow_out_temp", "sensor_hp_flow_out_temp", "")
    if hp_in:
        s["hp_flow_in_c"]  = ha_float(hp_in)
    if hp_out:
        s["hp_flow_out_c"] = ha_float(hp_out)
    if hp_in and hp_out and s.get("hp_flow_in_c") and s.get("hp_flow_out_c"):
        s["hp_delta_t_c"] = round(s["hp_flow_in_c"] - s["hp_flow_out_c"], 2)

    return s

# ── Battery direct control ───────────────────────────────────────────────────
def _batt(role: str, cfg_key: str, hardcoded: str) -> str:
    """Resolve a battery control entity: wizard > setup > hardcoded UUID/id."""
    return _wiz(role, cfg_key, hardcoded)

def set_battery_charge_target(target_soc: int, charge_power_w: int = 3000) -> bool:
    ops = [
        ha_set_number(_batt("battery_cutoff_soc",   "number_battery_charge_cutoff", "number.battery_grid_charge_cutoff_soc"), target_soc),
        ha_set_number(_batt("battery_backup_soc",   "number_battery_backup_soc",   ""),       target_soc),
    ]
    # Mode select — opt-in. GoodWe handles charging through standalone fast-charge
    # registers and has no mode entity for this; firing select.select_option against
    # a missing/foreign entity returns 400 and taints the whole op as ok=False (#7).
    mode_entity = _batt("battery_mode_select", "select_battery_mode", "")
    if mode_entity:
        ops.append(ha_set_select(mode_entity, "time_of_use_luna2000"))
    # Charge power — only if entity is available
    pwr_entity = _batt("battery_charge_power", "number_battery_charge_power", "")
    if pwr_entity:
        ops.append(ha_set_number(pwr_entity, charge_power_w))
    # Force charge switch
    force_entity = _batt("battery_force_charge", "switch_battery_force_charge", "")
    if force_entity:
        ops.append(ha_switch(force_entity, True))
    ok = all(ops)
    log.info(f"  Battery → charge to {target_soc}% @ {charge_power_w}W — {'OK' if ok else 'ERROR'}")
    return ok

def set_battery_self_consumption(min_soc: int | None = None) -> bool:
    # When no explicit floor is passed, honour the user's Battery Health Mode.
    # Previous default of hardcoded 20 silently shadowed bill_reducer (→ 10)
    # and battery_guard (→ 25), so the post-peak release was always at 20
    # regardless of what the user picked on the Tweak page (#7, andredp).
    if min_soc is None:
        min_soc = _health_mode_limits()[0]
    # Reset both floors that set_battery_charge_target raised during valley.
    # battery_backup_soc was previously left at the valley target, which on GoodWe
    # maps to goodwe_battery_soc_protection and locked the battery above min_soc
    # all peak (#7).
    ops = [
        ha_set_number(_batt("battery_cutoff_soc",  "number_battery_charge_cutoff", "number.battery_grid_charge_cutoff_soc"), min_soc),
        ha_set_number(_batt("battery_backup_soc",  "number_battery_backup_soc",   ""), min_soc),
    ]
    mode_entity = _batt("battery_mode_select", "select_battery_mode", "")
    if mode_entity:
        ops.append(ha_set_select(mode_entity, "maximise_self_consumption"))
    ok = all(ops)
    # Disengage force charge if entity is set
    force_entity = _batt("battery_force_charge", "switch_battery_force_charge", "")
    if force_entity:
        ha_switch(force_entity, False)
    log.info(f"  Battery → self-consumption (min {min_soc}%) — {'OK' if ok else 'ERROR'}")
    return ok

# ── Consumption estimation & optimal SOC ─────────────────────────────────────
_consumption_cache: dict = {"kw": 0.5, "updated": None}

def _refresh_consumption_cache():
    """Refresh the average-night-house-consumption baseline.

    Pre-#8 this used `abs(grid_power)` only, which is silently wrong for
    self-consumption installs: when the battery covers the night load the
    grid measurement collapses to ~0 W and the baseline goes near zero,
    falsely shrinking the smart-target valley-charge plan. Reported by
    @andredp.

    The correct quantity is total house consumption:

        house = solar + battery_discharge − grid_export
              = solar_w + batt_w  (positive = discharging)
                                  − grid_w (positive = exporting)

    All three series get resampled to a 15-min grid, aligned, filtered to
    the night window [22:00, 08:00) and averaged over their positive
    values. Falls back to the legacy `abs(grid)` calculation if solar or
    battery_power is not wired (grid-only setups).
    """
    global _consumption_cache

    grid_entity = _wiz("grid_power",     "sensor_grid_power",    "")
    solar_entity = _wiz("solar_power",   "sensor_solar_power",   "")
    batt_entity  = _wiz("battery_power", "sensor_battery_power", "")

    if not grid_entity:
        log.warning("  Consumption cache: no grid_power entity wired — skip")
        return

    grid_rows = _load_history_best_effort(grid_entity, 14)

    house_kw = None
    if solar_entity and batt_entity and grid_rows:
        try:
            solar_rows = _load_history_best_effort(solar_entity, 14)
            batt_rows  = _load_history_best_effort(batt_entity,  14)
            if solar_rows and batt_rows:
                g = _rows_to_15min_series(grid_rows,  "grid")
                s = _rows_to_15min_series(solar_rows, "solar")
                b = _rows_to_15min_series(batt_rows,  "batt")
                if bool(_WIZARD.get("grid_flip")):
                    g = -g
                df = pd.concat([g, s, b], axis=1).dropna()
                # In UTC; align to local hours via index offset isn't worth the
                # complexity here — the 22-08 window is a daily band that picks
                # up the same physical hours regardless of tz drift up to 1 h.
                df["house"] = df["solar"] + df["batt"] - df["grid"]
                df["hour"]  = df.index.hour
                night = df[((df["hour"] >= 22) | (df["hour"] < 8)) & (df["house"] > 0)]
                if len(night) >= 20:
                    house_kw = float(night["house"].mean()) / 1000.0
                    log.info(f"  Consumption cache (house = solar + batt − grid): "
                             f"{house_kw:.3f} kW avg night ({len(night)} samples)")
        except Exception as e:
            log.warning(f"  House-balance baseline failed ({e}); falling back to |grid|")

    if house_kw is None:
        # Legacy fallback for installs whose wizard does not have solar/battery
        # wired (pure-grid setups). Preserves prior behaviour exactly.
        night_watts = []
        for row in grid_rows:
            try:
                ts = datetime.fromisoformat(row["last_changed"].replace("Z", "+00:00"))
                ts_local = ts.astimezone().replace(tzinfo=None)
                val = float(row["state"])
                if ts_local.hour >= 22 or ts_local.hour < 8:
                    night_watts.append(abs(val))
            except (KeyError, ValueError, TypeError):
                continue
        house_kw = (sum(night_watts) / len(night_watts) / 1000) if len(night_watts) >= 20 else 0.5
        log.info(f"  Consumption cache (|grid| fallback): {house_kw:.3f} kW avg "
                 f"night ({len(night_watts)} grid-only samples)")

    _consumption_cache = {"kw": round(house_kw, 3), "updated": datetime.now()}

def _get_avg_night_consumption_kw() -> float:
    now = datetime.now()
    if (_consumption_cache["updated"] is None or
            (now - _consumption_cache["updated"]).total_seconds() > 3600):
        threading.Thread(target=_refresh_consumption_cache, daemon=True).start()
    return _consumption_cache["kw"]

def _explain(what: str, why: str,
             inputs: list[dict] | None = None,
             formula: str = "",
             calculation: str = "",
             alternatives_rejected: list[dict] | None = None) -> dict:
    """Build a structured `explanation` dict for the Decisions tab in the UI.

    Every `decide_*` function attaches one of these to its return value so the
    UI can render the full reasoning (inputs used, formula applied, alternatives
    considered) instead of a single-line summary. The legacy `reason` string is
    kept on every decision for back-compat with the saved decisions.json history.

    Conventions:
      what — one short sentence: what the engine decided to do
      why  — one short sentence: the single most-load-bearing reason
      inputs — list of `{label, value}` dicts, in causal order
      formula — symbolic representation (LaTeX-free, plain ASCII)
      calculation — the formula with the actual numbers substituted in
      alternatives_rejected — list of `{option, why}` for paths NOT taken
    """
    return {
        "what":                what,
        "why":                 why,
        "inputs":              inputs or [],
        "formula":             formula,
        "calculation":         calculation,
        "alternatives_rejected": alternatives_rejected or [],
    }


def calculate_optimal_soc(sensors: dict, tariff: dict | None = None) -> dict:
    """Target SOC for valley charging — store cheap electricity to cover tomorrow's peak demand.

    Strategy: importing during valley tariff is cheap; what matters is avoiding
    grid import during the user's PEAK hours. The previous implementation hard-
    coded `peak_hours = 8` (Spanish 2.0TD) which under- or over-estimates by a
    constant factor on every non-Spanish tariff. Bug surfaced by @Karplyak in
    issue #5: with a Ukrainian 16h-peak tariff the target came out at 36 %
    when the rational answer was ~70 %.

    Formula (scaled to actual tariff geometry):
        peak_h_count     = len(tariff["peak_hours"])
        peak_base_kwh    = base_load_kw × peak_h_count
        peak_solar_share = overlap(peak_hours, sunlight_window) / sunlight_window_hours
        solar_during_peak = solar_tomorrow × peak_solar_share
        battery_needed   = peak_base_kwh + temp_adj − solar_during_peak

    Both temperature adjustment and solar share now scale with peak_h_count so
    16h-peak installs get appropriately larger targets, and short-peak markets
    (Australia, some German tariffs) get correspondingly smaller ones.
    """
    battery_kwh  = float(cfg("battery_capacity_kwh", 10.0))

    # Tariff geometry — read once, used to scale every per-hour quantity below.
    if tariff is None:
        tariff = current_tariff()
    peak_hours_cfg = tariff.get("peak_hours", DEFAULT_TARIFF["peak_hours"])
    peak_h_count   = max(1, len(peak_hours_cfg))   # avoid div-by-zero on empty tariff
    peak_h_ratio   = peak_h_count / 8.0            # 1.0 for Spanish default, 2.0 for Karplyak

    # ── Solar (terrain-corrected) ────────────────────────────────────────────
    correction   = _compute_solar_correction_factor()
    solar_tm_raw = sensors.get("solar_tomorrow", 0)
    solar_tm     = round(solar_tm_raw * correction, 2)
    # Compute the fraction of useful daylight (7-19h, 12 hours) that falls
    # inside this user's peak window. Drops to ~0 for a tariff with peak only
    # 18-22 (almost no sun overlap); rises to ~1 for a tariff with peak 07-22
    # (almost the whole sun window is peak). Replaces the old hardcoded 0.45.
    SUN_WINDOW = range(7, 19)
    peak_sun_overlap   = sum(1 for h in peak_hours_cfg if h in SUN_WINDOW)
    peak_solar_share   = peak_sun_overlap / len(SUN_WINDOW)
    solar_during_peak  = round(solar_tm * peak_solar_share, 2)

    # ── Base consumption (kW) — night avg is the best proxy for base household load ──
    base_kw = _get_avg_night_consumption_kw()
    peak_base_kwh = round(base_kw * peak_h_count, 2)

    # ── Temperature correction (HVAC load) ──────────────────────────────────
    # The bands below describe an extra kWh figure that was tuned for an 8h
    # Spanish peak. Scale by peak_h_ratio so longer peaks get proportionally
    # larger adjustments. Cooling in summer follows the same scaling for
    # symmetry, even though hot-climate inverter A/C COP behaviour differs;
    # users on extreme tariffs can override via wizard advanced settings.
    temp = sensors.get("temp_outdoor", 15.0)
    if temp < 5:
        temp_adj_kwh = 3.0 * peak_h_ratio   # heat pump near full power all peak hours
    elif temp < 10:
        temp_adj_kwh = 2.0 * peak_h_ratio
    elif temp < 15:
        temp_adj_kwh = 1.0 * peak_h_ratio
    elif temp > 30:
        temp_adj_kwh = 1.5 * peak_h_ratio
    elif temp > 25:
        temp_adj_kwh = 0.5 * peak_h_ratio
    else:
        temp_adj_kwh = 0.0
    temp_adj_kwh = round(temp_adj_kwh, 2)

    peak_total_kwh = peak_base_kwh + temp_adj_kwh

    # ── Battery gap (what solar doesn't cover during peak) ───────────────────
    battery_needed = max(0.0, peak_total_kwh - solar_during_peak)
    soc_needed = (battery_needed / battery_kwh) * 100

    # +5 % safety buffer, then clamp to the user's health-mode SOC limits.
    # The floor is the configured health-mode minimum — that is the value the
    # user explicitly chose on the Tweak page and represents their own risk
    # boundary. The pre-#8 code added a hardcoded 30 % paternalistic floor on
    # top, which silently overrode bill_reducer (→10) for any user who wanted
    # to use the full battery capacity. Reported by @andredp.
    _hmin, _hmax = _health_mode_limits()
    target = max(_hmin, min(_hmax, round(soc_needed + 5)))

    return {
        "target_soc":         target,
        "needed_kwh":         round(battery_needed, 2),
        "peak_total_kwh":     round(peak_total_kwh, 2),
        "peak_h_count":       peak_h_count,
        "peak_solar_share":   round(peak_solar_share, 3),
        "solar_during_peak":  solar_during_peak,
        "solar_tomorrow":     solar_tm,
        "solar_tomorrow_raw": solar_tm_raw,
        "solar_correction":   correction,
        "base_kw":            base_kw,
        "temp_outdoor":       round(temp, 1),
        "temp_adj_kwh":       temp_adj_kwh,
        "health_mode_min":    _hmin,
        "health_mode_max":    _hmax,
    }

# ── Storm detection (AEMET OpenData) ─────────────────────────────────────────
STORM_CONDITIONS = {"lightning", "lightning-rainy", "exceptional", "hail", "pouring"}

def is_storm_forecast() -> bool:
    entity = cfg("sensor_weather", "")
    s = ha_state(entity)
    if not s:
        return False
    if s.get("state", "").lower() in STORM_CONDITIONS:
        return True
    for entry in s.get("attributes", {}).get("forecast", [])[:8]:
        if entry.get("condition", "").lower() in STORM_CONDITIONS:
            return True
    return False

# ── ML Model — dynamic feature engineering ───────────────────────────────────
#
# Feature set is built at training time from wizard-configured sensors.
# The exact feature list is saved in the model artifact and reused at inference.
# Adding a new sensor in the wizard triggers a retrain (MODEL_FEATURE_VER bump
# alone forces retrain on the next cycle that loads a stale artifact).
#
# Base features (always present):
#   hour, weekday, month       — calendar position
#   lag1, lag4, roll4          — SOC autoregressive lags at 15-min intervals
#   solar_proxy                — geometric solar elevation proxy (0–1)
#
# Dynamic features (added when wizard sensors are configured):
#   temp_outdoor               — outdoor temperature → HVAC load signal
#   solar_lag1, solar_roll4    — short-term solar production lags
#   grid_lag1, grid_roll4      — grid import magnitude lags (unsigned)
#   sm_<name>                  — one feature per configured sub-meter

BASE_FEATURES = ["hour", "weekday", "month", "lag1", "lag4", "roll4", "solar_proxy"]

# ── History helpers ───────────────────────────────────────────────────────────

def _influx_wizard_history(entity: str, days: int, influx_cfg: dict) -> list:
    """Fetch entity history from InfluxDB v1 (InfluxQL) configured in the wizard.
    The official HA InfluxDB integration uses v1 user/password auth.

    HA→Influx integration stores rows with measurement = unit (`%`, `W`, `kWh`,
    `°C`) and entity_id as a TAG without the domain prefix. Querying
    `FROM "<entity_short>"` returns nothing — must scan all measurements with
    `FROM /.*/` and filter by the tag, same pattern used in `ha_history_influx`.

    Returns HA-style rows: [{"last_changed": iso_str, "state": str_value}, ...]"""
    try:
        url  = f"http://{influx_cfg['host']}:{influx_cfg.get('port', 8086)}"
        db   = influx_cfg.get("db", "homeassistant")
        user = influx_cfg.get("username", "")
        pwd  = influx_cfg.get("password", "")
        entity_short = entity.split(".")[-1] if "." in entity else entity
        q = (
            f'SELECT mean("value") AS "value" FROM /.*/ '
            f"WHERE \"entity_id\" = '{entity_short}' "
            f'AND time > now()-{days}d '
            f'GROUP BY time(15m) fill(previous)'
        )
        resp, err, _ = _influx_query(url, db, q, user, pwd)
        if err or not resp:
            return []
        rows = []
        # Multiple measurements may match (same entity_id reported under several
        # units, e.g. a power sensor that also has kWh). Concat all series; the
        # 15-min resampler downstream will collapse duplicate timestamps.
        for series in resp.json().get("results", [{}])[0].get("series", []):
            cols = series.get("columns", [])
            try:
                ti = cols.index("time")
                vi = cols.index("value")
            except ValueError:
                continue
            for point in series.get("values", []):
                ts, val = point[ti], point[vi]
                if val is None:
                    continue
                rows.append({"last_changed": ts, "state": str(val)})
        return rows
    except Exception as e:
        log.debug(f"InfluxDB wizard history {entity}: {e}")
        return []

def _mariadb_wizard_history(entity: str, days: int, mariadb_cfg: dict) -> list:
    """Fetch entity history directly from a HA Recorder running on MariaDB/MySQL.

    Bypasses HA REST API for users whose /api/history endpoint returns empty
    (issue #2). Compatible with HA recorder schemas before and after the
    states_meta migration (HA 2023.4+).
    Returns HA-style rows: [{"last_changed": iso_str, "state": str}, ...]"""
    try:
        import pymysql
    except ImportError:
        log.warning("  pymysql not installed — MariaDB direct access unavailable")
        return []
    from datetime import timezone
    try:
        conn = pymysql.connect(
            host=mariadb_cfg["host"],
            port=int(mariadb_cfg.get("port", 3306)),
            user=mariadb_cfg.get("username", ""),
            password=mariadb_cfg.get("password", ""),
            database=mariadb_cfg.get("db", "homeassistant"),
            connect_timeout=10,
            read_timeout=60,
            charset="utf8mb4",
        )
    except Exception as e:
        log.debug(f"MariaDB connect {entity}: {e}")
        return []
    try:
        with conn.cursor() as cur:
            # Detect post-2023.4 schema (states_meta JOIN) vs legacy (entity_id in states)
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_name = 'states_meta'"
            )
            has_meta = cur.fetchone()[0] > 0
            # NOTE: HA recorder writes `last_changed_ts` only when the state
            # actually changes; for sensors that emit the same value repeatedly
            # (or when only attributes update) it stays NULL, and the timestamp
            # filter below silently drops those rows. `last_updated_ts` is
            # populated on every write and is what we want for history queries.
            # Reported as a fix by @Karplyak in issue #2.
            if has_meta:
                q = (
                    "SELECT s.state, s.last_updated_ts "
                    "FROM states s JOIN states_meta sm ON s.metadata_id = sm.metadata_id "
                    "WHERE sm.entity_id = %s "
                    "AND s.last_updated_ts > UNIX_TIMESTAMP(NOW() - INTERVAL %s DAY) "
                    "AND s.state NOT IN ('unknown', 'unavailable', '') "
                    "ORDER BY s.last_updated_ts ASC"
                )
            else:
                q = (
                    "SELECT state, last_updated FROM states "
                    "WHERE entity_id = %s "
                    "AND last_updated > NOW() - INTERVAL %s DAY "
                    "AND state NOT IN ('unknown', 'unavailable', '') "
                    "ORDER BY last_updated ASC"
                )
            cur.execute(q, (entity, days))
            raw = cur.fetchall()
        rows = []
        for state, ts in raw:
            if isinstance(ts, (int, float)):
                iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            elif isinstance(ts, datetime):
                iso = ts.isoformat() if ts.tzinfo else ts.replace(tzinfo=timezone.utc).isoformat()
            else:
                iso = str(ts)
            rows.append({"last_changed": iso, "state": str(state)})
        return rows
    except Exception as e:
        log.debug(f"MariaDB query {entity}: {e}")
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _load_history_best_effort(entity: str, days: int) -> list:
    """Unified history loader: wizard InfluxDB v1 → legacy InfluxDB v1 cfg
    → wizard MariaDB direct → HA recorder REST.
    Always returns HA-style rows; empty list if nothing is available."""
    # 1. Wizard InfluxDB (v1 user/password, configured in wizard Data Sources step)
    influx_cfg = _WIZARD.get("influxdb", {})
    if influx_cfg.get("host"):
        rows = _influx_wizard_history(entity, days, influx_cfg)
        if rows:
            log.debug(f"  [{entity}] {len(rows)} rows from wizard InfluxDB ({days}d)")
            return rows
    # 2. Legacy InfluxDB v1 (cfg-based, old add-on config)
    if cfg("influxdb_url", ""):
        rows, _ = ha_history_influx(entity, days=min(days, 365))
        if rows:
            log.debug(f"  [{entity}] {len(rows)} rows from InfluxDB v1 cfg")
            return rows
    # 3. Wizard MariaDB direct (bypasses HA REST for recorder-on-MariaDB setups)
    mariadb_cfg = _WIZARD.get("mariadb", {})
    if mariadb_cfg.get("host"):
        rows = _mariadb_wizard_history(entity, days, mariadb_cfg)
        if rows:
            log.debug(f"  [{entity}] {len(rows)} rows from wizard MariaDB ({days}d)")
            return rows
    # 4. HA recorder REST (capped at 14 days — typical recorder retention)
    rows = ha_history(entity, days=min(days, 14))
    log.debug(f"  [{entity}] {len(rows)} rows from HA recorder")
    return rows

def _rows_to_15min_series(rows: list, col: str) -> "pd.Series":
    """Convert HA-style history rows to a 15-min resampled, forward-filled Series."""
    records = []
    for row in rows:
        try:
            lc = row["last_changed"]
            if isinstance(lc, str):
                ts = datetime.fromisoformat(lc.replace("Z", "+00:00"))
            elif isinstance(lc, (int, float)):
                # InfluxDB path returns Unix timestamp. _influx_query asks for
                # epoch=ms, but we accept seconds and nanoseconds too so this
                # function stays correct if the upstream epoch ever changes.
                if lc > 1e15:     # nanoseconds (~year 33658 in s)
                    ts_val = lc / 1e9
                elif lc > 1e11:   # milliseconds (~year 5138 in s)
                    ts_val = lc / 1e3
                else:             # seconds
                    ts_val = lc
                ts = datetime.fromtimestamp(ts_val, tz=timezone.utc)
            else:
                continue
            val = float(row["state"])
            records.append({"ts": ts.replace(tzinfo=None), "value": val})
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
    if not records:
        return pd.Series(dtype=float, name=col)
    s = pd.DataFrame(records).set_index("ts")["value"]
    return s.resample("15min").mean().ffill().rename(col)

# ── Dynamic training DataFrame ────────────────────────────────────────────────

def _build_training_df(days: int = 90) -> "tuple[pd.DataFrame, list[str]] | tuple[None, None]":
    """Build a training DataFrame joining all wizard-configured sensor histories.

    Returns (df, feature_names) or (None, None) if insufficient data.
    The feature_names list is stored in the model artifact so predict_soc()
    can reconstruct the exact same feature vector at inference time.
    """
    # ── Primary target: battery SOC ──────────────────────────────────────────
    batt_entity = _wiz("battery_soc", "sensor_battery_soc", "sensor.battery_state_of_capacity")
    batt_rows   = _load_history_best_effort(batt_entity, days)
    if not batt_rows:
        log.warning("  No battery SOC history found — training aborted")
        return None, None

    soc_series = _rows_to_15min_series(batt_rows, "value")
    if len(soc_series) < 48:
        log.warning(f"  SOC series too short ({len(soc_series)} pts) — training aborted")
        return None, None

    # ── Base DataFrame (SOC + time features + autoregressive lags) ───────────
    df = soc_series.to_frame("value")
    df["hour"]        = df.index.hour
    df["weekday"]     = df.index.weekday
    df["month"]       = df.index.month
    df["lag1"]        = df["value"].shift(1)
    df["lag4"]        = df["value"].shift(4)
    df["roll4"]       = df["value"].rolling(4).mean()
    df["solar_proxy"] = [_solar_proxy_for_hour(ts.hour, ts.month) for ts in df.index]
    features: list[str] = list(BASE_FEATURES)  # copy

    # ── Optional sensor enrichment ────────────────────────────────────────────
    wiz_sensors = _WIZARD.get("sensors", {})

    # Outdoor temperature → accounts for HVAC-driven SOC swings
    temp_entity = _wiz("temp_outdoor", "sensor_temp_outdoor",
                       "")
    if wiz_sensors.get("temp_outdoor") or cfg("sensor_temp_outdoor", ""):
        temp_rows = _load_history_best_effort(temp_entity, days)
        if temp_rows:
            s = _rows_to_15min_series(temp_rows, "temp_outdoor")
            df = df.join(s, how="left")
            df["temp_outdoor"] = df["temp_outdoor"].ffill().fillna(15.0)
            features.append("temp_outdoor")
            log.info(f"  Feature added: temp_outdoor ({len(temp_rows)} rows)")

    # Solar production lags → captures daytime energy availability
    solar_entity = _wiz("solar_power", "sensor_solar_power", "")
    if wiz_sensors.get("solar_power") or cfg("sensor_solar_power", ""):
        solar_rows = _load_history_best_effort(solar_entity, days)
        if solar_rows:
            s = _rows_to_15min_series(solar_rows, "solar")
            df = df.join(s, how="left")
            df["solar"] = df["solar"].ffill().fillna(0.0)
            df["solar_lag1"]  = df["solar"].shift(1)
            df["solar_roll4"] = df["solar"].rolling(4).mean()
            features.extend(["solar_lag1", "solar_roll4"])
            log.info(f"  Features added: solar_lag1, solar_roll4 ({len(solar_rows)} rows)")

    # Grid power lags (unsigned magnitude) → captures household consumption pattern
    grid_entity = _wiz("grid_power", "sensor_grid_power", "")
    if wiz_sensors.get("grid_power") or cfg("sensor_grid_power", ""):
        grid_rows = _load_history_best_effort(grid_entity, days)
        if grid_rows:
            s = _rows_to_15min_series(grid_rows, "grid")
            df = df.join(s, how="left")
            df["grid"] = df["grid"].ffill().fillna(0.0).abs()
            df["grid_lag1"]  = df["grid"].shift(1)
            df["grid_roll4"] = df["grid"].rolling(4).mean()
            features.extend(["grid_lag1", "grid_roll4"])
            log.info(f"  Features added: grid_lag1, grid_roll4 ({len(grid_rows)} rows)")

    # Sub-meter features — one per configured extra power sensor
    for sm in _WIZARD.get("grid_submeters", []):
        sm_name   = sm.get("name", "").strip().lower().replace(" ", "_").replace("-", "_")
        sm_entity = sm.get("entity", "")
        if not (sm_name and sm_entity):
            continue
        feat = f"sm_{sm_name}"
        sm_rows = _load_history_best_effort(sm_entity, days)
        if sm_rows:
            s = _rows_to_15min_series(sm_rows, feat)
            df = df.join(s, how="left")
            df[feat] = df[feat].ffill().fillna(0.0).abs()
            features.append(feat)
            log.info(f"  Feature added: {feat} ({len(sm_rows)} rows)")

    df = df.dropna(subset=features + ["value"])
    if len(df) < 48:
        log.warning(f"  After join/dropna: only {len(df)} rows — training aborted")
        return None, None

    log.info(f"  Training set: {len(df)} rows × {len(features)} features — {features}")
    return df, features

# ── Training & prediction ─────────────────────────────────────────────────────

def train_model() -> bool:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    import numpy as np

    log.info("═══ ML Training started (dynamic features v3) ═══")

    # Days of history to request — more data available when InfluxDB is configured
    influx_cfg = _WIZARD.get("influxdb", {})
    days = 90 if influx_cfg.get("host") else \
           60 if cfg("influxdb_url", "") else 14

    df, features = _build_training_df(days=days)
    if df is None:
        log.warning("  Insufficient training data — staying in rules-only mode")
        return False

    X, y = df[features], df["value"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr",    GradientBoostingRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.07, subsample=0.8,
            min_samples_leaf=5, random_state=42,
        )),
    ])
    scores = cross_val_score(pipe, X, y, cv=min(5, len(df) // 96), scoring="r2")
    pipe.fit(X, y)

    artifact = {
        "pipeline":        pipe,
        "features":        features,
        "trained_at":      datetime.now().isoformat(),
        "r2_cv_mean":      float(np.mean(scores)),
        "r2_cv_std":       float(np.std(scores)),
        "n_samples":       len(df),
        "days_of_data":    days,
        "feature_version": MODEL_FEATURE_VER,
        "wizard_sensors":  list(_WIZARD.get("sensors", {}).keys()),
        "submeters":       [sm.get("name","") for sm in _WIZARD.get("grid_submeters", [])],
    }
    joblib.dump(artifact, MODEL_FILE)
    log.info(f"  Model saved: {len(df)} samples · R²={np.mean(scores):.3f}±{np.std(scores):.3f} · {len(features)} features")
    return True

def predict_soc(sensors: dict) -> dict:
    if MODEL_FILE.exists():
        try:
            art = joblib.load(MODEL_FILE)
            if art.get("feature_version", 1) != MODEL_FEATURE_VER:
                log.info("  Feature version mismatch — triggering background retrain")
                threading.Thread(target=train_model, daemon=True).start()
                raise ValueError("stale model")

            features = art.get("features", BASE_FEATURES)
            now      = datetime.now()

            # Build the feature row — include all possible features,
            # then select only those the trained model expects.
            row: dict = {
                "hour":        now.hour,
                "weekday":     now.weekday(),
                "month":       now.month,
                "lag1":        sensors.get("battery_soc", 50.0),
                "lag4":        sensors.get("battery_soc", 50.0),
                "roll4":       sensors.get("battery_soc", 50.0),
                "solar_proxy": _solar_proxy_for_hour(now.hour, now.month),
                # Dynamic — default to live sensor reading or neutral fallback
                "temp_outdoor": sensors.get("temp_outdoor", 15.0),
                "solar_lag1":   max(0, sensors.get("solar_power", 0)),
                "solar_roll4":  max(0, sensors.get("solar_power", 0)),
                "grid_lag1":    abs(sensors.get("grid_power", 0)),
                "grid_roll4":   abs(sensors.get("grid_power", 0)),
            }
            # Submeter features from live readings
            for sm in _WIZARD.get("grid_submeters", []):
                name = sm.get("name", "").strip().lower().replace(" ", "_").replace("-", "_")
                feat = f"sm_{name}"
                if feat in features:
                    row[feat] = abs(sensors.get(f"submeter_{sm.get('name','')}", 0))

            X    = pd.DataFrame([{f: row.get(f, 0.0) for f in features}])
            pred = float(art["pipeline"].predict(X)[0])
            return {
                "predicted_soc":  round(max(0.0, min(100.0, pred)), 1),
                "method":         "ml",
                "r2":             art.get("r2_cv_mean"),
                "r2_std":         art.get("r2_cv_std"),
                "trained_at":     art.get("trained_at"),
                "features_used":  features,
                "n_features":     len(features),
            }
        except Exception as e:
            log.warning(f"  ML prediction error: {e}")

    # Rules-based fallback
    solar_tm = sensors.get("solar_tomorrow", 0)
    pred = 80.0 if solar_tm < cfg("solar_tomorrow_irrisoria_kwh", 2.0) else 50.0
    return {"predicted_soc": pred, "method": "rules_fallback"}

# ── Heat pump logic ──────────────────────────────────────────────────────────
def _season() -> str:
    """Return 'summer' or 'winter'. Priority: wizard season_select entity →
    month-based fallback. Lets the user flip the seasonal regime from a HA
    helper (input_select.season) when the calendar disagrees with reality —
    notably for radiant cooling installs where the summer regime is opt-in.
    """
    season_entity = _wiz("season_select", "input_select_season", "")
    if season_entity:
        s = ha_str(season_entity, "").strip().lower()
        if s in ("summer", "verano"):
            return "summer"
        if s in ("winter", "invierno"):
            return "winter"
    m = datetime.now().month
    return "summer" if cfg("summer_start_month", 6) <= m <= cfg("summer_end_month", 9) else "winter"

def _is_summer() -> bool:
    return _season() == "summer"

# ── Solar surplus detector with hysteresis ──────────────────────────────────
# Previous version used `soc >= 99 and batt_power > 0` evaluated cycle-by-cycle.
# In practice the SOC sits at 99–100 % for only a few minutes per day around the
# peak — the engine almost never sampled the right cycle window, so radiant
# cooling never engaged via the surplus path. Now we cross at SOC ≥ 95 % with
# the battery no longer charging, hold the state across cycles, and only drop
# out when SOC falls below 90 % OR the battery starts charging hard again.
# State persists across addon restarts to avoid losing context on rebuild.
_surplus_state: dict = {"active": False, "since": None, "last_soc": None, "last_pwr": None}

def _load_surplus_state():
    global _surplus_state
    if SURPLUS_STATE_FILE.exists():
        try:
            _surplus_state = json.loads(SURPLUS_STATE_FILE.read_text())
        except Exception:
            pass

def _save_surplus_state():
    try:
        SURPLUS_STATE_FILE.write_text(json.dumps(_surplus_state))
    except Exception as e:
        log.warning(f"  Could not persist surplus_state.json: {e}")

def _has_surplus_power(soc: float, batt_power_w: float) -> bool:
    """Return True if there's enough solar surplus to spend on opportunistic
    loads (radiant cooling, heat pump pre-heating, etc.) with hysteresis.

    Tunables (cfg() keys):
        surplus_enter_soc      – default 95 %, SOC needed to enter surplus
        surplus_exit_soc       – default 90 %, SOC drop that ends surplus
        surplus_exit_charge_w  – default -500 W, hard-charging threshold that
                                  also ends surplus (battery soaking up solar
                                  again means there isn't really surplus).
    """
    global _surplus_state
    enter_soc = float(cfg("surplus_enter_soc", 95.0))
    exit_soc  = float(cfg("surplus_exit_soc",  90.0))
    exit_pwr  = float(cfg("surplus_exit_charge_w", -500.0))

    active = bool(_surplus_state.get("active", False))
    now_iso = datetime.now().isoformat()

    if active:
        # Stay active while SOC isn't depleted AND we're not charging hard.
        stay = (soc >= exit_soc) and (batt_power_w >= exit_pwr)
        if not stay:
            _surplus_state = {"active": False, "since": now_iso,
                              "last_soc": soc, "last_pwr": batt_power_w}
            _save_surplus_state()
            log.info(f"  Surplus → OFF (SOC={soc:.0f}% pwr={batt_power_w:.0f}W "
                     f"thresholds: <{exit_soc:.0f}% or <{exit_pwr:.0f}W)")
            return False
        _surplus_state.update({"last_soc": soc, "last_pwr": batt_power_w})
        return True

    # Inactive — require crossing the entry bar
    enter = (soc >= enter_soc) and (batt_power_w >= 0)
    if enter:
        _surplus_state = {"active": True, "since": now_iso,
                          "last_soc": soc, "last_pwr": batt_power_w}
        _save_surplus_state()
        log.info(f"  Surplus → ON (SOC={soc:.0f}% pwr={batt_power_w:.0f}W "
                 f"thresholds: ≥{enter_soc:.0f}% and ≥0W)")
        return True
    return False

# ── Free-cooling gate state (v5.0.8) ────────────────────────────────────────
# Holds the last "open the windows" notify timestamp so we don't spam Telegram
# every 15-min cycle while the conditions persist.
_cooling_gate_state: dict = {"last_open_windows_notify": None}

def _load_cooling_gate_state():
    global _cooling_gate_state
    if COOLING_GATE_STATE_FILE.exists():
        try:
            _cooling_gate_state = json.loads(COOLING_GATE_STATE_FILE.read_text())
        except Exception:
            pass

def _save_cooling_gate_state():
    try:
        COOLING_GATE_STATE_FILE.write_text(json.dumps(_cooling_gate_state))
    except Exception as e:
        log.warning(f"  Could not persist cooling_gate_state.json: {e}")

def _maybe_notify_open_windows(temp_in: float, temp_out: float, zone_name: str) -> None:
    """Send a 'open the windows' Telegram alert, rate-limited to one per
    `notify_open_windows_cooldown_h` hours (default 4h)."""
    if not cfg("notify_open_windows_enabled", True):
        return
    cooldown_h = float(cfg("notify_open_windows_cooldown_h", 4.0))
    now = datetime.now()
    last_iso = _cooling_gate_state.get("last_open_windows_notify")
    if last_iso:
        try:
            last = datetime.fromisoformat(last_iso)
            if (now - last).total_seconds() < cooldown_h * 3600:
                return
        except Exception:
            pass
    msg = (f"🌬️ *Free-cooling disponible* — {zone_name}\n"
           f"Casa a {temp_in:.1f}°C, fuera {temp_out:.1f}°C.\n"
           f"Abre ventanas — la aerotermia no se enciende.")
    send_telegram_alert(msg)
    _cooling_gate_state["last_open_windows_notify"] = now.isoformat()
    _save_cooling_gate_state()

def _hp_zone_decision(sensors: dict, zone: dict, zone_idx: int) -> dict:
    soc        = sensors["battery_soc"]
    batt_power = sensors["battery_power"]
    summer     = _is_summer()
    free_power = _has_surplus_power(soc, batt_power)
    temp_in    = sensors.get(f"temp_zone_{zone_idx}", sensors.get("temp_indoor", 20.0))
    temp_out_raw = sensors.get("temp_outdoor")
    temp_out   = float(temp_out_raw) if temp_out_raw is not None else None
    hour       = datetime.now().hour
    sched      = zone.get("sched") or {}
    sched_mode = sched.get(str(hour), sched.get(hour, "comfort"))
    t_comfort  = float(zone.get("temp_comfort", 21.0))
    t_surplus  = float(zone.get("temp_surplus", 23.0))
    t_min      = float(zone.get("temp_min",     18.0))
    zone_name  = zone.get("name", f"Zone {zone_idx + 1}")
    skip       = False
    target     = None
    reason     = ""

    if summer:
        entity = (zone.get("temp_cool") or
                  _wiz("hvac_temp_cool", "number_hvac_cool", ""))
        # Radiant-cooling regime: the floor stores cold poorly compared to a
        # split AC, so running it overnight/valley wastes energy. Default is
        # NOT to act unless (a) we have a clear solar surplus, or (b) the
        # house has crossed the configurable thermal override threshold.
        override_c = float(cfg("hvac_summer_override_c", 29.0))
        # Free-cooling gates (v5.0.8). When the outdoor air is already cold
        # enough, opening windows beats spending energy on the heat pump.
        # If `sensor_temp_outdoor` isn't wired, temp_out is None and these
        # gates are inert — behaviour matches v5.0.7.
        cooling_outdoor_max_c = float(cfg("cooling_outdoor_max_c", 22.0))
        free_cooling_delta_c  = float(cfg("free_cooling_delta_c",  2.0))
        outdoor_known       = temp_out is not None
        outdoor_cool_enough = outdoor_known and temp_out <= cooling_outdoor_max_c
        outdoor_cooler_than_indoor = (
            outdoor_known and temp_out <= (temp_in - free_cooling_delta_c)
        )
        # Forecast-nocturno gate (v5.0.13). If AEMET predicts a cool night
        # (default ≤ 18 °C, Guadarrama typical), don't burn surplus on
        # cooling — the house will free-cool overnight via open windows +
        # radiant-floor inertia. Inert when sensor not mapped (value None
        # or 0 / placeholder values dismissed).
        night_low_raw = sensors.get("aemet_night_low")
        night_low = float(night_low_raw) if night_low_raw not in (None, 0, 0.0) else None
        night_low_threshold = float(cfg("cooling_skip_if_night_low_c", 18.0))
        night_will_be_cool  = night_low is not None and night_low <= night_low_threshold
        if temp_in > override_c and outdoor_cooler_than_indoor:
            skip = True
            reason = (f"Free-cooling override: house {temp_in:.1f}°C, "
                      f"outdoor {temp_out:.1f}°C → open windows [{zone_name}]")
            log.info(f"  [COOLING-GATE] {reason}")
            _maybe_notify_open_windows(temp_in, temp_out, zone_name)
        elif temp_in > override_c:
            target = max(16.0, t_comfort - 2)
            reason = f"Thermal override ({temp_in:.1f}°C > {override_c:.1f}°C) [{zone_name}]"
        elif (free_power or sched_mode == "surplus") and outdoor_cool_enough:
            skip = True
            reason = (f"Free-cooling surplus: outdoor {temp_out:.1f}°C ≤ "
                      f"{cooling_outdoor_max_c:.1f}°C, save surplus for battery "
                      f"[{zone_name}]")
            log.info(f"  [COOLING-GATE] {reason}")
        elif (free_power or sched_mode == "surplus") and night_will_be_cool:
            skip = True
            reason = (f"Cool night forecast ({night_low:.1f}°C ≤ "
                      f"{night_low_threshold:.1f}°C): skip cooling, the house "
                      f"will free-cool tonight [{zone_name}]")
            log.info(f"  [COOLING-GATE] {reason}")
        elif free_power or sched_mode == "surplus":
            target = max(16.0, t_comfort - 3)
            reason = f"Surplus cooling [{zone_name}]"
        else:
            skip = True
            reason = f"Inactive (no surplus, {temp_in:.1f}°C ≤ {override_c:.1f}°C) [{zone_name}]"
    else:
        entity = (zone.get("temp_heat") or
                  _wiz("hvac_temp_heat", "number_hvac_heat", ""))
        if free_power or sched_mode == "surplus":
            target, reason = t_surplus, f"Surplus heating [{zone_name}]"
        elif sched_mode == "comfort":
            target, reason = t_comfort, f"Comfort heating ({temp_in:.1f}°C) [{zone_name}]"
        else:
            target, reason = t_min, f"Minimum heating [{zone_name}]"

    return {"entity": entity, "climate": zone.get("climate") or None,
            "target": None if skip else round(target, 1),
            "skip": skip,
            "reason": reason, "zone": zone_name,
            "sched_mode": sched_mode, "temp_indoor": round(temp_in, 1),
            "season": "summer" if summer else "winter"}

def _hp_legacy_decision(sensors: dict) -> dict:
    soc        = sensors["battery_soc"]
    batt_power = sensors["battery_power"]
    temp_in    = sensors.get("temp_indoor", 20.0)
    temp_out_raw = sensors.get("temp_outdoor")
    temp_out   = float(temp_out_raw) if temp_out_raw is not None else None
    sun        = get_sun_status()
    daytime    = sun["is_day"]
    free_power = _has_surplus_power(soc, batt_power)
    skip       = False
    target     = None
    reason     = ""
    summer     = _is_summer()
    if summer:
        entity = _wiz("hvac_temp_cool", "number_hvac_cool", "")
        override_c = float(cfg("hvac_summer_override_c", 29.0))
        cooling_outdoor_max_c = float(cfg("cooling_outdoor_max_c", 22.0))
        free_cooling_delta_c  = float(cfg("free_cooling_delta_c",  2.0))
        outdoor_known       = temp_out is not None
        outdoor_cool_enough = outdoor_known and temp_out <= cooling_outdoor_max_c
        outdoor_cooler_than_indoor = (
            outdoor_known and temp_out <= (temp_in - free_cooling_delta_c)
        )
        night_low_raw = sensors.get("aemet_night_low")
        night_low = float(night_low_raw) if night_low_raw not in (None, 0, 0.0) else None
        night_low_threshold = float(cfg("cooling_skip_if_night_low_c", 18.0))
        night_will_be_cool  = night_low is not None and night_low <= night_low_threshold
        if temp_in > override_c and outdoor_cooler_than_indoor:
            skip = True
            reason = (f"Free-cooling override: house {temp_in:.1f}°C, "
                      f"outdoor {temp_out:.1f}°C → open windows")
            log.info(f"  [COOLING-GATE] {reason}")
            _maybe_notify_open_windows(temp_in, temp_out, "default")
        elif temp_in > override_c:
            target, reason = 20.0, f"Thermal override ({temp_in:.1f}°C > {override_c:.1f}°C)"
        elif free_power and outdoor_cool_enough:
            skip = True
            reason = (f"Free-cooling surplus: outdoor {temp_out:.1f}°C ≤ "
                      f"{cooling_outdoor_max_c:.1f}°C, save surplus for battery")
            log.info(f"  [COOLING-GATE] {reason}")
        elif free_power and night_will_be_cool:
            skip = True
            reason = (f"Cool night forecast ({night_low:.1f}°C ≤ "
                      f"{night_low_threshold:.1f}°C): skip cooling, the house "
                      f"will free-cool tonight")
            log.info(f"  [COOLING-GATE] {reason}")
        elif free_power:
            target, reason = 16.0, "Solar surplus (hysteresis)"
        else:
            skip = True
            reason = f"Inactive (no surplus, {temp_in:.1f}°C ≤ {override_c:.1f}°C)"
    else:
        entity = _wiz("hvac_temp_heat", "number_hvac_heat", "")
        if free_power:
            target, reason = 18.5, "Solar surplus (hysteresis)"
        elif temp_in < 16 and daytime:
            target, reason = 17.0, f"Indoor too cold ({temp_in:.1f}°C)"
        else:
            target, reason = 16.0, "Winter base setpoint"
    return {"entity": entity, "climate": None,
            "target": None if skip else target,
            "skip": skip,
            "reason": reason,
            "zone": "default", "season": "summer" if summer else "winter"}

def decide_heat_pump(sensors: dict) -> list:
    zones = _WIZARD.get("hvac_zones", [])
    if zones:
        return [_hp_zone_decision(sensors, z, i) for i, z in enumerate(zones)]
    return [_hp_legacy_decision(sensors)]

# ── Battery decision ─────────────────────────────────────────────────────────
def _battery_charge_power_w(default_w: int = 2000) -> int:
    """Resolve the inverter's grid-charge power limit from the wizard's
    `battery_charge_power` entity (typically `number.battery_grid_charge_maximum_power`
    on a Huawei Luna). Falls back to `default_w` if the entity isn't configured or
    reports nothing. Caps emergency charges to the resolved value so we never push
    more than the user's hardware/comfort limit."""
    pwr_entity = _batt("battery_charge_power", "number_battery_charge_power", "")
    if pwr_entity:
        try:
            val = ha_float(pwr_entity)
            if val and val > 100:   # entity is in W; ignore garbage near zero
                return int(val)
        except Exception:
            pass
    return default_w


def decide_battery(sensors: dict, tariff: dict, prediction: dict) -> dict:
    """Decide the battery's action for this cycle. Every threshold and every
    target SOC is resolved against the user's configured `battery_health_mode`
    and `battery_*_threshold` options — previous versions had `30/40/95/20`
    hardcoded which silently overrode Battery Guard / Optimized perfiles."""
    soc       = sensors["battery_soc"]
    period    = tariff["period"]
    prices    = tariff["prices"]
    thr_emerg = cfg("battery_emergency_threshold", 10)
    thr_low   = cfg("battery_low_threshold",       30)
    storm_thr = cfg("battery_storm_threshold",     80)
    hmin, hmax = _health_mode_limits()
    # Emergency / recovery targets are anchored to the health-mode minimum
    # itself. Up until #8 these were `max(30, hmin)` and `max(40, hmin + 10)`
    # which silently overrode bill_reducer (hmin=10): the emergency floor was
    # effectively 30 even though the user explicitly chose 10. Now we honour
    # the user's pick — bill_reducer emergency target = 10, recovery low = 20.
    emerg_target = hmin
    low_target   = hmin + 10
    # Battery "full" threshold for self-consumption hand-off is the user's
    # health-mode ceiling, not a hardcoded 95 %. For Battery Guard (max 85 %)
    # we consider the battery full at 85 % and stop trying to charge further.
    full_thr     = hmax
    # Default charge power resolved from the wizard's configured entity (or
    # falls back to 2 kW). Each return below scales it down only when the
    # remaining gap is small, to avoid burst-charging tiny deltas.
    base_power_w = _battery_charge_power_w(default_w=2000)

    base_inputs = [
        {"label": "Current SOC",              "value": f"{soc:.0f}%"},
        {"label": "Tariff period",            "value": f"{period} ({prices.get(period, '?')}€/kWh)"},
        {"label": "Battery health mode",      "value": f"{cfg('battery_health_mode','bill_reducer')} ({hmin}-{hmax}%)"},
        {"label": "Configured charge power",  "value": f"{base_power_w} W"},
        {"label": "Emergency threshold",      "value": f"{thr_emerg}%"},
        {"label": "Low threshold",            "value": f"{thr_low}%"},
    ]

    storm = is_storm_forecast()
    if storm and soc < storm_thr and period != "peak":
        return {
            "action": "charge", "target_soc": storm_thr, "power_w": base_power_w,
            "reason": f"⚡ Storm forecast — maintaining {storm_thr}% reserve",
            "storm": True, "alert": True,
            "alert_msg": (f"⚡ *STORM MODE ACTIVATED*\n"
                          f"Charging battery to {storm_thr}% storm reserve\n"
                          f"Current SOC: {soc:.0f}%"),
            "explanation": _explain(
                what=f"Charge battery to {storm_thr}% as storm reserve",
                why="Storm forecast detected with SOC below the storm threshold — pre-emptive grid charge to survive a possible outage.",
                inputs=base_inputs + [
                    {"label": "Storm forecast",   "value": "yes"},
                    {"label": "Storm threshold",  "value": f"{storm_thr}%"},
                ],
                formula="if storm and SOC < storm_threshold and not peak → charge to storm_threshold",
                calculation=f"storm=yes AND SOC {soc:.0f}% < storm_thr {storm_thr}% AND period={period} → charge to {storm_thr}%",
                alternatives_rejected=[
                    {"option": "self-consumption", "why": "would let the battery drain through a possible outage"},
                ],
            ),
        }

    if period == "peak":
        if soc < thr_emerg:
            return {
                "action": "charge", "target_soc": emerg_target, "power_w": min(1500, base_power_w),
                "reason": f"EMERGENCY during peak: SOC={soc:.0f}% < {thr_emerg}%",
                "alert": True,
                "alert_msg": (f"🚨 *EMERGENCY GRID CHARGE*\n"
                              f"SOC critical: {soc:.0f}% (threshold: {thr_emerg}%)\n"
                              f"Forced charge during PEAK tariff ({prices['peak']}€/kWh)"),
                "explanation": _explain(
                    what=f"Emergency grid charge to {emerg_target}% during PEAK tariff",
                    why=f"SOC dropped below emergency threshold ({soc:.0f}% < {thr_emerg}%) — paying peak-tariff grid to prevent reaching 0% is the lesser evil.",
                    inputs=base_inputs + [
                        {"label": "Emergency target", "value": f"{emerg_target}% (health-mode floor)"},
                    ],
                    formula="if peak and SOC < emergency_threshold → charge to health_mode_min",
                    calculation=f"period=peak AND SOC {soc:.0f}% < {thr_emerg}% → charge to {emerg_target}%",
                    alternatives_rejected=[
                        {"option": "no action", "why": "battery would hit 0% and we'd import full peak load from grid"},
                    ],
                ),
            }
        return {
            "action": "none",
            "reason": f"Peak ({prices['peak']}€/kWh): no grid charging",
            "explanation": _explain(
                what="No grid action during peak",
                why=f"SOC {soc:.0f}% is above the emergency threshold ({thr_emerg}%), so no expensive peak-tariff grid charge needed.",
                inputs=base_inputs,
                formula="if peak and SOC ≥ emergency_threshold → no action",
                calculation=f"period=peak AND SOC {soc:.0f}% ≥ {thr_emerg}% → battery self-discharges to cover load",
                alternatives_rejected=[
                    {"option": "grid charge", "why": f"would buy at peak price ({prices['peak']}€/kWh) when not needed"},
                ],
            ),
        }

    if period == "valley":
        if soc >= full_thr:
            return {
                "action": "self_consumption",
                "reason": f"Valley, battery full ({soc:.0f}% ≥ health-mode ceiling {full_thr}%)",
                "explanation": _explain(
                    what="Self-consumption mode (battery already full)",
                    why=f"SOC {soc:.0f}% has reached the health-mode ceiling ({full_thr}%) — further charging would stress the cells.",
                    inputs=base_inputs + [
                        {"label": "Health-mode ceiling", "value": f"{full_thr}%"},
                    ],
                    formula="if valley and SOC ≥ health_mode_max → self_consumption",
                    calculation=f"period=valley AND SOC {soc:.0f}% ≥ {full_thr}% → stop charging",
                    alternatives_rejected=[
                        {"option": "continue charging", "why": f"would push past health-mode max ({full_thr}%) and stress the battery"},
                    ],
                ),
            }
        optimal = calculate_optimal_soc(sensors, tariff)
        target  = optimal["target_soc"]
        if soc < target:
            power = base_power_w if (target - soc) > 20 else max(1000, base_power_w // 2)
            return {
                "action": "charge", "target_soc": target, "power_w": power,
                "reason": (f"Valley — smart target {target}% "
                           f"(peak gap {optimal['needed_kwh']:.1f} kWh · "
                           f"solar peak {optimal['solar_during_peak']:.1f} kWh · "
                           f"temp adj {optimal['temp_adj_kwh']:+.1f} kWh · "
                           f"{optimal['peak_h_count']}h peak)"),
                "optimal": optimal,
                "explanation": _explain(
                    what=f"Charge battery to {target}% at {power} W (smart target)",
                    why=f"Valley tariff is the cheapest grid energy of the day. The engine estimates {optimal['peak_total_kwh']:.1f} kWh of peak-window consumption tomorrow against {optimal['solar_during_peak']:.1f} kWh of solar in that window, leaving a {optimal['needed_kwh']:.1f} kWh gap to pre-fill.",
                    inputs=base_inputs + [
                        {"label": "Peak hours configured",  "value": f"{optimal['peak_h_count']}h"},
                        {"label": "Base load (night avg)",  "value": f"{optimal['base_kw']:.2f} kW"},
                        {"label": "Peak consumption est.",  "value": f"{optimal['peak_total_kwh']:.1f} kWh (= {optimal['base_kw']:.2f}kW × {optimal['peak_h_count']}h + {optimal['temp_adj_kwh']:+.1f} kWh temp adj)"},
                        {"label": "Solar tomorrow",         "value": f"{optimal['solar_tomorrow']:.1f} kWh (terrain×{optimal['solar_correction']:.2f})"},
                        {"label": "Solar in peak window",   "value": f"{optimal['solar_during_peak']:.1f} kWh ({100*optimal['peak_solar_share']:.0f}% of total)"},
                        {"label": "Outdoor temp",           "value": f"{optimal['temp_outdoor']:.1f}°C"},
                        {"label": "Battery capacity",       "value": f"{cfg('battery_capacity_kwh', 10.0)} kWh"},
                    ],
                    formula="target_soc = clamp(health_min, health_max, ((peak_consumption − solar_during_peak) / capacity × 100) + 5% buffer)",
                    calculation=f"((({optimal['peak_total_kwh']:.1f} − {optimal['solar_during_peak']:.1f}) / {cfg('battery_capacity_kwh', 10.0)}) × 100) + 5 buffer → clamped to [{hmin},{hmax}] = {target}%",
                    alternatives_rejected=[
                        {"option": f"charge to {hmax}%", "why": "wastes valley grid energy on capacity tomorrow's solar can fill for free"},
                        {"option": "self-consumption now", "why": f"would leave SOC at {soc:.0f}% which is below the {target}% target"},
                    ],
                ),
            }
        return {
            "action": "self_consumption",
            "reason": f"Valley, at optimal level ({soc:.0f}% ≥ {target}%)",
            "explanation": _explain(
                what="Self-consumption mode (smart target already reached)",
                why=f"SOC {soc:.0f}% already meets or exceeds the engine's smart target of {target}% for tomorrow's peak.",
                inputs=base_inputs + [
                    {"label": "Smart target", "value": f"{target}%"},
                ],
                formula="if valley and SOC ≥ smart_target → self_consumption",
                calculation=f"period=valley AND SOC {soc:.0f}% ≥ {target}% → no further charging needed",
                alternatives_rejected=[
                    {"option": "keep charging", "why": "the target is already met; extra grid energy at valley would not be used in peak"},
                ],
            ),
        }

    if period == "mid":
        if soc < thr_emerg:
            return {
                "action": "charge", "target_soc": emerg_target, "power_w": min(1500, base_power_w),
                "reason": f"Mid + critical SOC ({soc:.0f}%) → emergency charge",
                "alert": True,
                "alert_msg": (f"🚨 *EMERGENCY GRID CHARGE*\n"
                              f"SOC critical: {soc:.0f}% during mid-peak\n"
                              f"Forced charge to {emerg_target}%"),
                "explanation": _explain(
                    what=f"Emergency grid charge to {emerg_target}% during MID tariff",
                    why=f"SOC critical ({soc:.0f}% < {thr_emerg}%). Even at mid tariff, recovery is worth the cost to avoid hitting 0%.",
                    inputs=base_inputs,
                    formula="if mid and SOC < emergency_threshold → charge to health_mode_min",
                    calculation=f"period=mid AND SOC {soc:.0f}% < {thr_emerg}% → charge to {emerg_target}%",
                    alternatives_rejected=[
                        {"option": "wait for valley", "why": "could take hours; SOC would hit 0% before then"},
                    ],
                ),
            }
        if soc < thr_low:
            return {
                "action": "charge", "target_soc": low_target, "power_w": min(1500, base_power_w),
                "reason": f"Mid + low SOC ({soc:.0f}% < {thr_low}%) → charge to {low_target}%",
                "alert": True,
                "alert_msg": (f"⚠️ *GRID CHARGE (mid-peak)*\n"
                              f"Low battery: {soc:.0f}% — charging to {low_target}%\n"
                              f"Tariff: {prices.get('mid', '?')}€/kWh"),
                "explanation": _explain(
                    what=f"Top-up to {low_target}% during MID tariff",
                    why=f"SOC ({soc:.0f}%) below the low threshold ({thr_low}%) — top-up at mid-tariff is preferable to risking emergency charge at peak.",
                    inputs=base_inputs,
                    formula="if mid and SOC < low_threshold → charge to max(40, health_mode_min+10)",
                    calculation=f"period=mid AND SOC {soc:.0f}% < {thr_low}% → charge to {low_target}%",
                    alternatives_rejected=[
                        {"option": "wait for valley", "why": "may take hours; risk of dropping into emergency range"},
                    ],
                ),
            }
        optimal = calculate_optimal_soc(sensors, tariff)
        target  = optimal["target_soc"]
        if soc < target:
            return {
                "action": "charge", "target_soc": target,
                "power_w": max(1000, base_power_w // 2),
                "reason": (f"Mid — opportunistic top-up to smart target {target}% "
                           f"(gap {optimal['needed_kwh']:.1f} kWh)"),
                "optimal": optimal,
                "explanation": _explain(
                    what=f"Opportunistic top-up to {target}% during MID tariff",
                    why=f"SOC {soc:.0f}% is below the smart target {target}%. Mid tariff is cheaper than peak — preferable to fill the gap now than import at peak later.",
                    inputs=base_inputs + [
                        {"label": "Smart target", "value": f"{target}%"},
                        {"label": "Peak gap",     "value": f"{optimal['needed_kwh']:.1f} kWh"},
                    ],
                    formula="if mid and SOC < smart_target → charge to smart_target at half-power",
                    calculation=f"period=mid AND SOC {soc:.0f}% < {target}% → charge to {target}%",
                    alternatives_rejected=[
                        {"option": "self-consumption", "why": f"would leave SOC at {soc:.0f}% below the smart target {target}%"},
                        {"option": "wait for valley", "why": "may not arrive before next peak; mid is cheaper than peak"},
                    ],
                ),
            }
        return {
            "action": "self_consumption",
            "reason": f"Mid period, self-consumption (SOC={soc:.0f}% ≥ {target}%)",
            "explanation": _explain(
                what="Self-consumption mode (mid tariff, target reached)",
                why=f"SOC {soc:.0f}% already meets the smart target {target}%. No mid-tariff grid charge needed.",
                inputs=base_inputs + [{"label": "Smart target", "value": f"{target}%"}],
                formula="if mid and SOC ≥ smart_target → self_consumption",
                calculation=f"period=mid AND SOC {soc:.0f}% ≥ {target}% → no grid action",
                alternatives_rejected=[
                    {"option": "top-up to health-mode max", "why": "would buy mid-tariff energy beyond what peak demand needs"},
                ],
            ),
        }

    return {
        "action": "none",
        "reason": "No battery action",
        "explanation": _explain(
            what="No battery action",
            why="The tariff period did not match any known branch.",
            inputs=base_inputs,
        ),
    }

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

# ── Setup integrity check (v4.0.2) ────────────────────────────────────────────
def run_setup_integrity_check() -> None:
    """Run the wizard-config integrity check and publish results to HA.

    Detects entity_ids assigned to multiple actuable roles (e.g. the same
    switch as `pool_switch` and as a `custom_loads[].switch`), which causes
    the engine to issue contradictory on/off commands in the same cycle.

    Side effects:
      • Logs warnings/criticals.
      • Publishes `binary_sensor.energy_optimizer_setup_conflict` to HA.
      • Sends a Telegram alert if there are CRITICAL conflicts.
    """
    from eo.checks.setup_integrity import check as _check_integrity

    report = _check_integrity(_WIZARD)

    if report.ok and not report.conflicts:
        log.info("  Setup integrity: OK")
    else:
        for line in report.to_summary().splitlines():
            (log.error if not report.ok else log.warning)(line)

    state = "off" if report.ok else "on"  # binary_sensor: on = problem detected
    ha_post("/api/states/binary_sensor.energy_optimizer_setup_conflict", {
        "state": state,
        "attributes": {
            "friendly_name":   "Energy Optimizer Setup Conflict",
            "device_class":    "problem",
            "icon":            "mdi:alert-circle" if not report.ok else "mdi:check-circle",
            "critical_count":  len(report.critical_conflicts),
            "warning_count":   len(report.warnings),
            "conflicts":       [c.to_dict() for c in report.conflicts],
            "summary":         report.to_summary(),
        },
    })

    if report.critical_conflicts:
        msg_lines = ["🚨 *Energy Optimizer — Setup conflict detected*", ""]
        for c in report.critical_conflicts:
            roles = ", ".join(f"`{o.role}`" for o in c.occurrences)
            msg_lines.append(f"• `{c.entity_id}` is assigned to {len(c.occurrences)} roles: {roles}")
        msg_lines.append("")
        msg_lines.append("This causes contradictory on/off commands in the same cycle. Fix in the Wizard before relying on automation.")
        send_telegram_alert("\n".join(msg_lines))

# ── Pool pump + cleaner logic ────────────────────────────────────────────────
_scheduler_ref = None  # set in main()

# ── Pool manual-override state (v5.0.9 / v5.0.10) ───────────────────────────
# When Sergio flips the pool switch by hand (Lovelace, mobile app, physical
# button…), the addon runs the pump for exactly `pool_manual_min_runtime_min`
# minutes (default 50) and then turns it off via a deferred APScheduler job —
# regardless of what the automatic decision would say in between. State
# persists across rebuilds at /data/pool_manual_state.json; the job is
# re-scheduled on startup if the off-time is still in the future.
_pool_manual_state: dict = {
    "addon_last_action": None,
    "manual_on_since":   None,
    "manual_off_at":     None,
}
_POOL_MANUAL_OFF_JOB_ID = "pool_manual_off"

def _load_pool_manual_state():
    global _pool_manual_state
    if POOL_MANUAL_STATE_FILE.exists():
        try:
            _pool_manual_state = json.loads(POOL_MANUAL_STATE_FILE.read_text())
        except Exception:
            pass
    # Backfill key for installs that ran v5.0.9 and saved a 2-key state.
    _pool_manual_state.setdefault("manual_off_at", None)

def _save_pool_manual_state():
    try:
        POOL_MANUAL_STATE_FILE.write_text(json.dumps(_pool_manual_state))
    except Exception as e:
        log.warning(f"  Could not persist pool_manual_state.json: {e}")

def _pool_manual_timer_off() -> None:
    """Deferred APScheduler task — turn the pool pump off at the end of the
    manual-ON window. Logs to decisions.json so the Activity tab shows it."""
    pool_sw = _wiz("pool_switch", "switch_pool", "")
    if not pool_sw:
        return
    log.info("  [POOL] Manual timer reached → switching pump OFF")
    ok = ha_switch(pool_sw, False)
    _pool_manual_state["addon_last_action"] = "off"
    _pool_manual_state["manual_on_since"]   = None
    _pool_manual_state["manual_off_at"]     = None
    _save_pool_manual_state()
    _record_event({
        "type":   "pool",
        "action": "off",
        "switch": pool_sw,
        "reason": "Manual ON timer elapsed — pump auto-off",
        "ok":     ok,
        "explanation": _explain(
            what=f"Turn off pool pump {pool_sw}",
            why=("The user flipped the pool switch on by hand; the addon ran "
                 "the pump for the configured manual runtime and now turns "
                 "it off as promised."),
            inputs=[
                {"label": "Pool switch", "value": pool_sw},
                {"label": "Trigger",     "value": "APScheduler manual-OFF timer"},
            ],
            formula="manual ON detected at T → schedule OFF at T + pool_manual_min_runtime_min",
        ),
    }, source="scheduler")

def _schedule_pool_manual_off(when: datetime) -> None:
    """(Re)schedule the deferred OFF job. Idempotent — replaces any existing
    job with the same id, so re-flipping the switch resets the timer cleanly."""
    if not _scheduler_ref:
        log.warning("  [POOL] Scheduler not ready — cannot schedule manual OFF")
        return
    _scheduler_ref.add_job(
        _pool_manual_timer_off,
        "date", run_date=when,
        id=_POOL_MANUAL_OFF_JOB_ID, replace_existing=True,
        misfire_grace_time=120,
    )

def _cancel_pool_manual_off() -> None:
    if not _scheduler_ref:
        return
    try:
        _scheduler_ref.remove_job(_POOL_MANUAL_OFF_JOB_ID)
    except Exception:
        pass  # job already fired or never existed — fine

def _reschedule_pool_manual_off_after_restart() -> None:
    """Called from main() after the scheduler is alive. If a manual-OFF was
    pending when the addon stopped, restore it. If the deadline has passed
    while we were down, fire the OFF immediately."""
    off_at_iso = _pool_manual_state.get("manual_off_at")
    if not off_at_iso:
        return
    try:
        off_at = datetime.fromisoformat(off_at_iso)
    except Exception:
        log.warning(f"  [POOL] Bad manual_off_at on disk ({off_at_iso}) — clearing")
        _pool_manual_state["manual_off_at"] = None
        _pool_manual_state["manual_on_since"] = None
        _save_pool_manual_state()
        return
    now = datetime.now()
    if off_at > now:
        log.info(f"  [POOL] Restoring manual OFF timer "
                 f"(fires at {off_at.strftime('%H:%M:%S')})")
        _schedule_pool_manual_off(off_at)
    else:
        log.info(f"  [POOL] Manual OFF was due at {off_at.strftime('%H:%M:%S')} "
                 f"while addon was down → firing now")
        _pool_manual_timer_off()

def _check_pool_manual_override(pool_sw: str) -> dict | None:
    """Return a manual-override decision if a manual ON is in progress,
    otherwise None so `decide_pool` runs as usual. Side effects: on a fresh
    manual ON, persist `manual_off_at` and schedule the deferred OFF job."""
    if not pool_sw:
        return None
    runtime_min = float(cfg("pool_manual_min_runtime_min", 50.0))
    actual_on   = (ha_str(pool_sw, "").lower() == "on")
    addon_last  = _pool_manual_state.get("addon_last_action")
    manual_since_iso = _pool_manual_state.get("manual_on_since")
    now = datetime.now()

    # 1. Fresh manual ON detected
    if actual_on and addon_last != "on" and not manual_since_iso:
        off_at = now + timedelta(minutes=runtime_min)
        _pool_manual_state["manual_on_since"] = now.isoformat()
        _pool_manual_state["manual_off_at"]   = off_at.isoformat()
        _save_pool_manual_state()
        _schedule_pool_manual_off(off_at)
        log.info(f"  [POOL] Manual ON detected → auto-OFF at "
                 f"{off_at.strftime('%H:%M')} ({runtime_min:.0f} min)")
        manual_since_iso = _pool_manual_state["manual_on_since"]

    # 2. Switch went off (user or addon) → cancel timer + clear lock
    if not actual_on and manual_since_iso:
        _cancel_pool_manual_off()
        _pool_manual_state["manual_on_since"] = None
        _pool_manual_state["manual_off_at"]   = None
        _save_pool_manual_state()
        log.info("  [POOL] Manual lock cleared (switch off, timer cancelled)")
        return None

    # 3. Inside the manual window → force ON. The deferred APScheduler job
    #    will turn it off at the exact minute; the cycle just holds the line.
    if manual_since_iso:
        off_at_iso = _pool_manual_state.get("manual_off_at")
        try:
            off_at = datetime.fromisoformat(off_at_iso) if off_at_iso else None
        except Exception:
            off_at = None
        if off_at and now < off_at:
            try:
                since = datetime.fromisoformat(manual_since_iso)
                elapsed = (now - since).total_seconds() / 60.0
            except Exception:
                elapsed = 0.0
            remaining = (off_at - now).total_seconds() / 60.0
            return {
                "action": True,
                "reason": (f"Manual ON — {elapsed:.0f}/{runtime_min:.0f} min "
                           f"({remaining:.0f} min remaining)"),
                "manual_override": True,
            }
        # Window elapsed but the job didn't run (clock skew, scheduler issue) →
        # apagar y limpiar aquí mismo para no quedarse atascado en ON.
        if actual_on:
            log.warning("  [POOL] Manual window elapsed without timer firing "
                        "→ forcing OFF now")
            ha_switch(pool_sw, False)
            _pool_manual_state["addon_last_action"] = "off"
        _cancel_pool_manual_off()
        _pool_manual_state["manual_on_since"] = None
        _pool_manual_state["manual_off_at"]   = None
        _save_pool_manual_state()
        return None

    return None

# ── HVAC per-zone decision state (v5.0.11) ──────────────────────────────────
# Tracks whether the addon's last decision for each zone was "active" or
# "skip", so we can detect the edge `active → skip` and reset the cooling
# setpoint back to a safe off-value. Without this, the slider sits at the
# aggressive value the addon wrote during the last surplus cycle (e.g. 16 °C
# for Z1CoolingTemp) for hours/days, and any out-of-band trigger (manual
# flip, ebusd internal loop) would run the heat pump way too hard.
_hvac_decision_state: dict = {}

def _load_hvac_decision_state():
    global _hvac_decision_state
    if HVAC_DECISION_STATE_FILE.exists():
        try:
            _hvac_decision_state = json.loads(HVAC_DECISION_STATE_FILE.read_text())
        except Exception:
            pass

def _save_hvac_decision_state():
    try:
        HVAC_DECISION_STATE_FILE.write_text(json.dumps(_hvac_decision_state))
    except Exception as e:
        log.warning(f"  Could not persist hvac_decision_state.json: {e}")

# ── Quiet log mode (v5.0.12) ─────────────────────────────────────────────────
# When `log_quiet_mode` is on, suppress the routine per-category INFO lines
# inside run_cycle and replace them with a structured end-of-cycle summary:
#  - `[Δ <cat>] ...` only for categories whose verdict changed vs last cycle.
#  - `[HB] BAT=... HP=... POOL=... DW=...` once per hour as heartbeat.
# WARNING/ERROR always pass through. State persists in /data/log_summary_state.json.
_log_summary_state: dict = {"last_summary": {}, "last_heartbeat": None}

def _load_log_summary_state():
    global _log_summary_state
    if LOG_SUMMARY_STATE_FILE.exists():
        try:
            _log_summary_state = json.loads(LOG_SUMMARY_STATE_FILE.read_text())
        except Exception:
            pass

def _save_log_summary_state():
    try:
        LOG_SUMMARY_STATE_FILE.write_text(json.dumps(_log_summary_state))
    except Exception as e:
        log.warning(f"  Could not persist log_summary_state.json: {e}")

class _QuietCycleFilter(logging.Filter):
    """Suppress noisy INFO lines emitted during run_cycle when quiet mode is
    enabled. Whitelist:
       - any record at WARNING or above
       - the cycle banner
       - explicit summary lines emitted by _emit_cycle_summary (prefix `[Δ`
         or `[HB`)
    """
    def __init__(self):
        super().__init__()
        self.enabled = False

    def filter(self, record):
        if not self.enabled:
            return True
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        if "Decision cycle" in msg:
            return True
        stripped = msg.lstrip()
        if stripped.startswith("[Δ") or stripped.startswith("[HB"):
            return True
        return False

_QUIET_FILTER = _QuietCycleFilter()

def _verdict_for(item: dict) -> str:
    """Reduce an action/skip dict to a stable verdict for diffing — keep the
    first few words of the reason, dropping the trailing numbers/temps that
    flicker cycle-to-cycle."""
    reason = (item.get("reason") or "").strip()
    # Trim at the first opening parenthesis (where the numbers usually live).
    cut = reason.find(" (")
    bucket = reason[:cut] if cut > 0 else reason
    # Cap aggressively
    return bucket[:60]

def _emit_cycle_summary(decision: dict) -> None:
    """End of run_cycle in quiet mode: log only what changed vs last cycle,
    plus an hourly heartbeat with the current verdicts. No-op in verbose mode."""
    if not cfg("log_quiet_mode", False):
        return
    # Compute current summary keyed by 'type:zone'
    summary: dict = {}
    item_lookup: dict = {}
    for items in decision.get("actions", []) + decision.get("skipped", []):
        t = items.get("type", "") or ""
        zone = items.get("zone") or ""
        key = f"{t}:{zone}" if zone else t
        summary[key] = _verdict_for(items)
        item_lookup[key] = (items.get("reason") or "")[:140]
    # Compare with last
    last = _log_summary_state.get("last_summary", {}) or {}
    changes = [(k, last.get(k), v) for k, v in summary.items() if last.get(k) != v]
    for k, was, now_v in changes:
        # Use the rich reason on the new state, mark the transition
        was_short = (was or "(new)")[:40]
        log.info(f"  [Δ {k}] {was_short} → {item_lookup.get(k, now_v)}")
    # Heartbeat
    now = datetime.now()
    last_hb_iso = _log_summary_state.get("last_heartbeat")
    need_hb = True
    if last_hb_iso:
        try:
            last_hb = datetime.fromisoformat(last_hb_iso)
            need_hb = (now - last_hb).total_seconds() >= 3600
        except Exception:
            pass
    if need_hb or (changes and not last_hb_iso):
        compact = " | ".join(
            f"{k}={summary[k]}" for k in sorted(summary.keys())
        )
        log.info(f"  [HB] {compact[:240]}")
        _log_summary_state["last_heartbeat"] = now.isoformat()
    _log_summary_state["last_summary"] = summary
    _save_log_summary_state()

def _record_pool_addon_action(action: str) -> None:
    """Persist the last on/off action the addon issued for the pool pump so
    `_check_pool_manual_override` can tell apart its own turn-ons from the
    user's. Call right after every ha_switch(pool_sw, ...)."""
    if action not in ("on", "off"):
        return
    _pool_manual_state["addon_last_action"] = action
    _save_pool_manual_state()

def decide_pool(sensors: dict, tariff: dict) -> dict:
    summer  = _is_summer()
    hrs_day = sensors.get("pool_hours_day", 0)
    hrs_wk  = sensors.get("pool_hours_week", 0)
    prices  = tariff["prices"]

    if summer and hrs_day >= 1.0:
        return {"action": False, "reason": f"Daily hours met ({hrs_day:.1f}h ≥ 1h/day)"}
    if not summer and hrs_wk >= 1.0:
        return {"action": False, "reason": f"Weekly hours met ({hrs_wk:.1f}h ≥ 1h/week)"}

    solar_now  = sensors["solar_current_hour"]
    solar_live = sensors.get("solar_power", 0)   # actual panel watts (0 if sensor not set)
    soc        = sensors["battery_soc"]
    period     = tariff["period"]

    # Prefer real-time solar_power (W) for surplus check; fallback to hourly forecast (kWh/h)
    solar_surplus = (solar_live > 1200) or (solar_now > 1.2)
    if solar_surplus and soc > 50:
        src = f"{solar_live:.0f}W" if solar_live > 0 else f"{solar_now:.2f} kWh/h"
        return {"action": True, "reason": f"Solar surplus ({src}), SOC={soc:.0f}%"}
    if period == "valley":
        return {"action": True, "reason": f"Valley tariff ({prices['valley']}€/kWh), hours pending"}
    if summer and hrs_day < 0.1 and datetime.now().hour >= 20:
        return {"action": True, "reason": f"Urgent: only {hrs_day:.1f}h today, late hour"}

    return {"action": False, "reason": f"Waiting — period={period}, solar={solar_now:.2f} kWh/h"}

def _start_pool_with_cleaner(pool_sw: str, cleaner_sw: str) -> list[dict]:
    """Turn on pool pump; start limpiafondos (~1.5 kWh) and auto-stop after 15 min.

    Returns the list of actions performed so the caller can append them to
    the cycle's `decision["actions"]` — previously the limpiafondos start
    happened silently inside this function, and its 15-min auto-stop happened
    in a deferred APScheduler job that never reached decisions.json (the
    "black box" Sergio called out). Both are now traced.
    """
    actions = []
    pump_ok = ha_switch(pool_sw, True)
    # The pool pump itself is logged by the caller as part of the pool decision;
    # we return only the cleaner-related actions to avoid double-logging.
    if cleaner_sw and _scheduler_ref:
        clean_ok = ha_switch(cleaner_sw, True)
        log.info("  [POOL] Limpiafondos ON — will auto-stop in 15 min")
        actions.append({
            "type":     "pool_cleaner",
            "action":   "on",
            "switch":   cleaner_sw,
            "reason":   "Auto-started with pool pump (15-min run)",
            "ok":       clean_ok,
            "explanation": _explain(
                what=f"Turn on pool cleaner {cleaner_sw}",
                why="Pool pump just started — the cleaner runs for 15 min in parallel as configured by _start_pool_with_cleaner.",
                inputs=[
                    {"label": "Pool pump switch",      "value": pool_sw},
                    {"label": "Cleaner switch",        "value": cleaner_sw},
                    {"label": "Auto-stop after",       "value": "15 min"},
                    {"label": "Pool pump start ok",    "value": str(pump_ok)},
                ],
                formula="if pool pump turned on AND cleaner configured → cleaner on, schedule off in 15 min",
                calculation="pool ON → cleaner ON → scheduler.add_job(cleaner OFF at now+15min)",
            ),
        })
        shutoff_time = datetime.now() + timedelta(minutes=15)

        def _cleaner_shutoff():
            """Deferred task: turn off the cleaner 15 min after start, and
            persist the off-action to decisions.json via _record_event so it
            shows up in the Activity tab alongside cycle decisions."""
            ok = ha_switch(cleaner_sw, False)
            log.info("  [POOL] Limpiafondos OFF (15 min timer completed)")
            _record_event({
                "type":     "pool_cleaner",
                "action":   "off",
                "switch":   cleaner_sw,
                "reason":   "Auto-stop after 15-min run",
                "ok":       ok,
                "explanation": _explain(
                    what=f"Turn off pool cleaner {cleaner_sw}",
                    why="The 15-min timer scheduled when the cleaner started has now elapsed.",
                    inputs=[
                        {"label": "Cleaner switch",   "value": cleaner_sw},
                        {"label": "Trigger",          "value": "APScheduler timer (deferred)"},
                    ],
                    formula="scheduled 15 min ago when pool pump started → fire now",
                ),
            }, source="scheduler")

        _scheduler_ref.add_job(
            _cleaner_shutoff,
            "date", run_date=shutoff_time,
            id="cleaner_shutoff", replace_existing=True,
        )
    return actions

# ── Dishwasher logic ─────────────────────────────────────────────────────────
def decide_dishwasher(sensors: dict, tariff: dict) -> dict:
    state      = sensors.get("dishwasher_state", "")
    period     = tariff["period"]
    solar      = sensors["solar_current_hour"]
    solar_live = sensors.get("solar_power", 0)   # actual panel watts
    soc        = sensors["battery_soc"]
    prices     = tariff["prices"]

    if state.lower() == "running":
        return {"action": None, "reason": "Already running", "state": state}
    if state.lower() != "ready":
        return {"action": None, "reason": f"Not ready (state: {state or 'unknown'})", "state": state}

    solar_surplus = (solar_live > 1500) or (solar > 1.5)
    if solar_surplus and soc > 60:
        src = f"{solar_live:.0f}W" if solar_live > 0 else f"{solar:.2f} kWh/h"
        return {"action": True,
                "reason": f"Solar surplus ({src}) + good SOC ({soc:.0f}%)",
                "state": state}
    if period == "valley":
        return {"action": True, "reason": f"Valley tariff ({prices['valley']}€/kWh)", "state": state}
    if period == "peak":
        return {"action": False,
                "reason": f"Peak tariff ({prices['peak']}€/kWh) — waiting", "state": state}
    return {"action": False, "reason": "Mid tariff — waiting for solar or valley", "state": state}

# ── Custom loads (user-defined switch-controlled devices) ────────────────────
# The wizard's Loads step lets the user add arbitrary `custom_loads`: each one
# has a switch entity, an estimated wattage, and a scheduling mode (`valley`,
# `solar`, `both`, `hours`). Until now the wizard saved these but no Python
# code consumed them — reported as a bug by @Karplyak in issue #4 (a load
# configured to fire 04-07 was being switched 23-07 by something else
# entirely, because the engine was simply not driving the switch at all).

def _hours_active(hour: int, spec: str) -> bool:
    """Return True if `hour` falls inside any range in `spec`.

    `spec` is the comma-separated string the user entered (e.g. `10-14,22-06`).
    Ranges where start > end wrap around midnight (`22-06` covers 22, 23, 0..5).
    Single-hour bare values (`12`) and malformed segments are ignored.
    """
    if not spec:
        return False
    for part in spec.split(","):
        part = part.strip()
        if "-" not in part:
            continue
        try:
            s_raw, e_raw = part.split("-", 1)
            s, e = int(s_raw.strip()) % 24, int(e_raw.strip()) % 24
        except (ValueError, AttributeError):
            continue
        if s == e:
            continue
        if s < e:
            if s <= hour < e:
                return True
        else:  # wraps midnight
            if hour >= s or hour < e:
                return True
    return False


def decide_custom_loads(sensors: dict, tariff: dict) -> list:
    """Decide on/off for every configured custom_load. Returns a list of
    {switch, name, want_on, current, reason} for loads that need a state
    change; loads already in the desired state are NOT included so the cycle
    doesn't spam the bus with redundant service calls."""
    loads      = _WIZARD.get("custom_loads", []) or []
    grid       = sensors.get("grid_power", 0) or 0   # +ve = exporting
    soc        = sensors.get("battery_soc", 0) or 0
    period     = tariff.get("period", "mid")
    hour       = datetime.now().hour
    decisions  = []

    for ld in loads:
        switch = (ld.get("switch") or "").strip()
        if not switch:
            continue
        sched = (ld.get("schedule") or "valley").strip()
        try:
            watts = float(ld.get("watts") or 0)
        except (TypeError, ValueError):
            watts = 0.0
        name = ld.get("name") or switch.split(".")[-1]

        in_valley = (period == "valley")
        # Solar surplus = exporting more than this load would draw (so flipping
        # it on doesn't import). Require SOC > 30% as cushion against clouds.
        in_solar  = (grid > max(watts, 50.0)) and (soc > 30)

        if sched == "valley":
            want_on = in_valley
            reason  = ("Valley tariff active" if want_on
                       else f"Outside valley (period={period})")
        elif sched == "solar":
            want_on = in_solar
            reason  = (f"Solar surplus (export {grid:.0f}W ≥ {watts:.0f}W, SOC={soc:.0f}%)"
                       if want_on
                       else f"Insufficient surplus (export {grid:.0f}W, SOC={soc:.0f}%)")
        elif sched == "both":
            want_on = in_valley or in_solar
            if in_valley:
                reason = "Valley tariff active"
            elif in_solar:
                reason = f"Solar surplus (export {grid:.0f}W ≥ {watts:.0f}W)"
            else:
                reason = f"No valley, no surplus (period={period}, export {grid:.0f}W)"
        elif sched == "hours":
            spec    = ld.get("hours") or ""
            want_on = _hours_active(hour, spec)
            reason  = (f"In hours range {spec} (h={hour:02d})" if want_on
                       else f"Outside hours range {spec} (h={hour:02d})")
        else:
            log.warning(f"  [CUSTOM LOAD/{name}] Unknown schedule '{sched}' — skipped")
            continue

        st = ha_state(switch)
        if not st:
            log.debug(f"  [CUSTOM LOAD/{name}] {switch} unavailable — skipped")
            continue
        cur_on = st.get("state") == "on"
        if want_on == cur_on:
            continue  # already in desired state, no action

        decisions.append({"switch": switch, "name": name, "want_on": want_on,
                          "current": cur_on, "reason": reason, "schedule": sched})
    return decisions

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
    # Counterfactual: cost with battery vs cost without battery (same consumption).
    # grid_power: -ve = importing, +ve = exporting. battery_power: +ve = charging, -ve = discharging.
    # Without the battery, its flow would have hit the grid: grid_cf = grid + battery.
    grid = sensors.get("grid_power")
    bat  = sensors.get("battery_power")
    if grid is None or bat is None:
        return
    interval_h = cfg("decision_interval_minutes", 15) / 60
    prices     = tariff["prices"]
    p_imp      = prices.get(tariff["period"], prices.get("mid", 0.15))
    p_exp      = prices.get("export", 0.06)

    def _net_cost_eur(grid_w: float) -> float:
        kwh = grid_w * interval_h / 1000
        return -kwh * p_imp if kwh < 0 else -kwh * p_exp

    grid_cf  = grid + bat
    delta    = _net_cost_eur(grid_cf) - _net_cost_eur(grid)   # +ve = battery saved money
    if abs(delta) > 0.30:
        log.warning(f"  [SAVINGS] delta {delta:.3f}€ exceeds ±0.30€/cycle cap — skipped (grid={grid:.0f}W bat={bat:.0f}W)")
        return

    savings = _load_savings()
    savings["total_eur_saved"]      = round(savings.get("total_eur_saved", 0) + delta, 4)
    savings["total_kwh_throughput"] = round(savings.get("total_kwh_throughput", 0) + abs(bat) * interval_h / 1000, 3)
    if tariff["period"] == "peak" and bat < 0 and grid < -100:
        kwh_peak = min(abs(bat) * interval_h / 1000, 2.0)
        savings["total_kwh_avoided_peak"] = round(savings.get("total_kwh_avoided_peak", 0) + kwh_peak, 3)
    savings["decisions_count"]      = savings.get("decisions_count", 0) + 1
    SAVINGS_FILE.write_text(json.dumps(savings, indent=2))

# ── Decision cycle ───────────────────────────────────────────────────────────
def run_cycle() -> dict:
    quiet = cfg("log_quiet_mode", False)
    if quiet:
        _QUIET_FILTER.enabled = True
    try:
        return _run_cycle_inner()
    finally:
        if quiet:
            _QUIET_FILTER.enabled = False

def _run_cycle_inner() -> dict:
    log.info("━━━ Decision cycle ━━━━━━━━━━━━━━━━━━━━━━")
    sensors    = read_sensors()
    tariff     = current_tariff()
    prediction = predict_soc(sensors)

    # Hydraulic ground truth: log the heat-pump flow temps + ΔT every cycle.
    # ΔT = ida − vuelta (the user's convention). Heating gives ΔT > 0 (water
    # leaves hot, returns cooler — gave heat to the loop). Cooling gives
    # ΔT < 0 (water leaves cold, returns warmer — absorbed heat from the
    # house). Idle collapses both legs to ~0.
    if "hp_delta_t_c" in sensors:
        d = sensors["hp_delta_t_c"]
        verdict = ("HEATING" if d > 1.0 else
                   "COOLING" if d < -1.0 else
                   "idle")
        log.info(f"  [HYDRAULIC] flow_in={sensors['hp_flow_in_c']:.1f}°C  "
                 f"flow_out={sensors['hp_flow_out_c']:.1f}°C  "
                 f"ΔT={d:+.1f}°C → {verdict}")

    decision = {"timestamp": datetime.now().isoformat(), "sensors": sensors,
                "tariff": tariff, "prediction": prediction, "actions": [], "skipped": []}

    # 1. Battery
    bat = decide_battery(sensors, tariff, prediction)
    bat_expl = bat.get("explanation")
    if bat["action"] == "charge":
        ok = set_battery_charge_target(bat["target_soc"], bat.get("power_w", 3000))
        decision["actions"].append({"type": "battery", "action": "charge",
            "target_soc": bat["target_soc"], "reason": bat["reason"], "ok": ok,
            "explanation": bat_expl})
        log.info(f"  [BATTERY] Charge → {bat['target_soc']}% — {bat['reason']}")
        if bat.get("alert") and bat.get("alert_msg"):
            send_telegram_alert(bat["alert_msg"])
    elif bat["action"] == "self_consumption":
        ok = set_battery_self_consumption()
        decision["actions"].append({"type": "battery", "action": "self_consumption",
            "reason": bat["reason"], "ok": ok, "explanation": bat_expl})
        log.info(f"  [BATTERY] Self-consumption — {bat['reason']}")
    else:
        decision["skipped"].append({"type": "battery", "reason": bat["reason"],
            "explanation": bat_expl})
        log.info(f"  [BATTERY] No action — {bat['reason']}")

    # 2. Heat pump — multi-zone
    # Each zone may produce TWO side effects: the number setpoint (always)
    # and an optional climate.set_temperature (when a climate entity is also
    # configured). Previously these were collapsed into one log entry with the
    # `ok` of the number-write only — the climate call was a silent black box.
    # Now they get separate entries (`heat_pump` and `climate_setpoint`) so the
    # Activity tab traces both with their own ok/explanation.
    hp_zones = decide_heat_pump(sensors)
    # Dedup: when several zones fall back to the same global slider (because
    # they don't have their own `temp_cool` entity), we only want one write
    # per actual entity per cycle. Skipping the duplicates avoids spamming
    # the bus and triple-logging the same reset.
    hvac_written_entities: set[str] = set()
    for hp in hp_zones:
        zone_tag = f"/{hp['zone']}" if hp.get("zone", "default") != "default" else ""
        # Conservative summer regime can decide to NOT touch the setpoint at all
        # (e.g. radiant cooling: don't engage overnight without solar surplus).
        # The action is recorded as a no-op with ok=True so the Activity tab
        # makes the inaction visible instead of silent.
        zone_key = hp.get("zone") or "default"
        if hp.get("skip"):
            # Edge-triggered cooling-OFF write (v5.0.11): when the decision
            # transitions from active → skip in summer AND the zone writes to
            # a number entity (not just a climate mirror), bump the slider
            # back up to a safe "off" value so it doesn't sit at e.g. 16 °C
            # for hours after the surplus window ends.
            prev = _hvac_decision_state.get(zone_key, {}).get("last")
            off_setpoint = float(cfg("cooling_off_setpoint_c", 25.0))
            reset_reason = None
            if hp.get("season") == "summer" and hp.get("entity"):
                if prev == "active":
                    reset_reason = "active→skip transition"
                else:
                    # Defensive bootstrap / recovery: if the slider is still
                    # parked at an aggressive cooling value (lower than the
                    # off-setpoint by more than 1 °C), bring it up. Covers
                    # first cycle after a fresh deploy when the state file is
                    # empty but a previous version left a low setpoint.
                    current = ha_float(hp["entity"])
                    if current and current < off_setpoint - 1:
                        reset_reason = (f"slider parked at {current:.1f}°C "
                                        f"during skip — bootstrap reset")
            if reset_reason and hp["entity"] in hvac_written_entities:
                # Another zone in this cycle already reset this slider; skip
                # the duplicate write (the value is the same anyway).
                reset_reason = None
            if reset_reason:
                ok_reset = ha_set_number(hp["entity"], off_setpoint)
                hvac_written_entities.add(hp["entity"])
                log.info(f"  [HP{zone_tag}] {reset_reason} → reset "
                         f"{hp['entity']} to {off_setpoint:.1f}°C")
                decision["actions"].append({
                    "type": "heat_pump_off_reset", "entity": hp["entity"],
                    "zone": hp.get("zone"), "value": off_setpoint,
                    "reason": (f"{reset_reason} — setpoint reset to "
                               f"{off_setpoint:.1f}°C"),
                    "ok": ok_reset,
                    "explanation": _explain(
                        what=f"Reset cooling setpoint on {hp['entity']} to {off_setpoint}°C",
                        why=("Without this reset the slider would stay at the "
                             "aggressive value (e.g. 16 °C) the addon wrote "
                             "during the last surplus cycle. The addon is no "
                             "longer asking for cooling, so the slider must "
                             "match — otherwise any out-of-band trigger "
                             "(manual flip, ebusd internal loop) would run "
                             "the heat pump too hard."),
                        inputs=[
                            {"label": "Entity",       "value": hp["entity"]},
                            {"label": "Off setpoint", "value": f"{off_setpoint:.1f}°C"},
                            {"label": "Prev state",   "value": prev or "unknown"},
                        ],
                        formula=("if prev decision was active AND now skip "
                                 "AND summer → write cooling_off_setpoint_c"),
                    ),
                })
            else:
                decision["actions"].append({
                    "type": "heat_pump", "entity": hp.get("entity"),
                    "zone": hp.get("zone"), "sched_mode": hp.get("sched_mode"),
                    "value": None, "reason": hp["reason"], "ok": True,
                    "explanation": hp.get("explanation")})
                log.info(f"  [HP{zone_tag}] skip — {hp['reason']}")
            _hvac_decision_state[zone_key] = {
                "last": "skip", "ts": datetime.now().isoformat()}
            _save_hvac_decision_state()
            continue
        # Active branch — record state and proceed with the existing write.
        _hvac_decision_state[zone_key] = {
            "last": "active", "ts": datetime.now().isoformat()}
        _save_hvac_decision_state()
        if hp["entity"] and hp["entity"] in hvac_written_entities:
            # Same slider already written by another zone this cycle (last
            # value wins). Skip the redundant write — historically the addon
            # repeated the write 3× when zones shared a global fallback.
            ok = True
        else:
            ok = ha_set_number(hp["entity"], hp["target"]) if hp["entity"] else False
            if hp["entity"] and ok:
                hvac_written_entities.add(hp["entity"])
        decision["actions"].append({"type": "heat_pump", "entity": hp["entity"],
            "zone": hp.get("zone"), "sched_mode": hp.get("sched_mode"),
            "value": hp["target"], "reason": hp["reason"], "ok": ok,
            "explanation": hp.get("explanation")})
        # Second action: climate.set_temperature on the climate entity (if any)
        if hp.get("climate"):
            climate_ok = ha_service("climate", "set_temperature", {
                "entity_id": hp["climate"], "temperature": hp["target"],
            })
            decision["actions"].append({
                "type":   "climate_setpoint",
                "entity": hp["climate"],
                "zone":   hp.get("zone"),
                "value":  hp["target"],
                "reason": f"Mirror setpoint to climate entity {hp['climate']}",
                "ok":     climate_ok,
                "explanation": _explain(
                    what=f"Set {hp['climate']} target temperature to {hp['target']}°C",
                    why=(f"Zone {hp.get('zone')} has both a number entity ({hp['entity']}) and a climate "
                         f"entity ({hp['climate']}) configured. The number drives the actual hardware; "
                         f"the climate mirror keeps HA dashboards and other automations in sync."),
                    inputs=[
                        {"label": "Number entity (hardware)", "value": hp.get("entity", "—")},
                        {"label": "Climate entity (mirror)",  "value": hp["climate"]},
                        {"label": "Target temperature",       "value": f"{hp['target']}°C"},
                        {"label": "Number-write ok",          "value": str(ok)},
                    ],
                    formula="if zone.climate configured → climate.set_temperature(climate, target)",
                ),
            })
        log.info(f"  [HEAT PUMP/{hp['season'].upper()}{zone_tag}] {hp['reason']} → {hp['target']}°C")

    # 3. Pool pump + cleaner
    pool_sw  = _wiz("pool_switch",  "switch_pool",         "")
    clean_sw = _wiz("pool_cleaner", "switch_pool_cleaner", "")
    # Manual-override gate: if the user flipped the switch ON outside the
    # addon, lock it ON for the configured runtime (default 50 min) before
    # letting decide_pool weigh in again.
    manual_override = _check_pool_manual_override(pool_sw)
    pool = manual_override if manual_override else decide_pool(sensors, tariff)
    pool_expl = pool.get("explanation")
    if pool["action"]:
        if pool.get("manual_override"):
            # Don't re-trigger the limpiafondos on a manual ON — the user is
            # in charge of side-loads when they flipped the pump by hand.
            ha_switch(pool_sw, True)
            _record_pool_addon_action("on")
            decision["actions"].append({"type": "pool", "action": True,
                "reason": pool["reason"], "ok": True,
                "explanation": pool_expl})
            log.info(f"  [POOL] ON (manual) — {pool['reason']}")
        else:
            # The helper turns on the pool pump and returns the side actions
            # (limpiafondos on + scheduled auto-off in 15 min) for tracing.
            cleaner_actions = _start_pool_with_cleaner(pool_sw, clean_sw)
            _record_pool_addon_action("on")
            decision["actions"].append({"type": "pool", "action": True,
                "reason": pool["reason"] + " (+ limpiafondos 15 min)", "ok": True,
                "explanation": pool_expl})
            # Each cleaner sub-action is its own row in Activity (own type, own ▶).
            decision["actions"].extend(cleaner_actions)
            log.info(f"  [POOL] ON — {pool['reason']}")
    else:
        ha_switch(pool_sw, False)
        _record_pool_addon_action("off")
        decision["skipped"].append({"type": "pool", "reason": pool["reason"],
            "explanation": pool_expl})
        log.info(f"  [POOL] OFF — {pool['reason']}")

    # 4. Dishwasher
    dw = decide_dishwasher(sensors, tariff)
    dw_expl = dw.get("explanation")
    if dw["action"] is True:
        dw_sw = cfg("switch_dishwasher", "")
        if dw_sw:
            ok = ha_switch(dw_sw, True)
            decision["actions"].append({"type": "dishwasher", "action": "start",
                "reason": dw["reason"], "ok": ok, "explanation": dw_expl})
            log.info(f"  [DISHWASHER] START — {dw['reason']}")
        else:
            decision["skipped"].append({"type": "dishwasher",
                "reason": f"Ready ({dw['reason']}) — no control switch configured",
                "explanation": dw_expl})
    elif dw["action"] is False:
        decision["skipped"].append({"type": "dishwasher", "reason": dw["reason"],
            "explanation": dw_expl})
        log.info(f"  [DISHWASHER] Waiting — {dw['reason']}")
    else:
        log.info(f"  [DISHWASHER] {dw['reason']}")

    # 4b. Custom loads — user-defined switch-controlled devices (issue #4)
    for cl in decide_custom_loads(sensors, tariff):
        ok = ha_switch(cl["switch"], cl["want_on"])
        state_word = "ON" if cl["want_on"] else "OFF"
        decision["actions"].append({"type": "custom_load", "switch": cl["switch"],
            "name": cl["name"], "action": state_word.lower(),
            "schedule": cl["schedule"], "reason": cl["reason"], "ok": ok,
            "explanation": cl.get("explanation")})
        log.info(f"  [CUSTOM LOAD/{cl['name']}] {state_word} — {cl['reason']}")

    # 5. Washer / Dryer — state monitoring + valley recommendation
    for appliance, state_key, power_key in [
        ("washer", "washer_state", "washer_power"),
        ("dryer",  "dryer_state",  "dryer_power"),
    ]:
        state = sensors.get(state_key)
        if state is None:
            continue   # not configured in wizard
        power = sensors.get(power_key, 0)
        period = tariff["period"]
        if isinstance(state, str) and state.lower() == "running":
            log.info(f"  [{appliance.upper()}] Running — {power:.0f}W")
            decision["actions"].append({"type": appliance, "action": "running",
                "power_w": power, "reason": "In cycle"})
        elif isinstance(state, str) and state.lower() in ("ready", "standby", "idle"):
            if period == "valley":
                decision["skipped"].append({"type": appliance,
                    "reason": f"Ready + valley tariff — good time to run"})
                log.info(f"  [{appliance.upper()}] Ready + valley — recommend manual start")
            else:
                decision["skipped"].append({"type": appliance,
                    "reason": f"Ready, waiting for valley or surplus (period={period})"})
        else:
            decision["skipped"].append({"type": appliance, "reason": f"State: {state}"})

    _update_savings(sensors, tariff)
    _save_decision(decision)
    # Quiet-mode summary: a couple of compact INFO lines at the very end,
    # AFTER all the per-category logs have been suppressed by _QUIET_FILTER.
    _emit_cycle_summary(decision)
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


def _record_event(action: dict, source: str = "event"):
    """Append a single side-effect action to decisions.json outside of the
    normal 15-min cycle, so the Activity tab still traces it.

    Used for:
      - Deferred actions fired by the scheduler (e.g. limpiafondos auto-stop
        15 min after the pool pump starts)
      - Manual triggers from API endpoints or the UI (e.g. /api/battery/charge)
      - Anything the engine does that wasn't decided as part of a `run_cycle()`

    Stored as a minimal `decision` envelope so the existing UI rendering
    works unchanged — `sensors`, `tariff`, `prediction` are intentionally
    empty (the event is detached from a cycle context). The `source` field
    distinguishes these from the regular cycle decisions in case the UI
    wants to surface them differently later.
    """
    event = {
        "timestamp":  datetime.now().isoformat(),
        "source":     source,                  # "event" | "manual_api" | "scheduler"
        "sensors":    {},
        "tariff":     {},
        "prediction": {},
        "actions":    [action],
        "skipped":    [],
    }
    _save_decision(event)

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
<link href="https://fonts.googleapis.com/css2?family=Bangers&display=swap" rel="stylesheet">
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
.hm-btn{background:var(--b);color:var(--t);border:1px solid #334155;flex-direction:column;padding:.55rem .4rem;text-align:center;justify-content:center;width:100%;transition:.15s all}
.hm-btn.active{background:rgba(56,189,248,.15);border-color:var(--a);color:var(--a)}
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
/* Expandable decision explanation row (v4.0.0) */
.exp-toggle{cursor:pointer;color:var(--g);margin-right:.35rem;font-size:.78rem;user-select:none;transition:transform .12s}
.exp-toggle:hover{filter:brightness(1.4)}
.exp-row td{background:rgba(74,222,128,.04);border-top:1px solid rgba(74,222,128,.12);padding:.55rem .8rem !important}
.exp-body{display:flex;flex-direction:column;gap:.25rem;font-size:.78rem;line-height:1.45;color:var(--t)}
.exp-body .exp-what{font-weight:600;color:var(--g)}
.exp-body .exp-why{color:var(--t);opacity:.85;margin-bottom:.2rem}
.exp-body .exp-section{margin-top:.3rem}
.exp-body .exp-section b{color:var(--m);font-weight:600;font-size:.72rem;letter-spacing:.03em;text-transform:uppercase;display:block;margin-bottom:.15rem}
.exp-body code{font-size:.72rem;background:rgba(0,0,0,.35);padding:.1rem .35rem;border-radius:.25rem;color:#a5b4fc;font-family:'Fira Code',monospace}
.exp-body ul{margin:.1rem 0 .15rem 1rem;padding:0;list-style-type:'✗ '}
.exp-body ul li{font-size:.74rem;margin:.15rem 0}
.exp-body .kv-grid{display:grid;grid-template-columns:auto 1fr;gap:.1rem .7rem;font-size:.74rem}
.exp-body .kv-grid .k{color:var(--m);font-weight:500}
.exp-body .kv-grid .v{color:var(--t)}
.tag{display:inline-block;padding:.1rem .4rem;border-radius:.25rem;font-size:.68rem;background:var(--b);color:var(--m)}
.tag.bat{background:rgba(56,189,248,.12);color:var(--a)}
.tag.hp{background:rgba(74,222,128,.12);color:var(--g)}
.tag.pool{background:rgba(251,191,36,.12);color:var(--y)}
.tag.cl{background:rgba(168,85,247,.14);color:#c084fc}                     /* custom_load — purple */
.tag.pc{background:rgba(251,191,36,.18);color:var(--y);font-style:italic}  /* pool_cleaner — yellow italic */
.tag.dw{background:rgba(244,114,182,.12);color:#f472b6}                    /* dishwasher */
.tag.cli{background:rgba(56,189,248,.18);color:var(--a);font-style:italic} /* climate_setpoint — same family as battery */
.tag.man{background:rgba(239,68,68,.14);color:var(--r);font-weight:700}    /* manual / API trigger — red, prominent */
.tag.evt{background:rgba(148,163,184,.18);color:#94a3b8;font-style:italic} /* scheduler/deferred event */
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
/* ── Wizard comic style ──────────────────────────────────────────────────── */
.wiz-wrap{max-width:740px;margin:0 auto}
.wiz-header{font-family:'Bangers',cursive;font-size:2.2rem;color:#fbbf24;text-shadow:3px 3px 0 #0f172a,-1px -1px 0 #0f172a;letter-spacing:.04em;margin-bottom:.3rem}
.wiz-subtitle{font-size:.82rem;color:var(--m);margin-bottom:.6rem}
.wiz-steps{display:flex;gap:.3rem;margin-bottom:1.2rem;overflow-x:auto;padding-bottom:.3rem}
.wiz-step-dot{display:flex;flex-direction:column;align-items:center;gap:.2rem;min-width:55px;cursor:pointer}
.wiz-step-dot .dot{width:30px;height:30px;border-radius:50%;border:3px solid var(--b);background:var(--s);display:flex;align-items:center;justify-content:center;font-size:.8rem;font-weight:700;transition:.2s;color:var(--m)}
.wiz-step-dot.done .dot{background:rgba(74,222,128,.2);border-color:var(--g);color:var(--g)}
.wiz-step-dot.active .dot{background:rgba(251,191,36,.2);border-color:var(--y);color:var(--y);box-shadow:0 0 0 3px rgba(251,191,36,.15)}
.wiz-step-dot .dot-lbl{font-size:.55rem;color:var(--m);text-align:center;max-width:55px;line-height:1.2}
.wiz-sub-dot{display:flex;flex-direction:column;align-items:center;cursor:pointer;min-width:24px}
.wiz-sub-dot .sub-circle{width:22px;height:22px;border-radius:50%;border:2px solid #475569;background:var(--s);display:flex;align-items:center;justify-content:center;font-size:.62rem;transition:.2s}
.wiz-sub-dot.active .sub-circle{border-color:var(--y);background:rgba(251,191,36,.2);box-shadow:0 0 0 2px rgba(251,191,36,.15)}
.wiz-sub-dot.done .sub-circle{border-color:var(--g);background:rgba(74,222,128,.1)}
.wiz-step-dot.done .dot-lbl,.wiz-step-dot.active .dot-lbl{color:var(--t)}
.wiz-pane{display:none}.wiz-pane.active{display:block}
.wiz-card{background:var(--s);border:3px solid var(--b);border-radius:1rem;box-shadow:4px 4px 0 var(--b);padding:1.2rem;margin-bottom:1rem;position:relative}
.wiz-card-title{font-family:'Bangers',cursive;font-size:1.35rem;color:var(--a);letter-spacing:.03em;margin-bottom:.8rem;display:flex;align-items:center;gap:.4rem}
.wiz-robot{position:absolute;top:-14px;right:12px;font-size:1.8rem;filter:drop-shadow(2px 2px 0 var(--b))}
/* Bubble — enhanced with avatar */
.bubble{display:flex;align-items:flex-start;gap:.75rem;background:#fffef0;color:#1e293b;border:3px solid #1e293b;border-radius:1rem;padding:.8rem 1rem;font-size:.95rem;font-weight:500;line-height:1.55;margin-bottom:1.3rem;position:relative;box-shadow:4px 4px 0 #1e293b}
.bubble::after{content:"";position:absolute;bottom:-15px;left:22px;border:7px solid transparent;border-top-color:#1e293b}
.bubble::before{content:"";position:absolute;bottom:-12px;left:23px;border:6px solid transparent;border-top-color:#fffef0;z-index:1}
.bub-avatar{font-size:2.2rem;line-height:1;flex-shrink:0;filter:drop-shadow(1px 1px 0 rgba(0,0,0,.3))}
.bub-text{flex:1}
.bub-name{font-size:.6rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem;opacity:.5}
/* Bolt = default (yellow/green warm) */
.bolt-bubble{display:none!important}
.nexus-bubble{display:flex}
/* Nexus mode overrides */
.nexus-bubble{display:flex;background:#1e1b4b;color:#e0e7ff;border-color:#818cf8;box-shadow:4px 4px 0 #4c1d95}
.nexus-bubble::after{border-top-color:#818cf8}
.nexus-bubble::before{border-top-color:#1e1b4b}
/* Mode toggle bar */
.wiz-mode-bar{display:flex;align-items:center;gap:.6rem;margin-bottom:.8rem;flex-wrap:wrap}
.wiz-mode-label{font-size:.68rem;color:var(--m);text-transform:uppercase;letter-spacing:.06em}
.wiz-mode-btn{font-family:'Bangers',cursive;font-size:.95rem;padding:.28rem .8rem;border:2px solid var(--b);border-radius:.4rem;cursor:pointer;background:var(--s);color:var(--m);letter-spacing:.04em;transition:.15s}
.wiz-mode-btn.bolt-active{background:#fbbf24;color:#0f172a;border-color:#fbbf24;box-shadow:2px 2px 0 #0f172a}
.wiz-mode-btn.nexus-active{background:#818cf8;color:#fff;border-color:#818cf8;box-shadow:2px 2px 0 #1e1b4b}
/* Data Quality bar */
.dq-wrap{display:flex;align-items:center;gap:.7rem;background:var(--s);border:2px solid var(--b);border-radius:.6rem;padding:.5rem .8rem;margin-bottom:.8rem}
.dq-label{font-size:.68rem;color:var(--m);white-space:nowrap}
.dq-track{flex:1;height:9px;background:var(--b);border-radius:5px;overflow:hidden}
.dq-fill{height:100%;border-radius:5px;transition:width .6s,background .6s;background:#ef4444}
.dq-score{font-family:'Bangers',cursive;font-size:1.1rem;min-width:40px;text-align:right;letter-spacing:.03em}
/* Entity status badges */
.ent-ok{color:#4ade80;font-size:.78rem;margin-left:.35rem;font-weight:700}
.ent-miss{color:#ef4444;font-size:.78rem;margin-left:.35rem;font-weight:700}
.ent-samples{font-size:.62rem;color:var(--m);margin-left:.25rem;opacity:.75}
/* Advanced-only visibility */
.adv-only{display:none}
.adv-only{display:block}
.adv-only-flex{display:none}
.adv-only-flex{display:flex}
.adv-only-grid{display:none}
.adv-only-grid{display:grid}
/* HVAC zone blocks */
.hvac-zone{background:var(--bg);border:2px solid #334155;border-radius:.7rem;padding:.8rem;margin-bottom:.6rem;position:relative}
.hvac-zone-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:.5rem}
.hvac-zone-title{font-size:.78rem;font-weight:700;color:var(--a)}
.hvac-zone-del{font-size:.72rem;color:#ef4444;cursor:pointer;background:none;border:1px solid #ef4444;border-radius:.3rem;padding:.1rem .4rem}
.hvac-sched{display:grid;grid-template-columns:repeat(12,1fr);gap:.15rem;margin-top:.4rem}
.hvac-sched-cell{background:var(--b);border-radius:.2rem;padding:.18rem .1rem;text-align:center;font-size:.56rem;color:var(--m);cursor:pointer;transition:.1s;user-select:none}
.hvac-sched-cell.comfort{background:rgba(74,222,128,.25);color:#4ade80}
.hvac-sched-cell.surplus{background:rgba(56,189,248,.2);color:var(--a)}
.hvac-sched-cell.minimum{background:rgba(100,116,139,.15);color:#94a3b8}
/* Appliance zone blocks */
.appliance-zone{background:var(--bg);border:2px solid #334155;border-radius:.7rem;padding:.7rem;margin-bottom:.5rem}
/* Grid sub-meter list */
.submeter-row{display:flex;gap:.4rem;align-items:center;margin-bottom:.4rem}
.submeter-row input{flex:1;background:var(--b);border:1px solid #475569;border-radius:.4rem;padding:.35rem .5rem;color:var(--t);font-size:.8rem}
.submeter-del{background:none;border:1px solid #ef4444;color:#ef4444;border-radius:.3rem;padding:.2rem .5rem;cursor:pointer;font-size:.72rem}
.chart-subnav{background:var(--b);color:var(--m);border:1px solid #334155;font-size:.8rem;padding:.35rem .8rem;border-radius:.4rem;cursor:pointer;transition:.15s}
.chart-subnav.active{background:rgba(56,189,248,.15);color:var(--a);border-color:var(--a)}
.chart-subnav:hover{color:var(--t)}
/* ── SVG energy flow diagram ── */
.fl-ln{fill:none;stroke-width:3;stroke-linecap:round;stroke-dasharray:9 7;transition:opacity .4s}
.fl-ln.fl-dim{opacity:.1;animation:none!important}
.fl-ln.fl-active{animation:fl-dash linear infinite}
.fl-node-bg{fill:rgba(15,23,42,.55)}
@keyframes fl-dash{to{stroke-dashoffset:-32}}
.entity-picker{background:var(--bg);border:2px solid var(--b);border-radius:.6rem;padding:.7rem;margin-bottom:.7rem}
.entity-picker-label{font-size:.72rem;color:var(--m);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.4rem;display:flex;align-items:center;justify-content:space-between}
.entity-picker-label span{color:var(--g);font-size:.65rem}
.entity-select{width:100%;background:var(--b);border:1px solid #475569;border-radius:.4rem;padding:.4rem .6rem;color:var(--t);font-size:.82rem;margin-bottom:.35rem}
.entity-cands{display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.25rem}
.entity-cand{background:rgba(56,189,248,.1);border:1px solid rgba(56,189,248,.3);color:var(--a);padding:.2rem .55rem;border-radius:.3rem;font-size:.7rem;cursor:pointer;transition:.15s}
.entity-cand:hover{background:rgba(56,189,248,.2);border-color:var(--a)}
.entity-cand.best{background:rgba(74,222,128,.12);border-color:var(--g);color:var(--g)}
.custom-load-row{background:rgba(255,255,255,.03);border:1px solid var(--b);border-radius:.5rem;padding:.6rem;margin-bottom:.4rem}.loads-subdot{display:inline-flex;align-items:center;justify-content:center;width:1.6rem;height:1.6rem;border-radius:.3rem;font-size:.85rem;background:var(--bg);border:1px solid var(--b);cursor:pointer;transition:.15s}.loads-subdot.active{border-color:var(--g);background:rgba(74,222,128,.15)}.loads-subdot:hover{border-color:var(--a)}
.ent-search-btn{background:none;border:1px solid var(--b);border-radius:.4rem;padding:.2rem .45rem;cursor:pointer;color:var(--m);font-size:.85rem;flex-shrink:0;line-height:1;transition:.15s}.ent-search-btn:hover{border-color:var(--a);color:var(--a)}.entity-picker-row{display:flex;gap:.3rem;align-items:center}.entity-picker-row input{flex:1;min-width:0}
.entity-state{font-size:.68rem;color:var(--m);margin-top:.15rem}
.hw-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:.6rem;margin-bottom:.9rem}
.hw-card{background:var(--bg);border:3px solid var(--b);border-radius:.8rem;padding:.75rem;cursor:pointer;transition:.15s;box-shadow:3px 3px 0 var(--b);text-align:center}
.hw-card:hover{border-color:var(--a);box-shadow:3px 3px 0 var(--a)}
.hw-card.selected{border-color:var(--g);box-shadow:3px 3px 0 var(--g);background:rgba(74,222,128,.06)}
.hw-card .hw-icon{font-size:1.8rem;margin-bottom:.25rem}
.hw-card .hw-name{font-size:.76rem;font-weight:600;color:var(--t)}
.hw-card .hw-desc{font-size:.62rem;color:var(--m);margin-top:.12rem}
.wiz-nav{display:flex;justify-content:space-between;align-items:center;margin-top:1rem;gap:.5rem}
.wiz-pane .wiz-nav{display:none}
.wiz-progress{font-size:.72rem;color:var(--m)}
.comic-btn{font-family:'Bangers',cursive;font-size:1.1rem;letter-spacing:.05em;padding:.5rem 1.4rem;border:3px solid #1e293b;border-radius:.6rem;box-shadow:3px 3px 0 #1e293b;cursor:pointer;transition:.1s transform,.1s box-shadow}
.comic-btn:active{transform:translate(2px,2px);box-shadow:1px 1px 0 #1e293b}
.comic-btn.next{background:#fbbf24;color:#1e293b}
.comic-btn.back{background:var(--s);color:var(--t)}
.comic-btn.save{background:#4ade80;color:#0f172a}
.wiz-loc-row{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-bottom:.7rem}
.wiz-field label{font-size:.72rem;color:var(--m);display:block;margin-bottom:.2rem}
.wiz-field input{width:100%;background:var(--b);border:2px solid #475569;border-radius:.4rem;padding:.4rem .6rem;color:var(--t);font-size:.85rem}
.wiz-field input:focus{outline:none;border-color:var(--a)}
.summary-table{width:100%;border-collapse:collapse;font-size:.8rem}
.summary-table td{padding:.4rem .5rem;border-bottom:1px solid var(--b)}
.summary-table td:first-child{color:var(--m);width:40%}
.summary-table td:last-child{color:var(--t);font-weight:600}
.wiz-done-icon{font-size:3rem;text-align:center;margin:1rem 0}
</style>
</head>
<body>
<h1>⚡ Energy Optimizer <span id="ver" style="font-size:.75rem;color:var(--m);font-weight:400">__VERSION__</span></h1>
<div id="notify" class="toast"></div>
<div class="tabs">
  <button class="tab active" onclick="showTab('dashboard')">Dashboard</button>
  <button class="tab" onclick="showTab('charts')">Charts</button>
  <button class="tab" onclick="showTab('tariff')">Tariff</button>
  <button class="tab" onclick="showTab('setup')">Tweaks</button>
  <button class="tab" onclick="showTab('wizard')" style="color:#fbbf24">Wizard</button>
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
  <!-- Chart sub-nav -->
  <div style="display:flex;gap:.4rem;margin-bottom:1rem;flex-wrap:wrap">
    <button class="btn chart-subnav active" id="cn-live" onclick="chartView('live')">⚡ Live</button>
    <button class="btn chart-subnav" id="cn-day"  onclick="chartView('day')">📅 Day</button>
    <button class="btn chart-subnav" id="cn-hist" onclick="chartView('hist')">📈 History</button>
    <button class="btn btn-sm" style="margin-left:auto" onclick="refreshChartView()">🔄 Refresh</button>
  </div>

  <!-- ── LIVE ─────────────────────────────────────────────────────────── -->
  <div id="cv-live">
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin-bottom:.8rem">
      <!-- SOL -->
      <div class="card" style="text-align:center;padding:.7rem .5rem">
        <div style="font-size:1.4rem">☀️</div>
        <div id="lv-solar-w" style="font-size:1.3rem;font-weight:700;color:#facc15;line-height:1.2">—</div>
        <div id="lv-solar-avg" style="font-size:.65rem;color:#a16207;margin-top:.1rem">Ø —</div>
        <div style="font-size:.62rem;color:var(--m);margin-top:.15rem">Solar</div>
      </div>
      <!-- RED -->
      <div class="card" style="text-align:center;padding:.7rem .5rem">
        <div style="font-size:1.4rem">⚡</div>
        <div id="lv-grid-w" style="font-size:1.3rem;font-weight:700;line-height:1.2">—</div>
        <div id="lv-grid-avg" style="font-size:.65rem;color:var(--m);margin-top:.1rem">Ø —</div>
        <div style="font-size:.62rem;color:var(--m);margin-top:.15rem"><span id="lv-grid-dir">Red</span></div>
      </div>
      <!-- BAT -->
      <div class="card" style="text-align:center;padding:.7rem .5rem">
        <div style="font-size:1.4rem">🔋</div>
        <div id="lv-bat-w" style="font-size:1.3rem;font-weight:700;line-height:1.2;color:#a78bfa">—</div>
        <div id="lv-bat-avg" style="font-size:.65rem;color:#7c3aed;margin-top:.1rem">Ø —</div>
        <div style="font-size:.62rem;color:var(--m);margin-top:.15rem"><span id="lv-bat-dir">Bat</span> · <span id="lv-bat-soc" style="color:var(--g)">—</span></div>
      </div>
      <!-- CASA -->
      <div class="card" style="text-align:center;padding:.7rem .5rem">
        <div style="font-size:1.4rem">🏠</div>
        <div id="lv-house-w" style="font-size:1.3rem;font-weight:700;color:#f97316;line-height:1.2">—</div>
        <div id="lv-house-avg" style="font-size:.65rem;color:#c2410c;margin-top:.1rem">Ø —</div>
        <div style="font-size:.62rem;color:var(--m);margin-top:.15rem">Casa</div>
      </div>
    </div>
    <!-- SVG energy flow diagram -->
    <div class="card" style="padding:.7rem 1rem">
      <svg id="flow-svg" viewBox="0 0 440 270" style="width:100%;max-height:230px;display:block">
        <!-- Connection lines — endpoints stop at node borders (4px gap) -->
        <path id="fl-ln-sol-hse" class="fl-ln fl-dim" d="M220,76 L220,146" stroke="#facc15"/>
        <path id="fl-ln-sol-bat" class="fl-ln fl-dim" d="M187,76 C152,110 114,145 68,168" stroke="#facc15"/>
        <path id="fl-ln-sol-grd" class="fl-ln fl-dim" d="M253,76 C288,110 326,145 372,168" stroke="#4ade80"/>
        <path id="fl-ln-bat-hse" class="fl-ln fl-dim" d="M120,198 L148,193" stroke="#a78bfa"/>
        <path id="fl-ln-grd-hse" class="fl-ln fl-dim" d="M320,198 L292,193" stroke="#f87171"/>
        <path id="fl-ln-grd-bat" class="fl-ln fl-dim" d="M316,215 C260,252 180,252 124,215" stroke="#f87171"/>

        <!-- Solar node (top center) -->
        <rect x="165" y="14" width="110" height="58" rx="11" class="fl-node-bg" stroke="#facc15" stroke-width="1.5"/>
        <text x="220" y="36" text-anchor="middle" font-size="19">☀️</text>
        <text id="sv-solar-val" x="220" y="53" text-anchor="middle" fill="#facc15" font-size="13" font-weight="700" font-family="sans-serif">—</text>
        <text x="220" y="66" text-anchor="middle" fill="#64748b" font-size="9" font-family="sans-serif">SOLAR</text>

        <!-- House node (center, prominent) -->
        <rect x="152" y="150" width="136" height="68" rx="13" fill="rgba(249,115,22,.08)" stroke="#f97316" stroke-width="2.5"/>
        <text x="220" y="174" text-anchor="middle" font-size="22">🏠</text>
        <text id="sv-house-val" x="220" y="193" text-anchor="middle" fill="#f97316" font-size="15" font-weight="700" font-family="sans-serif">—</text>
        <text x="220" y="207" text-anchor="middle" fill="#64748b" font-size="9" font-family="sans-serif">CONSUMO CASA</text>

        <!-- Battery node (bottom left) -->
        <rect x="12" y="172" width="112" height="62" rx="11" class="fl-node-bg" stroke="#a78bfa" stroke-width="1.5"/>
        <text x="68" y="194" text-anchor="middle" font-size="18">🔋</text>
        <text id="sv-bat-val" x="68" y="212" text-anchor="middle" fill="#a78bfa" font-size="12" font-weight="700" font-family="sans-serif">—</text>
        <text id="sv-bat-lbl" x="68" y="226" text-anchor="middle" fill="#64748b" font-size="9" font-family="sans-serif">—</text>

        <!-- Grid node (bottom right) -->
        <rect x="316" y="172" width="112" height="62" rx="11" class="fl-node-bg" stroke="#94a3b8" stroke-width="1.5"/>
        <text x="372" y="194" text-anchor="middle" font-size="18">⚡</text>
        <text id="sv-grid-val" x="372" y="212" text-anchor="middle" fill="#94a3b8" font-size="12" font-weight="700" font-family="sans-serif">—</text>
        <text id="sv-grid-lbl" x="372" y="226" text-anchor="middle" fill="#64748b" font-size="9" font-family="sans-serif">—</text>

        <!-- Footer: tariff + timestamp -->
        <text id="sv-tariff" x="110" y="258" text-anchor="middle" fill="#475569" font-size="9" font-family="sans-serif">—</text>
        <text id="sv-ts"     x="330" y="258" text-anchor="middle" fill="#334155" font-size="9" font-family="sans-serif">—</text>
        <!-- Imbalance warning (only shown when sensors out of sync) -->
        <text id="sv-warn"   x="220" y="246" text-anchor="middle" fill="#fb923c" font-size="9" font-weight="600" font-family="sans-serif"></text>
      </svg>
    </div>
    <!-- 7-day averages integrated into cards above -->
    <!-- SOC bar -->
    <div class="card" style="margin-top:.7rem">
      <h2 style="margin-top:0;font-size:.85rem">🔋 Battery SOC — last 24h <span id="soc-mae" style="font-size:.72rem;color:var(--m);font-weight:400"></span></h2>
      <canvas id="socChart" style="max-height:180px"></canvas>
    </div>
  </div>

  <!-- ── DAY ──────────────────────────────────────────────────────────── -->
  <div id="cv-day" style="display:none">
    <!-- Date nav -->
    <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.8rem;flex-wrap:wrap">
      <button class="btn btn-sm" onclick="dayNav(-1)">←</button>
      <span id="day-label" style="font-weight:600;font-size:.95rem">—</span>
      <button class="btn btn-sm" onclick="dayNav(1)">→</button>
      <button class="btn btn-sm btn-g" onclick="dayNav(0)">Today</button>
    </div>
    <!-- KPI row -->
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:.5rem;margin-bottom:.8rem" id="day-kpis">
      <div class="card" style="text-align:center;padding:.6rem" title="Total solar energy captured today.&#10;Ø = all-time daily average (daylight hours only).">
        <div id="kpi-solar" style="font-size:1.1rem;font-weight:700;color:#facc15">—</div>
        <div id="kpi-solar-avg" style="font-size:.6rem;color:#a16207;min-height:.8rem"></div>
        <div style="font-size:.65rem;color:var(--m)">☀️ Solar</div>
      </div>
      <div class="card" style="text-align:center;padding:.6rem" title="Total house consumption today (solar + battery discharge + grid import).&#10;Ø = all-time daily average.">
        <div id="kpi-cons" style="font-size:1.1rem;font-weight:700;color:#f97316">—</div>
        <div id="kpi-cons-avg" style="font-size:.6rem;color:#c2410c;min-height:.8rem"></div>
        <div style="font-size:.65rem;color:var(--m)">🏠 Consumed</div>
      </div>
      <div class="card" style="text-align:center;padding:.6rem" title="Solar energy sold to the grid today.&#10;Ø = all-time daily average. High export = room to schedule more loads or charge battery more.">
        <div id="kpi-exp" style="font-size:1.1rem;font-weight:700;color:#4ade80">—</div>
        <div id="kpi-exp-avg" style="font-size:.6rem;color:#16a34a;min-height:.8rem"></div>
        <div style="font-size:.65rem;color:var(--m)">📤 Exported</div>
      </div>
      <div class="card" style="text-align:center;padding:.6rem" title="Energy bought from the grid today.&#10;Ø = all-time daily average. The optimizer minimises this during peak tariff hours.">
        <div id="kpi-imp" style="font-size:1.1rem;font-weight:700;color:#f87171">—</div>
        <div id="kpi-imp-avg" style="font-size:.6rem;color:#dc2626;min-height:.8rem"></div>
        <div style="font-size:.65rem;color:var(--m)">📥 Imported</div>
      </div>
      <div class="card" style="text-align:center;padding:.6rem" title="% of today's consumption covered by solar (direct or via battery).&#10;Ø = all-time average. 100% = zero grid imports.">
        <div id="kpi-ss" style="font-size:1.1rem;font-weight:700;color:var(--a)">—</div>
        <div id="kpi-ss-avg" style="font-size:.6rem;color:var(--m);min-height:.8rem"></div>
        <div style="font-size:.65rem;color:var(--m)">🔄 Self-suff.</div>
      </div>
    </div>
    <!-- Stacked hourly bar -->
    <div class="card">
      <h2 style="margin-top:0;font-size:.85rem">Hourly energy balance (kWh)
        <span style="font-size:.65rem;color:var(--m);font-weight:400"> above zero = sources · below zero = sinks</span>
      </h2>
      <canvas id="dayChart" style="max-height:260px"></canvas>
    </div>
    <!-- SOC day -->
    <div class="card" style="margin-top:.7rem;margin-bottom:1rem">
      <h2 style="margin-top:0;font-size:.85rem">Battery SOC during the day (%)</h2>
      <canvas id="socDayChart" style="max-height:160px"></canvas>
    </div>
  </div>

  <!-- ── HISTORY ───────────────────────────────────────────────────────── -->
  <div id="cv-hist" style="display:none">
    <!-- Average KPI summary row -->
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:.5rem;margin-bottom:.8rem">
      <div class="card" style="text-align:center;padding:.6rem" title="Average daily solar production.&#10;Top = all-time · Bottom = last 12 months.">
        <div id="hs-solar" style="font-size:1rem;font-weight:700;color:#facc15">—</div>
        <div id="hs-solar-12m" style="font-size:.6rem;color:#a16207;min-height:.8rem"></div>
        <div style="font-size:.62rem;color:var(--m)">☀️ Solar/day</div>
      </div>
      <div class="card" style="text-align:center;padding:.6rem" title="Average daily house consumption (solar + battery discharge + grid import).&#10;Top = all-time · Bottom = last 12 months.">
        <div id="hs-cons" style="font-size:1rem;font-weight:700;color:#f97316">—</div>
        <div id="hs-cons-12m" style="font-size:.6rem;color:#c2410c;min-height:.8rem"></div>
        <div style="font-size:.62rem;color:var(--m)">🏠 Consumed/day</div>
      </div>
      <div class="card" style="text-align:center;padding:.6rem" title="Average daily energy exported (sold) to the grid.&#10;High values = room to add loads or storage.">
        <div id="hs-exp" style="font-size:1rem;font-weight:700;color:#4ade80">—</div>
        <div id="hs-exp-12m" style="font-size:.6rem;color:#16a34a;min-height:.8rem"></div>
        <div style="font-size:.62rem;color:var(--m)">📤 Exported/day</div>
      </div>
      <div class="card" style="text-align:center;padding:.6rem" title="Average daily energy imported from the grid.&#10;The optimizer minimises peak-tariff imports.">
        <div id="hs-imp" style="font-size:1rem;font-weight:700;color:#f87171">—</div>
        <div id="hs-imp-12m" style="font-size:.6rem;color:#dc2626;min-height:.8rem"></div>
        <div style="font-size:.62rem;color:var(--m)">📥 Imported/day</div>
      </div>
      <div class="card" style="text-align:center;padding:.6rem" title="Average % of consumption covered by solar (directly or via battery).&#10;100% = zero grid imports.">
        <div id="hs-ss" style="font-size:1rem;font-weight:700;color:var(--a)">—</div>
        <div id="hs-ss-12m" style="font-size:.6rem;color:var(--m);min-height:.8rem"></div>
        <div style="font-size:.62rem;color:var(--m)">🔄 Self-suff.</div>
      </div>
    </div>
    <div class="grid2">
      <div class="card">
        <h2 style="margin-top:0;font-size:.85rem">☀️ Solar — last 7 days (kWh)</h2>
        <canvas id="sol7Chart" style="max-height:200px"></canvas>
      </div>
      <div class="card">
        <h2 style="margin-top:0;font-size:.85rem">💰 Daily savings — last 7 days (€)</h2>
        <canvas id="savingsChart" style="max-height:200px"></canvas>
      </div>
    </div>
    <div class="card" style="margin-bottom:.7rem">
      <h2 style="margin-top:0;font-size:.85rem">📅 Solar production — last 12 months (kWh)</h2>
      <canvas id="sol12Chart" style="max-height:220px"></canvas>
    </div>
    <!-- ROI Calculator -->
    <div class="card" style="margin-bottom:1rem">
      <h2 style="margin-top:0;font-size:.85rem">🔋 Battery expansion — ROI calculator</h2>
      <div style="font-size:.72rem;color:var(--m);margin-bottom:.8rem">
        Based on your actual average daily savings. Enter the cost and capacity of additional storage to estimate payback time.
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-bottom:.6rem">
        <div>
          <label style="font-size:.72rem;color:var(--m);display:block;margin-bottom:.25rem">Battery cost (€)</label>
          <input id="roi-cost" type="number" min="100" step="100" value="3000"
            style="background:var(--b);border:1px solid #475569;border-radius:.35rem;padding:.3rem .6rem;color:var(--t);font-size:.85rem;width:100%;box-sizing:border-box"
            oninput="calcROI()">
        </div>
        <div>
          <label style="font-size:.72rem;color:var(--m);display:block;margin-bottom:.25rem">Additional capacity (kWh)</label>
          <input id="roi-kwh" type="number" min="1" max="50" step="0.5" value="5"
            style="background:var(--b);border:1px solid #475569;border-radius:.35rem;padding:.3rem .6rem;color:var(--t);font-size:.85rem;width:100%;box-sizing:border-box"
            oninput="calcROI()">
        </div>
      </div>
      <div id="roi-result" style="font-size:.8rem;line-height:1.8;background:rgba(74,222,128,.05);border:1px solid #1e3a2f;border-radius:.5rem;padding:.7rem .9rem;display:none"></div>
    </div>
  </div>
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
    <h3>🔋 Battery health mode</h3>
    <div style="font-size:.72rem;color:var(--m);margin-bottom:.8rem">
      Controls the SOC operating range. A tighter range reduces deep discharge stress and extends battery lifetime, at a small cost in daily savings capacity.
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem">
      <button id="hm-bill" class="btn hm-btn" onclick="setHealthMode('bill_reducer')"
        title="SOC 10–95%. Uses full battery capacity. Best for minimising electricity bill.">
        <div style="font-size:1.1rem">⚡</div>
        <div style="font-size:.75rem;font-weight:700;margin:.15rem 0">Bill Reducer</div>
        <div style="font-size:.63rem;color:var(--m)">10% – 95%</div>
      </button>
      <button id="hm-opt" class="btn hm-btn" onclick="setHealthMode('optimized')"
        title="SOC 20–90%. Balanced: ~95% of savings benefit with moderate cycle protection.">
        <div style="font-size:1.1rem">⚖️</div>
        <div style="font-size:.75rem;font-weight:700;margin:.15rem 0">Optimized</div>
        <div style="font-size:.63rem;color:var(--m)">20% – 90%</div>
      </button>
      <button id="hm-guard" class="btn hm-btn" onclick="setHealthMode('battery_guard')"
        title="SOC 25–85%. Prioritises battery longevity. Recommended for batteries older than 3 years.">
        <div style="font-size:1.1rem">🛡️</div>
        <div style="font-size:.75rem;font-weight:700;margin:.15rem 0">Battery Guard</div>
        <div style="font-size:.63rem;color:var(--m)">25% – 85%</div>
      </button>
    </div>
    <div style="font-size:.7rem;color:var(--m);margin-top:.5rem;min-height:1.1rem" id="hm-desc"></div>
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

  <!-- DEBUG SECTION -->
  <div class="setup-section" style="margin-top:1rem" id="dbg-section">
    <h3>🐛 Debug — sensor resolution</h3>
    <div style="font-size:.72rem;color:var(--m);margin-bottom:.5rem">
      Shows how each sensor role resolves: wizard config → addon options → hardcoded fallback.
      Use this to diagnose why a sensor shows 0 or unavailable.
    </div>
    <button class="btn btn-sm" onclick="loadDebugSensors()">🔄 Refresh</button>
    <div id="dbg-result" style="margin-top:.7rem;overflow-x:auto"></div>
  </div>
</div>

<!-- WIZARD -->
<div id="tab-wizard" class="tab-content">
  <div class="wiz-wrap" id="wiz-wrap">
    <div class="wiz-header">Setup Wizard</div>
    <div class="wiz-subtitle">Configure your installation — the optimizer adapts to your hardware automatically</div>

    <!-- Mode toggle -->
    <div class="wiz-mode-bar">
      <span class="wiz-mode-label">Assistant:</span>
      <!-- Mode toggle removed: always full-control (Nexus) mode -->
    </div>

    <!-- Data Quality thermometer -->
    <div class="dq-wrap">
      <span class="dq-label">📊 Data quality</span>
      <div class="dq-track"><div class="dq-fill" id="dq-fill" style="width:0%"></div></div>
      <span class="dq-score" id="dq-score" style="color:#ef4444">0%</span>
    </div>

    <div class="wiz-steps" id="wiz-dots"></div>
    <!-- Global navigation bar — updated by _wizNavUpdate() on every step change -->
    <div id="wiz-nav" class="wiz-nav" style="margin-bottom:.9rem"></div>

    <!-- ── Step 0: Data Sources ─────────────────────────────────────────── -->
    <div class="wiz-pane active" id="wiz-pane-0">
      <div class="bubble bolt-bubble">
        <span class="bub-avatar">🤖</span>
        <div class="bub-text"><div class="bub-name">Bolt</div>
          Hey! I'm Bolt. First things first — where does your Home Assistant store history? More data = smarter predictions!
        </div>
      </div>
      <div class="bubble nexus-bubble">
        <span class="bub-avatar">⚡</span>
        <div class="bub-text"><div class="bub-name">Nexus</div>
          Initializing data pipeline. The prediction engine performs significantly better with long-term time-series data from InfluxDB (60+ days). HA Recorder is the fallback (14 days). Configure your source carefully.
        </div>
      </div>
      <div class="wiz-card">
        <div class="wiz-card-title">💾 History &amp; Data Source</div>
        <div class="wiz-robot">💾</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.6rem;margin-bottom:.8rem">
          <div class="hw-card" id="ds-card-influx" data-ds="influxdb" onclick="wizSelectDS('influxdb')" style="border-color:var(--g);background:rgba(74,222,128,.06)">
            <div class="hw-icon">🗄️</div>
            <div class="hw-name">InfluxDB</div>
            <div class="hw-desc">60+ days, best accuracy</div>
          </div>
          <div class="hw-card" id="ds-card-mariadb" data-ds="mariadb" onclick="wizSelectDS('mariadb')">
            <div class="hw-icon">🐬</div>
            <div class="hw-name">MariaDB / MySQL</div>
            <div class="hw-desc">Direct recorder DB access</div>
          </div>
          <div class="hw-card" id="ds-card-ha" data-ds="ha_recorder" onclick="wizSelectDS('ha_recorder')">
            <div class="hw-icon">🏠</div>
            <div class="hw-name">HA Recorder REST</div>
            <div class="hw-desc">Up to 14 days, always available</div>
          </div>
        </div>
        <!-- Already-connected banner (shown when host is saved + tested OK) -->
        <div id="wiz-influx-ok" style="display:none;background:rgba(74,222,128,.08);border:1px solid rgba(74,222,128,.3);border-radius:.5rem;padding:.6rem .8rem;margin-bottom:.6rem;align-items:center;justify-content:space-between;gap:.6rem">
          <div>
            <span style="color:var(--g);font-weight:600">✓ Connected</span>
            <span id="wiz-influx-ok-detail" style="font-size:.72rem;color:var(--m);margin-left:.5rem">—</span>
          </div>
          <button class="btn btn-sm" onclick="wizInfluxEdit()" style="flex-shrink:0">✏️ Edit</button>
        </div>
        <div id="wiz-influx-fields" style="display:none">
          <div style="display:grid;grid-template-columns:1fr auto;gap:.5rem;margin-bottom:.4rem">
            <div class="wiz-field"><label>Host / IP</label><input type="text" id="wiz-influx-host" placeholder="172.30.32.1"></div>
            <div class="wiz-field"><label>Port</label><input type="number" id="wiz-influx-port" value="8086" style="width:80px"></div>
          </div>
          <div class="wiz-field" style="margin-bottom:.4rem"><label>Database</label><input type="text" id="wiz-influx-db" placeholder="homeassistant"></div>
          <div class="wiz-field" style="margin-bottom:.4rem"><label>Username</label><input type="text" id="wiz-influx-user" placeholder="(leave blank if none)" autocomplete="off"></div>
          <div class="wiz-field" style="margin-bottom:.6rem"><label>Password</label><input type="password" id="wiz-influx-pass" placeholder="(leave blank if none)"></div>
          <button class="btn btn-sm" onclick="wizTestInflux()" id="btn-test-influx">🔌 Test connection</button>
          <span id="wiz-influx-status" style="font-size:.72rem;margin-left:.5rem;color:var(--m)"></span>
        </div>
        <!-- MariaDB direct connection (recorder backend) -->
        <div id="wiz-mariadb-ok" style="display:none;background:rgba(74,222,128,.08);border:1px solid rgba(74,222,128,.3);border-radius:.5rem;padding:.6rem .8rem;margin-bottom:.6rem;align-items:center;justify-content:space-between;gap:.6rem">
          <div>
            <span style="color:var(--g);font-weight:600">✓ Connected</span>
            <span id="wiz-mariadb-ok-detail" style="font-size:.72rem;color:var(--m);margin-left:.5rem">—</span>
          </div>
          <button class="btn btn-sm" onclick="wizMariadbEdit()" style="flex-shrink:0">✏️ Edit</button>
        </div>
        <div id="wiz-mariadb-fields" style="display:none">
          <div style="display:grid;grid-template-columns:1fr auto;gap:.5rem;margin-bottom:.4rem">
            <div class="wiz-field"><label>Host / IP</label><input type="text" id="wiz-mariadb-host" placeholder="core-mariadb"></div>
            <div class="wiz-field"><label>Port</label><input type="number" id="wiz-mariadb-port" value="3306" style="width:80px"></div>
          </div>
          <div class="wiz-field" style="margin-bottom:.4rem"><label>Database</label><input type="text" id="wiz-mariadb-db" placeholder="homeassistant"></div>
          <div class="wiz-field" style="margin-bottom:.4rem"><label>Username</label><input type="text" id="wiz-mariadb-user" placeholder="homeassistant" autocomplete="off"></div>
          <div class="wiz-field" style="margin-bottom:.6rem"><label>Password</label><input type="password" id="wiz-mariadb-pass" placeholder="(recorder DB password)"></div>
          <button class="btn btn-sm" onclick="wizTestMariadb()" id="btn-test-mariadb">🔌 Test connection</button>
          <span id="wiz-mariadb-status" style="font-size:.72rem;margin-left:.5rem;color:var(--m)"></span>
        </div>
      </div>
      <div class="wiz-nav">
        <span class="wiz-progress">Step 1 of 8</span>
        <button class="comic-btn next" onclick="wizNext()">Next →</button>
      </div>
    </div>

    <!-- ── Step 1: Location ─────────────────────────────────────────────── -->
    <div class="wiz-pane" id="wiz-pane-1">
      <div class="bubble bolt-bubble">
        <span class="bub-avatar">🤖</span>
        <div class="bub-text"><div class="bub-name">Bolt</div>
          Where is your home? I need this to calculate solar angles and sunrise/sunset times accurately.
        </div>
      </div>
      <div class="bubble nexus-bubble">
        <span class="bub-avatar">⚡</span>
        <div class="bub-text"><div class="bub-name">Nexus</div>
          Geographic coordinates are used for solar elevation geometry — the ML model uses a continuous solar proxy derived from the exact sun angle at your latitude. Precision matters here.
        </div>
      </div>
      <div class="wiz-card">
        <div class="wiz-card-title">📍 Location</div>
        <div class="wiz-robot">🤖</div>
        <div class="wiz-loc-row">
          <div class="wiz-field"><label>Latitude</label><input type="number" id="wiz-lat" step="0.0001" placeholder="40.4168"></div>
          <div class="wiz-field"><label>Longitude</label><input type="number" id="wiz-lon" step="0.0001" placeholder="-3.7038"></div>
        </div>
        <div class="wiz-field" style="margin-bottom:.7rem"><label>Timezone</label>
          <input type="text" id="wiz-tz" placeholder="Europe/Madrid">
        </div>
        <button class="btn btn-sm" onclick="wizDetectLocation()" style="margin-bottom:.5rem">🔍 Auto-detect from Home Assistant</button>
        <div id="wiz-loc-status" style="font-size:.72rem;color:var(--m);margin-top:.3rem"></div>
      </div>
      <div class="wiz-nav">
        <button class="comic-btn back" onclick="wizBack()">← Back</button>
        <span class="wiz-progress">Step 2 of 8</span>
        <button class="comic-btn next" onclick="wizNext()">Next →</button>
      </div>
    </div>

    <!-- ── Step 2: Grid ─────────────────────────────────────────────────── -->
    <div class="wiz-pane" id="wiz-pane-2">
      <div class="bubble bolt-bubble">
        <span class="bub-avatar">🤖</span>
        <div class="bub-text"><div class="bub-name">Bolt</div>
          Let's find your grid meter — the sensor that measures what you import and export from the utility.
        </div>
      </div>
      <div class="bubble nexus-bubble">
        <span class="bub-avatar">⚡</span>
        <div class="bub-text"><div class="bub-name">Nexus</div>
          The primary grid sensor provides the net import/export signal. Additional sub-meters for high-consumption devices (HVAC, EV, pool) feed the ML feature set and improve per-device predictions. Add as many as you have.
        </div>
      </div>
      <div class="wiz-card">
        <div class="wiz-card-title">⚡ Grid Power Meter</div>
        <div class="wiz-robot">⚡</div>
        <div class="entity-picker">
          <div class="entity-picker-label">
            Main grid sensor (W — import/export)
            <span id="wiz-grid-status"></span>
          </div>
          <input class="entity-select" id="wiz-grid-entity" placeholder="sensor.grid_power" oninput="wizEntityTyped('grid_power','wiz-grid-entity','wiz-grid-status','wiz-grid-state')">
          <div id="wiz-grid-cands" class="entity-cands"></div>
          <div id="wiz-grid-state" class="entity-state"></div>
        </div>
        <div style="font-size:.72rem;color:var(--m);margin-bottom:.5rem">Sign convention: <strong>negative = importing from grid</strong>, positive = exporting.</div>
        <label style="font-size:.72rem;color:var(--m);display:flex;align-items:center;gap:.5rem;margin-bottom:.7rem">
          <input type="checkbox" id="wiz-grid-flip" style="accent-color:var(--a)"> Flip sign (multiply by −1)
        </label>
        <!-- Advanced: sub-meters -->
        <div class="adv-only">
          <div style="font-size:.75rem;font-weight:600;color:var(--a);margin-bottom:.4rem">Additional power meters <span style="font-size:.62rem;color:var(--m);font-weight:400">(enrich ML predictions)</span></div>
          <div id="wiz-submeters-list"></div>
          <button class="btn btn-sm" onclick="wizAddSubmeter()" style="margin-top:.2rem">+ Add meter</button>
        </div>
      </div>
      <div class="wiz-nav">
        <button class="comic-btn back" onclick="wizBack()">← Back</button>
        <span class="wiz-progress">Step 3 of 8</span>
        <button class="comic-btn next" onclick="wizNext()">Next →</button>
      </div>
    </div>

    <!-- ── Step 3: Solar ────────────────────────────────────────────────── -->
    <div class="wiz-pane" id="wiz-pane-3">
      <div class="bubble bolt-bubble">
        <span class="bub-avatar">🤖</span>
        <div class="bub-text"><div class="bub-name">Bolt</div>
          Do you have solar panels? Let's connect your production sensor and forecast data.
        </div>
      </div>
      <div class="bubble nexus-bubble">
        <span class="bub-avatar">⚡</span>
        <div class="bub-text"><div class="bub-name">Nexus</div>
          Real-time production feeds the energy balance equation. Hourly forecasts (Forecast.Solar / Solcast) are used by the optimizer to pre-charge the battery before expected low-solar periods. All four forecast horizons improve decisions.
        </div>
      </div>
      <div class="wiz-card">
        <div class="wiz-card-title">☀️ Solar</div>
        <div class="wiz-robot">☀️</div>
        <div class="entity-picker">
          <div class="entity-picker-label">Solar production (W, real-time) <span id="wiz-solar-status"></span></div>
          <input class="entity-select" id="wiz-solar-entity" placeholder="sensor.solar_power" oninput="wizEntityTyped('solar_power','wiz-solar-entity','wiz-solar-status','wiz-solar-state')">
          <div id="wiz-solar-cands" class="entity-cands"></div>
          <div id="wiz-solar-state" class="entity-state"></div>
        </div>
        <div class="entity-picker">
          <div class="entity-picker-label">Forecast — today (kWh) <span style="color:var(--m);font-size:.62rem">Forecast.Solar / Solcast</span></div>
          <input class="entity-select" id="wiz-fc-today" placeholder="sensor.energy_production_today" oninput="wizEntityTyped('solar_fc_today','wiz-fc-today',null,null)">
          <div id="wiz-fc-today-cands" class="entity-cands"></div>
        </div>
        <!-- Advanced: hourly forecasts -->
        <div class="adv-only">
          <div class="entity-picker">
            <div class="entity-picker-label">Forecast — tomorrow (kWh)</div>
            <input class="entity-select" id="wiz-fc-tomorrow" placeholder="sensor.energy_production_tomorrow">
            <div id="wiz-fc-tomorrow-cands" class="entity-cands"></div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem">
            <div class="entity-picker">
              <div class="entity-picker-label">Forecast — current hour</div>
              <input class="entity-select" id="wiz-fc-h0" placeholder="sensor.energy_current_hour">
              <div id="wiz-fc-h0-cands" class="entity-cands"></div>
            </div>
            <div class="entity-picker">
              <div class="entity-picker-label">Forecast — next hour</div>
              <input class="entity-select" id="wiz-fc-h1" placeholder="sensor.energy_next_hour">
              <div id="wiz-fc-h1-cands" class="entity-cands"></div>
            </div>
          </div>
        </div>
        <div style="font-size:.7rem;color:var(--m);margin-top:.3rem">Forecast sensors are optional but significantly improve charging decisions.</div>
      </div>
      <div class="wiz-nav">
        <button class="comic-btn back" onclick="wizBack()">← Back</button>
        <span class="wiz-progress">Step 4 of 8</span>
        <button class="comic-btn next" onclick="wizNext()">Next →</button>
      </div>
    </div>

    <!-- ── Step 4: Battery ──────────────────────────────────────────────── -->
    <div class="wiz-pane" id="wiz-pane-4">
      <div class="bubble bolt-bubble">
        <span class="bub-avatar">🤖</span>
        <div class="bub-text"><div class="bub-name">Bolt</div>
          Battery storage is where the magic happens! Map your SOC sensor and I'll handle the rest.
        </div>
      </div>
      <div class="bubble nexus-bubble">
        <span class="bub-avatar">⚡</span>
        <div class="bub-text"><div class="bub-name">Nexus</div>
          Battery control requires SOC + power for monitoring, plus the mode select and cutoff entities for active management. Force-charge switch activates pre-emptive grid charging before peak tariff windows. If auto-detection fails, all switches are listed below.
        </div>
      </div>
      <div class="wiz-card">
        <div class="wiz-card-title">🔋 Battery</div>
        <div class="wiz-robot">🔋</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem">
          <div class="entity-picker">
            <div class="entity-picker-label">State of charge (%) <span id="wiz-batt-soc-status"></span></div>
            <input class="entity-select" id="wiz-batt-soc" placeholder="sensor.battery_state_of_capacity" oninput="wizEntityTyped('battery_soc','wiz-batt-soc','wiz-batt-soc-status','wiz-batt-soc-state')">
            <div id="wiz-batt-soc-cands" class="entity-cands"></div>
            <div id="wiz-batt-soc-state" class="entity-state"></div>
          </div>
          <div id="wiz-batt-power-row" class="entity-picker">
            <div class="entity-picker-label">Power (W, negative=charging) <span id="wiz-batt-pow-status"></span></div>
            <input class="entity-select" id="wiz-batt-power" placeholder="sensor.battery_charge_discharge_power" oninput="wizEntityTyped('battery_power','wiz-batt-power','wiz-batt-pow-status',null)">
            <div id="wiz-batt-power-cands" class="entity-cands"></div>
          </div>
        </div>
        <label style="font-size:.72rem;color:var(--m);display:flex;align-items:center;gap:.5rem;margin:.4rem 0">
          <input type="checkbox" id="wiz-batt-split" style="accent-color:var(--a)" onchange="wizToggleBattSplit()">
          Split sensors — my inverter exposes separate charge and discharge entities (Deye, Solarman, Fronius…)
        </label>
        <div id="wiz-batt-split-row" style="display:none;grid-template-columns:1fr 1fr;gap:.5rem;margin-bottom:.3rem">
          <div class="entity-picker">
            <div class="entity-picker-label">Charge power (W, always ≥ 0) <span id="wiz-batt-chg-status"></span></div>
            <input class="entity-select" id="wiz-batt-chg" placeholder="sensor.battery_charge_power"
              oninput="wizEntityTyped('battery_charge_entity','wiz-batt-chg','wiz-batt-chg-status',null)">
            <div id="wiz-batt-chg-cands" class="entity-cands"></div>
          </div>
          <div class="entity-picker">
            <div class="entity-picker-label">Discharge power (W, always ≥ 0) <span id="wiz-batt-dis-status"></span></div>
            <input class="entity-select" id="wiz-batt-dis" placeholder="sensor.battery_discharge_power"
              oninput="wizEntityTyped('battery_discharge_entity','wiz-batt-dis','wiz-batt-dis-status',null)">
            <div id="wiz-batt-dis-cands" class="entity-cands"></div>
          </div>
        </div>
        <div style="margin-bottom:.5rem">
          <label style="font-size:.72rem;color:var(--m)">Battery capacity (kWh)</label>
          <input type="number" id="wiz-batt-cap" value="10" min="1" max="100" step="0.5" style="width:100px;background:var(--b);border:2px solid #475569;border-radius:.4rem;padding:.35rem .5rem;color:var(--t);font-size:.85rem;margin-top:.2rem">
        </div>
        <!-- Advanced: control entities -->
        <div class="adv-only">
          <div style="font-size:.75rem;font-weight:600;color:var(--a);margin:.6rem 0 .4rem">Control entities — Huawei Luna2000 / SolarEdge / generic</div>
          <div class="entity-picker">
            <div class="entity-picker-label">Working mode select <span id="wiz-batt-mode-status"></span></div>
            <input class="entity-select" id="wiz-batt-mode" placeholder="select.battery_working_mode" oninput="wizEntityTyped('battery_mode_select','wiz-batt-mode','wiz-batt-mode-status',null)">
            <div id="wiz-batt-mode-cands" class="entity-cands"></div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem">
            <div class="entity-picker">
              <div class="entity-picker-label">Charge cutoff SOC <span id="wiz-batt-cutoff-status"></span></div>
              <input class="entity-select" id="wiz-batt-cutoff" placeholder="number.battery_grid_charge_cutoff_soc" oninput="wizEntityTyped('battery_cutoff_soc','wiz-batt-cutoff','wiz-batt-cutoff-status',null)">
              <div id="wiz-batt-cutoff-cands" class="entity-cands"></div>
            </div>
            <div class="entity-picker">
              <div class="entity-picker-label">Backup SOC <span id="wiz-batt-backup-status"></span></div>
              <input class="entity-select" id="wiz-batt-backup" placeholder="number.battery_backup_soc" oninput="wizEntityTyped('battery_backup_soc','wiz-batt-backup','wiz-batt-backup-status',null)">
              <div id="wiz-batt-backup-cands" class="entity-cands"></div>
            </div>
          </div>
          <div class="entity-picker">
            <div class="entity-picker-label">Force charge switch <span id="wiz-batt-force-status"></span></div>
            <input class="entity-select" id="wiz-batt-force" placeholder="switch.battery_force_charge" oninput="wizEntityTyped('battery_force_charge','wiz-batt-force','wiz-batt-force-status',null)">
            <div id="wiz-batt-force-cands" class="entity-cands"></div>
            <div style="font-size:.65rem;color:var(--m);margin-top:.2rem">If no candidates shown, all available switches are listed — pick yours manually.</div>
          </div>
        </div>
      </div>
      <div class="wiz-nav">
        <button class="comic-btn back" onclick="wizBack()">← Back</button>
        <span class="wiz-progress">Step 5 of 8</span>
        <button class="comic-btn next" onclick="wizNext()">Next →</button>
      </div>
    </div>

    <!-- ── Step 5: Loads ────────────────────────────────────────────────── -->
    <div class="wiz-pane" id="wiz-pane-5">
      <div class="bubble bolt-bubble">
        <span class="bub-avatar">🤖</span>
        <div class="bub-text"><div class="bub-name">Bolt</div>
          Which devices should I manage? Select all that apply and I'll find the entities automatically.
        </div>
      </div>
      <div class="bubble nexus-bubble">
        <span class="bub-avatar">⚡</span>
        <div class="bub-text"><div class="bub-name">Nexus</div>
          Each HVAC zone can have independent climate entities, temperature sensors, and a full 24h schedule with three setpoint tiers: Comfort (optimal), Surplus (boost when solar overproduces), Minimum (low-tariff floor). Appliances support state sensors, power meters, and schedule windows.
        </div>
      </div>
      <div class="hw-grid" id="wiz-hw-grid">
        <div class="hw-card selected" data-hw="hvac" onclick="wizToggleHw(this)">
          <div class="hw-icon">🌡️</div><div class="hw-name">HVAC</div><div class="hw-desc">Heat pump / AC zones</div>
        </div>
        <div class="hw-card selected" data-hw="pool" onclick="wizToggleHw(this)">
          <div class="hw-icon">🏊</div><div class="hw-name">Pool</div><div class="hw-desc">Filter + cleaner</div>
        </div>
        <div class="hw-card" data-hw="dishwasher" onclick="wizToggleHw(this)">
          <div class="hw-icon">🍽️</div><div class="hw-name">Dishwasher</div><div class="hw-desc">Smart scheduling</div>
        </div>
        <div class="hw-card" data-hw="washer" onclick="wizToggleHw(this)">
          <div class="hw-icon">👕</div><div class="hw-name">Washer</div><div class="hw-desc">Washing machine</div>
        </div>
        <div class="hw-card" data-hw="dryer" onclick="wizToggleHw(this)">
          <div class="hw-icon">🌀</div><div class="hw-name">Dryer</div><div class="hw-desc">Tumble dryer</div>
        </div>
        <div class="hw-card" data-hw="ev" onclick="wizToggleHw(this)">
          <div class="hw-icon">🚗</div><div class="hw-name">EV Charger</div><div class="hw-desc">Wallbox / OCPP</div>
        </div>
        <div class="hw-card" data-hw="custom" onclick="wizToggleHw(this)">
          <div class="hw-icon">⚙️</div><div class="hw-name">Custom</div><div class="hw-desc">Any switch load</div>
        </div>
      </div>
      <div id="wiz-loads-entities"></div>
      <div class="wiz-nav" id="wiz-loads-nav">
        <button class="comic-btn back" onclick="wizBack()">← Back</button>
        <span class="wiz-progress" id="wiz-loads-progress">Step 6 of 8</span>
        <button class="comic-btn next" onclick="wizLoadsNext()">Next →</button>
      </div>
    </div>

    <!-- ── Step 6: Tariff ───────────────────────────────────────────────── -->
    <div class="wiz-pane" id="wiz-pane-6">
      <div class="bubble bolt-bubble">
        <span class="bub-avatar">🤖</span>
        <div class="bub-text"><div class="bub-name">Bolt</div>
          Tell me about your electricity contract. I'll optimize battery charging around your pricing periods.
        </div>
      </div>
      <div class="bubble nexus-bubble">
        <span class="bub-avatar">⚡</span>
        <div class="bub-text"><div class="bub-name">Nexus</div>
          Currently locked to Spain 2.0TD (P1/P2/P3 + excedentes). A global tariff sub-assistant supporting PVPC, flat-rate, TOU, and export pricing is planned for v3.1. Fine-tune the exact schedule in the Tariff tab after this wizard.
        </div>
      </div>
      <div class="wiz-card">
        <div class="wiz-card-title">💶 Electricity Tariff</div>
        <div class="wiz-robot">💶</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-bottom:.8rem">
          <div class="wiz-field"><label>Contracted power (kW)</label>
            <input type="number" id="wiz-contracted-kw" value="5.75" step="0.25" min="1" max="30">
          </div>
          <div class="wiz-field"><label>Tariff type</label>
            <select id="wiz-tariff-type" class="entity-select" style="margin-bottom:0">
              <option value="es_2td">Spain 2.0TD (active)</option>
              <option value="es_pvpc" disabled>Spain PVPC — coming in v3.1</option>
              <option value="custom" disabled>Custom — coming in v3.1</option>
            </select>
          </div>
        </div>
        <div style="margin-bottom:.5rem">
          <div class="tariff-note" style="margin-bottom:.4rem">Click a day to toggle all-day valley (green = cheap all day). Click hours to cycle: <span class="badge valley">valley</span> → <span class="badge mid">mid</span> → <span class="badge peak">peak</span>.</div>
          <div class="day-row" id="wiz-tariff-days"></div>
          <div class="timeline" id="wiz-tariff-timeline" style="margin:.5rem 0"></div>
          <div class="price-grid">
            <div class="price-field"><label>Peak (€/kWh)</label><input id="wiz-p-peak" type="number" step="0.01" min="0"></div>
            <div class="price-field"><label>Mid (€/kWh)</label><input id="wiz-p-mid" type="number" step="0.01" min="0"></div>
            <div class="price-field"><label>Valley (€/kWh)</label><input id="wiz-p-valley" type="number" step="0.01" min="0"></div>
            <div class="price-field"><label>Export (€/kWh)</label><input id="wiz-p-export" type="number" step="0.01" min="0"></div>
          </div>
          <button class="btn btn-g btn-sm" style="margin-top:.5rem" onclick="wizSaveTariff()">💾 Save tariff</button>
        </div>
        <div>
          <label style="font-size:.72rem;color:var(--m)">Weather entity (storm protection)</label>
          <input class="entity-select" id="wiz-weather" placeholder="" oninput="wizEntityTyped('weather','wiz-weather',null,null)" style="margin-top:.3rem">
          <div id="wiz-weather-cands" class="entity-cands" style="margin-top:.3rem"></div>
        </div>
      </div>
      <div class="wiz-nav">
        <button class="comic-btn back" onclick="wizBack()">← Back</button>
        <span class="wiz-progress">Step 7 of 8</span>
        <button class="comic-btn next" onclick="wizNext()">Next →</button>
      </div>
    </div>

    <!-- ── Step 7: Summary + Save ───────────────────────────────────────── -->
    <div class="wiz-pane" id="wiz-pane-7">
      <div class="bubble bolt-bubble">
        <span class="bub-avatar">🤖</span>
        <div class="bub-text"><div class="bub-name">Bolt</div>
          Looking great! Here's your configuration summary. Hit Save to activate — I'll start optimizing immediately!
        </div>
      </div>
      <div class="bubble nexus-bubble">
        <span class="bub-avatar">⚡</span>
        <div class="bub-text"><div class="bub-name">Nexus</div>
          Configuration validated. Committing to wizard_config.json. The optimizer will reload settings on the next cycle. Data quality score will update as entities are confirmed against your history source.
        </div>
      </div>
      <div class="wiz-card">
        <div class="wiz-card-title">📋 Summary</div>
        <div class="wiz-robot">📋</div>
        <table class="summary-table" id="wiz-summary-table"></table>
        <!-- Data quality & prediction forecast -->
        <div id="wiz-dq-summary" style="margin-top:.8rem;padding:.7rem;background:rgba(56,189,248,.04);border:1px solid rgba(56,189,248,.15);border-radius:.5rem;font-size:.75rem;display:none">
          <div style="font-weight:600;color:var(--a);margin-bottom:.4rem">📊 Data quality & prediction estimate</div>
          <div id="wiz-dq-detail" style="color:var(--m);line-height:1.6"></div>
        </div>
        <div id="wiz-save-status" style="font-size:.78rem;margin-top:.8rem;color:var(--m)"></div>
      </div>
      <div class="wiz-nav">
        <button class="comic-btn back" onclick="wizBack()">← Back</button>
        <span class="wiz-progress">Step 8 of 8</span>
        <button class="comic-btn save" onclick="wizSave()">Save &amp; Activate</button>
      </div>
    </div>

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
  if (name==='charts') chartView('live');
  if (name==='tariff') loadTariff();
  if (name==='setup')  { loadSetup(); loadDebugSensors(); }
  if (name==='wizard') loadWizard();
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

let _healthMode='bill_reducer';
function setHealthMode(m){
  _healthMode=m;
  const ids={bill_reducer:'hm-bill',optimized:'hm-opt',battery_guard:'hm-guard'};
  Object.entries(ids).forEach(([k,id])=>document.getElementById(id)?.classList.toggle('active',k===m));
  const desc={
    bill_reducer:  'SOC operates 10%–95%. Maximum daily savings — uses the full battery capacity every cycle.',
    optimized:     'SOC operates 20%–90%. Sweet spot for most installations: ~95% of savings benefit with moderate cycle protection.',
    battery_guard: 'SOC operates 25%–85%. Reduces deep discharge stress — recommended for batteries older than 3 years or when longevity is the priority.',
  };
  const el=document.getElementById('hm-desc');
  if(el) el.textContent=desc[m]||'';
  markDirty();
}

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
    setHealthMode(s.battery_health_mode||'bill_reducer');
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
        battery_health_mode:             _healthMode,
      })
    }).then(r=>r.json());
    if(r.ok){notify('✓ Settings saved');_dirty=false;document.getElementById('setup-hint').textContent='Saved ✓';}
    else notify('✗ Error saving settings','err');
  } catch(e){ notify('✗ Error: '+e.message,'err'); }
  finally{btn.disabled=false;btn.textContent='💾 Save settings';}
}

// ── Charts ────────────────────────────────────────────────────────────────────
let socInst=null,sol7Inst=null,sol12Inst=null,savInst=null,pwrInst=null,dayInst=null,socDayInst=null;
let _chartView='live', _chartDate=new Date().toISOString().slice(0,10), _liveTimer=null;

const CHART_DEFAULTS = {responsive:true,maintainAspectRatio:true,
  interaction:{mode:'index',intersect:false},
  plugins:{legend:{labels:{color:'#94a3b8',font:{size:10},boxWidth:10}}},
  scales:{
    x:{ticks:{color:'#94a3b8',maxTicksLimit:12,font:{size:9}},grid:{color:'#1e293b'}},
    y:{ticks:{color:'#94a3b8',font:{size:10}},grid:{color:'#334155'}}
  }
};

function _mkOpts(extra={}){
  const o=JSON.parse(JSON.stringify(CHART_DEFAULTS));
  if(extra.yMin!==undefined) o.scales.y.min=extra.yMin;
  if(extra.yMax!==undefined) o.scales.y.max=extra.yMax;
  if(extra.stacked){ o.scales.y.stacked=true; o.scales.x.stacked=true; }
  if(extra.tooltip) o.plugins.tooltip=extra.tooltip;
  return o;
}

function chartView(v){
  _chartView=v;
  ['live','day','hist'].forEach(k=>{
    document.getElementById('cv-'+k).style.display = k===v?'':'none';
    document.getElementById('cn-'+k).classList.toggle('active', k===v);
  });
  if(_liveTimer){ clearInterval(_liveTimer); _liveTimer=null; }
  if(v==='live'){ loadLive(); _liveTimer=setInterval(loadLive,30000); }
  else if(v==='day') loadDay(_chartDate);
  else loadHistory();
}

function refreshChartView(){ chartView(_chartView); }

// ── LIVE view ─────────────────────────────────────────────────────────────────
async function loadLive(){
  try {
    const [s, cd] = await Promise.all([
      fetch(BASE+'/api/status').then(r=>r.json()),
      fetch(BASE+'/api/chart-data').then(r=>r.json())
    ]);
    const ss = s.sensors||{};
    const solar  = ss.solar_power  ?? 0;
    const grid   = ss.grid_power   ?? 0;
    const bat    = ss.battery_power?? 0;
    const soc    = ss.battery_soc  ?? 0;
    // Energy balance: solar = house + bat_charge + grid_export (sign convention bat>0=chg, grid>0=exp).
    // When sensors disagree (Huawei battery reading lags during transitions / taper near full SOC),
    // the implied house comes out negative. Detect and fall back to historical baseload so we don't
    // lie with "0 W" for casa. STALE_TH = 80W tolerates rounding/sampling jitter.
    const balanceRaw = solar - grid - bat;
    const STALE_TH   = 80;
    const stale      = balanceRaw < -STALE_TH;
    const baselineW  = (s.consumption_baseline_kw ?? 0.5) * 1000;
    const house      = stale ? Math.max(50, baselineW) : Math.max(0, balanceRaw);

    const fmt = w => Math.abs(w) >= 1000 ? (w/1000).toFixed(2)+' kW' : Math.round(w)+' W';

    document.getElementById('lv-solar-w').textContent = fmt(solar);
    const lvHouseEl = document.getElementById('lv-house-w');
    lvHouseEl.textContent = fmt(house) + (stale ? ' ?' : '');
    lvHouseEl.style.color = stale ? '#fb923c' : '#f97316';
    lvHouseEl.title = stale ? `Balance: faltan ${Math.round(-balanceRaw)} W. Mostrando estimación (baseload nocturno). Suele resolverse en 1-2 ciclos cuando se actualiza la lectura desfasada (típicamente la batería).` : '';
    // bat>0=charging(green/+), bat<0=discharging(red/-). Show signed value directly.
    const batSign = bat > 50 ? '+' : bat < -50 ? '-' : '';
    const lvBatEl = document.getElementById('lv-bat-w');
    lvBatEl.textContent = batSign + fmt(Math.abs(bat)) + (stale ? ' ⚠' : '');
    document.getElementById('lv-bat-soc').textContent  = soc.toFixed(0)+'%';
    lvBatEl.style.color = stale ? '#fb923c' : (bat>50?'#4ade80':bat<-50?'#f87171':'#a78bfa');
    lvBatEl.title = stale ? 'Lectura probablemente desfasada (Huawei Luna2000 lagea cerca del 100% SOC). Suele estabilizarse en 1-2 ciclos.' : '';
    document.getElementById('lv-bat-dir').textContent  = bat>50?'↑ carga':bat<-50?'↓ desc':'idle';

    const gridEl = document.getElementById('lv-grid-w');
    // Show signed value: + export (green), - import (red)
    const gridSign = grid > 50 ? '+' : '';
    gridEl.textContent = gridSign + fmt(Math.abs(grid));
    gridEl.style.color = grid>50?'#4ade80':grid<-50?'#f87171':'#94a3b8';
    document.getElementById('lv-grid-dir').textContent = grid>50?'↑ venta':grid<-50?'↓ compra':'idle';

    // SVG flow diagram
    const spd = p => Math.max(0.35, 2.4 - Math.abs(p)/1800)+'s';
    function setLine(id, on, reverse=false, dur='1.6s'){
      const el=document.getElementById(id); if(!el) return;
      if(!on){ el.classList.add('fl-dim'); el.classList.remove('fl-active');
               el.style.animationDuration=''; el.style.animationDirection=''; return; }
      el.classList.remove('fl-dim'); el.classList.add('fl-active');
      el.style.animationDuration=dur; el.style.animationDirection=reverse?'reverse':'normal';
    }
    // Convention: bat>0=charging, bat<0=discharging, grid>0=export, grid<0=import.
    // Conservation in: solar + bat_dis + grid_imp = house + bat_chg + grid_exp = out.
    // When sensors disagree (stale==true), we trust solar+grid (revenue meter is fast,
    // inverter direct measurement is fast) and re-derive battery as the residual; this
    // produces a coherent diagram even when battery_power is lagged.
    const NOISE = 10;
    let bat_chg  = Math.max(0,  bat);
    let bat_dis  = Math.max(0, -bat);
    if (stale) {
      const batDerived = solar - grid - house;   // forced by conservation
      bat_chg = Math.max(0,  batDerived);
      bat_dis = Math.max(0, -batDerived);
    }
    const grid_imp = Math.max(0, -grid);
    const grid_exp = Math.max(0,  grid);
    // Priority allocation (sources → sinks): solar→house, bat_dis→house, grid_imp→house,
    // solar→bat_chg, solar→grid_exp, bat_dis→grid_exp, grid_imp→bat_chg.
    let rSol = solar, rHse = house, rBatChg = bat_chg, rBatDis = bat_dis,
        rGrdImp = grid_imp, rGrdExp = grid_exp;
    const f_sol_hse = Math.min(rSol, rHse);     rSol -= f_sol_hse;    rHse    -= f_sol_hse;
    const f_bat_hse = Math.min(rBatDis, rHse);  rBatDis -= f_bat_hse; rHse    -= f_bat_hse;
    const f_grd_hse = Math.min(rGrdImp, rHse);  rGrdImp -= f_grd_hse; rHse    -= f_grd_hse;
    const f_sol_bat = Math.min(rSol, rBatChg);  rSol -= f_sol_bat;    rBatChg -= f_sol_bat;
    const f_sol_grd = Math.min(rSol, rGrdExp);  rSol -= f_sol_grd;    rGrdExp -= f_sol_grd;
    const f_bat_grd = Math.min(rBatDis, rGrdExp); rBatDis -= f_bat_grd; rGrdExp -= f_bat_grd;
    const f_grd_bat = Math.min(rGrdImp, rBatChg); rGrdImp -= f_grd_bat; rBatChg -= f_grd_bat;
    // Derived state
    const charging = bat > 50, discharging = bat < -50;
    const importing = grid < -50, exporting = grid > 50;
    // Animate — sol-grd line also carries battery→grid (rare: discharging while exporting)
    setLine('fl-ln-sol-hse', f_sol_hse > NOISE, false, spd(f_sol_hse));
    setLine('fl-ln-sol-bat', f_sol_bat > NOISE, false, spd(f_sol_bat));
    setLine('fl-ln-sol-grd', (f_sol_grd + f_bat_grd) > NOISE, false, spd(f_sol_grd + f_bat_grd));
    setLine('fl-ln-bat-hse', f_bat_hse > NOISE, false, spd(f_bat_hse));
    setLine('fl-ln-grd-bat', f_grd_bat > NOISE, false, spd(f_grd_bat));
    setLine('fl-ln-grd-hse', f_grd_hse > NOISE, false, spd(f_grd_hse));

    document.getElementById('sv-solar-val').textContent = fmt(solar);
    document.getElementById('sv-house-val').textContent = fmt(house);
    document.getElementById('sv-bat-val').textContent   = fmt(Math.abs(bat));
    document.getElementById('sv-bat-lbl').textContent   = soc.toFixed(0)+'% · '+(discharging?'↓ desc':charging?'↑ carga':'idle');
    const svGridVal=document.getElementById('sv-grid-val');
    if(svGridVal){svGridVal.textContent=fmt(Math.abs(grid)); svGridVal.setAttribute('fill',grid>50?'#4ade80':grid<-50?'#f87171':'#94a3b8');}
    document.getElementById('sv-grid-lbl').textContent = exporting?'↑ exportando':importing?'↓ importando':'idle';
    const tp=s.tariff?.period||'';
    document.getElementById('sv-tariff').textContent = tp?`${tp} · ${(s.tariff?.price_kwh??0).toFixed(3)} €/kWh`:'';
    document.getElementById('sv-ts').textContent = new Date().toLocaleTimeString();
    const warnEl = document.getElementById('sv-warn');
    if (warnEl) {
      warnEl.textContent = stale ? `⚠ datos sincronizando · balance Δ ${Math.round(-balanceRaw)} W (lectura desfasada, normal cerca del 100% SOC)` : '';
    }

    // SOC 24h chart
    const sdat = cd.soc||{};
    const futL = sdat.future_labels||[], futP = sdat.future_predicted||[];
    const nP   = (sdat.labels||[]).length;
    const allL = [...(sdat.labels||[]),...futL];
    const actD = [...(sdat.actual||[]),...Array(futL.length).fill(null)];
    const preD = [...(sdat.predicted||[]),...Array(futL.length).fill(null)];
    const futD = [...Array(nP).fill(null),...futP];
    if(socInst) socInst.destroy();
    const socCtx = document.getElementById('socChart')?.getContext('2d');
    if(socCtx) socInst = new Chart(socCtx,{type:'line',
      data:{labels:allL,datasets:[
        {label:'Actual %',data:actD,borderColor:'#38bdf8',backgroundColor:'rgba(56,189,248,.08)',fill:true,tension:.3,pointRadius:0},
        {label:'Predicted %',data:preD,borderColor:'#a78bfa',backgroundColor:'transparent',fill:false,tension:.3,pointRadius:0,borderDash:[4,3]},
        {label:'Forecast %',data:futD,borderColor:'#4ade80',backgroundColor:'rgba(74,222,128,.06)',fill:true,tension:.3,pointRadius:0,borderDash:[6,3]},
      ]},
      options:{..._mkOpts({yMin:0,yMax:100}),plugins:{...CHART_DEFAULTS.plugins,
        tooltip:{callbacks:{label:ctx=>` ${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(1)??''}%`}}}}
    });
    const maeEl=document.getElementById('soc-mae');
    if(maeEl) maeEl.textContent=sdat.mae!=null?`— MAE ${sdat.mae}%`:'';

    // 7-day averages row
    // 7-day W-level averages — integrated into each card
    const av = cd.averages||{};
    const fmtW = w => w == null ? '—' : (Math.abs(w) >= 1000 ? (w/1000).toFixed(1)+' kW' : Math.round(w)+' W');
    const fmtAvg = (v, suffix) => v != null ? `Ø7d ${fmtW(v)}${suffix||''}` : 'Ø7d —';
    // SOL: avg during solar hours (non-zero samples)
    const solAvgEl = document.getElementById('lv-solar-avg');
    if(solAvgEl) solAvgEl.textContent = fmtAvg(av.avg_sol, ' (sol)');
    // RED: net avg (+ export green, - import red)
    const redAvgEl = document.getElementById('lv-grid-avg');
    if(redAvgEl) {
      redAvgEl.textContent = fmtAvg(av.avg_red);
      redAvgEl.style.color = av.avg_red > 0 ? '#16a34a' : av.avg_red < 0 ? '#dc2626' : 'var(--m)';
    }
    // BAT: net avg (+ charge purple, - discharge light)
    const batAvgEl = document.getElementById('lv-bat-avg');
    if(batAvgEl) {
      batAvgEl.textContent = fmtAvg(av.avg_bat);
      batAvgEl.style.color = av.avg_bat > 0 ? '#4ade80' : av.avg_bat < 0 ? '#f87171' : 'var(--m)';
    }
    // CASA: avg consumption
    const hsAvgEl = document.getElementById('lv-house-avg');
    if(hsAvgEl) hsAvgEl.textContent = fmtAvg(av.avg_casa);
  } catch(e){ console.warn('Live error',e); }
}

// ── DAY view ──────────────────────────────────────────────────────────────────
function dayNav(dir){
  if(dir===0){ _chartDate=new Date().toISOString().slice(0,10); }
  else {
    const d=new Date(_chartDate+'T12:00:00');
    d.setDate(d.getDate()+dir);
    _chartDate=d.toISOString().slice(0,10);
  }
  loadDay(_chartDate);
}

async function loadDay(date){
  document.getElementById('day-label').textContent = new Date(date+'T12:00:00').toLocaleDateString(undefined,{weekday:'short',year:'numeric',month:'short',day:'numeric'});
  try {
    const cd = await fetch(BASE+'/api/chart-data?date='+date).then(r=>r.json());
    const h  = cd.hourly||{};
    const kpi = h.kpi||{};

    // KPIs
    document.getElementById('kpi-solar').textContent = (kpi.solar_kwh??0).toFixed(1)+' kWh';
    document.getElementById('kpi-cons').textContent  = (kpi.consumption_kwh??0).toFixed(1)+' kWh';
    document.getElementById('kpi-exp').textContent   = (kpi.export_kwh??0).toFixed(1)+' kWh';
    document.getElementById('kpi-imp').textContent   = (kpi.import_kwh??0).toFixed(1)+' kWh';
    document.getElementById('kpi-ss').textContent    = (kpi.self_sufficiency_pct??0).toFixed(0)+'%';

    // All-time averages below KPI values
    try {
      const av=await fetch(BASE+'/api/averages').then(r=>r.json());
      const at=av.all_time||{};
      if(at.n_days){
        const f1=v=>v!=null?'Ø '+v.toFixed(1)+' kWh':'';
        const fp=v=>v!=null?'Ø '+v.toFixed(0)+'%':'';
        document.getElementById('kpi-solar-avg').textContent = f1(at.avg_solar_kwh);
        document.getElementById('kpi-cons-avg').textContent  = f1(at.avg_consumption_kwh);
        document.getElementById('kpi-exp-avg').textContent   = f1(at.avg_export_kwh);
        document.getElementById('kpi-imp-avg').textContent   = f1(at.avg_import_kwh);
        document.getElementById('kpi-ss-avg').textContent    = fp(at.avg_self_sufficiency_pct);
      }
    } catch(_){}

    // Stacked hourly bar
    if(dayInst) dayInst.destroy();
    const dCtx = document.getElementById('dayChart')?.getContext('2d');
    if(dCtx && h.labels) {
      dayInst = new Chart(dCtx,{type:'bar',
        data:{labels:h.labels, datasets:[
          {label:'Solar (kWh)',   data:h.solar,        backgroundColor:'rgba(250,204,21,.7)', stack:'src'},
          {label:'Bat discharge', data:h.bat_discharge, backgroundColor:'rgba(167,139,250,.7)',stack:'src'},
          {label:'Grid import',  data:h.grid_import,  backgroundColor:'rgba(248,113,113,.7)',stack:'src'},
          {label:'Grid export',  data:(h.grid_export||[]).map(v=>v!=null?-v:null),  backgroundColor:'rgba(74,222,128,.6)', stack:'snk'},
          {label:'Bat charge',   data:(h.bat_charge||[]).map(v=>v!=null?-v:null),   backgroundColor:'rgba(56,189,248,.6)', stack:'snk'},
        ]},
        options:{..._mkOpts({stacked:true}),
          plugins:{...CHART_DEFAULTS.plugins,
            tooltip:{callbacks:{label:ctx=>{
              const v=Math.abs(ctx.parsed.y??0);
              return ` ${ctx.dataset.label}: ${v.toFixed(3)} kWh`;
            }}}
          }
        }
      });
    }

    // SOC line for the day
    const socHours = h.labels||[];
    const socVals  = h.soc||[];
    if(socDayInst) socDayInst.destroy();
    const sdCtx = document.getElementById('socDayChart')?.getContext('2d');
    if(sdCtx && socVals.some(v=>v!=null)){
      socDayInst = new Chart(sdCtx,{type:'line',
        data:{labels:socHours,datasets:[
          {label:'SOC %',data:socVals,borderColor:'#38bdf8',backgroundColor:'rgba(56,189,248,.1)',
           fill:true,tension:.4,pointRadius:3,pointBackgroundColor:'#38bdf8'}
        ]},
        options:_mkOpts({yMin:0,yMax:100})
      });
    }
  } catch(e){ console.warn('Day error',e); }
}

// ── ROI calculator ─────────────────────────────────────────────────────────────
let _roiDailySavings=0;
function calcROI(){
  const cost =parseFloat(document.getElementById('roi-cost')?.value)||0;
  const kwh  =parseFloat(document.getElementById('roi-kwh')?.value)||0;
  const el   =document.getElementById('roi-result');
  if(!el) return;
  if(!_roiDailySavings||_roiDailySavings<=0||!cost||!kwh){el.style.display='none';return;}
  const existCap=parseFloat(document.getElementById('rng-batt-cap')?.value)||10;
  const addedDaily=_roiDailySavings*(kwh/Math.max(existCap,1));
  const payDays=cost/Math.max(addedDaily,0.001);
  const payYears=payDays/365;
  el.style.display='block';
  el.innerHTML=
    `<b>Current avg savings:</b> €${_roiDailySavings.toFixed(2)}/day · `+
    `€${(_roiDailySavings*365).toFixed(0)}/year<br>`+
    `<b>Projected extra savings from +${kwh} kWh:</b> `+
    `+€${addedDaily.toFixed(2)}/day (proportional estimate)<br>`+
    `<b>Estimated payback:</b> `+
    (payDays<365?Math.round(payDays)+' days':payYears.toFixed(1)+' years')+`<br>`+
    `<span style="font-size:.7rem;color:var(--m)">⚠ Assumes linear savings scaling and current tariff/usage patterns.</span>`;
}

// ── HISTORY view ──────────────────────────────────────────────────────────────
async function loadHistory(){
  try {
    const cd=await fetch(BASE+'/api/chart-data').then(r=>r.json());

    // ── Solar 7-day line chart ──────────────────────────────────────────────
    if(sol7Inst) sol7Inst.destroy();
    const s7=cd.solar_7d||{labels:[],actual:[],forecast:[]};
    const s7Ctx=document.getElementById('sol7Chart')?.getContext('2d');
    if(s7Ctx){
      sol7Inst=new Chart(s7Ctx,{type:'bar',
        data:{labels:s7.labels,datasets:[
          {label:'Actual kWh',data:s7.actual,backgroundColor:'rgba(74,222,128,.6)',borderColor:'#4ade80',borderWidth:1,borderRadius:4},
          {label:'Forecast kWh',data:s7.forecast,backgroundColor:'rgba(167,139,250,.4)',borderColor:'#a78bfa',borderWidth:1,borderRadius:4},
        ]},
        options:_mkOpts({stacked:false})
      });
    }

    // ── Solar 12-month bar chart ────────────────────────────────────────────
    if(sol12Inst) sol12Inst.destroy();
    const s12=cd.solar_12m||{labels:[],actual:[],forecast:[]};
    const s12Ctx=document.getElementById('sol12Chart')?.getContext('2d');
    if(s12Ctx){
      sol12Inst=new Chart(s12Ctx,{type:'bar',
        data:{labels:s12.labels,datasets:[
          {label:'Actual kWh',data:s12.actual,backgroundColor:'rgba(74,222,128,.6)',borderColor:'#4ade80',borderWidth:1,borderRadius:4},
          {label:'Forecast kWh',data:s12.forecast,backgroundColor:'rgba(167,139,250,.4)',borderColor:'#a78bfa',borderWidth:1,borderRadius:4},
        ]},
        options:_mkOpts({stacked:false})
      });
    }

    // ── Daily savings bar chart ─────────────────────────────────────────────
    const dlPlugin={id:'dl',afterDatasetsDraw(chart){
      const ctx=chart.ctx; ctx.save();
      chart.data.datasets.forEach((ds,i)=>{
        chart.getDatasetMeta(i).data.forEach((bar,j)=>{
          const v=ds.data[j]; if(v>0.005){
            ctx.fillStyle='#e2e8f0'; ctx.font='bold 10px sans-serif';
            ctx.textAlign='center'; ctx.textBaseline='bottom';
            ctx.fillText('€'+v.toFixed(2), bar.x, bar.y-2);
          }
        });
      }); ctx.restore();
    }};
    if(savInst) savInst.destroy();
    const savEl=document.getElementById('savingsChart');
    if(savEl && cd.savings_daily?.labels?.length>0){
      savInst=new Chart(savEl.getContext('2d'),{type:'bar',
        data:{labels:cd.savings_daily.labels,datasets:[
          {label:'€ saved',data:cd.savings_daily.values,
           backgroundColor:'rgba(74,222,128,.6)',borderColor:'#4ade80',borderWidth:1,borderRadius:4}
        ]},
        options:_mkOpts({stacked:false}),
        plugins:[dlPlugin]
      });
    }
    // ── Average KPI stats block ─────────────────────────────────────────────
    try {
      const [av,sv]=await Promise.all([
        fetch(BASE+'/api/averages').then(r=>r.json()),
        fetch(BASE+'/api/savings').then(r=>r.json()),
      ]);
      const at=av.all_time||{}, l12=av.last_12m||{};
      const f1=v=>v!=null?v.toFixed(1)+' kWh':'—';
      const fp=v=>v!=null?v.toFixed(0)+'%':'—';
      const f1s=v=>v!=null?'12m: '+v.toFixed(1)+' kWh':'';
      const fps=v=>v!=null?'12m: '+v.toFixed(0)+'%':'';
      if(at.n_days){
        document.getElementById('hs-solar').textContent     =f1(at.avg_solar_kwh);
        document.getElementById('hs-cons').textContent      =f1(at.avg_consumption_kwh);
        document.getElementById('hs-exp').textContent       =f1(at.avg_export_kwh);
        document.getElementById('hs-imp').textContent       =f1(at.avg_import_kwh);
        document.getElementById('hs-ss').textContent        =fp(at.avg_self_sufficiency_pct);
        document.getElementById('hs-solar-12m').textContent =f1s(l12.avg_solar_kwh);
        document.getElementById('hs-cons-12m').textContent  =f1s(l12.avg_consumption_kwh);
        document.getElementById('hs-exp-12m').textContent   =f1s(l12.avg_export_kwh);
        document.getElementById('hs-imp-12m').textContent   =f1s(l12.avg_import_kwh);
        document.getElementById('hs-ss-12m').textContent    =fps(l12.avg_self_sufficiency_pct);
      }
      // ROI: daily savings rate from savings.json
      const since=new Date(sv.since||Date.now());
      const elapsed=Math.max(1,(Date.now()-since.getTime())/86400000);
      _roiDailySavings=(sv.total_eur_saved||0)/elapsed;
      calcROI();
    } catch(_){}
  } catch(e){ console.warn('History error',e); }
}

// ── Status & decisions ────────────────────────────────────────────────────────
const PC={peak:'peak',valley:'valley',mid:'mid'};
// Visual badge mapping for each action `type` recorded in decisions.json.
// Any type not listed here renders with the generic gray pill, so missing
// entries don't break the UI — they just look undifferentiated. The full
// set is kept in sync with what the cycle / event recorders emit:
//   battery, heat_pump, climate_setpoint, pool, pool_cleaner,
//   dishwasher, custom_load, manual_api, scheduler_event
const tagMap = {
  battery:          'bat',
  heat_pump:        'hp',
  climate_setpoint: 'cli',
  pool:             'pool',
  pool_cleaner:     'pc',
  dishwasher:       'dw',
  custom_load:      'cl',
  manual_api:       'man',
  scheduler_event:  'evt',
};

// Expand / collapse a decision's explanation panel. Used by the toggle that
// appears in front of each `reason` when the decision carries an `explanation`
// dict (v4.0.0+). Pure DOM, no framework.
function toggleExp(rowId, btn){
  const r = document.getElementById(rowId);
  if (!r) return;
  const open = r.style.display !== 'none';
  r.style.display = open ? 'none' : 'table-row';
  if (btn) btn.textContent = open ? '▶' : '▼';
}

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
        <div class="sub">Now: ${ss.solar_power>0?(ss.solar_power/1000).toFixed(2)+' kW':'—'} · Next hr: ${(ss.solar_next_hour??0).toFixed(2)} kWh</div></div>
      <div class="card"><div class="metric">${(ss.solar_tomorrow??0).toFixed(1)}</div>
        <div class="label">Solar tomorrow (kWh)</div>
        <div class="sub" id="sol-correction"></div></div>
      <div class="card"><div class="metric ${PC[t.period]}">${(t.price_kwh??0).toFixed(2)}€</div>
        <div class="label">Price/kWh &nbsp;<span class="badge ${PC[t.period]}">${t.period}</span></div>
        <div class="sub">${t.weekend?'Weekend — valley all day':''}</div></div>
      <div class="card"><div class="metric">${(ss.temp_indoor??0).toFixed(1)}°C</div>
        <div class="label">Indoor temp</div>
        <div class="sub">Outdoor: ${(ss.temp_outdoor??0).toFixed(1)}°C</div></div>
      <div class="card"><div class="metric" style="font-size:1.1rem"><span class="badge ${dwCls}">${dw}</span></div>
        <div class="label">Dishwasher</div></div>
      <div class="card">${(()=>{
        if(!m?.r2_cv_mean) return '<div class="metric" style="font-size:.9rem">Not trained</div><div class="label">ML model</div><div class="sub">Will train tonight at 3 AM</div>';
        const r=m.r2_cv_mean;
        const [qual,col]=r>=0.9?['Excellent','var(--g)']:r>=0.7?['Good','var(--g)']:r>=0.5?['Fair','var(--y)']:['Learning','var(--m)'];
        return `<div class="metric" style="font-size:1rem;color:${col}">${qual}</div>
          <div class="label">Consumption forecast</div>
          <div class="sub">Trained on ${m.n_samples||0} readings · accuracy R²=${r.toFixed(2)}</div>`;
      })()}</div>`;

    const bpow=ss.battery_power??0;
    const bdir=bpow>50?'▲ Charging':bpow<-50?'▼ Discharging':'● Idle';
    document.getElementById('batt-soc-big').textContent=Number(ss.battery_soc??0).toFixed(0)+'%';
    document.getElementById('batt-dir').textContent=bdir+'  '+Math.abs(bpow).toFixed(0)+' W';
    document.getElementById('storm-badge').style.display=storm?'inline-block':'none';
    if(opt){
      const tempStr=opt.temp_adj_kwh>0?` · temp +${opt.temp_adj_kwh}kWh (${opt.temp_outdoor}°C)`:'';
      const scStr=opt.solar_correction&&opt.solar_correction!==1?` · terrain ${(opt.solar_correction*100).toFixed(0)}%`:'';
      document.getElementById('batt-target-line').innerHTML=
        `Smart target: <span>${opt.target_soc}%</span>`
        +` — peak demand ${opt.peak_total_kwh}kWh · solar peak ${opt.solar_during_peak}kWh`
        +` · battery gap <b>${opt.needed_kwh}kWh</b>${tempStr}${scStr}`;
      const sc=opt.solar_correction??1;
      const scEl=document.getElementById('sol-correction');
      if(scEl) scEl.textContent=sc!==1?`Terrain factor: ${(sc*100).toFixed(0)}%`:'';
    }

    document.getElementById('savings-box').innerHTML=`
      <div class="savings">
        <div><div class="savings-num">€${(sv.total_eur_saved??0).toFixed(2)}</div>
          <div class="label">Estimated savings</div></div>
        <div><div class="metric" style="color:var(--g)">${(sv.total_kwh_avoided_peak??0).toFixed(1)}</div>
          <div class="label">kWh covered at peak</div></div>
        <div style="font-size:.72rem;color:var(--m)">Since ${sv.since??'--'}</div>
      </div>`;

    // v4.0.0 — each decision can carry a structured `explanation` dict. Render
    // an expandable row beneath the summary that exposes the engine's reasoning
    // (inputs, formula, calculation, alternatives_rejected). Backwards-compatible:
    // entries without `explanation` render exactly like before.
    let _expIdx = 0;
    const renderExpRow = (id, exp, isSkipped) => {
      const inputsHtml = (exp.inputs && exp.inputs.length)
        ? `<div class="exp-section"><b>Inputs</b><div class="kv-grid">${
            exp.inputs.map(i => `<div class="k">${i.label}</div><div class="v">${i.value}</div>`).join('')
          }</div></div>` : '';
      const formulaHtml = exp.formula
        ? `<div class="exp-section"><b>Formula</b><code>${exp.formula}</code></div>` : '';
      const calcHtml = exp.calculation
        ? `<div class="exp-section"><b>Calculation</b><code>${exp.calculation}</code></div>` : '';
      const altsHtml = (exp.alternatives_rejected && exp.alternatives_rejected.length)
        ? `<div class="exp-section"><b>Alternatives rejected</b><ul>${
            exp.alternatives_rejected.map(alt => `<li><b style="color:var(--t);text-transform:none;letter-spacing:0;display:inline">${alt.option}</b> — ${alt.why}</li>`).join('')
          }</ul></div>` : '';
      return `<tr id="${id}" class="exp-row" style="display:none"><td colspan="4">
        <div class="exp-body">
          <div class="exp-what">${exp.what||''}</div>
          ${exp.why ? `<div class="exp-why">${exp.why}</div>` : ''}
          ${inputsHtml}${formulaHtml}${calcHtml}${altsHtml}
        </div></td></tr>`;
    };

    const rows=d.slice().reverse().flatMap(dec=>{
      const ts=new Date(dec.timestamp).toLocaleTimeString('en',{hour:'2-digit',minute:'2-digit'});
      const buildRow = (a, isSkipped) => {
        const hasExp = a.explanation && a.explanation.what;
        const expId  = hasExp ? `exp-${_expIdx++}` : '';
        const toggle = hasExp ? `<span class="exp-toggle" onclick="toggleExp('${expId}',this)">▶</span>` : '';
        const okCell = isSkipped
          ? `<td class="skip">–</td>`
          : `<td class="${a.ok?'ok-c':'err-c'}">${a.ok?'✓':'✗'}</td>`;
        const reasonCls = isSkipped ? 'skip' : 'ok-c';
        const main = `<tr><td>${ts}</td><td><span class="tag ${tagMap[a.type]||''}">${a.type}</span></td><td class="${reasonCls}">${toggle}${a.reason}</td>${okCell}</tr>`;
        return hasExp ? [main, renderExpRow(expId, a.explanation, isSkipped)] : [main];
      };
      return [
        ...(dec.actions||[]).flatMap(a => buildRow(a, false)),
        ...(dec.skipped||[]).flatMap(a => buildRow(a, true))
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

// ── Sensor resolution debug ──────────────────────────────────────────────────
function loadDebugSensors() {
  const el = document.getElementById('dbg-result');
  el.innerHTML = '<span style="color:var(--m);font-size:.75rem">Loading…</span>';
  fetch(BASE+'/api/debug/sensors')
    .then(r => r.json())
    .then(data => {
      const ROLE_LABEL = {
        solar_power:       'Solar',
        grid_power:        'Grid (Red)',
        battery_power:     'Battery',
        battery_soc:       'Battery SoC',
        temp_outdoor:      'Temp outdoor',
        temp_indoor:       'Temp indoor',
        solar_fc_today:    'Solar FC today',
        solar_fc_tomorrow: 'Solar FC tomorrow',
        pool_switch:       'Pool switch',
      };
      const srcColor = {wizard:'#4ade80', options:'#60a5fa', fallback:'#facc15'};
      let rows = '';
      for (const [role, info] of Object.entries(data)) {
        const lbl = ROLE_LABEL[role] || role;
        const raw = info.raw_state;
        const val = info.parsed_float;
        const src = info.source;
        const eid = info.entity_id;
        let statusIcon, statusColor;
        if (raw === null || raw === 'unavailable' || raw === 'unknown') {
          statusIcon = '✗'; statusColor = '#f87171';
        } else if (val === null) {
          statusIcon = '⚠'; statusColor = '#fb923c';
        } else {
          statusIcon = '✓'; statusColor = '#4ade80';
        }
        const dispVal = val !== null
          ? (Math.abs(val) >= 1000 ? (val/1000).toFixed(2)+' k' : val.toFixed(1))
          : (raw || '—');
        rows += `<tr>
          <td style="padding:.3rem .5rem;white-space:nowrap">${lbl}</td>
          <td style="padding:.3rem .5rem;font-size:.68rem;color:#94a3b8;word-break:break-all">${eid}</td>
          <td style="padding:.3rem .5rem;text-align:center"><span style="color:${srcColor[src]||'#fff'};font-size:.7rem">${src}</span></td>
          <td style="padding:.3rem .5rem;text-align:right;font-weight:600">${dispVal}</td>
          <td style="padding:.3rem .5rem;text-align:center;font-size:1rem;color:${statusColor}">${statusIcon}</td>
        </tr>`;
      }
      el.innerHTML = `<table style="width:100%;border-collapse:collapse;font-size:.75rem">
        <thead><tr style="color:var(--m);font-size:.65rem;border-bottom:1px solid #334155">
          <th style="padding:.3rem .5rem;text-align:left">Role</th>
          <th style="padding:.3rem .5rem;text-align:left">Entity ID</th>
          <th style="padding:.3rem .5rem;text-align:center">Source</th>
          <th style="padding:.3rem .5rem;text-align:right">Value</th>
          <th style="padding:.3rem .5rem;text-align:center">OK</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
    })
    .catch(err => {
      el.innerHTML = `<span style="color:#f87171;font-size:.75rem">Error: ${err.message}</span>`;
    });
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

// ── Wizard state ──────────────────────────────────────────────────────────────
let wizMode = 'nexus';  // always advanced mode
let wizStep = 0;
let wizEntities = [];
let wizDQSamples = {};
let wizCfg = {
  location:{}, sensors:{}, hardware:['hvac','pool'], tariff:{},
  battery_capacity_kwh:10, data_source:'influxdb',
  influxdb:{host:'',port:8086,org:'homeassistant',token:'',bucket:'homeassistant'},
  mariadb:{host:'',port:3306,db:'homeassistant',username:'',password:''},
  hvac_zones:[], grid_submeters:[]
};

const WIZ_STEPS = ['Data','Location','Grid','Solar','Battery','Loads','Tariff','Done'];

// ── Assistant mode ─────────────────────────────────────────────────────────────
function wizSetMode(mode) {
  wizMode = mode;
  const wrap = document.getElementById('wiz-wrap');
  wrap.className = 'wiz-wrap' + (mode==='nexus' ? ' mode-nexus' : '');
  document.getElementById('btn-mode-bolt').className = 'wiz-mode-btn' + (mode==='bolt' ? ' bolt-active' : '');
  document.getElementById('btn-mode-nexus').className = 'wiz-mode-btn' + (mode==='nexus' ? ' nexus-active' : '');
  if (wizStep === 5) wizLoadsEnter();
}

// ── Data Quality bar ───────────────────────────────────────────────────────────
function wizUpdateDQ(score) {
  const fill = document.getElementById('dq-fill');
  const txt  = document.getElementById('dq-score');
  if (!fill) return;
  const pct = Math.max(0, Math.min(100, Math.round(score)));
  fill.style.width = pct + '%';
  fill.style.background = pct >= 70 ? '#4ade80' : pct >= 40 ? '#fbbf24' : '#ef4444';
  txt.style.color = pct >= 70 ? '#4ade80' : pct >= 40 ? '#fbbf24' : '#ef4444';
  txt.textContent = pct + '%';
}

async function wizFetchDQ() {
  try {
    const r = await fetch(BASE+'/api/wizard/data-quality').then(r=>r.json());
    if (r.score !== undefined) {
      wizUpdateDQ(r.score);
      wizDQSamples = r.samples || {};
      _wizRefreshStatusBadges();
    }
  } catch(e) {}
}

// ── Load wizard ────────────────────────────────────────────────────────────────
async function loadWizard() {
  try {
    const r = await fetch(BASE+'/api/wizard/config').then(r=>r.json());
    if (r.sensors || r.location) wizCfg = Object.assign(wizCfg, r);
    // restore influxdb sub-object
    if (r.influxdb) wizCfg.influxdb = Object.assign(wizCfg.influxdb, r.influxdb);
    if (r.mariadb)  wizCfg.mariadb  = Object.assign(wizCfg.mariadb,  r.mariadb);
    if (r.hvac_zones) wizCfg.hvac_zones = r.hvac_zones;
    if (r.grid_submeters) wizCfg.grid_submeters = r.grid_submeters;
  } catch(e) {}
  // Pre-fill influxdb from HA options when wizard config has no host
  if (!wizCfg.influxdb?.host) {
    try {
      const opts = await fetch(BASE+'/api/options').then(r=>r.json());
      const url = opts.influxdb_url || '';
      const m = url.match(/https?:\/\/([^:\/]+)(?::(\d+))?/);
      if (m) {
        wizCfg.influxdb.host = m[1];
        wizCfg.influxdb.port = m[2] ? parseInt(m[2]) : 8086;
        wizCfg.influxdb.db   = opts.influxdb_db || 'homeassistant';
        wizCfg.influxdb.username = opts.influxdb_user || '';
      }
    } catch(e) {}
  }
  _wizPopulateFields();
  try {
    const r = await fetch(BASE+'/api/ha/entities').then(r=>r.json());
    wizEntities = r.entities || [];
  } catch(e) { wizEntities = []; }
  wizRenderDots();
  wizGoTo(wizStep);
  wizFetchDQ();
}

function wizRenderDots() {
  const _HW_EMO = {hvac:'🌡️',pool:'🏊',dishwasher:'🍽️',washer:'👕',dryer:'🌀',ev:'🚗',custom:'⚙️'};
  const _selDevs = (wizCfg.hardware||[]).filter(h=>h&&h!=='');
  let html = '';
  WIZ_STEPS.forEach((lbl, i) => {
    const st = i < wizStep ? 'done' : i === wizStep ? 'active' : '';
    html += `<div class="wiz-step-dot ${st}" onclick="wizGoTo(${i})">
      <div class="dot">${i < wizStep ? '✓' : (i+1)}</div>
      <div class="dot-lbl">${lbl}</div>
    </div>`;
    if (i === 5 && _selDevs.length > 0) {
      _selDevs.forEach((hw, si) => {
        const emo  = _HW_EMO[hw] || '🔧';
        const done = wizStep > 5 || (wizStep === 5 && wizLoadsSub > si + 1);
        const act  = wizStep === 5 && wizLoadsSub === si + 1;
        html += `<div class="wiz-sub-dot ${act?'active':done?'done':''}"
          onclick="if(wizStep===5){wizLoadsGoTo(${si+1});}else{wizGoTo(5);}" title="${hw}">
          <div class="sub-circle">${emo}</div>
        </div>`;
      });
    }
  });
  document.getElementById('wiz-dots').innerHTML = html;
}

function _wizNavUpdate() {
  const nav = document.getElementById('wiz-nav');
  if (!nav) return;
  const isFirst = wizStep === 0;
  const isLast  = wizStep === WIZ_STEPS.length - 1;
  const isLoads = wizStep === 5;
  if (isLoads) return;  // loads manages its own nav via wizLoadsEnter/_wizLoadsUpdateNav
  nav.innerHTML =
    (isFirst ? '<span></span>' : `<button class="comic-btn back" onclick="wizBack()">← Back</button>`) +
    `<span class="wiz-progress">Step ${wizStep+1} of ${WIZ_STEPS.length} — ${WIZ_STEPS[wizStep]}</span>` +
    (isLast
      ? `<button class="comic-btn save" onclick="wizSave()">Save &amp; Activate</button>`
      : `<button class="comic-btn next" onclick="wizNext()">Next →</button>`);
}

function wizGoTo(step) {
  document.querySelectorAll('.wiz-pane').forEach(p=>p.classList.remove('active'));
  document.getElementById('wiz-pane-'+step).classList.add('active');
  wizStep = step;
  wizRenderDots();
  _wizNavUpdate();
  _wizShowAllCands();
  if (step === 6) wizRenderTariffInStep();
  if (step === 7) wizBuildSummary();
  setTimeout(_wizAddSearchButtons, 80);
}

function wizNext() { if(wizStep < WIZ_STEPS.length-1) wizGoTo(wizStep+1); }
function wizBack() { if(wizStep > 0) wizGoTo(wizStep-1); }

function wizRenderTariffInStep() {
  if (!tariffCfg) { loadTariff().then(()=>wizRenderTariffInStep()); return; }
  const wd = tariffCfg.weekend_days||[5,6];
  const daysEl = document.getElementById('wiz-tariff-days');
  if (daysEl) daysEl.innerHTML = DAY_NAMES.map((d,i)=>
    '<div class="day-chip'+(wd.includes(i)?' weekend':'')+'" onclick="wizTariffToggleDay('+i+')">'+
    (wd.includes(i)?'🌿 ':'')+d+'</div>'
  ).join('');
  const tlEl = document.getElementById('wiz-tariff-timeline');
  if (tlEl) {
    const {peak_hours,valley_hours} = tariffCfg;
    tlEl.innerHTML = Array.from({length:24},(_,h)=>{
      const p = peak_hours.includes(h)?'peak':valley_hours.includes(h)?'valley':'mid';
      return '<div class="t-hour '+p+'" onclick="wizTariffToggleHour('+h+')" title="'+h+':00">'+h+'</div>';
    }).join('');
  }
  ['peak','mid','valley','export'].forEach(k=>{
    const el=document.getElementById('wiz-p-'+k);
    if(el && tariffCfg.prices) el.value=tariffCfg.prices[k]||0;
  });
}

function wizTariffToggleDay(i) {
  const wd=tariffCfg.weekend_days||[5,6];
  tariffCfg.weekend_days=wd.includes(i)?wd.filter(x=>x!==i):[...wd,i].sort((a,b)=>a-b);
  wizRenderTariffInStep(); renderDayRow();
}

function wizTariffToggleHour(h) {
  const {peak_hours,valley_hours}=tariffCfg;
  const isPeak=peak_hours.includes(h), isValley=valley_hours.includes(h);
  if(isValley){ tariffCfg.valley_hours=valley_hours.filter(x=>x!==h); }
  else if(!isPeak){ tariffCfg.peak_hours=[...peak_hours,h].sort((a,b)=>a-b); }
  else { tariffCfg.peak_hours=peak_hours.filter(x=>x!==h); tariffCfg.valley_hours=[...valley_hours,h].sort((a,b)=>a-b); }
  wizRenderTariffInStep(); renderTariffTimeline();
}

async function wizSaveTariff() {
  tariffCfg.prices = {
    peak:   parseFloat(document.getElementById('wiz-p-peak').value)||0.30,
    mid:    parseFloat(document.getElementById('wiz-p-mid').value)||0.18,
    valley: parseFloat(document.getElementById('wiz-p-valley').value)||0.08,
    export: parseFloat(document.getElementById('wiz-p-export').value)||0.06,
  };
  const r=await fetch(BASE+'/api/tariff',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(tariffCfg)}).then(r=>r.json());
  notify(r.ok?'✓ Tariff saved':'✗ Error','r.ok?\'ok\':\'err\'');
}

// ── Data source step ───────────────────────────────────────────────────────────
function wizSelectDS(ds) {
  wizCfg.data_source = ds;
  ['influx','mariadb','ha'].forEach(k => {
    const c = document.getElementById('ds-card-'+k);
    if (c) { c.style.borderColor = ''; c.style.background = ''; }
  });
  const idMap = {influxdb:'ds-card-influx', mariadb:'ds-card-mariadb', ha_recorder:'ds-card-ha'};
  const card = document.getElementById(idMap[ds]);
  if (card) { card.style.borderColor = 'var(--g)'; card.style.background = 'rgba(74,222,128,.06)'; }
  // Hide all panels first, then show the selected one
  ['influx','mariadb'].forEach(k => {
    const ok = document.getElementById('wiz-'+k+'-ok');
    const fl = document.getElementById('wiz-'+k+'-fields');
    if (ok) ok.style.display = 'none';
    if (fl) fl.style.display = 'none';
  });
  if (ds === 'influxdb') {
    const host = wizCfg.influxdb?.host;
    const ft   = wizCfg.influxdb?.first_ts;
    if (host) {
      const detail = ft ? `${host} · data since ${ft}` : `${host} · checking…`;
      document.getElementById('wiz-influx-ok-detail').textContent = detail;
      const _okEl=document.getElementById('wiz-influx-ok'); if(_okEl){_okEl.style.display='flex';};
      if (!ft) setTimeout(_wizAutoTestInflux, 200);
    } else {
      document.getElementById('wiz-influx-fields').style.display = '';
    }
  } else if (ds === 'mariadb') {
    const host = wizCfg.mariadb?.host;
    const verified = wizCfg.mariadb?.verified;
    if (host && verified) {
      document.getElementById('wiz-mariadb-ok-detail').textContent = `${host} · connected`;
      const _okEl=document.getElementById('wiz-mariadb-ok'); if(_okEl){_okEl.style.display='flex';};
    } else {
      document.getElementById('wiz-mariadb-fields').style.display = '';
      const m = wizCfg.mariadb||{};
      if (m.host) document.getElementById('wiz-mariadb-host').value = m.host;
      if (m.port) document.getElementById('wiz-mariadb-port').value = m.port;
      if (m.db)   document.getElementById('wiz-mariadb-db').value   = m.db;
      if (m.username) document.getElementById('wiz-mariadb-user').value = m.username;
    }
  }
}
function wizMariadbEdit() {
  document.getElementById('wiz-mariadb-ok').style.display = 'none';
  document.getElementById('wiz-mariadb-fields').style.display = '';
  const m = wizCfg.mariadb||{};
  if (m.host) document.getElementById('wiz-mariadb-host').value = m.host;
  if (m.port) document.getElementById('wiz-mariadb-port').value = m.port;
  if (m.db)   document.getElementById('wiz-mariadb-db').value   = m.db;
  if (m.username) document.getElementById('wiz-mariadb-user').value = m.username;
}
async function wizTestMariadb() {
  const btn = document.getElementById('btn-test-mariadb');
  const st  = document.getElementById('wiz-mariadb-status');
  btn.disabled = true; st.textContent = 'Testing…';
  wizCfg.mariadb = {
    host:     document.getElementById('wiz-mariadb-host').value.trim(),
    port:     parseInt(document.getElementById('wiz-mariadb-port').value) || 3306,
    db:       document.getElementById('wiz-mariadb-db').value.trim() || 'homeassistant',
    username: document.getElementById('wiz-mariadb-user').value.trim(),
    password: document.getElementById('wiz-mariadb-pass').value,
  };
  try {
    const r = await fetch(BASE+'/api/wizard/data-quality', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({mariadb: wizCfg.mariadb, sensors: wizCfg.sensors, only: 'mariadb'})
    }).then(r=>r.json());
    if (r.ok && r.source === 'mariadb') {
      st.style.color = 'var(--g)';
      st.textContent = `✓ Connected — ${r.total_samples||0} samples found`;
      wizUpdateDQ(r.score||0);
      wizDQSamples = r.samples||{};
      wizCfg.mariadb.verified = true;
      document.getElementById('wiz-mariadb-ok-detail').textContent = `${wizCfg.mariadb.host} · ${r.total_samples||0} samples`;
      const _okEl=document.getElementById('wiz-mariadb-ok'); if(_okEl){_okEl.style.display='flex';};
      document.getElementById('wiz-mariadb-fields').style.display = 'none';
    } else {
      st.style.color = '#ef4444';
      st.textContent = '✗ ' + (r.source !== 'mariadb' ? 'Connection failed or no data found' : (r.error||'Connection failed'));
    }
  } catch(e) { st.style.color='#ef4444'; st.textContent='✗ Network error'; }
  btn.disabled = false;
}
function wizInfluxEdit() {
  document.getElementById('wiz-influx-ok').style.display = 'none';
  document.getElementById('wiz-influx-fields').style.display = '';
  const inf = wizCfg.influxdb||{};
  if (inf.host) document.getElementById('wiz-influx-host').value = inf.host;
  if (inf.port) document.getElementById('wiz-influx-port').value = inf.port;
  if (inf.db)   document.getElementById('wiz-influx-db').value   = inf.db;
  if (inf.username) document.getElementById('wiz-influx-user').value = inf.username;
}

async function wizTestInflux() {
  const btn = document.getElementById('btn-test-influx');
  const st  = document.getElementById('wiz-influx-status');
  btn.disabled = true; st.textContent = 'Testing…';
  wizCfg.influxdb = {
    host:     document.getElementById('wiz-influx-host').value.trim(),
    port:     parseInt(document.getElementById('wiz-influx-port').value) || 8086,
    db:       document.getElementById('wiz-influx-db').value.trim() || 'homeassistant',
    username: document.getElementById('wiz-influx-user').value.trim(),
    password: document.getElementById('wiz-influx-pass').value,
  };
  try {
    const r = await fetch(BASE+'/api/wizard/data-quality', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({influxdb: wizCfg.influxdb, sensors: wizCfg.sensors, only: 'influxdb'})
    }).then(r=>r.json());
    if (r.ok) {
      st.style.color = 'var(--g)';
      st.textContent = `✓ Connected — ${r.total_samples||0} samples found`;
      wizUpdateDQ(r.score||0);
      wizDQSamples = r.samples||{};
      wizCfg.influxdb.first_ts = r.first_ts || null;
      // Switch to "already connected" banner (always, as long as connection succeeded)
      const _detail = r.first_ts ? `${wizCfg.influxdb.host} · data since ${r.first_ts}` : `${wizCfg.influxdb.host} · connected (no history yet)`;
      document.getElementById('wiz-influx-ok-detail').textContent = _detail;
      const _okEl=document.getElementById('wiz-influx-ok'); if(_okEl){_okEl.style.display='flex';};
      document.getElementById('wiz-influx-fields').style.display = 'none';
    } else {
      st.style.color = '#ef4444'; st.textContent = '✗ ' + (r.error||'Connection failed');
    }
  } catch(e) { st.style.color='#ef4444'; st.textContent='✗ Network error'; }
  btn.disabled = false;
}

async function _wizAutoTestInflux() {
  try {
    const r = await fetch(BASE+'/api/wizard/data-quality', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({influxdb: wizCfg.influxdb, sensors: wizCfg.sensors})
    }).then(r=>r.json());
    const el = document.getElementById('wiz-influx-ok-detail');
    if (!el) return;
    if (r.ok) {
      wizCfg.influxdb.first_ts = r.first_ts || null;
      el.textContent = r.first_ts
        ? `${wizCfg.influxdb.host} · data since ${r.first_ts}`
        : `${wizCfg.influxdb.host} · connected (no history yet)`;
    } else {
      el.textContent = `${wizCfg.influxdb.host} · previously configured`;
    }
  } catch(e) {
    const el = document.getElementById('wiz-influx-ok-detail');
    if (el) el.textContent = (wizCfg.influxdb?.host||'') + ' · previously configured';
  }
}

// ── Location ───────────────────────────────────────────────────────────────────
async function wizDetectLocation() {
  document.getElementById('wiz-loc-status').textContent = 'Detecting…';
  try {
    const r = await fetch(BASE+'/api/ha/location').then(r=>r.json());
    if (r.latitude)  { document.getElementById('wiz-lat').value = r.latitude;  wizCfg.location.latitude  = r.latitude; }
    if (r.longitude) { document.getElementById('wiz-lon').value = r.longitude; wizCfg.location.longitude = r.longitude; }
    if (r.time_zone) { document.getElementById('wiz-tz').value  = r.time_zone; wizCfg.location.timezone  = r.time_zone; }
    document.getElementById('wiz-loc-status').textContent = r.latitude ? '✓ Location loaded from Home Assistant' : 'Not found in HA config';
  } catch(e) { document.getElementById('wiz-loc-status').textContent = 'Error contacting HA'; }
}

// ── Entity typed / selected ────────────────────────────────────────────────────
function wizEntityTyped(role, inputId, statusId, stateId) {
  const val = document.getElementById(inputId)?.value.trim() || '';
  const exactMatch = wizEntities.find(e=>e.entity_id===val);
  wizCfg.sensors[role] = exactMatch ? val : (val === '' ? '' : wizCfg.sensors[role]||'');
  _wizUpdateStatusBadge(statusId, role, exactMatch ? val : '');
  if (stateId) {
    const el = document.getElementById(stateId);
    if (el) el.textContent = exactMatch ? `Current: ${exactMatch.state} ${exactMatch.unit||''}` : '';
  }
  // Live-filter candidates as user types
  const m = WIZ_CANDS_MAP[role];
  const _candsId = m?.candsId || (inputId + '-cands');
  _wizRenderCands(role, inputId, _candsId, stateId||m?.stateId, statusId||m?.statusId);
}

function _wizUpdateStatusBadge(statusId, role, val) {
  if (!statusId) return;
  const el = document.getElementById(statusId);
  if (!el) return;
  if (!val) { el.innerHTML = '<span class="ent-miss">✗</span>'; return; }
  const samples = wizDQSamples[val] || wizDQSamples[role];
  const sampleTxt = samples ? `<span class="ent-samples">(${samples} pts)</span>` : '';
  el.innerHTML = `<span class="ent-ok">✓</span>${sampleTxt}`;
}

function _wizRefreshStatusBadges() {
  const statusMap = {
    grid_power:'wiz-grid-status', solar_power:'wiz-solar-status',
    battery_soc:'wiz-batt-soc-status', battery_power:'wiz-batt-pow-status',
    battery_mode_select:'wiz-batt-mode-status', battery_cutoff_soc:'wiz-batt-cutoff-status',
    battery_backup_soc:'wiz-batt-backup-status', battery_force_charge:'wiz-batt-force-status',
  };
  Object.entries(statusMap).forEach(([role, sid]) => {
    _wizUpdateStatusBadge(sid, role, wizCfg.sensors[role]||'');
  });
}

// ── Candidate rendering ────────────────────────────────────────────────────────
function _wizShowAllCands() {
  const stepRoles = [
    [],
    [],
    ['grid_power'],
    ['solar_power','solar_fc_today','solar_fc_tomorrow','solar_fc_h0','solar_fc_h1'],
    ['battery_soc','battery_power','battery_mode_select','battery_cutoff_soc','battery_backup_soc','battery_force_charge'],
    [],
    ['weather'],
    []
  ];
  const roleMappings = {
    grid_power:          {inputId:'wiz-grid-entity',   candsId:'wiz-grid-cands',       stateId:'wiz-grid-state',     statusId:'wiz-grid-status'},
    solar_power:         {inputId:'wiz-solar-entity',  candsId:'wiz-solar-cands',      stateId:'wiz-solar-state',    statusId:'wiz-solar-status'},
    solar_fc_today:      {inputId:'wiz-fc-today',      candsId:'wiz-fc-today-cands',   stateId:null,                 statusId:null},
    solar_fc_tomorrow:   {inputId:'wiz-fc-tomorrow',   candsId:'wiz-fc-tomorrow-cands',stateId:null,                 statusId:null},
    solar_fc_h0:         {inputId:'wiz-fc-h0',         candsId:'wiz-fc-h0-cands',      stateId:null,                 statusId:null},
    solar_fc_h1:         {inputId:'wiz-fc-h1',         candsId:'wiz-fc-h1-cands',      stateId:null,                 statusId:null},
    battery_soc:         {inputId:'wiz-batt-soc',      candsId:'wiz-batt-soc-cands',   stateId:'wiz-batt-soc-state', statusId:'wiz-batt-soc-status'},
    battery_power:       {inputId:'wiz-batt-power',    candsId:'wiz-batt-power-cands', stateId:null,                 statusId:'wiz-batt-pow-status'},
    battery_mode_select: {inputId:'wiz-batt-mode',     candsId:'wiz-batt-mode-cands',  stateId:null,                 statusId:'wiz-batt-mode-status'},
    battery_cutoff_soc:  {inputId:'wiz-batt-cutoff',   candsId:'wiz-batt-cutoff-cands',stateId:null,                 statusId:'wiz-batt-cutoff-status'},
    battery_backup_soc:  {inputId:'wiz-batt-backup',   candsId:'wiz-batt-backup-cands',stateId:null,                 statusId:'wiz-batt-backup-status'},
    battery_force_charge:{inputId:'wiz-batt-force',    candsId:'wiz-batt-force-cands', stateId:null,                 statusId:'wiz-batt-force-status'},
    weather:             {inputId:'wiz-weather',       candsId:'wiz-weather-cands',    stateId:null,                 statusId:null},
  };
  const roles = stepRoles[wizStep] || [];
  roles.forEach(role => {
    const m = roleMappings[role];
    if (!m) return;
    const inputEl = document.getElementById(m.inputId);
    const candsEl = document.getElementById(m.candsId);
    if (!inputEl || !candsEl) return;
    if (!inputEl.value && wizCfg.sensors[role]) inputEl.value = wizCfg.sensors[role];
    _wizRenderCands(role, m.inputId, m.candsId, m.stateId, m.statusId);
  });
  if (wizStep === 5) wizLoadsEnter();
  if (wizStep === 2) _wizRenderSubmeters();
}

const WIZ_ROLE_HINTS = {
  grid_power:          {kw:['grid','acometida','net','meter','import','red'],units:['W','kW'],dc:['power'],dom:['sensor']},
  solar_power:         {kw:['solar','pv','placas','inverter','photovoltaic','produccion'],units:['W','kW'],dc:['power'],dom:['sensor']},
  battery_soc:         {kw:['soc','battery','bateria','capacity','charge','estado'],units:['%'],dc:['battery'],dom:['sensor']},
  battery_power:       {kw:['battery','bateria','charge','discharge','luna','storage'],units:['W','kW'],dc:['power'],dom:['sensor']},
  solar_fc_today:      {kw:['today','hoy','production_today','forecast_today'],units:['kWh'],dc:['energy'],dom:['sensor']},
  solar_fc_tomorrow:   {kw:['tomorrow','manana','production_tomorrow'],units:['kWh'],dc:['energy'],dom:['sensor']},
  solar_fc_h0:         {kw:['current_hour','esta_hora','energy_now'],units:['kWh','W'],dc:['energy','power'],dom:['sensor']},
  solar_fc_h1:         {kw:['next_hour','proxima','energy_next'],units:['kWh','W'],dc:['energy','power'],dom:['sensor']},
  battery_mode_select: {kw:['working_mode','battery','luna','mode'],units:[],dc:[],dom:['select']},
  battery_cutoff_soc:  {kw:['cutoff','grid_charge','charge_cutoff','soc'],units:['%'],dc:[],dom:['number']},
  battery_backup_soc:  {kw:['backup','reserva','minimum','minimo'],units:['%'],dc:[],dom:['number']},
  battery_force_charge:{kw:['force_charge','forzar','force','forced','solar','storage','battery','luna','sungrow','huawei'],units:[],dc:[],dom:['switch','button','input_boolean']},
  temp_outdoor:        {kw:['outdoor','outside','exterior','outsidetemp','ambient'],units:['°C','°F'],dc:['temperature'],dom:['sensor']},
  temp_indoor:         {kw:['indoor','salon','room','living','temperatura','interior'],units:['°C','°F'],dc:['temperature'],dom:['sensor']},
  hvac_mode:           {kw:['hvac','clima','aerotermia','heat_pump','ebus','climate','bomba'],units:[],dc:[],dom:['climate','select']},
  hvac_temp_heat:      {kw:['heat','calor','heating','setpoint','manualtemp','z1manual'],units:['°C'],dc:['temperature'],dom:['number','input_number']},
  hvac_temp_cool:      {kw:['cool','frio','cooling','setpoint','coolingtemp','z1cooling'],units:['°C'],dc:['temperature'],dom:['number','input_number']},
  pool_switch:         {kw:['pool','piscina','pump','depuradora','filtro'],units:[],dc:[],dom:['switch']},
  pool_cleaner:        {kw:['cleaner','limpiafondos','robot','vacuum'],units:[],dc:[],dom:['switch']},
  dishwasher_state:    {kw:['dishwasher','lavavajillas','operation','estado'],units:[],dc:[],dom:['sensor']},
  washer_state:        {kw:['washer','washing','lavadora','wash','wm','laundry','operation'],units:[],dc:[],dom:['sensor']},
  washer_power:        {kw:['washer','lavadora','washing','laundry'],units:['W','kW'],dc:['power'],dom:['sensor']},
  dryer_state:         {kw:['dryer','secadora','dry','tumble','secar'],units:[],dc:[],dom:['sensor']},
  dryer_power:         {kw:['dryer','secadora','dry','tumble'],units:['W','kW'],dc:['power'],dom:['sensor']},
  ev_charger:          {kw:['ev','car','coche','charger','cargador','wallbox','zappi','ocpp'],units:['A','W'],dc:[],dom:['switch','number']},
  weather:             {kw:['aemet','openweather','weather','tiempo'],units:[],dc:[],dom:['weather']},
};

function _wizScoreEntities(role, forceShowSwitches) {
  const hints = WIZ_ROLE_HINTS[role] || {kw:[],units:[],dc:[],dom:[]};
  const scored = wizEntities.map(e => {
    let score = 0;
    const id   = (e.entity_id||'').toLowerCase();
    const name = (e.name||'').toLowerCase();
    const combined = id + ' ' + name;
    const dom = id.split('.')[0];
    if (hints.dom.length && !hints.dom.includes(dom)) return {...e, _score:0};
    hints.kw.forEach(k => { if(combined.includes(k.toLowerCase())) score += 25; });
    if (hints.dc.length && hints.dc.includes(e.device_class)) score += 50;
    if (hints.units.length && hints.units.some(u=>e.unit&&e.unit.includes(u))) score += 40;
    return {...e, _score:score};
  }).filter(e=>e._score>0).sort((a,b)=>b._score-a._score);
  // Fallback for force_charge: list all switches if no candidates found
  if (!scored.length && forceShowSwitches) {
    return wizEntities
      .filter(e=>(e.entity_id||'').startsWith('switch.'))
      .sort((a,b)=>(a.entity_id||'').localeCompare(b.entity_id||''))
      .map(e=>({...e, _score:1, _fallback:true}));
  }
  return scored;
}

function _wizCandHtml(e, i, isFallback, inputId, candsId, role, stateId, statusId) {
  const samp = wizDQSamples[e.entity_id] ? ` <span class="ent-samples">(${wizDQSamples[e.entity_id]}pts)</span>` : '';
  const best = i===0 && !isFallback;
  return `<span class="entity-cand${best?' best':''}" onclick="_wizSelectCand('${e.entity_id}','${inputId}','${candsId}','${role}','${stateId||''}','${statusId||''}')">
    ${best?'⭐ ':''}${e.name||e.entity_id}
    <span style="font-size:.6rem;opacity:.7">${e.entity_id} ${e.state||''} ${e.unit||''}</span>${samp}
  </span>`;
}

// Module-level role→candsId map used by wizEntityTyped for live search
const WIZ_CANDS_MAP = {
  grid_power:           {candsId:'wiz-grid-cands',          stateId:'wiz-grid-state',     statusId:'wiz-grid-status'},
  solar_power:          {candsId:'wiz-solar-cands',         stateId:'wiz-solar-state',    statusId:'wiz-solar-status'},
  solar_fc_today:       {candsId:'wiz-fc-today-cands',      stateId:null,                 statusId:null},
  solar_fc_tomorrow:    {candsId:'wiz-fc-tomorrow-cands',   stateId:null,                 statusId:null},
  solar_fc_h0:          {candsId:'wiz-fc-h0-cands',         stateId:null,                 statusId:null},
  solar_fc_h1:          {candsId:'wiz-fc-h1-cands',         stateId:null,                 statusId:null},
  battery_soc:          {candsId:'wiz-batt-soc-cands',      stateId:'wiz-batt-soc-state', statusId:'wiz-batt-soc-status'},
  battery_power:        {candsId:'wiz-batt-power-cands',    stateId:null,                 statusId:'wiz-batt-pow-status'},
  battery_mode_select:  {candsId:'wiz-batt-mode-cands',     stateId:null,                 statusId:'wiz-batt-mode-status'},
  battery_cutoff_soc:   {candsId:'wiz-batt-cutoff-cands',   stateId:null,                 statusId:'wiz-batt-cutoff-status'},
  battery_backup_soc:   {candsId:'wiz-batt-backup-cands',   stateId:null,                 statusId:'wiz-batt-backup-status'},
  battery_force_charge: {candsId:'wiz-batt-force-cands',    stateId:null,                 statusId:'wiz-batt-force-status'},
  weather:              {candsId:'wiz-weather-cands',        stateId:null,                 statusId:null},
};

function _wizRenderCands(role, inputId, candsId, stateId, statusId) {
  const candsEl = document.getElementById(candsId);
  if (!candsEl) return;
  const inputEl = document.getElementById(inputId);
  const q = (inputEl?.value||'').trim();
  // If user is typing a search query, filter all entities
  if (q && !wizEntities.find(e=>e.entity_id===q)) {
    const ql = q.toLowerCase();
    const matches = wizEntities.filter(e=>
      (e.entity_id||'').toLowerCase().includes(ql)||(e.name||'').toLowerCase().includes(ql)
    ).slice(0,20);
    candsEl.innerHTML = matches.length
      ? matches.map((e,i)=>_wizCandHtml(e,i,false,inputId,candsId,role,stateId,statusId)).join('')
      : '<span style="font-size:.68rem;color:var(--m)">No results</span>';
    return;
  }
  // Default: show top scored suggestions
  const isForceSw = role === 'battery_force_charge';
  const scored = _wizScoreEntities(role, isForceSw);
  const isFallback = scored.length && scored[0]._fallback;
  const topLabel = isFallback ? '<span style="font-size:.62rem;color:var(--y);margin-bottom:.2rem;display:block">⚠ No auto-match — all switches listed:</span>' : '';
  candsEl.innerHTML = scored.length
    ? topLabel + scored.slice(0, isFallback ? 15 : 5).map((e,i)=>_wizCandHtml(e,i,isFallback,inputId,candsId,role,stateId,statusId)).join('')
    : '<span style="font-size:.68rem;color:var(--m)">Type to search entities…</span>';
  if (inputEl && statusId) _wizUpdateStatusBadge(statusId, role, inputEl.value);
}

function _wizSelectCand(entityId, inputId, candsId, role, stateId, statusId) {
  const inputEl = document.getElementById(inputId);
  if (inputEl) inputEl.value = entityId;
  wizCfg.sensors[role] = entityId;
  if (stateId) {
    const ent = wizEntities.find(e=>e.entity_id===entityId);
    const el = document.getElementById(stateId);
    if (el && ent) el.textContent = `Current: ${ent.state} ${ent.unit||''}`;
  }
  _wizUpdateStatusBadge(statusId, role, entityId);
  document.querySelectorAll('#'+candsId+' .entity-cand').forEach(c=>{
    c.style.background = c.textContent.includes(entityId) ? 'rgba(74,222,128,.2)' : '';
  });
}

// ── Grid sub-meters (Nexus mode) ───────────────────────────────────────────────
function _wizRenderSubmeters() {
  const list = document.getElementById('wiz-submeters-list');
  if (!list) return;
  list.innerHTML = (wizCfg.grid_submeters||[]).map((m,i) =>
    `<div class="submeter-row">
      <input value="${m.name||''}" placeholder="Label (e.g. HVAC)" oninput="wizCfg.grid_submeters[${i}].name=this.value">
      <input value="${m.entity||''}" placeholder="sensor.entity_id" oninput="wizCfg.grid_submeters[${i}].entity=this.value">
      <button class="submeter-del" onclick="wizDelSubmeter(${i})">✕</button>
    </div>`
  ).join('');
}

function wizAddSubmeter() {
  wizCfg.grid_submeters = wizCfg.grid_submeters || [];
  wizCfg.grid_submeters.push({name:'',entity:''});
  _wizRenderSubmeters();
}
function wizDelSubmeter(i) {
  wizCfg.grid_submeters.splice(i,1);
  _wizRenderSubmeters();
}

// ── Loads step ─────────────────────────────────────────────────────────────────
let wizLoadsSub = 0;  // 0=selection screen, ≥1=device config index

function wizLoadsEnter() {
  wizLoadsSub = 0;
  _wizLoadsUpdateNav();
}

function _wizLoadsUpdateNav() {
  const devices = wizCfg.hardware.filter(h=>h!=='');
  const nav = document.getElementById('wiz-nav');  // use global nav
  if (!nav) return;
  const grid = document.getElementById('wiz-hw-grid');
  const ents = document.getElementById('wiz-loads-entities');
  if (wizLoadsSub === 0) {
    if (grid) grid.style.display = '';
    if (ents) ents.innerHTML = '';
    const _devIcons = {hvac:'🌡️',pool:'🏊',dishwasher:'🍽️',washer:'👕',dryer:'🌀',ev:'🚗',custom:'⚙️'};
    const _miniDots0 = devices.map(d=>`<span class="loads-subdot" title="${d}">${_devIcons[d]||d}</span>`).join('');
    nav.innerHTML = `
      <button class="comic-btn back" onclick="wizBack()">← Back</button>
      <span style="display:flex;flex-direction:column;align-items:center;gap:.15rem">
        <span style="font-size:.65rem;color:var(--m)">Step 6 — Select devices</span>
        ${_miniDots0 ? `<span style="display:flex;gap:.25rem">${_miniDots0}</span>` : ''}
      </span>
      <button class="comic-btn next" onclick="wizLoadsNext()">Configure →</button>`;
  } else {
    const dev = devices[wizLoadsSub-1];
    const label = {hvac:'HVAC',pool:'Pool',dishwasher:'Dishwasher',washer:'Washer',dryer:'Dryer',ev:'EV Charger'}[dev]||dev;
    if (grid) grid.style.display = 'none';
    _wizRenderSingleDevice(dev, ents);
    const isLast = wizLoadsSub >= devices.length;
    const _devIcons1 = {hvac:'🌡️',pool:'🏊',dishwasher:'🍽️',washer:'👕',dryer:'🌀',ev:'🚗',custom:'⚙️'};
    const _miniDots1 = devices.map((d,i)=>{
      const act = i===wizLoadsSub-1;
      return `<span class="loads-subdot${act?' active':''}" onclick="wizLoadsGoTo(${i+1})" title="${d}">${_devIcons1[d]||d}</span>`;
    }).join('');
    nav.innerHTML = `
      <button class="comic-btn back" onclick="wizLoadsBack()">← Back</button>
      <span style="display:flex;flex-direction:column;align-items:center;gap:.15rem">
        <span style="font-size:.65rem;color:var(--m)">${label} · ${wizLoadsSub}/${devices.length}</span>
        <span style="display:flex;gap:.25rem">${_miniDots1}</span>
      </span>
      ${isLast
        ? `<button class="comic-btn next" onclick="wizNext()">Next →</button>`
        : `<button class="comic-btn next" onclick="wizLoadsNext()">Next →</button>`}`;
    setTimeout(()=>{ _wizLoadsRenderCands(); _wizAddSearchButtons(); }, 60);
  }
}

function wizLoadsNext() {
  const devices = wizCfg.hardware.filter(h=>h!=='');
  if (wizLoadsSub === 0 && devices.length === 0) { wizNext(); return; }
  if (wizLoadsSub < devices.length) { wizLoadsSub++; _wizLoadsUpdateNav(); }
  else { wizNext(); }
}

function wizLoadsBack() {
  if (wizLoadsSub > 0) { wizLoadsSub--; _wizLoadsUpdateNav(); }
  else wizBack();
}
function wizLoadsGoTo(sub) { wizLoadsSub = sub; _wizLoadsUpdateNav(); wizRenderDots(); }

function _wizRenderSingleDevice(hw, container) {
  if (!container) return;
  const isAdv = wizMode === 'nexus';
  let html = '';
  if (hw === 'hvac') {
    const zones = wizCfg.hvac_zones.length ? wizCfg.hvac_zones : [{name:'Zone 1',climate:'',temp_heat:'',temp_cool:'',temp_sensor:'',sched:{}}];
    if (!wizCfg.hvac_zones.length) wizCfg.hvac_zones = zones;
    html = `<div class="wiz-card"><div class="wiz-card-title">🌡️ HVAC Zones</div>
      ${zones.map((z,zi)=>{
        const schedHtml = isAdv ? `
          <div style="font-size:.65rem;color:var(--m);margin-top:.4rem">24h — click to cycle: <span style="color:#4ade80">■ Comfort</span> <span style="color:var(--a)">■ Surplus</span> <span style="color:#94a3b8">■ Min</span></div>
          <div class="hvac-sched">${Array.from({length:24},(_,h)=>{const mode=(z.sched||{})[h]||'minimum';return `<div class="hvac-sched-cell ${mode}" title="${h}:00" onclick="wizCycleSched(${zi},${h})">${h}</div>`;}).join('')}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem;margin-top:.4rem">
            <div class="wiz-field"><label>Comfort °C</label><input type="number" value="${z.temp_comfort||21}" step="0.5" oninput="wizCfg.hvac_zones[${zi}].temp_comfort=+this.value"></div>
            <div class="wiz-field"><label>Surplus °C</label><input type="number" value="${z.temp_surplus||23}" step="0.5" oninput="wizCfg.hvac_zones[${zi}].temp_surplus=+this.value"></div>
            <div class="wiz-field"><label>Minimum °C</label><input type="number" value="${z.temp_min||18}" step="0.5" oninput="wizCfg.hvac_zones[${zi}].temp_min=+this.value"></div>
          </div>` : '';
        return `<div class="hvac-zone">
          <div class="hvac-zone-hdr"><span class="hvac-zone-title">🌡️ ${z.name||'Zone '+(zi+1)}</span>${zi>0?`<button class="hvac-zone-del" onclick="wizDelHvacZone(${zi})">Remove</button>`:''}</div>
          <input class="entity-select" value="${z.name||''}" placeholder="Zone name" style="margin-bottom:.4rem" oninput="wizCfg.hvac_zones[${zi}].name=this.value">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem">
            <div class="entity-picker"><div class="entity-picker-label">Climate entity</div>
              <input class="entity-select" id="wiz-hvac-cli-${zi}" value="${z.climate||''}" placeholder="climate.aerotermia" oninput="wizCfg.hvac_zones[${zi}].climate=this.value;wizEntityTyped('hvac_mode','wiz-hvac-cli-${zi}',null,null)">
              <div id="wiz-hvac-cli-${zi}-cands" class="entity-cands"></div></div>
            <div class="entity-picker"><div class="entity-picker-label">Temp sensor</div>
              <input class="entity-select" id="wiz-hvac-ts-${zi}" value="${z.temp_sensor||''}" placeholder="sensor.indoor_temp" oninput="wizCfg.hvac_zones[${zi}].temp_sensor=this.value;wizEntityTyped('temp_indoor','wiz-hvac-ts-${zi}',null,null)">
              <div id="wiz-hvac-ts-${zi}-cands" class="entity-cands"></div></div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-top:.3rem">
            <div class="entity-picker"><div class="entity-picker-label">Heat setpoint</div>
              <input class="entity-select" id="wiz-hvac-ht-${zi}" value="${z.temp_heat||''}" placeholder="number.ebusd_heat_setpoint" oninput="wizCfg.hvac_zones[${zi}].temp_heat=this.value;wizEntityTyped('hvac_temp_heat','wiz-hvac-ht-${zi}',null,null)">
              <div id="wiz-hvac-ht-${zi}-cands" class="entity-cands"></div></div>
            <div class="entity-picker"><div class="entity-picker-label">Cool setpoint</div>
              <input class="entity-select" id="wiz-hvac-cl-${zi}" value="${z.temp_cool||''}" placeholder="number.ebusd_cool_setpoint" oninput="wizCfg.hvac_zones[${zi}].temp_cool=this.value;wizEntityTyped('hvac_temp_cool','wiz-hvac-cl-${zi}',null,null)">
              <div id="wiz-hvac-cl-${zi}-cands" class="entity-cands"></div></div>
          </div>${schedHtml}</div>`;
      }).join('')}
      <button class="btn btn-sm" onclick="wizAddHvacZone()" style="margin-top:.4rem">+ Add zone</button>
    </div>`;
  } else if (hw === 'pool') {
    html = `<div class="wiz-card"><div class="wiz-card-title">🏊 Pool</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem">
        <div class="entity-picker"><div class="entity-picker-label">Filter pump switch</div>
          <input class="entity-select" id="wiz-pool-sw" value="${wizCfg.sensors.pool_switch||''}" placeholder="" oninput="wizCfg.sensors.pool_switch=this.value;wizEntityTyped('pool_switch','wiz-pool-sw',null,null)">
          <div id="wiz-pool-sw-cands" class="entity-cands"></div></div>
        <div class="entity-picker"><div class="entity-picker-label">Cleaner switch (optional)</div>
          <input class="entity-select" id="wiz-pool-cl" value="${wizCfg.sensors.pool_cleaner||''}" placeholder="" oninput="wizCfg.sensors.pool_cleaner=this.value;wizEntityTyped('pool_cleaner','wiz-pool-cl',null,null)">
          <div id="wiz-pool-cl-cands" class="entity-cands"></div></div>
      </div></div>`;
  } else if (hw === 'dishwasher') {
    html = _wizApplianceSection('dishwasher','🍽️','Dishwasher','dishwasher_state','washer_power');
  } else if (hw === 'washer') {
    html = _wizApplianceSection('washer','👕','Washing Machine','washer_state','washer_power');
  } else if (hw === 'dryer') {
    html = _wizApplianceSection('dryer','🌀','Dryer','dryer_state','dryer_power');
  } else if (hw === 'ev') {
    html = `<div class="wiz-card"><div class="wiz-card-title">🚗 EV Charger</div>
      <div class="entity-picker"><div class="entity-picker-label">EV charger entity</div>
        <input class="entity-select" id="wiz-ev" value="${wizCfg.sensors.ev_charger||''}" placeholder="switch.wallbox" oninput="wizCfg.sensors.ev_charger=this.value;wizEntityTyped('ev_charger','wiz-ev',null,null)">
        <div id="wiz-ev-cands" class="entity-cands"></div></div></div>`;
  } else if (hw === 'custom') {
    container.innerHTML = _wizCustomLoadsSection();
    setTimeout(_wizAddSearchButtons, 60);
    return;
  }
  container.innerHTML = html;
}

function wizToggleHw(card) {
  card.classList.toggle('selected');
  const hw = card.dataset.hw;
  if (card.classList.contains('selected')) {
    if (!wizCfg.hardware.includes(hw)) wizCfg.hardware.push(hw);
  } else {
    wizCfg.hardware = wizCfg.hardware.filter(h=>h!==hw);
  }
  // Stay on selection screen; don't auto-navigate
}

function wizRenderLoadsEntities() {
  const hw = wizCfg.hardware;
  document.querySelectorAll('.hw-card').forEach(c => c.classList.toggle('selected', hw.includes(c.dataset.hw)));
  const sections = [];
  const isAdv = wizMode === 'nexus';

  // ── HVAC ──
  if (hw.includes('hvac')) {
    const zones = wizCfg.hvac_zones.length ? wizCfg.hvac_zones : [{name:'Zone 1',climate:'',temp_heat:'',temp_cool:'',temp_sensor:'',sched:{}}];
    if (!wizCfg.hvac_zones.length) wizCfg.hvac_zones = zones;
    const zoneHtml = zones.map((z,zi) => {
      const schedHtml = isAdv ? `
        <div style="font-size:.65rem;color:var(--m);margin-top:.4rem">24h schedule — click to cycle: <span style="color:#4ade80">■ Comfort</span> <span style="color:var(--a)">■ Surplus</span> <span style="color:#94a3b8">■ Min</span></div>
        <div class="hvac-sched">${Array.from({length:24},(_,h)=>{
          const mode = (z.sched||{})[h]||'minimum';
          return `<div class="hvac-sched-cell ${mode}" title="${h}:00" onclick="wizCycleSched(${zi},${h})">${h}</div>`;
        }).join('')}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem;margin-top:.4rem">
          <div class="wiz-field"><label>Comfort °C</label><input type="number" value="${z.temp_comfort||21}" step="0.5" oninput="wizCfg.hvac_zones[${zi}].temp_comfort=+this.value"></div>
          <div class="wiz-field"><label>Surplus °C</label><input type="number" value="${z.temp_surplus||23}" step="0.5" oninput="wizCfg.hvac_zones[${zi}].temp_surplus=+this.value"></div>
          <div class="wiz-field"><label>Minimum °C</label><input type="number" value="${z.temp_min||18}" step="0.5" oninput="wizCfg.hvac_zones[${zi}].temp_min=+this.value"></div>
        </div>` : '';
      return `<div class="hvac-zone">
        <div class="hvac-zone-hdr">
          <span class="hvac-zone-title">🌡️ ${z.name||'Zone '+(zi+1)}</span>
          ${zi>0?`<button class="hvac-zone-del" onclick="wizDelHvacZone(${zi})">Remove</button>`:''}
        </div>
        <input class="entity-select" value="${z.name||''}" placeholder="Zone name (e.g. Ground floor)" style="margin-bottom:.4rem" oninput="wizCfg.hvac_zones[${zi}].name=this.value">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem">
          <div class="entity-picker"><div class="entity-picker-label">Climate entity</div>
            <input class="entity-select" id="wiz-hvac-cli-${zi}" value="${z.climate||''}" placeholder="climate.aerotermia" oninput="wizCfg.hvac_zones[${zi}].climate=this.value;_wizRenderCands('hvac_mode','wiz-hvac-cli-'+zi,'wiz-hvac-cli-'+zi+'-cands',null,null)">
            <div id="wiz-hvac-cli-${zi}-cands" class="entity-cands"></div></div>
          <div class="entity-picker"><div class="entity-picker-label">Temp sensor</div>
            <input class="entity-select" id="wiz-hvac-ts-${zi}" value="${z.temp_sensor||''}" placeholder="sensor.indoor_temp" oninput="wizCfg.hvac_zones[${zi}].temp_sensor=this.value;_wizRenderCands('temp_indoor','wiz-hvac-ts-'+zi,'wiz-hvac-ts-'+zi+'-cands',null,null)">
            <div id="wiz-hvac-ts-${zi}-cands" class="entity-cands"></div></div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-top:.3rem">
          <div class="entity-picker"><div class="entity-picker-label">Heat setpoint</div>
            <input class="entity-select" id="wiz-hvac-ht-${zi}" value="${z.temp_heat||''}" placeholder="number.ebusd_heat_setpoint" oninput="wizCfg.hvac_zones[${zi}].temp_heat=this.value;_wizRenderCands('hvac_temp_heat','wiz-hvac-ht-'+zi,'wiz-hvac-ht-'+zi+'-cands',null,null)">
            <div id="wiz-hvac-ht-${zi}-cands" class="entity-cands"></div></div>
          <div class="entity-picker"><div class="entity-picker-label">Cool setpoint</div>
            <input class="entity-select" id="wiz-hvac-cl-${zi}" value="${z.temp_cool||''}" placeholder="number.ebusd_cool_setpoint" oninput="wizCfg.hvac_zones[${zi}].temp_cool=this.value;_wizRenderCands('hvac_temp_cool','wiz-hvac-cl-'+zi,'wiz-hvac-cl-'+zi+'-cands',null,null)">
            <div id="wiz-hvac-cl-${zi}-cands" class="entity-cands"></div></div>
        </div>
        ${schedHtml}
      </div>`;
    }).join('');
    sections.push(`<div class="wiz-card" style="margin-top:.7rem">
      <div class="wiz-card-title" style="font-size:1rem">🌡️ HVAC Zones</div>
      ${zoneHtml}
      <button class="btn btn-sm" onclick="wizAddHvacZone()" style="margin-top:.4rem">+ Add zone</button>
    </div>`);
  }

  // ── Pool ──
  if (hw.includes('pool')) sections.push(`<div class="wiz-card" style="margin-top:.7rem">
    <div class="wiz-card-title" style="font-size:1rem">🏊 Pool</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem">
      <div class="entity-picker"><div class="entity-picker-label">Filter pump switch</div>
        <input class="entity-select" id="wiz-pool-sw" value="${wizCfg.sensors.pool_switch||''}" placeholder="" oninput="wizCfg.sensors.pool_switch=this.value;_wizRenderCands('pool_switch','wiz-pool-sw','wiz-pool-sw-cands',null,null)">
        <div id="wiz-pool-sw-cands" class="entity-cands"></div></div>
      <div class="entity-picker"><div class="entity-picker-label">Cleaner switch (optional)</div>
        <input class="entity-select" id="wiz-pool-cl" value="${wizCfg.sensors.pool_cleaner||''}" placeholder="" oninput="wizCfg.sensors.pool_cleaner=this.value;_wizRenderCands('pool_cleaner','wiz-pool-cl','wiz-pool-cl-cands',null,null)">
        <div id="wiz-pool-cl-cands" class="entity-cands"></div></div>
    </div></div>`);

  // ── Dishwasher ──
  if (hw.includes('dishwasher')) sections.push(_wizApplianceSection('dishwasher','🍽️','Dishwasher','dishwasher_state','washer_power'));

  // ── Washer ──
  if (hw.includes('washer')) sections.push(_wizApplianceSection('washer','👕','Washing Machine','washer_state','washer_power'));

  // ── Dryer ──
  if (hw.includes('dryer')) sections.push(_wizApplianceSection('dryer','🌀','Dryer','dryer_state','dryer_power'));

  // ── EV ──
  if (hw.includes('ev')) sections.push(`<div class="wiz-card" style="margin-top:.7rem">
    <div class="wiz-card-title" style="font-size:1rem">🚗 EV Charger</div>
    <div class="entity-picker"><div class="entity-picker-label">EV charger entity (switch / number)</div>
      <input class="entity-select" id="wiz-ev" value="${wizCfg.sensors.ev_charger||''}" placeholder="switch.wallbox" oninput="wizCfg.sensors.ev_charger=this.value;_wizRenderCands('ev_charger','wiz-ev','wiz-ev-cands',null,null)">
      <div id="wiz-ev-cands" class="entity-cands"></div></div></div>`);

  // ── Custom loads ──
  if (hw.includes('custom')) sections.push(_wizCustomLoadsSection());

  document.getElementById('wiz-loads-entities').innerHTML = sections.join('');
  setTimeout(()=>_wizLoadsRenderCands(), 50);
}

function _wizApplianceSection(key, icon, label, stateRole, powerRole) {
  return `<div class="wiz-card" style="margin-top:.7rem">
    <div class="wiz-card-title" style="font-size:1rem">${icon} ${label}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem">
      <div class="entity-picker"><div class="entity-picker-label">State sensor</div>
        <input class="entity-select" id="wiz-${key}-state" value="${wizCfg.sensors[stateRole]||''}" placeholder="sensor.${key}_operation_state" oninput="wizCfg.sensors['${stateRole}']=this.value;_wizRenderCands('${stateRole}','wiz-${key}-state','wiz-${key}-state-cands',null,null)">
        <div id="wiz-${key}-state-cands" class="entity-cands"></div></div>
      <div class="entity-picker"><div class="entity-picker-label">Power meter (W)</div>
        <input class="entity-select" id="wiz-${key}-power" value="${wizCfg.sensors[powerRole]||''}" placeholder="sensor.${key}_power" oninput="wizCfg.sensors['${powerRole}']=this.value;_wizRenderCands('${powerRole}','wiz-${key}-power','wiz-${key}-power-cands',null,null)">
        <div id="wiz-${key}-power-cands" class="entity-cands"></div></div>
    </div></div>`;
}

function _wizCustomLoadsSection() {
  if (!wizCfg.custom_loads) wizCfg.custom_loads = [];
  const items = wizCfg.custom_loads;
  const schedOpts = [
    {v:'valley',l:'🌙 Valley tariff only'},{v:'solar',l:'☀️ Solar surplus only'},
    {v:'both',l:'☀️🌙 Solar + Valley'},{v:'hours',l:'⏱ Custom hours'},
  ];
  const rows = items.map((item,i) => `
    <div class="custom-load-row">
      <div style="display:grid;grid-template-columns:1fr auto;gap:.4rem;align-items:center;margin-bottom:.35rem">
        <input class="entity-select" value="${item.name||''}" placeholder="Device name (e.g. Irrigation pump)"
          oninput="wizCfg.custom_loads[${i}].name=this.value" style="font-weight:600">
        <div style="display:flex;align-items:center;gap:.3rem">
          <input type="number" value="${item.watts||0}" min="0" max="20000" step="50" style="width:72px"
            oninput="wizCfg.custom_loads[${i}].watts=+this.value">
          <span style="font-size:.65rem;color:var(--m)">W</span>
        </div>
      </div>
      <div class="entity-picker">
        <div class="entity-picker-label">Switch entity</div>
        <input class="entity-select" id="wiz-cl-sw-${i}" value="${item.switch||''}" placeholder="switch.my_device"
          oninput="wizCfg.custom_loads[${i}].switch=this.value;_wizRenderCands('pool_switch','wiz-cl-sw-'+${i},'wiz-cl-sw-'+${i}+'-cands',null,null)">
        <div id="wiz-cl-sw-${i}-cands" class="entity-cands"></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-top:.3rem">
        <div class="wiz-field"><label>Schedule</label>
          <select onchange="wizCfg.custom_loads[${i}].schedule=this.value;_wizRenderCustomLoads()">
            ${schedOpts.map(o=>`<option value="${o.v}"${item.schedule===o.v?' selected':''}>${o.l}</option>`).join('')}
          </select>
        </div>
        ${item.schedule==='hours'?`<div class="wiz-field"><label>Hours (e.g. 10-14,22-06)</label>
          <input value="${item.hours||''}" placeholder="10-14" oninput="wizCfg.custom_loads[${i}].hours=this.value"></div>`:'<div></div>'}
      </div>
      <button class="btn btn-sm" style="margin-top:.3rem;color:#ef4444;border-color:#ef4444"
        onclick="wizDelCustomLoad(${i})">✕ Remove</button>
    </div>`).join('<hr style="border-color:var(--b);margin:.4rem 0">');
  return `<div class="wiz-card" style="margin-top:.7rem">
    <div class="wiz-card-title" style="font-size:1rem">⚙️ Custom Loads</div>
    <div style="font-size:.7rem;color:var(--m);margin-bottom:.6rem">
      Any switch-controlled load. The engine schedules them during cheapest / greenest windows.</div>
    <div id="wiz-custom-loads-list">${rows}</div>
    <button class="btn btn-sm" onclick="wizAddCustomLoad()" style="margin-top:.5rem">+ Add device</button>
  </div>`;
}
function _wizRenderCustomLoads() {
  const el = document.getElementById('wiz-loads-entities');
  if (el) { el.innerHTML = _wizCustomLoadsSection(); setTimeout(_wizAddSearchButtons,60); }
}
function wizAddCustomLoad() {
  if (!wizCfg.custom_loads) wizCfg.custom_loads = [];
  wizCfg.custom_loads.push({name:'',switch:'',watts:0,schedule:'valley',hours:''});
  _wizRenderCustomLoads();
}
function wizDelCustomLoad(i) {
  wizCfg.custom_loads.splice(i,1);
  _wizRenderCustomLoads();
}

function _wizLoadsRenderCands() {
  const map = [
    ['hvac_mode','wiz-hvac-cli-0','wiz-hvac-cli-0-cands'],
    ['hvac_temp_heat','wiz-hvac-ht-0','wiz-hvac-ht-0-cands'],
    ['hvac_temp_cool','wiz-hvac-cl-0','wiz-hvac-cl-0-cands'],
    ['temp_indoor','wiz-hvac-ts-0','wiz-hvac-ts-0-cands'],
    ['pool_switch','wiz-pool-sw','wiz-pool-sw-cands'],
    ['pool_cleaner','wiz-pool-cl','wiz-pool-cl-cands'],
    ['ev_charger','wiz-ev','wiz-ev-cands'],
    ['dishwasher_state','wiz-dishwasher-state','wiz-dishwasher-state-cands'],
    ['washer_power','wiz-dishwasher-power','wiz-dishwasher-power-cands'],
    ['washer_state','wiz-washer-state','wiz-washer-state-cands'],
    ['washer_power','wiz-washer-power','wiz-washer-power-cands'],
    ['dryer_state','wiz-dryer-state','wiz-dryer-state-cands'],
    ['dryer_power','wiz-dryer-power','wiz-dryer-power-cands'],
  ];
  map.forEach(([role,inp,cands]) => {
    if (document.getElementById(inp)) _wizRenderCands(role,inp,cands,null,null);
  });
  // Extra HVAC zones if present
  (wizCfg.hvac_zones||[]).forEach((_,i) => {
    if (i===0) return;
    [['hvac_mode',`wiz-hvac-cli-${i}`,`wiz-hvac-cli-${i}-cands`],
     ['hvac_temp_heat',`wiz-hvac-ht-${i}`,`wiz-hvac-ht-${i}-cands`],
     ['hvac_temp_cool',`wiz-hvac-cl-${i}`,`wiz-hvac-cl-${i}-cands`],
     ['temp_indoor',`wiz-hvac-ts-${i}`,`wiz-hvac-ts-${i}-cands`],
    ].forEach(([r,ip,cp])=>{ if(document.getElementById(ip)) _wizRenderCands(r,ip,cp,null,null); });
  });
}

// ── HVAC zone management ───────────────────────────────────────────────────────
function wizAddHvacZone() {
  wizCfg.hvac_zones.push({name:'Zone '+(wizCfg.hvac_zones.length+1),climate:'',temp_heat:'',temp_cool:'',temp_sensor:'',sched:{},temp_comfort:21,temp_surplus:23,temp_min:18});
  wizRenderLoadsEntities();
}
function wizDelHvacZone(i) {
  wizCfg.hvac_zones.splice(i,1);
  wizRenderLoadsEntities();
}
function wizCycleSched(zi, h) {
  const z = wizCfg.hvac_zones[zi];
  if (!z.sched) z.sched = {};
  const order = ['minimum','comfort','surplus'];
  const cur = z.sched[h]||'minimum';
  z.sched[h] = order[(order.indexOf(cur)+1)%3];
  wizRenderLoadsEntities();
}

// ── Summary & save ─────────────────────────────────────────────────────────────
function wizBuildSummary() {
  wizCfg.tariff = wizCfg.tariff || {};
  wizCfg.tariff.contracted_kw = parseFloat(document.getElementById('wiz-contracted-kw')?.value) || 5.75;
  wizCfg.tariff.type = document.getElementById('wiz-tariff-type')?.value || 'es_2td';
  wizCfg.location.latitude  = parseFloat(document.getElementById('wiz-lat')?.value) || null;
  wizCfg.location.longitude = parseFloat(document.getElementById('wiz-lon')?.value) || null;
  wizCfg.location.timezone  = document.getElementById('wiz-tz')?.value || null;
  wizCfg.battery_capacity_kwh = parseFloat(document.getElementById('wiz-batt-cap')?.value) || 10;
  wizCfg.grid_flip = document.getElementById('wiz-grid-flip')?.checked || false;
  const ok = v => v ? `<span class="ent-ok">✓</span> ${v}` : `<span class="ent-miss">✗</span> <em>not set</em>`;
  const rows = [
    ['Data source',    wizCfg.data_source === 'influxdb' ? '🗄️ InfluxDB' : '🏠 HA Recorder'],
    ['Location',       wizCfg.location.latitude ? `${wizCfg.location.latitude}, ${wizCfg.location.longitude}` : '—'],
    ['Timezone',       wizCfg.location.timezone || '—'],
    ['Grid sensor',    ok(wizCfg.sensors.grid_power)],
    ['Solar sensor',   ok(wizCfg.sensors.solar_power)],
    ['Battery SOC',    ok(wizCfg.sensors.battery_soc)],
    ['Battery power',  ok(wizCfg.sensors.battery_power)],
    ['Battery capacity',(wizCfg.battery_capacity_kwh||'—') + ' kWh'],
    ['HVAC zones',     wizCfg.hvac_zones.length ? wizCfg.hvac_zones.map(z=>z.name||'Zone').join(', ') : '—'],
    ['Other devices',  wizCfg.hardware.filter(h=>h!=='hvac').join(', ') || '—'],
    ['Weather entity', ok(wizCfg.sensors.weather)],
    ['Tariff',         (wizCfg.tariff.type||'—') + ' · ' + (wizCfg.tariff.contracted_kw||'—') + ' kW'],
  ];
  document.getElementById('wiz-summary-table').innerHTML =
    rows.map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');

  // Data quality summary panel
  const totalSamples = Object.values(wizDQSamples).reduce((a,b)=>a+(b||0),0);
  const src = wizCfg.data_source === 'influxdb' ? 'InfluxDB' : 'HA Recorder';
  const days = wizCfg.data_source === 'influxdb' ? 90 : 14;
  const keySensors = ['grid_power','solar_power','battery_soc','battery_power'];
  const missing = keySensors.filter(r=>!wizCfg.sensors[r]);
  const hasAll = missing.length === 0;
  const qualLabel = totalSamples > 20000 ? ['Excellent 🟢','R² likely > 0.90 — highly accurate forecasts']
                  : totalSamples > 5000  ? ['Good 🟡','R² likely 0.75–0.90 — solid predictions']
                  : totalSamples > 500   ? ['Fair 🟠','R² likely 0.50–0.75 — usable but will improve over time']
                  :                        ['Learning 🔴','Not enough data yet — model will improve as data accumulates'];
  const dqEl = document.getElementById('wiz-dq-summary');
  const detEl = document.getElementById('wiz-dq-detail');
  if (dqEl && detEl) {
    dqEl.style.display = '';
    detEl.innerHTML = `
      <b>Source:</b> ${src} (up to ${days} days of history)<br>
      <b>Samples found:</b> ${totalSamples.toLocaleString()} across ${Object.keys(wizDQSamples).length} entities<br>
      <b>Prediction quality:</b> ${qualLabel[0]} — ${qualLabel[1]}<br>
      ${!hasAll ? `<span style="color:var(--y)">⚠ Missing sensors: ${missing.join(', ')} — accuracy will be lower</span><br>` : ''}
      <span style="color:var(--m);font-size:.68rem">Model trains automatically tonight at 03:00. More history = better predictions.</span>`;
  }
}

async function wizSave() {
  wizBuildSummary();
  document.getElementById('wiz-save-status').textContent = 'Saving…';
  try {
    const r = await fetch(BASE+'/api/wizard/config', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(wizCfg)
    }).then(r=>r.json());
    if (r.ok) {
      document.getElementById('wiz-save-status').innerHTML =
        '<span style="color:var(--g);font-size:1rem">✓ Configuration saved! The optimizer will use your new settings on the next cycle.</span>';
      notify('✓ Wizard configuration saved!', 'ok', 5000);
      setTimeout(wizFetchDQ, 1000);
    } else {
      document.getElementById('wiz-save-status').textContent = 'Error: ' + (r.error||'unknown');
    }
  } catch(e) {
    document.getElementById('wiz-save-status').textContent = 'Network error: ' + e.message;
  }
}

function wizToggleBattSplit() {
  const on = document.getElementById('wiz-batt-split')?.checked;
  document.getElementById('wiz-batt-power-row').style.display  = on ? 'none' : '';
  document.getElementById('wiz-batt-split-row').style.display  = on ? 'grid' : 'none';
  wizCfg.battery_power_split = !!on;
}

function _wizPopulateFields() {
  if (wizCfg.location) {
    if (wizCfg.location.latitude)  document.getElementById('wiz-lat').value  = wizCfg.location.latitude;
    if (wizCfg.location.longitude) document.getElementById('wiz-lon').value  = wizCfg.location.longitude;
    if (wizCfg.location.timezone)  document.getElementById('wiz-tz').value   = wizCfg.location.timezone;
  }
  if (wizCfg.battery_capacity_kwh) {
    const el = document.getElementById('wiz-batt-cap');
    if (el) el.value = wizCfg.battery_capacity_kwh;
  }
  if (wizCfg.hardware) {
    document.querySelectorAll('.hw-card').forEach(c => c.classList.toggle('selected', wizCfg.hardware.includes(c.dataset.hw)));
  }
  // Influx fields
  if (wizCfg.influxdb) {
    const f = wizCfg.influxdb;
    if (f.host)     { const el=document.getElementById('wiz-influx-host'); if(el) el.value=f.host; }
    if (f.port)     { const el=document.getElementById('wiz-influx-port'); if(el) el.value=f.port; }
    if (f.db)       { const el=document.getElementById('wiz-influx-db');   if(el) el.value=f.db; }
    if (f.username) { const el=document.getElementById('wiz-influx-user'); if(el) el.value=f.username; }
    if (f.password) { const el=document.getElementById('wiz-influx-pass'); if(el) el.value=f.password; }
  }
  wizSelectDS(wizCfg.data_source || 'influxdb');
  const fieldMap = {
    grid_power:'wiz-grid-entity', solar_power:'wiz-solar-entity',
    solar_fc_today:'wiz-fc-today', solar_fc_tomorrow:'wiz-fc-tomorrow',
    solar_fc_h0:'wiz-fc-h0', solar_fc_h1:'wiz-fc-h1',
    battery_soc:'wiz-batt-soc', battery_power:'wiz-batt-power',
    battery_mode_select:'wiz-batt-mode', battery_cutoff_soc:'wiz-batt-cutoff',
    battery_backup_soc:'wiz-batt-backup', battery_force_charge:'wiz-batt-force',
    weather:'wiz-weather',
  };
  Object.entries(fieldMap).forEach(([role,id]) => {
    const el = document.getElementById(id);
    if (el && wizCfg.sensors?.[role]) el.value = wizCfg.sensors[role];
  });
  // Restore split battery sensor mode
  if (wizCfg.battery_power_split) {
    const cb = document.getElementById('wiz-batt-split');
    if (cb) { cb.checked = true; wizToggleBattSplit(); }
  }
  const chgEl = document.getElementById('wiz-batt-chg');
  const disEl = document.getElementById('wiz-batt-dis');
  if (chgEl && wizCfg.sensors?.battery_charge_entity)    chgEl.value = wizCfg.sensors.battery_charge_entity;
  if (disEl && wizCfg.sensors?.battery_discharge_entity) disEl.value = wizCfg.sensors.battery_discharge_entity;
}

function _wizAddSearchButtons() {
  document.querySelectorAll('.entity-picker').forEach(picker => {
    if (picker.querySelector('.ent-search-btn')) return;
    const input = picker.querySelector('input.entity-select');
    if (!input || !input.id) return;
    const candsId = input.id + '-cands';
    const oninp = input.getAttribute('oninput') || '';
    const rm = oninp.match(/_wizRenderCands\('["']?([^'"]+)['"]?/);
    const wm = oninp.match(/wizEntityTyped\('["']?([^'"]+)['"]?/);
    const role = (rm && rm[1]) || (wm && wm[1]) || '';
    const row = document.createElement('div');
    row.className = 'entity-picker-row';
    input.parentNode.insertBefore(row, input);
    row.appendChild(input);
    const btn = document.createElement('button');
    btn.className = 'ent-search-btn';
    btn.title = 'Browse entities';
    btn.textContent = '🔍';
    btn.type = 'button';
    btn.onclick = () => { input.value = ''; _wizRenderCands(role, input.id, candsId, null, null); input.focus(); };
    row.appendChild(btn);
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
load();
loadWeather();
setInterval(load,30000);
setInterval(loadWeather,300000);
</script>
</body></html>"""

# ── API endpoints ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    base = request.headers.get("X-Ingress-Path", "").rstrip("/")
    ver = "v" + ADD_ON_VERSION
    return PANEL.replace("__BASE__", base).replace("__VERSION__", ver)

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
    # Energy-balance sanity check: solar = house + bat_charge + grid_export.
    # Surface the residual so the UI can flag stale sensor reads (Huawei Luna2000
    # modbus often lags during transitions, especially in taper near full SOC).
    solar_w = sensors.get("solar_power") or 0
    grid_w  = sensors.get("grid_power")  or 0
    bat_w   = sensors.get("battery_power") or 0
    balance = {
        "implied_house_w": round(solar_w - grid_w - bat_w, 1),
        "residual_w":      round(min(0, solar_w - grid_w - bat_w), 1),
    }
    if balance["implied_house_w"] < -80:
        log.warning(f"  Energy balance off by {balance['implied_house_w']:.0f}W "
                    f"(solar={solar_w:.0f}, grid={grid_w:+.0f}, bat={bat_w:+.0f}) "
                    f"— sensor likely stale (battery reading typically lags)")
    return jsonify({"sensors": sensors, "tariff": tariff, "model": model,
                    "optimal": optimal, "storm": storm, "balance": balance,
                    "consumption_baseline_kw": _get_avg_night_consumption_kw()})

@app.route("/api/debug/sensors")
def api_debug_sensors():
    """Show resolved entity IDs and raw HA values for every key sensor role."""
    roles = {
        "solar_power":        ("sensor_solar_power",       ""),
        "grid_power":         ("sensor_grid_power",        ""),
        "battery_soc":        ("sensor_battery_soc",       "sensor.battery_state_of_capacity"),
        "temp_outdoor":       ("sensor_temp_outdoor",      ""),
        "temp_indoor":        ("sensor_temp_salon",        ""),
        "solar_fc_today":     ("sensor_solar_today",       "sensor.energy_production_today"),
        "solar_fc_tomorrow":  ("sensor_solar_tomorrow",    "sensor.energy_production_tomorrow"),
        "pool_switch":        ("switch_pool",              ""),
    }
    result = {}
    for role, (cfg_key, fallback) in roles.items():
        entity_id = _wiz(role, cfg_key, fallback)
        state_obj = ha_state(entity_id)
        raw_state  = state_obj.get("state") if state_obj else None
        try:
            parsed = float(raw_state) if raw_state not in (None, "unavailable", "unknown") else None
        except (ValueError, TypeError):
            parsed = None
        result[role] = {
            "entity_id": entity_id,
            "source": ("wizard" if _WIZARD.get("sensors", {}).get(role)
                       else "options" if cfg(cfg_key, "")
                       else "fallback"),
            "raw_state": raw_state,
            "parsed_float": parsed,
        }
    # Battery power: single signed sensor OR split charge/discharge (Deye/Solarman)
    if _WIZARD.get("battery_power_split"):
        sc = _WIZARD.get("sensors", {})
        for sub_role, wiz_key in (("battery_charge_entity", "battery_charge_entity"),
                                   ("battery_discharge_entity", "battery_discharge_entity")):
            entity_id = sc.get(wiz_key, "")
            state_obj = ha_state(entity_id) if entity_id else None
            raw_state = state_obj.get("state") if state_obj else None
            try:
                parsed = float(raw_state) if raw_state not in (None, "unavailable", "unknown") else None
            except (ValueError, TypeError):
                parsed = None
            result[sub_role] = {
                "entity_id": entity_id,
                "source": "wizard" if entity_id else "fallback",
                "raw_state": raw_state,
                "parsed_float": parsed,
            }
    else:
        entity_id = _wiz("battery_power", "sensor_battery_power",
                         "sensor.battery_charge_discharge_power")
        state_obj = ha_state(entity_id)
        raw_state = state_obj.get("state") if state_obj else None
        try:
            parsed = float(raw_state) if raw_state not in (None, "unavailable", "unknown") else None
        except (ValueError, TypeError):
            parsed = None
        result["battery_power"] = {
            "entity_id": entity_id,
            "source": ("wizard" if _WIZARD.get("sensors", {}).get("battery_power")
                       else "options" if cfg("sensor_battery_power", "")
                       else "fallback"),
            "raw_state": raw_state,
            "parsed_float": parsed,
        }
    return jsonify(result)

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
        "battery_health_mode":            lambda v: str(v) if str(v) in ("bill_reducer","optimized","battery_guard") else "bill_reducer",
        "influxdb_url":                   lambda v: str(v)[:200].strip(),
        "influxdb_db":                    lambda v: str(v)[:100].strip(),
        "influxdb_user":                  lambda v: str(v)[:100].strip(),
        "influxdb_password":              lambda v: str(v)[:200],
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

def _hourly_from_decisions(date_str: str) -> dict:
    """Aggregate decisions.json into hourly kWh buckets for a given YYYY-MM-DD date."""
    K = 0.25 / 1000  # 15-min interval → kWh factor
    hours: dict = {h: {"sol": [], "grid": [], "bat": [], "soc": []} for h in range(24)}
    if DECISIONS_FILE.exists():
        try:
            for dec in json.loads(DECISIONS_FILE.read_text()):
                ts_str = dec.get("timestamp", "")
                if not ts_str or ts_str[:10] != date_str:
                    continue
                try:
                    h = datetime.fromisoformat(ts_str).hour
                except ValueError:
                    continue
                s = dec.get("sensors", {})
                hours[h]["sol"].append(s.get("solar_power") or 0)
                hours[h]["grid"].append(s.get("grid_power") or 0)
                hours[h]["bat"].append(s.get("battery_power") or 0)
                if s.get("battery_soc") is not None:
                    hours[h]["soc"].append(float(s["battery_soc"]))
        except Exception:
            pass

    res: dict = {k: [] for k in ("labels","solar","grid_import","grid_export",
                                  "bat_charge","bat_discharge","consumption","soc")}
    for h in range(24):
        res["labels"].append(f"{h:02d}:00")
        d = hours[h]
        if not d["sol"]:
            for k in res:
                if k != "labels":
                    res[k].append(None)
            continue
        sol  = sum(d["sol"]) * K
        gimp = sum(max(-g, 0) for g in d["grid"]) * K
        gexp = sum(max(g, 0) for g in d["grid"]) * K
        bchg = sum(max(-b, 0) for b in d["bat"]) * K
        bdis = sum(max(b, 0) for b in d["bat"]) * K
        cons = max(0.0, sol + bdis + gimp - gexp - bchg)
        res["solar"].append(round(sol, 3))
        res["grid_import"].append(round(gimp, 3))
        res["grid_export"].append(round(gexp, 3))
        res["bat_charge"].append(round(bchg, 3))
        res["bat_discharge"].append(round(bdis, 3))
        res["consumption"].append(round(cons, 3))
        res["soc"].append(round(sum(d["soc"]) / len(d["soc"]), 1) if d["soc"] else None)

    # Daily KPIs
    nn = lambda lst: [v for v in lst if v is not None]
    total_sol  = sum(nn(res["solar"]))
    total_imp  = sum(nn(res["grid_import"]))
    total_exp  = sum(nn(res["grid_export"]))
    total_cons = sum(nn(res["consumption"]))
    self_suff  = round(min(100, (1 - total_imp / max(total_cons, 0.01)) * 100), 1) if total_cons else 0
    solar_sc   = round(min(100, (1 - total_exp / max(total_sol, 0.01)) * 100), 1) if total_sol else 0
    res["kpi"] = {
        "solar_kwh":             round(total_sol,  2),
        "import_kwh":            round(total_imp,  2),
        "export_kwh":            round(total_exp,  2),
        "consumption_kwh":       round(total_cons, 2),
        "self_sufficiency_pct":  self_suff,
        "solar_self_consumed_pct": solar_sc,
    }
    return res


def _live_averages_7d() -> dict:
    """W-level 7-day averages from 15-min decision samples.

    SOL: mean only when > 0  (real production during daylight hours)
    RED: net mean  (+ = export, - = import)
    BAT: net mean  (+ = charging / bat>0, - = discharging / bat<0)
    CASA: mean of (solar - grid - bat) per sample
    """
    cutoff = datetime.now() - timedelta(days=7)
    sol_v, red_v, bat_v, casa_v = [], [], [], []
    if DECISIONS_FILE.exists():
        try:
            for dec in json.loads(DECISIONS_FILE.read_text()):
                ts_str = dec.get("timestamp", "")
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str)
                except ValueError:
                    continue
                if ts < cutoff:
                    continue
                s    = dec.get("sensors", {})
                sol  = s.get("solar_power")  or 0
                red  = s.get("grid_power")   or 0
                bat  = s.get("battery_power") or 0
                casa = sol - red - bat
                if sol > 0:
                    sol_v.append(sol)
                red_v.append(red)
                bat_v.append(bat)
                if casa > 0:
                    casa_v.append(casa)
        except Exception:
            pass
    def avg(lst): return round(sum(lst) / len(lst)) if lst else None
    return {
        "samples":  len(red_v),
        "avg_sol":  avg(sol_v),   # W, solar-hours only
        "avg_red":  avg(red_v),   # W, net (+ export / - import)
        "avg_bat":  avg(bat_v),   # W, net (+ charge / - discharge)
        "avg_casa": avg(casa_v),  # W, house consumption
    }


def _daily_kpi_averages() -> dict:
    """Average daily kWh KPIs aggregated from all decisions.json history.

    decisions.json convention: bat > 0 = discharging (raw Huawei sensor).
    Returns separate stats for all_time and last_12m.
    """
    K = 0.25 / 1000  # 15-min interval → kWh
    days: dict = {}
    if DECISIONS_FILE.exists():
        try:
            for dec in json.loads(DECISIONS_FILE.read_text()):
                ts_str = dec.get("timestamp", "")
                if not ts_str:
                    continue
                date = ts_str[:10]
                s    = dec.get("sensors", {})
                sol  = s.get("solar_power") or 0
                grid = s.get("grid_power") or 0
                bat  = s.get("battery_power") or 0
                d = days.setdefault(date, {"sol": 0.0, "gimp": 0.0, "gexp": 0.0,
                                           "bchg": 0.0, "bdis": 0.0})
                d["sol"]  += max(sol, 0) * K
                d["gimp"] += max(-grid, 0) * K
                d["gexp"] += max(grid, 0) * K
                d["bdis"] += max(bat, 0) * K   # bat > 0 = discharging
                d["bchg"] += max(-bat, 0) * K  # bat < 0 = charging
        except Exception:
            pass

    if not days:
        return {}

    cutoff_12m = (datetime.now().date() - timedelta(days=365)).isoformat()

    def _stats(day_list: list) -> dict:
        if not day_list:
            return {}
        n     = len(day_list)
        conss = [max(0.0, d["sol"] + d["bdis"] + d["gimp"] - d["gexp"] - d["bchg"])
                 for d in day_list]
        ss    = [min(100.0, (1 - d["gimp"] / max(c, 0.01)) * 100)
                 for d, c in zip(day_list, conss)]
        return {
            "n_days":                   n,
            "avg_solar_kwh":            round(sum(d["sol"] for d in day_list) / n, 2),
            "avg_import_kwh":           round(sum(d["gimp"] for d in day_list) / n, 2),
            "avg_export_kwh":           round(sum(d["gexp"] for d in day_list) / n, 2),
            "avg_consumption_kwh":      round(sum(conss) / n, 2),
            "avg_self_sufficiency_pct": round(sum(ss) / n, 1),
        }

    all_vals    = list(days.values())
    recent_vals = [v for k, v in days.items() if k >= cutoff_12m]
    return {
        "all_time": _stats(all_vals),
        "last_12m": _stats(recent_vals),
    }


@app.route("/api/averages")
def api_averages():
    return jsonify(_daily_kpi_averages())


@app.route("/api/chart-data")
def api_chart_data():
    from datetime import timezone as _tz
    date_param = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    soc_entity = _wiz("battery_soc", "sensor_battery_soc", "sensor.battery_state_of_capacity")
    # InfluxDB source: wizard config takes precedence over legacy cfg
    _wiz_influx = _WIZARD.get("influxdb", {})
    influx_u    = (_wiz_influx.get("host") and
                   f"http://{_wiz_influx['host']}:{_wiz_influx.get('port',8086)}") or \
                  cfg("influxdb_url", "").strip()
    influx_db   = _wiz_influx.get("db") or cfg("influxdb_db", "homeassistant").strip()
    influx_user = _wiz_influx.get("username") or cfg("influxdb_user", "").strip()
    influx_pass = _wiz_influx.get("password") or cfg("influxdb_password", "")

    # ── SOC — 15-min bucket grid, align actual + predicted ────────────────────
    if influx_u:
        rows, _ = ha_history_influx(soc_entity, days=1)
    if not influx_u or not rows:
        rows = ha_history(soc_entity, days=1)

    # Parse actual points with local naive timestamps
    actual_pts = []  # list of (local_naive_dt, float)
    for r in rows:
        try:
            ts_raw   = datetime.fromisoformat(r["last_changed"].replace("Z", "+00:00"))
            ts_local = ts_raw.astimezone().replace(tzinfo=None)
            val      = float(r["state"])
            actual_pts.append((ts_local, val))
        except (KeyError, ValueError, TypeError):
            continue

    # Build 15-min bucket grid for past 24h
    now_local  = datetime.now()
    last_min   = (now_local.minute // 15) * 15
    period_end = now_local.replace(minute=last_min, second=0, microsecond=0)
    buckets    = [period_end - timedelta(minutes=15 * i) for i in range(95, -1, -1)]
    soc_labels = [b.strftime("%H:%M") for b in buckets]
    HALF       = 450  # ±7.5 min in seconds

    # Actual SOC: nearest raw point per bucket
    soc_actual = []
    for b in buckets:
        if actual_pts:
            diffs = [(abs((pt[0] - b).total_seconds()), pt[1]) for pt in actual_pts]
            gap, val = min(diffs)
            soc_actual.append(round(val, 1) if gap <= HALF else None)
        else:
            soc_actual.append(None)

    # Predicted SOC: nearest decision per bucket
    decision_pts = []  # list of (local_naive_dt, float)
    cutoff = period_end - timedelta(hours=24)
    if DECISIONS_FILE.exists():
        try:
            history = json.loads(DECISIONS_FILE.read_text())
            for dec in history:
                ts_str = dec.get("timestamp", "")
                pred   = dec.get("prediction", {}).get("predicted_soc")
                if pred is not None and ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if ts >= cutoff:
                            decision_pts.append((ts, pred))
                    except ValueError:
                        pass
        except Exception:
            pass

    soc_predicted = []
    for b in buckets:
        if decision_pts:
            diffs = [(abs((pt[0] - b).total_seconds()), pt[1]) for pt in decision_pts]
            gap, val = min(diffs)
            soc_predicted.append(val if gap <= HALF else None)
        else:
            soc_predicted.append(None)

    # MAE between predicted and actual (where both available)
    errors  = [abs(a - p) for a, p in zip(soc_actual, soc_predicted)
               if a is not None and p is not None]
    soc_mae = round(sum(errors) / len(errors), 1) if errors else None

    # ── SOC forecast (future 8h) — chained ML predictions ────────────────────
    future_labels, future_predicted = [], []
    sensors = read_sensors()
    if MODEL_FILE.exists():
        try:
            art      = joblib.load(MODEL_FILE)
            mdl_feats = art.get("features", BASE_FEATURES)
            now      = datetime.now()
            soc_f    = sensors.get("battery_soc", 50.0) or 50.0
            for h in range(1, 9):
                ft = now + timedelta(hours=h)
                row = {
                    "hour":        ft.hour,
                    "weekday":     ft.weekday(),
                    "month":       ft.month,
                    "lag1":        soc_f,
                    "lag4":        soc_f,
                    "roll4":       soc_f,
                    "solar_proxy": _solar_proxy_for_hour(ft.hour, ft.month),
                    "temp_outdoor": sensors.get("temp_outdoor", 15.0),
                    "solar_lag1":  sensors.get("solar_power", 0) / 1000,
                    "solar_roll4": sensors.get("solar_power", 0) / 1000,
                    "grid_lag1":   sensors.get("grid_power", 0) / 1000,
                    "grid_roll4":  sensors.get("grid_power", 0) / 1000,
                }
                X    = pd.DataFrame([row])
                cols = [c for c in mdl_feats if c in X.columns]
                pred = float(art["pipeline"].predict(X[cols])[0])
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
        yest_date   = (now_local - timedelta(days=1)).date()
        yest_start  = datetime(yest_date.year, yest_date.month, yest_date.day,
                               tzinfo=_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        today_start = datetime(now_local.year, now_local.month, now_local.day,
                               tzinfo=_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        q_sol = (f'SELECT MAX("value") FROM /.*/ WHERE "entity_id" = \'{solar_entity_id}\' '
                 f'AND time >= \'{yest_start}\' AND time < \'{today_start}\'')
        r_sol, _, _ = _influx_query(influx_u, influx_db, q_sol, influx_user, influx_pass)
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

    # ── Solar forecast accuracy — yesterday's forecast for today ──────────────
    solar_today_forecast = None
    if DECISIONS_FILE.exists():
        try:
            history  = json.loads(DECISIONS_FILE.read_text())
            yest_str = (now_local - timedelta(days=1)).date().isoformat()
            yest_dec = [d for d in history if d.get("timestamp", "")[:10] == yest_str]
            if yest_dec:
                v = yest_dec[-1].get("sensors", {}).get("solar_tomorrow")
                if v is not None:
                    solar_today_forecast = round(float(v), 1)
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
            p_export = prices.get("export", 0.040)
            for dec in history:
                day    = dec.get("timestamp", "")[:10]
                sens   = dec.get("sensors", {})
                period = dec.get("tariff", {}).get("period", "mid")
                grid   = sens.get("grid_power", 0)       # -ve = buying (import), +ve = selling (export)
                bat    = sens.get("battery_power", 0)    # +ve = charging, -ve = discharging
                p_imp  = prices.get(period, prices.get("mid", 0.1483))
                kwh_f  = interval_h / 1000
                # Counterfactual: without smart battery, battery_power would be 0, so grid_nb = grid + bat
                # (if battery was discharging -2000W, without it we'd import 2000W more → grid shifts more negative)
                grid_nb  = grid + bat
                # Cost function: negative grid = paying p_imp; positive grid = receiving p_export (income, negative cost)
                cost_nb  = (-grid_nb * kwh_f * p_imp) if grid_nb < 0 else (-grid_nb * kwh_f * p_export)
                cost_act = (-grid    * kwh_f * p_imp) if grid    < 0 else (-grid    * kwh_f * p_export)
                saving   = cost_nb - cost_act
                if abs(saving) > 0.0001:
                    daily[day] = round(daily.get(day, 0) + saving, 4)
            for i in range(6, -1, -1):
                day = (datetime.now() - timedelta(days=i)).date().isoformat()
                savings_labels.append(day[5:])
                savings_values.append(round(daily.get(day, 0), 2))
        except Exception:
            pass

    # ── Solar 7-day & 12-month line charts ───────────────────────────────────
    solar_7d_labels,  solar_7d_actual,  solar_7d_forecast  = [], [], []
    solar_12m_labels, solar_12m_actual, solar_12m_forecast = [], [], []

    if influx_u:
        sol_tmrw_id = _wiz("solar_fc_tomorrow", "sensor_solar_tomorrow",
                           "sensor.energy_production_tomorrow").split(".")[-1]

        # Daily MAX actual + daily LAST forecast — 395d covers 13 months
        q_act = (f'SELECT MAX("value") FROM /.*/ WHERE "entity_id" = \'{solar_entity_id}\' '
                 f'AND time >= now() - 395d GROUP BY time(1d) fill(null)')
        q_fc  = (f'SELECT LAST("value") FROM /.*/ WHERE "entity_id" = \'{sol_tmrw_id}\' '
                 f'AND time >= now() - 396d GROUP BY time(1d) fill(null)')
        r_act, _, _ = _influx_query(influx_u, influx_db, q_act, influx_user, influx_pass)
        r_fc,  _, _ = _influx_query(influx_u, influx_db, q_fc,  influx_user, influx_pass)

        # Parse into date → value dicts
        actual_d: dict   = {}
        forecast_d: dict = {}
        if r_act:
            try:
                s = r_act.json()["results"][0].get("series", [])
                if s:
                    ci = {c: i for i, c in enumerate(s[0]["columns"])}
                    for pt in s[0]["values"]:
                        if pt[ci["max"]] is not None:
                            d = datetime.fromtimestamp(pt[ci["time"]] / 1000).date()
                            actual_d[d] = round(float(pt[ci["max"]]), 1)
            except Exception:
                pass
        if r_fc:
            try:
                s = r_fc.json()["results"][0].get("series", [])
                if s:
                    ci = {c: i for i, c in enumerate(s[0]["columns"])}
                    for pt in s[0]["values"]:
                        if pt[ci["last"]] is not None:
                            # forecast stored on day D is for day D+1
                            d = datetime.fromtimestamp(pt[ci["time"]] / 1000).date() + timedelta(days=1)
                            forecast_d[d] = round(float(pt[ci["last"]]), 1)
            except Exception:
                pass

        # Patch live sensor values for today and tomorrow
        today_d    = now_local.date()
        tomorrow_d = today_d + timedelta(days=1)
        actual_d[today_d]      = round(solar_today,    1)
        forecast_d[tomorrow_d] = round(solar_tomorrow, 1)

        # ── 7-day: 7 past days + tomorrow ────────────────────────────────────
        for i in range(6, -2, -1):   # i=6 → 6 days ago; i=-1 → tomorrow
            d   = (now_local - timedelta(days=i)).date()
            lbl = "Tomorrow" if i == -1 else d.strftime("%m-%d")
            solar_7d_labels.append(lbl)
            solar_7d_actual.append(actual_d.get(d) if i >= 0 else None)
            solar_7d_forecast.append(forecast_d.get(d))

        # ── 12-month: aggregate daily values by calendar month ───────────────
        m_actual:   dict = {}
        m_forecast: dict = {}
        m_cnt_a:    dict = {}
        m_cnt_f:    dict = {}
        for delta in range(395):
            d = today_d - timedelta(days=delta)
            m = d.strftime("%Y-%m")
            if d in actual_d:
                m_actual[m]  = m_actual.get(m, 0.0) + actual_d[d]
                m_cnt_a[m]   = m_cnt_a.get(m, 0) + 1
            if d in forecast_d:
                m_forecast[m] = m_forecast.get(m, 0.0) + forecast_d[d]
                m_cnt_f[m]    = m_cnt_f.get(m, 0) + 1

        months = sorted(set(m_actual) | set(m_forecast))[-13:]
        for m in months:
            dt  = datetime.strptime(m, "%Y-%m")
            lbl = dt.strftime("%b") if dt.month != 1 else dt.strftime("%b'%y")
            solar_12m_labels.append(lbl)
            solar_12m_actual.append(round(m_actual[m], 1) if m_cnt_a.get(m) else None)
            solar_12m_forecast.append(round(m_forecast[m], 1) if m_cnt_f.get(m) else None)

    # ── Power flow 24h from decisions.json ───────────────────────────────────
    pf_labels, pf_solar, pf_grid, pf_battery = [], [], [], []
    now_local_pf = datetime.now()
    cutoff_pf    = now_local_pf - timedelta(hours=24)
    if DECISIONS_FILE.exists():
        try:
            history_pf = json.loads(DECISIONS_FILE.read_text())
            for dec in sorted(history_pf, key=lambda d: d.get("timestamp", "")):
                ts_str = dec.get("timestamp", "")
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str)
                except ValueError:
                    continue
                if ts < cutoff_pf:
                    continue
                sens = dec.get("sensors", {})
                pf_labels.append(ts.strftime("%H:%M"))
                pf_solar.append(round((sens.get("solar_power") or 0) / 1000, 2))
                pf_grid.append(round((sens.get("grid_power") or 0) / 1000, 2))
                pf_battery.append(round((sens.get("battery_power") or 0) / 1000, 2))
        except Exception:
            pass

    return jsonify({
        "soc": {
            "labels":           soc_labels,
            "actual":           soc_actual,
            "predicted":        soc_predicted,
            "future_labels":    future_labels,
            "future_predicted": future_predicted,
            "mae":              soc_mae,
        },
        "solar":         {"yesterday": round(solar_yesterday, 1),
                          "today": round(solar_today, 1),
                          "today_forecast": solar_today_forecast,
                          "tomorrow": round(solar_tomorrow, 1)},
        "solar_7d":      {"labels": solar_7d_labels,  "actual": solar_7d_actual,
                          "forecast": solar_7d_forecast},
        "solar_12m":     {"labels": solar_12m_labels, "actual": solar_12m_actual,
                          "forecast": solar_12m_forecast},
        "savings_daily": {"labels": savings_labels, "values": savings_values},
        "power_flow":    {"labels": pf_labels, "solar": pf_solar,
                          "grid": pf_grid, "battery": pf_battery},
        "hourly":        _hourly_from_decisions(date_param),
        "averages":      _live_averages_7d(),
        "date":          date_param,
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
    """Manual override: force a battery charge to a given target. Records
    the action in decisions.json as a `manual_api` event so it doesn't bypass
    the Activity tab (no black box)."""
    data       = request.get_json() or {}
    target_soc = max(10, min(100, int(data.get("target_soc", 80))))
    requester  = request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown"
    ok         = set_battery_charge_target(target_soc)
    _record_event({
        "type":    "manual_api",
        "action":  "battery_charge",
        "target_soc": target_soc,
        "reason":  f"Manual override via /api/battery/charge (requested by {requester})",
        "ok":      ok,
        "explanation": _explain(
            what=f"Force battery charge to {target_soc}% (manual override)",
            why="Triggered by an external POST to /api/battery/charge — bypasses the normal decision cycle.",
            inputs=[
                {"label": "Endpoint",      "value": "/api/battery/charge"},
                {"label": "Target SOC",    "value": f"{target_soc}%"},
                {"label": "Requester IP",  "value": requester},
            ],
            formula="external HTTP POST → set_battery_charge_target(target_soc)",
        ),
    }, source="manual_api")
    return jsonify({"ok": ok, "target_soc": target_soc})

@app.route("/api/battery/self-consumption", methods=["POST"])
def api_battery_self_consumption():
    """Manual override: switch the battery back to self-consumption mode.
    Records the action in decisions.json as a `manual_api` event."""
    requester = request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown"
    ok        = set_battery_self_consumption()
    _record_event({
        "type":   "manual_api",
        "action": "battery_self_consumption",
        "reason": f"Manual override via /api/battery/self-consumption (requested by {requester})",
        "ok":     ok,
        "explanation": _explain(
            what="Switch battery to self-consumption (manual override)",
            why="Triggered by an external POST to /api/battery/self-consumption — bypasses the normal decision cycle.",
            inputs=[
                {"label": "Endpoint",     "value": "/api/battery/self-consumption"},
                {"label": "Requester IP", "value": requester},
            ],
            formula="external HTTP POST → set_battery_self_consumption()",
        ),
    }, source="manual_api")
    return jsonify({"ok": ok})

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
    entity = cfg("sensor_weather", "")
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

# ── Wizard / entity-discovery API endpoints ──────────────────────────────────
@app.route("/api/ha/entities")
def api_ha_entities():
    role   = request.args.get("role")
    domain = request.args.get("domain")
    all_states = ha_get("/api/states") or []
    if domain:
        return jsonify({"entities": get_entities_by_domain(domain, all_states)})
    # Return all entities with scores for every role, or a single role
    if role and role in ROLE_HINTS:
        cands = get_entity_candidates(role, all_states)
        return jsonify({"entities": cands})
    # No filter: return lightweight list of all entities for client-side scoring
    entities = [
        {
            "entity_id":    s.get("entity_id"),
            "name":         s.get("attributes", {}).get("friendly_name", s.get("entity_id")),
            "state":        s.get("state"),
            "unit":         s.get("attributes", {}).get("unit_of_measurement", ""),
            "device_class": s.get("attributes", {}).get("device_class", ""),
        }
        for s in all_states
    ]
    return jsonify({"entities": entities})

@app.route("/api/ha/location")
def api_ha_location():
    cfg_data = ha_get("/api/config") or {}
    return jsonify({
        "latitude":  cfg_data.get("latitude"),
        "longitude": cfg_data.get("longitude"),
        "time_zone": cfg_data.get("time_zone"),
        "elevation": cfg_data.get("elevation"),
    })

@app.route("/api/wizard/config", methods=["GET"])
def api_wizard_get():
    if WIZARD_FILE.exists():
        try:
            return jsonify(json.loads(WIZARD_FILE.read_text()))
        except Exception:
            pass
    return jsonify({})

@app.route("/api/wizard/config", methods=["POST"])
def api_wizard_post():
    data = request.get_json(silent=True) or {}
    try:
        save_wizard(data)
        _load_wizard_cache()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/wizard/data-quality", methods=["GET", "POST"])
def api_wizard_data_quality():
    """Test data sources and return sample counts + quality score per entity."""
    data = {}
    if request.method == "POST":
        data = request.get_json(silent=True) or {}

    influx_cfg  = data.get("influxdb") or _WIZARD.get("influxdb", {})
    mariadb_cfg = data.get("mariadb")  or _WIZARD.get("mariadb",  {})
    sensors     = data.get("sensors")  or _WIZARD.get("sensors",  {})
    submeters   = data.get("grid_submeters") or _WIZARD.get("grid_submeters", [])
    data_source = _WIZARD.get("data_source", "ha_recorder")
    # When the wizard tests one source explicitly (Test connection button),
    # restrict the cascade to that source so the user gets a clear OK/FAIL
    # for the source they're configuring instead of falling through to a
    # different one that already worked.
    only = (data.get("only") or "").strip().lower() or None

    # Build a single entity dict spanning both sensor roles AND submeter
    # entities so the data-quality counts include the Meross/etc that are
    # configured as `grid_submeters`. Without this, the wizard score reports
    # OK while the submeter feeds for ML training are silently ignored.
    all_entities: dict[str, str] = dict(sensors)
    for sm in submeters:
        sm_name = (sm.get("name") or "").strip()
        sm_eid  = sm.get("entity", "")
        if sm_name and sm_eid:
            all_entities[f"submeter_{sm_name}"] = sm_eid

    samples: dict[str, int] = {}
    source_ok = False
    source_label = "none"

    # ── Try InfluxDB v1 (official HA integration, user/password auth) ────────
    # In the old HA→Influx integration the MEASUREMENT is the unit (%, W, kWh…)
    # and entity_id is a TAG (without domain prefix). Querying FROM "<entity>"
    # therefore returns nothing — we must regex-match all measurements and
    # filter by the entity_id tag, same pattern used in ha_history_influx().
    if (only in (None, "influxdb")) and influx_cfg.get("host"):
        try:
            url  = f"http://{influx_cfg['host']}:{influx_cfg.get('port', 8086)}"
            db   = influx_cfg.get("db", "homeassistant")
            user = influx_cfg.get("username", "")
            pwd  = influx_cfg.get("password", "")
            any_response = False
            for role, entity_id in all_entities.items():
                if not entity_id:
                    continue
                entity_short = entity_id.split(".")[-1] if "." in entity_id else entity_id
                q = (f'SELECT count("value") FROM /.*/ '
                     f'WHERE "entity_id" = \'{entity_short}\' AND time > now() - 60d')
                resp, err, _ = _influx_query(url, db, q, user, pwd)
                if err or not resp:
                    continue
                any_response = True
                try:
                    # Sum across every series (one per matching measurement).
                    total = 0
                    for series in resp.json()["results"][0].get("series", []):
                        cols = series.get("columns", [])
                        ci   = cols.index("count") if "count" in cols else 1
                        for v in series.get("values", []):
                            total += int(v[ci] or 0)
                    if total:
                        samples[entity_id] = total
                except (KeyError, IndexError, TypeError, ValueError):
                    pass
            # Only claim "influxdb" as the source when at least one query
            # returned data; otherwise let the HA recorder fallback try.
            if any_response and samples:
                source_ok    = True
                source_label = "influxdb"
        except Exception as e:
            log.warning(f"InfluxDB quality check failed: {e}")

    # ── Try MariaDB direct (HA recorder backed by MariaDB/MySQL) ─────────────
    if (only in (None, "mariadb")) and not source_ok and mariadb_cfg.get("host"):
        try:
            for role, entity_id in all_entities.items():
                if not entity_id:
                    continue
                rows = _mariadb_wizard_history(entity_id, 60, mariadb_cfg)
                if rows:
                    samples[entity_id] = len(rows)
            if samples:
                source_ok    = True
                source_label = "mariadb"
        except Exception as e:
            log.warning(f"MariaDB quality check failed: {e}")

    # ── Fall back to HA recorder REST ────────────────────────────────────────
    if (only in (None, "ha_recorder")) and not source_ok:
        source_label = "ha_recorder"
        for role, entity_id in all_entities.items():
            if not entity_id:
                continue
            try:
                rows = ha_history(entity_id, days=14)
                if rows:
                    samples[entity_id] = len(rows)
            except Exception:
                pass

    # ── Score 0-100 ──────────────────────────────────────────────────────────
    key_roles   = ["grid_power", "solar_power", "battery_soc", "battery_power"]
    bonus_roles = ["solar_fc_today", "temp_outdoor", "temp_indoor", "battery_mode_select"]
    filled_key   = sum(1 for r in key_roles   if sensors.get(r))
    filled_bonus = sum(1 for r in bonus_roles if sensors.get(r))
    source_bonus = (30 if source_label == "influxdb"
                    else 20 if source_label == "mariadb"
                    else 10)
    avg_samples  = (sum(samples.values()) / max(len(samples), 1)) if samples else 0
    sample_score = min(30, int(avg_samples / 500 * 30))   # max 30 pts at 500+ samples/entity
    sensor_score = int(filled_key / len(key_roles) * 25) + int(filled_bonus / len(bonus_roles) * 15)
    score = min(100, source_bonus + sensor_score + sample_score)

    # ── Push sensor to HA ────────────────────────────────────────────────────
    ha_post("/api/states/sensor.energy_optimizer_data_quality", {
        "state": str(score),
        "attributes": {
            "unit_of_measurement": "%",
            "friendly_name":       "Energy Optimizer Data Quality",
            "icon":                "mdi:database-check",
            "source":              source_label,
            "total_samples":       sum(samples.values()),
            "entities_with_data":  len(samples),
        }
    })

    # ── Get oldest record (first_ts) from InfluxDB ──────────────────────────
    # Query by entity_id tag across all measurements (see comment above on
    # the wrong "FROM <entity>" pattern that this used to use).
    first_ts = None
    if influx_cfg.get("host"):
        try:
            url  = f"http://{influx_cfg['host']}:{influx_cfg.get('port', 8086)}"
            db   = influx_cfg.get("db", "homeassistant")
            user = influx_cfg.get("username", "")
            pwd  = influx_cfg.get("password", "")
            probe = next((v for v in sensors.values() if v), None) or cfg("sensor_battery_soc", "")
            if probe:
                short = probe.split(".")[-1] if "." in probe else probe
                resp_f, _, _ = _influx_query(
                    url, db,
                    f'SELECT "value" FROM /.*/ WHERE "entity_id" = \'{short}\' '
                    f'ORDER BY time ASC LIMIT 1',
                    user, pwd)
                if resp_f:
                    try:
                        data = resp_f.json()
                        first_ts = data["results"][0]["series"][0]["values"][0][0][:10]
                    except (KeyError, IndexError, TypeError, ValueError):
                        pass
        except Exception as e:
            log.debug(f"first_ts query failed: {e}")

    return jsonify({
        "ok":            True,
        "score":         score,
        "source":        source_label,
        "samples":       samples,
        "total_samples": sum(samples.values()),
        "first_ts":      first_ts,
        "error":         None,
    })


@app.route("/api/wizard/test-entity")
def api_wizard_test_entity():
    entity_id = request.args.get("entity_id", "")
    if not entity_id:
        return jsonify({"ok": False, "error": "entity_id required"}), 400
    state = ha_state(entity_id)
    if state:
        attrs = state.get("attributes", {})
        return jsonify({
            "ok":           True,
            "state":        state.get("state"),
            "unit":         attrs.get("unit_of_measurement", ""),
            "device_class": attrs.get("device_class", ""),
            "friendly_name":attrs.get("friendly_name", entity_id),
        })
    return jsonify({"ok": False, "error": "Entity not found or unavailable"})

# ── Startup ──────────────────────────────────────────────────────────────────
def main():
    global _scheduler_ref
    _load_setup_cache()
    _load_wizard_cache()
    _load_surplus_state()
    _load_cooling_gate_state()
    _load_pool_manual_state()
    _load_hvac_decision_state()
    _load_log_summary_state()
    # Attach the quiet-mode filter to the energy-optimizer logger AND to the
    # root logger so it intercepts any nested logger that propagates up
    # (Flask, APScheduler, libraries). The filter is a no-op until run_cycle
    # turns `enabled` on for the duration of the cycle.
    log.addFilter(_QUIET_FILTER)
    logging.getLogger().addFilter(_QUIET_FILTER)

    _check_version_sync()
    log.info("═══════════════════════════════════════")
    log.info(f"   Energy Optimizer v{ADD_ON_VERSION} — HAOS")
    log.info("═══════════════════════════════════════")
    log.info(f"  Supervisor token:        {'OK' if HA_TOKEN else 'NOT FOUND'}")
    log.info(f"  Email enabled:           {cfg('notify_email_enabled', True)}")
    log.info(f"  Telegram daily:          {cfg('notify_telegram_daily_enabled', True)}")
    log.info(f"  Telegram instant alerts: {cfg('notify_telegram_alerts_enabled', True)}")

    v5_enabled = bool(cfg("v5_engine_enabled", False))
    log.info(f"  v5 engine:               {'ENABLED' if v5_enabled else 'disabled (legacy v4 cycle active)'}")
    if v5_enabled:
        log.warning(
            "  v5 engine is enabled but the integration adapter is not wired yet "
            "in this release. Falling back to legacy v4 cycle. See docs/v5-wiring.md."
        )

    try:
        run_setup_integrity_check()
    except Exception as e:
        log.error(f"Setup integrity check failed to run: {e}")

    if not MODEL_FILE.exists():
        log.info("No saved model — starting initial training...")
        threading.Thread(target=train_model, daemon=True).start()

    threading.Thread(target=_refresh_consumption_cache, daemon=True).start()

    scheduler = BackgroundScheduler(timezone="Europe/Madrid")
    _scheduler_ref = scheduler

    # If a manual pool OFF was pending when the addon stopped, restore it.
    _reschedule_pool_manual_off_after_restart()

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
