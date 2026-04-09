#!/usr/bin/env python3
"""
Energy Optimizer — Add-on para Home Assistant OS
Motor de gestión energética: batería, aerotermia, depuradora
Lógica: reglas fijas de tarifa + modelo ML scikit-learn
Historial: API REST /api/history del propio HA (recorder nativo)
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, request

# ─── Rutas de datos persistentes (volumen /data del add-on) ─────────────────
DATA_DIR       = Path("/data")
OPTIONS_FILE   = DATA_DIR / "options.json"
MODEL_FILE     = DATA_DIR / "model.pkl"
DECISIONS_FILE = DATA_DIR / "decisions.json"
DATA_DIR.mkdir(exist_ok=True)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("energy-optimizer")

# ─── Opciones ─────────────────────────────────────────────────────────────────
def load_options() -> dict:
    """
    En HAOS el Supervisor inyecta las opciones del add-on en /data/options.json.
    El token de la API de HA está disponible como variable de entorno SUPERVISOR_TOKEN.
    """
    path = Path("/data/options.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    log.warning("options.json no encontrado, usando valores por defecto")
    return {}

OPT = load_options()

# ─── Cliente HA (usa Supervisor token, no necesita token manual) ──────────────
# En HAOS el Supervisor expone HA en http://supervisor/core
HA_BASE   = "http://supervisor/core"
HA_TOKEN  = os.environ.get("SUPERVISOR_TOKEN", "")
HEADERS   = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type":  "application/json",
}

def ha_get(path: str, params: dict = None) -> dict | list | None:
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

def ha_float(entity_id: str, default: float = 0.0) -> float:
    s = ha_state(entity_id)
    if s:
        try:
            return float(s["state"])
        except (ValueError, KeyError, TypeError):
            pass
    log.debug(f"No se pudo leer float de {entity_id}, usando {default}")
    return default

def ha_service(domain: str, service: str, data: dict = None) -> bool:
    log.info(f"  → Servicio: {domain}.{service} {data or ''}")
    return ha_post(f"/api/services/{domain}/{service}", data or {})

def ha_set_number(entity_id: str, value: float) -> bool:
    return ha_service("number", "set_value", {"entity_id": entity_id, "value": value})

def ha_call_script(script_entity: str) -> bool:
    return ha_service("script", "turn_on", {"entity_id": script_entity})

def ha_switch(entity_id: str, turn_on: bool) -> bool:
    action = "turn_on" if turn_on else "turn_off"
    return ha_service("switch", action, {"entity_id": entity_id})

# ─── Historial desde API REST de HA (recorder nativo) ────────────────────────
def ha_history(entity_id: str, days: int = 45) -> list[dict]:
    """
    Llama a /api/history/period/{start} filtrando por entity_id.
    Devuelve lista de dicts {last_changed, state}.
    Compatible con el recorder SQLite por defecto de HAOS.
    """
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    result = ha_get(
        f"/api/history/period/{start}",
        params={"filter_entity_id": entity_id, "minimal_response": "true"},
    )
    if result and isinstance(result, list) and len(result) > 0:
        return result[0]
    return []

# ─── Precios de electricidad ──────────────────────────────────────────────────
# Tarifa discriminación horaria estándar peninsular
PRICE = {
    "punta": 0.30,   # 10-14h y 18-22h (lun-vie)
    "llano": 0.18,   # resto horas laborables
    "valle": 0.08,   # 00-08h + sáb/dom completo
    "venta": 0.06,   # venta excedentes (fijo siempre)
}

def precio_actual() -> dict:
    """Determina la franja tarifaria según hora y día de la semana."""
    now     = datetime.now()
    hora    = now.hour
    festivo = now.weekday() >= 5  # sábado=5, domingo=6

    if festivo or (0 <= hora < 8):
        periodo = "valle"
    elif (10 <= hora < 14) or (18 <= hora < 22):
        periodo = "punta"
    else:
        periodo = "llano"

    return {
        "periodo":     periodo,
        "precio_kwh":  PRICE[periodo],
        "venta_kwh":   PRICE["venta"],
        "hora":        hora,
        "festivo":     festivo,
    }

# ─── Lectura unificada de sensores ────────────────────────────────────────────
def leer_sensores() -> dict:
    o = OPT
    return {
        # Eléctricos
        "battery_soc":          ha_float(o.get("sensor_battery_soc",      "sensor.battery_state_of_capacity")),
        "battery_power":        ha_float(o.get("sensor_battery_power",     "sensor.battery_charge_discharge_power")),
        "grid_power":           ha_float(o.get("sensor_grid_power",        "sensor.acometida_general_power")),
        # Solar forecast
        "solar_current_hour":   ha_float(o.get("sensor_solar_current_hour","sensor.energy_current_hour")),
        "solar_next_hour":      ha_float(o.get("sensor_solar_next_hour",   "sensor.energy_next_hour")),
        "solar_today":          ha_float(o.get("sensor_solar_today",       "sensor.energy_production_today")),
        "solar_tomorrow":       ha_float(o.get("sensor_solar_tomorrow",    "sensor.energy_production_tomorrow")),
        # Temperatura
        "temp_outdoor":         ha_float(o.get("sensor_temp_outdoor",      "sensor.ebusd_broadcast_outsidetemp_temp2")),
        "temp_salon":           ha_float(o.get("sensor_temp_salon",        "sensor.media_salon")),
        # Depuradora
        "pool_hours_day":       ha_float(o.get("sensor_pool_hours_day",    "sensor.depuradora_encendida_24h")),
        "pool_hours_week":      ha_float(o.get("sensor_pool_hours_week",   "sensor.depuradora_encendida_semana")),
        # Timestamp
        "ts": datetime.now().isoformat(),
    }

# ─── Modelo ML ────────────────────────────────────────────────────────────────
def _history_to_df(rows: list[dict]) -> pd.DataFrame | None:
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
    df["hour"]          = df.index.hour
    df["weekday"]       = df.index.weekday
    df["month"]         = df.index.month
    df["lag1"]          = df["value"].shift(1)
    df["lag4"]          = df["value"].shift(4)
    df["roll4"]         = df["value"].rolling(4).mean()
    df["solar_proxy"]   = df.index.hour.map(lambda h: 1 if 8 <= h <= 19 else 0)
    return df.dropna()

FEATURES = ["hour", "weekday", "month", "lag1", "lag4", "roll4", "solar_proxy"]

def entrenar_modelo() -> bool:
    """Entrena GradientBoostingRegressor con historial del SOC de batería."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    import numpy as np

    log.info("═══ Entrenamiento ML iniciado ═══")
    entity = OPT.get("sensor_battery_soc", "sensor.battery_state_of_capacity")
    rows   = ha_history(entity, days=60)
    df     = _history_to_df(rows)

    if df is None:
        log.warning(f"Historial insuficiente ({len(rows)} muestras). Solo reglas fijas.")
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
    log.info(f"Modelo entrenado: {len(df)} muestras, R² CV={np.mean(scores):.3f}")
    return True

def predecir_soc_optimo(sensores: dict) -> dict:
    """Predice el SOC necesario para la próxima hora. Fallback a reglas si no hay modelo."""
    if MODEL_FILE.exists():
        try:
            art   = joblib.load(MODEL_FILE)
            pipe  = art["pipeline"]
            now   = datetime.now()
            X = pd.DataFrame([{
                "hour":       now.hour,
                "weekday":    now.weekday(),
                "month":      now.month,
                "lag1":       sensores["battery_soc"],
                "lag4":       sensores["battery_soc"],
                "roll4":      sensores["battery_soc"],
                "solar_proxy": 1 if 8 <= now.hour <= 19 else 0,
            }])
            pred = float(pipe.predict(X[FEATURES])[0])
            pred = max(0.0, min(100.0, pred))
            return {
                "soc_predicho": pred,
                "metodo":       "ml",
                "r2":           art.get("r2_cv_mean"),
                "entrenado":    art.get("trained_at"),
            }
        except Exception as e:
            log.warning(f"Error en predicción ML: {e}. Fallback a reglas.")

    # Fallback heurístico
    solar_manana = sensores.get("solar_tomorrow", 0)
    pred = 80.0 if solar_manana < OPT.get("solar_tomorrow_irrisoria_kwh", 2.0) else 50.0
    return {"soc_predicho": pred, "metodo": "reglas_fallback"}

# ─── Lógica aerotermia ────────────────────────────────────────────────────────
def _es_verano() -> bool:
    m = datetime.now().month
    return OPT.get("summer_start_month", 6) <= m <= OPT.get("summer_end_month", 9)

def decidir_aerotermia(sensores: dict) -> dict:
    """
    Decide el objetivo de temperatura de la aerotermia.
    La propia máquina tiene inercia térmica y actúa inteligentemente;
    aquí solo fijamos el setpoint objetivo.
    """
    soc         = sensores["battery_soc"]
    batt_power  = sensores["battery_power"]
    temp_salon  = sensores["temp_salon"]
    hora        = datetime.now().hour
    es_dia      = 8 <= hora <= 20

    # "Electricidad gratis": batería cargando con placas y SOC ≥ 99%
    electricidad_gratis = (soc >= 99 and batt_power > 0)

    if _es_verano():
        entity = OPT.get("number_hvac_cool", "number.ebusd_ctls2_z1coolingtemp_tempv")
        if electricidad_gratis:
            target, reason = 16.0, "Electricidad gratis (SOC≥99% cargando con placas)"
        elif temp_salon > 26 and es_dia:
            target, reason = 20.0, f"Salón muy caliente ({temp_salon:.1f}°C > 26°C)"
        else:
            target, reason = 25.0, "Temperatura base verano"
    else:
        entity = OPT.get("number_hvac_heat", "number.ebusd_ctls2_z1manualtemp_tempv")
        if electricidad_gratis:
            target, reason = 18.5, "Electricidad gratis (SOC≥99% cargando con placas)"
        elif temp_salon < 16 and es_dia:
            target, reason = 17.0, f"Salón muy frío ({temp_salon:.1f}°C < 16°C)"
        else:
            target, reason = 16.0, "Temperatura base invierno"

    return {
        "entity": entity,
        "target": target,
        "reason": reason,
        "estacion": "verano" if _es_verano() else "invierno",
    }

# ─── Lógica batería ───────────────────────────────────────────────────────────
def decidir_bateria(sensores: dict, precio: dict, prediccion: dict) -> dict:
    """
    Prioridad 1: minimizar gasto €
    Prioridad 2: mantener ≥80% si producción mañana es irrisoria
    """
    soc           = sensores["battery_soc"]
    periodo       = precio["periodo"]
    solar_manana  = sensores["solar_tomorrow"]
    irrisoria     = solar_manana < OPT.get("solar_tomorrow_irrisoria_kwh", 2.0)
    thr_emerg     = OPT.get("battery_emergency_threshold", 10)
    thr_low       = OPT.get("battery_low_threshold",       30)
    thr_med       = OPT.get("battery_medium_threshold",    50)

    s30 = OPT.get("script_charge_30", "script.1700687735587")
    s50 = OPT.get("script_charge_50", "script.1700687757430")
    s80 = OPT.get("script_charge_80", "script.cargar_bateria_80")
    s99 = OPT.get("script_charge_99", "script.cargar_bateria_99")

    # Punta: NO cargar de red salvo emergencia absoluta
    if periodo == "punta":
        if soc < thr_emerg:
            return {"script": s30, "reason": f"EMERGENCIA en punta: SOC={soc:.0f}% < {thr_emerg}%"}
        return {"script": None, "reason": f"Punta ({PRICE['punta']}€/kWh): no cargar de red"}

    # Valle: hora más barata → máxima oportunidad de carga
    if periodo == "valle":
        if irrisoria and soc < 80:
            return {"script": s80, "reason": f"Valle + producción mañana baja ({solar_manana:.1f} kWh) → cargar al 80%"}
        if soc < thr_low:
            return {"script": s30, "reason": f"Valle + SOC bajo ({soc:.0f}%) → cargar mínimo 30%"}
        if soc < thr_med:
            return {"script": s50, "reason": f"Valle + SOC moderado ({soc:.0f}%) → cargar 50%"}
        return {"script": None, "reason": f"Valle, batería suficiente ({soc:.0f}%)"}

    # Llano: actuar solo si batería baja
    if periodo == "llano":
        if soc < thr_emerg:
            return {"script": s30, "reason": f"Llano + SOC crítico ({soc:.0f}%) → carga emergencia 30%"}
        if soc < 20:
            return {"script": s50, "reason": f"Llano + SOC muy bajo ({soc:.0f}%) → cargar 50%"}
        return {"script": None, "reason": f"Llano, sin acción de carga (SOC={soc:.0f}%)"}

    return {"script": None, "reason": "Sin acción de carga"}

# ─── Lógica depuradora ────────────────────────────────────────────────────────
def decidir_depuradora(sensores: dict, precio: dict) -> dict:
    """
    Jun–Sep: 1h/día encendida  (~1.2 kWh)
    Oct–May: 1h/semana encendida
    Preferencia: excedente solar > valle > llano (nunca punta si se puede evitar)
    """
    month    = datetime.now().month
    verano   = OPT.get("summer_start_month", 6) <= month <= OPT.get("summer_end_month", 9)
    horas_dia  = sensores.get("pool_hours_day", 0)
    horas_sem  = sensores.get("pool_hours_week", 0)

    # ¿Ya cumplidas las horas mínimas?
    if verano and horas_dia >= 1.0:
        return {"action": False, "reason": f"Horas diarias cumplidas ({horas_dia:.1f}h ≥ 1h/día)"}
    if not verano and horas_sem >= 1.0:
        return {"action": False, "reason": f"Horas semanales cumplidas ({horas_sem:.1f}h ≥ 1h/semana)"}

    # Horas pendientes → decidir cuándo encender
    solar_now   = sensores["solar_current_hour"]
    soc         = sensores["battery_soc"]
    periodo     = precio["periodo"]

    # Mejor momento: excedente solar + batería alta
    if solar_now > 1.2 and soc > 50:
        return {"action": True, "reason": f"Excedente solar disponible ({solar_now:.2f} kWh/h), SOC={soc:.0f}%"}

    # Segundo mejor: precio valle
    if periodo == "valle":
        return {"action": True, "reason": f"Precio valle ({PRICE['valle']}€/kWh), horas pendientes"}

    # Urgencia: casi fin del día/semana sin horas
    if verano and horas_dia < 0.1 and datetime.now().hour >= 20:
        return {"action": True, "reason": f"Urgente: {horas_dia:.1f}h acumuladas, hora tardía"}

    return {"action": False, "reason": f"Esperando mejor momento (periodo={periodo}, solar={solar_now:.2f} kWh/h)"}

# ─── Ciclo principal de decisión ──────────────────────────────────────────────
def ciclo_decision() -> dict:
    log.info("━━━ Ciclo de decisión ━━━━━━━━━━━━━━━━━━━━━━")
    sensores   = leer_sensores()
    precio     = precio_actual()
    prediccion = predecir_soc_optimo(sensores)

    decision = {
        "timestamp":   datetime.now().isoformat(),
        "sensores":    sensores,
        "precio":      precio,
        "prediccion":  prediccion,
        "acciones":    [],
        "omisiones":   [],
    }

    # ── 1. Batería ────────────────────────────────────────────
    bat = decidir_bateria(sensores, precio, prediccion)
    if bat["script"]:
        ok = ha_call_script(bat["script"])
        decision["acciones"].append({"tipo": "bateria", "script": bat["script"], "reason": bat["reason"], "ok": ok})
        log.info(f"  [BATERÍA] {bat['reason']}")
    else:
        decision["omisiones"].append({"tipo": "bateria", "reason": bat["reason"]})
        log.info(f"  [BATERÍA] Sin acción — {bat['reason']}")

    # ── 2. Aerotermia ─────────────────────────────────────────
    hvac = decidir_aerotermia(sensores)
    ok   = ha_set_number(hvac["entity"], hvac["target"])
    decision["acciones"].append({
        "tipo": "aerotermia", "entity": hvac["entity"],
        "valor": hvac["target"], "reason": hvac["reason"], "ok": ok,
    })
    log.info(f"  [AEROTERMIA/{hvac['estacion'].upper()}] {hvac['reason']} → {hvac['target']}°C")

    # ── 3. Depuradora ─────────────────────────────────────────
    pool        = decidir_depuradora(sensores, precio)
    switch_ent  = OPT.get("switch_pool", "switch.depuradora")
    ok          = ha_switch(switch_ent, pool["action"])
    key         = "acciones" if pool["action"] else "omisiones"
    decision[key].append({"tipo": "depuradora", "action": pool["action"], "reason": pool["reason"], "ok": ok})
    log.info(f"  [DEPURADORA] {'ON' if pool['action'] else 'OFF'} — {pool['reason']}")

    # ── Guardar decisión ──────────────────────────────────────
    _guardar_decision(decision)
    return decision

def _guardar_decision(d: dict):
    history = []
    if DECISIONS_FILE.exists():
        try:
            history = json.loads(DECISIONS_FILE.read_text())
        except Exception:
            history = []
    history.append(d)
    history = history[-500:]  # mantener últimas 500
    DECISIONS_FILE.write_text(json.dumps(history, indent=2, default=str))

# ─── Panel web (Flask) ────────────────────────────────────────────────────────
app = Flask(__name__)

PANEL = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Energy Optimizer</title>
<style>
:root{--bg:#0f172a;--s:#1e293b;--b:#334155;--a:#38bdf8;--g:#4ade80;--y:#fbbf24;--r:#f87171;--t:#e2e8f0;--m:#94a3b8}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t);font-family:system-ui,sans-serif;padding:1rem;font-size:14px}
h1{color:var(--a);font-size:1.3rem;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem}
h2{font-size:.7rem;color:var(--m);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.6rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.6rem;margin-bottom:1.4rem}
.card{background:var(--s);border-radius:.75rem;padding:.9rem;border:1px solid var(--b)}
.metric{font-size:1.8rem;font-weight:700;color:var(--a);line-height:1}
.label{font-size:.72rem;color:var(--m);margin-top:.3rem}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:.3rem;font-size:.72rem;font-weight:600}
.punta{color:var(--r);background:rgba(248,113,113,.12)}
.valle{color:var(--g);background:rgba(74,222,128,.12)}
.llano{color:var(--y);background:rgba(251,191,36,.12)}
.actions{display:flex;gap:.5rem;margin-bottom:1.4rem;flex-wrap:wrap}
button{background:var(--a);color:#0f172a;border:none;padding:.45rem 1rem;border-radius:.5rem;font-weight:700;cursor:pointer;font-size:.8rem}
button:hover{opacity:.85}
button.sec{background:var(--y)}
table{width:100%;border-collapse:collapse;font-size:.78rem}
td,th{padding:.45rem .6rem;border-bottom:1px solid var(--b);text-align:left}
th{color:var(--m);font-weight:500}
.ok{color:var(--g)}.skip{color:var(--m)}.err{color:var(--r)}
.tag{display:inline-block;padding:.1rem .4rem;border-radius:.25rem;font-size:.68rem;background:var(--b);color:var(--m)}
.tag.bat{background:rgba(56,189,248,.12);color:var(--a)}
.tag.hvac{background:rgba(74,222,128,.12);color:var(--g)}
.tag.pool{background:rgba(251,191,36,.12);color:var(--y)}
</style>
</head>
<body>
<h1>⚡ Energy Optimizer</h1>
<div id="kpis" class="grid"></div>
<div class="actions">
  <button onclick="runCycle()">▶ Ejecutar ciclo ahora</button>
  <button class="sec" onclick="retrain()">🔁 Re-entrenar modelo</button>
</div>
<h2>Últimas decisiones</h2>
<div class="card" style="overflow-x:auto">
  <table><thead><tr><th>Hora</th><th>Tipo</th><th>Motivo</th><th>OK</th></tr></thead>
  <tbody id="log"></tbody></table>
</div>
<script>
const PC={punta:'punta',valle:'valle',llano:'llano'};
async function load(){
  const [s,d]=await Promise.all([fetch('/api/status').then(r=>r.json()),fetch('/api/decisions?limit=20').then(r=>r.json())]);
  const {sensores:ss,precio:p,modelo:m}=s;
  document.getElementById('kpis').innerHTML=`
    <div class="card"><div class="metric">${ss.battery_soc?.toFixed(0)||'--'}%</div><div class="label">Batería SOC</div></div>
    <div class="card"><div class="metric">${ss.solar_today?.toFixed(1)||'--'}</div><div class="label">Solar hoy (kWh)</div></div>
    <div class="card"><div class="metric">${ss.solar_tomorrow?.toFixed(1)||'--'}</div><div class="label">Solar mañana (kWh)</div></div>
    <div class="card"><div class="metric ${PC[p.periodo]}">${p.precio_kwh?.toFixed(2)||'--'}€</div><div class="label">Precio · <span class="badge ${PC[p.periodo]}">${p.periodo}</span></div></div>
    <div class="card"><div class="metric">${ss.temp_salon?.toFixed(1)||'--'}°C</div><div class="label">Temp salón</div></div>
    <div class="card"><div class="metric">${ss.temp_outdoor?.toFixed(1)||'--'}°C</div><div class="label">Temp exterior</div></div>
    <div class="card"><div class="metric" style="font-size:1rem">${m?.r2_cv_mean!=null?'R²='+m.r2_cv_mean.toFixed(2):'Sin modelo'}</div><div class="label">Modelo ML</div></div>`;
  const rows=d.slice().reverse().flatMap(dec=>{
    const t=new Date(dec.timestamp).toLocaleTimeString('es',{hour:'2-digit',minute:'2-digit'});
    return [
      ...(dec.acciones||[]).map(a=>`<tr><td>${t}</td><td><span class="tag ${a.tipo==='bateria'?'bat':a.tipo==='aerotermia'?'hvac':'pool'}">${a.tipo}</span></td><td class="ok">${a.reason}</td><td class="${a.ok?'ok':'err'}">${a.ok?'✓':'✗'}</td></tr>`),
      ...(dec.omisiones||[]).map(a=>`<tr><td>${t}</td><td><span class="tag">${a.tipo}</span></td><td class="skip">${a.reason}</td><td class="skip">–</td></tr>`)
    ];
  }).join('');
  document.getElementById('log').innerHTML=rows||'<tr><td colspan="4" style="color:var(--m)">Sin decisiones aún</td></tr>';
}
async function runCycle(){
  const d=await fetch('/api/run',{method:'POST'}).then(r=>r.json());
  alert(`Ciclo ejecutado\n✓ ${d.acciones?.length||0} acciones\n– ${d.omisiones?.length||0} omisiones`);
  load();
}
async function retrain(){
  const d=await fetch('/api/retrain',{method:'POST'}).then(r=>r.json());
  alert(d.message);load();
}
load();setInterval(load,30000);
</script>
</body></html>"""

@app.route("/")
def index():
    return PANEL

@app.route("/api/status")
def api_status():
    sensores = leer_sensores()
    precio   = precio_actual()
    modelo   = {}
    if MODEL_FILE.exists():
        try:
            art    = joblib.load(MODEL_FILE)
            modelo = {k: art.get(k) for k in ("trained_at", "r2_cv_mean", "n_samples")}
        except Exception:
            pass
    return jsonify({"sensores": sensores, "precio": precio, "modelo": modelo})

@app.route("/api/decisions")
def api_decisions():
    limit = int(request.args.get("limit", 50))
    if DECISIONS_FILE.exists():
        return jsonify(json.loads(DECISIONS_FILE.read_text())[-limit:])
    return jsonify([])

@app.route("/api/run", methods=["POST"])
def api_run():
    return jsonify(ciclo_decision())

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    ok = entrenar_modelo()
    return jsonify({
        "ok": ok,
        "message": "Modelo reentrenado correctamente ✓" if ok else "Historial insuficiente — el sistema usará solo reglas fijas"
    })

@app.route("/api/options")
def api_options():
    safe = {k: v for k, v in OPT.items() if "token" not in k.lower()}
    return jsonify(safe)

# ─── Arranque ─────────────────────────────────────────────────────────────────
def main():
    log.info("═══════════════════════════════════════")
    log.info("   Energy Optimizer v1.0 — HAOS")
    log.info("═══════════════════════════════════════")
    log.info(f"  Supervisor token: {'OK' if HA_TOKEN else 'NO ENCONTRADO'}")

    # Entrenamiento inicial si no hay modelo guardado
    if not MODEL_FILE.exists():
        log.info("Sin modelo previo — intentando entrenamiento inicial...")
        threading.Thread(target=entrenar_modelo, daemon=True).start()

    # Scheduler APScheduler
    scheduler = BackgroundScheduler(timezone="Europe/Madrid")
    interval  = OPT.get("decision_interval_minutes", 15)
    scheduler.add_job(ciclo_decision, "interval", minutes=interval, id="ciclo",
                      next_run_time=datetime.now() + timedelta(seconds=15))

    cron = OPT.get("retrain_cron", "0 3 * * *").split()
    if len(cron) == 5:
        scheduler.add_job(
            entrenar_modelo, "cron",
            minute=cron[0], hour=cron[1], day=cron[2], month=cron[3], day_of_week=cron[4],
            id="retrain",
        )

    scheduler.start()
    log.info(f"  Ciclo: cada {interval} min · Reentrenamiento: {OPT.get('retrain_cron','0 3 * * *')}")

    # Servidor Flask (Ingress de HAOS escucha en 8765)
    app.run(host="0.0.0.0", port=8765, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
