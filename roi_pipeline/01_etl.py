"""ROI pipeline · stage 1.1 — ETL from InfluxDB to hourly DataFrame.

Pulls 12 months of canonical sensor data from the HA InfluxDB v1 add-on
(192.168.1.131:8086, db=homeassistant), resamples to 1h means, joins
into a single DataFrame, and runs sanity checks.

Sign convention (verified against memory project_sensor_truth.md):
- inverter_input_power: DC bruto; always >= 0
- battery_charge_discharge_power: +ve = charging, -ve = discharging
- battery_state_of_capacity: %
- acometida_general_power: +ve = exporting, -ve = importing

Output:
- data/etl_12m_1h.parquet: full DataFrame
- stdout: stats summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

INFLUX_URL = "http://192.168.1.131:8086"
INFLUX_DB  = "homeassistant"
DAYS       = 365
RESAMPLE   = "1h"

SENSORS = {
    "solar_dc_w":   "inverter_input_power",
    "battery_w":    "battery_charge_discharge_power",
    "soc_pct":      "battery_state_of_capacity",
    "grid_w":       "acometida_general_power",
    "ac_out_w":     "inverter_active_power",
    "temp_out_c":   "ebusd_broadcast_outsidetemp_temp2",
}

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def _influx_query(q: str) -> dict:
    r = requests.get(f"{INFLUX_URL}/query",
                     params={"db": INFLUX_DB, "q": q, "epoch": "ms"},
                     timeout=120)
    r.raise_for_status()
    return r.json()


def _fetch_series(entity_id: str, days: int, resample: str, label: str) -> pd.Series:
    """One InfluxQL query that resamples server-side. The HA→Influx integration
    stores entity_id as a TAG and the unit as the measurement, so we use
    `FROM /.*/` to scan all measurements and filter by tag."""
    # InfluxQL granularity: 1h, 15m, etc.
    q = (f'SELECT mean("value") FROM /.*/ '
         f'WHERE "entity_id" = \'{entity_id}\' '
         f'AND time > now()-{days}d '
         f'GROUP BY time({resample}) fill(null)')
    res = _influx_query(q)
    rows = []
    for r in res.get("results", [{}])[0].get("series", []):
        for ts, val in r.get("values", []):
            rows.append((ts, val))
    if not rows:
        return pd.Series(dtype=float, name=label)
    df = pd.DataFrame(rows, columns=["ts_ms", "value"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    s = df.set_index("ts")["value"].rename(label)
    # Multiple measurements may have produced parallel rows for same ts;
    # group and take mean to collapse them into a single time series.
    return s.groupby(level=0).mean()


def main() -> int:
    print(f"[ETL] Pulling {DAYS} days of {len(SENSORS)} sensors at {RESAMPLE} resolution")
    series = {}
    for label, entity in SENSORS.items():
        s = _fetch_series(entity, DAYS, RESAMPLE, label)
        series[label] = s
        first = s.index.min() if len(s) else None
        last  = s.index.max() if len(s) else None
        print(f"  · {label:12s} ({entity:45s}): {len(s):6d} rows  "
              f"[{first} → {last}]")

    df = pd.concat(series.values(), axis=1)
    df = df.sort_index()
    df.index.name = "ts_utc"

    print(f"\n[ETL] Joined frame (raw): {df.shape[0]} rows × {df.shape[1]} cols  "
          f"[{df.index.min()} → {df.index.max()}]")

    # ── Semantic fill ────────────────────────────────────────────────────────
    # HA→Influx writes ~every 60-90 s when active. Hourly nulls happen for two
    # reasons: (a) the sensor sits idle near 0 W and the integration debounces
    # writes, (b) the source went offline (inverter down, HA reboot). Naive
    # ffill propagates a one-off discharge spike across the rest of the day,
    # inflating battery throughput >100% round-trip. Fix: ffill with a 1-hour
    # cap, then fill the rest with 0 (assume idle for true gaps).
    for c in ["solar_dc_w", "ac_out_w", "battery_w", "grid_w"]:
        df[c] = df[c].ffill(limit=1).fillna(0.0)
    # State sensors (SOC, outdoor temp) change slowly and ffill over longer
    # gaps is the right call — battery doesn't drift on its own without flow.
    for c in ["soc_pct", "temp_out_c"]:
        df[c] = df[c].ffill().bfill()

    # ── Derived energy fields (Wh per hour = power_W * 1.0h) ─────────────────
    df["solar_kwh"]    = df["solar_dc_w"]  / 1000.0
    df["batt_kwh"]     = df["battery_w"]   / 1000.0  # +ve = charging
    df["batt_chg_kwh"] = df["batt_kwh"].clip(lower=0)
    df["batt_dis_kwh"] = (-df["batt_kwh"]).clip(lower=0)
    df["grid_kwh"]     = df["grid_w"]      / 1000.0  # +ve = exporting
    df["export_kwh"]   = df["grid_kwh"].clip(lower=0)
    df["import_kwh"]   = (-df["grid_kwh"]).clip(lower=0)

    # House consumption from Kirchhoff: solar = battery_charge + grid_export + house
    # → house = solar - battery - grid (all in same sign convention)
    df["house_kwh"]    = df["solar_kwh"] - df["batt_kwh"] - df["grid_kwh"]

    # ── Time features ────────────────────────────────────────────────────────
    idx_local = df.index.tz_convert("Europe/Madrid")
    df["hour"]       = idx_local.hour
    df["weekday"]    = idx_local.weekday
    df["month"]      = idx_local.month
    df["is_weekend"] = (idx_local.weekday >= 5).astype(int)

    # ── Sanity checks ────────────────────────────────────────────────────────
    print("\n[QC] Null counts per column:")
    for c in df.columns:
        null_pct = df[c].isna().mean() * 100
        print(f"  · {c:14s}: {df[c].isna().sum():5d} nulls ({null_pct:5.1f}%)")

    # Conservation check: house_kwh should be ≥ 0 except for sensor lag noise
    neg_house = df[df["house_kwh"] < -0.05]  # tolerate -50 W·h slack
    print(f"\n[QC] House consumption negative (sensor lag): {len(neg_house)} rows "
          f"({len(neg_house)/len(df)*100:.1f}%)")
    if len(neg_house):
        print(f"     Worst: {neg_house['house_kwh'].min():.3f} kWh  "
              f"Median |neg|: {neg_house['house_kwh'].abs().median():.3f} kWh")

    # ── Stats per natural unit ───────────────────────────────────────────────
    print("\n[STATS] Last 12 months totals (no extrapolation needed after ffill):")
    for col, label in [("solar_kwh",   "Solar production    "),
                       ("batt_chg_kwh","Battery charge      "),
                       ("batt_dis_kwh","Battery discharge   "),
                       ("export_kwh",  "Grid export         "),
                       ("import_kwh",  "Grid import         "),
                       ("house_kwh",   "House consumption   ")]:
        total = df[col].sum()
        print(f"  · {label}: {total:8.0f} kWh/year")

    # Battery efficiency check (charge → discharge ratio)
    chg = df["batt_chg_kwh"].sum()
    dis = df["batt_dis_kwh"].sum()
    if chg > 0:
        rt_eff = dis / chg
        print(f"\n[STATS] Battery round-trip efficiency: "
              f"{dis:.0f} dis / {chg:.0f} chg = {rt_eff*100:.1f}%")

    # Self-sufficiency: 1 - import / consumption
    cons = df["house_kwh"].sum()
    imp  = df["import_kwh"].sum()
    if cons > 0:
        ss = 1 - imp / cons
        print(f"[STATS] Self-sufficiency: {ss*100:.1f}% "
              f"({cons-imp:.0f}/{cons:.0f} kWh covered locally)")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = OUT_DIR / "etl_12m_1h.parquet"
    df.to_parquet(out, compression="snappy")
    print(f"\n[ETL] Wrote {out}  ({out.stat().st_size/1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
