"""HVAC efficiency curve — empirical COP vs outdoor temperature.

Sergio's hypothesis: heating cost is not just `price_kWh × hours`, because
the Genia Air's COP collapses as outdoor temperature drops. The naive
"always heat at valley tariff" rule may lose money on cold nights:

    cost_per_kWh_thermal = price_kWh / COP(T_outdoor)

This script pulls 365 days of (T_outdoor, electrical_consumed_kW,
thermal_yield_kW) from InfluxDB, filters to periods where the compressor
was actually running, and produces:

  1. A COP curve binned by outdoor temperature with confidence bands
  2. A second curve cost_per_kWh_thermal vs hour-of-day to identify the
     true sweet spot (accounts for tariff AND efficiency simultaneously)
  3. A CSV / JSON the add-on can later consume from the live system

Output: `data/cop_curve.csv`, `data/cop_curve.json`, `data/cop_analysis_summary.md`.

Run: `python 07_cop_analysis.py [--days N]`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

INFLUX_URL = "http://192.168.1.131:8086"
INFLUX_DB  = "homeassistant"

# Tariff bands (Madrid, weekday) used to compute effective €/kWh_thermal.
# Weekend simplification: P3 all day. Holidays not modelled here; pulled
# from the addon's logic in production.
BUY = {"P1": 0.222745, "P2": 0.150591, "P3": 0.114377}
P1_HOURS = set(range(10, 14)) | set(range(18, 22))
P2_HOURS = set(range(8, 10))  | set(range(14, 18)) | {22, 23}
# anything else = P3 (valley)

# Realistic Genia-Air-like operating envelope. Discards rows where the
# compressor isn't producing meaningful heat (idle, defrost, ACS, etc.).
MIN_ELEC_KW = 0.15   # below this we treat compressor as off
MIN_YIELD_KW = 0.10
MAX_COP_PLAUSIBLE = 7.0   # COP>7 is impossible in air-water HP, discard noise

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def fetch_series(entity_short: str, days: int, label: str) -> pd.Series:
    """InfluxQL `FROM /.*/ WHERE entity_id = ?` pattern. 15-min server-side
    mean to keep payload reasonable."""
    q = (f'SELECT mean("value") AS "value" FROM /.*/ '
         f"WHERE \"entity_id\" = '{entity_short}' "
         f"AND time > now()-{days}d "
         f'GROUP BY time(15m) fill(none)')
    r = requests.get(f"{INFLUX_URL}/query",
                     params={"db": INFLUX_DB, "q": q, "epoch": "ms"},
                     timeout=300)
    r.raise_for_status()
    out = r.json().get("results", [{}])[0]
    rows = []
    for series in out.get("series", []):
        for ts_ms, val in series.get("values", []):
            if val is None:
                continue
            rows.append((ts_ms, val))
    if not rows:
        return pd.Series(dtype=float, name=label)
    df = pd.DataFrame(rows, columns=["ts_ms", "value"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    s = df.set_index("ts")["value"].rename(label)
    return s.groupby(level=0).mean()   # collapse any duplicate slots


def tariff_eur_per_kwh(idx_local: pd.DatetimeIndex) -> pd.Series:
    """€/kWh paid for grid-imported energy at each hour. Weekends treated
    as P3 entirely (good first approximation; national holidays ignored)."""
    eur = []
    for ts in idx_local:
        if ts.weekday() >= 5:
            eur.append(BUY["P3"])
        elif ts.hour in P1_HOURS:
            eur.append(BUY["P1"])
        elif ts.hour in P2_HOURS:
            eur.append(BUY["P2"])
        else:
            eur.append(BUY["P3"])
    return pd.Series(eur, index=idx_local, name="eur_per_kwh")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365,
                    help="History window (default 365 = full year)")
    args = ap.parse_args()

    print(f"[COP] Pulling {args.days}d of aerotermia sensors from InfluxDB")
    series = {
        "T_out":      fetch_series("ebusd_broadcast_outsidetemp_temp2",            args.days, "T_out"),
        "elec_kW":    fetch_series("aerotermia_exterior_currentconsumedpower",     args.days, "elec_kW"),
        "yield_kW":   fetch_series("aerotermia_exterior_currentyieldpower",        args.days, "yield_kW"),
        "compr_util": fetch_series("ebusd_hmu_currentcompressorutil",              args.days, "compr_util"),
    }
    for k, s in series.items():
        first = s.index.min() if len(s) else None
        last  = s.index.max() if len(s) else None
        print(f"  · {k:12s}: {len(s):6d} rows  [{first} → {last}]")

    df = pd.concat(series.values(), axis=1).sort_index()
    df.index.name = "ts_utc"
    df = df.ffill(limit=1)
    df = df.dropna(subset=["T_out", "elec_kW", "yield_kW"])
    print(f"\n[COP] Joined frame: {len(df)} rows after dropna")

    # ── Active-heating mask ──────────────────────────────────────────────────
    active = (df["elec_kW"] >= MIN_ELEC_KW) & (df["yield_kW"] >= MIN_YIELD_KW)
    df_active = df[active].copy()
    print(f"[COP] {len(df_active)} rows where compressor is actually heating "
          f"({100 * len(df_active) / max(len(df), 1):.1f}% of all samples)")

    if len(df_active) < 100:
        print("[COP] WARNING: not enough active-heating samples to build a "
              "credible curve. Re-run after a winter month with real usage.")
        # Still continue so the output files exist (with whatever we have).

    df_active["cop"] = df_active["yield_kW"] / df_active["elec_kW"]
    df_active = df_active[df_active["cop"] <= MAX_COP_PLAUSIBLE]
    df_active = df_active[df_active["cop"] >= 0.5]   # drop garbage on the low end too

    print(f"[COP] {len(df_active)} samples after COP-plausibility filter "
          f"({MIN_ELEC_KW:.2f}–{MAX_COP_PLAUSIBLE:.1f} window)")

    # ── COP vs outdoor temperature, binned ───────────────────────────────────
    edges = list(range(-10, 36, 2))   # 2 °C buckets from -10 to 34
    df_active["T_bin"] = pd.cut(df_active["T_out"], bins=edges,
                                labels=[(edges[i] + edges[i+1]) / 2
                                        for i in range(len(edges) - 1)])
    by_temp = df_active.groupby("T_bin", observed=True)["cop"].agg(
        ["count", "mean", "median", "std",
         lambda s: s.quantile(0.1), lambda s: s.quantile(0.9)])
    by_temp.columns = ["n", "cop_mean", "cop_median", "cop_std", "cop_p10", "cop_p90"]
    by_temp = by_temp[by_temp["n"] >= 5].reset_index()
    by_temp = by_temp.rename(columns={"T_bin": "T_out_c"})

    print(f"\n[COP] Curve by 2°C buckets (n ≥ 5 samples):")
    print(by_temp.to_string(index=False))

    # ── Effective cost €/kWh_thermal per hour-of-day ─────────────────────────
    idx_local = df_active.index.tz_convert("Europe/Madrid")
    df_active["hour"]        = idx_local.hour
    df_active["eur_per_kwh"] = tariff_eur_per_kwh(idx_local).values
    df_active["eur_per_kwh_thermal"] = (df_active["eur_per_kwh"]
                                         / df_active["cop"].replace(0, np.nan))

    by_hour = df_active.groupby("hour")["eur_per_kwh_thermal"].agg(
        ["count", "mean", "median"]).reset_index()
    by_hour.columns = ["hour", "n", "eur_per_kwh_thermal_mean", "eur_per_kwh_thermal_median"]

    print(f"\n[COST] Effective €/kWh_thermal by hour-of-day:")
    print(by_hour.to_string(index=False))

    # ── Persist ──────────────────────────────────────────────────────────────
    by_temp.to_csv(OUT_DIR / "cop_curve.csv", index=False)
    by_hour.to_csv(OUT_DIR / "cost_per_kwh_thermal_by_hour.csv", index=False)

    summary = {
        "days_window":         args.days,
        "ts_first":             str(df.index.min()),
        "ts_last":              str(df.index.max()),
        "rows_total":           int(len(df)),
        "rows_active_heating":  int(len(df_active)),
        "cop_curve":            by_temp.to_dict(orient="records"),
        "cost_per_kwh_thermal_by_hour": by_hour.to_dict(orient="records"),
    }
    # Convert pandas-native types to plain Python so json.dump doesn't choke
    summary = json.loads(json.dumps(summary, default=str))
    (OUT_DIR / "cop_curve.json").write_text(json.dumps(summary, indent=2))

    # ── Human-readable summary ───────────────────────────────────────────────
    md_lines = [
        f"# HVAC COP analysis — {args.days}-day window",
        "",
        f"Window: {df.index.min()} → {df.index.max()}",
        f"Total samples: {len(df)} (15-min mean)",
        f"Active-heating samples: {len(df_active)} "
        f"({100 * len(df_active) / max(len(df), 1):.1f}% of total)",
        "",
        "## COP curve (binned by outdoor temperature)",
        "",
        "| T_out (°C) | n samples | COP median | COP P10 | COP P90 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_temp.iterrows():
        md_lines.append(
            f"| {row['T_out_c']:.0f} | {int(row['n'])} | "
            f"{row['cop_median']:.2f} | {row['cop_p10']:.2f} | {row['cop_p90']:.2f} |")
    md_lines += ["", "## Effective €/kWh_thermal by hour-of-day", "",
                 "| Hour | n samples | €/kWh_thermal (median) |",
                 "|---:|---:|---:|"]
    for _, row in by_hour.iterrows():
        md_lines.append(
            f"| {int(row['hour'])} | {int(row['n'])} | "
            f"{row['eur_per_kwh_thermal_median']:.3f} |")
    (OUT_DIR / "cop_analysis_summary.md").write_text("\n".join(md_lines))
    print(f"\n[OUT] Wrote {OUT_DIR / 'cop_curve.csv'}, "
          f"{OUT_DIR / 'cop_curve.json'}, "
          f"{OUT_DIR / 'cop_analysis_summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
