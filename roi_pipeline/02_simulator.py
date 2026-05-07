"""ROI pipeline · stage 1.2 — Deterministic battery simulator + tariff.

Given hourly solar production and house consumption (historical or forecast),
simulate what a battery of given capacity would do under a "self-consume
first, valley-charge second" policy. Outputs hourly flows + cumulative
financials under the Spanish 2.0TD Nufri tariff structure.

Tariff (€/kWh, from project_sensor_truth memory):
  Buy:  P1=0.222745 (peak)  P2=0.150591 (mid)  P3=0.114377 (valley)
  Sell: S1=0.010916          S2=0.010643          S3=0.000169
Calendar (weekday Madrid local):
  P1: 10-14, 18-22
  P2: 08-10, 14-18, 22-24
  P3: 00-08 + Sat/Sun + national holidays
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"

BUY  = {"P1": 0.222745, "P2": 0.150591, "P3": 0.114377}
SELL = {"P1": 0.010916, "P2": 0.010643, "P3": 0.000169}

# Spanish national holidays that always fall on the same calendar day
FIXED_HOLIDAYS = {(1,1),(1,6),(5,1),(8,15),(10,12),(11,1),(12,6),(12,8),(12,25)}


def tariff_period(ts_local: pd.Timestamp) -> str:
    """Map a Madrid-local timestamp to the 2.0TD period (P1/P2/P3)."""
    if ts_local.weekday() >= 5 or (ts_local.month, ts_local.day) in FIXED_HOLIDAYS:
        return "P3"
    h = ts_local.hour
    if (10 <= h < 14) or (18 <= h < 22):
        return "P1"
    if (8 <= h < 10) or (14 <= h < 18) or h >= 22:
        return "P2"
    return "P3"


@dataclass
class BatteryConfig:
    capacity_kwh:  float = 10.0
    p_max_kw:      float = 5.0     # max charge/discharge power
    eff_chg:       float = 0.96    # one-way efficiency (round-trip ≈ 0.92)
    eff_dis:       float = 0.96
    soc_min:       float = 0.10    # 10% reserve
    soc_max:       float = 0.95
    initial_soc:   float = 0.50


def simulate(df: pd.DataFrame, cfg: BatteryConfig) -> pd.DataFrame:
    """Simulate the battery hour-by-hour against the given hourly load profile.
    Expects df with columns: solar_kwh, house_kwh (positive scalars). Index is
    UTC. Returns a copy with sim_* columns added and per-hour € flows."""
    out = df.copy()
    soc = cfg.initial_soc * cfg.capacity_kwh
    soc_min_kwh = cfg.soc_min * cfg.capacity_kwh
    soc_max_kwh = cfg.soc_max * cfg.capacity_kwh

    sim_chg, sim_dis, sim_imp, sim_exp, sim_soc = [], [], [], [], []
    for solar, house in zip(out["solar_kwh"].values, out["house_kwh"].clip(lower=0).values):
        net = solar - house  # +ve = surplus, -ve = deficit
        chg = dis = imp = exp = 0.0
        if net > 0:
            # Surplus → charge battery first (within power and SOC limits),
            # any leftover exports to grid.
            room      = soc_max_kwh - soc
            chg_kwh   = min(net, cfg.p_max_kw, room / cfg.eff_chg)
            soc      += chg_kwh * cfg.eff_chg
            chg       = chg_kwh
            exp       = net - chg_kwh
        else:
            # Deficit → discharge battery first, rest comes from grid.
            need      = -net
            avail     = (soc - soc_min_kwh) * cfg.eff_dis
            dis_kwh   = min(need, cfg.p_max_kw, max(avail, 0.0))
            soc      -= dis_kwh / cfg.eff_dis
            dis       = dis_kwh
            imp       = need - dis_kwh
        sim_chg.append(chg)
        sim_dis.append(dis)
        sim_imp.append(imp)
        sim_exp.append(exp)
        sim_soc.append(soc / cfg.capacity_kwh * 100)

    out["sim_chg_kwh"] = sim_chg
    out["sim_dis_kwh"] = sim_dis
    out["sim_imp_kwh"] = sim_imp
    out["sim_exp_kwh"] = sim_exp
    out["sim_soc_pct"] = sim_soc

    # Tariff per row (use Madrid-local time)
    idx_local = out.index.tz_convert("Europe/Madrid")
    out["period"] = [tariff_period(t) for t in idx_local]
    out["buy_eur_kwh"]  = out["period"].map({k: BUY[k]  for k in BUY})
    out["sell_eur_kwh"] = out["period"].map({k: SELL[k] for k in SELL})

    out["sim_eur_imp"] = out["sim_imp_kwh"] * out["buy_eur_kwh"]
    out["sim_eur_exp"] = out["sim_exp_kwh"] * out["sell_eur_kwh"]
    out["sim_eur_net"] = out["sim_eur_imp"] - out["sim_eur_exp"]

    # No-battery counterfactual: surplus → export, deficit → import
    no_batt_imp = out["house_kwh"].clip(lower=0) - out["solar_kwh"].clip(upper=out["house_kwh"].clip(lower=0))
    no_batt_exp = (out["solar_kwh"] - out["house_kwh"].clip(lower=0)).clip(lower=0)
    out["nb_imp_kwh"] = no_batt_imp
    out["nb_exp_kwh"] = no_batt_exp
    out["nb_eur_imp"] = no_batt_imp * out["buy_eur_kwh"]
    out["nb_eur_exp"] = no_batt_exp * out["sell_eur_kwh"]
    out["nb_eur_net"] = out["nb_eur_imp"] - out["nb_eur_exp"]

    # Battery saving = (no-battery net cost) - (with-battery net cost)
    out["sim_eur_saved"] = out["nb_eur_net"] - out["sim_eur_net"]

    return out


def summarize(out: pd.DataFrame, cfg: BatteryConfig) -> dict:
    days = (out.index.max() - out.index.min()).total_seconds() / 86400
    return {
        "capacity_kwh":     cfg.capacity_kwh,
        "days":             round(days, 1),
        "solar_kwh":        round(out["solar_kwh"].sum(), 0),
        "house_kwh":        round(out["house_kwh"].clip(lower=0).sum(), 0),
        "sim_chg_kwh":      round(out["sim_chg_kwh"].sum(), 0),
        "sim_dis_kwh":      round(out["sim_dis_kwh"].sum(), 0),
        "sim_imp_kwh":      round(out["sim_imp_kwh"].sum(), 0),
        "sim_exp_kwh":      round(out["sim_exp_kwh"].sum(), 0),
        "nb_imp_kwh":       round(out["nb_imp_kwh"].sum(), 0),
        "nb_exp_kwh":       round(out["nb_exp_kwh"].sum(), 0),
        "sim_eur_net":      round(out["sim_eur_net"].sum(), 1),
        "nb_eur_net":       round(out["nb_eur_net"].sum(), 1),
        "sim_eur_saved":    round(out["sim_eur_saved"].sum(), 1),
        "saved_per_day":    round(out["sim_eur_saved"].sum() / max(days, 1), 2),
        "self_suff_sim":    round((1 - out["sim_imp_kwh"].sum() / max(out["house_kwh"].clip(lower=0).sum(), 1)) * 100, 1),
        "self_suff_nb":     round((1 - out["nb_imp_kwh"].sum() / max(out["house_kwh"].clip(lower=0).sum(), 1)) * 100, 1),
    }


def main() -> int:
    df = pd.read_parquet(DATA / "etl_12m_1h.parquet")
    print(f"[SIM] Loaded {len(df)} hourly rows  "
          f"[{df.index.min()} → {df.index.max()}]")

    # Sweep across battery capacities
    print(f"\n{'cap':>5}  {'imp_kwh':>9}  {'exp_kwh':>9}  "
          f"{'eur_net':>10}  {'eur_saved':>10}  {'€/day':>7}  {'self_suff':>9}")
    print("-" * 70)
    nb = simulate(df, BatteryConfig(capacity_kwh=0.001))  # ~no battery
    nb_sum = summarize(nb, BatteryConfig(capacity_kwh=0))
    print(f"{0:>5d}  {nb_sum['nb_imp_kwh']:>9.0f}  {nb_sum['nb_exp_kwh']:>9.0f}  "
          f"{nb_sum['nb_eur_net']:>10.1f}  {0:>10.1f}  {0:>7.2f}  "
          f"{nb_sum['self_suff_nb']:>8.1f}%")
    rows = [nb_sum]
    for cap in [5, 8, 10, 12, 14, 16, 20]:
        out = simulate(df, BatteryConfig(capacity_kwh=cap))
        s = summarize(out, BatteryConfig(capacity_kwh=cap))
        rows.append(s)
        print(f"{cap:>5d}  {s['sim_imp_kwh']:>9.0f}  {s['sim_exp_kwh']:>9.0f}  "
              f"{s['sim_eur_net']:>10.1f}  {s['sim_eur_saved']:>10.1f}  "
              f"{s['saved_per_day']:>7.2f}  {s['self_suff_sim']:>8.1f}%")

    # Persist sweep + 10 kWh full hourly trace for downstream analysis
    pd.DataFrame(rows).to_csv(DATA / "sim_sweep.csv", index=False)
    out10 = simulate(df, BatteryConfig(capacity_kwh=10))
    out10.to_parquet(DATA / "sim_10kwh.parquet", compression="snappy")
    print(f"\n[SIM] Wrote {DATA/'sim_sweep.csv'} and {DATA/'sim_10kwh.parquet'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
