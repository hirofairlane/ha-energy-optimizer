"""ROI pipeline · stage 1.3 — Validate simulator + TOU arbitrage variant.

Two checks:
1. Energy-balance plausibility — feed the historical battery_w into a
   minimal SOC tracker (no policy) and compare against the real SOC sensor.
   MAE in % SOC tells us how well our model of efficiency / capacity fits.

2. TOU arbitrage policy — extend simulate() to pre-charge from grid during
   P3 valley to discharge during P1 peak when forecast solar is insufficient.
   Re-run sweep across capacities and report extra savings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"

# Reuse simulator config
import importlib.util, sys
spec = importlib.util.spec_from_file_location("sim2", Path(__file__).parent / "02_simulator.py")
sim2 = importlib.util.module_from_spec(spec)
sys.modules["sim2"] = sim2  # py3.14 dataclasses needs the module registered
spec.loader.exec_module(sim2)
BatteryConfig, simulate, summarize, tariff_period, BUY, SELL = (
    sim2.BatteryConfig, sim2.simulate, sim2.summarize, sim2.tariff_period,
    sim2.BUY, sim2.SELL)


# ── 1. Validate against real SOC ─────────────────────────────────────────────

def validate_soc(df: pd.DataFrame, capacity_kwh: float = 10.0,
                 eff_chg: float = 0.96, eff_dis: float = 0.96) -> dict:
    """Track SOC by integrating real battery_w over time and compare to soc_pct.
    Calibrates initial SOC to the first valid reading."""
    soc_kwh   = (df["soc_pct"].iloc[0] / 100.0) * capacity_kwh
    sim_soc   = []
    for batt_kw in df["battery_w"].values / 1000.0:  # +ve = charging
        if batt_kw > 0:
            soc_kwh += batt_kw * eff_chg  # 1 hour = 1 kWh per kW
        else:
            soc_kwh += batt_kw / eff_dis
        soc_kwh = min(max(soc_kwh, 0.0), capacity_kwh)
        sim_soc.append(soc_kwh / capacity_kwh * 100)
    sim_soc = pd.Series(sim_soc, index=df.index)
    real    = df["soc_pct"]
    err     = (sim_soc - real).dropna()
    return {
        "n_hours":     len(err),
        "mae_pct":     round(err.abs().mean(), 2),
        "rmse_pct":    round((err ** 2).mean() ** 0.5, 2),
        "bias_pct":    round(err.mean(), 2),
        "p95_pct":     round(err.abs().quantile(0.95), 2),
        "real_mean":   round(real.mean(), 1),
        "real_std":    round(real.std(), 1),
    }


# ── 2. TOU-aware policy: peak shaving + valley pre-charge ────────────────────

def simulate_tou(df: pd.DataFrame, cfg: BatteryConfig,
                 peak_target_soc: float = 0.85) -> pd.DataFrame:
    """Same as simulate() but with a valley→peak arbitrage layer:
    during P3 hours, top up battery from grid up to peak_target_soc so we
    have margin to cover P1 hours from battery instead of grid."""
    out = df.copy()
    soc_min_kwh = cfg.soc_min * cfg.capacity_kwh
    soc_max_kwh = cfg.soc_max * cfg.capacity_kwh
    target_kwh  = peak_target_soc * cfg.capacity_kwh
    soc = cfg.initial_soc * cfg.capacity_kwh

    idx_local = out.index.tz_convert("Europe/Madrid")
    periods = [tariff_period(t) for t in idx_local]

    sim_chg, sim_dis, sim_imp, sim_exp, sim_soc, sim_grid_chg = [], [], [], [], [], []
    for solar, house, period in zip(out["solar_kwh"].values,
                                     out["house_kwh"].clip(lower=0).values,
                                     periods):
        net = solar - house
        chg = dis = imp = exp = grid_chg = 0.0
        if net > 0:
            room    = soc_max_kwh - soc
            chg_kwh = min(net, cfg.p_max_kw, room / cfg.eff_chg)
            soc    += chg_kwh * cfg.eff_chg
            chg     = chg_kwh
            exp     = net - chg_kwh
        else:
            need    = -net
            avail   = (soc - soc_min_kwh) * cfg.eff_dis
            dis_kwh = min(need, cfg.p_max_kw, max(avail, 0.0))
            soc    -= dis_kwh / cfg.eff_dis
            dis     = dis_kwh
            imp     = need - dis_kwh
        # Valley grid charge — only if room remains and we're below target
        if period == "P3" and soc < target_kwh:
            room    = target_kwh - soc
            chg_kwh = min(cfg.p_max_kw, room / cfg.eff_chg)
            soc    += chg_kwh * cfg.eff_chg
            grid_chg = chg_kwh
            imp    += chg_kwh
        sim_chg.append(chg); sim_dis.append(dis); sim_imp.append(imp); sim_exp.append(exp)
        sim_soc.append(soc / cfg.capacity_kwh * 100); sim_grid_chg.append(grid_chg)

    out["sim_chg_kwh"]      = sim_chg
    out["sim_dis_kwh"]      = sim_dis
    out["sim_imp_kwh"]      = sim_imp
    out["sim_exp_kwh"]      = sim_exp
    out["sim_soc_pct"]      = sim_soc
    out["sim_grid_chg_kwh"] = sim_grid_chg
    out["period"]           = periods
    out["buy_eur_kwh"]      = out["period"].map(BUY)
    out["sell_eur_kwh"]     = out["period"].map(SELL)
    out["sim_eur_imp"]      = out["sim_imp_kwh"] * out["buy_eur_kwh"]
    out["sim_eur_exp"]      = out["sim_exp_kwh"] * out["sell_eur_kwh"]
    out["sim_eur_net"]      = out["sim_eur_imp"] - out["sim_eur_exp"]

    # No-battery counterfactual (same as in simulate())
    no_batt_imp = out["house_kwh"].clip(lower=0) - out["solar_kwh"].clip(upper=out["house_kwh"].clip(lower=0))
    no_batt_exp = (out["solar_kwh"] - out["house_kwh"].clip(lower=0)).clip(lower=0)
    out["nb_imp_kwh"]   = no_batt_imp
    out["nb_exp_kwh"]   = no_batt_exp
    out["nb_eur_net"]   = no_batt_imp * out["buy_eur_kwh"] - no_batt_exp * out["sell_eur_kwh"]
    out["sim_eur_saved"] = out["nb_eur_net"] - out["sim_eur_net"]
    return out


def main() -> int:
    df = pd.read_parquet(DATA / "etl_12m_1h.parquet")

    # ── Validation ────────────────────────────────────────────────────────────
    print("[VAL] Reconstructing SOC from real battery_w flow (no policy)…")
    val = validate_soc(df)
    print(f"  Real SOC:  mean={val['real_mean']}%  std={val['real_std']}%")
    print(f"  Sim error: MAE={val['mae_pct']}%  RMSE={val['rmse_pct']}%  "
          f"P95={val['p95_pct']}%  bias={val['bias_pct']:+.2f}%")
    target = 8.0
    verdict = "✓ GOOD" if val['mae_pct'] <= target else "✗ TOO HIGH"
    print(f"  vs target MAE ≤ {target}% → {verdict}")

    # ── Self-consume vs TOU arbitrage sweep ──────────────────────────────────
    print(f"\n[SWEEP]  Self-consume only vs TOU arbitrage (valley pre-charge)")
    print(f"{'cap':>5}  {'SC saved':>10}  {'TOU saved':>10}  {'Δ':>8}  "
          f"{'TOU €/d':>8}  {'TOU SS%':>8}")
    print("-" * 62)
    for cap in [5, 8, 10, 12, 14, 16, 20]:
        cfg   = BatteryConfig(capacity_kwh=cap)
        out_sc  = simulate(df, cfg)
        out_tou = simulate_tou(df, cfg)
        sc      = summarize(out_sc, cfg)
        tou     = summarize(out_tou, cfg)
        delta   = tou["sim_eur_saved"] - sc["sim_eur_saved"]
        print(f"{cap:>5d}  {sc['sim_eur_saved']:>10.1f}  "
              f"{tou['sim_eur_saved']:>10.1f}  {delta:>+8.1f}  "
              f"{tou['saved_per_day']:>8.2f}  {tou['self_suff_sim']:>7.1f}%")

    # ── Marginal value of incremental capacity (using TOU policy) ────────────
    print(f"\n[MARGINAL] TOU-policy savings per added kWh (vs current 10 kWh):")
    base = summarize(simulate_tou(df, BatteryConfig(capacity_kwh=10)),
                     BatteryConfig(capacity_kwh=10))
    print(f"  Baseline @ 10 kWh:  saved={base['sim_eur_saved']:.1f}€/yr")
    for cap in [12, 14, 16, 20]:
        cur  = summarize(simulate_tou(df, BatteryConfig(capacity_kwh=cap)),
                         BatteryConfig(capacity_kwh=cap))
        marg_total = cur['sim_eur_saved'] - base['sim_eur_saved']
        added_kwh  = cap - 10
        marg_per_kwh = marg_total / added_kwh if added_kwh else 0
        # Cost assumption: ~430 €/kWh installed for Huawei Luna2000 expansion
        cost_per_kwh = 430
        payback = (added_kwh * cost_per_kwh) / marg_total if marg_total > 0 else float('inf')
        print(f"  +{added_kwh:>2d} kWh → +{marg_total:>5.1f}€/yr "
              f"({marg_per_kwh:>5.1f}€/kWh·yr) — payback {payback:>5.1f}y "
              f"@ 430€/kWh installed")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
