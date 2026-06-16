#!/usr/bin/env python3
"""
02_vrp_rigor.py -- robustness + full rigor gauntlet for the crypto VRP harvest.

Stress-tests the script-01 finding (premium real & strong early, decays to ~0 by
2024-2026) with:
  1. RV-estimator robustness  (close-to-close / Parkinson / Garman-Klass)
  2. phase robustness         (Sharpe across all 60 entry phases)
  3. decay: date split + purged walk-forward OOS
  4. selection rigor: deflated Sharpe + PBO (CSCV) over the config grid

Results -> experiments/paper1_vrp_rigor.json
"""
import os, sys, json, itertools
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import vrp          # noqa: E402
from alpha_research.eval import rigor           # noqa: E402

COINS = ["BTC", "ETH"]
METHODS = ["c2c", "parkinson", "gk"]
COSTS = [0.0, 1.0, 2.0, 3.0, 5.0]


def per_period_sharpe(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std()) if x.size > 2 and x.std() > 0 else float("nan")


def run(coin):
    res = {"coin": coin}
    panels = {m: vrp.build(coin, m) for m in METHODS}

    # 1) RV-estimator robustness (gate+voltgt, cost 2vp)
    rv_rob = {}
    for m in METHODS:
        p = panels[m]; ov = p.dropna(subset=["vrp_vol", "vrp_var"])
        ent = vrp.harvest_pnl(p, phase=0, gate=True, voltgt=True, cost_vp=2.0)
        rv_rob[m] = dict(
            mean_vrp_volpts=float(ov["vrp_vol"].mean() * 100),
            pct_pos=float((ov["vrp_vol"] > 0).mean()),
            t_var=rigor.nonoverlapping_tstat(ov["vrp_var"].to_numpy(), thin=vrp.H)["t"],
            harvest_sharpe=vrp.ann_sharpe(ent["pnl"]))
    res["rv_estimator_robustness"] = rv_rob

    # 2) phase robustness (c2c, gate+voltgt, cost 2vp)
    p = panels["c2c"]
    ph = np.array([vrp.ann_sharpe(vrp.harvest_pnl(p, phase=k, gate=True, voltgt=True, cost_vp=2.0)["pnl"])
                   for k in range(vrp.H)])
    ph = ph[np.isfinite(ph)]
    res["phase_robustness"] = dict(mean=float(ph.mean()), std=float(ph.std()),
                                   min=float(ph.min()), max=float(ph.max()), n=int(ph.size))

    # 3) decay: date split + purged walk-forward OOS
    ent = vrp.harvest_pnl(p, phase=0, gate=True, voltgt=True, cost_vp=2.0)
    pnl = ent["pnl"].to_numpy()
    half = (ent["time"] < pd.Timestamp("2024-01-01", tz="UTC")).to_numpy()
    splits = rigor.purged_walkforward_splits(len(ent), n_splits=3, embargo=0.0, purge=1)
    oos = np.concatenate([pnl[s.test] for s in splits]) if splits else np.array([])
    res["decay"] = dict(
        sharpe_2021_2023=vrp.ann_sharpe(pnl[half]),
        sharpe_2024_2026=vrp.ann_sharpe(pnl[~half]),
        walkforward_oos_sharpe=vrp.ann_sharpe(oos), n_oos=int(oos.size),
        n_2021_2023=int(half.sum()), n_2024_2026=int((~half).sum()))

    # 4) selection rigor: DSR + PBO over the config grid (phase 0; vary method,gate,voltgt,cost)
    grid = list(itertools.product(METHODS, [True, False], [True, False], COSTS))
    cols, trial_sh, labels = [], [], []
    for (m, g, vt, c) in grid:
        s = vrp.harvest_pnl(panels[m], phase=0, gate=g, voltgt=vt, cost_vp=c).set_index("time")["pnl"]
        cols.append(s.rename(str((m, g, vt, c))))
        trial_sh.append(per_period_sharpe(s.to_numpy()))
        labels.append((m, g, vt, c))
    mat = pd.concat(cols, axis=1).dropna()
    M = mat.to_numpy()
    pp = [per_period_sharpe(M[:, i]) for i in range(M.shape[1])]
    best = int(np.nanargmax(pp))
    pbo = rigor.probability_of_backtest_overfitting(M, n_splits=8)
    dsr = rigor.deflated_sharpe_ratio(M[:, best], trial_sh, n_trials=len(grid))
    res["selection"] = dict(
        n_configs=len(grid), best_config=str(labels[best]),
        best_ann_sharpe=float(pp[best] * np.sqrt(vrp.WIN_PER_YEAR)),
        deflated_sharpe=dsr, pbo=pbo["pbo"], pbo_n_comb=pbo["n_combinations"],
        n_windows=int(M.shape[0]))
    return res


def main():
    out = {}
    for c in COINS:
        out[c] = run(c)
    json.dump(out, open(f"{ROOT}/experiments/paper1_vrp_rigor.json", "w"), indent=2, default=str)
    for c in COINS:
        r = out[c]
        print(f"\n===== {c} RIGOR =====")
        print(" RV-estimator robustness (meanVRP vp / %pos / t(var) / harvest Sharpe):")
        for m, d in r["rv_estimator_robustness"].items():
            print(f"   {m:9s}: {d['mean_vrp_volpts']:+.2f}  {d['pct_pos']*100:.0f}%  "
                  f"t={d['t_var']:.2f}  Sh={d['harvest_sharpe']:.2f}")
        pr = r["phase_robustness"]
        print(f" Phase robustness Sharpe: mean={pr['mean']:.2f} std={pr['std']:.2f} "
              f"[{pr['min']:.2f},{pr['max']:.2f}] n={pr['n']}")
        dc = r["decay"]
        print(f" DECAY: 2021-23={dc['sharpe_2021_2023']:.2f} (n={dc['n_2021_2023']})  "
              f"2024-26={dc['sharpe_2024_2026']:.2f} (n={dc['n_2024_2026']})  "
              f"walk-fwd OOS={dc['walkforward_oos_sharpe']:.2f} (n={dc['n_oos']})")
        se = r["selection"]
        print(f" SELECTION: {se['n_configs']} configs, {se['n_windows']} windows; best={se['best_config']}")
        print(f"   ann Sharpe={se['best_ann_sharpe']:.2f}  DEFLATED Sharpe={se['deflated_sharpe']:.2f}  "
              f"PBO={se['pbo']:.2f} ({se['pbo_n_comb']} combos)")
    print("\nWrote experiments/paper1_vrp_rigor.json")


if __name__ == "__main__":
    main()
