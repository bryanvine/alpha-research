#!/usr/bin/env python3
"""
13_carry_rigor.py -- rigor gauntlet for the cross-sectional crypto funding-carry factor.

Deflated Sharpe + PBO (CSCV) over the config grid (lookback x quantile x rebalance),
purged walk-forward OOS, and a drop-one-coin robustness sweep -- to decide whether the
~0.71 net Sharpe is a real edge or a selection/overfitting artifact.

Results -> experiments/paper2_carry_rigor.json
"""
import os, sys, json, itertools
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import carry as C   # noqa: E402
from alpha_research.eval import rigor            # noqa: E402

COST = 6.0  # realistic one-way bps (taker + slippage)


def main():
    F = C.load_daily_funding(); R = C.load_spot_returns()
    LB, Q, RB = [3, 7, 14, 21], [0.2, 0.33, 0.5], [1, 5, 10]
    grid = list(itertools.product(LB, Q, RB))

    cols, labels, sr_pp = [], [], []
    for lb, q, rb in grid:
        s = C.cross_sectional_factor(F, R, lookback=lb, q=q, cost_oneway_bps=COST, rebal=rb).dropna()
        cols.append(s.rename(str((lb, q, rb)))); labels.append((lb, q, rb))
        sr_pp.append(float(s.mean() / s.std()) if s.std() > 0 else np.nan)
    mat = pd.concat(cols, axis=1).dropna(); M = mat.to_numpy()
    pp = [M[:, i].mean() / M[:, i].std() if M[:, i].std() > 0 else np.nan for i in range(M.shape[1])]
    best = int(np.nanargmax(pp))

    pbo = rigor.probability_of_backtest_overfitting(M, n_splits=10)
    dsr_best = rigor.deflated_sharpe_ratio(M[:, best], sr_pp, n_trials=len(grid))

    default = C.cross_sectional_factor(F, R, 7, 0.33, COST, 5).dropna()
    dsr_def = rigor.deflated_sharpe_ratio(default.values, sr_pp, n_trials=len(grid))

    splits = rigor.purged_walkforward_splits(len(default), n_splits=4, embargo=0.0, purge=7)
    oos = np.concatenate([default.values[s.test] for s in splits]) if splits else np.array([])

    # drop-one-coin robustness on the default config
    drop = {}
    for s in F.columns:
        Fd, Rd = F.drop(columns=[s]), R.drop(columns=[s], errors="ignore")
        drop[s] = round(C.ann_sharpe(C.cross_sectional_factor(Fd, Rd, 7, 0.33, COST, 5).dropna().values), 2)
    dvals = sorted(drop.items(), key=lambda kv: kv[1])

    out = dict(
        n_configs=len(grid), n_days=int(M.shape[0]),
        default_ann_sharpe=round(C.ann_sharpe(default.values), 2),
        best_config=str(labels[best]), best_ann_sharpe=round(pp[best] * np.sqrt(365), 2),
        deflated_sharpe_default=round(dsr_def, 3), deflated_sharpe_best=round(dsr_best, 3),
        pbo=round(pbo["pbo"], 3), pbo_combinations=pbo["n_combinations"],
        walkforward_oos_sharpe=round(C.ann_sharpe(oos), 2), n_oos=int(oos.size),
        dropone_min=dvals[0], dropone_max=dvals[-1],
        config_sharpe_range=[round(np.nanmin(pp) * np.sqrt(365), 2), round(np.nanmax(pp) * np.sqrt(365), 2)],
        frac_configs_positive=round(float(np.mean([x > 0 for x in pp if np.isfinite(x)])), 2),
    )
    json.dump(out, open(f"{ROOT}/experiments/paper2_carry_rigor.json", "w"), indent=2, default=str)

    print(f"Cross-sectional carry factor — RIGOR ({out['n_configs']} configs, {out['n_days']} days, cost {COST}bp):")
    print(f"  default (lb7,q.33,rebal5) ann Sharpe = {out['default_ann_sharpe']}")
    print(f"  best config {out['best_config']} ann Sharpe = {out['best_ann_sharpe']}")
    print(f"  config Sharpe range {out['config_sharpe_range']}, frac positive {out['frac_configs_positive']}")
    print(f"  DEFLATED Sharpe: default={out['deflated_sharpe_default']}, best={out['deflated_sharpe_best']}  (want >0.5)")
    print(f"  PBO = {out['pbo']}  ({out['pbo_combinations']} combos; want <0.5)")
    print(f"  purged walk-forward OOS Sharpe = {out['walkforward_oos_sharpe']} (n={out['n_oos']})")
    print(f"  drop-one-coin Sharpe range: {out['dropone_min']} .. {out['dropone_max']}")
    print("\nWrote experiments/paper2_carry_rigor.json")


if __name__ == "__main__":
    main()
