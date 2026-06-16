#!/usr/bin/env python3
"""
15_statarb_rigor.py -- Paper 3 rigor: does the apparent hourly stat-arb edge survive?

  1. tercile cost sweep -- show the illiquid-coin "edge" dies under realistic (coin-
     appropriate) costs, i.e. it is a stale-price artifact, not tradable alpha
  2. DSR + PBO over the hourly config grid (window x k x entry)
  3. cointegration pairs (daily): IS-selected pair count (multiple testing) + OOS net Sharpe
  4. purged walk-forward OOS on the hourly default

Results -> experiments/paper3_statarb_rigor.json
"""
import os, sys, json, itertools, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import statarb as SA  # noqa: E402
from alpha_research.eval import rigor              # noqa: E402

PPY_H = 8760


def main():
    out = {}
    Rh, dv = SA.load_hourly_panel()
    medv = dv.median().dropna().sort_values(); third = max(1, len(medv) // 3)
    terc = {"low_liq": list(medv.index[:third]), "mid_liq": list(medv.index[third:2 * third]),
            "high_liq": list(medv.index[2 * third:])}

    # 1) tercile cost sweep -- the stale-price-artifact test
    sweep = {}
    for name, coins in terc.items():
        coins = [c for c in coins if c in Rh.columns]
        sweep[name] = {f"{c}bp": round(SA.ann_sharpe(
            SA.pca_sscore_pnl(Rh[coins], 336, 2, 1.25, c, 24).values, PPY_H), 2) for c in [5, 20, 50, 80]}
    out["tercile_cost_sweep_net_sharpe"] = sweep

    # 2) DSR + PBO over the hourly config grid
    grid = list(itertools.product([168, 336], [2, 3], [1.0, 1.25, 1.5]))   # window, k, entry
    cols, sr_pp, labels = [], [], []
    for w, k, e in grid:
        s = SA.pca_sscore_pnl(Rh, w, k, e, 5.0, 24)
        cols.append(s.rename(str((w, k, e)))); labels.append((w, k, e))
        v = s.values[s.values != 0]
        sr_pp.append(float(v.mean() / v.std()) if len(v) > 2 and v.std() > 0 else np.nan)
    mat = pd.concat(cols, axis=1).fillna(0.0)
    M = mat.to_numpy(); pp = [M[:, i][M[:, i] != 0].mean() / M[:, i][M[:, i] != 0].std()
                              if (M[:, i] != 0).sum() > 2 else np.nan for i in range(M.shape[1])]
    best = int(np.nanargmax(pp))
    pbo = rigor.probability_of_backtest_overfitting(M, n_splits=10)
    dsr = rigor.deflated_sharpe_ratio(M[:, best][M[:, best] != 0], sr_pp, n_trials=len(grid))
    out["hourly_selection"] = dict(n_configs=len(grid), best_config=str(labels[best]),
        best_ann_sharpe=round(pp[best] * np.sqrt(PPY_H), 2), deflated_sharpe=round(dsr, 3),
        pbo=round(pbo["pbo"], 3), config_sharpe_range=[round(np.nanmin(pp) * np.sqrt(PPY_H), 2),
                                                       round(np.nanmax(pp) * np.sqrt(PPY_H), 2)])

    # 4) purged walk-forward OOS on the hourly default
    dfl = SA.pca_sscore_pnl(Rh, 336, 3, 1.25, 5.0, 24)
    nz = dfl[dfl != 0]
    splits = rigor.purged_walkforward_splits(len(nz), n_splits=4, embargo=0.0, purge=24)
    oos = np.concatenate([nz.values[s.test] for s in splits]) if splits else np.array([])
    out["hourly_walkforward_oos_sharpe"] = round(SA.ann_sharpe(oos, PPY_H), 2)

    # 3) cointegration pairs (daily) -- multiple-testing collapse
    Pd = pd.DataFrame({os.path.basename(f).replace("_spot1d.parquet", ""):
                       pd.read_parquet(f).assign(time=lambda d: pd.to_datetime(d["time"], utc=True)).set_index("time")["close"]
                       for f in __import__("glob").glob(f"{SA.FUND}/*_spot1d.parquet")}).sort_index().dropna(how="all")
    cp, is_pairs = SA.coint_pairs_pnl(Pd, form=90, entry_z=2.0, exit_z=0.5, adf_p=0.05, cost_bps=5.0, max_pairs=15)
    out["coint_pairs_daily"] = dict(is_selected_pairs_total=int(is_pairs),
        oos_net_sharpe=round(SA.ann_sharpe(cp.values, 365), 2),
        oos_net_sharpe_0cost=round(SA.ann_sharpe(SA.coint_pairs_pnl(Pd, 90, 2.0, 0.5, 0.05, 0.0, 15)[0].values, 365), 2))

    json.dump(out, open(f"{ROOT}/experiments/paper3_statarb_rigor.json", "w"), indent=2, default=str)

    print("TERCILE COST SWEEP (net Sharpe) -- the stale-price-artifact test:")
    for name, sw in sweep.items():
        print(f"  {name:9s}: " + "  ".join(f"{c}={v}" for c, v in sw.items()))
    s = out["hourly_selection"]
    print(f"\nHOURLY SELECTION ({s['n_configs']} configs): best {s['best_config']} ann Sharpe {s['best_ann_sharpe']}, "
          f"range {s['config_sharpe_range']}")
    print(f"  DEFLATED Sharpe {s['deflated_sharpe']}, PBO {s['pbo']}")
    print(f"  purged walk-forward OOS Sharpe: {out['hourly_walkforward_oos_sharpe']}")
    cp_ = out["coint_pairs_daily"]
    print(f"\nCOINTEGRATION PAIRS (daily): {cp_['is_selected_pairs_total']} IS-selected pairs (multiple testing); "
          f"OOS net Sharpe {cp_['oos_net_sharpe']} (gross {cp_['oos_net_sharpe_0cost']})")
    print("\nWrote experiments/paper3_statarb_rigor.json")


if __name__ == "__main__":
    main()
