#!/usr/bin/env python3
"""
12_carry_core.py -- Paper 2 core: crypto funding-rate carry, net of costs, with decay.

  * degenerate-signal check on every coin's funding (must exclude constant predictors)
  * cross-sectional funding-carry factor: full-sample net/gross Sharpe, by-year decay,
    tail (maxDD/worst-day/skew), the Oct-2025 cascade window, and a cost sweep
  * time-series cash-and-carry (delta-neutral) net return by year
  * H5: does trailing funding predict forward 7d returns (crowded-long mean reversion)?

Results -> experiments/paper2_carry_core.json
"""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import carry as C        # noqa: E402
from alpha_research.eval import rigor                 # noqa: E402
from alpha_research.eval.metrics import spearman_ic   # noqa: E402


def by_year(pnl):
    return {int(y): round(C.ann_sharpe(g.values), 3) for y, g in pnl.groupby(pnl.index.year)}


def tail(pnl):
    x = pnl.values[np.isfinite(pnl.values)]
    cum = np.cumsum(x); dd = cum - np.maximum.accumulate(cum)
    return dict(sharpe=C.ann_sharpe(x), ann_ret_pct=float(np.mean(x) * 365 * 100),
                ann_vol_pct=float(np.std(x) * np.sqrt(365) * 100), maxdd_pct=float(dd.min() * 100),
                worst_day_pct=float(x.min() * 100), skew=float(pd.Series(x).skew()), n=int(x.size))


def main():
    F = C.load_daily_funding(); R = C.load_spot_returns()
    out = {"universe": int(F.shape[1]), "span": [str(F.index.min())[:10], str(F.index.max())[:10]]}

    # 1) degenerate-signal check on each coin's funding
    deg = {}
    for s in F.columns:
        rep = rigor.degenerate_signal_check(F[s].dropna().values)
        if rep.is_degenerate:
            deg[s] = rep.reasons
    out["degenerate_funding"] = deg

    # 2) cross-sectional carry factor
    f = C.cross_sectional_factor(F, R, lookback=7, q=0.33, cost_oneway_bps=6.0, rebal=5).dropna()
    fg = C.cross_sectional_factor(F, R, lookback=7, q=0.33, cost_oneway_bps=0.0, rebal=5).dropna()
    xs = dict(net=tail(f), gross_sharpe=C.ann_sharpe(fg.values), by_year_net=by_year(f))
    octw = f[(f.index >= pd.Timestamp("2025-10-01", tz="UTC")) & (f.index <= pd.Timestamp("2025-11-15", tz="UTC"))]
    xs["oct2025_cum_pct"] = float(octw.sum() * 100) if len(octw) else None
    xs["oct2025_worst_day_pct"] = float(octw.min() * 100) if len(octw) else None
    out["xs_factor"] = xs
    out["xs_cost_sweep_sharpe"] = {f"{c}bp": round(C.ann_sharpe(
        C.cross_sectional_factor(F, R, cost_oneway_bps=c).dropna().values), 3) for c in [0, 3, 6, 10, 15]}

    # 3) cash-and-carry (time-series structural harvest)
    cc = C.cash_and_carry(F, lookback=7, thresh=0.0, cost_roundtrip_bps=18.0)
    ccg = C.cash_and_carry(F, lookback=7, thresh=0.0, cost_roundtrip_bps=0.0)
    out["cash_and_carry"] = dict(net=tail(cc), gross_ann_ret_pct=float(np.nanmean(ccg) * 365 * 100),
                                 by_year_net=by_year(cc[cc != 0]))

    # 4) H5: trailing funding -> forward 7d return (pooled cross-sectional rank-IC)
    sig = F.rolling(7).mean().shift(1); fwd = R.rolling(7).sum().shift(-7)
    idx = sig.index.intersection(fwd.index)
    ics = []
    for d in idx:
        a, b = sig.loc[d], fwd.loc[d]; m = a.notna() & b.notna()
        if m.sum() >= 8:
            ic = spearman_ic(a[m].values, b[m].values)
            if np.isfinite(ic):
                ics.append(ic)
    ics = np.array(ics)
    out["funding_predicts_fwd7d"] = dict(mean_rank_ic=float(ics.mean()),
        t=float(ics.mean() / ics.std() * np.sqrt(len(ics))) if ics.std() > 0 else None, n_days=len(ics))

    json.dump(out, open(f"{ROOT}/experiments/paper2_carry_core.json", "w"), indent=2, default=str)

    # ---- summary ----
    print(f"Universe {out['universe']} perps, {out['span'][0]}..{out['span'][1]}")
    print(f"Degenerate funding (excluded/flagged): {list(deg) or 'none'}")
    n = xs["net"]
    print(f"\nCROSS-SECTIONAL CARRY FACTOR (long-low / short-high funding, dollar-neutral):")
    print(f"  net: Sharpe {n['sharpe']:.2f}, ann {n['ann_ret_pct']:+.1f}% / vol {n['ann_vol_pct']:.1f}%, "
          f"maxDD {n['maxdd_pct']:.1f}%, worst day {n['worst_day_pct']:.1f}%, skew {n['skew']:.2f}  (gross Sharpe {xs['gross_sharpe']:.2f})")
    print(f"  by year (net Sharpe): {xs['by_year_net']}")
    print(f"  Oct-2025 cascade window: cum {xs['oct2025_cum_pct']}%, worst day {xs['oct2025_worst_day_pct']}%")
    print(f"  cost sweep (one-way bps -> net Sharpe): {out['xs_cost_sweep_sharpe']}")
    cn = out["cash_and_carry"]["net"]
    print(f"\nCASH-AND-CARRY (delta-neutral, collect funding):")
    print(f"  net: Sharpe {cn['sharpe']:.2f}, ann {cn['ann_ret_pct']:+.1f}%  (gross ann {out['cash_and_carry']['gross_ann_ret_pct']:+.1f}%)")
    print(f"  by year (net Sharpe): {out['cash_and_carry']['by_year_net']}")
    fp = out["funding_predicts_fwd7d"]
    print(f"\nH5 funding->fwd7d return rank-IC: {fp['mean_rank_ic']:+.4f} (t={fp['t']:.2f}, n={fp['n_days']})  "
          f"[negative = crowded longs underperform]")
    print("\nWrote experiments/paper2_carry_core.json")


if __name__ == "__main__":
    main()
