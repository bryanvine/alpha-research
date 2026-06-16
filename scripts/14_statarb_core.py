#!/usr/bin/env python3
"""
14_statarb_core.py -- Paper 3 core: PCA-residual (s-score) crypto statistical arbitrage,
walk-forward by construction, net of costs.

  * daily 30-coin panel (2023-2026): net/gross Sharpe, win-rate, by-year, cost sweep, degenerate check
  * hourly ~50-coin panel (2025+): net Sharpe + LIQUIDITY-TERCILE split (is any edge a
    stale-price artifact in illiquid coins?)

Results -> experiments/paper3_statarb_core.json
"""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import statarb as SA  # noqa: E402
from alpha_research.eval import rigor              # noqa: E402


def stats(pnl, ppy):
    x = pnl.values[np.isfinite(pnl.values)]; nz = x[x != 0]
    cum = np.cumsum(x); dd = cum - np.maximum.accumulate(cum)
    return dict(sharpe=SA.ann_sharpe(x, ppy), ann_ret_pct=float(np.mean(x) * ppy * 100),
                win_rate=float((nz > 0).mean()) if len(nz) else None,
                maxdd_pct=float(dd.min() * 100), n_active=int(len(nz)))


def by_year(pnl, ppy):
    return {int(y): round(SA.ann_sharpe(g.values, ppy), 2) for y, g in pnl.groupby(pnl.index.year)}


def main():
    out = {}

    # ---- DAILY ----
    Rd = SA.load_daily_panel()
    net = SA.pca_sscore_pnl(Rd, window=60, k=3, entry=1.25, cost_bps=5.0, rebal=1)
    gross = SA.pca_sscore_pnl(Rd, window=60, k=3, entry=1.25, cost_bps=0.0, rebal=1)
    deg = rigor.degenerate_signal_check(net.values[net.values != 0])
    out["daily"] = dict(universe=int(Rd.shape[1]), span=[str(Rd.index.min())[:10], str(Rd.index.max())[:10]],
                        net=stats(net, 365), gross_sharpe=SA.ann_sharpe(gross.values, 365),
                        by_year=by_year(net, 365),
                        degenerate=dict(flagged=deg.is_degenerate, reasons=deg.reasons))
    out["daily_cost_sweep"] = {f"{c}bp": round(SA.ann_sharpe(
        SA.pca_sscore_pnl(Rd, 60, 3, 1.25, c, 1).values, 365), 2) for c in [0, 5, 10, 20]}

    # ---- HOURLY + liquidity tercile ----
    Rh, dv = SA.load_hourly_panel()
    nh = SA.pca_sscore_pnl(Rh, window=336, k=3, entry=1.25, cost_bps=5.0, rebal=24)
    out["hourly"] = dict(universe=int(Rh.shape[1]), span=[str(Rh.index.min())[:10], str(Rh.index.max())[:10]],
                        net=stats(nh, 8760), by_year=by_year(nh, 8760))
    medv = dv.median().dropna().sort_values()
    third = max(1, len(medv) // 3)
    terc = {"low_liq": list(medv.index[:third]), "mid_liq": list(medv.index[third:2 * third]),
            "high_liq": list(medv.index[2 * third:])}
    out["hourly_liquidity_tercile"] = {}
    for name, coins in terc.items():
        coins = [c for c in coins if c in Rh.columns]
        p = SA.pca_sscore_pnl(Rh[coins], window=336, k=2, entry=1.25, cost_bps=5.0, rebal=24)
        out["hourly_liquidity_tercile"][name] = dict(n_coins=len(coins),
            net_sharpe=round(SA.ann_sharpe(p.values, 8760), 2),
            gross_sharpe=round(SA.ann_sharpe(SA.pca_sscore_pnl(Rh[coins], 336, 2, 1.25, 0.0, 24).values, 8760), 2))

    json.dump(out, open(f"{ROOT}/experiments/paper3_statarb_core.json", "w"), indent=2, default=str)

    d = out["daily"]
    print(f"DAILY PCA s-score stat-arb ({d['universe']} coins, {d['span'][0]}..{d['span'][1]}):")
    print(f"  net Sharpe {d['net']['sharpe']:.2f} (gross {d['gross_sharpe']:.2f}), ann {d['net']['ann_ret_pct']:+.1f}%, "
          f"win {d['net']['win_rate']*100:.0f}%, maxDD {d['net']['maxdd_pct']:.1f}%")
    print(f"  by year: {d['by_year']}")
    print(f"  cost sweep (bps->net Sharpe): {out['daily_cost_sweep']}")
    print(f"  degenerate: {d['degenerate']['flagged']}")
    h = out["hourly"]
    print(f"\nHOURLY PCA s-score ({h['universe']} coins, {h['span'][0]}..{h['span'][1]}):")
    print(f"  net Sharpe {h['net']['sharpe']:.2f}, ann {h['net']['ann_ret_pct']:+.1f}%, win {h['net']['win_rate']*100:.0f}%")
    print(f"  by year: {h['by_year']}")
    print(f"  LIQUIDITY TERCILE (net | gross Sharpe):")
    for k, v in out["hourly_liquidity_tercile"].items():
        print(f"    {k:9s} ({v['n_coins']} coins): {v['net_sharpe']:+.2f} | {v['gross_sharpe']:+.2f}")
    print("\nWrote experiments/paper3_statarb_core.json")


if __name__ == "__main__":
    main()
