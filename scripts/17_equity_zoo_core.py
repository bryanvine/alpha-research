#!/usr/bin/env python3
"""
17_equity_zoo_core.py -- Paper 4, Part B: directional anomalies on our own data, net of cost.

Equity (FnSpID, 2016-2025): momentum 12-1, short-term reversal, low-vol, news-sentiment
(follow + contrarian). Crypto (top-30 daily, 2023-2026): momentum, short-term reversal.
Cross-sectional dollar-neutral long-short, net of realistic costs, with sub-period decay,
turnover, cost sweep, degenerate check, and the +0.3 floor.

Results -> experiments/paper4_ourdata.json
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import equity_zoo as EZ   # noqa: E402
from alpha_research.factors import carry as C          # noqa: E402
from alpha_research.eval import rigor                  # noqa: E402

FLOOR = 0.3


def subperiods(pnl, ppy, splits):
    out = {}
    for lbl, a, b in splits:
        seg = pnl[(pnl.index >= pd.Timestamp(a, tz=pnl.index.tz)) & (pnl.index < pd.Timestamp(b, tz=pnl.index.tz))]
        out[lbl] = round(EZ.ann_sharpe(seg.values, ppy), 2)
    return out


def main():
    out = {"floor": FLOOR}

    # ---- EQUITY ----
    ret = EZ.load_returns(start="2016-01-01", min_obs_frac=0.6)
    sent = EZ.load_sentiment(start="2016-01-01")
    out["equity"] = dict(universe=int(ret.shape[1]), span=[str(ret.index.min())[:10], str(ret.index.max())[:10]])
    sigs = {
        "momentum_12_1": EZ.sig_momentum(ret), "short_term_reversal": EZ.sig_st_reversal(ret),
        "low_vol": EZ.sig_low_vol(ret),
        "sentiment_follow": EZ.sig_sentiment(sent, sign=+1), "sentiment_contrarian": EZ.sig_sentiment(sent, sign=-1),
    }
    sp = [("2016_2019", "2016-01-01", "2020-01-01"), ("2020_2022", "2020-01-01", "2023-01-01"),
          ("2023_2025", "2023-01-01", "2026-07-01")]
    eq = {}
    for name, sg in sigs.items():
        pnl, turn = EZ.ls_backtest(sg, ret, q=0.3, rebal=5, cost_bps=10.0)
        pnlg, _ = EZ.ls_backtest(sg, ret, q=0.3, rebal=5, cost_bps=0.0)
        deg = rigor.degenerate_signal_check(pnl.values[pnl.values != 0])
        eq[name] = dict(net_sharpe=round(EZ.ann_sharpe(pnl.values), 2),
                        gross_sharpe=round(EZ.ann_sharpe(pnlg.values), 2),
                        avg_turnover=round(turn, 3), by_period=subperiods(pnl, 252, sp),
                        degenerate=deg.is_degenerate)
    out["equity"]["factors"] = eq
    # cost sweep on momentum (representative)
    out["equity"]["momentum_cost_sweep"] = {f"{c}bp": round(EZ.ann_sharpe(
        EZ.ls_backtest(sigs["momentum_12_1"], ret, 0.3, 5, c)[0].values), 2) for c in [0, 5, 10, 20]}

    # ---- CRYPTO (top-30 daily) ----
    Rc = C.load_spot_returns()
    cmom = -np.log1p(Rc.fillna(0)).rolling(7).sum() * 0 + np.log1p(Rc.fillna(0)).rolling(30).sum()  # 30d momentum
    crev = -np.log1p(Rc.fillna(0)).rolling(7).sum()                                                 # 7d reversal
    cr = {}
    for name, sg in {"crypto_momentum_30d": cmom, "crypto_reversal_7d": crev}.items():
        pnl, turn = EZ.ls_backtest(sg, Rc, q=0.33, rebal=7, cost_bps=10.0)
        pnlg, _ = EZ.ls_backtest(sg, Rc, q=0.33, rebal=7, cost_bps=0.0)
        cr[name] = dict(net_sharpe=round(EZ.ann_sharpe(pnl.values, 365), 2),
                        gross_sharpe=round(EZ.ann_sharpe(pnlg.values, 365), 2), avg_turnover=round(turn, 3))
    out["crypto"] = dict(universe=int(Rc.shape[1]), factors=cr)

    json.dump(out, open(f"{ROOT}/experiments/paper4_ourdata.json", "w"), indent=2, default=str)

    print(f"EQUITY (FnSpID, {out['equity']['universe']} syms, {out['equity']['span'][0]}..{out['equity']['span'][1]}) "
          f"net of 10bp/side; floor +{FLOOR}:")
    print(f"  {'factor':22s} {'net':>6} {'gross':>6} {'turn':>6}  by-period(net)")
    for name, d in eq.items():
        print(f"  {name:22s} {d['net_sharpe']:>6} {d['gross_sharpe']:>6} {d['avg_turnover']:>6}  {d['by_period']}")
    print(f"  momentum cost sweep: {out['equity']['momentum_cost_sweep']}")
    print(f"\nCRYPTO (top-{out['crypto']['universe']} daily) net of 10bp/side:")
    for name, d in cr.items():
        print(f"  {name:22s} net {d['net_sharpe']:>6}  gross {d['gross_sharpe']:>6}  turn {d['avg_turnover']}")
    print("\nWrote experiments/paper4_ourdata.json")


if __name__ == "__main__":
    main()
