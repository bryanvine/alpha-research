#!/usr/bin/env python3
"""
21_macro_carry_core.py -- Paper 6: FX carry/value + commodity (energy) roll yield / basis-momentum.

FX (G10, monthly, net 5bp): carry (rate-differential sort) and value (REER), with crash skew
and pre/post-2008 decay. Commodity (4 EIA energy futures, 2004-2024, net 10bp): roll-yield carry
and basis-momentum, with pre/post-2010 (financialization) decay. Benchmarks: +0.3 floor and the
AQR QSPIX ~0.41 live multi-factor net Sharpe.

Results -> experiments/paper6_macro.json
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import skew as sp_skew
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import macro_carry as MC   # noqa: E402

FLOOR, QSPIX = 0.3, 0.41


def stats(pnl, splits=None):
    x = pnl.values[np.isfinite(pnl.values)]
    out = dict(sharpe=round(MC.ann_sharpe_m(pnl.values), 2),
               ann_ret_pct=round(float(np.nanmean(x) * 12 * 100), 2),
               skew=round(float(sp_skew(x[x != 0])), 2) if (x != 0).sum() > 8 else None,
               n_months=int((x != 0).sum()))
    if splits:
        out["by_period"] = {}
        for lbl, a, b in splits:
            seg = pnl[(pnl.index >= pd.Timestamp(a)) & (pnl.index < pd.Timestamp(b))]
            out["by_period"][lbl] = round(MC.ann_sharpe_m(seg.values), 2)
    return out


def main():
    out = {"floor": FLOOR, "qspix_live_benchmark": QSPIX}

    # ---- FX ----
    carry, spot, reer = MC.load_fx()
    ret = MC.fx_monthly_excess_returns(spot, carry)
    fx_split = [("pre_2008", "1990-01-01", "2008-01-01"), ("2008_2015", "2008-01-01", "2015-01-01"),
                ("post_2015", "2015-01-01", "2027-01-01")]
    carry_sig = carry.resample("ME").last().reindex(ret.index, method="ffill")
    fx_carry = MC.ls_factor(carry_sig, ret, n=3, cost_bps=5.0)
    # value: low REER (cheap) = long; demean cross-sectionally
    reer_m = reer.resample("ME").last().reindex(ret.index, method="ffill")
    val_sig = -(reer_m.sub(reer_m.mean(axis=1), axis=0))           # cheap (low REER) -> high signal -> long
    fx_value = MC.ls_factor(val_sig[ret.columns.intersection(val_sig.columns)], ret, n=3, cost_bps=5.0)
    out["fx"] = dict(span=[str(ret.index.min())[:7], str(ret.index.max())[:7]],
                     n_currencies=int(ret.shape[1]),
                     carry=stats(fx_carry, fx_split), value=stats(fx_value, fx_split))

    # ---- Commodity (energy) ----
    cret, roll, bm = MC.commodity_panels(MC.load_commodity())
    com_split = [("pre_2010", "2004-01-01", "2010-01-01"), ("post_2010", "2010-01-01", "2025-01-01")]
    roll_f = MC.ls_factor(roll, cret, n=2, cost_bps=10.0)
    bm_f = MC.ls_factor(bm, cret, n=2, cost_bps=10.0)
    out["commodity"] = dict(span=[str(cret.index.min())[:7], str(cret.index.max())[:7]],
                            n_commodities=int(cret.shape[1]), members=list(cret.columns),
                            roll_yield_carry=stats(roll_f, com_split),
                            basis_momentum=stats(bm_f, com_split))

    json.dump(out, open(f"{ROOT}/experiments/paper6_macro.json", "w"), indent=2, default=str)

    fx = out["fx"]
    print(f"FX G10 ({fx['n_currencies']} ccys, {fx['span'][0]}..{fx['span'][1]}) net 5bp; floor {FLOOR}, QSPIX {QSPIX}:")
    for name in ["carry", "value"]:
        d = fx[name]
        print(f"  {name:7s}: Sharpe {d['sharpe']:>5}  ann {d['ann_ret_pct']:>5}%  skew {d['skew']}  by-period {d['by_period']}")
    co = out["commodity"]
    print(f"\nCOMMODITY energy ({co['n_commodities']}: {co['members']}, {co['span'][0]}..{co['span'][1]}) net 10bp:")
    for name in ["roll_yield_carry", "basis_momentum"]:
        d = co[name]
        print(f"  {name:16s}: Sharpe {d['sharpe']:>5}  ann {d['ann_ret_pct']:>5}%  skew {d['skew']}  by-period {d['by_period']}")
    print("\nWrote experiments/paper6_macro.json")


if __name__ == "__main__":
    main()
