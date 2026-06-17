#!/usr/bin/env python3
"""
19_microstructure_core.py -- Paper 5: liquidity provision vs the fill-model illusion.

On SPY 100ms LOB (Nov-Dec 2025), per day then pooled:
  H2  OFI is contemporaneous, not predictive: R2(dMid ~ OFI) now vs at forward horizons;
      a marketable (taker) OFI strategy nets NEGATIVE after the spread.
  H1  fill-model illusion: a naive maker captures the half-spread (positive); marking the
      fill at the mid 5s later (adverse selection) flips it negative.
  H3  execution overlay: micro-price-gated timing shaves a fraction of the half-spread vs
      an immediate market order (the real, harvestable use).
  H4  reversal = liquidity provision: short-term reversal's Sharpe rises with VIX.

Results -> experiments/paper5_microstructure.json
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import microstructure as MS   # noqa: E402
from alpha_research.factors import equity_zoo as EZ        # noqa: E402

HZ = [1, 10, 50, 100, 300]    # 0.1s, 1s, 5s, 10s, 30s
NADV = 50                     # 5s adverse-selection horizon


def r2(y, x):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 100 or x.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def analyze_day(f):
    d = MS.load_day(f, "SPY")
    if d is None:
        return None
    F = MS.features(d)
    mid, micro, spr, ofi = F["mid"], F["micro"], F["spr"], F["ofi"]
    n = len(mid); half_bps = (spr / mid * 1e4) / 2.0
    dmid = np.diff(mid, prepend=mid[0])
    out = {"n": int(n), "half_spread_bps": float(np.mean(half_bps))}
    # H2: contemporaneous vs predictive
    out["r2_contemp"] = r2(dmid, ofi)
    out["r2_pred"] = {}
    for h in HZ:
        fwd = np.full(n, np.nan); fwd[:-h] = mid[h:] - mid[:-h]
        out["r2_pred"][h] = r2(fwd, ofi)
    # H2 taker: trade sign(OFI), hold NADV bars, pay full spread
    sig = np.sign(ofi)
    fwd = np.full(n, np.nan); fwd[:-NADV] = (mid[NADV:] - mid[:-NADV]) / mid[:-NADV] * 1e4
    taker = sig * fwd - (spr / mid * 1e4)            # cross the spread
    out["taker_net_bps"] = float(np.nanmean(taker[sig != 0]))
    # H1: maker naive vs adverse-selection-adjusted
    midN = np.full(n, np.nan); midN[:-NADV] = mid[NADV:]
    down = dmid < 0; up = dmid > 0                   # bid hit (maker buys) / ask lifted (maker sells)
    naive = float(np.nanmean(half_bps))              # captures half-spread by construction
    buy_pnl = (midN[down] - d["b"][down]) / mid[down] * 1e4     # bought at bid, mark at mid+5s
    sell_pnl = (d["a"][up] - midN[up]) / mid[up] * 1e4          # sold at ask, mark at mid+5s
    realistic = float(np.nanmean(np.concatenate([buy_pnl, sell_pnl])))
    out["maker_naive_bps"], out["maker_realistic_bps"] = naive, realistic
    # H3: execution overlay (buying): immediate cross vs micro-price-gated wait
    K = 20
    idx = np.arange(0, n - NADV, K)
    cost_naive = (d["a"][idx] - mid[idx]) / mid[idx] * 1e4
    wait_cost = (d["a"][idx + NADV] - mid[idx]) / mid[idx] * 1e4
    smart = np.where(micro[idx] > mid[idx], cost_naive, wait_cost)   # cross if rising, else wait
    out["exec_cost_naive_bps"] = float(np.nanmean(cost_naive))
    out["exec_cost_overlay_bps"] = float(np.nanmean(smart))
    return out


def main():
    files = MS.day_files()
    days = [a for a in (analyze_day(f) for f in files) if a]
    nd = len(days)

    def avg(key, sub=None):
        vals = [(d[key][sub] if sub is not None else d[key]) for d in days]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    out = {"symbol": "SPY", "n_days": nd, "total_snapshots": int(sum(d["n"] for d in days))}
    out["H2_ofi"] = dict(
        r2_contemporaneous=round(avg("r2_contemp"), 3),
        r2_predictive_by_horizon={f"{h}bar": round(avg("r2_pred", h), 4) for h in HZ},
        taker_net_bps=round(avg("taker_net_bps"), 2),
        half_spread_bps=round(avg("half_spread_bps"), 3))
    out["H1_fill_illusion"] = dict(maker_naive_bps=round(avg("maker_naive_bps"), 3),
                                   maker_realistic_bps=round(avg("maker_realistic_bps"), 3))
    cn, co = avg("exec_cost_naive_bps"), avg("exec_cost_overlay_bps")
    out["H3_execution_overlay"] = dict(cost_naive_bps=round(cn, 3), cost_overlay_bps=round(co, 3),
                                       slippage_reduction_pct=round((1 - co / cn) * 100, 1))

    # H4: reversal = liquidity provision (Sharpe rises with VIX)
    try:
        ret = EZ.load_returns(start="2016-01-01", min_obs_frac=0.6)
        rev = EZ.sig_st_reversal(ret)
        pnl, _ = EZ.ls_backtest(rev, ret, q=0.3, rebal=5, cost_bps=10.0)
        vix = pd.read_csv(f"{ROOT}/data/equity_vol/vixcls.csv")
        vix.columns = ["date", "vix"]; vix["date"] = pd.to_datetime(vix["date"], errors="coerce")
        vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")
        vix = vix.dropna().set_index("date")["vix"]
        p = pnl.copy(); p.index = pd.to_datetime(p.index).tz_localize(None)
        v = vix.reindex(p.index, method="ffill")
        hi = v > v.median()
        out["H4_reversal_vs_vix"] = dict(
            sharpe_low_vix=round(EZ.ann_sharpe(p[~hi].values), 2),
            sharpe_high_vix=round(EZ.ann_sharpe(p[hi].values), 2),
            vix_median=round(float(v.median()), 1))
    except Exception as e:
        out["H4_reversal_vs_vix"] = {"error": str(e)}

    json.dump(out, open(f"{ROOT}/experiments/paper5_microstructure.json", "w"), indent=2, default=str)

    print(f"SPY microstructure: {nd} days, {out['total_snapshots']:,} snapshots (100ms, RTH)")
    h2 = out["H2_ofi"]
    print(f"\nH2 OFI contemporaneous vs predictive (half-spread {h2['half_spread_bps']}bp):")
    print(f"  R2 contemporaneous (dMid~OFI now): {h2['r2_contemporaneous']}")
    print(f"  R2 predictive by horizon: {h2['r2_predictive_by_horizon']}")
    print(f"  taker strategy on OFI, net of spread: {h2['taker_net_bps']} bps  (expect <0)")
    h1 = out["H1_fill_illusion"]
    print(f"\nH1 fill-model illusion: naive maker +{h1['maker_naive_bps']}bp  ->  "
          f"realistic (5s adverse selection) {h1['maker_realistic_bps']}bp")
    h3 = out["H3_execution_overlay"]
    print(f"\nH3 execution overlay: cost naive {h3['cost_naive_bps']}bp -> overlay {h3['cost_overlay_bps']}bp "
          f"({h3['slippage_reduction_pct']}% reduction)")
    h4 = out["H4_reversal_vs_vix"]
    if "error" not in h4:
        print(f"\nH4 reversal Sharpe: low-VIX {h4['sharpe_low_vix']}  high-VIX {h4['sharpe_high_vix']} "
              f"(VIX median {h4['vix_median']}) -- liquidity-provision premium rises with vol")
    print("\nWrote experiments/paper5_microstructure.json")


if __name__ == "__main__":
    main()
