#!/usr/bin/env python3
"""
22_portfolio_construction.py -- Paper 7 (synthesis): do the modest edges add up?

Combine the surviving ~0.4-Sharpe premia into one risk-parity, vol-targeted, NO-LOOK-AHEAD
book and test whether diversification lifts the combined net Sharpe above the individual
sleeves and clears the AQR-QSPIX ~0.41 live ceiling.

Sleeves (each a monthly net-of-cost long-short return series):
  LONG-HISTORY book (no crypto): FX carry, FX value, equity quality (RMW), reversal (ST_Rev).
  FULL book (recent overlap):    + crypto funding carry, + crypto trend.

Weights = inverse trailing-36m vol (risk parity), lagged one month; book scaled to 10%/yr
target vol on trailing vol; net of a small overlay rebalancing cost.

Results -> experiments/paper7_portfolio.json
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import macro_carry as MC   # noqa: E402
from alpha_research.factors import carry as C           # noqa: E402
from alpha_research.factors import equity_zoo as EZ     # noqa: E402

QSPIX, FLOOR, TGT = 0.41, 0.30, 0.10 / np.sqrt(12)      # monthly target vol


def to_m(s):
    """Any-frequency return series -> monthly PeriodIndex (sum within month), tz-naive."""
    s = s.dropna().copy()
    idx = pd.to_datetime(s.index)
    idx = idx.tz_convert(None) if getattr(idx, "tz", None) is not None else idx
    s.index = idx.to_period("M")
    return s.groupby(level=0).sum()


def sharpe_m(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * np.sqrt(12)) if x.size >= 12 and x.std() > 0 else float("nan")


def risk_parity(B, cost_bps=5.0):
    """Inverse-trailing-vol risk-parity combine (lagged weights) -> (combined, vol_targeted)."""
    vol = B.rolling(36, min_periods=12).std()
    w = (1.0 / vol)
    w = w.div(w.sum(axis=1), axis=0).shift(1)             # weights known at t-1
    combined = (w * B).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    combined = combined - turn * cost_bps / 1e4           # overlay rebalancing cost
    tvol = combined.rolling(36, min_periods=12).std().shift(1)
    scale = (TGT / tvol).clip(upper=3.0)
    return combined.dropna(), (scale * combined).dropna()


def build_sleeves():
    sl = {}
    # FX carry + value (monthly)
    carry_d, spot, reer = MC.load_fx()
    ret = MC.fx_monthly_excess_returns(spot, carry_d)
    csig = carry_d.resample("ME").last().reindex(ret.index, method="ffill")
    sl["fx_carry"] = to_m(MC.ls_factor(csig, ret, n=3, cost_bps=5.0))
    rsig = reer.resample("ME").last().reindex(ret.index, method="ffill")
    vsig = -(rsig.sub(rsig.mean(axis=1), axis=0))
    sl["fx_value"] = to_m(MC.ls_factor(vsig[ret.columns.intersection(vsig.columns)], ret, n=3, cost_bps=5.0))
    # Equity quality (RMW) + reversal (ST_Rev) from Ken French (monthly, decimal)
    F = pd.read_parquet(f"{ROOT}/data/equity_factors/famafrench_monthly.parquet")
    F.index = pd.to_datetime(F.index)
    sl["quality"] = to_m(F["RMW"]); sl["reversal"] = to_m(F["ST_Rev"])
    # Crypto funding carry + trend (daily -> monthly, 2023+)
    Rc = C.load_spot_returns()
    sl["crypto_carry"] = to_m(C.cross_sectional_factor(C.load_daily_funding(), Rc, 7, 0.33, 6.0, 5))
    cmom = np.log1p(Rc.fillna(0)).rolling(30).sum()
    sl["crypto_trend"] = to_m(EZ.ls_backtest(cmom, Rc, q=0.33, rebal=7, cost_bps=10.0)[0])
    return pd.DataFrame(sl)


def book_stats(B, combined, vt, label):
    x = combined.values
    cum = np.cumsum(x); dd = cum - np.maximum.accumulate(cum)
    ind = {c: round(sharpe_m(B[c].dropna().values), 2) for c in B.columns}
    return {
        "label": label, "sleeves": list(B.columns),
        "span": [str(B.dropna().index.min()), str(B.dropna().index.max())], "n_months": int(len(combined)),
        "individual_sharpe": ind, "avg_individual_sharpe": round(np.nanmean(list(ind.values())), 2),
        "best_individual_sharpe": round(np.nanmax(list(ind.values())), 2),
        "combined_sharpe": round(sharpe_m(x), 2),
        "diversification_ratio": round(sharpe_m(x) / np.nanmean(list(ind.values())), 2),
        "vol_targeted_realized_vol_pct": round(float(np.nanstd(vt.values) * np.sqrt(12) * 100), 1),
        "vol_targeted_maxdd_pct": round(float((np.cumsum(vt.values) - np.maximum.accumulate(np.cumsum(vt.values))).min() * 100), 1),
        "corr": B.dropna().corr().round(2).to_dict(),
    }


def main():
    S = build_sleeves()
    out = {"benchmarks": {"qspix_live": QSPIX, "floor": FLOOR}}

    # LONG-HISTORY book (no crypto)
    longcols = ["fx_carry", "fx_value", "quality", "reversal"]
    BL = S[longcols].dropna()
    cL, vL = risk_parity(BL)
    out["long_history_book"] = book_stats(BL.loc[cL.index.min():], cL, vL, "FX carry+value, quality, reversal")
    # by-decade
    cL_ts = cL.copy(); cL_ts.index = cL_ts.index.to_timestamp()
    out["long_history_book"]["by_decade_sharpe"] = {
        d: round(sharpe_m(cL_ts[(cL_ts.index.year // 10 * 10) == d].values), 2)
        for d in sorted(set(cL_ts.index.year // 10 * 10))}

    # FULL book incl. crypto (recent overlap)
    BF = S.dropna()
    if len(BF) >= 18:
        cF, vF = risk_parity(BF)
        out["full_book_with_crypto"] = book_stats(BF.loc[cF.index.min():] if len(cF) else BF, cF, vF, "all six sleeves")
    else:
        out["full_book_with_crypto"] = {"note": f"insufficient overlap ({len(BF)} months) for a stable estimate"}

    json.dump(out, open(f"{ROOT}/experiments/paper7_portfolio.json", "w"), indent=2, default=str)

    L = out["long_history_book"]
    print(f"=== LONG-HISTORY diversified book ({L['span'][0]}..{L['span'][1]}, {L['n_months']} months) ===")
    print(f"  sleeves: {L['sleeves']}")
    print(f"  individual Sharpes: {L['individual_sharpe']}  (avg {L['avg_individual_sharpe']}, best {L['best_individual_sharpe']})")
    print(f"  COMBINED Sharpe = {L['combined_sharpe']}  (diversification ratio {L['diversification_ratio']}x)")
    print(f"  vs QSPIX {QSPIX} live / floor {FLOOR}")
    print(f"  vol-targeted: realized vol {L['vol_targeted_realized_vol_pct']}%/yr, maxDD {L['vol_targeted_maxdd_pct']}%")
    print(f"  by decade: {L['by_decade_sharpe']}")
    print(f"  sleeve correlations:")
    cm = pd.DataFrame(L["corr"])
    print("   " + cm.to_string().replace("\n", "\n   "))
    FB = out["full_book_with_crypto"]
    print(f"\n=== FULL book incl. crypto ===")
    if "combined_sharpe" in FB:
        print(f"  span {FB['span'][0]}..{FB['span'][1]} ({FB['n_months']} months); sleeves {FB['sleeves']}")
        print(f"  individual {FB['individual_sharpe']}")
        print(f"  COMBINED Sharpe = {FB['combined_sharpe']} (diversification {FB['diversification_ratio']}x)  [thin sample — illustrative]")
    else:
        print(f"  {FB['note']}")
    print("\nWrote experiments/paper7_portfolio.json")


if __name__ == "__main__":
    main()
