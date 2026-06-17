"""FX carry/value + commodity (energy) roll-yield / basis-momentum (Paper 6).

FX: G10 vs USD. Monthly excess return of holding a foreign currency =
interest differential + currency appreciation = carry_diff/12 - dlog(FXperUSD).
Carry = sort on the rate differential; value = sort on REER deviation.

Commodity: EIA energy futures (WTI, NatGas, HeatingOil, RBOB), front-through-4th.
Roll yield = log(C1/C2) (>0 backwardation); basis-momentum (Boons-Prado) =
12m return of C1 minus 12m return of C2. Cross-sectional long-short on the front contract.

NO LOOK-AHEAD: signals lagged one month before being applied to next month's return.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FX = "/apps/alpha-research/data/fx"
COMM = "/apps/alpha-research/data/commodity"


def load_fx():
    def rd(n):
        return pd.read_csv(f"{FX}/{n}", index_col=0, parse_dates=True).sort_index()
    return rd("panel_carry_diff_OECD3M_vs_USD.csv"), rd("panel_spot_FXperUSD.csv"), rd("panel_reer_broad.csv")


def fx_monthly_excess_returns(spot, carry):
    sp = spot.resample("ME").last()
    appr = -np.log(sp / sp.shift(1))                               # foreign appreciation (FXperUSD down = up)
    car = (carry.resample("ME").last().reindex(sp.index, method="ffill") / 100.0 / 12.0)
    return (appr + car.reindex(columns=appr.columns)).dropna(how="all")


def load_commodity():
    p = pd.read_parquet(f"{COMM}/eia_energy.parquet")
    p["date"] = pd.to_datetime(p["date"])
    return p


def commodity_panels(p):
    c1, c2, roll, bm = {}, {}, {}, {}
    for c in p["commodity"].unique():
        piv = p[p["commodity"] == c].pivot_table(index="date", columns="contract", values="price").sort_index()
        piv = piv.resample("ME").last()
        if not {1, 2}.issubset(piv.columns):
            continue
        c1[c] = piv[1]; c2[c] = piv[2]
        roll[c] = np.log(piv[1] / piv[2])                          # >0 backwardation
        bm[c] = np.log(piv[1] / piv[1].shift(12)) - np.log(piv[2] / piv[2].shift(12))
    C1, C2 = pd.DataFrame(c1), pd.DataFrame(c2)
    # ROLL-ADJUSTED held-front return: the contract held as 2nd-nearest last month is the
    # front this month (avoids the spurious splice jump in a naive C1_t/C1_{t-1} series).
    cret = np.log(C1 / C2.shift(1))
    return cret, pd.DataFrame(roll), pd.DataFrame(bm)


def ls_factor(signal_m, ret_m, n=3, cost_bps=5.0):
    """Monthly cross-sectional long top-n / short bottom-n of (lagged) signal, dollar-neutral."""
    sig = signal_m.shift(1)
    idx = ret_m.index
    pnl = np.zeros(len(idx)); prevw = pd.Series(0.0, index=ret_m.columns)
    for i, d in enumerate(idx):
        s = (sig.loc[d].dropna() if d in sig.index else pd.Series(dtype=float))
        s = s[[c for c in s.index if c in ret_m.columns]]
        w = pd.Series(0.0, index=ret_m.columns)
        if len(s) >= 4:
            k = min(n, len(s) // 2)
            w[s.nlargest(k).index] = 0.5 / k
            w[s.nsmallest(k).index] = -0.5 / k
        r = ret_m.loc[d].reindex(w.index).fillna(0.0)
        pnl[i] = float((w * r).sum() - (w - prevw).abs().sum() * cost_bps / 1e4)
        prevw = w
    return pd.Series(pnl, index=idx)


def ann_sharpe_m(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[x != 0]
    return float(x.mean() / x.std() * np.sqrt(12)) if x.size >= 12 and x.std() > 0 else float("nan")
