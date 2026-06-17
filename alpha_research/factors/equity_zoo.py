"""Directional anomaly zoo on our OWN data (Paper 4, Part B).

Price-based equity factors (momentum 12-1, short-term reversal, low-volatility) and a
news-sentiment factor, built on the FnSpID daily adjusted-price + sentiment panels, traded
cross-sectionally (dollar-neutral long-short) net of realistic costs. We have no
fundamentals / market-cap locally, so value / quality / size / BAB are audited via published
factor returns in Part A (OSAP + Ken French), not here.

NO LOOK-AHEAD: signals are shifted one day before being applied to the next day's return.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FN = "/apps/jepa-trader/data/raw_fnspid"


def load_returns(start="2016-01-01", min_obs_frac=0.6, winsor=0.5) -> pd.DataFrame:
    """date x sym daily returns from FnSpID adjusted prices, restricted to a liquid-ish
    survivor universe (>= min_obs_frac coverage since `start`), winsorized to kill
    split/data-error spikes."""
    df = pd.read_parquet(f"{FN}/prices_panel.parquet", columns=["date", "sym", "adj"])
    df["date"] = pd.to_datetime(df["date"]); df = df[df["date"] >= pd.Timestamp(start)]
    px = df.pivot_table(index="date", columns="sym", values="adj").sort_index()
    ret = px.pct_change().clip(-winsor, winsor)
    cov = ret.notna().mean()
    return ret[cov[cov >= min_obs_frac].index]


def load_sentiment(start="2016-01-01") -> pd.DataFrame:
    s = pd.read_parquet(f"{FN}/sentiment_panel.parquet")
    s["date"] = pd.to_datetime(s["date"]); s = s[s["date"] >= pd.Timestamp(start)]
    return s.pivot_table(index="date", columns="sym", values="sent").sort_index()


# ---- signals (higher = more attractive to go LONG) ----
def sig_momentum(ret, lb=252, skip=21):
    lr = np.log1p(ret.fillna(0.0))
    return lr.rolling(lb - skip).sum().shift(skip)                 # 12-1 momentum


def sig_st_reversal(ret, lb=21):
    return -np.log1p(ret.fillna(0.0)).rolling(lb).sum()           # contrarian on last month


def sig_low_vol(ret, lb=60):
    return -ret.rolling(lb).std()                                  # long low-vol


def sig_sentiment(sent, lb=5, sign=+1):
    return sign * sent.rolling(lb).mean()                          # +1 follow / -1 contrarian


def ls_backtest(sig: pd.DataFrame, ret: pd.DataFrame, q=0.3, rebal=5, cost_bps=10.0):
    """Cross-sectional dollar-neutral long-short PnL (gross exposure 1.0), net of turnover cost."""
    idx = sig.index.intersection(ret.index)
    S = sig.reindex(idx).shift(1)                                  # no look-ahead
    R = ret.reindex(idx)
    W = pd.DataFrame(0.0, index=idx, columns=R.columns)
    rebal_days = idx[::rebal]
    for d in rebal_days:
        s = S.loc[d].dropna()
        s = s[[c for c in s.index if c in R.columns]]
        if len(s) < 20:
            continue
        lo, hi = s.quantile(q), s.quantile(1 - q)
        longs, shorts = s[s >= hi].index, s[s <= lo].index
        w = pd.Series(0.0, index=R.columns)
        if len(longs):
            w[longs] = 0.5 / len(longs)
        if len(shorts):
            w[shorts] = -0.5 / len(shorts)
        W.loc[d] = w
    W = W.replace(0.0, np.nan).ffill().fillna(0.0)
    pnl = (W * R.fillna(0.0)).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(0.0)
    return (pnl - turn * cost_bps / 1e4).rename("pnl"), float(turn.mean())


def ann_sharpe(x, ppy=252.0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[x != 0]
    if x.size < 20 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(ppy))
