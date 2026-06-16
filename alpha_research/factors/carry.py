"""Crypto perpetual funding-rate carry factors (Paper 2).

Built from the Binance USD-M backfill (data/funding/): an event-level 8h funding
panel (4h for TIA) summed to a daily funding-accrued grid, and per-symbol daily
spot. Two harvests:
  * cross_sectional_factor : dollar-neutral long-low-funding / short-high-funding
    perp book (the relative-value carry factor). Funding is *received* on both
    legs (short the rich, long the cheap); price leg bets crowded longs mean-revert.
  * cash_and_carry         : the time-series structural harvest -- hold
    long-spot/short-perp while trailing funding > threshold and collect funding.

NO LOOK-AHEAD: the ranking/position signal is the trailing funding mean shifted by
one day, applied to the *next* day's funding accrual and price return.

Funding-sign convention: a LONG perp pays funding when funding_rate>0, a SHORT
receives it, so portfolio funding PnL = -(w . funding).

MATIC funding is degenerate (constant +0.0001/8h) after the POL rename (~2024-09-10)
and is masked out; TIA's 4h cadence is handled by daily summation.
"""
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd

FUND = "/apps/alpha-research/data/funding"


def load_daily_funding() -> pd.DataFrame:
    """date x symbol daily funding accrued (sum of intraday settlements)."""
    p = pd.read_parquet(f"{FUND}/panel_funding_8h.parquet")
    p["time"] = pd.to_datetime(p["time"], utc=True)
    p["date"] = p["time"].dt.floor("D")
    daily = p.groupby(["date", "symbol"])["funding_rate"].sum().unstack("symbol").sort_index()
    if "MATICUSDT" in daily.columns:                       # degenerate after POL rename
        daily.loc[daily.index >= pd.Timestamp("2024-09-10", tz="UTC"), "MATICUSDT"] = np.nan
    return daily


def load_spot_returns() -> pd.DataFrame:
    """date x symbol daily log return."""
    frames = {}
    for f in glob.glob(f"{FUND}/*_spot1d.parquet"):
        sym = os.path.basename(f).replace("_spot1d.parquet", "")
        s = pd.read_parquet(f); s["time"] = pd.to_datetime(s["time"], utc=True)
        frames[sym] = s.set_index("time")["close"]
    px = pd.DataFrame(frames).sort_index()
    px.index = px.index.floor("D")
    return np.log(px / px.shift(1))


def cross_sectional_factor(F: pd.DataFrame, R: pd.DataFrame, lookback: int = 7,
                           q: float = 0.33, cost_oneway_bps: float = 6.0,
                           rebal: int = 5) -> pd.Series:
    """Daily PnL of the dollar-neutral funding-carry factor (gross exposure 1.0)."""
    idx = F.index.intersection(R.index)
    F = F.reindex(idx); R = R.reindex(idx)
    sig = F.rolling(lookback).mean().shift(1)               # trailing funding, no look-ahead
    W = pd.DataFrame(np.nan, index=idx, columns=F.columns)
    for d in idx[lookback + 1::rebal]:
        s = sig.loc[d].dropna()
        s = s[[c for c in s.index if c in R.columns]]
        if len(s) < 6:
            continue
        lo, hi = s.quantile(q), s.quantile(1 - q)
        longs, shorts = s[s <= lo].index, s[s >= hi].index
        w = pd.Series(0.0, index=F.columns)
        if len(longs):
            w[longs] = 0.5 / len(longs)
        if len(shorts):
            w[shorts] = -0.5 / len(shorts)
        W.loc[d] = w
    W = W.ffill().fillna(0.0)
    price = (W * R).sum(axis=1)
    fund = -(W * F).sum(axis=1)                             # receive funding: -(w . f)
    cost = W.diff().abs().sum(axis=1).fillna(0.0) * cost_oneway_bps / 1e4
    return (price + fund - cost).rename("pnl")


def cash_and_carry(F: pd.DataFrame, lookback: int = 7, thresh: float = 0.0,
                   cost_roundtrip_bps: float = 18.0) -> pd.Series:
    """Daily PnL of the equal-weight time-series cash-and-carry harvest (delta-neutral)."""
    sig = F.rolling(lookback).mean().shift(1)
    inpos = (sig > thresh).astype(float)
    carry = inpos * F                                       # receive funding while positioned
    entries_up = (inpos.diff() == 1).astype(float)         # charge round-trip on each entry
    net = carry - entries_up * cost_roundtrip_bps / 1e4
    n_active = inpos.sum(axis=1).replace(0, np.nan)
    return (net.sum(axis=1) / n_active).fillna(0.0).rename("pnl")


def ann_sharpe(x, ppy: float = 365.0) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 20 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(ppy))
