"""Volatility-risk-premium construction + short-variance-swap harvest (Paper 1).

Canonical, reusable version of the logic first prototyped in
``scripts/01_vrp_core.py``. The time-series harvest runs off DVOL (risk-neutral
30d implied vol) vs OHLCV realized variance -- the Deribit option surface is a
single end-of-sample snapshot and cannot price the options leg historically.

NO LOOK-AHEAD: ``impl_vol`` is the DVOL known at/before each decision bar
(merge_asof backward); the gate uses *trailing* realized vol; the swap payoff
uses realized variance over the forward 30d window, which is the trade (resolved
later), not leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DERIBIT = "/apps/jepa-trader/data/raw_deribit"
H = 60                          # 30 days @ 12h bars
BARS_PER_YEAR = 365.25 * 2      # 730.5
WIN_PER_YEAR = 365.25 / 30.0    # ~12.17 non-overlapping 30d windows / year
ANN = BARS_PER_YEAR / H         # annualization for a 30d realized-variance window


def load_coin(coin: str, deribit: str = DERIBIT) -> pd.DataFrame:
    """Load Deribit index OHLCV + DVOL for ``coin`` ('BTC'/'ETH'), DVOL attached
    by the most-recent value known at/before each price bar (no look-ahead)."""
    px = pd.read_parquet(f"{deribit}/price_{coin}.parquet").sort_values("time").reset_index(drop=True)
    dv = pd.read_parquet(f"{deribit}/dvol_{coin}.parquet").sort_values("time").reset_index(drop=True)
    px["time"] = pd.to_datetime(px["time"]); dv["time"] = pd.to_datetime(dv["time"])
    return pd.merge_asof(px[["time", "open", "high", "low", "close"]],
                         dv[["time", "dvol_c"]], on="time", direction="backward")


def realized_var(df: pd.DataFrame, method: str = "c2c"):
    """Per-bar realized variance proxy → (forward-sum, trailing-sum) annualized.

    method: 'c2c' (close-to-close), 'parkinson' (high-low range), or
    'gk' (Garman-Klass, uses OHLC). Forward = sum over (i, i+H]; trailing =
    sum over (i-H, i] (no look-ahead)."""
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float); op = df["open"].to_numpy(float)
    if method == "c2c":
        per = np.concatenate([[np.nan], np.log(close[1:] / close[:-1])]) ** 2
    elif method == "parkinson":
        per = (1.0 / (4 * np.log(2))) * (np.log(high / low)) ** 2
    elif method == "gk":
        per = 0.5 * (np.log(high / low)) ** 2 - (2 * np.log(2) - 1) * (np.log(close / op)) ** 2
    else:
        raise ValueError(f"unknown RV method {method!r}")
    s = pd.Series(per)
    fwd = s.rolling(H).sum().shift(-H).to_numpy() * ANN
    trail = s.rolling(H).sum().to_numpy() * ANN
    return fwd, trail


def build(coin: str, method: str = "c2c", deribit: str = DERIBIT) -> pd.DataFrame:
    """VRP panel for ``coin``: implied (DVOL) vs forward realized variance."""
    df = load_coin(coin, deribit)
    fwd, trail = realized_var(df, method)
    closeS = pd.Series(df["close"].to_numpy(float))
    out = pd.DataFrame(dict(
        time=df["time"], impl_vol=df["dvol_c"].to_numpy(float) / 100.0,
        fwd_rv_var=fwd, trail_rv_var=trail,
        fwd_ret=np.log(closeS.shift(-H) / closeS).to_numpy()))
    out["impl_var"] = out["impl_vol"] ** 2
    out["vrp_var"] = out["impl_var"] - out["fwd_rv_var"]            # variance-swap PnL (short)
    out["vrp_vol"] = out["impl_vol"] - np.sqrt(out["fwd_rv_var"])   # vol-points PnL (short)
    return out


def harvest_pnl(panel: pd.DataFrame, phase: int = 0, gate: bool = True,
                voltgt: bool = True, cost_vp: float = 2.0) -> pd.DataFrame:
    """Non-overlapping short-variance-swap harvest PnL series.

    phase: entry offset in [0, H). gate: sell only when implied > trailing
    realized vol (positive ex-ante carry). voltgt: inverse-vol (≈constant-vega)
    sizing. cost_vp: round-trip cost in annualized vol points (applied to the
    strike). Returns the entry rows with a ``pnl`` column."""
    ent = panel.iloc[phase::H].dropna(subset=["impl_vol", "fwd_rv_var", "trail_rv_var"]).copy()
    impl = ent["impl_vol"].to_numpy(); rv = ent["fwd_rv_var"].to_numpy()
    trail_rvol = np.sqrt(ent["trail_rv_var"].to_numpy())
    cc = cost_vp / 100.0
    pnl = (np.maximum(impl - cc, 0.0)) ** 2 - rv                     # short variance swap, net strike
    g = (impl > trail_rvol) if gate else np.ones_like(impl, bool)
    size = np.clip(np.nanmedian(impl) / impl, 0.25, 4.0) if voltgt else np.ones_like(impl)
    ent["pnl"] = np.where(g, size * pnl, 0.0)
    ent["in_market"] = g
    return ent


def ann_sharpe(x) -> float:
    """Annualized Sharpe of a per-30d-window PnL series."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 8 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(WIN_PER_YEAR))
