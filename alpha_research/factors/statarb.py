"""Crypto statistical arbitrage (Paper 3).

PCA-residual mean-reversion (Avellaneda & Lee 2010 "s-score"): rolling-window PCA
extracts common factors, each coin's residual is modelled as an OU process, and we
trade the standardized residual deviation (buy oversold residuals, sell overbought),
dollar-neutral. The strategy is walk-forward BY CONSTRUCTION -- every s-score uses
only the trailing window, and weights set from info through bar t-1 earn the t-1->t
return (no look-ahead).

Also: cointegration pairs (Engle-Granger) for the multiple-testing / PBO illustration.

Data: daily 30-coin panel (Binance spot backfill, 2023-2026) and an hourly ~50-coin
panel (jepa raw_crypto bars_1h, 2025+); the hourly panel carries volume for a
liquidity-tercile split (does any edge live only in illiquid coins?).
"""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd

FUND = "/apps/alpha-research/data/funding"
BARS1H = "/apps/jepa-trader/data/raw_crypto/bars_1h.csv"


def load_daily_panel():
    """date x coin daily log-return panel (30 coins, 2023-2026)."""
    frames = {}
    for f in glob.glob(f"{FUND}/*_spot1d.parquet"):
        sym = os.path.basename(f).replace("_spot1d.parquet", "")
        s = pd.read_parquet(f); s["time"] = pd.to_datetime(s["time"], utc=True)
        frames[sym] = s.set_index("time")["close"]
    px = pd.DataFrame(frames).sort_index()
    return np.log(px / px.shift(1)).dropna(how="all")


def load_hourly_panel():
    """(returns, dollar_volume) hourly panels (~50 coins, 2025+)."""
    d = pd.read_csv(BARS1H); d["time"] = pd.to_datetime(d["time"], utc=True)
    px = d.pivot_table(index="time", columns="symbol", values="close").sort_index()
    vol = d.pivot_table(index="time", columns="symbol", values="volume").sort_index()
    dollar_vol = (vol * px)
    ret = np.log(px / px.shift(1))
    return ret.dropna(how="all"), dollar_vol


def _sscores(Rw: np.ndarray, k: int) -> np.ndarray:
    """s-score per coin from a window of returns (window x n). NaN if not OU-reverting."""
    Rw = Rw - np.nanmean(Rw, axis=0)
    sd = np.nanstd(Rw, axis=0); sd[sd == 0] = 1.0
    Z = np.nan_to_num(Rw / sd)
    try:
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full(Rw.shape[1], np.nan)
    V = Vt[:k].T                       # n x k eigenvectors (eigenportfolios)
    F = Z @ V                          # window x k factor returns
    s = np.full(Rw.shape[1], np.nan)
    for i in range(Rw.shape[1]):
        y = Z[:, i]
        beta, *_ = np.linalg.lstsq(F, y, rcond=None)
        X = np.cumsum(y - F @ beta)    # cumulative residual ~ OU
        x0, x1 = X[:-1], X[1:]
        if x0.std() == 0:
            continue
        b1, b0 = np.polyfit(x0, x1, 1)
        if not (0 < b1 < 1):           # require mean reversion
            continue
        m = b0 / (1 - b1)
        zeta = x1 - (b1 * x0 + b0)
        sigeq = zeta.std() / np.sqrt(1 - b1 ** 2)
        if sigeq > 0:
            s[i] = (X[-1] - m) / sigeq
    return s


def pca_sscore_pnl(R: pd.DataFrame, window: int = 60, k: int = 3, entry: float = 1.25,
                   cost_bps: float = 5.0, rebal: int = 1) -> pd.Series:
    """Walk-forward PCA-residual mean-reversion PnL (dollar-neutral, gross exposure 1)."""
    idx = R.index; M = R.values
    n = M.shape[0]; pnl = np.zeros(n); w_prev = np.zeros(M.shape[1])
    for t in range(window, n):
        if (t - window) % rebal == 0:
            s = _sscores(M[t - window:t], k)
            raw = np.nan_to_num(np.where(np.abs(s) > entry, -s, 0.0))  # contrarian on residual
            w = raw - raw.mean()                                       # dollar-neutral
            g = np.abs(w).sum()
            w = w / g if g > 0 else np.zeros_like(w)                   # gross 1.0
            turn = np.abs(w - w_prev).sum()
        else:
            w, turn = w_prev, 0.0
        pnl[t] = np.nansum(w * np.nan_to_num(M[t])) - turn * cost_bps / 1e4
        w_prev = w
    return pd.Series(pnl, index=idx)


def coint_pairs_pnl(P: pd.DataFrame, form: int = 90, entry_z: float = 2.0, exit_z: float = 0.5,
                    adf_p: float = 0.05, cost_bps: float = 5.0, max_pairs: int = 20):
    """Engle-Granger cointegration pairs, pairs re-selected each formation window (OOS).
    Returns (pnl series, n_selected_in_sample, in_sample_pair_count)."""
    from statsmodels.tsa.stattools import adfuller, coint
    cols = list(P.columns); idx = P.index; logP = np.log(P)
    n = len(idx); pnl = np.zeros(n); is_pairs_total = 0
    pos = {}                                              # (i,j)->(side, hedge)
    step = form
    for t in range(form, n, step):
        win = logP.iloc[t - form:t]
        cand = []
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                x, y = win[cols[a]], win[cols[b]]
                if x.isna().any() or y.isna().any():
                    continue
                try:
                    _, pval, _ = coint(x, y)
                except Exception:
                    continue
                if pval < adf_p:
                    beta = np.polyfit(y, x, 1)[0]
                    cand.append((pval, cols[a], cols[b], beta))
        cand.sort(key=lambda c: c[0])
        sel = cand[:max_pairs]; is_pairs_total += len(sel)
        # trade selected pairs over the NEXT window (OOS)
        for tt in range(t, min(t + step, n)):
            day = 0.0; ntr = 0
            for _, ca, cb, beta in sel:
                sp_win = logP[ca].iloc[t - form:t] - beta * logP[cb].iloc[t - form:t]
                mu, sd = sp_win.mean(), sp_win.std()
                if sd == 0 or tt == 0:
                    continue
                z = (logP[ca].iloc[tt] - beta * logP[cb].iloc[tt] - mu) / sd
                r_sp = (P[ca].iloc[tt] / P[ca].iloc[tt - 1] - 1) - beta * (P[cb].iloc[tt] / P[cb].iloc[tt - 1] - 1)
                key = (ca, cb)
                if key in pos:
                    side = pos[key]
                    day += -side * r_sp                    # short spread if it was high
                    if abs(z) < exit_z:
                        del pos[key]; ntr += 1
                elif abs(z) > entry_z:
                    pos[key] = np.sign(z); ntr += 1
            denom = max(1, len(sel))
            pnl[tt] = day / denom - ntr * cost_bps / 1e4 / denom
    return pd.Series(pnl, index=idx), is_pairs_total


def ann_sharpe(x, ppy: float = 365.0) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[x != 0]
    if x.size < 20 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(ppy))
