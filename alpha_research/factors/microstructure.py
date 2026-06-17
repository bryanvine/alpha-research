"""LOB microstructure for liquidity-provision analysis (Paper 5).

Built on raw 10-level SPY/QQQ snapshots (100ms, Nov-Dec 2025). Provides order-flow
imbalance (OFI; Cont-Kukanov-Stoikov), an imbalance-tilted micro-price (Stoikov), and
the primitives for the fill-model-illusion and execution-overlay tests. Regular-hours
only (14:30-21:00 UTC) with crossed/auction/wide quotes filtered out.
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd

LOB_DIR = "/apps/trading-system/data/training"
L1 = ["time", "symbol", "bid_price_1", "bid_size_1", "ask_price_1", "ask_size_1"]


def day_files():
    return sorted(glob.glob(f"{LOB_DIR}/lob_*.parquet"))


def load_day(path, sym="SPY", max_spread_bps=50.0):
    df = pd.read_parquet(path, columns=L1)
    df = df[df["symbol"] == sym]
    if df.empty:
        return None
    t = pd.to_datetime(df["time"], utc=True)
    h = t.dt.hour + t.dt.minute / 60.0
    b = df["bid_price_1"].to_numpy(float); a = df["ask_price_1"].to_numpy(float)
    bs = df["bid_size_1"].to_numpy(float); as_ = df["ask_size_1"].to_numpy(float)
    mid = (a + b) / 2.0
    ok = ((h.to_numpy() >= 14.5) & (h.to_numpy() <= 21.0) & (a > b) & (b > 0)
          & np.isfinite(bs) & np.isfinite(as_) & (bs > 0) & (as_ > 0)
          & ((a - b) / mid * 1e4 < max_spread_bps))
    if ok.sum() < 5000:
        return None
    return dict(b=b[ok], a=a[ok], bs=bs[ok], as_=as_[ok])


def features(d):
    b, a, bs, as_ = d["b"], d["a"], d["bs"], d["as_"]
    mid = (a + b) / 2.0
    spr = a - b
    imb = bs / (bs + as_)
    micro = mid + (imb - 0.5) * spr                       # Stoikov-style imbalance-tilted price
    db = np.diff(b); da = np.diff(a)
    bid_c = np.where(db > 0, bs[1:], np.where(db < 0, -bs[:-1], bs[1:] - bs[:-1]))
    ask_c = np.where(da < 0, as_[1:], np.where(da > 0, -as_[:-1], as_[1:] - as_[:-1]))
    ofi = np.concatenate([[0.0], bid_c - ask_c])
    return dict(b=b, a=a, mid=mid, spr=spr, imb=imb, micro=micro, ofi=ofi)
