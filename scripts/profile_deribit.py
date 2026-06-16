#!/usr/bin/env python3
"""Profile Deribit raw data for Paper 1 data foundation.

Profiles dvol/price parquet files and surface JSON files at
/apps/jepa-trader/data/raw_deribit/. Reports columns, dtypes, time span,
cadence, gaps, value sanity, and whether the option surface is a single
snapshot or a time series. Pure stdlib + pandas + pyarrow.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("/apps/jepa-trader/data/raw_deribit")


def find_time_col(df):
    for c in df.columns:
        lc = c.lower()
        if lc in ("timestamp", "time", "ts", "date", "datetime", "t"):
            return c
    # fall back: any datetime-typed column
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def to_datetime_series(s):
    """Best-effort conversion of a column to tz-naive UTC datetimes."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    # numeric epoch?
    if pd.api.types.is_numeric_dtype(s):
        v = float(s.dropna().iloc[0])
        # ms vs s vs ns heuristics
        if v > 1e17:
            unit = "ns"
        elif v > 1e14:
            unit = "us"
        elif v > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(s, unit=unit)
    return pd.to_datetime(s, errors="coerce")


def profile_parquet(path):
    print(f"\n{'='*70}\nFILE: {path.name}  ({path.stat().st_size/1024:.0f} KiB)\n{'='*70}")
    df = pd.read_parquet(path)
    print(f"rows={len(df)}  cols={list(df.columns)}")
    print("dtypes:")
    for c in df.columns:
        print(f"  {c:20s} {df[c].dtype}")

    # If a datetime index exists, surface it
    idx_is_dt = pd.api.types.is_datetime64_any_dtype(df.index)
    if idx_is_dt:
        print(f"INDEX is datetime: {df.index.min()} .. {df.index.max()}")
        tcol_series = pd.Series(df.index)
        tname = "<index>"
    else:
        tcol = find_time_col(df)
        if tcol is None:
            print("!! no time column found")
            return
        tname = tcol
        tcol_series = to_datetime_series(df[tcol])

    tcol_series = tcol_series.sort_values().reset_index(drop=True)
    print(f"\nTIME col: {tname}")
    print(f"  min={tcol_series.min()}   max={tcol_series.max()}")
    deltas = tcol_series.diff().dropna()
    if len(deltas):
        med = deltas.median()
        print(f"  median delta = {med}  (mode={deltas.mode().iloc[0] if len(deltas.mode()) else 'n/a'})")
        # gaps = deltas > 1.5x median
        thr = med * 1.5
        gaps = deltas[deltas > thr]
        print(f"  gap count (delta > 1.5x median = {thr}): {len(gaps)}")
        if len(gaps):
            big = deltas.nlargest(5)
            print(f"  largest 5 deltas: {[str(x) for x in big.tolist()]}")
        dup = (deltas == pd.Timedelta(0)).sum()
        print(f"  zero-delta (dup ts) count: {dup}")
        span_days = (tcol_series.max() - tcol_series.min()).total_seconds() / 86400
        print(f"  span = {span_days:.1f} days; expected rows @ median ~ {span_days*86400/med.total_seconds():.0f}")

    # value sanity on numeric cols
    print("\nVALUE SANITY (numeric cols):")
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c]):
            s = df[c].astype(float)
            print(f"  {c:20s} min={s.min():.4g} p50={s.median():.4g} max={s.max():.4g} "
                  f"nan={s.isna().sum()} nonpos={(s<=0).sum()}")


def profile_surface(path):
    print(f"\n{'='*70}\nSURFACE: {path.name}  ({path.stat().st_size/1024:.0f} KiB)\n{'='*70}")
    with open(path) as f:
        obj = json.load(f)
    print(f"top-level type: {type(obj).__name__}")

    # Normalize to a list of contract dicts + capture any top-level snapshot meta
    contracts = None
    meta = {}
    if isinstance(obj, dict):
        print(f"top-level keys: {list(obj.keys())[:30]}")
        # common shapes: {'timestamp':..., 'contracts':[...]} or {'result':[...]}
        for k in ("contracts", "result", "data", "options", "surface"):
            if k in obj and isinstance(obj[k], list):
                contracts = obj[k]
                break
        # capture scalar meta
        for k, v in obj.items():
            if not isinstance(v, (list, dict)):
                meta[k] = v
        if contracts is None:
            # maybe dict keyed by instrument -> dict
            vals = list(obj.values())
            if vals and isinstance(vals[0], dict):
                contracts = vals
    elif isinstance(obj, list):
        contracts = obj

    if meta:
        print(f"top-level scalar meta: {meta}")
    if contracts is None:
        print("!! could not locate contract list; raw sample:")
        print(json.dumps(obj, indent=2)[:1500])
        return

    print(f"number of contracts: {len(contracts)}")
    if not contracts:
        return
    sample = contracts[0]
    if isinstance(sample, dict):
        print(f"contract fields ({len(sample)}): {list(sample.keys())}")
        print("sample contract:")
        print(json.dumps(sample, indent=2)[:1200])

    # Try to extract timestamps to determine snapshot vs time-series
    df = pd.DataFrame(contracts)
    ts_candidates = [c for c in df.columns if any(
        k in c.lower() for k in ("timestamp", "time", "snapshot", "creation", "asof", "as_of", "date"))]
    print(f"\ntimestamp-like fields: {ts_candidates}")
    for c in ts_candidates:
        try:
            uniq = df[c].nunique()
            print(f"  {c}: n_unique={uniq}  sample={df[c].dropna().unique()[:5].tolist()}")
            conv = to_datetime_series(df[c])
            print(f"      -> as datetime: {conv.min()} .. {conv.max()}  ({uniq} distinct)")
        except Exception as e:
            print(f"  {c}: (could not parse: {e})")

    # expiry / strike / moneyness ranges
    for key in ("expiry", "expiration", "expiration_timestamp", "expiry_timestamp", "exp"):
        cols = [c for c in df.columns if key in c.lower()]
        for c in cols:
            try:
                conv = to_datetime_series(df[c])
                print(f"EXPIRY[{c}]: {conv.min()} .. {conv.max()}  n_unique={df[c].nunique()}")
            except Exception:
                print(f"EXPIRY[{c}]: distinct={sorted(df[c].dropna().unique())[:10]} ...")
            break
    for key in ("strike",):
        cols = [c for c in df.columns if key in c.lower()]
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce")
            print(f"STRIKE[{c}]: min={s.min()} max={s.max()} n_unique={s.nunique()}")
    for key in ("moneyness", "mny", "delta", "iv", "mark_iv", "bid_iv", "ask_iv"):
        cols = [c for c in df.columns if key == c.lower() or key in c.lower()]
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                print(f"FIELD[{c}]: min={s.min():.4g} p50={s.median():.4g} max={s.max():.4g} nan={s.isna().sum()}")


def main():
    print("DERIBIT DATA PROFILE")
    print(f"dir: {RAW}")
    for name in ["dvol_BTC.parquet", "dvol_ETH.parquet", "price_BTC.parquet", "price_ETH.parquet"]:
        p = RAW / name
        if p.exists():
            profile_parquet(p)
        else:
            print(f"MISSING: {p}")
    for name in ["surface_BTC.json", "surface_ETH.json"]:
        p = RAW / name
        if p.exists():
            profile_surface(p)
        else:
            print(f"MISSING: {p}")
    # summary.json
    sp = RAW / "summary.json"
    if sp.exists():
        print(f"\n{'='*70}\nsummary.json\n{'='*70}")
        with open(sp) as f:
            print(json.dumps(json.load(f), indent=2))


if __name__ == "__main__":
    main()
