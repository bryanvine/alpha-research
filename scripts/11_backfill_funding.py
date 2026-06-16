#!/usr/bin/env python3
"""
11_backfill_funding.py -- backfill Binance USD-M perpetual funding + matched spot
to 2023-01 for the top ~30 liquid USDT perps (Paper 2: Crypto Carry).

WHY THIS EXISTS
---------------
The existing funding panel under /apps/crypto-trader/data/funding_history/ (228
perps) is Hyperliquid-sourced (hourly, with a `premium` col) and, for the majors,
only reaches ~2023-06. We need a clean, cross-sectionally comparable Binance USD-M
funding cross-section spanning 2023 -> 2026 so a *decay test of the 2024 carry boom*
is possible.

DATA SOURCE -- IMPORTANT DEVIATION FROM THE BRIEF
-------------------------------------------------
The brief specified the live REST endpoints:
    GET https://fapi.binance.com/fapi/v1/fundingRate   (funding)
    GET https://api.binance.com/api/v3/klines           (spot)
Both are HTTP 451 ("restricted location") from this host -- Binance geo-blocks the
live api/fapi hosts here, and every live mirror tested (api1/api-gcp/data-api fapi,
binance.us futures) is either 451 or 404.

WORKING SOURCE: the official Binance public *data archive* on
    https://data.binance.vision/...
which is NOT geo-blocked and serves the identical Binance USD-M funding history as
monthly CSV-in-ZIP files. This is strictly better than paginating the REST endpoint
(no 429/418, deterministic, checksummed) and returns the same `last_funding_rate`
series. Layout:

  funding (8h):  data/futures/um/monthly/fundingRate/<SYM>/<SYM>-fundingRate-YYYY-MM.zip
                 -> csv cols: calc_time(ms), funding_interval_hours, last_funding_rate
  spot daily:    data/spot/monthly/klines/<SYM>/1d/<SYM>-1d-YYYY-MM.zip   (+ daily/ for tail)
                 -> csv cols: open_time, open, high, low, close, volume, close_time, ...

The current (in-progress) calendar month is published only after it closes, so the
funding tail typically ends at the last *completed* month. We fill the spot tail
from daily archives where available. We still fully span the 2024 carry boom either
way.

OUTPUTS (all under data/funding/)
  <COIN>USDT_funding.parquet   cols: time(UTC), funding_rate          (8h)
  <COIN>USDT_spot1d.parquet    cols: time(UTC), close                  (1d)
  panel_funding_8h.parquet     cols: time, symbol, funding_rate        (long)
  panel_funding_1d.parquet     cols: time(date), symbol, funding_rate  (mean daily)
  _coverage.json               per-symbol coverage summary (also -> research/profile)

Run:
  ./.venv/bin/python scripts/11_backfill_funding.py
Idempotent: re-running re-downloads and overwrites (cheap; archives are small).
"""
from __future__ import annotations

import io
import json
import time
import zipfile
import argparse
import datetime as dt
import concurrent.futures as cf
from pathlib import Path

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
OUT = Path("/apps/alpha-research/data/funding")
PROFILE = Path("/apps/alpha-research/research/paper2_data_profile.md")
HOST = "https://data.binance.vision"

START_YM = (2023, 1)                     # backfill floor
TODAY = dt.date(2026, 6, 16)             # currentDate; only used to bound the month loop
SLEEP = 0.05                             # small per-request sleep (politeness); see WORKERS
WORKERS = 6                              # symbols fetched concurrently against the archive CDN
MAX_RETRY = 5

UNIVERSE = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "TRX",
    "DOT", "MATIC", "LTC", "BCH", "NEAR", "APT", "ARB", "OP", "SUI", "INJ",
    "TIA", "SEI", "RUNE", "FIL", "ATOM", "UNI", "AAVE", "ETC", "XLM", "ALGO",
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "alpha-research/paper2-funding-backfill"})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def months(start: tuple[int, int], end_date: dt.date):
    """Yield (year, month) from start (inclusive) through end_date's month (inclusive)."""
    y, m = start
    while (y, m) <= (end_date.year, end_date.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch_zip_csv(url: str) -> pd.DataFrame | None:
    """GET a Binance-archive .zip, return its single CSV as a DataFrame.

    Returns None on a clean 404 (month/symbol simply not published). Retries
    429/418/5xx and transient network errors with exponential backoff.
    """
    backoff = 1.0
    for attempt in range(MAX_RETRY):
        try:
            r = SESSION.get(url, timeout=60)
        except requests.RequestException as e:
            if attempt == MAX_RETRY - 1:
                print(f"      ! network error (giving up): {e!r}")
                return None
            time.sleep(backoff)
            backoff *= 2
            continue

        if r.status_code == 404:
            return None
        if r.status_code == 200:
            try:
                zf = zipfile.ZipFile(io.BytesIO(r.content))
            except zipfile.BadZipFile:
                print(f"      ! bad zip at {url}")
                return None
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                # archives are sometimes header-less, sometimes headered; detect.
                head = fh.read(64)
            with zf.open(name) as fh:
                has_header = head[:1].isalpha()  # 'calc_time'/'open_time' vs a digit
                return pd.read_csv(fh, header=0 if has_header else None)
        if r.status_code in (429, 418) or r.status_code >= 500:
            wait = backoff
            ra = r.headers.get("Retry-After")
            if ra and ra.isdigit():
                wait = max(wait, float(ra))
            print(f"      ... {r.status_code}; backing off {wait:.1f}s")
            time.sleep(wait)
            backoff *= 2
            continue
        # any other status: log and stop trying this url
        print(f"      ! unexpected {r.status_code} at {url}")
        return None
    return None


def funding_url(sym: str, y: int, m: int) -> str:
    return (f"{HOST}/data/futures/um/monthly/fundingRate/{sym}/"
            f"{sym}-fundingRate-{y}-{m:02d}.zip")


def spot_month_url(sym: str, y: int, m: int) -> str:
    return f"{HOST}/data/spot/monthly/klines/{sym}/1d/{sym}-1d-{y}-{m:02d}.zip"


def spot_day_url(sym: str, d: dt.date) -> str:
    return (f"{HOST}/data/spot/daily/klines/{sym}/1d/"
            f"{sym}-1d-{d.isoformat()}.zip")


# Binance occasionally changed schemas; normalise by position when header-less.
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "n_trades",
              "taker_base", "taker_quote", "ignore"]


def normalise_funding(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] >= 3 and not isinstance(df.columns[0], str):
        df = df.iloc[:, :3]
        df.columns = FUNDING_COLS
    elif "calc_time" not in df.columns:
        # headered but unexpected -> take first 3 positionally
        df = df.iloc[:, :3]
        df.columns = FUNDING_COLS
    out = pd.DataFrame({
        "time": pd.to_datetime(df["calc_time"].astype("int64"), unit="ms", utc=True),
        "funding_rate": pd.to_numeric(df["last_funding_rate"], errors="coerce"),
    })
    return out.dropna(subset=["funding_rate"])


def normalise_kline(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns[0], str):
        df.columns = KLINE_COLS[: df.shape[1]]
    # Some recent monthly kline files ship microsecond open_time; detect by magnitude.
    ot = df["open_time"].astype("int64")
    unit = "us" if ot.iloc[0] > 10_000_000_000_000 else "ms"
    out = pd.DataFrame({
        "time": pd.to_datetime(ot, unit=unit, utc=True),
        "close": pd.to_numeric(df["close"], errors="coerce"),
    })
    return out.dropna(subset=["close"])


# ---------------------------------------------------------------------------
# per-symbol backfill
# ---------------------------------------------------------------------------
def backfill_funding(coin: str) -> pd.DataFrame:
    sym = f"{coin}USDT"
    frames, first_seen = [], False
    for y, m in months(START_YM, TODAY):
        df = fetch_zip_csv(funding_url(sym, y, m))
        time.sleep(SLEEP)
        if df is None:
            # before listing: keep probing (some perps list mid-2023+).
            # after listing: a single missing recent month = not yet published -> stop.
            if first_seen and (y, m) >= (TODAY.year, TODAY.month - 1 if TODAY.month > 1 else 12):
                break
            continue
        first_seen = True
        frames.append(normalise_funding(df))
    if not frames:
        return pd.DataFrame(columns=["time", "funding_rate"])
    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=["time"])
             .sort_values("time")
             .reset_index(drop=True))
    return out


def backfill_spot(coin: str, fund_end: pd.Timestamp | None) -> pd.DataFrame:
    sym = f"{coin}USDT"
    frames, first_seen = [], False
    for y, m in months(START_YM, TODAY):
        df = fetch_zip_csv(spot_month_url(sym, y, m))
        time.sleep(SLEEP)
        if df is None:
            continue
        first_seen = True
        frames.append(normalise_kline(df))
    # tail: daily archives for the current (unpublished-as-monthly) month
    cur_first = dt.date(TODAY.year, TODAY.month, 1)
    d = cur_first
    while d <= TODAY:
        df = fetch_zip_csv(spot_day_url(sym, d))
        time.sleep(SLEEP)
        if df is not None:
            frames.append(normalise_kline(df))
        d += dt.timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["time", "close"])
    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=["time"])
             .sort_values("time")
             .reset_index(drop=True))
    return out


def expected_8h_obs(start: pd.Timestamp, end: pd.Timestamp) -> int:
    span_h = (end - start).total_seconds() / 3600.0
    return int(round(span_h / 8.0)) + 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def process_symbol(idx_coin):
    """Backfill funding + spot for one coin. Returns (coverage_dict, panel_frame_or_None)."""
    i, coin = idx_coin
    sym = f"{coin}USDT"
    print(f"[{i:>2}/{len(UNIVERSE)}] {sym} ...", flush=True)

    fund = backfill_funding(coin)
    if fund.empty:
        print(f"      !! no funding archive for {sym} -- UNAVAILABLE", flush=True)
        return sym, {"available": False}, None

    fund.to_parquet(OUT / f"{sym}_funding.parquet", index=False)
    f_start, f_end = fund["time"].iloc[0], fund["time"].iloc[-1]

    spot = backfill_spot(coin, f_end)
    if not spot.empty:
        spot.to_parquet(OUT / f"{sym}_spot1d.parquet", index=False)
        s_start = spot["time"].iloc[0].date().isoformat()
        s_end = spot["time"].iloc[-1].date().isoformat()
        s_n = len(spot)
    else:
        s_start = s_end = None
        s_n = 0
        print(f"      ~ no spot klines for {sym} (perp without matched spot pair?)", flush=True)

    # gap diagnostics on the 8h funding series
    exp = expected_8h_obs(f_start, f_end)
    n = len(fund)
    missing = max(0, exp - n)
    # count distinct large gaps (>1 funding interval) as a coarse gap signal
    deltas = fund["time"].diff().dropna().dt.total_seconds() / 3600.0
    big_gaps = int((deltas > 8.5).sum())     # >~8h interval (allow small drift)
    max_gap_h = float(deltas.max()) if len(deltas) else 0.0

    cov = {
        "available": True,
        "funding_earliest": f_start.isoformat(),
        "funding_latest": f_end.isoformat(),
        "funding_n_obs": int(n),
        "funding_expected_8h": int(exp),
        "funding_missing_est": int(missing),
        "funding_n_gaps_gt8h": big_gaps,
        "funding_max_gap_hours": round(max_gap_h, 1),
        "spot_earliest": s_start,
        "spot_latest": s_end,
        "spot_n_obs": int(s_n),
    }
    print(f"      {sym}: funding {f_start.date()} -> {f_end.date()}  n={n} "
          f"(exp~{exp}, miss~{missing}, gaps>8h={big_gaps})  spot n={s_n}", flush=True)

    fr = fund.copy()
    fr["symbol"] = sym
    return sym, cov, fr[["time", "symbol", "funding_rate"]]


def main(workers: int = WORKERS):
    OUT.mkdir(parents=True, exist_ok=True)
    coverage = {}
    panel_rows = []
    unavailable = []

    tasks = list(enumerate(UNIVERSE, 1))
    if workers <= 1:
        results = [process_symbol(t) for t in tasks]
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(process_symbol, tasks))

    for sym, cov, fr in results:
        coverage[sym] = cov
        if not cov.get("available"):
            unavailable.append(sym)
        if fr is not None:
            panel_rows.append(fr)

    # ----- merged panels -----
    if panel_rows:
        panel8h = (pd.concat(panel_rows, ignore_index=True)
                     .sort_values(["time", "symbol"])
                     .reset_index(drop=True))
        panel8h.to_parquet(OUT / "panel_funding_8h.parquet", index=False)

        d1 = panel8h.copy()
        d1["date"] = d1["time"].dt.floor("D")
        panel1d = (d1.groupby(["date", "symbol"], as_index=False)["funding_rate"]
                     .mean()
                     .rename(columns={"date": "time"})
                     .sort_values(["time", "symbol"])
                     .reset_index(drop=True))
        panel1d.to_parquet(OUT / "panel_funding_1d.parquet", index=False)
        print(f"\npanel_funding_8h: {len(panel8h):,} rows, "
              f"{panel8h['symbol'].nunique()} symbols")
        print(f"panel_funding_1d: {len(panel1d):,} rows")
        total_obs = len(panel8h)
        n_sym = panel8h["symbol"].nunique()
    else:
        total_obs = 0
        n_sym = 0

    # ----- coverage json (consumed by the profile writer) -----
    earliest_dates = sorted(
        v["funding_earliest"][:10] for v in coverage.values() if v.get("available")
    )
    median_earliest = (earliest_dates[len(earliest_dates) // 2]
                       if earliest_dates else None)
    summary = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": f"{HOST} (Binance USD-M public data archive; live api/fapi geo-blocked 451)",
        "n_symbols_requested": len(UNIVERSE),
        "n_symbols_backfilled": n_sym,
        "unavailable": unavailable,
        "median_earliest_date": median_earliest,
        "total_funding_obs_8h": total_obs,
        "per_symbol": coverage,
    }
    (OUT / "_coverage.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'_coverage.json'}")
    print(f"symbols backfilled: {n_sym}/{len(UNIVERSE)}  "
          f"median earliest: {median_earliest}  total 8h obs: {total_obs:,}")
    if unavailable:
        print(f"unavailable: {unavailable}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="concurrent symbols fetched against the archive CDN")
    args = ap.parse_args()
    main(workers=args.workers)
