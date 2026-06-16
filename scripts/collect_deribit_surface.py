#!/usr/bin/env python3
"""
collect_deribit_surface.py -- forward Deribit option-surface collector.

Appends a timestamped snapshot of the FULL BTC & ETH option book summary (bid/ask
price, mark_iv, underlying, open interest) so a future Paper-1 v2 can backtest the
true options-execution leg (historical bid-ask spreads by moneyness/tenor) that the
single end-of-sample snapshot cannot support. Run on a schedule (e.g. cron every 6h):

  0 */6 * * * /apps/alpha-research/.venv/bin/python /apps/alpha-research/scripts/collect_deribit_surface.py \
              >> /apps/alpha-research/data/deribit_surface/collector.log 2>&1

Output: one parquet per run under data/deribit_surface/ (gitignored). Public API, no auth.
"""
import os, json, urllib.request, datetime
import pandas as pd

OUT = "/apps/alpha-research/data/deribit_surface"
API = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={}&kind=option"


def fetch(cur):
    with urllib.request.urlopen(API.format(cur), timeout=30) as r:
        return json.load(r)["result"]


def main():
    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for cur in ("BTC", "ETH"):
        for o in fetch(cur):
            o["_currency"] = cur
            rows.append(o)
    df = pd.DataFrame(rows)
    df["_snapshot"] = stamp
    path = f"{OUT}/surface_{stamp}.parquet"
    df.to_parquet(path)
    print(f"[{stamp}] wrote {len(df)} option rows ({(df['_currency']=='BTC').sum()} BTC / "
          f"{(df['_currency']=='ETH').sum()} ETH) -> {path}")


if __name__ == "__main__":
    main()
