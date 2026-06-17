#!/usr/bin/env python3
"""
20_fetch_eia.py -- fetch EIA energy-futures term structure for Paper 6 (commodity roll yield).

Front-through-4th-contract daily settlement prices for WTI, Henry Hub natural gas, NY Harbor
No.2 heating oil, and RBOB gasoline -- a genuine multi-contract energy curve for roll-yield
and basis-momentum factors (the binding gap that free continuous-futures feeds could not fill).

Reads EIA_API_KEY from /apps/alpha-research/.env (gitignored). Output -> data/commodity/eia_energy.parquet.
"""
import os, sys, json, time, urllib.request
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = None
for line in open(f"{ROOT}/.env"):
    if line.strip().startswith("EIA_API_KEY="):
        KEY = line.strip().split("=", 1)[1]
assert KEY, "EIA_API_KEY not found in .env"

SERIES = {
    "WTI":        ["PET.RCLC1.D", "PET.RCLC2.D", "PET.RCLC3.D", "PET.RCLC4.D"],
    "NatGas":     ["NG.RNGC1.D", "NG.RNGC2.D", "NG.RNGC3.D", "NG.RNGC4.D"],
    "HeatingOil": ["PET.EER_EPD2F_PE1_Y35NY_DPG.D", "PET.EER_EPD2F_PE2_Y35NY_DPG.D",
                   "PET.EER_EPD2F_PE3_Y35NY_DPG.D", "PET.EER_EPD2F_PE4_Y35NY_DPG.D"],
    "RBOB":       ["PET.EER_EPMRR_PE1_Y35NY_DPG.D", "PET.EER_EPMRR_PE2_Y35NY_DPG.D",
                   "PET.EER_EPMRR_PE3_Y35NY_DPG.D", "PET.EER_EPMRR_PE4_Y35NY_DPG.D"],
}


def fetch(sid):
    url = f"https://api.eia.gov/v2/seriesid/{sid}?api_key={KEY}"
    with urllib.request.urlopen(url, timeout=90) as r:
        d = json.load(r)
    rows = d.get("response", {}).get("data", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    vcol = "value" if "value" in df.columns else [c for c in df.columns if c not in ("period",)][0]
    out = df[["period", vcol]].rename(columns={"period": "date", vcol: "price"})
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna().sort_values("date")


def main():
    frames, report = [], {}
    for comm, sids in SERIES.items():
        for k, sid in enumerate(sids, 1):
            try:
                df = fetch(sid)
                time.sleep(0.3)
                if df is None or len(df) < 100:
                    report[f"{comm}_C{k}"] = f"empty/short ({sid})"; continue
                df["commodity"] = comm; df["contract"] = k
                frames.append(df)
                report[f"{comm}_C{k}"] = f"{len(df)} rows {df['date'].min().date()}..{df['date'].max().date()}"
            except Exception as e:
                report[f"{comm}_C{k}"] = f"ERROR {e}"
    if not frames:
        print("NO DATA fetched:", report); sys.exit(1)
    panel = pd.concat(frames, ignore_index=True)
    os.makedirs(f"{ROOT}/data/commodity", exist_ok=True)
    panel.to_parquet(f"{ROOT}/data/commodity/eia_energy.parquet")
    print("EIA energy futures fetched (contract 1-4 daily):")
    for k, v in report.items():
        print(f"  {k:14s} {v}")
    print(f"\nTotal {len(panel):,} rows -> data/commodity/eia_energy.parquet")
    # quick roll-yield sanity (front vs second, latest)
    for comm in SERIES:
        sub = panel[panel.commodity == comm]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="date", columns="contract", values="price").dropna()
        if {1, 2}.issubset(piv.columns) and len(piv):
            ry = ((piv[1] / piv[2] - 1) * 12 * 100).iloc[-60:].mean()  # ~annualized front/second roll, last 60d
            print(f"  {comm}: recent annualized roll (C1/C2-1)x12 = {ry:+.1f}%  "
                  f"({'backwardation' if ry > 0 else 'contango'})")


if __name__ == "__main__":
    main()
