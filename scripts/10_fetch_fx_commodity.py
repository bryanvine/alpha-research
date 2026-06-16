#!/usr/bin/env python3
"""
10_fetch_fx_commodity.py -- Paper 6 (FX Carry & Commodity Roll Yield) data sourcing.

Acquires FREE FX + commodity data, normalizes it to documented conventions, writes
raw CSVs under data/fx/ and data/commodity/, builds normalized panels, and emits a
machine-readable manifest the registry + profile are built from.

ALL HTTP IS DONE WITH `requests` (the bash sandbox on this host has no network; only
Python egress works -- mirrors the Paper-1 fetch approach). Nothing here needs an API
key. Re-running is idempotent (files are overwritten).

------------------------------------------------------------------------------------
FX
------------------------------------------------------------------------------------
SPOT (FRED daily USD bilateral rates). FRED publishes different pairs in DIFFERENT
USD directions:
   * "USD per 1 foreign"  : DEXUSEU(EUR), DEXUSUK(GBP), DEXUSAL(AUD), DEXUSNZ(NZD)
   * "foreign per 1 USD"  : DEXJPUS(JPY), DEXCAUS(CAD), DEXSZUS(CHF), DEXSDUS(SEK),
                            DEXNOUS(NOK)
We NORMALIZE EVERY SERIES TO **foreign currency units per 1 USD** ("FX_per_USD").
   -> for the "USD per foreign" quotes we store 1/rate.
   -> a RISE in the normalized series = USD APPRECIATES / foreign currency DEPRECIATES.
Each raw file also keeps the untouched native column so the convention is auditable.

CARRY (3M interest-rate differential vs USD, in percentage points):
   USD leg : DGS3MO (3M Treasury CMT).                       [primary USD short rate]
   Foreign : OECD 3M interbank IR3TIB01<CC>M156N (MONTHLY) on FRED, where <CC> is
             EZ/JP/GB/CA/AU/CH/SE/NO/NZ. We also pull IR3TIB01USM156N so a fully
             "OECD-consistent" differential is available too.
   diff_i = rate_foreign_i - rate_USD  (positive => foreign pays more => long-carry).
   NOTE: foreign legs are MONTHLY -> carry is a monthly panel; spot/value handled
   separately. EUR uses the euro-area aggregate (EZ).

VALUE (BIS real effective exchange rate, REER -- monthly, index 2020=100 on FRED):
   Broad basket  RB<CC>BIS (from 1994): US,JP,GB,CA,AU,CH,SE,NO,NZ,XM(euro area).
   We use the BROAD index as primary (wider, modern basket) and ALSO pull the NARROW
   index RN<CC>BIS (from 1964) for the long-history robustness check. EUR uses XM.
   A higher REER = currency relatively EXPENSIVE in real terms (value signal: short).

POSITIONING (optional bonus) -- CFTC Commitments of Traders, legacy futures-only,
   Socrata public API (no key): non-commercial (speculator) net length as a sentiment
   overlay. Weekly. Market names were renamed by CFTC ~2022-02, so each market stitches
   an OLD + NEW name where needed.

------------------------------------------------------------------------------------
COMMODITIES
------------------------------------------------------------------------------------
FRONT-MONTH / SPOT (FRED daily, no key):
   WTI  DCOILWTICO, Brent DCOILBRENTEU, Henry Hub gas DHHNGSP (all $/bbl or $/MMBtu).
MONTHLY IMF prices (FRED) for breadth: copper PCOPPUSDM, maize PMAIZMTUSDM,
   wheat PWHEAMTUSDM, EU nat-gas PNGASEUUSDM, all-commodity index PALLFNFINDEXM,
   Brent monthly POILBREUSDM.

ROLL YIELD / TERM STRUCTURE -- *** THE BINDING CONSTRAINT FOR PAPER 6 *** :
   Roll yield needs FRONT + SECOND contract (or a spot-vs-excess-return pair). The free
   continuous-futures feeds are dead/blocked:
     * Nasdaq Data Link CHRIS/CME_CL1 & CL2  -> HTTP 403 (Quandl decommissioned the free
       CHRIS continuous tier; robots NOINDEX wall). Verified this run.
     * Stooq front-month CSV (cl.f, gc.f, ng.f, ...) -> JS proof-of-work anti-bot wall
       (same wall the Paper-1 work hit). Verified this run.
     * BCOM/GSCI total-return vs spot on FRED (BCOMTR/SPGSCITR/DJUBSTR) -> HTTP 404.
     * EIA open-data v2 -> 403 API_KEY_MISSING (needs a free key; out of scope here).
   => With only FRED front-month spot, ROLL YIELD IS NOT DIRECTLY COMPUTABLE from free
      sources. We DOCUMENT this as Blocker #1 and provide what IS free instead.

GOLD PRICE: FRED's $/oz London fix series (GOLDAMGBD228NLBM, GOLDPMGBD228NLBM,
   PGOLDUSDM, ...) all 404 (discontinued). No free FRED $/oz gold remains -> Blocker #2.
   We capture GOLD futures POSITIONING via COT, and note the price gap.
"""
import os
import sys
import io
import json
import time
import csv as _csv
import datetime as dt

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FX_DIR = os.path.join(ROOT, "data", "fx")
CM_DIR = os.path.join(ROOT, "data", "commodity")
MANIFEST = os.path.join(ROOT, "data", "_paper6_manifest.json")

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) alpha-research/paper6"}
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
COT_BASE = "https://publicreporting.cftc.gov/resource/6dca-aqww.csv"

os.makedirs(FX_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

# Accumulates per-series provenance for the manifest the profile/registry read.
MANI = {"generated": dt.datetime.utcnow().isoformat() + "Z", "fx": {}, "commodity": {}, "failures": []}


# ----------------------------------------------------------------------------------
# Low-level fetch helpers
# ----------------------------------------------------------------------------------
def http_get(url, params=None, tries=3, timeout=45):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, params=params, timeout=timeout)
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise last


def fred_csv(series_id):
    """Return a clean DataFrame[date, <id>] for a FRED series, or None on 404/garbage.

    FRED returns an HTML page (not CSV) for an unknown id; the valid CSV header always
    starts with 'observation_date'. '.' is FRED's missing-value sentinel.
    """
    r = http_get(FRED.format(id=series_id))
    txt = r.text
    head = txt.splitlines()[:1]
    if r.status_code != 200 or not head or not head[0].lower().startswith("observation_date"):
        return None
    df = pd.read_csv(io.StringIO(txt), na_values=["."])
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def span_of(df, col="date"):
    if df is None or len(df) == 0:
        return (None, None, 0)
    s = df[col]
    return (str(s.iloc[0].date()), str(s.iloc[-1].date()), int(len(df)))


def save_csv(df, path):
    df.to_csv(path, index=False)
    return os.path.relpath(path, ROOT)


# ==================================================================================
# FX SPOT
# ==================================================================================
# id -> (currency, native_convention). native 'USD_per_FX' means rate = USD per 1 unit
# of foreign; 'FX_per_USD' means rate = foreign units per 1 USD.
FX_SPOT_IDS = {
    "DEXUSEU": ("EUR", "USD_per_FX"),
    "DEXJPUS": ("JPY", "FX_per_USD"),
    "DEXUSUK": ("GBP", "USD_per_FX"),
    "DEXCAUS": ("CAD", "FX_per_USD"),
    "DEXUSAL": ("AUD", "USD_per_FX"),
    "DEXSZUS": ("CHF", "FX_per_USD"),
    "DEXSDUS": ("SEK", "FX_per_USD"),
    "DEXNOUS": ("NOK", "FX_per_USD"),
    "DEXUSNZ": ("NZD", "USD_per_FX"),
}


def fetch_fx_spot():
    print("\n=== FX SPOT (FRED daily) -> normalize to FX-per-USD ===")
    panel = {}
    for sid, (ccy, conv) in FX_SPOT_IDS.items():
        df = fred_csv(sid)
        if df is None:
            print(f"  [FAIL] {sid} ({ccy})")
            MANI["failures"].append({"kind": "fx_spot", "id": sid, "ccy": ccy})
            continue
        native = df[sid].astype(float)
        if conv == "USD_per_FX":
            norm = 1.0 / native          # -> foreign per USD
        else:
            norm = native                # already foreign per USD
        out = pd.DataFrame({"date": df["date"], f"{ccy}_native": native, f"{ccy}_per_USD": norm})
        rel = save_csv(out, os.path.join(FX_DIR, f"spot_{ccy}.csv"))
        a, b, n = span_of(out)
        print(f"  [ ok ] {sid:8s} {ccy}  native={conv:11s} n={n:<6d} {a}..{b}  -> {rel}")
        MANI["fx"][f"spot_{ccy}"] = {
            "file": rel, "fred_id": sid, "ccy": ccy, "native_convention": conv,
            "normalized": "FX_per_USD", "start": a, "end": b, "rows": n,
        }
        panel[ccy] = out.set_index("date")[f"{ccy}_per_USD"]
    if panel:
        wide = pd.DataFrame(panel).sort_index()
        wide.index.name = "date"
        rel = save_csv(wide.reset_index(), os.path.join(FX_DIR, "panel_spot_FXperUSD.csv"))
        a, b, n = (str(wide.index[0].date()), str(wide.index[-1].date()), len(wide))
        print(f"  [PANEL] spot FX-per-USD  cols={list(wide.columns)}  n={n}  {a}..{b}  -> {rel}")
        MANI["fx"]["panel_spot"] = {"file": rel, "cols": list(wide.columns), "start": a, "end": b,
                                    "rows": n, "convention": "FX_per_USD (rise=USD up)"}
    return panel


# ==================================================================================
# FX CARRY (3M rate differential vs USD)
# ==================================================================================
USD_SHORT = {"DGS3MO": "USD 3M Treasury CMT (daily)"}
# OECD 3M interbank rate (monthly), FRED id IR3TIB01<CC>M156N
OECD_3M = {
    "EUR": "EZ", "JPY": "JP", "GBP": "GB", "CAD": "CA", "AUD": "AU",
    "CHF": "CH", "SEK": "SE", "NOK": "NO", "NZD": "NZ", "USD": "US",
}


def fetch_fx_carry():
    print("\n=== FX CARRY (3M rate differential vs USD) ===")
    # USD daily short rate (primary)
    usd = fred_csv("DGS3MO")
    if usd is not None:
        rel = save_csv(usd.rename(columns={"DGS3MO": "USD_3M"}), os.path.join(FX_DIR, "rate_USD_DGS3MO.csv"))
        a, b, n = span_of(usd)
        print(f"  [ ok ] DGS3MO USD 3M (daily) n={n} {a}..{b} -> {rel}")
        MANI["fx"]["rate_USD_DGS3MO"] = {"file": rel, "fred_id": "DGS3MO", "freq": "daily",
                                         "units": "percent", "start": a, "end": b, "rows": n}
    else:
        MANI["failures"].append({"kind": "usd_rate", "id": "DGS3MO"})

    # OECD 3M interbank (monthly) for each currency incl. USD
    rates = {}
    for ccy, cc in OECD_3M.items():
        sid = f"IR3TIB01{cc}M156N"
        df = fred_csv(sid)
        if df is None:
            print(f"  [FAIL] {sid} ({ccy})")
            MANI["failures"].append({"kind": "oecd_3m", "id": sid, "ccy": ccy})
            continue
        col = f"{ccy}_3M"
        df = df.rename(columns={sid: col})
        rel = save_csv(df, os.path.join(FX_DIR, f"rate_{ccy}_OECD3M.csv"))
        a, b, n = span_of(df)
        print(f"  [ ok ] {sid:16s} {ccy} OECD 3M IB (monthly) n={n:<4d} {a}..{b}")
        MANI["fx"][f"rate_{ccy}_OECD3M"] = {"file": rel, "fred_id": sid, "freq": "monthly",
                                            "units": "percent", "start": a, "end": b, "rows": n}
        rates[ccy] = df.set_index("date")[col]

    # Build monthly differential panel: foreign_OECD3M - USD_OECD3M (OECD-consistent)
    if "USD" in rates:
        usd_m = rates["USD"]
        diffs = {}
        for ccy, s in rates.items():
            if ccy == "USD":
                continue
            d = (s - usd_m).dropna()
            diffs[ccy] = d
        wide = pd.DataFrame(diffs).sort_index()
        wide.index.name = "date"
        rel = save_csv(wide.reset_index(), os.path.join(FX_DIR, "panel_carry_diff_OECD3M_vs_USD.csv"))
        a, b, n = (str(wide.index[0].date()), str(wide.index[-1].date()), len(wide))
        print(f"  [PANEL] carry diff (OECD 3M, foreign-USD)  cols={list(wide.columns)}  n={n}  {a}..{b}")
        MANI["fx"]["panel_carry"] = {
            "file": rel, "cols": list(wide.columns), "start": a, "end": b, "rows": n,
            "definition": "rate_foreign_OECD3M - rate_USD_OECD3M (pct pts); +ve => long-carry",
            "freq": "monthly",
        }


# ==================================================================================
# FX VALUE (BIS REER)
# ==================================================================================
REER_CC = {"USD": "US", "JPY": "JP", "GBP": "GB", "CAD": "CA", "AUD": "AU",
           "CHF": "CH", "SEK": "SE", "NOK": "NO", "NZD": "NZ", "EUR": "XM"}


def fetch_fx_value():
    print("\n=== FX VALUE (BIS REER on FRED) ===")
    for label, prefix, freq_note in [("broad", "RB", "broad basket, from 1994"),
                                     ("narrow", "RN", "narrow basket, from 1964")]:
        panel = {}
        for ccy, cc in REER_CC.items():
            sid = f"{prefix}{cc}BIS"
            df = fred_csv(sid)
            if df is None:
                print(f"  [FAIL] {sid} ({ccy}, {label})")
                MANI["failures"].append({"kind": f"reer_{label}", "id": sid, "ccy": ccy})
                continue
            col = f"{ccy}_REER"
            df = df.rename(columns={sid: col})
            rel = save_csv(df, os.path.join(FX_DIR, f"reer_{label}_{ccy}.csv"))
            a, b, n = span_of(df)
            MANI["fx"][f"reer_{label}_{ccy}"] = {"file": rel, "fred_id": sid, "basket": label,
                                                 "units": "index 2020=100", "start": a, "end": b, "rows": n}
            panel[ccy] = df.set_index("date")[col]
        if panel:
            wide = pd.DataFrame(panel).sort_index()
            wide.index.name = "date"
            rel = save_csv(wide.reset_index(), os.path.join(FX_DIR, f"panel_reer_{label}.csv"))
            a, b, n = (str(wide.index[0].date()), str(wide.index[-1].date()), len(wide))
            print(f"  [PANEL] REER {label} ({freq_note})  cols={list(wide.columns)}  n={n}  {a}..{b}")
            MANI["fx"][f"panel_reer_{label}"] = {"file": rel, "cols": list(wide.columns),
                                                 "start": a, "end": b, "rows": n,
                                                 "note": f"{freq_note}; higher = real-expensive"}


# ==================================================================================
# COMMODITY PRICES (FRED)
# ==================================================================================
CM_DAILY = {
    "DCOILWTICO": ("WTI", "USD/bbl", "WTI Cushing front spot, daily"),
    "DCOILBRENTEU": ("BRENT", "USD/bbl", "Brent Europe front spot, daily"),
    "DHHNGSP": ("HENRYHUB", "USD/MMBtu", "Henry Hub natural gas spot, daily"),
}
CM_MONTHLY = {
    "PCOPPUSDM": ("COPPER", "USD/mt", "IMF global copper price, monthly"),
    "PMAIZMTUSDM": ("MAIZE", "USD/mt", "IMF maize (corn) price, monthly"),
    "PWHEAMTUSDM": ("WHEAT", "USD/mt", "IMF wheat price, monthly"),
    "PNGASEUUSDM": ("NGAS_EU", "USD/MMBtu", "IMF EU natural-gas price, monthly"),
    "POILBREUSDM": ("BRENT_M", "USD/bbl", "IMF Brent price, monthly"),
    "PALLFNFINDEXM": ("ALLCOMM_IDX", "index 2016=100", "IMF all-commodity price index, monthly"),
}


def fetch_commodity_prices():
    print("\n=== COMMODITY PRICES (FRED) ===")
    daily_panel = {}
    for sid, (name, units, desc) in CM_DAILY.items():
        df = fred_csv(sid)
        if df is None:
            print(f"  [FAIL] {sid} ({name})")
            MANI["failures"].append({"kind": "commodity_daily", "id": sid, "name": name})
            continue
        df = df.rename(columns={sid: name})
        rel = save_csv(df, os.path.join(CM_DIR, f"price_{name}.csv"))
        a, b, n = span_of(df)
        print(f"  [ ok ] {sid:14s} {name:9s} daily   n={n:<6d} {a}..{b} -> {rel}")
        MANI["commodity"][f"price_{name}"] = {"file": rel, "fred_id": sid, "freq": "daily",
                                              "units": units, "desc": desc, "start": a, "end": b, "rows": n}
        daily_panel[name] = df.set_index("date")[name]
    if daily_panel:
        wide = pd.DataFrame(daily_panel).sort_index()
        wide.index.name = "date"
        rel = save_csv(wide.reset_index(), os.path.join(CM_DIR, "panel_price_daily.csv"))
        a, b, n = (str(wide.index[0].date()), str(wide.index[-1].date()), len(wide))
        print(f"  [PANEL] daily commodity prices cols={list(wide.columns)} n={n} {a}..{b}")
        MANI["commodity"]["panel_price_daily"] = {"file": rel, "cols": list(wide.columns),
                                                  "start": a, "end": b, "rows": n}

    for sid, (name, units, desc) in CM_MONTHLY.items():
        df = fred_csv(sid)
        if df is None:
            print(f"  [FAIL] {sid} ({name})")
            MANI["failures"].append({"kind": "commodity_monthly", "id": sid, "name": name})
            continue
        df = df.rename(columns={sid: name})
        rel = save_csv(df, os.path.join(CM_DIR, f"price_{name}.csv"))
        a, b, n = span_of(df)
        print(f"  [ ok ] {sid:14s} {name:11s} monthly n={n:<5d} {a}..{b}")
        MANI["commodity"][f"price_{name}"] = {"file": rel, "fred_id": sid, "freq": "monthly",
                                              "units": units, "desc": desc, "start": a, "end": b, "rows": n}


# ==================================================================================
# COMMODITY TERM STRUCTURE PROBES (document the blocker)
# ==================================================================================
def probe_term_structure():
    print("\n=== COMMODITY TERM-STRUCTURE / ROLL-YIELD probes (expected to fail) ===")
    blockers = {}

    # 1) Nasdaq Data Link CHRIS continuous front + second
    for q in ["CHRIS/CME_CL1", "CHRIS/CME_CL2"]:
        try:
            r = http_get(f"https://data.nasdaq.com/api/v3/datasets/{q}.csv", tries=1, timeout=20)
            ok = r.status_code == 200 and r.text[:20].lower().startswith("date")
            blockers[q] = {"status": r.status_code, "usable": bool(ok), "bytes": len(r.text)}
            print(f"  CHRIS {q}: http={r.status_code} usable={ok}")
        except Exception as e:  # noqa: BLE001
            blockers[q] = {"status": "err", "usable": False, "error": type(e).__name__}
            print(f"  CHRIS {q}: ERR {type(e).__name__}")

    # 2) Stooq front-month
    for s in ["cl.f", "ng.f"]:
        try:
            r = http_get(f"https://stooq.com/q/d/l/?s={s}&i=d", tries=1, timeout=20)
            txt = r.text
            usable = r.status_code == 200 and txt[:5].lower().startswith("date")
            wall = "noindex" in txt[:200].lower() or "async()" in txt[:200].lower()
            blockers[f"stooq_{s}"] = {"status": r.status_code, "usable": bool(usable),
                                      "antibot_wall": bool(wall)}
            print(f"  Stooq {s}: http={r.status_code} usable={usable} antibot_wall={wall}")
        except Exception as e:  # noqa: BLE001
            blockers[f"stooq_{s}"] = {"status": "err", "usable": False, "error": type(e).__name__}
            print(f"  Stooq {s}: ERR {type(e).__name__}")

    # 3) BCOM/GSCI total-return vs spot on FRED
    for sid in ["BCOMTR", "SPGSCITR", "DJUBSTR"]:
        df = fred_csv(sid)
        blockers[f"fred_{sid}"] = {"usable": df is not None}
        print(f"  FRED {sid}: usable={df is not None}")

    # 4) EIA open-data (needs key)
    try:
        r = http_get("https://api.eia.gov/v2/", tries=1, timeout=15)
        needs_key = "API_KEY_MISSING" in r.text
        blockers["eia_v2"] = {"status": r.status_code, "needs_key": bool(needs_key), "usable": False}
        print(f"  EIA v2: http={r.status_code} needs_key={needs_key}")
    except Exception as e:  # noqa: BLE001
        blockers["eia_v2"] = {"status": "err", "usable": False, "error": type(e).__name__}

    MANI["commodity"]["term_structure_probes"] = blockers
    any_usable = any(v.get("usable") for v in blockers.values())
    MANI["commodity"]["roll_yield_computable_from_free_sources"] = bool(any_usable)
    print(f"  => roll-yield computable from FREE front+second sources: {any_usable}")


# ==================================================================================
# CFTC COT POSITIONING (optional bonus)
# ==================================================================================
# tag -> list of (market_and_exchange_names) to stitch (old first, new last).
COT_MARKETS = {
    # commodities
    "WTI_CRUDE": ["CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
                  "WTI FINANCIAL CRUDE OIL - NEW YORK MERCANTILE EXCHANGE"],
    "GOLD": ["GOLD - COMMODITY EXCHANGE INC."],
    "HENRYHUB_GAS": ["HENRY HUB LAST DAY FIN - NEW YORK MERCANTILE EXCHANGE"],
    # FX (CME)
    "EUR": ["EURO FX - CHICAGO MERCANTILE EXCHANGE"],
    "JPY": ["JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE"],
    "GBP": ["BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE",
            "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE"],
    "CAD": ["CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE"],
    "AUD": ["AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE"],
    "CHF": ["SWISS FRANC - CHICAGO MERCANTILE EXCHANGE"],
    "NZD": ["NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE",
            "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE"],
}
COT_SELECT = ("report_date_as_yyyy_mm_dd,open_interest_all,"
              "noncomm_positions_long_all,noncomm_positions_short_all,"
              "comm_positions_long_all,comm_positions_short_all")


def _cot_one(name):
    rows = []
    offset = 0
    while True:
        r = http_get(COT_BASE, params={
            "$select": COT_SELECT,
            "$where": f"market_and_exchange_names='{name}'",
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$limit": 5000, "$offset": offset}, timeout=90)
        if r.status_code != 200:
            break
        chunk = list(_csv.DictReader(io.StringIO(r.text)))
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < 5000:
            break
        offset += 5000
    return rows


def fetch_cot():
    print("\n=== CFTC COT positioning (legacy futures-only; optional) ===")
    panel = {}
    for tag, names in COT_MARKETS.items():
        all_rows = []
        for nm in names:
            all_rows.extend(_cot_one(nm))
        if not all_rows:
            print(f"  [FAIL] {tag}")
            MANI["failures"].append({"kind": "cot", "tag": tag})
            continue
        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce")
        for c in ["noncomm_positions_long_all", "noncomm_positions_short_all",
                  "comm_positions_long_all", "comm_positions_short_all", "open_interest_all"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        df["noncomm_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
        df["noncomm_net_pct_oi"] = df["noncomm_net"] / df["open_interest_all"]
        out = df[["date", "open_interest_all", "noncomm_positions_long_all",
                  "noncomm_positions_short_all", "noncomm_net", "noncomm_net_pct_oi"]]
        sub = "fx" if tag in ("EUR", "JPY", "GBP", "CAD", "AUD", "CHF", "NZD") else "commodity"
        d = FX_DIR if sub == "fx" else CM_DIR
        rel = save_csv(out, os.path.join(d, f"cot_{tag}.csv"))
        a, b, n = span_of(out)
        print(f"  [ ok ] COT {tag:13s} n={n:<5d} {a}..{b} -> {rel}")
        MANI[sub][f"cot_{tag}"] = {"file": rel, "markets": names, "source": "CFTC COT legacy (6dca-aqww)",
                                   "freq": "weekly", "start": a, "end": b, "rows": n,
                                   "cols": "noncomm_net = spec long-short; noncomm_net_pct_oi"}
        panel[tag] = out.set_index("date")["noncomm_net_pct_oi"]


# ==================================================================================
def main():
    fetch_fx_spot()
    fetch_fx_carry()
    fetch_fx_value()
    fetch_commodity_prices()
    probe_term_structure()
    fetch_cot()

    with open(MANIFEST, "w") as f:
        json.dump(MANI, f, indent=2)
    print(f"\nManifest -> {os.path.relpath(MANIFEST, ROOT)}  "
          f"(fx series={len(MANI['fx'])}, commodity series={len(MANI['commodity'])}, "
          f"failures={len(MANI['failures'])})")


if __name__ == "__main__":
    main()
