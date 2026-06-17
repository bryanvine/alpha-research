#!/usr/bin/env python3
"""
16_fetch_factor_zoo.py -- Paper 4 ("The Cost of Direction") factor-zoo data sourcing.

Acquires the canonical PUBLISHED-FACTOR benchmark datasets so the directional-anomaly
audit (how many survive out-of-sample + net of costs) has clean inputs:

  (1) OPEN-SOURCE ASSET PRICING  (Chen & Zimmermann, openassetpricing.com)
      The replication zoo: ~200 published predictors, as PRE-BUILT LONG-SHORT
      PORTFOLIO MONTHLY RETURNS (not raw firm signals) + the SignalDoc metadata.
      Pulled via the `openassetpricing` pip package (module `openassetpricing`,
      class `openassetpricing.OpenAP`), which reads the hosted Google-Drive release
      directly -- NO WRDS / NO Google login needed for the portfolio + doc files.
        * dl_port('op', 'pandas')   -> PredictorPortsFull.csv (date x signalname x port)
                                       LONG-SHORT leg is port == 'LS'.  RETURNS IN PERCENT.
        * dl_signal_doc('pandas')   -> SignalDoc.csv  (Acronym, Cat.Signal=clear/likely/
                                       maybe, SampleStartYear/SampleEndYear, t-stat, Sign...)
      We pivot the LS leg to a tidy DATE x PREDICTOR wide frame and CONVERT TO DECIMAL.

      NOTE on the package: it is pinned to pandas<3 and on install DOWNGRADES the venv
      from pandas 3.0.3 -> 2.2.3 and pulls in polars/wrds. pyarrow/numpy unaffected.
      Its OpenAP(release_year=...) wants the release TAG, not a plain year:
      valid tags are 2022, 2023, 202408, 202410, 202510 (we use 202510 = v2.00, 2025-10).
      Passing OpenAP(2025) raises TypeError (no release2025_url) -- a package quirk.

  (2) KEN FRENCH DATA LIBRARY  (free, no key) -- the out-of-sample yardstick factors.
      Monthly CSV zips: each has a text preamble, then a MONTHLY block of `YYYYMM,vals`
      IN PERCENT, a blank line, then an ANNUAL block (and sometimes more sub-tables).
      We parse ONLY the leading monthly block and CONVERT TO DECIMAL.
        * F-F_Research_Data_5_Factors_2x3   -> Mkt-RF, SMB, HML, RMW, CMA, RF
        * F-F_Momentum_Factor               -> Mom
        * F-F_ST_Reversal_Factor            -> ST_Rev
        * F-F_LT_Reversal_Factor            -> LT_Rev

ALL HTTP IS DONE WITH PYTHON (requests / the OAP package). The bash sandbox on this
host has no network egress; only in-process Python egress works (mirrors the Paper-1/6
fetch approach). Re-running is idempotent (files are overwritten).

OUTPUTS (all under data/equity_factors/):
  oap_ls_returns.parquet     date x predictor, monthly, DECIMAL long-short returns
  oap_signaldoc.csv          predictor metadata (verbatim from OSAP)
  famafrench_monthly.parquet date x factor, monthly, DECIMAL
  _paper4_manifest.json      machine-readable summary (spans, units, counts, blockers)
"""
import os
import io
import sys
import json
import zipfile
import datetime as dt

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQ_DIR = os.path.join(ROOT, "data", "equity_factors")
MANIFEST = os.path.join(EQ_DIR, "_paper4_manifest.json")

OAP_RELEASE = 202510  # v2.00, 2025-10 (latest in package)
UA = {"User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120 Safari/537.36 alpha-research/paper4")}

# Dedicated OSAP "Download" page artifacts (separate Drive file_ids => separate quota
# buckets than the 212-portfolio PredictorPortsFull.csv the package pulls). These are
# the PREFERRED route: PredictorLSretWide.csv is ALREADY a tidy date x predictor wide
# frame of long-short monthly returns (in PERCENT). Verified live on the OSAP data page.
OSAP_LSWIDE_FILEID = "10sOryk_ddjkXagaajTKUk1nwJs2ZLRiI"  # "Monthly long-short returns (wide csv)"
OSAP_SIGNALDOC_FILEID = "1Sev9s6cPFUGgxp1pFiej0lGzpsMqJCI2"  # SignalDoc.csv (direct)

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FRENCH_FILES = {
    # logical_zip_name : (zip url, list of expected output column names)
    "5factors": (f"{FRENCH_BASE}/F-F_Research_Data_5_Factors_2x3_CSV.zip",
                 ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]),
    "momentum": (f"{FRENCH_BASE}/F-F_Momentum_Factor_CSV.zip", ["Mom"]),
    "st_rev":   (f"{FRENCH_BASE}/F-F_ST_Reversal_Factor_CSV.zip", ["ST_Rev"]),
    "lt_rev":   (f"{FRENCH_BASE}/F-F_LT_Reversal_Factor_CSV.zip", ["LT_Rev"]),
}

manifest = {
    "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    "paper": "Paper 4 -- The Cost of Direction (net-of-cost audit of the directional anomaly zoo)",
    "oap": {},
    "famafrench": {},
    "blockers": [],
}


# ---------------------------------------------------------------------------
# (1) OPEN-SOURCE ASSET PRICING
# ---------------------------------------------------------------------------
def _drive_download(file_id, timeout=240):
    """Download a public Google-Drive file, handling the large-file confirm-token
    interstitial. Returns (bytes_or_None, info_str). info=='QUOTA' on the Drive
    'Too many users have viewed or downloaded this file' wall."""
    import re as _re
    sess = requests.Session()
    sess.headers.update(UA)
    base = "https://drive.google.com/uc?export=download"
    r = sess.get(base, params={"id": file_id}, stream=True, timeout=120)
    if "text/html" not in r.headers.get("Content-Type", ""):
        return r.content, "direct"
    html = r.content.decode("utf-8", "ignore")
    if "Quota exceeded" in html or "Too many users" in html:
        return None, "QUOTA"
    form = _re.search(r'<form[^>]+action="([^"]+)"[^>]*>(.*?)</form>', html, _re.S)
    if form:
        action = form.group(1).replace("&amp;", "&")
        inputs = dict(_re.findall(r'name="([^"]+)"\s+value="([^"]*)"', form.group(2)))
        r2 = sess.get(action, params=inputs, stream=True, timeout=timeout)
        if "text/html" not in r2.headers.get("Content-Type", ""):
            return r2.content, "confirm-form"
        h2 = r2.content.decode("utf-8", "ignore")
        if "Quota exceeded" in h2 or "Too many users" in h2:
            return None, "QUOTA"
        return None, "HTML2:" + h2[:160]
    return None, "HTML:" + html[:160]


def _oap_lswide_from_drive():
    """Preferred path: pull the dedicated PredictorLSretWide.csv (wide LS returns)
    + SignalDoc.csv directly from the OSAP Download-page Drive file_ids.
    Returns (wide_df_or_None, doc_df_or_None)."""
    wide = None
    doc = None
    # SignalDoc
    try:
        print("  [drive] SignalDoc.csv ...", flush=True)
        content, info = _drive_download(OSAP_SIGNALDOC_FILEID)
        if content is not None:
            doc = pd.read_csv(io.BytesIO(content))
            print(f"    ok ({info}) {doc.shape[0]}x{doc.shape[1]}", flush=True)
        else:
            print(f"    drive SignalDoc unavailable: {info}", flush=True)
    except Exception as e:
        print(f"    drive SignalDoc error: {e!r}", flush=True)
    # LS-wide returns
    try:
        print("  [drive] PredictorLSretWide.csv (wide LS returns) ...", flush=True)
        content, info = _drive_download(OSAP_LSWIDE_FILEID)
        if content is not None:
            wide = pd.read_csv(io.BytesIO(content))
            print(f"    ok ({info}) {wide.shape[0]}x{wide.shape[1]}", flush=True)
        else:
            print(f"    drive LS-wide unavailable: {info}", flush=True)
    except Exception as e:
        print(f"    drive LS-wide error: {e!r}", flush=True)
    return wide, doc


def _finalize_oap_wide(wide_raw, source_desc):
    """Take a raw date x predictor PERCENT wide frame -> tidy DECIMAL parquet + manifest."""
    wide = wide_raw.copy()
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.set_index("date").sort_index()
    # Coerce all predictor columns numeric (some files ship object cols w/ "NA").
    wide = wide.apply(pd.to_numeric, errors="coerce")
    wide.index = wide.index + pd.offsets.MonthEnd(0)  # normalize to month-end
    wide.index.name = "date"
    wide_dec = wide / 100.0  # OSAP LS returns ship in PERCENT -> DECIMAL
    out = os.path.join(EQ_DIR, "oap_ls_returns.parquet")
    wide_dec.to_parquet(out)
    print(f"    -> {out}", flush=True)
    print(f"       {wide_dec.shape[0]} months x {wide_dec.shape[1]} predictors, DECIMAL; "
          f"{wide_dec.index.min().date()} .. {wide_dec.index.max().date()}", flush=True)
    return wide_dec, source_desc


def fetch_oap():
    print("\n=== (1) Open-Source Asset Pricing (Chen & Zimmermann) ===", flush=True)

    doc = None
    wide_dec = None
    source_desc = None

    # ---- PREFERRED: dedicated Drive files (PredictorLSretWide.csv + SignalDoc.csv) ----
    # Separate file_ids => separate per-file quota buckets than the big 212-portfolio
    # PredictorPortsFull.csv the pip package pulls (which is currently quota-walled).
    try:
        wide_raw, doc_raw = _oap_lswide_from_drive()
    except Exception as e:
        print(f"  drive route error: {e!r}", flush=True)
        wide_raw, doc_raw = None, None

    if doc_raw is not None:
        doc = doc_raw
    if wide_raw is not None and "date" in wide_raw.columns:
        try:
            wide_dec, source_desc = _finalize_oap_wide(
                wide_raw,
                "openassetpricing.com Download page -- PredictorLSretWide.csv "
                "(dedicated wide long-short returns file), fetched directly from Drive")
        except Exception as e:
            print(f"  finalize (drive LS-wide) failed: {e!r}", flush=True)

    # ---- FALLBACK: pip openassetpricing package (PredictorPortsFull.csv, port=='LS') ----
    if wide_dec is None or doc is None:
        try:
            import openassetpricing as oap
            print(f"  [pkg] OpenAP(release_year={OAP_RELEASE}) "
                  f"(parses Drive folder listing)...", flush=True)
            o = oap.OpenAP(OAP_RELEASE)
            if doc is None:
                print("  [pkg] dl_signal_doc ...", flush=True)
                doc = o.dl_signal_doc("pandas")
            if wide_dec is None:
                print("  [pkg] dl_port('op') -- prebuilt portfolios (port=='LS') ...",
                      flush=True)
                port = o.dl_port("op", "pandas")
                ls = port.loc[port["port"].astype(str) == "LS"].copy()
                wide_raw = (ls.pivot_table(index="date", columns="signalname", values="ret")
                              .reset_index())
                wide_dec, source_desc = _finalize_oap_wide(
                    wide_raw,
                    "openassetpricing.com via pip openassetpricing -- "
                    "PredictorPortsFull.csv (port=='LS'), pivoted to wide")
        except Exception as e:
            msg = (f"OSAP package fallback failed ({e!r}). The pip route pulls the big "
                   f"PredictorPortsFull.csv whose Drive file is currently QUOTA-WALLED "
                   f"('Too many users have viewed or downloaded this file recently'). "
                   f"Dedicated wide file (preferred): "
                   f"https://drive.google.com/file/d/{OSAP_LSWIDE_FILEID}/view  "
                   f"(PredictorLSretWide.csv). SignalDoc: "
                   f"https://drive.google.com/file/d/{OSAP_SIGNALDOC_FILEID}/view . "
                   f"Manual page: https://www.openassetpricing.com/data/ . "
                   f"Release folder (v2.00, 2025-10): "
                   f"https://drive.google.com/drive/folders/1qQDuTsnyvWfEJR6nPBQZ8xxlq6bkLG_y")
            print("  NOTE:", msg, flush=True)
            if wide_dec is None:
                manifest["blockers"].append({"dataset": "oap_ls_returns", "blocker": msg})

    # ---- Write SignalDoc + manifest ----
    if doc is not None:
        doc_path = os.path.join(EQ_DIR, "oap_signaldoc.csv")
        doc.to_csv(doc_path, index=False)
        print(f"  SignalDoc -> {doc_path}  ({doc.shape[0]} rows x {doc.shape[1]} cols)",
              flush=True)
    else:
        manifest["blockers"].append({"dataset": "oap_signaldoc",
                                     "blocker": "SignalDoc.csv could not be fetched."})

    if wide_dec is not None:
        # Cross-reference predictors vs SignalDoc Cat.Signal where possible.
        cat_counts = None
        if doc is not None and {"Acronym", "Cat.Signal"}.issubset(doc.columns):
            in_doc = doc.set_index("Acronym")["Cat.Signal"]
            present = [c for c in wide_dec.columns if c in in_doc.index]
            cat_counts = (in_doc.loc[present].value_counts(dropna=False)
                          .astype(int).to_dict())
            cat_counts = {str(k): int(v) for k, v in cat_counts.items()}
        manifest["oap"] = {
            "release_package": OAP_RELEASE,
            "source": source_desc,
            "ls_returns_path": "data/equity_factors/oap_ls_returns.parquet",
            "ls_returns_schema": ("DatetimeIndex 'date' (month-end) x predictor columns "
                                  "(OSAP Acronym); values = long-short monthly return"),
            "units": "DECIMAL (OSAP long-short returns ship in PERCENT; divided by 100)",
            "n_predictors": int(wide_dec.shape[1]),
            "n_months": int(wide_dec.shape[0]),
            "date_min": str(wide_dec.index.min().date()),
            "date_max": str(wide_dec.index.max().date()),
            "predictor_cat_signal_counts": cat_counts,
            "signaldoc_path": "data/equity_factors/oap_signaldoc.csv",
            "signaldoc_present": doc is not None,
            "signaldoc_cols": (list(doc.columns) if doc is not None else None),
            "signaldoc_rows": (int(doc.shape[0]) if doc is not None else None),
            "signaldoc_key_metadata": ["Acronym", "Cat.Signal (Predictor/Placebo/Drop)",
                                       "SampleStartYear", "SampleEndYear", "T-Stat", "Sign"],
        }


# ---------------------------------------------------------------------------
# (2) KEN FRENCH DATA LIBRARY
# ---------------------------------------------------------------------------
def _parse_french_monthly(raw_bytes, expected_cols):
    """Parse a French Data Library zip's MONTHLY block only -> DataFrame in PERCENT.

    Layout: preamble lines, then a header line, then rows `YYYYMM, v, v, ...`. The
    monthly block ends at the first blank line (or a non-YYYYMM token); annual block
    and any further sub-tables follow and are IGNORED.
    """
    zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
    name = zf.namelist()[0]
    text = zf.read(name).decode("latin-1")
    lines = text.splitlines()

    rows = []
    header = None
    started = False
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        token = cells[0]
        is_monthly_key = len(token) == 6 and token.isdigit()  # YYYYMM
        if is_monthly_key:
            started = True
            # numeric payload; some cells can be -99.99 / -999 (French missing codes)
            vals = []
            for c in cells[1:]:
                if c == "":
                    continue
                try:
                    vals.append(float(c))
                except ValueError:
                    vals.append(np.nan)
            rows.append((token, vals))
        else:
            if started:
                # first non-YYYYMM row AFTER the monthly block => block finished
                break
            # before the monthly block: capture the last comma-bearing line as header
            if len(cells) > 1 and any(cells[1:]):
                header = cells

    if not rows:
        raise ValueError(f"No monthly YYYYMM rows parsed from {name}")

    width = max(len(v) for _, v in rows)
    # Derive column names: prefer the file's own header tokens, else expected_cols.
    colnames = None
    if header is not None:
        cand = [h for h in header[-width:] if h != ""]
        if len(cand) == width:
            colnames = cand
    if colnames is None or len(colnames) != width:
        colnames = list(expected_cols)[:width]
    if len(colnames) < width:  # pad if file has more cols than expected
        colnames += [f"col{i}" for i in range(len(colnames), width)]

    idx = pd.to_datetime([r[0] for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
    data = [r[1] + [np.nan] * (width - len(r[1])) for r in rows]
    df = pd.DataFrame(data, index=idx, columns=colnames)
    df.index.name = "date"
    # French missing codes
    df = df.mask((df <= -99.99) & (df >= -999.0) | (df == -99.99) | (df == -999.0))
    df = df.replace([-99.99, -999.0, -9999.0], np.nan)
    return df


def fetch_french():
    print("\n=== (2) Ken French Data Library ===", flush=True)
    frames = []
    spans = {}
    for key, (url, expected) in FRENCH_FILES.items():
        try:
            print(f"  GET {url}", flush=True)
            r = requests.get(url, headers=UA, timeout=120)
            r.raise_for_status()
            df = _parse_french_monthly(r.content, expected)
            df = df / 100.0  # PERCENT -> DECIMAL
            # Rename to the expected canonical names if the count lines up.
            if len(df.columns) == len(expected):
                df.columns = expected
            else:
                # keep whatever parsed; should not happen for these files
                print(f"    WARN: parsed {len(df.columns)} cols, expected {len(expected)}: "
                      f"{list(df.columns)}", flush=True)
            frames.append(df)
            spans[key] = {"cols": list(df.columns),
                          "start": str(df.index.min().date()),
                          "end": str(df.index.max().date()),
                          "n_months": int(df.shape[0])}
            print(f"    parsed {df.shape[0]} months {df.index.min().date()}.."
                  f"{df.index.max().date()}; cols={list(df.columns)}", flush=True)
        except Exception as e:
            msg = f"French '{key}' fetch/parse failed ({e!r}); url={url}"
            print("  BLOCKER:", msg, flush=True)
            manifest["blockers"].append({"dataset": f"french_{key}", "blocker": msg})

    if not frames:
        manifest["blockers"].append({"dataset": "famafrench", "blocker": "no French frames parsed"})
        return

    merged = pd.concat(frames, axis=1).sort_index()
    # If RF appeared more than once (it won't here, only 5F has it) dedupe.
    merged = merged.loc[:, ~merged.columns.duplicated()]
    out = os.path.join(EQ_DIR, "famafrench_monthly.parquet")
    merged.to_parquet(out)
    print(f"  -> {out}", flush=True)
    print(f"     {merged.shape[0]} months x {merged.shape[1]} factors, DECIMAL; "
          f"{merged.index.min().date()} .. {merged.index.max().date()}; "
          f"cols={list(merged.columns)}", flush=True)

    manifest["famafrench"] = {
        "path": "data/equity_factors/famafrench_monthly.parquet",
        "source": "Ken French Data Library (Tuck/Dartmouth), free, no key",
        "schema": "DatetimeIndex 'date' (month-end) x factor columns",
        "units": "DECIMAL (source CSVs are PERCENT; divided by 100)",
        "factors": list(merged.columns),
        "n_factors": int(merged.shape[1]),
        "n_months": int(merged.shape[0]),
        "date_min": str(merged.index.min().date()),
        "date_max": str(merged.index.max().date()),
        "per_source_spans": spans,
    }


def main():
    os.makedirs(EQ_DIR, exist_ok=True)
    fetch_oap()
    fetch_french()
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nManifest -> {MANIFEST}", flush=True)
    if manifest["blockers"]:
        print(f"BLOCKERS: {len(manifest['blockers'])}", flush=True)
        for b in manifest["blockers"]:
            print(f"  - [{b['dataset']}] {b['blocker'][:160]}", flush=True)
    else:
        print("No blockers.", flush=True)


if __name__ == "__main__":
    main()
