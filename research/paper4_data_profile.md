# Paper 4 — "The Cost of Direction": published-factor benchmark data profile

**Generated:** 2026-06-17 UTC by `scripts/16_fetch_factor_zoo.py`
**Output dir:** `/apps/alpha-research/data/equity_factors/`
**Registry entry:** `configs/data_sources.yaml -> equity_factor_zoo`
**Manifest:** `data/equity_factors/_paper4_manifest.json`

---

## 0. TL;DR

| dataset | file | shape | span | units |
|---|---|---|---|---|
| OSAP long-short returns | `oap_ls_returns.parquet` | 1188 mo × **212 predictors** | 1926-01 .. 2024-12 | **DECIMAL** |
| OSAP predictor metadata | `oap_signaldoc.csv` | 331 rows × 28 cols | — | — |
| Fama-French factors | `famafrench_monthly.parquet` | 1203 mo × **9 factors** | 1926-02 .. 2026-04 | **DECIMAL** |

**Everything stored is DECIMAL** (a monthly return of 1% is `0.01`). Both sources ship in
PERCENT; the fetcher divides by 100. Sanity check that confirms the conversion: Mkt-RF
annualizes to **7.17%/yr**, RF to **4.36%/yr**, and the average OSAP predictor long-short
to **6.0%/yr** — all single-digit, i.e. decimal (had we left them in percent these would
read as 717%/yr, etc.).

No blockers in the final run. (One transient Google-Drive quota wall on the *large*
portfolio file was routed around — see §2.3.)

---

## 1. Why this dataset exists

Paper 4 ("The Cost of Direction") audits the directional **anomaly zoo**: of the ~200
published cross-sectional equity predictors, how many actually survive (a) **out of
sample**, past their original publication window, and (b) **net of realistic trading
costs**? To run that audit we need, as clean inputs:

1. **The returns of the published long-short anomalies themselves** — not raw firm-level
   signals, but the *realized monthly long-short portfolio returns* for each predictor,
   on a common calendar. → Chen–Zimmermann **Open-Source Asset Pricing** (OSAP).
2. **The original publication metadata** — each anomaly's in-sample window, original
   t-stat, sign, and whether the authors classed it a real predictor vs a placebo — so
   "out-of-sample" can be defined per-anomaly. → OSAP **SignalDoc**.
3. **A canonical risk-model yardstick** — the Fama-French factors + momentum + the two
   reversal factors — to (a) define the standard "known" factors the zoo is measured
   against and (b) provide the OOS benchmark legs. → **Ken French Data Library**.

---

## 2. Source 1 — Open-Source Asset Pricing (Chen & Zimmermann)

**Provider:** Andrew Y. Chen & Tom Zimmermann, *Open Source Cross-Sectional Asset
Pricing* (Critical Finance Review). Site `openassetpricing.com`; code at GitHub
`OpenSourceAP/CrossSection`. The canonical free replication of the published anomaly zoo.

### 2.1 What we pulled (RETURNS, not signals)

The brief wants **returns (date × predictor, monthly)**, not raw firm signals. OSAP
publishes prebuilt **long-short portfolio returns** directly, so no portfolio
construction (and no WRDS/CRSP access) was needed.

- **`oap_ls_returns.parquet`** — the realized **long-short** monthly returns.
  - Schema: `DatetimeIndex 'date'` (month-end) × **212 predictor columns** (named by OSAP
    *Acronym*, e.g. `AM`, `Accruals`, `Mom12m`, `STreversal`, `AnnouncementReturn`).
  - **1188 months, 1926-01-31 .. 2024-12-31.**
  - **Units: DECIMAL.** Source file `PredictorLSretWide.csv` is in PERCENT (verified:
    median |monthly value| ≈ 1.64%, cross-predictor mean ≈ 0.50%/mo); divided by 100.
  - This is the dedicated **"Monthly long-short returns (wide csv)"** artifact from the
    OSAP Download page — already a tidy date × predictor wide frame (no pivot needed).
  - **NaN by design (~31% of cells):** predictors have **staggered start dates** (each
    begins when its underlying data does). 27 predictors reach back to 1926; the big
    cohorts begin ~1951, ~1963, ~1972, ~1996; only 3 are fully populated from 1926.
    Per-predictor first-valid dates must be respected in the audit.

### 2.2 Metadata — `oap_signaldoc.csv` (the OOS / placebo key)

331 rows × 28 cols, verbatim from OSAP `SignalDoc.csv`. **All key fields present.** The
columns Paper 4 will lean on:

| column | meaning |
|---|---|
| `Acronym` | predictor id; **joins 212/212 to the return columns** |
| `Cat.Signal` | **Predictor = 212 / Placebo = 114 / Drop = 5**. *(v2 schema — note this release uses Predictor/Placebo/Drop, NOT the older clear/likely/maybe labels.)* All 212 traded return columns are `Predictor`. |
| `SampleStartYear`, `SampleEndYear` | the **original in-sample window** (starts 1926–2002; ends **1968–2014**). Everything after `SampleEndYear` is genuine post-publication OOS. |
| `T-Stat` | original published t-stat (median \|t\| = **4.0** among Predictors). |
| `Sign` | +1/−1 — the long-short direction as published. |
| `Authors`, `Year`, `Journal`, `LongDescription`, `Cat.Form/Data/Economic`, `GScholarCites202509` | provenance + taxonomy + citation count. |

Because every anomaly's in-sample window ends by 2014 while the return series runs to
2024-12, there is **10+ years of clean out-of-sample** for the most recent anomalies and
much more for older ones — exactly what the OOS-survival audit needs.

### 2.3 How it was acquired + the quota wall we routed around

Acquired by fetching the OSAP Download-page **Google-Drive files directly** with
`requests` (handling Drive's large-file confirm-token interstitial). The pip
`openassetpricing` package is the documented path and the **fallback** in the script.

Three package quirks worth recording (all handled):

1. **`pip install openassetpricing` (v0.0.2) downgrades the venv:** it pins `pandas<3`
   and so replaced **pandas 3.0.3 → 2.2.3**, and pulled in `polars`, `wrds`,
   `sqlalchemy`, `ipython`. `numpy` / `pyarrow` / `requests` are unaffected, and the
   parquet outputs are engine-neutral. (If pandas 3.x is required elsewhere, reinstall it
   *after* running this fetch, or run the fetch in a throwaway venv.)
2. **`OpenAP(release_year=...)` wants the release _tag_, not a plain year.** Valid tags:
   `2022, 2023, 202408, 202410, 202510`. `OpenAP(2025)` raises `TypeError`. We pin
   **202510** (v2.00, Oct-2025).
3. **The big portfolio file is Drive-quota-walled.** The package's `dl_port('op')`
   downloads `PredictorPortsFull.csv` (all legs incl. deciles, file_id
   `1g7w-yQ6Cg2qbMEkER9Q3vgns4JszXQo6`), which returned *"Too many users have viewed or
   downloaded this file recently"* during this run. We therefore **prefer the dedicated
   `PredictorLSretWide.csv`** (file_id `10sOryk_ddjkXagaajTKUk1nwJs2ZLRiI`) and
   `SignalDoc.csv` (`1Sev9s6cPFUGgxp1pFiej0lGzpsMqJCI2`) — separate file_ids = separate
   per-file quota buckets, both downloaded cleanly. Note the package's
   `dl_all_signals`/`dl_signal` also need **WRDS/CRSP credentials**; we do not use them
   (we want returns, not firm signals).

**Manual fallback** (if every Drive route is quota-walled): download
`PredictorLSretWide.csv` + `SignalDoc.csv` by hand from
`https://www.openassetpricing.com/data/` into this dir and re-run the script — the
normalize step will pick them up. Release folder (v2.00):
`https://drive.google.com/drive/folders/1qQDuTsnyvWfEJR6nPBQZ8xxlq6bkLG_y`.

### 2.4 Quick descriptive read (full-sample, decimal)

- Mean annualized long-short return across the 212 predictors: **6.0%** (median 5.2%).
- Mean annualized Sharpe across predictors: **0.51** (median **0.45**) — *full-sample,
  gross, in-and-out-of-sample combined*. The whole point of Paper 4 is that this almost
  certainly **shrinks OOS and net of costs**.
- Highest full-sample gross Sharpe: `SmileSlope` 2.08, `AnnouncementReturn` 1.92,
  `EarningsStreak` 1.68, `dCPVolSpread` 1.55, `STreversal` 1.40.
- Lowest: `VarCF` −0.26, `PatentsRD` −0.20, `Governance` −0.14, `FirmAge` −0.14.

---

## 3. Source 2 — Ken French Data Library

**Provider:** Ken French Data Library (Tuck/Dartmouth), free, no API key. This vintage is
built from the **202604 CRSP database** (per the file preambles).

**Acquired via** `requests` GET of the monthly CSV zips. Each zip is a single text file
with a preamble, a **monthly block** of `YYYYMM,vals` (in PERCENT), a blank line, then an
**annual block** (and sometimes further sub-tables). We parse **only the leading monthly
block** and divide by 100 → DECIMAL. (Verified the parser stops exactly at the blank line
before *"Annual Factors: January-December"*, and that columns align: the 1963-07 raw row
`-0.39,-0.48,-0.81,0.64,-1.15,0.27` → `-0.0039,...,0.0027`.)

**`famafrench_monthly.parquet`** — `DatetimeIndex 'date'` (month-end) × 9 factor columns,
**DECIMAL**, **1203 months 1926-02-28 .. 2026-04-30**:

| factor(s) | source zip | span | n months |
|---|---|---|---|
| `Mkt-RF, SMB, HML, RMW, CMA, RF` | `F-F_Research_Data_5_Factors_2x3_CSV.zip` | **1963-07** .. 2026-04 | 754 |
| `Mom` | `F-F_Momentum_Factor_CSV.zip` | 1927-01 .. 2026-04 | 1192 |
| `ST_Rev` | `F-F_ST_Reversal_Factor_CSV.zip` | 1926-02 .. 2026-04 | 1203 |
| `LT_Rev` | `F-F_LT_Reversal_Factor_CSV.zip` | 1931-01 .. 2026-04 | 1144 |

**Caveat for the audit:** the **5-factor block (incl. RMW/CMA) only begins 1963-07**, so
any benchmark/regression using the full FF5 model is constrained to **1963-07 onward**.
Mom/ST_Rev/LT_Rev go back to the late 1920s. NaNs before each factor's first-valid date
are expected (the columns share one index).

Per-factor full-sample mean monthly return (decimal): Mkt-RF 0.00597, SMB 0.00185, HML
0.00296, RMW 0.00257, CMA 0.00243, RF 0.00363, Mom 0.00620, ST_Rev 0.00632, LT_Rev
0.00279 — all the right magnitude and sign for these series.

---

## 4. How the two sources line up

- **OSAP returns ∩ French factors = 1187 months, 1926-02 .. 2024-12.** The binding
  overlap **end is the OSAP wide-file tail, 2024-12** (French extends to 2026-04).
- For anything using the **FF5 risk model**, the common usable span is **1963-07 ..
  2024-12**.
- Join key: OSAP predictor `Acronym` ↔ `oap_signaldoc.csv['Acronym']` (212/212 matched).
  French factors join on the month-end `date` index.

---

## 5. Reproduce / refresh

```bash
./.venv/bin/python scripts/16_fetch_factor_zoo.py        # idempotent; overwrites
```

Network on this host is **Python-only** (the bash sandbox has no egress; all HTTP is via
`requests` / the OSAP package in-process — mirrors the Paper-1/Paper-6 fetch approach).
If the OSAP Drive files are quota-walled, retry later or use the manual fallback in §2.3.

## 6. Known caveats (carry into the audit)

1. **Units are DECIMAL** everywhere. Do not re-divide; do not assume percent.
2. **OSAP returns are GROSS** long-short portfolio returns (equal-weight deciles in the
   wide file). The "net of costs" half of Paper 4 must *impose* a cost model — turnover ×
   spread/impact — on top of these; the file itself carries no cost adjustment. OSAP does
   publish liquidity-screened / value-weighted portfolio variants
   (`PredictorAltPorts_*` on the same Drive release) if a more tradable baseline is
   wanted later.
3. **Staggered predictor starts (~31% NaN)** — respect each predictor's first-valid date;
   do not treat NaN as zero.
4. **Define OOS per-anomaly** using `SampleEndYear` from SignalDoc (ends 1968–2014), not a
   single global cutoff.
5. **`Cat.Signal` uses Predictor/Placebo/Drop** in this release (212/114/5), not the older
   clear/likely/maybe taxonomy. The 114 **Placebos** are a built-in null distribution —
   useful as a control arm for the survival audit (not in the returns file, which is the
   212 Predictors only; pull them via the alt-ports/firm-char route if needed).
6. **FF5 starts 1963-07.** Constrain full-factor-model benchmarks accordingly.
7. **OSAP return tail = 2024-12** (the Oct-2024 wide-file vintage), while French runs to
   2026-04 — the audit's OSAP-side OOS ends 2024-12.
