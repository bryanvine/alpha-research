# Paper 1 — Data Profile: The Volatility Risk Premium, Cross-Asset

**Date:** 2026-06-16
**Author:** data-foundation pass (alpha-research)
**Scope:** Verify the two prior-review data risks, profile the Deribit crypto-vol data,
locate a realized-vol source spanning the DVOL window, and acquire the free equity-VRP
benchmark data.

---

## 0. Executive summary (the four questions)

1. **BTC & ETH have DVOL + matching realized-vol coverage over the FULL 2021–2026 window — YES**, but only via the Deribit index-price parquet.
   - DVOL (risk-neutral 30d implied): `dvol_{BTC,ETH}.parquet` — 2021-03-24 → 2026-06-16, 12h bars, zero gaps.
   - Realized vol over the **full** span: **only** `price_{BTC,ETH}.parquet` (Deribit index, OHLCV @ 12h) reaches back to 2021-03. All 1h sources (jepa `bars_1h.csv`, crypto TimescaleDB `bars`) start **2025-01-30**; deepest sub-daily anywhere is 15m-binance from **2024-03-07**. So **DATA RISK #1 is CONFIRMED** for high-frequency bars, but **refuted at 12h** — we can still compute 30d realized variance over the entire DVOL span from the Deribit price file.

2. **Surface = single end-of-sample SNAPSHOT, not a time series — DATA RISK #2 CONFIRMED.**
   `surface_{BTC,ETH}.json` each hold one cross-section of live contracts (no timestamp field, stale spot index, all expiries in the future). Unusable for a time-series options-execution-cost backtest as-is.

3. **Equity VRP benchmark downloaded** (all succeeded): FRED VIXCLS, VXVCLS(VIX3M) + 7 other VIX-family series; Cboe CDN VIX/VIX9D/VIX3M/VIX6M/VVIX/SKEW + benchmark indices PUT/BXM/BXMD/WPUT/PPUT. Spans below.

4. **Blockers:** (a) crypto option **surface time series** must be re-collected from Deribit (the existing snapshot can't price the options leg historically); (b) **sub-daily realized vol pre-2024 is unavailable** — the 12h Deribit close is the only RV proxy for 2021-2023, so HF realized-vol estimators (e.g. 5-min RV) are off the table for the early sample; (c) TimescaleDB is **not host-reachable** (internal port only). Details in §5.

---

## 1. Environment

- venv: `/apps/alpha-research/.venv` (Python 3.12.3, stdlib `venv`; `uv` not installed).
- Installed: numpy, pandas (3.0.3), scipy (1.17.1), pyarrow (24.0.0), statsmodels (0.14.6), matplotlib, requests, pyyaml.
- Verified: `/apps/alpha-research/.venv/bin/python -c 'import pyarrow,pandas,scipy,statsmodels'` → OK.
- Profiler script: `/apps/alpha-research/scripts/profile_deribit.py`.

---

## 2. Deribit data profile  (`/apps/jepa-trader/data/raw_deribit/`)

### 2.1 DVOL — risk-neutral 30d implied vol (the harvest's short leg)

| file | rows | cols | time span (UTC) | cadence | gaps | dups | NaN | DVOL close range (vol pts) |
|---|---|---|---|---|---|---|---|---|
| `dvol_BTC.parquet` | 3822 | ts(int64 ms), dvol_o/h/l/c(f64), time(datetime ms,UTC) | 2021-03-24 00:00 → 2026-06-16 12:00 | 12h (median Δ, mode 12h) | 0 | 0 | 0 | 32.2 – 156.2 (high 167.8) |
| `dvol_ETH.parquet` | 3822 | same | 2021-03-24 00:00 → 2026-06-16 12:00 | 12h | 0 | 0 | 0 | 30.5 – 196.9 (high 206.4) |

- Span = 1910.5 days; expected rows @ 12h ≈ 3821 → **observed 3822 = effectively complete, no missing bars.**
- **Value sanity: PASS.** DVOL is in annualized vol points; BTC ~32–168, ETH ~30–206 — within the expected ~10–200 range (crypto runs hotter than equities; ETH 206 spike is plausible for a 2021/2022 stress bar). Medians BTC ≈ 56, ETH ≈ 70.

### 2.2 Index price (OHLCV — the harvest's realized-vol leg)

| file | rows | cols | time span (UTC) | cadence | gaps | NaN | close range |
|---|---|---|---|---|---|---|---|
| `price_BTC.parquet` | 3822 | ts, open, high, low, close, volume, time | 2021-03-23 20:00 → 2026-06-16 08:00 | 12h | 0 | 0 | $15,400 – $125,500 |
| `price_ETH.parquet` | 3822 | same | 2021-03-23 20:00 → 2026-06-16 08:00 | 12h | 0 | 0 | $877 – $4,957 |

- **Value sanity: PASS.** BTC $15.4k–$126k and ETH $877–$4,957 bracket the real 2021–2026 ranges. Volume positive throughout. **OHLC present** → Garman–Klass / Parkinson estimable at 12h, not just close-to-close.
- RV check (this profile): annualized close-to-close full-sample RV = BTC 54.8%, ETH 73.3%; 30d-rolling RV median BTC 49.7% / ETH 64.7%; 30d Parkinson median BTC 53.3% / ETH 69.0%. These sit just **below** the DVOL medians (BTC 56, ETH 70) → a small **positive** implied-minus-realized VRP, consistent with the harvest thesis.

### 2.3 Option surfaces — SNAPSHOT, NOT A TIME SERIES  (DATA RISK #2 = CONFIRMED)

`surface_BTC.json` / `surface_ETH.json` top-level shape: `{currency, index, n, options[]}`.

| file | n contracts | fields per contract | "index" (spot) | expiry range | strike range | moneyness | mark_iv (vol pts) | timestamp field? |
|---|---|---|---|---|---|---|---|---|
| `surface_BTC.json` | 932 | instrument, exp, strike, cp, mark_iv, moneyness | 65,691.31 | 2026-06-17 → 2027-03-26 (11 expiries) | 20,000 – 380,000 | 0.30 – 5.79 | 28.2 – 172.8 | **none** |
| `surface_ETH.json` | 762 | same | 1,778.24 | 2026-06-17 → 2027-03-26 (11 expiries) | 500 – 16,000 | 0.28 – 9.00 | 46.7 – 183.9 | **none** |

- **It is a single live snapshot.** No `timestamp`/`creation`/`as_of` field exists anywhere in the file; there is exactly one `index` (spot) value per currency, and **every expiry is in the future** relative to today (2026-06-16). The snapshot spot ($65,691 BTC / $1,778 ETH) matches the last `bars_1h.csv` BTC-USD/ETH-USD close (~65,711 / ~1,769), i.e. it was captured at end-of-pull, not historically.
- **Implication:** options-level execution-cost modelling (bid/ask, IV smile, delta) is only available for one instant. A historical options-execution backtest needs a re-collected **surface time series** (see Blockers).

`summary.json` corroborates: per coin `dvol_rows=3822`, `price_rows=3822`, `surface_opts=932/762`, plus the same single spot index.

---

## 3. Realized-vol source over the DVOL window (2021-03 → 2026-06)

Checked, in priority order:

| source | symbols | granularity | span | rows (BTC) | cols | gaps | covers 2021? |
|---|---|---|---|---|---|---|---|
| **`/apps/jepa-trader/data/raw_deribit/price_*.parquet`** | BTC, ETH (index) | **12h OHLCV** | **2021-03-23 → 2026-06-16** | 3822 | O/H/L/C/V | **0** | **YES** |
| `/apps/jepa-trader/data/raw_crypto/bars_1h.csv` | BTC-USD, ETH-USD (+~50 coins) | 1h OHLCV | 2025-01-30 → 2026-06-16 | 14,588 | O/H/L/C/V | n/a | NO |
| crypto TimescaleDB `bars` (15m binance) | BTC-USD, ETH-USD | 15m | 2024-03-07 → 2026-03-07 | 70,080 | O/H/L/C/V | — | NO (only to 2024-03) |
| crypto TimescaleDB `bars` (1h/4h/1d) | BTC-USD, ETH-USD | 1h/4h/1d | 2025-01-30 → 2026-06-16 | 8,923 (1h) | O/H/L/C/V | — | NO |
| `/apps/crypto-trader/data/historical/*_kraken.csv` | BTC, ETH, +8 | 1h/4h/1d | rolling 721-row windows (1h ≈ last 30d; 1d from 2024-01-29) | 721 | O/H/L/C/V | — | NO |

**CONCLUSION.** We **can** compute 30d realized variance over the full DVOL span for **both BTC and ETH**, but **only** from `price_{BTC,ETH}.parquet` (Deribit index, 12h OHLCV, zero gaps). 60×12h-bars = 30 days; close-to-close, Parkinson, and Garman–Klass are all feasible (OHLC present).

**Fallback note (DATA RISK #1).** Higher-frequency bars are a recent phenomenon: 1h only exists from **2025-01-30** (jepa `bars_1h.csv` and the TimescaleDB), and the deepest sub-daily history anywhere is **15m binance from 2024-03-07**. For 2021–2024 the **12h Deribit close is the finest available RV input** — true intraday/HF realized-vol estimators (5-min RV, etc.) are not possible for the early sample. For 2024-03+ (15m) and 2025+ (1h), HF realized vol can optionally be layered in for robustness checks.

---

## 4. Equity VRP benchmark inventory  (`/apps/alpha-research/data/equity_vol/`)

All downloads on 2026-06-16 succeeded. Daily EOD. Columns: FRED = `observation_date,<ID>`; Cboe = `DATE,[OPEN,HIGH,LOW,]CLOSE` or `DATE,<SYM>`.

### 4.1 FRED (no API key) — base URL `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`

| file | series | span | rows |
|---|---|---|---|
| `vixcls.csv` | VIXCLS (VIX, 30d) | 1990-01-02 → 2026-06-15 | 9510 |
| `vxvcls.csv` | VXVCLS (VIX 3-month) | 2007-12-04 → 2026-06-15 | 4835 |
| `fred_VXDCLS.csv` | VXD (DJIA vol) | 1997-10-07 → 2026-06-15 | 7486 |
| `fred_VXNCLS.csv` | VXN (Nasdaq-100 vol) | 2001-02-02 → 2026-06-15 | 6618 |
| `fred_VXOCLS.csv` | VXO (old S&P100 vol) | 1986-01-02 → 2021-09-23 (discontinued) | 9321 |
| `fred_OVXCLS.csv` | OVX (crude-oil VIX) | 2007-05-10 → 2026-06-15 | 4983 |
| `fred_GVZCLS.csv` | GVZ (gold VIX) | 2008-06-03 → 2026-06-15 | 4705 |
| `fred_EVZCLS.csv` | EVZ (EUR/USD VIX) | 2007-11-01 → 2025-03-11 | 4529 |
| `fred_VXTYN.csv` | VXTYN (10y Treasury-note vol) | 2003-01-02 → 2020-05-15 (discontinued) | 4532 |

> No `VIX9D`/`VIX6M`/`VVIX`/`SKEW` IDs exist on FRED (404) — those come from Cboe (below). FRED IDs `VIXN`, `VIX9D` returned 404 (removed).

### 4.2 Cboe CDN (no key) — base URL `https://cdn.cboe.com/api/global/us_indices/daily_prices/<SYM>_History.csv`

Volatility / term-structure / skew indices:

| file | index | span | rows | cols |
|---|---|---|---|---|
| `cboe_VIX.csv` | VIX | 1990-01-02 → 2026-06-15 | 9207 | OHLC |
| `cboe_VIX9D.csv` | VIX9D (9-day) | 2011-01-04 → 2026-06-15 | 3884 | OHLC |
| `cboe_VIX3M.csv` | VIX3M (3-month) | 2009-09-18 → 2026-06-15 | 4210 | OHLC |
| `cboe_VIX6M.csv` | VIX6M (6-month) | 2008-01-02 → 2026-06-15 | 4642 | OHLC |
| `cboe_VVIX.csv` | VVIX (vol-of-vol) | 2006-03-06 → 2026-06-15 | 5041 | DATE,VVIX |
| `cboe_SKEW.csv` | SKEW | 1990-01-02 → 2026-06-15 | 9164 | DATE,SKEW |

CBOE option-writing benchmark indices (the "is equity VRP dead net?" workhorses for H6):

| file | index | span | rows |
|---|---|---|---|
| `cboe_PUT.csv` | PUT (S&P 500 PutWrite) | 1991-03-04 → 2026-06-15 | 4900 |
| `cboe_BXM.csv` | BXM (S&P 500 BuyWrite) | 2002-03-22 → 2026-06-15 | 6094 |
| `cboe_BXMD.csv` | BXMD (BuyWrite 30-delta) | 1986-06-20 → 2026-06-15 | 10068 |
| `cboe_WPUT.csv` | WPUT (Weekly PutWrite) | 2006-01-31 → 2026-06-15 | 5122 |
| `cboe_PPUT.csv` | PPUT (5% Put-Protection) | 1986-06-30 → 2026-06-15 | 10062 |

All Cboe and FRED series run **through 2026-06-15**, fully overlapping the crypto DVOL window (the crypto sample starts 2021-03-24, so VIX/PUT/BXM all cover it with deep pre-history for the long-run "dead net" baseline).

---

## 5. BLOCKERS

1. **Crypto option SURFACE is a snapshot, not a time series (HIGH — blocks the options-execution-cost test).**
   `surface_{BTC,ETH}.json` is a single live cross-section (no timestamp; all expiries future-dated; stale spot). The DVOL-vs-RV **time-series** harvest is fully unblocked (§2–3), but pricing the **options-execution** leg historically (bid/ask spreads, smile, delta-hedge slippage) needs a re-collected Deribit surface **time series** (periodic snapshots, ideally with bid/ask IV and greeks, over 2021–2026 or at least a representative live forward-collection window). *Action:* schedule a Deribit surface collector, or buy/obtain historical surface data.

2. **No sub-daily realized vol before 2024-03 (MEDIUM — bounds RV-estimator choice).**
   Finest RV input for 2021-03→2024-03 is the **12h Deribit close** (close-to-close / Parkinson / Garman–Klass at 12h). 15m exists only from 2024-03-07 (binance), 1h only from 2025-01-30. HF realized-vol estimators (5-min RV, realized kernels) are infeasible for the early sample; the main result should use the 12h estimator end-to-end, with HF RV as a post-2024 robustness check only.

3. **TimescaleDB not reachable from host (LOW — worked around).**
   `crypto-timescaledb` (user `trading`, db `crypto`, **port 12432 inside the container**, `listen_addresses='*'`) is healthy but the port is **not published to the host** (host `localhost:12432`/`16432` refused; no `psql` client installed). Reached it via `docker exec`. It adds nothing beyond the files for Paper 1 (its BTC/ETH `bars` start 2024-03 at best). The yaml `port: 12432` is correct for *inside* the container; mark host-access as requiring a published port or a docker-exec tunnel.

4. **No equity-options *surface* / per-strike data, only indices (LOW for H6).**
   We have VIX-family indices + CBOE writing-strategy total-return indices — sufficient to characterise the (largely dead) net equity VRP via VIX−RV spreads and PUT/BXM excess returns. We do **not** have an equity index-option surface for an apples-to-apples options-execution-cost comparison vs Deribit; flagged as out of scope unless cross-asset execution parity is required (the yaml `equity_options_vix` TODO remains open for that).

---

## 6. Provenance / repro

- Deribit profile: `/apps/alpha-research/.venv/bin/python /apps/alpha-research/scripts/profile_deribit.py`
- Equity downloads: `curl -sL` to the FRED and `cdn.cboe.com` URLs in §4 (no auth).
- TimescaleDB: `docker exec crypto-timescaledb` → `psql -h 127.0.0.1 -p 12432 -U trading -d crypto`.
- Registry updated: `/apps/alpha-research/configs/data_sources.yaml` (`datasets.equity_vol`).
