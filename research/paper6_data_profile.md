# Paper 6 — Data Profile: FX Carry & Commodity Roll Yield

**Date:** 2026-06-16
**Author:** data-foundation pass (alpha-research)
**Scope:** Source FREE FX (G10 carry + value) and commodity (price + roll-yield/basis)
data, normalize to documented conventions, register in `configs/data_sources.yaml`,
and profile coverage + blockers. All fetches: no API key.

**Fetcher:** `scripts/10_fetch_fx_commodity.py` (idempotent; re-run to refresh).
**Raw data:** `data/fx/` (51 files, ~6.1M), `data/commodity/` (13 files, ~1.1M).
**Manifest (machine-readable provenance):** `data/_paper6_manifest.json`.
**HTTP note:** the bash sandbox on this host has **no network**; only Python `requests`
egress works. All downloads are done in-process with `requests` (mirrors the Paper-1
fetch). `curl` from a shell returns `http=000` here — not a data problem.

---

## 0. Executive summary

| Block | Got it? | Span | Notes |
|---|---|---|---|
| **FX spot** (G10, 9 ccys) | **YES** | 1971-01 → 2026-06 daily (EUR from 1999) | All normalized to **foreign-per-USD**; native col kept for audit |
| **FX carry** (3M diff vs USD) | **YES** | 1994-01 → 2026-05 monthly (per-ccy starts vary) | OECD 3M interbank − USD; USD daily `DGS3MO` also stored |
| **FX value** (BIS REER) | **YES** | broad 1994-01→2026-04, narrow 1964-01→2026-04 (monthly) | 9 ccys + euro-area; higher = real-expensive |
| **FX positioning** (CFTC COT) | **YES (bonus)** | 2000/2004 → 2026-06 weekly | spec net length; old+new market names stitched |
| **Commodity price** (energy daily) | **YES** | WTI 1986, Brent 1987, Henry Hub 1997 → 2026-06 daily | front-month spot |
| **Commodity price** (metals/ags monthly) | **YES** | 1992-01 → 2026-05 monthly | IMF copper/maize/wheat/EU-gas/Brent/all-comm index |
| **Commodity positioning** (COT) | **YES (bonus)** | WTI 2000, Gold 1986, HH-gas 2011 → 2026-06 weekly | incl. **GOLD** (offsets missing gold spot) |
| **Commodity ROLL YIELD** (front+second futures) | **NO — BLOCKED** | — | **Binding constraint for Paper 6. See Blocker #1.** |
| **Gold spot $/oz** (FRED) | **NO — BLOCKED** | — | All FRED $/oz IDs discontinued (404). See Blocker #2. |

**Bottom line for Paper 6.** The **FX carry+value** leg is fully sourced and clean
(spot, rate differential, REER, plus COT positioning). The **commodity roll-yield**
leg is the problem: **roll yield needs front + second contract (or spot-vs-excess-return),
and every free continuous-futures feed is dead or walled** (Quandl CHRIS 403; Stooq
anti-bot; FRED BCOM/GSCI total-return 404; EIA needs a key). We have **front-month
spot only** for energy. **Roll yield is therefore NOT computable from free sources as
delivered** — this is Blocker #1 and the single most important finding for Paper 6.

---

## 1. FX — spot (the normalized panel)

**FRED daily USD bilateral rates** — base URL
`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`.

### Sign convention (DOCUMENTED, enforced in code)
FRED publishes different pairs in **different USD directions**. We normalize **every
series to foreign-currency units per 1 USD** (`<CCY>_per_USD`):
- IDs quoted "USD per 1 foreign" (`DEXUSEU`, `DEXUSUK`, `DEXUSAL`, `DEXUSNZ`) are **inverted** (store `1/rate`).
- IDs quoted "foreign per 1 USD" (`DEXJPUS`, `DEXCAUS`, `DEXSZUS`, `DEXSDUS`, `DEXNOUS`) are kept as-is.
- **A RISE in the normalized series = USD appreciates / foreign currency depreciates.**
- Each raw `spot_<CCY>.csv` keeps the untouched native column (`<CCY>_native`) so the transform is auditable.

**Normalization verified** (last obs 2026-06-12, each within ratio ≈1.00 of the known
spot): EUR 0.864, JPY 160.24, GBP 0.746, CAD 1.397, AUD 1.419, CHF 0.796, SEK 9.414,
NOK 9.500, NZD 1.716.

| file | ccy | FRED id | native convention | span | rows (valid) |
|---|---|---|---|---|---|
| `data/fx/spot_EUR.csv` | EUR | DEXUSEU | USD per FX → inverted | 1999-01-04 → 2026-06-12 | 6883 |
| `data/fx/spot_JPY.csv` | JPY | DEXJPUS | FX per USD | 1971-01-04 → 2026-06-12 | 13898 |
| `data/fx/spot_GBP.csv` | GBP | DEXUSUK | USD per FX → inverted | 1971-01-04 → 2026-06-12 | 13904 |
| `data/fx/spot_CAD.csv` | CAD | DEXCAUS | FX per USD | 1971-01-04 → 2026-06-12 | 13910 |
| `data/fx/spot_AUD.csv` | AUD | DEXUSAL | USD per FX → inverted | 1971-01-04 → 2026-06-12 | 13897 |
| `data/fx/spot_CHF.csv` | CHF | DEXSZUS | FX per USD | 1971-01-04 → 2026-06-12 | 13904 |
| `data/fx/spot_SEK.csv` | SEK | DEXSDUS | FX per USD | 1971-01-04 → 2026-06-12 | 13903 |
| `data/fx/spot_NOK.csv` | NOK | DEXNOUS | FX per USD | 1971-01-04 → 2026-06-12 | 13903 |
| `data/fx/spot_NZD.csv` | NZD | DEXUSNZ | USD per FX → inverted | 1971-01-04 → 2026-06-12 | 13888 |

**Combined panel:** `data/fx/panel_spot_FXperUSD.csv` — cols `date,EUR,JPY,GBP,CAD,AUD,CHF,SEK,NOK,NZD`, 14465 rows (1971-01-04 → 2026-06-12). EUR is NaN before 1999 (euro didn't exist); ~560 NaNs/ccy over 55y are US/foreign market holidays (clean — not gaps).

---

## 2. FX — carry (3M interest-rate differential vs USD)

- **USD leg (daily):** `DGS3MO` (3M Treasury CMT), 1981-09-01 → 2026-06-15, 11685 rows → `data/fx/rate_USD_DGS3MO.csv`.
- **Foreign legs (monthly):** OECD 3M interbank rate, FRED id `IR3TIB01<CC>M156N`. **All 9 worked**, plus the US (`IR3TIB01USM156N`) for an OECD-consistent differential.

| ccy | FRED id | span | rows |
|---|---|---|---|
| EUR (euro area) | IR3TIB01EZM156N | 1994-01 → 2026-01 | 385 |
| JPY | IR3TIB01JPM156N | 2002-04 → 2026-04 | 289 |
| GBP | IR3TIB01GBM156N | 1957-01 → 2026-01 | 829 |
| CAD | IR3TIB01CAM156N | 1956-01 → 2026-05 | 845 |
| AUD | IR3TIB01AUM156N | 1968-01 → 2026-05 | 701 |
| CHF | IR3TIB01CHM156N | 1999-07 → 2026-05 | 323 |
| SEK | IR3TIB01SEM156N | 1982-01 → 2026-05 | 533 |
| NOK | IR3TIB01NOM156N | 1979-01 → 2026-05 | 569 |
| NZD | IR3TIB01NZM156N | 1973-12 → 2026-05 | 630 |
| USD | IR3TIB01USM156N | 1964-06 → 2026-05 | 744 |

**Differential panel:** `data/fx/panel_carry_diff_OECD3M_vs_USD.csv` — `diff_ccy = rate_foreign_OECD3M − rate_USD_OECD3M` (percentage points; **+ve ⇒ foreign pays more ⇒ long-carry target**), monthly, 743 rows (1964-06 → 2026-05).

**Sign sanity PASS:** JPY diff ∈ [−5.47, +0.30], CHF ∈ [−4.60, −0.02] (classic **funding** currencies, ≤0 vs USD); AUD ∈ [−8.46, +12.37], NZD ∈ [−4.35, +19.89] (classic **carry targets**, large positive). Latest-month diffs: CAD −1.43, CHF −3.76, AUD +0.71, NOK +0.84.

**Caveat (staggered publication):** the OECD EZ/JP/GB 3M series end at 2026-01/04 while CAD/AUD/etc. reach 2026-05, so the most recent 1–4 monthly rows are NaN for EUR/JPY/GBP in the panel (a leg simply hasn't published yet). EUR is the **euro-area aggregate**; JPY carry starts only 2002-04 and CHF 1999-07. For a daily carry signal you must forward-fill the monthly foreign rate against the daily USD `DGS3MO` (left as a modeling choice, not baked in).

---

## 3. FX — value (BIS real effective exchange rate, REER)

BIS REER mirrored on FRED, monthly, **index 2020 = 100**. Both baskets pulled:

- **Broad** (`RB<CC>BIS`, wider/modern basket, from 1994) — **primary**.
- **Narrow** (`RN<CC>BIS`, from 1964) — long-history robustness.
- All 9 G10 + **euro area (`XM`)** worked. `RBEUBIS`/`RNEUBIS` 404 → euro area uses `XM` (mapped to EUR).

| panel | file | cols | span | rows |
|---|---|---|---|---|
| broad | `data/fx/panel_reer_broad.csv` | USD,JPY,GBP,CAD,AUD,CHF,SEK,NOK,NZD,EUR | 1994-01 → 2026-04 | 388 |
| narrow | `data/fx/panel_reer_narrow.csv` | (same 10) | 1964-01 → 2026-04 | 748 |

Per-ccy raw files: `data/fx/reer_{broad,narrow}_<CCY>.csv`.

**Value signal convention:** higher REER = currency relatively **expensive** in real terms (mean-reversion → short the rich, long the cheap). **Sanity PASS** (broad, latest 2026-04): JPY 65.7 (historically very cheap; basket range max 194 in the 1990s), USD 107.1 / AUD 115.9 / GBP 111.5 (rich). Ranges all economically plausible.

---

## 4. FX — positioning (CFTC COT, bonus)

Legacy futures-only, Socrata public API (no key): `publicreporting.cftc.gov/resource/6dca-aqww.csv`. Weekly. Non-commercial (speculator) net = `noncomm_long − noncomm_short`; also `noncomm_net_pct_oi`. CFTC **renamed several markets ~2022-02**, so each market stitches an OLD + NEW name (verified). Files `data/fx/cot_<CCY>.csv`.

| ccy | span | rows | latest spec net (% OI) |
|---|---|---|---|
| EUR | 2000-08 → 2026-06 | 1346 | +13,932 (+1.6%) |
| JPY | 2000-08 → 2026-06 | 1346 | −145,818 (−28.9%) |
| GBP | 2004-01 → 2026-06 | 1171 | −64,213 (−22.0%) |
| CAD | 2000-08 → 2026-06 | 1346 | −119,999 (−31.3%) |
| AUD | 2004-01 → 2026-06 | 1170 | +18,160 (+5.8%) |
| CHF | 2000-08 → 2026-06 | 1344 | −36,665 (−31.8%) |
| NZD | 2004-08 → 2026-06 | 1085 | −31,571 (−21.2%) |

(NZD stitches `NEW ZEALAND DOLLAR…` → `NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE`; GBP stitches `BRITISH POUND STERLING…` → `BRITISH POUND…`.)

---

## 5. Commodity — prices

### 5.1 Energy front-month / spot (FRED daily)

| file | name | FRED id | units | span | rows | range |
|---|---|---|---|---|---|---|
| `data/commodity/price_WTI.csv` | WTI | DCOILWTICO | $/bbl | 1986-01-02 → 2026-06-08 | 10177 | −36.98 … 145.31 |
| `data/commodity/price_BRENT.csv` | Brent | DCOILBRENTEU | $/bbl | 1987-05-20 → 2026-06-08 | 9907 | 9.10 … 143.95 |
| `data/commodity/price_HENRYHUB.csv` | Henry Hub gas | DHHNGSP | $/MMBtu | 1997-01-07 → 2026-06-08 | 7387 | 1.05 … 30.72 |

**Combined:** `data/commodity/panel_price_daily.csv` (1986-01-02 → 2026-06-08). **Value sanity PASS:** WTI correctly captures the **−$36.98** print on **2020-04-20** (the historic negative-oil day) — confirms this is genuine front-month spot, not a smoothed index. WTI coverage = 96.5% of business days in span (remainder = market holidays); within-span NaNs: WTI 371, Brent 282, HH 288 (holidays only, no internal gaps).

### 5.2 Metals / ags / index (FRED monthly, IMF/PINK series)

| file | name | FRED id | units | span | rows |
|---|---|---|---|---|---|
| `data/commodity/price_COPPER.csv` | Copper | PCOPPUSDM | $/mt | 1992-01 → 2026-05 | 413 |
| `data/commodity/price_MAIZE.csv` | Maize (corn) | PMAIZMTUSDM | $/mt | 1992-01 → 2026-05 | 413 |
| `data/commodity/price_WHEAT.csv` | Wheat | PWHEAMTUSDM | $/mt | 1992-01 → 2026-05 | 413 |
| `data/commodity/price_NGAS_EU.csv` | EU nat-gas | PNGASEUUSDM | $/MMBtu | 1992-01 → 2026-05 | 413 |
| `data/commodity/price_BRENT_M.csv` | Brent (monthly) | POILBREUSDM | $/bbl | 1992-01 → 2026-05 | 413 |
| `data/commodity/price_ALLCOMM_IDX.csv` | All-commodity index | PALLFNFINDEXM | index 2016=100 | 1992-01 → 2026-06 | 414 |

### 5.3 Commodity positioning (CFTC COT, bonus)

`data/commodity/cot_<TAG>.csv` (same schema/source as §4):

| market | span | rows | latest spec net (% OI) |
|---|---|---|---|
| WTI crude | 2000-02 → 2026-06 | 1374 | +19,500 (+9.3%) |
| **Gold** | 1986-01 → 2026-06 | 1920 | +173,837 (+52.2%) |
| Henry Hub gas | 2011-11 → 2026-06 | 756 | +48,516 (+11.6%) |

(WTI stitches `CRUDE OIL, LIGHT SWEET…` → `WTI FINANCIAL CRUDE OIL…`. The **gold** COT partially offsets the missing gold spot — see Blocker #2.)

---

## 6. BLOCKERS

### 1. Commodity ROLL YIELD needs front+second futures — NO FREE SOURCE (HIGH; the binding constraint for Paper 6)
Roll yield / basis requires the **front AND second** contract (or a spot-vs-excess-return pair). Every free continuous-futures path was tested this run and is dead or walled:

| source tried | result | meaning |
|---|---|---|
| Nasdaq Data Link `CHRIS/CME_CL1`, `CHRIS/CME_CL2` (front+second) | **HTTP 403** (robots NOINDEX wall) | Quandl **decommissioned** the free CHRIS continuous tier |
| Stooq `cl.f`, `ng.f`, `gc.f` front-month CSV | **HTTP 200 but JS proof-of-work anti-bot wall** (`async()` token page, not CSV) | same wall Paper-1 hit; not scriptable without a headless browser |
| FRED `BCOMTR` / `SPGSCITR` / `DJUBSTR` (total-return vs spot) | **HTTP 404** | FRED does not carry the commodity TR indices (would have let us back out roll via TR−spot) |
| EIA open-data v2 | **HTTP 403 `API_KEY_MISSING`** | needs a free key; out of scope for "no-key" sourcing |

**Consequence:** with **front-month spot only** (FRED WTI/Brent/Henry Hub), **roll yield is NOT directly computable from free sources as delivered.**

*Mitigations / paths forward (for the next pass, in rough order of effort):*
1. **Register a free Nasdaq Data Link API key** and pull a current continuous-futures table (CHRIS is gone, but the free tier still serves some CME continuous via the v3 API with a key). — likely the cleanest fix.
2. **EIA free API key** (`api.eia.gov`) — gives WTI/Brent/Henry Hub **futures contracts 1–4** (`PET.RCLC1..4`, `NG.RNGC1..4`) as separate daily series → a real front-vs-second roll for energy, no scraping. Trivial to add once a key exists; the brief deferred it only because it isn't *key-free*.
3. **CFTC COT spreading positions** (already downloadable here) are a positioning proxy, **not** a price-based roll yield — do not substitute.
4. A **headless-browser** Stooq fetch could clear the proof-of-work wall, but that is brittle and likely ToS-adjacent — not recommended.

Until one of (1)/(2) lands, Paper 6's commodity leg can do **outright price momentum / level** and **positioning (COT)**, but **not** the roll-yield/basis carry signal that is the paper's thesis.

### 2. No free FRED gold (or silver/platinum) spot price $/oz (MEDIUM)
Every FRED $/oz precious-metal ID is **discontinued (404)**: `GOLDAMGBD228NLBM`, `GOLDPMGBD228NLBM`, `PGOLDUSDM`, `PSILVERUSDM`, `PPLATUSDM`. Survivors are only **producer/CPI indices** (`WPU10210501`, `IQ12260`, `PCU212221212221`) — not tradable $/oz prices. So **gold/metals price level** is missing from the free FRED set (copper $/mt via IMF `PCOPPUSDM` is the one metal we do have). We **do** have **gold COT positioning** (1986→2026) as a partial offset. *Fix:* the same EIA/Nasdaq key route, or Yahoo `GC=F`/`GLD` chart JSON (the Paper-1 work already uses Yahoo v8 chart JSON successfully for ^GSPC — a known-good keyless fallback worth adding next pass).

### 3. FX carry is monthly + staggered (LOW; modeling note)
Foreign short rates (OECD 3M interbank) are **monthly**, and publication lags differ (EZ/JP/GB end 2026-01/04 vs others 2026-05), so the most recent 1–4 differential rows are NaN for some currencies. JPY carry only starts 2002-04, CHF 1999-07. The USD leg is daily (`DGS3MO`). A daily carry signal requires forward-filling the monthly foreign rate (left to the model). No forward-points/FX-swap series were sourced (FRED carries none free); the rate differential is the carry proxy. Dukascopy/HistData tick bid-ask was **not** attempted (known anti-bot / heavy) — flagged as a separate blocker if execution-cost modeling on FX is later required.

---

## 7. Provenance / repro
- Fetch + normalize + QA: `./.venv/bin/python scripts/10_fetch_fx_commodity.py` (writes all CSVs + `data/_paper6_manifest.json`; prints per-series spans and the term-structure probe verdict).
- Registry updated: `configs/data_sources.yaml` → `fx:` and `commodity:` sections (appended).
- All FRED via `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`; COT via `https://publicreporting.cftc.gov/resource/6dca-aqww.csv`. No API keys used.
- 66 series fetched, **0 failures**. Term-structure/roll-yield probes recorded under `commodity.term_structure_probes` in the manifest with `roll_yield_computable_from_free_sources: false`.
