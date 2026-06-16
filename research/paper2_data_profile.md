# Paper 2 — Crypto Carry: funding + spot data profile

**Generated:** 2026-06-16 22:41:58 UTC by `scripts/11_backfill_funding.py`
**Output dir:** `/apps/alpha-research/data/funding/`
**Registry entry:** `configs/data_sources.yaml -> crypto_funding_backfill`

---

## 1. What this backfill is and why it exists

The pre-existing funding panel (`/apps/crypto-trader/data/funding_history/`, 228 perps)
is **Hyperliquid-sourced**: hourly cadence, a `premium` column, and — for the majors —
only reaches **~2023-06**. It is therefore not a clean instrument for a *decay test of
the 2024 carry boom* (wrong exchange/cadence for a Binance study, and the early-2023
window is incomplete for several names).

This step builds a fresh, cross-sectionally comparable **Binance USD-M perpetual
funding** cross-section plus **matched Binance spot daily close**, spanning **2023-01 ->
2026**, for the top ~30 liquid USDT perps.

## 2. Source — IMPORTANT deviation from the brief

The brief specified the live REST endpoints
`https://fapi.binance.com/fapi/v1/fundingRate` (funding) and
`https://api.binance.com/api/v3/klines` (spot). **Both return HTTP 451 "Service
unavailable from a restricted location" from this host**, and every live mirror tested
(`api1`, `api-gcp`, `data-api` fapi, `binance.us` futures) is 451 or 404. Binance
geo-blocks its live trading API here.

**Working source actually used:** the official **Binance public _data archive_** at
`https://data.binance.vision`, which is **not** geo-blocked and serves the *identical*
Binance USD-M funding series as monthly CSV-in-ZIP files (no API key, no 429/418,
checksummed). This is strictly more reliable than paginating the REST endpoint.

- Funding (8h): `data/futures/um/monthly/fundingRate/<SYM>/<SYM>-fundingRate-YYYY-MM.zip`
  → cols `calc_time(ms), funding_interval_hours, last_funding_rate`.
- Spot (1d): `data/spot/{monthly,daily}/klines/<SYM>/1d/...zip` → kline; we keep `open_time -> time`, `close`.

The current calendar month is published only after it closes, so the **funding tail ends
at the last completed month = 2026-05-31**. Spot daily archives are current to T-1, so
**spot runs to 2026-06-15**. (Both fully span the 2024 boom regardless.)

## 3. Outputs

| file | rows | notes |
|---|---|---|
| `<COIN>USDT_funding.parquet` (x30) | per-symbol | cols `time`(UTC), `funding_rate`; 8h (TIA 4h) |
| `<COIN>USDT_spot1d.parquet` (x30) | per-symbol | cols `time`(UTC), `close` |
| `panel_funding_8h.parquet` | 112,853 | long: `time, symbol, funding_rate`; 30 symbols; 0 NaN; 0 dup(time,symbol) |
| `panel_funding_1d.parquet` | 36,677 | daily MEAN funding per symbol: `time, symbol, funding_rate` |
| `_coverage.json` | — | machine-readable per-symbol coverage |

## 4. Per-symbol coverage

`miss` = estimated missing 8h obs vs a uniform-interval expectation; `gaps` = count of
intra-series gaps > 8h; `maxgap` = largest gap (== the funding interval when there are no holes).

| symbol | funding first | funding last | n obs | miss | gaps | maxgap | spot first | spot last | spot n | note |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| ETHUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| SOLUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| BNBUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| XRPUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| DOGEUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| ADAUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| AVAXUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| LINKUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| TRXUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| DOTUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| MATICUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2024-09-10 | 619 | spot delisted 2024-09-10 (POL rename); funding pinned flat after ~2024-10 |
| LTCUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| BCHUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| NEARUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| APTUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| ARBUSDT | 2023-03-23 | 2026-05-31 | 3497 | 0 | 0 | 8h | 2023-03-23 | 2026-06-15 | 1181 | late lister |
| OPUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| SUIUSDT | 2023-05-03 | 2026-05-31 | 3373 | 0 | 0 | 8h | 2023-05-03 | 2026-06-15 | 1140 | late lister |
| INJUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| TIAUSDT | 2023-10-31 | 2026-05-31 | 5659 | 0 | 0 | 4h | 2023-10-31 | 2026-06-15 | 959 | 4h interval (2x rows) |
| SEIUSDT | 2023-08-16 | 2026-05-31 | 3058 | 0 | 0 | 8h | 2023-08-15 | 2026-06-15 | 1036 | late lister |
| RUNEUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| FILUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| ATOMUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| UNIUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| AAVEUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| ETCUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| XLMUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |
| ALGOUSDT | 2023-01-01 | 2026-05-31 | 3741 | 0 | 0 | 8h | 2023-01-01 | 2026-06-15 | 1262 |  |

**Universe outcome:** 30 / 30 requested symbols backfilled. **None dropped** — every coin
in the universe has a Binance USD-M `<COIN>USDT` perp. **Median earliest date = 2023-01-01.**
**Total funding observations (8h panel) = 112,853.**

- **26 / 30** reach the 2023-01-01 floor with **zero gaps**.
- **4 late listers** captured at their true Binance listing date: **ARB 2023-03-23, SUI
  2023-05-03, SEI 2023-08-16, TIA 2023-10-31**. These are genuine listing dates, not data holes.
- **No >8h gaps anywhere** — `maxgap` equals the funding interval for every symbol.

## 5. Known data caveats (must read before use)

1. **TIA is on a 4-HOUR funding interval** (all 29 others are 8h, verified at every month
   sampled). TIA therefore has ~2x rows/day (n=5659). **Annualise TIA at 6 settlements/day
   (not 3); do NOT assume a uniform 8h grid across the raw panel.** `panel_funding_1d`
   averages within-day and is comparable across all symbols.
2. **MATIC is a delisting/rename artifact.** `MATICUSDT_funding` runs to 2026-05-31, but
   from **~2024-10** the rate is **pinned to the flat default +0.0001/8h** (no real price
   discovery), and the **MATICUSDT _spot_ pair delisted 2024-09-10** (MATIC->POL rename;
   matched spot n=619). **Treat MATIC carry as effectively ending ~2024-09.** `POLUSDT`
   exists in the archive if a continuation series is desired.
3. **Funding tail = 2026-05-31** (last published monthly archive); spot tail = 2026-06-15.

## 6. Do we now span the 2024 carry boom?  — YES (explicit)

**Yes. The panel fully spans the 2024 crypto carry boom and now permits a decay test.**
The 8h funding floor is 2023-01-01 for 26/30 names (the other 4 list in H1-H2 2023), and
all series run continuously through 2026-05-31 with zero gaps. The 2024 calendar year is
covered in full for every symbol.

Cross-sectional **annualized carry** (mean 8h funding x 3 settlements/day x 365; TIA x6),
by year, makes the boom-and-decay visible directly in the data:

| year | median across universe | mean across universe |
|---|---|---|
| 2023 | +7.3% | +5.6% |
| **2024** | **+13.4%** | **+11.0%** |
| 2025 | +2.8% | +1.2% |
| 2026 (to May) | +0.6% | -1.6% |

The cross-sectional median carry roughly **doubles into the 2024 boom (~13%/yr) and then
decays by ~5-10x by 2025-2026** — the exact regime shift Paper 2's decay test targets.

Per-symbol annualized carry by year (TIA on 4h basis; MATIC 2025-26 values are the stale
flat-default artifact noted above):

| symbol | 2023 | 2024 | 2025 | 2026(→May) |
|---|---|---|---|---|
| BTCUSDT | +7.9 | +11.9 | +5.1 | +0.9 |
| ETHUSDT | +8.3 | +13.0 | +4.9 | +0.4 |
| SOLUSDT | +1.3 | +13.6 | +0.4 | -3.2 |
| BNBUSDT | -8.3 | -3.3 | -2.1 | +1.8 |
| XRPUSDT | +8.2 | +14.2 | +3.5 | -1.8 |
| DOGEUSDT | +10.0 | +14.0 | +4.3 | +1.1 |
| ADAUSDT | +7.2 | +14.0 | +4.1 | +0.7 |
| AVAXUSDT | +7.5 | +10.5 | +0.1 | -0.3 |
| LINKUSDT | +10.2 | +13.3 | +5.1 | +3.8 |
| TRXUSDT | -3.6 | +3.4 | -0.5 | -4.9 |
| DOTUSDT | +1.6 | +11.5 | +1.1 | -10.5 |
| MATICUSDT | +7.2 | +14.3 | +11.0 | +11.0 |
| LTCUSDT | +9.3 | +14.4 | +5.6 | +2.0 |
| BCHUSDT | -9.0 | -1.1 | -2.6 | -4.7 |
| NEARUSDT | +6.9 | +14.0 | +2.1 | +2.8 |
| APTUSDT | +2.6 | +16.3 | -12.7 | -11.5 |
| ARBUSDT | +10.9 | +15.9 | +4.9 | +2.6 |
| OPUSDT | +9.2 | +16.0 | +3.2 | -2.1 |
| SUIUSDT | +4.3 | +10.3 | +3.4 | +1.7 |
| INJUSDT | -11.8 | +14.4 | +0.8 | -20.5 |
| TIAUSDT | +32.9 | -9.7 | -4.3 | +1.9 |
| SEIUSDT | +0.0 | +6.0 | -15.0 | -11.1 |
| RUNEUSDT | +0.3 | +15.5 | +3.9 | +2.5 |
| FILUSDT | +11.5 | +15.6 | +2.5 | -1.7 |
| ATOMUSDT | +1.3 | +8.3 | -9.0 | -12.7 |
| UNIUSDT | +9.9 | +13.5 | +6.3 | +2.4 |
| AAVEUSDT | +9.7 | +11.0 | +5.3 | +1.9 |
| ETCUSDT | +10.2 | +14.8 | +4.3 | +1.6 |
| XLMUSDT | +4.2 | +11.8 | -2.3 | -1.5 |
| ALGOUSDT | +8.2 | +11.8 | +2.0 | -1.6 |

## 7. Reproduce / refresh

```
./.venv/bin/python scripts/11_backfill_funding.py            # all 30, ~few min
./.venv/bin/python scripts/11_backfill_funding.py --workers 1  # serial (max politeness)
```
Idempotent: re-downloads and overwrites. Polite to the archive (small per-request sleep,
concurrency-limited, exponential backoff on 429/418/5xx).
