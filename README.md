# Alpha Research

A cross-asset quantitative research program hunting for **genuinely harvestable
alpha** — and reporting honest negative results when there isn't any. It covers
crypto, equities, FX, commodities, and volatility / options, and publishes a
rigorous, reproducible, cost-aware series of papers, in the same spirit as
[`jepa-trader`](https://github.com/bryanvine/jepa-trader).

The unit of output is a paper backed by a fully reproducible pipeline: every
claimed edge must clear a fixed statistical-rigor bar, and every backtest is net
of realistic transaction costs. Negative results are first-class outcomes.

## Paper series

1. **The Volatility Risk Premium, Cross-Asset** — extend jepa-trader's one
   genuine win (Deribit DVOL beating trailing realized vol); is the premium
   harvestable net of costs across crypto and equities?
2. **Crypto Carry: The Funding-Rate Cross-Section** — cross-sectional funding
   carry + basis term structure on 228 perps, net of funding / taker costs.
3. **Statistical Arbitrage in Crypto** — cointegration / PCA stat-arb where
   inefficiency is highest.
4. **The Cost of Direction** — an honest net-of-cost replication audit of the
   directional anomaly zoo (momentum, value, BAB, reversal, seasonality, PEAD).
5. **Liquidity Provision as Alpha** — the LOB / Databento L2 data reframed as
   execution and market-making alpha, not directional prediction.
6. **FX Carry & Commodity Roll Yield** — the most-documented anomalies in
   finance, on newly-sourced FX and commodity data.

## Repository structure

```
alpha_research/          Python package
  data/                  data-access loaders (see configs/data_sources.yaml)
  eval/                  backtest engine + predictive metrics + rigor protocol
  factors/               signal / factor implementations (per paper)
scripts/                 numbered CLI pipeline (see scripts/README.md)
experiments/             results JSON (gitignored)
data/                    datasets (gitignored)
docs/                    GitHub Pages papers (HTML) + shared template
research/                literature review + design notes
configs/                 data_sources.yaml and other config
papers/                  working notes / drafts per paper
```

## Reused from jepa-trader (battle-tested assets)

This project deliberately stands on assets proven out in `jepa-trader`:

- **Rigor protocol** (`alpha_research/eval/rigor.py`) — deflated Sharpe ratio,
  probability of backtest overfitting (PBO via CSCV), non-overlapping t-stats,
  purged + embargoed walk-forward splits, RankMe effective rank, and a
  **mandatory degenerate-signal check** (a prior in-house crypto system lost
  money because 67% of its "winning" models were constant predictors — no signal
  is ranked here without passing this check first).
- **Backtest engine** (`alpha_research/eval/backtest.py`) — event-driven,
  cost-aware, segment-aware non-overlapping holds, full bid-ask spread + fee
  round-trip crossing, and a rank-based selectivity sweep. Asset-agnostic:
  it operates purely on `signal` / `returns` / `spread` / `segment` arrays.
- **Paper template** (`docs/_template.html`) — the shared academic HTML scaffold
  (CSS, MathJax, Open Graph / Twitter cards, favicons, and the standard section
  structure) used to publish the series on GitHub Pages.

## Rigor bar (every study)

Purged / embargoed walk-forward CV · deflated Sharpe · PBO via CSCV ·
non-overlapping t-stats (**t > 3** for a real claim) · realistic transaction
costs (full spread + fees + funding / slippage) · prediction-diversity /
degenerate-signal check · survivorship-free / point-in-time universes · every
signal benchmarked against its trivial baseline **and** the +0.3-Sharpe daily
mean-reversion floor (the only equity survivor of prior work).

## Install

```bash
pip install -e .
```

Requires Python >= 3.11. See `RESEARCH_LOG.md` for the dated decision log and
the hard-won lessons from prior in-house work.
