# Alpha Research

A cross-asset quantitative research program hunting for **genuinely harvestable
alpha** — and reporting honest negative results when there isn't any. It covers
crypto, equities, FX, commodities, and volatility / options, and publishes a
rigorous, reproducible, cost-aware series of papers, in the same spirit as
[`jepa-trader`](https://github.com/bryanvine/jepa-trader).

The unit of output is a paper backed by a fully reproducible pipeline: every
claimed edge must clear a fixed statistical-rigor bar, and every backtest is net
of realistic transaction costs. Negative results are first-class outcomes.

## The series (9 papers, published)

Live at **[bryanvine.github.io/alpha-research](https://bryanvine.github.io/alpha-research/)**.
Each title links to the paper; the one-line verdict is in *italics*.

1. **[The Volatility Risk Premium, Cross-Asset](https://bryanvine.github.io/alpha-research/paper1.html)**
   — *real and statistically robust (BTC t=4.45, deflated Sharpe 0.88), but not
   harvestable net of options-execution cost, and decaying in crypto.*
2. **[Crypto Carry: The Funding-Rate Cross-Section](https://bryanvine.github.io/alpha-research/paper2.html)**
   — *the "Sharpe-6" level trade is an artifact; the cross-sectional funding factor
   is the program's one genuine net-of-cost, out-of-sample edge (~0.4).*
3. **[Statistical Arbitrage in Crypto](https://bryanvine.github.io/alpha-research/paper3.html)**
   — *illusory: a stale-price artifact in illiquid coins + a multiple-testing mirage
   (210 in-sample pairs lose out-of-sample).*
4. **[The Cost of Direction](https://bryanvine.github.io/alpha-research/paper4.html)**
   — *a replication crisis: 212 published anomalies decay 53% out-of-sample; only
   low-turnover quality survives.*
5. **[Liquidity Provision as Alpha](https://bryanvine.github.io/alpha-research/paper5.html)**
   — *no standalone microstructure alpha (a sub-0.1 bp HFT game), but a real execution
   edge (~13% slippage) + the vol-conditional liquidity-provision premium.*
6. **[FX Carry & Commodity Roll Yield](https://bryanvine.github.io/alpha-research/paper6.html)**
   — *finance's most-documented anomalies are modest, decayed, crash-prone (FX carry
   0.44→0.2, FX value 0.46, energy roll ≈ 0).*
7. **[Synthesis: Do the Modest Edges Add Up?](https://bryanvine.github.io/alpha-research/paper7.html)**
   — *capstone: four near-uncorrelated ~0.4 premia diversify into a combined 0.53
   (above the QSPIX ceiling) — but the premia are decaying (2020s ≈ 0.13).*
8. **[On-Chain Liquidity & the Crypto Premia: An Autonomous Search](https://bryanvine.github.io/alpha-research/paper8.html)**
   — *the first paper found by a machine: an autonomous LLM loop, gated by this program's
   rigor harness, surfaced a stablecoin-flow overlay (in-window OOS 1.18 vs 0.66, 5-day
   mechanism t=3.22) that **died out-of-window** (t→0.48, PBO 0.11→0.51) — a fit, not an
   edge. The automatic, honest kill is the deliverable.*
9. **[The Machine Analyst: Two LLM Fundamental-Valuation Bots](https://bryanvine.github.io/alpha-research/paper9.html)**
   — *the machine as analyst, not researcher: two bots value the whole S&P 500 daily —
   **[Vinebot](https://dbot.vineai.tech)** (Damodaran, prices everything) and
   **[Buffybot](https://buffy.bot)** (Buffett, waits for the fat pitch) — with the LLM
   confined to clamped assumptions while Python does the math. They triangulate (Spearman
   0.71, 85% sign agreement) and both call mid-2026 rich, but an LLM analyst **cannot be
   backtested** (it has already read the future); skill is unproven and the test is a
   multi-year forward one. The honest, auditable apparatus is the deliverable.*

**Program verdict.** The durable edges are modest, uncorrelated, *decaying* risk
premia — compensation for **providing** liquidity / insurance / financing, never
directional forecasts. Every spectacular backtest alpha proved an artifact; honest
cost / decay / capacity modeling was the real edge. A diversified book of the
survivors clears a live multi-style benchmark historically (~0.53 Sharpe) but should
be sized to the recent, fading regime.

**Live tracking.** The diversified book is paper-traded forward (daily, vol-targeted,
net of costs and borrow) — see `scripts/50_live_book.py` and the dated `RESEARCH_LOG.md`.

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
