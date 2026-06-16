# scripts/ — numbered CLI pipeline

Each research step is a standalone, runnable script with a **two-digit numeric
prefix** (`NN_short_name.py`). The convention is borrowed from `jepa-trader`:

- **Numbers impose a reading / execution order.** Lower numbers come first; a
  script generally consumes the artifacts produced by lower-numbered ones.
- **Number *ranges* group a pipeline stage.** A loose convention:
  - `00–09` data acquisition / dataset construction
  - `10–29` feature / factor / signal construction
  - `30–49` evaluation, backtests, rigor (deflated Sharpe, PBO, walk-forward)
  - `50–69` per-paper analyses and ablations
  - `70–89` figures / tables for the papers
  - `90–99` misc / one-off diagnostics
- **Each script is self-contained and re-runnable**: it reads from the paths in
  `configs/data_sources.yaml`, writes results JSON to `experiments/` (gitignored),
  and figures to the relevant `docs/figures/` or `papers/` location.
- **Gaps in the numbering are fine** — numbers are slots, not a contiguous count.

Example (illustrative):
```
01_fetch_deribit_vol.py     # acquire / refresh DVOL + surfaces
20_build_vrp_signal.py      # construct the volatility-risk-premium signal
30_backtest_vrp.py          # cost-aware backtest + rigor protocol
70_make_paper1_figures.py   # figures for docs/paper1.html
```

Every signal MUST be run through `alpha_research.eval.rigor.degenerate_signal_check`
before it is ranked or backtested (see the mandatory note in `rigor.py`).
