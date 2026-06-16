"""Evaluation layer: cost-aware backtest, predictive metrics, and the
statistical-rigor protocol (deflated Sharpe, PBO/CSCV, purged walk-forward,
degenerate-signal check). Reused from ``jepa-trader``."""

from . import backtest, metrics, rigor  # noqa: F401
