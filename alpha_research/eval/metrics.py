"""Predictive metrics with NaN/boundary masking.

Ported from ``jepa-trader`` (``jepa_trader/eval/metrics.py``). The information
coefficient (IC), directional accuracy, and R^2 helpers are preserved verbatim;
NaN/degenerate inputs are masked out and near-constant inputs return NaN (a
deliberate guard — a constant predictor has undefined rank correlation and must
not be scored as if it had signal). A NaN-masked Sharpe helper is added for
convenience.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def _valid(pred: np.ndarray, y: np.ndarray, mask: np.ndarray | None):
    m = np.isfinite(pred) & np.isfinite(y)
    if mask is not None:
        m = m & mask.astype(bool)
    return pred[m], y[m]


def spearman_ic(pred, y, mask=None) -> float:
    """Rank (Spearman) information coefficient. NaN if too few or constant."""
    p, t = _valid(pred, y, mask)
    if p.size < 10 or p.std() < 1e-12 or t.std() < 1e-12:
        return float("nan")
    return float(spearmanr(p, t).statistic)


def pearson_ic(pred, y, mask=None) -> float:
    """Pearson information coefficient. 0.0 if too few or constant."""
    p, t = _valid(pred, y, mask)
    if p.size < 10 or p.std() < 1e-12 or t.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def directional_accuracy(pred, y, mask=None, deadband: float = 0.0) -> float:
    """Sign accuracy on samples whose realized |return| exceeds the dead-band."""
    p, t = _valid(pred, y, mask)
    sel = np.abs(t) > deadband
    if sel.sum() < 10:
        return float("nan")
    return float((np.sign(p[sel]) == np.sign(t[sel])).mean())


def r2(pred, y, mask=None) -> float:
    """Out-of-sample R^2 of the prediction against the realized target."""
    p, t = _valid(pred, y, mask)
    if p.size < 10:
        return float("nan")
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def sharpe(returns, mask=None, periods_per_year: float | None = None) -> float:
    """NaN-masked Sharpe ratio of a per-period return series.

    Per-period Sharpe = mean / std. If ``periods_per_year`` is given, the result
    is annualized by sqrt(periods_per_year). Returns NaN for <10 finite points or
    zero dispersion. (Deflated / probabilistic Sharpe live in ``rigor.py``.)
    """
    r = np.asarray(returns, dtype=float)
    if mask is not None:
        r = r[mask.astype(bool)]
    r = r[np.isfinite(r)]
    if r.size < 10 or r.std() < 1e-12:
        return float("nan")
    sr = float(r.mean() / r.std())
    if periods_per_year is not None:
        sr *= float(np.sqrt(periods_per_year))
    return sr


def evaluate(pred, y, mask=None, deadband: float = 0.0) -> dict:
    """Bundle IC / Pearson / directional accuracy / R^2 / n for a prediction."""
    p, _ = _valid(pred, y, mask)
    return dict(
        ic=spearman_ic(pred, y, mask),
        pearson=pearson_ic(pred, y, mask),
        dir_acc=directional_accuracy(pred, y, mask, deadband),
        r2=r2(pred, y, mask),
        n=int(p.size),
    )
