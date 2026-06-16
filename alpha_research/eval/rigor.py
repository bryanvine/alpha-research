"""
Statistical-rigor protocol for alpha research.

Every study in this project must pass this protocol before any edge is claimed.
It is a port + extension of the rigor machinery proven out in ``jepa-trader``
(see ``jepa_trader/eval/xs_eval.py`` and ``scripts/57_rankme.py``), combined with
the standard literature for the pieces that lived only as prose in the jepa
research log.

Provenance of each function (see module-level note in the project report):
  * ``deflated_sharpe_ratio`` / ``probabilistic_sharpe_ratio`` / ``expected_max_sharpe``
        -- PORTED from jepa-trader xs_eval (``deflate_sr0`` + ``psr``), generalized.
  * ``nonoverlapping_tstat``        -- PORTED from jepa-trader xs_eval (``ic_stats``).
  * ``rankme``                      -- PORTED from jepa-trader scripts/57_rankme.py.
  * ``probability_of_backtest_overfitting`` (CSCV)
        -- IMPLEMENTED FRESH from Bailey, Borwein, Lopez de Prado & Zhu (2017);
           the jepa log specifies "PBO via CSCV" as a requirement but ships no
           standalone CSCV implementation.
  * ``purged_walkforward_splits``   -- IMPLEMENTED FRESH (purged + embargoed
           expanding walk-forward) from Lopez de Prado, *Advances in Financial
           ML* (2018), ch. 7; jepa used hard-coded calendar folds + within-segment
           purging, but no reusable splitter.
  * ``degenerate_signal_check``     -- IMPLEMENTED FRESH. THE most important guard
           in this project (see the prominent note below).

The headline hurdles used across the project:
  * non-overlapping t-stat hurdle for a *real* claim: ``T_STAT_HURDLE`` = 3.0
    (the jepa log explicitly raised this from the usual t>2 to t>3).
  * deflated Sharpe must be materially > 0 after deflating for the number of
    configurations tried.
  * PBO (probability of backtest overfitting) should be < 0.5; near 0 is good.
  * the +0.3-Sharpe daily mean-reversion floor is the practical "is this even
    worth it" benchmark (see ``MEAN_REVERSION_SHARPE_FLOOR``).

References
----------
[1] D. Bailey & M. Lopez de Prado, *The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality*, J. Portfolio
    Management, 2014. (SSRN 2460551)
[2] D. Bailey, J. Borwein, M. Lopez de Prado & Q. Zhu, *The Probability of
    Backtest Overfitting*, J. Computational Finance, 2017. (CSCV)
[3] M. Lopez de Prado, *Advances in Financial Machine Learning*, Wiley 2018
    (purging, embargoing, walk-forward, combinatorial CV).
[4] Q. Garrido et al., *RankMe: Assessing the Downstream Performance of
    Pretrained Representations by their Rank*, ICML 2023. (arXiv 2210.02885)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import kurtosis, norm, skew

# ---------------------------------------------------------------------------
# Project-wide hurdles / constants
# ---------------------------------------------------------------------------
EULER = 0.5772156649015329          # Euler-Mascheroni constant (gamma)
T_STAT_HURDLE = 3.0                 # non-overlapping t hurdle for a real claim
PBO_HURDLE = 0.5                    # probability-of-overfitting must be below this
MEAN_REVERSION_SHARPE_FLOOR = 0.3   # the only equity survivor; the practical floor


# ===========================================================================
# 1. Deflated / probabilistic Sharpe ratio  (Bailey & Lopez de Prado)
#    PORTED from jepa-trader xs_eval.py (`psr`, `deflate_sr0`), generalized.
# ===========================================================================
def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true per-period SR > ``sr_benchmark``).

    Skew/kurtosis-aware (Bailey & Lopez de Prado 2014). ``returns`` and
    ``sr_benchmark`` are in *per-period* units (do NOT pass an annualized SR).
    Returns NaN for <10 finite points or zero dispersion.

        PSR(SR*) = Phi( (SR - SR*) * sqrt(n - 1)
                        / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    where SR is the sample per-period Sharpe, g3 is skew, g4 is (non-Fisher)
    kurtosis, and n is the sample size.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))
    denom = math.sqrt(max(1e-9, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(r.size - 1) / denom))


def expected_max_sharpe(trial_sharpes, n_trials: int | None = None) -> float:
    """Expected maximum *per-period* Sharpe under the null, given N trials.

    This is the deflation benchmark SR0 of Bailey & Lopez de Prado: the Sharpe
    you would expect to see as the best of ``n_trials`` independent strategies
    whose true Sharpe is zero, given the observed cross-trial dispersion of
    Sharpes. Ported from jepa-trader's ``deflate_sr0``.

        SR0 = sigma_SR * [ (1 - gamma) * Z(1 - 1/N) + gamma * Z(1 - 1/(N*e)) ]

    where sigma_SR is the std of the trial Sharpes, gamma is Euler-Mascheroni,
    Z is the standard-normal inverse CDF, and N is the number of trials.
    """
    v = np.asarray([s for s in np.asarray(trial_sharpes, dtype=float)
                    if np.isfinite(s)])
    n = int(n_trials) if n_trials is not None else int(v.size)
    if v.size < 2 or n < 2:
        return 0.0
    var = float(v.var(ddof=1))
    z1 = norm.ppf(1 - 1.0 / n)
    z2 = norm.ppf(1 - 1.0 / (n * math.e))
    return math.sqrt(var) * ((1 - EULER) * z1 + EULER * z2)


def deflated_sharpe_ratio(returns, trial_sharpes, n_trials: int | None = None) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    The probability that the strategy's true Sharpe exceeds the *expected max
    Sharpe under the null* for the number of configurations tried -- i.e. PSR
    benchmarked against ``expected_max_sharpe`` rather than against zero. This
    is the headline statistic for any claimed edge: it directly penalizes
    selection over many backtests.

    Parameters
    ----------
    returns : per-period return series of the *selected* strategy.
    trial_sharpes : per-period Sharpe of every configuration that was tried
        (used to estimate cross-trial Sharpe dispersion). Include the winner.
    n_trials : number of independent trials; defaults to ``len(trial_sharpes)``.

    Returns a probability in [0, 1]; a real edge wants this materially > 0.5
    (and ideally near 1). 0 means the result is fully explained by selection.
    """
    sr0 = expected_max_sharpe(trial_sharpes, n_trials)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)


# Backwards-compatible alias matching the jepa-trader name.
def deflate_sr0(trial_sharpes, n_trials):
    """Alias for :func:`expected_max_sharpe` (jepa-trader name)."""
    return expected_max_sharpe(trial_sharpes, n_trials)


def psr(returns, sr0: float = 0.0) -> float:
    """Alias for :func:`probabilistic_sharpe_ratio` (jepa-trader name)."""
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)


# ===========================================================================
# 2. Non-overlapping t-statistic
#    PORTED from jepa-trader xs_eval.py (`ic_stats`).
# ===========================================================================
def nonoverlapping_tstat(series, thin: int = 1) -> dict:
    """Non-overlapping t-stat of a per-period statistic (e.g. per-period IC).

    Overlapping multi-horizon labels inflate t-stats because consecutive
    observations share information. Sub-sampling every ``thin`` observations
    (set ``thin = horizon``) restores (approximate) independence before the
    t-test, so the t-stat is honest.

        t = mean(sub) / std(sub, ddof=1) * sqrt(n_sub),  sub = series[::thin]

    Parameters
    ----------
    series : 1-D array of the per-period statistic (e.g. one rank-IC per bar).
    thin : sub-sampling stride; use the label horizon so windows don't overlap.

    Returns dict(mean, t, n, pos) where ``mean`` is the full-sample mean,
    ``t`` the non-overlapping t-stat, ``n`` the sub-sampled count, and ``pos``
    the fraction of the full series that is positive. Compare ``|t|`` against
    ``T_STAT_HURDLE`` (3.0) for a real claim.
    """
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 5:
        return dict(mean=float("nan"), t=float("nan"), n=int(s.size), pos=float("nan"))
    mean = float(s.mean())
    sub = s[::max(1, int(thin))]
    sd = float(sub.std(ddof=1)) if sub.size > 1 else 0.0
    t = sub.mean() / sd * math.sqrt(sub.size) if sd > 0 else float("nan")
    return dict(mean=mean, t=float(t), n=int(sub.size), pos=float((s > 0).mean()))


# ===========================================================================
# 3. Probability of Backtest Overfitting (PBO) via CSCV
#    IMPLEMENTED FRESH from Bailey, Borwein, Lopez de Prado & Zhu (2017).
# ===========================================================================
def probability_of_backtest_overfitting(perf_matrix, n_splits: int = 16,
                                         higher_is_better: bool = True) -> dict:
    """Probability of Backtest Overfitting (PBO) via CSCV.

    Combinatorially-Symmetric Cross-Validation (Bailey et al. 2017). Given the
    per-period performance of ``N`` competing configurations over ``T`` periods,
    PBO estimates how often the configuration that looks best *in sample* fails
    to beat the median *out of sample* -- i.e. how often a backtest "winner" is
    a selection artifact.

    Method:
      1. Split the T periods into ``n_splits`` contiguous blocks.
      2. For every way to choose half the blocks as IS (the other half OOS):
           - rank configs by mean performance on IS; pick the IS-best;
           - find its *rank* among all configs on the OOS half;
           - logit of its relative OOS rank -> lambda.
      3. PBO = fraction of splits with lambda <= 0 (IS-best lands in the bottom
         half OOS). Also returns the logit distribution and a performance-
         degradation slope (OOS vs IS regression of the selected config).

    Parameters
    ----------
    perf_matrix : array (T_periods, N_configs) of per-period performance
        (e.g. per-period returns or per-period IC), one column per configuration.
    n_splits : even number of contiguous blocks S; C(S, S/2) combinations are
        evaluated. 16 -> 12,870 combinations (the standard choice).
    higher_is_better : if False, performance is negated first.

    Returns dict(pbo, n_combinations, logits, oos_below_median_rate). A healthy
    strategy family has ``pbo`` well below ``PBO_HURDLE`` (0.5).
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T_periods, N_configs)")
    if not higher_is_better:
        M = -M
    T, N = M.shape
    if N < 2:
        raise ValueError("need >= 2 configurations to assess overfitting")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    n_splits = min(n_splits, T)
    if n_splits < 2:
        raise ValueError("need >= 2 periods to split")

    # Contiguous, near-equal blocks of period indices.
    blocks = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2
    all_idx = set(range(n_splits))

    logits = []
    oos_below = 0
    for is_blocks in combinations(range(n_splits), half):
        oos_blocks = sorted(all_idx - set(is_blocks))
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in oos_blocks])

        is_perf = M[is_rows].mean(axis=0)
        oos_perf = M[oos_rows].mean(axis=0)

        best = int(np.argmax(is_perf))
        # Relative rank of the IS-best config on OOS, in (0, 1).
        order = np.argsort(oos_perf)          # ascending: worst..best
        rank = int(np.where(order == best)[0][0]) + 1
        w = rank / (N + 1.0)                   # relative rank, avoids 0/1
        if w <= 0.5:
            oos_below += 1
        logits.append(math.log(w / (1.0 - w)))

    logits = np.asarray(logits, dtype=float)
    n_comb = logits.size
    pbo = float((logits <= 0).mean()) if n_comb else float("nan")
    return dict(
        pbo=pbo,
        n_combinations=int(n_comb),
        logits=logits,
        oos_below_median_rate=float(oos_below / n_comb) if n_comb else float("nan"),
    )


# ===========================================================================
# 4. Purged + embargoed walk-forward splitter
#    IMPLEMENTED FRESH (Lopez de Prado 2018, ch. 7).
# ===========================================================================
@dataclass
class WalkForwardSplit:
    """One expanding walk-forward fold (integer index arrays into the sample).

    ``fold`` is 0-based. ``train`` excludes anything within the purge window of
    the test block and anything inside the post-test embargo of an *earlier*
    fold; ``test`` is a contiguous out-of-sample block.
    """
    fold: int
    train: np.ndarray
    test: np.ndarray


def purged_walkforward_splits(n_samples: int, n_splits: int = 4,
                              embargo: float = 0.01, purge: int = 0,
                              expanding: bool = True) -> list[WalkForwardSplit]:
    """Purged + embargoed walk-forward CV splits (Lopez de Prado 2018).

    Walk-forward respects time (train is always strictly before test). Two
    leakage controls are layered on:
      * **purge**: drop the ``purge`` samples immediately *before* each test
        block from training -- their labels (built over a forward horizon) would
        otherwise overlap the test period. Set ``purge`` to the label horizon.
      * **embargo**: after each test block, embargo a fraction of the sample so
        that a *later* fold's training set cannot start right after this fold's
        test set (serial correlation leaks across the boundary). ``embargo`` is
        a fraction of ``n_samples`` (e.g. 0.01 = 1%).

    Parameters
    ----------
    n_samples : length of the (time-ordered) sample.
    n_splits : number of test folds; the sample tail is divided into this many
        contiguous test blocks.
    embargo : embargo size as a fraction of ``n_samples``.
    purge : number of samples to purge immediately before each test block.
    expanding : if True, training is everything allowed up to the test block
        (expanding window); if False, training is only the contiguous allowed
        block immediately preceding it (rolling window).

    Returns a list of :class:`WalkForwardSplit`. Designed so a fitted model only
    ever sees information strictly prior to (and outside the purge/embargo of)
    the block it is scored on.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if n_samples < n_splits + 1:
        raise ValueError("n_samples too small for the requested n_splits")
    idx = np.arange(n_samples)
    emb = int(round(embargo * n_samples))

    # Carve the *tail* of the sample into n_splits contiguous test blocks, so
    # the first fold always has a non-trivial training history before it.
    fold_starts = np.linspace(n_samples // (n_splits + 1), n_samples, n_splits + 1,
                              dtype=int)
    splits: list[WalkForwardSplit] = []
    for k in range(n_splits):
        test_start = int(fold_starts[k])
        test_end = int(fold_starts[k + 1])
        if test_end <= test_start:
            continue
        test = idx[test_start:test_end]

        # Training candidates: everything before the (purged) test start.
        purge_start = max(0, test_start - max(0, int(purge)))
        if expanding:
            train_lo = 0
        else:
            prev_end = int(fold_starts[k - 1]) if k > 0 else 0
            train_lo = max(0, prev_end + emb)
        train = idx[train_lo:purge_start]

        # Embargo: also drop any training samples that fall within the embargo
        # window *after* an earlier test block (relevant for expanding windows).
        if emb > 0 and expanding:
            keep = np.ones(train.shape, dtype=bool)
            for j in range(k):
                e_start = int(fold_starts[j + 1])
                e_end = min(n_samples, e_start + emb)
                keep &= ~((train >= e_start) & (train < e_end))
            train = train[keep]

        if train.size == 0:
            continue
        splits.append(WalkForwardSplit(fold=k, train=train, test=test))
    return splits


# ===========================================================================
# 5. RankMe effective rank
#    PORTED from jepa-trader scripts/57_rankme.py.
# ===========================================================================
def rankme(Z, eps: float = 1e-7) -> float:
    """RankMe effective rank (Garrido et al. 2023): exp(entropy of L1-normalized
    singular values).

    A representation/embedding diagnostic: a value close to the embedding
    dimension ``D`` means high rank (healthy, diverse); a value near 1 means the
    representation has collapsed onto a single direction. Useful here as a
    *feature/embedding* diversity check (the scalar-signal analogue is
    :func:`degenerate_signal_check`).

    Parameters
    ----------
    Z : array (n_samples, D) of embeddings / features.
    eps : numerical floor.
    """
    Z = np.asarray(Z, dtype=float)
    Z = Z - Z.mean(0, keepdims=True)
    sv = np.linalg.svd(Z, compute_uv=False)
    p = sv / (sv.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


# ===========================================================================
# 6. *** DEGENERATE / CONSTANT-SIGNAL CHECK ***  <-- READ THIS
#    IMPLEMENTED FRESH. The single most important guard in this project.
# ===========================================================================
#
#   WHY THIS EXISTS / WHY IT IS MANDATORY
#   -------------------------------------
#   The in-house crypto ML system (`/apps/crypto-trader/`) lost real money live
#   (-$409, -16-21% over 104 days). A post-mortem found that **67% of its
#   "winning" models were degenerate constant predictors** -- they emitted an
#   almost-constant prediction that gamed a backtester which never checked
#   prediction diversity. A constant signal can post a flattering backtest (it
#   effectively becomes "always long" or "always flat"), sails through naive
#   ranking, and then does nothing -- or quietly takes on undiversified
#   directional risk -- in production.
#
#   THEREFORE: every signal MUST pass `degenerate_signal_check(...)` BEFORE it is
#   ranked, backtested for a headline number, or compared against a baseline. A
#   flagged signal is disqualified, not "penalized". Do not rank degenerate
#   signals. This check is cheap; run it first, always.
#
@dataclass
class DegeneracyReport:
    """Result of :func:`degenerate_signal_check`.

    ``is_degenerate`` is the verdict; ``reasons`` lists every triggered rule;
    the remaining fields expose the diagnostics so they can be logged.
    """
    is_degenerate: bool
    reasons: list[str]
    n: int
    n_finite: int
    n_unique: int
    frac_unique: float
    std: float
    frac_at_mode: float
    iqr: float

    def __bool__(self) -> bool:  # truthy == degenerate, so `if report:` flags it
        return self.is_degenerate


def degenerate_signal_check(signal,
                            min_unique: int = 10,
                            min_frac_unique: float = 0.01,
                            min_std: float = 1e-8,
                            max_frac_at_mode: float = 0.95,
                            min_finite_frac: float = 0.5) -> DegeneracyReport:
    """Flag near-constant / low-diversity / near-zero-variance signals.

    *** RUN THIS ON EVERY SIGNAL BEFORE RANKING OR BACKTESTING IT. ***

    A signal is flagged ``is_degenerate=True`` if ANY of the following hold
    (each appended to ``reasons``):

      * too many non-finite values  (finite fraction < ``min_finite_frac``);
      * too few distinct values     (n_unique < ``min_unique``);
      * too few distinct values relative to sample size
                                    (frac_unique < ``min_frac_unique``);
      * near-zero dispersion        (std < ``min_std``  OR  IQR == 0);
      * a single value dominates    (mode share > ``max_frac_at_mode``).

    Rationale: see the module-level note. This guard exists specifically because
    a prior in-house system lost money when 67% of its "winning" models were
    constant predictors that a diversity-blind backtester happily ranked.

    Parameters
    ----------
    signal : array-like of signal values (any shape; flattened).
    min_unique : minimum number of distinct finite values required.
    min_frac_unique : minimum (#unique / #finite) required.
    min_std : minimum standard deviation required.
    max_frac_at_mode : maximum allowed share of the single most-common value.
    min_finite_frac : minimum fraction of values that must be finite.

    Returns a :class:`DegeneracyReport` (truthy iff degenerate).
    """
    s = np.asarray(signal, dtype=float).ravel()
    n = int(s.size)
    finite = s[np.isfinite(s)]
    n_finite = int(finite.size)
    reasons: list[str] = []

    if n == 0:
        return DegeneracyReport(True, ["empty signal"], 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    finite_frac = n_finite / n
    if finite_frac < min_finite_frac:
        reasons.append(f"too many non-finite values (finite_frac={finite_frac:.3f} "
                       f"< {min_finite_frac})")

    if n_finite == 0:
        return DegeneracyReport(True, reasons or ["no finite values"], n, 0, 0,
                                0.0, 0.0, 0.0, 0.0)

    uniq, counts = np.unique(finite, return_counts=True)
    n_unique = int(uniq.size)
    frac_unique = n_unique / n_finite
    std = float(finite.std())
    frac_at_mode = float(counts.max() / n_finite)
    q75, q25 = np.percentile(finite, [75, 25])
    iqr = float(q75 - q25)

    if n_unique < min_unique:
        reasons.append(f"too few unique values (n_unique={n_unique} < {min_unique})")
    if frac_unique < min_frac_unique:
        reasons.append(f"low value diversity (frac_unique={frac_unique:.4f} "
                       f"< {min_frac_unique})")
    if std < min_std:
        reasons.append(f"near-zero variance (std={std:.2e} < {min_std:.0e})")
    if iqr == 0.0:
        reasons.append("zero inter-quartile range (>=50% of mass at one value)")
    if frac_at_mode > max_frac_at_mode:
        reasons.append(f"single value dominates (mode_share={frac_at_mode:.3f} "
                       f"> {max_frac_at_mode})")

    return DegeneracyReport(
        is_degenerate=bool(reasons),
        reasons=reasons,
        n=n,
        n_finite=n_finite,
        n_unique=n_unique,
        frac_unique=frac_unique,
        std=std,
        frac_at_mode=frac_at_mode,
        iqr=iqr,
    )


def assert_signal_not_degenerate(signal, name: str = "signal", **kwargs) -> DegeneracyReport:
    """Raise ``ValueError`` if ``signal`` is degenerate; else return the report.

    Convenience wrapper for pipelines that should hard-fail rather than silently
    rank a degenerate predictor.
    """
    rep = degenerate_signal_check(signal, **kwargs)
    if rep.is_degenerate:
        raise ValueError(f"DEGENERATE SIGNAL '{name}' disqualified: "
                         + "; ".join(rep.reasons))
    return rep
