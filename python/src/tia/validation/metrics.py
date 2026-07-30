"""Performance, significance and calibration metrics.

This module is deliberately paranoid. Every quantity a backtest reports is an
*estimate* with a standard error, and most of the folklore metrics in trading
(Sharpe, hit rate, profit factor) are quoted as if they were measurements. The
functions here are grouped accordingly:

*   **Descriptive** -- returns, volatility, Sharpe, Sortino, Calmar, drawdown,
    hit rate, profit factor, expectancy, tail ratio, worst streaks. Useful for
    describing what happened; useless for deciding whether it will happen again.
*   **Inferential** -- probabilistic and deflated Sharpe ratios, minimum track
    record length, win-rate standard errors, required sample sizes. These are
    what decide whether the edge is real. See :func:`win_rate_standard_error`
    for the single most important sanity check in this file.
*   **Stability** -- regime breakdown, cross-fold stability, population
    stability index.
*   **Calibration** -- Brier score and its Murphy decomposition, reliability
    curves, expected calibration error. A directional model that is *accurate*
    but not *calibrated* cannot be sized, because Kelly needs a probability, not
    a score.

Conventions
-----------
*   Inputs named ``returns`` are **per-period natural log returns** (SPEC §1).
    Aggregates convert to simple returns where that is the meaningful quantity
    (``total_return``, ``annualized_return``).
*   Inputs named ``trade_returns`` are per-trade returns; the per-trade
    functions do not care whether they are log, simple or sigma units, only
    that the unit is consistent.
*   Sharpe ratios passed to :func:`probabilistic_sharpe_ratio` and friends are
    **non-annualised, per-observation** ratios, and ``n`` is the number of
    observations behind them. Annualising first and then passing ``n`` in years
    is the most common way to get these formulas wrong by a factor of
    ``sqrt(periods_per_year)``.

References
----------
Bailey, D. and López de Prado, M. (2012), "The Sharpe Ratio Efficient
Frontier", *Journal of Risk* 15(2) -- PSR and MinTRL.
Bailey, D. and López de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
*Journal of Portfolio Management* 40(5).
Murphy, A. (1973), "A New Vector Partition of the Probability Score",
*Journal of Applied Meteorology* 12 -- Brier decomposition.
Lo, A. (2002), "The Statistics of Sharpe Ratios", *Financial Analysts Journal*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..types import REGIME_NAMES

__all__ = [
    "EULER_MASCHERONI",
    "norm_cdf",
    "norm_ppf",
    "skewness",
    "kurtosis",
    "total_return",
    "annualized_return",
    "volatility",
    "sharpe_ratio",
    "sharpe_standard_error",
    "sortino_ratio",
    "DrawdownStats",
    "drawdown_stats",
    "max_drawdown",
    "calmar_ratio",
    "hit_rate",
    "profit_factor",
    "expectancy_sigma",
    "average_win_loss",
    "tail_ratio",
    "worst_streak",
    "max_consecutive_losses",
    "summarize_returns",
    "summarize_trades",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "minimum_track_record_length",
    "win_rate_standard_error",
    "win_rate_confidence_interval",
    "trades_needed_to_distinguish",
    "regime_breakdown",
    "FoldStability",
    "stability_across_folds",
    "psi",
    "brier_score",
    "BrierDecomposition",
    "brier_decomposition",
    "reliability_curve",
    "expected_calibration_error",
]

#: Euler--Mascheroni constant, used in the Gumbel approximation to the expected
#: maximum of ``n_trials`` independent Sharpe ratios (Bailey & López de Prado
#: 2014, eq. 5).
EULER_MASCHERONI: float = 0.577215664901532860606512090082


# ---------------------------------------------------------------------------
# Normal distribution helpers (no scipy: these must work in the port)
# ---------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    """Standard normal CDF via ``math.erf``; exact to double precision."""
    return 0.5 * math.erfc(-float(x) / math.sqrt(2.0))


_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF.

    Acklam's rational approximation followed by one Halley refinement against
    :func:`norm_cdf`, which brings the error to the order of machine epsilon.
    Implemented here rather than imported from scipy because SPEC §10 forbids
    optional dependencies on any code path that might be ported.
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        return math.nan
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = _poly(_ACKLAM_C, q) / (_poly(_ACKLAM_D, q) * q + 1.0)
    elif p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -_poly(_ACKLAM_C, q) / (_poly(_ACKLAM_D, q) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        x = _poly(_ACKLAM_A, r) * q / (_poly(_ACKLAM_B, r) * r + 1.0)
    # Halley step.
    e = norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


def _poly(coeffs: Sequence[float], x: float) -> float:
    out = 0.0
    for c in coeffs:
        out = out * x + c
    return out


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


def _clean(x: Iterable[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def skewness(x: Iterable[float]) -> float:
    """Sample skewness, moment estimator ``m3 / m2**1.5`` (no bias correction).

    The uncorrected estimator is what Bailey & López de Prado's PSR formula
    assumes, so it is what is used here; the correction is O(1/n) and would
    silently shift PSR at small ``n``.
    """
    a = _clean(x)
    n = a.size
    if n < 3:
        return math.nan
    d = a - a.mean()
    m2 = float(np.mean(d**2))
    if m2 <= 0.0:
        return 0.0
    return float(np.mean(d**3) / m2**1.5)


def kurtosis(x: Iterable[float], excess: bool = False) -> float:
    """Sample kurtosis. ``excess=False`` returns the *raw* fourth moment ratio.

    The PSR/DSR formulas take raw kurtosis (3 for a Gaussian), which is the
    single most common transcription error in implementations of them.
    """
    a = _clean(x)
    n = a.size
    if n < 4:
        return math.nan
    d = a - a.mean()
    m2 = float(np.mean(d**2))
    if m2 <= 0.0:
        return 3.0 - (3.0 if excess else 0.0)
    k = float(np.mean(d**4) / (m2 * m2))
    return k - 3.0 if excess else k


# ---------------------------------------------------------------------------
# Descriptive performance
# ---------------------------------------------------------------------------


def total_return(returns: Iterable[float]) -> float:
    """Simple total return implied by a series of per-period log returns."""
    a = _clean(returns)
    if a.size == 0:
        return 0.0
    return float(np.expm1(a.sum()))


def annualized_return(returns: Iterable[float], periods_per_year: float = 252.0) -> float:
    """Geometric annualised return from per-period log returns."""
    a = _clean(returns)
    if a.size == 0:
        return math.nan
    return float(np.expm1(a.mean() * periods_per_year))


def volatility(returns: Iterable[float], periods_per_year: float = 252.0) -> float:
    """Annualised standard deviation of per-period log returns (``ddof=1``)."""
    a = _clean(returns)
    if a.size < 2:
        return math.nan
    return float(a.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: Iterable[float],
    periods_per_year: float = 1.0,
    risk_free_per_period: float = 0.0,
    ddof: int = 1,
) -> float:
    """Sharpe ratio, annualised by ``sqrt(periods_per_year)``.

    The default ``periods_per_year=1`` returns the **per-observation** ratio,
    which is what the PSR/DSR functions in this module expect. Annualise only
    for human display.
    """
    a = _clean(returns)
    if a.size < 2:
        return math.nan
    ex = a - risk_free_per_period
    sd = ex.std(ddof=ddof)
    if not np.isfinite(sd) or sd <= 0.0:
        return math.nan
    return float(ex.mean() / sd * math.sqrt(periods_per_year))


def sharpe_standard_error(n: int, sr: float, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of a Sharpe ratio estimate under non-IID, non-normal returns.

    ``se = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (n - 1))`` (Mertens 2002;
    the same variance expression that sits inside PSR). Quoting a Sharpe without
    this number is how a 0.4 in-sample Sharpe on 200 observations gets mistaken
    for evidence: the standard error there is roughly 0.07 even before any
    multiple-testing charge.
    """
    if n < 2:
        return math.nan
    var = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr * sr
    if var <= 0.0:
        return math.nan
    return math.sqrt(var / (n - 1))


def sortino_ratio(
    returns: Iterable[float],
    periods_per_year: float = 1.0,
    target_per_period: float = 0.0,
) -> float:
    """Mean excess return over downside deviation, annualised.

    Downside deviation uses the *full* sample in the denominator (zeroing the
    upside), not only the losing observations; using only losers inflates the
    ratio and is the common error.
    """
    a = _clean(returns)
    if a.size < 2:
        return math.nan
    ex = a - target_per_period
    downside = np.minimum(ex, 0.0)
    dd = math.sqrt(float(np.mean(downside**2)))
    if dd <= 0.0:
        return math.inf if ex.mean() > 0.0 else math.nan
    return float(ex.mean() / dd * math.sqrt(periods_per_year))


@dataclass(frozen=True)
class DrawdownStats:
    """Drawdown geometry of one equity path.

    ``max_drawdown`` is a positive fraction (0.2 means a 20% peak-to-trough
    loss) computed on the *compounded* equity curve implied by the log returns.
    ``drawdown_duration`` is peak-to-trough; ``time_to_recover`` is
    trough-to-new-high and is ``inf`` when the path never recovers, which is the
    number that actually decides whether a live system gets switched off.
    """

    max_drawdown: float
    peak_index: int
    trough_index: int
    drawdown_duration: int
    time_to_recover: float
    max_underwater: int

    def as_dict(self) -> dict[str, float]:
        return {
            "max_drawdown": self.max_drawdown,
            "dd_peak_index": float(self.peak_index),
            "dd_trough_index": float(self.trough_index),
            "dd_duration": float(self.drawdown_duration),
            "dd_time_to_recover": float(self.time_to_recover),
            "dd_max_underwater": float(self.max_underwater),
        }


def drawdown_stats(returns: Iterable[float]) -> DrawdownStats:
    """Maximum drawdown and its duration from per-period log returns."""
    a = np.asarray(returns, dtype=float).ravel()
    a = np.where(np.isfinite(a), a, 0.0)
    if a.size == 0:
        return DrawdownStats(0.0, 0, 0, 0, 0.0, 0)
    equity = np.exp(np.cumsum(a))
    running_max = np.maximum.accumulate(equity)
    dd = 1.0 - equity / running_max
    trough = int(np.argmax(dd))
    max_dd = float(dd[trough])
    # The peak that the trough is measured from.
    peak = int(np.argmax(equity[: trough + 1])) if trough >= 0 else 0
    underwater = dd > 0.0
    # Longest contiguous underwater run.
    max_uw = 0
    run = 0
    for flag in underwater:
        run = run + 1 if flag else 0
        max_uw = max(max_uw, run)
    recovered = np.nonzero(equity[trough:] >= running_max[trough])[0]
    ttr = float(recovered[0]) if recovered.size else math.inf
    return DrawdownStats(
        max_drawdown=max_dd,
        peak_index=peak,
        trough_index=trough,
        drawdown_duration=int(trough - peak),
        time_to_recover=ttr,
        max_underwater=int(max_uw),
    )


def max_drawdown(returns: Iterable[float]) -> float:
    """Maximum peak-to-trough loss as a positive fraction."""
    return drawdown_stats(returns).max_drawdown


def calmar_ratio(returns: Iterable[float], periods_per_year: float = 252.0) -> float:
    """Annualised return divided by maximum drawdown."""
    mdd = max_drawdown(returns)
    if mdd <= 0.0:
        return math.inf
    ar = annualized_return(returns, periods_per_year)
    return float(ar / mdd)


def hit_rate(trade_returns: Iterable[float]) -> float:
    """Fraction of trades with strictly positive return.

    Read :func:`win_rate_standard_error` before drawing any conclusion from
    this number.
    """
    a = _clean(trade_returns)
    if a.size == 0:
        return math.nan
    return float(np.mean(a > 0.0))


def profit_factor(trade_returns: Iterable[float]) -> float:
    """Gross profit over gross loss. ``inf`` when there are no losing trades."""
    a = _clean(trade_returns)
    if a.size == 0:
        return math.nan
    gains = float(a[a > 0.0].sum())
    losses = float(-a[a < 0.0].sum())
    if losses <= 0.0:
        return math.inf if gains > 0.0 else math.nan
    return gains / losses


def expectancy_sigma(ret_sigma: Iterable[float]) -> float:
    """Mean per-trade outcome in sigma units -- the quantity the gate acts on.

    Expressing expectancy in sigma rather than currency is what lets one
    parameter set be shared across instruments and decades (SPEC §1), and it is
    the unit :class:`~tia.types.TargetSpec` uses.
    """
    a = _clean(ret_sigma)
    if a.size == 0:
        return math.nan
    return float(a.mean())


def average_win_loss(trade_returns: Iterable[float]) -> tuple[float, float, float]:
    """``(avg_win, avg_loss, |avg_win / avg_loss|)``; losses are negative."""
    a = _clean(trade_returns)
    wins = a[a > 0.0]
    losses = a[a < 0.0]
    avg_w = float(wins.mean()) if wins.size else 0.0
    avg_l = float(losses.mean()) if losses.size else 0.0
    ratio = abs(avg_w / avg_l) if avg_l != 0.0 else math.inf
    return avg_w, avg_l, ratio


def tail_ratio(returns: Iterable[float], quantile: float = 0.05) -> float:
    """``|q(1-quantile)| / |q(quantile)|``: right tail over left tail.

    Below 1 means the losing tail is fatter than the winning tail, which a
    Sharpe ratio cannot see.
    """
    a = _clean(returns)
    if a.size < 4:
        return math.nan
    right = abs(float(np.quantile(a, 1.0 - quantile)))
    left = abs(float(np.quantile(a, quantile)))
    if left <= 0.0:
        return math.inf if right > 0.0 else math.nan
    return right / left


def worst_streak(trade_returns: Iterable[float], n: int) -> float:
    """Worst cumulative outcome over any ``n`` consecutive trades.

    The honest answer to "how bad can this feel", and the number that should
    set the drawdown throttle rather than the observed maximum drawdown, which
    is a single realisation of an extreme-value distribution.
    """
    a = _clean(trade_returns)
    if a.size == 0 or n <= 0:
        return math.nan
    if a.size < n:
        return float(a.sum())
    csum = np.concatenate(([0.0], np.cumsum(a)))
    windows = csum[n:] - csum[:-n]
    return float(windows.min())


def max_consecutive_losses(trade_returns: Iterable[float]) -> int:
    """Longest run of non-positive trades."""
    a = _clean(trade_returns)
    run = best = 0
    for v in a:
        run = run + 1 if v <= 0.0 else 0
        best = max(best, run)
    return int(best)


def summarize_returns(
    returns: Iterable[float], periods_per_year: float = 252.0
) -> dict[str, float]:
    """Canonical descriptive bundle for a per-period log return series."""
    a = _clean(returns)
    n = int(a.size)
    sr_per_obs = sharpe_ratio(a, 1.0)
    sk = skewness(a)
    ku = kurtosis(a)
    out: dict[str, float] = {
        "n_periods": float(n),
        "total_return": total_return(a),
        "annualized_return": annualized_return(a, periods_per_year),
        "annualized_vol": volatility(a, periods_per_year),
        "sharpe_per_period": sr_per_obs,
        "sharpe_annualized": sr_per_obs * math.sqrt(periods_per_year)
        if np.isfinite(sr_per_obs)
        else math.nan,
        "sharpe_se_per_period": sharpe_standard_error(
            n, sr_per_obs, 0.0 if not np.isfinite(sk) else sk, 3.0 if not np.isfinite(ku) else ku
        ),
        "sortino_annualized": sortino_ratio(a, periods_per_year),
        "calmar": calmar_ratio(a, periods_per_year),
        "skew": sk,
        "kurtosis": ku,
        "tail_ratio": tail_ratio(a),
    }
    out.update(drawdown_stats(a).as_dict())
    return out


def summarize_trades(
    trade_returns: Iterable[float],
    ret_sigma: Iterable[float] | None = None,
    streak_n: int = 10,
) -> dict[str, float]:
    """Canonical descriptive bundle for a sequence of per-trade outcomes."""
    a = _clean(trade_returns)
    avg_w, avg_l, wl = average_win_loss(a)
    p = hit_rate(a)
    out: dict[str, float] = {
        "n_trades": float(a.size),
        "hit_rate": p,
        "hit_rate_se": win_rate_standard_error(int(a.size), p) if a.size else math.nan,
        "profit_factor": profit_factor(a),
        "avg_win": avg_w,
        "avg_loss": avg_l,
        "win_loss_ratio": wl,
        "mean_trade": float(a.mean()) if a.size else math.nan,
        "median_trade": float(np.median(a)) if a.size else math.nan,
        f"worst_{streak_n}_trade_streak": worst_streak(a, streak_n),
        "max_consecutive_losses": float(max_consecutive_losses(a)),
    }
    if ret_sigma is not None:
        s = _clean(ret_sigma)
        out["expectancy_sigma"] = expectancy_sigma(s)
        out["expectancy_sigma_se"] = (
            float(s.std(ddof=1) / math.sqrt(s.size)) if s.size > 1 else math.nan
        )
        if s.size > 1 and s.std(ddof=1) > 0.0:
            out["expectancy_t"] = float(s.mean() / (s.std(ddof=1) / math.sqrt(s.size)))
        else:
            out["expectancy_t"] = math.nan
    return out


# ---------------------------------------------------------------------------
# Inferential: PSR, DSR, MinTRL
# ---------------------------------------------------------------------------


def _psr_variance_term(sr: float, skew: float, kurt: float) -> float:
    """``1 - g3*SR + (g4-1)/4 * SR^2``: the non-normality correction.

    Negative skew and fat tails *inflate* this term, which *reduces* PSR: a
    strategy whose returns are many small wins and rare large losses needs a
    higher Sharpe than a Gaussian one to make the same claim.
    """
    return 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr * sr


def probabilistic_sharpe_ratio(
    sr: float, n: int, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0
) -> float:
    """Probability that the true Sharpe exceeds ``sr_benchmark``.

    Bailey & López de Prado (2012), eq. 3::

        PSR = Phi( (SR - SR*) * sqrt(n - 1)
                   / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    ``sr`` and ``sr_benchmark`` are **per-observation**, ``n`` is the number of
    observations, ``skew``/``kurt`` are the sample skewness and *raw* kurtosis
    (3 for a Gaussian) of those observations.

    A failing result (PSR below, say, 0.95) means the track record is simply too
    short, too skewed or too fat-tailed to distinguish the observed Sharpe from
    the benchmark -- irrespective of how large the point estimate is. It does
    *not* mean the strategy is bad; it means the evidence is not yet evidence.
    """
    if n < 2 or not np.isfinite(sr):
        return math.nan
    skew = 0.0 if not np.isfinite(skew) else float(skew)
    kurt = 3.0 if not np.isfinite(kurt) else float(kurt)
    var = _psr_variance_term(sr, skew, kurt)
    if var <= 0.0:
        return math.nan
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(var)
    return norm_cdf(z)


def expected_max_sharpe(n_trials: int, variance_of_trial_srs: float) -> float:
    """Expected maximum of ``n_trials`` IID null Sharpe ratios.

    Bailey & López de Prado (2014), eq. 5, the Gumbel/extreme-value
    approximation::

        E[max SR] = sqrt(Var[SR_trials]) * ( (1 - gamma) * Phi^-1(1 - 1/N)
                                             + gamma * Phi^-1(1 - 1/(N e)) )

    with ``gamma`` the Euler--Mascheroni constant. This is the Sharpe ratio you
    should *expect* to find by searching ``N`` variants of a strategy that has
    no edge at all. With 100 trials and a cross-trial Sharpe dispersion of 0.5
    per-observation-units, the best of them looks impressive by construction.

    ``n_trials <= 1`` returns 0.0: with a single pre-registered configuration
    there is no selection bias to charge for.
    """
    if n_trials <= 1:
        return 0.0
    sd = math.sqrt(max(float(variance_of_trial_srs), 0.0))
    if sd == 0.0:
        return 0.0
    z1 = norm_ppf(1.0 - 1.0 / n_trials)
    z2 = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sd * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    sr: float,
    n: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    n_trials: int = 1,
    variance_of_trial_srs: float = 0.0,
) -> float:
    """PSR against the *expected maximum* Sharpe under the null.

    Bailey & López de Prado (2014). The deflated Sharpe ratio is exactly the
    probabilistic Sharpe ratio with the benchmark raised from zero to
    :func:`expected_max_sharpe`, so it answers the only question that matters
    after a research programme: is this result better than the best result you
    would have found if nothing worked?

    ``n_trials`` must be the honest count of configurations ever evaluated
    against validation data -- ``Config.trials_ledger_count`` plus every
    parameter that was moved after looking at a result
    (``Config.fitted_dof()`` is a lower bound on that). Under-reporting trials
    is the single most effective way to fool this test, which is why the count
    lives in the frozen manifest and is hashed.
    """
    sr0 = expected_max_sharpe(n_trials, variance_of_trial_srs)
    return probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=sr0)


def minimum_track_record_length(
    sr: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    target_confidence: float = 0.95,
    sr_benchmark: float = 0.0,
) -> float:
    """Observations needed before PSR would reach ``target_confidence``.

    Bailey & López de Prado (2012), eq. 5::

        MinTRL = 1 + (1 - g3*SR + (g4-1)/4 * SR^2) * (Z_alpha / (SR - SR*))^2

    Returns ``inf`` when ``sr <= sr_benchmark`` -- no amount of additional data
    makes a non-edge significant. Compare the result with the number of
    observations actually available; if MinTRL exceeds it, the correct verdict
    is "not yet decidable", and any acceptance test that passes anyway is
    mis-specified.
    """
    if not np.isfinite(sr) or sr <= sr_benchmark:
        return math.inf
    skew = 0.0 if not np.isfinite(skew) else float(skew)
    kurt = 3.0 if not np.isfinite(kurt) else float(kurt)
    z = norm_ppf(target_confidence)
    var = _psr_variance_term(sr, skew, kurt)
    if var <= 0.0:
        return math.inf
    return 1.0 + var * (z / (sr - sr_benchmark)) ** 2


# ---------------------------------------------------------------------------
# Inferential: how many trades do you actually need?
# ---------------------------------------------------------------------------


def win_rate_standard_error(n_trades: int, p: float = 0.5) -> float:
    """Standard error of an observed win rate: ``sqrt(p (1-p) / n)``.

    **Read this before quoting a hit rate.** The binomial standard error is
    ruthless at the sample sizes discretionary trading actually produces:

    ======  =======  ==========================================
    trades  p        standard error (percentage points)
    ======  =======  ==========================================
    50      0.60     6.9
    95      0.60     5.0
    200     0.60     3.5
    1000    0.60     1.5
    ======  =======  ==========================================

    At 95 trades the 95% confidence interval around an observed 60% win rate is
    roughly [50.2%, 69.8%]. **Ninety-five trades cannot distinguish a 60% edge
    from a coin flip.** :func:`trades_needed_to_distinguish` puts the required
    sample at about 194 trades for 80% power, and that is for a *pre-specified*
    hypothesis with no parameter search.

    The practical consequence for this system is structural, not cosmetic:
    statistical power has to come from **pooling across instruments** (many
    short, independent-ish samples sharing one parameter set), not from a longer
    history of one instrument. A longer single-instrument history buys fewer
    effectively independent observations than it appears to, because the same
    regime persists across it, and it buys them from a market whose
    microstructure has changed. This is also why every parameter in
    :class:`~tia.config.Config` is scale-free: pooling is only legitimate if one
    parameter set is genuinely meant to apply everywhere.
    """
    if n_trades <= 0:
        return math.nan
    p = float(p)
    if not np.isfinite(p):
        return math.nan
    p = min(max(p, 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / n_trades)


def win_rate_confidence_interval(
    n_trades: int, wins: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a win rate.

    Wilson rather than the normal approximation because at these sample sizes
    the normal interval is both too narrow and capable of leaving [0, 1].
    """
    if n_trades <= 0:
        return (math.nan, math.nan)
    z = norm_ppf(0.5 * (1.0 + confidence))
    phat = wins / n_trades
    denom = 1.0 + z * z / n_trades
    centre = (phat + z * z / (2 * n_trades)) / denom
    half = (
        z
        * math.sqrt(phat * (1.0 - phat) / n_trades + z * z / (4 * n_trades * n_trades))
        / denom
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def trades_needed_to_distinguish(
    p0: float, p1: float, power: float = 0.80, alpha: float = 0.05, two_sided: bool = True
) -> int:
    """Sample size to reject ``p0`` in favour of ``p1`` at the stated power.

    Standard unpooled two-proportion-free formula for a single binomial::

        n = ( z_{alpha'} sqrt(p0 q0) + z_{power} sqrt(p1 q1) )^2 / (p1 - p0)^2

    with ``alpha' = alpha/2`` when two-sided. For the canonical case
    ``p0=0.5, p1=0.6, power=0.8, alpha=0.05`` this returns **194** -- roughly
    twice the ~95 trades a year of discretionary trading produces, which is the
    quantitative statement of why single-instrument track records are not
    evidence. Halving the effect size to ``p1=0.55`` quadruples the requirement.
    """
    if not (0.0 < p0 < 1.0 and 0.0 < p1 < 1.0):
        raise ValueError("probabilities must be strictly inside (0, 1)")
    if p1 == p0:
        return 2**31 - 1
    a = alpha / 2.0 if two_sided else alpha
    z_a = norm_ppf(1.0 - a)
    z_b = norm_ppf(power)
    num = z_a * math.sqrt(p0 * (1.0 - p0)) + z_b * math.sqrt(p1 * (1.0 - p1))
    return int(math.ceil((num * num) / ((p1 - p0) ** 2)))


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def regime_breakdown(
    returns: Iterable[float],
    regime_labels: Iterable[int] | Iterable[str],
    periods_per_year: float = 252.0,
) -> dict[str, dict[str, float]]:
    """Per-regime performance breakdown.

    Integer labels in ``0..3`` are mapped through
    :data:`~tia.types.REGIME_NAMES`, so passing :class:`~tia.types.Regime`
    values works directly.

    What a failure looks like: a strategy whose entire expectancy comes from one
    regime is a regime bet with extra steps, and its Sharpe is a statement about
    how much of the sample that regime occupied rather than about the edge. The
    two directional effects this system is built on point in opposite
    directions by regime (SPEC, :class:`~tia.types.Regime`), so an honest
    breakdown should show *both* TREND and REVERT contributing.
    """
    r = np.asarray(returns, dtype=float).ravel()
    labels = list(regime_labels)
    if len(labels) != r.size:
        raise ValueError(f"length mismatch: {r.size} returns vs {len(labels)} labels")
    keys: list[str] = []
    for lab in labels:
        if isinstance(lab, str):
            keys.append(lab)
        else:
            i = int(lab)
            keys.append(REGIME_NAMES[i] if 0 <= i < len(REGIME_NAMES) else f"regime_{i}")
    arr = np.asarray(keys, dtype=object)
    out: dict[str, dict[str, float]] = {}
    for key in sorted(set(keys)):
        sub = r[arr == key]
        sub = sub[np.isfinite(sub)]
        out[key] = {
            "n": float(sub.size),
            "share": float(sub.size / max(r.size, 1)),
            "mean": float(sub.mean()) if sub.size else math.nan,
            "total": float(sub.sum()) if sub.size else 0.0,
            "sharpe_per_period": sharpe_ratio(sub, 1.0),
            "sharpe_annualized": sharpe_ratio(sub, periods_per_year),
            "hit_rate": hit_rate(sub),
        }
    return out


@dataclass(frozen=True)
class FoldStability:
    """Dispersion of one metric across independent folds or CPCV paths.

    ``t_stat`` treats the folds as independent draws, which is optimistic even
    after purging; treat it as an upper bound on significance. ``fraction_positive``
    is the more robust statement and the one the acceptance test uses.
    """

    name: str
    n_folds: int
    mean: float
    std: float
    t_stat: float
    fraction_positive: float
    minimum: float
    median: float
    maximum: float
    iqr: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n_folds": float(self.n_folds),
            "mean": self.mean,
            "std": self.std,
            "t_stat": self.t_stat,
            "fraction_positive": self.fraction_positive,
            "min": self.minimum,
            "median": self.median,
            "max": self.maximum,
            "iqr": self.iqr,
        }


def stability_across_folds(values: Iterable[float], name: str = "metric") -> FoldStability:
    """Summarise a metric measured once per fold / per CPCV path."""
    a = _clean(values)
    k = int(a.size)
    if k == 0:
        return FoldStability(name, 0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if k > 1 else 0.0
    t = float(mean / (std / math.sqrt(k))) if (k > 1 and std > 0.0) else math.nan
    q75, q25 = (float(np.quantile(a, 0.75)), float(np.quantile(a, 0.25))) if k > 1 else (mean, mean)
    return FoldStability(
        name=name,
        n_folds=k,
        mean=mean,
        std=std,
        t_stat=t,
        fraction_positive=float(np.mean(a > 0.0)),
        minimum=float(a.min()),
        median=float(np.median(a)),
        maximum=float(a.max()),
        iqr=q75 - q25,
    )


def psi(
    expected: Iterable[float], actual: Iterable[float], bins: int = 10, epsilon: float = 1e-6
) -> float:
    """Population stability index between a reference and a current sample.

    ``PSI = sum_i (a_i - e_i) * ln(a_i / e_i)`` over quantile bins of the
    *expected* sample. Conventional reading: below 0.10 stable, 0.10--0.25 a
    warning, above 0.25 the distribution has moved and any model calibrated on
    the reference sample is now extrapolating. ``Config.psi_alarm`` defaults to
    0.25 for exactly this reason.

    Bins are quantiles of ``expected`` rather than equal width, so the statistic
    is invariant to monotone rescaling of the feature -- which matters here
    because most features in :class:`~tia.types.FeatureSnapshot` are already
    ranks or robust z-scores.
    """
    e = _clean(expected)
    a = _clean(actual)
    if e.size < bins or a.size == 0 or bins < 2:
        return math.nan
    qs = np.quantile(e, np.linspace(0.0, 1.0, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)
    if qs.size < 3:
        return 0.0
    e_hist, _ = np.histogram(e, bins=qs)
    a_hist, _ = np.histogram(a, bins=qs)
    e_frac = np.maximum(e_hist / e.size, epsilon)
    a_frac = np.maximum(a_hist / a.size, epsilon)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _check_prob_pairs(p: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    pa = np.asarray(p, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if pa.size != ya.size:
        raise ValueError(f"length mismatch: {pa.size} probabilities vs {ya.size} outcomes")
    ok = np.isfinite(pa) & np.isfinite(ya)
    pa, ya = pa[ok], ya[ok]
    if pa.size and (pa.min() < -1e-9 or pa.max() > 1.0 + 1e-9):
        raise ValueError("probabilities must lie in [0, 1]")
    if pa.size and not np.all((ya == 0.0) | (ya == 1.0)):
        raise ValueError("outcomes must be 0 or 1")
    return np.clip(pa, 0.0, 1.0), ya


def brier_score(p: Iterable[float], y: Iterable[float]) -> float:
    """Mean squared error of probabilistic forecasts.

    0.25 is the score of a constant 0.5 forecast, which is why
    ``Config.brier_alarm`` sits just above it: a probability model scoring worse
    than a coin flip is not a probability model, and the system demotes itself
    rather than sizing on it.
    """
    pa, ya = _check_prob_pairs(p, y)
    if pa.size == 0:
        return math.nan
    return float(np.mean((pa - ya) ** 2))


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy (1973) three-term partition: ``brier = reliability - resolution + uncertainty``.

    ``reliability`` (lower is better) is miscalibration: how far bin frequencies
    sit from the probabilities claimed in them. ``resolution`` (higher is
    better) is discrimination: how much the bin frequencies differ from the base
    rate. ``uncertainty`` is the base rate's own variance and is a property of
    the problem, not the model.

    The failure mode this separates is the important one: a model can improve
    its Brier score by becoming *less confident* (shrinking toward the base
    rate), which lowers reliability while destroying resolution. Such a model is
    well calibrated and useless -- it can never clear a probability gate like
    ``Config.p_min``.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    n_bins_used: int

    @property
    def residual(self) -> float:
        """Identity check; should be ~0 up to floating point."""
        return self.brier - (self.reliability - self.resolution + self.uncertainty)

    def as_dict(self) -> dict[str, float]:
        return {
            "brier": self.brier,
            "reliability": self.reliability,
            "resolution": self.resolution,
            "uncertainty": self.uncertainty,
            "bins_used": float(self.n_bins_used),
        }


def _bin_edges(bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(bins) + 1)


def _bin_assign(pa: np.ndarray, bins: int) -> np.ndarray:
    edges = _bin_edges(bins)
    idx = np.digitize(pa, edges[1:-1], right=False)
    return np.clip(idx, 0, bins - 1)


def brier_decomposition(
    p: Iterable[float], y: Iterable[float], bins: int = 10
) -> BrierDecomposition:
    """Decompose the Brier score into reliability, resolution and uncertainty."""
    pa, ya = _check_prob_pairs(p, y)
    n = pa.size
    if n == 0:
        return BrierDecomposition(math.nan, math.nan, math.nan, math.nan, 0)
    base = float(ya.mean())
    unc = base * (1.0 - base)
    assign = _bin_assign(pa, bins)
    rel = 0.0
    res = 0.0
    used = 0
    for b in range(bins):
        m = assign == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        used += 1
        pbar = float(pa[m].mean())
        obar = float(ya[m].mean())
        w = cnt / n
        rel += w * (pbar - obar) ** 2
        res += w * (obar - base) ** 2
    return BrierDecomposition(
        brier=brier_score(pa, ya),
        reliability=rel,
        resolution=res,
        uncertainty=unc,
        n_bins_used=used,
    )


def reliability_curve(
    p: Iterable[float], y: Iterable[float], bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean_forecast, observed_frequency, count)`` per non-empty bin.

    The diagonal is perfect calibration. A curve consistently *below* the
    diagonal means overconfidence, which in this system translates directly into
    oversizing, because the risk layer takes ``p_success`` at face value.
    """
    pa, ya = _check_prob_pairs(p, y)
    assign = _bin_assign(pa, bins)
    mp: list[float] = []
    freq: list[float] = []
    cnt: list[float] = []
    for b in range(bins):
        m = assign == b
        k = int(m.sum())
        if k == 0:
            continue
        mp.append(float(pa[m].mean()))
        freq.append(float(ya[m].mean()))
        cnt.append(float(k))
    return np.asarray(mp), np.asarray(freq), np.asarray(cnt)


def expected_calibration_error(
    p: Iterable[float], y: Iterable[float], bins: int = 10
) -> float:
    """Count-weighted mean absolute gap between claimed and realised frequency.

    ``ECE = sum_b (n_b / n) * |mean(p in b) - freq(y in b)|``. Unlike
    reliability (a squared term), ECE is in probability units and can be read
    directly: 0.04 means "the stated probabilities are off by four percentage
    points on average", which for a gate set at ``p_min = 0.58`` is material.
    """
    pa, ya = _check_prob_pairs(p, y)
    if pa.size == 0:
        return math.nan
    mp, freq, cnt = reliability_curve(pa, ya, bins)
    if cnt.size == 0:
        return math.nan
    return float(np.sum(cnt * np.abs(mp - freq)) / cnt.sum())
