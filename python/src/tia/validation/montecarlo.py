"""Null-hypothesis machinery: bootstraps, permutations and surrogate data.

This is the file that decides whether the edge is real, and it does so by making
the null hypothesis *concrete*. A p-value is only as meaningful as the null it is
computed against, and in trading almost every interesting failure mode is a null
that was chosen for convenience:

*   **IID bootstrap** (:func:`iid_bootstrap`) destroys all serial dependence.
    Too easy a null: any strategy that merely rides volatility clustering beats
    it, because the surrogate has none.
*   **Stationary block bootstrap** (:func:`stationary_block_bootstrap`) preserves
    dependence up to the block length. The right null for "is the mean return
    distinguishable from zero given how autocorrelated and heteroskedastic the
    series is".
*   **Trade-order permutation** (:func:`trade_order_permutation`) keeps the exact
    multiset of trade outcomes and destroys their order. It tests a specific and
    very common artifact: performance that came from *when* the good trades
    happened (compounding, drawdown-throttled sizing, a lucky ordering that kept
    the equity curve above a kill switch) rather than from the trades themselves.
*   **GARCH surrogates** (:func:`garch_surrogate`) preserve volatility clustering
    and fat tails while destroying directional structure. Read that function's
    docstring; it is the most diagnostic test in the module.
*   **Sign scrambling** (:func:`sign_scramble_surrogate`) is the cheap version of
    the same idea and needs no model fit.

One-sided by default
--------------------
:func:`mc_pvalue` and :func:`surrogate_pvalue` use the "greater is better"
convention with the ``(1 + #{null >= observed}) / (1 + n_paths)`` estimator
(Davison & Hinkley 1997, §4.2). The ``+1`` keeps the p-value away from an
impossible exact zero and makes it exact under the randomisation hypothesis:
with 2000 paths the smallest reportable p-value is 1/2001.

References
----------
Politis, D. and Romano, J. (1994), "The Stationary Bootstrap", *JASA* 89(428).
Bollerslev, T. (1986), "Generalized Autoregressive Conditional
Heteroskedasticity", *Journal of Econometrics* 31.
Theiler, J. et al. (1992), "Testing for Nonlinearity in Time Series: the Method
of Surrogate Data", *Physica D* 58 -- the surrogate-data logic this module
applies to returns.
White, H. (2000), "A Reality Check for Data Snooping", *Econometrica* 68(5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "stationary_block_bootstrap",
    "iid_bootstrap",
    "trade_order_permutation",
    "GarchFit",
    "fit_garch11",
    "garch_surrogate",
    "sign_scramble_surrogate",
    "mc_pvalue",
    "surrogate_pvalue",
    "bootstrap_ci",
    "autocorrelation",
    "path_statistics",
    "SCIPY_AVAILABLE",
]


try:  # pragma: no cover - environment dependent
    from scipy.optimize import minimize as _scipy_minimize

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _scipy_minimize = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


def _as_returns(returns: Sequence[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(returns, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("empty return series")
    if not np.all(np.isfinite(a)):
        raise ValueError("return series contains non-finite values; clean it first")
    return a


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def stationary_block_bootstrap(
    returns: Sequence[float] | np.ndarray,
    block_len: float,
    n_paths: int,
    rng: np.random.Generator,
    length: int | None = None,
) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap.

    Blocks have **geometrically distributed** lengths with mean ``block_len``
    (restart probability ``p = 1 / block_len``) and wrap around the end of the
    series. The random block length is what makes the resampled series
    stationary; fixed-length blocks (Künsch's moving-block bootstrap) impose a
    periodicity of the block length on the surrogate, which shows up as spurious
    structure at exactly the horizon a trading strategy is looking at.

    ``block_len`` should be long enough to carry volatility clustering through
    the resample -- ``Config.bootstrap_block_bars`` defaults to 60 for that
    reason. Too short and the null loses the heteroskedasticity that the
    strategy might be harvesting, which flatters the strategy.

    Returns an array of shape ``(n_paths, length)``, ``length`` defaulting to the
    input length. Circular wrapping makes the resample's expected mean exactly
    the sample mean, so a mean-based statistic is unbiased under this null -- the
    test is about dispersion, not centring.
    """
    a = _as_returns(returns)
    n = a.size
    m = int(length) if length is not None else n
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if not (block_len >= 1.0):
        raise ValueError("block_len must be at least 1")
    p = 1.0 / float(block_len)
    idx = np.empty((n_paths, m), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_paths)
    if m > 1:
        restart = rng.random((n_paths, m - 1)) < p
        fresh = rng.integers(0, n, size=(n_paths, m - 1))
        for t in range(1, m):
            cont = (idx[:, t - 1] + 1) % n
            idx[:, t] = np.where(restart[:, t - 1], fresh[:, t - 1], cont)
    return a[idx]


def iid_bootstrap(
    returns: Sequence[float] | np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
    length: int | None = None,
) -> np.ndarray:
    """Plain resampling with replacement: the *deliberately weak* null.

    Reported alongside the block bootstrap only as a contrast. If a strategy's
    p-value is small against the IID null and large against the block bootstrap,
    the "edge" is serial dependence that the IID null threw away -- i.e. the
    strategy is a volatility or autocorrelation harvester, and its returns will
    behave nothing like the IID null's when it is levered.
    """
    a = _as_returns(returns)
    m = int(length) if length is not None else a.size
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    return a[rng.integers(0, a.size, size=(n_paths, m))]


def trade_order_permutation(
    trade_returns: Sequence[float] | np.ndarray, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle the *sequence* of trade returns, keeping the multiset exact.

    Every permuted path has, by construction, the same number of trades, the same
    mean, the same standard deviation, the same hit rate and the same profit
    factor as the original. Anything that differs between the observed path and
    the permutations is therefore attributable purely to **order**: compounding,
    the drawdown throttle in ``risk/limits.py``, the kill switch, sizing that
    depends on recent equity, or a maximum drawdown that happened to arrive
    early rather than late.

    Use it with path-dependent statistics -- terminal compounded equity, maximum
    drawdown, longest losing streak, time under water. Using it with the mean is
    a null operation, and a p-value near 0.5 there is a correct result, not a
    failure.
    """
    a = _as_returns(trade_returns)
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    out = np.empty((n_paths, a.size), dtype=float)
    for i in range(n_paths):
        out[i] = rng.permutation(a)
    return out


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GarchFit:
    """Quasi-maximum-likelihood GARCH(1,1) fit.

    ``sigma2`` is the fitted conditional variance path, ``std_resid`` the
    standardised residuals ``(r - mu) / sqrt(sigma2)``. ``persistence`` is
    ``alpha + beta``; values above ~0.99 mean the variance process is close to
    integrated, which is normal for financial returns and is precisely the
    property a surrogate must reproduce.
    """

    mu: float
    omega: float
    alpha: float
    beta: float
    loglik: float
    sigma2: np.ndarray
    std_resid: np.ndarray
    method: str
    converged: bool

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        pers = self.persistence
        if pers >= 1.0:
            return math.inf
        return self.omega / (1.0 - pers)

    def as_dict(self) -> dict[str, float]:
        return {
            "mu": self.mu,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "persistence": self.persistence,
            "loglik": self.loglik,
            "converged": float(self.converged),
        }


_MAX_PERSISTENCE = 0.9995


def _garch_filter(
    e: np.ndarray, omega: float, alpha: float, beta: float, h0: float
) -> np.ndarray:
    n = e.size
    h = np.empty(n, dtype=float)
    prev = h0
    for t in range(n):
        h[t] = prev
        prev = omega + alpha * e[t] * e[t] + beta * prev
    return h


def _garch_negloglik(
    params: np.ndarray, e: np.ndarray, h0: float
) -> float:
    omega, alpha, beta = float(params[0]), float(params[1]), float(params[2])
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= _MAX_PERSISTENCE:
        return 1e12
    h = _garch_filter(e, omega, alpha, beta, h0)
    if not np.all(np.isfinite(h)) or np.any(h <= 0.0):
        return 1e12
    ll = -0.5 * np.sum(np.log(2.0 * math.pi) + np.log(h) + e * e / h)
    if not np.isfinite(ll):
        return 1e12
    return float(-ll)


def fit_garch11(
    returns: Sequence[float] | np.ndarray,
    zero_mean: bool = False,
    method: str = "auto",
) -> GarchFit:
    """Fit GARCH(1,1) by Gaussian quasi-maximum likelihood.

    Model, Bollerslev (1986)::

        r_t   = mu + e_t
        e_t   = sqrt(h_t) * z_t,     z_t iid, mean 0, variance 1
        h_t   = omega + alpha * e_{t-1}^2 + beta * h_{t-1}

    ``mu`` is fixed at the sample mean (or 0 with ``zero_mean=True``) rather than
    jointly estimated: the mean is a nuisance parameter here, its QML estimate is
    the sample mean anyway to first order, and profiling it out makes the
    likelihood surface far better behaved at the sample sizes involved.

    ``method``:

    ``"auto"``
        Use scipy's L-BFGS-B from several starting points when scipy is
        importable, otherwise fall back to the grid search.
    ``"grid"``
        Deterministic profile-likelihood grid over ``(alpha, beta)`` with
        variance targeting ``omega = var * (1 - alpha - beta)``, then a local
        refinement. No optional dependency, and accurate to a few percent in the
        parameters -- which is ample, because the surrogate only needs the
        *shape* of the volatility process, not a publication-grade estimate.

    Gaussian QML is used even though returns are not Gaussian: the QML estimator
    is consistent for the variance dynamics under fairly weak conditions, and the
    surrogate does not use the Gaussian assumption at all -- it bootstraps the
    empirical standardised residuals, so the fitted fat tails come along for the
    ride.
    """
    a = _as_returns(returns)
    if a.size < 32:
        raise ValueError("GARCH(1,1) needs at least 32 observations to be meaningful")
    mu = 0.0 if zero_mean else float(a.mean())
    e = a - mu
    var = float(np.mean(e * e))
    if var <= 0.0:
        raise ValueError("degenerate (zero-variance) return series")
    h0 = var

    use_scipy = SCIPY_AVAILABLE and method in ("auto", "scipy")
    if method == "scipy" and not SCIPY_AVAILABLE:
        raise RuntimeError("method='scipy' requested but scipy is not importable")

    best: tuple[float, tuple[float, float, float], str, bool] | None = None

    if use_scipy:
        starts = [
            (var * 0.10, 0.08, 0.85),
            (var * 0.05, 0.05, 0.92),
            (var * 0.30, 0.15, 0.70),
            (var * 0.50, 0.30, 0.40),
        ]
        bounds = [(1e-14, 10.0 * var + 1e-12), (0.0, 0.999), (0.0, 0.999)]
        for s in starts:
            try:
                res = _scipy_minimize(  # type: ignore[misc]
                    _garch_negloglik,
                    np.asarray(s, dtype=float),
                    args=(e, h0),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
            except Exception:  # pragma: no cover - optimiser robustness
                continue
            nll = float(res.fun)
            p = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
            if p[1] + p[2] >= _MAX_PERSISTENCE or p[0] <= 0.0:
                continue
            if best is None or nll < best[0]:
                best = (nll, p, "scipy-lbfgsb", bool(res.success))

    if best is None:
        best = _fit_garch11_grid(e, var, h0)

    nll, (omega, alpha, beta), used_method, converged = best
    h = _garch_filter(e, omega, alpha, beta, h0)
    z = e / np.sqrt(h)
    return GarchFit(
        mu=mu,
        omega=omega,
        alpha=alpha,
        beta=beta,
        loglik=-nll,
        sigma2=h,
        std_resid=z,
        method=used_method,
        converged=converged,
    )


def _fit_garch11_grid(
    e: np.ndarray, var: float, h0: float
) -> tuple[float, tuple[float, float, float], str, bool]:
    """Deterministic variance-targeted grid search, the scipy-free fallback."""

    def evaluate(alphas: np.ndarray, betas: np.ndarray) -> tuple[float, tuple[float, float, float]]:
        best_nll = math.inf
        best_p = (var * 0.1, 0.05, 0.85)
        for al in alphas:
            for be in betas:
                if al < 0.0 or be < 0.0 or al + be >= _MAX_PERSISTENCE:
                    continue
                om = var * (1.0 - al - be)
                if om <= 0.0:
                    continue
                nll = _garch_negloglik(np.asarray([om, al, be]), e, h0)
                if nll < best_nll:
                    best_nll = nll
                    best_p = (om, al, be)
        return best_nll, best_p

    coarse_a = np.linspace(0.005, 0.40, 24)
    coarse_b = np.linspace(0.30, 0.99, 36)
    nll, p = evaluate(coarse_a, coarse_b)
    # Local refinement around the coarse optimum.
    fine_a = np.clip(np.linspace(p[1] - 0.02, p[1] + 0.02, 9), 0.0, 0.95)
    fine_b = np.clip(np.linspace(p[2] - 0.03, p[2] + 0.03, 13), 0.0, 0.999)
    nll2, p2 = evaluate(fine_a, fine_b)
    if nll2 < nll:
        nll, p = nll2, p2
    # Finally relax the variance-targeting constraint on omega only.
    om_grid = p[0] * np.asarray([0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.6])
    for om in om_grid:
        cand = _garch_negloglik(np.asarray([om, p[1], p[2]]), e, h0)
        if cand < nll:
            nll, p = cand, (float(om), p[1], p[2])
    return nll, p, "grid-variance-targeted", math.isfinite(nll)


def garch_surrogate(
    returns: Sequence[float] | np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
    fit: GarchFit | None = None,
    include_drift: bool = False,
    burn_in: int = 250,
    length: int | None = None,
    method: str = "auto",
) -> np.ndarray:
    """Simulate GARCH(1,1) surrogate return paths with bootstrapped residuals.

    **The point of this test, stated loudly.** A GARCH surrogate preserves the
    two features of financial returns that are genuinely easy to exploit by
    accident -- **volatility clustering** and **fat tails** -- while destroying
    everything about *direction*: no autocorrelation in signed returns, no trend,
    no mean reversion, no structure, no relationship between any feature and the
    sign of the next return. Under this null there is no directional edge to
    find, by construction.

    Therefore, if the strategy **still makes money on the surrogates**, it is not
    trading direction. It is doing one of these instead, and each has a specific
    fingerprint:

    *   **Harvesting volatility.** Barriers stated in sigma units, a stop wider
        than the target or vice versa, and a vertical barrier together define a
        payoff that has non-zero expectation under *any* symmetric random walk
        with clustered volatility. A ``2.6 sigma`` target against a ``1.6 sigma``
        stop is a bet on hitting the nearer barrier first; that bet has a
        computable, direction-free expectation which is *not* zero once
        volatility is time-varying.
    *   **A cost-model artifact.** If modelled costs are subtracted at entry but
        the exit fill is optimistic (a favourable gap credited at the open, a
        stop filled at the stop price when it gapped through), the strategy earns
        the difference on every trade regardless of direction. This is the single
        most common reason a surrogate test comes back positive, which is why
        :mod:`tia.labeling.triple_barrier` refuses to credit favourable gaps.
    *   **Selection through the volatility filter.** Entering only in
        low-volatility states and exiting on expansion is a positive-expectation
        *volatility* trade under GARCH, with no directional content at all.

    None of these are frauds -- a volatility harvester can be a real business --
    but they are not the strategy this system claims to be, they will not
    survive a change in the cost model, and they must never be reported as
    directional edge. Hence :func:`surrogate_pvalue` must come back **small**:
    the observed performance has to be *outside* the surrogate distribution.

    Parameters
    ----------
    include_drift
        ``False`` (default) sets the surrogate mean to zero, so the null is "no
        directional edge *and* no drift". ``True`` retains the fitted ``mu``,
        which is the right null for a long-biased strategy that should be
        charged for the market's own drift. Choose before looking at the result
        and record the choice: switching afterwards is p-hacking.
    burn_in
        Discarded warm-up steps so the simulated variance process starts from its
        stationary distribution rather than from the unconditional variance.

    Returns
    -------
    ndarray of shape ``(n_paths, length)``.
    """
    a = _as_returns(returns)
    g = fit if fit is not None else fit_garch11(a, method=method)
    m = int(length) if length is not None else a.size
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")

    z_pool = g.std_resid[np.isfinite(g.std_resid)]
    if z_pool.size < 8:
        raise ValueError("too few standardised residuals to bootstrap")
    # Re-standardise the pool so the simulated variance is not biased by a
    # residual sample whose variance is not exactly 1.
    z_pool = (z_pool - z_pool.mean()) / z_pool.std(ddof=0)

    total = burn_in + m
    z = z_pool[rng.integers(0, z_pool.size, size=(n_paths, total))]
    h = np.full(n_paths, g.unconditional_variance if np.isfinite(g.unconditional_variance) else float(np.var(a)))
    out = np.empty((n_paths, m), dtype=float)
    mu = g.mu if include_drift else 0.0
    for t in range(total):
        e = np.sqrt(h) * z[:, t]
        if t >= burn_in:
            out[:, t - burn_in] = mu + e
        h = g.omega + g.alpha * e * e + g.beta * h
    return out


def sign_scramble_surrogate(
    returns: Sequence[float] | np.ndarray,
    block_len: int,
    n_paths: int,
    rng: np.random.Generator,
    permute_within_block: bool = False,
) -> np.ndarray:
    """Randomise return *signs* inside volatility-matched blocks.

    The cheap sibling of :func:`garch_surrogate`, with the same purpose and no
    model fit. The series is cut into contiguous blocks of ``block_len`` bars --
    short enough that volatility is roughly constant within a block, which is
    what "volatility matched" means -- and each return's sign is flipped
    independently with probability one half.

    What is preserved and what is destroyed:

    *   Preserved **exactly**: the sequence of absolute returns, hence the
        realised volatility path, the volatility clustering, and the
        unconditional fat tails.
    *   Destroyed: the sign sequence, therefore all directional predictability,
        all return autocorrelation and all trend.

    So the interpretation is identical to the GARCH surrogate: a strategy that
    still profits here is trading volatility or exploiting an accounting
    asymmetry, not direction. Because the magnitude path is held fixed rather
    than resampled, this null is *tighter* than the GARCH one (less
    path-to-path variation in volatility), which makes it a slightly harsher
    test of the directional claim and a slightly weaker test of robustness to
    volatility regimes. Run both.

    ``permute_within_block=True`` additionally shuffles the order of returns
    inside each block, which decorrelates within-block structure while keeping
    the block's volatility level -- use it when the strategy's horizon is shorter
    than ``block_len``.
    """
    a = _as_returns(returns)
    n = a.size
    if block_len < 1:
        raise ValueError("block_len must be at least 1")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    out = np.empty((n_paths, n), dtype=float)
    signs = np.where(rng.random((n_paths, n)) < 0.5, -1.0, 1.0)
    base = np.broadcast_to(a, (n_paths, n)).copy()
    if permute_within_block:
        for start in range(0, n, block_len):
            stop = min(start + block_len, n)
            width = stop - start
            if width < 2:
                continue
            order = np.argsort(rng.random((n_paths, width)), axis=1)
            base[:, start:stop] = np.take_along_axis(base[:, start:stop], order, axis=1)
    np.multiply(base, signs, out=out)
    return out


# ---------------------------------------------------------------------------
# p-values and intervals
# ---------------------------------------------------------------------------


def mc_pvalue(
    observed: float,
    null_stats: Sequence[float] | np.ndarray,
    alternative: str = "greater",
) -> float:
    """Monte Carlo p-value with the ``(1 + count) / (1 + n)`` estimator.

    ``alternative``:

    ``"greater"``
        ``P(null >= observed)``. Use when large values of the statistic are the
        claim (Sharpe, expectancy, total return). This is the default because it
        is the case that matters.
    ``"less"``
        ``P(null <= observed)``. Use for statistics where small is the claim
        (maximum drawdown, Brier score).
    ``"two-sided"``
        ``P(|null - median(null)| >= |observed - median(null)|)``, centred on the
        null's own median so that an asymmetric null distribution does not bias
        the test.

    The ``+1`` in numerator and denominator is not a fudge: it makes the p-value
    the exact randomisation-test p-value when the observed value is included in
    the reference set, and it prevents reporting ``p = 0``, which no finite
    simulation can support.
    """
    s = np.asarray(null_stats, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0 or not np.isfinite(observed):
        return math.nan
    if alternative == "greater":
        count = int(np.sum(s >= observed))
    elif alternative == "less":
        count = int(np.sum(s <= observed))
    elif alternative == "two-sided":
        centre = float(np.median(s))
        count = int(np.sum(np.abs(s - centre) >= abs(observed - centre)))
    else:
        raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")
    return (1.0 + count) / (1.0 + s.size)


def surrogate_pvalue(
    observed_stat: float, surrogate_stats: Sequence[float] | np.ndarray
) -> float:
    """One-sided p-value of an observed statistic against surrogate paths.

    Thin, explicit alias for ``mc_pvalue(..., alternative="greater")``, provided
    because the surrogate test's direction is the thing most often reversed by
    accident. The claim being tested is *"the strategy does better than it would
    on data with no directional structure"*, so:

    *   **small p (say <= 0.05)** -- the observed performance is outside the
        surrogate distribution: consistent with a genuine directional edge.
    *   **large p** -- volatility-only surrogates reproduce the result. The
        strategy is not measuring direction. This is a **fail**, however good the
        headline Sharpe.
    """
    return mc_pvalue(observed_stat, surrogate_stats, alternative="greater")


def bootstrap_ci(
    samples: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    method: str = "percentile",
    observed: float | None = None,
) -> tuple[float, float]:
    """Confidence interval from bootstrap replicates of a statistic.

    ``method="percentile"``
        The ``[alpha/2, 1 - alpha/2]`` quantiles of the replicates. Simple, and
        adequate when the replicate distribution is roughly symmetric.
    ``method="basic"``
        The basic (reverse-percentile) interval
        ``[2*observed - q_{1-alpha/2}, 2*observed - q_{alpha/2}]``, which
        corrects first-order bias and requires ``observed``. Prefer it when the
        replicate distribution is visibly shifted away from ``observed`` --
        Sharpe ratios on short samples usually are.

    Both are two-sided; halve ``alpha`` yourself for a one-sided bound.
    """
    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size < 2:
        return (math.nan, math.nan)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    lo_q, hi_q = float(np.quantile(s, alpha / 2.0)), float(np.quantile(s, 1.0 - alpha / 2.0))
    if method == "percentile":
        return (lo_q, hi_q)
    if method == "basic":
        if observed is None:
            raise ValueError("method='basic' requires the observed statistic")
        return (2.0 * observed - hi_q, 2.0 * observed - lo_q)
    raise ValueError("method must be 'percentile' or 'basic'")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def autocorrelation(x: Sequence[float] | np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Sample autocorrelation at lags ``1..max_lag`` (biased 1/n estimator).

    Used to *verify* what a surrogate destroyed and what it preserved: after a
    sign scramble or a GARCH surrogate, the autocorrelation of the signed series
    must collapse toward zero while the autocorrelation of squared or absolute
    returns must survive. The test suite asserts exactly that, because a
    surrogate generator that quietly destroys volatility clustering turns the
    surrogate test into an easier null and therefore into a rubber stamp.
    """
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    if n < 3 or max_lag < 1:
        return np.zeros(0, dtype=float)
    d = a - a.mean()
    denom = float(np.sum(d * d))
    if denom <= 0.0:
        return np.zeros(max_lag, dtype=float)
    out = np.empty(int(max_lag), dtype=float)
    for k in range(1, int(max_lag) + 1):
        out[k - 1] = float(np.sum(d[k:] * d[:-k]) / denom) if k < n else 0.0
    return out


def path_statistics(
    paths: np.ndarray, stat_fn: Callable[[np.ndarray], float]
) -> np.ndarray:
    """Apply a scalar statistic to every row of a ``(n_paths, n)`` array.

    A plain loop rather than ``np.apply_along_axis`` so that ``stat_fn`` may be
    any Python callable -- typically a closure that runs the whole strategy over
    a surrogate price path rather than merely summarising returns.
    """
    p = np.asarray(paths, dtype=float)
    if p.ndim != 2:
        raise ValueError("paths must be a 2-D (n_paths, n) array")
    return np.asarray([float(stat_fn(p[i])) for i in range(p.shape[0])], dtype=float)
