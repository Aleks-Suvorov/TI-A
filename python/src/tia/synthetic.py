"""Synthetic market generators, for tests and for null-hypothesis checks.

Two uses, and they pull in opposite directions:

*   :func:`generate_null` produces a market with **no** exploitable directional
    structure but with realistic volatility clustering, fat tails and an intraday
    volume profile. Running the system on it must produce approximately nothing.
    A framework that finds trades here has a bug or an implicit bias, and this is
    the cheapest test that catches it.
*   :func:`generate_with_regimes` produces a market that genuinely alternates
    between trending and mean-reverting states. Running the system on it must
    produce something, and its regime posterior should track the truth. This is a
    *recovery* test: it verifies that the machinery can find structure that is
    definitely there, which is the necessary complement to the null test.

Both are seeded and deterministic. Neither is a claim about real markets: a
generator that produced realistic returns would be a solved research problem, and
passing a test against one's own simulator is weak evidence at best. Their value
is as a floor, not a ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import Bar

__all__ = ["generate_null", "generate_with_regimes", "bars_from_arrays", "SyntheticTruth"]


@dataclass(slots=True)
class SyntheticTruth:
    """Ground truth accompanying a generated series, for recovery tests."""

    regime: np.ndarray  #: 0 trending, 1 mean-reverting, per bar
    drift: np.ndarray
    sigma: np.ndarray


def _garch_path(
    n: int, rng: np.random.Generator, omega: float, alpha: float, beta: float, nu: float
) -> np.ndarray:
    """GARCH(1,1) with Student-t innovations: clustering plus fat tails."""
    eps = np.zeros(n)
    var = omega / max(1.0 - alpha - beta, 1e-3)
    t = rng.standard_t(nu, size=n) / math.sqrt(nu / (nu - 2.0))
    for i in range(n):
        var = omega + alpha * (eps[i - 1] ** 2 if i else 0.0) + beta * var
        eps[i] = math.sqrt(var) * t[i]
    return eps


def _intraday_profile(n: int, period: int) -> np.ndarray:
    """U-shaped volume multiplier: high at the open and close, low midday."""
    if period <= 1:
        return np.ones(n)
    x = (np.arange(n) % period) / period
    return 0.6 + 1.4 * (np.cos(2.0 * math.pi * x) * 0.5 + 0.5) ** 2


def bars_from_arrays(
    close: np.ndarray,
    volume: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator,
    start_ts: float = 1_600_000_000.0,
    step_s: float = 86_400.0,
    substeps: int = 12,
    half_spread: float = 2.5e-4,
) -> list[Bar]:
    """Build OHLC bars by simulating the intrabar path, not by drawing a range.

    This matters more than it looks. Drawing a high and a low independently of
    the close-to-close move produces bars whose ranges are not the running
    extremes of any diffusion, and every range-based estimator in the system
    assumes they are. Corwin-Schultz in particular then over-estimates the
    effective spread by roughly an order of magnitude, which would make the cost
    model reject every trade -- a test failure caused entirely by the test data.

    So each bar is a Brownian bridge from the previous close to this bar's close,
    sampled at ``substeps`` points, with the true high and low taken from the
    path and a half-spread added to the high and subtracted from the low. The
    injected ``half_spread`` also gives the spread estimators a known ground
    truth to be validated against.
    """
    n = close.size
    bars: list[Bar] = []
    prev = float(close[0])
    for i in range(n):
        c = float(close[i])
        s = float(max(sigma[i], 1e-9))
        o = prev if i else c
        # Brownian bridge in log space from o to c over `substeps` increments.
        lo_o, lo_c = math.log(o), math.log(c)
        w = np.cumsum(rng.standard_normal(substeps)) / math.sqrt(substeps)
        t = np.arange(1, substeps + 1) / substeps
        bridge = w - t * w[-1]
        path = lo_o + t * (lo_c - lo_o) + s * bridge
        hi_true = float(np.exp(max(path.max(), lo_o, lo_c)))
        lo_true = float(np.exp(min(path.min(), lo_o, lo_c)))
        hi = hi_true * (1.0 + half_spread)
        lo = lo_true * (1.0 - half_spread)
        # The close is a *transaction* price, not the mid: it prints on the bid
        # or the ask with equal probability. This is what a real close is, and
        # the Abdi-Ranaldo estimator's derivation depends on it -- fed a series
        # of mids, that estimator understates the spread by roughly half.
        c_obs = c * (1.0 + half_spread * (1.0 if rng.random() < 0.5 else -1.0))
        c_obs = min(max(c_obs, lo), hi)
        bars.append(
            Bar(
                timestamp=start_ts + i * step_s,
                open=o,
                high=max(hi, o, c_obs),
                low=min(lo, o, c_obs),
                close=c_obs,
                volume=float(max(volume[i], 1.0)),
            )
        )
        prev = c_obs
    return bars


def generate_null(
    n: int = 3000,
    seed: int = 0,
    start_price: float = 100.0,
    bars_per_session: int = 1,
) -> list[Bar]:
    """A martingale with realistic second-moment structure and no edge.

    Returns are conditionally heteroskedastic and fat-tailed, volume covaries
    with volatility and follows an intraday profile, but the conditional mean is
    exactly zero. Any strategy that profits here beyond sampling noise is either
    reading the future or exploiting an artefact of the cost model.
    """
    rng = np.random.default_rng(seed)
    eps = _garch_path(n, rng, omega=2e-6, alpha=0.09, beta=0.89, nu=5.0)
    sigma = np.abs(eps) * 0.0 + np.sqrt(
        np.convolve(eps**2, np.ones(20) / 20.0, mode="same") + 1e-8
    )
    close = start_price * np.exp(np.cumsum(eps))
    profile = _intraday_profile(n, bars_per_session)
    volume = (
        1e6
        * profile
        * np.exp(0.8 * (sigma / max(float(np.mean(sigma)), 1e-12) - 1.0))
        * np.exp(0.3 * rng.standard_normal(n))
    )
    return bars_from_arrays(close, volume, sigma, rng)


def generate_with_regimes(
    n: int = 6000,
    seed: int = 1,
    start_price: float = 100.0,
    mean_dwell: int = 250,
    trend_strength: float = 0.06,
    revert_strength: float = 0.10,
    bars_per_session: int = 1,
) -> tuple[list[Bar], SyntheticTruth]:
    """A market that genuinely alternates between persistence and reversion.

    In the trending state, returns carry a small positive autocorrelation via a
    persistent drift. In the reverting state, an Ornstein-Uhlenbeck pull toward a
    slow-moving anchor produces negative autocorrelation. ``trend_strength`` and
    ``revert_strength`` are deliberately modest -- comparable to the information
    coefficients that are actually achievable, per ``docs/01-THEORY.md`` §0 -- so
    that a system which only performs here at implausible effect sizes is
    revealed as such.
    """
    rng = np.random.default_rng(seed)
    eps = _garch_path(n, rng, omega=2e-6, alpha=0.08, beta=0.90, nu=5.0)
    sigma = np.sqrt(np.convolve(eps**2, np.ones(20) / 20.0, mode="same") + 1e-8)

    regime = np.zeros(n, dtype=np.int64)
    state = 0
    switch_p = 1.0 / max(mean_dwell, 2)
    for i in range(n):
        if rng.random() < switch_p:
            state = 1 - state
        regime[i] = state

    logp = np.zeros(n)
    drift = np.zeros(n)
    anchor = math.log(start_price)
    mom = 0.0
    lp = math.log(start_price)
    for i in range(n):
        s = float(sigma[i])
        if regime[i] == 0:
            mom = 0.94 * mom + trend_strength * s * rng.standard_normal()
            d = mom
            anchor = 0.99 * anchor + 0.01 * lp
        else:
            mom = 0.0
            d = -revert_strength * (lp - anchor)
            anchor = 0.995 * anchor + 0.005 * lp
        drift[i] = d
        lp = lp + d + float(eps[i])
        logp[i] = lp

    close = np.exp(logp)
    profile = _intraday_profile(n, bars_per_session)
    volume = (
        1e6
        * profile
        * np.exp(0.8 * (sigma / max(float(np.mean(sigma)), 1e-12) - 1.0))
        * np.exp(0.3 * rng.standard_normal(n))
    )
    bars = bars_from_arrays(close, volume, sigma, rng)
    return bars, SyntheticTruth(regime=regime, drift=drift, sigma=sigma)
