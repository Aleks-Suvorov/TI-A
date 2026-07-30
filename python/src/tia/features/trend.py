"""Trend measured as a significance rather than a level.

``docs/01-THEORY.md`` §4. The complaint against conventional trend indicators is
not that they are wrong but that they are unthresholdable: a moving-average
slope has units of price per bar, so any threshold on it encodes an implicit and
usually unexamined assumption about volatility. Since volatility varies by an
order of magnitude within a single instrument's history, such a threshold is
wrong most of the time.

Both estimators here return t-statistics against explicit nulls:

*   :class:`KalmanTrend` gives the filtered slope divided by its own posterior
    standard deviation -- "how sure is the filter that a trend exists?"
*   :class:`VarianceRatio` gives the Lo-MacKinlay heteroskedasticity-robust
    statistic, asymptotically N(0,1) under the martingale null -- "are
    increments positively or negatively autocorrelated, significantly?"

Only forward recursions are used. The Kalman *smoother*, which is what most
published "Kalman trend" examples actually plot, conditions on future
observations and would repaint every historical value.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import safe_div
from .rolling import RingBuffer, RollingMoments, RollingSum

__all__ = ["KalmanTrend", "VarianceRatio", "PathEfficiency", "MultiHorizonAgreement"]


class KalmanTrend:
    """Local linear trend filter on log price, with volatility-tied covariances.

    State :math:`x_t=(\\mu_t,\\beta_t)^\\top` -- level and slope -- with

    .. math::
        x_t = \\begin{pmatrix}1&1\\\\0&1\\end{pmatrix}x_{t-1}+\\eta_t,
        \\qquad \\ln C_t = (1\\;0)x_t + \\varepsilon_t.

    Both :math:`Q=\\mathrm{diag}(q_\\mu,q_\\beta)\\hat\\sigma^2` and
    :math:`R=\\hat\\sigma^2` scale with the *forecast* volatility, which is what
    makes the filter adapt without any user setting: the same dimensionless
    signal-to-noise ratio produces a slow filter in turbulent conditions and a
    responsive one in calm conditions. The two ratios are the only free
    quantities, and because they are ratios rather than periods, one value serves
    every instrument.
    """

    __slots__ = ("_q_mu", "_q_beta", "_x", "_P", "_init", "_n")

    def __init__(self, snr_level: float = 0.02, snr_slope: float = 0.002) -> None:
        if snr_level <= 0.0 or snr_slope <= 0.0:
            raise ValueError("signal-to-noise ratios must be positive")
        self._q_mu = float(snr_level)
        self._q_beta = float(snr_slope)
        self._x = np.zeros(2)
        self._P = np.eye(2)
        self._init = False
        self._n = 0

    def reset(self) -> None:
        self._x = np.zeros(2)
        self._P = np.eye(2)
        self._init = False
        self._n = 0

    def update(self, log_price: float, sigma: float) -> None:
        if log_price != log_price or not (sigma > 0.0):
            return
        s2 = sigma * sigma

        if not self._init:
            # Diffuse but not improper: ten sigma of level uncertainty and one
            # sigma of slope uncertainty. A truly diffuse prior would make the
            # first few t-statistics meaningless rather than merely wide.
            self._x[:] = (log_price, 0.0)
            self._P = np.diag([100.0 * s2, s2])
            self._init = True
            self._n = 1
            return

        # Predict.
        self._x[0] += self._x[1]
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self._P = F @ self._P @ F.T
        self._P[0, 0] += self._q_mu * s2
        self._P[1, 1] += self._q_beta * s2

        # Update.
        resid = log_price - self._x[0]
        S = self._P[0, 0] + s2
        if S <= 0.0:
            return
        K = self._P[:, 0] / S
        self._x = self._x + K * resid
        self._P = self._P - np.outer(K, self._P[0, :])
        # Enforce symmetry and positivity; the rank-one downdate above can drift
        # a few ulps and, over a long run, produce a negative variance.
        self._P = 0.5 * (self._P + self._P.T)
        np.fill_diagonal(self._P, np.clip(np.diag(self._P), 1e-300, None))
        self._n += 1

    @property
    def level(self) -> float:
        return float(self._x[0]) if self._init else math.nan

    @property
    def slope(self) -> float:
        return float(self._x[1]) if self._init else math.nan

    @property
    def slope_var(self) -> float:
        return float(self._P[1, 1]) if self._init else math.nan

    @property
    def slope_t(self) -> float:
        """The output that matters: slope over its posterior standard deviation."""
        if not self._init or self._n < 5:
            return math.nan
        v = self._P[1, 1]
        if not (v > 0.0):
            return math.nan
        return float(self._x[1] / math.sqrt(v))

    @property
    def count(self) -> int:
        return self._n


class VarianceRatio:
    """Lo-MacKinlay (1988) variance-ratio test, heteroskedasticity-robust form.

    Under the martingale null :math:`\\mathrm{VR}(q)=1`. The robust variance
    :math:`\\theta(q)` is essential rather than optional here: returns are
    strongly heteroskedastic, and the homoskedastic statistic rejects the null in
    every volatility cluster for that reason alone, producing a spurious "trend"
    reading precisely when volatility rises.

    Positive ``z`` means positively autocorrelated increments -- persistence, so
    condition (C) of the theory doc applies. Negative ``z`` means mean reversion,
    so condition (B) applies. This sign is the single most important input to the
    regime posterior.
    """

    __slots__ = ("_buf", "_lags", "_window", "_z", "_vr", "_min_obs")

    def __init__(self, window: int = 120, lags: tuple[int, ...] = (2, 4, 8, 16)) -> None:
        self._window = int(window)
        self._lags = tuple(int(q) for q in lags)
        if min(self._lags) < 2:
            raise ValueError("variance-ratio lags must be >= 2")
        # Need enough non-overlapping blocks at the longest horizon for the
        # asymptotics to be even roughly applicable.
        self._min_obs = max(3 * max(self._lags), 40)
        self._buf = RingBuffer(self._window)
        self._z: dict[int, float] = {q: math.nan for q in self._lags}
        self._vr: dict[int, float] = {q: math.nan for q in self._lags}

    def reset(self) -> None:
        self._buf.reset()
        self._z = {q: math.nan for q in self._lags}
        self._vr = {q: math.nan for q in self._lags}

    def update(self, ret: float) -> None:
        if ret != ret:
            return
        self._buf.push(ret)
        if len(self._buf) < self._min_obs:
            return
        r = self._buf.values()
        self._compute(r)

    def _compute(self, r: np.ndarray) -> None:
        n = r.size
        mu = float(r.mean())
        d = r - mu
        ss = float((d * d).sum())
        if ss <= 0.0:
            return
        # Unbiased one-period variance.
        var_1 = ss / (n - 1)

        # delta_j for the heteroskedasticity-robust variance of the ratio.
        deltas: dict[int, float] = {}
        max_j = max(self._lags) - 1
        d2 = d * d
        denom = ss * ss
        for j in range(1, max_j + 1):
            if j >= n:
                deltas[j] = 0.0
                continue
            deltas[j] = float((d2[j:] * d2[:-j]).sum()) / denom

        for q in self._lags:
            if n <= q + 2:
                continue
            # Overlapping q-period sums via cumulative sums.
            cs = np.concatenate(([0.0], np.cumsum(r)))
            qs = cs[q:] - cs[:-q]  # length n - q + 1
            m = q * (n - q + 1) * (1.0 - q / n)
            if m <= 0.0:
                continue
            var_q = float(((qs - q * mu) ** 2).sum()) / m
            vr = safe_div(var_q, var_1, math.nan)
            if vr != vr:
                continue
            theta = 0.0
            for j in range(1, q):
                w = 2.0 * (q - j) / q
                theta += w * w * deltas.get(j, 0.0)
            self._vr[q] = vr
            self._z[q] = safe_div(vr - 1.0, math.sqrt(theta), math.nan) if theta > 0.0 else math.nan

    def z(self, q: int) -> float:
        return self._z.get(q, math.nan)

    def vr(self, q: int) -> float:
        return self._vr.get(q, math.nan)

    @property
    def z_mean(self) -> float:
        """Average of the per-horizon statistics.

        A plain average rather than a weighted one: the horizons are strongly
        dependent, so a variance-optimal weighting would need the full
        covariance of the statistics across ``q``, which is not reliably
        estimable in a 120-bar window. Averaging is the conservative choice and
        errs toward a smaller-magnitude statistic.
        """
        vals = [v for v in self._z.values() if v == v]
        return float(np.mean(vals)) if vals else math.nan

    @property
    def hurst_implied(self) -> float:
        """``0.5 + ln VR(q) / (2 ln q)`` averaged over horizons.

        Reported for display only and used in no decision. The R/S and
        variance-ratio routes to Hurst are both badly biased at these sample
        sizes; the honest object is the test statistic, not the exponent.
        """
        vals = []
        for q, vr in self._vr.items():
            if vr == vr and vr > 0.0:
                vals.append(0.5 + math.log(vr) / (2.0 * math.log(q)))
        return float(np.mean(vals)) if vals else math.nan

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self._min_obs


class PathEfficiency:
    """Net displacement over path length, in ``[0, 1]``.

    1.0 is a straight line, 0.0 is a random walk that ends where it started.
    Dimensionless and directly comparable across instruments, which is why it is
    one of the regime measurement features.
    """

    __slots__ = ("_net", "_abs")

    def __init__(self, window: int = 22) -> None:
        self._net = RollingSum(window)
        self._abs = RollingSum(window)

    def reset(self) -> None:
        self._net.reset()
        self._abs.reset()

    def update(self, ret: float) -> None:
        if ret != ret:
            return
        self._net.update(ret)
        self._abs.update(abs(ret))

    @property
    def value(self) -> float:
        if len(self._abs) < 5:
            return math.nan
        return abs(safe_div(self._net.total, self._abs.total, math.nan))

    @property
    def signed(self) -> float:
        """Efficiency carrying the direction of the net move."""
        if len(self._abs) < 5:
            return math.nan
        return safe_div(self._net.total, self._abs.total, math.nan)


class MultiHorizonAgreement:
    """Mean sign of the return over several lookbacks, in ``[-1, +1]``.

    Deliberately crude and deliberately parameter-free: no thresholds, no
    weights, just whether price is above where it was. Its value is as a
    robustness check on the Kalman slope -- when a filtered slope and simple sign
    agreement disagree, the slope is being driven by the filter's own dynamics
    rather than by the data.
    """

    __slots__ = ("_buf", "_horizons")

    def __init__(self, horizons: tuple[int, ...] = (5, 10, 20, 40, 80)) -> None:
        self._horizons = tuple(sorted(int(h) for h in horizons))
        self._buf = RingBuffer(max(self._horizons) + 2)

    def reset(self) -> None:
        self._buf.reset()

    def update(self, log_price: float) -> None:
        if log_price != log_price:
            return
        self._buf.push(log_price)

    @property
    def value(self) -> float:
        n = len(self._buf)
        if n < self._horizons[0] + 1:
            return math.nan
        now = self._buf.last(1)
        signs = []
        for h in self._horizons:
            if n >= h + 1:
                prev = self._buf.last(h + 1)
                if prev == prev:
                    diff = now - prev
                    signs.append(0.0 if diff == 0.0 else math.copysign(1.0, diff))
        return float(np.mean(signs)) if signs else math.nan
