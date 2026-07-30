"""Streaming rolling statistics.

Every rolling quantity in TI-A is computed by one of these objects. That is a
deliberate architectural choice rather than a convenience: an object that only
ever receives one observation at a time cannot accidentally read the future, so
the causality guarantee of ``SPEC.md`` §2 holds by construction for anything
built out of these primitives.

Cost per update is O(1) for the moment-based estimators and O(log n) for the
order-statistic ones. Nothing here is O(history).
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from typing import Deque, Iterable, Sequence

import numpy as np

__all__ = [
    "RingBuffer",
    "RollingMoments",
    "RollingSum",
    "EWMA",
    "EWVar",
    "RollingQuantile",
    "CausalRank",
    "RollingExtreme",
    "RobustZ",
    "BucketedMedian",
    "RollingCorrelation",
]


class RingBuffer:
    """Fixed-capacity FIFO of floats with cheap array access.

    ``push`` returns the evicted value (or ``None``), which several estimators
    use to maintain their sums incrementally.
    """

    __slots__ = ("_buf", "_n", "_i", "_cap", "_filled")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._cap = int(capacity)
        self._buf = np.zeros(self._cap, dtype=np.float64)
        self._i = 0
        self._n = 0
        self._filled = False

    def __len__(self) -> int:
        return self._n

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def full(self) -> bool:
        return self._filled

    def push(self, x: float) -> float | None:
        evicted: float | None = None
        if self._filled:
            evicted = float(self._buf[self._i])
        self._buf[self._i] = x
        self._i = (self._i + 1) % self._cap
        if not self._filled:
            self._n += 1
            if self._n == self._cap:
                self._filled = True
        return evicted

    def values(self) -> np.ndarray:
        """Oldest-first view of the contents. Allocates; use sparingly."""
        if not self._filled:
            return self._buf[: self._n].copy()
        return np.concatenate((self._buf[self._i :], self._buf[: self._i]))

    def last(self, k: int = 1) -> float:
        """The value ``k`` pushes ago; ``k=1`` is the most recent."""
        if k < 1 or k > self._n:
            return math.nan
        return float(self._buf[(self._i - k) % self._cap])

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._i = 0
        self._n = 0
        self._filled = False


class RollingMoments:
    """Windowed mean, variance, skewness and kurtosis by incremental sums.

    Uses raw power sums, which is O(1) per update. Raw-sum accumulation is
    numerically fragile over very long runs, so sums are periodically rebuilt
    from the buffer; ``rebuild_every`` trades a little cost for stability. This
    matters in practice: without it, a million-bar backtest can accumulate
    enough error in the fourth moment to flip the sign of a kurtosis estimate.
    """

    __slots__ = ("_b", "_s1", "_s2", "_s3", "_s4", "_since", "_rebuild")

    def __init__(self, window: int, rebuild_every: int = 4096) -> None:
        self._b = RingBuffer(window)
        self._s1 = self._s2 = self._s3 = self._s4 = 0.0
        self._since = 0
        self._rebuild = int(rebuild_every)

    def __len__(self) -> int:
        return len(self._b)

    @property
    def full(self) -> bool:
        return self._b.full

    def update(self, x: float) -> None:
        if x != x:  # NaN never enters the accumulators
            return
        ev = self._b.push(x)
        self._s1 += x
        self._s2 += x * x
        self._s3 += x**3
        self._s4 += x**4
        if ev is not None:
            self._s1 -= ev
            self._s2 -= ev * ev
            self._s3 -= ev**3
            self._s4 -= ev**4
        self._since += 1
        if self._since >= self._rebuild:
            self._resum()

    def _resum(self) -> None:
        v = self._b.values()
        self._s1 = float(v.sum())
        self._s2 = float((v**2).sum())
        self._s3 = float((v**3).sum())
        self._s4 = float((v**4).sum())
        self._since = 0

    @property
    def mean(self) -> float:
        n = len(self._b)
        return self._s1 / n if n else math.nan

    @property
    def var(self) -> float:
        """Sample variance with Bessel's correction."""
        n = len(self._b)
        if n < 2:
            return math.nan
        m = self._s1 / n
        v = (self._s2 - n * m * m) / (n - 1)
        return max(v, 0.0)

    @property
    def std(self) -> float:
        v = self.var
        return math.sqrt(v) if v == v else math.nan

    @property
    def skew(self) -> float:
        n = len(self._b)
        if n < 3:
            return math.nan
        m = self._s1 / n
        m2 = self._s2 / n - m * m
        if m2 <= 1e-300:
            return 0.0
        m3 = self._s3 / n - 3.0 * m * self._s2 / n + 2.0 * m**3
        return m3 / m2**1.5

    @property
    def kurtosis(self) -> float:
        """Excess kurtosis; 0 for a Gaussian."""
        n = len(self._b)
        if n < 4:
            return math.nan
        m = self._s1 / n
        m2 = self._s2 / n - m * m
        if m2 <= 1e-300:
            return 0.0
        m4 = self._s4 / n - 4.0 * m * self._s3 / n + 6.0 * m * m * self._s2 / n - 3.0 * m**4
        return m4 / (m2 * m2) - 3.0

    def values(self) -> np.ndarray:
        return self._b.values()

    def reset(self) -> None:
        self._b.reset()
        self._s1 = self._s2 = self._s3 = self._s4 = 0.0
        self._since = 0


class RollingSum:
    """Windowed sum. Separate from :class:`RollingMoments` because the variance
    estimators only need the first power and this halves their cost."""

    __slots__ = ("_b", "_s")

    def __init__(self, window: int) -> None:
        self._b = RingBuffer(window)
        self._s = 0.0

    def update(self, x: float) -> None:
        if x != x:
            return
        ev = self._b.push(x)
        self._s += x
        if ev is not None:
            self._s -= ev

    def __len__(self) -> int:
        return len(self._b)

    @property
    def full(self) -> bool:
        return self._b.full

    @property
    def total(self) -> float:
        return self._s

    @property
    def mean(self) -> float:
        n = len(self._b)
        return self._s / n if n else math.nan

    def values(self) -> np.ndarray:
        return self._b.values()

    def reset(self) -> None:
        self._b.reset()
        self._s = 0.0


class EWMA:
    """Exponentially weighted mean, specified by half-life in observations.

    Bias correction by the accumulated weight makes early values usable rather
    than pinned to the first observation, which matters because warmup periods
    are long enough here that a biased start would persist into live signals.
    """

    __slots__ = ("_lam", "_num", "_den", "_n")

    def __init__(self, halflife: float) -> None:
        if halflife <= 0.0:
            raise ValueError("halflife must be positive")
        self._lam = math.exp(-math.log(2.0) / float(halflife))
        self._num = 0.0
        self._den = 0.0
        self._n = 0

    def update(self, x: float) -> float:
        if x != x:
            return self.value
        self._num = self._lam * self._num + x
        self._den = self._lam * self._den + 1.0
        self._n += 1
        return self.value

    @property
    def value(self) -> float:
        return self._num / self._den if self._den > 0.0 else math.nan

    @property
    def count(self) -> int:
        return self._n

    def reset(self) -> None:
        self._num = self._den = 0.0
        self._n = 0


class EWVar:
    """Exponentially weighted mean and variance, bias-corrected."""

    __slots__ = ("_m", "_v", "_lam", "_n")

    def __init__(self, halflife: float) -> None:
        self._m = EWMA(halflife)
        self._v = EWMA(halflife)
        self._lam = self._m._lam
        self._n = 0

    def update(self, x: float) -> None:
        if x != x:
            return
        prev = self._m.value
        self._m.update(x)
        ref = prev if prev == prev else x
        self._v.update((x - ref) ** 2)
        self._n += 1

    @property
    def mean(self) -> float:
        return self._m.value

    @property
    def var(self) -> float:
        v = self._v.value
        return v if v == v else math.nan

    @property
    def std(self) -> float:
        v = self.var
        return math.sqrt(v) if v == v and v >= 0.0 else math.nan

    @property
    def count(self) -> int:
        return self._n

    def reset(self) -> None:
        self._m.reset()
        self._v.reset()
        self._n = 0


class RollingQuantile:
    """Windowed quantiles from a sorted list kept alongside the window.

    O(log n) search plus O(n) list splice per update. For the window sizes used
    here (60-500) this beats a heap pair in practice and is far simpler to
    verify. ``quantile`` uses linear interpolation between order statistics.
    """

    __slots__ = ("_win", "_sorted", "_cap")

    def __init__(self, window: int) -> None:
        self._cap = int(window)
        self._win: Deque[float] = deque()
        self._sorted: list[float] = []

    def __len__(self) -> int:
        return len(self._win)

    @property
    def full(self) -> bool:
        return len(self._win) >= self._cap

    def update(self, x: float) -> None:
        if x != x:
            return
        self._win.append(x)
        bisect.insort(self._sorted, x)
        if len(self._win) > self._cap:
            old = self._win.popleft()
            i = bisect.bisect_left(self._sorted, old)
            # Guard against float identity issues: bisect_left lands on the
            # first equal value, which is the correct one to remove for a
            # multiset.
            if i < len(self._sorted) and self._sorted[i] == old:
                self._sorted.pop(i)

    def quantile(self, q: float) -> float:
        n = len(self._sorted)
        if n == 0:
            return math.nan
        if n == 1:
            return self._sorted[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return self._sorted[lo] * (1.0 - frac) + self._sorted[hi] * frac

    @property
    def median(self) -> float:
        return self.quantile(0.5)

    def mad(self) -> float:
        """Median absolute deviation, scaled to a Gaussian-consistent sigma."""
        med = self.median
        if med != med or not self._sorted:
            return math.nan
        dev = sorted(abs(v - med) for v in self._sorted)
        n = len(dev)
        m = dev[n // 2] if n % 2 else 0.5 * (dev[n // 2 - 1] + dev[n // 2])
        return 1.4826 * m

    def rank_of(self, x: float) -> float:
        """Fraction of the window strictly below ``x``, plus half the ties.

        The mid-rank convention keeps the transform unbiased when a feature has
        atoms -- volume-based features often do, and a naive ``<`` convention
        makes their ranks systematically low.
        """
        n = len(self._sorted)
        if n == 0:
            return math.nan
        lo = bisect.bisect_left(self._sorted, x)
        hi = bisect.bisect_right(self._sorted, x)
        return (lo + 0.5 * (hi - lo)) / n

    def reset(self) -> None:
        self._win.clear()
        self._sorted.clear()


class CausalRank:
    """Rolling empirical quantile of the *current* observation.

    The workhorse of the rank-transform discipline described in
    ``docs/01-THEORY.md`` §10. The current value is ranked against the trailing
    window **including itself**, so the transform is a function of
    :math:`\\mathcal{F}_t` alone.

    Returns NaN until ``min_obs`` observations exist. Callers must treat that
    NaN as "not yet valid" rather than substituting 0.5, because a fabricated
    neutral rank is an opinion and the fusion layer would act on it.
    """

    __slots__ = ("_q", "_min")

    def __init__(self, window: int, min_obs: int = 30) -> None:
        self._q = RollingQuantile(window)
        self._min = int(min_obs)

    def update(self, x: float) -> float:
        if x != x:
            return math.nan
        self._q.update(x)
        if len(self._q) < self._min:
            return math.nan
        return self._q.rank_of(x)

    @property
    def ready(self) -> bool:
        return len(self._q) >= self._min

    @property
    def median(self) -> float:
        return self._q.median

    def quantile(self, q: float) -> float:
        return self._q.quantile(q)

    def reset(self) -> None:
        self._q.reset()


class RollingExtreme:
    """Windowed min and max via monotonic deques. O(1) amortised."""

    __slots__ = ("_cap", "_i", "_mind", "_maxd", "_n")

    def __init__(self, window: int) -> None:
        self._cap = int(window)
        self._i = 0
        self._n = 0
        self._mind: Deque[tuple[int, float]] = deque()
        self._maxd: Deque[tuple[int, float]] = deque()

    def update(self, x: float) -> None:
        if x != x:
            return
        i = self._i
        while self._mind and self._mind[-1][1] >= x:
            self._mind.pop()
        self._mind.append((i, x))
        while self._maxd and self._maxd[-1][1] <= x:
            self._maxd.pop()
        self._maxd.append((i, x))
        cutoff = i - self._cap + 1
        while self._mind and self._mind[0][0] < cutoff:
            self._mind.popleft()
        while self._maxd and self._maxd[0][0] < cutoff:
            self._maxd.popleft()
        self._i += 1
        self._n = min(self._n + 1, self._cap)

    def __len__(self) -> int:
        return self._n

    @property
    def full(self) -> bool:
        return self._n >= self._cap

    @property
    def min(self) -> float:
        return self._mind[0][1] if self._mind else math.nan

    @property
    def max(self) -> float:
        return self._maxd[0][1] if self._maxd else math.nan

    def reset(self) -> None:
        self._i = 0
        self._n = 0
        self._mind.clear()
        self._maxd.clear()


class RobustZ:
    """Median/MAD standard score. Resistant to the outliers that make a
    mean/std z-score useless on return data."""

    __slots__ = ("_q",)

    def __init__(self, window: int) -> None:
        self._q = RollingQuantile(window)

    def update(self, x: float) -> float:
        if x != x:
            return math.nan
        self._q.update(x)
        if len(self._q) < 10:
            return math.nan
        med = self._q.median
        mad = self._q.mad()
        if mad <= 1e-300 or mad != mad:
            return 0.0
        return (x - med) / mad

    def reset(self) -> None:
        self._q.reset()


class BucketedMedian:
    """One rolling median per time-of-day bucket.

    Volume and volatility follow a pronounced and extremely well-replicated
    intraday U-shape. Comparing a 09:35 volume to a 12:15 volume without
    conditioning on the hour mostly measures what time it is, so every volume
    feature in the system is normalised by its own bucket's median.
    """

    __slots__ = ("_buckets", "_n")

    def __init__(self, n_buckets: int, window: int) -> None:
        self._n = int(n_buckets)
        self._buckets = [RollingQuantile(window) for _ in range(self._n)]

    def update(self, bucket: int, x: float) -> None:
        if x != x:
            return
        self._buckets[bucket % self._n].update(x)

    def median(self, bucket: int) -> float:
        return self._buckets[bucket % self._n].median

    def count(self, bucket: int) -> int:
        return len(self._buckets[bucket % self._n])

    def normalise(self, bucket: int, x: float, fallback: float = math.nan) -> float:
        """``x`` divided by its bucket's median, or ``fallback`` if too thin."""
        b = self._buckets[bucket % self._n]
        if len(b) < 5:
            return fallback
        med = b.median
        if med is None or med != med or med <= 0.0:
            return fallback
        return x / med

    def reset(self) -> None:
        for b in self._buckets:
            b.reset()


class RollingCorrelation:
    """Rolling correlation matrix of an m-vector, by exponential weighting.

    Used for the engine correlation matrix of ``docs/01-THEORY.md`` §6.4 and for
    the feature-redundancy tempering of §5.3. Exponential rather than windowed
    weighting because the quantity of interest -- how much engines are currently
    duplicating each other -- changes fastest exactly when it matters most, in
    a crisis.

    The known deficiency is stated in the theory doc §14: the estimate lags. A
    trailing estimator cannot report a correlation spike until it has seen it,
    so effective breadth is overstated during the first days of a regime break.
    """

    __slots__ = ("_m", "_lam", "_mean", "_cov", "_w", "_n")

    def __init__(self, m: int, halflife: float) -> None:
        self._m = int(m)
        self._lam = math.exp(-math.log(2.0) / float(halflife))
        self._mean = np.zeros(self._m)
        self._cov = np.eye(self._m)
        self._w = 0.0
        self._n = 0

    def update(self, x: Sequence[float] | np.ndarray) -> None:
        v = np.asarray(x, dtype=np.float64)
        if v.shape != (self._m,) or not np.all(np.isfinite(v)):
            return
        lam = self._lam
        self._w = lam * self._w + 1.0
        alpha = 1.0 / self._w
        delta = v - self._mean
        self._mean += alpha * delta
        # Exponentially weighted second moment about the *updated* mean; the
        # (1-alpha) factor keeps the estimate consistent as the weight grows.
        self._cov = (1.0 - alpha) * (self._cov + alpha * np.outer(delta, delta))
        self._n += 1

    @property
    def count(self) -> int:
        return self._n

    def covariance(self) -> np.ndarray:
        return self._cov.copy()

    def correlation(self, ridge: float = 0.0) -> np.ndarray:
        """Correlation matrix, optionally shrunk toward the identity.

        ``ridge`` is the ``fusion_shrink`` of the config. Without it,
        near-collinear engines make the matrix ill-conditioned and its inverse
        produces enormous opposing weights -- the pooled estimate becomes a
        difference of nearly equal large numbers.
        """
        d = np.sqrt(np.clip(np.diag(self._cov), 1e-300, None))
        c = self._cov / np.outer(d, d)
        np.fill_diagonal(c, 1.0)
        c = np.clip(c, -0.999, 0.999)
        np.fill_diagonal(c, 1.0)
        if ridge > 0.0:
            c = (1.0 - ridge) * c + ridge * np.eye(self._m)
        return c

    def reset(self) -> None:
        self._mean[:] = 0.0
        self._cov = np.eye(self._m)
        self._w = 0.0
        self._n = 0


def _as_array(x: Iterable[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=np.float64)
