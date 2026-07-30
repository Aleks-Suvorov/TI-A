"""Information-theoretic and run-structure features.

These answer a question no volatility or trend measure answers: *is there any
exploitable ordering in the recent sequence at all?* A series can have high
volatility, a significant trend t-statistic and still be, in the ordinal sense,
maximally disordered -- and in that state neither of the theory's conditions (B)
nor (C) is present, which is what the `QUIET` regime is for.

Permutation entropy (Bandt & Pompe, 2002) is the right tool because it depends
only on the *ordinal pattern* of consecutive values. It is therefore invariant to
any monotone transformation of price, needs no binning choice, is robust to
outliers, and is cheap. Sample and approximate entropy were considered and
rejected: both require a tolerance parameter whose scale-dependence would have to
be estimated, adding a fitted degree of freedom for no measurable gain.
"""

from __future__ import annotations

import math
from itertools import permutations

import numpy as np

from .rolling import RingBuffer

__all__ = ["PermutationEntropy", "RunAsymmetry"]


class PermutationEntropy:
    """Normalised permutation entropy over a rolling window.

    For embedding dimension :math:`d`, each window of :math:`d` consecutive
    values maps to one of :math:`d!` ordinal patterns. The Shannon entropy of the
    empirical pattern distribution, divided by :math:`\\ln d!`, lies in
    :math:`[0, 1]`: 0 is perfectly ordered (monotone), 1 is maximally disordered.

    ``d = 3`` by default. Higher dimensions need exponentially more data to
    populate :math:`d!` patterns -- at ``d = 5`` there are 120 patterns and a
    60-bar window cannot estimate their distribution at all, so a larger
    embedding would produce a number that looks more sophisticated and means
    less.
    """

    __slots__ = ("_d", "_w", "_buf", "_patterns", "_index", "_counts", "_hist", "_norm")

    def __init__(self, order: int = 3, window: int = 60) -> None:
        if not 2 <= order <= 5:
            raise ValueError("permutation entropy order must be in [2, 5]")
        self._d = int(order)
        self._w = int(window)
        self._buf = RingBuffer(self._d)
        self._patterns = list(permutations(range(self._d)))
        self._index = {p: i for i, p in enumerate(self._patterns)}
        self._counts = np.zeros(len(self._patterns), dtype=np.int64)
        # Ring of recent pattern ids so counts can be decremented on eviction,
        # making the update O(d log d) rather than O(window).
        self._hist = RingBuffer(self._w)
        self._norm = math.log(math.factorial(self._d))

    def reset(self) -> None:
        self._buf.reset()
        self._hist.reset()
        self._counts[:] = 0

    def update(self, x: float) -> None:
        if x != x:
            return
        self._buf.push(x)
        if len(self._buf) < self._d:
            return
        v = self._buf.values()
        # argsort gives the ordinal pattern. Ties are broken by index, which is
        # the standard convention and is only reached on exactly equal prices.
        pat = tuple(int(i) for i in np.argsort(v, kind="stable"))
        pid = self._index[pat]
        ev = self._hist.push(float(pid))
        self._counts[pid] += 1
        if ev is not None:
            self._counts[int(ev)] -= 1

    @property
    def ready(self) -> bool:
        return int(self._counts.sum()) >= max(20, self._w // 3)

    @property
    def value(self) -> float:
        total = int(self._counts.sum())
        if total < max(20, self._w // 3):
            return math.nan
        p = self._counts[self._counts > 0] / total
        h = float(-(p * np.log(p)).sum())
        return h / self._norm

    @property
    def complexity(self) -> float:
        """One minus entropy: how much ordinal structure is present."""
        v = self.value
        return (1.0 - v) if v == v else math.nan


class RunAsymmetry:
    """Asymmetry between the lengths of up-runs and down-runs.

    Under a symmetric random walk, mean up-run and down-run lengths are equal.
    A persistent difference is a weak but genuinely model-free signature of
    directional persistence, and its usefulness here is as a cross-check on the
    variance-ratio statistic: the two measure related things by unrelated routes,
    so agreement raises the effective breadth of evidence and disagreement
    lowers it.

    Output is in ``[-1, +1]``, positive when up-runs are longer.
    """

    __slots__ = ("_up", "_down", "_cur", "_sign", "_max")

    def __init__(self, max_runs: int = 40) -> None:
        self._max = int(max_runs)
        self._up = RingBuffer(self._max)
        self._down = RingBuffer(self._max)
        self._cur = 0
        self._sign = 0

    def reset(self) -> None:
        self._up.reset()
        self._down.reset()
        self._cur = 0
        self._sign = 0

    def update(self, ret: float) -> None:
        if ret != ret or ret == 0.0:
            return
        s = 1 if ret > 0.0 else -1
        if s == self._sign:
            self._cur += 1
            return
        if self._sign > 0 and self._cur > 0:
            self._up.push(float(self._cur))
        elif self._sign < 0 and self._cur > 0:
            self._down.push(float(self._cur))
        self._sign = s
        self._cur = 1

    @property
    def value(self) -> float:
        if len(self._up) < 5 or len(self._down) < 5:
            return math.nan
        mu_u = float(self._up.values().mean())
        mu_d = float(self._down.values().mean())
        tot = mu_u + mu_d
        if tot <= 0.0:
            return math.nan
        return (mu_u - mu_d) / tot

    @property
    def current_run(self) -> int:
        """Signed length of the run in progress."""
        return self._sign * self._cur
