"""Probability calibration.

A stated confidence of 97% has to mean that, over many such statements, about 97
of every 100 came true. Nothing in the pooling of :mod:`tia.fusion.calop`
guarantees that -- it guarantees the *ordering* and the variance treatment. The
map from pooled evidence to a frequency is an empirical object and has to be
estimated, monitored, and refitted.

This module owns the online map. It is deliberately independent of
:mod:`tia.validation.metrics`, which owns the offline diagnostics: the live path
must not import the research path.

Two stages:

1.  A parametric link, ``sigmoid(T x + b)``, fitted once on the development
    universe and frozen. Cheap, monotone, and defined everywhere.
2.  An isotonic correction fitted online by pool-adjacent-violators once enough
    realised outcomes exist. Isotonic makes no functional-form assumption beyond
    monotonicity, which is exactly the assumption we are willing to make.

**Causality.** The isotonic map is fitted only on outcomes that have already
resolved, and is applied only to bars after the fit. ``IsotonicCalibrator.fit``
takes a snapshot; the caller is responsible for never passing it a future
outcome, and :class:`OnlineCalibrator` enforces that by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..types import clip

__all__ = [
    "pav",
    "IsotonicCalibrator",
    "fit_platt",
    "OnlineCalibrator",
    "brier_score",
    "BrierDecomposition",
    "brier_decomposition",
    "reliability_table",
    "expected_calibration_error",
]


# ---------------------------------------------------------------------------
# Isotonic regression
# ---------------------------------------------------------------------------


def pav(
    y: np.ndarray, w: np.ndarray | None = None, return_blocks: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators: the isotonic (non-decreasing) least-squares fit.

    O(n) with the standard block-merging implementation. Returns fitted values
    aligned with the input order, which must already be sorted by the predictor.
    With ``return_blocks``, also returns the number of observations backing each
    fitted value -- which :class:`IsotonicCalibrator` needs in order to smooth
    blocks that rest on very few points.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n == 0:
        return (y.copy(), np.zeros(0)) if return_blocks else y.copy()
    w = np.ones(n) if w is None else np.asarray(w, dtype=np.float64).copy()

    vals = y.copy()
    wts = w.copy()
    sizes = np.ones(n, dtype=np.int64)
    k = 0
    for i in range(1, n):
        k += 1
        vals[k], wts[k], sizes[k] = y[i], w[i], 1
        while k > 0 and vals[k - 1] > vals[k]:
            tw = wts[k - 1] + wts[k]
            vals[k - 1] = (wts[k - 1] * vals[k - 1] + wts[k] * vals[k]) / tw
            wts[k - 1] = tw
            sizes[k - 1] += sizes[k]
            k -= 1
    out = np.empty(n)
    counts = np.empty(n)
    pos = 0
    for j in range(k + 1):
        out[pos : pos + sizes[j]] = vals[j]
        counts[pos : pos + sizes[j]] = sizes[j]
        pos += sizes[j]
    return (out, counts) if return_blocks else out


@dataclass
class IsotonicCalibrator:
    """Piecewise-constant monotone map, stored as interpolation knots."""

    x: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    y: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    n_fit: int = 0

    def fit(self, p: np.ndarray, outcome: np.ndarray, weights: np.ndarray | None = None) -> "IsotonicCalibrator":
        p = np.asarray(p, dtype=np.float64)
        o = np.asarray(outcome, dtype=np.float64)
        ok = np.isfinite(p) & np.isfinite(o)
        p, o = p[ok], o[ok]
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)[ok]
        else:
            w = np.ones_like(p)
        if p.size < 20:
            return self
        order = np.argsort(p, kind="stable")
        ps, os_, ws = p[order], o[order], w[order]
        fitted = pav(os_, ws)  # type: ignore[assignment]

        # Bound the map's *endpoints* away from certainty. On binary outcomes
        # PAVA leaves the leading run of failures at exactly 0 and the trailing
        # run of successes at exactly 1 -- it only pools where the ordering is
        # violated, and a run of identical values never violates it. So the map
        # ends up asserting p = 0 and p = 1, which is a claim of certainty, and
        # it propagates straight into position sizing.
        #
        # The evidence behind each endpoint is the length of that run, and the
        # honest bound is its Laplace posterior mean: n successes out of n
        # supports (n+1)/(n+2), which is 0.857 on five observations and 0.99 on
        # a hundred. Clipping preserves monotonicity for free and leaves the
        # interior untouched -- isotonic's ordering constraint already handles
        # ordinary small-sample noise there, and flattening the interior would
        # destroy resolution, which is the property that actually earns anything.
        if fitted[0] <= 1e-12:
            n_lo = int(np.argmax(fitted > 1e-12)) or fitted.size
            fitted = np.maximum(fitted, 1.0 / (n_lo + 2.0))
        if fitted[-1] >= 1.0 - 1e-12:
            rev = fitted[::-1]
            n_hi = int(np.argmax(rev < 1.0 - 1e-12)) or fitted.size
            fitted = np.minimum(fitted, (n_hi + 1.0) / (n_hi + 2.0))
        # Collapse to the distinct knots of the step function; this is what
        # makes the map small enough to export to Pine Script.
        keep = np.concatenate(([True], np.diff(fitted) > 1e-12))
        self.x = np.concatenate(([0.0], ps[keep], [1.0]))
        self.y = np.concatenate(([fitted[keep][0]], fitted[keep], [fitted[keep][-1]]))
        self.x, idx = np.unique(self.x, return_index=True)
        self.y = self.y[idx]
        self.n_fit = int(p.size)
        return self

    def __call__(self, p: float) -> float:
        if self.n_fit == 0:
            return p
        return float(clip(np.interp(p, self.x, self.y), 1e-6, 1.0 - 1e-6))

    def knots(self, max_knots: int = 24) -> list[tuple[float, float]]:
        """Down-sampled knots, for export to environments without arrays."""
        if self.x.size <= max_knots:
            return [(float(a), float(b)) for a, b in zip(self.x, self.y)]
        idx = np.linspace(0, self.x.size - 1, max_knots).round().astype(int)
        return [(float(self.x[i]), float(self.y[i])) for i in idx]


# ---------------------------------------------------------------------------
# Parametric link
# ---------------------------------------------------------------------------


def fit_platt(scores: np.ndarray, outcome: np.ndarray, max_iter: int = 100) -> tuple[float, float]:
    """Fit ``sigmoid(T x + b)`` by Newton-Raphson on the log-likelihood.

    Implemented directly rather than via sklearn so the live path keeps its
    numpy-only dependency. Newton on a two-parameter logistic is a handful of
    lines and converges in well under ten iterations.
    """
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 30:
        return 1.0, 0.0

    T, b = 1.0, 0.0
    for _ in range(max_iter):
        z = T * x + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        g = np.array([np.sum((p - y) * x), np.sum(p - y)])
        w = p * (1.0 - p) + 1e-12
        H = np.array(
            [[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]]
        ) + 1e-9 * np.eye(2)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:  # pragma: no cover
            break
        T -= float(step[0])
        b -= float(step[1])
        if float(np.max(np.abs(step))) < 1e-9:
            break
    # A negative temperature would invert the map, which means the evidence is
    # anti-predictive. That is a finding to report, not to encode.
    return (max(T, 1e-3), b)


# ---------------------------------------------------------------------------
# Online calibrator
# ---------------------------------------------------------------------------


class OnlineCalibrator:
    """Maintains the live probability map from resolved outcomes.

    Outcomes arrive late -- a trade opened now resolves in up to
    ``max_holding_bars`` bars -- so the caller records ``(p, outcome)`` pairs only
    at resolution. The map is refitted every ``refit_every`` resolutions, never
    on every bar, both for cost and because a map that moves continuously makes
    live behaviour irreproducible.
    """

    def __init__(self, min_samples: int = 200, capacity: int = 5000, refit_every: int = 50) -> None:
        self.min_samples = int(min_samples)
        self.capacity = int(capacity)
        self.refit_every = int(refit_every)
        self._p: list[float] = []
        self._y: list[float] = []
        self._iso = IsotonicCalibrator()
        self._since_fit = 0
        self.active = False

    def reset(self) -> None:
        self._p.clear()
        self._y.clear()
        self._iso = IsotonicCalibrator()
        self._since_fit = 0
        self.active = False

    def record(self, p: float, success: bool | float) -> None:
        if p != p:
            return
        self._p.append(float(p))
        self._y.append(float(success))
        if len(self._p) > self.capacity:
            # Drop the oldest. The calibration map should describe the market we
            # are in, and an eight-year-old reliability curve does not.
            del self._p[: len(self._p) - self.capacity]
            del self._y[: len(self._y) - self.capacity]
        self._since_fit += 1
        if len(self._p) >= self.min_samples and self._since_fit >= self.refit_every:
            self._iso.fit(np.array(self._p), np.array(self._y))
            self._since_fit = 0
            self.active = self._iso.n_fit > 0

    def __call__(self, p: float) -> float:
        return self._iso(p) if self.active else p

    @property
    def n_samples(self) -> int:
        return len(self._p)

    def health(self) -> dict[str, float]:
        if len(self._p) < 30:
            return {"brier": math.nan, "ece": math.nan, "n": float(len(self._p))}
        p = np.array(self._p)
        y = np.array(self._y)
        d = brier_decomposition(p, y)
        return {
            "brier": brier_score(p, y),
            "reliability": d.reliability,
            "resolution": d.resolution,
            "uncertainty": d.uncertainty,
            "skill": d.skill,
            "ece": expected_calibration_error(p, y),
            "n": float(p.size),
        }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((p - y) ** 2)) if p.size else math.nan


@dataclass(frozen=True, slots=True)
class BrierDecomposition:
    """Murphy's decomposition: ``BS = reliability - resolution + uncertainty``.

    ``reliability`` is calibration error, lower is better. ``resolution`` is how
    far forecasts move from the base rate in a way that is borne out, higher is
    better. ``uncertainty`` is the base rate's own variance and is a property of
    the problem, not of the forecaster.

    The distinction matters operationally: a forecaster can improve its Brier
    score by becoming better calibrated *or* by becoming more discriminating,
    and only the second is worth anything to a trading system. A perfectly
    calibrated forecaster that always predicts the base rate has zero
    reliability error and zero resolution, and is useless.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float

    @property
    def skill(self) -> float:
        """Brier skill score against the base-rate forecast."""
        return 1.0 - self.brier / self.uncertainty if self.uncertainty > 0.0 else math.nan


def brier_decomposition(p: np.ndarray, y: np.ndarray, bins: int = 15) -> BrierDecomposition:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = p.size
    if n == 0:
        return BrierDecomposition(math.nan, math.nan, math.nan, math.nan)
    base = float(y.mean())
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rel = res = 0.0
    for k in range(bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        pk = float(p[m].mean())
        ok = float(y[m].mean())
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - base) ** 2
    rel /= n
    res /= n
    unc = base * (1.0 - base)
    return BrierDecomposition(brier_score(p, y), rel, res, unc)


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Rows of a reliability diagram: predicted vs observed frequency per bin."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for k in range(bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        obs = float(y[m].mean())
        # Wilson interval: correct at the small counts these bins actually have,
        # where the normal approximation is badly wrong near 0 and 1.
        z = 1.959963984540054
        den = 1.0 + z * z / nk
        centre = (obs + z * z / (2 * nk)) / den
        half = z * math.sqrt(obs * (1 - obs) / nk + z * z / (4 * nk * nk)) / den
        rows.append(
            {
                "bin_low": float(edges[k]),
                "bin_high": float(edges[k + 1]),
                "n": float(nk),
                "predicted": float(p[m].mean()),
                "observed": obs,
                "ci_low": max(0.0, centre - half),
                "ci_high": min(1.0, centre + half),
            }
        )
    return rows


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Weighted mean absolute gap between predicted and observed frequency."""
    rows = reliability_table(p, y, bins)
    n = sum(r["n"] for r in rows)
    if n <= 0:
        return math.nan
    return float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rows) / n)
