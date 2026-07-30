"""Anchored and rolling walk-forward evaluation.

Walk-forward answers a question cross-validation cannot: *would this system,
refitted only on information available at the time, have worked as it went
forward?* It is weaker than combinatorial purged CV as a significance test --
there is only one ordering of history, so it yields one path -- but it is the
only scheme whose folds are in the same causal order the live system will meet,
and it is the only one in which **parameter drift** is observable.

Two variants, and when each is the honest one:

*   :func:`anchored_walkforward` -- training window grows from a fixed origin.
    The right choice when the parameters are meant to be structural constants
    (which, per :class:`~tia.config.Config`, most of TI-A's are): more data
    should make them *more* stable, and if it does not, they are noise.
*   :func:`rolling_walkforward` -- fixed-length training window slides forward.
    The right choice when the quantity being estimated is genuinely
    non-stationary, and the honest test of whether the recent past is a better
    teacher than the distant past.

Both insert an embargo between the end of training and the start of testing, for
the same reason as :mod:`tia.validation.cpcv`: the last training label's outcome
is not yet known at the moment the first test decision would be taken, and
overlapping labels correlate across the boundary.

Parameter drift
---------------
:class:`ParameterDrift` is the part of this module that most often kills a
strategy, and it is the part usually left out of walk-forward reports. If a
parameter refitted per fold swings by a factor of three, or changes sign, then
each fold's fit is describing that fold's noise. The fold-by-fold performance can
still look acceptable -- each fold was fitted to itself -- while the *deployed*
parameter, whatever single value gets frozen, matches none of them. The
diagnostic is deliberately blunt: coefficient of variation, max/min ratio,
mean absolute fold-to-fold relative change, and sign flips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Sequence

import numpy as np

from .metrics import FoldStability, stability_across_folds

__all__ = [
    "anchored_walkforward",
    "rolling_walkforward",
    "ParameterDrift",
    "parameter_drift",
    "WalkForwardResult",
]


# ---------------------------------------------------------------------------
# Split generators
# ---------------------------------------------------------------------------


def anchored_walkforward(
    n: int, n_splits: int, min_train: int, embargo: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward over ``n`` ordered observations.

    The remaining ``n - min_train`` observations are cut into ``n_splits``
    contiguous test blocks of near-equal size. Fold ``i`` trains on
    ``[0, test_start - embargo)`` and tests on the ``i``-th block.

    Raises when ``min_train <= embargo`` (there would be no training data) or
    when there are fewer than ``n_splits`` observations to test on.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if not (0 < min_train < n):
        raise ValueError("min_train must satisfy 0 < min_train < n")
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    if min_train <= embargo:
        raise ValueError("min_train must exceed embargo, else the first fold has no training data")
    if n - min_train < n_splits:
        raise ValueError(
            f"only {n - min_train} test observations available for {n_splits} splits"
        )
    bounds = np.linspace(min_train, n, n_splits + 1)
    edges = np.unique(np.round(bounds).astype(int))
    for i in range(edges.size - 1):
        test_start, test_end = int(edges[i]), int(edges[i + 1])
        train_end = test_start - embargo
        if train_end <= 0 or test_end <= test_start:  # pragma: no cover - guarded above
            continue
        yield np.arange(0, train_end, dtype=np.int64), np.arange(
            test_start, test_end, dtype=np.int64
        )


def rolling_walkforward(
    n: int, train_bars: int, test_bars: int, step: int | None = None, embargo: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Fixed-length sliding-window walk-forward.

    ``step`` defaults to ``test_bars``, which makes the test blocks tile the
    sample without overlap -- the only setting under which the concatenated test
    blocks form a legitimate single out-of-sample path. A smaller ``step``
    produces overlapping test blocks: useful for measuring stability, invalid for
    computing an aggregate Sharpe, and the docstring says so because the mistake
    is easy and flattering.
    """
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive")
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    stride = int(test_bars if step is None else step)
    if stride <= 0:
        raise ValueError("step must be positive")
    start = 0
    while start + train_bars + embargo + test_bars <= n:
        train = np.arange(start, start + train_bars, dtype=np.int64)
        t0 = start + train_bars + embargo
        test = np.arange(t0, t0 + test_bars, dtype=np.int64)
        yield train, test
        start += stride


# ---------------------------------------------------------------------------
# Parameter drift
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterDrift:
    """Stability of one refitted parameter across folds.

    ``coefficient_of_variation``
        ``std / |mean|``. The headline number. Above ~0.5 the parameter is not
        being estimated, it is being fitted to fold noise.
    ``max_over_min``
        Ratio of the largest to the smallest value (``inf`` if any value is 0 or
        the sign changes). Immune to the mean being near zero.
    ``mean_abs_relative_change``
        Mean ``|x_{i+1} - x_i| / |x_i|``: how much the parameter moves from one
        refit to the next, which is what a live system would actually have to
        follow.
    ``sign_flips``
        Number of sign changes. For any parameter whose sign has a meaning
        (a slope, a bias, a skew), one flip is disqualifying.
    """

    name: str
    values: tuple[float, ...]
    mean: float
    std: float
    coefficient_of_variation: float
    max_over_min: float
    mean_abs_relative_change: float
    sign_flips: int
    threshold_cv: float = 0.5

    @property
    def stable(self) -> bool:
        cv = self.coefficient_of_variation
        return bool(np.isfinite(cv) and cv <= self.threshold_cv and self.sign_flips == 0)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.coefficient_of_variation):
            return "undefined"
        if self.sign_flips > 0:
            return "sign-unstable"
        return "stable" if self.stable else "drifting"

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "cv": self.coefficient_of_variation,
            "max_over_min": self.max_over_min,
            "mean_abs_rel_change": self.mean_abs_relative_change,
            "sign_flips": float(self.sign_flips),
            "stable": float(self.stable),
        }


def parameter_drift(
    values: Sequence[float], name: str = "parameter", threshold_cv: float = 0.5
) -> ParameterDrift:
    """Compute :class:`ParameterDrift` for one parameter's per-fold values."""
    a = np.asarray(values, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return ParameterDrift(name, (), math.nan, math.nan, math.nan, math.nan, math.nan, 0, threshold_cv)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    cv = float(std / abs(mean)) if mean != 0.0 else (0.0 if std == 0.0 else math.inf)
    signs = np.sign(a)
    nz = signs[signs != 0.0]
    flips = int(np.sum(nz[1:] != nz[:-1])) if nz.size > 1 else 0
    lo, hi = float(np.min(a)), float(np.max(a))
    if lo > 0.0:
        mom = hi / lo
    elif hi < 0.0:
        mom = lo / hi
    else:
        mom = math.inf
    if a.size > 1:
        denom = np.abs(a[:-1])
        rel = np.abs(np.diff(a)) / np.where(denom > 0.0, denom, np.nan)
        marc = float(np.nanmean(rel)) if np.any(np.isfinite(rel)) else math.inf
    else:
        marc = 0.0
    return ParameterDrift(
        name=name,
        values=tuple(float(v) for v in a),
        mean=mean,
        std=std,
        coefficient_of_variation=cv,
        max_over_min=mom,
        mean_abs_relative_change=marc,
        sign_flips=flips,
        threshold_cv=threshold_cv,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Per-fold metrics and refitted parameters, plus their stability.

    Mutable by design: folds are appended as they are evaluated. ``fold_metrics``
    entries are plain ``name -> value`` mappings so that this class needs to know
    nothing about which metrics the caller cares about.
    """

    scheme: str = "walk-forward"
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    fold_params: list[dict[str, float]] = field(default_factory=list)
    train_sizes: list[int] = field(default_factory=list)
    test_sizes: list[int] = field(default_factory=list)

    def add_fold(
        self,
        metrics: Mapping[str, float],
        params: Mapping[str, float] | None = None,
        train_size: int | None = None,
        test_size: int | None = None,
    ) -> None:
        self.fold_metrics.append({str(k): float(v) for k, v in metrics.items()})
        self.fold_params.append({str(k): float(v) for k, v in (params or {}).items()})
        self.train_sizes.append(int(train_size) if train_size is not None else -1)
        self.test_sizes.append(int(test_size) if test_size is not None else -1)

    @property
    def n_folds(self) -> int:
        return len(self.fold_metrics)

    def metric_names(self) -> list[str]:
        names: list[str] = []
        for m in self.fold_metrics:
            for k in m:
                if k not in names:
                    names.append(k)
        return names

    def metric(self, name: str) -> np.ndarray:
        """Per-fold values of one metric; ``nan`` for folds that lack it."""
        return np.asarray([m.get(name, math.nan) for m in self.fold_metrics], dtype=float)

    def stability(self, name: str) -> FoldStability:
        """Cross-fold stability of one metric.

        ``fraction_positive`` is the statement to quote. A mean that is positive
        because one fold carried it is not evidence of an edge; it is evidence
        that one fold happened.
        """
        return stability_across_folds(self.metric(name), name=name)

    def parameter_drift(self, threshold_cv: float = 0.5) -> dict[str, ParameterDrift]:
        """Drift diagnostics for every refitted parameter seen in any fold."""
        names: list[str] = []
        for p in self.fold_params:
            for k in p:
                if k not in names:
                    names.append(k)
        out: dict[str, ParameterDrift] = {}
        for k in names:
            vals = [p[k] for p in self.fold_params if k in p]
            out[k] = parameter_drift(vals, name=k, threshold_cv=threshold_cv)
        return out

    def max_parameter_cv(self, threshold_cv: float = 0.5) -> float:
        """Worst coefficient of variation across refitted parameters.

        The single number the acceptance test uses. ``nan`` when no parameters
        were recorded, which the acceptance test must treat as missing evidence
        rather than as a pass.
        """
        drifts = self.parameter_drift(threshold_cv)
        if not drifts:
            return math.nan
        cvs = [d.coefficient_of_variation for d in drifts.values()]
        finite = [c for c in cvs if np.isfinite(c)]
        if not finite:
            return math.inf
        return float(max(finite))

    def __str__(self) -> str:
        if self.n_folds == 0:
            return f"{self.scheme}: no folds"
        names = self.metric_names()
        widths = {k: max(len(k), 9) for k in names}
        head = "fold  " + "  ".join(f"{k:>{widths[k]}}" for k in names)
        lines = [f"{self.scheme}: {self.n_folds} folds", head, "-" * len(head)]
        for i, m in enumerate(self.fold_metrics):
            row = f"{i:>4}  " + "  ".join(
                f"{m.get(k, float('nan')):>{widths[k]}.4f}" for k in names
            )
            lines.append(row)
        lines.append("-" * len(head))
        mean_row = "mean  " + "  ".join(
            f"{float(np.nanmean(self.metric(k))):>{widths[k]}.4f}" for k in names
        )
        lines.append(mean_row)
        return "\n".join(lines)
