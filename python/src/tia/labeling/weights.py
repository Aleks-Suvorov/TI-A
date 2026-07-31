"""Sample weights for overlapping labels.

Why this file exists
--------------------
Triple-barrier labels **overlap**. A label opened at bar 100 with a 30-bar
vertical barrier is still open when the label at bar 105 is created, and both
are functions of the same returns over bars 105-130. They are not independent
observations, and nothing about calling them "trades" changes that.

Two consequences, both quantitative and both fatal if ignored:

1.  **The effective sample size is far smaller than the label count.** If, on
    average, ``c`` labels are open on every bar, then roughly ``m / c``
    independent observations are hiding inside ``m`` labels. With
    ``max_holding_bars = 30`` and a signal firing every fifth bar, ``c`` is
    about 6: three hundred labels carry the information of fifty.

2.  **Every t-statistic computed as if labels were independent is inflated by
    about ``sqrt(m / m_eff)``.** In the example above that is a factor of ~2.4 --
    enough to turn a t-statistic of 1.0 into an apparently decisive 2.4, and it
    is the single most common way a backtest lies about significance. The same
    inflation infects Sharpe standard errors, bootstrap intervals computed by
    resampling labels independently, and any cross-validation that does not
    purge (see :mod:`tia.validation.cpcv`).

The remedies implemented here are López de Prado's (*Advances in Financial
Machine Learning*, 2018, ch. 4):

*   :func:`concurrency` -- how many labels span each bar.
*   :func:`average_uniqueness` -- each label's mean ``1 / concurrency`` over its
    own span: 1.0 for a label that never overlaps anything, ``1/c`` for one that
    shares every bar with ``c-1`` others.
*   :func:`time_decay_weights` and :func:`linear_decay_weights` -- forgetting, so
    that a decade-old market microstructure does not outvote the current one.
*   :func:`combined_weights` -- the product, normalised to mean 1 so that the
    weighted mean of anything stays on the same scale as its unweighted mean.
*   :func:`effective_sample_size` and :func:`tstat_inflation_factor` -- the
    numbers to quote next to any t-statistic.

Sample weights are *not* a fix for overlap. They correct the first moment; they
do not restore independence. Purged cross-validation and block bootstrapping
still have to do their jobs.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "concurrency",
    "average_uniqueness",
    "time_decay_weights",
    "linear_decay_weights",
    "combined_weights",
    "effective_sample_size",
    "overlap_effective_sample_size",
    "combined_effective_sample_size",
    "tstat_inflation_factor",
    "weighted_mean",
    "weighted_tstat",
]


def _check_spans(
    n_bars: int, entry_indices: Sequence[int] | np.ndarray, exit_indices: Sequence[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(entry_indices, dtype=np.int64).ravel()
    x = np.asarray(exit_indices, dtype=np.int64).ravel()
    if e.size != x.size:
        raise ValueError("entry_indices and exit_indices must have the same length")
    if n_bars <= 0:
        raise ValueError("n_bars must be positive")
    if e.size == 0:
        return e, x
    if e.min() < 0 or x.max() >= n_bars:
        raise ValueError(f"label spans must lie inside [0, {n_bars - 1}]")
    if np.any(x < e):
        raise ValueError("exit_index must be >= entry_index for every label")
    return e, x


def concurrency(
    n_bars: int,
    entry_indices: Sequence[int] | np.ndarray,
    exit_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Number of labels spanning each bar, inclusive of both endpoints.

    Computed as a difference array plus a cumulative sum, so it is O(n + m)
    rather than O(n * m). Bars spanned by no label get 0.

    The mean of this array over the bars that any label touches is the overlap
    multiplier that inflates naive t-statistics.
    """
    e, x = _check_spans(n_bars, entry_indices, exit_indices)
    delta = np.zeros(n_bars + 1, dtype=np.int64)
    if e.size:
        np.add.at(delta, e, 1)
        np.add.at(delta, x + 1, -1)
    return np.cumsum(delta)[:n_bars]


def average_uniqueness(
    n_bars: int,
    entry_indices: Sequence[int] | np.ndarray,
    exit_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Per-label mean of ``1 / concurrency`` over the label's span.

    López de Prado ch. 4, snippets 4.1-4.2, but computed with prefix sums
    instead of the reference implementation's nested loop.

    Interpretation: a value of 0.25 says that on average this label shared each
    of its bars with three others, so it should count as a quarter of an
    observation. ``sum(average_uniqueness)`` is therefore an estimate of the
    number of *non-overlapping* labels the sample contains -- report it next to
    the label count in any results table.
    """
    e, x = _check_spans(n_bars, entry_indices, exit_indices)
    if e.size == 0:
        return np.zeros(0, dtype=float)
    c = concurrency(n_bars, e, x)
    inv = np.where(c > 0, 1.0 / np.maximum(c, 1), 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(inv)))
    spans = (x - e + 1).astype(float)
    return (prefix[x + 1] - prefix[e]) / spans


def time_decay_weights(
    anchor_indices: Sequence[int] | np.ndarray,
    halflife_bars: float,
    reference_index: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Exponential forgetting with an explicit half-life, in bars.

    ``w_i = 2 ** (-(reference_index - anchor_i) / halflife_bars)``

    A half-life is stated rather than a decay rate because it is the only form
    of the parameter that can be sanity-checked against a market fact: "how long
    ago did the microstructure that produced this observation stop resembling
    today's?". ``Config.edge_halflife_bars`` answers the same question for the
    online edge book, and deliberately answers it with a *long* number -- the
    cost of forgetting a real regime you have not seen recently exceeds the cost
    of stale data.

    ``halflife_bars <= 0`` or ``inf`` disables decay (all weights 1.0).
    """
    a = np.asarray(anchor_indices, dtype=float).ravel()
    if a.size == 0:
        return np.zeros(0, dtype=float)
    if not np.isfinite(halflife_bars) or halflife_bars <= 0.0:
        w = np.ones_like(a)
    else:
        ref = float(a.max()) if reference_index is None else float(reference_index)
        age = np.maximum(ref - a, 0.0)
        w = np.power(2.0, -age / float(halflife_bars))
    if normalize:
        mean = float(w.mean())
        if mean > 0.0:
            w = w / mean
    return w


def linear_decay_weights(avg_uniqueness: Sequence[float] | np.ndarray, last_weight: float = 0.0) -> np.ndarray:
    """López de Prado's piecewise-linear time decay (ch. 4, snippet 4.11).

    Decay is linear in *cumulative average uniqueness* rather than in bars, which
    is the right clock: a stretch of history in which many overlapping labels
    were open contains less new information than the bar count suggests, and
    should therefore age faster.

    ``last_weight`` is the weight given to the oldest observation:

    *   ``1.0``  -- no decay.
    *   ``0.0``  -- the oldest observation gets zero weight, the newest 1.0.
    *   ``c < 0`` -- observations older than a fraction ``|c|`` of the cumulative
        uniqueness are dropped entirely (weight clipped at 0).

    Labels must be supplied in chronological order.
    """
    u = np.asarray(avg_uniqueness, dtype=float).ravel()
    if u.size == 0:
        return np.zeros(0, dtype=float)
    cum = np.cumsum(u)
    total = float(cum[-1])
    if total <= 0.0:
        return np.ones_like(u)
    c = float(last_weight)
    if c >= 0.0:
        slope = (1.0 - c) / total
    else:
        slope = 1.0 / ((c + 1.0) * total)
    const = 1.0 - slope * total
    w = const + slope * cum
    return np.clip(w, 0.0, None)


def combined_weights(
    n_bars: int,
    entry_indices: Sequence[int] | np.ndarray,
    exit_indices: Sequence[int] | np.ndarray,
    halflife_bars: float | None = None,
    decay: str = "exponential",
    last_weight: float = 0.0,
    reference_index: int | None = None,
) -> np.ndarray:
    """Uniqueness times decay, normalised to mean 1.

    ``decay`` is ``"exponential"`` (needs ``halflife_bars``), ``"linear"``
    (López de Prado's uniqueness clock, needs ``last_weight``) or ``"none"``.

    Normalising to mean 1 rather than to sum 1 is deliberate: weighted means and
    weighted regressions then keep the same numeric scale as their unweighted
    counterparts, so a weighted expectancy in sigma units is still readable as an
    expectancy in sigma units. It does *not* restore the sample size --
    :func:`effective_sample_size` is the number to quote for that.
    """
    e, x = _check_spans(n_bars, entry_indices, exit_indices)
    if e.size == 0:
        return np.zeros(0, dtype=float)
    u = average_uniqueness(n_bars, e, x)
    if decay == "none":
        d = np.ones_like(u)
    elif decay == "exponential":
        if halflife_bars is None:
            raise ValueError("exponential decay requires halflife_bars")
        d = time_decay_weights(x, halflife_bars, reference_index=reference_index, normalize=False)
    elif decay == "linear":
        order = np.argsort(e, kind="stable")
        d = np.empty_like(u)
        d[order] = linear_decay_weights(u[order], last_weight=last_weight)
    else:
        raise ValueError("decay must be 'exponential', 'linear' or 'none'")
    w = u * d
    mean = float(w.mean())
    if mean <= 0.0:
        return np.ones_like(w)
    return w / mean


def effective_sample_size(weights: Sequence[float] | np.ndarray) -> float:
    """Kish's effective sample size ``(sum w)^2 / sum w^2``.

    **This measures weight DISPERSION only, and by itself it does not correct for
    overlap.** Kish is scale-invariant, so a set of labels that all share the same
    average uniqueness -- the normal case when entries are regularly spaced --
    carries no dispersion at all and this returns ``n``, however severe the
    overlap. Measured on 300 labels with a 20-bar horizon entered every 3 bars,
    Kish reported 293.5 where the genuinely independent count was 43.7, an
    inflation of 2.59x in any t-statistic built on it.

    Use :func:`combined_effective_sample_size` for the number that belongs in a
    standard error. This function remains available because dispersion is a real
    and separate effect -- time decay creates it even with no overlap at all.
    """
    w = np.asarray(weights, dtype=float).ravel()
    w = w[np.isfinite(w)]
    if w.size == 0:
        return 0.0
    s1 = float(w.sum())
    s2 = float(np.sum(w * w))
    if s2 <= 0.0:
        return 0.0
    return s1 * s1 / s2


def overlap_effective_sample_size(
    n_bars: int,
    entry_indices: Sequence[int] | np.ndarray,
    exit_indices: Sequence[int] | np.ndarray,
) -> float:
    """Number of genuinely independent observations, ``sum(average uniqueness)``.

    Lopez de Prado's measure: a label that shares its span with six others
    contributes about one seventh of an observation. Unlike Kish's statistic this
    is *not* scale-invariant and does not care whether the weights are uniform --
    it counts non-overlapping information directly, which is the quantity a
    standard error needs.
    """
    e, x = _check_spans(n_bars, entry_indices, exit_indices)
    if e.size == 0:
        return 0.0
    return float(np.sum(average_uniqueness(n_bars, e, x)))


def combined_effective_sample_size(
    n_bars: int,
    entry_indices: Sequence[int] | np.ndarray,
    exit_indices: Sequence[int] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Effective sample size accounting for **both** overlap and weight dispersion.

    Two independent effects shrink a sample:

    * *overlap* -- labels sharing a span carry the same information, measured by
      :func:`overlap_effective_sample_size`;
    * *dispersion* -- unequal weights (time decay) concentrate the sample on
      fewer observations, measured by Kish.

    They compose multiplicatively as retained fractions:

    ``ESS = n * (sum(u)/n) * (Kish(w)/n) = sum(u) * Kish(w) / n``

    With no overlap this reduces to Kish; with uniform weights it reduces to
    ``sum(u)``. This is the ``n`` that belongs in every standard error, in
    :func:`~tia.validation.metrics.probabilistic_sharpe_ratio` and in
    :func:`~tia.validation.metrics.minimum_track_record_length`.
    """
    e, x = _check_spans(n_bars, entry_indices, exit_indices)
    n = float(e.size)
    if n <= 0.0:
        return 0.0
    ess_overlap = overlap_effective_sample_size(n_bars, e, x)
    if weights is None:
        return ess_overlap
    kish = effective_sample_size(weights)
    return float(ess_overlap * kish / n)


def tstat_inflation_factor(
    weights: Sequence[float] | np.ndarray,
    n_bars: int | None = None,
    entry_indices: Sequence[int] | np.ndarray | None = None,
    exit_indices: Sequence[int] | np.ndarray | None = None,
) -> float:
    """``sqrt(n / n_eff)``: how much a naive t-statistic overstates significance.

    Divide a naively computed t-statistic by this. A value of 2.4 means a
    reported t of 2.4 is really a t of 1.0, i.e. no evidence at all.

    Pass the label spans whenever they are available. Without them only weight
    dispersion can be measured, and for regularly spaced overlapping labels that
    is close to no correction at all.
    """
    w = np.asarray(weights, dtype=float).ravel()
    n = float(w[np.isfinite(w)].size)
    if n <= 0.0:
        return math.nan
    if n_bars is not None and entry_indices is not None and exit_indices is not None:
        ess = combined_effective_sample_size(n_bars, entry_indices, exit_indices, w)
    else:
        ess = effective_sample_size(w)
    if ess <= 0.0:
        return math.nan
    return math.sqrt(n / ess)


def weighted_mean(
    values: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray
) -> float:
    """Weight-normalised mean, ignoring non-finite pairs."""
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if v.size != w.size:
        raise ValueError("values and weights must have the same length")
    ok = np.isfinite(v) & np.isfinite(w)
    if not np.any(ok):
        return math.nan
    tot = float(w[ok].sum())
    if tot <= 0.0:
        return math.nan
    return float(np.sum(v[ok] * w[ok]) / tot)


def weighted_tstat(
    values: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray
) -> float:
    """t-statistic of a weighted mean against zero, using the *effective* n.

    ``t = mean_w / (sd_w / sqrt(n_eff))``. This is the only t-statistic in the
    codebase that should be quoted for overlapping labels; the unweighted
    version is inflated by :func:`tstat_inflation_factor`.
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if v.size != w.size:
        raise ValueError("values and weights must have the same length")
    ok = np.isfinite(v) & np.isfinite(w)
    v, w = v[ok], w[ok]
    if v.size < 2:
        return math.nan
    mu = weighted_mean(v, w)
    tot = float(w.sum())
    if tot <= 0.0:
        return math.nan
    var = float(np.sum(w * (v - mu) ** 2) / tot)
    ess = effective_sample_size(w)
    if var <= 0.0 or ess <= 1.0:
        return math.nan
    return float(mu / math.sqrt(var / ess))
