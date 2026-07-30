"""Purged and combinatorial purged cross-validation.

The problem
-----------
Ordinary k-fold cross-validation is invalid for financial labels for two
independent reasons, and both of them leak *future* information into training:

1.  **Overlap.** A triple-barrier label spans ``[entry, exit]``. If a training
    label's span intersects the test span, the training set contains the very
    returns the test set is scored on. López de Prado (*Advances in Financial
    Machine Learning*, 2018, ch. 7) calls the remedy **purging**: drop every
    training label whose span overlaps the test span.

2.  **Serial correlation.** Even a training label that ends just before the test
    block begins is informative about it, because volatility and structure are
    persistent. The remedy is an **embargo**: additionally drop training labels
    that begin within a short window *after* each test block. (Only after: a
    label ending before the test block starts and not overlapping it is
    legitimate past information, and dropping it costs data for nothing.)

Shuffling, the other habit imported from cross-sectional machine learning, is
never applied here at all: groups are contiguous in time.

Why combinatorial
-----------------
A single k-fold split produces one out-of-sample path, so it yields one Sharpe
ratio with no distribution attached, and the temptation is to treat that single
number as the answer. Combinatorial purged CV (López de Prado ch. 12) instead
holds out every combination of ``k`` of the ``K`` groups, producing
``C(K, k)`` fits and, from them, ``phi = C(K, k) * k / K = C(K-1, k-1)``
distinct full-length backtest paths. What comes out is a *distribution* of
out-of-sample performance, which supports the only statement worth making:
"the median path Sharpe was X and 80% of paths were positive", rather than
"the backtest Sharpe was X".

With the defaults in :class:`~tia.config.Config` (``cv_folds=8``,
``cv_test_groups=2``) that is 28 fits and 7 paths.

Leakage is asserted, not assumed
--------------------------------
:meth:`PurgedKFold.leakage_report` re-checks, split by split, that no surviving
training label's span intersects any test span. It is cheap and it has caught
real bugs; run it in CI. ``PurgedKFold(..., purge=False)`` exists solely so that
the test suite can demonstrate the check firing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterator, Sequence

import numpy as np

__all__ = [
    "LeakageReport",
    "PurgedKFold",
    "CombinatorialPurgedCV",
]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakageReport:
    """Audit of one cross-validation scheme.

    ``ok`` is ``True`` only when no training label in any split overlaps that
    split's test span *and* no index appears in both train and test. A ``False``
    here invalidates every number computed from the scheme, so it is treated as
    an error rather than a warning.
    """

    scheme: str
    n_splits: int
    n_samples: int
    n_bars: int
    embargo_bars: int
    purged_total: int
    embargoed_total: int
    overlap_violations: int
    duplicate_violations: int
    train_sizes: tuple[int, ...] = field(default_factory=tuple)
    test_sizes: tuple[int, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.overlap_violations == 0 and self.duplicate_violations == 0

    @property
    def mean_train_fraction(self) -> float:
        if not self.train_sizes or self.n_samples == 0:
            return math.nan
        return float(np.mean(self.train_sizes) / self.n_samples)

    @property
    def dropped_fraction(self) -> float:
        """Share of the sample sacrificed to purging and embargo, per split."""
        if self.n_splits == 0 or self.n_samples == 0:
            return math.nan
        return (self.purged_total + self.embargoed_total) / (self.n_splits * self.n_samples)

    def as_dict(self) -> dict[str, float]:
        return {
            "n_splits": float(self.n_splits),
            "n_samples": float(self.n_samples),
            "embargo_bars": float(self.embargo_bars),
            "purged_total": float(self.purged_total),
            "embargoed_total": float(self.embargoed_total),
            "overlap_violations": float(self.overlap_violations),
            "duplicate_violations": float(self.duplicate_violations),
            "mean_train_fraction": self.mean_train_fraction,
            "dropped_fraction": self.dropped_fraction,
            "leakage_ok": float(self.ok),
        }

    def __str__(self) -> str:
        lines = [
            f"{self.scheme}: {self.n_splits} splits over {self.n_samples} labels "
            f"/ {self.n_bars} bars",
            f"  embargo                 : {self.embargo_bars} bars",
            f"  purged (total)          : {self.purged_total}",
            f"  embargoed (total)       : {self.embargoed_total}",
            f"  mean train fraction     : {self.mean_train_fraction:.3f}",
            f"  train/test overlap      : {self.overlap_violations} violations",
            f"  index appears in both   : {self.duplicate_violations} violations",
            f"  VERDICT                 : {'no leakage detected' if self.ok else 'LEAKAGE'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base machinery
# ---------------------------------------------------------------------------


class _PurgedBase:
    """Shared group construction, purging and embargo logic.

    Labels are ordered by ``entry_index`` internally; the indices yielded are
    always positions in the *caller's* arrays, so unsorted input is safe.
    """

    scheme = "purged"

    def __init__(
        self,
        n_groups: int,
        entry_indices: Sequence[int] | np.ndarray,
        exit_indices: Sequence[int] | np.ndarray,
        embargo_frac: float = 0.0,
        n_bars: int | None = None,
        purge: bool = True,
        embargo: bool = True,
    ) -> None:
        e = np.asarray(entry_indices, dtype=np.int64).ravel()
        x = np.asarray(exit_indices, dtype=np.int64).ravel()
        if e.size != x.size:
            raise ValueError("entry_indices and exit_indices must have the same length")
        if e.size == 0:
            raise ValueError("no labels")
        if np.any(x < e):
            raise ValueError("exit_index must be >= entry_index for every label")
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        if n_groups > e.size:
            raise ValueError(f"n_groups={n_groups} exceeds the label count {e.size}")
        if not (0.0 <= embargo_frac < 1.0):
            raise ValueError("embargo_frac must lie in [0, 1)")

        self.entry = e
        self.exit = x
        self.n_samples = int(e.size)
        self.n_bars = int(n_bars) if n_bars is not None else int(x.max()) + 1
        if self.n_bars <= int(x.max()):
            raise ValueError("n_bars must exceed the largest exit index")
        self.n_groups = int(n_groups)
        self.embargo_frac = float(embargo_frac)
        self.embargo_bars = (
            int(math.ceil(self.embargo_frac * self.n_bars)) if embargo else 0
        )
        self.purge = bool(purge)
        self.use_embargo = bool(embargo)

        order = np.argsort(e, kind="stable")
        self._order = order
        self._groups: tuple[np.ndarray, ...] = tuple(
            np.asarray(g, dtype=np.int64) for g in np.array_split(order, self.n_groups)
        )
        if any(g.size == 0 for g in self._groups):  # pragma: no cover - guarded above
            raise ValueError("empty group; reduce n_groups")

    # -- geometry ----------------------------------------------------------

    def group_positions(self, g: int) -> np.ndarray:
        """Positions (in caller order) of the labels forming group ``g``."""
        return self._groups[g]

    def group_span(self, g: int) -> tuple[int, int]:
        """Bar interval ``[first entry, last exit]`` covered by group ``g``."""
        pos = self._groups[g]
        return int(self.entry[pos].min()), int(self.exit[pos].max())

    def _train_for(self, test_groups: Sequence[int]) -> tuple[np.ndarray, np.ndarray, int, int]:
        test_pos = np.sort(np.concatenate([self._groups[g] for g in test_groups]))
        spans = [self.group_span(g) for g in test_groups]

        keep = np.ones(self.n_samples, dtype=bool)
        keep[test_pos] = False
        purged = np.zeros(self.n_samples, dtype=bool)
        embargoed = np.zeros(self.n_samples, dtype=bool)

        for start, end in spans:
            if self.purge:
                # Interval intersection: [a, b] meets [start, end].
                overlap = (self.entry <= end) & (self.exit >= start)
                purged |= overlap & keep
            if self.embargo_bars > 0:
                window = (self.entry > end) & (self.entry <= end + self.embargo_bars)
                embargoed |= window & keep

        n_purged = int(np.sum(purged & keep))
        n_embargoed = int(np.sum(embargoed & keep & ~purged))
        keep &= ~purged
        keep &= ~embargoed
        train_pos = np.nonzero(keep)[0]
        return train_pos, test_pos, n_purged, n_embargoed

    # -- iteration ---------------------------------------------------------

    def _test_group_sets(self) -> list[tuple[int, ...]]:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def n_splits(self) -> int:
        return len(self._test_group_sets())

    def split(self) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        """Yield ``(train_idx, test_idx, path_id)`` for every split."""
        for split_id, groups in enumerate(self._test_group_sets()):
            train_pos, test_pos, _, _ = self._train_for(groups)
            yield train_pos, test_pos, split_id

    # -- audit -------------------------------------------------------------

    def leakage_report(self) -> LeakageReport:
        """Verify that no surviving training label overlaps any test span.

        Also verifies the trivially necessary condition that train and test
        index sets are disjoint. Both checks are exhaustive over splits.
        """
        purged_total = 0
        embargoed_total = 0
        overlaps = 0
        dupes = 0
        train_sizes: list[int] = []
        test_sizes: list[int] = []
        for groups in self._test_group_sets():
            train_pos, test_pos, np_, ne_ = self._train_for(groups)
            purged_total += np_
            embargoed_total += ne_
            train_sizes.append(int(train_pos.size))
            test_sizes.append(int(test_pos.size))
            dupes += int(np.intersect1d(train_pos, test_pos).size)
            for g in groups:
                start, end = self.group_span(g)
                te = self.entry[train_pos]
                tx = self.exit[train_pos]
                overlaps += int(np.sum((te <= end) & (tx >= start)))
        return LeakageReport(
            scheme=self.scheme,
            n_splits=len(self._test_group_sets()),
            n_samples=self.n_samples,
            n_bars=self.n_bars,
            embargo_bars=self.embargo_bars,
            purged_total=purged_total,
            embargoed_total=embargoed_total,
            overlap_violations=overlaps,
            duplicate_violations=dupes,
            train_sizes=tuple(train_sizes),
            test_sizes=tuple(test_sizes),
        )


# ---------------------------------------------------------------------------
# Plain purged k-fold
# ---------------------------------------------------------------------------


class PurgedKFold(_PurgedBase):
    """K contiguous test groups, purged and embargoed, no shuffling.

    Example
    -------
    >>> import numpy as np
    >>> entry = np.arange(0, 100, 5)
    >>> exit_ = entry + 8                     # deliberately overlapping labels
    >>> cv = PurgedKFold(5, entry, exit_, embargo_frac=0.02, n_bars=110)
    >>> report = cv.leakage_report()
    >>> report.ok
    True

    ``purge=False`` disables the overlap purge and is a diagnostic only: it
    exists so that the leakage assertion can be shown to fire, and so that the
    cost of purging (how much training data it removes) can be measured.
    """

    scheme = "purged k-fold"

    def _test_group_sets(self) -> list[tuple[int, ...]]:
        return [(g,) for g in range(self.n_groups)]


# ---------------------------------------------------------------------------
# Combinatorial purged CV
# ---------------------------------------------------------------------------


class CombinatorialPurgedCV(_PurgedBase):
    """Hold out every combination of ``n_test_groups`` of ``n_groups`` groups.

    Parameters
    ----------
    n_groups, n_test_groups
        ``K`` and ``k``. ``Config.cv_folds`` and ``Config.cv_test_groups``.
    entry_indices, exit_indices
        Label spans, as produced by
        :meth:`~tia.labeling.triple_barrier.BarrierOutcome.spans`.
    embargo_frac
        ``Config.cv_embargo_frac``; embargo length is
        ``ceil(embargo_frac * n_bars)`` bars applied after each test block.

    Path counting
    -------------
    ``n_splits = C(K, k)`` fits are performed. Each group appears in the test set
    of ``C(K-1, k-1)`` of them, so the out-of-sample blocks can be reassembled
    into ``phi = C(K, k) * k / K = C(K-1, k-1)`` distinct full-length backtest
    paths (:attr:`n_paths`). :meth:`path_assignments` and :meth:`path_blocks`
    give the mapping.

    ``split()`` yields ``(train_idx, test_idx, path_id)`` where ``path_id`` is
    the index of the test-group *combination* -- i.e. the split id. It is
    reported under that name because it is what a caller keys per-split results
    on; the ``phi`` reassembled paths each draw blocks from ``k`` different
    combinations, so a split does not correspond one-to-one with a path. Use
    :meth:`path_blocks` when you need the paths themselves, for instance to
    compute the distribution of path Sharpe ratios that
    :func:`~tia.validation.metrics.stability_across_folds` summarises.
    """

    scheme = "combinatorial purged CV"

    def __init__(
        self,
        n_groups: int,
        n_test_groups: int,
        entry_indices: Sequence[int] | np.ndarray,
        exit_indices: Sequence[int] | np.ndarray,
        embargo_frac: float = 0.0,
        n_bars: int | None = None,
        purge: bool = True,
        embargo: bool = True,
    ) -> None:
        super().__init__(
            n_groups,
            entry_indices,
            exit_indices,
            embargo_frac=embargo_frac,
            n_bars=n_bars,
            purge=purge,
            embargo=embargo,
        )
        if not (1 <= n_test_groups < n_groups):
            raise ValueError("n_test_groups must satisfy 1 <= k < K")
        self.n_test_groups = int(n_test_groups)
        self._combos: tuple[tuple[int, ...], ...] = tuple(
            combinations(range(self.n_groups), self.n_test_groups)
        )

    def _test_group_sets(self) -> list[tuple[int, ...]]:
        return list(self._combos)

    # -- path bookkeeping --------------------------------------------------

    @property
    def n_paths(self) -> int:
        """``C(K, k) * k / K``: the number of reconstructable backtest paths."""
        return math.comb(self.n_groups - 1, self.n_test_groups - 1)

    def path_assignments(self) -> dict[int, list[tuple[int, int]]]:
        """``path_id -> [(split_id, group_id), ...]``, one group per path per group slot.

        Construction: for each group ``g``, the splits that test ``g`` are
        enumerated in order and the ``j``-th of them is assigned to path ``j``.
        Every path therefore covers each of the ``K`` groups exactly once, which
        is what makes it a full-length, non-overlapping out-of-sample path.
        """
        paths: dict[int, list[tuple[int, int]]] = {p: [] for p in range(self.n_paths)}
        for g in range(self.n_groups):
            j = 0
            for split_id, combo in enumerate(self._combos):
                if g in combo:
                    paths[j].append((split_id, g))
                    j += 1
        return paths

    def path_blocks(self) -> list[list[tuple[int, int, np.ndarray]]]:
        """``[(split_id, group_id, test_positions), ...]`` per path, in time order.

        ``test_positions`` are positions in the caller's label arrays. To build
        a path's out-of-sample return series, take the model fitted on
        ``split_id``'s training set and score exactly those positions.
        """
        assign = self.path_assignments()
        out: list[list[tuple[int, int, np.ndarray]]] = []
        for p in range(self.n_paths):
            blocks = [
                (split_id, g, self.group_positions(g)) for split_id, g in assign[p]
            ]
            blocks.sort(key=lambda t: self.group_span(t[1])[0])
            out.append(blocks)
        return out

    def summary(self) -> str:
        """One-line description, for the report header."""
        return (
            f"CPCV K={self.n_groups} k={self.n_test_groups}: "
            f"{self.n_splits} fits, {self.n_paths} backtest paths, "
            f"embargo {self.embargo_bars} bars"
        )
