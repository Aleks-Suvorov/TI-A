"""Training the Edge Book and the calibrator.

The system has a genuine cold-start problem, and naming it is worth more than
hiding it. The trade gate acts on the lower credible bound of expectancy. An
untrained Edge Book has no observations, so every bound sits below zero, so the
system never trades, so it never observes an outcome. Left alone the deadlock is
permanent -- and a design that quietly seeded the book with an optimistic prior
to break it would be smuggling in exactly the assumption the whole architecture
exists to avoid.

The correct resolution is counterfactual evaluation, which is meta-labeling in
López de Prado's sense. A *primary* pass runs the full pipeline with trading
disabled and records every bar that clears the **structural** gates -- regime,
breadth, probability, session, valid features -- but not the expectancy gate,
because that is the thing being learned. Those candidates are then labelled by
the triple barrier, weighted for overlap, and written into the Edge Book. A
*secondary* pass trades with the trained book.

Two properties make this legitimate rather than circular:

*   The primary pass never uses an outcome to decide whether to record a
    candidate. It records everything that was structurally eligible, so the
    Edge Book learns the outcome distribution of *all* such setups, including the
    bad ones. Selecting candidates on their outcomes would be exactly the
    look-ahead this whole document is written against.
*   The labels are purged and weighted. Candidates overlap heavily -- a
    twenty-bar horizon means twenty overlapping labels -- so uniqueness weighting
    is mandatory, not optional. Without it the effective sample size is
    overstated and every credible interval is too narrow: measured here, 66
    candidates carry 30.2 independent observations, and under denser overlap the
    t-statistic inflation reaches 2.6x. The correction must use
    :func:`~tia.labeling.weights.combined_effective_sample_size`; Kish's
    statistic alone is scale-invariant and cannot see uniform overlap.

For a real deployment the training pass runs on the development universe and the
resulting book is frozen or slowly updated. Training and trading on the *same*
bars is in-sample by construction and is only appropriate for smoke tests;
:func:`walk_forward_train` is the honest version and is what the validation suite
calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .config import Config
from .data.sessions import DAILY, SessionSpec
from .engines.edgebook import EdgeBook, SetupFamily
from .fusion.calibration import IsotonicCalibrator, fit_platt
from .labeling.triple_barrier import apply_triple_barrier_to_bars
from .labeling.weights import combined_effective_sample_size, combined_weights
from .pipeline import TIA
from .types import Bar, ExogenousSnapshot, Regime

__all__ = ["Candidate", "CandidateSet", "collect_candidates", "train_edge_book", "walk_forward_train"]


@dataclass(slots=True)
class Candidate:
    """A bar that was structurally eligible for a trade, before expectancy."""

    index: int
    direction: int
    regime: int
    setup: int
    bucket: int
    p_success: float
    log_odds: float
    ebe: float
    stop_sigma: float
    target_sigma: float
    sigma: float
    max_holding: int
    cost_sigma: float


@dataclass(slots=True)
class CandidateSet:
    candidates: list[Candidate] = field(default_factory=list)
    n_bars: int = 0
    n_valid_bars: int = 0

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def eligibility_rate(self) -> float:
        return len(self.candidates) / self.n_valid_bars if self.n_valid_bars else math.nan

    def arrays(self) -> dict[str, np.ndarray]:
        c = self.candidates
        return {
            "index": np.array([x.index for x in c], dtype=np.int64),
            "direction": np.array([x.direction for x in c], dtype=np.int64),
            "regime": np.array([x.regime for x in c], dtype=np.int64),
            "setup": np.array([x.setup for x in c], dtype=np.int64),
            "bucket": np.array([x.bucket for x in c], dtype=np.int64),
            "p_success": np.array([x.p_success for x in c]),
            "stop_sigma": np.array([x.stop_sigma for x in c]),
            "target_sigma": np.array([x.target_sigma for x in c]),
            "sigma": np.array([x.sigma for x in c]),
            "max_holding": np.array([x.max_holding for x in c], dtype=np.int64),
            "cost_sigma": np.array([x.cost_sigma for x in c]),
        }


def collect_candidates(
    bars: Sequence[Bar],
    cfg: Config | None = None,
    session: SessionSpec = DAILY,
    exog: Sequence[ExogenousSnapshot] | None = None,
) -> CandidateSet:
    """Run the pipeline with trading disabled and record eligible bars.

    Structural eligibility means: features valid, a direction, a tradeable
    session phase, a non-quiet regime, transition hazard within tolerance,
    effective breadth above the gate, and calibrated probability above the floor.
    Expectancy is deliberately *not* checked -- it is what we are here to measure.
    """
    cfg = cfg or Config()
    sysm = TIA(cfg, session=session, learn=False)
    out = CandidateSet()

    for i, bar in enumerate(bars):
        e = exog[i] if exog is not None and i < len(exog) else None
        dec = sysm.on_bar(bar, e)
        out.n_bars += 1
        f = dec.engine_outputs and sysm.kernel.snapshot
        snap = sysm.kernel.snapshot
        if not snap.valid:
            continue
        out.n_valid_bars += 1

        fu = dec.fusion
        if fu is None or fu.direction == 0:
            continue
        if dec.regime is Regime.QUIET:
            continue
        if not snap.phase.is_tradeable or snap.phase.is_auction:
            continue
        if fu.ebe < cfg.ebe_min or fu.p_success < cfg.p_min:
            continue
        haz = float(sysm.regime_engine.state.hazard)
        if haz > cfg.regime_hazard_max:
            continue

        spec = sysm.planner.plan(
            fu.direction, bar.close, snap, dec.regime, dec.regime_posterior
        )
        if spec is None:
            continue
        cost = dec.costs.round_trip_sigma if dec.costs is not None else math.inf
        if not math.isfinite(cost):
            continue

        out.candidates.append(
            Candidate(
                index=i,
                direction=fu.direction,
                regime=int(dec.regime),
                setup=int(
                    __import__("tia.decision.policy", fromlist=["classify_setup"]).classify_setup(
                        dec.engine_outputs, dec.regime, dec.regime_posterior
                    )
                ),
                bucket=sysm.edge_book.bucket_of(fu.log_odds),
                p_success=fu.p_success,
                log_odds=fu.log_odds,
                ebe=fu.ebe,
                stop_sigma=spec.stop_sigma,
                target_sigma=spec.target_sigma,
                sigma=spec.sigma,
                max_holding=spec.max_holding_bars,
                cost_sigma=cost,
            )
        )
    return out


def train_edge_book(
    bars: Sequence[Bar],
    candidates: CandidateSet,
    cfg: Config | None = None,
    book: EdgeBook | None = None,
    calibrator: IsotonicCalibrator | None = None,
) -> tuple[EdgeBook, IsotonicCalibrator, dict[str, float]]:
    """Label the candidates and write the outcomes into an Edge Book.

    Returns the book, a fitted isotonic calibrator, and a diagnostics dict. The
    diagnostics include the *effective* sample size after uniqueness weighting,
    which is typically a small fraction of the raw candidate count and is the
    number any subsequent t-statistic must be computed against.
    """
    cfg = cfg or Config()
    book = book if book is not None else EdgeBook(cfg)
    a = candidates.arrays()
    diag: dict[str, float] = {"n_candidates": float(len(candidates))}
    if len(candidates) == 0:
        return book, calibrator or IsotonicCalibrator(), diag

    res = apply_triple_barrier_to_bars(
        bars,
        entry_indices=a["index"],
        directions=a["direction"],
        stop_sigma=a["stop_sigma"],
        target_sigma=a["target_sigma"],
        sigma=a["sigma"],
        max_holding=a["max_holding"],
        execution_lag=cfg.execution_lag_bars,
    )

    w = combined_weights(
        n_bars=len(bars),
        entry_indices=res.entry_index,
        exit_indices=res.exit_index,
        halflife_bars=cfg.edge_halflife_bars,
    )
    # Overlap-aware, not Kish. Kish is scale-invariant and therefore blind to
    # the uniform down-weighting that regularly spaced overlapping labels
    # produce; using it inflated this t-statistic by up to 2.6x.
    ess = combined_effective_sample_size(len(bars), res.entry_index, res.exit_index, w)
    diag["effective_sample_size"] = float(ess)
    diag["uniqueness_ratio"] = float(ess / max(len(candidates), 1))

    ok = np.isfinite(res.ret_sigma)
    for j in np.nonzero(ok)[0]:
        book.observe(
            int(a["regime"][j]),
            int(a["setup"][j]),
            int(a["bucket"][j]),
            float(res.ret_sigma[j]),
            weight=float(w[j]),
        )

    # Calibrate the probability against what actually happened. "Success" is
    # reaching the target barrier before the stop, which is the event the
    # probability claims to describe -- timeouts count as failures for this
    # purpose even though they are not losses, because the forecast was about
    # the barrier, not the sign.
    success = (res.outcome[ok] > 0).astype(float)
    praw = a["p_success"][ok]
    cal = calibrator or IsotonicCalibrator()
    if praw.size >= cfg.calibration_min_samples:
        cal.fit(praw, success, weights=w[ok])
        diag["calibration_fitted"] = 1.0
    else:
        diag["calibration_fitted"] = 0.0

    diag["hit_rate"] = float(success.mean()) if success.size else math.nan
    diag["mean_ret_sigma"] = float(np.average(res.ret_sigma[ok], weights=w[ok])) if ok.any() else math.nan
    diag["mean_cost_sigma"] = float(np.mean(a["cost_sigma"][ok])) if ok.any() else math.nan
    diag["net_expectancy_sigma"] = diag["mean_ret_sigma"] - diag["mean_cost_sigma"]
    # The t-statistic that matters uses the *effective* sample size, not the
    # candidate count. Reporting the naive one would overstate significance by
    # roughly sqrt(n / ess), which for a twenty-bar horizon is a factor of four.
    if ok.any():
        r = res.ret_sigma[ok]
        wt = w[ok]
        mu = float(np.average(r, weights=wt))
        var = float(np.average((r - mu) ** 2, weights=wt))
        diag["t_stat_effective"] = mu / math.sqrt(var / max(ess, 1.0)) if var > 0 else math.nan
    return book, cal, diag


def walk_forward_train(
    bars: Sequence[Bar],
    cfg: Config | None = None,
    session: SessionSpec = DAILY,
    n_folds: int = 4,
    min_train_frac: float = 0.35,
) -> tuple[list[TIA], dict[str, float]]:
    """Anchored walk-forward: train on the past, trade the next block, repeat.

    This is the only training protocol whose results mean anything. Each fold's
    Edge Book and calibrator are fitted strictly on bars preceding the block they
    are used to trade, so no outcome influences the decision that produced it.
    The per-fold systems are returned so their trades can be concatenated into a
    single out-of-sample record.
    """
    cfg = cfg or Config()
    n = len(bars)
    start = int(n * min_train_frac)
    if start < cfg.rank_window or n - start < n_folds * 50:
        raise ValueError(
            f"need substantially more than {cfg.rank_window} bars for walk-forward training; got {n}"
        )
    edges = np.linspace(start, n, n_folds + 1).astype(int)

    systems: list[TIA] = []
    agg: dict[str, float] = {"folds": float(n_folds)}
    for k in range(n_folds):
        tr_end, te_end = int(edges[k]), int(edges[k + 1])
        train_bars = bars[:tr_end]
        cands = collect_candidates(train_bars, cfg, session)
        book, cal, diag = train_edge_book(train_bars, cands, cfg)

        # Trade the out-of-sample block with the trained book. The system is
        # re-warmed on the full prefix so its streaming state matches what a live
        # system would hold, but `learn=False` over the prefix keeps the book
        # from seeing the block it is about to trade.
        sysm = TIA(cfg, session=session, edge_book=book, learn=False)
        sysm.calibrator._iso = cal
        sysm.calibrator.active = cal.n_fit > 0
        for i in range(tr_end):
            sysm.on_bar(bars[i])
        sysm.learn = True
        for i in range(tr_end, te_end):
            sysm.on_bar(bars[i])
        # Only trades opened inside the test block are out of sample.
        sysm.trades = [t for t in sysm.trades if t.entry_index >= tr_end]
        systems.append(sysm)
        for kk, vv in diag.items():
            agg[f"fold{k}_{kk}"] = vv
    return systems, agg
