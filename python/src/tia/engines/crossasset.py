"""Cross-asset engine: peer confirmation and the volatility environment.

Optional and degrading gracefully. When ``ExogenousSnapshot`` carries nothing,
the engine reports ``reliability = 0`` and drops out of the pool rather than
contributing a fabricated neutral reading.

What it measures is deliberately modest. Cross-sectional information is genuinely
valuable, but its main value is in *ranking* instruments against each other,
which is a portfolio construction problem this single-instrument system does not
solve (see ``docs/16-ROADMAP.md``). What is available here is narrower: whether an
instrument is moving with or against the peers it usually moves with, and whether
the broad volatility environment is deteriorating.

An instrument moving *against* its usual peers is the more informative case. It
means the move is idiosyncratic, which makes a mechanical or liquidity-driven
explanation more likely than an informational one -- and mechanical moves revert.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..features.rolling import EWMA, RollingCorrelation
from ..types import BarContext, EngineOutput, clip
from .base import BaseEngine, squash

__all__ = ["CrossAssetEngine"]


class CrossAssetEngine(BaseEngine):
    name = "crossasset"

    def __init__(self, cfg: Config | None = None, max_peers: int = 8) -> None:
        super().__init__(cfg)
        self.warmup = 40
        self.max_peers = int(max_peers)
        self._peer_names: list[str] = []
        self._corr: RollingCorrelation | None = None
        self._iv_ewma = EWMA(60.0)
        self._iv_prev = math.nan

    def _reset(self) -> None:
        self._peer_names = []
        self._corr = None
        self._iv_ewma.reset()
        self._iv_prev = math.nan

    def _compute(self, ctx: BarContext) -> EngineOutput:
        if not self.cfg.enable_crossasset_engine:
            return self.abstain("disabled")

        f = ctx.features
        peers = ctx.exog.peers or {}
        iv = ctx.exog.implied_vol
        if not peers and iv is None:
            return self.abstain("no exogenous data")

        notes: list[str] = []
        score = 0.0
        weight = 0.0

        if peers and f.ret == f.ret and f.sigma_bp == f.sigma_bp and f.sigma_bp > 0.0:
            names = sorted(peers)[: self.max_peers]
            if self._peer_names != names:
                # The peer set changed, so the correlation history no longer
                # describes the same object and must be discarded rather than
                # reinterpreted.
                self._peer_names = names
                self._corr = RollingCorrelation(len(names) + 1, halflife=120.0)
            vec = np.array([f.ret / f.sigma_bp] + [peers[n] for n in names], dtype=np.float64)
            if np.all(np.isfinite(vec)) and self._corr is not None:
                self._corr.update(vec)
                if self._corr.count > 40:
                    C = self._corr.correlation(ridge=0.05)
                    betas = C[0, 1:]
                    peer_moves = vec[1:]
                    # Expected co-move given the usual relationships.
                    expected = float(np.dot(betas, peer_moves) / max(len(betas), 1))
                    actual = float(vec[0])
                    residual = actual - expected
                    # Idiosyncratic excursions revert; shared moves persist.
                    score += -squash(residual, 2.0) * 0.6
                    weight += 0.6
                    if abs(residual) > 2.0:
                        notes.append("move is idiosyncratic versus peers")
                    elif abs(expected) > 1.0 and math.copysign(1.0, expected) == math.copysign(
                        1.0, actual
                    ):
                        notes.append("move confirmed by peers")

        if iv is not None and math.isfinite(iv):
            prev = self._iv_prev
            self._iv_prev = float(iv)
            self._iv_ewma.update(float(iv))
            base = self._iv_ewma.value
            if prev == prev and base == base and base > 0.0:
                # A jump in implied volatility is a broad risk-off signal. It is
                # not directional for an arbitrary instrument, so it enters as a
                # reliability discount, not as a score.
                shock = (float(iv) - prev) / base
                if shock > 0.15:
                    notes.append("implied volatility spiking")

        if weight <= 0.0:
            return self.emit(
                0.0, 0.15, features={"peers": float(len(peers))}, notes=("insufficient peer history",)
            )

        score = clip(score / weight, -1.0, 1.0)
        reliability = clip(0.20 + 0.45 * min(len(peers) / 4.0, 1.0), 0.0, 0.65)
        if iv is not None and math.isfinite(iv):
            iv_base = self._iv_ewma.value
            if iv_base == iv_base and iv_base > 0.0 and iv / iv_base > 1.35:
                reliability *= 0.6

        return self.emit(
            score,
            reliability,
            features={"peers": float(len(peers)), "implied_vol": float(iv) if iv else math.nan},
            diagnostics={"weight": weight},
            notes=notes,
        )
