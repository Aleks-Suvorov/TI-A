"""Positioning engine: dealer gamma and open interest. Disabled by default.

The mechanism is real and well understood. A dealer who is short gamma must hedge
in the direction of the move, amplifying it; long gamma forces the opposite,
damping it. The effect on realised volatility around large option expiries is
documented.

The data is not. Aggregate dealer positioning is not published. Every retail and
most commercial "gamma exposure" figures are reconstructions built on assumptions
about who is on which side of open interest -- typically that customers buy calls
and sell puts, or some variant -- and those assumptions are neither verifiable nor
stable. The reconstruction error is unknown and probably large, and it is
correlated with exactly the market conditions in which one would want to use the
signal.

So this engine exists, is correct as far as it goes, is documented, and is
``False`` in the default configuration. ``Config.enable_positioning_engine``
turns it on for users who have a positioning feed they can actually defend. It
may never be load-bearing: ``SPEC.md`` §9 forbids any optional module from
appearing in the decision gate's conjunctive conditions.
"""

from __future__ import annotations

import math

from ..config import Config
from ..features.rolling import CausalRank
from ..types import BarContext, EngineOutput, clip
from .base import BaseEngine, squash

__all__ = ["PositioningEngine"]


class PositioningEngine(BaseEngine):
    name = "positioning"

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__(cfg)
        self.warmup = 40
        self._oi_rank = CausalRank(self.cfg.rank_window, 40)

    def _reset(self) -> None:
        self._oi_rank.reset()

    def _compute(self, ctx: BarContext) -> EngineOutput:
        if not self.cfg.enable_positioning_engine:
            return self.abstain("disabled: see docs/17-EVIDENCE-REVIEW.md on data quality")

        g = ctx.exog.gamma_imbalance
        oi = ctx.exog.oi_change
        if g is None and oi is None:
            return self.abstain("no positioning data")

        f = ctx.features
        notes: list[str] = []
        score = 0.0
        weight = 0.0

        if g is not None and math.isfinite(g):
            gi = clip(float(g), -1.0, 1.0)
            if f.ret_norm == f.ret_norm:
                # Short dealer gamma (negative imbalance) amplifies the move in
                # progress; long gamma damps it. The sign convention is stated
                # in the ExogenousSnapshot contract and must be honoured by the
                # feed adapter.
                amplify = -gi
                score += amplify * math.tanh(f.ret_norm / 2.0) * 0.7
                weight += 0.7
                if abs(gi) > 0.5:
                    notes.append(
                        "dealers likely short gamma: moves amplified"
                        if gi < 0
                        else "dealers likely long gamma: moves damped"
                    )

        if oi is not None and math.isfinite(oi):
            r = self._oi_rank.update(float(oi))
            if r == r and f.ret == f.ret:
                # Rising open interest alongside a directional move is new
                # positioning rather than closing, which is the weakly
                # continuation-favouring case.
                if r > 0.75 and f.ret != 0.0:
                    score += math.copysign(0.3 * (r - 0.75) / 0.25, f.ret)
                    weight += 0.3
                    notes.append("open interest building with the move")

        if weight <= 0.0:
            return self.emit(0.0, 0.1, notes=("positioning data present but uninformative",))

        # Capped hard. Even with a good feed, the inferential chain from open
        # interest to dealer inventory to hedging flow has several unverifiable
        # links, and the reliability should reflect the weakest one.
        reliability = clip(0.15 + 0.35 * (weight / 1.0), 0.0, 0.45)

        return self.emit(
            clip(score / weight, -1.0, 1.0),
            reliability,
            features={
                "gamma_imbalance": float(g) if g is not None else math.nan,
                "oi_change": float(oi) if oi is not None else math.nan,
            },
            diagnostics={"weight": weight},
            notes=notes,
        )
