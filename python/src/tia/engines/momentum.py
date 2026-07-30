"""Momentum engine: displacement that was cheap in liquidity terms.

This is condition (C) of ``docs/01-THEORY.md`` §0.1 made operational. The claim is
not "price went up, so it will keep going up" -- that has a weak and unstable
empirical basis at these horizons. The claim is narrower and better supported:
*price moved far while consuming little liquidity*, which under any Kyle-type
model is the signature of informed flow, and informed flow has permanent impact.

So the engine's evidence is high :math:`|\\mathrm{LADR}|`, not a large return. A
large return on enormous volume is the opposite signal, and the liquidity engine
picks it up as absorption.
"""

from __future__ import annotations

import math

from ..config import Config
from ..types import BarContext, EngineOutput, Regime, clip
from .base import BaseEngine, squash

__all__ = ["MomentumEngine"]


class MomentumEngine(BaseEngine):
    name = "momentum"

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__(cfg)
        self.warmup = 20
        self.regime_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
        self._ladr_ema = 0.0
        self._ema_n = 0

    def _reset(self) -> None:
        self._ladr_ema = 0.0
        self._ema_n = 0

    def _compute(self, ctx: BarContext) -> EngineOutput:
        f = ctx.features
        if f.ladr_rank != f.ladr_rank:
            return self.abstain("displacement ratio unavailable")

        p_trend = self.regime_probs[int(Regime.TREND)]
        p_revert = self.regime_probs[int(Regime.REVERT)]
        p_quiet = self.regime_probs[int(Regime.QUIET)]

        # Only the upper half of the LADR distribution is evidence. A bar with a
        # median cost of displacement tells us nothing, so it maps to zero rather
        # than to a weak opinion.
        strength = max(0.0, 2.0 * f.ladr_rank - 1.0)
        direction = 0.0
        if f.ladr == f.ladr and f.ladr != 0.0:
            direction = math.copysign(1.0, f.ladr)

        # Persistence of cheap displacement over a few bars, so that a single
        # anomalous print does not carry the engine.
        inst = direction * strength
        alpha = 0.35
        self._ladr_ema = (1.0 - alpha) * self._ladr_ema + alpha * inst if self._ema_n else inst
        self._ema_n += 1

        parts = [(inst, 0.5), (self._ladr_ema, 0.3)]
        if f.clv_ema == f.clv_ema:
            # Where bars have been closing within their ranges: cheap
            # corroboration that the displacement is being held rather than
            # given back.
            parts.append((clip(f.clv_ema, -1.0, 1.0), 0.2))
        score = sum(v * w for v, w in parts) / sum(w for _, w in parts)

        # Jumps are displacement without a tradeable path. They are real
        # information but our stop cannot be assumed to have survived them, so
        # jump-dominated conditions reduce trust rather than raise it.
        jump_penalty = 1.0
        if f.jump_share == f.jump_share:
            jump_penalty = clip(1.0 - 0.5 * f.jump_share, 0.4, 1.0)

        reliability = (
            clip(0.20 + 0.80 * strength, 0.0, 1.0)
            * clip(0.35 + 0.65 * p_trend, 0.0, 1.0)
            * (1.0 - 0.5 * p_revert)
            * (1.0 - 0.7 * p_quiet)
            * jump_penalty
        )

        notes = []
        if f.ladr_rank > 0.9:
            notes.append(f"displacement in top decile of liquidity cost ({f.ladr_rank:.2f})")
        if f.jump_share == f.jump_share and f.jump_share > 0.5:
            notes.append("variance is jump-dominated")

        return self.emit(
            score,
            reliability,
            features={
                "ladr_rank": f.ladr_rank,
                "ladr": f.ladr,
                "ret_norm": f.ret_norm,
                "clv_ema": f.clv_ema,
                "jump_share": f.jump_share,
            },
            diagnostics={"strength": strength, "ladr_ema": self._ladr_ema},
            notes=notes,
        )
