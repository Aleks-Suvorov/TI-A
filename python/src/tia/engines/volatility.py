"""Volatility engine: mostly a statement about whether to act, not which way.

Volatility is the most forecastable quantity available to this system and it is
almost entirely directionless. Pretending otherwise is a standard error: a
compression that resolves into an expansion has, ex ante, no reliable direction,
and an indicator that assigns one is manufacturing a signal from a real
observation.

So this engine usually returns a score near zero with a *confident* reliability.
That is not the same as abstaining. Under the pooling of
``docs/01-THEORY.md`` §6, a confident-neutral engine contributes nothing to the
pooled log-odds while adding to its variance, which correctly widens the
credible interval and pulls the reported probability toward 50%. An engine that
abstained instead would leave the pool falsely narrow.

Where it does contribute direction is narrow and defensible: an expansion already
under way, whose displacement is being held, is weak confirming evidence for the
direction that is expanding.
"""

from __future__ import annotations

import math

from ..config import Config
from ..types import BarContext, EngineOutput, Regime, clip
from .base import BaseEngine, band_score

__all__ = ["VolatilityEngine"]


class VolatilityEngine(BaseEngine):
    name = "volatility"

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__(cfg)
        self.warmup = 30
        self.regime_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    def _compute(self, ctx: BarContext) -> EngineOutput:
        f = ctx.features
        notes: list[str] = []

        # Favourability of the volatility state for *taking a position at all*.
        # Too low and the target is unreachable inside the vertical barrier; too
        # high and the stop is noise and the cost model degrades.
        fav = band_score(f.vol_rank, 0.25, 0.80) if f.vol_rank == f.vol_rank else 0.0

        expanding = f.vol_ratio == f.vol_ratio and f.vol_ratio > 1.15
        compressed = f.compression == f.compression and f.compression > 0.75

        score = 0.0
        if expanding and f.ret_norm == f.ret_norm and abs(f.ret_norm) > 0.5:
            # Confirming only, and deliberately small: the expansion is real
            # information, its direction is barely so.
            score = clip(0.35 * math.tanh(f.ret_norm / 2.0), -0.4, 0.4)
            notes.append("volatility expanding with displacement")
        if compressed:
            notes.append("range compressed; expansion hazard elevated, direction unknown")

        vov_penalty = 1.0
        if f.vol_of_vol == f.vol_of_vol:
            # Unstable volatility means the forecast that sets every barrier is
            # itself unreliable, which is a reason to trust everything less.
            vov_penalty = clip(1.0 - 0.8 * clip(f.vol_of_vol / 0.7, 0.0, 1.0), 0.3, 1.0)

        p_stress = self.regime_probs[int(Regime.STRESS)]
        reliability = clip(0.35 + 0.45 * clip((fav + 1.0) / 2.0, 0.0, 1.0), 0.0, 1.0) * vov_penalty
        reliability *= 1.0 - 0.4 * p_stress

        if f.vol_rank == f.vol_rank and f.vol_rank > 0.9:
            notes.append("volatility in top decile; barriers wide, size reduced")

        return self.emit(
            score,
            reliability,
            features={
                "vol_rank": f.vol_rank,
                "vol_ratio": f.vol_ratio,
                "compression": f.compression,
                "sigma_fcst": f.sigma_fcst,
                "favourable": fav,
            },
            diagnostics={"vov_penalty": vov_penalty, "vol_of_vol": f.vol_of_vol},
            notes=notes,
        )
