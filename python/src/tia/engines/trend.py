"""Trend engine: direction from filtered slope, trust from measured persistence.

The separation between the two channels is the whole design. ``score`` says which
way the estimated trend points; ``reliability`` says whether trend-following is
the right thing to be doing at all, and that question is answered by the
variance-ratio statistic and the regime posterior rather than by the slope.

A steep slope in a mean-reverting regime is not a weak buy signal. It is a strong
signal that should not be acted on, and encoding that as a small score would let
several such engines sum into a large one.
"""

from __future__ import annotations

import math

from ..config import Config
from ..types import BarContext, EngineOutput, Regime, clip
from .base import BaseEngine, squash

__all__ = ["TrendEngine"]


class TrendEngine(BaseEngine):
    name = "trend"

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__(cfg)
        self.warmup = 20
        self.regime_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    def _compute(self, ctx: BarContext) -> EngineOutput:
        f = ctx.features
        p_trend = self.regime_probs[int(Regime.TREND)]
        p_revert = self.regime_probs[int(Regime.REVERT)]

        parts: list[tuple[float, float]] = []
        if f.kalman_slope_t == f.kalman_slope_t:
            # Scale 2.0: a two-sigma filtered slope maps to 0.76, which is a
            # strong but not saturated opinion. Saturation matters because the
            # fusion layer inverts tanh.
            parts.append((squash(f.kalman_slope_t, 2.0), 0.45))
        if f.trend_agree == f.trend_agree:
            parts.append((f.trend_agree, 0.25))
        if f.htf_slope_t == f.htf_slope_t:
            parts.append((squash(f.htf_slope_t, 2.0), 0.30))
        if not parts:
            return self.abstain("no trend estimate available")

        wsum = sum(w for _, w in parts)
        score = sum(v * w for v, w in parts) / wsum

        # Persistence factor from the Lo-MacKinlay statistic. At z = 0 (the
        # martingale null) this is 0.5: we neither trust nor distrust the slope.
        # At z = +2 it is 1.0, at z <= -2 it is 0.
        pers = 0.5
        if f.vr_z == f.vr_z:
            pers = clip(0.5 + f.vr_z / 4.0, 0.0, 1.0)

        # Efficiency corroborates: a slope achieved by a direct path is more
        # believable than the same slope achieved by a wandering one.
        eff = f.efficiency if f.efficiency == f.efficiency else 0.3

        agreement = 1.0
        if f.htf_slope_t == f.htf_slope_t and f.kalman_slope_t == f.kalman_slope_t:
            same = (f.htf_slope_t >= 0.0) == (f.kalman_slope_t >= 0.0)
            # Timeframe disagreement is not a reason to trade the other way; it
            # is a reason to be less sure.
            agreement = 1.0 if same else 0.45

        reliability = (
            clip(0.25 + 0.75 * p_trend, 0.0, 1.0)
            * clip(0.15 + 0.85 * pers, 0.0, 1.0)
            * clip(0.45 + 0.55 * eff, 0.0, 1.0)
            * agreement
            * (1.0 - 0.6 * p_revert)
        )

        notes = []
        if f.vr_z == f.vr_z and abs(f.vr_z) > 2.0:
            notes.append(
                f"increments {'persistent' if f.vr_z > 0 else 'mean-reverting'} (z={f.vr_z:+.1f})"
            )
        if agreement < 1.0:
            notes.append("execution and higher timeframe disagree")

        return self.emit(
            score,
            reliability,
            features={
                "slope_t": f.kalman_slope_t,
                "htf_slope_t": f.htf_slope_t,
                "vr_z": f.vr_z,
                "efficiency": eff,
                "agreement": f.trend_agree,
            },
            diagnostics={"persistence": pers, "tf_agreement": agreement},
            notes=notes,
        )
