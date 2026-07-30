"""Liquidity engine: sweeps that fail, and volume that goes nowhere.

Condition (B) of ``docs/01-THEORY.md`` §0.1. Two distinct measurements, both from
the microstructure primitive:

**Sweep-and-fail.** Price penetrates a confirmed level on anomalous volume and
then fails to hold it. This is the measurable core of what practitioners call a
stop run, and its economic content is compensated liquidity provision: someone
had to be forced out, and someone was paid to take the other side. The engine
scores in the *reclaim* direction, and only after the reclaim is confirmed on a
closed bar -- which is late, and is the actual information arrival time.

**Absorption.** Heavy participation producing little displacement, at a
structural level. This is "effort versus result" made computable. It is a
statement about volume and price, not a claim about anyone's intent, and it is
scored as evidence that the level will hold rather than as evidence of a
particular actor.

The rest of the Smart Money vocabulary is discarded; see
``docs/17-EVIDENCE-REVIEW.md``.
"""

from __future__ import annotations

import math

from ..config import Config
from ..types import BarContext, EngineOutput, Regime, clip
from .base import BaseEngine, squash

__all__ = ["LiquidityEngine"]


class LiquidityEngine(BaseEngine):
    name = "liquidity"

    #: Bars after a confirmed sweep during which it still counts as evidence.
    #: Short, because the reversal effect it exploits decays fast -- carrying it
    #: longer would be claiming information we do not have.
    SWEEP_LIFETIME = 4

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__(cfg)
        self.warmup = 30
        self.regime_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
        self._sweep_dir = 0
        self._sweep_age = 10**6
        self._sweep_strength = 0.0

    def _reset(self) -> None:
        self._sweep_dir = 0
        self._sweep_age = 10**6
        self._sweep_strength = 0.0

    def _compute(self, ctx: BarContext) -> EngineOutput:
        f = ctx.features
        p_revert = self.regime_probs[int(Regime.REVERT)]
        p_stress = self.regime_probs[int(Regime.STRESS)]
        p_trend = self.regime_probs[int(Regime.TREND)]
        notes: list[str] = []

        self._sweep_age += 1
        if f.swept_low > 0.0 or f.swept_high > 0.0:
            self._sweep_dir = 1 if f.swept_low > 0.0 else -1
            self._sweep_age = 0
            part = f.participation if f.participation == f.participation else 1.0
            self._sweep_strength = clip(0.45 + 0.25 * math.log1p(max(part - 1.0, 0.0)), 0.0, 1.0)
            notes.append(
                f"swept and reclaimed a confirmed {'low' if self._sweep_dir > 0 else 'high'}"
            )

        sweep_score = 0.0
        if self._sweep_age <= self.SWEEP_LIFETIME and self._sweep_dir != 0:
            decay = 1.0 - self._sweep_age / (self.SWEEP_LIFETIME + 1.0)
            sweep_score = self._sweep_dir * self._sweep_strength * decay

        # Absorption near a range extreme: heavy volume that failed to move
        # price, where price is at a level that matters. Scores toward the middle
        # of the range.
        absorb_score = 0.0
        absorbing = False
        if f.absorption_rank == f.absorption_rank and f.range_pos == f.range_pos:
            edge = abs(f.range_pos - 0.5) * 2.0  # 0 mid-range, 1 at an extreme
            if f.absorption_rank > 0.65 and edge > 0.5:
                absorbing = True
                mag = (f.absorption_rank - 0.65) / 0.35 * edge
                absorb_score = -math.copysign(clip(mag, 0.0, 1.0), f.range_pos - 0.5)
                notes.append("volume absorbed at range extreme")

        if sweep_score != 0.0 and absorb_score != 0.0:
            score = 0.7 * sweep_score + 0.3 * absorb_score
        else:
            score = sweep_score + absorb_score

        has_evidence = sweep_score != 0.0 or absorbing
        if not has_evidence:
            # No liquidity event is a confident statement of no evidence, not an
            # absence of opinion. Reported with moderate reliability so it
            # correctly dampens the pooled confidence rather than vanishing.
            return self.emit(
                0.0,
                0.35,
                features={"absorption_rank": f.absorption_rank, "range_pos": f.range_pos},
                diagnostics={"sweep_age": float(min(self._sweep_age, 999))},
                notes=("no liquidity event",),
            )

        reliability = (
            clip(0.30 + 0.70 * abs(score), 0.0, 1.0)
            * clip(0.40 + 0.60 * (p_revert + 0.6 * p_stress), 0.0, 1.0)
            * (1.0 - 0.35 * p_trend)
        )
        # In stress the reversal is better paid and far more dangerous. The
        # engine keeps its opinion; the risk layer cuts the size.
        if p_stress > 0.5:
            notes.append("stress regime: reversal edge larger, tails fatter")

        return self.emit(
            score,
            reliability,
            features={
                "absorption_rank": f.absorption_rank,
                "participation": f.participation,
                "range_pos": f.range_pos,
                "sweep_age": float(min(self._sweep_age, 999)),
            },
            diagnostics={
                "sweep_score": sweep_score,
                "absorb_score": absorb_score,
                "sweep_dir": float(self._sweep_dir),
            },
            notes=notes,
        )
