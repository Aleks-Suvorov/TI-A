"""Structure engine: confirmed swing geometry only.

Every level this engine reads was confirmed at a bar that has already closed, so
nothing here can repaint. The cost is that its view of structure is several bars
stale, which is correct rather than unfortunate -- the alternative is a level that
was invisible at the time it supposedly mattered.

The engine's reading is regime-dependent in a way that follows directly from
theory §0.1: the same location inside a range is bullish in one regime and
bearish in the other. Near the top of a confirmed range, a trending market is
breaking out and a mean-reverting one is failing. Rather than pick, the engine
scores both and weights by the regime posterior, which is what the posterior is
for.
"""

from __future__ import annotations

import math

from ..config import Config
from ..types import BarContext, EngineOutput, Regime, clip
from .base import BaseEngine, squash

__all__ = ["StructureEngine"]


class StructureEngine(BaseEngine):
    name = "structure"

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__(cfg)
        self.warmup = 40
        self.regime_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
        self.structure_bias: float = math.nan

    def _compute(self, ctx: BarContext) -> EngineOutput:
        f = ctx.features
        if f.range_pos != f.range_pos:
            return self.abstain("no confirmed range")

        p_trend = self.regime_probs[int(Regime.TREND)]
        p_revert = self.regime_probs[int(Regime.REVERT)]
        notes: list[str] = []

        # Position inside the confirmed range, centred: -1 at the low, +1 at
        # the high.
        pos = 2.0 * f.range_pos - 1.0

        # Continuation reading: high in the range is strength.
        continuation = clip(pos, -1.0, 1.0)
        # Reversion reading: high in the range is where supply lives.
        reversion = -continuation

        w_t = p_trend + 1e-9
        w_r = p_revert + 1e-9
        score = (w_t * continuation + w_r * reversion) / (w_t + w_r)

        # Higher-timeframe location is a slow, independent corroboration.
        if f.htf_range_pos == f.htf_range_pos:
            htf_pos = 2.0 * f.htf_range_pos - 1.0
            htf_read = (w_t * htf_pos - w_r * htf_pos) / (w_t + w_r)
            score = 0.7 * score + 0.3 * clip(htf_read, -1.0, 1.0)

        # Confirmed higher-highs / higher-lows, when available.
        if self.structure_bias == self.structure_bias:
            score = 0.75 * score + 0.25 * self.structure_bias
            if abs(self.structure_bias) > 0.9:
                notes.append(
                    "higher highs and higher lows" if self.structure_bias > 0 else
                    "lower highs and lower lows"
                )

        # Staleness: a range whose extremes were confirmed long ago is a weaker
        # description of the present than a fresh one.
        ages = [a for a in (f.pivot_high_age, f.pivot_low_age) if a == a]
        staleness = 1.0
        if ages:
            oldest = max(ages)
            staleness = clip(1.0 - (oldest - 20.0) / 120.0, 0.25, 1.0)
            if oldest > 80:
                notes.append("confirmed range is stale")

        # Ambiguity between the two readings is itself informative: when the
        # regime posterior is split, structure genuinely does not say anything.
        decisiveness = abs(p_trend - p_revert) / max(p_trend + p_revert, 1e-9)
        reliability = (
            clip(0.25 + 0.75 * abs(pos), 0.0, 1.0)
            * clip(0.35 + 0.65 * decisiveness, 0.0, 1.0)
            * staleness
        )

        return self.emit(
            score,
            reliability,
            features={
                "range_pos": f.range_pos,
                "htf_range_pos": f.htf_range_pos,
                "pivot_high": f.pivot_high,
                "pivot_low": f.pivot_low,
                "structure_bias": self.structure_bias,
            },
            diagnostics={
                "continuation": continuation,
                "decisiveness": decisiveness,
                "staleness": staleness,
            },
            notes=notes,
        )
