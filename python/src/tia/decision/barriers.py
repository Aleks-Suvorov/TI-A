"""Target construction: barriers in sigma units, converted to price last.

``docs/01-THEORY.md`` §8.1. Setting the stop at "1.6 sigma" and only then turning
that into a price is what allows one parameter set to serve an instrument
trading at 4 and one trading at 40,000, in 1995 and in 2026. A stop set as a
percentage, a tick count, or a fixed ATR multiple with a fitted coefficient does
not travel.

The vertical barrier is not a convenience either. The conditioning information
that justified the trade decays; after ``max_holding_bars`` the position is no
longer the one that was analysed, and continuing to hold it is a different bet
made silently.
"""

from __future__ import annotations

import math

from ..config import Config
from ..types import FeatureSnapshot, Regime, TargetSpec, clip

__all__ = ["BarrierPlanner"]


class BarrierPlanner:
    """Builds a :class:`TargetSpec` from the volatility forecast and the regime."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()

    def plan(
        self,
        direction: int,
        entry_ref: float,
        f: FeatureSnapshot,
        regime: Regime,
        regime_posterior: tuple[float, float, float, float] | None = None,
    ) -> TargetSpec | None:
        cfg = self.cfg
        sigma = f.sigma_fcst
        if not (sigma == sigma and sigma > 0.0) or direction == 0:
            return None

        stop = cfg.stop_sigma
        target = cfg.target_sigma
        holding = cfg.max_holding_bars

        p = regime_posterior or (0.0, 0.0, 0.0, 0.0)
        p_stress = p[int(Regime.STRESS)]
        p_trend = p[int(Regime.TREND)]
        p_revert = p[int(Regime.REVERT)]

        # Stress widens the stop, because a jump-dominated tape produces gaps
        # that a tight stop converts into a guaranteed adverse fill rather than
        # into protection. The widening is paid for by the risk layer sizing
        # down, not by moving the target further away.
        if p_stress > 0.2:
            stop *= 1.0 + 0.6 * p_stress

        # Jump variation has the same implication at a shorter horizon.
        if f.jump_share == f.jump_share and f.jump_share > 0.4:
            stop *= 1.0 + 0.4 * (f.jump_share - 0.4) / 0.6

        # Mean-reverting setups aim at the middle of the range, not beyond it, so
        # their targets are nearer and their holds shorter. Trend setups get the
        # opposite treatment. Both adjustments are bounded and are driven by the
        # posterior rather than by a switch.
        if p_revert > p_trend:
            target *= 1.0 - 0.25 * (p_revert - p_trend)
            holding = int(holding * (1.0 - 0.3 * (p_revert - p_trend)))
        else:
            target *= 1.0 + 0.20 * (p_trend - p_revert)

        # A target inside a confirmed structural level is more plausible than one
        # beyond it. Where the level is close, clip the target to just short of
        # it rather than pretending price will sail through.
        level = f.pivot_high if direction > 0 else f.pivot_low
        if level == level and level > 0.0 and entry_ref > 0.0:
            dist = math.log(level / entry_ref) * direction / sigma
            if 0.3 < dist < target:
                target = max(0.8 * dist, 0.6 * stop)

        stop = clip(stop, 0.4, 8.0)
        target = clip(target, 0.4, 12.0)
        holding = int(clip(float(holding), 3.0, 500.0))

        return TargetSpec(
            direction=int(direction),
            entry_ref=float(entry_ref),
            stop_sigma=float(stop),
            target_sigma=float(target),
            max_holding_bars=holding,
            sigma=float(sigma),
        )

    @staticmethod
    def expectancy_from_probability(p_success: float, spec: TargetSpec) -> float:
        """Naive expectancy in sigma units, assuming a barrier is always reached.

        Used only as a sanity cross-check against the Edge Book, which models the
        realised return distribution directly and therefore also captures
        timeouts. Where the two disagree materially, the Edge Book is right and
        the disagreement is a signal that the barriers are rarely being touched
        -- which is itself worth knowing.
        """
        return p_success * spec.target_sigma - (1.0 - p_success) * spec.stop_sigma
