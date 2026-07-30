"""Transaction costs, estimated rather than assumed.

``docs/01-THEORY.md`` §8.2. Costs are expressed in sigma units so that the trade
gate compares like with like: an expectancy of 0.12 sigma against a round trip of
0.09 sigma is a decision one can actually make, whereas comparing a percentage
expectancy against a basis-point spread requires a conversion that is easy to get
silently wrong.

Every component is either measured from the data or derived from a law with
published support. Nothing here is a plug number chosen to make a backtest work,
which matters because cost assumptions are the single easiest place to
manufacture a strategy: halving the assumed spread turns many losing systems into
winning ones, and the change is invisible in an equity curve.
"""

from __future__ import annotations

import math

from ..config import Config
from ..types import CostEstimate, FeatureSnapshot, safe_div

__all__ = ["CostModel"]


class CostModel:
    """Assembles a :class:`CostEstimate` for a contemplated trade."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()

    def estimate(
        self,
        f: FeatureSnapshot,
        spread_rel: float,
        participation_fraction: float = 0.01,
        stress_multiplier: float = 1.0,
    ) -> CostEstimate:
        """Cost of one round trip, in sigma units.

        ``spread_rel`` is the Corwin-Schultz relative effective spread. When it is
        unavailable the fallback is deliberately *pessimistic* -- a fixed five
        basis points, which is wide for a liquid future and narrow for a small
        cap. An optimistic fallback would let the system trade instruments whose
        costs it cannot measure.

        ``participation_fraction`` is our order size as a fraction of the bar's
        dollar volume. It drives the square-root impact term, and it is the
        channel through which capacity limits enter the decision: at large enough
        size the impact term alone exceeds the expectancy and the gate refuses.
        """
        cfg = self.cfg
        sigma = f.sigma_fcst
        if not (sigma == sigma and sigma > 0.0):
            # Without a volatility forecast there is no sigma-unit accounting to
            # be had, so the cost is reported as prohibitive rather than as zero.
            return CostEstimate(math.inf, math.inf, math.inf, math.inf)

        s = spread_rel if (spread_rel == spread_rel and spread_rel > 0.0) else 5e-4
        half_spread_sigma = safe_div(0.5 * s, sigma, math.inf)
        # Range-based spread estimators cannot resolve a spread far below a bar's
        # own volatility. Validated against a known injected spread, this one
        # tracks the truth down to roughly twenty basis points and over-estimates
        # (conservatively) below that. Above the cap the estimate carries no
        # information about the real cost at all, so the instrument is reported
        # as untradeable rather than merely expensive, and the gate refuses it.
        if half_spread_sigma > cfg.cost_max_half_spread_sigma:
            half_spread_sigma = math.inf

        # Square-root impact law. eta is the one DEV-fitted parameter here,
        # calibrated against observed slippage on the development universe.
        q = max(float(participation_fraction), 0.0)
        impact_sigma = cfg.cost_impact_eta * math.sqrt(q)

        # Adverse-fill assumption: a fraction of the execution bar's *range* is
        # charged as slippage. The range rather than a fixed tick, so the
        # assumption scales with conditions and grows in exactly the
        # circumstances where real slippage grows.
        range_sigma = f.range_over_sigma
        if not (range_sigma == range_sigma and range_sigma > 0.0):
            # A typical bar's range is a little over one forecast sigma. Used
            # only before the range statistic is available.
            range_sigma = 1.3
        slippage_sigma = cfg.cost_slippage_range_frac * min(range_sigma, 5.0)

        fee_sigma = safe_div(cfg.cost_fee_bps * 1e-4, sigma, 0.0)

        # In stress, spreads widen, depth thins and the impact law's coefficient
        # rises. Applying the multiplier to the liquidity-sensitive components
        # only -- fees do not widen in a crisis.
        k = max(1.0, float(stress_multiplier))
        return CostEstimate(
            half_spread_sigma=half_spread_sigma * k,
            impact_sigma=impact_sigma * k,
            slippage_sigma=slippage_sigma * k,
            fee_sigma=fee_sigma,
        )

    def capacity_participation(self, f: FeatureSnapshot, edge_sigma: float) -> float:
        """Largest participation fraction at which the edge still survives impact.

        Solves ``2 * eta * sqrt(q) = edge_sigma`` for ``q``. Beyond this the
        strategy is trading against itself, and the honest response is to refuse
        the size rather than to accept a worse fill -- which is why the policy
        layer treats this as a hard constraint.
        """
        if edge_sigma <= 0.0 or self.cfg.cost_impact_eta <= 0.0:
            return 0.0
        return float((edge_sigma / (2.0 * self.cfg.cost_impact_eta)) ** 2)
