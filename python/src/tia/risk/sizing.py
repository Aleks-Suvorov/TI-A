"""Position sizing: fractional Kelly with an explicit uncertainty penalty.

``docs/01-THEORY.md`` §9.1. The growth-optimal fraction for approximately
Gaussian outcomes is :math:`\\mu/\\varsigma^2`, but :math:`\\mu` is estimated and
plugging in the point estimate overbets systematically. Integrating over the
posterior of :math:`\\mu` gives, to second order,

.. math:: f^\\star_{\\text{Bayes}} \\approx \\frac{\\mu_n}{\\varsigma_n^2 + \\mathrm{Var}(\\mu_n)}

so parameter uncertainty enters as additional variance. That is a clean and
useful statement: an edge you are unsure about deserves a smaller bet for exactly
the same reason a volatile edge does.

Three further reductions are applied on top, and the stacking is deliberate:

*   a fractional-Kelly multiplier of 0.25, the conventional compromise between
    growth rate and the fact that full Kelly's drawdowns are intolerable for
    anyone who has to keep trading through them;
*   a haircut by the ratio of the lower credible bound to the posterior mean, so
    that a wide posterior shrinks the bet even when its centre is attractive;
*   the drawdown throttle from :mod:`tia.risk.limits`.

Full Kelly on an estimated edge is not aggressive, it is a category error: the
Kelly criterion is optimal for a *known* distribution, and betting it on an
estimated one is reliably worse than betting a fraction of it.
"""

from __future__ import annotations

import math

from ..config import Config
from ..engines.edgebook import EdgeEstimate
from ..types import CostEstimate, RiskDecision, TargetSpec, clip, safe_div

__all__ = ["Sizer"]


class Sizer:
    """Converts an expectancy posterior into a risk fraction and a unit count."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()

    def size(
        self,
        edge: EdgeEstimate,
        spec: TargetSpec,
        costs: CostEstimate,
        equity: float,
        price: float,
        drawdown_throttle: float = 1.0,
        extra_throttles: dict[str, float] | None = None,
        blocks: tuple[str, ...] = (),
    ) -> RiskDecision:
        cfg = self.cfg
        throttles: dict[str, float] = dict(extra_throttles or {})

        net_mean = edge.mean - costs.round_trip_sigma
        net_lcb = edge.lcb - costs.round_trip_sigma
        if blocks or net_lcb <= 0.0 or not (spec.stop_sigma > 0.0):
            return RiskDecision(
                risk_fraction=0.0,
                units=0.0,
                throttles=throttles,
                blocks=blocks or ("expectancy lower bound not positive",),
            )

        # Bayesian Kelly in sigma units. Parameter uncertainty is added to the
        # outcome variance rather than ignored.
        denom = edge.outcome_var + (edge.var_mean if math.isfinite(edge.var_mean) else 1e6)
        kelly = safe_div(max(net_mean, 0.0), denom, 0.0)
        throttles["kelly_raw"] = kelly

        # Uncertainty haircut: the ratio of what we can defend to what we
        # believe. At a wide posterior this is near zero even when the mean is
        # attractive.
        haircut = clip(safe_div(net_lcb, max(net_mean, 1e-9), 0.0), 0.0, 1.0)
        throttles["uncertainty_haircut"] = haircut

        # Thin cells are shrunk further. The Edge Book's prior already pulls them
        # toward zero expectancy; this makes the sizing consequence explicit
        # rather than relying on the prior alone.
        data_factor = clip(edge.n_eff / 30.0, 0.15, 1.0)
        throttles["data_sufficiency"] = data_factor

        throttles["drawdown"] = drawdown_throttle
        throttles["kelly_fraction"] = cfg.kelly_fraction

        frac = kelly * cfg.kelly_fraction * haircut * data_factor * drawdown_throttle
        for k, v in (extra_throttles or {}).items():
            frac *= v

        # The baseline risk_per_trade is a *scale*, not a target: the Kelly term
        # decides how much of it to use. Anchoring to it keeps position sizes in
        # a range the operator recognises, which matters more for adherence than
        # it does for growth.
        frac = frac * (cfg.risk_per_trade / 0.005)
        frac = clip(frac, 0.0, cfg.max_risk_per_trade)

        # Convert a risk fraction to units via the stop distance. The stop is in
        # sigma; the price distance is what the position must be sized against.
        stop_dist = abs(price * (1.0 - math.exp(-spec.stop_sigma * spec.sigma)))
        units = safe_div(frac * equity, stop_dist, 0.0)

        return RiskDecision(
            risk_fraction=float(frac),
            units=float(max(units, 0.0)),
            throttles=throttles,
            blocks=(),
        )

    def volatility_target_scalar(self, sigma_per_bar: float, bars_per_year: float) -> float:
        """Multiplier that brings realised volatility toward the annual target.

        Reported separately from the Kelly term because they answer different
        questions: Kelly asks how much of the edge to press, volatility targeting
        asks how much risk the book should carry regardless of the edge. In a
        single-instrument deployment the two are combined; in a portfolio the
        second belongs at the portfolio level.
        """
        if not (sigma_per_bar > 0.0) or bars_per_year <= 0.0:
            return 1.0
        ann = sigma_per_bar * math.sqrt(bars_per_year)
        return float(clip(self.cfg.vol_target_ann / ann, 0.1, 3.0))
