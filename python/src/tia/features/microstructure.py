"""The microstructure primitive: displacement per unit of consumed liquidity.

``docs/01-THEORY.md`` §3. This module answers one question from OHLCV data alone:
*did this move cost a lot of liquidity, or a little?* Kyle (1985) says the answer
discriminates informed flow, whose impact is permanent, from uninformed pressure,
whose impact is transient. That is the same distinction practitioners reach for
when they talk about "effort versus result", and it is the only version of that
idea which is measurable.

The two outputs are

.. math::
    \\mathrm{LADR}_t = \\mathrm{sign}(r_t)\\,\\frac{|r_t/\\sigma^{bp}_t|}{p_t^{\\psi}},
    \\qquad
    \\mathrm{ABS}_t = \\frac{p_t}{|r_t/\\sigma^{bp}_t| + \\epsilon}

with :math:`p_t` the bar's dollar volume divided by the median for its own
time-of-day bucket, and :math:`\\psi=1/2` the square-root impact law.

Both are consumed as causal ranks so that their marginal distributions are
identical on every instrument in every era.

**Stated limitation.** :math:`p_t` is *total* volume, not signed order flow.
Without a footprint feed we cannot separate buy- from sell-initiated volume, so
LADR is a confirming measurement rather than an independent alpha source, and it
misreads in situations where volume and consumed liquidity decouple: cross-venue
sweeps, auction prints, index rebalances.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import Bar, safe_div
from .rolling import EWMA, BucketedMedian, CausalRank, RollingQuantile, RollingSum

__all__ = ["MicrostructureKernel", "MicroReadout", "CorwinSchultzSpread"]

_EPS = 1e-6
#: ``3 - 2*sqrt(2)``, the Corwin-Schultz denominator constant.
_CS_K = 3.0 - 2.0 * math.sqrt(2.0)


class CorwinSchultzSpread:
    """Effective spread from high/low ranges, with a second estimator to bound it.

    **Corwin & Schultz (2012).** A bar's high is more likely buyer-initiated and
    its low seller-initiated, so the observed one-bar and two-bar ranges
    decompose into volatility plus twice the half-spread. Solving gives an
    estimator needing no quote data at all -- which is what makes cost estimation
    possible in a system restricted to OHLCV.

    **Abdi & Ranaldo (2017).** A different route to the same quantity: the close
    should sit inside the range, and the covariance between the close's deviation
    from the mid-range on consecutive bars is driven by the spread.

    Both are used because both are *upward* biased in practice -- the
    non-negativity truncation guarantees it -- and their biases have different
    sources. Taking the smaller of the two robust aggregates is therefore closer
    to the truth than either alone, and this is the choice made here. It is
    validated in ``tests/test_features.py`` against a known injected spread.

    Aggregation is by rolling **median**, not mean. The per-pair estimates have a
    heavy right tail, and a mean over 22 of them is dominated by two or three
    outliers; on a validation series with a known 8 bp spread, mean aggregation
    over-estimated by roughly 7x where the median did not.

    **Causality.** Both estimators use the bar pair :math:`(t-1, t)`, never
    :math:`(t, t+1)`. Written the natural way they are trivially forward-looking,
    and published implementations frequently are.
    """

    __slots__ = ("_cs", "_ar", "_prev", "_out", "_out_cs", "_out_ar")

    def __init__(self, window: int = 22) -> None:
        self._cs = RollingQuantile(int(window))
        self._ar = RollingQuantile(int(window))
        self._prev: Bar | None = None
        self._out = math.nan
        self._out_cs = math.nan
        self._out_ar = math.nan

    def reset(self) -> None:
        self._cs.reset()
        self._ar.reset()
        self._prev = None
        self._out = self._out_cs = self._out_ar = math.nan

    def update(self, bar: Bar) -> float:
        prev = self._prev
        self._prev = bar
        if prev is None:
            return self._out

        try:
            hl_prev = math.log(prev.high / prev.low)
            hl_now = math.log(bar.high / bar.low)
            h2 = max(prev.high, bar.high)
            l2 = min(prev.low, bar.low)
            hl_two = math.log(h2 / l2)
        except (ValueError, ZeroDivisionError):  # pragma: no cover - guarded by Bar
            return self._out

        beta = hl_prev * hl_prev + hl_now * hl_now
        gamma = hl_two * hl_two
        alpha = (
            (math.sqrt(2.0 * beta) - math.sqrt(beta)) / _CS_K
            - math.sqrt(max(gamma, 0.0) / _CS_K)
        )
        s_cs = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
        self._cs.update(max(s_cs, 0.0))

        # Abdi-Ranaldo: 2 * sqrt(max(0, (c_{t-1} - eta_{t-1})(c_{t-1} - eta_t)))
        # with eta the log mid-range. Uses only bars t-1 and t.
        eta_prev = 0.5 * (math.log(prev.high) + math.log(prev.low))
        eta_now = 0.5 * (math.log(bar.high) + math.log(bar.low))
        c_prev = math.log(prev.close)
        cov = (c_prev - eta_prev) * (c_prev - eta_now)
        self._ar.update(2.0 * math.sqrt(cov) if cov > 0.0 else 0.0)

        if len(self._cs) >= 5:
            self._out_cs = self._cs.median
            self._out_ar = self._ar.median
            cands = [v for v in (self._out_cs, self._out_ar) if v == v and v > 0.0]
            self._out = min(cands) if cands else 0.0
        return self._out

    @property
    def value(self) -> float:
        """Relative effective spread, e.g. 0.0004 for four basis points."""
        return self._out

    @property
    def half_spread(self) -> float:
        return 0.5 * self._out if self._out == self._out else math.nan

    @property
    def components(self) -> tuple[float, float]:
        """The two underlying estimates, for the monitoring layer."""
        return self._out_cs, self._out_ar


class MicroReadout:
    __slots__ = (
        "participation",
        "participation_rank",
        "ladr",
        "ladr_abs",
        "ladr_rank",
        "absorption",
        "absorption_rank",
        "amihud",
        "amihud_rank",
        "kyle_lambda",
        "clv",
        "clv_ema",
        "delta_norm",
        "spread",
        "ret_norm",
        "valid",
    )

    def __init__(self) -> None:
        for s in self.__slots__:
            setattr(self, s, math.nan)
        self.valid = False


class MicrostructureKernel:
    """Streaming computation of the LADR/absorption pair and friends."""

    def __init__(
        self,
        impact_exponent: float = 0.5,
        participation_window: int = 60,
        tod_buckets: int = 13,
        rank_window: int = 252,
        rank_min_obs: int = 60,
        spread_window: int = 22,
        clv_halflife: float = 5.0,
    ) -> None:
        self.psi = float(impact_exponent)
        self._tod_vol = BucketedMedian(int(tod_buckets), int(participation_window))
        self._part_rank = CausalRank(rank_window, rank_min_obs)
        self._ladr_rank = CausalRank(rank_window, rank_min_obs)
        self._abs_rank = CausalRank(rank_window, rank_min_obs)
        self._amihud_rank = CausalRank(rank_window, rank_min_obs)
        self._clv_ema = EWMA(clv_halflife)
        self._spread = CorwinSchultzSpread(spread_window)
        self._delta_scale = RollingQuantile(int(participation_window))
        self._count = 0
        self.out = MicroReadout()

    @property
    def warmup(self) -> int:
        return 70

    def reset(self) -> None:
        self._tod_vol.reset()
        for r in (self._part_rank, self._ladr_rank, self._abs_rank, self._amihud_rank):
            r.reset()
        self._clv_ema.reset()
        self._spread.reset()
        self._delta_scale.reset()
        self._count = 0
        self.out = MicroReadout()

    # ------------------------------------------------------------------ #
    def update(self, bar: Bar, ret: float, sigma_bp: float, tod_bucket: int) -> MicroReadout:
        o = self.out = MicroReadout()
        self._count += 1

        o.clv = bar.clv
        o.clv_ema = self._clv_ema.update(bar.clv)
        o.spread = self._spread.update(bar)

        dv = bar.dollar_volume
        # Update the time-of-day reference *before* normalising, so the current
        # bar is included in its own reference class -- consistent with the rank
        # convention and avoiding a one-bar asymmetry between the two.
        self._tod_vol.update(tod_bucket, dv)
        part = self._tod_vol.normalise(tod_bucket, dv, fallback=math.nan)
        o.participation = part
        if part == part and part > 0.0:
            o.participation_rank = self._part_rank.update(part)

        if ret != ret or not (sigma_bp > 0.0):
            return o

        d = ret / sigma_bp
        o.ret_norm = d
        ad = abs(d)

        if part == part and part > 0.0:
            # Square-root impact law: displacement achieved per unit of
            # sqrt(consumed liquidity). Large means the move was cheap in
            # liquidity terms, which under Kyle is the signature of information.
            denom = part**self.psi
            ladr = safe_div(ad, denom, math.nan)
            if ladr == ladr:
                o.ladr_abs = ladr
                o.ladr = math.copysign(ladr, ret) if ret != 0.0 else 0.0
                o.ladr_rank = self._ladr_rank.update(ladr)

            # Absorption: volume transacted without price consequence. This is
            # the measurable core of what is usually called accumulation or
            # distribution -- an observable statement about volume and price, not
            # an unobservable claim about intent.
            absorb = safe_div(part, ad + _EPS, math.nan)
            if absorb == absorb:
                o.absorption = absorb
                o.absorption_rank = self._abs_rank.update(math.log1p(absorb))

        if dv > 0.0:
            o.amihud = abs(ret) / dv
            o.amihud_rank = self._amihud_rank.update(math.log(o.amihud + 1e-300))
            # Regression-free Kyle lambda: price move per square root of traded
            # value, the same functional form as the impact law above but in
            # price rather than sigma units, so it is comparable through time on
            # one instrument even when sigma shifts.
            o.kyle_lambda = abs(ret) / math.sqrt(dv)

        if bar.delta is not None and bar.volume > 0.0:
            # Signed volume, scaled by its own robust dispersion so that the
            # feature is comparable across instruments with different footprint
            # conventions.
            raw = bar.delta / bar.volume
            self._delta_scale.update(abs(raw))
            mad = self._delta_scale.mad()
            o.delta_norm = raw / mad if (mad == mad and mad > 1e-12) else raw

        o.valid = self._count >= self.warmup and o.ladr_rank == o.ladr_rank
        return o
