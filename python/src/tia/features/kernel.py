"""The feature kernel: one streaming object producing one FeatureSnapshot per bar.

Every engine reads from the snapshot; no engine computes its own scales. That
centralisation is the point. If each engine estimated its own volatility, the
system would hold several mutually inconsistent opinions about how large a move
is, and the fusion layer would silently treat that inconsistency as independent
evidence.

The kernel owns the higher-timeframe views as well. They are built by *causal
aggregation* (:class:`tia.features.structure.HTFAggregator`) rather than by
resampling a history, so a higher-timeframe reading changes only when a
higher-timeframe bar actually completes -- exactly what a live system sees, and
the reason the multi-timeframe layer here cannot repaint.
"""

from __future__ import annotations

import math

from ..config import Config
from ..data.sessions import DAILY, SessionSpec, SessionTagger
from ..types import Bar, FeatureSnapshot, safe_div
from .entropy import PermutationEntropy, RunAsymmetry
from .jumps import JumpTest
from .microstructure import MicrostructureKernel
from .rolling import CausalRank, RollingMoments
from .scales import VolatilityKernel
from .structure import HTFAggregator, PivotTracker, SweepDetector
from .trend import KalmanTrend, MultiHorizonAgreement, PathEfficiency, VarianceRatio

__all__ = ["FeatureKernel", "HTFView"]


class HTFView:
    """A causal higher-timeframe view: aggregate, then run trend and structure.

    Only completed higher-timeframe bars update the estimators. Between
    completions the readings are held, which is correct: on a five-times
    aggregation, four out of five execution bars genuinely carry no new
    higher-timeframe information, and pretending otherwise is how multi-timeframe
    indicators come to repaint.
    """

    def __init__(self, multiple: int, cfg: Config) -> None:
        self.multiple = int(multiple)
        self._agg = HTFAggregator(multiple)
        self._vol = VolatilityKernel(
            window=max(8, cfg.vol_window // 2),
            window_short=max(3, cfg.vol_window_short),
            window_long=max(16, cfg.vol_window // 2 * 3),
            ewma_halflife=max(3.0, cfg.ewma_halflife / 2.0),
            har_weights=cfg.har_weights,
            rank_window=max(60, cfg.rank_window // multiple),
            rank_min_obs=max(20, cfg.rank_min_obs // 2),
        )
        self._kal = KalmanTrend(cfg.kalman_snr, cfg.kalman_slope_snr)
        self._piv = PivotTracker(cfg.pivot_atr_mult, cfg.pivot_confirm_bars)
        self.slope_t = math.nan
        self.range_pos = math.nan
        self.structure_bias = math.nan
        self.n_bars = 0

    def reset(self) -> None:
        self._agg.reset()
        self._vol.reset()
        self._kal.reset()
        self._piv.reset()
        self.slope_t = math.nan
        self.range_pos = math.nan
        self.structure_bias = math.nan
        self.n_bars = 0

    def update(self, bar: Bar) -> bool:
        htf = self._agg.update(bar)
        if htf is None:
            return False
        v = self._vol.update(htf)
        sigma = v.sigma_bp if v.sigma_bp == v.sigma_bp and v.sigma_bp > 0.0 else math.nan
        if sigma == sigma:
            self._kal.update(math.log(htf.close), sigma)
            self._piv.update(htf, sigma)
            self.slope_t = self._kal.slope_t
            self.range_pos = self._piv.range_position(htf.close)
            self.structure_bias = self._piv.structure_bias()
        self.n_bars += 1
        return True


class FeatureKernel:
    """Streaming feature computation for one instrument."""

    def __init__(self, cfg: Config | None = None, session: SessionSpec = DAILY) -> None:
        self.cfg = cfg or Config()
        c = self.cfg

        self.sessions = SessionTagger(session, c.tod_buckets)
        self.vol = VolatilityKernel(
            window=c.vol_window,
            window_short=c.vol_window_short,
            window_long=c.vol_window_long,
            ewma_halflife=c.ewma_halflife,
            har_weights=c.har_weights,
            rank_window=c.rank_window,
            rank_min_obs=c.rank_min_obs,
        )
        self.micro = MicrostructureKernel(
            impact_exponent=c.impact_exponent,
            participation_window=c.participation_window,
            tod_buckets=c.tod_buckets,
            rank_window=c.rank_window,
            rank_min_obs=c.rank_min_obs,
            spread_window=c.spread_estimator_window,
        )
        self.jumps = JumpTest(c.vol_window)
        self.kalman = KalmanTrend(c.kalman_snr, c.kalman_slope_snr)
        self.vratio = VarianceRatio(c.vr_window, c.vr_lags)
        self.efficiency = PathEfficiency(c.vol_window)
        self.agreement = MultiHorizonAgreement()
        self.entropy = PermutationEntropy(c.perm_entropy_order, c.perm_entropy_window)
        self.runs = RunAsymmetry()
        self.pivots = PivotTracker(c.pivot_atr_mult, c.pivot_confirm_bars)
        self.sweeps = SweepDetector()
        self.htf1 = HTFView(c.htf_multiple, c)
        self.htf2 = HTFView(c.htf_multiple_2, c)
        self._ret_moments = RollingMoments(c.rank_window // 2)
        self._vr_rank = CausalRank(c.rank_window, c.rank_min_obs)

        self._prev_close = math.nan
        self._i = -1
        self.snapshot = FeatureSnapshot(index=-1, timestamp=0.0, close=math.nan)

    # ------------------------------------------------------------------ #
    @property
    def warmup(self) -> int:
        """Bars before the snapshot is marked valid.

        Dominated by the rank window's minimum observation count, because a rank
        computed from thirty observations is not a rank -- it is a coarse
        histogram bucket with an unstable boundary.
        """
        return max(
            self.vol.warmup,
            self.micro.warmup,
            self.cfg.rank_min_obs + 10,
            self.cfg.vr_window // 2,
            self.cfg.htf_multiple_2 * 4,
        )

    def reset(self) -> None:
        for obj in (
            self.sessions, self.vol, self.micro, self.jumps, self.kalman,
            self.vratio, self.efficiency, self.agreement, self.entropy,
            self.runs, self.pivots, self.sweeps, self.htf1, self.htf2,
            self._ret_moments, self._vr_rank,
        ):
            obj.reset()
        self._prev_close = math.nan
        self._i = -1
        self.snapshot = FeatureSnapshot(index=-1, timestamp=0.0, close=math.nan)

    # ------------------------------------------------------------------ #
    def update(self, bar: Bar) -> FeatureSnapshot:
        self._i += 1
        i = self._i
        f = FeatureSnapshot(index=i, timestamp=bar.timestamp, close=bar.close)

        sess = self.sessions.update(bar)
        f.phase = sess.phase
        f.tod_frac = sess.tod_frac
        f.dow = sess.dow
        f.overnight_ret = sess.overnight_ret
        f.bars_since_session_open = sess.bars_since_open

        ret = math.nan
        if self._prev_close == self._prev_close and self._prev_close > 0.0:
            ret = math.log(bar.close / self._prev_close)
        self._prev_close = bar.close
        f.ret = ret

        v = self.vol.update(bar)
        f.sigma_bp = v.sigma_bp
        f.sigma_rv = v.sigma_rv
        f.sigma_rs = v.sigma_rs
        f.sigma_ew = v.sigma_ew
        f.sigma_fcst = v.sigma_fcst
        f.jump_share = v.jump_share
        f.vol_of_vol = v.vol_of_vol
        f.vol_rank = v.vol_rank
        f.vol_ratio = v.vol_ratio
        f.compression = v.compression
        f.range_over_sigma = v.range_over_sigma

        sigma = v.sigma_bp if (v.sigma_bp == v.sigma_bp and v.sigma_bp > 0.0) else math.nan
        m = self.micro.update(bar, ret, sigma, sess.tod_bucket)
        f.ret_norm = m.ret_norm
        f.participation = m.participation
        f.ladr = m.ladr
        f.ladr_rank = m.ladr_rank
        f.absorption = m.absorption
        f.absorption_rank = m.absorption_rank
        f.amihud = m.amihud
        f.kyle_lambda = m.kyle_lambda
        f.clv_ema = m.clv_ema
        f.delta_norm = m.delta_norm

        if ret == ret:
            self.jumps.update(ret)
            self.vratio.update(ret)
            self.efficiency.update(ret)
            self.runs.update(ret)
            self._ret_moments.update(ret)

        self.entropy.update(bar.close)
        self.agreement.update(math.log(bar.close))
        if sigma == sigma:
            self.kalman.update(math.log(bar.close), sigma)

        f.kalman_level = self.kalman.level
        f.kalman_slope = self.kalman.slope
        f.kalman_slope_t = self.kalman.slope_t
        f.vr_z = self.vratio.z_mean
        f.hurst_implied = self.vratio.hurst_implied
        f.efficiency = self.efficiency.value
        f.trend_agree = self.agreement.value
        f.perm_entropy = self.entropy.value
        f.run_asym = self.runs.value
        f.ret_kurtosis = self._ret_moments.kurtosis if len(self._ret_moments) > 30 else math.nan

        if sigma == sigma:
            self.pivots.update(bar, sigma)
        f.pivot_high = self.pivots.high_price()
        f.pivot_low = self.pivots.low_price()
        f.pivot_high_age = self.pivots.high_age()
        f.pivot_low_age = self.pivots.low_age()
        f.range_pos = self.pivots.range_position(bar.close)

        ev = self.sweeps.update(
            bar, sigma, f.pivot_high, f.pivot_low, m.participation_rank, m.absorption_rank
        )
        if ev is not None:
            if ev.direction > 0:
                f.swept_low = 1.0
            else:
                f.swept_high = 1.0

        self.htf1.update(bar)
        self.htf2.update(bar)
        f.htf_slope_t = self.htf1.slope_t
        f.htf_range_pos = self.htf1.range_pos

        need = self.warmup
        f.warmup_remaining = max(0, need - (i + 1))
        f.valid = (
            f.warmup_remaining == 0
            and v.valid
            and m.valid
            and f.sigma_fcst == f.sigma_fcst
            and f.vol_rank == f.vol_rank
            and f.ladr_rank == f.ladr_rank
        )
        self.snapshot = f
        return f

    # ------------------------------------------------------------------ #
    def regime_features(self) -> dict[str, float]:
        """The eight rank features consumed by the regime measurement model.

        All are in ``[0, 1]`` and all are causal. ``vr_rank`` converts the
        variance-ratio z-statistic to a rank rather than using it raw, because
        the Beta measurement model of ``docs/01-THEORY.md`` §5.2 requires
        arguments on the unit interval; the raw statistic is retained separately
        for the trend engine, which wants its magnitude.
        """
        f = self.snapshot
        vr_rank = self._vr_rank.update(f.vr_z) if f.vr_z == f.vr_z else math.nan
        return {
            "vol_rank": f.vol_rank,
            "vr_rank": vr_rank,
            "ladr_rank": f.ladr_rank,
            "absorption_rank": f.absorption_rank,
            "participation_rank": self.micro.out.participation_rank,
            "entropy": f.perm_entropy,
            "jump_share": f.jump_share,
            "efficiency": f.efficiency,
        }
