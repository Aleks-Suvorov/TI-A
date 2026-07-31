"""Volatility scale estimation and forecasting.

This is the foundation layer of ``docs/01-THEORY.md`` §2. Every barrier, every
cost and every normalised feature in the system is denominated in the sigma this
module produces, so an error here is not a small mis-sizing -- it simultaneously
corrupts the system's notion of a large move, an expensive trade and an
acceptable loss.

Four estimators are maintained rather than one, because they fail in different
places and their disagreement is itself informative:

*   ``sigma_bp``  bipower variation, consistent for the *continuous* part alone
    and therefore the correct denominator when asking "was this move large?"
*   ``sigma_rv``  plain realised volatility, which includes jumps
*   ``sigma_rs``  Rogers-Satchell range volatility, drift-independent
*   ``sigma_gk``  Garman-Klass range volatility, most efficient under zero drift
*   ``sigma_ew``  EWMA, fastest to react

``sigma_rv`` exceeding ``sigma_bp`` isolates jump variation. Range estimators
exceeding close-to-close estimators indicates intrabar reversal -- the bar's path
mattered more than its endpoints.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import Bar, safe_div
from .rolling import EWMA, CausalRank, RollingMoments, RollingSum

__all__ = ["VolatilityKernel", "VolReadout", "fit_har_weights", "GK_CONST"]

#: Garman-Klass second-term coefficient, ``2 ln 2 - 1``.
GK_CONST = 2.0 * math.log(2.0) - 1.0

#: ``pi/2``, the bipower scaling constant from Barndorff-Nielsen & Shephard.
_BP_SCALE = math.pi / 2.0


class VolReadout:
    """Plain record of one bar's volatility state. Mutable for cheap reuse."""

    __slots__ = (
        "sigma_bp",
        "sigma_rv",
        "sigma_rs",
        "sigma_gk",
        "sigma_ew",
        "sigma_fcst",
        "sigma_short",
        "sigma_long",
        "jump_share",
        "vol_of_vol",
        "vol_rank",
        "vol_ratio",
        "compression",
        "range_over_sigma",
        "valid",
    )

    def __init__(self) -> None:
        for s in self.__slots__:
            setattr(self, s, math.nan)
        self.valid = False


class VolatilityKernel:
    """Streaming volatility estimation.

    One instance per instrument. ``update`` is called once per closed bar and is
    O(1) amortised in every component.
    """

    def __init__(
        self,
        window: int = 22,
        window_short: int = 5,
        window_long: int = 66,
        ewma_halflife: float = 10.0,
        har_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        rank_window: int = 252,
        rank_min_obs: int = 60,
        vov_window: int = 60,
    ) -> None:
        self.n = int(window)
        self.n_s = int(window_short)
        self.n_l = int(window_long)
        w = np.asarray(har_weights, dtype=np.float64)
        if w.size != 3 or not np.isclose(w.sum(), 1.0):
            raise ValueError("har_weights must be three numbers summing to 1")
        self.har_w = w

        # Close-to-close accumulators.
        self._rv = RollingSum(self.n)
        self._rv_s = RollingSum(self.n_s)
        self._rv_l = RollingSum(self.n_l)
        # Bipower needs the product of adjacent absolute returns.
        self._bp = RollingSum(self.n)
        self._prev_abs_ret = math.nan

        # Range-based accumulators.
        self._rs = RollingSum(self.n)
        self._gk = RollingSum(self.n)

        self._ew = EWMA(ewma_halflife)  # tracks r^2
        self._log_sigma_short = RollingMoments(vov_window)
        self._vol_rank = CausalRank(rank_window, rank_min_obs)
        self._compression_rank = CausalRank(rank_window, rank_min_obs)

        self._prev_close = math.nan
        self._count = 0
        self.out = VolReadout()

    # ------------------------------------------------------------------ #
    @property
    def warmup(self) -> int:
        """Bars before the readout can be trusted."""
        return max(self.n_l, 30) + 2

    def reset(self) -> None:
        for obj in (
            self._rv,
            self._rv_s,
            self._rv_l,
            self._bp,
            self._rs,
            self._gk,
            self._log_sigma_short,
            self._vol_rank,
            self._compression_rank,
        ):
            obj.reset()
        self._ew.reset()
        self._prev_close = math.nan
        self._prev_abs_ret = math.nan
        self._count = 0
        self.out = VolReadout()

    # ------------------------------------------------------------------ #
    def update(self, bar: Bar) -> VolReadout:
        o = self.out = VolReadout()
        c = bar.close

        ret = math.nan
        if self._prev_close == self._prev_close and self._prev_close > 0.0:
            ret = math.log(c / self._prev_close)
        self._prev_close = c
        self._count += 1

        if ret == ret:
            a = abs(ret)
            self._rv.update(ret * ret)
            self._rv_s.update(ret * ret)
            self._rv_l.update(ret * ret)
            if self._prev_abs_ret == self._prev_abs_ret:
                self._bp.update(a * self._prev_abs_ret)
            self._prev_abs_ret = a
            self._ew.update(ret * ret)

        # Range estimators use only within-bar information, so they are
        # available on the very first bar.
        lo_c = math.log(bar.low / c)
        hi_c = math.log(bar.high / c)
        hi_o = math.log(bar.high / bar.open)
        lo_o = math.log(bar.low / bar.open)
        self._rs.update(hi_c * hi_o + lo_c * lo_o)
        hl = math.log(bar.high / bar.low)
        co = math.log(c / bar.open)
        self._gk.update(0.5 * hl * hl - GK_CONST * co * co)

        n_rv = len(self._rv)
        if n_rv < 2 or len(self._rv_l) < min(self.n_l, 20):
            return o

        # ---- close-to-close scales --------------------------------------
        o.sigma_rv = math.sqrt(max(self._rv.total / n_rv, 0.0))
        n_bp = len(self._bp)
        if n_bp >= 2:
            # BPV over the window estimates integrated variance; dividing by the
            # number of contributing pairs gives the per-bar continuous variance.
            bpv_per_bar = _BP_SCALE * self._bp.total / n_bp
            o.sigma_bp = math.sqrt(max(bpv_per_bar, 0.0))
        else:
            o.sigma_bp = o.sigma_rv

        # A degenerate bipower estimate (a run of identical closes) would make
        # every normalised displacement infinite. Fall back to the range
        # estimator, which cannot be zero unless the bar had no range at all.
        if not (o.sigma_bp > 0.0):
            o.sigma_bp = max(o.sigma_rv, 1e-8)

        o.sigma_rs = math.sqrt(max(self._rs.mean, 0.0)) if len(self._rs) >= 2 else math.nan
        o.sigma_gk = math.sqrt(max(self._gk.mean, 0.0)) if len(self._gk) >= 2 else math.nan
        ew = self._ew.value
        o.sigma_ew = math.sqrt(max(ew, 0.0)) if ew == ew else math.nan

        # ---- jump decomposition -----------------------------------------
        rv2, bp2 = o.sigma_rv**2, o.sigma_bp**2
        o.jump_share = 0.0 if rv2 <= 0.0 else min(max((rv2 - bp2) / rv2, 0.0), 1.0)

        # ---- multi-horizon forecast -------------------------------------
        s_short = math.sqrt(max(self._rv_s.mean, 0.0)) if len(self._rv_s) >= 2 else math.nan
        s_long = math.sqrt(max(self._rv_l.mean, 0.0)) if len(self._rv_l) >= 3 else math.nan
        o.sigma_short, o.sigma_long = s_short, s_long

        # Geometric blend across horizons: the HAR insight that volatility is
        # driven by several persistence scales, without fitting coefficients.
        parts, weights = [], []
        for v, w in zip((s_short, o.sigma_rv, s_long), self.har_w):
            if v == v and v > 0.0:
                parts.append(math.log(v))
                weights.append(w)
        wsum = sum(weights)

        if wsum > 0.0:
            o.sigma_fcst = math.exp(sum(p * w for p, w in zip(parts, weights)) / wsum)
        else:
            # Every close-to-close horizon is exactly zero. This is not
            # hypothetical: a halted instrument, a pegged rate, a stablecoin or a
            # dead overnight session all produce runs of identical closes. Fall
            # back to whichever range-based scale survived, and only then to the
            # floor. Returning early here (an earlier version's behaviour) left
            # sigma_fcst as NaN and every consumer downstream had to guess.
            fallback = [
                v for v in (o.sigma_rs, o.sigma_gk, o.sigma_bp) if v == v and v > 0.0
            ]
            o.sigma_fcst = min(fallback) if fallback else 1e-6

        # Guard against a forecast so small that sigma-normalised quantities
        # explode; one basis point per bar is below any tradeable instrument.
        o.sigma_fcst = max(o.sigma_fcst, 1e-6)

        # ---- derived state ----------------------------------------------
        if s_short == s_short and s_short > 0.0:
            self._log_sigma_short.update(math.log(s_short))
            if len(self._log_sigma_short) >= 10:
                o.vol_of_vol = self._log_sigma_short.std

        o.vol_ratio = safe_div(o.sigma_ew, o.sigma_fcst, math.nan)
        r = self._vol_rank.update(math.log(o.sigma_fcst))
        o.vol_rank = r

        if s_long == s_long and s_long > 0.0 and s_short == s_short:
            comp_raw = self._compression_rank.update(s_short / s_long)
            o.compression = (1.0 - comp_raw) if comp_raw == comp_raw else math.nan

        o.range_over_sigma = safe_div(hl, o.sigma_fcst, math.nan)
        o.valid = self._count >= self.warmup and o.vol_rank == o.vol_rank
        return o


# ---------------------------------------------------------------------- #
# Offline HAR fitting, for the development universe only
# ---------------------------------------------------------------------- #


def fit_har_weights(
    log_rv_short: np.ndarray,
    log_rv_mid: np.ndarray,
    log_rv_long: np.ndarray,
    log_rv_next: np.ndarray,
) -> tuple[float, float, float]:
    """Fit HAR blend weights by constrained least squares on pooled dev data.

    Provided for completeness and for the sensitivity study, **not** used by
    default. The default weights ``(0.5, 0.3, 0.2)`` are a ``THEORY`` parameter
    precisely so that this regression -- and the temptation to run it per
    instrument -- contributes nothing to the fitted degree-of-freedom count.

    Weights are constrained to be non-negative and to sum to one, which is what
    makes the result a *blend* rather than an unconstrained regression capable
    of large offsetting coefficients.
    """
    X = np.column_stack([log_rv_short, log_rv_mid, log_rv_long])
    y = np.asarray(log_rv_next, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if X.shape[0] < 100:
        raise ValueError("need at least 100 usable observations to fit HAR weights")

    # Sum-to-one constraint by reparameterising on the simplex interior.
    def loss(w2: np.ndarray) -> float:
        w = np.concatenate(([1.0 - w2.sum()], w2))
        return float(np.mean((X @ w - y) ** 2))

    try:
        from scipy.optimize import minimize  # type: ignore

        res = minimize(
            loss,
            x0=np.array([0.3, 0.2]),
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            constraints=[{"type": "ineq", "fun": lambda w: 1.0 - w.sum()}],
            method="SLSQP",
        )
        w2 = np.clip(res.x, 0.0, 1.0)
    except Exception:  # pragma: no cover - scipy is optional
        best, w2 = math.inf, np.array([0.3, 0.2])
        for a in np.linspace(0.0, 1.0, 51):
            for b in np.linspace(0.0, 1.0 - a, 51):
                v = loss(np.array([a, b]))
                if v < best:
                    best, w2 = v, np.array([a, b])
    w0 = max(0.0, 1.0 - float(w2.sum()))
    total = w0 + float(w2.sum())
    return (w0 / total, float(w2[0]) / total, float(w2[1]) / total)
