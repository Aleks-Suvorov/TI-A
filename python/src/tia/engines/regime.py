"""Regime posterior with no parameters fitted to returns.

``docs/01-THEORY.md`` §5. This engine sits at the top of the architecture because
the two directional effects with real out-of-sample evidence -- momentum
continuation and short-horizon liquidity-provision reversal -- issue *opposite*
instructions, and nothing downstream can be interpreted without knowing which is
in force. Systems that average trend and reversion logic together are not
hedging; they are cancelling their own signal into noise.

The design's distinguishing feature is that the measurement model is *specified*
rather than estimated. A Gaussian HMM fitted to returns needs O(K² + KJ²)
parameters estimated on the target series, is notoriously unstable across
restarts, and leaks badly if fitted on a full sample. Here each regime is a fixed
product of Beta densities over rank features, chosen a priori from what each
regime *means*. Parameters fitted to market outcomes in this engine: zero. The
stickiness is estimated once on the development universe and frozen; the Beta
concentration is a theory parameter.

Note the encoding trick that makes the design table expressible: with
concentration ``c = 2`` and mean ``m = 0.5``, ``Beta(1, 1)`` is exactly uniform,
so "this regime makes no claim about this feature" lives inside the same
parametric family and contributes a likelihood factor of exactly 1.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..features.rolling import EWMA, RollingCorrelation
from ..types import BarContext, EngineOutput, Regime, clip
from .base import BaseEngine

__all__ = ["RegimeEngine", "RegimeState", "FEATURE_ORDER", "DESIGN", "beta_logpdf"]

#: Order of the rank features consumed by the measurement model. Fixed, because
#: the design table below is indexed positionally.
FEATURE_ORDER: tuple[str, ...] = (
    "vol_rank",
    "vr_rank",
    "ladr_rank",
    "absorption_rank",
    "participation_rank",
    "entropy",
    "jump_share",
    "efficiency",
)

#: ``None`` means "this regime makes no claim about this feature".
_HI, _LO, _MID = 0.8, 0.2, 0.5

#: The measurement design. Rows are regimes in :class:`Regime` order, columns
#: follow :data:`FEATURE_ORDER`. Every entry is a statement about what the regime
#: *means*, defensible from the theory, and none was chosen by looking at
#: outcomes.
DESIGN: dict[Regime, tuple[float | None, ...]] = {
    # A trend is: increments positively autocorrelated, displacement cheap in
    # liquidity terms, little absorption, efficient path, low ordinal disorder.
    # It makes no claim about the volatility level -- trends occur at every
    # volatility -- nor about raw participation.
    Regime.TREND: (None, _HI, _HI, _LO, None, 0.35, None, _HI),
    # Mean reversion is the mirror: negatively autocorrelated increments, heavy
    # absorption, inefficient paths, higher ordinal disorder.
    Regime.REVERT: (None, _LO, 0.35, _HI, None, 0.65, 0.35, _LO),
    # Stress is defined by the volatility and jump state, and by participation.
    # It deliberately makes no claim about autocorrelation or efficiency,
    # because in stress those flip fast and any claim would be wrong half the
    # time.
    Regime.STRESS: (0.85, None, 0.7, 0.3, _HI, None, _HI, None),
    # Quiet is thin and disordered: low volatility, low participation, weak
    # displacement, high entropy.
    Regime.QUIET: (0.15, None, 0.3, None, _LO, 0.7, _LO, 0.3),
}


def beta_logpdf(x: float, a: float, b: float) -> float:
    """Log density of ``Beta(a, b)``, guarded away from the open interval's ends.

    Ranks can legitimately be exactly 0 or 1 -- the current bar is the extreme of
    its own window more often than one might expect -- and the Beta density
    diverges there for shape parameters below 1. Clamping to a small interior
    margin is the standard remedy and costs nothing that matters.
    """
    if x != x:
        return 0.0
    x = clip(x, 1e-6, 1.0 - 1e-6)
    return (
        (a - 1.0) * math.log(x)
        + (b - 1.0) * math.log1p(-x)
        - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    )


class RegimeState:
    """Public read model of the engine, consumed across the system."""

    __slots__ = ("posterior", "dominant", "hazard", "drift", "ambiguity", "temper", "valid")

    def __init__(self) -> None:
        self.posterior = np.full(4, 0.25)
        self.dominant = Regime.QUIET
        self.hazard = 1.0
        self.drift = 0.0
        self.ambiguity = 0.75
        self.temper = 1.0
        self.valid = False

    def p(self, r: Regime) -> float:
        return float(self.posterior[int(r)])

    def as_tuple(self) -> tuple[float, float, float, float]:
        v = self.posterior
        return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))


class RegimeEngine(BaseEngine):
    """Sticky discrete Bayes filter over four regimes. Forward recursion only.

    The Baum-Welch forward-backward posterior would be smoother and would also
    repaint every historical regime label the instant a new bar arrived, so only
    :math:`p(k_t \\mid f_{1:t})` is admissible here.

    The engine's own *directional* score is deliberately near zero. Knowing the
    regime says which kind of evidence to trust, not which way to trade. What it
    contributes to fusion is a small trend/revert lean and, much more
    importantly, the hazard that gates entries.
    """

    name = "regime"

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.warmup = 5
        self._shapes = self._build_shapes(c.regime_beta_concentration)
        self._log_prior = np.full(4, -math.log(4.0))
        self._logA = self._build_transition(c.regime_stickiness)
        self._log_alpha = self._log_prior.copy()
        self._ref = [EWMA(c.regime_shift_hazard_halflife) for _ in range(4)]
        self._featcorr = RollingCorrelation(len(FEATURE_ORDER), halflife=250.0)
        self.state = RegimeState()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_shapes(concentration: float) -> np.ndarray:
        """Shape parameters ``(a, b)`` per regime and feature.

        ``None`` in the design becomes ``Beta(1, 1)``, the uniform density, via
        mean 0.5 and concentration 2.
        """
        out = np.zeros((4, len(FEATURE_ORDER), 2))
        for reg, row in DESIGN.items():
            for j, m in enumerate(row):
                if m is None:
                    a = b = 1.0
                else:
                    a = m * concentration
                    b = (1.0 - m) * concentration
                out[int(reg), j] = (max(a, 1e-3), max(b, 1e-3))
        return out

    @staticmethod
    def _build_transition(stickiness: float) -> np.ndarray:
        s = clip(stickiness, 0.5, 0.9999)
        off = (1.0 - s) / 3.0
        A = np.full((4, 4), off)
        np.fill_diagonal(A, s)
        return np.log(A)

    def _reset(self) -> None:
        self._log_alpha = self._log_prior.copy()
        for e in self._ref:
            e.reset()
        self._featcorr.reset()
        self.state = RegimeState()

    # ------------------------------------------------------------------ #
    def step(self, feats: dict[str, float]) -> RegimeState:
        """Advance the filter by one bar. Separated from :meth:`_compute` so the
        pipeline can update the regime before the engines that condition on it."""
        st = RegimeState()
        vals = np.array([feats.get(k, math.nan) for k in FEATURE_ORDER], dtype=np.float64)
        usable = np.isfinite(vals)
        if usable.sum() < 4:
            self.state = st
            return st

        # Tempering exponent: correlated features would otherwise be counted as
        # independent evidence, the same error CALOP corrects in the fusion
        # layer. eta = 1 when features are independent, smaller when redundant.
        if usable.all():
            self._featcorr.update(vals)
        eta = 1.0
        if self._featcorr.count > 60:
            C = self._featcorr.correlation(ridge=0.15)
            try:
                ebe_f = float(np.ones(C.shape[0]) @ np.linalg.solve(C, np.ones(C.shape[0])))
                eta = clip(ebe_f / len(FEATURE_ORDER), 0.15, 1.0)
            except np.linalg.LinAlgError:  # pragma: no cover - ridge prevents this
                eta = 1.0
        st.temper = eta

        loglik = np.zeros(4)
        for k in range(4):
            s = 0.0
            for j, x in enumerate(vals):
                if usable[j]:
                    a, b = self._shapes[k, j]
                    s += beta_logpdf(float(x), float(a), float(b))
            loglik[k] = eta * s

        # Forward recursion in log space: alpha_t(k) ∝ p(f|k) * sum_j A_jk alpha_{t-1}(j)
        prev = self._log_alpha
        pred = np.empty(4)
        for k in range(4):
            v = prev + self._logA[:, k]
            mx = float(v.max())
            pred[k] = mx + math.log(float(np.exp(v - mx).sum()))
        post = pred + loglik
        mx = float(post.max())
        post = post - (mx + math.log(float(np.exp(post - mx).sum())))
        self._log_alpha = post

        alpha = np.exp(post)
        alpha = alpha / alpha.sum()
        st.posterior = alpha
        st.dominant = Regime(int(np.argmax(alpha)))

        ref = np.array([e.update(float(alpha[k])) for k, e in enumerate(self._ref)])
        tot = float(ref.sum())
        if tot > 0.0:
            ref = ref / tot
            st.drift = 0.5 * float(np.abs(alpha - ref).sum())
        st.ambiguity = 1.0 - float(alpha.max())
        # Either a posterior in motion or one that never committed is a reason
        # to stand aside. Taking the max rather than a blend means one alone
        # suffices, which is the conservative reading.
        st.hazard = max(st.drift, st.ambiguity)
        st.valid = True
        self.state = st
        return st

    # ------------------------------------------------------------------ #
    def _compute(self, ctx: BarContext) -> EngineOutput:
        st = self.state
        if not st.valid:
            return self.abstain("regime not yet estimated")

        f = ctx.features
        # A small directional lean only: in TREND follow the measured slope, in
        # REVERT fade the position inside the confirmed range. The magnitude is
        # capped low because this engine's job is routing, not direction.
        lean = 0.0
        if f.kalman_slope_t == f.kalman_slope_t:
            lean += st.p(Regime.TREND) * math.tanh(f.kalman_slope_t / 2.0)
        if f.range_pos == f.range_pos:
            lean += st.p(Regime.REVERT) * (0.5 - f.range_pos) * 2.0
        score = clip(0.5 * lean, -1.0, 1.0)

        # Reliability is the posterior's confidence discounted by the hazard: a
        # committed, stable regime is worth listening to; one in transition is
        # not.
        reliability = clip((1.0 - st.ambiguity) * (1.0 - st.hazard), 0.0, 1.0)

        notes = [f"regime {st.dominant.label.lower()} p={st.posterior.max():.2f}"]
        if st.hazard > self.cfg.regime_hazard_max:
            notes.append(f"transition hazard {st.hazard:.2f} above gate")

        return self.emit(
            score,
            reliability,
            features={
                "p_trend": st.p(Regime.TREND),
                "p_revert": st.p(Regime.REVERT),
                "p_stress": st.p(Regime.STRESS),
                "p_quiet": st.p(Regime.QUIET),
                "hazard": st.hazard,
            },
            diagnostics={
                "drift": st.drift,
                "ambiguity": st.ambiguity,
                "temper": st.temper,
            },
            notes=notes,
        )
