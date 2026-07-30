"""Correlation-Aware Log-Odds Pooling.

``docs/01-THEORY.md`` §6. This module exists to prevent one specific and very
expensive error.

Take ten engines, each mildly bullish with score 0.5. Each maps to log-odds
:math:`2\\,\\mathrm{artanh}(0.5) = 1.0986`. Summing them -- which is what any
"confluence score" implicitly does -- gives 10.99, and :math:`\\varsigma(10.99) =
0.99998`. A system reporting 99.998% confidence from ten mild opinions is not
confident; it is broken.

Summation of log-odds is the correct Bayesian operation *only* under conditional
independence of the evidence. Ten engines built on overlapping windows of the
same price series are nowhere near independent. The error is not a rounding
detail: it is the difference between 75% and 99.998%, and it is invisible in a
backtest because the inflated probability is a monotone transform of the honest
one -- ranking is preserved, so hit rates look fine while position sizing and
confidence gating are wildly wrong.

The fix models each engine as a noisy observation of a common latent log-odds
:math:`L`, with :math:`\\ell = \\mathbf 1 L + \\varepsilon`,
:math:`\\varepsilon \\sim N(0, \\sigma_e^2 D^{-1/2} C D^{-1/2})`, where :math:`C`
is the engine correlation matrix and :math:`D` holds the reliabilities as
precisions. Writing :math:`M = D^{1/2} C^{-1} D^{1/2}`:

.. math::
    \\hat L = \\mathbf 1^\\top M \\ell, \\qquad
    \\mathrm{EBE} = \\mathbf 1^\\top M \\mathbf 1, \\qquad
    \\mathrm{Var}(\\hat L) = \\sigma_e^2\\,\\mathrm{EBE}.

This interpolates between the two correct answers: at :math:`C = I` it *is* the
Bayesian sum, and for perfectly duplicated engines it counts them once. For
equicorrelated engines :math:`\\mathrm{EBE} = m/(1+(m-1)\\bar\\rho)`, so ten
engines at :math:`\\bar\\rho = 0.5` supply an effective breadth of 1.82 -- fewer
than two independent opinions.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..features.rolling import RollingCorrelation
from ..types import EngineOutput, FusionResult, clip

__all__ = ["CALOP", "logistic", "evidence_from_score", "effective_breadth"]

#: Scores are clamped inside the open interval before artanh, so that a
#: saturated engine maps to a large but finite log-odds rather than infinity.
_SCORE_CLAMP = 0.999

#: Engines below this effective magnitude are treated as expressing no direction
#: when computing aligned breadth. They still contribute to total breadth, and
#: hence to variance, which is the correct treatment of a confident abstainer.
_ALIGN_EPS = 0.05


def logistic(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def evidence_from_score(score: float) -> float:
    """Map a score in ``[-1, 1]`` to log-odds evidence.

    ``2 artanh(s)`` is the natural link: it is the inverse of the tanh squashing
    the engines already apply, so an engine that formed its opinion as
    ``tanh(t)`` on some t-statistic contributes ``2t`` -- linear in the
    underlying statistic, which is what a log-odds contribution should be.
    """
    s = clip(score, -_SCORE_CLAMP, _SCORE_CLAMP)
    return 2.0 * math.atanh(s)


def effective_breadth(C: np.ndarray, reliabilities: np.ndarray) -> float:
    """``1' D^{1/2} C^{-1} D^{1/2} 1`` -- independent opinions actually present."""
    d = np.sqrt(np.clip(reliabilities, 0.0, 1.0))
    try:
        x = np.linalg.solve(C, d)
    except np.linalg.LinAlgError:  # pragma: no cover - ridge should prevent this
        return float(np.sum(reliabilities))
    return float(max(d @ x, 0.0))


class CALOP:
    """Stateful pooler. Owns the causally-estimated engine correlation matrix."""

    def __init__(self, engine_names: list[str], cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.names = list(engine_names)
        self.m = len(self.names)
        self._idx = {n: i for i, n in enumerate(self.names)}
        self._corr = RollingCorrelation(self.m, halflife=float(self.cfg.correlation_window) / 2.0)
        self._last_C = np.eye(self.m)

    def reset(self) -> None:
        self._corr.reset()
        self._last_C = np.eye(self.m)

    # ------------------------------------------------------------------ #
    def observe(self, outputs: dict[str, EngineOutput]) -> None:
        """Feed the current scores to the correlation estimator.

        Called every bar regardless of whether a trade results, because the
        correlation structure is a property of the engines, not of the trades.
        Estimating it only on signal bars would sample exactly the tail where
        engines agree and would understate correlation badly.
        """
        v = np.zeros(self.m)
        ok = True
        for n, i in self._idx.items():
            o = outputs.get(n)
            if o is None or not o.valid:
                ok = False
                break
            v[i] = o.score
        if ok:
            self._corr.update(v)

    def correlation(self) -> np.ndarray:
        if self._corr.count > 60:
            self._last_C = self._corr.correlation(ridge=self.cfg.fusion_shrink)
        else:
            # Before enough history exists, assume a uniform moderate
            # correlation rather than independence. Assuming independence early
            # would let the system be most overconfident exactly when it knows
            # least about its own engines.
            C = np.full((self.m, self.m), 0.35)
            np.fill_diagonal(C, 1.0)
            self._last_C = (1.0 - self.cfg.fusion_shrink) * C + self.cfg.fusion_shrink * np.eye(
                self.m
            )
        return self._last_C

    # ------------------------------------------------------------------ #
    def fuse(self, outputs: dict[str, EngineOutput], calibrate=None) -> FusionResult:
        """Pool engine outputs into a direction, a probability and a breadth.

        ``calibrate`` is an optional callable mapping a raw probability to a
        recalibrated one (see :mod:`tia.fusion.calibration`). It is applied last,
        after the uncertainty correction, so that the isotonic map corrects
        residual miscalibration rather than doing the variance shrinkage's job.
        """
        cfg = self.cfg
        e = np.zeros(self.m)
        rho = np.zeros(self.m)
        raw_scores = np.zeros(self.m)
        for n, i in self._idx.items():
            o = outputs.get(n)
            if o is None or not o.valid or o.reliability <= 0.0:
                continue
            raw_scores[i] = o.score
            e[i] = evidence_from_score(o.score)
            rho[i] = o.reliability

        if not np.any(rho > 0.0):
            return FusionResult(
                log_odds=0.0, log_odds_se=math.inf, p_success=0.5, p_low=0.0, p_high=1.0,
                ebe=0.0, direction=0,
            )

        C = self.correlation()
        d = np.sqrt(rho)
        try:
            Cinv_d = np.linalg.solve(C, d)
            Cinv_de = np.linalg.solve(C, d * e)
        except np.linalg.LinAlgError:  # pragma: no cover
            Cinv_d, Cinv_de = d, d * e

        L = float(d @ Cinv_de)
        ebe = float(max(d @ Cinv_d, 1e-9))
        var_L = (cfg.fusion_evidence_sd**2) * ebe
        se_L = math.sqrt(var_L)

        direction = 0 if L == 0.0 else (1 if L > 0.0 else -1)

        # Aligned breadth: how many independent engines actually support the
        # pooled direction. Total breadth includes confident abstainers, which
        # rightly widen the interval but must not satisfy the gate.
        aligned = [
            i
            for i in range(self.m)
            if rho[i] > 0.0
            and abs(raw_scores[i]) * rho[i] > _ALIGN_EPS
            and (direction == 0 or math.copysign(1.0, raw_scores[i]) == direction)
        ]
        if aligned:
            sub = C[np.ix_(aligned, aligned)]
            ebe_aligned = effective_breadth(sub, rho[aligned])
        else:
            ebe_aligned = 0.0

        # Predictive probability under uncertainty in L, via MacKay's probit
        # approximation to the Gaussian-integrated logistic. This is the
        # mechanism that pulls stated confidence toward 50% when the evidence is
        # narrow: a reported 97% requires evidence that is both strong and broad.
        T, b = cfg.fusion_temperature, cfg.fusion_bias
        kappa = 1.0 / math.sqrt(1.0 + (math.pi / 8.0) * (T * T) * var_L)
        centre = (T * L + b) * kappa
        p_up = logistic(centre)

        z = 1.6448536269514722  # 90% two-sided
        p_lo_up = logistic((T * (L - z * se_L) + b) * kappa)
        p_hi_up = logistic((T * (L + z * se_L) + b) * kappa)

        if calibrate is not None:
            p_up = calibrate(p_up)
            p_lo_up = calibrate(p_lo_up)
            p_hi_up = calibrate(p_hi_up)

        # Report probabilities for the *proposed* direction, per SPEC.md §1.
        if direction >= 0:
            p_success, p_low, p_high = p_up, p_lo_up, p_hi_up
        else:
            p_success, p_low, p_high = 1.0 - p_up, 1.0 - p_hi_up, 1.0 - p_lo_up

        contributions: dict[str, float] = {}
        weights: dict[str, float] = {}
        for n, i in self._idx.items():
            w = float(d[i] * Cinv_d[i]) if ebe > 0.0 else 0.0
            weights[n] = w / ebe if ebe > 0.0 else 0.0
            contributions[n] = float(d[i] * Cinv_de[i])

        return FusionResult(
            log_odds=L,
            log_odds_se=se_L,
            p_success=clip(p_success, 1e-6, 1.0 - 1e-6),
            p_low=clip(min(p_low, p_high), 0.0, 1.0),
            p_high=clip(max(p_low, p_high), 0.0, 1.0),
            ebe=ebe_aligned,
            direction=direction,
            weights=weights,
            contributions=contributions,
        )

    # ------------------------------------------------------------------ #
    def diagnostics(self) -> dict[str, float]:
        """Correlation health, for the monitoring layer.

        Rising mean correlation means falling effective breadth, which is the
        early warning that the system is about to become overconfident. It is
        also the failure the trailing estimator is slowest to see -- see
        ``docs/01-THEORY.md`` §14.3.
        """
        C = self.correlation()
        off = C[~np.eye(self.m, dtype=bool)]
        return {
            "mean_abs_corr": float(np.mean(np.abs(off))) if off.size else 0.0,
            "max_abs_corr": float(np.max(np.abs(off))) if off.size else 0.0,
            "ebe_at_full_reliability": effective_breadth(C, np.ones(self.m)),
            "condition_number": float(np.linalg.cond(C)),
            "samples": float(self._corr.count),
        }
