"""Jump detection and the continuous/discontinuous variance split.

``sigma_bp`` and ``jump_share`` are produced by :mod:`tia.features.scales`; this
module adds the *test*, which is a different and more useful object than the
share. A jump share of 0.3 in a 22-bar window might be one genuine discontinuity
or might be sampling noise, and the Barndorff-Nielsen & Shephard ratio statistic
distinguishes them with an asymptotic null.

The distinction matters for the system because jumps and continuous variation
behave differently downstream. A jump is, by construction, a price change that
occurred without a tradeable path -- our stop was not "hit", it was skipped over.
So jump-dominated conditions call for wider barriers and smaller size, which is
the `STRESS` regime's job.
"""

from __future__ import annotations

import math

import numpy as np

from .rolling import RingBuffer, RollingSum

__all__ = ["JumpTest", "MU_43", "BNS_THETA"]

#: ``mu_{4/3} = 2^{2/3} * Gamma(7/6) / Gamma(1/2)``, the tripower constant.
MU_43 = (2.0 ** (2.0 / 3.0)) * math.gamma(7.0 / 6.0) / math.gamma(0.5)

#: ``(pi^2)/4 + pi - 5``, the asymptotic variance constant of the BNS ratio test.
BNS_THETA = (math.pi**2) / 4.0 + math.pi - 5.0


class JumpTest:
    """Barndorff-Nielsen & Shephard ratio test for jumps in a rolling window.

    The statistic

    .. math::
        z = \\frac{1 - \\mathrm{BPV}/\\mathrm{RV}}
                  {\\sqrt{\\theta\\, n^{-1}\\max(1,\\ \\mathrm{TQ}/\\mathrm{BPV}^2)}}

    is asymptotically standard normal under the null of no jumps, with
    :math:`\\theta = \\pi^2/4 + \\pi - 5` and TQ the realised tripower
    quarticity. Large positive ``z`` says the window contains discontinuities
    that plain realised variance is attributing to diffusion.

    Tripower quarticity is used rather than realised quarticity because the
    latter is itself jump-contaminated, which would bias the test toward
    non-rejection exactly when jumps are present.
    """

    __slots__ = ("_n", "_rv", "_bp", "_tp", "_a1", "_a2", "_ret", "_jump_k")

    def __init__(self, window: int = 22, jump_sigma_mult: float = 4.0) -> None:
        self._n = int(window)
        self._rv = RollingSum(self._n)
        self._bp = RollingSum(self._n)
        self._tp = RollingSum(self._n)
        self._a1 = math.nan  # |r_{t-1}|
        self._a2 = math.nan  # |r_{t-2}|
        self._ret = RingBuffer(self._n)
        self._jump_k = float(jump_sigma_mult)

    def reset(self) -> None:
        self._rv.reset()
        self._bp.reset()
        self._tp.reset()
        self._ret.reset()
        self._a1 = self._a2 = math.nan

    def update(self, ret: float) -> None:
        if ret != ret:
            return
        a = abs(ret)
        self._ret.push(ret)
        self._rv.update(ret * ret)
        if self._a1 == self._a1:
            self._bp.update(a * self._a1)
        if self._a1 == self._a1 and self._a2 == self._a2:
            self._tp.update((a ** (4.0 / 3.0)) * (self._a1 ** (4.0 / 3.0)) * (self._a2 ** (4.0 / 3.0)))
        self._a2 = self._a1
        self._a1 = a

    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        return len(self._tp) >= max(10, self._n // 3)

    @property
    def z(self) -> float:
        if not self.ready:
            return math.nan
        n_bp = len(self._bp)
        rv = self._rv.total
        if rv <= 0.0 or n_bp < 2:
            return math.nan
        # Scale both to the same per-window integrated-variance footing.
        bpv = (math.pi / 2.0) * self._bp.total * (n_bp + 1) / n_bp
        if bpv <= 0.0:
            return math.nan
        n_tp = len(self._tp)
        tq = (n_tp * (MU_43**-3.0)) * self._tp.total
        ratio = max(1.0, tq / (bpv * bpv)) if bpv > 0.0 else 1.0
        denom = math.sqrt(BNS_THETA * ratio / max(n_bp, 1))
        if denom <= 0.0:
            return math.nan
        return (1.0 - bpv / rv) / denom

    @property
    def significant(self) -> bool:
        """``z > 2.33``, i.e. rejection at the one-sided 1% level."""
        z = self.z
        return bool(z == z and z > 2.326)

    def is_jump_bar(self, ret: float, sigma_bp: float) -> bool:
        """Per-bar flag: this bar's move exceeds ``jump_sigma_mult`` sigma.

        A crude complement to the window-level test, but it is the version the
        risk layer needs -- a single bar that moved four continuous-sigma is one
        our stop may have gapped through regardless of what the window says.
        """
        if ret != ret or not (sigma_bp > 0.0):
            return False
        return abs(ret) / sigma_bp > self._jump_k
