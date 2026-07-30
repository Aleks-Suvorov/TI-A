"""Drawdown throttles and kill switches.

The throttle is asymmetric on purpose. A drawdown is weak evidence that the model
has degraded and strong evidence that our estimate of the edge was too high.
Reducing size is the right response to both, and -- this is the part that matters
-- it is the right response *without needing to know which*. Any rule that
requires diagnosing the cause before acting will act too late.

Kill switches are separate from throttles and are not negotiable in the moment.
Each has a pre-committed threshold from the configuration, and tripping one stops
new entries until a human reviews. The discipline this enforces is the point:
``docs/08-FAILURE-MODES.md`` argues that the most likely cause of this system's
death is not a statistical failure but a human adjusting it after a drawdown, and
a switch that can be reasoned away during the drawdown is not a switch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Deque
from collections import deque

from ..config import Config
from ..types import clip

__all__ = ["RiskLimits", "LimitState"]


@dataclass(slots=True)
class LimitState:
    equity: float = 1.0
    peak_equity: float = 1.0
    drawdown: float = 0.0
    throttle: float = 1.0
    halted: bool = False
    triggered: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RiskLimits:
    """Tracks equity, drawdown and the health signals that stop trading."""

    def __init__(self, cfg: Config | None = None, initial_equity: float = 1.0) -> None:
        self.cfg = cfg or Config()
        self.state = LimitState(equity=initial_equity, peak_equity=initial_equity)
        self._trade_returns: Deque[float] = deque(maxlen=self.cfg.monitor_window)
        self._slippage_ratios: Deque[float] = deque(maxlen=50)
        self._brier: float = math.nan
        self._psi: float = math.nan
        self._stale_bars: int = 0

    def reset(self, initial_equity: float = 1.0) -> None:
        self.state = LimitState(equity=initial_equity, peak_equity=initial_equity)
        self._trade_returns.clear()
        self._slippage_ratios.clear()
        self._brier = math.nan
        self._psi = math.nan
        self._stale_bars = 0

    # ------------------------------------------------------------------ #
    def on_equity(self, equity: float) -> None:
        s = self.state
        s.equity = float(equity)
        s.peak_equity = max(s.peak_equity, s.equity)
        s.drawdown = (
            0.0 if s.peak_equity <= 0.0 else max(0.0, 1.0 - s.equity / s.peak_equity)
        )

    def on_trade(self, ret_sigma: float) -> None:
        if ret_sigma == ret_sigma:
            self._trade_returns.append(float(ret_sigma))

    def on_slippage(self, realised: float, modelled: float) -> None:
        if modelled > 0.0 and realised == realised:
            self._slippage_ratios.append(realised / modelled)

    def on_calibration(self, brier: float) -> None:
        self._brier = float(brier)

    def on_feature_drift(self, psi: float) -> None:
        self._psi = float(psi)

    def on_data_staleness(self, bars: int) -> None:
        self._stale_bars = int(bars)

    # ------------------------------------------------------------------ #
    def throttle(self) -> float:
        """Size multiplier from the current drawdown.

        Falls linearly from 1 at ``dd_throttle_start`` to ``dd_throttle_floor``
        at ``dd_throttle_stop``, then to zero at ``dd_kill``.
        """
        cfg = self.cfg
        dd = self.state.drawdown
        if dd <= cfg.dd_throttle_start:
            return 1.0
        if dd >= cfg.dd_kill:
            return 0.0
        if dd >= cfg.dd_throttle_stop:
            span = max(cfg.dd_kill - cfg.dd_throttle_stop, 1e-9)
            f = (dd - cfg.dd_throttle_stop) / span
            return float(clip(cfg.dd_throttle_floor * (1.0 - f), 0.0, 1.0))
        span = max(cfg.dd_throttle_stop - cfg.dd_throttle_start, 1e-9)
        f = (dd - cfg.dd_throttle_start) / span
        return float(clip(1.0 - f * (1.0 - cfg.dd_throttle_floor), 0.0, 1.0))

    # ------------------------------------------------------------------ #
    def check(self) -> LimitState:
        """Evaluate every kill switch and update the state."""
        cfg = self.cfg
        s = self.state
        s.triggered = []
        s.warnings = []

        if s.drawdown >= cfg.dd_kill:
            s.triggered.append(
                f"drawdown {100 * s.drawdown:.1f}% at or beyond the {100 * cfg.dd_kill:.0f}% kill level"
            )

        if self._brier == self._brier and self._brier > cfg.brier_alarm:
            # A Brier score of 0.25 is a coin flip. Above the alarm the
            # probability model has stopped being informative, and every gate
            # downstream of it is acting on a number that means nothing.
            s.triggered.append(
                f"rolling Brier {self._brier:.3f} above {cfg.brier_alarm:.3f}: "
                "probability model unreliable"
            )

        if self._psi == self._psi and self._psi > cfg.psi_alarm:
            s.triggered.append(
                f"feature population stability index {self._psi:.2f} above {cfg.psi_alarm:.2f}: "
                "live inputs no longer resemble the development distribution"
            )

        if len(self._slippage_ratios) >= 10:
            med = sorted(self._slippage_ratios)[len(self._slippage_ratios) // 2]
            if med > cfg.slippage_alarm_mult:
                s.triggered.append(
                    f"realised slippage {med:.1f}x the model: cost assumptions are wrong"
                )

        if self._stale_bars > 3:
            s.triggered.append(f"data stale for {self._stale_bars} bars")

        # A losing streak is checked against its binomial tail rather than
        # against a round number, so the threshold adapts to the trade rate and
        # does not fire on a run that is unremarkable for this many trades.
        streak = self._loss_streak()
        if streak >= self._streak_alarm_level():
            s.warnings.append(
                f"{streak} consecutive losses, beyond the 1-in-100 level for this sample"
            )

        s.throttle = self.throttle()
        s.halted = bool(s.triggered)
        return s

    def _loss_streak(self) -> int:
        n = 0
        for r in reversed(self._trade_returns):
            if r < 0.0:
                n += 1
            else:
                break
        return n

    def _streak_alarm_level(self) -> int:
        """Streak length whose probability under a 50% hit rate is about 1%.

        With ``n`` trades the expected longest run is roughly ``log2(n)``, so a
        fixed "five losses in a row" alarm fires constantly on a long sample and
        never on a short one. This scales.
        """
        n = max(len(self._trade_returns), 8)
        return int(math.ceil(math.log2(n)) + 3)

    # ------------------------------------------------------------------ #
    def report(self) -> str:
        s = self.check()
        lines = [
            f"equity {s.equity:.4f}  peak {s.peak_equity:.4f}  drawdown {100 * s.drawdown:.2f}%",
            f"size throttle {s.throttle:.2f}  halted={s.halted}",
        ]
        for t in s.triggered:
            lines.append(f"  KILL   {t}")
        for w in s.warnings:
            lines.append(f"  warn   {w}")
        return "\n".join(lines)
