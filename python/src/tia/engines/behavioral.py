"""Behavioural engine: when, not which way.

Time-of-day and session effects are among the most reliably replicated patterns
in the literature -- the U-shaped intraday profile of volume and volatility, the
midday liquidity trough, the different character of overnight and intraday
returns. What survives replication is the *volume and volatility* structure. The
directional content of calendar effects is far weaker, has decayed since
publication, and is not something this system leans on.

So the engine's main output is reliability, and its main job is to make the
system stand aside in the phases where the cost model is worst and the
information content lowest: auctions, the first minutes, and the run into the
close. A small directional tilt is retained only for the midday trough, where the
mean-reversion evidence is comparatively solid, and it is deliberately capped.
"""

from __future__ import annotations

import math

from ..config import Config
from ..types import BarContext, EngineOutput, SessionPhase, clip
from .base import BaseEngine

__all__ = ["BehavioralEngine"]

#: Reliability multiplier by session phase. Auctions are not merely risky, they
#: are a different price-formation mechanism: our microstructure features assume
#: continuous trading and do not describe a call auction at all.
_PHASE_TRUST: dict[SessionPhase, float] = {
    SessionPhase.ALL_DAY: 1.00,
    SessionPhase.PRE_OPEN: 0.00,
    SessionPhase.OPENING_AUCTION: 0.00,
    SessionPhase.MORNING: 0.95,
    SessionPhase.MIDDAY: 0.70,
    SessionPhase.AFTERNOON: 0.90,
    SessionPhase.CLOSING_AUCTION: 0.00,
    SessionPhase.POST_CLOSE: 0.00,
    SessionPhase.OVERNIGHT: 0.25,
}


class BehavioralEngine(BaseEngine):
    name = "behavioral"

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__(cfg)
        self.warmup = 10

    def _compute(self, ctx: BarContext) -> EngineOutput:
        f = ctx.features
        notes: list[str] = []

        trust = _PHASE_TRUST.get(f.phase, 0.5)
        if trust <= 0.0:
            return self.emit(
                0.0,
                0.0,
                features={"phase": float(int(f.phase))},
                notes=(f"{f.phase.name.lower().replace('_', ' ')}: not tradeable",),
            )

        score = 0.0
        # Midday reversion: the one calendar tilt with evidence solid enough to
        # act on, and capped well below what the raw effect size might suggest
        # because published intraday effects decay after publication.
        if f.phase is SessionPhase.MIDDAY and f.range_pos == f.range_pos:
            score = clip(-0.30 * (2.0 * f.range_pos - 1.0), -0.3, 0.3)
            notes.append("midday liquidity trough: fade extension")

        # Proximity to a scheduled event. Absent a calendar this is None and the
        # engine simply does not apply the adjustment -- it does not assume the
        # calendar is clear.
        event_penalty = 1.0
        dte = ctx.exog.days_to_event
        if dte is not None:
            if dte <= 0.0:
                event_penalty = 0.0
                notes.append("scheduled event lands inside this bar")
            elif dte < 1.0:
                event_penalty = 0.25
                notes.append("scheduled event within one session")
            elif dte < 2.0:
                event_penalty = 0.65

        # Very early in a session the rolling windows contain a discontinuity
        # and the microstructure features are least reliable.
        open_penalty = 1.0
        if 0 <= f.bars_since_session_open < 3:
            open_penalty = 0.5 + 0.15 * f.bars_since_session_open
            notes.append("first bars of session")

        reliability = clip(trust * event_penalty * open_penalty, 0.0, 1.0)

        return self.emit(
            score,
            reliability,
            features={
                "phase": float(int(f.phase)),
                "tod_frac": f.tod_frac,
                "dow": float(f.dow),
                "days_to_event": float(dte) if dte is not None else math.nan,
            },
            diagnostics={
                "phase_trust": trust,
                "event_penalty": event_penalty,
                "open_penalty": open_penalty,
            },
            notes=notes,
        )
