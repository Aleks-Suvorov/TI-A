"""Session and time-of-day tagging.

Time-of-day is not a cosmetic feature. The intraday U-shape in volume and
volatility is among the most robustly replicated facts in market
microstructure -- documented since the late 1980s and still present -- and it is
large. An unconditional comparison of a 09:35 volume against a 12:15 volume
mostly measures what time it is, so every volume-derived feature in TI-A is
normalised within its own time-of-day bucket (see
:class:`tia.features.rolling.BucketedMedian`).

The tagger is deliberately profile-driven rather than clever. Guessing an
instrument's session from its data is possible and is a source of subtle,
hard-to-debug errors; declaring it is not.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass

from ..types import Bar, SessionPhase

__all__ = [
    "SessionSpec",
    "SessionState",
    "SessionTagger",
    "US_EQUITY",
    "US_FUTURES",
    "CRYPTO",
    "FX",
    "DAILY",
    "PROFILES",
]


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """Trading session definition in a fixed UTC offset.

    ``utc_offset_hours`` is the offset of the instrument's *reference* market.
    Daylight-saving handling is deliberately out of scope here: for a research
    system, a fixed offset with an explicit warning beats a timezone database
    dependency that silently changes historical feature values when it updates.
    Production deployments should supply exchange-provided session boundaries
    instead -- see ``docs/12-DEPLOYMENT.md``.
    """

    name: str
    utc_offset_hours: float
    open_minute: int  #: minutes from local midnight
    close_minute: int
    #: Continuous instruments have no auction and no overnight gap.
    continuous: bool = False
    #: Minutes at the start/end treated as auction phases.
    auction_minutes: int = 5
    #: Whether the instrument trades on weekends.
    weekend: bool = False
    #: True for bar intervals of a day or more, where intraday phase is moot.
    daily: bool = False

    @property
    def length_minutes(self) -> int:
        if self.continuous:
            return 24 * 60
        return max(1, self.close_minute - self.open_minute)


US_EQUITY = SessionSpec("us_equity", -5.0, 9 * 60 + 30, 16 * 60)
US_FUTURES = SessionSpec("us_futures", -5.0, 0, 24 * 60, continuous=True)
CRYPTO = SessionSpec("crypto", 0.0, 0, 24 * 60, continuous=True, weekend=True)
FX = SessionSpec("fx", 0.0, 0, 24 * 60, continuous=True)
DAILY = SessionSpec("daily", 0.0, 0, 24 * 60, continuous=True, daily=True, weekend=True)

PROFILES: dict[str, SessionSpec] = {
    s.name: s for s in (US_EQUITY, US_FUTURES, CRYPTO, FX, DAILY)
}


@dataclass(slots=True)
class SessionState:
    phase: SessionPhase = SessionPhase.ALL_DAY
    tod_frac: float = math.nan
    tod_bucket: int = 0
    dow: int = -1
    minutes_into_session: float = math.nan
    bars_since_open: int = -1
    new_session: bool = False
    overnight_ret: float = math.nan


class SessionTagger:
    """Streaming session tagger. One instance per instrument."""

    def __init__(self, spec: SessionSpec = DAILY, n_buckets: int = 13) -> None:
        self.spec = spec
        self.n_buckets = int(n_buckets)
        self._prev_day: int | None = None
        self._prev_close: float = math.nan
        self._bars_since_open = -1
        self.state = SessionState()

    def reset(self) -> None:
        self._prev_day = None
        self._prev_close = math.nan
        self._bars_since_open = -1
        self.state = SessionState()

    # ------------------------------------------------------------------ #
    def update(self, bar: Bar) -> SessionState:
        st = self.state = SessionState()
        local = _dt.datetime.fromtimestamp(
            bar.timestamp + self.spec.utc_offset_hours * 3600.0, tz=_dt.timezone.utc
        )
        st.dow = local.weekday()
        day_id = local.toordinal()
        st.new_session = self._prev_day is not None and day_id != self._prev_day

        if st.new_session and self._prev_close == self._prev_close and self._prev_close > 0.0:
            # The overnight return is a genuinely different object from an
            # intraday return: it accrues without a tradeable path, and the
            # literature finds its properties differ systematically. Reported
            # separately rather than mixed into the return series.
            st.overnight_ret = math.log(bar.open / self._prev_close)
            self._bars_since_open = 0
        elif self._prev_day is None:
            self._bars_since_open = 0
        else:
            self._bars_since_open += 1

        self._prev_day = day_id
        self._prev_close = bar.close
        st.bars_since_open = self._bars_since_open

        if self.spec.daily:
            st.phase = SessionPhase.ALL_DAY
            st.tod_frac = 0.5
            st.tod_bucket = 0
            st.minutes_into_session = 0.0
            return st

        minute = local.hour * 60 + local.minute
        if self.spec.continuous:
            st.minutes_into_session = float(minute)
            st.tod_frac = minute / (24.0 * 60.0)
            st.phase = self._continuous_phase(minute)
        else:
            rel = minute - self.spec.open_minute
            st.minutes_into_session = float(rel)
            span = self.spec.length_minutes
            st.tod_frac = min(max(rel / span, 0.0), 1.0) if span > 0 else math.nan
            st.phase = self._session_phase(rel, span)

        # Buckets index the intraday clock, so they must span the *whole* day for
        # continuous instruments and the session for scheduled ones.
        frac = st.tod_frac if st.tod_frac == st.tod_frac else 0.0
        st.tod_bucket = min(int(frac * self.n_buckets), self.n_buckets - 1)
        return st

    # ------------------------------------------------------------------ #
    def _session_phase(self, rel: int, span: int) -> SessionPhase:
        a = self.spec.auction_minutes
        if rel < -a:
            return SessionPhase.PRE_OPEN
        if rel < a:
            return SessionPhase.OPENING_AUCTION
        if rel > span + a:
            return SessionPhase.POST_CLOSE
        if rel > span - a:
            return SessionPhase.CLOSING_AUCTION
        f = rel / span
        if f < 0.30:
            return SessionPhase.MORNING
        if f < 0.65:
            # The lunchtime liquidity trough: thinner books, lower volume, and
            # documented mean reversion. Treated as its own phase because the
            # behavioural engine conditions on it.
            return SessionPhase.MIDDAY
        return SessionPhase.AFTERNOON

    def _continuous_phase(self, minute: int) -> SessionPhase:
        # Even 24-hour instruments inherit the cash-market rhythm of their
        # underlying, so the phase labels track the reference market's hours.
        h = minute / 60.0
        if 9.5 <= h < 10.0:
            return SessionPhase.OPENING_AUCTION
        if 10.0 <= h < 12.0:
            return SessionPhase.MORNING
        if 12.0 <= h < 14.0:
            return SessionPhase.MIDDAY
        if 14.0 <= h < 15.75:
            return SessionPhase.AFTERNOON
        if 15.75 <= h < 16.25:
            return SessionPhase.CLOSING_AUCTION
        return SessionPhase.OVERNIGHT
