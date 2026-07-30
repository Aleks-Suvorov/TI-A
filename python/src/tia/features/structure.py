"""Market structure that cannot repaint.

``docs/01-THEORY.md`` §4.3. Structure-based systems are the worst offenders for
fictitious backtest performance, and the mechanism is always the same: a swing
pivot that is obvious in hindsight is invisible in real time. The fix is not
subtle but it must be applied without exception -- every pivot records the bar at
which it became *knowable*, and consumers may only read pivots whose
``confirmed_at`` has already passed.

The cost is latency. A pivot is confirmed several bars after it occurred, and a
liquidity sweep is only knowable after it has failed. That delay is not a
deficiency of this implementation; it is the actual information arrival time, and
any system that appears not to pay it is reading the future.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Deque, Literal
from collections import deque

from ..types import Bar, clip, safe_div

__all__ = ["Pivot", "PivotTracker", "SweepDetector", "HTFAggregator", "SweepEvent"]


@dataclass(frozen=True, slots=True)
class Pivot:
    """A confirmed swing extreme.

    ``index`` is where it happened; ``confirmed_at`` is where it became visible.
    The gap between them is the non-repainting tax, and it is reported rather
    than hidden.
    """

    kind: Literal["high", "low"]
    index: int
    price: float
    confirmed_at: int
    #: Retracement, in sigma units, that triggered confirmation.
    retrace_sigma: float

    @property
    def lag(self) -> int:
        return self.confirmed_at - self.index


class PivotTracker:
    """Causal, volatility-scaled zigzag.

    Maintains a candidate extreme in the current search direction. The candidate
    becomes a confirmed :class:`Pivot` when price has retraced from it by at
    least ``atr_mult`` sigma **and** at least ``confirm_bars`` bars have elapsed
    since the extreme. Both conditions are needed: the retracement test alone
    fires on a single violent bar, and the elapsed-bars test alone fires on
    sideways drift.

    Scaling the retracement threshold by sigma rather than by a price amount is
    what lets one parameter serve every instrument and every era.
    """

    def __init__(self, atr_mult: float = 1.5, confirm_bars: int = 3, history: int = 32) -> None:
        self.atr_mult = float(atr_mult)
        self.confirm_bars = int(confirm_bars)
        self._pivots: Deque[Pivot] = deque(maxlen=int(history))
        self._seeking: Literal["high", "low"] = "high"
        self._cand_price = math.nan
        self._cand_index = -1
        self._cand_sigma = math.nan
        self._i = -1
        self.last_high: Pivot | None = None
        self.last_low: Pivot | None = None

    def reset(self) -> None:
        self._pivots.clear()
        self._seeking = "high"
        self._cand_price = math.nan
        self._cand_index = -1
        self._cand_sigma = math.nan
        self._i = -1
        self.last_high = None
        self.last_low = None

    # ------------------------------------------------------------------ #
    def update(self, bar: Bar, sigma: float) -> Pivot | None:
        """Feed one closed bar. Returns a pivot only on the bar it is confirmed."""
        self._i += 1
        i = self._i
        if not (sigma > 0.0):
            return None

        if self._cand_index < 0:
            self._cand_price = bar.high if self._seeking == "high" else bar.low
            self._cand_index = i
            self._cand_sigma = sigma
            return None

        thresh = self.atr_mult * self._cand_sigma

        if self._seeking == "high":
            if bar.high >= self._cand_price:
                # Candidate extends; the clock restarts, which is correct -- the
                # swing has not turned.
                self._cand_price = bar.high
                self._cand_index = i
                self._cand_sigma = sigma
                return None
            retrace = math.log(self._cand_price / bar.low) if bar.low > 0.0 else 0.0
            if retrace >= thresh and (i - self._cand_index) >= self.confirm_bars:
                piv = Pivot("high", self._cand_index, self._cand_price, i, retrace / self._cand_sigma)
                self._pivots.append(piv)
                self.last_high = piv
                self._seeking = "low"
                self._cand_price = bar.low
                self._cand_index = i
                self._cand_sigma = sigma
                return piv
        else:
            if bar.low <= self._cand_price:
                self._cand_price = bar.low
                self._cand_index = i
                self._cand_sigma = sigma
                return None
            retrace = math.log(bar.high / self._cand_price) if self._cand_price > 0.0 else 0.0
            if retrace >= thresh and (i - self._cand_index) >= self.confirm_bars:
                piv = Pivot("low", self._cand_index, self._cand_price, i, retrace / self._cand_sigma)
                self._pivots.append(piv)
                self.last_low = piv
                self._seeking = "high"
                self._cand_price = bar.high
                self._cand_index = i
                self._cand_sigma = sigma
                return piv
        return None

    # ------------------------------------------------------------------ #
    @property
    def index(self) -> int:
        return self._i

    def high_price(self) -> float:
        return self.last_high.price if self.last_high else math.nan

    def low_price(self) -> float:
        return self.last_low.price if self.last_low else math.nan

    def high_age(self) -> float:
        """Bars since the pivot high was *confirmed*, not since it occurred."""
        return float(self._i - self.last_high.confirmed_at) if self.last_high else math.nan

    def low_age(self) -> float:
        return float(self._i - self.last_low.confirmed_at) if self.last_low else math.nan

    def range_position(self, close: float) -> float:
        """Where price sits inside the last confirmed range, in ``[0, 1]``."""
        if not (self.last_high and self.last_low):
            return math.nan
        hi, lo = self.last_high.price, self.last_low.price
        if hi <= lo:
            return math.nan
        return clip((close - lo) / (hi - lo), 0.0, 1.0)

    def range_sigma(self, sigma: float) -> float:
        """Width of the confirmed range in sigma units."""
        if not (self.last_high and self.last_low) or not (sigma > 0.0):
            return math.nan
        hi, lo = self.last_high.price, self.last_low.price
        if hi <= lo:
            return math.nan
        return math.log(hi / lo) / sigma

    def recent(self, n: int = 4) -> list[Pivot]:
        return list(self._pivots)[-n:]

    def structure_bias(self) -> float:
        """Higher-highs/higher-lows score in ``[-1, +1]`` from confirmed pivots.

        Compares the last two confirmed highs and the last two confirmed lows.
        Uses four pivots, so it is slow -- appropriately, since this is meant to
        describe structure rather than momentum.
        """
        highs = [p.price for p in self._pivots if p.kind == "high"][-2:]
        lows = [p.price for p in self._pivots if p.kind == "low"][-2:]
        score = 0.0
        n = 0
        if len(highs) == 2:
            score += 1.0 if highs[1] > highs[0] else -1.0
            n += 1
        if len(lows) == 2:
            score += 1.0 if lows[1] > lows[0] else -1.0
            n += 1
        return score / n if n else math.nan


@dataclass(frozen=True, slots=True)
class SweepEvent:
    """A completed penetration-and-failure at a confirmed level."""

    direction: int  #: +1 = swept a low, bullish reclaim; -1 = swept a high
    level: float
    penetration_index: int
    reclaim_index: int
    penetration_sigma: float
    participation_rank: float
    absorption_rank: float

    @property
    def bars_to_reclaim(self) -> int:
        return self.reclaim_index - self.penetration_index


class SweepDetector:
    """Penetration of a confirmed level, on anomalous volume, that fails.

    The four conditions of ``docs/01-THEORY.md`` §3.2, all evaluated on closed
    bars:

    1. price penetrates a confirmed pivot by more than ``min_penetration_sigma``,
       so that ordinary noise does not qualify;
    2. participation on the penetrating bar is in its upper tail -- the level was
       hit, not drifted through;
    3. the bar shows absorption, or a close location value opposed to the
       penetration, i.e. it gave back its excursion;
    4. a later closed bar reclaims the level.

    Condition 4 is what makes this non-repainting *and* late. Marking a sweep at
    the moment of penetration is reading the future.
    """

    def __init__(
        self,
        max_bars_to_reclaim: int = 6,
        min_penetration_sigma: float = 0.25,
        participation_rank_min: float = 0.75,
    ) -> None:
        self.max_bars = int(max_bars_to_reclaim)
        self.min_pen = float(min_penetration_sigma)
        self.part_min = float(participation_rank_min)
        self._pending: list[dict[str, float]] = []
        self._i = -1
        self.last_event: SweepEvent | None = None

    def reset(self) -> None:
        self._pending.clear()
        self._i = -1
        self.last_event = None

    def update(
        self,
        bar: Bar,
        sigma: float,
        pivot_high: float,
        pivot_low: float,
        participation_rank: float,
        absorption_rank: float,
    ) -> SweepEvent | None:
        self._i += 1
        i = self._i
        event: SweepEvent | None = None

        # Expire stale candidates first, so a reclaim outside the window cannot
        # resurrect one.
        self._pending = [c for c in self._pending if i - int(c["idx"]) <= self.max_bars]

        # Check whether any pending penetration has now been reclaimed.
        still: list[dict[str, float]] = []
        for c in self._pending:
            d = int(c["dir"])
            reclaimed = (d > 0 and bar.close > c["level"]) or (d < 0 and bar.close < c["level"])
            if reclaimed and event is None:
                event = SweepEvent(
                    direction=d,
                    level=c["level"],
                    penetration_index=int(c["idx"]),
                    reclaim_index=i,
                    penetration_sigma=c["pen"],
                    participation_rank=c["part"],
                    absorption_rank=c["absorb"],
                )
            elif not reclaimed:
                still.append(c)
        self._pending = still

        # Register a new penetration if this bar qualifies.
        if sigma > 0.0 and participation_rank == participation_rank:
            heavy = participation_rank >= self.part_min
            if heavy:
                if pivot_low == pivot_low and bar.low < pivot_low:
                    pen = math.log(pivot_low / bar.low) / sigma
                    # Gave back the excursion: closed in the upper half, or the
                    # bar absorbed unusual volume without going anywhere.
                    gave_back = bar.clv > 0.0 or (
                        absorption_rank == absorption_rank and absorption_rank >= 0.6
                    )
                    if pen >= self.min_pen and gave_back:
                        self._pending.append(
                            {
                                "dir": 1.0,
                                "level": pivot_low,
                                "idx": float(i),
                                "pen": pen,
                                "part": participation_rank,
                                "absorb": absorption_rank if absorption_rank == absorption_rank else 0.5,
                            }
                        )
                if pivot_high == pivot_high and bar.high > pivot_high:
                    pen = math.log(bar.high / pivot_high) / sigma
                    gave_back = bar.clv < 0.0 or (
                        absorption_rank == absorption_rank and absorption_rank >= 0.6
                    )
                    if pen >= self.min_pen and gave_back:
                        self._pending.append(
                            {
                                "dir": -1.0,
                                "level": pivot_high,
                                "idx": float(i),
                                "pen": pen,
                                "part": participation_rank,
                                "absorb": absorption_rank if absorption_rank == absorption_rank else 0.5,
                            }
                        )

        if event is not None:
            self.last_event = event
        return event

    def age_of_last(self) -> float:
        if self.last_event is None:
            return math.inf
        return float(self._i - self.last_event.reclaim_index)


class HTFAggregator:
    """Builds higher-timeframe bars causally from execution-timeframe bars.

    A higher-timeframe bar is emitted only when it is *complete*. Between
    emissions, downstream consumers see the previous completed bar's derived
    values held constant, which is exactly what a live system would see and is
    the reason this is implemented by aggregation rather than by resampling a
    full history.

    The multiple is derived from configuration rather than chosen by the trader,
    per the design requirement that nothing be configured.
    """

    def __init__(self, multiple: int) -> None:
        if multiple < 2:
            raise ValueError("higher-timeframe multiple must be at least 2")
        self.multiple = int(multiple)
        self._count = 0
        self._o = math.nan
        self._h = -math.inf
        self._l = math.inf
        self._v = 0.0
        self._ts = 0.0
        self.completed: Bar | None = None
        self.n_completed = 0

    def reset(self) -> None:
        self._count = 0
        self._o = math.nan
        self._h = -math.inf
        self._l = math.inf
        self._v = 0.0
        self._ts = 0.0
        self.completed = None
        self.n_completed = 0

    def update(self, bar: Bar) -> Bar | None:
        if self._count == 0:
            self._o = bar.open
            self._h = -math.inf
            self._l = math.inf
            self._v = 0.0
        self._h = max(self._h, bar.high)
        self._l = min(self._l, bar.low)
        self._v += bar.volume
        self._ts = bar.timestamp
        self._count += 1

        if self._count >= self.multiple:
            out = Bar(
                timestamp=self._ts,
                open=self._o,
                high=self._h,
                low=self._l,
                close=bar.close,
                volume=self._v,
            )
            self._count = 0
            self.completed = out
            self.n_completed += 1
            return out
        return None
