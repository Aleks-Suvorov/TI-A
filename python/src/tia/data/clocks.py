"""Alternative sampling clocks.

Sampling by the calendar is a convention, not a law. Information does not arrive
uniformly in time -- it arrives with trading activity -- so returns sampled on a
clock that advances with transacted value are closer to independent and
identically distributed, and closer to normal, than returns sampled every five
minutes. This is the "volume clock" argument (Easley, López de Prado & O'Hara).

The practical consequences are real: every statistical procedure downstream of
sampling, from the variance-ratio test's asymptotics to the Normal-Inverse-Gamma
likelihood of the Edge Book, assumes something closer to IID normality than clock
time delivers.

**Why it is off by default.** Dollar bars need a finer-grained feed than the
execution timeframe to construct without severe rounding, they make bar timestamps
irregular (complicating session tagging and any comparison with a chart), and
they change the meaning of "bars" in every window parameter. The gain is real but
the cost is architectural, so ``Config.use_volume_clock`` defaults to ``False``
and the roadmap treats enabling it as a deliberate project.
"""

from __future__ import annotations

import math
from typing import Iterable, Iterator, Literal

from ..types import Bar
from ..features.rolling import EWMA

__all__ = ["ActivityClock", "resample_by_activity"]


class ActivityClock:
    """Aggregates bars until a target amount of activity has transacted.

    ``mode`` selects the activity measure:

    ``"dollar"``  dollar volume -- the default and the best-behaved, because it is
                  comparable across price levels and across instruments
    ``"volume"``  share/contract volume -- appropriate when the instrument's price
                  level is stable and the contract size is the natural unit
    ``"trades"``  trade count -- the closest available proxy for the number of
                  information events, when the feed provides it

    The target is adaptive: it tracks an exponentially weighted average of
    activity per input bar, multiplied by ``bars_per_output``. A fixed target
    would produce hundreds of bars a day in an active period and none in a quiet
    one, which reintroduces the very non-uniformity the clock exists to remove.
    """

    def __init__(
        self,
        bars_per_output: float = 1.0,
        mode: Literal["dollar", "volume", "trades"] = "dollar",
        target_halflife: float = 500.0,
        min_input_bars: int = 1,
        warmup_bars: int = 50,
    ) -> None:
        self.mode = mode
        self.bars_per_output = float(bars_per_output)
        self.min_input_bars = int(min_input_bars)
        self.warmup_bars = int(warmup_bars)
        self._activity_ewma = EWMA(target_halflife)
        self._acc = 0.0
        self._n_in = 0
        self._o = math.nan
        self._h = -math.inf
        self._l = math.inf
        self._v = 0.0
        self._trades = 0.0
        self._delta = 0.0
        self._has_delta = True
        self._ts = 0.0
        self._seen = 0

    def reset(self) -> None:
        self._activity_ewma.reset()
        self._acc = 0.0
        self._n_in = 0
        self._o = math.nan
        self._h = -math.inf
        self._l = math.inf
        self._v = 0.0
        self._trades = 0.0
        self._delta = 0.0
        self._has_delta = True
        self._ts = 0.0
        self._seen = 0

    # ------------------------------------------------------------------ #
    def _activity(self, bar: Bar) -> float:
        if self.mode == "dollar":
            return bar.dollar_volume
        if self.mode == "volume":
            return bar.volume
        return float(bar.trades) if bar.trades is not None else 1.0

    @property
    def target(self) -> float:
        a = self._activity_ewma.value
        if a != a or a <= 0.0:
            return math.inf
        return a * self.bars_per_output

    @property
    def warm(self) -> bool:
        return self._seen >= self.warmup_bars

    # ------------------------------------------------------------------ #
    def update(self, bar: Bar) -> Bar | None:
        """Feed one input bar; returns an output bar when the target is met."""
        act = self._activity(bar)
        self._activity_ewma.update(act)
        self._seen += 1

        if self._n_in == 0:
            self._o = bar.open
            self._h = -math.inf
            self._l = math.inf
            self._v = 0.0
            self._trades = 0.0
            self._delta = 0.0
            self._has_delta = True
            self._acc = 0.0

        self._h = max(self._h, bar.high)
        self._l = min(self._l, bar.low)
        self._v += bar.volume
        if bar.trades is not None:
            self._trades += float(bar.trades)
        if bar.delta is None:
            self._has_delta = False
        else:
            self._delta += float(bar.delta)
        self._acc += act
        self._ts = bar.timestamp
        self._n_in += 1

        if not self.warm:
            # During warmup, pass input bars straight through one-for-one so the
            # downstream warmup is not doubled.
            return self._emit(bar.close)

        if self._acc >= self.target and self._n_in >= self.min_input_bars:
            return self._emit(bar.close)
        return None

    def _emit(self, close: float) -> Bar:
        out = Bar(
            timestamp=self._ts,
            open=self._o,
            high=self._h,
            low=self._l,
            close=close,
            volume=self._v,
            trades=self._trades if self._trades > 0.0 else None,
            delta=self._delta if self._has_delta else None,
        )
        self._n_in = 0
        self._acc = 0.0
        return out

    def flush(self, close: float) -> Bar | None:
        """Emit a partially-filled bar. For end-of-history only.

        A flushed bar has less activity behind it than its peers and so is not
        comparable; the backtester discards it rather than trading it.
        """
        if self._n_in == 0:
            return None
        return self._emit(close)


def resample_by_activity(bars: Iterable[Bar], **kw: object) -> Iterator[Bar]:
    """Convenience wrapper: stream input bars, yield activity bars."""
    clock = ActivityClock(**kw)  # type: ignore[arg-type]
    for bar in bars:
        out = clock.update(bar)
        if out is not None:
            yield out
