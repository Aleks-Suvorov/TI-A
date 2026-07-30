"""Bar validation and data-integrity screening.

Every one of these checks exists because the corresponding defect silently
manufactures or destroys backtest performance. A duplicated bar doubles a
return; a non-monotonic timestamp reorders cause and effect; an unadjusted split
appears as a 50% single-bar move that a momentum engine will read as the strongest
displacement in the sample.

The validator is streaming and *rejecting* rather than repairing. Repairing bad
data means guessing, and a guess that enters the feature kernel is indistinguishable
from a signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from ..types import Bar

__all__ = ["ValidationIssue", "ValidationReport", "BarValidator", "clean_stream"]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    index: int
    timestamp: float
    kind: str
    detail: str
    fatal: bool


@dataclass(slots=True)
class ValidationReport:
    n_seen: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.issues:
            out[i.kind] = out.get(i.kind, 0) + 1
        return out

    def summary(self) -> str:
        lines = [
            f"bars seen {self.n_seen}  accepted {self.n_accepted}  rejected {self.n_rejected}",
        ]
        for kind, n in sorted(self.counts().items(), key=lambda kv: -kv[1]):
            lines.append(f"  {kind:24s} {n}")
        return "\n".join(lines)

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / self.n_seen if self.n_seen else math.nan


class BarValidator:
    """Streaming validator.

    ``check`` returns ``None`` for an acceptable bar or a
    :class:`ValidationIssue` describing why it was rejected. Non-fatal issues are
    recorded and the bar is still accepted -- a zero-volume bar is suspicious but
    real on thin instruments, whereas a timestamp that moves backwards is not
    recoverable.
    """

    def __init__(
        self,
        max_abs_return: float = 0.75,
        split_ratio_tolerance: float = 0.02,
        max_gap_multiple: float = 20.0,
        allow_zero_volume: bool = True,
    ) -> None:
        self.max_abs_return = float(max_abs_return)
        self.split_tol = float(split_ratio_tolerance)
        self.max_gap_multiple = float(max_gap_multiple)
        self.allow_zero_volume = bool(allow_zero_volume)
        self.report = ValidationReport()
        self._prev: Bar | None = None
        self._prev_dt: float = math.nan
        self._i = -1

    def reset(self) -> None:
        self.report = ValidationReport()
        self._prev = None
        self._prev_dt = math.nan
        self._i = -1

    # ------------------------------------------------------------------ #
    def check(self, bar: Bar) -> ValidationIssue | None:
        self._i += 1
        i = self._i
        self.report.n_seen += 1

        def issue(kind: str, detail: str, fatal: bool = True) -> ValidationIssue:
            iss = ValidationIssue(i, bar.timestamp, kind, detail, fatal)
            self.report.add(iss)
            return iss

        if bar.partial:
            self.report.n_rejected += 1
            return issue("partial_bar", "bar not closed")

        if not all(math.isfinite(v) for v in (bar.open, bar.high, bar.low, bar.close, bar.volume)):
            self.report.n_rejected += 1
            return issue("non_finite", "NaN or infinity in bar fields")

        prev = self._prev
        if prev is not None:
            if bar.timestamp < prev.timestamp:
                self.report.n_rejected += 1
                return issue(
                    "time_reversal",
                    f"timestamp {bar.timestamp} precedes previous {prev.timestamp}",
                )
            if bar.timestamp == prev.timestamp:
                self.report.n_rejected += 1
                return issue("duplicate_timestamp", f"repeated timestamp {bar.timestamp}")

            dt = bar.timestamp - prev.timestamp
            if self._prev_dt == self._prev_dt and self._prev_dt > 0.0:
                if dt > self.max_gap_multiple * self._prev_dt:
                    # Non-fatal: weekends, holidays and halts are legitimate.
                    # Recorded because a gap invalidates rolling windows that
                    # assume uniform spacing, and the monitoring layer needs to
                    # know it happened.
                    issue(
                        "large_time_gap",
                        f"gap of {dt:.0f}s vs typical {self._prev_dt:.0f}s",
                        fatal=False,
                    )
            self._prev_dt = dt if not (self._prev_dt == self._prev_dt) else 0.5 * (self._prev_dt + dt)

            if prev.close > 0.0:
                r = math.log(bar.close / prev.close)
                if abs(r) > self.max_abs_return:
                    ratio = bar.close / prev.close
                    kind, detail = self._classify_extreme(ratio, r)
                    self.report.n_rejected += 1
                    return issue(kind, detail)

        if bar.volume <= 0.0 and not self.allow_zero_volume:
            self.report.n_rejected += 1
            return issue("zero_volume", "volume is zero")
        if bar.volume <= 0.0:
            issue("zero_volume", "volume is zero", fatal=False)

        if bar.high == bar.low:
            issue("zero_range", "high equals low", fatal=False)

        self._prev = bar
        self.report.n_accepted += 1
        return None

    # ------------------------------------------------------------------ #
    def _classify_extreme(self, ratio: float, log_ret: float) -> tuple[str, str]:
        """Distinguish a probable unadjusted corporate action from a real move.

        Splits and reverse splits land on simple ratios. A 2:1 split shows a
        close ratio near 0.5; a 1:10 reverse split near 10. Real single-bar moves
        of that magnitude happen, but they do not cluster on round numbers, so
        proximity to a simple ratio is genuine evidence about which we are seeing.
        """
        for num, den in (
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 10), (1, 20),
            (2, 1), (3, 1), (4, 1), (5, 1), (10, 1), (20, 1),
            (2, 3), (3, 2),
        ):
            target = num / den
            if abs(ratio / target - 1.0) < self.split_tol:
                return (
                    "probable_split",
                    f"close ratio {ratio:.4f} is within tolerance of {num}:{den}; "
                    "history is likely unadjusted",
                )
        return (
            "extreme_return",
            f"log return {log_ret:+.3f} exceeds max_abs_return {self.max_abs_return}",
        )


def clean_stream(
    bars: Iterable[Bar], validator: BarValidator | None = None
) -> Iterator[tuple[Bar, ValidationIssue | None]]:
    """Yield only acceptable bars, along with any non-fatal issue recorded.

    Rejected bars are dropped entirely rather than interpolated. Downstream
    rolling windows therefore see a shorter but internally consistent series,
    which is the right trade: a fabricated bar propagates into every feature for
    the length of every window that contains it.
    """
    v = validator or BarValidator()
    for bar in bars:
        iss = v.check(bar)
        if iss is not None and iss.fatal:
            continue
        yield bar, (iss if iss is not None else None)
