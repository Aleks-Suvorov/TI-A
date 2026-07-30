"""Shared engine scaffolding.

Engines are small and there are many of them, so the boilerplate that enforces
the contract lives here rather than being reimplemented ten times with ten
subtly different bugs. :class:`BaseEngine` guarantees:

*   ``valid=False`` and ``score=0`` until warmup completes, so no engine can vote
    on a half-filled window;
*   scores and reliabilities are clamped into range before the ``EngineOutput``
    constructor's assertions see them, turning what would be a crash on a NaN
    into a defensible abstention;
*   a NaN anywhere in the inputs produces an abstention rather than propagating.

The last point is the important one. The natural failure of a numeric pipeline is
to emit NaN, and NaN compared against a threshold is silently ``False`` -- which
in a trading system means a filter that has stopped working looks exactly like a
filter that is passing. Abstention has to be explicit.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from ..config import Config
from ..types import BarContext, EngineOutput, clip

__all__ = ["BaseEngine", "squash", "linear_score", "band_score"]


def squash(x: float, scale: float = 1.0) -> float:
    """Map an unbounded statistic to ``[-1, 1]`` via ``tanh``.

    ``scale`` is the value of ``x`` mapped to ``tanh(1) = 0.762``. Using tanh
    rather than clipping matters downstream: the fusion layer inverts it with
    ``artanh``, so a clipped score would map to infinite log-odds and a single
    saturated engine could dominate the pool.
    """
    if x != x or scale <= 0.0:
        return 0.0
    return math.tanh(x / scale)


def linear_score(x: float, lo: float, hi: float) -> float:
    """Map ``[lo, hi]`` linearly onto ``[-1, 1]``, clamped outside."""
    if x != x or hi <= lo:
        return 0.0
    return clip(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def band_score(x: float, low: float, high: float) -> float:
    """+1 inside ``[low, high]``, falling to -1 outside it.

    For features where the *middle* is favourable -- moderate volatility, moderate
    participation -- rather than the extremes.
    """
    if x != x:
        return 0.0
    if low <= x <= high:
        return 1.0
    width = max(high - low, 1e-9)
    d = (low - x) / width if x < low else (x - high) / width
    return clip(1.0 - 2.0 * d, -1.0, 1.0)


class BaseEngine:
    """Base class implementing the :class:`tia.types.Engine` protocol."""

    name: str = "base"
    warmup: int = 0

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._n = 0
        self.last: EngineOutput = EngineOutput.abstain(self.name)

    # -- lifecycle ------------------------------------------------------ #
    def reset(self) -> None:
        self._n = 0
        self.last = EngineOutput.abstain(self.name)
        self._reset()

    def _reset(self) -> None:  # pragma: no cover - overridden where needed
        """Subclass hook. Base state is cleared by :meth:`reset`."""

    @property
    def count(self) -> int:
        return self._n

    @property
    def warm(self) -> bool:
        return self._n >= self.warmup

    # -- main entry point ----------------------------------------------- #
    def update(self, ctx: BarContext) -> EngineOutput:
        self._n += 1
        if not ctx.features.valid or not self.warm:
            self.last = EngineOutput.abstain(
                self.name, "warmup" if not self.warm else "features invalid"
            )
            return self.last
        try:
            out = self._compute(ctx)
        except Exception as exc:  # pragma: no cover - defensive
            # An engine that raises must not take the system down mid-session;
            # it must fall silent and be visible in the diagnostics.
            self.last = EngineOutput(
                name=self.name,
                score=0.0,
                reliability=0.0,
                notes=(f"error: {type(exc).__name__}",),
                valid=False,
            )
            return self.last
        self.last = out
        return out

    def _compute(self, ctx: BarContext) -> EngineOutput:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- helpers -------------------------------------------------------- #
    def emit(
        self,
        score: float,
        reliability: float,
        features: Mapping[str, float] | None = None,
        diagnostics: Mapping[str, float] | None = None,
        notes: Sequence[str] = (),
    ) -> EngineOutput:
        """Build a contract-valid output, converting NaN into abstention."""
        if score != score or reliability != reliability:
            return EngineOutput.abstain(self.name, "non-finite score")
        return EngineOutput(
            name=self.name,
            score=clip(score, -1.0, 1.0),
            reliability=clip(reliability, 0.0, 1.0),
            features=dict(features or {}),
            diagnostics=dict(diagnostics or {}),
            notes=tuple(notes),
            valid=True,
        )

    def abstain(self, reason: str) -> EngineOutput:
        return EngineOutput.abstain(self.name, reason)
