"""The decision gate and the position state machine.

``SPEC.md`` §3 and §7. Two responsibilities:

**The gate.** Ten conjunctive conditions, of which the first is primary: the
*lower credible bound* of net expectancy must clear a threshold. Everything else
is a filter on top of it. Conjunctive rather than scored, because a weighted
score lets a spectacular reading on one axis buy its way past a disqualifying
reading on another, and the disqualifying readings here -- a regime in
transition, an untradeable session phase, evidence resting on one engine -- are
disqualifying for reasons that do not trade off against expectancy.

**The state machine.** One position at a time, with `LONG -> SHORT` in a single
bar forbidden. A reversal costs two bars: `SELL`, then possibly `SHORT`. This is
realistic and is also a whipsaw brake.

**Division of labour on exits.** Stop and target are *resting orders* placed at
entry; they fill intrabar and are reported back through :meth:`Policy.notify_exit`.
This module emits `SELL`/`COVER` only for the exits that require a decision at a
bar close: the vertical barrier, and a signal reversal that has survived the
minimum hold. Modelling a stop as a bar-close decision would be a look-ahead
error in the optimistic direction, since it silently assumes the close was
available at the stop price.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Mapping, Sequence

from ..config import Config
from ..engines.edgebook import EdgeEstimate, SetupFamily
from ..types import (
    Action,
    BarContext,
    CostEstimate,
    Decision,
    EngineOutput,
    FusionResult,
    Position,
    Regime,
    RiskDecision,
    SessionPhase,
    TargetSpec,
)

__all__ = ["Policy", "GateResult", "classify_setup", "OpenPosition"]


@dataclass(slots=True)
class GateResult:
    """Outcome of the ten conditions, with every failure named."""

    passed: bool
    ev_mean: float
    ev_lcb: float
    reasons: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> "GateResult":
        self.passed = False
        self.reasons.append(reason)
        return self


@dataclass(slots=True)
class OpenPosition:
    direction: int
    entry_index: int
    entry_price: float
    spec: TargetSpec
    setup: SetupFamily
    regime: int
    bucket: int
    p_at_entry: float
    bars_held: int = 0
    #: The round-trip cost modelled at decision time, in sigma units. Charged to
    #: equity when the trade is booked, so realized P&L is net of frictions.
    cost_sigma: float = 0.0
    #: False until the executor fills the entry at the next bar's open. Barriers
    #: in ``spec`` are re-anchored to the actual fill at that moment.
    filled: bool = False


def classify_setup(
    outputs: Mapping[str, EngineOutput],
    regime: Regime,
    posterior: Sequence[float],
) -> SetupFamily:
    """Assign the bar to one of four named setup families.

    The families are the Edge Book's conditioning variable, so this function
    determines which historical outcomes a contemplated trade is compared
    against. Ordering matters: a sweep is checked first because it is the most
    specific and most distinctive configuration, and a bar that is both a sweep
    and a continuation should be judged as a sweep.
    """
    liq = outputs.get("liquidity")
    if liq is not None and liq.valid:
        # The liquidity engine reports sweep_age in .features when a sweep is
        # active (it is human-facing then) and in .diagnostics when idle. An
        # earlier version read only .diagnostics, which made SWEEP_REVERSAL
        # unreachable: across 20,000 audited bars with 2,193 active-sweep bars,
        # zero were classified into the family, so a quarter of the Edge Book's
        # taxonomy silently never existed. Read both, features first.
        age = float(liq.features.get("sweep_age", liq.diagnostics.get("sweep_age", 999)))
        if age <= 4 and abs(float(liq.diagnostics.get("sweep_score", 0.0))) > 0.05:
            return SetupFamily.SWEEP_REVERSAL

    vol = outputs.get("volatility")
    if vol is not None and vol.valid:
        comp = float(vol.features.get("compression", float("nan")))
        ratio = float(vol.features.get("vol_ratio", float("nan")))
        if comp == comp and comp > 0.7 and ratio == ratio and ratio > 1.1:
            return SetupFamily.COMPRESSION_BREAK

    if regime is Regime.REVERT or (
        len(posterior) == 4 and posterior[int(Regime.REVERT)] > posterior[int(Regime.TREND)]
    ):
        return SetupFamily.RANGE_FADE
    return SetupFamily.CONTINUATION


class Policy:
    """Gate plus state machine. One instance per instrument."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.position: Position = Position.FLAT
        self.open: OpenPosition | None = None
        self._recent_entries: Deque[int] = deque()
        self._index = -1
        self._last_exit_index = -(10**9)
        self.n_entries = 0
        self.n_exits = 0
        #: The position most recently closed by a bar-close decision (vertical
        #: barrier, reversal, session close). The executor reads and books it.
        #: An earlier version discarded it here, so those exits produced no
        #: trade record and no equity change -- 4 of 62 positions in the audit
        #: run simply vanished from the accounting.
        self.last_closed: OpenPosition | None = None

    def reset(self) -> None:
        self.position = Position.FLAT
        self.open = None
        self._recent_entries.clear()
        self._index = -1
        self._last_exit_index = -(10**9)
        self.n_entries = 0
        self.n_exits = 0
        self.last_closed = None

    # ------------------------------------------------------------------ #
    @property
    def trade_rate(self) -> float:
        """Entries per hundred bars over the trailing window."""
        return len(self._recent_entries) * 1.0

    def _prune(self) -> None:
        cutoff = self._index - 100
        while self._recent_entries and self._recent_entries[0] < cutoff:
            self._recent_entries.popleft()

    # ------------------------------------------------------------------ #
    def evaluate_gate(
        self,
        ctx: BarContext,
        fusion: FusionResult,
        regime: Regime,
        posterior: Sequence[float],
        hazard: float,
        edge: EdgeEstimate,
        costs: CostEstimate,
        spec: TargetSpec | None,
        risk: RiskDecision | None,
        outputs: Mapping[str, EngineOutput],
    ) -> GateResult:
        cfg = self.cfg
        # Expectancy is the Edge Book's posterior for this cell, in sigma units,
        # net of the modelled round trip. The gate acts on the lower credible
        # bound: we require 90% posterior credence that net expectancy is
        # positive, not merely a positive point estimate.
        rt = costs.round_trip_sigma
        ev_mean = edge.mean - rt
        ev_lcb = edge.lcb - rt
        g = GateResult(passed=True, ev_mean=ev_mean, ev_lcb=ev_lcb)

        if self.position is not Position.FLAT:
            return g.fail("already in a position")
        if not ctx.features.valid:
            return g.fail("features not valid")
        if fusion.direction == 0:
            return g.fail("no directional evidence")
        if spec is None:
            return g.fail("no volatility forecast for barriers")
        if fusion.direction < 0 and not cfg.allow_shorts:
            return g.fail("shorts disabled")

        phase = ctx.features.phase
        if not phase.is_tradeable or phase.is_auction:
            return g.fail(f"session phase {phase.name.lower()}")

        if regime is Regime.QUIET:
            g.fail("quiet regime: no exploitable structure")
        if hazard > cfg.regime_hazard_max:
            g.fail(f"regime transition hazard {hazard:.2f} > {cfg.regime_hazard_max:.2f}")

        if not (ev_lcb > cfg.ev_lcb_min_sigma):
            g.fail(
                f"expectancy lower bound {ev_lcb:+.3f}s does not clear "
                f"{cfg.ev_lcb_min_sigma:+.3f}s (mean {ev_mean:+.3f}s, cost {rt:.3f}s)"
            )
        if not (fusion.p_success >= cfg.p_min):
            g.fail(f"probability {fusion.p_success:.3f} below {cfg.p_min:.2f}")
        if not (fusion.ebe >= cfg.ebe_min):
            g.fail(
                f"effective breadth {fusion.ebe:.2f} below {cfg.ebe_min:.2f}: "
                "agreement rests on too few independent engines"
            )

        invalid = [n for n, o in outputs.items() if not o.valid]
        if len(invalid) > max(2, len(outputs) // 3):
            g.fail(f"{len(invalid)} engines invalid")

        self._prune()
        if len(self._recent_entries) >= cfg.max_trades_per_100_bars:
            g.fail(
                f"trade rate limiter: {len(self._recent_entries)} entries in the last 100 bars"
            )

        if risk is not None:
            if risk.risk_fraction <= 0.0:
                g.fail("risk layer returned zero size: " + (", ".join(risk.blocks) or "throttled"))

        return g

    # ------------------------------------------------------------------ #
    def check_exit(self, ctx: BarContext, fusion: FusionResult) -> tuple[Action, str] | None:
        """Bar-close exits only: vertical barrier and confirmed reversal."""
        if self.position is Position.FLAT or self.open is None:
            return None
        op = self.open
        act = Action.SELL if self.position is Position.LONG else Action.COVER

        if op.bars_held >= op.spec.max_holding_bars:
            return act, f"vertical barrier at {op.spec.max_holding_bars} bars"

        if op.bars_held >= self.cfg.min_holding_bars:
            # A reversal must be a positive statement about the other side, not
            # merely a weakening of this one -- otherwise every position is
            # closed by noise on the second bar.
            opposite = fusion.direction == -op.direction
            conviction = fusion.p_success >= self.cfg.p_exit
            if opposite and conviction and fusion.ebe >= self.cfg.ebe_min * 0.7:
                return act, f"reversal signal at p={fusion.p_success:.2f}"

        if not ctx.features.phase.is_tradeable:
            return act, "session closing"
        return None

    # ------------------------------------------------------------------ #
    def step(
        self,
        ctx: BarContext,
        fusion: FusionResult,
        regime: Regime,
        posterior: Sequence[float],
        hazard: float,
        edge: EdgeEstimate,
        costs: CostEstimate,
        spec: TargetSpec | None,
        risk: RiskDecision | None,
        outputs: Mapping[str, EngineOutput],
        setup: SetupFamily,
        bucket: int,
    ) -> tuple[Action, GateResult, str]:
        """Advance one bar and return the action to emit."""
        self._index = ctx.features.index
        if self.open is not None:
            self.open.bars_held += 1

        exit_sig = self.check_exit(ctx, fusion)
        if exit_sig is not None:
            action, why = exit_sig
            self.last_closed = self.open
            self._close(self._index)
            g = GateResult(passed=False, ev_mean=edge.mean, ev_lcb=edge.lcb, reasons=[why])
            return action, g, why

        g = self.evaluate_gate(
            ctx, fusion, regime, posterior, hazard, edge, costs, spec, risk, outputs
        )
        if not g.passed or spec is None:
            return Action.NO_TRADE, g, "; ".join(g.reasons) or "no setup"

        action = Action.BUY if fusion.direction > 0 else Action.SHORT
        self.position = Position.LONG if fusion.direction > 0 else Position.SHORT
        # entry_price here is PROVISIONAL (the decision close). The executor
        # fills at the next bar's open and re-anchors the barriers to the fill
        # -- which is what SPEC.md section 2 rule 3 requires, what
        # execute_at_index has always claimed, and what the triple-barrier
        # labeller the Edge Book is trained on actually does. An earlier
        # version filled at this close, so the live system was trading a
        # different game from the one it had learned.
        self.open = OpenPosition(
            direction=fusion.direction,
            entry_index=self._index,
            entry_price=ctx.features.close,
            spec=spec,
            setup=setup,
            regime=int(regime),
            bucket=bucket,
            p_at_entry=fusion.p_success,
            cost_sigma=costs.round_trip_sigma if math.isfinite(costs.round_trip_sigma) else 0.0,
            filled=False,
        )
        self._recent_entries.append(self._index)
        self.n_entries += 1
        return action, g, "entry"

    # ------------------------------------------------------------------ #
    def notify_exit(self, index: int) -> OpenPosition | None:
        """Called by the executor when a resting stop or target filled."""
        op = self.open
        self._close(index)
        return op

    def _close(self, index: int) -> None:
        self.position = Position.FLAT
        self.open = None
        self._last_exit_index = index
        self.n_exits += 1
