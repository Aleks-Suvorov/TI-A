"""The end-to-end streaming pipeline.

One object, one bar at a time, one :class:`Decision` out. This is simultaneously
the research path and the live path, which is not a convenience -- it is the
mechanism that makes backtest and live agree. There is no separate "vectorised
backtest" implementation that could drift from the streaming one, and no place
for a look-ahead to enter on one side only.

Order of operations within a bar, which matters:

1. validate and reject the bar if it is not usable
2. update the feature kernel
3. advance the regime filter, and publish the posterior to the engines
4. update every engine with the same :class:`BarContext`
5. feed engine scores to the correlation estimator, then pool
6. resolve any open position whose resting barriers were touched
7. classify the setup, look up the Edge Book cell, price the costs
8. size, gate, and emit an action

Step 6 sits after pooling so that the resolution of a position and the evaluation
of a new one see the same information, and before step 8 so that a position
closed this bar frees the book for a new entry no earlier than the next bar --
which is what the state machine's two-bar reversal rule requires anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from .config import Config
from .data.sessions import DAILY, SessionSpec
from .data.validate import BarValidator
from .decision.barriers import BarrierPlanner
from .decision.costs import CostModel
from .decision.explain import build_card
from .decision.policy import OpenPosition, Policy, classify_setup
from .engines.base import BaseEngine
from .engines.behavioral import BehavioralEngine
from .engines.crossasset import CrossAssetEngine
from .engines.edgebook import EdgeBook, EdgeEstimate, SetupFamily
from .engines.liquidity import LiquidityEngine
from .engines.momentum import MomentumEngine
from .engines.positioning import PositioningEngine
from .engines.regime import RegimeEngine
from .engines.structure import StructureEngine
from .engines.trend import TrendEngine
from .engines.volatility import VolatilityEngine
from .features.kernel import FeatureKernel
from .fusion.calibration import OnlineCalibrator
from .fusion.calop import CALOP
from .risk.limits import RiskLimits
from .risk.sizing import Sizer
from .types import (
    Action,
    Bar,
    BarContext,
    Decision,
    EngineOutput,
    ExogenousSnapshot,
    ExplainCard,
    FusionResult,
    Position,
    Regime,
)

__all__ = ["TIA", "TradeRecord"]


@dataclass(slots=True)
class TradeRecord:
    """A resolved trade, in the form the Edge Book and the metrics both want."""

    entry_index: int
    exit_index: int
    direction: int
    entry_price: float
    exit_price: float
    ret_sigma: float
    outcome: int  #: +1 target, -1 stop, 0 timeout or discretionary exit
    setup: int
    regime: int
    bucket: int
    p_at_entry: float
    risk_fraction: float
    equity_after: float
    reason: str = ""
    #: Modelled round-trip cost at decision time, sigma units. Equity is charged
    #: this; ret_sigma stays GROSS because the gate nets costs separately and
    #: the Edge Book must learn the same (gross) quantity it is compared to.
    cost_sigma: float = 0.0

    @property
    def net_sigma(self) -> float:
        return self.ret_sigma - self.cost_sigma


class TIA:
    """The complete system for one instrument."""

    def __init__(
        self,
        cfg: Config | None = None,
        session: SessionSpec = DAILY,
        edge_book: EdgeBook | None = None,
        initial_equity: float = 1.0,
        learn: bool = True,
        history_limit: int | None = None,
    ) -> None:
        self.cfg = cfg or Config()
        self.kernel = FeatureKernel(self.cfg, session)
        self.validator = BarValidator()

        self.regime_engine = RegimeEngine(self.cfg)
        self.engines: list[BaseEngine] = [
            self.regime_engine,
            TrendEngine(self.cfg),
            MomentumEngine(self.cfg),
            LiquidityEngine(self.cfg),
            VolatilityEngine(self.cfg),
            StructureEngine(self.cfg),
            BehavioralEngine(self.cfg),
            CrossAssetEngine(self.cfg),
            PositioningEngine(self.cfg),
        ]
        self.calop = CALOP([e.name for e in self.engines], self.cfg)
        self.calibrator = OnlineCalibrator(min_samples=self.cfg.calibration_min_samples)
        # The Edge Book may be shared across instruments. That sharing is the
        # whole point of docs/01-THEORY.md §12: selectivity is per instrument,
        # statistical power comes from pooling across the universe.
        self.edge_book = edge_book if edge_book is not None else EdgeBook(self.cfg)
        self.planner = BarrierPlanner(self.cfg)
        self.costs = CostModel(self.cfg)
        self.policy = Policy(self.cfg)
        self.sizer = Sizer(self.cfg)
        self.limits = RiskLimits(self.cfg, initial_equity)
        self.learn = bool(learn)

        # Research keeps everything; a live service must not. At one-minute bars
        # an unbounded decision log grows by roughly half a gigabyte a year, and
        # a signal system that quietly consumes memory for months is a system
        # that fails at an unpredictable moment for an unrelated-looking reason.
        # None means unbounded, which is the right default for a backtest.
        self.history_limit = history_limit
        self.equity = float(initial_equity)
        self.trades: list[TradeRecord] = []
        self.decisions: list[Decision] = []
        self._index = -1
        self._open_risk: float = 0.0
        self._open_entry_price: float = math.nan

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.kernel.reset()
        self.validator.reset()
        for e in self.engines:
            e.reset()
        self.calop.reset()
        self.calibrator.reset()
        self.edge_book.reset()
        self.policy.reset()
        self.limits.reset(self.equity)
        self.trades.clear()
        self.decisions.clear()
        self._index = -1
        self._open_risk = 0.0
        self._open_entry_price = math.nan

    # ------------------------------------------------------------------ #
    def on_bar(self, bar: Bar, exog: ExogenousSnapshot | None = None) -> Decision:
        """Process one closed bar and return the decision for it."""
        issue = self.validator.check(bar)
        if issue is not None and issue.fatal:
            return self._null_decision(bar, f"bar rejected: {issue.kind}")

        self._index += 1
        f = self.kernel.update(bar)
        self.edge_book.tick()

        st = self.regime_engine.step(self.kernel.regime_features())
        posterior = st.as_tuple()
        for e in self.engines:
            if hasattr(e, "regime_probs"):
                e.regime_probs = posterior  # type: ignore[attr-defined]
        # The structure engine wants the confirmed higher-high/higher-low bias,
        # which lives on the pivot tracker rather than in the flat snapshot.
        for e in self.engines:
            if isinstance(e, StructureEngine):
                e.structure_bias = self.kernel.pivots.structure_bias()

        ctx = BarContext(
            bar=bar,
            features=f,
            exog=exog or ExogenousSnapshot(),
            position=self.policy.position,
            bars_in_position=self.policy.open.bars_held if self.policy.open else 0,
        )

        outputs: dict[str, EngineOutput] = {}
        for e in self.engines:
            outputs[e.name] = e.update(ctx)

        self.calop.observe(outputs)
        fusion = self.calop.fuse(outputs, calibrate=self.calibrator)

        self._resolve_open_position(bar, f)

        setup = classify_setup(outputs, st.dominant, posterior)
        bucket = self.edge_book.bucket_of(fusion.log_odds)
        edge = self.edge_book.estimate(int(st.dominant), int(setup), bucket)

        stress_mult = 1.0 + 1.5 * posterior[int(Regime.STRESS)]
        # A measured spread always beats an estimated one. The range-based
        # estimator is a fallback for bar-only environments and is conservative
        # by construction; using it when a real quote feed exists would refuse
        # trades on liquid instruments for no reason.
        ex = ctx.exog
        spread_rel = (
            float(ex.spread)
            if ex.spread is not None and math.isfinite(ex.spread) and ex.spread >= 0.0
            else self.kernel.micro.out.spread
        )
        cost = self.costs.estimate(
            f,
            spread_rel=spread_rel,
            participation_fraction=self.cfg.default_participation,
            stress_multiplier=stress_mult,
        )

        spec = None
        if fusion.direction != 0:
            spec = self.planner.plan(fusion.direction, bar.close, f, st.dominant, posterior)

        self.limits.on_equity(self.equity)
        lim = self.limits.check()
        risk = None
        if spec is not None:
            risk = self.sizer.size(
                edge,
                spec,
                cost,
                equity=self.equity,
                price=bar.close,
                drawdown_throttle=lim.throttle,
                blocks=tuple(lim.triggered),
            )

        action, gate, why = self.policy.step(
            ctx, fusion, st.dominant, posterior, st.hazard, edge, cost, spec, risk, outputs,
            setup, bucket,
        )

        if action.is_exit and self.policy.last_closed is not None:
            # A bar-close exit: vertical barrier, confirmed reversal, or session
            # close. Booked at this bar's close, matching the labeller's
            # vertical-barrier convention, with outcome 0 (neither barrier).
            # An earlier version dropped these entirely -- the position left the
            # state machine but no trade record and no equity change existed,
            # and in learn mode the Edge Book never saw a timeout outcome,
            # biasing it toward barrier touches.
            closed = self.policy.last_closed
            self.policy.last_closed = None
            self._book_trade(closed, bar.close, 0, why)

        if action.is_entry and risk is not None:
            self._open_risk = risk.risk_fraction
            self._open_entry_price = bar.close

        card = build_card(
            action, f, fusion, st.dominant, outputs, edge, cost, spec, risk, setup,
            veto_reasons=() if action.is_entry else tuple(gate.reasons[:3]),
        )

        dec = Decision(
            action=action,
            decided_at_index=self._index,
            execute_at_index=self._index + self.cfg.execution_lag_bars,
            timestamp=bar.timestamp,
            position_before=ctx.position,
            position_after=self.policy.position,
            target=spec if action.is_entry else None,
            fusion=fusion,
            risk=risk if action.is_entry else None,
            costs=cost,
            expected_value_sigma=gate.ev_mean,
            expected_value_lcb=gate.ev_lcb,
            regime=st.dominant,
            regime_posterior=posterior,
            engine_outputs=outputs,
            card=card,
        )
        self.decisions.append(dec)
        if self.history_limit is not None:
            # Trim in blocks rather than one at a time: a list pop from the front
            # is O(n), and doing it every bar would make the pipeline quadratic.
            if len(self.decisions) > 2 * self.history_limit:
                del self.decisions[: len(self.decisions) - self.history_limit]
            if len(self.trades) > 2 * self.history_limit:
                del self.trades[: len(self.trades) - self.history_limit]
        return dec

    # ------------------------------------------------------------------ #
    def _resolve_open_position(self, bar: Bar, f) -> None:
        """Check the resting stop and target against this bar's range.

        Pessimistic throughout, per ``SPEC.md`` §2.7: if the bar's range spans
        both barriers the stop is taken, and if the bar opened beyond the stop
        the fill is the open rather than the stop price. Bar data cannot resolve
        the intrabar sequence, so the ambiguity is always resolved against us.
        The optimistic convention can manufacture most of a strategy's apparent
        edge, and it does so invisibly.
        """
        op = self.policy.open
        if op is None:
            return

        if not op.filled:
            # Execution: the decision was made at the previous close; the fill
            # is THIS bar's open, exactly as execute_at_index has always
            # claimed and exactly as the triple-barrier labeller that trained
            # the Edge Book assumes. Barriers keep their sigma distances but
            # are re-anchored to the real fill, so they always straddle it and
            # a "gap at fill" is impossible by construction.
            import dataclasses as _dc

            op.entry_price = bar.open
            op.spec = _dc.replace(op.spec, entry_ref=bar.open)
            op.filled = True
            # Fall through: the fill bar's own range is barrier-checked below,
            # matching the labeller, which begins checking on the fill bar.

        spec = op.spec
        stop_p, tgt_p = spec.stop_price, spec.target_price
        d = spec.direction

        hit_stop = (bar.low <= stop_p) if d > 0 else (bar.high >= stop_p)
        hit_tgt = (bar.high >= tgt_p) if d > 0 else (bar.low <= tgt_p)
        if not (hit_stop or hit_tgt):
            return

        if hit_stop:
            gapped = (bar.open < stop_p) if d > 0 else (bar.open > stop_p)
            exit_price = bar.open if gapped else stop_p
            outcome = -1
            reason = "stop" + (" (gapped through)" if gapped else "")
        else:
            exit_price = tgt_p
            outcome = 1
            reason = "target"

        self._book_trade(op, exit_price, outcome, reason)
        self.policy.notify_exit(self._index)

    def _book_trade(self, op: OpenPosition, exit_price: float, outcome: int, reason: str) -> None:
        spec = op.spec
        ret = math.log(exit_price / op.entry_price) * spec.direction
        ret_sigma = ret / spec.sigma if spec.sigma > 0.0 else 0.0

        # Equity is charged the round trip modelled at decision time. An
        # earlier version charged nothing: the gate netted costs for the
        # DECISION while the reported P&L was cost-free -- the single easiest
        # way for a backtest to flatter itself. ret_sigma stays gross because
        # the Edge Book must learn the same quantity the gate later nets.
        net_sigma = ret_sigma - op.cost_sigma
        pnl_frac = 0.0
        if spec.stop_sigma > 0.0 and self._open_risk > 0.0:
            pnl_frac = (net_sigma / spec.stop_sigma) * self._open_risk
        self.equity *= 1.0 + pnl_frac
        self.limits.on_equity(self.equity)
        self.limits.on_trade(net_sigma)

        if self.learn:
            # Because only one position is open at a time, labels never overlap
            # and the uniqueness weight of docs/01-THEORY.md §8.3 is exactly 1.
            # The batch research path still computes uniqueness explicitly,
            # because there it evaluates every candidate bar, not just the ones
            # actually traded.
            self.edge_book.observe(op.regime, int(op.setup), op.bucket, ret_sigma, weight=1.0)
            self.calibrator.record(op.p_at_entry, 1.0 if outcome > 0 else 0.0)

        self.trades.append(
            TradeRecord(
                entry_index=op.entry_index,
                exit_index=self._index,
                direction=spec.direction,
                entry_price=op.entry_price,
                exit_price=exit_price,
                ret_sigma=ret_sigma,
                outcome=outcome,
                setup=int(op.setup),
                regime=op.regime,
                bucket=op.bucket,
                p_at_entry=op.p_at_entry,
                risk_fraction=self._open_risk,
                equity_after=self.equity,
                reason=reason,
                cost_sigma=op.cost_sigma,
            )
        )
        self._open_risk = 0.0

    # ------------------------------------------------------------------ #
    def _null_decision(self, bar: Bar, why: str) -> Decision:
        """A decision for a bar that never reached the kernel.

        The reason is carried on the card rather than discarded. A rejected bar
        is one of the few places where the system produces NO TRADE for a reason
        that is not a gate condition, and an operator reading a run of silent
        NO TRADEs deserves to be told the feed is the problem.
        """
        return Decision(
            action=Action.NO_TRADE,
            decided_at_index=self._index,
            execute_at_index=self._index + self.cfg.execution_lag_bars,
            timestamp=bar.timestamp,
            position_before=self.policy.position,
            position_after=self.policy.position,
            card=ExplainCard(
                headline="NO TRADE",
                confidence_pct=math.nan,
                rows=(("Data", "bar rejected"),),
                veto_reasons=(why,),
            ),
        )

    # ------------------------------------------------------------------ #
    def run(
        self, bars: Iterable[Bar], exog: Sequence[ExogenousSnapshot] | None = None
    ) -> list[Decision]:
        out: list[Decision] = []
        for i, bar in enumerate(bars):
            e = exog[i] if exog is not None and i < len(exog) else None
            out.append(self.on_bar(bar, e))
        return out

    def stream(
        self, bars: Iterable[Bar], exog: Sequence[ExogenousSnapshot] | None = None
    ) -> Iterator[Decision]:
        for i, bar in enumerate(bars):
            e = exog[i] if exog is not None and i < len(exog) else None
            yield self.on_bar(bar, e)

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        n = len(self.trades)
        acted = sum(1 for d in self.decisions if d.is_actionable)
        lines = [
            f"TI-A {self.cfg.version}  manifest={self.cfg.manifest_hash()}",
            f"bars processed        {len(self.decisions)}",
            f"actionable signals    {acted}"
            + (f"  ({100.0 * acted / len(self.decisions):.2f}% of bars)" if self.decisions else ""),
            f"completed trades      {n}",
            f"final equity          {self.equity:.4f}",
        ]
        if n:
            wins = sum(1 for t in self.trades if t.ret_sigma > 0)
            exp = sum(t.ret_sigma for t in self.trades) / n
            lines += [
                f"hit rate              {100.0 * wins / n:.1f}%  ({wins}/{n})",
                f"expectancy            {exp:+.4f} sigma/trade",
            ]
            # The standard error is printed next to the estimate, always, because
            # the whole argument of docs/01-THEORY.md §12 is that a selective
            # system's trade count is too small for the estimate to stand alone.
            sd = (
                sum((t.ret_sigma - exp) ** 2 for t in self.trades) / max(n - 1, 1)
            ) ** 0.5
            se = sd / math.sqrt(n)
            lines.append(f"  standard error      {se:.4f}  (t = {exp / se:+.2f})" if se > 0 else "")
        lines.append(self.limits.report())
        cal = self.calibrator.health()
        if cal["n"] >= 30:
            lines.append(
                f"calibration           Brier {cal['brier']:.4f}  ECE {cal['ece']:.4f}  n={cal['n']:.0f}"
            )
        d = self.calop.diagnostics()
        lines.append(
            f"engine correlation    mean|r| {d['mean_abs_corr']:.3f}  "
            f"EBE at full reliability {d['ebe_at_full_reliability']:.2f} of {len(self.engines)}"
        )
        return "\n".join(x for x in lines if x)
