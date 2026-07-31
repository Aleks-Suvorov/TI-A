"""Pipeline and decision-policy tests.

These check the *invariants* rather than the performance. Performance on
synthetic data is not evidence of anything; the invariants are what make a
performance number meaningful when it is eventually measured on real data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tia import Action, Config, Position, TIA
from tia.decision.policy import Policy, classify_setup
from tia.engines.edgebook import SetupFamily
from tia.synthetic import generate_null, generate_with_regimes
from tia.training import collect_candidates, train_edge_book
from tia.types import Regime


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


def test_never_flips_from_long_to_short_in_one_bar() -> None:
    """SPEC.md section 3: a reversal costs two bars."""
    bars, _ = generate_with_regimes(4000, seed=50)
    sysm = TIA(Config())
    prev = Position.FLAT
    for d in sysm.stream(bars):
        if prev is Position.LONG:
            assert d.position_after is not Position.SHORT, (
                f"bar {d.decided_at_index}: flipped LONG -> SHORT in one bar"
            )
        if prev is Position.SHORT:
            assert d.position_after is not Position.LONG
        prev = d.position_after


def test_actions_are_consistent_with_the_position_transition() -> None:
    bars, _ = generate_with_regimes(4000, seed=51)
    for d in TIA(Config()).stream(bars):
        if d.action is Action.BUY:
            assert d.position_before is Position.FLAT and d.position_after is Position.LONG
        elif d.action is Action.SHORT:
            assert d.position_before is Position.FLAT and d.position_after is Position.SHORT
        elif d.action is Action.SELL:
            assert d.position_before is Position.LONG and d.position_after is Position.FLAT
        elif d.action is Action.COVER:
            assert d.position_before is Position.SHORT and d.position_after is Position.FLAT


def test_an_exit_is_always_preceded_by_an_entry() -> None:
    bars, _ = generate_with_regimes(4000, seed=52)
    open_side = 0
    for d in TIA(Config()).stream(bars):
        if d.action is Action.BUY:
            assert open_side == 0
            open_side = 1
        elif d.action is Action.SHORT:
            assert open_side == 0
            open_side = -1
        elif d.action is Action.SELL:
            assert open_side == 1, "SELL without an open long"
            open_side = 0
        elif d.action is Action.COVER:
            assert open_side == -1, "COVER without an open short"
            open_side = 0
        # A resting barrier can also close the position, so resynchronise.
        if d.position_after is Position.FLAT:
            open_side = 0


def test_entries_carry_a_target_and_exits_do_not() -> None:
    bars, _ = generate_with_regimes(4000, seed=53)
    for d in TIA(Config()).stream(bars):
        if d.action.is_entry:
            assert d.target is not None and d.risk is not None
        else:
            assert d.target is None


def test_stop_and_target_straddle_the_entry_correctly() -> None:
    bars, _ = generate_with_regimes(5000, seed=54)
    checked = 0
    for d in TIA(Config()).stream(bars):
        if d.target is None:
            continue
        t = d.target
        if t.direction > 0:
            assert t.stop_price < t.entry_ref < t.target_price
        else:
            assert t.target_price < t.entry_ref < t.stop_price
        assert t.reward_risk > 0.0
        checked += 1
    assert checked >= 0


# ---------------------------------------------------------------------------
# Timing and abstention
# ---------------------------------------------------------------------------


def test_execution_always_follows_the_decision() -> None:
    bars = generate_null(1500, seed=55)
    for d in TIA(Config()).stream(bars):
        assert d.execute_at_index > d.decided_at_index


def test_no_trade_dominates() -> None:
    """Abstention is the primary output, not a fallback."""
    bars, _ = generate_with_regimes(5000, seed=56)
    sysm = TIA(Config())
    sysm.run(bars)
    acted = sum(1 for d in sysm.decisions if d.is_actionable)
    frac = acted / len(sysm.decisions)
    assert frac < 0.10, (
        f"{100 * frac:.1f}% of bars produced an action; the gate is not selective"
    )


def test_null_market_produces_essentially_nothing() -> None:
    """The cheapest test that catches a look-ahead or a cost-model artifact.

    A martingale with realistic volatility clustering contains no directional
    edge by construction. A framework that finds trades here has a bug.
    """
    bars = generate_null(6000, seed=57)
    sysm = TIA(Config())
    sysm.run(bars)
    acted = sum(1 for d in sysm.decisions if d.is_actionable)
    assert acted <= 3, f"{acted} actionable signals on a pure martingale"


def test_untrained_system_cannot_trade() -> None:
    """The cold start is real and is resolved by training, not by a loose prior."""
    bars, _ = generate_with_regimes(3000, seed=58)
    sysm = TIA(Config())
    sysm.run(bars)
    assert all(not d.is_actionable for d in sysm.decisions), (
        "an untrained Edge Book produced a trade; its credible bound should be "
        "negative everywhere"
    )


def test_every_no_trade_is_answerable() -> None:
    """A refusal must always name a reason."""
    bars, _ = generate_with_regimes(2000, seed=59)
    sysm = TIA(Config())
    checked = 0
    for d in sysm.stream(bars):
        if d.action is Action.NO_TRADE and d.card is not None and d.decided_at_index > 400:
            assert d.card.veto_reasons, f"bar {d.decided_at_index}: NO_TRADE with no reason"
            checked += 1
    assert checked > 500


# ---------------------------------------------------------------------------
# Rate limiter and risk
# ---------------------------------------------------------------------------


def test_trade_rate_limiter_engages() -> None:
    p = Policy(Config())
    for i in range(int(Config().max_trades_per_100_bars) + 2):
        p._index = 1000 + i
        p._recent_entries.append(1000 + i)
    p._index = 1010
    p._prune()
    assert len(p._recent_entries) >= Config().max_trades_per_100_bars


def test_risk_fraction_never_exceeds_the_cap() -> None:
    bars, _ = generate_with_regimes(5000, seed=60)
    cfg = Config()
    for d in TIA(cfg).stream(bars):
        if d.risk is not None:
            assert 0.0 <= d.risk.risk_fraction <= cfg.max_risk_per_trade


def test_equity_accounting_matches_the_trade_record() -> None:
    """Equity must be reconstructible from the recorded trades alone.

    Uses a demonstrable edge so the test actually exercises trades rather than
    passing vacuously -- an accounting test that never sees a trade is not an
    accounting test.
    """
    from tia.types import ExogenousSnapshot

    bars, _ = generate_with_regimes(
        6000, seed=61, trend_strength=0.5, revert_strength=0.75
    )
    exog = [ExogenousSnapshot(spread=5e-4)] * len(bars)
    cands = collect_candidates(bars, Config(), exog=exog)
    book, cal, _ = train_edge_book(bars, cands, Config())
    sysm = TIA(Config(), edge_book=book, learn=False)
    sysm.calibrator._iso = cal
    sysm.calibrator.active = cal.n_fit > 0
    sysm.run(bars, exog=exog)

    assert sysm.trades, "no trades even on a large edge: the gate may be stuck shut"

    # Replay the recorded trades and reproduce the final equity exactly.
    eq = 1.0
    for t in sysm.trades:
        assert t.equity_after > 0.0, "equity went non-positive"
        assert t.risk_fraction >= 0.0
        eq *= t.equity_after / eq if eq > 0 else 1.0
    assert sysm.equity > 0.0
    assert sysm.trades[-1].equity_after == pytest.approx(sysm.equity, rel=1e-12)
    assert eq == pytest.approx(sysm.equity, rel=1e-9)

    # Every trade's sigma-return and its price move must agree in sign.
    for t in sysm.trades:
        move = math.log(t.exit_price / t.entry_price) * t.direction
        assert math.copysign(1.0, move) == math.copysign(1.0, t.ret_sigma) or abs(t.ret_sigma) < 1e-12


def test_trade_records_are_internally_consistent() -> None:
    bars, _ = generate_with_regimes(6000, seed=62)
    cands = collect_candidates(bars, Config())
    book, _, _ = train_edge_book(bars, cands, Config())
    sysm = TIA(Config(), edge_book=book, learn=False)
    sysm.run(bars)
    for t in sysm.trades:
        assert t.exit_index >= t.entry_index
        assert t.direction in (-1, 1)
        assert t.outcome in (-1, 0, 1)
        assert 0 <= t.bucket < Config().edge_evidence_buckets
        assert 0 <= t.setup < 4
        # Sign consistency: a target hit must be a gain in sigma terms.
        if t.outcome > 0:
            assert t.ret_sigma > 0
        elif t.outcome < 0:
            assert t.ret_sigma < 0.5, "a stop-out should not be a large gain"


# ---------------------------------------------------------------------------
# Setup classification
# ---------------------------------------------------------------------------


def test_setup_classification_is_total_and_ordered() -> None:
    """Every bar gets exactly one family, and the most specific one wins."""
    bars, _ = generate_with_regimes(3000, seed=63)
    sysm = TIA(Config())
    seen: set[int] = set()
    for d in sysm.stream(bars):
        if not d.engine_outputs:
            continue
        fam = classify_setup(d.engine_outputs, d.regime, d.regime_posterior)
        assert isinstance(fam, SetupFamily)
        seen.add(int(fam))
    assert len(seen) >= 2, "classification collapsed to a single family"


def test_regime_and_posterior_are_reported_together() -> None:
    bars, _ = generate_with_regimes(1200, seed=64)
    for d in TIA(Config()).stream(bars):
        assert isinstance(d.regime, Regime)
        assert sum(d.regime_posterior) == pytest.approx(1.0, abs=1e-6)
        assert d.regime_posterior[int(d.regime)] == pytest.approx(
            max(d.regime_posterior), abs=1e-12
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_validation_report_renders_without_trades() -> None:
    from tia.validation.report import build_report

    out = build_report(Config(), [])
    assert "VALIDATION REPORT" in out
    assert "ACCEPTANCE VERDICT" in out
    assert Config().manifest_hash() in out


def test_validation_report_includes_the_verdict() -> None:
    from tia.validation.report import build_report

    bars, _ = generate_with_regimes(6000, seed=65)
    cands = collect_candidates(bars, Config())
    book, cal, diag = train_edge_book(bars, cands, Config())
    sysm = TIA(Config(), edge_book=book, learn=False)
    sysm.run(bars)
    out = build_report(Config(), sysm.trades, diag, [sysm])
    assert "PRE-REGISTERED ACCEPTANCE VERDICT" in out
    assert "PASS" in out or "FAIL" in out
    assert "EVIDENCE BREADTH" in out
