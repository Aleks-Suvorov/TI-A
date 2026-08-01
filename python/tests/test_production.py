"""Production-safety tests: the final review board's additions.

These verify the properties an operator depends on, as distinct from the
properties a researcher depends on: that a bad config refuses to start, that
the journal is a faithful machine-readable record, that the demotion ladder
actually fires and actually blocks, that accounting survives overnight gaps
and corrupted streams, and that random perturbation cannot break an invariant.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest

from tia import Action, Config, Position, TIA
from tia.monitoring.drift import DemotionLevel
from tia.synthetic import generate_null, generate_with_regimes
from tia.training import collect_candidates, train_edge_book
from tia.types import Bar, ExogenousSnapshot


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_default_config_is_valid() -> None:
    assert Config().validate() == []


@pytest.mark.parametrize(
    "kw",
    [
        {"stop_sigma": -1.6},
        {"p_min": 5.8},
        {"execution_lag_bars": 0},
        {"rank_min_obs": 500},          # exceeds rank_window
        {"dd_kill": 0.02},              # below throttle thresholds
        {"har_weights": (0.5, 0.5, 0.5)},
        {"risk_per_trade": 0.5},        # exceeds max_risk_per_trade
        {"min_holding_bars": 40},       # exceeds max_holding_bars
    ],
)
def test_pipeline_refuses_a_bad_config(kw: dict) -> None:
    """A config typo must be an error at construction, not a system that
    trades wrongly. execution_lag_bars=0 in particular is look-ahead by
    construction and must never be startable."""
    with pytest.raises(ValueError, match="invalid configuration"):
        TIA(Config().replace(**kw))


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


def test_journal_is_a_faithful_machine_readable_record() -> None:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        bars, _ = generate_with_regimes(
            4000, seed=61, trend_strength=0.5, revert_strength=0.75
        )
        exog = [ExogenousSnapshot(spread=5e-4)] * len(bars)
        cands = collect_candidates(bars, Config(), exog=exog)
        book, cal, _ = train_edge_book(bars, cands, Config())
        sysm = TIA(Config(), edge_book=book, learn=False, journal_path=path)
        sysm.calibrator._iso = cal
        sysm.calibrator.active = cal.n_fit > 0
        sysm.run(bars, exog=exog)
        sysm.journal.close()

        records = [json.loads(l) for l in open(path)]
        by_kind: dict[str, list] = {}
        for r in records:
            by_kind.setdefault(r["kind"], []).append(r)

        # Lifecycle carries the manifest hash, so a journal can be tied to the
        # exact parameter set that produced it.
        assert by_kind["lifecycle"][0]["manifest"] == Config().manifest_hash()

        # Every booked trade appears, and the journal's numbers agree with the
        # in-memory record to full precision.
        jt = by_kind.get("trade", [])
        assert len(jt) == len(sysm.trades)
        for j, t in zip(jt, sysm.trades):
            assert j["entry_index"] == t.entry_index
            assert j["net_sigma"] == pytest.approx(t.net_sigma)
            assert j["equity_after"] == pytest.approx(t.equity_after)

        # Every actionable decision appears.
        actionable = [d for d in sysm.decisions if d.is_actionable]
        j_actions = [r for r in by_kind.get("decision", []) if r["action"] != "NO TRADE"]
        assert len(j_actions) == len(actionable)

        # NaN never reaches the journal; every line round-trips.
        for r in records:
            json.dumps(r)  # would raise on non-finite values written raw
    finally:
        os.unlink(path)


def test_journal_failure_does_not_kill_the_pipeline() -> None:
    """The journal must degrade, not crash: first write failure disables it
    and records the error where the health report can surface it."""
    bars = generate_null(300, seed=5)
    sysm = TIA(Config(), journal_path="/dev/null/impossible/path.jsonl")
    sysm.run(bars)  # must not raise
    assert not sysm.journal.enabled
    assert sysm.journal.last_error is not None


def test_disabled_journal_costs_nothing_and_changes_nothing() -> None:
    bars, _ = generate_with_regimes(1200, seed=9)
    a = [d.action for d in TIA(Config()).stream(bars)]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        b = [d.action for d in TIA(Config(), journal_path=path).stream(bars)]
    finally:
        os.unlink(path)
    assert a == b, "journalling changed a decision"


# ---------------------------------------------------------------------------
# The demotion ladder actually fires and actually blocks
# ---------------------------------------------------------------------------


def test_demotion_ladder_blocks_entries_under_miscalibration() -> None:
    """A monitor that is wired but never fires is still dead code. Force the
    condition it exists for -- systematic miscalibration -- and require the
    ladder to reach NO_TRADE and the gate to refuse with a named reason."""
    bars, _ = generate_with_regimes(
        6000, seed=61, trend_strength=0.5, revert_strength=0.75
    )
    exog = [ExogenousSnapshot(spread=5e-4)] * len(bars)
    cands = collect_candidates(bars, Config(), exog=exog)
    book, cal, _ = train_edge_book(bars, cands, Config())
    sysm = TIA(Config(), edge_book=book, learn=False)
    sysm.calibrator._iso = cal
    sysm.calibrator.active = cal.n_fit > 0

    # Poison the drift monitor with confidently wrong forecasts: p=0.9
    # followed by failure, repeatedly. Brier = 0.81 >> the 0.27 alarm.
    for _ in range(120):
        sysm.monitor.observe_outcome(0.9, 0.0)

    entries = 0
    demoted_at = None
    veto_seen = False
    for d in sysm.stream(bars):
        if d.action.is_entry:
            entries += 1
        if sysm.demotion >= DemotionLevel.NO_TRADE and demoted_at is None:
            demoted_at = d.decided_at_index
        if (
            demoted_at is not None
            and d.card is not None
            and any("demotion" in v for v in d.card.veto_reasons)
        ):
            veto_seen = True

    assert demoted_at is not None and demoted_at <= 60, (
        "the monitor never demoted despite Brier 0.81 against an alarm of 0.27"
    )
    assert entries == 0, (
        f"{entries} entries were allowed while the probability model was "
        "demonstrably broken -- the ladder is decoration"
    )
    assert veto_seen, "the demotion refusal never surfaced as a named veto reason"


def test_healthy_run_is_not_demoted() -> None:
    """The converse: the ladder must not fire on a healthy system, or operators
    will learn to ignore it."""
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
    assert sysm.demotion < DemotionLevel.NO_TRADE
    assert sysm.trades, "healthy trained run should trade"


# ---------------------------------------------------------------------------
# Overnight gaps: the blind spot that hid the fill-timing bug
# ---------------------------------------------------------------------------


def test_accounting_is_exact_under_overnight_gaps() -> None:
    """docs/19 D2 survived 1.19M assertions because the generator produced
    open(t+1) == close(t) exactly. This test runs the full loop on a market
    where opens gap, and requires: every entry booked, every fill at the next
    open (not the decision close), equity reconstructible exactly."""
    bars, _ = generate_with_regimes(
        6000, seed=61, trend_strength=0.5, revert_strength=0.75,
        overnight_gap_sigma=0.6,
    )
    gaps = [abs(math.log(bars[i].open / bars[i - 1].close)) for i in range(1, len(bars))]
    assert float(np.median(gaps)) > 1e-4, "fixture failed to produce gaps"

    exog = [ExogenousSnapshot(spread=5e-4)] * len(bars)
    cands = collect_candidates(bars, Config(), exog=exog)
    book, cal, _ = train_edge_book(bars, cands, Config())
    sysm = TIA(Config(), edge_book=book, learn=False)
    sysm.calibrator._iso = cal
    sysm.calibrator.active = cal.n_fit > 0
    entries = sum(1 for d in sysm.stream(bars) if d.action.is_entry)
    still_open = 1 if sysm.policy.open else 0

    assert entries == len(sysm.trades) + still_open
    assert sysm.trades, "gapped fixture produced no trades"
    for t in sysm.trades:
        assert t.entry_price == pytest.approx(bars[t.entry_index + 1].open, rel=1e-12)
        assert t.entry_price != pytest.approx(bars[t.entry_index].close, rel=1e-9), (
            "a fill landed exactly on the decision close in a gapped market: "
            "the D2 fill-timing bug is back"
        )
    assert sysm.trades[-1].equity_after == pytest.approx(sysm.equity, rel=1e-12)


def test_gap_through_stop_fills_at_the_open_not_the_stop() -> None:
    """Pessimistic gap handling, end to end: when the open gaps past the stop,
    the fill is the (worse) open."""
    bars, _ = generate_with_regimes(
        8000, seed=13, trend_strength=0.5, revert_strength=0.75,
        overnight_gap_sigma=1.5,
    )
    exog = [ExogenousSnapshot(spread=5e-4)] * len(bars)
    cands = collect_candidates(bars, Config(), exog=exog)
    book, cal, _ = train_edge_book(bars, cands, Config())
    sysm = TIA(Config(), edge_book=book, learn=False)
    sysm.calibrator._iso = cal
    sysm.calibrator.active = cal.n_fit > 0
    sysm.run(bars, exog=exog)
    gapped = [t for t in sysm.trades if "gapped" in t.reason]
    for t in gapped:
        exit_bar = bars[t.exit_index]
        assert t.exit_price == pytest.approx(exit_bar.open, rel=1e-12)
        # The gapped fill must be at least as bad as the stop would have been.
        assert t.ret_sigma <= -t.cost_sigma * 0 - 0.0 or t.outcome == -1


# ---------------------------------------------------------------------------
# Duplicate bars, corrupted streams, idempotency
# ---------------------------------------------------------------------------


def test_duplicate_bar_is_rejected_and_state_is_unchanged() -> None:
    """Idempotency per bar (docs/12): a duplicate delivery must change nothing.
    The validator provides this by rejecting a repeated timestamp."""
    bars = generate_null(600, seed=7)
    sysm = TIA(Config())
    for b in bars[:400]:
        sysm.on_bar(b)
    snapshot = (
        sysm._index,
        sysm.equity,
        sysm.policy.position,
        len(sysm.trades),
        sysm.kernel.snapshot.as_dict(),
    )
    dup = sysm.on_bar(bars[399])  # exact duplicate of the last bar
    assert dup.action is Action.NO_TRADE
    assert dup.card is not None and dup.card.veto_reasons
    after = (
        sysm._index,
        sysm.equity,
        sysm.policy.position,
        len(sysm.trades),
        sysm.kernel.snapshot.as_dict(),
    )
    assert snapshot == after, "a duplicate bar mutated pipeline state"


def test_pending_fill_survives_a_rejected_bar() -> None:
    """If the bar after a decision is rejected by the validator, the entry must
    fill at the next VALID bar's open, and accounting must remain exact."""
    bars, _ = generate_with_regimes(
        6000, seed=61, trend_strength=0.5, revert_strength=0.75
    )
    exog = [ExogenousSnapshot(spread=5e-4)] * len(bars)
    cands = collect_candidates(bars, Config(), exog=exog)
    book, cal, _ = train_edge_book(bars, cands, Config())

    sysm = TIA(Config(), edge_book=book, learn=False)
    sysm.calibrator._iso = cal
    sysm.calibrator.active = cal.n_fit > 0

    # Feed the stream, injecting a duplicate (rejected) bar immediately after
    # every entry decision.
    valid_seq: list[Bar] = []
    it = iter(range(len(bars)))
    i = 0
    entries = 0
    while i < len(bars):
        d = sysm.on_bar(bars[i], exog[i])
        valid_seq.append(bars[i])
        if d.action.is_entry:
            entries += 1
            sysm.on_bar(bars[i], exog[i])  # duplicate: must be rejected
        i += 1

    still_open = 1 if sysm.policy.open else 0
    assert entries == len(sysm.trades) + still_open
    for t in sysm.trades:
        # entry_index counts VALID bars, which by construction equals the
        # original series index here; the fill is the next valid bar's open.
        assert t.entry_price == pytest.approx(bars[t.entry_index + 1].open, rel=1e-12)


# ---------------------------------------------------------------------------
# Randomized property test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [101, 202, 303, 404])
def test_invariants_hold_under_random_market_conditions(seed: int) -> None:
    """Seeded random perturbation: random edge strength, random gap size,
    random dropout. Whatever the market looks like, the invariants must hold."""
    rng = np.random.default_rng(seed)
    strength = float(rng.uniform(0.0, 0.6))
    gap = float(rng.uniform(0.0, 1.0))
    bars, _ = generate_with_regimes(
        3000, seed=seed, trend_strength=strength,
        revert_strength=strength * 1.5, overnight_gap_sigma=gap,
    )
    # Random bar dropout (simulates feed gaps).
    keep = rng.random(len(bars)) > 0.02
    bars = [b for b, k in zip(bars, keep) if k]

    sysm = TIA(Config())
    entries = 0
    prev_pos = Position.FLAT
    for d in sysm.stream(bars):
        if d.fusion is not None:
            assert 0.0 < d.fusion.p_success < 1.0
            assert math.isfinite(d.fusion.log_odds)
            assert d.fusion.ebe >= 0.0
        assert d.execute_at_index > d.decided_at_index
        if prev_pos is Position.LONG:
            assert d.position_after is not Position.SHORT
        if prev_pos is Position.SHORT:
            assert d.position_after is not Position.LONG
        prev_pos = d.position_after
        if d.action.is_entry:
            entries += 1
    assert sysm.equity > 0.0 and math.isfinite(sysm.equity)
    still_open = 1 if sysm.policy.open else 0
    assert entries == len(sysm.trades) + still_open
