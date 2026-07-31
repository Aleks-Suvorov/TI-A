"""Regression tests. Each one locks in a bug that actually shipped.

Every test here has a story: it failed once, in this repository, in a way that
was invisible from reading the code. The comments record what went wrong, because
a regression test whose reason has been forgotten is the first one someone
deletes.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tia import Config, TIA
from tia.labeling.weights import (
    average_uniqueness,
    combined_effective_sample_size,
    combined_weights,
    effective_sample_size,
    overlap_effective_sample_size,
    tstat_inflation_factor,
)
from tia.types import Bar

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Kish's ESS is blind to uniform overlap
# ---------------------------------------------------------------------------


def test_kish_ess_alone_does_not_correct_for_overlap() -> None:
    """The bug: Kish is scale-invariant, so uniformly overlapping labels -- the
    normal case when entries are regularly spaced -- carry no weight dispersion
    and it returns ~n however severe the overlap. `training.py` reported that
    number as the effective sample size, inflating its t-statistic by 2.6x."""
    e = np.arange(0, 900, 3)
    x = e + 20
    w = combined_weights(1000, e, x, halflife_bars=5000.0)

    kish = effective_sample_size(w)
    assert kish > 0.9 * len(e), (
        "this test documents that Kish does NOT see overlap; if it now does, "
        "the combined measure below needs revisiting"
    )
    # Kish is scale-invariant, which is exactly why it cannot see uniform
    # down-weighting.
    assert effective_sample_size(w) == pytest.approx(effective_sample_size(w * 7.3))

    overlap = overlap_effective_sample_size(1000, e, x)
    assert overlap < 0.25 * len(e), f"overlap-aware count {overlap} too high"
    assert overlap == pytest.approx(float(average_uniqueness(1000, e, x).sum()))


def test_combined_ess_accounts_for_overlap_and_dispersion() -> None:
    e = np.arange(0, 900, 3)
    x = e + 20
    w = combined_weights(1000, e, x, halflife_bars=5000.0)
    combined = combined_effective_sample_size(1000, e, x, w)
    assert combined < 0.25 * len(e)
    assert combined <= overlap_effective_sample_size(1000, e, x) + 1e-9


def test_combined_ess_reduces_to_kish_without_overlap() -> None:
    """With non-overlapping labels the only shrinkage is dispersion."""
    e = np.arange(0, 900, 30)
    x = e + 20
    w = combined_weights(1000, e, x, halflife_bars=5000.0)
    assert combined_effective_sample_size(1000, e, x, w) == pytest.approx(
        effective_sample_size(w), rel=1e-9
    )


def test_tstat_inflation_uses_spans_when_given() -> None:
    e = np.arange(0, 900, 3)
    x = e + 20
    w = combined_weights(1000, e, x, halflife_bars=5000.0)
    naive = tstat_inflation_factor(w)
    aware = tstat_inflation_factor(w, 1000, e, x)
    assert naive < 1.1, "dispersion alone barely corrects"
    assert aware > 2.0, f"span-aware inflation {aware} too small"


def test_training_reports_the_overlap_aware_sample_size() -> None:
    from tia.synthetic import generate_with_regimes
    from tia.training import collect_candidates, train_edge_book

    bars, _ = generate_with_regimes(3000, seed=12)
    cands = collect_candidates(bars, Config())
    if len(cands) < 10:
        pytest.skip("too few candidates on this sample")
    _, _, diag = train_edge_book(bars, cands, Config())
    assert diag["effective_sample_size"] <= len(cands) + 1e-9
    assert diag["uniqueness_ratio"] < 1.0


# ---------------------------------------------------------------------------
# 2. A perfectly flat market crashed the volatility kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("volume", [0.0, 1e6])
def test_flat_market_does_not_crash(volume: float) -> None:
    """The bug: with every close-to-close horizon exactly zero, the HAR blend's
    weight list was empty and `sum(...)/wsum` divided by zero. A halted
    instrument, a pegged rate, a stablecoin or a dead session all produce runs of
    identical closes, so this is an ordinary live input, not a corner case."""
    bars = [Bar(i * 86400.0, 100.0, 100.0, 100.0, 100.0, volume) for i in range(400)]
    sysm = TIA(Config())
    sysm.run(bars)  # must not raise
    assert sysm.equity > 0.0
    assert all(not d.is_actionable for d in sysm.decisions)


def test_flat_market_yields_a_usable_volatility_forecast() -> None:
    from tia.features.scales import VolatilityKernel

    k = VolatilityKernel()
    for i in range(300):
        k.update(Bar(i * 86400.0, 100.0, 100.0, 100.0, 100.0, 1e6))
    assert k.out.sigma_fcst > 0.0, "forecast must never be zero or NaN"
    assert math.isfinite(k.out.sigma_fcst)


def test_near_flat_market_with_occasional_ticks() -> None:
    """A pegged instrument that moves one tick a week."""
    bars = []
    px = 100.0
    for i in range(500):
        if i % 40 == 0:
            px += 0.01
        bars.append(Bar(i * 86400.0, px, px, px, px, 1e5))
    sysm = TIA(Config())
    sysm.run(bars)
    assert sysm.equity > 0.0


# ---------------------------------------------------------------------------
# 3. NaN exported as a plausible-looking number
# ---------------------------------------------------------------------------


def test_export_maps_missing_to_null_not_to_a_sentinel() -> None:
    """The bug: `_r(nan)` returned -1e15, because `nan > 0` is False. A missing
    calibration Brier score reached the manifest as a finite, plausible number
    that no consumer could distinguish from a measurement."""
    from tia.export import _r, _pine_num

    assert _r(float("nan")) is None
    assert _pine_num(float("nan")) == "na"
    assert _pine_num(None) == "na"
    # True infinities keep a directional sentinel: they carry meaning.
    assert _r(float("inf")) == 1e15
    assert _r(float("-inf")) == -1e15
    assert _r(0.25) == pytest.approx(0.25)


def test_export_vectors_reject_non_finite_values() -> None:
    """Vectors are consumed numerically by the port, so a NaN there is a defect
    rather than a value and must be loud."""
    from tia.export import _rv

    assert _rv([1.0, 2.0]) == [1.0, 2.0]
    with pytest.raises(ValueError, match="non-finite"):
        _rv([1.0, float("nan")])


# ---------------------------------------------------------------------------
# 4. The Pine port drifting out of sync with the exported model
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (REPO / "pine" / "frozen_model.json").exists(), reason="no export")
def test_pine_constants_match_the_exported_model() -> None:
    """The bug: the JSON was regenerated after the isotonic fix while the .pine
    files kept the previous constants, so the published chart asserted 0% and
    100% confidence for a full commit. Export now emits by default; this is the
    guard that keeps it honest."""
    model = json.loads((REPO / "pine" / "frozen_model.json").read_text())
    for name in ("TIA.pine", "TIA_strategy.pine"):
        text = (REPO / "pine" / name).read_text()
        assert model["config_manifest_hash"] in text, f"{name}: stale manifest hash"
        line = next(l for l in text.splitlines() if l.startswith("FM_ISO_Y"))
        emitted = [float(v) for v in line.split('"')[1].split(",")]
        expected = model["fusion"]["knots_y"]
        assert len(emitted) == len(expected)
        assert all(abs(a - b) < 1e-6 for a, b in zip(emitted, expected)), f"{name}: stale isotonic map"


@pytest.mark.skipif(not (REPO / "pine").exists(), reason="no pine directory")
def test_pine_calibration_map_never_asserts_certainty() -> None:
    for name in ("TIA.pine", "TIA_strategy.pine"):
        line = next(
            l for l in (REPO / "pine" / name).read_text().splitlines() if l.startswith("FM_ISO_Y")
        )
        ys = [float(v) for v in line.split('"')[1].split(",")]
        assert min(ys) > 0.0 and max(ys) < 1.0, f"{name}: calibrated map contains 0 or 1"


@pytest.mark.skipif(not (REPO / "python" / "tools" / "pinelint.py").exists(), reason="no linter")
def test_pine_static_check_passes() -> None:
    """Guards the two compile errors that shipped: a parameter named `str`
    shadowing the namespace whose `.split` the function then called, and
    Python-style comma-separated multiple assignment."""
    out = subprocess.run(
        [sys.executable, str(REPO / "python" / "tools" / "pinelint.py")],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, f"pinelint reported problems:\n{out.stdout}"


# ---------------------------------------------------------------------------
# 5. Isotonic endpoints asserting certainty
# ---------------------------------------------------------------------------


def test_isotonic_endpoints_are_bounded_by_their_run_length() -> None:
    """The bug: PAVA pools only where the ordering is *violated*, and a run of
    identical values never violates it -- so a trailing run of successes stayed
    at exactly 1.0 and the card read 'Confidence: 100%' beside 'Evidence
    Breadth: 0.8 independent'."""
    from tia.fusion.calibration import IsotonicCalibrator

    # Ten guaranteed successes at the top, five guaranteed failures at the bottom.
    p = np.concatenate([np.linspace(0.01, 0.05, 5), np.linspace(0.3, 0.7, 40),
                        np.linspace(0.95, 0.99, 10)])
    y = np.concatenate([np.zeros(5), (np.arange(40) % 2).astype(float), np.ones(10)])
    cal = IsotonicCalibrator().fit(p, y)
    assert cal.y.max() < 1.0 and cal.y.min() > 0.0

    # The top bound is the Laplace mean (n+1)/(n+2) for the trailing run of
    # successes. Compute the run length rather than assuming it: the alternating
    # middle block ends on a success, so the run is one longer than the block of
    # guaranteed ones.
    order = np.argsort(p, kind="stable")
    ys = y[order]
    n_hi = 0
    for v in ys[::-1]:
        if v != 1.0:
            break
        n_hi += 1
    assert cal.y.max() == pytest.approx((n_hi + 1.0) / (n_hi + 2.0), abs=1e-9)
    assert cal(0.999) < 1.0


# ---------------------------------------------------------------------------
# 6. The Edge Book hierarchy counting a cell's own data in its own prior
# ---------------------------------------------------------------------------


def test_edge_book_prior_excludes_the_cells_own_observations() -> None:
    """The bug: every observation was written to leaf, node and root, and each
    level passed its already-updated mean down as the child's prior."""
    from tia.engines.edgebook import EdgeBook

    book = EdgeBook(Config())
    for _ in range(5):
        book.observe(0, 0, 3, 2.0, weight=1.0)
    est = book.estimate(0, 0, 3)
    # Leave-one-out: prior is 0, so the mean is 10/(kappa0 + 5) = 10/30.
    assert est.mean == pytest.approx(10.0 / 30.0, abs=0.02)


def test_edge_book_sibling_evidence_still_informs_an_empty_cell() -> None:
    """Leave-one-out must not become no-pooling: siblings should still inform."""
    from tia.engines.edgebook import EdgeBook

    book = EdgeBook(Config())
    rng = np.random.default_rng(1)
    for b in range(4):
        for _ in range(400):
            book.observe(2, 1, b, float(rng.normal(0.5, 1.0)))
    assert book.estimate(2, 1, 4).mean > 0.1


# ---------------------------------------------------------------------------
# 7. Barrier outcomes misaligned with their candidates
# ---------------------------------------------------------------------------


def test_barrier_outcomes_are_realigned_by_source_position() -> None:
    """The bug: the labeller DROPS candidates it cannot fill -- one whose
    execution bar is past the end of the sample, or whose sigma is not finite --
    so its output is shorter than the candidate arrays and not positionally
    aligned with them. `train_edge_book` indexed the candidate arrays with
    output-relative positions, attributing every outcome after the first drop to
    the wrong regime, setup and bucket. It corrupted the Edge Book silently; the
    length mismatch only happened to raise further down, and only in the
    walk-forward path, where prefixes truncate the trailing candidates.
    """
    from tia.labeling.triple_barrier import apply_triple_barrier

    n = 60
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)

    entries = np.array([5, 10, 15, 20, n - 1])  # the last cannot be filled
    dirs = np.array([1, 1, -1, 1, 1])
    sig = np.array([0.01, 0.01, np.nan, 0.01, 0.01])  # index 2 has no sigma

    res = apply_triple_barrier(
        high, low, close, entries, dirs, stop_sigma=1.0, target_sigma=2.0,
        sigma=sig, max_holding=10, execution_lag=1, open_=open_,
    )

    assert res.n_dropped >= 2, f"expected drops, got {res.n_dropped}"
    assert len(res) < entries.size

    # The realignment invariant the training pass depends on.
    assert np.array_equal(res.entry_index, entries[res.source_position])
    assert np.array_equal(res.direction, dirs[res.source_position])

    # And the property that makes the old code wrong: row index != source index.
    assert np.any(res.source_position != np.arange(len(res))), (
        "this fixture no longer forces a misalignment, so it cannot detect the bug"
    )


def test_training_attributes_outcomes_to_the_right_cells_after_a_drop() -> None:
    """End-to-end version: a prefix that truncates trailing candidates must not
    shift outcomes into neighbouring Edge Book cells."""
    from tia.synthetic import generate_with_regimes
    from tia.training import collect_candidates, train_edge_book

    bars, _ = generate_with_regimes(2500, seed=44, trend_strength=0.4, revert_strength=0.6)
    cands = collect_candidates(bars, Config())
    if len(cands) < 20:
        pytest.skip("too few candidates on this sample")

    book, _, diag = train_edge_book(bars, cands, Config())
    assert diag["n_labelled"] + diag["n_dropped"] == pytest.approx(float(len(cands)))

    # Total weighted observations in the book must equal what was labelled;
    # a misattribution would still balance, but a length bug would not.
    root = book._root
    assert root.n_w > 0.0
    assert root.n_w <= diag["n_labelled"] + 1e-9


# ---------------------------------------------------------------------------
# 8. The execution model (adversarial audit findings)
# ---------------------------------------------------------------------------


def _trained_system():
    from tia.synthetic import generate_with_regimes
    from tia.training import collect_candidates, train_edge_book
    from tia.types import ExogenousSnapshot

    bars, _ = generate_with_regimes(6000, seed=61, trend_strength=0.5, revert_strength=0.75)
    exog = [ExogenousSnapshot(spread=5e-4)] * len(bars)
    cands = collect_candidates(bars, Config(), exog=exog)
    book, cal, _ = train_edge_book(bars, cands, Config())
    sysm = TIA(Config(), edge_book=book, learn=False)
    sysm.calibrator._iso = cal
    sysm.calibrator.active = cal.n_fit > 0
    return sysm, bars, exog


def test_every_entry_produces_exactly_one_trade_record() -> None:
    """The bug: vertical-barrier, reversal and session-close exits removed the
    position from the state machine but were never booked -- no trade record,
    no equity change. 4 of 62 positions in the audit run simply vanished, and
    in learn mode the Edge Book never saw a timeout outcome."""
    sysm, bars, exog = _trained_system()
    entries = sum(1 for d in sysm.stream(bars) if d.action.is_entry)
    still_open = 1 if sysm.policy.open else 0
    assert entries == len(sysm.trades) + still_open, (
        f"{entries} entries but {len(sysm.trades)} trades booked "
        f"({still_open} open): positions are vanishing from the accounting"
    )
    reasons = {t.reason.split(" ")[0] for t in sysm.trades}
    # Timeout-class exits must appear in the record, not only barrier fills.
    assert reasons - {"target", "stop"}, (
        f"only barrier exits were booked ({reasons}); bar-close exits are lost"
    )


def test_entries_fill_at_the_next_bars_open() -> None:
    """The bug: the live pipeline filled at the decision bar's CLOSE while the
    triple-barrier labeller that trains the Edge Book fills at the next bar's
    OPEN (execution_lag=1). The system was trading a different game from the
    one it had learned, and SPEC.md section 2 rule 3 -- which the Decision's
    own execute_at_index field states -- was violated by the backtest itself."""
    sysm, bars, exog = _trained_system()
    sysm.run(bars, exog=exog)
    assert sysm.trades, "fixture produced no trades"
    for t in sysm.trades:
        want = bars[t.entry_index + 1].open
        assert t.entry_price == pytest.approx(want, rel=1e-12), (
            f"trade at {t.entry_index} filled at {t.entry_price}, "
            f"next open is {want}"
        )


def test_equity_is_charged_the_modelled_round_trip() -> None:
    """The bug: the gate netted costs for the decision while realized P&L was
    cost-free -- the single easiest way for a backtest to flatter itself."""
    sysm, bars, exog = _trained_system()
    sysm.run(bars, exog=exog)
    assert sysm.trades
    total_cost = sum(t.cost_sigma for t in sysm.trades)
    assert total_cost > 0.0, "no cost was charged to any trade"
    for t in sysm.trades:
        assert t.net_sigma == pytest.approx(t.ret_sigma - t.cost_sigma)
    gross = sum(t.ret_sigma for t in sysm.trades)
    net = sum(t.net_sigma for t in sysm.trades)
    assert net < gross, "net must be strictly below gross when costs are positive"


def test_timeout_outcomes_reach_the_edge_book_in_learn_mode() -> None:
    """Learning from barrier touches only biases the book optimistic: timeouts
    are the mediocre outcomes, and a learner that never sees them concludes the
    world is made of wins and losses at full barrier distance."""
    from tia.synthetic import generate_with_regimes
    from tia.training import collect_candidates, train_edge_book
    from tia.types import ExogenousSnapshot

    bars, _ = generate_with_regimes(6000, seed=61, trend_strength=0.5, revert_strength=0.75)
    exog = [ExogenousSnapshot(spread=5e-4)] * len(bars)
    cands = collect_candidates(bars, Config(), exog=exog)
    book, cal, _ = train_edge_book(bars, cands, Config())

    # Count observe() calls directly: comparing n_w before/after is confounded
    # by the book's own exponential decay over the replay.
    observed: list[float] = []
    original = book.observe

    def counting_observe(regime, setup, bucket, ret_sigma, weight=1.0):
        observed.append(float(ret_sigma))
        return original(regime, setup, bucket, ret_sigma, weight)

    book.observe = counting_observe  # type: ignore[method-assign]
    sysm = TIA(Config(), edge_book=book, learn=True)
    sysm.calibrator._iso = cal
    sysm.calibrator.active = cal.n_fit > 0
    sysm.run(bars, exog=exog)

    timeouts = [t for t in sysm.trades if t.outcome == 0]
    if not timeouts:
        pytest.skip("no timeout exits on this sample")
    assert len(observed) == len(sysm.trades), (
        f"{len(sysm.trades)} trades closed but only {len(observed)} reached the "
        "Edge Book: timeout outcomes are being dropped from learning"
    )


def test_sweep_reversal_family_is_reachable() -> None:
    """The bug: classify_setup read sweep_age from .diagnostics, but the
    liquidity engine reports it in .features when a sweep is active. Across
    20,000 audited bars with 2,193 active-sweep bars, zero were classified
    SWEEP_REVERSAL -- a quarter of the Edge Book's taxonomy did not exist."""
    from collections import Counter

    from tia.decision.policy import classify_setup
    from tia.engines.edgebook import SetupFamily
    from tia.synthetic import generate_with_regimes

    counts: Counter = Counter()
    for seed in (5, 21, 44):
        bars, _ = generate_with_regimes(4000, seed=seed, trend_strength=0.3, revert_strength=0.5)
        sysm = TIA(Config())
        for d in sysm.stream(bars):
            if d.engine_outputs:
                counts[classify_setup(d.engine_outputs, d.regime, d.regime_posterior)] += 1
    assert counts.get(SetupFamily.SWEEP_REVERSAL, 0) > 0, (
        f"SWEEP_REVERSAL never classified: {dict(counts)}"
    )


def test_pine_lognorm_sign_matches_the_reference() -> None:
    """The export ships ln B(a,b) -- the value beta_logpdf SUBTRACTS. The
    audit's own first fix ADDED it, inverting the correction and doubling the
    damage. This locks the sign against the Python reference at sample points."""
    import json
    import re

    from tia.engines.regime import beta_logpdf

    model_path = REPO / "pine" / "frozen_model.json"
    if not model_path.exists():
        pytest.skip("no exported model")
    m = json.loads(model_path.read_text())
    a = m["regime"]["design_a"]; b = m["regime"]["design_b"]
    n = m["regime"]["design_log_beta_norm"]
    fa = [v for r in a for v in r] if isinstance(a[0], list) else a
    fb = [v for r in b for v in r] if isinstance(b[0], list) else b
    fn = [v for r in n for v in r] if isinstance(n[0], list) else n
    for ai, bi, ni in zip(fa, fb, fn):
        for x in (0.1, 0.5, 0.9):
            pine = (ai - 1) * math.log(x) + (bi - 1) * math.log(1 - x) - ni
            assert pine == pytest.approx(beta_logpdf(x, ai, bi), abs=1e-9)
    # And the .pine source must subtract, not add.
    for name in ("TIA.pine", "TIA_strategy.pine"):
        text = (REPO / "pine" / name).read_text()
        assert re.search(r"f_betaLogPdf\([^)]*\)[^\n]*-\s*array\.get\(DESIGN_LOGN", text), (
            f"{name}: the lognorm correction is not being subtracted"
        )
