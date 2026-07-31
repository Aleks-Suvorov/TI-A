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
