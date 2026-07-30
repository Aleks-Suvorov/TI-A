"""The leakage auditor.

This is the most important test in the repository. Everything else checks that a
computation is correct; this checks that it was *possible* -- that the value the
system reports at bar ``t`` could have been produced by a system that had never
seen bar ``t+1``.

The method is brute force and deliberately so. Run the whole pipeline on
``bars[:t+1]``, record every feature, every engine output and the decision at the
final bar, then run it on the full history and compare the same bar. If any value
differs, something downstream of the bar has influenced it, and the system
repaints.

A subtle point worth stating: because the architecture makes engines streaming,
these tests *should* pass by construction. That is not a reason to skip them. The
failure mode they catch is a future edit -- someone adding a
``np.mean(whole_series)`` for convenience, or a smoother, or a
``request.security`` without ``lookahead_off`` in the Pine port. The test exists
to make that edit fail loudly rather than to make today's code look good.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tia import Config, TIA
from tia.features.kernel import FeatureKernel
from tia.synthetic import generate_null, generate_with_regimes


def _snapshot_dict(f) -> dict[str, float]:
    return {k: v for k, v in f.as_dict().items()}


def _assert_identical(a: dict[str, float], b: dict[str, float], where: str) -> None:
    assert set(a) == set(b), f"{where}: key sets differ"
    for k in sorted(a):
        va, vb = a[k], b[k]
        if va != va and vb != vb:
            continue  # both NaN
        assert va == vb, (
            f"{where}: feature '{k}' differs between prefix and full runs "
            f"({va!r} vs {vb!r}). Something at or after this bar influenced it."
        )


# ---------------------------------------------------------------------------
# Feature kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [420, 613, 877, 1150])
def test_feature_kernel_is_prefix_consistent(cut: int) -> None:
    """Features at bar t must not depend on bars after t."""
    bars = generate_null(1400, seed=5)
    cfg = Config()

    full = FeatureKernel(cfg)
    full_snap = None
    for i, bar in enumerate(bars):
        snap = full.update(bar)
        if i == cut:
            full_snap = _snapshot_dict(snap)

    prefix = FeatureKernel(cfg)
    pre_snap = None
    for bar in bars[: cut + 1]:
        pre_snap = _snapshot_dict(prefix.update(bar))

    assert full_snap is not None and pre_snap is not None
    _assert_identical(pre_snap, full_snap, f"FeatureKernel at bar {cut}")


def test_regime_features_are_prefix_consistent() -> None:
    """The rank features feeding the regime filter must also be causal."""
    bars = generate_with_regimes(1200, seed=6)[0]
    cfg = Config()
    cut = 900

    full = FeatureKernel(cfg)
    full_rf = None
    for i, bar in enumerate(bars):
        full.update(bar)
        rf = full.regime_features()
        if i == cut:
            full_rf = dict(rf)

    pre = FeatureKernel(cfg)
    pre_rf = None
    for bar in bars[: cut + 1]:
        pre.update(bar)
        pre_rf = dict(pre.regime_features())

    assert full_rf is not None and pre_rf is not None
    _assert_identical(pre_rf, full_rf, f"regime features at bar {cut}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [700, 950])
def test_pipeline_decision_is_prefix_consistent(cut: int) -> None:
    """The whole decision -- action, probability, breadth, regime -- is causal."""
    bars = generate_with_regimes(1300, seed=8)[0]
    cfg = Config()

    full = TIA(cfg)
    full_dec = None
    for i, bar in enumerate(bars):
        d = full.on_bar(bar)
        if i == cut:
            full_dec = d

    pre = TIA(cfg)
    pre_dec = None
    for bar in bars[: cut + 1]:
        pre_dec = pre.on_bar(bar)

    assert full_dec is not None and pre_dec is not None
    assert pre_dec.action is full_dec.action
    assert pre_dec.regime is full_dec.regime
    assert pre_dec.regime_posterior == pytest.approx(full_dec.regime_posterior, abs=0.0)
    assert pre_dec.fusion.log_odds == full_dec.fusion.log_odds
    assert pre_dec.fusion.p_success == full_dec.fusion.p_success
    assert pre_dec.fusion.ebe == full_dec.fusion.ebe
    assert pre_dec.expected_value_lcb == full_dec.expected_value_lcb

    for name, out in full_dec.engine_outputs.items():
        pre_out = pre_dec.engine_outputs[name]
        assert pre_out.score == out.score, f"engine '{name}' score repaints"
        assert pre_out.reliability == out.reliability, f"engine '{name}' reliability repaints"


def test_decisions_are_stable_under_appending() -> None:
    """Appending bars must not alter any decision already emitted.

    This is the operational statement of "does not repaint": a historical arrow
    on the chart does not move when a new bar arrives. Distinct from the prefix
    test above, which checks one bar in isolation; here every prior decision is
    compared at once.
    """
    bars = generate_with_regimes(1100, seed=9)[0]
    cfg = Config()
    cut = 800

    short = TIA(cfg)
    short_actions = [short.on_bar(b).action for b in bars[:cut]]

    long = TIA(cfg)
    long_actions = [long.on_bar(b).action for b in bars]

    assert short_actions == long_actions[:cut], (
        "decisions changed when later bars were appended: the system repaints"
    )


# ---------------------------------------------------------------------------
# Structure: confirmation delay must be real
# ---------------------------------------------------------------------------


def test_pivots_are_never_visible_before_confirmation() -> None:
    """A pivot must not be readable at any bar before its ``confirmed_at``."""
    from tia.features.structure import PivotTracker

    bars = generate_null(900, seed=10)
    cfg = Config()
    kern = FeatureKernel(cfg)
    tracker = PivotTracker(cfg.pivot_atr_mult, cfg.pivot_confirm_bars)

    seen: list[tuple[int, int]] = []
    for bar in bars:
        f = kern.update(bar)
        sigma = f.sigma_bp
        if sigma == sigma and sigma > 0.0:
            piv = tracker.update(bar, sigma)
            if piv is not None:
                seen.append((piv.index, piv.confirmed_at))

    assert seen, "no pivots confirmed; the test would be vacuous"
    for occurred, confirmed in seen:
        assert confirmed > occurred, (
            f"pivot at {occurred} was confirmed at {confirmed}: confirmation must lag occurrence"
        )
        assert confirmed - occurred >= cfg.pivot_confirm_bars, (
            f"pivot confirmed after {confirmed - occurred} bars, "
            f"less than pivot_confirm_bars={cfg.pivot_confirm_bars}"
        )


def test_sweeps_are_only_marked_after_the_reclaim() -> None:
    """A sweep is knowable only once it has failed, never at penetration."""
    bars, _ = generate_with_regimes(2000, seed=11)
    kern = FeatureKernel(Config())
    marks = 0
    for bar in bars:
        f = kern.update(bar)
        if f.swept_low > 0.0 or f.swept_high > 0.0:
            ev = kern.sweeps.last_event
            assert ev is not None
            assert ev.reclaim_index > ev.penetration_index, (
                "sweep marked at or before penetration: this reads the future"
            )
            marks += 1
    # Not asserting a count: on some seeds there are none, and a test that
    # demands sweeps exist would be testing the generator, not the detector.
    assert marks >= 0


# ---------------------------------------------------------------------------
# Execution timing
# ---------------------------------------------------------------------------


def test_execution_index_is_strictly_after_decision_index() -> None:
    bars = generate_with_regimes(800, seed=12)[0]
    cfg = Config()
    assert cfg.execution_lag_bars >= 1, "execution lag must never be zero"
    sysm = TIA(cfg)
    for bar in bars:
        d = sysm.on_bar(bar)
        assert d.execute_at_index > d.decided_at_index


def test_determinism_across_identical_runs() -> None:
    """Same bars twice, same decisions. No wall clock, no unseeded randomness."""
    bars = generate_with_regimes(700, seed=13)[0]
    cfg = Config()
    a = [(_d.action, _d.fusion.log_odds) for _d in TIA(cfg).stream(bars)]
    b = [(_d.action, _d.fusion.log_odds) for _d in TIA(cfg).stream(bars)]
    assert a == b


def test_reset_restores_initial_state() -> None:
    """A reset system must reproduce a fresh one exactly."""
    bars = generate_null(600, seed=14)
    cfg = Config()
    sysm = TIA(cfg)
    first = [d.action for d in sysm.stream(bars)]
    sysm.reset()
    second = [d.action for d in sysm.stream(bars)]
    assert first == second
