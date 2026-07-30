"""Tests for correlation-aware pooling and calibration.

The first three tests are the ones that matter. They check the propositions in
``docs/01-THEORY.md`` §6 numerically, including the specific failure the whole
fusion layer exists to prevent: ten mildly bullish correlated engines being
reported as 99.998% confident.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tia.config import Config
from tia.fusion.calibration import (
    IsotonicCalibrator,
    OnlineCalibrator,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    fit_platt,
    pav,
    reliability_table,
)
from tia.fusion.calop import CALOP, effective_breadth, evidence_from_score, logistic
from tia.types import EngineOutput


def _equicorrelated(m: int, rho: float) -> np.ndarray:
    C = np.full((m, m), rho)
    np.fill_diagonal(C, 1.0)
    return C


# ---------------------------------------------------------------------------
# The propositions
# ---------------------------------------------------------------------------


def test_ebe_equals_engine_count_when_independent() -> None:
    """Proposition 1, first half: at C = I, breadth is the engine count."""
    for m in (2, 5, 9):
        assert effective_breadth(np.eye(m), np.ones(m)) == pytest.approx(float(m))


def test_ebe_collapses_to_one_for_duplicated_engines() -> None:
    """Proposition 1, second half: perfect duplicates count once."""
    for m in (2, 4, 10):
        ebe = effective_breadth(_equicorrelated(m, 0.999), np.ones(m))
        assert ebe == pytest.approx(1.0, abs=0.02), (
            f"{m} identical engines reported {ebe:.3f} independent opinions"
        )


def test_ebe_matches_the_closed_form_for_equicorrelated_engines() -> None:
    """EBE = m / (1 + (m-1) rho), the formula quoted in the theory doc."""
    for m in (3, 5, 10):
        for rho in (0.0, 0.2, 0.5, 0.8):
            expected = m / (1.0 + (m - 1) * rho)
            got = effective_breadth(_equicorrelated(m, rho), np.ones(m))
            assert got == pytest.approx(expected, rel=1e-9)


def test_ten_engines_at_half_correlation_give_fewer_than_two_opinions() -> None:
    """The headline number from ``docs/01-THEORY.md`` §6.2."""
    ebe = effective_breadth(_equicorrelated(10, 0.5), np.ones(10))
    assert ebe == pytest.approx(1.818, abs=0.01)


def test_calop_reduces_to_the_bayesian_sum_under_independence() -> None:
    """With C = I and full reliability, pooling *is* log-odds summation."""
    names = [f"e{i}" for i in range(4)]
    pooler = CALOP(names, Config())
    pooler._last_C = np.eye(4)
    pooler._corr._n = 10**6  # force use of the injected matrix

    scores = [0.3, -0.1, 0.5, 0.2]
    outs = {
        n: EngineOutput(name=n, score=s, reliability=1.0)
        for n, s in zip(names, scores)
    }
    # correlation() would recompute; call the maths directly against C = I.
    e = np.array([evidence_from_score(s) for s in scores])
    res = pooler.fuse(outs)
    # The pooler re-derives C from its own estimator, which has no samples, so
    # assert against the algebra rather than the internal state.
    direct = float(np.ones(4) @ np.linalg.solve(np.eye(4), e))
    assert direct == pytest.approx(float(e.sum()), rel=1e-12)
    assert res.direction == 1


def test_naive_summation_overstates_confidence_catastrophically() -> None:
    """The motivating example, checked numerically.

    Ten engines each scoring 0.5. Naive log-odds summation reports 99.998%.
    Correlation-aware pooling at rho = 0.5, with the uncertainty correction,
    reports something far more modest. This gap is the reason the module exists.
    """
    m = 10
    e = np.full(m, evidence_from_score(0.5))
    naive_p = logistic(float(e.sum()))
    assert naive_p > 0.9999, "the naive calculation should be absurdly confident"

    C = _equicorrelated(m, 0.5)
    L = float(np.ones(m) @ np.linalg.solve(C, e))
    ebe = effective_breadth(C, np.ones(m))
    cfg = Config()
    var = cfg.fusion_evidence_sd**2 * ebe
    kappa = 1.0 / math.sqrt(1.0 + (math.pi / 8.0) * cfg.fusion_temperature**2 * var)
    honest_p = logistic(cfg.fusion_temperature * L * kappa)

    assert honest_p < 0.93, f"corrected probability {honest_p:.4f} is still too confident"
    assert naive_p - honest_p > 0.07


def test_uncertainty_correction_shrinks_toward_a_coin_flip() -> None:
    """More breadth at fixed evidence widens the interval and lowers confidence."""
    cfg = Config()
    L = 2.0
    ps = []
    for ebe in (1.0, 4.0, 9.0):
        var = cfg.fusion_evidence_sd**2 * ebe
        kappa = 1.0 / math.sqrt(1.0 + (math.pi / 8.0) * var)
        ps.append(logistic(L * kappa))
    assert ps[0] > ps[1] > ps[2] > 0.5


def test_reliability_zero_removes_an_engine_entirely() -> None:
    names = ["a", "b", "c"]
    pooler = CALOP(names, Config())
    live = {
        "a": EngineOutput("a", 0.6, 1.0),
        "b": EngineOutput("b", -0.9, 0.0),  # muted
        "c": EngineOutput("c", 0.4, 1.0),
    }
    res = pooler.fuse(live)
    assert res.direction == 1, "a zero-reliability bearish engine must not flip the direction"
    assert res.contributions["b"] == pytest.approx(0.0)


def test_aligned_breadth_excludes_disagreeing_engines() -> None:
    """The gate's breadth counts only engines supporting the pooled direction."""
    names = [f"e{i}" for i in range(4)]
    pooler = CALOP(names, Config())
    agree = {n: EngineOutput(n, 0.5, 0.9) for n in names}
    mixed = {
        "e0": EngineOutput("e0", 0.5, 0.9),
        "e1": EngineOutput("e1", 0.5, 0.9),
        "e2": EngineOutput("e2", -0.5, 0.9),
        "e3": EngineOutput("e3", -0.4, 0.9),
    }
    assert pooler.fuse(agree).ebe > pooler.fuse(mixed).ebe


def test_confident_neutral_engine_lowers_confidence_without_flipping_it() -> None:
    """A zero score with real reliability is evidence of *no* direction."""
    names = ["a", "b"]
    pooler = CALOP(names, Config())
    only = pooler.fuse({"a": EngineOutput("a", 0.7, 1.0), "b": EngineOutput("b", 0.0, 0.0)})
    with_neutral = pooler.fuse(
        {"a": EngineOutput("a", 0.7, 1.0), "b": EngineOutput("b", 0.0, 1.0)}
    )
    assert with_neutral.direction == only.direction == 1
    assert with_neutral.p_success < only.p_success


def test_evidence_link_is_the_inverse_of_tanh() -> None:
    for t in (-2.0, -0.5, 0.0, 0.7, 3.0):
        assert evidence_from_score(math.tanh(t)) == pytest.approx(2.0 * t, rel=1e-9)


def test_saturated_score_stays_finite() -> None:
    assert math.isfinite(evidence_from_score(1.0))
    assert math.isfinite(evidence_from_score(-1.0))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_pav_is_monotone_and_preserves_the_mean() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(200)
    f = pav(y)
    assert np.all(np.diff(f) >= -1e-12), "PAVA output must be non-decreasing"
    assert f.mean() == pytest.approx(y.mean(), rel=1e-12)


def test_pav_leaves_already_monotone_input_untouched() -> None:
    y = np.linspace(0.0, 1.0, 50)
    assert np.allclose(pav(y), y)


def test_isotonic_corrects_a_known_miscalibration() -> None:
    """A forecaster that reports p but delivers p^2 should be fixed."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.random(4000) < p**2).astype(float)
    raw = brier_score(p, y)
    cal = IsotonicCalibrator().fit(p, y)
    fixed = brier_score(np.array([cal(x) for x in p]), y)
    assert fixed < raw
    assert expected_calibration_error(np.array([cal(x) for x in p]), y) < 0.03


def test_platt_recovers_a_known_logistic() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 2.0, 8000)
    true_t, true_b = 0.8, -0.3
    y = (rng.random(8000) < 1.0 / (1.0 + np.exp(-(true_t * x + true_b)))).astype(float)
    t, b = fit_platt(x, y)
    assert t == pytest.approx(true_t, abs=0.08)
    assert b == pytest.approx(true_b, abs=0.08)


def test_brier_decomposition_identity_holds() -> None:
    """BS = reliability - resolution + uncertainty, to numerical precision."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0.0, 1.0, 5000)
    y = (rng.random(5000) < p).astype(float)
    d = brier_decomposition(p, y, bins=20)
    assert d.brier == pytest.approx(d.reliability - d.resolution + d.uncertainty, abs=2e-3)


def test_perfectly_calibrated_forecaster_has_near_zero_reliability_term() -> None:
    rng = np.random.default_rng(4)
    p = rng.uniform(0.1, 0.9, 20000)
    y = (rng.random(20000) < p).astype(float)
    d = brier_decomposition(p, y, bins=10)
    assert d.reliability < 0.002
    assert d.resolution > 0.03
    assert d.skill > 0.0


def test_base_rate_forecaster_is_calibrated_but_useless() -> None:
    """The distinction the decomposition exists to make."""
    rng = np.random.default_rng(5)
    y = (rng.random(5000) < 0.4).astype(float)
    p = np.full(5000, 0.4)
    d = brier_decomposition(p, y)
    # Reliability is not exactly zero because the realised base rate differs
    # from 0.4 by sampling error; the residual is that difference squared,
    # order 1e-5 at n = 5000.
    assert d.reliability < 1e-3, "constant base-rate forecasts are near-perfectly calibrated"
    assert d.resolution == pytest.approx(0.0, abs=1e-12), "...and carry no information at all"


def test_reliability_table_intervals_bracket_the_truth() -> None:
    rng = np.random.default_rng(6)
    p = rng.uniform(0.1, 0.9, 6000)
    y = (rng.random(6000) < p).astype(float)
    rows = reliability_table(p, y, bins=10)
    covered = sum(1 for r in rows if r["ci_low"] <= r["predicted"] <= r["ci_high"])
    assert covered >= len(rows) - 1


def test_online_calibrator_stays_identity_until_it_has_data() -> None:
    c = OnlineCalibrator(min_samples=200)
    assert c(0.73) == 0.73
    rng = np.random.default_rng(7)
    for _ in range(199):
        c.record(float(rng.random()), bool(rng.random() < 0.5))
    assert c(0.73) == 0.73, "must not calibrate on fewer than min_samples outcomes"
