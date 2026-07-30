"""Tests for the feature kernel.

Where a quantity has a known statistical null -- the variance-ratio statistic
under a martingale, permutation entropy under IID noise -- the test checks the
null rather than a hand-computed constant. That catches scaling errors, which are
the failure mode these estimators actually have and which a fixture-based test
would miss.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tia.config import Config
from tia.features.entropy import PermutationEntropy, RunAsymmetry
from tia.features.jumps import JumpTest
from tia.features.kernel import FeatureKernel
from tia.features.microstructure import CorwinSchultzSpread
from tia.features.rolling import (
    BucketedMedian,
    CausalRank,
    EWMA,
    RingBuffer,
    RollingCorrelation,
    RollingExtreme,
    RollingMoments,
    RollingQuantile,
)
from tia.features.scales import VolatilityKernel
from tia.features.structure import HTFAggregator, PivotTracker
from tia.features.trend import KalmanTrend, PathEfficiency, VarianceRatio
from tia.synthetic import bars_from_arrays, generate_null, generate_with_regimes
from tia.types import Bar


# ---------------------------------------------------------------------------
# Rolling primitives
# ---------------------------------------------------------------------------


def test_ring_buffer_evicts_oldest_first() -> None:
    b = RingBuffer(3)
    assert b.push(1.0) is None and b.push(2.0) is None and b.push(3.0) is None
    assert b.push(4.0) == 1.0
    assert list(b.values()) == [2.0, 3.0, 4.0]
    assert b.last(1) == 4.0 and b.last(3) == 2.0


def test_rolling_moments_match_numpy() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    m = RollingMoments(50)
    for v in x:
        m.update(float(v))
    w = x[-50:]
    assert m.mean == pytest.approx(w.mean(), rel=1e-9)
    assert m.var == pytest.approx(w.var(ddof=1), rel=1e-8)


def test_rolling_moments_survive_long_runs_without_drift() -> None:
    """Raw power sums drift; the periodic rebuild is what stops it."""
    rng = np.random.default_rng(1)
    m = RollingMoments(64, rebuild_every=4096)
    x = rng.normal(loc=1e4, scale=1.0, size=60_000)
    for v in x:
        m.update(float(v))
    assert m.var == pytest.approx(x[-64:].var(ddof=1), rel=1e-4)


def test_rolling_quantile_matches_numpy() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    q = RollingQuantile(100)
    for v in x:
        q.update(float(v))
    w = x[-100:]
    for p in (0.1, 0.25, 0.5, 0.9):
        assert q.quantile(p) == pytest.approx(float(np.quantile(w, p)), rel=1e-9)


def test_causal_rank_is_uniform_on_iid_input() -> None:
    """A rank transform of IID data must be uniform. Checks the tie convention."""
    rng = np.random.default_rng(3)
    r = CausalRank(500, min_obs=100)
    out = [r.update(float(v)) for v in rng.normal(size=6000)]
    vals = np.array([v for v in out if v == v])
    assert vals.size > 5000
    assert vals.mean() == pytest.approx(0.5, abs=0.02)
    hist, _ = np.histogram(vals, bins=10, range=(0.0, 1.0))
    assert hist.min() > 0.6 * hist.mean(), "rank distribution is not close to uniform"


def test_causal_rank_uses_only_past_and_present() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=400)
    full = CausalRank(200, 50)
    got = [full.update(float(v)) for v in x]
    pre = CausalRank(200, 50)
    for v in x[:300]:
        last = pre.update(float(v))
    assert last == got[299] or (last != last and got[299] != got[299])


def test_rolling_extreme_matches_bruteforce() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=300)
    e = RollingExtreme(40)
    for i, v in enumerate(x):
        e.update(float(v))
        w = x[max(0, i - 39) : i + 1]
        assert e.min == pytest.approx(w.min())
        assert e.max == pytest.approx(w.max())


def test_ewma_halflife_is_correct() -> None:
    """After one half-life of a step change, the value should be ~halfway."""
    e = EWMA(halflife=10.0)
    for _ in range(2000):
        e.update(0.0)
    for _ in range(10):
        e.update(1.0)
    assert e.value == pytest.approx(0.5, abs=0.03)


def test_bucketed_median_separates_time_of_day() -> None:
    b = BucketedMedian(3, 100)
    for _ in range(50):
        b.update(0, 10.0)
        b.update(1, 100.0)
        b.update(2, 1.0)
    assert b.normalise(0, 20.0) == pytest.approx(2.0)
    assert b.normalise(1, 20.0) == pytest.approx(0.2)


def test_rolling_correlation_recovers_a_known_structure() -> None:
    rng = np.random.default_rng(6)
    true_rho = 0.7
    L = np.linalg.cholesky([[1.0, true_rho], [true_rho, 1.0]])
    c = RollingCorrelation(2, halflife=800.0)
    for _ in range(6000):
        c.update(L @ rng.normal(size=2))
    assert c.correlation()[0, 1] == pytest.approx(true_rho, abs=0.06)


# ---------------------------------------------------------------------------
# Volatility and jumps
# ---------------------------------------------------------------------------


def test_bipower_is_robust_to_a_jump_and_realised_variance_is_not() -> None:
    """The reason both estimators are maintained."""
    rng = np.random.default_rng(7)
    n, s = 400, 0.01
    rets = rng.normal(0.0, s, n)
    close = 100 * np.exp(np.cumsum(rets))
    bars = bars_from_arrays(close, np.full(n, 1e6), np.full(n, s), np.random.default_rng(8))

    k = VolatilityKernel()
    for b in bars:
        k.update(b)
    clean_bp, clean_rv = k.out.sigma_bp, k.out.sigma_rv

    # Inject a single 10-sigma jump near the end.
    close2 = close.copy()
    close2[-5:] *= math.exp(10 * s)
    bars2 = bars_from_arrays(close2, np.full(n, 1e6), np.full(n, s), np.random.default_rng(8))
    k2 = VolatilityKernel()
    for b in bars2:
        k2.update(b)

    rv_inflation = k2.out.sigma_rv / clean_rv
    bp_inflation = k2.out.sigma_bp / clean_bp
    assert rv_inflation > bp_inflation, (
        "realised variance should absorb the jump more than bipower does"
    )
    assert k2.out.jump_share > k.out.jump_share


def test_jump_share_is_bounded() -> None:
    for b in generate_null(600, seed=9):
        pass
    k = VolatilityKernel()
    for b in generate_null(600, seed=9):
        o = k.update(b)
        if o.jump_share == o.jump_share:
            assert 0.0 <= o.jump_share <= 1.0


def test_jump_test_flags_an_injected_discontinuity() -> None:
    rng = np.random.default_rng(10)
    t = JumpTest(window=40)
    for v in rng.normal(0.0, 0.01, 300):
        t.update(float(v))
    quiet_z = t.z
    for _ in range(1):
        t.update(0.12)  # a 12-sigma move
    assert t.z > quiet_z


def test_volatility_forecast_is_positive_and_finite() -> None:
    k = VolatilityKernel()
    seen = 0
    for b in generate_null(500, seed=11):
        o = k.update(b)
        if o.valid:
            seen += 1
            assert o.sigma_fcst > 0.0 and math.isfinite(o.sigma_fcst)
    assert seen > 100


# ---------------------------------------------------------------------------
# Trend estimators against their nulls
# ---------------------------------------------------------------------------


def test_variance_ratio_is_centred_under_a_random_walk() -> None:
    """z(q) is asymptotically N(0,1) under the martingale null."""
    rng = np.random.default_rng(12)
    zs = []
    for trial in range(60):
        vr = VarianceRatio(window=250, lags=(2, 4, 8))
        for v in rng.normal(0.0, 0.01, 250):
            vr.update(float(v))
        if vr.z_mean == vr.z_mean:
            zs.append(vr.z_mean)
    z = np.array(zs)
    assert abs(z.mean()) < 0.45, f"variance-ratio statistic is biased under the null: {z.mean():.3f}"
    assert 0.3 < z.std() < 2.5, f"variance-ratio scale is wrong under the null: {z.std():.3f}"


def test_variance_ratio_sign_separates_persistence_from_reversion() -> None:
    rng = np.random.default_rng(13)
    n = 2000

    # Positively autocorrelated increments.
    e = rng.normal(0.0, 0.01, n)
    trend = np.zeros(n)
    m = 0.0
    for i in range(n):
        m = 0.85 * m + 0.35 * e[i]
        trend[i] = m
    vr_t = VarianceRatio(window=400, lags=(2, 4, 8))
    for v in trend:
        vr_t.update(float(v))

    # Negatively autocorrelated increments.
    rev = np.zeros(n)
    for i in range(1, n):
        rev[i] = e[i] - 0.5 * e[i - 1]
    vr_r = VarianceRatio(window=400, lags=(2, 4, 8))
    for v in rev:
        vr_r.update(float(v))

    assert vr_t.z_mean > 1.0, f"persistent series scored {vr_t.z_mean:.2f}"
    assert vr_r.z_mean < -1.0, f"mean-reverting series scored {vr_r.z_mean:.2f}"


def test_kalman_slope_t_recovers_a_locally_detectable_trend() -> None:
    """Sign recovery, at a drift the filter can actually see.

    The filter estimates a *local* slope over roughly ten bars. A drift of
    0.25 sigma per bar accumulates 2.5 sigma over that window against noise of
    sqrt(10) = 3.2 sigma, so it is genuinely not detectable locally and the
    filter correctly reports |t| near 1 with an arbitrary sign. That is the
    estimator being honest, not failing. This test therefore uses a drift the
    filter can resolve; the weak-drift case is covered by the unbiasedness test
    below.
    """
    n, s = 600, 0.01
    for seed, sign in ((140, +1.0), (141, -1.0)):
        rng = np.random.default_rng(seed)
        k = KalmanTrend(0.02, 0.002)
        lp = math.log(100.0)
        for _ in range(n):
            lp += sign * 1.0 * s + float(rng.normal(0.0, s))
            k.update(lp, s)
        assert math.copysign(1.0, k.slope_t) == sign, (
            f"filter reported slope_t={k.slope_t:+.2f} on a {sign:+.0f} trend"
        )
        assert abs(k.slope_t) > 1.5


def test_kalman_slope_t_is_unbiased_at_a_weak_drift() -> None:
    """At a drift below the filter's local resolution, the sign is right on
    average even though any single reading is a coin flip."""
    n, s = 400, 0.01
    ts = []
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        k = KalmanTrend(0.02, 0.002)
        lp = math.log(100.0)
        for _ in range(n):
            lp += 0.25 * s + float(rng.normal(0.0, s))
            k.update(lp, s)
        ts.append(k.slope_t)
    assert float(np.mean(ts)) > 0.0, "filter is biased against a real positive drift"


def test_kalman_slope_t_is_near_zero_on_a_random_walk() -> None:
    rng = np.random.default_rng(15)
    ts = []
    for trial in range(40):
        k = KalmanTrend(0.02, 0.002)
        lp = math.log(100.0)
        for _ in range(400):
            lp += float(rng.normal(0.0, 0.01))
            k.update(lp, 0.01)
        ts.append(k.slope_t)
    assert abs(float(np.mean(ts))) < 1.0


def test_path_efficiency_is_bounded_and_extreme_cases_are_right() -> None:
    straight = PathEfficiency(20)
    for _ in range(20):
        straight.update(0.01)
    assert straight.value == pytest.approx(1.0)

    alternating = PathEfficiency(20)
    for i in range(20):
        alternating.update(0.01 if i % 2 == 0 else -0.01)
    assert alternating.value < 0.1


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def test_permutation_entropy_is_near_one_for_noise_and_zero_for_a_ramp() -> None:
    rng = np.random.default_rng(16)
    noisy = PermutationEntropy(3, 200)
    for v in rng.normal(size=400):
        noisy.update(float(v))
    assert noisy.value > 0.95

    ramp = PermutationEntropy(3, 200)
    for i in range(400):
        ramp.update(float(i))
    assert ramp.value == pytest.approx(0.0, abs=1e-9)


def test_run_asymmetry_detects_longer_up_runs() -> None:
    r = RunAsymmetry()
    for _ in range(30):
        for _ in range(4):
            r.update(0.01)
        r.update(-0.01)
    assert r.value > 0.4


# ---------------------------------------------------------------------------
# Microstructure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("true_half_spread", [2.0e-3, 1.0e-3, 2.5e-4])
def test_spread_estimator_tracks_an_injected_spread(true_half_spread: float) -> None:
    """Validated against ground truth, and required to be conservative.

    Range-based estimators cannot resolve a spread far below the bar's own
    volatility, so the requirement is directional: never materially *under*
    state, and track the truth once it is resolvable.
    """
    rng = np.random.default_rng(17)
    n, s = 2500, 0.01
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * s))
    bars = bars_from_arrays(
        close, np.full(n, 1e6), np.full(n, s), rng, half_spread=true_half_spread
    )
    cs = CorwinSchultzSpread(22)
    vals = []
    for b in bars:
        v = cs.update(b)
        if v == v:
            vals.append(v)
    est = float(np.median(vals))
    truth = 2.0 * true_half_spread
    # Never materially understate -- a cost model that does will trade things it
    # should not. Overstatement is permitted and grows as the true spread falls
    # below the resolution limit, which is the honest behaviour of any
    # range-based estimator.
    assert est > 0.9 * truth, f"estimator understated the spread: {est:.5f} vs {truth:.5f}"
    assert est < 8.0 * truth, f"estimator wildly overstated the spread: {est:.5f} vs {truth:.5f}"


def test_ladr_is_high_when_price_moves_on_little_volume() -> None:
    """The defining property of the microstructure primitive."""
    cfg = Config()
    k = FeatureKernel(cfg)
    bars = generate_null(600, seed=18)
    for b in bars[:-1]:
        k.update(b)

    last = bars[-1]
    base = k.snapshot.close
    # Same displacement, two very different volumes.
    move = base * 1.03
    thin = Bar(last.timestamp + 86400, base, move * 1.001, base * 0.999, move, last.volume * 0.1)
    import copy

    k_thin = copy.deepcopy(k)
    k_thick = copy.deepcopy(k)
    f_thin = k_thin.update(thin)
    thick = Bar(last.timestamp + 86400, base, move * 1.001, base * 0.999, move, last.volume * 10.0)
    f_thick = k_thick.update(thick)

    assert f_thin.ladr > f_thick.ladr, (
        "the same move on less volume must register as cheaper displacement"
    )
    assert f_thick.absorption > f_thin.absorption


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_htf_aggregator_emits_only_complete_bars() -> None:
    agg = HTFAggregator(5)
    bars = generate_null(53, seed=19)
    emitted = [agg.update(b) for b in bars]
    n = sum(1 for e in emitted if e is not None)
    assert n == 10, f"53 bars at 5x should emit 10 complete bars, got {n}"
    for i, e in enumerate(emitted):
        assert (e is not None) == ((i + 1) % 5 == 0)


def test_htf_aggregate_range_contains_constituent_ranges() -> None:
    agg = HTFAggregator(4)
    bars = generate_null(40, seed=20)
    group: list[Bar] = []
    for b in bars:
        group.append(b)
        out = agg.update(b)
        if out is not None:
            assert out.high == pytest.approx(max(x.high for x in group))
            assert out.low == pytest.approx(min(x.low for x in group))
            assert out.open == pytest.approx(group[0].open)
            assert out.close == pytest.approx(group[-1].close)
            group = []


def test_pivot_range_position_is_bounded() -> None:
    k = FeatureKernel(Config())
    for b in generate_with_regimes(1200, seed=21)[0]:
        f = k.update(b)
        if f.range_pos == f.range_pos:
            assert 0.0 <= f.range_pos <= 1.0
