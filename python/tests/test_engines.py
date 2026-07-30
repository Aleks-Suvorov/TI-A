"""Tests for the engines, the regime filter and the Edge Book."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tia.config import Config
from tia.engines.base import band_score, linear_score, squash
from tia.engines.edgebook import EdgeBook, SetupFamily, norm_ppf, student_t_ppf
from tia.engines.regime import DESIGN, FEATURE_ORDER, RegimeEngine, beta_logpdf
from tia.features.kernel import FeatureKernel
from tia.pipeline import TIA
from tia.synthetic import generate_null, generate_with_regimes
from tia.types import Regime


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def test_squash_is_bounded_and_monotone() -> None:
    xs = [-100.0, -3.0, 0.0, 1.0, 100.0]
    ys = [squash(x, 2.0) for x in xs]
    assert all(-1.0 <= y <= 1.0 for y in ys)
    assert ys == sorted(ys)
    assert squash(0.0) == 0.0
    assert squash(float("nan")) == 0.0


def test_band_score_prefers_the_middle() -> None:
    assert band_score(0.5, 0.25, 0.8) == 1.0
    assert band_score(0.05, 0.25, 0.8) < 0.5
    assert band_score(0.99, 0.25, 0.8) < 0.5


def test_linear_score_maps_endpoints() -> None:
    assert linear_score(0.0, 0.0, 1.0) == pytest.approx(-1.0)
    assert linear_score(1.0, 0.0, 1.0) == pytest.approx(1.0)
    assert linear_score(0.5, 0.0, 1.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Regime measurement model
# ---------------------------------------------------------------------------


def test_beta_logpdf_matches_the_analytic_uniform_case() -> None:
    """Beta(1,1) must be exactly uniform -- this is the 'no claim' encoding."""
    for x in (0.01, 0.3, 0.5, 0.9, 0.99):
        assert beta_logpdf(x, 1.0, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_beta_logpdf_integrates_to_one() -> None:
    for a, b in ((4.8, 1.2), (1.2, 4.8), (3.0, 3.0)):
        xs = np.linspace(1e-4, 1 - 1e-4, 20001)
        dens = np.exp([beta_logpdf(float(x), a, b) for x in xs])
        assert float(np.trapezoid(dens, xs)) == pytest.approx(1.0, abs=2e-3)


def test_regime_design_covers_every_feature() -> None:
    for reg, row in DESIGN.items():
        assert len(row) == len(FEATURE_ORDER), f"{reg.name} design row has wrong width"
    assert set(DESIGN) == set(Regime)


def test_regime_design_encodes_the_two_opposing_theories() -> None:
    """TREND and REVERT must disagree on autocorrelation, which is the whole
    point of separating them (``docs/01-THEORY.md`` §0.1)."""
    j = FEATURE_ORDER.index("vr_rank")
    assert DESIGN[Regime.TREND][j] is not None
    assert DESIGN[Regime.REVERT][j] is not None
    assert DESIGN[Regime.TREND][j] > DESIGN[Regime.REVERT][j]

    k = FEATURE_ORDER.index("absorption_rank")
    assert DESIGN[Regime.TREND][k] < DESIGN[Regime.REVERT][k]


def test_regime_posterior_is_a_probability_vector() -> None:
    eng = RegimeEngine(Config())
    kern = FeatureKernel(Config())
    for bar in generate_with_regimes(800, seed=30)[0]:
        kern.update(bar)
        st = eng.step(kern.regime_features())
        assert st.posterior.shape == (4,)
        assert float(st.posterior.sum()) == pytest.approx(1.0, abs=1e-9)
        assert np.all(st.posterior >= 0.0)
        assert 0.0 <= st.hazard <= 1.0


def test_regime_filter_separates_a_trending_from_a_reverting_market() -> None:
    """Recovery test: given markets that really do differ, the posterior must.

    Not a test that the labels are 'correct' in an absolute sense -- the design
    table defines what TREND means -- but that the filter discriminates two
    genuinely different data-generating processes, which is the necessary
    condition for the regime layer to be doing anything at all.
    """
    cfg = Config()

    def mean_posterior(bars) -> np.ndarray:
        eng, kern = RegimeEngine(cfg), FeatureKernel(cfg)
        acc, n = np.zeros(4), 0
        for i, bar in enumerate(bars):
            kern.update(bar)
            st = eng.step(kern.regime_features())
            if i > 400 and st.valid:
                acc += st.posterior
                n += 1
        return acc / max(n, 1)

    rng = np.random.default_rng(31)
    n, s = 3000, 0.01
    # A strongly persistent series.
    e = rng.normal(0.0, s, n)
    mom, lp, trend_close = 0.0, math.log(100.0), []
    for i in range(n):
        mom = 0.9 * mom + 0.5 * e[i]
        lp += mom
        trend_close.append(math.exp(lp))
    # A strongly mean-reverting series.
    lp, anchor, rev_close = math.log(100.0), math.log(100.0), []
    for i in range(n):
        lp += -0.35 * (lp - anchor) + float(e[i])
        rev_close.append(math.exp(lp))

    from tia.synthetic import bars_from_arrays

    tb = bars_from_arrays(np.array(trend_close), np.full(n, 1e6), np.full(n, s), np.random.default_rng(32))
    rb = bars_from_arrays(np.array(rev_close), np.full(n, 1e6), np.full(n, s), np.random.default_rng(32))

    pt, pr = mean_posterior(tb), mean_posterior(rb)
    trend_lean = pt[int(Regime.TREND)] - pt[int(Regime.REVERT)]
    revert_lean = pr[int(Regime.TREND)] - pr[int(Regime.REVERT)]
    assert trend_lean > revert_lean, (
        f"filter does not discriminate: trend-market lean {trend_lean:+.3f}, "
        f"revert-market lean {revert_lean:+.3f}"
    )


def test_regime_hazard_rises_when_the_posterior_moves() -> None:
    eng = RegimeEngine(Config())
    kern = FeatureKernel(Config())
    hazards = []
    for bar in generate_with_regimes(1500, seed=33, mean_dwell=120)[0]:
        kern.update(bar)
        st = eng.step(kern.regime_features())
        if st.valid:
            hazards.append(st.hazard)
    h = np.array(hazards)
    assert h.size > 500
    assert h.max() > 0.2, "hazard never rises; transitions are not being detected"
    assert h.min() < 0.5, "hazard never falls; the filter never commits"


def test_regime_tempering_stays_in_range_on_real_features() -> None:
    """On the actual feature set the exponent sits at its cap.

    That is a finding about the features, not a broken mechanism: the eight
    regime features include strongly *negatively* correlated pairs -- LADR and
    absorption are near-reciprocal by construction -- so their effective count
    is not below eight and no down-weighting is called for. The exponent is
    capped at 1.0 so that tempering can only ever soften the likelihood, never
    sharpen it. The mechanism itself is exercised by the next test.
    """
    eng = RegimeEngine(Config())
    kern = FeatureKernel(Config())
    for bar in generate_null(900, seed=34):
        kern.update(bar)
        st = eng.step(kern.regime_features())
    assert 0.15 <= st.temper <= 1.0


def test_regime_tempering_engages_on_redundant_features() -> None:
    """Feed the filter eight copies of one feature; it must discount them."""
    eng = RegimeEngine(Config())
    rng = np.random.default_rng(35)
    st = None
    for _ in range(400):
        v = float(rng.random())
        st = eng.step({k: v for k in FEATURE_ORDER})
    assert st is not None
    assert st.temper < 0.4, (
        f"tempering exponent {st.temper:.3f}: eight identical features are "
        "being counted as eight independent pieces of evidence"
    )


# ---------------------------------------------------------------------------
# Quantile functions
# ---------------------------------------------------------------------------


def test_norm_ppf_matches_known_quantiles() -> None:
    for p, expected in ((0.5, 0.0), (0.975, 1.959964), (0.95, 1.644854), (0.1, -1.281552)):
        assert norm_ppf(p) == pytest.approx(expected, abs=1e-5)


def test_student_t_ppf_approaches_normal_at_high_dof() -> None:
    assert student_t_ppf(0.1, 10_000) == pytest.approx(norm_ppf(0.1), abs=1e-3)


def test_student_t_ppf_has_fatter_tails_than_normal() -> None:
    for dof in (5.0, 10.0, 30.0):
        assert student_t_ppf(0.1, dof) < norm_ppf(0.1), f"t({dof}) tail is not fatter"


def test_student_t_ppf_is_reasonably_accurate() -> None:
    """Against known table values for the 10th percentile."""
    known = {5.0: -1.476, 10.0: -1.372, 20.0: -1.325, 30.0: -1.310}
    for dof, expected in known.items():
        assert student_t_ppf(0.1, dof) == pytest.approx(expected, abs=0.02)


# ---------------------------------------------------------------------------
# Edge Book
# ---------------------------------------------------------------------------


def test_untrained_edge_book_reports_no_edge() -> None:
    """The cold-start property: absent evidence, the bound is below zero."""
    book = EdgeBook(Config())
    est = book.estimate(int(Regime.TREND), int(SetupFamily.CONTINUATION), 2)
    assert est.mean == pytest.approx(0.0, abs=1e-9)
    assert est.lcb < 0.0, "an empty cell must not clear the expectancy gate"
    assert est.is_thin


def test_edge_book_shrinks_a_thin_cell_toward_zero() -> None:
    """Five spectacular observations must not produce a spectacular estimate."""
    book = EdgeBook(Config())
    for _ in range(5):
        book.observe(0, 0, 3, 2.0, weight=1.0)
    est = book.estimate(0, 0, 3)
    # Leave-one-out shrinkage: the prior must not be informed by these same five
    # observations, so the posterior mean is 10/(25+5) = 0.333, not 0.77.
    assert est.mean == pytest.approx(10.0 / 30.0, abs=0.02), (
        f"posterior mean {est.mean:.3f}; the hierarchy may be double-counting"
    )
    assert est.lcb < 0.35, "five observations should not produce a confident bound"


def test_edge_book_learns_from_sufficient_evidence() -> None:
    book = EdgeBook(Config())
    rng = np.random.default_rng(40)
    for _ in range(1200):
        book.observe(0, 0, 3, float(rng.normal(0.45, 1.0)), weight=1.0)
    est = book.estimate(0, 0, 3)
    assert est.mean == pytest.approx(0.45, abs=0.10)
    assert est.lcb > 0.0, "with 1200 clean observations the bound should clear zero"
    assert est.lcb < est.mean < est.ucb


def test_edge_book_lower_bound_lies_below_the_mean() -> None:
    book = EdgeBook(Config())
    rng = np.random.default_rng(41)
    for n in (10, 100, 1000):
        b = EdgeBook(Config())
        for _ in range(n):
            b.observe(1, 1, 2, float(rng.normal(0.3, 1.2)))
        e = b.estimate(1, 1, 2)
        assert e.lcb < e.mean

    widths = []
    for n in (20, 200, 2000):
        b = EdgeBook(Config())
        r = np.random.default_rng(42)
        for _ in range(n):
            b.observe(1, 1, 2, float(r.normal(0.3, 1.2)))
        e = b.estimate(1, 1, 2)
        widths.append(e.mean - e.lcb)
    assert widths[0] > widths[1] > widths[2], "credible interval must narrow with data"


def test_edge_book_hierarchy_pools_across_cells() -> None:
    """A cell with no data borrows from its parent, not from the global prior."""
    book = EdgeBook(Config())
    rng = np.random.default_rng(43)
    for b in range(4):
        for _ in range(400):
            book.observe(2, 1, b, float(rng.normal(0.5, 1.0)))
    empty = book.estimate(2, 1, 4)
    assert empty.mean > 0.1, (
        f"empty cell mean {empty.mean:.3f} did not borrow from a strongly positive parent"
    )


def test_edge_book_decays_old_evidence() -> None:
    cfg = Config().replace(edge_halflife_bars=100.0)
    book = EdgeBook(cfg)
    for _ in range(500):
        book.observe(0, 0, 0, 1.0)
    before = book.estimate(0, 0, 0)
    for _ in range(1000):
        book.tick()
    after = book.estimate(0, 0, 0)
    assert after.n_eff < 0.01 * before.n_eff
    assert abs(after.mean) < abs(before.mean)


def test_edge_book_state_round_trips() -> None:
    book = EdgeBook(Config())
    rng = np.random.default_rng(44)
    for _ in range(200):
        book.observe(int(rng.integers(4)), int(rng.integers(4)), int(rng.integers(5)),
                     float(rng.normal(0.2, 1.0)))
    state = book.state_dict()
    other = EdgeBook(Config())
    other.load_state_dict(state)
    for key in ((0, 0, 0), (1, 2, 3), (3, 3, 4)):
        a, b = book.estimate(*key), other.estimate(*key)
        assert a.mean == pytest.approx(b.mean)
        assert a.lcb == pytest.approx(b.lcb)
        assert a.n_eff == pytest.approx(b.n_eff)


def test_bucket_boundaries_are_monotone_in_evidence() -> None:
    book = EdgeBook(Config())
    buckets = [book.bucket_of(x) for x in (0.0, 0.3, 0.7, 1.2, 3.0)]
    assert buckets == sorted(buckets)
    assert book.bucket_of(2.0) == book.bucket_of(-2.0), "buckets must be sign-symmetric"


# ---------------------------------------------------------------------------
# Engine contract compliance
# ---------------------------------------------------------------------------


def test_every_engine_respects_the_output_contract() -> None:
    sysm = TIA(Config())
    checked = 0
    for bar in generate_with_regimes(900, seed=45)[0]:
        d = sysm.on_bar(bar)
        for name, out in d.engine_outputs.items():
            assert -1.0 <= out.score <= 1.0, f"{name} score out of range"
            assert 0.0 <= out.reliability <= 1.0, f"{name} reliability out of range"
            assert out.score == out.score, f"{name} emitted NaN score"
            if not out.valid:
                assert out.score == 0.0 and out.reliability == 0.0
            checked += 1
    assert checked > 5000


def test_engines_abstain_during_warmup() -> None:
    sysm = TIA(Config())
    bars = generate_null(300, seed=46)
    for i, bar in enumerate(bars[:80]):
        d = sysm.on_bar(bar)
        for name, out in d.engine_outputs.items():
            assert not out.valid, f"{name} claimed validity at bar {i}, before warmup"


def test_optional_engines_abstain_without_their_data() -> None:
    sysm = TIA(Config())
    for bar in generate_null(500, seed=47):
        d = sysm.on_bar(bar)
    assert d.engine_outputs["positioning"].reliability == 0.0
    assert d.engine_outputs["crossasset"].reliability == 0.0
