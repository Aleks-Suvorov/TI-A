# Statistical and Economic Assumptions

Every assumption the system rests on, tagged **TESTABLE** (with the test and
where it lives) or **UNTESTABLE** (with the consequence if false, and what keeps
the outcome merely unprofitable rather than catastrophic).

---

## 1. Distributional

### A1. Rank-transformed features are approximately stationary — **TESTABLE**
Weaker than assuming return stationarity: a rank is uniform by construction over
its own window, so level and scale shifts are absorbed. What is assumed is that
the *joint* dependence structure is stable.
**Test.** Population stability index against a frozen development reference
(`monitoring/drift.py`), alarm at `psi_alarm = 0.25`.
**Fails when.** A structural break makes the trailing 252 bars an irrelevant
reference class. The rank of a feature then describes a world that no longer
exists, and it does so without any NaN or error to signal it.

### A2. Outcome returns are conditionally Gaussian within an Edge Book cell — **TESTABLE, and known to be false**
The Normal-Inverse-Gamma model assumes Gaussian outcomes. Triple-barrier outcomes
are **truncated by construction**: most mass sits at exactly `+target_sigma` and
`−stop_sigma`, with a continuous timeout component between. The true distribution
is closer to a three-point mixture.

**Direction of the error.** A Gaussian fitted to a bimodal, truncated
distribution *over*-estimates the variance, because it must span both atoms. An
over-estimated `ς²` makes the posterior on `μ` wider than it should be, which
makes the lower credible bound **too low** and the Kelly fraction **too small**.
So the approximation error is *conservative*: it costs opportunity, not capital.
This is why it is tolerated rather than fixed.
**Test.** Compare the NIG posterior against a Dirichlet-multinomial over
`{target, stop, timeout}` on the same cells. Not currently implemented.

### A3. Engine evidence is jointly Gaussian around a common latent log-odds — **UNTESTABLE in practice**
The generative model behind CALOP. Engine scores are bounded in `[-1,1]` and
mapped through `artanh`, so the transformed evidence is unbounded, but there is
no reason it is Gaussian.
**Consequence if false.** The variance propagation and hence the credible
interval are wrong. The *point* estimate `L̂` is a GLS estimator and remains the
best linear unbiased combination regardless of Gaussianity, so the direction and
ranking survive; only the stated uncertainty is affected.
**Containment.** The interval is displayed, monitored against realised
calibration, and the Brier alarm catches systematic error.

---

## 2. Independence and conditioning

### A4. Conditional independence of engines, *after* accounting for correlation — **PARTIALLY TESTABLE**
CALOP does not assume independence; it estimates and corrects for linear
correlation. What remains assumed is that there is no *higher-order* dependence —
that two engines cannot be linearly uncorrelated yet fail together.
**Test.** Ablation studies; comparing realised joint failure rates against those
implied by the correlation matrix.
**Consequence if false.** Effective breadth is overstated and confidence with it.
See `docs/08-FAILURE-MODES.md` §2.1 — this is the severe one.

### A5. The triple-barrier outcome is a sufficient statistic for a trade's value — **UNTESTABLE**
Assumes a trade's worth is fully captured by which barrier it hit and when.
Ignores path: two trades reaching the same target, one smoothly and one after a
90%-of-stop excursion, are recorded identically.
**Consequence if false.** Risk is understated, because maximum adverse excursion
carries information about slippage, margin and the psychological sustainability
of the strategy.
**Containment.** The vertical barrier bounds holding time; the drawdown throttle
responds to realised equity rather than to labels.

### A6. Sample weights correct for overlap adequately — **TESTABLE**
Uniqueness weighting reduces the effective sample; whether it reduces it *enough*
is an empirical question.
**Test.** `effective_sample_size` vs candidate count (measured: ratio ~0.75); the
t-statistic computed against the effective count is reported separately.

---

## 3. Market microstructure and economics

### A7. The square-root price-impact law — **TESTABLE, well supported**
`impact = η√q`. Among the better-supported empirical regularities in execution.
**Test.** Realised vs modelled slippage, `slippage_alarm_mult`.
**Fails when.** Order sizes approach a large fraction of available liquidity, or
in stressed conditions where the exponent rises.

### A8. Our own trading does not move the market — **UNTESTABLE from inside**
The cost model prices *our* impact but assumes no strategic response and no
crowding by others running similar logic.
**Consequence if false.** Edge decays, and it decays fastest in exactly the setups
that work best.
**Containment.** The capacity bound (`capacity_participation`) refuses size the
cost model says is too large. This limits self-impact but does nothing about
crowding.

### A9. Kyle's interpretation of price impact per unit volume — **UNTESTABLE**
LADR's entire meaning rests on the claim that displacement per unit of consumed
liquidity discriminates informed from uninformed flow.
**Consequence if false.** The momentum engine measures nothing, and the
architecture loses condition (C).
**Containment.** LADR is confirming, not originating; ablation would reveal it.

### A10. The cost model remains valid in stress — **TESTABLE, and the test is unreliable**
`stress_multiplier` scales liquidity-sensitive components by up to 2.5x, a number
chosen by judgement rather than measurement.
**Consequence if false.** Every expectancy in a crisis is overstated.
**Containment.** The `STRESS` regime cuts size; the slippage alarm trips.
**Honest note.** The data needed to calibrate crisis costs is sparse by
definition, and estimating it from the few available episodes is itself a
small-sample problem.

---

## 4. Regime model

### A11. The Beta measurement design describes what the regimes mean — **UNTESTABLE as stated**
The 32 design cells are assertions, not estimates. They cannot be "wrong" in a
statistical sense because they *define* the regimes.
**Consequence if poorly chosen.** The posterior tracks something, but not the
thing that separates continuation from reversal, and evidence is misrouted.
**Containment.** The falsifiable prediction that profitable action differs
between `TREND` and `REVERT` (`docs/00-PREREGISTRATION.md`) tests the taxonomy's
*usefulness* even though the design itself is not estimable.

### A12. Regime transitions are Markov with constant stickiness — **TESTABLE, certainly false in detail**
Real regime persistence is duration-dependent; a regime that has lasted 200 bars
is not equally likely to end as one that has lasted 5.
**Consequence.** Transition timing is mis-estimated, mostly making the filter
slower to switch than it should be.
**Containment.** The hazard gate stands the system down through transitions, so
the cost of a slow switch is missed opportunity rather than a wrong-way position.

---

## 5. Meta-assumptions

### A13. Historical data is representative of the future — **UNTESTABLE, and the load-bearing one**
Everything rests on this. It cannot be tested from inside the sample.
**Containment.** Cross-decade and cross-market validation raises the bar; the
demotion ladder responds when live behaviour diverges; position sizing is
fractional-Kelly with an uncertainty haircut so that being wrong is survivable.

### A14. Ergodicity — **UNTESTABLE**
Time averages over one instrument's history are assumed to approximate ensemble
averages. This is why cross-sectional pooling is used for inference: it converts
a partly-ergodic assumption into a partly-cross-sectional one, which is weaker
but not eliminated.

### A15. Determinism and reproducibility — **TESTABLE**
No wall clock, no unseeded randomness, no dict-ordering dependence.
**Test.** `tests/test_causality.py::test_determinism_across_identical_runs`.

### A16. Causality — **TESTABLE, and tested by brute force**
No computation at bar `t` uses information from after `t`.
**Test.** `tests/test_causality.py` — prefix replay against full-history replay,
requiring bit-identical output, plus a stability test asserting that appending
bars never alters an emitted decision.

---

## 6. Summary

| Class | Testable | Untestable |
|---|---|---|
| Distributional | A1, A2 | A3 |
| Independence | A4 (partly), A6 | A5 |
| Microstructure | A7, A10 (weakly) | A8, A9 |
| Regime | A12 | A11 |
| Meta | A15, A16 | A13, A14 |

The untestable assumptions cluster in economics, not in statistics. That is the
expected shape: statistical assumptions can be checked against data, while claims
about *why* a pattern exists cannot. The system's defence against all of them is
the same and is not clever — trade a fraction of Kelly on a lower credible bound,
and be willing to stop.
