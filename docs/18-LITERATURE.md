# Annotated Bibliography

Organised by the role each work plays in TI-A. `docs/17-EVIDENCE-REVIEW.md`
grades the underlying *concepts*; this document records the *sources*.

**Citation integrity.** Entries are restricted to work whose authorship and
approximate date are well established. Where a finding is well replicated but no
single canonical citation is appropriate, it is described without one rather than
attached to a plausible guess. Anything uncertain is marked `[verify]`.

---

## Market microstructure and price impact

**Kyle, A. (1985). "Continuous Auctions and Insider Trading." *Econometrica*.**
The model behind the system's central microstructure primitive. Equilibrium price
impact per unit of order flow, λ, increases with the informativeness of flow — so
displacement per unit of consumed liquidity discriminates informed flow (permanent
impact) from uninformed pressure (transient). This is what LADR measures and the
entire justification for condition (C) in `docs/01-THEORY.md` §0.1.

**Amihud, Y. (2002). "Illiquidity and Stock Returns." *Journal of Financial
Markets*.** Operationalises the reciprocal of Kyle's λ as |return| ÷ dollar
volume. Computed directly in `features/microstructure.py` and used as a ranked
illiquidity feature.

**Glosten, L. and Milgrom, P. (1985).** Sequential-trade model of the spread as
compensation for adverse selection. Background for why spread must be estimated
rather than assumed, and why it widens in exactly the conditions where signals
fire.

**Corwin, S. and Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads
from Daily High and Low Prices." *Journal of Finance*.** Recovers the effective
spread from the observation that highs are more likely buyer-initiated and lows
seller-initiated, so one- and two-bar ranges decompose into volatility plus
spread. Implemented, and the single reason cost estimation is possible in an
OHLCV-only system. Measured behaviour against injected spreads is recorded in
`tests/test_features.py`: monotone and never understating, from 0.97× at 40bp to
3.6× at 2bp.

**Abdi, F. and Ranaldo, A. (2017). "A Simple Estimation of Bid-Ask Spreads from
Daily Close, High, and Low Prices." *Review of Financial Studies*.** A second
route to the same quantity via the covariance of the close's deviation from the
mid-range. Implemented as a cross-check. Not used for the estimate itself: it is
not monotone in the truth over the range tested and understates at wide spreads.

**Almgren, R., Thum, C., Hauptmann, E. and Li, H. (2005). "Direct Estimation of
Equity Market Impact."** Empirical support for the square-root impact law used in
`decision/costs.py` and in the capacity bound.

**Hasbrouck, J. (2009). "Trading Costs and Returns for US Equities."**
Establishes that low-frequency spread proxies can be estimated from daily data,
and how well. Supports the OHLCV-only cost approach and bounds expectations for
its accuracy.

---

## Volatility: estimation and forecasting

The strongest-evidence area in the entire review, and the reason the risk
denominator is estimated first.

**Engle, R. (1982)** and **Bollerslev, T. (1986).** ARCH and GARCH. The
foundation of volatility clustering as a modelling target. GARCH(1,1) is fitted in
`validation/montecarlo.py` to generate surrogate nulls.

**Barndorff-Nielsen, O. and Shephard, N. (2004).** Bipower variation: a
jump-robust estimator of integrated variance, and the RV−BPV difference as jump
variation. `sigma_bp` is the system's primary denominator precisely because
dividing by a jump-contaminated scale makes genuine displacement look ordinary.
The associated ratio test is implemented in `features/jumps.py`.

**Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized
Volatility." *Journal of Financial Econometrics*.** The HAR specification. TI-A
uses a fixed geometric blend across the same three horizons rather than fitted
coefficients, deliberately trading a small accuracy loss for a `THEORY` rather
than `DEV` parameter tag.

**Garman, M. and Klass, M. (1980)** and **Rogers, L. and Satchell, S. (1991).**
Range-based volatility estimators. Rogers–Satchell is drift-independent, which
matters in exactly the trending conditions where a position would be sized. Both
implemented; their disagreement with close-to-close estimators is itself
informative about intrabar reversal.

**Andersen, T. and Bollerslev, T. (1998).** Establishes that realised volatility
from high-frequency data is a far better proxy for true volatility than daily
squared returns — the empirical grounding for treating volatility as the
forecastable quantity.

---

## Return predictability

**Lo, A. and MacKinlay, A. (1988). "Stock Market Prices Do Not Follow Random
Walks: Evidence from a Simple Specification Test." *Review of Financial
Studies*.** The variance-ratio test with its heteroskedasticity-robust variant.
Used in place of a Hurst exponent because it comes with an actual null
distribution, and the robust form specifically because the homoskedastic version
rejects the martingale null in every volatility cluster for that reason alone.

**Moskowitz, T., Ooi, Y. and Pedersen, L. (2012). "Time Series Momentum."
*Journal of Financial Economics*.** Time-series momentum across asset classes and
decades. One of the two directional effects the system rests on, and the
justification for the `TREND` regime.

**Jegadeesh, N. (1990)** and **Lehmann, B. (1990).** Short-horizon reversal in
individual securities. Early evidence for the effect the `REVERT` regime targets.

**Nagel, S. (2012). "Evaporating Liquidity." *Review of Financial Studies*.**
The most important single citation for the liquidity engine. Short-horizon
reversal returns are compensation for liquidity provision and are strongest when
liquidity provision is most expensive. This is what the sweep-and-fail detector
measures, and it is the defensible version of what practitioners call a stop run.

**Jegadeesh, N. and Titman, S. (1993).** Cross-sectional momentum. Not used
directly — the system is single-instrument — but the primary motivation for the
cross-sectional ranking item at the top of `docs/16-ROADMAP.md`.

---

## Regime detection and filtering

**Hamilton, J. (1989). "A New Approach to the Economic Analysis of Nonstationary
Time Series and the Business Cycle." *Econometrica*.** The Markov regime-switching
framework. TI-A adopts the *structure* — a sticky transition matrix and a forward
filter — while rejecting the *estimation*: the measurement model is specified a
priori rather than fitted, which removes the instability and leakage that fitting
a hidden Markov model to returns introduces.

**Kalman, R. (1960).** The linear filter. Used for the local-linear-trend model in
`features/trend.py`, forward pass only. The smoother, which most published
"Kalman trend" examples actually plot, conditions on future observations and would
repaint.

---

## Information theory

**Bandt, C. and Pompe, B. (2002). "Permutation Entropy: A Natural Complexity
Measure for Time Series." *Physical Review Letters*.** Ordinal-pattern entropy.
Chosen over sample and approximate entropy because it needs no tolerance
parameter — which would have added a fitted degree of freedom — and is invariant
to any monotone transform of price.

---

## Financial machine learning and validation

**López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.**
Four techniques used directly and non-negotiably: the **triple-barrier method**
(`labeling/triple_barrier.py`), **sample uniqueness weighting** for overlapping
labels (`labeling/weights.py`), **purged and embargoed cross-validation**
(`validation/cpcv.py`), and **meta-labelling**, which is the pattern the
counterfactual training pass in `training.py` implements. The book's central
warning — that overlapping labels inflate every t-statistic computed as if they
were independent — is the reason effective sample size is reported everywhere
alongside trade counts.

**Bailey, D. and López de Prado, M. (2012, 2014).** The **probabilistic Sharpe
ratio** and the **deflated Sharpe ratio**, correcting for non-normality, sample
length, and the number of trials. Implemented in `validation/metrics.py`;
`trials_ledger_count` feeds the latter.

**Politis, D. and Romano, J. (1994). "The Stationary Bootstrap." *Journal of the
American Statistical Association*.** Geometric-block resampling that preserves
serial dependence. Used for confidence intervals on performance statistics.

**White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*.**
The formal framing of multiple-testing in strategy evaluation; background for the
trials ledger discipline.

---

## Probability calibration

**Brier, G. (1950).** The Brier score.

**Murphy, A. (1973).** Decomposition of the Brier score into reliability,
resolution and uncertainty. The distinction is used operationally: a forecaster
that always predicts the base rate is perfectly calibrated and completely useless,
and only *resolution* is worth anything to a trading system.

**Platt, J. (1999).** Sigmoid fitting of classifier outputs to probabilities.
Implemented directly by Newton–Raphson in `fusion/calibration.py` so the live path
keeps its numpy-only dependency.

**Zadrozny, B. and Elkan, C. (2002).** Isotonic regression for calibration. The
pool-adjacent-violators map used on top of the parametric link. Its degenerate
endpoints — exactly 0 and 1 — are a documented pathology fixed in
`docs/05-CONFIDENCE-CALIBRATION.md` §3.

**MacKay, D. (1992).** The probit approximation to the Gaussian-integrated
logistic, `p ≈ σ(μ/√(1 + πσ²/8))`. This is the mechanism that pulls stated
confidence toward 50% when evidence is narrow, and it is what makes a reported 97%
require evidence that is both strong *and* broad. `[verify — the approximation is
standard and appears in MacKay's work on Bayesian neural networks; the precise
1992 attribution should be checked before citing formally.]`

---

## Portfolio and risk

**Kelly, J. (1956).** The growth-optimal betting fraction. Applied with the
Bayesian correction of `docs/01-THEORY.md` §9.1: parameter uncertainty enters as
additional variance, so an uncertain edge deserves a smaller bet for the same
reason a volatile one does.

**Grinold, R. (1989). "The Fundamental Law of Active Management."**
`IR ≈ IC · √breadth`. `docs/01-THEORY.md` §6 Proposition 2 shows this relation
falls out of the CALOP pooling algebra with *measured* effective breadth rather
than an assumed count of independent bets — which is the more useful form, since
the assumed count is almost always wrong.

---

## Market behaviour and seasonality

**Admati, A. and Pfleiderer, P. (1988). "A Theory of Intraday Patterns."
*Review of Financial Studies*.** Theoretical account of the intraday U-shape in
volume and volatility. The empirical pattern is among the most robustly replicated
in the literature and is the reason every volume feature is normalised within a
time-of-day bucket — an unconditional comparison largely measures what hour it is.

**Harris, L. (1986).** Intraday and day-of-week return patterns. Supports the
session structure in `data/sessions.py`; the *directional* content of calendar
effects is treated as far weaker than the volume and volatility structure.

**Savor, P. and Wilson, M. (2013).** Elevated returns on scheduled macroeconomic
announcement days. The basis for the event-proximity term in the behavioural
engine, which is applied only when a calendar is supplied and is otherwise
omitted rather than assumed clear.

**Easley, D., López de Prado, M. and O'Hara, M. (2012).** The volume clock and
volume-synchronised probability of informed trading. The argument for sampling by
transacted value rather than by the calendar; implemented in `data/clocks.py` and
disabled by default for the architectural reasons in `docs/16-ROADMAP.md` §6.

---

## On what is *not* cited

The Smart Money / ICT literature has no peer-reviewed base and is largely not
falsifiable as stated. Two of its components have measurable analogues with real
supporting work — the stop-run reversal (Nagel above) and displacement (which maps
onto jump variation, Barndorff-Nielsen & Shephard above) — and those are kept
under their proper names. `docs/17-EVIDENCE-REVIEW.md` grades the rest and says
why each is discarded.

Similarly, aggregate dealer-gamma estimates are not cited because the mechanism's
academic support concerns *dealer hedging in general*, not the specific
reconstructions circulated publicly, whose error is unknown and unverifiable.
That gap is why `enable_positioning_engine` defaults to `False`.
