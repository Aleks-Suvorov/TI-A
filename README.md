# TI-A

An adaptive, abstention-first trading signal framework.

The trader sees five words — **BUY**, **SELL**, **SHORT**, **COVER**,
**NO TRADE** — a confidence figure with a credible interval, a stop, a target,
and a nine-row explanation. Everything else happens underneath.

> **Status: research framework, not a trading system.** Every number in this
> repository comes from synthetic data. Nothing here has been validated on real
> markets, and `docs/00-PREREGISTRATION.md` describes the protocol that would
> have to be executed before it could be. Read `docs/08-FAILURE-MODES.md` before
> forming an opinion about whether this would work.

---

## The short version

Directional predictability in liquid markets, at horizons of hours to days, has
an information coefficient of roughly 0.02–0.06. That is an R² of a few tenths of
one percent. Three consequences drive the entire design:

1. **Any method with enough capacity to fit R² = 0.3% can fit noise.** So
   capacity is spent deliberately and *counted*: `Config.fitted_dof()` reports
   the number of parameters ever fitted to market outcomes, and it is **7**.
   Everything else is fixed a priori, updated by causal recursion, or an
   operational knob.

2. **The denominator is more forecastable than the numerator.** Volatility is
   the most reliably predictable quantity in financial time series, so it is
   estimated first and everything — barriers, costs, features — is denominated
   in it.

3. **Most bars contain nothing.** Abstention is the primary output, not a
   defensive afterthought. Roughly 1% of bars are even *structurally eligible*
   before the expectancy gate applies.

### What conditions actually precede high-probability moves?

Stripped of folklore, the evidence supports three, and the last two **contradict
each other**:

- **(A) A resolved risk denominator** — conditional volatility estimable with low
  relative error, so a stop can sit where noise will not reach it.
- **(B) A liquidity event that fails** — price penetrates a level where resting
  orders concentrate, consumes anomalous volume, and fails to hold. This is
  compensated liquidity provision, and it says *fade*.
- **(C) Cheap displacement in an autocorrelated regime** — price travels far per
  unit of volume consumed, which under any Kyle-type model is the signature of
  informed flow. It says *follow*.

Because (B) and (C) issue opposite instructions, a **regime posterior sits at the
top of the architecture**, not off to the side as a filter. Without knowing which
applies, the two signals cancel into noise — which is exactly what happens to
indicator systems that average trend and reversion logic together.

---

## Three ideas worth stealing

**A regime model with zero parameters fitted to returns.** Rather than fitting a
Gaussian HMM — unstable, leaky, and needing O(K² + KJ²) parameters — each of four
regimes is a *specified* product of Beta densities over rank features. The
encoding trick: `Beta(1,1)` is exactly uniform, so "this regime makes no claim
about this feature" lives inside the same parametric family and contributes a
likelihood factor of exactly 1. Forward recursion only; the smoother would
repaint every historical label. → `docs/01-THEORY.md` §5

**CALOP: correlation-aware log-odds pooling.** Ten mildly bullish engines, each
scoring 0.5. Summing log-odds — which is what any "confluence score" implicitly
does — reports **99.998%** confidence. That is correct Bayes *only* under
conditional independence, and ten engines built on overlapping windows of one
price series are nowhere near independent. Modelling them as noisy observations
of a common latent log-odds gives

> **L̂ = 1ᵀMℓ,  EBE = 1ᵀM1,  Var(L̂) = σ_e²·EBE,  M = D^½C⁻¹D^½**

which *is* the Bayesian sum when `C = I` and counts duplicates once when they are
identical. For equicorrelated engines `EBE = m/(1+(m−1)ρ̄)`, so **ten engines at
ρ̄ = 0.5 supply an effective breadth of 1.82** — fewer than two independent
opinions. Measured on this system's nine engines: mean |correlation| 0.30,
effective breadth **2.66**. → `docs/01-THEORY.md` §6

**The gate acts on a lower credible bound.** Expectancy is estimated by a
hierarchical Normal-Inverse-Gamma model over `(regime, setup, evidence bucket)`
cells with leave-one-out shrinkage toward a root that asserts *zero* edge. A
trade requires 90% posterior credence that expectancy net of modelled costs is
positive. Selectivity is a *consequence* of this rule, not a threshold anyone
chose — and it tightens automatically when data is thin, variance is high, or
costs rise. → `docs/01-THEORY.md` §7

---

## The uncomfortable arithmetic

With 95 trades and a 60% hit rate, the standard error is 5.0 percentage points.
The 95% interval is [50%, 70%]. **Ninety-five trades cannot distinguish a real
60% edge from a coin flip.** Establishing a per-trade expectancy of 0.15σ against
an outcome standard deviation of 1.5σ at t = 2 needs ~400 trades — about 40 years
of daily data on one instrument.

So selectivity is applied **per instrument** and inference is pooled **across**
instruments. Fifty instruments over twenty years gives ~10⁴ trades: enough power
to establish the effect and break it down by regime. Each instrument still trades
rarely, which is what the trader experiences.

**Corollary:** a single-instrument backtest of this system is statistically
uninformative. It can falsify, never confirm. → `docs/01-THEORY.md` §12

---

## Quick start

```bash
python3 -m pip install numpy            # the only hard dependency
cd python && PYTHONPATH=src python3 - <<'PY'
from tia import TIA, Config
from tia.synthetic import generate_null

system = TIA(Config())
system.run(generate_null(3000))
print(system.summary())
PY
```

On a martingale with realistic volatility clustering, this prints zero actionable
signals. That is the single most instructive thing the framework does: **a
system that finds trades in a null market has a bug**, and this is the cheapest
test that catches it.

To see it trade, train the Edge Book first:

```python
from tia.training import collect_candidates, train_edge_book
from tia.synthetic import generate_with_regimes

bars, truth = generate_with_regimes(6000)
cands = collect_candidates(bars)
book, cal, diag = train_edge_book(bars, cands)
print(diag)     # includes effective_sample_size and t_stat_effective
```

`walk_forward_train` is the honest protocol — train on the prefix, trade the next
block. Training and trading the same bars is in-sample by construction.

```bash
python3 -m pytest python/tests -q       # 90 tests, ~18s
```

---

## Repository map

```
SPEC.md                    the binding interface contract
docs/
  00-PREREGISTRATION.md    hypotheses, universe split, acceptance thresholds
  01-THEORY.md             the mathematics — start here
  02-ARCHITECTURE.md       component map, data flow, recorded deviations
  03-ENGINES.md            per-engine reference, incl. the regime design table
  04-SIGNAL-LOGIC.md       the ten gate conditions, barriers, exits
  05..06                   confidence model, risk and sizing
  07-VALIDATION.md         the protocol: CPCV, surrogate nulls, stress, ablation
  08-FAILURE-MODES.md      how this dies, ranked
  09-ASSUMPTIONS.md        every assumption, tagged testable or not
  10-PSEUDOCODE.md         language-independent reference
  11-PINE-PLAN.md          the port: what maps, what is approximated, tolerances
  12..16                   deployment, monitoring, learning, complexity, roadmap
  17-EVIDENCE-REVIEW.md    every concept graded A-F, kept or discarded, with reasons
  18-LITERATURE.md         annotated bibliography
python/src/tia/            the implementation (numpy only in the live path)
python/tests/              90 tests; test_causality.py is the leakage auditor
pine/                      TradingView port + the frozen model it evaluates
```

---

## Non-repainting, structurally

Every estimator is **streaming**: it receives one closed bar at a time and holds
its own state. An object that is never handed the future cannot read it. This is
why the guarantee is architectural rather than a matter of discipline, and why
`tests/test_causality.py` can verify it by brute force — replay a prefix, replay
the whole history, require bit-identical output at the shared bar.

Additionally: decisions at bar `t` execute at `t+1` at the earliest; pivots are
invisible until confirmed; a sweep is knowable only after it has failed; if a
bar's range spans both barriers the **stop** is taken; a gap through the stop
fills at the **open**. Bar data cannot resolve intrabar sequence, so the
ambiguity is always resolved against us.

---

## What this project refuses to claim

- No backtest equity curve is presented, because one produced on synthetic data
  or a single instrument would be meaningless and one produced on real data has
  not been earned yet.
- The `positioning` engine (dealer gamma) is **off by default**. The mechanism is
  real; the publicly available data is a reconstruction resting on unverifiable
  assumptions, and its error is correlated with exactly the conditions where you
  would want the signal.
- Most of the Smart Money / ICT vocabulary is **discarded**. Two components
  survive — the stop-run reversal and displacement — because they have measurable
  analogues with supporting literature. The rest are not falsifiable as usually
  stated. `docs/17-EVIDENCE-REVIEW.md` grades every concept and says why.
- Machine learning is used for meta-labelling and calibration, not for predicting
  returns. At this signal-to-noise ratio a high-capacity learner fits noise.

Two real bugs were found by the test suite during development and are documented
in the commit history: the Edge Book hierarchy double-counted observations
(a cell's own data helped form its own prior), and the spread estimator combined
two biased estimators in a way that inherited the worse behaviour of each. Both
are the kind of error that inflates apparent edge silently.

---

## Licence and intent

Research code. Not investment advice, not a recommendation, and not fit to trade
as-is. If you deploy it anyway, read `docs/08-FAILURE-MODES.md` first and note
its conclusion: the most likely cause of this system's death is not a statistical
failure but a human adjusting it after a drawdown.
