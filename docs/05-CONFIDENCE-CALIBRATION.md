# The Confidence Model

What "97% confident" is allowed to mean, and the machinery that makes it mean it.

---

## 1. Three distinct quantities

Trading systems routinely conflate these. They are not the same, and the design
keeps them separate all the way to the display:

| Quantity | Symbol | Question |
|---|---|---|
| **Evidence** | `L̂` | how far does the pooled log-odds sit from zero? |
| **Breadth** | `EBE` | how many independent opinions does `L̂` rest on? |
| **Probability** | `p` | over many such statements, how often is it right? |

Strong evidence from one engine is not the same as moderate evidence from five.
The first should produce a *narrow* interval around a *modest* probability; the
second a narrower interval around a larger one. Only a model that carries breadth
explicitly can express that difference — which is why `EBE` is a first-class
output rather than a diagnostic.

---

## 2. From score to probability

**Step 1 — evidence.** Each engine's score maps to log-odds by the natural link:

$$\ell_i = 2\,\mathrm{artanh}\big(\mathrm{clip}(s_i,\ \pm 0.999)\big)$$

This is the inverse of the `tanh` the engines apply, so an engine that formed its
opinion as `tanh(t)` on some t-statistic contributes `2t` — linear in the
underlying statistic, which is what a log-odds contribution should be. The clip
keeps a saturated engine finite; without it one engine could map to infinity and
dominate the pool by itself.

**Step 2 — pooling.** With `C` the causally-estimated engine correlation matrix,
`D = diag(ρ_i)` the reliabilities as precisions, and `M = D^½ C⁻¹ D^½`:

$$\hat L = \mathbf 1^\top M \ell, \qquad \mathrm{EBE} = \mathbf 1^\top M \mathbf 1, \qquad \mathrm{Var}(\hat L) = \sigma_e^2\,\mathrm{EBE}$$

Derived in `docs/01-THEORY.md` §6, with the two propositions: this *is* the
Bayesian sum when engines are independent, and it counts duplicates once when
they are identical.

**Step 3 — the uncertainty correction.** A plug-in `p = σ(L̂)` ignores that `L̂`
is itself an estimate. Integrating over its Gaussian uncertainty (MacKay's probit
approximation) gives the predictive probability:

$$p = \varsigma\!\left(\frac{T\hat L + b}{\sqrt{1 + \tfrac{\pi}{8}T^2\,\mathrm{Var}(\hat L)}}\right)$$

**This denominator is the whole confidence model.** It is the mechanism by which
a reported 97% requires evidence that is both *strong* and *broad*. Holding
evidence fixed at `L̂ = 2` and varying breadth:

| EBE | Var(L̂) | reported p |
|---|---|---|
| 1 | 0.36 | 0.86 |
| 4 | 1.44 | 0.78 |
| 9 | 3.24 | 0.71 |

More engines agreeing produces *lower* stated confidence at fixed evidence,
because more engines contributing means more accumulated estimation noise. That
is the opposite of how confluence scoring behaves, and it is correct.

**Step 4 — isotonic recalibration.** Once 200 outcomes exist, a monotone map
fitted by pool-adjacent-violators is composed on top. Isotonic assumes only
monotonicity, which is the only functional-form assumption we are willing to
make.

---

## 3. The certainty bug, and why it is worth documenting

Raw PAVA on binary outcomes drives its first fitted value to exactly 0 and its
last to exactly 1. The reason is easy to miss: PAVA pools only where the ordering
is *violated*, and a run of identical values never violates it — so a leading run
of failures stays at 0 and a trailing run of successes stays at 1.

The consequence is a calibrated probability of exactly 100%, asserting certainty
on the strength of however many observations back that run. It reached the
explanation card during development, reading `Confidence: 100% [100%-100%]`
alongside `Evidence Breadth: 0.8 independent` — the exact pathology this entire
layer exists to prevent, arriving through a different door.

The fix bounds only the map's endpoints, at the Laplace posterior mean for the
run length behind each: `n` successes out of `n` supports `(n+1)/(n+2)` — 0.857
on five observations, 0.99 on a hundred. Two earlier attempts were worse and are
recorded because the failure modes are instructive:

* **Smoothing every block** crushed the map's range to `[0.18, 0.88]` and drove
  expected calibration error from 0.018 to 0.045. Interior smoothing destroys
  *resolution*, which is the property that actually earns anything.
* **Using PAVA's internal block sizes** as the evidence count was simply wrong,
  for the reason above: those blocks are size 1 for every element of an
  unviolated run.

---

## 4. Reading calibration properly

Murphy's decomposition splits the Brier score into three parts, and the split
matters operationally:

$$\mathrm{BS} = \underbrace{\text{reliability}}_{\text{lower is better}} - \underbrace{\text{resolution}}_{\text{higher is better}} + \underbrace{\text{uncertainty}}_{\text{a property of the problem}}$$

A forecaster can improve its Brier score by becoming better *calibrated* or by
becoming more *discriminating*, and only the second is worth anything to a
trading system. **A forecaster that always predicts the base rate has zero
reliability error and zero resolution, and is completely useless.**
`tests/test_fusion.py` asserts exactly this, because it is the distinction most
easily lost.

Monitored quantities and their thresholds:

| Metric | Alarm | Meaning of a breach |
|---|---|---|
| rolling Brier | `brier_alarm = 0.27` | 0.25 is a coin flip; above this the probability model is uninformative and every gate downstream acts on a meaningless number |
| Brier skill vs base rate | `< 0` | forecasts are worse than predicting the base rate |
| expected calibration error | `> 0.05` | stated probabilities are systematically off |
| reliability term | rising | calibration drifting; refit or demote |
| resolution term | falling | the model is losing its ability to discriminate — worse than miscalibration, and not fixable by recalibrating |

Reliability-diagram bins use **Wilson** intervals rather than the normal
approximation, which is badly wrong near 0 and 1 at the counts these bins
actually have.

---

## 5. What the trader sees

```
Confidence:                    64%  [55%-73%]
Evidence Breadth:              1.6 independent
```

The interval is not decoration. It is the visible output of `Var(L̂)`, and a wide
interval is the system saying that its engines agree for what may be a single
reason. A trader who learns to read the breadth row alongside the confidence row
has the whole confidence model in two lines.

The gate does not act on `p` as its primary test — `p_min = 0.58` is a floor.
The primary test is the Edge Book's lower credible bound on expectancy
(`docs/06-RISK-SIZING.md` §1), because a high probability on a poor
reward-to-risk ratio is not an opportunity.
