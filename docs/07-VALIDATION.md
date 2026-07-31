# Validation Methodology

A protocol someone else could execute. Implemented in `python/src/tia/validation/`.

---

## 1. Why ordinary cross-validation is invalid here

Three reasons, all fatal:

1. **Serial correlation.** Adjacent bars are not independent draws. A random
   train/test split puts near-copies of test observations into training.
2. **Overlapping labels.** A trade opened at `t` and closed at `t+20` shares
   outcome information with one opened at `t+5`. Treating them as independent
   inflates every t-statistic — by roughly `√(n/n_eff)`, a factor near 4 at a
   twenty-bar horizon.
3. **Non-stationarity.** Testing on the past using a model fitted on the future
   is not a weaker test, it is a different and meaningless one.

**Purging** removes any training label whose span overlaps a test span.
**Embargo** additionally drops a small window after each test block, because
serial correlation extends beyond the label span itself.

Neither is optional. Skipping them is the most common way a leakage-free-*looking*
pipeline leaks.

---

## 2. Combinatorial purged cross-validation

`validation/cpcv.py`. Split into `cv_folds = 8` contiguous groups, hold out
`cv_test_groups = 2` at a time, purge and embargo, and enumerate all `C(8,2) = 28`
combinations. This yields `28 × 2/8 = 7` distinct backtest *paths* rather than one.

Why paths matter: a single walk-forward gives one equity curve, and one curve
cannot distinguish a robust strategy from a lucky ordering. Seven paths give a
distribution, and the *spread* across paths is a direct estimate of how much of
the headline result is sequencing luck.

`LeakageReport` asserts zero overlap between any training label span and any test
span, and reports purged and embargoed counts. If purging removes almost nothing,
the label horizon or the embargo is mis-specified.

---

## 3. Walk-forward

`validation/walkforward.py` and `training.walk_forward_train`. Anchored (expanding
training window) and rolling (fixed window) variants.

The diagnostic that matters is **parameter drift across folds**. If re-fitted
values swing wildly fold to fold, the fit is noise regardless of aggregate
performance. `parameter_drift` reports the dispersion; a coefficient of variation
above ~0.5 on a `DEV` parameter means that parameter is not identified by the
data.

---

## 4. Monte Carlo and surrogate nulls

`validation/montecarlo.py`. The most informative part of the suite.

### 4.1 GARCH surrogate nulls — the single best test

Fit a GARCH(1,1) to the returns, then simulate paths by bootstrapping the
standardised residuals. The surrogates **preserve volatility clustering and fat
tails** while **destroying directional structure**.

Run the entire strategy on them. If it still profits:

> the strategy is harvesting volatility, or exploiting an artifact of the cost
> model, or reading the future — but it is **not** predicting direction.

That inference is unusually clean, because the surrogate matches the real series
on precisely the properties a volatility-harvesting artifact would exploit.
`surrogate_pvalue(observed, surrogates)` gives the fraction of surrogates
matching or exceeding the observed statistic; acceptance requires p ≤ 0.05.

`sign_scramble_surrogate` is a cheaper variant with the same logic: randomise
return signs within volatility-matched blocks.

### 4.2 Stationary block bootstrap

Politis–Romano with geometric block lengths (`bootstrap_block_bars = 60`,
long enough to carry volatility clustering through the resample). Gives
confidence intervals on performance statistics that respect serial dependence.

### 4.3 Trade-order permutation

Shuffle the sequence of trade returns. Tests whether path dependence and
compounding — rather than per-trade edge — produced the result. A strategy whose
performance collapses under permutation was relying on ordering.

---

## 5. Stress

`validation/stress.py`.

| Test | What it catches |
|---|---|
| **noise injection** | perturb OHLC within a plausible spread, preserving consistency; a strategy sensitive to the last tick is fitting quote noise |
| **timing shift** | shift signals ±1..5 bars — **a real edge decays smoothly and monotonically; an artifact collapses discontinuously or, tellingly, improves** |
| **bar dropout** | random bars removed; tests robustness to feed gaps |
| **latency injection** | extra execution delay; measures how much edge lives in the first bar |
| **ablation** | leave-one-engine-out and one-engine-only; any engine whose removal *improves* results is removed |

The timing-shift profile is the most diagnostic single plot in the suite. Real
predictive information is smeared across nearby bars because the underlying
process is continuous; an artifact is not, and its profile has a spike or a
discontinuity.

---

## 6. Parameter sensitivity

`parameter_plateau` scans each parameter over the multiplicative neighbourhood
declared in `Config.sensitivity_for(name)` and reports:

* **plateau ratio** = performance at the chosen point ÷ median performance across
  the neighbourhood. Near 1 means a flat region; well above 1 means a spike,
  which is what a tuned parameter looks like.
* **interiority** — the chosen value must **not** be the best point in its own
  scan. This catches tuning that a flat-looking plateau would hide.

Both are pre-registered criteria (C8, C9).

---

## 7. Multiple testing

`probabilistic_sharpe_ratio` corrects for non-normality and sample length;
`deflated_sharpe_ratio` additionally charges the trial count via the expected
maximum Sharpe under the null. `trials_ledger_count` feeds it, and honest
accounting depends on honest counting.

`minimum_track_record_length` answers the operationally useful question: how long
must this run before the record distinguishes it from zero?

---

## 8. Calibration validation

Reliability diagrams with **Wilson** intervals (the normal approximation is badly
wrong near 0 and 1 at these counts), the Brier decomposition into
reliability/resolution/uncertainty, and expected calibration error. See
`docs/05-CONFIDENCE-CALIBRATION.md` §4 for why resolution matters more than
reliability.

---

## 9. Tests we expect to fail

Named in advance, because a suite that only contains tests one expects to pass is
theatre:

* **Trade count (C2).** At ~1% eligibility, reaching 400 trades needs either a
  long history or a wide universe. This is the criterion most likely to make the
  primary test inconclusive rather than negative.
* **Engine ablation (H4).** Nine engines with mean correlation 0.30 almost
  certainly contains at least one that contributes nothing. The honest response
  is deletion, not defence.
* **Cross-decade (C12).** Microstructure has changed enough since 1990 that the
  rank-transform portability claim is genuinely at risk.
* **Plateau interiority (C9).** Several `THEORY` parameters were set by
  convention rather than by dimensional argument, and convention is a weaker
  guarantee of interiority than it feels like.

---

## 10. Execution order

Cheapest and most likely to kill it first. There is no reason to spend compute
establishing the significance of a result that a two-minute test would have
refuted.

1. **Null-market check** (seconds). The system must find nothing on a martingale.
2. **Causality audit** (seconds). Prefix replay, bit-identical.
3. **Timing shift** (minutes). Artifacts show up here immediately.
4. **Surrogate nulls** (minutes–hours). The most informative test in the suite.
5. **Ablation** (hours). Establishes what is actually contributing.
6. **Purged CPCV across the development universe** (hours).
7. **Parameter sensitivity scan** (hours–days; the dominant compute cost).
8. **Walk-forward with parameter-drift diagnostics.**
9. **Validation universe — once.**

Steps 1–5 are diagnostics and may be re-run freely. Step 9 consumes a trial.
