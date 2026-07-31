# Computational Complexity

Measured, not estimated. Profile: `cProfile` over 2,000 bars of synthetic daily
data, single instrument, default configuration, on one core.

```
2,000 bars in 3.53 s   →  1.8 ms/bar,  ~555 bars/second/instrument
```

---

## 1. Where the time actually goes

| Component | Cumulative | Share | Per bar |
|---|---|---|---|
| `FeatureKernel.update` | 1.51 s | 42% | 0.76 ms |
| ├─ `VarianceRatio._compute` | 0.33 s | 9% | 0.17 ms |
| ├─ `VolatilityKernel.update` ×2 HTF | 0.26 s | 7% | 0.13 ms |
| ├─ `KalmanTrend.update` | 0.15 s | 4% | 0.07 ms |
| └─ `MicrostructureKernel.update` | 0.18 s | 5% | 0.09 ms |
| `RegimeEngine.step` | 0.73 s | 20% | 0.37 ms |
| `CALOP.fuse` | 0.49 s | 13% | 0.24 ms |
| nine engines (`BaseEngine.update`) | 0.32 s | 9% | 0.16 ms |
| everything else | ~0.5 s | 14% | |

**The bottleneck is the variance-ratio statistic**, as expected: it is the only
O(window × q_max) computation in the kernel, recomputing 120-bar sums at four
horizons plus the heteroskedasticity-robust `δ_j` corrections every bar. It alone
is more expensive than all nine engines combined.

The two surprises are worth noting:

* **`RegimeEngine.step` at 20%** is dominated by 61,616 `beta_logpdf` calls
  (4 regimes × 8 features × 2,000 bars, in a Python loop) plus one
  `np.linalg.solve` per bar for the tempering exponent — which, as
  `docs/03-ENGINES.md` records, currently sits at its cap and changes nothing.
* **`CALOP.fuse` at 13%** performs two `np.linalg.solve` calls per bar on a 9×9
  matrix. The matrix changes slowly; the solve does not need to be repeated.

---

## 2. Per-component complexity

`W` = window, `m` = engine count, `K` = 4 regimes, `J` = 8 regime features.

| Component | Time/bar | Space | Notes |
|---|---|---|---|
| `RingBuffer`, `RollingSum`, `EWMA` | O(1) | O(W) | incremental |
| `RollingMoments` | O(1) amortised | O(W) | raw power sums, periodic rebuild for stability |
| `RollingQuantile`, `CausalRank` | O(W) insert | O(W) | `bisect.insort` — list splice dominates |
| `RollingExtreme` | O(1) amortised | O(W) | monotonic deques |
| `BucketedMedian` | O(W/B) | O(W) | one quantile structure per bucket |
| `RollingCorrelation` | O(m²) | O(m²) | exponentially weighted |
| `VolatilityKernel` | O(1) | O(W) | all incremental |
| `CorwinSchultzSpread` | O(W) | O(W) | two rolling medians |
| `MicrostructureKernel` | O(W) | O(W) | dominated by four `CausalRank` |
| `KalmanTrend` | O(1) | O(1) | 2×2 matrices |
| **`VarianceRatio`** | **O(W·q_max)** | O(W) | **the bottleneck** |
| `PermutationEntropy` | O(d log d) | O(W) | ring-buffered counts, already O(1) in W |
| `PivotTracker`, `SweepDetector` | O(1) | O(1) | |
| `HTFView` ×2 | O(1)/O(W) at completion | O(W) | only on completed HTF bars |
| `RegimeEngine.step` | O(K·J + K² + J³) | O(K + J²) | `J³` is the tempering solve |
| each engine | O(1) | O(1) | |
| `CALOP.fuse` | O(m³) | O(m²) | two solves; `m = 9` so the constant dominates |
| `EdgeBook.estimate` | O(1) | O(cells) | ≤ 80 cells, a few kB |
| decision, risk | O(1) | O(1) | |

Nothing is O(history). The whole system is streaming.

---

## 3. Optimisations, ranked by payoff

1. **Incremental variance ratio** (≈ −9% wall clock, the single largest win).
   The `δ_j` cross-products and the overlapping `q`-sums can both be maintained
   incrementally as bars enter and leave the window, turning O(W·q) into O(q).
2. **Cache the Cholesky factor in CALOP** (≈ −10%). The correlation matrix is
   updated every bar but changes slowly; factor it once every `n` bars and reuse.
   Two `solve` calls become two triangular back-substitutions.
3. **Vectorise `beta_logpdf`** (≈ −10%). The 4×8 grid is a natural numpy
   operation; shape parameters are constant, so `lgamma` normalisers can be
   precomputed entirely — they already are in the Pine export.
4. **Skip the tempering solve when it is capped.** Recompute the feature
   correlation every `n` bars rather than every bar; it moves far more slowly
   than the posterior.
5. **Histogram or order-statistic tree for `CausalRank`** (O(log W) instead of
   O(W) splice). Worth it only at `rank_window` ≫ 252.
6. **Numba or Rust for the streaming inner loop** (10–50×). The right move only
   if sub-millisecond bars are needed; it costs the property that the reference
   implementation is readable, which is currently doing real work.

Realistically, items 1–4 together would roughly double throughput to ~1,100
bars/s for a day's effort. That is worth doing before a large sensitivity scan
and not before.

---

## 4. Validation-suite cost — where the compute really goes

Per-bar cost is irrelevant next to this.

For a 50-instrument universe at 5,000 bars each (250,000 bars, ~7.5 min of
single-core streaming):

| Stage | Multiplier | Core-hours |
|---|---|---|
| single pass | 1× | 0.13 |
| CPCV, C(8,2) = 28 splits | 28× | 3.5 |
| × Monte Carlo, 2,000 paths | 2,000× | **~250** |
| **parameter sensitivity**, ~40 scannable params × 9 points | 360× | **~45** (on a single pass each) |
| ablation, 9 leave-one-out + 9 one-only | 18× | 2.3 |

**Total for a full acceptance run: on the order of 300 core-hours**, dominated by
the Monte Carlo surrogates. On 32 cores that is ~10 hours.

Parallelise over the **instrument** axis first — it is embarrassingly parallel,
needs no shared state except the Edge Book (which is written once per fold), and
maps directly onto the pooling design. Parallelise over Monte Carlo paths second.
Do *not* parallelise within a bar; the streaming dependency chain forbids it.

The `mc_paths = 2000` default is generous. For a p-value threshold of 0.05, 500
paths gives adequate resolution and cuts the dominant cost by 4×; use 2,000 only
for the final acceptance run.

---

## 5. Memory

| Scope | Footprint |
|---|---|
| one `FeatureKernel` (252-bar windows, 13 buckets, 2 HTF views) | ~90 kB |
| nine engines + CALOP (9×9) | ~15 kB |
| `EdgeBook`, 80 cells | ~10 kB |
| calibrator, 5,000 retained outcomes | ~80 kB |
| **per instrument, total** | **~200 kB** |
| 500 instruments live | ~100 MB |

Memory is not a constraint at any plausible universe size. The Edge Book is
shared across instruments, so it does not multiply.

---

## 6. Latency budget for a live bar-close decision

| Step | Budget |
|---|---|
| bar arrival and validation | 1 ms |
| feature kernel | 1 ms |
| regime + engines + fusion | 1 ms |
| Edge Book, costs, gate, sizing | < 0.5 ms |
| **decision available** | **< 5 ms after bar close** |
| order transmission | broker-dependent |

Ample. The system decides at bar close and executes at the next open, so the
budget is measured in seconds, not microseconds. **This architecture is not
suitable for latency-sensitive execution and does not pretend to be.**

---

## 7. What would have to change for one-second bars

Daily and hourly bars are comfortable. One-second bars are a different system:

* **Throughput.** 86,400 bars/day/instrument at 1.8 ms/bar is 2.6 minutes/day of
  compute per instrument — fine for one, impossible for 500 in real time. Items
  1–4 above plus a compiled inner loop become mandatory.
* **The rank window becomes wrong.** 252 one-second bars is four minutes, which
  is not a reference class for anything. The windows would need re-derivation,
  and that is a research task, not a configuration change.
* **The cost model breaks down.** At one-second granularity, spread and queue
  position dominate and the square-root impact law is the wrong parameterisation.
* **Corwin–Schultz becomes unusable.** It needs a bar range that reflects a
  diffusion; at one second the range is mostly spread.
* **The intrabar-sequence problem gets worse, not better.** More bars means more
  same-bar barrier ties, and the pessimistic tie-break becomes a larger fraction
  of measured performance.

The honest summary: the design targets bar intervals from minutes to days. Below
that it is not a tuning exercise.
