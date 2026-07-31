# System Architecture

Companion to `docs/01-THEORY.md`, which derives *why*. This document describes
*what*, in enough detail to navigate the code or reimplement it elsewhere.

---

## 1. The shape of the thing

```
                         ┌──────────────────────────────┐
   closed bar  ────────► │ L0  Data & Sanity            │
                         │  validate · sessions · clocks│
                         └──────────────┬───────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │ L1  Feature Kernel           │
                         │  scales · jumps · micro ·    │
                         │  structure · trend · entropy │
                         └──────────────┬───────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │ L2a Regime Posterior         │  ◄── routes everything
                         │  4-state causal Bayes filter │      downstream
                         └──────────────┬───────────────┘
                                        ▼
        ┌────────┬────────┬────────┬────┴───┬────────┬────────┬────────┐
        │ trend  │momentum│liquidity│volatil │structur│behavior│cross/  │  L2b Engines
        │        │        │        │        │        │        │position│
        └────┬───┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┘
             └───────┴────────┴────────┴────────┴────────┴────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │ L3  CALOP Fusion             │
                         │  L̂ = 1ᵀMℓ · EBE · calibrate  │
                         └──────────────┬───────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │ L4  Edge Book                │
                         │  hierarchical NIG expectancy │
                         └──────────────┬───────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │ L5  Decision                 │
                         │  barriers · costs · gate ·   │
                         │  state machine · card        │
                         └──────────────┬───────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │ L6  Risk                     │
                         │  Bayes-Kelly · throttle ·    │
                         │  kill switches               │
                         └──────────────┬───────────────┘
                                        ▼
                          BUY · SELL · SHORT · COVER · NO TRADE
                                        │
                         ┌──────────────▼───────────────┐
                         │ L7  Monitoring               │
                         │  calibration · PSI · breadth │
                         │  · demotion ladder           │
                         └──────────────────────────────┘
```

Two things about this diagram matter more than the boxes.

**The regime posterior is at the top, not the side.** It is not a filter applied
to a signal; it decides which of two contradictory bodies of evidence
(continuation versus reversal) is admissible at all. Every engine below it
receives the posterior and adjusts its own reliability accordingly. Theory §0.1
and §5.

**The Edge Book sits between fusion and the decision.** It is downstream of the
engines, not one of them. Engines produce a *direction and a confidence*; the
Edge Book converts that into an *expectancy with a credible interval*, and the
gate acts on the interval's lower end. This is why the system is selective
without a selectivity parameter.

---

## 2. Streaming, not vectorised

Every stateful object in `features/`, `engines/`, `fusion/`, `decision/` and
`risk/` exposes the same shape:

```python
obj.reset()          # back to pre-warmup
obj.update(...)      # consume exactly one closed bar, in order
```

An object that is only ever handed one bar at a time cannot read the future.
This is why the causality guarantee is structural rather than a matter of
discipline, and why `tests/test_causality.py` can verify it by brute force:
replay a prefix, replay the whole history, require bit-identical output at the
shared final bar.

The cost is that there is no fast vectorised path. On this hardware the pipeline
runs about 1,000 bars per second per instrument, which is ample for research on
daily and hourly data and is the wrong architecture for tick data. That trade is
made deliberately and is revisited in `docs/15-COMPLEXITY.md`.

There is exactly **one** implementation of the decision logic. Research and live
call the same `TIA.on_bar`. A system with separate backtest and live code paths
will eventually have them disagree, and the disagreement will be discovered in
production.

---

## 3. Layer by layer

### L0 — Data and sanity (`tia/data/`)

| Module | Responsibility |
|---|---|
| `validate.py` | monotone timestamps, duplicate detection, OHLC consistency, split detection, gap flagging |
| `sessions.py` | session phase, time-of-day fraction and bucket, day of week, overnight return |
| `clocks.py` | equal-dollar-volume resampling (`ActivityClock`), off by default |

The validator **rejects** rather than repairs. Repairing bad data means guessing,
and a guess that enters the feature kernel is indistinguishable from a signal.
Split detection deserves a note: a 2:1 split appears as a 50% single-bar move,
which a momentum engine reads as the strongest displacement in the sample. The
validator flags close ratios near simple fractions because real moves of that
size do not cluster on round numbers.

### L1 — Feature kernel (`tia/features/`)

One `FeatureKernel` per instrument, producing one `FeatureSnapshot` per bar.
Centralisation is the point: if each engine estimated its own volatility, the
system would hold several mutually inconsistent opinions about how large a move
is, and fusion would treat that inconsistency as independent evidence.

| Module | Produces |
|---|---|
| `rolling.py` | the streaming primitives everything is built from |
| `scales.py` | `sigma_bp`, `sigma_rv`, `sigma_rs`, `sigma_ew`, `sigma_fcst`, `jump_share`, `compression`, `vol_ratio`, `vol_of_vol` |
| `jumps.py` | Barndorff-Nielsen–Shephard jump test statistic |
| `microstructure.py` | `participation`, `ladr`, `absorption`, `amihud`, `kyle_lambda`, `clv_ema`, spread |
| `structure.py` | confirmed pivots, `range_pos`, sweep events, higher-timeframe aggregation |
| `trend.py` | `kalman_slope_t`, `vr_z`, `efficiency`, `trend_agree` |
| `entropy.py` | `perm_entropy`, `run_asym` |
| `kernel.py` | assembles all of the above plus the two higher-timeframe views |

### L2a — Regime (`tia/engines/regime.py`)

Four states, a fixed Beta measurement design over eight rank features, tempered
likelihood, sticky forward Bayes recursion. **Zero parameters fitted to
returns.** Emits a posterior, a dominant state, and a transition hazard. Full
design table in `docs/03-ENGINES.md`.

### L2b — Engines (`tia/engines/`)

Nine, each emitting `score ∈ [-1,1]`, `reliability ∈ [0,1]`, human-facing
features, machine-facing diagnostics, and short notes. Detailed in
`docs/03-ENGINES.md`.

The `score` / `reliability` split is load-bearing. A steep trend in a
mean-reverting regime is not a *weak* buy signal — it is a strong signal that
should not be acted on. Encoding that as a small score would let several such
engines sum into a large one; encoding it as low reliability makes the fusion
layer discount it and widen its interval, which is the correct treatment.

### L3 — Fusion (`tia/fusion/`)

`calop.py` pools with `L̂ = 1ᵀMℓ`, `EBE = 1ᵀM1`, `Var(L̂) = σ_e²·EBE`, then
applies the MacKay probit correction so that stated confidence shrinks toward
50% when evidence is narrow. `calibration.py` owns the online isotonic map and
the Brier diagnostics.

Two breadth numbers are computed and they answer different questions. *Total*
breadth includes confidently-neutral engines and drives the variance, correctly
widening the interval. *Aligned* breadth counts only engines supporting the
pooled direction and is what the gate tests, so a confident abstainer cannot
help satisfy a threshold about agreement.

### L4 — Edge Book (`tia/engines/edgebook.py`)

Three-level hierarchy `(regime, setup, evidence bucket) → (regime, setup) →
global`, Normal-Inverse-Gamma per cell, exponentially decayed weighted
sufficient statistics, **leave-one-out** shrinkage so a cell's own observations
never inform its own prior. Closed-form Student-t lower credible bound.

### L5 — Decision (`tia/decision/`)

`barriers.py` sizes the triple barrier from the volatility forecast and the
regime; `costs.py` prices the round trip in sigma units; `policy.py` runs the ten
conjunctive gate conditions and the position state machine; `explain.py` renders
the nine-row card.

### L6 — Risk (`tia/risk/`)

`sizing.py` computes the Bayesian Kelly fraction with parameter uncertainty
added to the variance, then applies the fractional-Kelly multiplier, the
credible-bound haircut, a data-sufficiency factor and the drawdown throttle.
`limits.py` tracks equity and evaluates the kill switches.

### L7 — Monitoring (`tia/monitoring/drift.py`)

Rolling calibration, population stability on the feature distribution, engine
correlation drift, execution slippage, and a five-rung demotion ladder from
`NORMAL` through `TIGHTENED`, `RESTRICTED`, `NO_TRADE` to `HALTED`.

---

## 4. Training, and the cold start

The gate acts on a lower credible bound. An untrained Edge Book has no
observations, so every bound is negative, so the system never trades and never
learns. This deadlock is real and `tia/training.py` resolves it honestly:

1. **Primary pass.** Run the pipeline with trading disabled. Record every bar
   clearing the *structural* gates — valid features, a direction, tradeable
   session, non-quiet regime, hazard within tolerance, breadth and probability
   above their floors — but **not** the expectancy gate, which is the thing
   being learned.
2. **Label.** Apply the triple barrier to every candidate.
3. **Weight.** Uniqueness × time decay. Candidates overlap heavily; treating
   them as independent overstates the effective sample by roughly a factor of
   four at a twenty-bar horizon.
4. **Write.** Into the Edge Book, and fit the calibrator on realised outcomes.

The pass never uses an outcome to decide whether to *record* a candidate — it
records everything structurally eligible, including the bad ones. Selecting
candidates on their outcomes would be the exact look-ahead this system exists to
avoid.

`walk_forward_train` is the honest protocol: train on the prefix, trade the next
block, repeat. Training and trading the same bars is in-sample by construction
and is only appropriate for smoke tests.

---

## 5. Where state lives

Everything that must survive a restart, and where it is:

| State | Owner |
|---|---|
| rolling windows, ring buffers | `FeatureKernel` and its sub-kernels |
| Kalman level/slope and covariance | `features/trend.py:KalmanTrend` |
| regime log-posterior, reference EWMA | `engines/regime.py:RegimeEngine` |
| engine score correlation matrix | `fusion/calop.py:CALOP` |
| calibration samples and isotonic knots | `fusion/calibration.py:OnlineCalibrator` |
| Edge Book sufficient statistics | `engines/edgebook.py:EdgeBook` (`state_dict`) |
| position, entry, barriers | `decision/policy.py:Policy` |
| equity, high-water mark, alarms | `risk/limits.py:RiskLimits` |

A cold start must replay at least `FeatureKernel.warmup` bars before emitting
anything. See `docs/12-DEPLOYMENT.md`.

---

## 6. Deviations from `SPEC.md`, recorded

* **`EdgeBookEngine` is not an `Engine`.** `SPEC.md` §4 lists it under
  `engines/` for discoverability, but it consumes fused evidence rather than
  voting alongside the engines, has no `score`, and is never pooled. The
  topology in this document is the correct one.
* **`ebe_min` was recalibrated from 2.5 to 1.25** after measuring that the nine
  engines carry a mean absolute score correlation near 0.30, giving a total
  breadth at full reliability of about 2.66. The original value was unreachable
  and would have made the gate a de facto "never trade" rule. The threshold is
  now derived from a measurement, and the measurement is reproducible via
  `CALOP.diagnostics()`.
* **`cost_slippage_range_frac` was reduced from 0.10 to 0.02** for the same
  class of reason: charged against a real bar range of roughly 1.3 sigma, the
  original produced a round-trip cost near 0.28 sigma, several times any
  plausible edge.

Each of these is a change to a `THEORY`-tagged default and therefore a research
decision. They are recorded here, in the config docstrings, and in the trials
ledger in `docs/00-PREREGISTRATION.md`.
