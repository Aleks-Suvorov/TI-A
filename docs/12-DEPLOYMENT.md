# Live Deployment

---

## 1. Topologies

| Topology | Can | Cannot |
|---|---|---|
| **TradingView** (`pine/`) | evaluate a frozen model on one chart, alert | learn, pool across instruments, validate |
| **Python research** | train, validate, sensitivity-scan | trade |
| **Python live service** | everything the research path does, plus emit orders | — |

The frozen-model workflow connects the first two: `tia.export` writes
`pine/frozen_model.json` and rewrites the constants region inside the `.pine`
files. `FM_HASH` carries `Config.manifest_hash()`; if it disagrees with the
running research config, the chart and the research describe different systems.

---

## 2. The bar-close race

```
bar closes ──► feed delivers ──► validate ──► features ──► decide ──► transmit ──► fill
   t              t+~1s           <5 ms total (docs/15)                        t+1 open
```

The decision is formed at the close of bar `t` and executes at the open of
`t + execution_lag_bars`. That gives a full bar interval of slack, so latency is
not the constraint. What *is* the constraint is correctness under imperfect
delivery:

* **Late data.** If bar `t` has not arrived by the time bar `t+1` opens, do not
  guess. Emit nothing and mark the feature kernel stale; `on_data_staleness`
  trips a kill switch after 3 bars.
* **Out-of-order data.** The validator rejects non-monotone timestamps outright.
  A feed that delivers out of order must be buffered and sequenced upstream.
* **Revised bars.** A vendor that revises a closed bar has invalidated every
  feature computed from it. Either replay from the revision point or accept the
  divergence explicitly — do not patch state in place.

**Idempotency.** `on_bar` must be safe to call exactly once per bar and unsafe to
call twice. Enforce it with a guard on the last processed timestamp; a duplicate
delivery that reaches the kernel double-counts a return and corrupts every
window that contains it.

---

## 3. Data hygiene

* **Vendor differences are real.** Two vendors genuinely disagree about a bar's
  high, because they consolidate different venues. Pin one vendor for research
  and live, and treat a vendor change as a re-development.
* **Corporate actions — the silent trap.** A retroactively split-adjusted history
  changes *every* historical feature value. A live-vs-backtest comparison run
  after an adjustment will diverge for reasons that have nothing to do with the
  model, and nothing in the system will flag it. Snapshot the adjusted history
  used for each research run, alongside its manifest hash.
* **Futures rolls.** Pick a convention (ratio back-adjustment is usual) and
  record it. The choice changes results, and an unrecorded choice makes results
  irreproducible.
* **Sessions and DST.** `data/sessions.py` uses fixed UTC offsets by design: a
  timezone database that updates would silently change historical feature values.
  Production deployments should supply exchange session boundaries instead.
* **Halts and limit moves.** The validator flags large time gaps as non-fatal.
  A halted instrument should be excluded rather than traded through.

---

## 4. State

Must be persisted between restarts:

| State | Owner |
|---|---|
| rolling windows and ring buffers | `FeatureKernel` |
| Kalman level, slope, covariance | `KalmanTrend` |
| regime log-posterior, reference EWMA | `RegimeEngine` |
| engine correlation matrix | `CALOP` |
| calibration samples and isotonic knots | `OnlineCalibrator` |
| Edge Book sufficient statistics | `EdgeBook.state_dict()` |
| position, entry price, barriers, bars held | `Policy` |
| equity, high-water mark, alarm windows | `RiskLimits` |

Only `EdgeBook` currently has explicit serialisation. The rest is reconstructible
by **replay**, which is the recommended warm-start:

> **A cold start must replay at least `FeatureKernel.warmup` bars before emitting
> a signal.** Not doing so means the first signals come from half-filled windows,
> and half-filled windows produce ranks that are coarse histogram buckets with
> unstable boundaries.

Replay is deterministic (`tests/test_causality.py`), so a restart that replays the
same history reconstructs bit-identical state. That property is what makes replay
an acceptable substitute for serialisation.

---

## 5. Failover and reconciliation

* **Heartbeat.** Emit a liveness signal every bar, including `NO_TRADE` bars. A
  system that stands aside for weeks is indistinguishable from a dead one without
  this.
* **Reconciliation loop.** Compare intended position against broker position every
  bar. On disagreement: **default to flat.** Do not attempt to reason about which
  is right while the market is open.
* **Kill switch state must survive restart.** A halted system that restarts
  un-halted has defeated the switch.

---

## 6. Order handling

* **Market-on-open is the default** and is the least sophisticated possible
  execution. It is modelled pessimistically (`cost_slippage_range_frac` of the
  bar's range) so the backtest does not flatter it.
* **A limit at the open** improves the fill and introduces non-fill risk, which
  the barrier logic does not model. If used, unfilled signals must be recorded as
  such, not silently dropped — dropping them biases the record toward the trades
  that were easy to fill, which are the ones where the edge was weakest.
* **Participation caps.** `CostModel.capacity_participation` returns the largest
  fraction of bar volume at which the edge survives impact. At `η = 0.6` and a
  0.15σ edge that is 1.6%. **The system must refuse size its own cost model says
  is too large** — a system that will not refuse does not have a capacity limit,
  it has an undiscovered one.

---

## 7. Staged go-live

| Stage | Size | Graduation criteria |
|---|---|---|
| **Paper** | 0 | 3 months; live features match research replay to 1e-6; slippage model within 2× of realised |
| **Minimum** | smallest tradeable | ≥ 30 trades **and** ≥ 6 months; no kill switch tripped; realised expectancy within the pre-registered interval |
| **Quarter** | 25% target | ≥ 100 trades; calibration Brier below alarm; per-regime signs consistent with H2 |
| **Full** | target | ≥ 400 trades pooled across the universe (the power threshold from `docs/01-THEORY.md` §12) |

Two honest points about this table:

1. **A go-live decision after 20 trades is a coin flip.** The standard error on a
   hit rate at n = 20 is 11 percentage points. The trade counts above are minima
   for the *decision* to mean anything, not targets.
2. **The counts are reached across the universe, not per instrument.** At ~1%
   eligibility, 400 trades on one instrument is decades. This is the deployment
   consequence of the architecture's central constraint.

---

## 8. Capacity

From the square-root law, the edge survives up to
`q_max = (edge_sigma / 2η)²` of a bar's dollar volume. At a 0.15σ edge and
`η = 0.6`, that is 1.6% of bar volume — for a liquid future trading $10bn/day on
daily bars, roughly $160m per signal, before any consideration of how many
signals fire simultaneously.

That number is an upper bound derived from a model, not a measurement. The way to
find the real one is to increase size gradually and watch realised slippage
against the model; `slippage_alarm_mult` exists for exactly this.
