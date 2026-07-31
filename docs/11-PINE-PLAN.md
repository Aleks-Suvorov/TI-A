# Pine Script Port — Design and Verification

## 1. The frozen-model split

Pine Script cannot learn. There is no cross-session persistence, no optimiser, no
way to accumulate an Edge Book across a universe of instruments, and no way to
fit an isotonic calibration map. So the port is an *evaluator*:

```
Python (research)                          Pine (evaluation)
─────────────────                          ─────────────────
counterfactual training pass
  → Edge Book posteriors        ──┐
  → isotonic calibration map    ──┤
  → engine correlation matrix   ──┼──►  pine/frozen_model.json
  → DEV parameters              ──┤          │
  → regime tempering exponent   ──┘          │  tia.export.emit_pine_constants
                                             ▼
                                    constants block inside TIA.pine
                                             │
                            features + engines + fusion + gate, live
```

This is not a compromise forced by the platform. It is the split the
architecture already has: `Provenance.DEV` parameters are fitted once and frozen,
and `Provenance.ONLINE` quantities are causal recursions a streaming port
reproduces exactly.

**Workflow.**

```bash
cd python
PYTHONPATH=src python3 -m tia.export                      # writes pine/frozen_model.json
PYTHONPATH=src python3 -c "from tia.export import emit_pine_constants; \
    emit_pine_constants('../pine/frozen_model.json', '../pine/TIA.pine', '../pine/TIA_strategy.pine')"
```

The emitter rewrites only the region between the `BEGIN/END FROZEN MODEL`
markers. Everything outside is hand-written and is never touched.

---

## 2. What maps one-to-one

| Component | Python | Pine | Notes |
|---|---|---|---|
| Bipower variation, jump share | `features/scales.py` | ✓ | identical formula |
| Rogers–Satchell | `features/scales.py` | ✓ | |
| EWMA volatility | `features/scales.py` | ✓ | `alpha = 2/(2·hl+1)` |
| Geometric HAR forecast | `features/scales.py` | ✓ | weights frozen |
| Corwin–Schultz spread | `features/microstructure.py` | ✓ | bars `(t-1,t)`, rolling **median** |
| LADR, absorption | `features/microstructure.py` | ✓ | |
| Close location value | `types.Bar.clv` | ✓ | |
| Kalman local linear trend | `features/trend.py` | ✓ | same recursion, forward only |
| Path efficiency | `features/trend.py` | ✓ | |
| Permutation entropy, order 3 | `features/entropy.py` | ✓ | |
| Regime Beta measurement model | `engines/regime.py` | ✓ | design table frozen, `Beta(1,1)` contributes exactly 0 |
| Sticky forward Bayes recursion | `engines/regime.py` | ✓ | log space, forward only |
| Transition hazard | `engines/regime.py` | ✓ | `max(drift, ambiguity)` |
| CALOP pooling, total and aligned breadth | `fusion/calop.py` | ✓ | `C⁻¹` frozen, `D` live |
| MacKay probit correction | `fusion/calop.py` | ✓ | |
| Isotonic map | `fusion/calibration.py` | ✓ | knots interpolated, `numpy.interp` semantics |
| Student-t quantile | `engines/edgebook.py` | ✓ | same Cornish–Fisher expansion |
| Cost model | `decision/costs.py` | ✓ | |
| Barrier construction | `decision/barriers.py` | ✓ | |
| Ten gate conditions | `decision/policy.py` | ✓ | |
| Position state machine | `decision/policy.py` | ✓ | including the two-bar reversal rule |

---

## 3. What is approximated, and by how much

**Lo–MacKinlay variance ratio.** Python computes the heteroskedasticity-robust
statistic with the full `θ(q) = Σ [2(q−j)/q]² δ_j` correction at four horizons.
That is `O(window × q_max)` per bar — roughly 2,000 operations — and Pine's
per-bar budget does not accommodate it four times over. The port uses the
*homoskedastic* ratio divided by a frozen null standard deviation calibrated on
the development universe.

*Direction of the error*: the homoskedastic statistic over-rejects the martingale
null during volatility clusters, so the Pine `vrZ` reads more extreme than
Python's exactly when volatility rises. The consequence is that the Pine regime
posterior leans toward `TREND` more readily in turbulent conditions than the
reference does. **This is the largest single fidelity gap in the port** and is the
first thing to check when comparing outputs.

**Participation normalisation.** Python normalises dollar volume within a
time-of-day bucket, because the intraday U-shape means an unconditional volume
comparison largely measures what hour it is. Pine uses a single rolling median.
On daily bars the two are identical. On intraday bars the Pine `participation`
carries a systematic intraday cycle, which inflates `ladr_rank` near the open and
close and deflates it midday. *Mitigation*: use the daily timeframe, or accept
that intraday sweep detection will fire more readily at the session edges.

**Regime tempering exponent.** Python re-estimates `η = EBE_f / J` every bar from
a rolling 8×8 feature correlation. Pine takes the frozen value. The exported
`tempering_exponent_sd` records the realised dispersion over the training run so
the approximation error is auditable; where that standard deviation is small the
approximation is harmless.

**Engine set.** Pine implements seven engines: `regime`, `trend`, `momentum`,
`liquidity`, `volatility`, `structure`, `behavioral`. The `crossasset` and
`positioning` engines are omitted — both are optional in the reference
(`SPEC.md` §9 forbids them from the gate's conjunctive conditions), both need
exogenous data Pine would have to fetch per bar, and `positioning` is disabled by
default anyway. The exported correlation matrix and `ebe_at_unit_reliability` are
computed over exactly the seven ported engines, so the breadth arithmetic is
internally consistent; `ebe_min` remains valid because it was calibrated as a
fraction of available independence, not as an absolute count.

**Behavioral engine.** Reduced to session trust. The event-proximity term needs a
calendar Pine does not have; consistent with the reference's behaviour of not
applying the adjustment rather than assuming the calendar is clear.

**Higher timeframe.** One view (`htf_multiple = 5`), not two. The 25× structural
view would need a second `request.security` pair and adds little at Pine's
warmup budget.

---

## 4. What cannot be ported at all

* **The counterfactual training pass.** Requires labelling every structurally
  eligible bar with a triple barrier and weighting for overlap. Pine has no
  concept of a label set.
* **Universe pooling.** The Edge Book's statistical power comes from sharing one
  book across many instruments (theory §12). Pine sees one chart.
* **Purged, embargoed cross-validation.** No.
* **Online Edge Book updates and isotonic refitting.** No persistence.
* **The uncertainty haircut on position size**, in its exact form. The Pine
  strategy substitutes the frozen `lcb/mean` ratio per cell.

The practical consequence: **the Pine script cannot tell you whether the strategy
works.** It can only show you what a trained model does on one chart. Validation
lives in `docs/07-VALIDATION.md` and runs in Python.

---

## 5. Repainting traps, and how each is avoided

| Trap | Avoided by |
|---|---|
| `request.security` without `lookahead_off` | serves the *completed* higher-timeframe bar to historical bars that could not have seen it — the classic backtest-only miracle |
| `lookahead_off` but no `[1]` | the in-progress higher-timeframe bar still updates intrabar, so a historical signal changes as the bar forms |
| **both together** | only fully-closed higher-timeframe bars reach the script — this is what the port does |
| `ta.pivothigh(l, r)` consumed at the pivot bar | the value is only *known* `r` bars later; the port tracks `pivotHighBar` and consumes the level from the bar it became visible |
| Signals evaluated intrabar | every emission is guarded by `barstate.isconfirmed` |
| Sweep marked at penetration | the port requires a later closed bar to reclaim the level; a sweep is knowable only after it fails |
| Forward-backward regime smoothing | forward recursion only; the smoother would relabel all history on every new bar |
| Corwin–Schultz over `(t, t+1)` | the port uses `(t-1, t)`; written the natural way this estimator is trivially forward-looking |

---

## 6. Cost per bar

Dominated by two loops:

| Component | Cost |
|---|---|
| Permutation entropy | 6 × 60 pattern comparisons = 360 ops |
| Regime forward recursion | 4 × 8 Beta densities + 4 × 4 transitions ≈ 50 ops |
| CALOP total breadth | 7 × 7 = 49 multiply-adds |
| CALOP aligned breadth | ≤ 49 multiply-adds |
| Everything else | `ta.*` built-ins, effectively free |

Roughly 500 operations per bar, comfortably inside Pine's budget. The permutation
entropy loop is the one to optimise first if limits are hit; a ring-buffer count
would make it O(1) as it is in the Python reference.

`max_bars_back = 1000` and warmup is ~350 bars.

---

## 7. Verification protocol

The port and the reference must agree. Concretely:

1. **Export a reference trace.** Run the Python pipeline on the same instrument
   and timeframe and dump per-bar features to CSV:

   ```python
   from tia.features.kernel import FeatureKernel
   from tia import Config
   k = FeatureKernel(Config())
   rows = [k.update(b).as_dict() for b in bars]
   ```

2. **Plot the same quantities in Pine** with `plot(..., display=display.data_window)`
   and read them off the data window bar by bar, or temporarily `plotchar` the
   value as a label.

3. **Compare, with these tolerances.** They differ by quantity because the
   approximation sources differ:

   | Quantity | Tolerance | Rationale |
   |---|---|---|
   | `sigmaBp`, `sigmaRv`, `sigmaRs`, `sigmaFcst` | 1e-6 relative | identical closed forms |
   | `jumpShare`, `efficiency`, `clv` | 1e-6 absolute | identical |
   | `slopeT` | 1e-4 relative | same recursion, different accumulation order |
   | `spreadRel` | 5% relative | `ta.median` vs the reference's sorted-list median treat ties differently |
   | `volRank`, `ladrRank`, `partRank` | 0.02 absolute | `ta.percentrank` uses a strict `<` convention; the reference uses mid-rank for ties |
   | `permEntropy` | 1e-6 absolute | identical |
   | `vrZ` | **no tolerance — expect disagreement** | different estimator, see §3 |
   | `posterior[k]` | 0.05 absolute | inherits `vrZ` and rank differences |
   | `logOdds` | 0.10 absolute | inherits the above |
   | `pSuccess` | 0.03 absolute | the probit correction compresses upstream error |

4. **Compare the decisions, not just the features.** The acceptance criterion is
   that **at least 90% of Python's actionable signals appear in Pine within ±1
   bar, and Pine emits no more than 1.25× Python's signal count.** A port that
   emits *more* signals than the reference has lost a gate somewhere, and that is
   the failure mode to watch for.

5. **Re-run after every export.** The constants block changes with every
   retraining, and a stale block silently pairs new logic with old posteriors.
   `FM_HASH` carries the config manifest hash; compare it against
   `Config.manifest_hash()` before trusting any comparison.

---

## 8. Known deficiencies, stated plainly

* The variance-ratio approximation is the weakest link and biases the regime
  posterior toward `TREND` in volatile conditions.
* Intraday participation carries a session cycle the reference removes.
* The Edge Book is frozen, so Pine's expectancy estimates go stale between
  retrainings while a live Python system's do not.
* TradingView's backtester cannot resolve intrabar order; the strategy wrapper
  submits the stop first so the ambiguity resolves against us, but that is a
  convention, not a measurement.
* None of this is validated on real market data in this repository. Every number
  in `frozen_model.json` currently comes from a synthetic training run, which is
  a smoke test and nothing more.
