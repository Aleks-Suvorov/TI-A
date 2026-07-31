# Pre-Registration

Written before validation data was touched. Its purpose is to make the results
falsifiable: thresholds fixed in advance cannot be moved afterwards, and a
redesign that follows a failure is charged to the trial count.

**Manifest at registration:** `Config.manifest_hash()` recorded in the ledger
below. Any result reported against a different hash describes a different system.

---

## 1. Hypotheses

### H1 — Primary
The gated signal has **positive expectancy net of modelled costs**, pooled across
the development universe, at **t ≥ 3** after uniqueness weighting.

The t-statistic must be computed against the **effective** sample size, not the
trade count. Measured on synthetic data the uniqueness ratio is ~0.75, so the
naive statistic overstates significance by ~1.15x; at longer horizons the factor
approaches 4.

### H2 — Regime conditionality (the strong, falsifiable one)
**The sign of the profitable action flips between the `TREND` and `REVERT`
regimes.** In `TREND`, continuation setups carry positive expectancy and fade
setups do not; in `REVERT`, the reverse.

This is the architecture's central claim, and it is the hypothesis most able to
fail. If both regimes show the same behaviour, the regime layer is not doing the
job assigned to it and the design is wrong — not mis-tuned.

### H3 — Calibration
Stated probabilities are calibrated: expected calibration error ≤ 0.05 and Brier
skill against the base rate > 0.

### H4 — Engine contribution
Each engine has non-negative marginal contribution under leave-one-out ablation.
Any engine whose removal *improves* out-of-sample expectancy is removed — and
because adding correlated engines lowers effective breadth while appearing to
raise confidence, the prior expectation is that at least one will fail this.

### H5 — Breadth
Signals are not generated when aligned effective breadth is below `ebe_min`.
Verified structurally by the gate, and reported so that the distribution of
breadth at signal time can be inspected.

---

## 2. Universe split

**Development** — parameters fitted, Edge Book trained, calibration fitted:

| Class | Instruments | Period |
|---|---|---|
| Equity index futures | ES, NQ, FESX, NK | 2005–2016 |
| FX majors | EURUSD, USDJPY, GBPUSD | 2005–2016 |

**Validation — quarantined, touched exactly once:**

| Class | Instruments | Period |
|---|---|---|
| Single-name equities | 30 large-cap, survivorship-corrected | 2005–2016 |
| Commodities | CL, GC, HG | 2005–2016 |
| Crypto | BTC, ETH | 2017–present |
| Cross-decade | all development instruments | 1990–2004 and 2017–present |

Disjoint by **asset class** and by **era**. The cross-decade split matters most:
it tests whether the rank-transform discipline (`docs/01-THEORY.md` §10) delivers
what it claims — that a parameter set chosen on 1990s futures is meaningful on
2024 crypto because the *features* have been made comparable.

**Single-use rule.** Each evaluation against validation data increments
`trials_ledger_count`. There is no "quick check".

---

## 3. Acceptance criteria

Pre-committed. All must pass.

| # | Criterion | Threshold |
|---|---|---|
| C1 | expectancy t-statistic (effective n) | ≥ 3.0 |
| C2 | trade count for power | ≥ 400 |
| C3 | deflated Sharpe ratio | ≥ 0.5 |
| C4 | expected calibration error | ≤ 0.05 |
| C5 | Brier skill vs base rate | > 0 |
| C6 | maximum drawdown | ≤ 25% |
| C7 | per-regime consistency (H2) | sign flip present, both directions t ≥ 2 |
| C8 | worst parameter plateau ratio | ≤ 1.35 |
| C9 | plateau interiority | chosen point is **not** the best in its own scan |
| C10 | GARCH surrogate null p-value | ≤ 0.05 |
| C11 | timing-shift monotonicity | performance decays smoothly at ±1..5 bars |
| C12 | cross-market consistency | ≥ 3 of 4 validation classes with positive expectancy |

C2 deserves emphasis: **below 400 trades the primary test does not have the power
to conclude, and a "pass" on C1 with fewer is not a pass.** See
`docs/01-THEORY.md` §12.

C9 is easy to overlook and is the one most likely to catch a tuned parameter. A
plateau can be flat and still have the chosen value sitting at its best point,
which is what tuning looks like when the scan is too narrow.

---

## 4. Stopping and redesign rules

* **Any criterion fails** → the configuration is rejected. It is not adjusted and
  re-tested against the same data.
* **A redesign following a failure** increments `trials_ledger_count`, requires a
  new ledger entry, and the deflated Sharpe ratio is recharged for every trial to
  date.
* **No partial credit.** Reporting "passes 10 of 12 criteria" is reporting a
  failure with extra words.
* **The validation universe is consumed.** After a failure, a genuinely
  independent re-test requires new data — a later era, or instruments held back
  from both universes.

---

## 5. Trials ledger

| # | Date | Manifest | Change | Evaluated against | Outcome |
|---|---|---|---|---|---|
| 1 | initial | `20499f7708b16f27` | initial specification | development only (synthetic) | not yet evaluated against validation |

Recorded amendments to `THEORY`-tagged defaults, made before any validation
evaluation and on the basis of *measurement*, not performance:

* `ebe_min` 2.5 → 1.25. The original was unreachable: nine engines with measured
  mean |correlation| 0.30 give a total breadth at full reliability of 2.66, so a
  threshold of 2.5 on *aligned* breadth was a de facto never-trade rule.
* `cost_slippage_range_frac` 0.10 → 0.02. Charged against a real bar range of
  ~1.3σ, the original produced a round-trip cost near 0.28σ, several times any
  plausible edge.
* `cost_max_half_spread_sigma` added at 0.20, with instruments above it declared
  untradeable rather than merely expensive.

These are recorded rather than hidden because each is a change to a parameter the
manifest calls `THEORY`, and the discipline is worth nothing if amendments are
silent.

---

## 6. What would falsify the whole thesis

Stated plainly, so that it can happen:

1. **H2 fails.** The regime layer does not separate the two effects, and the
   architecture's organising claim is wrong.
2. **Expectancy is positive gross and negative net across every reasonable cost
   assumption.** The edge exists and is not harvestable, which is the most common
   fate of published anomalies.
3. **The GARCH surrogate null is not rejected.** The strategy profits on paths
   with volatility clustering and no directional structure, meaning it harvests
   volatility or exploits a cost-model artifact rather than predicting direction.
4. **Cross-decade validation fails while within-decade succeeds.** The
   rank-transform portability claim is false, and every "cross-market" result is
   a disguised refit.
5. **Effective breadth at signal time is consistently near 1.** The nine engines
   are one engine, and the entire fusion apparatus is decoration.

Any of these is a reason to stop, not to reparameterise.
