# Failure Modes

For each: mechanism, early-warning signature, severity, mitigation, and the
residual risk that remains after mitigation. Nothing here is hypothetical.

---

## 1. Statistical

### 1.1 Overfitting despite the protocol
**Mechanism.** Six `DEV` parameters is few, but the *research program* has many
more implicit degrees of freedom: which features to compute, which four regimes,
which setup families, the design table's 32 cells. None are fitted to outcomes,
but all were chosen by a designer who knows what markets look like.
**Signature.** Validation-universe performance materially below development.
Fold-to-fold parameter drift in walk-forward.
**Severity.** High. This is the default outcome for systems of this kind.
**Mitigation.** `Config.fitted_dof()` is auditable; the manifest is hashed;
`docs/00-PREREGISTRATION.md` fixes acceptance thresholds in advance; the trials
ledger charges every re-evaluation to the deflated Sharpe ratio.
**Residual.** Substantial and unquantifiable. Design choices made before any
data was touched cannot be charged to a trial count, and we cannot prove they
were uninformed by prior exposure to market data.

### 1.2 The regime taxonomy is wrong
**Mechanism.** Four regimes may not carve the space where the
continuation/reversal distinction actually lies. Then the top of the architecture
misroutes evidence and both directional engines contribute noise.
**Signature.** The falsifiable prediction in §00 fails: the profitable action does
*not* differ in kind between `TREND` and `REVERT`. Per-regime expectancies
statistically indistinguishable.
**Severity.** High — it invalidates the architecture's central claim.
**Mitigation.** Regime-stratified reporting is built into
`validation/report.py`. The prediction is stated in advance so it can fail.
**Residual.** If it fails, the correct response is redesign, not reparameterise.

### 1.3 Small-sample inference at high selectivity
**Mechanism.** The system trades ~1% of bars. 95 trades cannot distinguish a 60%
hit rate from a coin flip (`docs/01-THEORY.md` §12).
**Signature.** Any confident claim about performance from fewer than ~400 trades.
**Severity.** High, and mostly a risk of *self-deception* rather than of loss.
**Mitigation.** Standard errors are printed beside every estimate, in
`TIA.summary()` and in the validation report. The pre-registration sets a
minimum trade count for the primary test to conclude at all.
**Residual.** Real. The honest position is that a deployment must run for years,
or across dozens of instruments, before its live record means anything.

### 1.4 Multiple testing across the whole program
**Signature.** A deflated Sharpe ratio far below the nominal one.
**Mitigation.** `trials_ledger_count` is a config parameter and is charged.
**Residual.** Honest accounting depends on honest counting, which depends on
whoever maintains the ledger.

### 1.5 Non-stationarity outrunning the rank windows
**Mechanism.** Rank transforms assume the last 252 bars are a relevant reference
class. After a structural break they are not.
**Signature.** `psi_alarm` breach; rising population stability index.
**Mitigation.** PSI monitoring against a frozen development reference; the
`STRESS` regime; auto-demotion.
**Residual.** The window cannot shorten without becoming noise. There is a
genuine floor on how fast this can adapt.

---

## 2. Model-structural

### 2.1 Engine correlation converges faster than the estimator can see — **severe**
**Mechanism.** `C` is estimated on a trailing window with a 250-bar half-life.
In a crisis, engine correlations converge within days. Effective breadth is
therefore *overstated* for weeks, and the MacKay correction under-shrinks — so
the system is most overconfident exactly when overconfidence is most expensive.
**Signature.** Rising `mean_abs_corr` in `CALOP.diagnostics()`, falling
`ebe_at_full_reliability`, both lagging realised conditions.
**Severity.** Severe. This is the failure mode most likely to produce a large,
fast loss rather than a slow bleed.
**Mitigation.** The `STRESS` regime cuts size and widens stops; the hazard gate
stands the system down through transitions; `ebe_min` is a hard conjunctive
condition.
**Residual.** **Not fixed.** A trailing estimator cannot report a correlation
spike it has not yet seen. A forward-looking proxy (implied correlation,
cross-asset dispersion) would help and is not implemented.

### 2.2 LADR's denominator is unsigned
**Mechanism.** Participation is *total* volume, not signed order flow. Where
volume and consumed liquidity decouple — cross-venue sweeps, auction prints,
index rebalances, closing crosses — the primitive misreads, and it misreads in
the direction of calling mechanical flow "informed".
**Signature.** Momentum-engine reliability high on days with known mechanical
flow; systematic underperformance on rebalance dates.
**Mitigation.** LADR is a *confirming* input, never an originating one; the
momentum engine's reliability is discounted outside `TREND`.
**Residual.** Real. A footprint feed would resolve it; see `docs/16-ROADMAP.md`.

### 2.3 The cold-start dependence
**Mechanism.** The gate acts on a lower credible bound, so an untrained Edge Book
never opens it. Training is therefore load-bearing, and a training pass on
unrepresentative data produces a book that is confidently wrong rather than
appropriately empty.
**Signature.** Edge Book cells with high `n_eff` and expectancies that do not
replicate out of sample.
**Mitigation.** `walk_forward_train`; the counterfactual pass records *all*
structurally eligible bars, never selecting on outcome.
**Residual.** The book's quality is bounded by the development universe's
representativeness, which cannot be checked from inside it.

### 2.4 The spread estimator's resolution floor
**Mechanism.** Corwin–Schultz cannot resolve a spread far below the bar's own
volatility; below a few basis points it returns a conservative floor. Measured:
0.97x at 40bp, 3.6x at 2bp.
**Signature.** `half_spread_sigma` clustering near the estimator's floor;
liquid instruments never clearing the expectancy gate.
**Severity.** Moderate — it costs opportunity, not money.
**Mitigation.** `ExogenousSnapshot.spread` overrides the estimator and is the
production path. Above `cost_max_half_spread_sigma` the instrument is declared
untradeable rather than merely expensive.
**Residual.** OHLC-only deployments (TradingView) inherit the floor.

### 2.5 Calibration decay and Edge Book staleness
**Signature.** Rising Brier; falling *resolution* specifically — worse than
miscalibration and not fixable by recalibrating.
**Mitigation.** Rolling Brier decomposition, `brier_alarm`, demotion ladder.
**Residual.** The Edge Book half-life is deliberately long (5,040 bars). Long
memory is the right default but guarantees slow adaptation.

---

## 3. Market-structural

* **Liquidity evaporation invalidating the cost model.** Spreads widen and depth
  thins precisely when signals fire. The `stress_multiplier` scales
  liquidity-sensitive cost components by up to 2.5x; whether that is enough is
  unknown, because the estimator that would tell us is the one that fails.
* **Halts, limit moves, gaps through stops.** Modelled pessimistically
  (fill at the open), but no sizing rule anticipates them.
* **Microstructure regime change.** Decimalisation, the HFT transition, the 2020
  retail-flow shift each changed what a bar *means*. Rank normalisation absorbs
  level shifts but not changes in the relationship between volume and
  information.
* **Crowding and alpha decay.** The two effects the system rests on —
  time-series momentum and short-horizon reversal — are both published and both
  show evidence of post-publication decay.
* **Adverse selection.** Entering at the next bar's open with a market order is
  the least sophisticated possible execution.

---

## 4. Execution and operations

| Failure | Signature | Mitigation | Residual |
|---|---|---|---|
| slippage above model | `slippage_alarm_mult` breach | kill switch | cost model still wrong until refit |
| vendor bar differences | live/backtest divergence on identical dates | pin one vendor | silent if unmonitored |
| retroactive split adjustment | every historical feature changes | validator flags split-like ratios | a re-adjusted history invalidates prior comparisons *silently* |
| contract rolls | discontinuity in every window | documented convention | back-adjustment choice changes results |
| timezone/DST | session mistagging | fixed UTC offsets, exchange calendar in production | the fixed-offset default is wrong twice a year |
| outage and restart | position/intent mismatch | reconciliation loop, default-to-flat | a cold start must replay the full warmup |

---

## 5. Human

**This is where the system most likely dies.**

* **The psychological difficulty of abstention.** A system that stands aside for
  weeks feels broken. The pressure to "make it trade" is constant and is the
  proximate cause of most of the changes that would destroy it.
* **Override.** Taking the signals one likes.
* **Parameter tinkering after a drawdown.** Adjusting `ev_lcb_min_sigma` or
  `ebe_min` after a losing run converts a pre-registered system into an
  unregistered one, invalidates every acceptance test retroactively, and does so
  without leaving a trace unless the manifest hash is logged.

Mitigation is procedural, not technical: the change-control protocol in
`docs/14-CONTINUOUS-LEARNING.md`, mandatory waiting periods, and an immutable
change log tied to `Config.manifest_hash()`.

---

## 6. How this system most likely dies

Ranked by probability × severity:

**1. A human adjusts it after a drawdown.** Most likely by a wide margin.
Every technical mitigation in this repository is downstream of someone choosing
not to edit a config file during a bad month. The drawdown that triggers it will
almost certainly be inside the pre-registered distribution — which is precisely
why it feels like evidence and is not.

**2. Engine correlation converges in a crisis before the estimator notices**
(§2.1). The system sizes up on apparent breadth that has already evaporated and
takes a large loss in a short window. This is the most likely *technical* killer,
and it is not fully mitigated.

**3. The edge was never there.** The development universe produced an Edge Book
that reflects design choices rather than market structure, and the validation
universe reveals it. This is the most likely outcome *overall* — but it is a
death by discovery rather than by loss, and the protocol exists to make it happen
on paper rather than in an account.

Note what is *not* in the top three: a coding error. The test suite, the
causality auditor and the null-market check address that class directly, and
three real bugs were caught during development by exactly those means.
