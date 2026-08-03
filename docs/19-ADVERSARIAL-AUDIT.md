# Adversarial Audit

Independent audit conducted with the explicit mandate to prove the system
should **not** be traded. Everything that appeared to work was presumed
overfit or broken until demonstrated otherwise. This document is the complete
deliverable: every defect found, every fix applied, every optimization
rejected, and the final verdict.

**Method.** Rather than re-reading code the audit *attacked* it: eight
targeted attacks against the execution model, the training/trading interface,
the port, and the statistical claims, each designed so that a specific class
of failure would produce a measurable kill. Five attacks drew blood.

---

## 1. Defects found (all fixed, all regression-locked)

### D1 — Positions vanished from the accounting — **CRITICAL**

Bar-close exits (vertical barrier, confirmed reversal, session close) removed
the position from the state machine but were never booked: no trade record,
no equity change, no learning signal. **4 of 62 closed positions in the attack
run simply did not exist financially.**

Severity is worse than the count suggests, in three compounding ways:
timeout-class exits are systematically the *mediocre* outcomes, so the
reported record was censored toward barrier touches (wins at full target,
losses at full stop); in learn mode the Edge Book never observed a timeout,
biasing online learning optimistic; and the reported trade list showed
`{'target', 'stop'}` as the only exit reasons, which reads as clean when it
is actually the signature of the censoring.

*Fix:* `Policy` exposes the closed position (`last_closed`); the pipeline
books it at the decision bar's close with outcome 0 — matching the labeller's
vertical-barrier convention — and feeds it to the Edge Book and calibrator.
*Locked by:* `test_every_entry_produces_exactly_one_trade_record`,
`test_timeout_outcomes_reach_the_edge_book_in_learn_mode`.

### D2 — The live pipeline traded a different game than it trained on — **CRITICAL**

Entries filled at the **decision bar's close**. The triple-barrier labeller
that trains the Edge Book fills at the **next bar's open** (`execution_lag=1`),
which is also what `SPEC.md` §2 rule 3 requires and what the `Decision`'s own
`execute_at_index` field has claimed since the first commit. The backtest
ignored its own field: 20 of 20 sampled fills were at the decision close.

Why every prior test missed it: the synthetic generator produces
`open(t+1) == close(t)` exactly — no overnight gaps — so on synthetic data the
two fill conventions are *numerically identical*. The audit measured the
fill-gap-to-cost ratio at 0.00 on this data. On real data with gaps, the
close-fill convention silently pockets the overnight move on every entry, in
the direction of the signal, which is a look-ahead-adjacent optimism that
grows with exactly the volatility conditions where signals fire.

*Fix:* deferred fills. The position is opened unfilled at decision; the
executor fills at the next bar's open and **re-anchors the barriers to the
actual fill** (sigma distances preserved), then barrier-checks that same
bar's range — byte-for-byte the labeller's convention.
*Locked by:* `test_entries_fill_at_the_next_bars_open`.

### D3 — Realized equity charged zero transaction costs — **CRITICAL**

The gate netted the modelled round trip for the *decision*; the booked P&L
was cost-free. In the attack run this flattered the equity curve by
**10.18σ of round trips across 62 trades**. The single easiest way for a
backtest to flatter itself, hiding in the gap between two layers that were
each individually correct.

*Fix:* the decision-time `round_trip_sigma` is stored on the position and
charged to equity at booking. `ret_sigma` deliberately stays **gross**,
because the Edge Book must learn the same gross quantity the gate later nets —
charging costs twice would double-count them against the edge.
*Locked by:* `test_equity_is_charged_the_modelled_round_trip`.

### D4 — A quarter of the setup taxonomy was unreachable

`classify_setup` read `sweep_age` from `EngineOutput.diagnostics`; the
liquidity engine reports it in `.features` while a sweep is active. Result:
across 20,000 audited bars containing 2,193 active-sweep bars, **zero** were
classified `SWEEP_REVERSAL`. The Edge Book cells for one of four families
never existed, every sweep trade was judged against the wrong reference
class, and — worse for the port — the Pine implementation classifies sweeps
correctly, so Python and Pine disagreed about which cell a trade belongs to.

*Fix:* read `.features` first with a `.diagnostics` fallback.
*Locked by:* `test_sweep_reversal_family_is_reachable`.

### D5 — The Pine regime posterior was permanently tilted

`f_betaLogPdf` omitted the Beta log-normalizer `ln B(a,b)`. The omitted
constants differ **by regime row** — TREND +11.04, REVERT +15.20, STRESS
+10.99, QUIET +13.80 — so every bar, regardless of data, the Pine posterior
carried a fixed tilt of up to 4.2 log units (≈ e⁴ in odds) between regimes.
The export had shipped the correct constants all along, and the emitted file
even declared them; the consumption code ignored them.

**The audit's own first fix was also wrong**: the export ships `ln B(a,b)` —
the value the reference *subtracts* — and the first patch *added* it,
inverting the correction. The cross-check against the Python `beta_logpdf`
caught the auditor's fix before it shipped. Final agreement: 4×10⁻¹².
*Locked by:* `test_pine_lognorm_sign_matches_the_reference`, which checks
both the constants and the sign of the subtraction in the `.pine` source.

---

## 2. Statistical audit

### What the attacks established

**Null discipline holds.** 0 signals across 6 × 2,500 martingale bars with
realistic volatility clustering, *after* the execution fixes. The gate is not
finding structure in noise, and the cost model is not manufacturing trades.

**The machinery transfers across independent samples.** Trained on three
disjoint seeds, traded on two never-seen seeds of the same process:
t = +7.22 (55 trades) and t = +5.67 (40 trades), net of costs, at next-open
fills. The learning-and-gating machinery captures a real edge when one exists
and carries it out of sample.

**The gate refuses undetectable edges.** At an edge strength comparable to
real markets (0.06, cf. `docs/01-THEORY.md` §0's IC ≈ 0.04), the corrected
system emits **zero** trades — gross expectancy positive, evidence
insufficient, lower credible bound negative. This is the design's central
claim behaving correctly under attack.

**The result is not cost-marginal.** On the OOS run, expectancy survives a
5× multiplication of every modelled cost (t falls +7.22 → +4.44). The
machinery's margin over its own friction model is wide — *on this synthetic
edge*.

**Regime routing is real but H2 is not yet testable.** The trained book's
cell occupancy differs sharply by regime (CONTINUATION carries n=129 in
TREND vs 0 in REVERT; RANGE_FADE 12.6 in REVERT vs 0 in TREND) — the
taxonomy routes. But the *sign-flip* prediction (H2, the pre-registration's
strongest test) cannot be evaluated on a generator whose both regimes reward
their own setup positively. H2 remains open until real data.

### What no attack here can establish

Every number above comes from a generator whose edge is 5–10× anything real.
These attacks validate the **machinery** — learning, gating, transfer,
accounting — and say **nothing about whether real markets contain the edge**.
The synthetic data also has two structural blind spots now documented as
such: opens equal prior closes (fill-timing bugs invisible — this is exactly
how D2 survived), and both regimes reward their own setups (H2 untestable).

---

## 3. Phase-by-phase findings where no change is recommended

| Area | Verdict |
|---|---|
| Causality / repainting | **No defect found.** Prefix-replay and append-stability tests are bit-exact; pivots, sweeps and HTF views pay their confirmation delays; the forward-only recursions are genuinely forward-only. |
| `request.security` usage | **Superseded in rc5.** The audit checked the flags and found both `[1]` and `lookahead_off` present, but did not check whether the *expression* was legal: it was a mutable variable, which Pine rejects, so the file this row cleared could never have compiled. The port now contains no `request.security` at all. See `pine/CHANGELOG_TRADINGVIEW.md` C1. |
| Session/timezone | Fixed-offset design is documented and intentional; production is told to supply exchange calendars. No change. |
| Numerical stability | 40k-bar run: correlation condition number 4.8, no drift, bounded state. Periodic rebuilds in `RollingMoments` are doing their job. No change. |
| Parameter sensitivity | 295 configurations across every scannable parameter's declared neighbourhood run without failure; plateau protocol pre-registered. No change. |
| Risk engine | Bayesian-Kelly with uncertainty haircut, drawdown throttle, kill switches: each term traced to a stated justification; nothing statistically unjustified found. The missing piece (portfolio correlation) is already the roadmap's #2 and is a scope boundary, not a defect. |
| Architecture | No superior architecture identified *at this evidence level*. The one structural criticism that survives: the regime measurement design table (32 asserted cells) is the largest unvalidated object in the system — but replacing assertion with estimation before real data exists would convert the system's strongest anti-overfit property into 32 fitted parameters. Rejected. |

## 4. Optimizations applied and rejected

**Applied:** only the five defect fixes. Net line-count change is modest and
every added line is regression-locked.

**Rejected, deliberately:**
- *Incremental variance-ratio / CALOP Cholesky caching* (≈20% CPU): behavior-
  preserving but touches the two most delicate numerical paths in the same
  change-set as five semantic fixes. Optimizing and bug-fixing in one pass is
  how regressions hide. Documented in `docs/15-COMPLEXITY.md` §3; do it in a
  quiet change-set.
- *Refitting anything after the fixes*: the corrected execution model changes
  measured expectancies; retuning any parameter against the new numbers would
  be optimization against the test. Parameters stand.
- *Softening the gate to make the 0.06 row trade*: the refusal is the design.

## 5. Estimated impact of each fix

| Fix | Impact on reported results |
|---|---|
| D1 booking | removes optimistic censoring; adds timeout outcomes (mediocre by construction) to every future expectancy estimate |
| D2 fills | zero on this synthetic data; on real gapped data, removes signal-directional overnight optimism from every entry |
| D3 costs | −10.2σ over 62 trades in the attack run (~0.16σ/trade); scales with trade count |
| D4 taxonomy | sweep trades now judged against their own reference class; Python/Pine now agree on cell assignment |
| D5 Pine regime | removes a permanent ≈e⁴ inter-regime odds tilt from every chart evaluation |

## 6. Remaining limitations (unchanged by this audit)

1. **No real-market evidence.** Every number in this repository remains
   synthetic. The pre-registered protocol (`docs/00-PREREGISTRATION.md`) is
   still unexecuted; H2 is untested.
2. Engine-correlation lag in crises (`docs/08-FAILURE-MODES.md` §2.1) is
   still the top technical risk and still unmitigated.
3. The Pine port still cannot be compiled here; `pinelint` + the sign tests
   are static approximations. One TradingView compile remains the highest-
   value outstanding check.
4. The spread estimator's resolution floor still makes OHLC-only deployments
   conservative to the point of refusal on tight-spread instruments.
5. Synthetic OOS transfer is machinery validation, not market validation.

## 7. Verdict

**Should this be traded today? No.** Not because the audit found the
machinery unsound — after the five fixes, 135 tests, 1.19M invariant checks
and the attack battery all pass — but because the only evidence of *market*
edge is synthetic, the pre-registered acceptance protocol has never been run
on real data, and the system's own validation report currently returns
`FAIL` on statistical power (25 trades of the 400 required). The system
itself agrees with this verdict; that is its best property.

**Is it ready for the pre-registered validation campaign on real data?
Yes** — with the execution model now consistent between training, backtest,
spec, and (pending one compile) the chart port. Confidence that the
*machinery* is correct: high, and earned adversarially. Confidence that the
*edge* exists: none is claimed, and none should be until the protocol runs.

The most important sentence in this audit: **two of the five critical bugs
(D2, D3) were invisible to every test written by the system's own author,
because the test data and the accounting shared the author's blind spots.**
The fill-timing bug survived a 1.19M-assertion audit suite because the
synthetic generator cannot express an overnight gap. Whoever runs the real-
data campaign should assume the same class of blind spot exists in this
audit, and budget for a third pair of eyes.
