# Final Review Board Report

The last review before this system is permitted near real capital. No further
AI review is assumed. This document contains the readiness assessment, the
risk register, the validation summary, the operational checklist, the phased
deployment protocol with objective promotion criteria, and the board's answers
to the six governing questions.

**Scope of this pass.** Three prior passes (build, bug-hunt, adversarial
audit) had already found and fixed fourteen defects. This pass verified their
fixes, closed four *production* gaps that all prior passes left open, corrected
one defect in the safety layer itself, and — for the first time — put the
system in contact with real market data.

---

## 1. What this pass found and changed

### F1 — The documented safety layer was not wired (fixed)

`docs/13-MONITORING.md` documented a five-rung demotion ladder and four kill
switches. Inspection showed the ladder was **never instantiated by the
pipeline**, and of the four kill switches only drawdown was ever fed — the
calibration, feature-drift and slippage switches were dead code paths. A
documented safety property that is not wired is worse than an undocumented
one, because operators plan around it.

Wired now: `TIA` owns a `DriftMonitor`; every valid feature snapshot and every
resolved outcome feeds it (outcomes regardless of learn mode — a frozen-model
deployment is precisely where decay must be caught); health refreshes every 50
bars; the rolling Brier feeds the `RiskLimits` calibration switch; and the
ladder acts through the existing gate so each refusal carries a named reason —
TIGHTENED halves size, RESTRICTED requires a committed regime posterior
(≥ 0.75), NO_TRADE and above block entries.

Proven, both directions: a poisoned monitor (Brier 0.81) demotes within 60
bars and produces **zero** entries with a named veto; a healthy trained run is
**not** demoted and trades normally.

### F2 — The safety layer itself had a defect (fixed)

The negative-Brier-skill rung demoted a demonstrably healthy run at n=62
outcomes. Cause: gated selection lifts realized success above the population
forecast, so small-sample "skill vs base rate" reads negative even when
miscalibration is in the *safe* direction (under-confidence). A ladder that
fires on healthy systems trains operators to ignore it — which
`docs/08-FAILURE-MODES.md` ranks as the way safety systems actually die.
Skill-based demotion now requires the isotonic map's own sample floor
(`calibration_min_samples = 200`); below it the finding is reported but not
acted on. The absolute Brier alarm is unchanged and catches genuine breakage
at any sample size.

### F3 — No journal existed (fixed)

A live divergence, a broker dispute, or a post-mortem all require knowing what
the system saw and decided, bar by bar. Added an append-only JSONL journal
(stdlib only, disabled by default): decisions with NO_TRADE heartbeats — a
system that stands aside for weeks must be distinguishable from a dead one —
trades with gross/cost/net, health transitions, and lifecycle events tied to
the config manifest hash. Verified: every journal number agrees with the
in-memory record to full precision; a failed journal disables itself without
taking the pipeline down; journalling changes no decision.

### F4 — Config accepted garbage (fixed)

`Config.validate()` now refuses to start on nonsensical values. The cases are
not hypothetical: a negative stop, a probability floor above one, or
`execution_lag_bars = 0` (look-ahead **by construction**) each previously
produced not an error but a system that traded wrongly.

### F5 — The gap blind spot is closed (fixed)

The adversarial audit showed the fill-timing bug (D2) survived 1.19M
assertions because the synthetic generator could not express an overnight gap.
The generator now can (`overnight_gap_sigma`, default 0, byte-identical to
existing seeds when off), and the accounting is verified exact on gapped
markets — every entry booked, every fill at the true next open, gap-through
stops filled at the (worse) open.

Real data quantified what was at stake: **SPY's median overnight gap is
27.9 bp (p95: 128.6 bp)**. Against modelled round trips of 10–20 bp, the old
close-fill convention would have injected more signal-directional optimism per
trade than the entire cost model — sufficient, on its own, to manufacture a
fake edge on real data.

### F6 — First contact with real market data (new evidence)

Six broad asset-class ETFs chosen a priori (SPY, QQQ, IWM, GLD, TLT, EEM),
15 years of daily bars, 22,632 bars total (`tools/realdata_check.py`,
reproducible):

| Check | Result |
|---|---|
| Validator on raw vendor bars | **0 fatal rejects** |
| Untrained system (cold-start discipline) | **0 signals** on all six |
| Structural eligibility in training | 0.0–1.7% of bars (design target ≈1%) |
| Pooled walk-forward (train 60%, trade 40%, shared book) | 66 training candidates → **0 out-of-sample trades** |

The zero is not a malfunction; it is the theory's power arithmetic
(`docs/01-THEORY.md` §12) arriving on schedule. Sixty-six weighted candidates
across ~13,500 training bars cannot move a lower credible bound past a cost
hurdle, so the gate stays shut. The pre-registered bar is ~400 trades; six
instruments of daily data cannot reach it. **The system's refusal to trade on
real data at realistic edge sizes is it working as designed, and the honest
conclusion is the pre-registered one: stop optimizing; widen the universe and
extend the history.**

Caveats on this data, stated: one vendor's bars; split-adjusted but not
dividend-adjusted closes (ex-dividend days appear as small opening gaps);
survivorship-lite universe (broad ETFs chosen a priori, not winners ex post).

---

## 2. Validation summary

| Verification | Status |
|---|---|
| Unit + regression + production tests | **157 passed, 0 skipped** |
| Deep invariant audit (`tools/audit.py`) | **1,189,356 checks, 0 violations** |
| Pine static checks (`tools/pinelint.py`) | **0 problems** |
| Causality: prefix-replay bit-identical; append-stability | pass |
| Null markets (6 seeds × 2,500 bars + audit's 8 × 2,500) | 0 signals |
| Cross-seed OOS transfer (machinery) | t = +7.2 / +5.7 net of costs |
| Every entry booked; fills at next open; equity cost-inclusive; reconstructible to 1e-12 | pass, incl. gapped markets |
| Config sanity, journal fidelity, ladder both directions | pass |
| Real data: integrity, cold start, gap measurement, pooled walk-forward | pass (see F6) |
| Pine ↔ Python equivalence | static + constant-level only; **no TradingView compile available here** — the one outstanding mechanical check |

Cumulative defect ledger across all passes: **19 verified defects found and
fixed** (8 in the bug-hunt, 5 in the adversarial audit, 6 in this pass
counting F1–F5 and the F2 meta-defect), each with a regression test naming the
incident.

## 3. Risk register

| # | Risk | Likelihood | Severity | Mitigation | Residual |
|---|---|---|---|---|---|
| R1 | No demonstrated market edge | — | — | pre-registered protocol, gate acts on LCB | **the** open question; everything else is machinery |
| R2 | Human overrides/tinkers after drawdown | high | high | change control, manifest hash, 30-day rule (`docs/14`) | procedural only; highest-ranked killer |
| R3 | Engine correlation converges in crisis faster than the trailing estimator sees | medium | severe | STRESS regime, hazard gate, `ebe_min`, correlation drift is the #1 monitored number | **not fixed**; accepted and documented |
| R4 | Pine port diverges from reference | medium | medium | frozen-model default emit, hash check, lint, sign tests | needs one TradingView compile + §7 verification protocol |
| R5 | Vendor data revisions invalidate live-vs-backtest comparison | medium | medium | snapshot-by-hash discipline (`docs/12`) | silent if operators skip the discipline |
| R6 | Cost model wrong in stress | medium | high | 2.5× stress multiplier, slippage kill switch | multiplier is judgement, not measurement |
| R7 | Monitor blind spots (cf. F2) | medium | medium | both-directions ladder tests | a monitor can only catch what its author imagined |
| R8 | Single-developer provenance; blind-spot correlation across all passes | certain | medium | three adversarial passes, real-data contact | **irreducible here**: the same author audited their own work; independent human review is required |

## 4. Operational checklist (go-live gate, Phase 1+)

Before any signal reaches even a paper account:

1. `python3 -m pytest tests/` — 157/157.
2. `python3 tools/audit.py` — 0 violations.
3. `python3 tools/pinelint.py` — 0 problems (if the chart port is used).
4. `python3 tools/realdata_check.py` — integrity + cold-start pass on the
   *deployment* universe.
5. `tia config` — manifest hash recorded in the run log; matches the trained
   Edge Book's provenance and (if used) `FM_HASH` in the Pine files.
6. Journal enabled (`journal_path=...`), write-verified, and shipped to
   durable storage.
7. Warm-start rule honoured: cold starts replay ≥ `FeatureKernel.warmup` bars
   before the first signal is honoured.
8. Kill-switch drill: poison a copy's monitor, verify demotion + named veto.
9. Reconciliation loop against broker positions live; default-to-flat on
   disagreement.
10. The pre-registered thresholds (`docs/00`) printed and signed by whoever
    owns the stop-doing-this decision.

## 5. Phased deployment protocol

| Phase | Capital | Entry criteria | Exit / promotion criteria |
|---|---|---|---|
| **0 — historical** | none | now | pre-registered campaign on the full universe: n ≥ 400 pooled trades, DSR ≥ 0.5, calibration ECE ≤ 0.05, per-regime consistency (H2), plateau + interiority, surrogate p ≤ 0.05. **Currently: FAILS on power — 0 OOS real-data trades.** |
| **1 — paper** | none | Phase 0 passes | ≥ 3 months **and** ≥ 30 paper trades; live features match research replay to 1e-6; realized slippage within 2× model; zero unexplained journal gaps |
| **2 — minimum live** | smallest tradeable size | Phase 1 passes | ≥ 6 months **and** ≥ 30 live trades; no kill switch tripped; realized expectancy inside the pre-registered interval; slippage within 2× model |
| **3 — scaled** | stepwise to target | Phase 2 passes | each step: ≥ 100 additional trades with SPRT against the null not rejecting decay; capacity bound (`capacity_participation`) respected at every step |

Non-negotiables at every phase: a trade count below the phase minimum is a
**hold**, not a judgement call — at n=20 the hit-rate standard error is 11
points and any decision is a coin flip; demotion to NO_TRADE at any phase
resets that phase's clock; any change to a THEORY/DEV parameter returns the
system to Phase 0 with a new ledger entry.

## 6. Remaining known limitations

1. **No statistically sufficient evidence of a market edge exists.** 66 real
   candidates against a 400-trade requirement. Per the mandate: further
   optimization should stop until additional real-market data — more
   instruments, longer history, or intraday bars — is collected.
2. H2 (the regime sign-flip, the architecture's strongest falsifiable claim)
   remains untested: synthetic generators can't express it and the real-data
   sample can't power it.
3. The Pine port has never been compiled by TradingView.
4. Portfolio-level correlation across simultaneous instruments is unmanaged
   (roadmap #2).
5. R3 (crisis correlation convergence) is documented, monitored, and unfixed.
6. Every review pass shares one author. R8 is irreducible from inside.

---

## 7. The six answers

**Is the implementation technically correct?**
Yes, to the limit of what this environment can verify — with the verification
being the strong part of the claim: bit-exact causality under prefix replay
and appending, exact accounting under gaps and corrupted streams (equity
reconstructible to 1e-12), every entry booked, probabilities bounded and
normalized everywhere including under saturated/degenerate inputs, 1.19M
invariant checks and 157 tests passing, and clean first contact with 22,632
real vendor bars. Two caveats: the Pine port is verified only statically, and
19 defects were found across the passes — several invisible to the tests that
existed when they shipped — so "correct" means "no known defect and strong
guards," not "proven."

**Is it production-ready from a software engineering perspective?**
Yes, at signal-service scope, as of this pass: config validation, structured
journaling with manifest provenance, a wired and two-way-tested demotion
ladder, fed kill switches, bounded memory, deterministic replay for
warm-starts, and a reproducible real-data protocol. What it is *not*: a
complete trading stack — order management, broker adapters, and portfolio
aggregation are out of scope and documented as such.

**Is there statistically sufficient evidence of a genuine trading edge?**
**No.** Stated without hedging. Synthetic results validate machinery, not
markets. The one real-data test produced 66 training candidates and zero
out-of-sample trades against a pre-registered requirement of ~400 at t ≥ 3.
The system's own acceptance report returns FAIL on power. Per the mandate:
further optimization should stop here; the only path to evidence is more real
data — wider universe, longer history, or finer bars — through the
pre-registered protocol, charging the trials ledger.

**Is it ready for paper trading?**
Engineering: yes. Protocol: paper trading (Phase 1) is gated behind Phase 0,
which currently fails on power. Running unofficial paper *now* is useful only
as an operations rehearsal — feeds, journal, reconciliation, kill-switch
drills — and its P&L must be treated as meaningless at the trade counts it
will produce. It cannot count toward promotion.

**Is it ready for live trading with small capital?**
**No.** Not as a hedge, not at minimum size. Small capital does not repair
absent evidence; it converts an unanswered statistical question into a slow,
noisy, unfalsifiable experiment with real money. The gate refuses at realistic
edge sizes on real data, and overriding one's own system's refusal is the
exact failure mode `docs/08-FAILURE-MODES.md` ranks first.

**What objective milestones must be met before increasing capital?**
In order, each gating the next: (1) Phase 0 passes in full on real data —
n ≥ 400 pooled OOS trades, DSR ≥ 0.5, ECE ≤ 0.05, H2 sign-flip confirmed,
plateau/interiority, surrogate p ≤ 0.05, all charged to the trials ledger;
(2) one successful TradingView compile plus the §7 Pine-vs-Python verification
protocol, if the chart port is deployed; (3) independent human review (R8);
(4) Phase 1: 3 months / ≥ 30 paper trades with 1e-6 replay fidelity and
slippage within 2× model; (5) Phase 2: 6 months / ≥ 30 live trades at minimum
size, no kill switch, expectancy inside the pre-registered interval; (6) each
scaling step thereafter: ≥ 100 further trades with a non-rejecting SPRT and
the capacity bound respected. A quantified summary of the board's confidence:
**machinery correctness — high, adversarially earned; existence of the edge —
unknown, currently unfalsifiable at the available sample size, and honestly
labelled as such by the system itself.**
