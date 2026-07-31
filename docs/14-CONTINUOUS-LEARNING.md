# Continuous Learning

---

## 1. What learns, and what does not

Tied directly to the `Provenance` tags in `config.py`:

| Provenance | Count | Updates | Rule |
|---|---|---|---|
| `THEORY` | ~45 | never | changing one is a research decision requiring a new pre-registration |
| `DEV` | **7** | at re-development only | frozen after the development fit; charged to the deflated Sharpe ratio |
| `ONLINE` | ~5 | every bar, causally | Edge Book counts, calibration map, engine correlation |
| `OPS` | ~10 | freely | account size, risk appetite, symbol list |

**Why the boundary sits there.** Online updating of *structural* parameters is
continuous overfitting with extra steps. It also destroys attribution: if the
model that produced a live result is not the model that was pre-registered, the
result cannot be evidence for or against the hypothesis. The `ONLINE` quantities
are exempt because at every point in time they use strictly past data — they add
estimation variance, not in-sample optimism, and `Config.online_dof()` reports how
much.

---

## 2. Causal online updating

All three online objects use exponentially-decayed sufficient statistics:

* **Edge Book** — three floats per cell (`n_w`, `Σwr`, `Σwr²`), decayed lazily on
  access so the update is O(1) and exact.
* **Calibrator** — a bounded ring of `(p, outcome)` pairs, refitted every 50
  resolutions rather than every bar. A map that moves continuously makes live
  behaviour irreproducible.
* **Engine correlation** — exponentially weighted, half-life 250 bars.

**On half-life choice.** The Edge Book's is deliberately long — 5,040 bars, about
twenty years of daily data. The reasoning is asymmetric: forgetting a real regime
you have not seen recently is worse than carrying stale data, because the stale
data is diluted by the hierarchy's shrinkage while the forgotten regime leaves a
cell empty and therefore *silently* untradeable. The engine correlation's is
short (250 bars) for the opposite reason: it must track a crisis, and it already
does not track one fast enough.

---

## 3. Periodic re-development

**Cadence:** annually, or on a demotion to `HALTED`, whichever comes first.

**The hard rule:** a re-development is a **new pre-registration** with a **new
trials-ledger entry**, and the deflated Sharpe ratio is recharged for every trial
to date.

**The ratchet to avoid.** Each re-development that "just looks at" the validation
set to check something incorporates knowledge of it into the next design. After a
few cycles the validation set is a second development set and its independence is
gone, without any single step having been obviously wrong. Defences:

* the validation universe is consumed by each evaluation and must be *extended*
  (later era, held-back instruments) rather than reused;
* every evaluation is logged in the ledger, including ones that were "only a
  sanity check";
* the manifest hash of what was evaluated is recorded, so an undocumented
  evaluation is detectable after the fact.

---

## 4. Drift, shift, or bad luck

Three different things that look identical in the short run:

| | What changed | Test | Response |
|---|---|---|---|
| **Concept drift** | the mapping from features to outcomes | rolling Brier and *resolution* falling while PSI is stable | re-develop |
| **Covariate shift** | the input distribution | PSI breach with calibration intact | restrict to familiar regions; re-develop if persistent |
| **Bad luck** | nothing | drawdown inside the pre-registered distribution; sequential test not rejecting | **do nothing** |

**They are often indistinguishable at the sample sizes available.** With ~50
trades a year, separating drift from bad luck at reasonable power takes years.
The practical implication is uncomfortable and should be stated: for most of a
deployment's life, the correct action on ambiguous evidence is to reduce size and
wait, not to diagnose. The demotion ladder implements exactly that.

---

## 5. Champion / challenger

A candidate runs in **shadow mode**, producing signals that are logged and not
traded, against the same data as the incumbent.

**How long before promotion is meaningful?** Longer than anyone wants. To detect
a 0.05σ improvement in per-trade expectancy against a 1.5σ outcome standard
deviation, at 80% power:

```
n ≈ 2 · (1.96 + 0.84)² · 1.5² / 0.05² ≈ 14,000 paired trades
```

Paired comparison on the same bars reduces this substantially — the two systems
share most of their variance — but even a generous correlation of 0.9 leaves
~1,400 paired trades. **At realistic trade rates that is years for a single
instrument and months for a broad universe**, which is another argument for
breadth.

Promotion criteria: the challenger must win on the paired sequential test, must
not have a worse Brier score, and must not have lower effective breadth at signal
time. A challenger that wins on expectancy while lowering breadth has probably
added a correlated engine and is buying apparent performance with real
overconfidence.

---

## 6. The anti-tinkering protocol

`docs/08-FAILURE-MODES.md` ranks a human adjusting the system after a drawdown as
the most likely cause of its death. This section is the mitigation, and it is
procedural because the failure is procedural.

* **Who may change what.** `OPS` parameters: the operator, logged. `ONLINE`:
  nobody, they update themselves. `DEV`: only at a re-development. `THEORY`: only
  with a new pre-registration.
* **Evidence required.** A `THEORY` or `DEV` change requires a written hypothesis,
  a pre-registered test, and a ledger entry — *before* the change is made.
* **Mandatory waiting period.** No parameter change within **30 days** of a new
  drawdown high-water mark. This is the single most important rule on the page,
  because it targets the exact moment when the temptation is strongest and the
  evidence is weakest.
* **Immutable change log.** Every change records the previous and new
  `Config.manifest_hash()`, the date, the author, and the justification. A
  running system whose hash is absent from the log is unauthorised.

None of this is technically enforced by the code, and it should not be — a
technical lock that can be removed is not a control. What the code provides is
*detectability*: the hash makes an undocumented change visible after the fact,
which is what turns a norm into an auditable one.
