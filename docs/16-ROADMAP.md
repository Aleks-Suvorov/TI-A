# Roadmap

Ordered by expected value. The ordering is argued, not assumed: items near the
top change what the system *is*, items near the bottom change how well it does
what it already does.

---

## 1. Cross-sectional ranking across a universe — **the largest single win**

**What.** Instead of asking "does this instrument clear an absolute expectancy
threshold?", rank all instruments at each bar and act on the extremes.

**Why it dominates everything else.** Three compounding reasons:

1. **Common noise cancels.** Much of a single instrument's return is market-wide.
   Ranking differences out a factor the absolute threshold has to fight through.
   The information coefficient on a cross-sectional signal is typically higher
   than on the same signal used absolutely, for exactly this reason.
2. **It converts a calibration problem into an ordering problem.** The absolute
   gate needs `EdgeBook.lcb` to be right *in level*, which requires the cost model
   and the volatility forecast to be right in level. A ranking needs only the
   ordering to be right, which is a far weaker requirement and far more robust to
   the estimator biases documented in `docs/08-FAILURE-MODES.md`.
3. **The architecture already assumes it.** The Edge Book is shared across
   instruments precisely because statistical power comes from pooling
   (`docs/01-THEORY.md` §12). Ranking is the natural consumer of that design; not
   doing it leaves the pooling half-used.

**Expected magnitude.** Large — plausibly the difference between an edge that is
detectable and one that is not.
**Cost.** Moderate: a universe-level scheduler, simultaneous position management,
and a portfolio-level risk layer that does not yet exist.
**Mirage risk.** Low. The mechanism is well understood.
**Validation.** Cross-sectional information coefficient by regime; compare against
the absolute gate on the same universe and period.

---

## 2. Portfolio construction across simultaneous signals

**What.** Correlation-aware sizing when several instruments signal at once.
**Why.** Currently sizing is per instrument, so twenty correlated signals produce
twenty positions that are not twenty independent bets. This is the largest
*unmitigated* risk exposure in the system (`docs/06-RISK-SIZING.md` §6).
**Expected magnitude.** Large on risk, neutral-to-positive on return.
**Cost.** Moderate. The CALOP machinery already computes exactly the right
object — effective breadth from a correlation matrix — and it applies unchanged
to positions instead of engines.
**Mirage risk.** None; this is risk control, not alpha.

---

## 3. Signed order flow, if a footprint feed becomes available

**What.** Replace total volume with buy/sell-initiated volume in LADR and
absorption.
**Why.** LADR's unsigned denominator is its known weakness
(`docs/08-FAILURE-MODES.md` §2.2). Signed flow is the quantity Kyle's model
actually concerns; total volume is a proxy for it.
**Expected magnitude.** Moderate on the momentum and liquidity engines, small on
the system as a whole — those engines are confirming, not originating.
**Cost.** High: the data is expensive, vendor-specific, and its classification
rules differ.
**Mirage risk.** Moderate. Trade-classification errors are systematic and
correlate with volatility.
**Validation.** Ablation against the unsigned version on the same bars.

---

## 4. Meta-labelling with a small constrained learner

**What.** A low-capacity model (logistic regression or a depth-2 gradient boosting
model with strong regularisation) predicting `P(the gated signal is right)` from
the engine outputs, sitting on top of the existing gate.
**Why.** This is the one use of machine learning defensible at this
signal-to-noise ratio: the primary model proposes, the secondary model estimates
whether to act. It cannot invent signals, only decline them.
**Expected magnitude.** Small-to-moderate — the literature reports meaningful
precision gains from meta-labelling.
**Cost.** Low. The training pass already produces exactly the labelled dataset it
needs.
**Mirage risk.** **High**, and this is the item most likely to look good and be
worthless. Mitigations: cap capacity hard, require it to survive CPCV with
purging, and require the ablation to show it *only* removes trades.

---

## 5. Fitted or realised-kernel volatility forecasting

**What.** Replace the fixed `(0.5, 0.3, 0.2)` geometric blend with a fitted HAR,
or a realised-kernel estimator if intraday data is available.
**Why.** Volatility is the denominator of everything.
**Expected magnitude.** Small. The documented gap between a fixed blend and a
fitted HAR is real but modest, and the fixed weights buy a `THEORY` tag instead of
a `DEV` one — a genuine reduction in fitted degrees of freedom.
**Mirage risk.** Low.
**Note.** `fit_har_weights` is already implemented and disabled. Enabling it
should be a pre-registered decision, since it moves a parameter from `THEORY` to
`DEV` and increments the DSR charge.

---

## 6. Volume-clock sampling

**What.** `ActivityClock` already exists and is off by default. Sampling by equal
dollar volume makes returns closer to IID and normal.
**Why.** Every downstream statistical procedure — the variance-ratio asymptotics,
the NIG likelihood — assumes something closer to IID normality than clock time
delivers.
**Expected magnitude.** Small-to-moderate, and mostly on the *validity* of the
statistics rather than on returns.
**Cost.** Architectural: needs a finer feed, makes timestamps irregular, and
changes the meaning of "bars" in every window parameter.
**Validation.** Compare the variance-ratio statistic's null distribution under
both clocks; the volume clock should be closer to N(0,1).

---

## 7. A particle filter for the regime state

**What.** Replace the discrete four-state Bayes filter with a particle filter over
a continuous regime space.
**Why.** Only if the Beta measurement model proves too rigid — i.e. if H2 in the
pre-registration fails in a way suggesting the taxonomy, not the idea, is wrong.
**Mirage risk.** High. A continuous state space is far more expressive and far
easier to overfit; the current design's main virtue is that it has zero parameters
fitted to returns.
**Sequencing.** Do not attempt before H2 has been tested.

---

## 8. A learned regime measurement model

**What.** Estimate the Beta design table from data rather than specifying it.
**Why.** The 32 design cells are the largest set of unvalidated assertions in the
system.
**Mirage risk.** **Very high.** This converts the system's strongest anti-overfit
property — zero parameters fitted to returns in the regime layer — into 32 fitted
parameters, and it must be charged accordingly. Quantify before attempting: 32
parameters against ~400 effective observations is not a favourable ratio.

---

## 9. Options-implied information

Term structure, skew, and variance risk premium as regime and volatility inputs.
Real information content, but the data quality and cost problems that keep the
positioning engine disabled apply in weaker form. Behind a flag, never
load-bearing.

## 10. Execution improvements

Limit placement, participation scheduling, and opportunistic fills. Market-on-open
is the least sophisticated possible execution and there is measurable slippage to
recover. Requires modelling non-fill risk, which the barrier logic does not
currently do — and unfilled signals must be recorded, not dropped, or the record
biases toward easy fills.

## 11. Research infrastructure

Deterministic data snapshots keyed by manifest hash; a results database keyed to
the trials ledger; automated report generation. Raises the quality of every future
test rather than the performance of this one, which is why it belongs here rather
than at the top — but its value compounds.

---

## Things we deliberately will not do

**Deep learning on raw prices.** At an information coefficient near 0.04, a
high-capacity model has more than enough capacity to fit noise, and no amount of
regularisation makes an R² of 0.3% a good target for a network. The defensible
uses of learning here are meta-labelling and calibration, both of which are on the
roadmap above.

**Reinforcement learning for execution, without an impact-respecting simulator.**
RL needs an environment. A backtest that replays historical prices is not an
environment, because it does not respond to the agent's orders — so the agent
learns to exploit a simulator artifact. Building a market simulator that respects
impact is a large research project in its own right, and without it RL results are
not evidence.

**Higher-frequency operation.** `docs/15-COMPLEXITY.md` §7 explains why sub-minute
bars are a different system, not a tuning exercise: the rank windows lose meaning,
the cost model's parameterisation is wrong, and Corwin–Schultz becomes unusable.

**Adding engines for their own sake.** This is the one that looks most like
progress and is most harmful. Adding a tenth correlated engine *lowers* effective
breadth per engine while *raising* apparent agreement — measured, nine engines
already supply the independent evidence of 2.66. The correct move when the system
seems under-confident is to find a genuinely *uncorrelated* information source, or
to accept that the confidence is honest. `docs/01-THEORY.md` §6 is the argument;
`ebe_min` is the enforcement.
