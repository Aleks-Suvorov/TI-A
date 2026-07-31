# The Engines

Nine engines, each a streaming estimator emitting a direction, a
trustworthiness, and an explanation. This document is the reference for what
each measures, what it deliberately does not, and how it fails.

Read `docs/01-THEORY.md` §5 and §6 first; the reliability channel and the regime
routing only make sense against that background.

---

## 0. The contract, restated

```python
EngineOutput(
    name: str,
    score: float,         # [-1, +1]; sign = direction, magnitude = strength
    reliability: float,   # [0, 1]; "should you listen to me right now"
    features: dict,       # human-facing, for the explain card
    diagnostics: dict,    # machine-facing, for monitoring
    notes: tuple[str],    # short lowercase phrases
    valid: bool,
)
```

`score` and `reliability` answer different questions and must never be merged.
`score = 0.0` with `reliability = 0.8` is a *confident statement that there is no
directional evidence*, which is information: it contributes nothing to the pooled
log-odds while adding to its variance, correctly widening the interval. An engine
that abstained instead (`reliability = 0`) would leave the pool falsely narrow.

Three properties are tested for every engine (`tests/test_engines.py`,
`tests/test_causality.py`): determinism, prefix consistency, and graceful
degradation when optional inputs are missing.

---

## 1. Regime engine — `engines/regime.py`

**Measures.** A posterior over four states, plus a transition hazard.

**The design table.** Each cell is a statement about what the regime *means*.
`—` denotes "this regime makes no claim about this feature", encoded as
`Beta(1,1)`, which is exactly uniform and contributes a likelihood factor of
exactly 1. Values are the Beta mean `m`; concentration is
`regime_beta_concentration = 6` everywhere else.

| Feature | TREND | REVERT | STRESS | QUIET |
|---|---|---|---|---|
| `vol_rank` | — | — | 0.85 | 0.15 |
| `vr_rank` (autocorrelation) | **0.80** | **0.20** | — | — |
| `ladr_rank` (cheap displacement) | 0.80 | 0.35 | 0.70 | 0.30 |
| `absorption_rank` | 0.20 | 0.80 | 0.30 | — |
| `participation_rank` | — | — | 0.80 | 0.20 |
| `entropy` | 0.35 | 0.65 | — | 0.70 |
| `jump_share` | — | 0.35 | 0.80 | 0.20 |
| `efficiency` | 0.80 | 0.20 | — | 0.30 |

The bolded row is the axis that matters. TREND and REVERT are separated
primarily by the sign of return autocorrelation, because that is what decides
whether momentum or liquidity provision is the paid behaviour. STRESS
deliberately makes no claim about autocorrelation or efficiency: in stress those
flip fast and any claim would be wrong half the time.

**Parameters fitted to returns: zero.** The concentration is a `THEORY`
parameter; the stickiness (`regime_stickiness = 0.985`, implying a mean dwell of
about 67 bars) is estimated once on the development universe and frozen.

**Tempering.** The likelihood is raised to `η = EBE_features / J`, capped at 1.
On the actual feature set this sits at the cap, because the eight features
include near-reciprocal pairs (LADR and absorption) and are therefore not
redundant in the relevant sense. Fed eight copies of one feature, `η` falls below
0.4 — the mechanism works and simply is not needed here. That is a finding about
the feature design, not a dormant component.

**Hazard.** `max(posterior drift, ambiguity)`, where drift is total variation
against an EWMA of past posteriors and ambiguity is `1 − max_k α(k)`. Taking the
max means either alone suffices to stand aside, which is the conservative
reading. Entries require `hazard ≤ 0.35`, i.e. a dominant regime held with at
least 65% mass and not currently in motion.

**Score.** Deliberately small. Knowing the regime says which evidence to trust,
not which way to trade. What it contributes is routing and the hazard veto.

**Fails when.** The four-state taxonomy does not carve the space where the
continuation/reversal distinction actually lies; the rank features saturate in a
tail event; a transition completes faster than the sticky filter can follow.

---

## 2. Trend engine — `engines/trend.py`

**Measures.** Direction from the Kalman local-linear-trend slope t-statistic,
multi-horizon sign agreement, and the higher-timeframe slope. Trust from the
Lo–MacKinlay variance-ratio statistic, path efficiency, and the regime posterior.

**Score** = weighted blend, 0.45 execution-timeframe slope (via `tanh(t/2)`),
0.25 sign agreement, 0.30 higher-timeframe slope.

**Reliability** multiplies four factors: the trend-regime posterior, a
persistence factor `clip(0.5 + z/4, 0, 1)` from the variance ratio (0.5 at the
martingale null — neither trusting nor distrusting), path efficiency, and a
timeframe-agreement factor that falls to 0.45 when execution and higher
timeframes disagree. Disagreement is not a reason to trade the other way; it is
a reason to be less sure.

**Deliberately not measured.** Any moving-average crossover, any fixed-period
oscillator. The output is a significance, not a level, precisely so that no
threshold encodes an unexamined assumption about volatility.

**Fails when.** The drift is real but below the filter's local resolution — at
0.25 sigma per bar against noise of 1 sigma, the filter honestly reports |t| ≈ 1
with an arbitrary sign, which is correct behaviour and a genuine limitation.

---

## 3. Momentum engine — `engines/momentum.py`

**Measures.** Displacement that was *cheap in liquidity terms*: the LADR rank,
its short EMA, and the smoothed close location value.

The claim is narrower and better supported than "price went up so it will
continue". It is that price moved far while consuming little liquidity, which
under any Kyle-type model is the signature of informed flow, and informed flow
has permanent impact. Only the upper half of the LADR distribution counts as
evidence: `strength = max(0, 2·ladr_rank − 1)`. A bar at the median cost of
displacement says nothing, and maps to zero rather than to a weak opinion.

**Reliability** rises with strength and the trend posterior, falls with the
revert and quiet posteriors, and is penalised by jump share. Jumps are
displacement without a tradeable path: real information, but our stop cannot be
assumed to have survived them.

**Fails when.** Volume and consumed liquidity decouple — cross-venue sweeps,
auction prints, index rebalances. `participation` is total volume, not signed
order flow, and this is the engine most exposed to that limitation.

---

## 4. Liquidity engine — `engines/liquidity.py`

**Measures.** Two distinct things.

*Sweep-and-fail*: price penetrates a confirmed pivot by more than 0.25 sigma, on
participation in the upper quartile, with the bar giving back its excursion, and
then a later closed bar reclaims the level. Scores in the **reclaim** direction,
decaying over four bars. This is the measurable core of what is usually called a
stop run, and its economic content is compensated liquidity provision.

*Absorption*: `absorption_rank > 0.65` while price sits near a range extreme.
Scores toward the middle of the range. This is "effort versus result" made
computable — a statement about volume and price, not a claim about anyone's
intent.

**Reliability** rises with the revert and stress posteriors and falls with the
trend posterior. In stress the reversal is better paid *and* far more dangerous;
the engine keeps its opinion and the risk layer cuts the size.

**Deliberately not measured.** Order blocks, breaker blocks, mitigation blocks,
premium/discount arrays, and the rest of the vocabulary — see
`docs/17-EVIDENCE-REVIEW.md`. Two components of that tradition survive because
they have measurable analogues with supporting literature; the rest do not.

**The latency is real.** A sweep is knowable only after it has failed. Systems
that mark sweeps at penetration are reading the future, and the several-bar
delay here is the actual information arrival time.

---

## 5. Volatility engine — `engines/volatility.py`

**Measures.** Whether conditions favour taking a position at all: volatility
rank inside a favourable band, expansion ratio, compression, volatility of
volatility.

**Usually scores near zero with confident reliability.** Volatility is the most
forecastable quantity available and is almost entirely directionless; a
compression that resolves into an expansion has, ex ante, no reliable direction.
An indicator that assigns one is manufacturing a signal from a real observation.

Where it does contribute direction is narrow: an expansion already under way,
whose displacement is being held, is weak confirming evidence, capped at ±0.4.

**Fails when.** Volatility of volatility is high, which means the forecast that
sets every barrier is itself unreliable — handled by an explicit penalty, but
only partially.

---

## 6. Structure engine — `engines/structure.py`

**Measures.** Position inside the last confirmed range, higher-timeframe range
position, and the confirmed higher-high/higher-low bias.

**Regime-dependent by construction.** The same location is bullish in one regime
and bearish in the other: near the top of a range, a trending market is breaking
out and a mean-reverting one is failing. Rather than pick, the engine computes
both readings and weights by the regime posterior. When the posterior is split,
the readings cancel and the reliability falls — which is correct, because in
that state structure genuinely does not say anything.

**Staleness.** A range whose extremes were confirmed eighty bars ago is a weaker
description of the present, and reliability decays accordingly.

**Every level is confirmed.** Nothing here can repaint, at the cost of a
`pivot_confirm_bars` delay.

---

## 7. Behavioral engine — `engines/behavioral.py`

**Measures.** Session phase trust, event proximity, and session-open effects.

**Mostly a statement about when *not* to act.** What survives replication in the
calendar literature is the volume and volatility structure — the intraday
U-shape, the midday trough — not the directional content, which is weaker and
has decayed since publication. So the engine's main output is reliability:
auctions and pre/post-session get exactly 0, midday 0.70, morning 0.95.

Auctions get zero rather than a low number because they are a *different price
formation mechanism*: every microstructure feature in the system assumes
continuous trading and does not describe a call auction at all.

The one directional tilt retained is midday mean reversion, capped at ±0.3.

**Event proximity** requires a calendar. Absent one, the adjustment is simply
not applied — the engine does not assume the calendar is clear.

---

## 8. Cross-asset engine — `engines/crossasset.py`

**Measures.** Whether the instrument is moving with or against the peers it
usually moves with, and whether the broad volatility environment is
deteriorating.

The informative case is moving *against* the usual peers: an idiosyncratic move
makes a mechanical or liquidity-driven explanation more likely than an
informational one, and mechanical moves revert. Implied-volatility shocks enter
as a reliability discount, never as a score — a VIX spike is not directional for
an arbitrary instrument.

**Optional.** With no `ExogenousSnapshot` data it reports `reliability = 0` and
drops out of the pool. Capped at 0.65 reliability even when fully fed.

**What it is not.** Genuine cross-sectional information is mostly about *ranking*
instruments against each other, which is a portfolio problem this
single-instrument system does not solve. See `docs/16-ROADMAP.md`, where it is
argued to be the largest available improvement.

---

## 9. Positioning engine — `engines/positioning.py`

**Measures.** Dealer gamma imbalance and open-interest change, when supplied.

**Disabled by default, and this is the honest part.** The mechanism is real and
documented: a dealer short gamma must hedge with the move, amplifying it. The
*data* is not. Aggregate dealer positioning is not published, and every widely
available "gamma exposure" figure is a reconstruction resting on unverifiable
assumptions about who is on which side of open interest. The reconstruction error
is unknown, probably large, and correlated with exactly the conditions in which
one would want the signal.

So the engine exists, is correct as far as it goes, is capped at 0.45
reliability, and `enable_positioning_engine` defaults to `False`. `SPEC.md` §9
forbids any optional module from appearing in the gate's conjunctive conditions,
so it can never become load-bearing.

---

## 10. Edge Book — `engines/edgebook.py`

Not an engine. It consumes fused evidence rather than voting alongside the
engines, has no score, and is never pooled. Documented here because `SPEC.md`
lists it in this directory.

**Cells.** `(regime, setup family, evidence bucket)`, where setup families are
four named configurations:

| Family | Fires when |
|---|---|
| `SWEEP_REVERSAL` | the liquidity engine has a confirmed sweep within four bars |
| `COMPRESSION_BREAK` | compression above 0.7 with the expansion ratio above 1.1 |
| `RANGE_FADE` | the revert posterior exceeds the trend posterior |
| `CONTINUATION` | otherwise |

Ordering matters: a bar that is both a sweep and a continuation is judged as a
sweep, the more specific configuration.

**Model.** Normal-Inverse-Gamma on the realised outcome return in sigma units,
signed by direction. Modelling the *return* rather than a win/loss label handles
target, stop and timeout uniformly; a win-rate model has to special-case
timeouts, which are common and are neither wins nor losses.

**Shrinkage is leave-one-out.** A cell's prior comes from its parent's evidence
*excluding the cell's own*. Without that exclusion the same observations set the
prior and then update against it: five observations of +2 sigma produced a
posterior mean of 0.77 instead of 0.33 and a bound that cleared a gate it had no
business clearing. This was a real bug, caught by
`tests/test_engines.py::test_edge_book_shrinks_a_thin_cell_toward_zero`.

**Output.** Posterior mean, closed-form Student-t lower credible bound at the
10th percentile, outcome variance, and effective observation count. The gate acts
on the bound.

---

## 11. Measured engine correlation

From `CALOP.diagnostics()` on 3,000 synthetic bars:

```
mean |correlation|          0.297
effective breadth at full
  reliability               2.66   (of 9 engines)
```

Nine engines supply the independent evidence of fewer than three. This is not a
defect to be fixed by adding engines — adding a tenth correlated engine *lowers*
effective breadth per engine while appearing to raise confidence. It is the
quantity `ebe_min = 1.25` is calibrated against, and it is the single most
important number to watch in production: rising correlation means falling
breadth, which means the system is becoming overconfident.
