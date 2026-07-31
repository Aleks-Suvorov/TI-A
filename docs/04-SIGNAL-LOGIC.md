# Signal Generation Logic

How the system gets from nine engine opinions to one of five words.

---

## 1. The five words

```
FLAT  ──BUY───►  LONG        LONG  ──SELL───►  FLAT
FLAT  ──SHORT─►  SHORT       SHORT ──COVER──►  FLAT
                    anything ──NO_TRADE──► unchanged
```

`SELL` closes a long; it never opens a short. `COVER` closes a short; it never
opens a long. **`LONG → SHORT` in one bar is forbidden.** A reversal costs two
bars: `SELL`, then possibly `SHORT` on the next. This is both realistic — no
discretionary book flips instantly on one bar's close — and a whipsaw brake.

`NO_TRADE` is the overwhelmingly common output. On synthetic data, roughly 1% of
bars are even *structurally eligible* for consideration, before the expectancy
gate is applied. That is the design working, not a malfunction.

---

## 2. Timing

A decision formed at the close of bar `t` carries `decided_at_index = t` and
`execute_at_index = t + execution_lag_bars`, which is never less than `t + 1`.
Both indices are on the `Decision` object, and the backtester and the live
adapter honour the same two fields. There is exactly one notion of when an order
may be sent.

Consequently the entry fill is the **open of the next bar**, plus adverse
slippage. Nothing in the decision may read a price at or after
`execute_at_index`.

---

## 3. The gate — ten conjunctive conditions

An entry requires **all** of these. Conjunctive rather than scored, because a
weighted score lets a spectacular reading on one axis buy past a disqualifying
reading on another, and the disqualifying readings here do not trade off against
expectancy.

| # | Condition | Config key | Why |
|---|---|---|---|
| 1 | `EV_lcb − costs > ev_lcb_min_sigma` | `ev_lcb_min_sigma` | **the primary gate** |
| 2 | `p_success ≥ p_min` | `p_min` | a floor, not the main test |
| 3 | aligned `EBE ≥ ebe_min` | `ebe_min` | agreement must rest on independent evidence |
| 4 | `hazard ≤ regime_hazard_max` | `regime_hazard_max` | transitions are where strategies die |
| 5 | regime is not `QUIET` | — | no exploitable structure |
| 6 | features valid, engines mostly valid | — | no acting on a half-filled window |
| 7 | trade rate under the limiter | `max_trades_per_100_bars` | if it fires constantly, something broke |
| 8 | risk layer returns non-zero size | `risk/limits.py` | no kill switch tripped |
| 9 | position is `FLAT` | — | one position at a time |
| 10 | session phase tradeable, not an auction | `sessions.py` | auctions are a different mechanism |

**Condition 1 is the system.** Everything else is a filter on top of it. It reads:

```
EdgeBook.estimate(regime, setup, bucket).lcb  −  costs.round_trip_sigma  >  0.05
```

That is: *90% posterior credence that expectancy net of modelled costs is
positive by at least 0.05 sigma*. Selectivity is a consequence of this rule, not
a separate threshold. The rule tightens automatically when data is thin, when
outcome variance is high, and when costs rise — which are precisely the
circumstances in which one should trade less.

Every failed condition is recorded on the `ExplainCard` as a veto reason, so a
`NO_TRADE` is always answerable.

---

## 4. Setup classification

The bar is assigned to one of four named families, which is the Edge Book's
conditioning variable and therefore determines *which historical outcomes this
trade is being compared against*.

```
if liquidity engine has a confirmed sweep within 4 bars     → SWEEP_REVERSAL
elif compression > 0.7 and expansion ratio > 1.1            → COMPRESSION_BREAK
elif P(REVERT) > P(TREND)                                   → RANGE_FADE
else                                                        → CONTINUATION
```

Ordering is deliberate: the most specific configuration wins. A bar that is both
a sweep and a continuation is judged as a sweep.

Evidence buckets are fixed log-odds boundaries at `|L̂| ∈ {0.25, 0.6, 1.0, 1.6}`,
not empirical quantiles. Quantile boundaries would drift through time, silently
reassigning history to different cells and making the research and live paths
disagree.

---

## 5. Barriers

Set in sigma units, converted to price last. Base values `stop_sigma = 1.6`,
`target_sigma = 2.6`, `max_holding_bars = 30`, then adjusted:

* **Stress widens the stop** by up to 60% of the stress posterior. A
  jump-dominated tape produces gaps that a tight stop converts into a guaranteed
  adverse fill rather than into protection. The widening is paid for by sizing
  down, not by moving the target further away.
* **Jump share widens the stop** similarly above a 0.4 share.
* **Mean-reverting setups aim nearer and hold shorter**; trending setups the
  reverse. Both bounded, both driven by the posterior rather than a switch.
* **A confirmed structural level clips the target.** If a pivot sits between
  entry and target, the target moves to just short of it. Price may sail
  through, but assuming it will is not free.

The vertical barrier is not a convenience. After `max_holding_bars` the
conditioning information that justified the trade has decayed, and continuing to
hold is a different bet made silently.

---

## 6. Exits

Two mechanisms, deliberately separated.

**Resting orders.** Stop and target are placed at entry and fill intrabar. They
are checked against each bar's range with pessimistic tie-breaks: if the range
spans both barriers, the **stop** is recorded; if the bar opened beyond the stop,
the fill is the **open**, not the stop price. Bar data cannot resolve intrabar
sequence, so the ambiguity is always resolved against us. The optimistic
convention can manufacture most of a strategy's apparent edge, invisibly.

**Bar-close decisions.** `SELL` / `COVER` are emitted only for exits that
genuinely require a decision at a close:

* the vertical barrier is reached;
* a reversal signal survives the minimum hold — requiring `direction` to have
  flipped, `p_success ≥ p_exit`, and aligned breadth still above 70% of the entry
  gate. A reversal must be a positive statement about the other side, not merely
  a weakening of this one, or every position is closed by noise on the second bar;
* the session is closing.

Modelling a stop as a bar-close decision would be a look-ahead error in the
optimistic direction, since it silently assumes the close was available at the
stop price.

---

## 7. Worked example

A `BUY` on a sweep reversal, with the card the trader would see:

```
BUY  (long)
----------------------------------------------
Trend Alignment:                      31% long
Liquidity:                    Confirmed sweep
Momentum Expansion:                        58%
Volatility:                        Favourable
Regime:                        Mean-Reverting
Institutional Flow:                Accumulation
Evidence Breadth:              1.6 independent
Confidence:                    64%  [55%-73%]
Expected R:R:                        1.44 : 1
Expectancy (net):        +0.082s at 90% credence
Setup:                     Sweep Reversal (n=173)
Stop / Target:                 98.41 / 101.87
Risk:                        0.31% of equity
```

Note what this card does *not* claim. Confidence is 64%, not 97%, because
aligned breadth is 1.6 — the engines agree, but they are not independent, and the
MacKay correction pulls the number toward 50% accordingly. The trend engine reads
only 31% because the regime is mean-reverting and its reliability is discounted
there. The expectancy is stated at the *lower* credible bound, and the cell it
came from is named along with its effective sample size, so a reader can see the
estimate rests on 173 observations rather than 5.

A `NO_TRADE` card carries up to three veto reasons, e.g.:

```
NO TRADE
----------------------------------------------
...
  stood aside: effective breadth 0.94 below 1.25
  stood aside: regime transition hazard 0.41 > 0.35
```
