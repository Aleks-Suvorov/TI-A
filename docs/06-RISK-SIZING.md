# Risk and Position Sizing

---

## 1. The gate is a risk decision, not a signal decision

Most systems generate a signal and then size it. TI-A does the opposite: the
question "is this worth risking anything on?" *is* the signal. The primary gate
is

```
EdgeBook.estimate(regime, setup, bucket).lcb  −  costs.round_trip_sigma  >  ev_lcb_min_sigma
```

— 90% posterior credence that expectancy net of modelled costs exceeds 0.05σ.
Selectivity is a consequence of this rule, not a threshold anyone chose, and it
tightens automatically in exactly the circumstances where one should trade less:
thin data (wide posterior), high outcome variance, or rising costs.

The observed behaviour under the corrected execution model (next-open fills,
cost-charged equity; 5,000 bars, seed 77, measured spread supplied):

| synthetic edge | trades | net/trade | t (effective n) | equity |
|---|---|---|---|---|
| 0.06 (realistic) | **0** | — | — | 1.0000 |
| 0.30 | 32 | +1.47σ | +4.22 | 1.66 |
| 0.80 | 19 | +2.30σ | +7.53 | 1.75 |

The first row is the design. At an edge strength comparable to what is actually
achievable, the system emits nothing — not because the edge is absent but
because it cannot be demonstrated, and the gate acts on the lower credible
bound rather than the point estimate.

The t-statistics are computed against `combined_effective_sample_size`, not the
candidate count — Kish's statistic alone is scale-invariant and blind to
uniform overlap; see `docs/09-ASSUMPTIONS.md` A6. Equity is net of the modelled
round trip charged at booking; see `docs/19-ADVERSARIAL-AUDIT.md` D3 for the
version of this table that was produced before costs were charged.

These numbers move with the sample. At 5,000 bars rather than 6,000 the same
seed gives t = 2.50 and 6.61 for the two lower rows. That instability across a
20% change in sample length is not a defect in the measurement — it is the
§12 power argument arriving in person.

---

## 2. Kelly under parameter uncertainty

For approximately Gaussian outcomes the growth-optimal fraction is `f* = μ/ς²`.
But `μ` is estimated, and plugging in the point estimate overbets systematically.
Integrating over the posterior of `μ` gives, to second order,

```
f*_Bayes ≈ μ_n / (ς_n² + Var(μ_n))
```

Parameter uncertainty enters as **additional variance** — a clean statement of
why an edge you are unsure about deserves a smaller bet for exactly the same
reason a volatile edge does. Full Kelly on an *estimated* edge is not aggressive,
it is a category error: Kelly is optimal for a known distribution.

Four further reductions stack on top, and the stacking is deliberate:

| Factor | Default | Rationale |
|---|---|---|
| fractional Kelly | `kelly_fraction = 0.25` | quarter-Kelly is the conventional compromise; full-Kelly drawdowns are intolerable for anyone who must keep trading through them |
| uncertainty haircut | `LCB/mean` | a wide posterior shrinks the bet even when its centre is attractive |
| data sufficiency | `clip(n_eff/30, 0.15, 1)` | makes the thin-cell consequence explicit rather than relying on the prior alone |
| drawdown throttle | `g(DD)` | §3 |

Then a hard cap at `max_risk_per_trade = 0.02`.

Conversion to units goes through the *stop distance*, so one loss costs
`risk_fraction` of equity regardless of the instrument's volatility:

```
stop_dist = |price · (1 − exp(−stop_sigma · sigma))|
units     = risk_fraction · equity / stop_dist
```

---

## 3. The drawdown throttle is deliberately asymmetric

`g(DD)` falls linearly from 1.0 at `dd_throttle_start = 4%` to
`dd_throttle_floor = 0.25` at `dd_throttle_stop = 15%`, then to zero at
`dd_kill = 25%`.

A drawdown is *weak* evidence that the model has degraded and *strong* evidence
that our estimate of the edge was too high. Reducing size is the correct response
to both — and, critically, it is the right response **without needing to know
which**. Any rule that requires diagnosing the cause before acting will act too
late.

---

## 4. Kill switches

Separate from throttles and not negotiable in the moment. Each has a
pre-committed threshold; tripping one stops new entries until a human reviews.

| Switch | Threshold | What a breach means |
|---|---|---|
| drawdown | `dd_kill = 25%` | outside the pre-registered distribution |
| calibration | `brier_alarm = 0.27` | the probability model is uninformative; every gate acts on a meaningless number |
| feature drift | `psi_alarm = 0.25` | live inputs no longer resemble the development distribution |
| execution | `slippage_alarm_mult = 2.0` | the cost model is wrong, so every expectancy is overstated |
| data staleness | 3 bars | acting on stale state |

A losing streak is checked against its **binomial tail** rather than a round
number: with `n` trades the expected longest run is roughly `log₂ n`, so a fixed
"five losses in a row" alarm fires constantly on a long sample and never on a
short one. The threshold scales as `⌈log₂ n⌉ + 3`.

The discipline these enforce is the point. `docs/08-FAILURE-MODES.md` argues the
most likely cause of this system's death is a human adjusting it after a
drawdown, and a switch that can be reasoned away during the drawdown is not a
switch.

---

## 5. Capacity

Impact enters the cost model as `η√q` with `q` the order size as a fraction of
bar dollar volume. Setting the round-trip impact equal to the edge and solving:

```
q_max = (edge_sigma / (2·η))²
```

At `η = 0.6` and an edge of 0.15σ, `q_max = 1.6%` of a bar's dollar volume. Above
that the strategy is trading against itself, and `CostModel.capacity_participation`
returns the bound so the policy can refuse the size rather than accept a worse
fill. **A system that will not refuse size does not have a capacity limit; it has
an undiscovered one.**

---

## 6. What is not modelled

Stated plainly, because each is a real exposure:

* **Portfolio correlation.** Sizing is per instrument. Running the system on
  twenty correlated instruments produces twenty positions that are not twenty
  independent bets, and nothing here prevents that. Correlation-aware portfolio
  construction is the largest missing risk component — see
  `docs/16-ROADMAP.md`.
* **Overnight and weekend gap risk** beyond what the stop-widening in
  `STRESS` provides. A gap through the stop fills at the open, which the
  backtester models pessimistically, but no position-size reduction anticipates
  it.
* **Funding, borrow and carry.** Not in the cost model.
* **Margin and liquidation mechanics.** The system reasons in equity fractions
  and assumes the account can hold the position.
