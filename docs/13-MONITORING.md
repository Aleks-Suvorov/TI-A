# Monitoring

A selective system cannot be judged by P&L on any timescale a human finds
satisfying. So what is monitored is the *model*, which has far more statistical
power per unit time than returns do.

Implemented in `python/src/tia/monitoring/drift.py`.

---

## Tier 1 — model health (every bar)

| Metric | Window | Alarm | Meaning of a breach | Response |
|---|---|---|---|---|
| feature validity / NaN | 1 bar | any | kernel broken or data stale | suppress |
| data staleness | 1 bar | 3 bars | acting on stale state | kill switch |
| regime ambiguity `1 − max α` | 1 bar | `> 0.35` | no committed regime | veto entries |
| regime transition hazard | 1 bar | `regime_hazard_max = 0.35` | regime in motion — where strategies die | veto entries |
| aligned effective breadth | per signal | `ebe_min = 1.25` | agreement rests on too few independent engines | veto entry |
| rolling Brier | 100 outcomes | `brier_alarm = 0.27` | 0.25 is a coin flip; every gate now acts on a meaningless number | demote to NO TRADE |
| Brier **resolution** | 100 outcomes | falling trend | losing discriminating power — worse than miscalibration and *not* fixable by recalibrating | investigate |
| expected calibration error | 100 outcomes | `> 0.05` | stated probabilities systematically off | refit calibrator |
| feature PSI vs development | 500 bars | `psi_alarm = 0.25` | live inputs no longer resemble the development distribution | restrict |

The PSI reference is **frozen** once `reference_window` observations exist.
Letting it keep updating would make drift undetectable by construction — the
comparison would always be against a distribution that had already absorbed the
drift.

---

## Tier 2 — execution quality (every trade)

| Metric | Alarm | Why |
|---|---|---|
| realised ÷ modelled slippage | `slippage_alarm_mult = 2.0` | the cost model is wrong, so every expectancy in the gate is overstated |
| effective spread paid vs Corwin–Schultz estimate | 2× | validates the estimator that stands in for a quote feed |
| fill rate | < 95% | limit orders are not filling; the record is biased toward easy fills |
| post-fill drift against the position | persistent | adverse selection: we are the liquidity |
| participation achieved | > `capacity_participation` | trading above our own capacity bound |

---

## Tier 3 — strategy performance (rolling window of trades)

Reported **always with standard errors**, because the whole argument of
`docs/01-THEORY.md` §12 is that the trade count is too small for a point estimate
to stand alone.

* expectancy in σ units, with standard error and t-statistic
* hit rate with a binomial (Wilson) interval
* per-regime breakdown — the falsifiable H2 prediction
* drawdown and its duration
* trade frequency vs the expected rate (a *rise* is as concerning as a fall:
  `max_trades_per_100_bars` is a brake, and hitting it means something broke)
* realised holding times vs the vertical barrier — if most trades time out, the
  barriers are mis-set relative to the horizon the edge actually lives on

---

## Tier 4 — decay detection (monthly / quarterly)

* Live expectancy vs the walk-forward expectation, by a **sequential** test
  (SPRT or CUSUM against the pre-registered null) rather than a repeated t-test.
  Repeatedly t-testing an accumulating sample is a multiple-testing violation
  that will find decay that is not there.
* Edge Book cell drift: posterior means moving materially between refreshes.
* **Engine correlation drift.** Rising `mean_abs_corr` means falling effective
  breadth, which means growing overconfidence. This is the leading indicator for
  the severe failure mode in `docs/08-FAILURE-MODES.md` §2.1, and it is the single
  most important number on this page.

---

## The auto-demotion ladder

Implemented as `DemotionLevel` in `monitoring/drift.py`. Degradation is gradual,
and a monitor whose only action is "halt" gets overridden long before it fires.

| Level | Trigger | Action |
|---|---|---|
| `NORMAL` | — | continue |
| `TIGHTENED` | Brier within 10% of alarm | raise `ev_lcb_min_sigma` 50%, halve size |
| `RESTRICTED` | PSI breach, or effective breadth below `ebe_min` at full reliability | trade only the dominant regime at full confidence |
| `NO_TRADE` | Brier above alarm, negative Brier skill, or slippage above `slippage_alarm_mult` | suppress signals; keep computing and logging |
| `HALTED` | `dd_kill`, or manual | stop; human review required to resume |

Demotion is automatic. **Promotion is not** — returning to `NORMAL` requires a
human decision recorded in the change log.

---

## What NOT to monitor

* **Short-run P&L.** At ~1% eligibility a month may contain no trades. Reacting
  to a month is reacting to noise.
* **A drawdown inside the pre-registered distribution.** Here is the arithmetic
  that should be printed and kept: with a per-trade expectancy of 0.15σ, a
  per-trade standard deviation of 1.5σ, and 400 trades, the expected maximum
  drawdown is roughly `0.5 · σ_trade · √(2 ln n) ≈ 2.7σ` of cumulative trade
  units — several times the *mean* trade outcome and entirely unremarkable. A
  drawdown of that size is not information.
* **Individual losing trades.** The system's stated hit rate is well under
  certainty by construction; a stop-out is the system working.
* **Any metric on fewer than 30 outcomes.** The interval is wider than the effect.

The failure mode this section exists to prevent is the one
`docs/08-FAILURE-MODES.md` ranks first: a human adjusting the system after a
drawdown that was inside its own predicted distribution.
