# 17 — Evidence Review

An audit of every concept a designer of a system like TI-A is tempted to reach
for, and a keep/discard decision for each. The document exists to be *used
against* the system: if a component is not defensible here, it does not enter
`features/`, `engines/`, or `decision/`.

The bibliography supporting every claim is `docs/18-LITERATURE.md`. Where this
document says "grade A" it means the literature entry exists and says what is
claimed. Where a specific citation could not be verified it is marked
`[recollection — verify]` in 18 and treated as one grade lower here.

An audit that concluded "everything works" would be worthless. Of the 48
concepts below, **9 are CORE, 12 CONFIRMING, 8 OPTIONAL and 19 DISCARDED.**
The single most important conclusion is in §12: almost all of the reliable
statistical structure in financial price data lives in the second moment, not
the first, and the surviving directional edge is small enough that the system's
trade rate, barrier geometry and universe size are determined by statistical
power rather than by opportunity.

---

## 1. Grading scale

The same five-level scale is used throughout. It grades **the evidence for the
claim as stated**, not the plausibility of the mechanism and not our
implementation.

| Grade | Meaning |
|---|---|
| **A** | Replicated across multiple markets and multiple decades in peer-reviewed work by independent authors. For directional claims, the effect survives realistic transaction costs. For measurement, risk and cost claims, the estimator remains accurate out of sample. |
| **B** | Solid published evidence, but the effect is fragile, shrinking over time, concentrated in a subsample, or only marginally survives costs. Real, but not something to bet the architecture on. |
| **C** | Mixed or actively contested. Published results exist on both sides, or the positive results do not survive a data-snooping correction. |
| **D** | Suggestive only. No clean test exists — usually because the concept has never been stated precisely enough to have a null distribution, or because the required data has never been assembled. |
| **F** | Unfalsifiable as usually stated, or tested and refuted. |

Two conventions:

* **Split grades** (`A (L2) / C (OHLCV)`) are used where the grade genuinely
  depends on the data feed. This is not hedging — it is the single most
  important distinction in this document, because TI-A's core must run on
  OHLCV, and several concepts that are grade A on order-book data are grade C
  or worse once reconstructed from bars.
* **"Survives costs"** is a real bar. Several effects that are grade A as
  *statistical facts about returns* are downgraded because the published
  effect size is smaller than the round-trip friction a retail-to-mid-tier
  participant pays. Where that is the reason for a downgrade it is stated.

Verdicts:

| Verdict | Contract |
|---|---|
| **CORE** | Load-bearing. May originate a signal or set a barrier/size. Failure of a CORE component is a system failure. |
| **CONFIRMING** | May modulate `reliability` or contribute log-odds *once a CORE component has already produced a direction*. May never originate a direction on its own. Enforced by construction: a CONFIRMING engine that fires alone cannot reach `ebe_min`. |
| **OPTIONAL** | Behind a config flag, **default off**, with the data deficiency written into `docs/09-ASSUMPTIONS.md`. Never appears in conditions 1–10 of SPEC §7. |
| **DISCARDED** | Not implemented. The reason is recorded so the decision does not get relitigated every six months. |

---

## 2. Price formation and microstructure

### 2.1 Market microstructure theory

**Claim.** Prices are not exogenous draws from a distribution; they are the
output of a mechanism in which informed and uninformed traders interact through
a specific protocol, and the properties of that protocol (spread, depth, impact,
tick size, priority rules) determine measurable regularities in the price series.

**Mechanism.** Yes, and it is the best-founded mechanism in this document.
Kyle (1985) and Glosten–Milgrom (1985) derive spread and impact from adverse
selection under explicit strategic assumptions; the predictions (impact is
increasing in order size and in the probability of informed trading, spread
compensates for adverse selection) are direct consequences, not fitted
relationships.

**Grade.** A for the theory and for its impact predictions; the empirical
literature is thirty years deep and cross-market.

**Data.** The theory is feed-agnostic. Its *estimators* differ: Kyle's λ and
PIN want tick data, but Roll (1984), Corwin–Schultz (2012) and Hasbrouck's
(2009) Gibbs estimator recover effective spread from daily OHLC with reported
correlations to tick-based benchmarks near 0.9.

**Verdict.** **CORE** — but as a source of *estimators and cost discipline*,
not as a signal. Microstructure theory tells us how much we pay and how to
normalise displacement by participation. It does not tell us which way to bet.
Nothing in `engines/` should claim otherwise.

**Proxy.** `amihud = |r| / dollar_volume`; `kyle_lambda` as a regression-free
impact proxy; Corwin–Schultz effective spread over `spread_estimator_window`;
`participation = dollar_volume / rolling_median(dollar_volume)`.

### 2.2 Order flow theory

**Claim.** Signed order flow, not price history, is the proximate cause of
price change. Aggregate net buying pressure predicts the next price increment
with far higher R² than any transformation of past prices.

**Mechanism.** Yes. Cont, Kukanov and Stoikov (2014) show order-flow imbalance
at the best quotes explains contemporaneous mid-price moves with a roughly
linear coefficient scaled by depth; Chordia, Roll and Subrahmanyam (2002) show
the same at daily aggregation. Bouchaud, Farmer and Lillo's propagator
framework gives the decay structure.

**Grade.** **A (tick/L2) / C (OHLCV).** The high grade is for *contemporaneous
explanation*, which is not the same as prediction: OFI's explanatory R² on the
same-interval return is large; its predictive R² on the next interval is small
and decays within seconds to minutes. Reconstructing order flow from bars —
close-location value, up/down volume splits, "buying pressure" indicators — is
a lossy inversion with no published validation of comparable quality.

**Data.** L2 quote-update stream for the real thing. OHLCV gives only a coarse
proxy.

**Verdict.** **CONFIRMING.** The OHLCV proxy (`clv_ema`, `absorption`) is
allowed to modulate confidence. It may not originate a signal, because the one
thing we know about the proxy is that it discards the information that made the
original grade A.

**Proxy.** `clv_ema` = EWMA of close-location value `(2C − H − L)/(H − L)`, a
scale-free bar-level buying-pressure estimate; `absorption` =
`participation / |displacement|`.

### 2.3 Auction market theory / Market Profile

**Claim.** The market is a two-sided auction seeking "value"; price rotates
around a developing value area, and trades initiated away from value tend to
return to it. The Market Profile / TPO distribution makes value observable.

**Mechanism.** Partly. The *academic* auction literature — Madhavan (1992) on
call versus continuous mechanisms, Amihud–Mendelson (1987) on open versus close
return variance, Biais, Hillion and Spatt (1999) on pre-opening price
discovery, Barclay–Hendershott (2003) on after-hours — establishes real and
large heterogeneity in price-discovery efficiency *across session phases*.
That is a genuine finding. It does not establish that a volume-at-price
histogram identifies a mean-reverting attractor.

**Grade.** **A** for session-phase heterogeneity in price discovery;
**C** for Market Profile as a trading framework. Steidlmayer's construction has
no published out-of-sample test with a null distribution; the "value area"
boundaries (one standard deviation of the TPO distribution) are conventions,
and the framework's core prediction ("price returns to value") is not
distinguishable from mean reversion measured any other way.

**Data.** Tick data for a real TPO/volume distribution. OHLCV supports only a
degenerate surrogate.

**Verdict.** **DISCARDED as a trading framework.** The surviving content —
session phases differ, and auction periods are not tradeable — is kept under
its own name in §11.3 and is already enforced by SPEC §7 condition 10.
Retaining Market Profile alongside it would double-count.

**Proxy.** None retained. The session-phase content is `SessionPhase` tagging in
`data/sessions.py`.

### 2.4 "Liquidity engineering" / engineered liquidity, inducements

**Claim.** Large participants deliberately manufacture price movement toward
regions of resting orders in order to source liquidity for their own position,
so those regions are predictable destinations.

**Mechanism.** Two very different claims are bundled here, and they must be
separated.

1. *Resting orders cluster at identifiable prices.* This is **true and
   documented.** Osler (2003) shows currency take-profit orders cluster at
   round numbers and stop-loss orders cluster just beyond them; Kavajecz and
   Odders-White (2004) show support and resistance levels coincide with depth
   peaks in the limit order book. Grade **B**, arguably A.
2. *An identifiable agent intentionally engineers moves to those clusters.*
   This is unfalsifiable from price data. Intent is not observable, and the
   observable consequence is identical to a mechanical cascade with no author.
   Grade **F**.

**Grade.** **B** (order clustering) / **F** (the engineering narrative).

**Data.** Claim 1's *evidence* came from proprietary order-book data. Its
*consequence* — that prior extremes and round numbers are where cascades start
and stop — is observable in OHLCV.

**Verdict.** **DISCARDED as a named concept.** Its one testable component is
level clustering, which is already carried by confirmed pivots (`pivot_high`,
`pivot_low`) and consumed by the sweep detector in §10.1. Keeping "liquidity
engineering" as a separate concept would add vocabulary without adding a
measurable quantity, and would import the unfalsifiable half.

**Proxy.** None additional. Absorbed into confirmed-pivot proximity and the
sweep-reversal feature.

### 2.5 Hidden liquidity

**Claim.** A large fraction of available size is not displayed — icebergs,
reserve orders, dark venues — so displayed depth understates true liquidity,
and inferring hidden size is informative.

**Mechanism.** Yes. Bessembinder, Panayides and Venkataraman (2009) find hidden
orders are 44% of order volume on Euronext-Paris and document the exposure
trade-off (hiding lowers implementation shortfall but reduces fill probability).
Zhu (2014) and Comerton-Forde and Putniņš (2015) analyse dark trading's effect
on price discovery.

**Grade.** **B.** Well documented that it exists and is large; much thinner on
whether *inferred* hidden liquidity predicts returns.

**Data.** By construction not observable in any feed, including L3. The only
observable is the *aftermath*: large volume that produced small displacement
implies size was resting.

**Verdict.** **CONFIRMING** via the aftermath measurement only. Direct
detection is **DISCARDED** — it is unobservable by definition, and any indicator
claiming to show it is measuring something else.

**Proxy.** `absorption = participation / |ret_norm|`, ranked causally. High
absorption rank with a close rejected back into range is the measurable
signature of size having been absorbed.

### 2.6 Adverse selection

**Claim.** A passive counterparty systematically loses to informed flow, so the
price at which one transacts is conditionally worse than the price one observes,
and the conditional penalty grows with the information content of one's own
reason for trading.

**Mechanism.** Yes — this is Glosten–Milgrom (1985) and Kyle (1985) directly,
with Easley, Kiefer, O'Hara and Paperman (1996) providing the PIN estimator.

**Grade.** **A.**

**Data.** Theory is feed-agnostic; the implication for us is a cost-model
constraint, not a measurement.

**Verdict.** **CORE — in the cost model, not as a signal.** This is the single
most under-appreciated item in the list for a system designer, and it is why
SPEC §2.7 mandates pessimistic tie-breaks and why `cost_slippage_range_frac`
is deliberately punitive. The trades TI-A most wants to take are precisely the
trades where the resting side has the most reason to step away: a signal that
fires on displacement plus participation is, by construction, correlated with
informed flow. Optimistic fill assumptions in a backtest of such a signal are
not a small error; they are the whole result.

**Proxy.** Not a feature. Enforced as: stop assumed hit first when both barriers
fall in one bar; gapped stops filled at the worse of stop and open; slippage
charged as `cost_slippage_range_frac` of the execution bar's range; realised
versus modelled slippage tracked in `monitoring/drift.py` with
`slippage_alarm_mult`.

### 2.7 Imbalance detection

**Claim.** Local excesses of buying over selling interest are detectable and
predict short-horizon continuation.

**Mechanism.** Yes at the order level (see §2.2). At the bar level the
mechanism is weaker: a bar's imbalance is already impounded in its own close, so
what remains to predict is only the *unfinished* portion of a metaorder.

**Grade.** **A (L2 order-flow imbalance, contemporaneous) / C (bar-level
proxies, predictive).** The bar-level version is the classic
"up-volume/down-volume" family, which has no published test surviving a
data-snooping correction.

**Data.** L2 for the real thing; OHLCV for proxies; footprint for signed volume.

**Verdict.** **CONFIRMING** through `clv_ema` and `absorption`. **OPTIONAL** for
true OFI if an L2 feed is ever attached (`enable_delta_features` and the
`Bar.delta` field are the hooks).

**Proxy.** `clv_ema`; `run_asym` (asymmetry of up-run versus down-run lengths,
which is a nonparametric imbalance statistic with a computable null under an
IID-sign hypothesis).

---

## 3. Execution and transaction cost

### 3.1 Institutional execution and metaorder footprints

**Claim.** Institutions cannot trade their size at once, so they split
metaorders over hours or days. The resulting serially correlated participation
leaves a detectable footprint, and a detected footprint predicts continuation
until the metaorder completes.

**Mechanism.** Yes, and strongly. Almgren–Chriss (2000) and Bertsimas–Lo (1998)
derive the optimal schedule; Lillo, Farmer and Mantegna (2003), Almgren, Thum,
Hauptmann and Li (2005) and Tóth et al. (2011) establish the square-root impact
law empirically across markets; Bouchaud, Farmer and Lillo (2009) explain the
long autocorrelation of order signs as the shadow of order splitting.

**Grade.** **B.** The impact law and the order-splitting fact are grade A. The
*inference* — "I can identify from bars that a metaorder is in progress and it
has not finished" — is grade B at best. Every bar with high participation is a
candidate; most are not metaorders; and by the time participation is
statistically anomalous, most of the impact has been paid.

**Data.** The strong evidence comes from proprietary broker metaorder databases.
OHLCV gives participation and its persistence.

**Verdict.** **CONFIRMING.** Persistence of elevated participation may raise
confidence in a direction already established. It may not originate one,
because the base rate of "high participation" that is *not* an unfinished
metaorder is high.

**Proxy.** `participation` and its rank; autocorrelation of
`sign(ret) × participation` over a short window; `impact_exponent = 0.5`
normalisation of displacement by participation, which is what makes `ladr`
comparable across volume regimes.

### 3.2 VWAP behaviour

**Claim.** Because institutions are benchmarked against VWAP, flow is
mechanically attracted to VWAP, so price mean-reverts to session VWAP and
anchored VWAP acts as support and resistance.

**Mechanism.** The first half is real: VWAP became the standard institutional
execution benchmark after Berkowitz, Logue and Noser (1988), and
benchmark-tracking flow does exist. The second half does not follow. A trader
minimising tracking error against VWAP trades *proportionally to volume*, which
is a schedule, not a price-reverting force. If anything, VWAP-tracking flow is
*price-taking* and therefore adds to impact rather than damping it.

**Grade.** **C** for "price reverts to session VWAP"; **D** for anchored VWAP as
support/resistance. There is no peer-reviewed test of anchored VWAP as a
predictive level that we could locate, and the construction has a free parameter
(the anchor) chosen after the fact, which is fatal.

**Data.** True VWAP needs intrabar volume-weighted prices. From OHLCV the best
available is a typical-price-weighted running average, which is not VWAP.

**Verdict.** **DISCARDED as a signal.** **CORE as an execution benchmark** in
`decision/costs.py` and in slippage monitoring — which is what VWAP was invented
for. The distance-from-VWAP variable is additionally near-collinear with
`range_pos` and `clv_ema`, so including it would inflate apparent confluence
while lowering effective breadth.

**Proxy.** None as a signal. As a benchmark: realised fill versus the execution
bar's volume-weighted typical price, logged for `slippage_alarm_mult`.

### 3.3 Execution quality

**Claim.** Realised performance is dominated by the difference between the
decision price and the achieved price, and that difference is measurable,
attributable and controllable.

**Mechanism.** Yes. Perold (1988) defines implementation shortfall; Kissell and
Glantz (2003) give the practitioner framework; the impact-law literature gives
the functional form.

**Grade.** **A.**

**Data.** OHLCV suffices for a conservative model; live fills for validation.

**Verdict.** **CORE.** With the gross edges established in §12, cost modelling
error of a few basis points is the difference between a positive and a negative
expectancy. A backtest that models costs less pessimistically than reality is
not a weak backtest; it is a different strategy.

**Proxy.** `CostEstimate` = half-spread (Corwin–Schultz) + impact
(`cost_impact_eta × participation^0.5`) + slippage
(`cost_slippage_range_frac × range`) + fees, all expressed in σ units so they
are directly comparable to the barrier geometry.

---

## 4. Volume-derived inference

### 4.1 Volume profile / volume at price

**Claim.** Prices at which large volume previously traded are levels of
significance: they attract price, and they act as support and resistance
because participants have positions and reference points there.

**Mechanism.** Partially supported, but not by the volume-profile literature —
by Kavajecz and Odders-White (2004), who show that levels technical analysts
identify coincide with depth peaks in the limit order book. That gives a real
microstructural reason why *some* levels matter. It does not privilege
volume-weighted price histograms over other level-identification methods, and
the two need not agree.

**Grade.** **C.** The underlying "levels coincide with depth" result is B. The
specific volume-profile construction has no independent published test.

**Data.** **Requires intrabar price–volume distribution (tick or footprint).**
OHLCV gives four prices and one volume per bar; any "volume profile" built from
that is a histogram of bar closes, which is a different object and a poor
approximation of it.

**Verdict.** **OPTIONAL**, default off, with the data deficiency documented. If
a footprint feed exists, high-volume-node proximity is admissible as a
CONFIRMING input. On OHLCV it is not admissible at all, and the honest reason
is that we cannot compute it.

**Proxy.** With footprint: distance in σ to the nearest high-volume node in the
trailing `rank_window`. Without: nothing — confirmed pivots already carry the
level information at the fidelity the data supports.

### 4.2 Volume delta (per-bar signed volume)

**Claim.** Splitting each bar's volume into buy-initiated and sell-initiated
components reveals aggressor intent that price alone conceals.

**Mechanism.** Yes, when the signing is accurate. This is the bar-level version
of order-flow imbalance, and §2.2's grade-A contemporaneous result applies.

**Grade.** **C**, for two reasons that are both about the signing, not the
concept. First, trade signing is inferential: Lee–Ready (1991) is the standard,
its accuracy was measured around 85% on 1980s–90s NYSE data, and it degraded
after decimalisation and with the rise of sub-penny and off-exchange
execution. Second, the derived toxicity metrics are contested: VPIN (Easley,
López de Prado and O'Hara, 2012) was directly challenged by Andersen and
Bondarenko (2014), who find it is a poor short-run volatility predictor and
that it peaked *after* rather than before the 2010 flash crash.

**Data.** **Requires tick data with trade-direction inference, or an exchange
feed with aggressor flags.** Not derivable from OHLCV.

**Verdict.** **OPTIONAL.** `Bar.delta` exists as a nullable field;
`enable_delta_features` defaults to `False`; `delta_norm` is `NaN` without a
feed and lowers `reliability` rather than substituting a neutral reading, per
SPEC §1.

**Proxy.** `delta_norm = delta / (volume × participation^0.5)`, ranked causally
and de-seasonalised by time-of-day bucket.

### 4.3 Cumulative delta

**Claim.** Accumulating signed volume produces a running measure of net
aggression; divergence between cumulative delta and price reveals absorption and
predicts reversal.

**Mechanism.** No coherent one at the level of the *cumulative* series. Signed
volume is a noisy estimate with a signing error that is not mean-zero
(misclassification is systematically related to spread and to trade size).
Cumulating a series with a biased error produces a random walk with drift in
the *error*, so the level of cumulative delta is dominated by accumulated
misclassification, not by accumulated aggression. Where the origin is placed
determines the level, and there is no principled origin.

**Grade.** **D.** "Delta divergence" has no null distribution as usually
stated: any two series that both trend will diverge somewhere, and the pattern
is identified visually after the fact.

**Data.** Same requirement as §4.2, plus an arbitrary origin.

**Verdict.** **DISCARDED.** The non-stationary level is not usable. The
differenced, normalised, per-bar quantity is exactly `delta_norm` in §4.2 and is
already available under a flag. Nothing is lost by dropping the cumulative
form; what is gained is that we do not have to defend a choice of origin.

**Proxy.** None. Superseded by `delta_norm`.

### 4.4 Footprint charts

**Claim.** Displaying per-price bid/ask volume within each bar lets a trader see
absorption, exhaustion and initiative in a way aggregated bars cannot.

**Mechanism.** The *data* is legitimate and strictly more informative than
OHLCV. The *method* — reading the display — is a discretionary human process
with no specification.

**Grade.** **D** as a method; the data it displays supports better estimators
(§4.1, §4.2) which are graded separately. There is no peer-reviewed evaluation
of footprint reading, and there could not be, because there is no stated rule to
evaluate.

**Data.** Tick or footprint feed.

**Verdict.** **DISCARDED as a method; OPTIONAL as a data source.** If the feed
exists, use it to compute `delta_norm` and high-volume nodes, both of which have
null distributions. Do not implement a "footprint pattern" detector — that would
be encoding one person's reading habits as if they were a measurement.

**Proxy.** None beyond §4.1–4.2.

### 4.5 Accumulation / distribution

**Claim.** Sustained, quiet buying by informed participants before a markup, and
its mirror before a markdown, is detectable in the relationship between volume
and price before the move is visible in price alone.

**Mechanism.** There is a genuine and well-formalised mechanism here, and it is
not the one the classic indicators encode. Campbell, Grossman and Wang (1993)
show that price declines on high volume reverse more strongly than declines on
low volume, consistent with volume identifying risk-sharing (liquidity-motivated)
trade rather than information. Llorente, Michaely, Saar and Wang (2002) sharpen
this into a testable cross-sectional prediction: **return autocorrelation
conditional on volume is negative for stocks where trade is liquidity-driven and
positive where it is information-driven.** That is exactly the
accumulation/distribution intuition, stated so it can be measured, with a sign
that is itself an estimate.

**Grade.** **B** for volume-conditioned return autocorrelation.
**D/F** for the classic constructions. The A/D line and OBV are arbitrary
cumulative sums with the same origin problem as §4.3 and no derivation; Granville's
OBV assigns a bar's entire volume to one direction based on the close, which is
a maximally lossy estimator of the quantity it is trying to measure.

**Data.** OHLCV suffices for the defensible version.

**Verdict.** **CONFIRMING**, under the name *volume-conditioned return
autocorrelation*. The A/D line and OBV are **DISCARDED**.

**Proxy.** Causal rolling estimate of `corr(r_t, r_{t−1} × participation_{t−1})`
over `rank_window`; its sign classifies the instrument's current regime as
liquidity-driven (expect reversal) or information-driven (expect continuation),
and it feeds the regime posterior rather than the direction directly.

### 4.6 Wyckoff method

**Claim.** Markets move through a repeating cycle — accumulation, markup,
distribution, markdown — driven by the relationship between "effort" (volume)
and "result" (price movement); a divergence between effort and result signals
that the campaign is failing and reversal is near.

The framework and the folklore must be separated, because they have completely
different evidence grades.

**Mechanism — the framework.** No testable content. The phase taxonomy
(springs, upthrusts, secondary tests, signs of strength, last points of supply)
is applied retrospectively; every price path can be labelled as some phase, and
the labelling is not unique. Nothing in the framework specifies, in advance and
without discretion, which bars constitute a spring.

**Mechanism — effort versus result.** Fully coherent, and it is the central
quantity of modern microstructure. "Price impact per unit volume" is Kyle's
λ (1985); its daily-data cousin is Amihud (2002) illiquidity. Low result per
unit effort means size is resting: liquidity is abundant, or a large passive
participant is absorbing. High result per unit effort means the book is thin.
Both are measurable, both have theory, and both are known to predict volatility
and returns in the cross-section (Amihud 2002; Hasbrouck 2009).

**Grade.** **F** for the framework and phase taxonomy — unfalsifiable as
usually stated. **A** for the effort/result relationship as price impact per
unit volume, on theory and cross-sectional evidence; **B** as a time-series
conditioning variable at the single-instrument level, where the evidence is
thinner.

**Data.** OHLCV suffices.

**Verdict.** **Framework DISCARDED. Effort-versus-result CORE**, under its
proper names. Wyckoff deserves credit for stating the relationship in 1931; he
does not get a vocabulary in the codebase, because the vocabulary is where the
unfalsifiability lives. `notes` may say `"absorbing"`, never `"spring"`.

**Proxy.** `absorption = participation / |ret_norm|`;
`amihud = |ret| / dollar_volume`; `kyle_lambda`; all ranked causally over
`rank_window` and de-seasonalised by time-of-day bucket, because the intraday
U-shape (§11.1) otherwise dominates the signal.

---

## 5. Options-market positioning

### 5.1 Options positioning (open interest, skew, put/call)

**Claim.** Option-market variables reveal informed positioning and expected
distributions, and predict underlying returns and volatility.

**Mechanism.** Partly. The volatility content is well founded: the
variance risk premium (Bakshi–Kapadia 2003; Bollerslev, Tauchen and Zhou 2009)
is one of the most robust premia in finance, and implied volatility is a
genuinely forward-looking input that no OHLCV transformation can replicate.
Directional content from open interest and put/call ratios is much weaker: open
interest does not reveal which side is long, and put/call ratios conflate
hedging with speculation.

**Grade.** **B** for implied-volatility level and term structure as inputs to a
volatility forecast; **C** for open-interest and put/call directional signals.

**Data.** **Paid options feed.** Not derivable from underlying OHLCV.

**Verdict.** **OPTIONAL.** `ExogenousSnapshot.implied_vol` is consumed by
`CrossAssetEngine` when available. `oi_change` is consumed only by
`PositioningEngine`, which defaults off.

**Proxy.** With a feed: implied-minus-realised volatility spread in σ units, and
its rank. Without: nothing — `sigma_fcst` already carries the best OHLCV-derived
volatility forecast, and pretending to infer positioning from price would be
worse than abstaining.

### 5.2 Gamma exposure

**Claim.** Aggregate dealer gamma determines whether hedging flow is
stabilising or destabilising. Under net long dealer gamma, hedging is
counter-trend and volatility is suppressed; under net short gamma, hedging is
pro-trend and moves amplify. Large strikes act as magnets and "gamma flip"
levels as regime boundaries.

**Mechanism.** **The mechanism is real and documented.** Gârleanu, Pedersen and
Poteshman (2009) establish demand-based option pricing with constrained
intermediaries; Ni, Pearson and Poteshman (2005) document expiration-date price
clustering (pinning) attributable to hedge rebalancing; Ni, Pearson, Poteshman
and White (2021) identify a non-informational channel through which market-maker
hedge rebalancing affects underlying return volatility and the probability of
large moves. This is not folklore.

**Grade.** **B for the mechanism. F for the publicly available data.** This
distinction is the entire verdict, so state it precisely: every widely
circulated retail "GEX" or "dealer gamma" figure is constructed by (i) taking
exchange open interest, which does not identify counterparties; (ii) assuming a
sign convention — customers buy puts and sell calls, dealers take the other
side — which is an assumption, not a measurement; (iii) applying a
Black–Scholes gamma at an assumed volatility; and (iv) summing across strikes
and expiries with an assumed spot. Steps (i) and (ii) are unverifiable. There
is no public dataset against which a GEX estimate can be validated, which means
there is no way to know whether an engine consuming it is responding to dealer
positioning or to the sign convention its vendor chose. Published results use
proprietary CBOE customer/firm-level open-interest data that is not available to
us.

**Data.** **Paid options chain plus an unverifiable positioning assumption.**
This is the worst data-quality position in the entire document: it is not that
the data is noisy, it is that its error cannot be bounded.

**Verdict.** **OPTIONAL, default off**, with the deficiency stated in
`docs/09-ASSUMPTIONS.md` in exactly these terms. `enable_positioning_engine`
defaults to `False`, and its config docstring already says "the mechanism is
real, the retail data is not". Never permitted in SPEC §7 conditions 1–10.

**Proxy.** If a chain is available: `gamma_imbalance`, explicitly labelled as
resting on an unverifiable dealer-sign assumption, capped at low `reliability`,
and monitored for drift. Preferred alternative: skip the estimate and measure
the *consequence* instead — see §5.3.

### 5.3 Dealer hedging

**Claim.** Delta-hedging flow from options and leveraged-ETF market makers is
mechanical, predictable in timing, and concentrated near the close.

**Mechanism.** Yes, and here there is a genuinely valuable finding for a system
constrained to OHLCV. Baltussen, Da, Lammers and Martens (2021) document, across
more than 60 futures markets in equities, bonds, commodities and currencies from
1974 to 2020, that the return in the last 30 minutes before the close is
positively predicted by the return over the rest of the day, and they link it to
gamma hedging demand from option and leveraged-ETF market makers. Gao, Han, Li
and Zhou (2018) establish the same market-intraday-momentum pattern on S&P 500
ETF data and show it is stronger on high-volatility, high-volume and
macro-release days.

**Grade.** **B.** Broad cross-market replication and a stated mechanism, but the
effect is a late-session conditional and reverts over subsequent days, so it is
a timing conditional rather than a standalone strategy; and the attribution to
hedging is inference, not identification.

**Data.** **Intraday OHLCV only.** No options data required. This is the point:
the *consequence* of dealer hedging is observable in bars, whereas the *cause*
(§5.2) is not.

**Verdict.** **CONFIRMING** — via session-phase-conditioned momentum, never via
a gamma estimate. When a system can measure an effect directly, it should not
route through an unverifiable estimate of the effect's cause.

**Proxy.** In `BehavioralEngine`: sign and magnitude of the
session-to-date return, active only when `phase` is the closing phase and
`tod_frac` exceeds a threshold, interacted with `vol_rank` (the effect is
documented to be stronger on volatile days). Requires intraday bars; on daily
bars this feature is unavailable and must lower `reliability`, not read neutral.

---

## 6. Volatility

### 6.1 Volatility clustering

**Claim.** Volatility is strongly autocorrelated: large moves follow large
moves. Returns are close to unpredictable in mean but far from independent.

**Mechanism.** Yes, from several directions: information arrival is clustered,
leverage and margin constraints create feedback (Brunnermeier–Pedersen 2009),
and heterogeneous-horizon trading generates cascades. Mandelbrot (1963) observed
it; Engle (1982) and Bollerslev (1986) formalised it; Cont (2001) catalogues it
among the stylized facts that hold universally.

**Grade.** **A.** This is as close to a law as empirical finance has. It holds
in every asset class, at every frequency from seconds to months, in every
decade, and it has never meaningfully attenuated despite being known for forty
years — which is itself informative: unlike a return anomaly, knowing about it
does not arbitrage it away.

**Data.** OHLCV. Range-based estimators (Parkinson 1980; Garman–Klass 1980;
Rogers–Satchell 1991; Yang–Zhang 2000) extract substantially more information
per bar than close-to-close, with Molnár (2012) finding Garman–Klass generally
best among them.

**Verdict.** **CORE.**

**Proxy.** `sigma_bp` (bipower, jump-robust), `sigma_rv`, `sigma_rs`
(Rogers–Satchell, drift-free), `sigma_ew` (EWMA), `vol_of_vol`, `vol_rank`,
`compression`.

### 6.2 Volatility forecasting

**Claim.** Next-period volatility is predictable with accuracy far exceeding
anything achievable for next-period returns.

**Mechanism.** Yes — clustering (§6.1) plus long-memory-like persistence
(Ding, Granger and Engle 1993) plus the leverage effect. Corsi's (2009) HAR
model gives an interpretation: volatility is generated by agents operating at
daily, weekly and monthly horizons, whose superposed persistence mimics long
memory with three parameters.

**Grade.** **A, and it is the strongest item in this document by a wide
margin.** Andersen and Bollerslev (1998) settled the "volatility models don't
forecast" objection by showing the low R² of early studies was a property of
the noisy proxy (squared daily returns), not the model. With realised measures
as the target, daily-horizon regressions on log realised variance routinely
achieve out-of-sample R² of roughly 0.5–0.7. Hansen and Lunde (2005) found no
model reliably beating GARCH(1,1) for exchange rates while asymmetric models
help for equities; Patton and Sheppard (2015) show signed jumps and
semivariances add real forecasting power, with negative realised semivariance
the more informative component. The result replicates across asset classes and
decades and is not competed away.

**Data.** OHLCV. Realised-variance-quality forecasts want intraday data
(Andersen, Bollerslev, Diebold and Labys 2003), but range estimators recover
much of it from daily bars.

**Verdict.** **CORE. And this is the architectural centre of gravity.**

Two orders of magnitude separate a volatility R² of ~0.6 from a return R² of
~0.005. That asymmetry has a direct design consequence, which the rest of this
document exists to enforce: **the reliable edge available to a system like TI-A
lives in the risk denominator, not the return numerator.** Concretely —

* barriers should be placed in units of *forecast* σ, so that stop and target
  distances are correct even when the direction is a coin flip
  (`TargetSpec.sigma` from `sigma_fcst`, not from trailing realised σ);
* position size should be inversely proportional to forecast σ, which is where
  Barroso and Santa-Clara (2015) locate essentially all of the improvement in
  risk-managed momentum, and Moreira and Muir (2017) in volatility-managed
  portfolios;
* the system should stand aside when forecast σ is far from its own recent
  distribution, because that is where the barrier geometry is least reliable;
* the honest caveat: Cederburg, O'Doherty, Wang and Yan (2020) show that most of
  the volatility-managed alpha in Moreira–Muir is not implementable in real time
  and that out-of-sample versions generally underperform, with the failure
  traced to instability in the spanning regressions. Volatility timing helps
  momentum, profitability and betting-against-beta and does not help most other
  factors. So: **use forecast volatility to scale risk, not to time the
  market.** The former is robust; the latter is the contested part.

There is one further consequence worth stating because it is easy to miss.
Christoffersen and Diebold (2006) show that volatility dependence *by itself*
induces sign predictability at some horizons, even with zero conditional mean.
That means a directional backtest can show a positive hit rate that is entirely
an artefact of volatility dynamics interacting with a fixed barrier geometry.
Any claimed directional edge must therefore be tested against a
volatility-matched null — which is why `validation/montecarlo.py` requires GARCH
surrogate nulls, not just IID bootstrap.

**Proxy.** `sigma_fcst` — geometric HAR-style blend of daily/weekly/monthly log
volatility with `har_weights`, computed from bipower variation. `vol_ratio =
sigma_ew / sigma_fcst` as the expansion indicator. `compression` for coiling.
`jump_share` to separate continuous from jump variation before forecasting, per
Andersen, Bollerslev and Diebold (2007).

---

## 7. Complexity, memory and information

### 7.1 Entropy measures

**Claim.** The entropy of a return series measures its predictability directly:
low entropy means structure is present and forecasting is possible; high entropy
means the series is effectively random.

**Mechanism.** Coherent in principle. Entropy of the ordinal-pattern
distribution (Bandt and Pompe 2002) is a genuine complexity measure with known
behaviour on stochastic processes, it is robust to monotone transformations of
the series, and it is cheap to compute in a streaming fashion.

**Grade.** **C.** The measure is sound; the finance evidence is thin and
scattered, mostly in physics-adjacent venues, usually without out-of-sample
tests or cost accounting, and typically on single markets. There is no
replicated result showing entropy predicts returns. There is better support for
entropy as a *state descriptor* — periods of low ordinal entropy do coincide
with trending and with volatility regimes — which is a much weaker claim.

**Data.** OHLCV.

**Verdict.** **CONFIRMING**, and specifically as an input to the regime
posterior, never to direction. Permutation entropy is directionless by
construction (it is a function of the pattern distribution, not of the sign of
the drift), so any attempt to derive a direction from it would be an
interpretation layered on top of the measurement.

**Proxy.** `perm_entropy` at `perm_entropy_order = 3` over
`perm_entropy_window = 60`, normalised to `[0, 1]` by `log(3!)`. Consumed by
`RegimeEngine` as one of the Beta-distributed rank features.

### 7.2 Hurst exponent

**Claim.** A single exponent H characterises the memory of a price series:
H > 0.5 means persistence (trending), H < 0.5 means anti-persistence (mean
reversion), H = 0.5 means a random walk. Estimating H therefore tells a system
which regime it is in.

**Mechanism.** The concept is well defined for fractional Brownian motion
(Mandelbrot and Van Ness 1968). The problem is entirely in the estimation and
in the null hypothesis, and it is severe enough to reject the concept as
usually applied.

**Grade.** **D as usually estimated.** The reasons deserve to be spelled out,
because "we compute the Hurst exponent" is one of the most common ways a
research process fools itself.

1. **R/S estimators are badly biased in small samples.** Hurst's (1951)
   rescaled range and the log-log regression built on it produce estimates
   biased *away* from 0.5 on short series. Feed a few hundred draws of IID
   Gaussian noise to a standard R/S estimator and it will typically report H in
   the region of 0.55–0.60. A practitioner computing H over a 200-bar rolling
   window and observing "H ≈ 0.58, the market is trending" is observing the
   estimator, not the market.
2. **Short-range dependence contaminates the estimate.** Volatility clustering
   and autocorrelated volume both inflate R/S-based H even when returns have no
   long memory. Lo (1991) built the modified R/S statistic specifically to be
   robust to short-range dependence and heteroskedasticity, and found no
   evidence of long memory in US stock returns once that correction was applied.
3. **Lo's own test is not the answer either.** Teverovsky, Taqqu and Willinger
   (1999) re-examined Lo's CRSP analysis and showed the modified R/S statistic
   has a strong bias *toward accepting* the no-long-memory null, whether or not
   long memory is present. So the literature's position is: R/S-family
   estimators are unreliable in both directions.
4. **The claim is also mostly false where it can be tested.** Long memory in
   *returns* is largely refuted for liquid markets. Long memory in *volatility*
   and in *absolute* returns is robustly established (Ding, Granger and Engle
   1993) — but that is §6, and HAR captures it with three parameters and no
   exponent estimation.
5. **No usable null distribution.** A rolling H with no standard error cannot
   support any statement of the form "significantly above 0.5", which is the
   only form in which the claim could gate a trade.

**The defensible replacement.** The Lo–MacKinlay (1988) variance-ratio test.
VR(q) = Var(q-period return) / (q × Var(1-period return)) equals 1 under a
random walk; the heteroskedasticity-robust test statistic has an asymptotic
standard normal distribution under a null that explicitly *allows* volatility
clustering — which is essential, since we know volatility clustering is present
(§6.1) and it is exactly what contaminates R/S. So the variance ratio delivers
what the Hurst exponent promises and cannot supply: a signed, standardised
statistic with a known null.

**Data.** OHLCV.

**Verdict.** **Hurst exponent DISCARDED as a decision input. Lo–MacKinlay
heteroskedasticity-robust variance-ratio z-statistic CORE.** A Hurst-like number
may be *displayed* for continuity with practitioner intuition, computed as
`0.5 + log(VR)/(2 log q)`, provided it is labelled biased and never read by an
engine. This is exactly the role of `hurst_implied` in `FeatureSnapshot`, whose
comment already reads "biased, display only".

**Proxy.** `vr_z` — heteroskedasticity-consistent variance-ratio statistic at
`vr_lags = (2, 4, 8, 16)` over `vr_window = 120`. Positive and significant is
evidence of persistence; negative and significant of reversion; insignificant is
the modal state and must be read as *no opinion*, not as weak evidence.

### 7.3 Information theory

**Claim.** Information-theoretic quantities — mutual information, transfer
entropy, KL divergence — measure dependence without assuming linearity, and can
detect predictive relationships that correlation misses.

**Mechanism.** Two distinct uses with different grades.

1. *Scoring and calibration.* Log loss is the KL divergence between the
   forecast and the truth; it is a strictly proper scoring rule (Gneiting and
   Raftery 2007); log-odds is the natural space in which independent evidence
   adds. This is grade **A** and is not a claim about markets at all — it is
   decision theory, and it is the mathematical basis of SPEC §6.
2. *Dependence detection.* Mutual information and transfer entropy on returns.
   Grade **C**. The estimators are severely biased in small samples — plug-in MI
   estimates on a few hundred observations are positive even for independent
   series, and the bias grows with the number of bins — so a reported nonzero MI
   is not evidence of dependence without a permutation null. Where careful
   permutation tests are done, the surviving nonlinear dependence in liquid
   return series is small and largely attributable to volatility.

**Grade.** **A** (scoring rules, log-odds pooling) / **C** (MI and transfer
entropy as dependence detectors).

**Data.** OHLCV.

**Verdict.** **CORE** for the scoring-rule and log-odds machinery: fusion
operates in log-odds, calibration is measured by Brier and log loss,
`brier_alarm` demotes the system when the probability model fails.
**DISCARDED** for MI/transfer-entropy feature screening — the bias problem means
any feature it selected would need a permutation test to defend, at which point
we have replaced one screening procedure with a more expensive one that adds a
trials-ledger entry.

**Proxy.** `l_i = atanh(clip(s_i, ±0.999)) × 2 × rho_i`, GLS-pooled against the
engine correlation matrix; rolling Brier score; effective breadth of evidence
`EBE = 1ᵀC⁻¹1`, which is itself an information-content measure — it counts how
many independent pieces of evidence a set of correlated engine outputs actually
contains.

### 7.4 Fractal geometry

**Claim.** Markets are self-similar: the same patterns recur at every scale, so
a method that works on one timeframe works on all, and fractal structure can be
traded directly.

**Mechanism.** Two claims again, and only one survives.

1. *Statistical scaling properties are non-Gaussian and roughly
   scale-invariant.* True and important. Mandelbrot (1963) established that
   speculative price changes have heavier tails than the normal distribution and
   that the tails do not thin as fast as √t aggregation would imply. Cont (2001)
   documents aggregational Gaussianity, volatility clustering and tail
   heaviness as universal. Grade **A**.
2. *Chart patterns are self-similar, so pattern recognition transfers across
   timeframes, and market structure has a fractal grammar.* Grade **F**. This
   is not a statement with a test. Approximate self-similarity of *statistical
   moments* implies nothing whatsoever about the recurrence of *visual
   patterns*, and the inference from one to the other is the error on which
   Peters' (1991, 1994) popularisation, Elliott wave theory, and the "fractal
   market structure" vocabulary in ICT-adjacent material all rest.

**Grade.** **A** (scaling of moments and tails) / **F** (fractal pattern
trading).

**Data.** OHLCV.

**Verdict.** **DISCARDED as a pattern language.** The surviving content is
already CORE elsewhere: fat tails are why we use bipower variation and a jump
decomposition rather than assuming Gaussian returns; approximate scale
invariance is why every TI-A feature is expressed in σ units or as a causal
rank, which is what makes `htf_multiple = 5` and `htf_multiple_2 = 25` derived
constants rather than user parameters. Multi-timeframe analysis in TI-A is
justified by the scale-free construction of the features, not by a claim that
patterns repeat.

**Proxy.** `ret_kurtosis`; `jump_share = (RV − BPV)/RV`; σ-normalisation of all
displacement features; nested higher-timeframe views at fixed multiples.

---

## 8. Latent state and filtering

### 8.1 Bayesian inference

**Claim.** Beliefs about unobservable quantities should be represented as
distributions, updated by evidence, and decisions taken with respect to the
posterior rather than a point estimate.

**Mechanism.** Not a market claim — an estimation framework. Its relevance here
is that it is the only framework that makes "how much should I trust this
number" a first-class quantity rather than an afterthought.

**Grade.** **A** as estimation machinery, with the specific properties TI-A
needs being long established: hierarchical shrinkage dominates unshrunk
per-cell estimates in mean squared error when cells are thin (the
James–Stein/empirical-Bayes result), and posterior credible bounds propagate
estimation uncertainty into decisions rather than discarding it.

**Data.** OHLCV.

**Verdict.** **CORE.** Three places specifically. (i) The regime posterior is a
discrete-state filter, not a classifier: `regime_posterior` is a distribution
and `regime_hazard_max` gates on transition probability. (ii) The edge book is
hierarchical: `edge_pooling_strength = 25` pseudo-observations of the parent are
injected into each child so a thin cell reports "no edge" rather than "huge
edge", and `edge_prior_mean_sigma = 0.0` makes the null hypothesis of no edge
everywhere the default that evidence must overcome. (iii) The decision gate acts
on `expected_value_lcb`, the 10th-percentile credible bound
(`edge_lcb_quantile = 0.10`), not the posterior mean — so an uncertain edge is
automatically treated as a smaller edge, which is the correct response to
estimation risk and requires no extra machinery.

**Proxy.** Beta–Bernoulli and Normal–Inverse-Gamma conjugate updates with
exponential forgetting (`edge_halflife_bars`); credible bounds from the
posterior; `log_odds_se` propagated to `p_low`/`p_high`.

### 8.2 Hidden Markov models / market regime detection

Treated together, because in practice they are the same claim.

**Claim.** The market occupies one of a small number of unobserved states with
different return and volatility dynamics; identifying the current state
conditions everything else.

**Mechanism.** Yes. Hamilton (1989) established regime-switching as a workable
econometric framework; Ang and Bekaert (2002) show regime-dependent correlations
and their asset-allocation consequences; the persistence of volatility states is
the same fact as §6.1 seen through a discrete lens.

**Grade.** **B**, with an important qualification that determines the verdict.
What regime models reliably recover is **volatility** states — high-vol and
low-vol, with high persistence and well-identified transitions. What they
recover much less reliably is **directional** states: "trending" versus
"mean-reverting" classifications are unstable, and an EM-fitted HMM re-estimated
on a rolling window will relabel its own states, flip their order, and produce
regime sequences that differ materially under small changes to the window. Ex
post, regimes look obvious; ex ante, only the volatility dimension is
dependable. Guidolin and Timmermann's work on regime-switching allocation finds
economically meaningful gains but with substantial parameter uncertainty
`[recollection — verify]`.

**Data.** OHLCV.

**Verdict.** **CORE, but constrained in three ways** that follow directly from
the qualification above.

1. **No EM refitting.** The transition matrix diagonal is a frozen constant
   (`regime_stickiness = 0.985`, implying a mean dwell time of ~67 bars) and the
   measurement densities are fixed Beta distributions on rank features with
   fixed concentration (`regime_beta_concentration = 6.0`). Only the *posterior*
   updates online. This trades some fit for label stability and for a fitted
   degrees-of-freedom count that stays in single digits.
2. **Regime gates, it does not direct.** The regime posterior modulates which
   engines are trusted and whether to trade at all (`QUIET` vetoes entry
   entirely). It never contributes a directional score.
3. **Transitions are for standing aside.** `regime_hazard_max = 0.35` vetoes
   entry when the shift hazard is high, because transitions are where
   conditional distributions are least like their recent history and where
   strategies fitted within a regime fail.

**Proxy.** Four-state discrete filter over `Regime` ∈ {TRENDING,
MEAN_REVERTING, STRESS, QUIET}, driven by ranked `vol_rank`, `vr_z`,
`efficiency`, `perm_entropy`, `absorption_rank`. Shift hazard from the
divergence between the current posterior and an exponentially smoothed
reference posterior at `regime_shift_hazard_halflife = 20`.

### 8.3 Kalman filters and adaptive filtering

**Claim.** A recursive state-space filter extracts the unobserved level and
slope of a noisy series optimally, adapting its responsiveness to the
signal-to-noise ratio, and dominates fixed-window moving averages.

**Mechanism.** Yes, with a precise scope. Kalman (1960) gives the minimum-MSE
linear estimator for a linear-Gaussian state space; Harvey (1989) develops the
structural-time-series formulation TI-A uses. A local-linear-trend filter is
the correct estimator of the level and slope *of the model it assumes*. It is
not a claim that the slope predicts returns.

**Grade.** **B.** For *variance*, adaptive filtering (EWMA is the simplest
adaptive filter) is grade A and is §6. For *level and slope*, the filter is a
better-behaved estimator than a fixed moving average — no fixed lag, an
automatic responsiveness/smoothness trade-off from one interpretable parameter,
and, decisively, a posterior variance. But it does not create predictability
that was not there. The published record on RLS/LMS-style adaptive filters
applied to return prediction is unimpressive and largely un-replicated.

The real value-add is the uncertainty output. A moving-average slope has no
standard error, so "the trend is up" is not a statement that can be gated. The
Kalman posterior gives `kalman_slope_t = slope / sqrt(posterior variance)`,
which turns the same intuition into a t-statistic that can be required to exceed
a threshold. That is the difference between an indicator and a measurement.

**Data.** OHLCV.

**Verdict.** **CORE** as the trend-level estimator, and **CORE for variance
adaptation** (EWMA in `sigma_ew`). Its two DEV-provenance parameters
(`kalman_snr`, `kalman_slope_snr`) are two of only six parameters in the whole
system ever fitted to market outcomes, and they are charged to
`fitted_dof()` accordingly.

**Proxy.** `kalman_level`, `kalman_slope`, `kalman_slope_t`. The engine consumes
`kalman_slope_t`, not `kalman_slope`.

### 8.4 Particle filters

**Claim.** Sequential Monte Carlo estimates the posterior of arbitrary
nonlinear, non-Gaussian state-space models, so it can capture regime and
volatility dynamics that a Kalman filter cannot.

**Mechanism.** Yes, and the method is correct — Gordon, Salmond and Smith (1993)
introduced the bootstrap filter and the theory is sound.

**Grade.** **C** in this application. The method is not in doubt; its marginal
value here is. Three specific objections, in order of severity for TI-A:

1. **The relevant alternative is exact, not approximate.** TI-A's latent state
   is a four-state discrete variable. For a finite discrete state space the
   forward recursion is *exact* and costs O(K²) per bar with K = 4. A particle
   filter would be a Monte Carlo approximation to a quantity we can compute in
   closed form. It cannot do better than exact, and it will do worse.
2. **It violates determinism.** SPEC §5 requires deterministic engines with no
   unseeded RNG, and SPEC §2.6 requires bit-identical output when a random
   prefix of history is replayed. A particle filter is stochastic; making it
   reproducible requires pinning a seed and a resampling order, at which point
   its output depends on that arbitrary choice. Reproducibility and Monte Carlo
   state estimation are in direct tension, and reproducibility wins.
3. **Degeneracy in exactly the wrong place.** Particle weights collapse when an
   observation is far in the tail of the predictive distribution — i.e. during
   the volatility jumps and regime transitions that are precisely the moments
   when we most need the state estimate to be trustworthy.

**Data.** OHLCV.

**Verdict.** **DISCARDED.** Not because sequential Monte Carlo is unsound, but
because for a small discrete state space it is strictly dominated by exact
filtering, and it breaks a non-negotiable architectural requirement. If the
state space were ever made continuous and nonlinear, this decision would need
revisiting; that would be a research decision requiring pre-registration.

**Proxy.** None. Superseded by the exact discrete forward recursion in
`engines/regime.py`.

---

## 9. Statistical learning

### 9.1 Machine learning

**Claim.** Flexible, high-capacity models discover nonlinear interactions in
market data that linear methods and hand-crafted rules miss, and with enough
data and regularisation they outperform.

**Mechanism.** Coherent, and the empirical record in *some* settings is
genuinely good. Gu, Kelly and Xiu (2020) show tree ensembles and neural networks
improve out-of-sample monthly return prediction relative to linear models in the
US cross-section, with the gains concentrated in interactions and in a small set
of predictors. Sirignano and Cont (2019) train on billions of limit-order-book
records and find a *universal and stationary* relation between order-flow
history and price-move direction — a strong result. Chinco, Clark-Joseph and Ye
(2019) show LASSO on the full cross-section of lagged one-minute returns
improves one-minute-ahead forecasts by about 23% over OLS, with the selected
predictors being sparse, short-lived and associated with news. Kelly, Malamud
and Zhou (2024) argue theoretically and empirically for a "virtue of
complexity" in ridgeless regression for market timing.

**Grade.** **C overall**, decomposing as: **B** for cross-sectional ranking with
heavy regularisation and many assets; **B** for order-book-scale learning with
tick data; **D** for raw price/return prediction from OHLCV at the scale
available to this project. The overall grade is C because the successes are in
regimes TI-A is not in, and the failures are in the regime it is in.

The honest reason, stated in terms of signal-to-noise rather than of algorithms.

* **The best available return R² is roughly 10⁻² to 10⁻³.** Gu, Kelly and Xiu
  report monthly out-of-sample R² around 0.3–0.4% for the pooled cross-section.
  Welch and Goyal (2008) show most published equity-premium predictors have
  *negative* out-of-sample R²; Campbell and Thompson (2008) recover roughly
  0.5% monthly with sign and sanity constraints imposed. An R² of 0.005
  corresponds to a correlation between forecast and outcome of about 0.07.
* **A high-capacity model with a target that is 99.5% noise fits the noise.**
  This is not a criticism of any particular architecture; it is a statement
  about the ratio of the number of functions a model can express to the number
  of effectively independent observations available to distinguish them. A
  gradient-boosted ensemble on 20 features and 5,000 daily bars has enough
  capacity to interpolate the sample many times over, and the regularisation
  strength that would prevent it is itself a hyperparameter chosen by looking at
  outcomes — which is a trials-ledger entry.
* **Financial samples are far smaller than they look.** Overlapping labels,
  volatility clustering and cross-sectional correlation mean 5,000 daily bars
  contain on the order of a few hundred effectively independent observations of
  a return-sign relationship. The purged, embargoed cross-validation of
  López de Prado (2018) is a partial remedy for the leakage, not for the
  smallness.
* **Non-stationarity caps what any amount of data can buy.** McLean and Pontiff
  (2016) document post-publication decay in predictor returns; Hou, Xue and
  Zhang (2020) fail to replicate a majority of published anomalies under uniform
  methodology; Chordia, Subrahmanyam and Tong (2014) find anomaly attenuation as
  liquidity improved. A relationship learned from 2010–2018 is not guaranteed to
  be a relationship in 2026, and there is no cross-validation scheme that fixes
  that, because the problem is in the world, not in the estimator.
* **The successful cases are the exceptions that prove the rule.** Sirignano and
  Cont succeed with billions of observations of a *mechanically* determined
  relationship (order flow to price). Chinco et al. succeed on one-minute
  horizons with hundreds of millions of observations, finding effects that are
  explicitly short-lived. Gu, Kelly and Xiu succeed on the cross-section with
  thousands of assets. Every success is bought with either a mechanical target
  or an enormous effective sample. TI-A has neither.
* **The reasonable counter-argument, stated fairly.** Kelly, Malamud and Zhou
  (2024) prove that in a high-dimensional limit, models with more parameters
  than observations can outperform sparse ones, and demonstrate market-timing
  results including divestment before 14 of 15 NBER recessions. If correct, this
  weakens the "capacity fits noise" argument considerably. It is also actively
  contested, depends on ridgeless-regression asymptotics whose applicability to
  a few hundred effective observations is exactly what is in question, and its
  demonstrated application is *timing a single series* with a small number of
  effective events — not the many-decision problem TI-A faces. It is not enough
  to reverse the verdict, and it is enough to keep the question open.

**Data.** OHLCV for the version we could run; tick or a large cross-section for
the versions that work.

**Verdict.** **CONFIRMING and OPTIONAL, with a hard prohibition.** ML is
permitted for exactly three roles, all of which are second-stage problems where
the target is far less noisy than a return:

1. **Calibration.** Mapping a raw score to a probability. Isotonic regression
   (PAVA) after `calibration_min_samples = 200` outcomes. The target here is a
   binary label whose base rate is well estimated, and the model class is a
   monotone step function with effectively one degree of freedom per bin — this
   is the good case.
2. **Meta-labeling.** Taking a signal produced by a CORE component and
   predicting whether *that specific signal* will succeed (López de Prado 2018).
   The primary model sets direction; the secondary model sets size or veto. The
   secondary problem has a much better base rate than direction prediction and
   its errors are bounded (a wrong meta-label costs a foregone trade, not a
   wrong-way trade).
3. **Cross-sectional ranking.** Ordering instruments by expected edge when a
   universe is available — the regime where the published evidence is best.

**Forbidden: originating a direction from a learned function of raw prices.** No
model may output a directional score that is not derived from a component graded
A or B in this document.

**Proxy.** Isotonic calibration map in `fusion/calibration.py`; the edge book
(`engines/edgebook.py`) is a deliberately low-capacity meta-labeler — a
hierarchical conditional-expectancy table over regime × setup × evidence bucket
with `edge_evidence_buckets = 5`, strong shrinkage toward a no-edge prior, and
no free functional form. It is a machine-learning model with the capacity dialled
down to where the signal-to-noise ratio can support it.

### 9.2 Reinforcement learning

**Claim.** An agent trained to maximise cumulative risk-adjusted return
discovers a trading policy end-to-end, including entry, exit and sizing, without
hand-specified rules — and can in principle exceed any hand-designed policy.

**Mechanism.** Coherent in the abstract; trading is a sequential decision
problem with delayed rewards, which is what RL is for. But every one of RL's
enabling assumptions fails in this application, and the failures are structural
rather than a matter of insufficient compute.

**Grade.** **D.** Moody and Saffell (2001) is the honest early attempt; the
literature since is dominated by results that do not survive realistic costs or
out-of-sample periods, and the replication rate is poor. The specific failures:

1. **No simulator that respects market impact.** RL needs an environment it can
   query counterfactually — "what if I had bought here?". A historical replay
   answers that question *wrongly*, because it returns the price path that
   occurred when we did *not* trade. The agent therefore learns a policy for a
   world in which its own actions are free, and the error is largest exactly
   where the agent chooses to be most active. Building an impact-respecting
   simulator requires a model of the order book's response, at which point the
   RL result is a property of that model, not of the market.
2. **One trajectory.** There is exactly one realised history. RL's sample
   complexity is stated in terms of *episodes*; we have one episode, resampled.
   Bootstrap resampling of that episode destroys the temporal dependence that
   the policy is supposed to exploit; not resampling it means the agent
   memorises the path. There is no way out of this dilemma from inside the
   method.
3. **Non-stationary environment.** RL's convergence guarantees assume a fixed
   MDP. Markets are not one — and worse, they are not one *partly because
   participants are also learning*, so the environment is adversarial and
   adapting.
4. **Credit assignment over long horizons with a near-zero signal.** Attributing
   a terminal P&L to one action among hundreds, when the per-action signal-to-noise
   ratio is what §9.1 describes, requires an enormous number of episodes to
   average out. The variance of the policy-gradient estimator scales badly in
   exactly this regime.
5. **Reward hacking against the backtester.** Any exploitable artefact of the
   simulation — optimistic fills, look-ahead in a feature, a barrier tie-break
   rule — will be found and maximised by an optimiser far more reliably than by
   a human designer. RL converts a subtle backtest bug into a spectacular
   backtest result. This is not hypothetical; it is the modal outcome.

**Where RL is legitimately defensible** — and this is worth stating so the
verdict is not read as blanket dismissal. **Optimal execution.** Nevmyvaka, Feng
and Kearns (2006) apply RL to trade scheduling with order-book data and get real
improvements, because there the horizon is minutes, the reward is dense and
attributable, the environment is much closer to stationary, the counterfactual
is much better approximated, and the action space is small. Every objection above
is weakened. That is a genuine RL success in finance, and it is *not the problem
TI-A is solving*.

**Data.** OHLCV would be the available feed; the defensible application needs
L2.

**Verdict.** **DISCARDED for signal generation.** Execution scheduling is out of
scope for TI-A (SPEC §2.3 fixes execution at `t + execution_lag_bars` at the
next bar's open), so the one defensible application does not arise. If TI-A ever
acquires an intrabar execution layer, RL becomes worth revisiting *there* and
nowhere else.

**Proxy.** None. The functions RL would perform are done explicitly and
auditably: sizing by fractional Kelly with an uncertainty haircut
(`risk/sizing.py`), exit by triple barrier, entry by the conjunctive gate.
Every one of those is inspectable and has a stated justification, which a
learned policy would not.

### 9.3 Probability calibration

**Claim.** A forecast probability is only useful if it is calibrated: among all
occasions on which the system says 60%, the event should occur about 60% of the
time. Discrimination without calibration is not decision-ready.

**Mechanism.** Not a market claim — a decision-theoretic requirement, and for
TI-A a load-bearing one. Every gate in SPEC §7 is stated in probability or
expectancy units. `ev_lcb_min_sigma` compares a probability-weighted payoff to a
cost; if `p_success` is not calibrated, that comparison is meaningless and the
system's primary gate is decoration.

**Grade.** **A.** Brier (1950) gives the scoring rule; Murphy (1973) decomposes
it into reliability, resolution and uncertainty, which is exactly the diagnostic
needed to tell "wrong" from "uninformative"; Gneiting and Raftery (2007)
establish the theory of proper scoring rules; Platt (1999) and Zadrozny and Elkan
(2002) give the two standard recalibration maps. Guo et al. (2017) document that
modern high-capacity classifiers are systematically *over*confident and that
calibration must be fitted separately — a warning that applies directly to any
confluence-style score, which behaves like an over-parameterised classifier.

**Data.** OHLCV plus realised labels.

**Verdict.** **CORE.** With one addition that is specific to this architecture
and is the most important calibration point in the design: **naive addition of
log-odds across engines is correct only for conditionally independent evidence,
and confluence scoring is naive log-odds addition wearing a different hat.**
When m engines are correlated, summing their contributions overstates
confidence by roughly √(m / EBE). Correcting this requires the engine
correlation matrix, GLS pooling, and a ridge (`fusion_shrink = 0.15`) to keep
near-collinear engines from producing explosive weights. The `ebe_min = 2.5`
gate then refuses trades where apparent agreement is one piece of information
counted several times. See `docs/05-CONFIDENCE-CALIBRATION.md`.

**Proxy.** Platt link then isotonic (PAVA) after `calibration_min_samples`;
reliability diagram at `calibration_bins = 15`; rolling Brier with
`brier_alarm = 0.27` triggering self-demotion to NO_TRADE (0.25 is the Brier
score of a coin flip, so the alarm fires slightly before the model is provably
worthless).

---

## 10. Directional return predictability

### 10.1 Liquidity sweeps — *compensated liquidity provision*

This is the one item in §10 that comes from practitioner folklore and survives
the audit intact, so it is worth being precise about what survives and why.

**Claim (practitioner form).** Price pushes through an obvious prior high or
low, triggering resting stop orders; that cascade of market orders is absorbed;
price then reverses back through the level and continues in the opposite
direction. The move through the level was "liquidity being taken", and its
failure is the signal.

**Claim (defensible form).** A liquidity-demanding price extreme, followed by a
failure to hold, is followed by a short-horizon reversal. The reversal is
compensation earned by whoever supplied liquidity into the cascade, and its size
is increasing in the contemporaneous cost of supplying liquidity.

**Mechanism.** Yes, and it is well specified.

* **Nagel (2012), "Evaporating Liquidity", RFS 25(7).** Short-term reversal
  strategy returns are a proxy for the returns to liquidity provision, and those
  returns are highly predictable by the VIX: expected returns and conditional
  Sharpe ratios from liquidity provision *spike* during turmoil, consistent
  with financially constrained intermediaries withdrawing supply. This does two
  things for us. It supplies the economic reason the effect exists (it is a risk
  premium for providing liquidity when providing it is expensive, not a free
  lunch), and it supplies the **conditioning variable**: the effect is large
  when volatility is high and small otherwise. An unconditional reversal signal
  is therefore the wrong specification; a volatility-conditioned one is the
  right one.
* **Campbell, Grossman and Wang (1993)** give the volume conditioning:
  high-volume declines reverse more than low-volume declines, because volume
  identifies liquidity-motivated selling.
* **Osler (2003, 2005)** give the microstructural reason the *level* matters:
  stop-loss orders cluster just beyond round numbers and prior extremes, and
  their triggering produces genuine price cascades in currency markets. This is
  measured from actual order data, not inferred from charts.
* **Kavajecz and Odders-White (2004)** show the levels technical analysts use
  coincide with depth peaks in the order book.
* **Brunnermeier and Pedersen (2005, 2009)** and **Carlin, Lobo and
  Viswanathan (2007)** supply the amplification theory — predatory trading and
  liquidity spirals — though note that these are *models*; empirical
  identification of intent is thin, which is why §10.2 grades lower.
* **Lehmann (1990), Jegadeesh (1990)** establish the underlying short-horizon
  reversal fact.

**Grade.** **B.** Solid, mechanistically explained, replicated across equities
and currencies and asset classes, with a stated conditioning variable. Not A,
for two honest reasons: unconditional short-horizon reversal has attenuated
substantially in liquid large-caps since 2000 (Chordia, Subrahmanyam and Tong
2014; Novy-Marx and Velikov 2016 on trading costs of high-turnover anomalies),
and reversal strategies are gross-return-heavy — turnover is high and the effect
is concentrated in less liquid names and stressed periods, both of which are
where costs are worst. The conditional version is what survives.

**Data.** OHLCV is sufficient — the required ingredients are prior confirmed
extremes, an intrabar violation, the closing location, participation, and
contemporaneous volatility rank. This is the rare case where the practitioner
concept, the academic mechanism, and the available data all line up.

**Verdict.** **CORE**, under the name *compensated liquidity provision* /
sweep-failure reversal. Not under the name "liquidity sweep", and specifically
not with any claim about who took the liquidity or why.

**Proxy.** `swept_high` / `swept_low`, set to 1.0 on a bar that (i) trades
beyond a *confirmed* pivot extreme (`pivot_high` / `pivot_low`, visible only
after `pivot_confirm_bars = 3`, so no repainting), (ii) closes back inside the
prior range with `clv_ema` opposing the violation direction, and (iii) has
`participation` in the upper tail of its causal rank. Signal strength scaled by
`vol_rank` per Nagel's conditioning result, and by `absorption_rank`. The
direction is *against* the violation. Consumed by `LiquidityEngine`.

### 10.2 Stop hunts

**Claim.** Large participants deliberately drive price to levels where retail
stop orders rest, in order to trigger them and trade against the resulting flow.

**Mechanism.** The theory exists — Brunnermeier and Pedersen (2005) model
predatory trading, where a predator trades in the same direction as a distressed
trader's forced liquidation and then reverses. Carlin, Lobo and Viswanathan
(2007) formalise episodic breakdowns of cooperative liquidity provision.

**Grade.** **C.** The models are respectable; empirical identification of
*intent* is thin. Osler (2005) documents that price cascades follow stop
triggering, but a cascade requires no author — it follows mechanically from the
existence of clustered stops plus finite depth. Distinguishing "an agent
deliberately caused this" from "clustered stops plus thin depth caused this" is
not possible from price data, and it is not clear that anyone has done it from
order data either.

**Data.** Intent would require identified-participant order data. The observable
consequence is identical to §10.1.

**Verdict.** **DISCARDED as a distinct concept.** Its entire measurable content
is already captured by §10.1, and it adds an unobservable (intent) to a
measurable. Merging it in would give the system a second engine firing on the
same event, inflating apparent confluence while lowering effective breadth —
which is exactly the failure mode `ebe_min` exists to catch. The behavioural
narrative is also actively harmful in a codebase, because it makes a
falsifiable feature feel confirmed by a story.

**Proxy.** None additional. Superseded by `swept_high` / `swept_low`.

### 10.3 Momentum ignition

**Claim.** A participant initiates a rapid series of orders to induce a sharp
price move and trigger other algorithms' momentum responses, then exits into the
flow it created.

**Mechanism.** Real enough to be a regulatory category. The SEC's *Concept
Release on Equity Market Structure* (Release No. 34-61358, January 2010)
discusses momentum ignition alongside order anticipation as a potentially
manipulative proprietary strategy, and it appears in ESMA's market-abuse
guidance `[recollection — verify]`. Kirilenko, Kyle, Samadi and Tuzun (2017)
document related HFT dynamics in the 2010 flash crash.

**Grade.** **C.** The phenomenon is real and named by regulators. What does not
exist is a validated detector: identifying it requires observing rapid order
submission and cancellation by a single participant, which needs order-level
data with participant identifiers — a regulatory dataset, not a market feed.

**Data.** L3 / order-audit-trail data with participant identifiers. Not
obtainable.

**Verdict.** **DISCARDED — unidentifiable from any feed we will have.** From
OHLCV, a momentum-ignition event and a genuine information-driven jump are the
same object: a large, fast, high-participation displacement. We already measure
that (`jump_share`, `ladr`) and we already condition on whether it holds or
reverses. Adding an "ignition detector" would be relabelling a measurement with
an unverifiable causal story.

**Proxy.** None. Its observable footprint is `jump_share` plus the subsequent
hold/fail conditioning in §10.1 and §10.5.

### 10.4 ICT concepts and Smart Money Concepts

Taken together, because SMC is the broader label and ICT the best-known
curriculum within it. This entry is longer than the others because both
credulity and dismissiveness are easy here, and neither is useful.

**Claim.** Price is driven by institutional order flow that leaves identifiable
footprints. Order blocks mark where institutions accumulated and price returns
to them. Fair value gaps are inefficiencies price must fill. Displacement marks
institutional intent. Liquidity is deliberately engineered above highs and below
lows and then swept. Market-maker models, kill zones, optimal trade entry,
premium/discount arrays and IPDA time-based delivery describe when and where
this happens.

**Assessment of the framework as a whole — grade F.** The reasons are specific,
not rhetorical.

1. **The core inferential move is unfalsifiable.** Every claim attributes
   observed price action to the intent of an unobservable agent ("smart money",
   "the algorithm", "IPDA"). No OHLCV feed identifies counterparties. A theory
   whose central variable is unobservable, and which can accommodate any
   outcome by relabelling the phase, does not make predictions that can fail.
2. **Level selection is discretionary and post hoc.** "The order block" is
   identified after the move that makes it interesting. Which of the several
   candidate bars becomes the order block is a judgement, and different
   practitioners mark different bars on the same chart. Without a
   selection rule stated in advance, backtesting is impossible in principle —
   not merely difficult.
3. **No null distribution, ever.** No ICT/SMC claim we are aware of is
   accompanied by a statement of what the pattern's frequency or success rate
   would be under a random walk with the same volatility. Since price
   frequently returns to recent levels *by construction* (that is what a
   mean-zero random walk with clustered volatility does), "price returned to the
   order block" has a high base rate that must be netted out before any claim of
   edge survives. None of the popular material does this.
4. **No peer-reviewed literature exists.** A targeted search returned only
   educational content and vendor material; no indexed peer-reviewed evaluation
   of order blocks or fair value gaps was located. This is not proof of absence,
   but for a framework with a very large practitioner following, the absence of
   even a negative published test is informative about the framework's
   testability rather than about academic neglect.
5. **The general prior is bad.** The broader technical-pattern literature, when
   tested with data-snooping corrections, is not encouraging: Brock, Lakonishok
   and LeBaron (1992) found simple rules profitable; Sullivan, Timmermann and
   White (1999) showed the result did not survive a bootstrap reality check
   across the full rule universe; Bajgrowicz and Scaillet (2012) found no
   persistent profitable rules after false-discovery control and costs; Marshall,
   Young and Rose (2006) found candlestick patterns valueless for DJIA stocks
   under a bootstrap that randomises OHLC; Park and Irwin (2007) survey the
   field and find the positive results concentrated in older, smaller and less
   liquid markets. Lo, Mamaysky and Wang (2000) is the honourable exception —
   they formalised pattern definitions with kernel smoothing precisely so the
   patterns *could* be tested, and found modest incremental information content.
   That is the standard ICT/SMC has not met: **state the rule so it can fail.**

**The two components that DO survive.** Being fair means identifying them
precisely, keeping them under their proper names, and giving credit for the
observation while withholding it from the framework.

**(a) The stop-run / liquidity-cascade reversal.** Under the ICT label this is
"liquidity sweep", "stop run", "turtle soup", "judas swing". Stripped of the
intent narrative it is a real, measurable, economically motivated effect:
compensated liquidity provision into a cascade. Nagel (2012) supplies the
mechanism and the volatility conditioning; Osler (2003, 2005) supplies the
order-clustering evidence that explains why prior extremes and round numbers are
where cascades happen; Campbell, Grossman and Wang (1993) supply the volume
conditioning. **Kept, as §10.1, graded B, CORE.** ICT's contribution is having
noticed the pattern; it is not the explanation, and the explanation matters
because it tells us the effect is conditional on volatility rather than
universal.

**(b) Displacement / imbalance, as jump variation.** Under the ICT label,
"displacement" is a large, fast, one-directional move, and a "fair value gap" is
the unfilled range left in the middle of a three-bar displacement. Stripped of
the narrative, this is the *jump component of quadratic variation*, and it has a
rigorous econometric treatment: Barndorff-Nielsen and Shephard (2004, 2006)
decompose realised variance into continuous and jump parts using bipower
variation; Lee and Mykland (2008) give a nonparametric jump test with a null
distribution and show jumps in individual stocks coincide with earnings and
company news while index jumps coincide with macro releases; Andersen,
Bollerslev and Diebold (2007) show separating the jump component materially
improves volatility forecasts. So there is a measurable quantity, a test
statistic, a null, and a documented economic content: a jump is usually news
being incorporated. **Kept, as `jump_share` and `ladr`, graded B, CORE as a
feature.** Note carefully what is *not* kept: the claim that the gap must be
"filled". Jump-variation literature says a jump is a permanent price revision
carrying information — the opposite of the gap-fill claim — and gap-fill has a
high base rate under a random walk that no popular treatment nets out.

**Everything else is DISCARDED**, with reasons:

| Concept | Why discarded |
|---|---|
| Order blocks, breaker blocks, mitigation blocks | Discretionary post-hoc level selection; no advance rule; base rate of revisiting a recent level not netted out. |
| Fair value gap as a *target* (as opposed to a jump measurement) | The measurable part is jump variation, already kept. "Must be filled" contradicts the jump literature and has an unstated high base rate. |
| Premium/discount arrays, optimal trade entry, Fibonacci zones | Fixed retracement ratios with no derivation and no null; equivalent to `range_pos`, which is kept and requires no numerology. |
| Kill zones | Time-of-day effects are real (§11.1) and are kept under that name, measured per-instrument from data rather than fixed to London/New York clock windows asserted a priori. |
| Judas swing, market-maker buy/sell models, AMD/PO3 | Narrative sequences with no advance identification rule. |
| SMT divergence | Cross-asset divergence between correlated instruments; the measurable version is peer correlation and lead-lag (§10.6), kept under that name with an actual estimator. |
| IPDA, algorithmic time-based delivery | Posits an unobservable mechanism; unfalsifiable. |
| "Smart money" as an agent | Not identifiable in any available feed. The named counterparty is the framework's core unobservable. |
| Liquidity voids, inducements, engineered liquidity | §2.4 — the order-clustering half is kept via confirmed pivots; the intent half is unfalsifiable. |

**Data.** OHLCV for the two survivors. The discarded material would require
identified-participant data to test even in principle, which is another way of
saying it is not a theory about prices.

**Verdict.** **Framework DISCARDED. Two components kept under their proper
names** (§10.1 sweep-failure reversal as compensated liquidity provision; §7.4
and this entry, displacement as jump variation). The renaming is not pedantry:
the proper names come with test statistics, null distributions and conditioning
variables, and the ICT names come with none of those. Renaming is what makes the
components testable.

**Proxy.** `swept_high`, `swept_low` (§10.1); `jump_share = (RV − BPV)/RV`;
`ladr` (liquidity-adjusted displacement ratio, signed) = displacement
normalised by `participation^impact_exponent`, which is the square-root-impact
correction that makes displacement comparable across volume regimes;
`range_pos`.

### 10.5 Trend persistence

**Claim.** Assets that have risen continue to rise over horizons of weeks to
months, across essentially every market, and the effect has been present for
over a century.

**Mechanism.** Yes, several, and their multiplicity is a mild worry (a fact
explained by four incompatible theories is a fact whose explanation is unknown).
Under-reaction to information via gradual diffusion (Hong and Stein 1999);
over-reaction via feedback trading (Barberis, Shleifer and Vishny 1998; Daniel,
Hirshleifer and Subrahmanyam 1998); the disposition effect creating a
capital-gains overhang (Grinblatt and Han 2005; Frazzini 2006); and risk-based
explanations.

**Grade.** **B**, downgraded from A, and the downgrade is the substance of this
entry.

*For A:* Jegadeesh and Titman (1993) established cross-sectional momentum;
Moskowitz, Ooi and Pedersen (2012) established time-series momentum across 58
instruments and four asset classes; Asness, Moskowitz and Pedersen (2013) show
momentum everywhere; Hurst, Ooi and Pedersen (2017) extend the evidence to 1880
and find positive returns in *every decade*, with good performance in 8 of the
10 largest 60/40 drawdowns. That is a genuinely remarkable record.

*Against A:* (i) **Crash risk.** Daniel and Moskowitz (2016) document momentum
crashes — the strategy has conditional negative skew and its worst episodes are
severe, which means realised Sharpe overstates the utility of the payoff.
(ii) **Post-2009 attenuation.** Managed-futures trend-following performance
since roughly 2010 has been materially below its long-run record, and while the
2022 rebound helps, "positive in every decade" is a low bar for a strategy
whose Sharpe has roughly halved. (iii) **Costs bind.** Korajczyk and Sadka
(2004), Lesmond, Schill and Zhou (2004) and Novy-Marx and Velikov (2016) show
momentum's profitability is materially reduced and in some specifications
eliminated by realistic trading costs. (iv) **Horizon mismatch — the one that
matters most for TI-A.** The strong evidence is at *monthly-to-annual* formation
horizons on futures and on cross-sectional equity portfolios. There is no
comparable body of evidence for intraday or few-bar trend persistence, and the
short-horizon evidence points the *other way* (§10.6 reversal). Importing
"momentum works" from the 12-month literature into a 30-bar holding period is a
category error, and it is one of the most common ways a system's designer
borrows credibility that does not transfer.

**Data.** OHLCV.

**Verdict.** **CORE at daily-and-slower execution timeframes; explicitly
downgraded to CONFIRMING at intraday timeframes**, where the supporting
literature is thin and the cost burden is proportionally much larger. This
should be a documented property of the deployed configuration, not a free
parameter.

**Proxy.** `kalman_slope_t` (slope divided by its own posterior standard error —
so the requirement is statistical, not merely directional); `vr_z > 0` and
significant, which is the direct test of persistence with a valid null under
heteroskedasticity; `efficiency = |net move| / path length` as a
directionality-of-path measure; `trend_agree` as multi-horizon sign agreement;
`htf_slope_t` for the higher-timeframe view. All four are correlated by
construction, which is precisely why they must enter fusion through the
correlation-aware pooling of SPEC §6 rather than being summed.

### 10.6 Mean reversion

**Claim.** Prices overshoot and revert; buying weakness and selling strength is
profitable at short horizons.

**Mechanism.** Yes, and it is the same mechanism as §10.1: inventory risk and
compensated liquidity provision. A dealer or arbitrageur who absorbs a
liquidity-motivated order flow takes on inventory risk and must be paid for it;
the payment appears as a short-horizon reversal.

**Grade.** **B, conditional; C, unconditional.** Jegadeesh (1990) and Lehmann
(1990) established weekly and monthly reversal. Nagel (2012) reinterprets it as
liquidity provision and shows the returns are strongly predictable by the VIX —
large in turmoil, small otherwise. Campbell, Grossman and Wang (1993) show the
volume conditioning. Against: Chordia, Subrahmanyam and Tong (2014) document
attenuation as liquidity improved; the effect requires high turnover, so it is
the anomaly most exposed to costs (Novy-Marx and Velikov 2016); and in liquid
large-caps the unconditional version is largely gone net of costs.

The conclusion is not "mean reversion works" or "mean reversion is dead". It is
that **the unconditional version is dead and the conditional version is alive**,
where the conditioning is on volatility, on liquidity demand, and on the
character of the volume (liquidity-driven versus information-driven, per
Llorente et al. 2002).

**Data.** OHLCV.

**Verdict.** **CORE, conditional only.** TI-A may take a reversion trade when
`vol_rank` is elevated, a liquidity-demand event has occurred (§10.1), and
`absorption_rank` is high. It may not take an unconditional "oversold" trade —
which also means no RSI/stochastic-style unconditional oscillator entries, since
those are exactly the unconditional version.

**Proxy.** `swept_high` / `swept_low` interacted with `vol_rank` and
`absorption_rank`; `vr_z < 0` and significant; volume-conditioned return
autocorrelation (§4.5) negative.

### 10.7 Statistical arbitrage

**Claim.** Relative-value positions in related instruments isolate a
mean-reverting spread whose reversion is more reliable than any outright
directional forecast, because the common factor is hedged out.

**Mechanism.** Yes. Cointegration (Engle and Granger 1987) gives the
econometrics; the economic content is that an economically linked pair has a
stationary relationship maintained by arbitrageurs, and that its deviations are
compensated for bearing convergence risk (Shleifer and Vishny 1997 on why the
compensation is needed).

**Grade.** **B, shrinking.** Gatev, Goetzmann and Rouwenhorst (2006) document
profitable distance-based pairs trading from 1962 with returns declining sharply
after the early 2000s; Avellaneda and Lee (2010) find similar decay for
PCA-residual stat-arb in US equities. The decline is exactly what
Shleifer–Vishny plus competition predicts, and McLean and Pontiff (2016)
generalise the pattern.

**Data.** OHLCV for a universe, plus reliable shorting, financing and corporate
action handling.

**Verdict.** **DISCARDED as a strategy family** — it is architecturally
incompatible with TI-A, which is single-instrument and single-position by
construction (SPEC §3: one position, `LONG → SHORT` forbidden in one bar). A
spread trade is two simultaneous positions with a joint risk model; retrofitting
it would change the type contract, not add a feature. **CONFIRMING** via the
residual machinery: the idea that an instrument's move can be decomposed into a
peer-explained component and an idiosyncratic residual is directly useful, and
it is what `CrossAssetEngine` implements.

**Proxy.** Causal rolling beta of the instrument on each peer in
`ExogenousSnapshot.peers`; residual return in σ units; peer-agreement score.
Consumed as confirmation, and — more importantly — as a *deflator*: if a move is
fully explained by peers, it carries less instrument-specific information, and
its contribution to fusion should shrink accordingly.

### 10.8 Cross-asset correlation

**Claim.** Correlations between instruments are estimable, predictable, and
carry information both about risk and about direction through lead-lag effects.

**Mechanism.** Yes, on both counts, but they are different quality.

*Risk.* Conditional correlation is modellable (Engle 2002, DCC), is asymmetric
— higher in downturns (Ang and Chen 2002; Longin and Solnik 2001) — and, most
importantly for TI-A, **conditional-correlation estimates are biased upward in
high-volatility windows.** Forbes and Rigobon (2002) show that what looks like
contagion is substantially heteroskedasticity: correlation estimated within a
high-variance subsample is mechanically higher. This directly threatens the
fusion layer, because `EBE = 1ᵀC⁻¹1` is computed from an estimated correlation
matrix, and if C is biased upward during stress then EBE is biased *downward*
during stress — which happens to be the conservative direction, and is therefore
acceptable, but must be known and documented rather than discovered later.

*Direction.* Lead-lag exists — Lo and MacKinlay (1990) on cross-autocorrelations,
Hou (2007) on intra-industry information diffusion, Cohen and Frazzini (2008) on
economically linked firms, Rapach, Strauss and Zhou (2013) on US leadership in
international returns — but the effects are small, concentrated in smaller and
less liquid names, and largely at weekly horizons.

**Grade.** **B** for both, with the risk application the stronger of the two.

**Data.** Synchronous OHLCV for peers. Synchronisation matters: non-synchronous
closes manufacture spurious lead-lag, which is the classic artefact in this
literature.

**Verdict.** **CORE for the risk and breadth application** — the correlation
matrix in fusion, `correlation_window = 500`, `fusion_shrink = 0.15`, and
`ebe_min`. Without correlation-aware pooling the system would systematically
overstate confidence whenever several engines respond to the same underlying
move, which is most of the time. **CONFIRMING for directional peer agreement,
with low weight.**

**Proxy.** Causal rolling correlations with `correlation_window`; the ridged
inverse for GLS weights; `EBE` reported on the explainability card so a human
can see when apparent agreement is illusory.

---

## 11. Behavioural, calendar and event conditioning

### 11.1 Time-of-day effects

**Claim.** Volume, volatility, spreads and returns follow systematic intraday
patterns; conditioning on time of day improves everything.

**Mechanism.** Yes. Admati and Pfleiderer (1988) explain the concentration of
volume and volatility at the open and close as an equilibrium in which
discretionary liquidity traders and informed traders cluster together; overnight
information accumulation plus the mechanics of opening and closing auctions do
the rest.

**Grade.** **A for the volume and volatility seasonality.** The intraday
U-shape has been documented since Wood, McInish and Ord (1985) and Harris
(1986), holds across markets and decades, and is one of the most robust facts in
the empirical literature. **B for intraday return predictability**: Heston,
Korajczyk and Sadka (2010) document return continuation at daily-frequency
intraday intervals; Gao, Han, Li and Zhou (2018) and Baltussen et al. (2021)
document market intraday momentum; Bogousslavsky (2016) supplies an
infrequent-rebalancing model that generates the observed autocorrelation
patterns.

**Grade note.** The U-shape's grade-A status is *why* it matters most as a
normaliser rather than a signal. A volume-based feature that is not
de-seasonalised by time of day is measuring the hour of the day. That is not a
subtle bias — the U-shape's amplitude routinely exceeds the cross-sectional
dispersion of the feature it is contaminating, so an un-normalised absorption or
participation feature will fire predominantly at the open and close for
mechanical reasons.

**Data.** Intraday OHLCV. On daily bars, unavailable — and must lower
`reliability`, not read neutral.

**Verdict.** **CORE as a normaliser** (`tod_buckets = 13`, per-time-of-day
rank and robust-z normalisers in `features/ranks.py`). **CONFIRMING as a
directional conditioner** (§5.3).

**Proxy.** `tod_frac`; per-bucket causal rank and robust-z for every volume and
volatility feature; `bars_since_session_open`.

### 11.2 Session effects

**Claim.** Overnight and intraday returns have different statistical
properties; sessions differ in liquidity and price-discovery efficiency; the
open and close are special.

**Mechanism.** Yes. Amihud and Mendelson (1987) show open and close return
variance differ, attributable to the trading mechanism; Barclay and Hendershott
(2003) show after-hours trading has less efficient price discovery per unit
volume; Biais, Hillion and Spatt (1999) trace price discovery through a
pre-opening period.

**Grade.** **A** for the existence of systematic session differences.
Lou, Polk and Skouras (2019) show a "tug of war": several well-known
cross-sectional premia accrue predominantly overnight or predominantly
intraday, and the two components partially offset. Bogousslavsky (2021)
documents the cross-section of intraday versus overnight returns. This is a
robust and replicated decomposition.

**Data.** Intraday OHLCV with session boundaries; a session calendar per
instrument.

**Verdict.** **CORE**, in two distinct roles. As a **gate**: SPEC §7 condition
10 refuses entry during auction periods and non-tradeable phases, which is a
risk-control decision (auction mechanics make fills unpredictable, and our cost
model does not describe them). As a **normaliser and conditioner**: overnight
return is a separate variable from intraday return, not a component of a
continuous series.

**Proxy.** `phase` (`SessionPhase`), `overnight_ret` as a distinct feature,
`dow`, session-relative normalisation of all volume features.

### 11.3 Macro event filtering

**Claim.** Scheduled macroeconomic events dominate returns and volatility on the
days they occur; a system should behave differently around them.

**Mechanism.** Yes, and it is well identified because the timing is exogenous
and known in advance. Savor and Wilson (2013, 2014) show a large fraction of the
equity risk premium is earned on the small number of days with scheduled
macroeconomic announcements; Lucca and Moench (2015) document the pre-FOMC
announcement drift; Ai and Bansal (2018) give a preference-based explanation for
the announcement premium.

**Grade.** **A for the risk claim** — variance rises sharply and predictably at
scheduled times, this replicates across decades and markets, and it needs no
cost argument because standing aside is free. **B for the directional claim**
(announcement-day premium, pre-FOMC drift): real and published, but there are
only a handful of events per year, the results are concentrated in specific
subperiods, and the trades are crowded.

**Data.** **An event calendar is exogenous** —
`ExogenousSnapshot.days_to_event`. Not derivable from OHLCV.

**Verdict.** **CORE as a veto when a calendar is available; OPTIONAL in the
sense that its absence must degrade gracefully.** Without a calendar the system
loses the veto and must not pretend otherwise: `reliability` falls, per SPEC §1.
The directional pre-announcement drift is **DISCARDED** — too few independent
events for our sample-size requirements (§12), and a crowded trade with a
known date is the worst possible combination for a system whose gross edge is
small.

**Proxy.** `days_to_event`; veto entries within a configured window; widen
barriers or refuse when forecast σ is elevated by event proximity. A secondary,
calendar-free fallback: `vol_of_vol` and a jump-arrival-rate estimate rise ahead
of scheduled events even without knowing the calendar, so the volatility layer
provides partial protection automatically.

### 11.4 News impact

**Claim.** News moves prices; identifying and reacting to news is an edge.

**Mechanism.** Yes for the first clause. Lee and Mykland (2008) show individual
stock jumps coincide with earnings and firm-specific news and index jumps with
macro releases; Tetlock (2007) and Tetlock, Saar-Tsechansky and Macskassy (2008)
show textual negativity in financial media predicts returns and volume.

**Grade.** **A that news causes jumps and volatility. D that news is tradeable
by us.** The gap between those two is the whole verdict. Reacting to news
requires low-latency, licensed, machine-readable feeds with entity resolution;
the participants who trade news profitably are competing on microseconds and on
data rights. By the time a retail or mid-tier feed delivers a headline, the jump
has occurred — and the jump *is* the news being incorporated. There is no
published evidence that a delayed, unlicensed news feed generates tradeable
directional signal net of costs.

**Data.** Licensed low-latency news feed with entity resolution. Not in scope.

**Verdict.** **DISCARDED as a signal source.** The residual is already CORE
elsewhere: `jump_share` measures the jump, which is the observable footprint of
the news, and the sweep/hold conditioning in §10.1 measures whether the market
accepted the revision. Measuring the consequence is strictly better than
consuming a delayed report of the cause — the same argument as §5.3.

**Proxy.** `jump_share`; `ret_kurtosis`; jump-arrival intensity as a
`reliability` deflator (a system should trust itself less when jumps are
frequent, because barrier-based labels behave badly under jumps).

### 11.5 Behavioural finance

**Claim.** Investors are systematically biased — overconfidence,
representativeness, loss aversion, the disposition effect, limited attention —
and these biases create predictable price patterns.

**Mechanism.** Yes, and it is well documented at the level of *behaviour*.
Kahneman and Tversky (1979) establish prospect theory; Odean (1998) documents
the disposition effect in brokerage data; Barber and Odean (2000) show retail
traders lose to costs and turnover; Grinblatt and Han (2005) and Frazzini (2006)
connect the disposition effect to momentum and post-earnings drift; Black (1986)
gives the framework in which noise traders can persist. Shleifer and Vishny
(1997) explain why arbitrage does not eliminate the resulting mispricing.

**Grade.** **B.** The biases are real and replicated. The step from bias to
tradeable signal is where the field is weakest, and the honest observation is
this: **almost every behavioural signal, when operationalised, turns out to be
momentum, reversal, or post-earnings drift measured differently.** The
disposition-effect signal is a capital-gains-overhang variable that correlates
strongly with momentum. Overreaction signals are reversal. Limited-attention
signals are drift. Behavioural finance is therefore best understood, for our
purposes, as an *explanation* for effects we already measure rather than as an
independent source of them.

**Data.** OHLCV for the price-based operationalisations; brokerage or
holdings data for the direct behavioural measures, which we do not have.

**Verdict.** **CONFIRMING as interpretation; DISCARDED as an independent signal
source and specifically as a separate engine.** Adding a "behavioural engine"
that fires on the same events as trend and reversal would inflate apparent
confluence while contributing no independent information — the failure mode
`ebe_min` and `fusion_shrink` exist to prevent. Note that `BehavioralEngine` in
the module map is *not* this: it handles session, day-of-week and event
proximity, which are calendar effects with independent evidence, and it should
arguably be renamed for that reason.

**Proxy.** None new. The behavioural literature's role is to justify why the
kept effects should be expected to persist — biases are stable, so effects
grounded in them decay more slowly than effects grounded in
inefficiency-that-arbitrage-can-close.

---

## 12. Summary tables and conclusions

### 12.1 Concept → grade → verdict, sorted by grade

| Concept | Grade | Verdict |
|---|---|---|
| Volatility clustering | A | CORE |
| Volatility forecasting | A | CORE |
| Time-of-day effects (volume/volatility U-shape) | A | CORE (normaliser) |
| Session effects | A | CORE (gate + normaliser) |
| Adverse selection | A | CORE (cost model) |
| Execution quality / implementation shortfall | A | CORE (cost model) |
| Probability calibration | A | CORE |
| Bayesian inference (machinery) | A | CORE |
| Market microstructure theory | A | CORE (estimators, not signals) |
| Macro event filtering — risk/veto | A | CORE (needs exogenous calendar) |
| Information theory — proper scoring rules | A | CORE |
| Wyckoff "effort vs result" as price impact per unit volume | A (theory) / B (time series) | CORE |
| Fractal geometry — scaling of moments and tails | A | CORE (already inside vol/jumps) |
| News impact — news causes jumps | A | CORE via `jump_share` |
| Auction theory — session-phase price-discovery heterogeneity | A | CORE via session effects |
| Order flow theory | A (L2) / C (OHLCV) | CONFIRMING |
| Imbalance detection | A (L2, contemporaneous) / C (bars) | CONFIRMING |
| Trend persistence | B | CORE (daily+) / CONFIRMING (intraday) |
| Mean reversion — conditional | B | CORE (conditional only) |
| Liquidity sweeps → compensated liquidity provision | B | CORE |
| ICT displacement → jump variation | B | CORE (renamed) |
| Hidden Markov models / regime detection | B | CORE (constrained: gate, not direction) |
| Kalman / adaptive filtering | B | CORE |
| Cross-asset correlation | B | CORE (risk/breadth) / CONFIRMING (direction) |
| Accumulation/distribution → volume-conditioned autocorrelation | B | CONFIRMING |
| Institutional execution / metaorder footprints | B | CONFIRMING |
| Hidden liquidity | B | CONFIRMING (via absorption only) |
| Dealer hedging → session-conditioned intraday momentum | B | CONFIRMING |
| Behavioural finance | B | CONFIRMING (interpretation only) |
| Statistical arbitrage | B (shrinking) | DISCARDED (architecture) / CONFIRMING (residuals) |
| Gamma exposure — mechanism | B (mechanism) / F (retail data) | OPTIONAL, default off |
| Options positioning — implied volatility | B | OPTIONAL (paid feed) |
| Liquidity engineering — order clustering at levels | B | DISCARDED as concept (absorbed into pivots/sweep) |
| Macro event filtering — directional pre-drift | B | DISCARDED (too few events) |
| Entropy measures | C | CONFIRMING (regime only) |
| Machine learning | C overall (B ranking / D raw prediction) | CONFIRMING + OPTIONAL, prohibited from originating direction |
| Volume delta | C | OPTIONAL (needs tick/footprint) |
| Volume profile | C | OPTIONAL (needs intrabar price-volume) |
| Particle filters | C | DISCARDED (dominated by exact discrete filter; breaks determinism) |
| Auction market theory as Market Profile | C | DISCARDED |
| Stop hunts | C | DISCARDED (merged into §10.1; intent unobservable) |
| Momentum ignition | C | DISCARDED (needs L3 with participant IDs) |
| VWAP behaviour as reversion target | C | DISCARDED as signal / CORE as cost benchmark |
| Information theory — MI / transfer entropy screening | C | DISCARDED (estimator bias) |
| Options positioning — OI / put-call directional | C | OPTIONAL, default off |
| Hurst exponent | D | DISCARDED (replaced by Lo–MacKinlay variance ratio) |
| Cumulative delta | D | DISCARDED (non-stationary, arbitrary origin) |
| Footprint charts as a method | D | DISCARDED as method / OPTIONAL as data source |
| Reinforcement learning | D | DISCARDED (for signals) |
| News impact as a tradeable feed | D | DISCARDED |
| A/D line, OBV | D/F | DISCARDED |
| Wyckoff framework and phase taxonomy | F | DISCARDED |
| Fractal pattern trading / self-similar chart grammar | F | DISCARDED |
| Liquidity engineering as intent narrative | F | DISCARDED |
| ICT concepts (framework) | F | DISCARDED (2 components kept, renamed) |
| Smart Money Concepts | F | DISCARDED |

### 12.2 What survives

The complete list of A/B-graded, OHLCV-computable effects TI-A is permitted to
rest on. If a proposed feature is not derivable from something on this list, it
does not go in.

**Risk and scale — the strongest ground.**

1. **Volatility clustering and forecastability.** `sigma_bp`, `sigma_rs`,
   `sigma_ew`, `sigma_fcst` (HAR-style), `vol_of_vol`, `compression`,
   `vol_rank`. Grade A. This sets every barrier, every position size and the
   unit in which every other feature is expressed.
2. **Jump/continuous decomposition of variation.** `jump_share` from realised
   minus bipower variation. Grade B, theory A. Improves the volatility forecast
   and is the proper name for "displacement".
3. **Intraday and session seasonality of volume and volatility.** Grade A.
   Mandatory normalisation, not optional polish.

**Conditioning and state.**

4. **Price impact per unit volume.** `absorption`, `amihud`, `kyle_lambda`,
   `participation`. Grade A on theory. This is "effort versus result" with a
   null distribution.
5. **Volatility-regime state with a fixed, non-refitted filter.** Grade B.
   Gates; never directs.
6. **Lo–MacKinlay heteroskedasticity-robust variance ratio.** `vr_z`. Grade A as
   a test statistic. The only defensible persistence/reversion measure in this
   document, because it is the only one with a valid null under volatility
   clustering.
7. **Volume-conditioned return autocorrelation.** Grade B. Classifies the
   current tape as liquidity-driven or information-driven.
8. **Cross-asset correlation for evidence-breadth deflation.** Grade B. Prevents
   the system from mistaking one piece of information for several.

**Direction — the short list, and it is short.**

9. **Sweep-failure reversal as compensated liquidity provision**, conditioned on
   volatility rank and absorption. Grade B. `swept_high`, `swept_low`.
10. **Time-series trend persistence at daily-and-slower horizons**, measured as
    a t-statistic rather than a sign. Grade B. `kalman_slope_t`, `vr_z`,
    `efficiency`, `trend_agree`, `htf_slope_t`.
11. **Conditional short-horizon mean reversion** — the same effect as (9) seen
    from the other side. Grade B, conditional only.
12. **Session-conditioned intraday momentum** (the observable consequence of
    dealer hedging). Grade B. Requires intraday bars.

**Decision machinery.**

13. **Calibration and proper scoring**, with correlation-aware log-odds pooling.
    Grade A.
14. **Bayesian hierarchical shrinkage toward a no-edge prior**, with decisions
    taken at a credible lower bound. Grade A as machinery.
15. **Adverse-selection-aware cost modelling.** Grade A.

That is the entire permitted foundation. Four directional items, all grade B,
none unconditional.

### 12.3 What we discard, and why

| Discarded | One-line reason |
|---|---|
| ICT framework (order blocks, PD arrays, OTE, judas swing, MMXM, IPDA, kill zones as branded) | Central variable ("smart money" intent) is unobservable; level selection is post hoc; no null distribution is ever stated. |
| Smart Money Concepts as an inferential framework | Attributes price action to an unidentifiable counterparty; unfalsifiable by construction. |
| Wyckoff phase taxonomy and vocabulary | Applied retrospectively with no advance identification rule; the one testable idea (effort vs result) is kept under its proper name. |
| Fractal pattern trading / self-similar chart grammar | Confuses approximate scale invariance of statistical moments with recurrence of visual patterns; no test. |
| Market Profile / value-area trading | No published out-of-sample test with a null; needs tick data; its testable residual duplicates mean reversion measured properly. |
| Hurst exponent as a decision input | R/S estimators are severely biased in small samples; Lo's correction over-accepts the null (Teverovsky et al. 1999); no usable standard error. Replaced by `vr_z`. |
| Cumulative delta and delta divergence | Cumulating a biased signed-volume estimate yields a level dominated by accumulated misclassification; arbitrary origin; no null. |
| A/D line, OBV | Arbitrary cumulative constructions with the same origin problem and a maximally lossy volume-signing rule. |
| Footprint chart reading as a method | No stated rule, therefore nothing to test; the underlying data is kept as an OPTIONAL source. |
| Stop hunts as a distinct concept | Intent is unobservable and unnecessary; the measurable content is identical to the sweep-failure reversal, so keeping both double-counts. |
| Momentum ignition | Detection requires L3 order data with participant identifiers; from OHLCV it is indistinguishable from an information jump. |
| Liquidity engineering / inducement narrative | The order-clustering half is real and kept via confirmed pivots; the intentional-engineering half is unfalsifiable. |
| Reinforcement learning for signal generation | No impact-respecting simulator, one historical trajectory, non-stationary and adversarial environment, credit assignment at near-zero SNR, and a strong tendency to maximise backtest artefacts. |
| ML originating direction from raw prices | Return R² of order 10⁻³ against a model class with far more capacity than the effective sample can distinguish; permitted only for calibration, meta-labeling and ranking. |
| Particle filters | Strictly dominated by exact forward recursion on a 4-state space, and stochastic estimation violates the reproducibility requirement in SPEC §2.6. |
| Mutual information / transfer entropy screening | Plug-in estimators are positively biased on independent series at our sample sizes; defending a selection would require a permutation test that costs more than it buys. |
| VWAP as a mean-reversion target; anchored VWAP as support | Benchmark-tracking flow is volume-proportional, not price-reverting; the anchor is chosen after the fact; near-collinear with `range_pos`. |
| News feeds as a directional signal | The jump *is* the news; profitable news trading is a latency-and-licensing competition we are not in. |
| Pre-announcement directional drift | Real but only a handful of independent events per year — far below the sample size §12.4 requires — and heavily crowded. |
| Statistical arbitrage as a strategy family | Architecturally incompatible with single-instrument, single-position semantics; the residual-decomposition machinery is kept. |
| Behavioural finance as a separate engine | Its operationalisations reduce to momentum, reversal and drift; a separate engine would inflate confluence with no independent information. |

### 12.4 The uncomfortable conclusion

Everything above can be summarised in one sentence: **fifteen items survive, four
of them are directional, all four are grade B, and none of them works
unconditionally.** The design consequences of that are arithmetic, and they are
more restrictive than they first appear.

**How small is the surviving directional edge?**

Take the barrier geometry in `config.py`: `target_sigma = 2.6`,
`stop_sigma = 1.6`. Under a driftless random walk, the probability of touching
the target before the stop is `stop / (stop + target) = 1.6 / 4.2 = 0.381`, and
the expected value is exactly zero. **0.381, not 0.5, is the null hit rate.**
Any reported hit rate must be compared to it, and a system reporting "42%
win rate" with these barriers is reporting a real edge while one reporting "52%"
with symmetric barriers is reporting nothing.

Expected value in σ units is `EV = 4.2p − 1.6`. So:

| Gross edge over null | p | Gross EV (σ) | Per-trade Sharpe |
|---|---|---|---|
| +1 pt | 0.391 | 0.042 | 0.020 |
| +3 pts | 0.411 | 0.126 | 0.061 |
| +5 pts | 0.431 | 0.210 | 0.100 |
| +13 pts | 0.511 | 0.546 | 0.253 |
| +20 pts | 0.581 | 0.836 | 0.403 |

Now the costs, in the same units. `cost_slippage_range_frac = 0.10` of the
execution bar's range: for a diffusion, `E[range] ≈ σ√(8/π) ≈ 1.6σ`, and in
practice 1.5–2.5σ for daily bars, so slippage is roughly 0.15–0.25σ per side, or
**0.3–0.5σ round trip.** Add a Corwin–Schultz half-spread of perhaps 0.02–0.05σ
per side and impact of similar order, and total friction is **≈ 0.35–0.55σ per
round trip** — 8–13% of the 4.2σ barrier span.

Setting net EV to the gate value `ev_lcb_min_sigma = 0.05` gives the requirement:

> **TI-A must find a gross directional edge of roughly 8 to 13 percentage points
> over the barrier-implied null in order to clear its own primary gate.**

Nothing in §12.2 delivers that unconditionally. Published unconditional
directional effects, converted to this scale, are worth low single digits at
best. Three consequences follow, and they are the design.

**(1) Trade rarely, and only in the conditional pockets.** An 8–13 point edge is
not available on demand. It plausibly exists in the narrow conditional regions
the surviving items identify — a volatility-conditioned sweep failure in the
upper tail of `vol_rank`, where Nagel (2012) documents that liquidity-provision
Sharpe ratios spike; a high-`slope_t` trend with confirming variance ratio at a
daily-or-slower horizon; the last-30-minute session conditional on a
high-volatility day. Outside those, the correct output is `score = 0.0`, which
SPEC §1 is careful to distinguish from a weak opinion. `max_trades_per_100_bars
= 4.0` should be read not as a permission but as a ceiling far above the
expected rate; if the system approaches it, something has broken.

**(2) Widen the barriers relative to the friction scale — hold longer.** This
is the highest-leverage change available, and it is pure arithmetic. Friction is
roughly *fixed* per round trip (one spread, one impact event, two slippage
charges) while the barrier span scales as √h with holding horizon h. At the
current geometry the expected first-passage time is `stop × target = 1.6 × 2.6 ≈
4.2` bars, so the system pays ~0.45σ of friction to play for ~4 bars of
diffusion. Placing barriers at `k·σ·√h` with h = 30 (the existing
`max_holding_bars`) would widen the span by √(30/4.2) ≈ 2.7×, cutting friction
from ~11% of span to ~4% and reducing the required gross edge from 8–13 points
to roughly 3–5 — which is inside what the surviving literature supports. **The
single most effective way to make a small edge tradeable is to stop paying for
it so often.**

**(3) The configured `p_min` is inconsistent with the arithmetic — a finding for
the designer, not a change made here.** `p_min = 0.58` is documented as "a
floor, not the primary gate". With these barriers it corresponds to a 20-point
edge over null and a gross EV of 0.836σ, which is roughly 17× the
`ev_lcb_min_sigma` gate value. It is therefore not a floor: it is by far the
binding constraint, and it is set above what any documented effect delivers.
Either `p_min` should be expressed relative to the barrier-implied null
(`p_min = b/(a+b) + margin`) so it scales when the geometry changes, or it should
be lowered to approximately 0.42. Flagging, not editing — `config.py` is outside
this document's ownership, and a `THEORY`-provenance change requires a
pre-registration entry.

**Why selectivity alone is not enough: the statistical power problem.**

Selectivity solves the edge problem and creates a worse one. The number of
trades needed to establish a per-trade Sharpe `s` at t-statistic τ is
`N = (τ/s)²`:

| Per-trade Sharpe | Edge over null | N for t = 2 |
|---|---|---|
| 0.024 | ~1.2 pts (the bare gate) | ~6,700 |
| 0.061 | 3 pts | ~1,100 |
| 0.100 | 5 pts | ~400 |
| 0.253 | 13 pts | ~63 |

At a realistic 5-point conditional edge, **~400 trades** are needed for
significance. With a mean holding period of a few bars and a trade rate far
below the 4-per-100-bar ceiling — call it 1 per 100 bars, which is what genuine
selectivity implies — one instrument with 5,000 bars of history yields ~50
trades. **A single instrument cannot establish its own edge.** Not "will
struggle to": cannot, by a factor of eight, within any history that is
plausibly stationary.

And multiple testing makes it worse, in a way that is easy to underestimate. The
expected maximum of T independent null Sharpe estimates is approximately
`2.25/√N` at T = 50 (from the expected maximum of T standard normals, ≈
`Φ⁻¹(1 − 1/(T+1))` plus a Gumbel correction). At N = 400 that is **0.113 —
larger than the 0.100 true per-trade Sharpe of a genuine 5-point edge.** In
other words: at 400 trades, the best of fifty honest attempts on pure noise
looks better than the real effect. Either the trade count rises to ~2,000 (which
makes `E[max SR_null] ≈ 0.05`, half the true edge) or the trial count is held
near single digits. This is exactly why `trials_ledger_count` defaults to 1 and
must be incremented honestly, why `fitted_dof()` is six, and why the deflated
Sharpe ratio (Bailey and López de Prado 2014) — not the Sharpe ratio — is the
acceptance statistic.

**The only resolution: cross-sectional pooling. Many instruments, few trades
each.**

The trade count must come from breadth, not from frequency. But instruments are
correlated, and correlated trades do not each count as an observation. For
average pairwise correlation `ρ̄` of trade outcomes, effective breadth is
`K_eff = K / (1 + (K−1)ρ̄)`:

* 20 correlated single stocks with `ρ̄ ≈ 0.3` → `K_eff ≈ 2.9`. Twenty
  instruments purchase three instruments' worth of statistical power.
* 20 instruments spanning equity indices, rates, FX, commodities and crypto with
  `ρ̄ ≈ 0.1` → `K_eff ≈ 6.9`.

So the universe design is not a deployment detail; it is a statistical
requirement, and it has three parts.

1. **Span asset classes, not sectors.** Diversity of `ρ̄` buys power; count does
   not. Twenty uncorrelated instruments are worth roughly seven, and twenty
   correlated ones roughly three.
2. **Charge for overlap explicitly.** Simultaneous trades in correlated
   instruments must not be counted as independent evidence. This is what
   `labeling/weights.py` sample-uniqueness weighting does for overlapping labels
   in time, and the same discipline must extend across instruments — the same
   argument that produces `EBE` in fusion produces effective trade count in
   validation. A pooled result that ignores it will overstate its t-statistic by
   roughly `√(K/K_eff)`, which at `ρ̄ = 0.3` and K = 20 is a factor of 2.6.
3. **Pool the edge estimate, not the parameters.** Parameters stay global and
   few (`fitted_dof() == 6`); only the edge-book posterior counts accumulate,
   shrunk toward a global no-edge prior (`edge_prior_mean_sigma = 0.0`,
   `edge_prior_strength = 40`). This is what makes pooling a power gain rather
   than a per-instrument overfit.

**The honest bottom line for the committee.**

TI-A's realistic target is a per-trade Sharpe around 0.10 — a 5-point edge over
the barrier-implied null — achieved on perhaps 1 to 2 trades per 100 bars per
instrument, across 15 to 25 instruments spanning at least four asset classes,
with a total trade count reaching a few hundred within two to three years and
approaching 2,000 within a decade. Aggregated at 15 instruments with
`K_eff ≈ 5`, that is an annual portfolio Sharpe in the region of **0.5 to 0.9**
before fees and after honest costs. That is a real result and a defensible one.
It is also nowhere near what a system with these components is usually claimed
to produce, and any backtest of this architecture reporting a Sharpe above about
1.5 should be treated as a bug report rather than a result — the most likely
causes, in order, being optimistic fills, a look-ahead in a feature, a barrier
tie-break favouring the target, and an uncharged trials ledger.

The corollary is the least comfortable part. **Most of what makes this system
worth building is not the directional signal.** It is the volatility forecast
that places barriers correctly, the cost model that refuses trades whose edge
does not clear friction, the calibration that makes the probability mean
something, the correlation-aware pooling that stops one piece of information
from being counted five times, and the credible lower bound that automatically
treats an uncertain edge as a small one. Those components are graded A and they
are what the rare grade-B directional signal is spent through. A system built
the other way round — a confident directional model with risk management bolted
on — would have the grades exactly inverted, and would be resting its
architecture on the weakest evidence in this document.
