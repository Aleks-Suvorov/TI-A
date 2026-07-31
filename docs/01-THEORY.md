# TI-A: Mathematical Theory

> The objective is not to maximise historical returns. It is to build an
> estimator of conditional expectancy that is honest about its own uncertainty
> and that declines to act when that uncertainty is large.

---

## 0. The problem, stated properly

Let $P_t$ be a price and $r_{t\rightarrow t+h} = \ln(P_{t+h}/P_t)$. A trading
system is a map from an information set $\mathcal{F}_t$ to an action. Its
long-run value is governed by expectancy per unit of risk taken.

The reason this is hard is a signal-to-noise problem, and it is worth putting a
number on it before designing anything. Suppose daily returns have volatility
$\sigma = 1\%$ and there exists a genuine predictable component with
$\mathbb{E}[r_{t+1}\mid\mathcal{F}_t] = \mu_t$, where $\mu_t$ has standard
deviation $s$. The best achievable correlation between forecast and outcome —
the information coefficient — is

$$\mathrm{IC} = \frac{s}{\sqrt{s^2+\sigma^2}} \approx \frac{s}{\sigma}.$$

Empirically, for liquid instruments on horizons of hours to days, credible
estimates of $\mathrm{IC}$ from public information sit in the range
$0.02$–$0.06$. That is an $R^2$ of $0.04\%$ to $0.36\%$. Directional
predictability is real but it is *tiny*, and it is buried in a distribution with
fat tails, time-varying scale, and structural breaks.

Three consequences follow immediately, and they dictate the entire architecture:

1. **Any method with enough capacity to fit $R^2 = 0.3\%$ has more than enough
   capacity to fit noise.** So capacity must be spent deliberately and counted.
   This is why §11 accounts for degrees of freedom explicitly and why the regime
   layer is built with *zero* parameters fitted to returns.

2. **The denominator is more forecastable than the numerator.** Volatility is
   the single most reliably predictable quantity in financial time series
   (§2). Since expectancy is $\mathbb{E}[R]/\sigma$ under any risk-normalised
   accounting, a system that forecasts $\sigma$ well and direction barely can
   still have positive expectancy — provided it only acts when the direction
   term clears the cost term. So volatility forecasting is not a filter bolted
   on the side; it is load-bearing, and it enters *first*.

3. **With $\mathrm{IC}\approx 0.04$, most bars contain no actionable
   information.** Abstention is not a defensive afterthought. It is the primary
   output. A system that trades on 2% of bars and stands aside on 98% is
   behaving correctly, not timidly.

### 0.1 The central question, answered

The design brief asks: *what measurable conditions consistently exist
immediately before high-probability directional moves?* Stripped of folklore,
the evidence supports exactly three, and the third splits into two mutually
exclusive cases:

**(A) A resolved risk denominator.** The conditional volatility is estimable
with low relative error, so a stop can be placed where noise will not reach it
and a target where the move can plausibly arrive. This is a *precondition* for
expectancy rather than a source of direction, and it is the most reliable of the
three. Operationalised in §2.

**(B) A liquidity event that fails.** Price penetrates a level at which resting
orders concentrate, consumes anomalous volume doing so, and then fails to hold
beyond it. This is the microstructural signature of forced or mechanical flow
being absorbed by a liquidity provider who is compensated for it. It has real
supporting evidence — short-horizon reversal is strongest exactly when
liquidity provision is most expensive — and it is measurable from OHLCV as
*penetration + high absorption + failure to hold*. Operationalised in §3.2 and
§5.

**(C) A displacement that is cheap in liquidity terms, inside an autocorrelated
regime.** Price travels far per unit of volume consumed. Under any
Kyle-type model, price impact per unit of order flow is increasing in the
probability that the flow is informed, so a large move on small consumed
liquidity is the signature of information rather than pressure — and
information has permanent impact, i.e. it continues. Measurable from OHLCV as
high $|\mathrm{LADR}|$ (§3.1) conditional on a positive variance-ratio
statistic (§4.2).

**(B) and (C) are contradictory instructions.** (B) says fade the move; (C) says
follow it. Their applicability is separated by the sign of return
autocorrelation and by volatility state. This is precisely why the regime
posterior sits at the *top* of the architecture (§5) rather than being a filter
applied afterward: without knowing which of (B) or (C) is in force, the two
signals cancel into noise, which is exactly what happens to indicator systems
that average trend and reversion logic together.

Everything else in this document is machinery for estimating (A), (B), (C) and
their reliability without deceiving ourselves.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $t$ | index of a *closed* bar on the execution timeframe |
| $O_t, H_t, L_t, C_t, V_t$ | bar open, high, low, close, volume |
| $r_t = \ln(C_t/C_{t-1})$ | log return |
| $\sigma_t$ | conditional per-bar return standard deviation, known at $t$ |
| $\hat\sigma_{t+1|t}$ | forecast of next-bar volatility |
| $\mathcal{F}_t$ | information available at the close of bar $t$ |
| $k \in \{1,\dots,K\}$ | regime index, $K=4$ |
| $\alpha_t(k)$ | posterior probability of regime $k$ at time $t$ |
| $s_i \in [-1,1]$ | engine $i$'s directional score |
| $\rho_i \in [0,1]$ | engine $i$'s reliability |
| $\ell_i$ | engine $i$'s log-odds evidence |
| $\hat L$ | pooled log-odds |
| $\mathrm{EBE}$ | effective breadth of evidence |
| $\mathcal{R}$ | outcome return in $\sigma$ units, signed by trade direction |

Everything a decision depends on is one of: a bounded dimensionless quantity, a
t-statistic with a known null distribution, or a causal rolling rank in $[0,1]$.
§10 explains why this restriction is what makes a single parameter set valid
across instruments and across decades.

---

## 2. Layer 1 — the risk denominator

### 2.1 Jump-robust scale

Realised variance over a window of $n$ bars,
$\mathrm{RV}_t = \sum_{i=t-n+1}^{t} r_i^2$, estimates total quadratic
variation, which includes jumps. Bipower variation
(Barndorff-Nielsen & Shephard, 2004),

$$\mathrm{BPV}_t = \frac{\pi}{2}\cdot\frac{n}{n-1}\sum_{i=t-n+2}^{t}|r_i|\,|r_{i-1}|,$$

is consistent for the *continuous* part alone. Their difference isolates jump
variation, giving a jump share

$$J_t = \mathrm{clip}\!\left(\frac{\mathrm{RV}_t-\mathrm{BPV}_t}{\mathrm{RV}_t},\,0,\,1\right).$$

This separation matters twice over. First, $\sigma^{\mathrm{bp}}_t=\sqrt{\mathrm{BPV}_t/n}$
is the correct denominator when asking "was this bar's move large?" — dividing
by a jump-contaminated scale makes genuine displacement look ordinary, which is
exactly backwards. Second, $J_t$ is itself informative: continuous and jump
variation have different persistence and different implications for what
follows.

Range-based estimators add efficiency, since $H$ and $L$ carry information a
close-to-close return discards. Rogers–Satchell (1991) is drift-independent,
which matters precisely in the trending conditions where a system might size
positions:

$$\mathrm{RS}_t = \ln\frac{H_t}{C_t}\ln\frac{H_t}{O_t} + \ln\frac{L_t}{C_t}\ln\frac{L_t}{O_t}.$$

Garman–Klass (1980) is more efficient still under zero drift and is retained as
a cross-check. Disagreement between range and close-to-close estimators is
itself a signal: it indicates intrabar reversal, i.e. that the bar's path
mattered more than its endpoints.

### 2.2 Forecast

Volatility's most robust empirical property is long-memory persistence. The HAR
specification (Corsi, 2009) captures it with three horizons:

$$\ln\hat\sigma_{t+1|t} = w_d\ln\sigma^{(d)}_t + w_w\ln\sigma^{(w)}_t + w_m\ln\sigma^{(m)}_t.$$

We fix $(w_d,w_w,w_m)=(0.5,0.3,0.2)$ a priori rather than fitting per
instrument. The loss versus a fitted HAR is small and well documented; the gain
is that this parameter contributes nothing to the fitted degree-of-freedom count
and cannot be the channel through which overfitting enters. A fitted variant is
available and disabled by default.

Two derived quantities matter downstream:

$$\text{expansion ratio } \quad \Xi_t = \frac{\sigma^{\mathrm{ew}}_t}{\hat\sigma_{t+1|t}}, \qquad
\text{compression } \quad \Gamma_t = 1-\mathrm{rank}_t\!\left(\frac{\sigma^{(d)}_t}{\sigma^{(m)}_t}\right).$$

$\Xi_t>1$ says volatility is currently above its own forecast — expansion in
progress. $\Gamma_t$ near 1 says short-horizon range is unusually small relative
to longer horizons — a coiled state. Because volatility mean-reverts on top of
being persistent, high $\Gamma_t$ raises the hazard of an expansion without
saying anything about its direction. That asymmetry is used honestly in §5: high
compression modulates *sizing and target distance*, never direction.

### 2.3 Why this comes first

Every barrier in the system is set in $\sigma$ units and only then converted to
price (§7). Every cost is expressed in $\sigma$ units (§8). Every engine score
is computed from $\sigma$-normalised inputs. If $\hat\sigma$ is wrong, the
system is not "slightly mis-sized" — its entire notion of what constitutes a
large move, an expensive trade, and an acceptable loss is wrong simultaneously.
Volatility estimation is therefore treated as the foundation rather than as a
feature.

---

## 3. Layer 2 — the microstructure primitive

### 3.1 Liquidity-Adjusted Displacement Ratio

This is the system's core novel construct, and it exists to make one question
computable from OHLCV: *did this move cost a lot of liquidity, or a little?*

Kyle (1985) gives the canonical answer to why that question matters. In a market
with informed and uninformed flow, the equilibrium price impact per unit of
order flow, $\lambda$, is increasing in the informativeness of flow. Amihud
(2002) operationalises the reciprocal as an illiquidity measure. Both point at
the same dimensionless object: price move per unit of volume consumed.

Define, per bar:

$$d_t = \frac{r_t}{\sigma^{\mathrm{bp}}_t} \qquad\text{(displacement in jump-robust }\sigma\text{ units)}$$

$$p_t = \frac{\tilde V_t}{\mathrm{median}_{60}\big(\tilde V \mid \text{same time-of-day bucket}\big)} \qquad\text{(participation)}$$

where $\tilde V_t$ is dollar volume. The time-of-day conditioning is not
cosmetic: intraday volume follows a pronounced and extremely well-replicated
U-shape, so an unconditional volume comparison mostly measures what hour it is.

Under a square-root impact law — the form with the strongest empirical support
across markets and order sizes — price displacement from consuming quantity $Q$
scales as $\sqrt{Q}$. Inverting, the *liquidity cost per unit of achieved
displacement* is captured by

$$\boxed{\ \mathrm{LADR}_t = \mathrm{sign}(r_t)\cdot\frac{|d_t|}{p_t^{\,\psi}},\qquad \psi=\tfrac12.\ }$$

$\mathrm{LADR}$ is dimensionless, scale-free, and comparable across instruments
and eras by construction. Its interpretation:

* $|\mathrm{LADR}|$ **large** — price moved far on little consumed liquidity.
  Under Kyle, this is the signature of *informed* flow, whose impact is
  permanent. Continuation evidence.
* $|\mathrm{LADR}|$ **small with large $p_t$** — heavy volume produced little
  displacement. Someone absorbed it. Under Kyle, uninformed pressure met a
  liquidity provider; the impact is transient. Reversal evidence.

The second case deserves its own name, because it is the measurable core of what
practitioners call accumulation, distribution, and "effort versus result":

$$\mathrm{ABS}_t = \frac{p_t}{|d_t| + \epsilon}.$$

High $\mathrm{ABS}$ *at a structural level* is the only defensible formalisation
of Wyckoffian absorption available from bar data. It is not a claim about
institutional intent, which is unobservable; it is a statement that volume was
transacted without price consequence, which is observable.

The pair $(\mathrm{LADR}_t,\ \mathrm{ABS}_t)$ is a two-dimensional summary of
the bar's auction that unifies Kyle's $\lambda$, Amihud illiquidity, and
effort-versus-result into quantities with consistent units. Both are consumed as
causal ranks, $\mathrm{LADR}^{\mathrm{rank}}$ and $\mathrm{ABS}^{\mathrm{rank}}$,
so their marginal distributions are identical on every instrument.

**Honest limitations.** $p_t$ is total volume, not signed order flow; without a
footprint feed we cannot separate buy- from sell-initiated volume, so
$\mathrm{LADR}$ conflates "informed buying" with "informed selling of the
opposite sign" only insofar as $\mathrm{sign}(r_t)$ resolves it. Bar aggregation
destroys sequence information within the bar. And genuine order-flow imbalance
predictivity decays over seconds to minutes, so on bar data this is a
*confirming* measurement, not an independent alpha source. It is used as such.

### 3.2 Sweep and failure

Combining §3.1 with structure (§4.3) gives the measurable version of the
liquidity-event condition (B). A sweep-and-fail at a reference level $\Lambda$
requires, all within a window of $m$ bars and all evaluable from closed bars:

1. **Penetration.** $L_t < \Lambda - \delta$ for a long setup at a prior low,
   with $\delta$ a small multiple of $\sigma^{\mathrm{bp}}$ so that ordinary
   noise does not qualify.
2. **Anomalous participation.** $p_t$ in its upper decile — the level was not
   drifted through, it was *hit*.
3. **Absorption.** $\mathrm{ABS}^{\mathrm{rank}}_t$ high, or $\mathrm{CLV}_t$
   strongly opposed to the penetration direction: the bar gave back its
   excursion.
4. **Failure to hold.** $C_{t+j} > \Lambda$ for some $j \le m$, confirmed on a
   closed bar.

Condition 4 is what makes this non-repainting and also what makes it late. A
sweep is only knowable after it has failed. Systems that mark sweeps at the
moment of penetration are reading the future; the delay is not a deficiency of
this implementation but the actual information arrival time.

### 3.3 Effective spread without quote data

Costs must be estimated, not assumed, and the system may only use OHLCV. The
Corwin–Schultz (2012) estimator recovers the effective spread from the insight
that the high is more likely buyer-initiated and the low seller-initiated, so
observed ranges over one and two bars decompose into volatility plus spread:

$$\hat S_t = \frac{2(e^{\alpha_t}-1)}{1+e^{\alpha_t}},\qquad
\alpha_t = \frac{\sqrt{2\beta_t}-\sqrt{\beta_t}}{3-2\sqrt2}-\sqrt{\frac{\gamma_t}{3-2\sqrt2}}$$

with $\beta_t$ from consecutive single-bar log ranges and $\gamma_t$ from the
two-bar range. Negative estimates are set to zero and the result is averaged.
Crucially, the pair $(t-1,t)$ is used, never $(t,t+1)$: the estimator is
perfectly capable of being implemented with look-ahead and must not be.

---

## 4. Layer 3 — trend as a testable proposition

Most trend indicators output a *level* — a moving-average distance, a slope.
A level cannot be thresholded meaningfully because its scale depends on
volatility, which varies by an order of magnitude. The system therefore outputs
*significances*.

### 4.1 Kalman local linear trend

Model the log price as a level plus slope with Gaussian innovations:

$$x_t = \begin{pmatrix}\mu_t\\ \beta_t\end{pmatrix},\quad
x_t = \underbrace{\begin{pmatrix}1&1\\0&1\end{pmatrix}}_{F}x_{t-1}+\eta_t,\quad
\ln C_t = \underbrace{\begin{pmatrix}1&0\end{pmatrix}}_{H}x_t+\varepsilon_t$$

with $\eta_t\sim N(0,Q)$, $\varepsilon_t\sim N(0,R_t)$,
$Q=\mathrm{diag}(q_\mu,q_\beta)\,\hat\sigma^2_{t}$ and $R_t=\hat\sigma^2_t$.
Tying both covariances to the *forecast* volatility is what makes the filter
adaptive with no user settings: in high volatility the observation is noisier in
absolute terms, so the filter slows; in low volatility it sharpens. The only
free quantities are the two dimensionless ratios $q_\mu, q_\beta$, and they are
signal-to-noise ratios rather than periods, so one value serves every
instrument.

The standard forward recursion yields the filtered slope $\hat\beta_t$ and its
posterior variance $P^{\beta\beta}_t$. The output is not the slope but

$$T^\beta_t = \frac{\hat\beta_t}{\sqrt{P^{\beta\beta}_t}},$$

a t-statistic: *how confident is the filter that a trend exists at all?* This is
directly comparable across instruments, across volatility levels, and across
eras — which a slope is not. Only the forward pass is used. The Kalman smoother,
which is what most published "Kalman trend" examples actually plot, uses future
observations and is inadmissible here.

### 4.2 Variance ratio instead of Hurst

Trend persistence is a statement about the autocorrelation of increments. The
common tool is the Hurst exponent via rescaled range, but R/S estimators are
severely biased in samples of the size available and have no usable null
distribution, which makes any threshold on them arbitrary.

The Lo–MacKinlay (1988) variance ratio does have a null distribution. With
$q$-period aggregation,

$$\mathrm{VR}(q)=\frac{\mathrm{Var}\big(\sum_{j=0}^{q-1}r_{t-j}\big)}{q\,\mathrm{Var}(r_t)},$$

and under the martingale null $\mathrm{VR}(q)=1$. The heteroskedasticity-robust
test statistic is

$$z(q)=\frac{\mathrm{VR}(q)-1}{\sqrt{\theta(q)}}\ \xrightarrow{d}\ N(0,1),\qquad
\theta(q)=\sum_{j=1}^{q-1}\left[\frac{2(q-j)}{q}\right]^2\delta_j,$$

$$\delta_j=\frac{\sum_t (r_t-\bar r)^2(r_{t-j}-\bar r)^2}{\left[\sum_t (r_t-\bar r)^2\right]^2}.$$

The robust form matters enormously here: financial returns are strongly
heteroskedastic, and the homoskedastic variance ratio rejects the martingale
null constantly for that reason alone, producing spurious "trend" readings in
every volatility cluster.

We compute $z(q)$ for $q\in\{2,4,8,16\}$ and aggregate. Positive $z$ means
positively autocorrelated increments (persistence, condition C applies);
negative $z$ means mean reversion (condition B applies). An implied Hurst
$H=\tfrac12+\ln \mathrm{VR}(q)/(2\ln q)$ is computed for display only, flagged
as biased, and used in no decision.

### 4.3 Structure without repainting

Swing pivots are defined by a scale-free zigzag: a candidate high at bar $i$ is
confirmed at bar $i+c$ if no bar in $(i, i+c]$ exceeded it and price has since
retraced by at least $\kappa\,\sigma^{\mathrm{bp}}_i$. Each pivot stores
`confirmed_at`, and **every consumer may only read pivots with
`confirmed_at` $\le t$.** This single rule eliminates the most common source of
fictitious backtest performance in structure-based systems: the pivot that is
obvious in hindsight and invisible in real time.

The higher-timeframe view is constructed by running the same causal pivot
detector on aggregated bars, at multiples $5\times$ and $25\times$ the execution
timeframe. The multiples are derived, not user-set — the trader configures
nothing — and the aggregation is itself causal, using only completed
higher-timeframe bars.

---

## 5. Layer 4 — the regime posterior

### 5.1 Why regimes, and why only four

§0.1 established that conditions (B) and (C) issue opposite instructions. The
regime layer's only job is to say which is in force. It has four states because
four is what the evidence supports distinguishing:

| Regime | Character | Which condition applies |
|---|---|---|
| `TREND` | positively autocorrelated increments, efficient paths, cheap displacement | (C) — follow |
| `REVERT` | negatively autocorrelated, absorptive, range-bound | (B) — fade |
| `STRESS` | high volatility, jump-dominated, correlations converge | (B) pays more but tails are fatal; size down hard |
| `QUIET` | low volatility, thin participation, high entropy | neither; stand aside |

More states would be discoverable by clustering, and would be fitted noise.

### 5.2 A measurement model with no fitted parameters

This is the design's second novel element. Rather than fitting a Gaussian HMM to
returns — which requires estimating $O(K^2 + K J^2)$ parameters on the target
series, is notoriously unstable, and leaks badly if fitted on the full sample —
we *specify* the measurement model a priori and fit nothing to returns.

Take $J$ features, each already a rank in $(0,1)$. For regime $k$, assert

$$p(f\mid k)=\prod_{j=1}^{J}\mathrm{Beta}\big(f_j;\ a_{kj},\ b_{kj}\big),\qquad
a_{kj}=m_{kj}c_{kj},\quad b_{kj}=(1-m_{kj})c_{kj}.$$

Each cell of the design is one of four statements, encoded by $(m,c)$:

| Statement | $m$ | $c$ | Density |
|---|---|---|---|
| this feature reads high in this regime | 0.8 | 6 | $\mathrm{Beta}(4.8,1.2)$ |
| reads low | 0.2 | 6 | $\mathrm{Beta}(1.2,4.8)$ |
| reads middling | 0.5 | 6 | $\mathrm{Beta}(3,3)$ |
| this regime makes no claim | 0.5 | 2 | $\mathrm{Beta}(1,1)$ = uniform |

The last row is the elegant part: setting $c=2, m=0.5$ yields *exactly* the
uniform density, so "no opinion" is expressible within the same parametric
family and contributes a likelihood factor of exactly 1. The full design table
lives in `engines/regime.py` and is reproduced in `docs/03-ENGINES.md`.

Total parameters fitted to market outcomes in this layer: **zero**. The
concentration $c$ is a theory parameter; the stickiness below is estimated once
on a development universe and frozen.

### 5.3 Tempered likelihood

The features are correlated, so the product $\prod_j p(f_j|k)$ overstates the
evidence — the same error the fusion layer corrects in §6. We therefore temper:

$$\tilde p(f\mid k)=\Big[\prod_j p(f_j\mid k)\Big]^{\eta},\qquad
\eta=\frac{\mathrm{EBE}_f}{J},\qquad \mathrm{EBE}_f=\mathbf{1}^\top C_f^{-1}\mathbf{1},$$

with $C_f$ the causally-estimated feature correlation matrix. If the $J$ features
were independent, $\eta=1$ and nothing changes. If they are effectively three
independent measurements dressed as eight, $\eta=3/8$ and the posterior is
correspondingly less certain. The same correction appears twice in this system
because it addresses the same error twice: **counting correlated evidence more
than once is the dominant failure mode of multi-signal designs.**

### 5.4 Causal filtering

Forward recursion only:

$$\alpha_t(k)\ \propto\ \tilde p(f_t\mid k)\sum_{j}A_{jk}\,\alpha_{t-1}(j),\qquad
A=\varsigma I+\frac{1-\varsigma}{K-1}(\mathbf{1}\mathbf{1}^\top-I).$$

Stickiness $\varsigma$ implies mean dwell time $1/(1-\varsigma)$ bars; at
$\varsigma=0.985$ that is about 67 bars, consistent with documented volatility
regime persistence. Computation is in log space. **No smoothing.** The
Baum–Welch forward-backward posterior $p(k_t\mid f_{1:T})$ conditions on the
future and would repaint every historical regime label the moment a new bar
arrived. Only $p(k_t \mid f_{1:t})$ is admissible.

### 5.5 Transition hazard

Regime transitions are where regime-conditional strategies lose money, because
the strategy is still acting on the old regime while the market has moved to the
new one. The system detects and stands aside during them:

$$h_t=\max\Big(\underbrace{\tfrac12\textstyle\sum_k|\alpha_t(k)-\bar\alpha_t(k)|}_{\text{drift}},\ \underbrace{1-\max_k\alpha_t(k)}_{\text{ambiguity}}\Big)$$

where $\bar\alpha_t$ is an EWMA of past posteriors. Both terms are in $[0,1]$.
Drift catches a posterior in motion; ambiguity catches one that has never
committed. Entries require $h_t \le 0.35$, i.e. a dominant regime held with at
least 65% posterior mass and not currently in flux.

---

## 6. Layer 5 — correlation-aware fusion

### 6.1 The failure this layer exists to prevent

Take ten engines, each mildly bullish with score $s_i=0.5$. Map each to log-odds
$\ell_i = 2\,\mathrm{artanh}(0.5)=1.0986$. Sum them, as any "confluence score"
implicitly does:

$$\hat L_{\text{naive}} = \sum_i \ell_i = 10.99 \quad\Longrightarrow\quad p = \varsigma(10.99) = 0.99998.$$

A system reporting 99.998% confidence from ten mild opinions is not confident,
it is broken. Summation of log-odds is the correct Bayesian operation **only**
under conditional independence of the evidence. Ten technical engines built on
overlapping windows of the same price series are nowhere near independent, and
the error is not small — it is the difference between 75% and 99.998%.

This is, in our assessment, the single most consequential defect in
multi-indicator trading systems, and it is invisible in a backtest because the
inflated probability is monotone in the honest one: ranking is preserved, so
*hit rates look fine* while *position sizing and confidence gating are wildly
wrong*.

### 6.2 CALOP

Model each engine's evidence as a noisy observation of a common latent log-odds
$L$:

$$\ell = \mathbf{1}L + \varepsilon,\qquad \varepsilon\sim N(0,\Sigma),\qquad
\Sigma = \sigma_e^2\, D^{-1/2} C\, D^{-1/2},\quad D=\mathrm{diag}(\rho_1,\dots,\rho_m).$$

$C$ is the causally-estimated correlation matrix of engine scores (ridge-
regularised, §6.4); reliability enters as precision, so an unreliable engine is
a noisy observation rather than a muted one.

Write $M = D^{1/2}C^{-1}D^{1/2}$. Then define

$$\boxed{\ \hat L = \mathbf{1}^\top M\,\ell,\qquad
\mathrm{EBE} = \mathbf{1}^\top M\,\mathbf{1},\qquad
\mathrm{Var}(\hat L)=\sigma_e^2\,\mathrm{EBE}.\ }$$

**Proposition 1 (CALOP interpolates between the two correct answers).**
*If $C=I$ and all $\rho_i=1$, then $\hat L=\sum_i \ell_i$ — the Bayesian sum
under conditional independence. If two engines are perfectly correlated, they
contribute exactly one engine's worth of evidence.*

*Proof of the second claim.* For $m=2$ with correlation $\rho$ and unit
reliabilities, $C^{-1}=\frac{1}{1-\rho^2}\begin{pmatrix}1&-\rho\\-\rho&1\end{pmatrix}$,
so $\mathbf 1^\top C^{-1} = \frac{1}{1+\rho}(1,1)$ and
$\hat L = (\ell_1+\ell_2)/(1+\rho)$. At $\rho=0$ this is $\ell_1+\ell_2$; at
$\rho\to1$ it is $(\ell_1+\ell_2)/2$, the single shared opinion. Correspondingly
$\mathrm{EBE}=2/(1+\rho)$ falls from 2 to 1. $\square$

So $\mathrm{EBE}$ is literally *how many independent engine opinions the pooled
number rests on*. For equicorrelated engines,

$$\mathrm{EBE}=\frac{m}{1+(m-1)\bar\rho}.$$

**Ten engines at $\bar\rho=0.5$ yield $\mathrm{EBE}=1.82$.** Not ten opinions —
fewer than two. The overconfidence factor of the naive treatment is
$\sqrt{m/\mathrm{EBE}}=\sqrt{1+(m-1)\bar\rho}=2.35$ in standard-error terms, and
far worse in probability terms as §6.1 showed.

**Proposition 2 (connection to the fundamental law of active management).**
*$\hat L$ scales with $\mathrm{EBE}$ while $\mathrm{SD}(\hat L)$ scales with
$\sqrt{\mathrm{EBE}}$, so the evidence z-statistic
$z=\hat L/(\sigma_e\sqrt{\mathrm{EBE}})$ grows as $\sqrt{\mathrm{EBE}}$.* This is
Grinold's $\mathrm{IR}\approx \mathrm{IC}\sqrt{\text{breadth}}$ with
$\mathrm{EBE}$ as the *measured* breadth rather than an assumed count of
independent bets. The relation is not imposed; it falls out of the pooling
algebra.

### 6.3 From pooled evidence to a calibrated probability

A plug-in $p=\varsigma(\hat L)$ ignores that $\hat L$ is itself uncertain. The
Gaussian-integrated logistic (MacKay's probit approximation) gives the
predictive probability:

$$\boxed{\ p = \varsigma\!\left(\frac{T\hat L+b}{\sqrt{1+\tfrac{\pi}{8}T^2\mathrm{Var}(\hat L)}}\right)\ }$$

with $(T,b)$ fit once by maximum likelihood on the development universe. The
denominator is the mechanism by which the system's stated confidence is
automatically pulled toward 50% when evidence is narrow — and it means a
reported 97% requires evidence that is both *strong* and *broad*. The credible
interval reported to the trader is
$\varsigma\big(T(\hat L \pm z_q\,\mathrm{SD}(\hat L))+b\big)$.

Once $\ge 200$ realised outcomes exist, an isotonic recalibration (PAVA) is
fitted on strictly past data and composed on top, correcting any residual
monotone miscalibration without imposing a functional form. Calibration quality
is monitored live via the Brier decomposition
$\mathrm{BS} = \text{reliability} - \text{resolution} + \text{uncertainty}$;
deteriorating reliability demotes the system to `NO TRADE` (§13).

### 6.4 Estimating $C$ causally

$C$ is a rolling correlation of engine scores over `correlation_window` bars,
computed by a causal recursion, then ridge-regularised:
$C_\lambda = (1-\lambda)C + \lambda I$ with $\lambda=0.15$. The ridge is not
cosmetic. Near-collinear engines make $C$ ill-conditioned, and $C^{-1}$ then
produces enormous weights of opposing sign — the pooled estimate becomes a
difference of nearly identical large numbers, which is numerically and
statistically indefensible. The shrinkage bounds $\|M\|$ and keeps
$\mathrm{EBE}$ conservative, which is the direction we want to err in.

---

## 7. Layer 6 — the Edge Book

### 7.1 Purpose

Layers 1–6 produce a direction and a calibrated probability. They do not produce
an *expectancy*, because expectancy depends on the joint distribution of outcome
returns given the setup, not on a win probability alone. The Edge Book estimates
that distribution, and it is what the trade/no-trade gate actually acts on.

The design choice here is to prefer a small, interpretable hierarchical Bayesian
model over a high-capacity learner. With $\mathrm{IC}\approx0.04$, a gradient-
boosted ensemble will find structure that is not there. A hierarchical model with
strong priors will instead report *no edge* when the data is thin, which is the
correct answer far more often than not.

### 7.2 Cells and hierarchy

Index cells by $(k, s, b)$: regime, setup family, and evidence bucket (a
quantile bucket of $\hat L$). Setup families are the small set of qualitatively
distinct configurations the engines can produce — continuation, sweep-reversal,
compression-breakout, range-fade — and are named, not learned.

The hierarchy is $(k,s,b) \rightarrow (k,s) \rightarrow \text{global}$, with the
global root prior asserting **zero** expectancy. The null hypothesis is built
into the model: absent evidence, every cell believes there is no edge.

### 7.3 Normal-Inverse-Gamma with weighted, decayed counts

Let $\mathcal{R}$ be the realised outcome in $\sigma$ units, signed by
direction, from the triple-barrier evaluation (§8). Model
$\mathcal{R}\sim N(\mu,\varsigma^2)$ within a cell, with the conjugate prior
$\mu\mid\varsigma^2\sim N(\mu_0,\varsigma^2/\kappa_0)$,
$\varsigma^2\sim\mathrm{InvGamma}(a_0,b_0)$. Using sample weights $w_i$
(uniqueness $\times$ time decay, §8.3), with $n_w=\sum w_i$ and
$\bar{\mathcal R}_w = \sum w_i\mathcal R_i/n_w$:

$$\kappa_n=\kappa_0+n_w,\qquad
\mu_n=\frac{\kappa_0\mu_0+n_w\bar{\mathcal R}_w}{\kappa_n},\qquad
a_n=a_0+\tfrac{n_w}{2},$$

$$b_n=b_0+\tfrac12\sum_i w_i(\mathcal R_i-\bar{\mathcal R}_w)^2
+\frac{\kappa_0 n_w(\bar{\mathcal R}_w-\mu_0)^2}{2\kappa_n}.$$

The marginal posterior of $\mu$ is Student-$t$ with $2a_n$ degrees of freedom,
location $\mu_n$, scale $\sqrt{b_n/(a_n\kappa_n)}$. Hence a closed-form lower
credible bound:

$$\boxed{\ \mathrm{LCB}_q(\mu)=\mu_n+t_{q,\,2a_n}\sqrt{\frac{b_n}{a_n\kappa_n}}\ }$$

Three properties make this the right tool:

* **Thin cells say "no edge."** With $n_w \ll \kappa_0$, $\mu_n\to\mu_0=0$ and
  the interval is wide, so $\mathrm{LCB}_q<0$ and the gate refuses. No special-
  casing needed; the shrinkage does it.
* **All three barrier outcomes are handled uniformly.** Because we model the
  realised return rather than a win/loss label, target hits, stop hits and
  timeouts all enter naturally. A win-rate model has to special-case timeouts.
* **Sufficient statistics are three scalars per cell**, decayed exponentially,
  so updates are $O(1)$ and the whole book is a few kilobytes.

### 7.4 The gate acts on the lower bound

$$\text{trade only if}\quad \mathrm{LCB}_{0.10}(\mu) - c_{\text{round-trip}} > \tau.$$

The system acts on the pessimistic end of what it believes, net of modelled
costs. This is the mechanism that produces "95 exceptional trades instead of
1000 mediocre ones" — not a hand-set confidence threshold, but a decision rule
that requires 90% posterior credence that net expectancy is positive. It
tightens automatically when data is thin, when outcome variance is high, or when
costs rise, all of which are exactly the circumstances in which one should trade
less.

---

## 8. Layer 7 — targets, labels, costs

### 8.1 Triple barrier

Each hypothetical trade gets an upper barrier at $+\,\theta_{\text{tgt}}\hat\sigma$,
a lower at $-\,\theta_{\text{stop}}\hat\sigma$, and a vertical barrier at
$T_{\max}$ bars, all measured from the *execution* price (the open of
$t+\text{lag}$). Setting barriers in $\sigma$ units before converting to price
is what allows one parameter set to serve every instrument.

The vertical barrier is not a convenience. The conditioning information that
justified the trade decays; after $T_{\max}$ bars the position is no longer the
one that was analysed, and holding it is a different bet made silently.

**Pessimistic tie-breaks, always.** If a bar's range spans both barriers, the
stop is recorded as hit. If the bar opens beyond the stop, the fill is the open,
not the stop. Bar data cannot resolve intrabar sequence, so the resolution is
always chosen against us. This is a material effect, not a rounding detail: the
optimistic convention can manufacture most of a strategy's apparent edge.

### 8.2 Costs in $\sigma$ units

$$c_{\text{round-trip}} = 2\big(\hat S_t/2 + \eta\sqrt{q} + \phi\,\mathrm{Range}_t/\hat\sigma_t + \text{fees}\big)/\hat\sigma_t$$

with $\hat S_t$ the Corwin–Schultz spread (§3.3), $q$ the order size as a
fraction of bar dollar volume, $\eta\sqrt q$ the square-root impact law, and
$\phi$ a fraction of the execution bar's range charged as adverse slippage.
Expressing costs in $\sigma$ units puts them on the same footing as expectancy,
so the gate compares like with like.

### 8.3 Overlapping labels

Labels overlap: a trade opened at $t$ and closed at $t+20$ shares outcome
information with one opened at $t+5$. Treating them as independent inflates
every t-statistic in the validation suite, sometimes by a factor of three or
more. Two corrections are mandatory:

* **Uniqueness weights.** For each label, the mean of $1/\text{concurrency}$ over
  its span, where concurrency counts labels spanning each bar.
* **Purging and embargo.** Any training label whose span overlaps a test span is
  removed, plus a small embargo after each test block.

Both are López de Prado's, both are non-negotiable, and skipping them is the
most common way a leakage-free-looking pipeline leaks.

---

## 9. Layer 8 — risk

### 9.1 Kelly under parameter uncertainty

For approximately Gaussian outcomes the growth-optimal fraction is
$f^\star=\mu/\varsigma^2$. But $\mu$ is estimated, and plugging in $\mu_n$
overbets systematically. Integrating over the posterior of $\mu$ gives, to
second order,

$$f^\star_{\text{Bayes}} \approx \frac{\mu_n}{\varsigma_n^2+\mathrm{Var}(\mu_n)}.$$

Parameter uncertainty enters as additional variance — a clean statement of why
uncertain edges deserve smaller bets. The deployed fraction adds a fractional-
Kelly multiplier $\varphi=0.25$, a haircut by
$\mathrm{LCB}_q(\mu)/\mu_n$, and a hard cap:

$$f = \min\left(f_{\max},\ \varphi\cdot\frac{\max(\mu_n,0)}{\varsigma_n^2+\mathrm{Var}(\mu_n)}\cdot\frac{\mathrm{LCB}_q(\mu)}{\mu_n}\right)\cdot g(\mathrm{DD}_t).$$

### 9.2 Drawdown throttle

$g(\mathrm{DD})$ falls linearly from 1 to a floor of 0.25 between 4% and 15%
drawdown, and to 0 at 25%. The asymmetry is deliberate: a drawdown is weak
evidence that the model has degraded and strong evidence that our estimate of
the edge was too high. Reducing size is the correct response to both, and it is
also the only response that is robust to *not knowing which*.

---

## 10. The rank-transform discipline

Every quantity entering a decision is one of:

1. bounded and dimensionless by construction — $\mathrm{CLV}$, path efficiency,
   jump share, permutation entropy, regime posteriors;
2. a t-statistic with a known null — $T^\beta$, $z(q)$;
3. a causal rolling rank in $[0,1]$ — volume, volatility, $\mathrm{LADR}$,
   absorption.

The consequence is that the *marginal* distribution of every feature is
approximately the same on every instrument in every era, by construction. A
parameter set chosen on 1990s equity index futures is therefore meaningful on
2024 crypto perpetuals, not because the markets are alike but because the
features have been made comparable. This is what turns cross-market and
cross-decade testing into a genuine out-of-sample test rather than a disguised
refit.

**The honest cost.** Rank transforms discard magnitude: a 10$\sigma$ day and a
4$\sigma$ day both rank 1.0. In tail events that information matters, so a small
number of raw t-statistics and $\sigma$-unit quantities are carried alongside the
ranks, and the `STRESS` regime exists partly to handle the cases where the rank
representation has saturated. Ranks are also non-stationary in a second-order
sense — the *window* over which they are computed imposes an implicit assumption
that the last 252 bars are a relevant reference class, which is exactly wrong
immediately after a structural break.

---

## 11. Degrees of freedom, counted

The defence against overfitting is not the validation suite. It is that there is
little to overfit *with*, and that the count is auditable.

| Provenance | Count | Charged to DSR? |
|---|---|---|
| `THEORY` — set a priori, never tuned on outcomes | ~45 | No |
| `DEV` — fit once on the development universe, then frozen | **7** | **Yes** |
| `ONLINE` — updated by causal recursion during operation | ~5 | No (adds variance, not optimism) |
| `OPS` — account size, risk appetite, symbol list | ~10 | No |

The seven fitted parameters are: two Kalman signal-to-noise ratios, regime
stickiness, the calibration temperature, bias and evidence standard deviation,
and the impact coefficient.
Everything else that *looks* like a tuning knob — window lengths, barrier
multiples, regime design table, pooling strengths — is fixed by dimensional
argument or by published convention, and `config.py` records which is which and
hashes the manifest so that a report and a running system can be proven to
match.

A system with ten engines and seven fitted parameters is a fundamentally
different statistical object from one with ten engines and sixty.

---

## 12. Statistical power — the uncomfortable arithmetic

The brief asks for extreme selectivity: prefer 95 exceptional trades to 1000
mediocre ones. That is correct as a *trading* policy and catastrophic as an
*inference* policy, and the tension has to be faced explicitly.

With $n=95$ trades and an observed hit rate of 60%, the standard error is
$\sqrt{0.6\cdot0.4/95}=5.0$ percentage points. The 95% confidence interval is
$[50\%,\,70\%]$ — **95 trades cannot distinguish a real 60% edge from a coin
flip.** Detecting a 10-point edge at 80% power and 5% one-sided significance
needs about 155 trades. Establishing a per-trade expectancy of $0.15\sigma$
against a per-trade outcome standard deviation of $1.5\sigma$ at $t=2$ needs

$$n \ge \left(\frac{2\times1.5}{0.15}\right)^2 = 400 \text{ trades}.$$

At four trades per hundred bars, 400 trades is 10,000 bars — roughly 40 years of
daily data on a single instrument. No amount of cleverness escapes this
arithmetic.

**The resolution is breadth, not history.** Selectivity is applied *per
instrument*; inference is pooled *across* instruments. Fifty instruments over
twenty years at four trades per hundred bars gives on the order of $10^4$
trades — enough power to establish the effect and to break it down by regime.
Each individual instrument still trades rarely, which is what the trader
experiences.

This is why the Edge Book is hierarchical and pooled across the universe rather
than fitted per symbol, why every feature is rank-normalised so that pooling is
legitimate, and why the validation protocol is cross-sectional. The
architecture is shaped by this constraint more than by any other single
consideration.

**Corollary, stated plainly:** a single-instrument backtest of this system, or of
any system this selective, is *statistically uninformative* about whether it
works. It can only falsify (by blowing up), never confirm. Anyone evaluating
this framework on one symbol and a few hundred bars is measuring noise, and the
validation suite is designed to make that impossible to do accidentally.

---

## 13. What the trader sees, and why that is enough

Five actions, a confidence figure with an interval, a stop, a target, and a
nine-row explanation card. The interval is not decoration: it is the visible
output of §6.3's variance term, and a wide interval is the system telling the
trader that its engines agree for a reason that may be a single reason.

Everything else — the posteriors, the pooled log-odds, the Edge Book cell counts,
the throttles — is internal, logged, and monitored, but never presented as
something to interpret. A system that requires the user to weigh eight
sub-indicators has not reduced their decision problem; it has renamed it.

---

## 14. Where this can fail

Enumerated properly in `docs/08-FAILURE-MODES.md`. The four that most threaten
the theory itself:

1. **The regime taxonomy is wrong.** Four regimes may not carve the space where
   the (B)/(C) distinction actually lies. Then the top of the architecture
   misroutes evidence and both engines contribute noise.
2. **$\mathrm{LADR}$'s denominator is unsigned.** Total volume is a weak proxy
   for consumed liquidity. In markets where volume and liquidity decouple —
   sweeps across venues, auction prints, index rebalances — the primitive
   misreads.
3. **Correlation estimation lags.** $C$ is estimated on a trailing window. In a
   crisis, engine correlations converge within days while the window takes
   hundreds of bars to notice, so $\mathrm{EBE}$ is overstated exactly when
   overconfidence is most expensive. The `STRESS` regime and the hazard gate are
   partial mitigations; they are not a fix.
4. **The reference class assumption.** Rank windows and Edge Book cells assume
   the recent past is a relevant reference class for the present. After a
   genuine structural break it is not, and the system will be confidently wrong
   for as long as the window takes to refill.

None of these is hypothetical. Each has a specific monitoring signature in
`docs/13-MONITORING.md`, and the honest response to all four is the same:
trade smaller, and be willing to stop.
