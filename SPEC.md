# TI-A Engineering Contract

Normative interface spec. Every module and every port (Python, Pine) must
conform. If code and this document disagree, the code in
`python/src/tia/types.py` and `python/src/tia/config.py` wins and this document
is the bug.

---

## 1. Naming, units, conventions

| Suffix | Meaning | Range |
|---|---|---|
| `_rank` | causal rolling empirical quantile over `Config.rank_window` | `[0, 1]` |
| `_z` | robust standard score, `(x - median) / (1.4826 * MAD)` | unbounded |
| `_t` | a t-statistic: estimate divided by its own standard error | unbounded |
| `_sigma` | measured in per-bar return standard deviations | unbounded |
| `_pct` | simple percentage return | unbounded |
| `_ann` | annualised | unbounded |

* Returns are natural logs: `r_t = ln(C_t / C_{t-1})`.
* `sigma` with no suffix is **per-bar** log-return standard deviation on the
  execution timeframe.
* Engine `score ∈ [-1, +1]`; sign is direction (positive = long); `0.0` means
  *no opinion*, which is distinct from a weak opinion.
* `reliability ∈ [0, 1]` answers "should you listen to me at all right now",
  never "which way".
* Probabilities are always stated **for the proposed direction**, never "for up".
* Missing data lowers `reliability` toward zero. It never raises, and it never
  substitutes a neutral reading — a neutral reading is an opinion.

## 2. Causality — the non-negotiable rules

1. **Streaming engines.** Every engine implements
   `update(ctx: BarContext) -> EngineOutput`, called exactly once per closed
   bar, in order. An engine holds its own state. It is never handed a sequence,
   an index into the future, or a DataFrame.
2. **Closed bars only.** `Bar.partial == True` bars are dropped by the data
   layer and never reach an engine.
3. **Decide at `t`, execute at `t + execution_lag_bars`.** A `Decision` carries
   both indices. The backtester and the live adapter honour the same two fields.
   `execution_lag_bars` is never 0.
4. **Confirmed structure only.** A swing pivot at bar `i` becomes visible at
   `i + pivot_confirm_bars`. Consumers read `pivot_*_age`, which is measured
   from confirmation, not from occurrence. This delay is the price of not
   repainting and is paid explicitly.
5. **No full-sample statistics.** Every mean, quantile, correlation and
   calibration map is computed from a trailing window or a causal recursion.
   Grep test: no call to a non-rolling `mean`, `std`, `quantile`, `corr`, `min`
   or `max` over a whole series inside `features/`, `engines/`, `fusion/`,
   `decision/`, or `risk/`.
6. **Prefix consistency is tested.** `tests/test_causality.py` replays random
   prefixes and requires bit-identical output at the shared final bar. Any
   module that fails this is broken, whatever else it does.
7. **Pessimistic tie-breaks.** If both barriers fall inside one bar, the **stop**
   is assumed hit first. If a stop gaps through, the fill is the *worse* of the
   stop price and the bar's open.

## 3. The five outputs

`Action` is the only user-facing vocabulary, with single-position semantics:

```
FLAT  --BUY-->   LONG    LONG  --SELL-->  FLAT
FLAT  --SHORT--> SHORT   SHORT --COVER--> FLAT
anything --NO_TRADE--> unchanged
```

`LONG → SHORT` in one bar is forbidden. A reversal costs two bars: `SELL`, then
possibly `SHORT`. This is both realistic and a whipsaw brake.

## 4. Module map and ownership

```
python/src/tia/
  types.py          contract: Bar, FeatureSnapshot, BarContext, EngineOutput,
                    FusionResult, TargetSpec, CostEstimate, RiskDecision,
                    Decision, ExplainCard, Engine, Action, Position, Regime
  config.py         frozen manifest, Provenance tags, manifest_hash()
  data/
    validate.py     bar sanity, gap and split detection, monotonic time
    sessions.py     SessionPhase tagging, time-of-day fraction, day of week
    clocks.py       equal-dollar-volume and equal-volume bar construction
  features/
    scales.py       bipower variation, Rogers-Satchell, Garman-Klass, EWMA,
                    HAR-style forecast, compression, vol-of-vol
    jumps.py        RV/BPV jump decomposition, jump share
    ranks.py        CausalRank, RobustZ, per-time-of-day normalisers
    microstructure.py  LADR, absorption, Amihud, Kyle proxy, Corwin-Schultz
                    spread, CLV smoothing, delta normalisation
    structure.py    causal ZigZag pivots with confirmation, range position,
                    nested higher-timeframe views, sweep detection
    trend.py        Kalman local-linear-trend, Lo-MacKinlay variance ratio,
                    path efficiency, multi-horizon sign agreement
    entropy.py      permutation entropy, run-length asymmetry, kurtosis
    kernel.py       FeatureKernel: one streaming object producing FeatureSnapshot
  engines/
    base.py         BaseEngine helper: warmup bookkeeping, score clamping
    regime.py       RegimeEngine  -> posterior over Regime + shift hazard
    structure.py    StructureEngine
    trend.py        TrendEngine
    momentum.py     MomentumEngine (displacement / LADR)
    liquidity.py    LiquidityEngine (sweep, absorption, illiquidity)
    volatility.py   VolatilityEngine (compression -> expansion, forecast)
    behavioral.py   BehavioralEngine (session, day, event proximity)
    crossasset.py   CrossAssetEngine (peers, implied vol) — optional
    positioning.py  PositioningEngine (gamma, OI) — optional, default OFF
    edgebook.py     EdgeBookEngine (hierarchical conditional expectancy)
  fusion/
    calop.py        correlation-aware log-odds pooling, effective breadth
    calibration.py  Platt link, isotonic (PAVA), reliability diagram, Brier
  labeling/
    triple_barrier.py  barrier outcomes with pessimistic tie-breaks
    weights.py      sample uniqueness and time-decay weights
  decision/
    barriers.py     TargetSpec construction from the volatility forecast
    costs.py        CostEstimate assembly
    policy.py       EV gate, hysteresis, state machine, rate limiter
    explain.py      ExplainCard rendering
  risk/
    sizing.py       fractional Kelly with uncertainty haircut, vol targeting
    limits.py       drawdown throttle, kill switches
  validation/
    cpcv.py         combinatorial purged cross-validation with embargo
    walkforward.py  anchored and rolling walk-forward
    montecarlo.py   stationary block bootstrap, GARCH surrogate nulls,
                    trade-order permutation
    stress.py       noise injection, timing shift, ablation, parameter plateau
    metrics.py      PSR, DSR, expectancy, tail metrics, regime breakdown
    report.py       text report assembly
  monitoring/
    drift.py        rolling Brier, PSI, slippage tracking, auto-demotion
  pipeline.py       TIA: end-to-end streaming runner
  backtest.py       execution simulator honouring §2 rules
  cli.py            command line entry points
```

## 5. Engine construction contract

```python
class MyEngine(BaseEngine):
    name = "my_engine"          # snake_case, stable, used as a dict key
    warmup = 120                # bars before valid=True is permitted

    def __init__(self, cfg: Config) -> None: ...
    def reset(self) -> None: ...
    def update(self, ctx: BarContext) -> EngineOutput: ...
```

Requirements:

* Deterministic. No wall clock, no unseeded RNG, no dict-ordering dependence.
* `update` must be O(1) or O(window) per call, never O(history).
* While `ctx.features.valid` is `False`, return `EngineOutput.abstain(name)`.
* `features` in the output holds only quantities that could be shown to a human;
  `diagnostics` holds internals for monitoring.
* `notes` are short lowercase phrases, e.g. `"swept 20-bar low and failed"`.

## 6. Fusion contract

Given `outputs: dict[str, EngineOutput]`:

1. Map each score to a log-odds contribution
   `l_i = atanh(clip(s_i, ±0.999)) * 2 * rho_i`.
2. Pool with GLS weights against the engine correlation matrix `C`
   (ridge-regularised by `fusion_shrink`):
   `w = C⁻¹1 / (1ᵀC⁻¹1)`, `L̂ = wᵀl`, `Var(L̂) = σ̄² / (1ᵀC⁻¹1)`.
3. Effective breadth of evidence `EBE = 1ᵀC⁻¹1`. Equals the engine count when
   engines are independent, tends to 1 when they are duplicates.
4. `p = sigmoid(T·L̂ + b)`, then isotonic recalibration once
   `calibration_min_samples` outcomes exist.
5. Propagate `Var(L̂)` into `p_low`/`p_high`.

Rationale is in `docs/05-CONFIDENCE-CALIBRATION.md`. The key claim: naive
log-odds summation, which is what "confluence" scoring implicitly does, is
optimal **only** for conditionally independent experts, and overstates
confidence by roughly `sqrt(m / EBE)` when they are not.

## 7. Decision gate — all conditions are conjunctive

An entry is emitted only when **every** one of these holds:

| # | Condition | Config key |
|---|---|---|
| 1 | `EV_lcb_net > ev_lcb_min_sigma` | `ev_lcb_min_sigma` |
| 2 | `p_success >= p_min` | `p_min` |
| 3 | `EBE >= ebe_min` | `ebe_min` |
| 4 | regime shift hazard `<= regime_hazard_max` | `regime_hazard_max` |
| 5 | regime is not `QUIET` | — |
| 6 | features valid, all required engines valid | — |
| 7 | trade rate under the limiter | `max_trades_per_100_bars` |
| 8 | risk layer returns non-zero size, no kill switch | `risk/limits.py` |
| 9 | position is `FLAT` | — |
| 10 | session phase tradeable, not an auction | `sessions.py` |

Condition 1 is the primary gate: expectancy at the **lower** credible bound,
net of modelled costs. Everything else is a filter on top of it.

## 8. Explainability card — fixed rows

```
Trend Alignment        <pct or n/a>
Liquidity              <Confirmed | Absorbing | None>
Momentum Expansion     <pct>
Volatility             <Favourable | Elevated | Compressed>
Regime                 <Trending | Mean-Reverting | Stress | Quiet>
Institutional Flow     <Accumulation | Distribution | Neutral>
Evidence Breadth       <EBE, 1 decimal>
Confidence             <pct, with credible interval>
Expected R:R           <ratio>
```

New engines map into existing rows. The card does not grow with the model.

## 9. Optional data contracts

Modules consuming `ExogenousSnapshot` must (a) default to disabled or
zero-reliability, (b) never be required by conditions 1–10 of §7, (c) document
their data source's known deficiencies in `docs/09-ASSUMPTIONS.md`.

## 10. Dependency policy

* Hard dependency: `numpy` only.
* Optional: `pandas` (IO convenience), `scipy` (optimisers, distributions),
  `sklearn`/`statsmodels` (validation cross-checks only).
* Every optional import is guarded and has a pure-Python or numpy fallback.
  Nothing in `features/`, `engines/`, `fusion/`, `decision/` or `risk/` may
  require an optional dependency, because those modules are what gets ported to
  a live trading environment.

## 11. Style

* `from __future__ import annotations` at the top of every module.
* Full type annotations on public functions.
* Docstrings state the *why* and cite the source of any formula.
* Comments only where the code cannot show the constraint itself.
* No print statements outside `cli.py` and `validation/report.py`.
