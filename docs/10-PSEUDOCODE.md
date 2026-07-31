# Pseudocode

Language-independent reference for the per-bar computation. Enough to reimplement
without reading Python. Notation follows `docs/01-THEORY.md`.

---

## Top level

```
function ON_BAR(bar, exog):
    # -- L0 validate -----------------------------------------------------
    issue ← VALIDATE(bar)                      # monotone time, OHLC sanity, splits
    if issue is fatal: return NO_TRADE

    # -- L1 features -----------------------------------------------------
    f ← FEATURE_KERNEL.update(bar)             # see below
    EDGE_BOOK.tick()                           # advance the forgetting clock

    # -- L2a regime ------------------------------------------------------
    state ← REGIME.step(f.rank_features)
    for each engine: engine.regime_probs ← state.posterior

    # -- L2b engines -----------------------------------------------------
    ctx ← (bar, f, exog, position, bars_in_position)
    outputs ← { e.name: e.update(ctx) for e in engines }

    # -- L3 fusion -------------------------------------------------------
    CALOP.observe(outputs)                     # every bar, not only signal bars
    fusion ← CALOP.fuse(outputs, calibrator)

    # -- resolve any open position BEFORE evaluating a new one ------------
    RESOLVE_BARRIERS(bar)

    # -- L4 expectancy ---------------------------------------------------
    setup  ← CLASSIFY_SETUP(outputs, state)
    bucket ← BUCKET_OF(fusion.log_odds)
    edge   ← EDGE_BOOK.estimate(state.dominant, setup, bucket)

    # -- L5 costs, barriers, gate ----------------------------------------
    cost ← COST_MODEL(f, spread := exog.spread ?? f.spread_estimate,
                      participation, stress_multiplier := 1 + 1.5·p_stress)
    spec ← PLAN_BARRIERS(fusion.direction, bar.close, f, state)
    risk ← SIZE(edge, spec, cost, equity, throttle := LIMITS.throttle())
    action, gate ← POLICY.step(ctx, fusion, state, edge, cost, spec, risk, outputs)

    return DECISION(action,
                    decided_at   := t,
                    execute_at   := t + execution_lag_bars,   # never 0
                    target := spec, fusion, risk, cost,
                    card := BUILD_CARD(...))
```

---

## L1 — feature kernel

```
function FEATURE_KERNEL.update(bar):
    r ← ln(bar.close / prev_close)

    # --- volatility (THEORY 2) ---
    RV  ← rolling_sum(r², 22) / 22
    BPV ← (π/2) · rolling_sum(|r_t|·|r_{t-1}|, 22) / 21
    σ_bp ← sqrt(BPV);  σ_rv ← sqrt(RV)
    jump_share ← clip((σ_rv² − σ_bp²) / σ_rv², 0, 1)
    RS ← ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)
    σ_rs ← sqrt(mean(RS, 22))
    σ_ew ← sqrt(EWMA(r², halflife=10))
    σ_5, σ_66 ← sqrt(mean(r², 5)), sqrt(mean(r², 66))
    σ̂ ← exp(0.5·ln σ_5 + 0.3·ln σ_rv + 0.2·ln σ_66)     # geometric HAR blend
    vol_rank    ← CAUSAL_RANK(ln σ̂, 252)
    compression ← 1 − CAUSAL_RANK(σ_5/σ_66, 252)
    vol_ratio   ← σ_ew / σ̂

    # --- microstructure (THEORY 3) ---
    dv ← ((H+L+C)/3) · V
    p  ← dv / median(dv | same time-of-day bucket, 60)      # removes the U-shape
    d  ← r / σ_bp
    LADR ← sign(r) · |d| / p^0.5                             # square-root impact
    ABS  ← p / (|d| + ε)
    ladr_rank, abs_rank, part_rank ← CAUSAL_RANK(·, 252)

    # Corwin-Schultz, on bars (t-1, t) ONLY. Never (t, t+1).
    β ← ln(H_{t-1}/L_{t-1})² + ln(H_t/L_t)²
    γ ← ln(max(H_{t-1},H_t) / min(L_{t-1},L_t))²
    α ← (√(2β) − √β)/(3−2√2) − √(γ/(3−2√2))
    S ← rolling_MEDIAN(max(2(e^α−1)/(1+e^α), 0), 22)         # median, not mean

    # --- trend (THEORY 4) ---
    KALMAN_LLT(ln C, σ̂)                                      # forward pass only
    slope_t ← β̂ / sqrt(P_ββ)                                 # a t-statistic
    for q in {2,4,8,16}:
        VR(q) ← Var(Σ_{j<q} r_{t-j}) / (q·Var(r))
        θ(q)  ← Σ_{j<q} [2(q−j)/q]² · δ_j                     # heteroskedasticity-robust
        z(q)  ← (VR(q) − 1)/√θ(q)
    vr_z ← mean over q
    efficiency ← |Σ r| / Σ|r|   over 22
    perm_entropy ← Bandt-Pompe order 3, window 60, normalised

    # --- structure (THEORY 4.3) ---
    PIVOTS.update(bar, σ_bp)     # confirmed only after `pivot_confirm_bars`
    range_pos ← (C − pivot_low) / (pivot_high − pivot_low)
    SWEEPS.update(...)           # penetration + volume + give-back, THEN reclaim
    HTF1, HTF2 ← causal aggregation at 5x and 25x

    f.valid ← (bars_seen ≥ warmup) AND all required ranks present
```

---

## L2a — regime posterior

```
function REGIME.step(rank_features):
    # η = EBE_features / J, capped at 1: tempering can soften the likelihood,
    # never sharpen it. Correlated features must not be counted as independent.
    η ← clip(1ᵀC_f⁻¹1 / J, 0.15, 1.0)

    for k in {TREND, REVERT, STRESS, QUIET}:
        loglik[k] ← η · Σ_j  BetaLogPdf(f_j ; a_kj, b_kj)
        # a = m·c, b = (1−m)·c. m = 0.5, c = 2 gives Beta(1,1) = uniform,
        # which encodes "this regime makes no claim about this feature".

    # Sticky FORWARD recursion in log space. No smoothing: forward-backward
    # would repaint every historical label on every new bar.
    for k:
        pred[k] ← logsumexp_j( logα[j] + logA[j][k] )
    logα ← normalise(pred + loglik)
    α ← exp(logα)

    ᾱ ← EWMA(α, halflife = 20)
    drift     ← ½ Σ_k |α_k − ᾱ_k|
    ambiguity ← 1 − max_k α_k
    hazard    ← max(drift, ambiguity)          # either alone is reason to stand aside
```

---

## L3 — CALOP fusion

```
function CALOP.fuse(outputs):
    for each engine i:
        e_i ← 2·artanh(clip(score_i, ±0.999))   # inverse of the engines' tanh
        ρ_i ← reliability_i

    C ← rolling correlation of engine SCORES, ridge-shrunk by fusion_shrink
    D ← diag(ρ)
    M ← D^½ · C⁻¹ · D^½

    L̂     ← 1ᵀ M e                              # = Σ e_i when C = I
    EBE   ← 1ᵀ M 1                              # independent opinions actually present
    Var   ← σ_e² · EBE
    dir   ← sign(L̂)

    # Aligned breadth: only engines supporting the pooled direction. Total
    # breadth includes confident abstainers, which rightly widen the interval
    # but must not satisfy a threshold about agreement.
    S ← { i : ρ_i > 0 AND |score_i|·ρ_i > align_eps AND sign(score_i) = dir }
    EBE_aligned ← 1ᵀ (D^½ C_S⁻¹ D^½)|_S 1

    # MacKay probit correction: stated confidence shrinks toward 50% when
    # evidence is narrow. A reported 97% needs evidence both strong AND broad.
    κ ← 1 / sqrt(1 + (π/8)·T²·Var)
    p ← σ(T·L̂ + b, scaled by κ)
    p ← ISOTONIC(p)   if ≥ 200 resolved outcomes
    report p for the PROPOSED DIRECTION, not "for up"
```

---

## L4 — Edge Book (leave-one-out hierarchical NIG)

```
function EDGE_BOOK.estimate(regime, setup, bucket):
    leaf ← cells[(regime, setup, bucket)]
    node ← nodes[(regime, setup)]
    decay leaf, node, root to the current clock          # lazy, exact

    # LEAVE-ONE-OUT: a cell's prior must not be informed by its own data.
    root_excl ← root − node
    node_excl ← node − leaf

    root_post ← NIG_POSTERIOR(root_excl, μ0 := 0, κ0 := edge_prior_strength)
    node_post ← NIG_POSTERIOR(node_excl, μ0 := root_post.mean, κ0 := pooling)
    return      NIG_POSTERIOR(leaf,      μ0 := node_post.mean, κ0 := pooling)

function NIG_POSTERIOR(cell, μ0, κ0):
    n, s1, s2 ← cell.weighted_count, cell.weighted_sum, cell.weighted_sumsq
    κ_n ← κ0 + n
    μ_n ← (κ0·μ0 + s1) / κ_n
    a_n ← a0 + n/2
    b_n ← b0 + ½(s2 − s1²/n) + κ0·n·(s1/n − μ0)² / (2κ_n)
    scale ← sqrt(b_n / (a_n·κ_n));   dof ← 2·a_n
    lcb   ← μ_n + t_quantile(0.10, dof) · scale        # the gate acts on THIS
```

---

## L5 — the gate and the state machine

```
function POLICY.step(...):
    if position ≠ FLAT:
        if bars_held ≥ max_holding: return SELL/COVER, "vertical barrier"
        if bars_held ≥ min_holding
           AND fusion.direction = −position
           AND fusion.p ≥ p_exit
           AND fusion.EBE_aligned ≥ 0.7·ebe_min:
            return SELL/COVER, "reversal"
        if session not tradeable: return SELL/COVER
        return NO_TRADE

    # Ten CONJUNCTIVE conditions. Not a weighted score: a spectacular reading on
    # one axis must not buy past a disqualifying reading on another.
    pass ← (edge.lcb − cost.round_trip > ev_lcb_min_sigma)   # ← the primary gate
       AND (fusion.p ≥ p_min)
       AND (fusion.EBE_aligned ≥ ebe_min)
       AND (regime.hazard ≤ regime_hazard_max)
       AND (regime.dominant ≠ QUIET)
       AND features.valid AND enough engines valid
       AND (entries in last 100 bars < max_trades_per_100_bars)
       AND (risk.fraction > 0 AND no kill switch)
       AND (position = FLAT)
       AND (session tradeable AND not an auction)

    if pass: return BUY if direction > 0 else SHORT
    else:    return NO_TRADE with every failed condition named
```

## Barrier resolution — pessimistic throughout

```
function RESOLVE_BARRIERS(bar):
    if no open position: return
    hit_stop   ← (dir > 0 ? bar.low ≤ stop  : bar.high ≥ stop)
    hit_target ← (dir > 0 ? bar.high ≥ target : bar.low ≤ target)

    if hit_stop:                      # STOP WINS a same-bar tie, always
        gapped    ← (dir > 0 ? bar.open < stop : bar.open > stop)
        exit_px   ← gapped ? bar.open : stop      # gap fills at the OPEN
        outcome   ← −1
    else if hit_target:
        exit_px ← target;  outcome ← +1

    # Bar data cannot resolve intrabar sequence, so the ambiguity is always
    # resolved against us. The optimistic convention can manufacture most of a
    # strategy's apparent edge, invisibly.
```
