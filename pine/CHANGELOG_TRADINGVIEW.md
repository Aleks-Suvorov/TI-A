# TradingView port — repair log

Every defect found in the previous `pine/` build, what it did, and what
replaced it; then every Python feature this port omits or approximates.

**Verification status is stated precisely and is not overstated: no file in
this repository has been compiled by TradingView.** No Pine compiler exists in
the environment these changes were made in. What *was* done is set out in
[Verification](#verification).

---

## Part 1 — compile-time defects

### C1. `request.security()` given a mutable-variable expression

```pine
htfSlopeRaw = request.security(syminfo.tickerid, htfStr, slopeT[1], lookahead=barmerge.lookahead_off)
htfRangeRaw = request.security(syminfo.tickerid, htfStr, rangePos[1], lookahead=barmerge.lookahead_off)
```

`slopeT` and `rangePos` were assigned inside `if` blocks, making them mutable
variables. Pine rejects mutable-variable expressions in `request.security()`
outright — this alone made the file uncompilable.

**Fixed by removing `request.security()` entirely.** The higher timeframe is
now aggregated inside the script: five closed chart bars are accumulated into
one higher-timeframe bar, which drives its own Kalman filter and its own
high/low rings. This is the same causal `HTFAggregator` the Python reference
uses, so the two agree by construction, and it cannot look ahead because the
higher-timeframe values only update when a higher-timeframe bar *completes*.

### C2. Invalid timeframe string on non-intraday charts

```pine
htfStr = str.tostring(htfTF) + (timeframe.isintraday ? "" : "D")
```

On a weekly chart this produced `"5D"`, and on a monthly chart worse. Not a
valid resolution. Removed along with C1 — there is no timeframe string left to
build.

### C3. Parameter shadowing the `str` namespace

```pine
f_parseFloats(str) =>
    parts = str.split(str, ",")     // `str` is now the parameter, not the namespace
```

A parameter named `str` shadows the built-in namespace, so `str.split` stops
resolving for the rest of the function. Renamed to `csv`. A rule in
`python/tools/pinelint.py` now fails the build on any parameter or local that
shadows any of 40 built-in namespaces.

### C4. Comma-separated multiple assignment (4 sites)

```pine
sweepDir := 1, sweepAge := 0, pendLoAge := 999
```

Valid Python, invalid Pine. Split onto separate lines; the linter checks for
this across both files.

### C5. Series-qualified lengths in `math.sum`

```pine
math.sum(ret, q)     // q is a loop variable
```

`math.sum` requires a *simple int* length. The four variance-ratio horizons are
now four literal calls. The linter now parses the argument list of 25 `ta.*`
and `math.*` calls and rejects any length that is not a literal, an input, or
a name bound to a literal and never reassigned.

### C6. Untyped function parameters

Every helper took untyped parameters, leaving Pine to infer a qualifier it
cannot always get right at the call site. All 11 helpers now declare explicit
types (`f_clip(float x, float lo, float hi)`). The linter now treats an
untyped parameter as an **error**, not a style note: TradingView rejected
`f_safediv(dollarVol, medVol, na)` against the old untyped signature with
*CE10189 — the argument "d" should be explicitly typified*, so typing every
parameter is what makes that whole class unreachable.

### C7. `//@version=6` not on the first line

It sat below a 30-line comment header. TradingView scans the opening lines for
the version annotation and silently falls back to **Pine v1** when it does not
find one, which produces a cascade of errors that name none of the real cause.
Moved to line 1 in both files; a test asserts it.

### C8. `math.tanh` does not exist in Pine — and failed silently

Found by compiling on TradingView (NQ1!, 5m). Pine's `math` namespace has
`tan` but **no hyperbolic functions at all** — no `tanh`, `sinh` or `cosh`.

What made this expensive is not the error itself but its blast radius. The
direct complaint was one line of CE10271 (*Could not find function or function
reference 'math.tanh'*). But every value downstream of a missing call takes
type **"unknown"**, and that propagated the whole length of the script:

```
math.tanh  ->  regScore / trScore / volScore  ->  bScore  ->  bDir
           ->  propDir  ->  buySig, sellSig, planStop, planTarget, propConf
```

so TradingView reported **seven** errors, six of them CE10122 complaints about
`str.format` arguments a hundred lines away in the alert block. The tell that
they were all one cause: in `str.format(..., str.tostring(close, ...), ...)`
argument 4 was accepted because `close` is a built-in, while arguments 5, 6 and
7 were rejected because `planStop`, `planTarget` and `propConf` all trace back
through `propDir`.

Fixed with an `f_tanh` helper, `(e^{2z} - 1)/(e^{2z} + 1)` with the argument
clipped to +/-20, which agrees with the true tanh to one ULP (max absolute
error 2.2e-16 over x in [-100, 100]) — so no measured behaviour changed.

Two guards were added so no invented built-in can ship again:

* `pinelint.py` now carries the **complete v6 function set** for `math`, `str`,
  `ta`, `array` and `table`, and rejects any call outside it with a
  did-you-mean hint. Re-introducing `math.tanh` now fails locally with
  *'math.tanh' is not a Pine v6 built-in; did you mean tan?*
* the alert bodies were rewritten as **string concatenation** instead of
  `str.format`. `str.format` resolves against a typed overload set, which is
  what turned one root cause into six misleading errors; `+` on strings has no
  overloads to resolve.

### C9. Continuation-line indentation

Pine distinguishes a wrapped line from a new local block purely by indentation:
a continuation must **not** be indented by a multiple of four spaces. Getting
this wrong produces an error pointing somewhere unrelated. `pinelint.py` now
checks it.

### C10. `for` counts downward when the end value is below the start

Found while auditing for the next compile round, not by the compiler. Pine's
`for a to b` **decrements** when `b < a`. So the ordinary-looking

```pine
for k = 0 to array.size(BUCKET_EDGE) - 1
```

does not "do nothing" on an empty array — it becomes `for k = 0 to -1` and
iterates `k = 0`, then `k = -1`, reading index 0 of an empty array and raising
a runtime error on the chart. Both `array.size`-bounded loops in the port are
now wrapped in an explicit size guard, with the reason written next to them.

This one never fired, because the arrays are populated from the generated
constants block and are never empty in practice. It would have fired on the one
occasion it mattered: a malformed or truncated frozen-model block, which is
exactly when a clear "model failed to load" message is wanted instead of an
opaque index error.

---

## Part 2 — runtime and logic defects

### R1. Permutation entropy re-read 360 bars of history through a dynamic index

The order-3 permutation entropy looped over a 60-bar window and, for each,
referenced `close[i]`, `close[i+1]`, `close[i+2]` with a **loop-variable
index** — 360 history references per bar. That is both a performance trap and
a `max_bars_back` runtime-error risk.

**Replaced with an O(1) ring buffer**: a 6-slot pattern-count array and a
60-slot ring. Each bar evicts one pattern and inserts one. Same statistic,
constant work.

### R2. Barrier exits closed the position without telling the user

The state machine detected a stop or target hit and set `posState := 0`, but
emitted no `SELL` or `COVER`. The trader watching the chart saw a `BUY`, then
silence — the position closed and nothing said so. Since the visible contract
of this system is exactly five words (BUY / SELL / SHORT / COVER / NO TRADE),
an exit that prints nothing is a hole in the contract.

**Fixed**: stop hits, target hits, time exits and reversal exits all now flow
through one exit path that emits `SELL` or `COVER`, sets the alert reason
(`stop hit`, `target hit`, `time exit at 30 bars`, `reversal signal`), and
prices the exit pessimistically — a gap through the stop books at the *open*,
not the stop. A test asserts entries and exits reconcile exactly.

### R3. Beta log-normaliser omitted from the regime likelihood

Carried over from `docs/19-ADVERSARIAL-AUDIT.md` defect D5 and preserved here
because it is easy to reintroduce: the regime measurement model needs
`ln B(a,b)` **subtracted** from the Beta kernel. Omitting it tilted the
posterior by up to 4.2 log units. The first attempted fix *added* the shipped
value where the reference subtracts it, which doubled the error rather than
removing it. The call site now reads

```pine
acc += f_betaLogPdf(v, a, b) - array.get(DESIGN_LOGN, k * NFEAT + j)
```

with a comment saying so, because this has now been got wrong twice.

### R4. State mutated on unconfirmed bars

The position state machine ran on every tick of the live bar. Pine rolls back
`var` state between realtime ticks, so the committed history was correct — but
signals flickered on and off intrabar before settling at the close.

**Fixed**: every mutation of position, barrier and exit state is now gated on
`barstate.isconfirmed`. Historical behaviour is unchanged (confirmed is always
true on closed bars); the live bar simply stops flickering.

### R5. Sizing could silently produce a leveraged position (strategy)

Risk-based sizing divides risk capital by the stop distance. A very tight stop
therefore produces a very large position. Added a hard notional cap
(`Cap notional at % of equity`, default 100%) applied before rounding, and a
card row that shows the next position size — including the case where it
rounds to zero, which is otherwise indistinguishable from "no signal".

### R6. Table size, indices and `na` typing

- The card is declared `table.new(..., 2, 20)` and every `table.cell` index is
  checked against that declaration by the linter.
- `bgcolor(... : na)` became `bgcolor(... : color(na))`.
- Every `var float` declaration is typed, and `float(na)` is used wherever a
  typed `na` is needed.

### R7. Constants hand-copied instead of generated

Several model constants were literals in the Pine source that no longer
matched `pine/frozen_model.json`. Everything the model owns now comes from the
generated `BEGIN/END FROZEN MODEL` block, and a test compares `FM_HASH`
against the exported manifest, so a stale file fails the build rather than
trading a model nobody has.

---

## Part 3 — what is new

### Two operating modes

**Mode A — Frozen Research Model** evaluates the exported model through its
original gate: expectancy's 90% lower credible bound, minus modelled costs,
must exceed 0.05σ. **This gate has not been relaxed.** Measured over 22,626
daily bars of six ETFs it produced **4-5 entries**; over 20,850 hourly bars,
zero. When it declines, the card prints the arithmetic
(`edge 0.169 - cost 0.176 = -0.007 sigma, under the 0.05 gate`) so the refusal
is legible rather than mysterious.

**Mode B — Practical Observation Mode** (default) runs the same engines and
the same regime posterior, gated on a reliability-weighted blend with a 1–5
strictness slider. At default strictness it produced ~94 entries over the same
22,626 daily bars.

Mode B is **not** a validated strategy, and one measurement makes that
concrete: replayed over a synthetic martingale — a market built to contain no
edge at all — Mode B produced about 21 entries per 4,000 bars, while Mode A
produced zero or one. This is stated in both script headers and in
`START_HERE_TRADINGVIEW.md`, because a mode that fires on noise while the
project's headline claim is "a system that finds trades in a null market has a
bug" needs the distinction made loudly.

### One shared computational core

Everything from the frozen-model block through the Mode A / Mode B gates is
delimited by `// ==== BEGIN SHARED CORE ====` / `// ==== END SHARED CORE ====`
and is **byte-identical** in both files;
`python/tests/test_pine_port.py::test_indicator_and_strategy_share_an_identical_core`
fails the build if it stops being. The strategy therefore cannot trade
something the indicator does not paint.

### Diagnostics card

Twenty rows: Mode, Warm-up, Position, Direction, Confidence, Regime, Regime
hazard, Setup, Trend, Momentum, Volatility, Liquidity, Higher TF, Evidence,
Edge − cost, Gates passed, Stop / target, **Why no trade**, and the model hash
with a "NOT market-validated" marker. The veto row reports the **first**
failing gate of twelve, with its numbers.

### Alerts

Six `alertcondition()`s (BUY, SELL, SHORT, COVER, any ENTRY, any EXIT) plus
dynamic `alert()` calls carrying symbol, timeframe, signal price, stop, target,
confidence, mode, and the literal string `NOT VALIDATED`. All fire once per bar
close.

### Strategy

`pyramiding=0`, 0.02% commission, 2-tick slippage, `process_orders_on_close=false`
so market orders fill at the **next** bar's open (the Python model's
`execution_lag_bars = 1`), stop-and-target bracket via `strategy.exit`,
risk-per-trade sizing with a notional cap, a date-range filter, and position
state read from `strategy.position_size` rather than a private counter that
could drift away from the emulator's.

---

## Part 4 — what this port omits or approximates

| Python feature | Pine status | Why |
| --- | --- | --- |
| Heteroskedasticity-robust variance ratio θ(q) | **Approximated** — homoskedastic form with a frozen null scale | The robust correction needs a second pass over the window; the largest known numerical gap in the port (`docs/11-PINE-PLAN.md` §3) |
| Corwin–Schultz / Abdi–Ranaldo spread estimation | **Omitted** — a fixed 5bp fallback is used | The estimator needs a 22-bar rolling median of a heavy-tailed quantity; the fallback is the Python model's own `spread_fallback_rel` |
| Behavioural / dealer-gamma engine | **Abstains** — score 0, reliability 0.9 | Its inputs do not exist inside TradingView. Kept so engine indices match the exported 7×7 precision matrix |
| Online correlation learning (CALOP) | **Frozen** — the exported precision matrix is fixed | Learning would make the script's output depend on how much history the chart happens to have loaded |
| Edge Book online updating | **Frozen** — posteriors are read, never written | Same reason; the Pine script evaluates a model, it does not train one |
| Isotonic recalibration | **Frozen** — 8 exported knots, linear interpolation | Matches `numpy.interp` semantics including end-clamping |
| Drift monitor / demotion ladder | **Omitted** | Requires the decision journal and a realised-outcome feed |
| Bayesian Kelly sizing | **Simplified** — fixed risk-per-trade with a notional cap | Kelly needs the posterior variance of the mean, which the frozen export does not carry per-cell |
| Triple-barrier labelling, purged CPCV | **Not applicable** | Training-time machinery |
| Decision journal (JSONL) | **Omitted** | No filesystem in Pine; alerts carry the same fields |
| Time-of-day buckets, cross-asset engine | **Omitted** | Need a second data feed, which would mean `request.security` |

None of these omissions makes the Pine script *look better* than the Python
model — the frozen constants, the unrelaxed Mode A gate and the fixed spread
fallback all cut the other way.

---

## Verification

**No `.pine` file here has been compiled by TradingView.** No Pine compiler
was available. Do not read anything below as a claim that it was.

What was actually done:

| Check | Result |
| --- | --- |
| `python/tools/pinelint.py` — 18 static rule families over both files | **0 problems** |
| `python/tests/test_pine_port.py` — 24 regression checks | **24 passed** |
| Shared core byte-identical between indicator and strategy | asserted by test |
| `FM_HASH` matches `pine/frozen_model.json` | asserted by test |
| No `request.security` in code (comments excluded) | asserted by test |
| Port arithmetic replayed over 22,626 real daily bars (6 ETFs, 15y) | no non-finite values; entries and exits reconcile |
| Same, over 20,850 real hourly bars | no non-finite values |
| Same, over 12,000 synthetic null-market bars | Mode A: 0–1 entries; Mode B: ~21 per 4,000 |
| Strictness monotonicity (raising it never adds signals) | asserted by test |
| Stops on the correct side of every entry | asserted by test |
| Warm-up blocks every bar before 300 | asserted by test |

The replay uses `python/tools/pinesim.py`, a transliteration of the Pine
arithmetic into Python driven by the same `pine/frozen_model.json` the Pine
constants are generated from, so no constant is copied by hand. It establishes
that the logic *does something* and does not divide by zero. **It does not
establish that the Pine file compiles**, and a disagreement between it and the
chart is evidence against the transliteration, not against TradingView.

To finish verification, follow the manual checklist below.

### Manual compile-verification checklist

1. Paste `pine/TIA.pine` into a **new indicator** in the Pine Editor. Click
   **Save**. Expect: no red error; the status line reads "Script saved".
2. Click **Add to chart** on **SPY, 1D**. Expect: the diagnostics card in the
   top-right within a few seconds.
3. Scroll back so at least 3,000 bars are loaded. Expect: the **Warm-up** row
   reads `ready`, and the **Why no trade** row shows a specific reason.
4. Confirm roughly **14 BUY/SHORT markers** on SPY daily since 2010 at Mode B
   strictness 3. A wildly different count means the transliteration and Pine
   disagree — report it.
5. Set strictness to 1, then 5. Expect: strictly more signals at 1, strictly
   fewer at 5.
6. Switch to **Mode A**. Expect: `NO TRADE` on almost every bar, and the
   **Edge − cost** row showing a negative number.
7. Change the timeframe to **1W**, then **1h**, then **5m**. Expect: no error
   at any resolution. (This is the C1/C2 regression.)
8. Load a symbol with no volume feed — e.g. `FX:EURUSD`. Expect: the
   **Liquidity** row reads `no volume feed`, and no error.
9. Paste `pine/TIA_strategy.pine` into a **new strategy**. Expect: compiles,
   and a **Strategy Tester** panel appears.
10. In the Strategy Tester, open **List of Trades**. Expect: entry prices sit at
    the **open of the bar *after* each signal**, shifted by the 2-tick slippage
    setting — never at the signal bar's close. If they equal the signal bar's
    close, `process_orders_on_close` has been changed and the results are
    look-ahead contaminated. (To check this cleanly, set slippage to 0 in
    Properties for one run, confirm the fills land exactly on the next open,
    then put it back.)

If any step fails, the exact error text plus the step number is enough to
diagnose it.
