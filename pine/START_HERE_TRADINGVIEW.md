# Start here — TI-A on TradingView

This page assumes you have never written a line of code and never want to.
Everything below is copy, paste, and click.

> **Before anything else.** This is a research indicator. Nobody has shown it
> makes money, this repository does not claim it does, and the two files here
> are not connected to a broker and must not be. Read
> [What you should *not* conclude](#what-you-should-not-conclude) before you
> risk anything.

---

## 1. Install the indicator (2 minutes)

1. Open [tradingview.com](https://www.tradingview.com) and open any chart.
2. At the bottom of the window, click **Pine Editor**.
3. Click **Open ▸ New indicator**. Select everything in the editor
   (`Ctrl-A` / `Cmd-A`) and delete it.
4. Open the file **`pine/TIA.pine`** from this repository. Select all of it and
   copy it. Paste it into the empty Pine Editor.
5. Click **Save**, give it the name `TI-A`, then click **Add to chart**.

That is the whole installation. There is nothing to configure to make it run.

### Install the strategy (optional, another 2 minutes)

Same steps, but in step 3 choose **Open ▸ New strategy**, and in step 4 paste
**`pine/TIA_strategy.pine`**. After **Add to chart**, a **Strategy Tester**
panel appears at the bottom of the screen.

The strategy and the indicator compute the *same* signals from the *same*
code — a test in this repository fails the build if they ever stop matching.
The indicator draws them; the strategy sends them to TradingView's simulated
broker so they can be counted.

---

## 2. The smoke test — do this first

Use these exact settings the first time, because this page tells you precisely
what you should see:

| Setting | Value |
| --- | --- |
| Symbol | **SPY** |
| Timeframe | **1D** (daily) |
| Mode | **B: Practical Observation Mode** (this is the default) |
| Signal strictness | **3** (the default) |
| Date range | scroll back to at least 2010 so the script has history |

### What you should see if it is working

Within a few seconds of adding it to the chart:

- **A panel in the top-right corner** with about twenty rows: Mode, Warm-up,
  Position, Direction, Confidence, Regime, Regime hazard, Setup, Trend,
  Momentum, Volatility, Liquidity, Higher TF, Evidence, Edge − cost, Gates
  passed, Stop / target, and **Why no trade**.
- **A faint background tint** that changes colour as the market's regime
  changes — green for trending, blue for mean-reverting, red for stress, grey
  for quiet. The chart is mostly grey, and that is correct: most bars are
  unremarkable.
- **Green `BUY` triangles below bars and red `SHORT` triangles above bars**,
  each with a label giving the signal price, the stop, the target and a
  confidence figure. Small orange `SELL` and cyan `COVER` crosses mark the
  exits.
- **On SPY daily since 2010, roughly 14 entries.** Not 14 per year — 14 in
  total, across about fifteen years. This system abstains on purpose.
- **The "Why no trade" row is almost never empty.** On any bar where nothing
  is happening, it tells you the single reason: `quiet regime, p=0.71`,
  `cooldown, 6 bars left`, `score 0.31 under strictness 0.45`, and so on.

If the top-right panel is there and "Why no trade" is giving you a reason, the
script is working. That is the test.

### How many signals to expect

Measured by replaying the port's own arithmetic over real vendor data
(`python/tools/pinesim.py`). These are counts of **entries over the whole
history**, not per year:

**Daily bars, ~15 years (~3,770 bars per symbol)**

| Strictness | SPY | QQQ | IWM | GLD | TLT | EEM | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (loose) | 42 | 48 | 45 | 57 | 33 | 42 | 267 |
| 2 | 37 | 46 | 38 | 45 | 31 | 41 | 238 |
| **3 (default)** | **14** | **24** | **15** | **22** | **10** | **12** | **97** |
| 4 | 6 | 10 | 1 | 6 | 2 | 4 | 29 |
| 5 (strict) | 6 | 9 | 1 | 5 | 0 | 3 | 24 |

**Hourly bars, ~2 years (~3,475 bars per symbol)** — 232 entries in total at
strictness 3, so shorter timeframes are busier.

**Mode A across all six ETFs and all 22,600 daily bars: 5 entries.** One on
SPY, three on QQQ, one on EEM, zero on the other three. On hourly bars, zero.
That is not a malfunction — see the next section.

---

## 3. The two modes, and why Mode A says no so often

**Mode B — Practical Observation Mode** (the default) is what makes the chart
usable. It runs the full engine stack — a four-state regime posterior, a
Kalman trend filter, a liquidity-adjusted displacement measure, a volume
absorption measure, a sweep-and-fail detector, a variance ratio and
permutation entropy — and requires the reliability-weighted blend of them to
clear a threshold you control with the **strictness** slider. Turn strictness
down for more signals, up for fewer.

**Mode A — Frozen Research Model** evaluates the model that the Python side of
this repository actually trained and exported, through the gate it was
designed with: it will only trade when there is 90% posterior credence that
expectancy, *after* modelled trading costs, is positive. Most of the time
there is not, so it says no.

When Mode A refuses, the card shows you the arithmetic:

```
Edge − cost    0.169 − 0.176 = -0.007 sigma
Why no trade   edge 0.169 - cost 0.176 = -0.007 sigma, under the 0.05 gate
```

That reads: the model's own 90% lower bound on the edge for this situation is
0.169 (in units of one bar's volatility), a round trip costs 0.176 of the same
unit, so trading it loses money in expectation. It is not broken. It has
looked at the situation and correctly declined.

**Mode A's gate has deliberately not been loosened to make the chart busier.**
That would be manufacturing signals by lowering the bar, which is the exact
thing this project exists to avoid.

---

## 4. Settings worth knowing

| Setting | What it does |
| --- | --- |
| **Operating mode** | A (frozen research model) or B (observation). Default B. |
| **Signal strictness** | 1–5, Mode B only. Lower = more signals. Default 3. |
| **Allow long / short entries** | Turn either direction off entirely. |
| **Require higher-timeframe agreement** | Vetoes a signal fighting the higher-timeframe trend. On by default. |
| **Limit entries to a session** | Only for intraday charts. Off by default. |
| **Cooldown bars after an exit** | Forced wait after any exit. Default 10. |
| **Max entries per 100 bars** | Hard rate limit. Default 6. |
| **Stop / target basis** | "Forecast volatility" (the model's own rule) or a plain ATR multiple, if you prefer stops you can reason about. |
| **Card position / text size** | Move the panel if it covers your price action. |

In the **strategy**, additionally: risk-per-trade sizing, a notional cap so a
tight stop cannot silently build a leveraged position, and a date-range filter.

---

## 5. Alerts

Right-click the chart ▸ **Add alert**, set **Condition** to `TI-A`, then pick
one of six: `TI-A BUY`, `TI-A SELL`, `TI-A SHORT`, `TI-A COVER`,
`TI-A any ENTRY`, `TI-A any EXIT`.

The script also fires a rich alert automatically, which looks like:

```
TI-A BUY | SPY 1D | signal 412.53 | stop 405.11 | target 424.59 |
confidence 61.4% | mode B-observation | NOT VALIDATED
```

Alerts fire **once per bar close** only. There is no intrabar alert, on
purpose: a signal that appears mid-bar and vanishes before the close is not a
signal, it is noise.

---

## 6. Troubleshooting

### "The Pine Editor shows a red error and won't compile"

Almost always one of three things:

- **You pasted only part of the file.** The file is ~1,100 lines. Click inside
  the editor, press `Ctrl-A`/`Cmd-A` and delete, then paste the whole thing
  again. Check that the very first line is `//@version=6` and the last line
  ends with `alert.freq_once_per_bar_close)`.
- **You pasted the strategy into a "New indicator" tab (or vice versa).** The
  first statement has to match: `indicator(` for `TIA.pine`, `strategy(` for
  `TIA_strategy.pine`.
- **Your editor reformatted the file.** Pine is indentation-sensitive, exactly
  like Python. Copy from the raw file, not from a rendered web page, and don't
  let anything convert tabs or strip leading spaces.

Copy the exact error text from the editor — it names a line number — and check
that line against the file.

### "It compiles but I see no signals at all"

Work down this list in order:

1. **Look at the "Why no trade" row.** It gives you the answer directly. Every
   remaining item on this list is really just a specific value of that row.
2. **Is "Warm-up" still counting down?** The script needs 300 bars before it
   will consider anything. See the next section.
3. **Are you in Mode A?** Then no signals is the expected result. Switch to
   Mode B to see the engines working.
4. **Is strictness at 4 or 5?** Drop it to 2. At strictness 5 on TLT daily,
   fifteen years produced zero entries.
5. **Not enough history on the chart.** Scroll left. With only 400 bars loaded,
   300 go to warm-up and 100 remain.
6. **Both directions disabled**, or a session filter set to hours the symbol
   does not trade.

### "The card says WARMING UP forever"

The counter needs 300 closed bars **on the current chart**. If it never
reaches zero:

- The symbol genuinely has fewer than 300 bars of history at this timeframe —
  a new listing, or a 1-minute chart on a free plan that only loads a few
  thousand bars. Switch to a longer timeframe.
- You are on a very high timeframe (monthly, quarterly) where 300 bars is more
  history than exists. 300 monthly bars is 25 years.
- Check the row underneath: if it reads `frozen model failed to load; re-run
  the exporter`, then the constants block in the file you pasted is a
  placeholder rather than a real export. Get a fresh copy of the file.

### "The diagnostics card is blank, or missing"

- The card only draws on the **last** bar of the chart. If you have scrolled
  far back in history, scroll to the right-hand edge.
- **"Show diagnostics card"** may be off in Settings ▸ Inputs ▸ Display.
- It may be *behind* something. Change **Card position** to `Bottom left`.
- If the card is there but every value reads `n/a`, the script is still in
  warm-up.

### "Invalid timeframe / it breaks on weekly or monthly charts"

It should not, and this was a real defect in an earlier build. That version
called `request.security()` with a timeframe string it assembled at runtime,
which produced things like `"5D"` on a weekly chart — not a valid resolution.

This build contains **no `request.security()` at all**. The higher timeframe
is built inside the script by grouping five closed chart bars into one, so it
works identically on 1-minute, hourly, daily, weekly and monthly charts. A
regression test in this repository fails if `request.security` ever reappears.

If you *do* see a timeframe error, you have an old copy of the file.

### "My symbol has no volume (FX, indices, some CFDs)"

Handled. The **Liquidity** row will read `no volume feed`, and the three
volume-dependent measures — participation, absorption, and the sweep detector's
volume confirmation — go neutral instead of poisoning everything downstream
with zeros.

The script still works, but with less evidence. Expect fewer signals, and be
aware that the sweep-and-fail setup effectively cannot fire without volume,
because "anomalous volume" is half of its definition.

### "The script is slow, or TradingView says it exceeds a limit"

The heaviest part is a 7×7 matrix operation on every bar. If you hit a
resource limit:

- Turn off **"Label entries with stop and target"** — labels are the most
  expensive thing drawn.
- Turn off **"Tint background by regime"**.
- Set **Card text size** to `Tiny`, or turn the card off once you trust it.
- Use a higher timeframe. A 1-minute chart with 20,000 bars is 20,000
  evaluations; the same period on hourly is 300.

The script declares `max_bars_back=500`. Do not lower it — the volatility rank
needs 252 bars of history, and shrinking the buffer produces a runtime error
rather than a wrong answer.

### "The Strategy Tester shows zero trades"

- **Check the "Gates passed" row on the strategy's own card** — it counts 13
  conditions, one more than the indicator (the date-range filter).
- **The date range filter may be excluding everything.** It is off by default;
  if you turned it on, confirm the window actually overlaps your chart.
- **"Next size" reads `0 - equity too small for this stop`.** Position size is
  derived from risk-per-trade and the stop distance, then rounded down to whole
  units. On a $100,000 account risking 0.5%, a $500 risk against a $40-wide
  stop is 12 shares — fine — but against a $600-wide stop it rounds to zero.
  Either raise risk-per-trade, raise initial capital in Properties, or turn off
  **"Round position size down to whole units"** (correct for crypto and FX).
- **You are in Mode A.** Five trades in fifteen years across six ETFs means
  most single-symbol tests will show zero. This is the expected result.
- **Not enough bars.** 300 go to warm-up before the first possible entry.

---

## What you should *not* conclude

Making this indicator run on TradingView proves that the code executes. It
proves **nothing whatsoever** about whether the signals are profitable, and
this repository does not claim they are.

Three specific traps:

1. **A green equity curve in the Strategy Tester is not evidence.** One symbol
   over one history is a single draw from a very wide distribution. The Python
   reference's own pre-registered test needs roughly 400 trades pooled across
   about 50 instruments before it can even be run
   (`docs/00-PREREGISTRATION.md`). You will not get there on one chart.

2. **Mode B fires on pure noise.** Replayed over a synthetic martingale — a
   market constructed to contain no edge at all — Mode B produced about 21
   entries per 4,000 bars at default strictness. Mode A produced zero or one.
   Mode B is a tool for *seeing what the engines react to*. It is not evidence
   that what they react to is tradeable.

3. **Do not tune the strictness slider until the backtest looks good.**
   `docs/08-FAILURE-MODES.md` ranks the most likely cause of this system's
   death, and it is not a statistical failure — it is a human adjusting it
   after a drawdown.

Do not connect this to a live account. Do not wire it to a webhook that places
orders. Neither file contains any brokerage integration, and that is
deliberate.

---

## Where the numbers come from

| Question | File |
| --- | --- |
| The mathematics | `docs/01-THEORY.md` |
| What each engine measures | `docs/03-ENGINES.md` |
| What the port approximates and why | `docs/11-PINE-PLAN.md` |
| Every defect fixed in this port | `pine/CHANGELOG_TRADINGVIEW.md` |
| How this system most likely dies | `docs/08-FAILURE-MODES.md` |
| The final review board's verdict | `docs/20-FINAL-REVIEW.md` |
| What would have to be true to trade it | `docs/00-PREREGISTRATION.md` |

**Not investment advice.** Research code, not a trading system.
