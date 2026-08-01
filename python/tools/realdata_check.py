"""First-contact protocol for real market data.

Reproduces the final review board's Phase-8 disproof attempt: fetch a small,
a-priori-chosen ETF universe, and run the four checks that real data can answer
and synthetic data cannot. Requires network access; everything else in the
repository does not.

Run:  cd python && PYTHONPATH=src python3 tools/realdata_check.py

The four checks, and what each outcome means:

1.  **Data integrity.** The validator and kernel must run clean on raw vendor
    bars. A crash or a flood of rejects here is an implementation kill.
2.  **Cold-start discipline.** An untrained system must emit zero signals on
    real data. Any signal is a bug -- the credible bound cannot clear zero
    without observations.
3.  **Gap reality.** Measures the overnight-gap distribution, i.e. how large
    the D2 fill-timing bug's optimism would have been on this data. This is
    the number that justifies the deferred-fill execution model.
4.  **Pooled walk-forward.** Train on the first 60% of every instrument into
    one shared Edge Book, trade the last 40% out of sample. The pre-registered
    bar is n >= 400 trades at t >= 3. On six ETFs of daily data the expected
    and honest outcome is *far too few trades to conclude anything* -- in which
    case the correct action, stated in advance, is to stop optimizing and
    widen the universe, not to loosen the gate until trades appear.

Caveats this script cannot remove: the vendor's raw closes are split-adjusted
but not dividend-adjusted, so ex-dividend days appear as small opening gaps;
the universe is survivorship-lite (broad asset-class ETFs chosen a priori, not
winners picked ex post); and one vendor's bars are one vendor's opinion.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tia import Config, TIA  # noqa: E402
from tia.data.validate import BarValidator  # noqa: E402
from tia.engines.edgebook import EdgeBook  # noqa: E402
from tia.fusion.calibration import IsotonicCalibrator  # noqa: E402
from tia.training import collect_candidates, train_edge_book  # noqa: E402
from tia.types import Bar, ExogenousSnapshot  # noqa: E402

#: Chosen a priori for asset-class breadth, not performance: US large cap,
#: US tech, US small cap, gold, long treasuries, emerging equity.
UNIVERSE = ("SPY", "QQQ", "IWM", "GLD", "TLT", "EEM")
RANGE = "15y"
#: Conservative 1bp relative spread for these ETFs; a real deployment supplies
#: the measured spread instead.
SPREAD_REL = 1e-4
TRAIN_FRAC = 0.6


def fetch(symbol: str) -> list[Bar]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?range={RANGE}&interval=1d"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    bars: list[Bar] = []
    for i, ts in enumerate(result["timestamp"]):
        o, h, l, c, v = (quote[k][i] for k in ("open", "high", "low", "close", "volume"))
        if None in (o, h, l, c, v) or min(o, h, l, c) <= 0:
            continue
        # Vendor rounding can violate OHLC consistency by a cent; widen rather
        # than fabricate. The validator would otherwise reject the bar outright.
        bars.append(
            Bar(float(ts), float(o), float(max(h, o, c)), float(min(l, o, c)),
                float(c), float(v))
        )
    return bars


def main() -> int:
    print(f"TI-A real-data first-contact protocol  ({', '.join(UNIVERSE)}, {RANGE} daily)")
    data: dict[str, list[Bar]] = {}
    for sym in UNIVERSE:
        try:
            data[sym] = fetch(sym)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {sym}: FETCH FAILED ({exc}); skipping")
    if not data:
        print("no data fetched; network unavailable. Nothing to conclude.")
        return 2

    print("\n1. data integrity")
    for sym, bars in data.items():
        v = BarValidator()
        fatal = sum(1 for b in bars if (i := v.check(b)) is not None and i.fatal)
        print(f"   {sym}: {len(bars)} bars, {fatal} fatal rejects")

    print("\n2. cold-start discipline (untrained system must emit nothing)")
    total = 0
    for sym, bars in data.items():
        sysm = TIA(Config())
        sysm.run(bars)
        total += sum(1 for d in sysm.decisions if d.is_actionable)
    print(f"   untrained signals across universe: {total}")
    if total:
        print("   KILL: an untrained system traded on real data. Investigate before")
        print("   anything else; the credible-bound gate is broken.")
        return 1

    print("\n3. overnight-gap reality (the D2 blind spot, measured)")
    for sym, bars in data.items():
        g = [abs(math.log(bars[i].open / bars[i - 1].close)) for i in range(1, len(bars))]
        print(f"   {sym}: median {np.median(g) * 1e4:5.1f} bp   p95 {np.percentile(g, 95) * 1e4:6.1f} bp")

    print("\n4. pooled walk-forward (train first 60% into one shared book, trade last 40%)")
    cfg = Config()
    book = EdgeBook(cfg)
    cal = IsotonicCalibrator()
    for sym, bars in data.items():
        cut = int(len(bars) * TRAIN_FRAC)
        ex = [ExogenousSnapshot(spread=SPREAD_REL)] * cut
        cands = collect_candidates(bars[:cut], cfg, exog=ex)
        book, cal, _ = train_edge_book(bars[:cut], cands, cfg, book=book, calibrator=cal)
        print(f"   {sym}: {len(cands)} training candidates "
              f"({100 * cands.eligibility_rate:.1f}% of valid bars)")

    all_net: list[float] = []
    for sym, bars in data.items():
        cut = int(len(bars) * TRAIN_FRAC)
        ex = [ExogenousSnapshot(spread=SPREAD_REL)] * len(bars)
        sysm = TIA(cfg, edge_book=book, learn=False)
        sysm.calibrator._iso = cal
        sysm.calibrator.active = cal.n_fit > 0
        for i, b in enumerate(bars):
            sysm.on_bar(b, ex[i])
        oos = [t for t in sysm.trades if t.entry_index >= cut]
        all_net += [t.net_sigma for t in oos]
        print(f"   {sym}: {len(oos)} out-of-sample trades")

    print("\nPOOLED OUT-OF-SAMPLE RESULT")
    n = len(all_net)
    if n > 2:
        net = np.array(all_net)
        se = net.std(ddof=1) / math.sqrt(n)
        t = net.mean() / se if se > 0 else math.nan
        print(f"   n={n}  net {net.mean():+.4f} sigma/trade  t={t:+.2f}")
        print(f"   pre-registered bar: n >= 400 and t >= 3")
    else:
        print(f"   {n} trades: statistically NOTHING can be concluded, for or against.")
        print("   Per the pre-registration, the correct action is to widen the universe")
        print("   and extend the history -- not to loosen the gate until trades appear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
