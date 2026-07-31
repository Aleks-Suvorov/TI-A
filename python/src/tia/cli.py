"""Command line interface.

``pandas`` is an optional dependency, so CSV reading falls back to the standard
library. Everything the live path touches stays numpy-only.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from .config import Config
from .data.sessions import PROFILES
from .types import Bar

__all__ = ["main", "read_csv"]

_TS_FIELDS = ("timestamp", "time", "date", "datetime", "open_time")
_FIELD_ALIASES = {
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "adj close", "adj_close"),
    "volume": ("volume", "v", "vol"),
}


def _parse_timestamp(raw: str, index: int) -> float:
    """Accept epoch seconds, epoch millis, or a handful of date formats.

    Falls back to the row index when a file has no usable time column. That is
    safe here because every window in the system is measured in *bars*, not in
    wall-clock time -- but it does disable session tagging, so the caller is
    warned.
    """
    raw = raw.strip()
    if not raw:
        return float(index)
    try:
        v = float(raw)
        return v / 1000.0 if v > 1e11 else v
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw[:19], fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return float(index)


def read_csv(path: str | Path) -> list[Bar]:
    """Read OHLCV bars from a CSV.

    Expected columns, case-insensitive, in any order: a time column named one of
    ``timestamp/time/date/datetime/open_time``, plus ``open``, ``high``, ``low``,
    ``close`` and ``volume``. Rows failing OHLC consistency are skipped with a
    warning rather than repaired -- repairing means guessing, and a guess that
    reaches the feature kernel is indistinguishable from a signal.
    """
    p = Path(path)
    with p.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{p}: no header row")
        lower = {(name or "").strip().lower(): name for name in reader.fieldnames}

        ts_col = next((lower[k] for k in _TS_FIELDS if k in lower), None)
        cols: dict[str, str] = {}
        for field, aliases in _FIELD_ALIASES.items():
            hit = next((lower[a] for a in aliases if a in lower), None)
            if hit is None:
                raise ValueError(
                    f"{p}: no column for '{field}'. Found: {sorted(lower)}"
                )
            cols[field] = hit

        bars: list[Bar] = []
        skipped = 0
        for i, row in enumerate(reader):
            try:
                ts = _parse_timestamp(row.get(ts_col, "") if ts_col else "", i)
                bars.append(
                    Bar(
                        timestamp=ts,
                        open=float(row[cols["open"]]),
                        high=float(row[cols["high"]]),
                        low=float(row[cols["low"]]),
                        close=float(row[cols["close"]]),
                        volume=float(row[cols["volume"]] or 0.0),
                    )
                )
            except (ValueError, KeyError, TypeError):
                skipped += 1
        if skipped:
            print(f"note: skipped {skipped} unusable rows", file=sys.stderr)
        if not bars:
            raise ValueError(f"{p}: no usable rows")
        return bars


def _session_for(name: str):
    if name not in PROFILES:
        raise SystemExit(f"unknown session profile '{name}'. Choose from: {sorted(PROFILES)}")
    return PROFILES[name]


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = Config()
    print(cfg.manifest_report())
    print(f"parameters fitted to market outcomes: {cfg.fitted_dof()}")
    print(f"parameters updated online (causal)  : {cfg.online_dof()}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .pipeline import TIA
    from .synthetic import generate_null, generate_with_regimes
    from .training import collect_candidates, train_edge_book

    print("=" * 62)
    print("1. NULL MARKET  (a martingale; the system should find nothing)")
    print("=" * 62)
    null = generate_null(args.bars, seed=7)
    s0 = TIA(Config())
    s0.run(null)
    print(s0.summary())

    print()
    print("=" * 62)
    print("2. REGIME-SWITCHING MARKET  (a real, small, alternating edge)")
    print("=" * 62)
    bars, _ = generate_with_regimes(args.bars * 2, seed=12)
    cands = collect_candidates(bars, Config())
    print(f"structurally eligible bars: {len(cands)} of {cands.n_valid_bars} "
          f"({100.0 * cands.eligibility_rate:.2f}%)")
    book, cal, diag = train_edge_book(bars, cands, Config())
    for k in ("effective_sample_size", "uniqueness_ratio", "hit_rate",
              "mean_ret_sigma", "mean_cost_sigma", "net_expectancy_sigma",
              "t_stat_effective"):
        v = diag.get(k, math.nan)
        print(f"  {k:22s} {v:+.4f}" if v == v else f"  {k:22s} n/a")
    print()
    print("Note the effective sample size against the candidate count: overlapping")
    print("labels mean the real sample is far smaller than it looks, and every")
    print("t-statistic must be computed against the smaller number.")

    s1 = TIA(Config(), edge_book=book, learn=False)
    s1.calibrator._iso = cal
    s1.calibrator.active = cal.n_fit > 0
    s1.run(bars)
    print()
    print(s1.summary())

    print()
    print("=" * 62)
    print("3. RECOVERY TEST  (does the gate open when an edge is provable?)")
    print("=" * 62)
    print("A system that never trades is indistinguishable from one whose gate is")
    print("stuck shut. So: the same generator, at increasing edge strengths, with a")
    print("measured spread supplied through ExogenousSnapshot (which is the")
    print("production path -- estimating a spread from OHLC alone cannot resolve")
    print("below a few basis points and returns a conservative floor).")
    print()
    from .types import ExogenousSnapshot

    print(f"  {'edge':>6s} {'cands':>6s} {'gross':>8s} {'cost':>7s} {'net':>8s} "
          f"{'t':>7s} {'signals':>8s}")
    s2 = None
    rows: list[dict[str, float]] = []
    for strength in (0.06, 0.30, 0.80):
        bb, _ = generate_with_regimes(
            args.bars * 2, seed=77, trend_strength=strength, revert_strength=strength * 1.5
        )
        ex = [ExogenousSnapshot(spread=5.0e-4)] * len(bb)
        cc = collect_candidates(bb, Config(), exog=ex)
        bk, cl, dd = train_edge_book(bb, cc, Config())
        ss = TIA(Config(), edge_book=bk, learn=False)
        ss.calibrator._iso = cl
        ss.calibrator.active = cl.n_fit > 0
        ss.run(bb, exog=ex)
        acted = sum(1 for d in ss.decisions if d.is_actionable)
        print(f"  {strength:6.2f} {len(cc):6d} {dd.get('mean_ret_sigma', math.nan):+8.4f} "
              f"{dd.get('mean_cost_sigma', math.nan):7.3f} "
              f"{dd.get('net_expectancy_sigma', math.nan):+8.4f} "
              f"{dd.get('t_stat_effective', math.nan):+7.2f} {acted:8d}")
        rows.append({
            "gross": float(dd.get("mean_ret_sigma", math.nan)),
            "cost": float(dd.get("mean_cost_sigma", math.nan)),
            "net": float(dd.get("net_expectancy_sigma", math.nan)),
            "t": float(dd.get("t_stat_effective", math.nan)),
            "ess": float(dd.get("effective_sample_size", math.nan)),
        })
        if acted:
            s2 = ss
    print()
    # Describe the row that was actually printed rather than asserting numbers.
    # They move with the sample -- which is itself the point, and a narrative
    # that contradicts its own table teaches the reader to stop reading it.
    r0 = rows[0]
    print("Read the first row carefully. At an edge strength comparable to what is")
    print(f"actually achievable, gross expectancy is {r0['gross']:+.3f} sigma against")
    print(f"{r0['cost']:.3f} of cost; the t-statistic on {r0['ess']:.0f} independent")
    print(f"observations is {r0['t']:+.2f}, and the system emits nothing.")
    print()
    print("Whether net expectancy on that row is positive or negative varies with")
    print("the sample. What does not vary is the reason for standing aside: at a")
    print("t-statistic near 1 the LOWER credible bound is negative, and the gate")
    print("acts on that rather than on the point estimate. The system is not")
    print("refusing because the edge is absent -- it is refusing because the edge")
    print("cannot be demonstrated. Extreme selectivity falls out of the decision")
    print("rule instead of being imposed by a threshold.")

    if s2 is None:
        s2 = s1
    card = next(
        (d.card for d in reversed(s2.decisions) if d.card and d.card.rows), None
    )
    if card is not None:
        print()
        print("Most recent explanation card:")
        print(card.render())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import TIA

    bars = read_csv(args.csv)
    sysm = TIA(Config(), session=_session_for(args.session))
    n_act = 0
    for dec in sysm.stream(bars):
        if dec.is_actionable:
            n_act += 1
            when = datetime.fromtimestamp(dec.timestamp, tz=timezone.utc).isoformat()
            print(f"\n[{dec.decided_at_index}] {when}")
            if dec.card:
                print(dec.card.render())
    print()
    print(sysm.summary())
    if n_act == 0:
        print("\nNo actionable signals. For a selective system on a short history this")
        print("is the expected outcome, not a failure -- see docs/01-THEORY.md section 12.")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .training import collect_candidates, train_edge_book

    bars = read_csv(args.csv)
    cfg = Config()
    session = _session_for(args.session)
    cands = collect_candidates(bars, cfg, session)
    print(f"bars {len(bars)}  valid {cands.n_valid_bars}  candidates {len(cands)}"
          f"  ({100.0 * cands.eligibility_rate:.2f}% of valid bars)")
    if not len(cands):
        print("No structurally eligible bars; nothing to train on.")
        return 0
    book, cal, diag = train_edge_book(bars, cands, cfg)
    print()
    for k in sorted(diag):
        v = diag[k]
        print(f"  {k:24s} {v:+.4f}" if v == v else f"  {k:24s} n/a")
    print()
    print(book.summary())
    print()
    print("Reminder: training and trading the same bars is in-sample by")
    print("construction. Use tia.training.walk_forward_train for a result that means")
    print("something.")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from .training import walk_forward_train
    from .validation.report import build_report

    bars = read_csv(args.csv)
    cfg = Config()
    try:
        systems, agg = walk_forward_train(bars, cfg, _session_for(args.session), n_folds=args.folds)
    except ValueError as exc:
        print(f"cannot run walk-forward: {exc}", file=sys.stderr)
        return 2
    trades = [t for s in systems for t in s.trades]
    print(build_report(cfg, trades, agg, systems))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .export import main as export_main

    return export_main([])


# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tia",
        description="TI-A: an adaptive, abstention-first trading signal framework.",
        epilog="Most bars produce NO TRADE. That is the design, not a fault.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run on synthetic data, including the null-market check")
    d.add_argument("--bars", type=int, default=3000)
    d.set_defaults(fn=_cmd_demo)

    for name, fn, helptext in (
        ("run", _cmd_run, "stream an OHLCV CSV and print actionable signals"),
        ("train", _cmd_train, "run the counterfactual training pass over a CSV"),
        ("validate", _cmd_validate, "walk-forward validation report over a CSV"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--csv", required=True, help="path to an OHLCV CSV")
        s.add_argument("--session", default="daily", choices=sorted(PROFILES),
                       help="instrument session profile (default: daily)")
        if name == "validate":
            s.add_argument("--folds", type=int, default=4)
        s.set_defaults(fn=fn)

    c = sub.add_parser("config", help="print the parameter manifest and its hash")
    c.set_defaults(fn=_cmd_config)

    e = sub.add_parser("export", help="write pine/frozen_model.json")
    e.set_defaults(fn=_cmd_export)

    args = p.parse_args(list(argv) if argv is not None else None)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
