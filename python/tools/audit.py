"""Deep invariant audit.

Complements the unit tests. The tests check that components are individually
correct; this checks that the *assembled system* never violates its own contract
across many seeds, configurations, market conditions and degenerate inputs.

Run:  cd python && PYTHONPATH=src python3 tools/audit.py

Exit code is the number of violations found, so it can gate a commit.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "src"))

from tia import Action, Config, Position, TIA  # noqa: E402
from tia.data.sessions import PROFILES  # noqa: E402
from tia.engines.edgebook import EdgeBook, student_t_ppf  # noqa: E402
from tia.fusion.calop import CALOP  # noqa: E402
from tia.synthetic import bars_from_arrays, generate_null, generate_with_regimes  # noqa: E402
from tia.training import collect_candidates, train_edge_book  # noqa: E402
from tia.types import Bar, ExogenousSnapshot  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(message)


def section(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


# ---------------------------------------------------------------------------
# 1. Frozen-model / Pine consistency
# ---------------------------------------------------------------------------


def audit_pine_freshness() -> None:
    section("Pine constants vs exported model")
    js = REPO / "pine" / "frozen_model.json"
    if not js.exists():
        check(False, "pine/frozen_model.json missing")
        return
    model = json.loads(js.read_text())

    for name in ("TIA.pine", "TIA_strategy.pine"):
        path = REPO / "pine" / name
        if not path.exists():
            check(False, f"{name} missing")
            continue
        text = path.read_text()

        # The constants block must carry the same manifest hash as the JSON.
        want = model["config_manifest_hash"]
        check(
            f'FM_HASH          = "{want}"' in text or f'FM_HASH = "{want}"' in text,
            f"{name}: FM_HASH does not match frozen_model.json ({want})",
        )

        # The isotonic map must match, and must not assert certainty.
        for key, jkey in (("FM_ISO_X", "knots_x"), ("FM_ISO_Y", "knots_y")):
            line = next((l for l in text.splitlines() if l.startswith(key)), None)
            if line is None:
                check(False, f"{name}: {key} not found")
                continue
            emitted = [float(v) for v in line.split('"')[1].split(",")]
            expected = model["fusion"][jkey]
            check(
                len(emitted) == len(expected)
                and all(abs(a - b) < 1e-6 for a, b in zip(emitted, expected)),
                f"{name}: {key} is STALE — does not match frozen_model.json",
            )
            if key == "FM_ISO_Y":
                check(
                    min(emitted) > 0.0 and max(emitted) < 1.0,
                    f"{name}: isotonic map asserts certainty (contains 0 or 1)",
                )

        # No sentinel values may reach Pine.
        for line in text.splitlines():
            if line.startswith("FM_") and "e+15" in line.replace("E+15", "e+15"):
                check(False, f"{name}: sentinel value leaked into constants: {line[:60]}")


def audit_export_sanity() -> None:
    section("Exported model internal sanity")
    model = json.loads((REPO / "pine" / "frozen_model.json").read_text())

    f = model["fusion"]
    for k in ("brier", "ece"):
        v = f.get(k)
        check(
            v is None or (isinstance(v, (int, float)) and (math.isnan(v) or -1e6 < v < 1e6)),
            f"fusion.{k} = {v!r} is a sentinel, not a value",
        )
    check(
        len(f["knots_x"]) == len(f["knots_y"]),
        "isotonic knot vectors have different lengths",
    )
    check(
        all(a <= b + 1e-12 for a, b in zip(f["knots_x"], f["knots_x"][1:])),
        "isotonic knots_x not sorted",
    )
    check(
        all(a <= b + 1e-12 for a, b in zip(f["knots_y"], f["knots_y"][1:])),
        "isotonic knots_y not monotone",
    )

    eb = model["edge_book"]
    n_expected = 4 * eb["n_setups"] * eb["n_buckets"]
    for key in ("mu_n", "scale", "dof", "n_eff"):
        arr = eb["cells"][key]
        check(len(arr) == n_expected, f"edge_book.cells.{key} has {len(arr)} entries, want {n_expected}")
        check(all(math.isfinite(v) for v in arr), f"edge_book.cells.{key} contains non-finite values")
    check(all(d > 2.0 for d in eb["cells"]["dof"]), "edge_book dof <= 2: variance undefined")
    check(all(s > 0.0 for s in eb["cells"]["scale"]), "edge_book scale <= 0")

    prec = np.array(model["engines"]["precision"]).reshape(
        model["engines"]["count"], model["engines"]["count"]
    )
    check(np.allclose(prec, prec.T, atol=1e-9), "engine precision matrix not symmetric")
    check(
        np.all(np.linalg.eigvalsh(prec) > 0),
        "engine precision matrix not positive definite",
    )
    check(
        abs(float(np.ones(prec.shape[0]) @ prec @ np.ones(prec.shape[0]))
            - model["engines"]["ebe_at_unit_reliability"]) < 1e-6,
        "exported EBE does not equal 1' P 1 from the exported precision matrix",
    )


# ---------------------------------------------------------------------------
# 2. Decision-stream invariants over many seeds
# ---------------------------------------------------------------------------


def audit_decision_invariants() -> None:
    section("Decision-stream invariants (12 seeds x 2 markets)")
    cfg = Config()
    for seed in range(12):
        for label, bars in (
            ("null", generate_null(1200, seed=seed)),
            ("regime", generate_with_regimes(1200, seed=100 + seed)[0]),
        ):
            sysm = TIA(cfg)
            prev_pos = Position.FLAT
            for d in sysm.stream(bars):
                tag = f"[{label} seed={seed} bar={d.decided_at_index}]"

                check(d.execute_at_index > d.decided_at_index, f"{tag} execution not after decision")

                fu = d.fusion
                if fu is not None:
                    check(0.0 < fu.p_success < 1.0, f"{tag} p_success={fu.p_success} not in (0,1)")
                    check(
                        fu.p_low <= fu.p_success + 1e-9 and fu.p_success <= fu.p_high + 1e-9,
                        f"{tag} credible interval does not bracket p "
                        f"({fu.p_low:.4f}, {fu.p_success:.4f}, {fu.p_high:.4f})",
                    )
                    check(math.isfinite(fu.log_odds), f"{tag} log_odds not finite")
                    check(fu.ebe >= 0.0, f"{tag} negative effective breadth {fu.ebe}")
                    check(
                        fu.ebe <= len(sysm.engines) + 1e-6,
                        f"{tag} breadth {fu.ebe} exceeds engine count",
                    )
                    check(fu.direction in (-1, 0, 1), f"{tag} bad direction {fu.direction}")

                check(
                    math.isfinite(d.expected_value_sigma) or d.expected_value_sigma == -math.inf,
                    f"{tag} expected_value_sigma is NaN",
                )
                check(
                    d.expected_value_lcb <= d.expected_value_sigma + 1e-9,
                    f"{tag} EV lower bound above the mean",
                )

                if d.costs is not None:
                    for name in ("half_spread_sigma", "impact_sigma", "slippage_sigma", "fee_sigma"):
                        v = getattr(d.costs, name)
                        check(v >= 0.0 and not math.isnan(v), f"{tag} cost.{name} = {v}")

                for nm, o in d.engine_outputs.items():
                    check(-1.0 <= o.score <= 1.0, f"{tag} engine {nm} score {o.score}")
                    check(0.0 <= o.reliability <= 1.0, f"{tag} engine {nm} reliability {o.reliability}")
                    check(o.score == o.score, f"{tag} engine {nm} NaN score")

                if d.target is not None:
                    t = d.target
                    check(t.sigma > 0.0, f"{tag} target sigma {t.sigma}")
                    check(t.stop_sigma > 0.0 and t.target_sigma > 0.0, f"{tag} non-positive barrier")
                    if t.direction > 0:
                        check(
                            t.stop_price < t.entry_ref < t.target_price,
                            f"{tag} long barriers not straddling entry",
                        )
                    else:
                        check(
                            t.target_price < t.entry_ref < t.stop_price,
                            f"{tag} short barriers not straddling entry",
                        )

                if prev_pos is Position.LONG:
                    check(d.position_after is not Position.SHORT, f"{tag} LONG->SHORT in one bar")
                if prev_pos is Position.SHORT:
                    check(d.position_after is not Position.LONG, f"{tag} SHORT->LONG in one bar")
                prev_pos = d.position_after

                check(d.action is not Action.NO_TRADE or d.target is None, f"{tag} NO_TRADE with a target")

            check(sysm.equity > 0.0, f"[{label} seed={seed}] equity went non-positive")
            for t in sysm.trades:
                check(math.isfinite(t.ret_sigma), f"[{label} seed={seed}] non-finite trade return")
                check(t.exit_index >= t.entry_index, f"[{label} seed={seed}] exit before entry")


# ---------------------------------------------------------------------------
# 3. Degenerate and adversarial inputs
# ---------------------------------------------------------------------------


def audit_degenerate_inputs() -> None:
    section("Degenerate and adversarial inputs")
    cfg = Config()

    cases: dict[str, list[Bar]] = {}

    # Perfectly flat market: zero range, zero return, zero volume.
    cases["flat"] = [Bar(i * 86400.0, 100.0, 100.0, 100.0, 100.0, 0.0) for i in range(400)]

    # Constant price with volume.
    cases["no_move_with_volume"] = [
        Bar(i * 86400.0, 100.0, 100.0, 100.0, 100.0, 1e6) for i in range(400)
    ]

    # Monotone ramp: maximum trend, zero noise.
    cases["ramp"] = [
        Bar(i * 86400.0, 100 + i * 0.1, 100 + i * 0.1 + 0.05, 100 + i * 0.1 - 0.01,
            100 + i * 0.1 + 0.04, 1e6)
        for i in range(400)
    ]

    # Alternating: maximum mean reversion.
    cases["sawtooth"] = [
        Bar(i * 86400.0, 100.0, 101.0, 99.0, 100.0 + (1.0 if i % 2 else -1.0), 1e6)
        for i in range(400)
    ]

    # Extreme volatility with huge gaps.
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.standard_normal(400) * 0.15))
    cases["extreme_vol"] = bars_from_arrays(px, np.full(400, 1e6), np.full(400, 0.15), rng)

    # Tiny prices (crypto dust) and huge prices (index levels).
    for name, scale in (("dust", 1e-6), ("huge", 1e6)):
        p = scale * np.exp(np.cumsum(rng.standard_normal(400) * 0.01))
        cases[name] = bars_from_arrays(p, np.full(400, 1e6), np.full(400, 0.01), rng)

    # Very short series (below warmup).
    cases["tiny"] = generate_null(20, seed=1)

    for name, bars in cases.items():
        try:
            sysm = TIA(cfg)
            for d in sysm.stream(bars):
                fu = d.fusion
                if fu is not None:
                    check(
                        0.0 < fu.p_success < 1.0 and math.isfinite(fu.log_odds),
                        f"[{name}] degenerate probability p={fu.p_success} L={fu.log_odds}",
                    )
                    check(fu.ebe >= 0.0, f"[{name}] negative breadth")
                check(math.isfinite(sysm.equity) and sysm.equity > 0, f"[{name}] equity broke")
            print(f"  {name:20s} ok  ({len(bars)} bars, {sum(1 for x in sysm.decisions if x.is_actionable)} signals)")
        except Exception:
            check(False, f"[{name}] raised: {traceback.format_exc(limit=3)}")

    # A bar stream containing corrupt bars the validator must reject.
    good = generate_null(300, seed=2)
    corrupt = list(good)
    corrupt.insert(150, good[149])  # exact duplicate timestamp
    try:
        sysm = TIA(cfg)
        sysm.run(corrupt)
        rep = sysm.validator.report
        check(rep.n_rejected >= 1, "validator did not reject a duplicate timestamp")
        print(f"  {'corrupt_stream':20s} ok  (rejected {rep.n_rejected})")
    except Exception:
        check(False, f"[corrupt_stream] raised: {traceback.format_exc(limit=3)}")


# ---------------------------------------------------------------------------
# 4. Optional data paths
# ---------------------------------------------------------------------------


def audit_optional_paths() -> None:
    section("Optional data paths and configurations")
    base = generate_with_regimes(700, seed=5)[0]

    # Footprint delta.
    rng = np.random.default_rng(7)
    with_delta = [
        Bar(b.timestamp, b.open, b.high, b.low, b.close, b.volume,
            trades=float(rng.integers(50, 500)), delta=float(rng.normal(0, 0.2) * b.volume))
        for b in base
    ]
    cfg_delta = Config().replace(enable_delta_features=True)
    try:
        TIA(cfg_delta).run(with_delta)
        print("  delta/footprint feed   ok")
    except Exception:
        check(False, f"delta path raised: {traceback.format_exc(limit=3)}")

    # Exogenous: peers, implied vol, events, spread, positioning.
    exog = [
        ExogenousSnapshot(
            peers={"SPY": float(rng.normal(0, 1)), "QQQ": float(rng.normal(0, 1))},
            implied_vol=15.0 + float(rng.normal(0, 2)),
            days_to_event=float(i % 30),
            spread=5e-4,
            gamma_imbalance=float(np.clip(rng.normal(0, 0.4), -1, 1)),
            oi_change=float(rng.normal(0, 1)),
        )
        for i in range(len(base))
    ]
    for name, cfg in (
        ("crossasset on", Config()),
        ("positioning on", Config().replace(enable_positioning_engine=True)),
        ("shorts disabled", Config().replace(allow_shorts=False)),
    ):
        try:
            s = TIA(cfg)
            s.run(base, exog=exog)
            if cfg.allow_shorts is False:
                check(
                    all(d.action is not Action.SHORT for d in s.decisions),
                    "shorts emitted while allow_shorts=False",
                )
            print(f"  {name:22s} ok")
        except Exception:
            check(False, f"{name} raised: {traceback.format_exc(limit=3)}")

    # Every session profile must run.
    for pname, spec in PROFILES.items():
        try:
            TIA(Config(), session=spec).run(base[:400])
            print(f"  session={pname:12s}   ok")
        except Exception:
            check(False, f"session {pname} raised: {traceback.format_exc(limit=3)}")


# ---------------------------------------------------------------------------
# 5. Numerical edge cases in the maths
# ---------------------------------------------------------------------------


def audit_numerics() -> None:
    section("Numerical edge cases")

    # Student-t quantile against known table values and limiting behaviour.
    for dof, want in ((5.0, -1.476), (10.0, -1.372), (30.0, -1.310), (100.0, -1.290)):
        got = student_t_ppf(0.1, dof)
        check(abs(got - want) < 0.02, f"student_t_ppf(0.1,{dof}) = {got:.4f}, want ~{want}")
    check(student_t_ppf(0.5, 10.0) == 0.0 or abs(student_t_ppf(0.5, 10.0)) < 1e-9,
          "student_t_ppf(0.5) should be 0")
    check(student_t_ppf(0.1, 1.5) < student_t_ppf(0.1, 30.0),
          "t quantile should widen as dof falls")

    # Edge Book must stay finite under extreme and adversarial inputs.
    book = EdgeBook(Config())
    for v in (0.0, 1e-12, -1e-12, 50.0, -50.0, 1e6, -1e6):
        book.observe(0, 0, 0, v, weight=1.0)
    est = book.estimate(0, 0, 0)
    for name in ("mean", "lcb", "ucb", "outcome_var"):
        check(math.isfinite(getattr(est, name)), f"EdgeBook.{name} not finite after extreme inputs")
    check(est.lcb <= est.mean <= est.ucb, "EdgeBook bound ordering violated")

    # Zero and negative weights must be ignored, not corrupt the cell.
    b2 = EdgeBook(Config())
    before = b2.estimate(1, 1, 1)
    b2.observe(1, 1, 1, 5.0, weight=0.0)
    b2.observe(1, 1, 1, 5.0, weight=-3.0)
    b2.observe(1, 1, 1, float("nan"), weight=1.0)
    b2.observe(1, 1, 1, float("inf"), weight=1.0)
    after = b2.estimate(1, 1, 1)
    check(abs(before.mean - after.mean) < 1e-12, "EdgeBook accepted a zero/negative/NaN observation")

    # CALOP with pathological inputs.
    from tia.types import EngineOutput

    names = [f"e{i}" for i in range(5)]
    pooler = CALOP(names, Config())
    for scores, rels, label in (
        ([0.0] * 5, [0.0] * 5, "all muted"),
        ([1.0] * 5, [1.0] * 5, "all saturated"),
        ([-1.0] * 5, [1.0] * 5, "all saturated short"),
        ([1.0, -1.0, 1.0, -1.0, 0.0], [1.0] * 5, "maximally split"),
        ([0.5] * 5, [1e-12] * 5, "vanishing reliability"),
    ):
        outs = {n: EngineOutput(n, s, r) for n, s, r in zip(names, scores, rels)}
        res = pooler.fuse(outs)
        check(math.isfinite(res.log_odds), f"CALOP [{label}] log_odds not finite")
        check(0.0 <= res.p_success <= 1.0, f"CALOP [{label}] p out of range")
        check(res.ebe >= 0.0, f"CALOP [{label}] negative breadth")
        check(res.p_low <= res.p_high + 1e-12, f"CALOP [{label}] interval inverted")


# ---------------------------------------------------------------------------
# 6. Determinism, reset, and state round-trips
# ---------------------------------------------------------------------------


def audit_determinism() -> None:
    section("Determinism, reset and state round-trips")
    bars = generate_with_regimes(700, seed=9)[0]
    cfg = Config()

    a = [(d.action, d.fusion.log_odds, d.expected_value_lcb) for d in TIA(cfg).stream(bars)]
    b = [(d.action, d.fusion.log_odds, d.expected_value_lcb) for d in TIA(cfg).stream(bars)]
    check(a == b, "two identical runs produced different results")

    s = TIA(cfg)
    first = [d.action for d in s.stream(bars)]
    s.reset()
    second = [d.action for d in s.stream(bars)]
    check(first == second, "reset did not restore initial state")

    # Edge Book state round-trip after real training.
    tb = generate_with_regimes(2500, seed=10)[0]
    cands = collect_candidates(tb, cfg)
    book, _, _ = train_edge_book(tb, cands, cfg)
    clone = EdgeBook(cfg)
    clone.load_state_dict(book.state_dict())
    mismatch = 0
    for r in range(4):
        for st in range(4):
            for bu in range(cfg.edge_evidence_buckets):
                x, y = book.estimate(r, st, bu), clone.estimate(r, st, bu)
                if abs(x.mean - y.mean) > 1e-12 or abs(x.lcb - y.lcb) > 1e-12:
                    mismatch += 1
    check(mismatch == 0, f"EdgeBook state round-trip mismatched in {mismatch} cells")


# ---------------------------------------------------------------------------
# 7. Behavioural properties the system claims
# ---------------------------------------------------------------------------


def audit_claims() -> None:
    section("Claimed behavioural properties")
    cfg = Config()

    # Null markets must yield essentially nothing, across many seeds.
    total = 0
    for seed in range(8):
        s = TIA(cfg)
        s.run(generate_null(2500, seed=200 + seed))
        total += sum(1 for d in s.decisions if d.is_actionable)
    check(total <= 5, f"null markets produced {total} signals across 8 seeds")
    print(f"  null markets: {total} signals over 8 x 2500 bars")

    # The gate must open on a demonstrable edge (guards against a stuck gate).
    opened = []
    for strength in (0.30, 0.80):
        bars, _ = generate_with_regimes(
            5000, seed=77, trend_strength=strength, revert_strength=strength * 1.5
        )
        ex = [ExogenousSnapshot(spread=5e-4)] * len(bars)
        c = collect_candidates(bars, cfg, exog=ex)
        book, cal, diag = train_edge_book(bars, c, cfg)
        s = TIA(cfg, edge_book=book, learn=False)
        s.calibrator._iso = cal
        s.calibrator.active = cal.n_fit > 0
        s.run(bars, exog=ex)
        n = sum(1 for d in s.decisions if d.is_actionable)
        opened.append(n)
        print(f"  edge={strength:.2f}: t={diag.get('t_stat_effective', float('nan')):+.2f}  signals={n}")
    check(max(opened) > 0, "the gate never opens even on a large, demonstrable edge")

    # Selectivity: eligibility must stay low on a realistic market.
    bars, _ = generate_with_regimes(4000, seed=11)
    c = collect_candidates(bars, cfg)
    check(
        c.eligibility_rate < 0.05,
        f"structural eligibility {100 * c.eligibility_rate:.1f}% is not selective",
    )
    print(f"  structural eligibility: {100 * c.eligibility_rate:.2f}%")


# ---------------------------------------------------------------------------
# 8. Static hygiene
# ---------------------------------------------------------------------------


def audit_static() -> None:
    section("Static hygiene")
    src = REPO / "python" / "src" / "tia"

    # No module in the live path may import an optional dependency.
    live = ["features", "engines", "fusion", "decision", "risk", "data"]
    for sub in live:
        for py in (src / sub).rglob("*.py"):
            text = py.read_text()
            for banned in ("import pandas", "import scipy", "import sklearn", "import statsmodels"):
                if banned in text and "try:" not in text.split(banned)[0][-200:]:
                    check(False, f"{py.relative_to(src)}: unguarded '{banned}' in the live path")

    # Every module must compile.
    for py in src.rglob("*.py"):
        try:
            compile(py.read_text(), str(py), "exec")
        except SyntaxError as exc:
            check(False, f"{py.relative_to(src)}: syntax error {exc}")

    # The CLI must expose every documented subcommand.
    out = subprocess.run(
        [sys.executable, "-m", "tia.cli", "--help"],
        capture_output=True, text=True, cwd=str(REPO / "python"),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    for cmd in ("demo", "run", "train", "validate", "config", "export"):
        check(cmd in out.stdout, f"CLI missing subcommand '{cmd}'")


# ---------------------------------------------------------------------------


def main() -> int:
    print("TI-A deep invariant audit")
    audit_pine_freshness()
    audit_export_sanity()
    audit_static()
    audit_numerics()
    audit_determinism()
    audit_degenerate_inputs()
    audit_optional_paths()
    audit_decision_invariants()
    audit_claims()

    print(f"\n{'=' * 70}")
    print(f"{CHECKS} checks run, {len(FAILURES)} violations")
    print("=" * 70)
    seen: set[str] = set()
    for f in FAILURES:
        key = f.split("]")[-1][:80]
        if key in seen:
            continue
        seen.add(key)
        print(f"  FAIL  {f}")
    if len(FAILURES) > len(seen):
        print(f"  ... {len(FAILURES) - len(seen)} further occurrences suppressed")
    return len(FAILURES)


if __name__ == "__main__":
    raise SystemExit(min(main(), 250))
