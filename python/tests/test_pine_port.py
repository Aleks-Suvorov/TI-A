"""Regression checks for the TradingView port.

TradingView's compiler is not available here, so these tests guard the
properties that a compile would otherwise catch late and expensively:

  * the two ``.pine`` files share a byte-identical computational core, so the
    strategy can never quietly trade something the indicator does not paint;
  * the static checker reports zero problems;
  * every ``FM_`` constant the scripts reference exists in the generated
    block, and the block matches ``pine/frozen_model.json``;
  * the port's arithmetic, transliterated into Python, produces signals on
    real-shaped data rather than sitting in a permanent veto -- the failure
    that a compiling-but-inert script exhibits;
  * the port abstains where it is supposed to abstain.

They do NOT prove the files compile. Only TradingView can do that, and the
manual checklist in ``pine/START_HERE_TRADINGVIEW.md`` is how that gets done.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PINE = ROOT / "pine"
TOOLS = ROOT / "python" / "tools"

sys.path.insert(0, str(TOOLS))

CORE_BEGIN = "// ==== BEGIN SHARED CORE ===="
CORE_END = "// ==== END SHARED CORE ===="


def core_of(path: Path) -> str:
    text = path.read_text()
    b = text.index(CORE_BEGIN)
    e = text.index(CORE_END) + len(CORE_END)
    return text[b:e]


@pytest.fixture(scope="module")
def model() -> dict:
    return json.loads((PINE / "frozen_model.json").read_text())


# --------------------------------------------------------------------------- #
def test_both_pine_files_exist_and_declare_v6():
    for name in ("TIA.pine", "TIA_strategy.pine"):
        text = (PINE / name).read_text()
        assert text.splitlines()[0].strip() == "//@version=6", (
            f"{name}: //@version=6 must be the first line, or TradingView "
            f"silently compiles the file as Pine v1"
        )


def test_indicator_and_strategy_share_an_identical_core():
    """The one invariant that keeps the two scripts honest with each other.

    If this fails, someone edited the computation in one file only, and the
    Strategy Tester is now measuring something the chart does not show.
    """
    a = core_of(PINE / "TIA.pine")
    b = core_of(PINE / "TIA_strategy.pine")
    if a != b:
        la, lb = a.splitlines(), b.splitlines()
        first = next((i for i in range(min(len(la), len(lb))) if la[i] != lb[i]), min(len(la), len(lb)))
        pytest.fail(
            f"shared core diverged at core line {first + 1}:\n"
            f"  TIA.pine          : {la[first] if first < len(la) else '<eof>'}\n"
            f"  TIA_strategy.pine : {lb[first] if first < len(lb) else '<eof>'}"
        )


def test_core_only_depends_on_two_host_names():
    """The core is spliceable only if it imports nothing but isModeA/strictIn."""
    core = core_of(PINE / "TIA.pine")
    for name in ("allowLong", "allowShort", "cooldownIn", "maxPer100In", "showCard"):
        assert name not in core, (
            f"'{name}' is a host-file input and must not appear inside the "
            f"shared core, or the splice into the strategy breaks"
        )


def test_pinelint_reports_no_problems():
    r = subprocess.run([sys.executable, str(TOOLS / "pinelint.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"pinelint found problems:\n{r.stdout}\n{r.stderr}"


def test_no_request_security_anywhere():
    """The port builds its higher timeframe in-script, on purpose.

    request.security with a mutable-variable expression is a compile error,
    and the timeframe string the previous build assembled was invalid on
    weekly charts. Neither can come back by accident.
    """
    for name in ("TIA.pine", "TIA_strategy.pine"):
        code = "\n".join(
            l.split("//")[0] for l in (PINE / name).read_text().splitlines()
        )
        assert "request.security" not in code, f"{name} reintroduced request.security"


def test_no_hyperbolic_math_builtins():
    """Pine has `math.tan` but no `tanh`, `sinh` or `cosh`.

    `math.tanh` shipped once. The direct error was CE10271 ("Could not find
    function or function reference"), but the damage was indirect: every value
    downstream of the missing call took type "unknown", so TradingView reported
    seven errors, six of them CE10122 complaints about `str.format` arguments
    a hundred lines away. The squash function is now `f_tanh`, which agrees
    with the real tanh to one ULP.
    """
    for name in ("TIA.pine", "TIA_strategy.pine"):
        code = "\n".join(
            l.split("//")[0] for l in (PINE / name).read_text().splitlines()
        )
        for fn in ("math.tanh", "math.sinh", "math.cosh"):
            assert fn not in code, f"{name} calls {fn}, which does not exist in Pine"
        assert "f_tanh(" in code, f"{name}: the f_tanh replacement is missing"


def test_alerts_avoid_str_format_overload_resolution():
    """Alert bodies are concatenated, not formatted.

    `str.format` resolves against a typed overload set, so a single upstream
    type error becomes a cascade of confusing complaints about its arguments
    rather than one error at the real cause. `+` on strings has no overloads.
    """
    for name in ("TIA.pine", "TIA_strategy.pine"):
        code = "\n".join(
            l.split("//")[0] for l in (PINE / name).read_text().splitlines()
        )
        assert "str.format" not in code, (
            f"{name}: use string concatenation in alert bodies"
        )


def test_every_frozen_constant_used_is_declared():
    import re
    for name in ("TIA.pine", "TIA_strategy.pine"):
        text = (PINE / name).read_text()
        block = text[text.index("BEGIN FROZEN MODEL"):text.index("END FROZEN MODEL")]
        declared = set(re.findall(r"^(FM_[A-Z_0-9]+)", block, re.M))
        used = set(re.findall(r"\b(FM_[A-Z_0-9]+)\b", text.replace(block, "")))
        assert not (used - declared), f"{name}: undeclared {sorted(used - declared)}"


def test_frozen_block_matches_the_exported_model(model):
    """A hand-edited constant no longer corresponds to any manifest hash."""
    import re
    for name in ("TIA.pine", "TIA_strategy.pine"):
        text = (PINE / name).read_text()
        m = re.search(r'FM_HASH\s*=\s*"([0-9a-f]+)"', text)
        assert m, f"{name}: no FM_HASH"
        assert m.group(1) == model["config_manifest_hash"], (
            f"{name} was generated from a different model than "
            f"pine/frozen_model.json; re-run the exporter"
        )


def test_signals_are_emitted_only_on_confirmed_bars():
    for name in ("TIA.pine", "TIA_strategy.pine"):
        text = (PINE / name).read_text()
        assert "barstate.isconfirmed" in text
        assert "confirmed and" in text, (
            f"{name}: the confirmed-bar guard exists but is not applied to a signal"
        )


def test_table_rows_cover_every_diagnostic_field():
    """The card must name at least fourteen fields including the veto reason."""
    text = (PINE / "TIA.pine").read_text()
    required = ["Mode", "Warm-up", "Position", "Direction", "Confidence", "Regime",
                "Regime hazard", "Setup", "Trend", "Momentum", "Volatility",
                "Liquidity", "Higher TF", "Evidence", "Gates passed",
                "Stop / target", "Why no trade"]
    missing = [r for r in required if f'"{r}"' not in text]
    assert not missing, f"diagnostics card is missing rows: {missing}"


def test_all_six_alertconditions_exist():
    for name in ("TIA.pine", "TIA_strategy.pine"):
        text = (PINE / name).read_text()
        for title in ("TI-A BUY", "TI-A SELL", "TI-A SHORT", "TI-A COVER",
                      "TI-A any ENTRY", "TI-A any EXIT"):
            assert f'title="{title}"' in text, f"{name}: missing alertcondition {title}"


def test_strategy_declares_costs_and_forbids_pyramiding():
    text = (PINE / "TIA_strategy.pine").read_text()
    decl = text[text.index("strategy("):text.index("// ═", text.index("strategy("))]
    assert "pyramiding=0" in decl, "pyramiding must be disabled"
    assert "commission_type" in decl and "commission_value" in decl
    assert "slippage=" in decl
    assert "process_orders_on_close=false" in decl, (
        "orders must fill on the NEXT bar's open, not the signal bar's close"
    )
    assert "calc_on_every_tick=false" in decl


def test_strategy_brackets_the_position_on_the_signal_bar():
    """The first bar of a trade must not be naked.

    An exit submitted only after the emulator reports the fill activates one
    bar late, leaving the opening bar -- the one most likely to gap -- with no
    stop. The bracket is therefore keyed off `liveDir`, which is set on the
    signal bar, not off `strategy.position_size`, which is not.
    """
    text = (PINE / "TIA_strategy.pine").read_text()
    tail = text[text.index(CORE_END):]
    assert "liveDir  := 1" in tail and "liveDir  := -1" in tail
    assert "if liveDir == 1 and not na(stopPx)" in tail, (
        "strategy.exit must be gated on liveDir (set on the signal bar), not on "
        "posState (set only once the emulator reports a fill)"
    )
    assert "if posState == 1 and not na(stopPx)" not in tail, (
        "bracket submission regressed to post-fill timing"
    )


def test_strategy_reads_position_from_the_broker_emulator():
    text = (PINE / "TIA_strategy.pine").read_text()
    tail = text[text.index(CORE_END):]
    assert "strategy.position_size" in tail, (
        "the strategy must take its position from the emulator, not a private counter"
    )


# --------------------------------------------------------------------------- #
# Behavioural checks, via the Python transliteration in tools/pinesim.py.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bars():
    from tia.synthetic import generate_with_regimes
    b, _ = generate_with_regimes(4000, seed=11)
    return [(x.open, x.high, x.low, x.close, x.volume) for x in b]


def run(model, bars, **kw):
    from pinesim import PineSim
    sim = PineSim(model, **kw)
    for b in bars:
        sim.on_bar(*b)
    return sim


def test_mode_b_produces_signals_rather_than_sitting_inert(model, bars):
    """A script that compiles and paints nothing is the failure being guarded.

    The count is deliberately loose: this asserts the gate is reachable, not
    that any particular frequency is correct.
    """
    sim = run(model, bars, mode_a=False, strict=3)
    entries = [s for s in sim.signals if s[1] in ("BUY", "SHORT")]
    assert len(entries) >= 5, (
        f"Mode B produced {len(entries)} entries on 4000 bars; the binding "
        f"constraint was {sim.vetoes.most_common(3)}"
    )


def test_strictness_is_monotone(model, bars):
    counts = []
    for s in (1, 2, 3, 4, 5):
        sim = run(model, bars, mode_a=False, strict=s)
        counts.append(sum(1 for x in sim.signals if x[1] in ("BUY", "SHORT")))
    assert counts == sorted(counts, reverse=True), (
        f"raising strictness must never increase the signal count: {counts}"
    )


def test_mode_a_is_far_more_selective_than_mode_b(model, bars):
    a = run(model, bars, mode_a=True)
    b = run(model, bars, mode_a=False, strict=3)
    na = sum(1 for x in a.signals if x[1] in ("BUY", "SHORT"))
    nb = sum(1 for x in b.signals if x[1] in ("BUY", "SHORT"))
    assert na <= nb, "the frozen model's gate must not be looser than observation mode"


def test_direction_switches_are_respected(model, bars):
    long_only = run(model, bars, mode_a=False, strict=2, allow_short=False)
    assert not any(x[1] == "SHORT" for x in long_only.signals)
    short_only = run(model, bars, mode_a=False, strict=2, allow_long=False)
    assert not any(x[1] == "BUY" for x in short_only.signals)


def test_no_non_finite_values_reach_the_gate(model, bars):
    sim = run(model, bars, mode_a=False, strict=3)
    assert not sim.nonfinite, f"non-finite features: {dict(sim.nonfinite)}"


def test_entries_and_exits_reconcile(model, bars):
    sim = run(model, bars, mode_a=False, strict=1)
    ent = sum(1 for x in sim.signals if x[1] in ("BUY", "SHORT"))
    ex = sum(1 for x in sim.signals if x[1] in ("SELL", "COVER"))
    assert ent - ex == (1 if sim.pos else 0), (
        f"{ent} entries but {ex} exits with position {sim.pos}: an exit was lost, "
        f"which is how a system silently censors its own losers"
    )


def test_cooldown_and_rate_limit_are_enforced(model, bars):
    sim = run(model, bars, mode_a=False, strict=1, cooldown=25, max_per_100=2)
    entries = [x[0] for x in sim.signals if x[1] in ("BUY", "SHORT")]
    for a, b in zip(entries, entries[1:]):
        assert b - a >= 1
    for k, bar in enumerate(entries):
        window = [e for e in entries if 0 <= bar - e <= 100]
        assert len(window) <= 3, f"rate limit exceeded near bar {bar}: {window}"


def test_warmup_blocks_every_early_bar(model, bars):
    sim = run(model, bars, mode_a=False, strict=1)
    first = min((x[0] for x in sim.signals), default=10 ** 9)
    assert first >= sim.WARMUP, (
        f"a signal was emitted at bar {first}, before the {sim.WARMUP}-bar warm-up"
    )


def test_stops_sit_on_the_correct_side_of_every_entry(model, bars):
    sim = run(model, bars, mode_a=False, strict=1)
    for _, kind, px, stop, target, _ in sim.signals:
        if kind == "BUY":
            assert stop < px < target, f"long at {px} with stop {stop}, target {target}"
        elif kind == "SHORT":
            assert target < px < stop, f"short at {px} with stop {stop}, target {target}"
