"""A static checker for the Pine Script port.

TradingView's compiler is not available in this environment, so this encodes
the Pine v6 constraints that a Python-trained eye reliably gets wrong. It is
not a parser and does not claim completeness: it catches the specific classes
of error that are easy to write, impossible to see by reading, and fatal at
compile time.

What it checks, and why each rule exists (every one earned its place by
catching a real defect in this repository's history):

  1.  ``//@version=6`` present, and the declaration statement present.
  2.  Balanced delimiters.
  3.  Comma-separated multiple assignment -- valid Python, invalid Pine.
  4.  Namespace shadowing. A parameter named ``str`` silently breaks every
      later ``str.split``; that bug shipped once and cost a compile cycle.
  5.  Python-isms: ``return``, ``True/False/None``, bare ``max/min/abs``,
      ``nan``, ``**``, ``elif``.
  6.  Indentation in multiples of four.
  7.  Series-qualified lengths in the TA calls that demand ``simple int``.
      ``math.sum(x, n)`` with a mutable ``n`` is the classic one.
  8.  Functions used before they are defined -- Pine resolves top-down.
  9.  Untyped function parameters, reported as a note. Explicit types stop
      Pine from having to infer a qualifier it cannot always get right.
  10. Table cell indices beyond the declared table size -- a runtime error
      TradingView reports only once the script is on a chart.
  11. ``plot``/``plotshape``/``bgcolor``/``alertcondition``/``hline``/
      ``fill``/``input.*`` inside a local block, which Pine forbids.
  12. ``:=`` to an identifier that was never declared.
  13. History depth beyond the declared ``max_bars_back``.
  14. ``request.security`` without ``lookahead_off`` and without an offset,
      scanned over CODE only -- the previous version of this rule flagged the
      comment that says the script contains no ``request.security`` at all.
  15. Every ``FM_`` constant used is declared in the generated block.
  16. A ``barstate.isconfirmed`` guard exists.

Run:  cd python && python3 tools/pinelint.py
Exit code is the number of problems found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Built-in namespaces and variables. Shadowing any of these with a parameter
#: or local silently breaks every later use of the namespace -- ``str.split``
#: stops resolving the moment a parameter is named ``str``.
RESERVED = {
    "str", "math", "array", "matrix", "map", "table", "ta", "request", "color",
    "syminfo", "timeframe", "session", "strategy", "ticker", "label", "line",
    "box", "polyline", "input", "plot", "alert", "runtime", "chart", "currency",
    "dayofweek", "display", "format", "hline", "location", "order", "position",
    "scale", "shape", "size", "text", "xloc", "yloc", "barstate", "bar_index",
    "open", "high", "low", "close", "volume", "time", "hl2", "hlc3", "ohlc4",
    "na", "int", "float", "bool", "string", "line_style", "extend",
}

#: Keywords that legitimately begin a line and are not assignments.
_NON_ASSIGN = ("if ", "else", "for ", "while ", "switch", "//", "type ", "import ", "method ")

#: Calls whose length/lookback argument Pine requires to be a *simple int*:
#: a literal, an ``input.int`` result, or a name bound to one. A series value
#: there is a compile error, and it is the single easiest mistake to make when
#: porting from Python where every length is just a number.
SIMPLE_LEN_CALLS = {
    "math.sum": 1, "ta.sma": 1, "ta.ema": 1, "ta.rma": 1, "ta.wma": 1,
    "ta.variance": 1, "ta.stdev": 1, "ta.percentrank": 1, "ta.percentile_linear_interpolation": 1,
    "ta.median": 1, "ta.mode": 1, "ta.highest": 1, "ta.lowest": 1, "ta.atr": 0,
    "ta.change": 1, "ta.mom": 1, "ta.roc": 1, "ta.correlation": 2, "ta.linreg": 1,
    "ta.cci": 1, "ta.rsi": 1, "ta.wpr": 0, "ta.cmo": 1, "ta.dev": 1,
}

#: Constructs Pine only accepts at the outermost scope.
GLOBAL_ONLY = ("plot(", "plotshape(", "plotchar(", "plotarrow(", "plotcandle(",
               "plotbar(", "bgcolor(", "alertcondition(", "hline(", "fill(",
               "barcolor(", "input.", "indicator(", "strategy(")

ERRORS: list[str] = []
NOTES: list[str] = []


def err(path: Path, lineno: int, msg: str, line: str = "") -> None:
    tail = f"\n        {line.strip()[:100]}" if line else ""
    ERRORS.append(f"{path.name}:{lineno}: {msg}{tail}")


def strip_strings_and_comments(line: str) -> str:
    """Blank out string literals and trailing comments so scans see only code."""
    out, i, in_str = [], 0, False
    while i < len(line):
        c = line[i]
        if in_str:
            out.append(" ")
            if c == '"' and line[i - 1] != "\\":
                in_str = False
        elif c == '"':
            in_str = True
            out.append(" ")
        elif c == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


def split_args(inner: str) -> list[str]:
    """Split a call's argument list on top-level commas."""
    args, depth, cur = [], 0, []
    for ch in inner:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return args


def call_args(code: str, start: int) -> tuple[str, int] | None:
    """Return the argument text of the call whose ``(`` follows ``start``."""
    i = code.find("(", start)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(code):
        if code[j] == "(":
            depth += 1
        elif code[j] == ")":
            depth -= 1
            if depth == 0:
                return code[i + 1:j], j
        j += 1
    return None


def check_file(path: Path) -> None:
    raw = path.read_text().splitlines()
    text = path.read_text()
    code_lines = [strip_strings_and_comments(l) for l in raw]
    code = "\n".join(code_lines)

    # --- version and declaration -------------------------------------------
    if not any(l.strip().startswith("//@version=6") for l in raw[:40]):
        err(path, 1, "missing //@version=6")
    if not re.search(r"^\s*(indicator|strategy)\s*\(", text, re.M):
        err(path, 1, "no indicator() or strategy() declaration")

    # --- balanced delimiters ------------------------------------------------
    for o, c, name in (("(", ")", "parentheses"), ("[", "]", "brackets")):
        if code.count(o) != code.count(c):
            err(path, 1, f"unbalanced {name}: {code.count(o)} vs {code.count(c)}")

    # --- names bound to compile-time constants ------------------------------
    # A name assigned a bare numeric literal at global scope is a `const int`
    # or `const float` and is therefore legal as a TA length. Anything else --
    # a var, a reassigned name, an expression -- is not.
    const_names: set[str] = set()
    reassigned: set[str] = set()
    declared: set[str] = set()
    functions: dict[str, int] = {}
    for n, c in enumerate(code_lines, 1):
        m = re.match(r"^(?:int|float)?\s*(\w+)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$", c.strip())
        if m and not c.startswith((" ", "\t")):
            const_names.add(m.group(1))
        m2 = re.match(r"^\s*(\w+)\s*:=", c)
        if m2:
            reassigned.add(m2.group(1))
    const_names -= reassigned

    depth_carry = 0
    for n, line in enumerate(raw, 1):
        c = code_lines[n - 1]
        stripped = c.strip()
        opened = depth_carry
        depth_carry += c.count("(") - c.count(")") + c.count("[") - c.count("]")
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        in_local = indent > 0

        # --- constructs Pine allows only at global scope --------------------
        # (a continuation line of a wrapped global call is indented but is not
        # a local scope, so only test lines that *start* a statement)
        if in_local and opened == 0:
            for g in GLOBAL_ONLY:
                if stripped.startswith(g):
                    err(path, n, f"`{g.rstrip('(.')}` cannot be called inside a local block", line)
                    break

        if opened > 0:
            # A continuation line of a wrapped call. Named arguments there look
            # exactly like assignments and are not.
            continue

        # --- comma-separated multiple assignment ----------------------------
        # `a = 1, b = 2` is valid Python and invalid Pine. Tuple unpacking
        # `[a, b] = f()` is the only comma form Pine accepts.
        if not stripped.startswith("[") and not stripped.startswith(_NON_ASSIGN):
            d, assigns, j = 0, 0, 0
            while j < len(c):
                ch = c[j]
                if ch in "([":
                    d += 1
                elif ch in ")]":
                    d -= 1
                elif d == 0 and ch == "=":
                    prev = c[j - 1] if j else " "
                    nxt = c[j + 1] if j + 1 < len(c) else " "
                    if prev not in "=!<>:+-*/%" and nxt != "=":
                        assigns += 1
                j += 1
            if assigns > 1:
                err(path, n, "comma-separated multiple assignment is not valid Pine", line)

        # --- namespace shadowing --------------------------------------------
        m = re.match(r"^\s*(?:var(?:ip)?\s+)?(?:\w+(?:\[\])?\s+)?(\w+)\s*(?::=|=)(?!=)", c)
        if m:
            name = m.group(1)
            declared.add(name)
            if name in RESERVED:
                err(path, n, f"'{name}' shadows a Pine built-in namespace/variable", line)

        # --- function definitions and their parameters ----------------------
        fm = re.match(r"^\s*(\w+)\s*\(([^)]*)\)\s*=>", c)
        if fm:
            functions[fm.group(1)] = n
            declared.add(fm.group(1))
            raw_params = [p.strip() for p in fm.group(2).split(",") if p.strip()]
            for p in raw_params:
                parts = p.split()
                name = parts[-1].split("=")[0].strip()
                declared.add(name)
                if name in RESERVED:
                    err(path, n, f"parameter '{name}' shadows a Pine built-in namespace", line)
                if len(parts) == 1 and "=" not in p:
                    NOTES.append(f"{path.name}:{n}: parameter '{name}' has no explicit type")

        # --- for-loop counters are declarations too --------------------------
        fl = re.match(r"^\s*for\s+(\w+)\s*=", c)
        if fl:
            declared.add(fl.group(1))

        # --- Python-isms that compile in neither language --------------------
        if re.search(r"\breturn\b", c):
            err(path, n, "Pine has no `return`; the last expression is the value", line)
        if re.search(r"\bTrue\b|\bFalse\b|\bNone\b", c):
            err(path, n, "Python literal (True/False/None); Pine uses true/false/na", line)
        if re.search(r"\bmax\s*\(|\bmin\s*\(|\babs\s*\(", c) and not re.search(
            r"math\.(max|min|abs)|array\.(max|min)|\w+_(max|min)|\.max\b|\.min\b", c
        ):
            err(path, n, "bare max/min/abs; Pine requires the math.* namespace", line)
        if re.search(r"\bnan\b", c):
            err(path, n, "`nan` is not Pine; use `na`", line)
        if "**" in c:
            err(path, n, "`**` is not a Pine operator; use math.pow", line)
        if re.search(r"\belif\b", c):
            err(path, n, "`elif` is not Pine; use `else if`", line)

        # --- indentation must be a multiple of four --------------------------
        if line.strip() and indent % 4 != 0 and not line.lstrip().startswith("//"):
            prev = raw[n - 2].rstrip() if n >= 2 else ""
            if not prev.endswith((",", "(", "?", ":", "and", "or")):
                err(path, n, f"indentation {indent} is not a multiple of 4", line)

    # --- series-qualified lengths where Pine demands a simple int ------------
    for n, c in enumerate(code_lines, 1):
        for fname, argpos in SIMPLE_LEN_CALLS.items():
            for m in re.finditer(re.escape(fname) + r"\s*\(", c):
                got = call_args(c, m.start() + len(fname) - 1)
                if got is None:
                    continue
                args = split_args(got[0])
                if len(args) <= argpos:
                    continue
                a = args[argpos].strip()
                if re.fullmatch(r"-?\d+", a):
                    continue                       # literal
                if a in const_names:
                    continue                       # bound to a literal, never reassigned
                if re.fullmatch(r"[A-Z_][A-Z_0-9]*", a):
                    continue                       # generated constant block
                if re.fullmatch(r"\w+\s*[-+]\s*\d+", a) and a.split()[0] in const_names:
                    continue                       # const arithmetic
                if a.startswith("input."):
                    continue                       # input.int is simple int
                err(path, n, f"{fname} length argument '{a}' may not be a simple int "
                             f"(Pine requires a literal, an input, or a const)", c)

    # --- functions must be defined before first use --------------------------
    for name, defline in functions.items():
        for n, c in enumerate(code_lines, 1):
            if n >= defline:
                break
            if re.search(r"(?<![.\w])" + re.escape(name) + r"\s*\(", c):
                err(path, n, f"'{name}' is used on line {n} but defined on line {defline}; "
                             f"Pine resolves top-down", raw[n - 1])
                break

    # --- := to an identifier that was never declared -------------------------
    for n, c in enumerate(code_lines, 1):
        m = re.match(r"^\s*(\w+)\s*:=", c)
        if m and m.group(1) not in declared:
            err(path, n, f"'{m.group(1)}' is assigned with := but never declared", raw[n - 1])

    # --- table cell indices within the declared table size -------------------
    tables: dict[str, tuple[int, int]] = {}
    for n, c in enumerate(code_lines, 1):
        m = re.search(r"(\w+)\s*=\s*table\.new\s*\(", c)
        if m:
            got = call_args(c, m.end() - 1)
            if got:
                args = [a for a in split_args(got[0]) if "=" not in a.split("(")[0]]
                if len(args) >= 3:
                    try:
                        tables[m.group(1)] = (int(args[1]), int(args[2]))
                    except ValueError:
                        pass
    for n, c in enumerate(code_lines, 1):
        for fn in ("table.cell", "table.clear"):
            for m in re.finditer(re.escape(fn) + r"\s*\(", c):
                got = call_args(c, m.end() - 1)
                if not got:
                    continue
                args = split_args(got[0])
                if not args or args[0] not in tables:
                    continue
                cols, rows = tables[args[0]]
                for idx, limit, what in ((1, cols, "column"), (2, rows, "row")):
                    if len(args) > idx and re.fullmatch(r"\d+", args[idx].strip()):
                        v = int(args[idx])
                        if v >= limit:
                            err(path, n, f"{fn} {what} {v} is outside the "
                                         f"{cols}x{rows} table '{args[0]}'", c)
                # table.clear takes end_column/end_row too
                if fn == "table.clear" and len(args) >= 5:
                    for idx, limit, what in ((3, cols, "end column"), (4, rows, "end row")):
                        if re.fullmatch(r"\d+", args[idx].strip()) and int(args[idx]) >= limit:
                            err(path, n, f"{fn} {what} {args[idx]} is outside the "
                                         f"{cols}x{rows} table '{args[0]}'", c)

    # --- history depth against the declared max_bars_back --------------------
    mbb = re.search(r"max_bars_back\s*=\s*(\d+)", code)
    limit = int(mbb.group(1)) if mbb else 300
    deepest = 0
    for n, c in enumerate(code_lines, 1):
        for m in re.finditer(r"\[(\d+)\]", c):
            deepest = max(deepest, int(m.group(1)))
    for n, c in enumerate(code_lines, 1):
        for fname, argpos in SIMPLE_LEN_CALLS.items():
            for m in re.finditer(re.escape(fname) + r"\s*\(", c):
                got = call_args(c, m.start() + len(fname) - 1)
                if got is None:
                    continue
                args = split_args(got[0])
                if len(args) > argpos and re.fullmatch(r"-?\d+", args[argpos].strip()):
                    deepest = max(deepest, int(args[argpos]))
    if deepest > limit:
        err(path, 1, f"deepest history reference is {deepest} bars but max_bars_back "
                     f"is {limit}; declare a larger max_bars_back")

    # --- every FM_ constant used must be declared in the generated block -----
    if "BEGIN FROZEN MODEL" in text:
        block = text[text.index("BEGIN FROZEN MODEL"):text.index("END FROZEN MODEL")]
        block_declared = set(re.findall(r"^(FM_[A-Z_0-9]+)", block, re.M))
        used = set(re.findall(r"\b(FM_[A-Z_0-9]+)\b", text.replace(block, "")))
        for name in sorted(used - block_declared):
            err(path, 1, f"{name} is used but not declared in the frozen-model block")
        unused = block_declared - used - {"FM_SCHEMA", "FM_HASH", "FM_GENERATED", "FM_PLACEHOLDER"}
        if unused:
            NOTES.append(f"{path.name}: {len(unused)} declared constants unused "
                         f"({', '.join(sorted(unused)[:6])}{'...' if len(unused) > 6 else ''})")

    # --- non-repainting requirements -----------------------------------------
    # Scanned over CODE, not comments. The previous version of this rule fired
    # on the comment stating that the script contains no request.security.
    for n, c in enumerate(code_lines, 1):
        if "request.security" in c:
            if "lookahead=barmerge.lookahead_off" not in c:
                err(path, n, "request.security without lookahead_off REPAINTS", raw[n - 1])
            if not re.search(r"\[\d+\]\s*,", c):
                err(path, n, "request.security without an [n] offset repaints intrabar", raw[n - 1])
    if "barstate.isconfirmed" not in code:
        err(path, 1, "no barstate.isconfirmed guard: signals may repaint intrabar")


def main() -> int:
    print("Pine static check")
    files = sorted((REPO / "pine").glob("*.pine"))
    if not files:
        print("  no .pine files found")
        return 1
    for f in files:
        check_file(f)
        print(f"  checked {f.name} ({len(f.read_text().splitlines())} lines)")
    for note in NOTES:
        print(f"  note {note}")
    print(f"\n{len(ERRORS)} problems")
    for e in ERRORS:
        print(f"  {e}")
    return len(ERRORS)


if __name__ == "__main__":
    raise SystemExit(min(main(), 250))
