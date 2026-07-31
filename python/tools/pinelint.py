"""A static checker for the Pine Script port.

TradingView's compiler is not available here, so this encodes the Pine v6
constraints that a Python-trained eye reliably gets wrong. It is not a parser and
does not claim completeness; it catches the specific classes of error that are
easy to write, impossible to see by reading, and fatal at compile time.

Run:  cd python && python3 tools/pinelint.py
Exit code is the number of problems found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Built-in namespaces and variables. Shadowing any of these with a parameter or
#: local silently breaks every later use of the namespace -- `str.split` stops
#: resolving the moment a parameter is named `str`.
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

ERRORS: list[str] = []


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
            if c == '"':
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


def check_file(path: Path) -> None:
    raw = path.read_text().splitlines()
    text = path.read_text()

    # --- version and declaration ------------------------------------------
    if not any(l.strip().startswith("//@version=6") for l in raw[:40]):
        err(path, 1, "missing //@version=6")
    if not re.search(r"^\s*(indicator|strategy)\s*\(", text, re.M):
        err(path, 1, "no indicator() or strategy() declaration")

    # --- balanced delimiters ----------------------------------------------
    code = "\n".join(strip_strings_and_comments(l) for l in raw)
    for o, c, name in (("(", ")", "parentheses"), ("[", "]", "brackets")):
        if code.count(o) != code.count(c):
            err(path, 1, f"unbalanced {name}: {code.count(o)} vs {code.count(c)}")

    defined: set[str] = set()
    fn_params: dict[int, list[str]] = {}
    depth_carry = 0  # unclosed delimiters carried in from previous lines

    for n, line in enumerate(raw, 1):
        c = strip_strings_and_comments(line)
        stripped = c.strip()
        opened = depth_carry
        depth_carry += c.count("(") - c.count(")") + c.count("[") - c.count("]")
        if not stripped:
            continue
        if opened > 0:
            # A continuation line of a wrapped call. Named arguments there look
            # exactly like assignments and are not.
            continue

        # --- comma-separated multiple assignment --------------------------
        # `a = 1, b = 2` is valid Python and invalid Pine. Tuple unpacking
        # `[a, b] = f()` is the only comma form Pine accepts.
        if not stripped.startswith("[") and not stripped.startswith(_NON_ASSIGN):
            # Count top-level `=` assignments outside any bracket nesting.
            depth, assigns = 0, 0
            j = 0
            while j < len(c):
                ch = c[j]
                if ch in "([":
                    depth += 1
                elif ch in ")]":
                    depth -= 1
                elif depth == 0 and ch == "=":
                    prev = c[j - 1] if j else " "
                    nxt = c[j + 1] if j + 1 < len(c) else " "
                    if prev not in "=!<>:" and nxt != "=":
                        assigns += 1
                j += 1
            if assigns > 1:
                err(path, n, "comma-separated multiple assignment is not valid Pine", line)

        # --- namespace shadowing ------------------------------------------
        m = re.match(r"^\s*(?:var\s+)?(?:\w+\s+)?(\w+)\s*(?::=|=)(?!=)", c)
        if m:
            name = m.group(1)
            defined.add(name)
            if name in RESERVED:
                err(path, n, f"'{name}' shadows a Pine built-in namespace/variable", line)

        # --- function definitions and their parameters --------------------
        fm = re.match(r"^\s*(\w+)\s*\(([^)]*)\)\s*=>", c)
        if fm:
            defined.add(fm.group(1))
            params = [p.strip().split()[-1] for p in fm.group(2).split(",") if p.strip()]
            fn_params[n] = params
            for p in params:
                if p in RESERVED:
                    err(path, n, f"parameter '{p}' shadows a Pine built-in namespace", line)
                defined.add(p)

        # --- Python-isms that compile in neither language ------------------
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

        # --- indentation must be a multiple of four ------------------------
        indent = len(line) - len(line.lstrip(" "))
        if line.strip() and indent % 4 != 0 and not line.lstrip().startswith("//"):
            # Continuation lines of a wrapped call are the legitimate exception.
            prev = raw[n - 2].rstrip() if n >= 2 else ""
            if not prev.endswith((",", "(", "?", ":", "and", "or")):
                err(path, n, f"indentation {indent} is not a multiple of 4", line)

    # --- every FM_ constant used must be defined in the generated block ----
    if "BEGIN FROZEN MODEL" in text:
        block = text[text.index("BEGIN FROZEN MODEL"):text.index("END FROZEN MODEL")]
        declared = set(re.findall(r"^(FM_[A-Z_0-9]+)", block, re.M))
        used = set(re.findall(r"\b(FM_[A-Z_0-9]+)\b", text.replace(block, "")))
        for name in sorted(used - declared):
            err(path, 1, f"{name} is used but not declared in the frozen-model block")
        unused = declared - used - {"FM_SCHEMA", "FM_HASH", "FM_GENERATED", "FM_PLACEHOLDER"}
        if unused:
            print(f"  note {path.name}: {len(unused)} declared constants unused "
                  f"({', '.join(sorted(unused)[:5])}{'...' if len(unused) > 5 else ''})")

    # --- non-repainting requirements --------------------------------------
    for n, line in enumerate(raw, 1):
        if "request.security" in line:
            if "lookahead=barmerge.lookahead_off" not in line:
                err(path, n, "request.security without lookahead_off REPAINTS", line)
            if not re.search(r"\[\d+\]\s*,", line):
                err(path, n, "request.security without an [n] offset repaints intrabar", line)
    if "barstate.isconfirmed" not in text:
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
    print(f"\n{len(ERRORS)} problems")
    for e in ERRORS:
        print(f"  {e}")
    return len(ERRORS)


if __name__ == "__main__":
    raise SystemExit(min(main(), 250))
