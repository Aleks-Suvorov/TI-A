"""Plain-text validation report assembly.

The report leads with the configuration manifest hash, because a result that
cannot be tied to an exact parameter set is not a result. It closes with the
pre-registered acceptance verdict, criterion by criterion, so that a reader sees
the pass/fail decision rather than having to derive it from tables.

Deliberately text-only: a report that needs a plotting stack is a report that
stops being generated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..config import Config
from ..types import REGIME_NAMES

__all__ = ["Criterion", "ValidationReport", "build_report"]

_RULE = "=" * 78
_THIN = "-" * 78


@dataclass(slots=True)
class Criterion:
    """One pre-registered acceptance test."""

    name: str
    observed: float
    threshold: float
    direction: str  # ">=" or "<="
    note: str = ""

    @property
    def passed(self) -> bool:
        if self.observed != self.observed:
            return False
        return self.observed >= self.threshold if self.direction == ">=" else self.observed <= self.threshold

    def render(self) -> str:
        obs = "n/a" if self.observed != self.observed else f"{self.observed:+.4f}"
        verdict = "PASS" if self.passed else "FAIL"
        line = f"  [{verdict}] {self.name:34s} {obs:>10s}  {self.direction} {self.threshold:+.4f}"
        return line + (f"\n         {self.note}" if self.note else "")


@dataclass
class ValidationReport:
    cfg: Config
    sections: list[str] = field(default_factory=list)
    criteria: list[Criterion] = field(default_factory=list)

    # -- assembly ------------------------------------------------------- #
    def header(self) -> "ValidationReport":
        c = self.cfg
        self.sections.append(
            "\n".join(
                [
                    _RULE,
                    f"TI-A VALIDATION REPORT   version {c.version}   manifest {c.manifest_hash()}",
                    _RULE,
                    f"parameters fitted to market outcomes : {c.fitted_dof()}",
                    f"parameters updated online (causal)   : {c.online_dof()}",
                    f"configurations charged to the DSR    : {c.trials_ledger_count}",
                    "",
                    "A result that cannot be tied to an exact parameter set is not a",
                    "result. If this hash disagrees with a running system's, this report",
                    "does not describe that system.",
                ]
            )
        )
        return self

    def section(self, title: str, body: str) -> "ValidationReport":
        self.sections.append(f"\n{_THIN}\n{title}\n{_THIN}\n{body}")
        return self

    def add_criterion(self, c: Criterion) -> "ValidationReport":
        self.criteria.append(c)
        return self

    # -- content -------------------------------------------------------- #
    def trades_section(self, trades: Sequence) -> "ValidationReport":
        n = len(trades)
        if n == 0:
            return self.section(
                "TRADES",
                "No trades. For a system whose gate requires 90% credence in positive\n"
                "net expectancy, this is a legitimate outcome on a short or featureless\n"
                "sample -- it is not evidence of a defect.",
            )
        r = np.array([t.ret_sigma for t in trades], dtype=np.float64)
        wins = int((r > 0).sum())
        mu = float(r.mean())
        sd = float(r.std(ddof=1)) if n > 1 else math.nan
        se = sd / math.sqrt(n) if sd == sd and n > 1 else math.nan
        t_stat = mu / se if se == se and se > 0 else math.nan
        hit = wins / n
        hit_se = math.sqrt(hit * (1 - hit) / n) if n else math.nan
        dd = _max_drawdown(np.array([t.equity_after for t in trades], dtype=np.float64))

        body = "\n".join(
            [
                f"  trades                {n}",
                f"  hit rate              {100 * hit:.1f}%  +/- {100 * hit_se:.1f} (1 s.e.)",
                f"  expectancy            {mu:+.4f} sigma/trade",
                f"  standard error        {se:.4f}   t = {t_stat:+.2f}",
                f"  outcome sd            {sd:.4f} sigma",
                f"  max drawdown          {100 * dd:.2f}%",
                "",
                "  The standard error is printed next to the estimate deliberately.",
                f"  At n = {n}, the 95% interval on the hit rate spans roughly",
                f"  {100 * (hit - 1.96 * hit_se):.0f}%-{100 * (hit + 1.96 * hit_se):.0f}%.",
                "  See docs/01-THEORY.md section 12: establishing a 0.15-sigma expectancy",
                "  at t = 2 needs about 400 trades, which is why inference is pooled",
                "  across instruments rather than extracted from one.",
            ]
        )
        self.add_criterion(
            Criterion("expectancy t-statistic", t_stat, 3.0, ">=",
                      "pre-registered primary hypothesis")
        )
        self.add_criterion(
            Criterion("trade count for power", float(n), 400.0, ">=",
                      "below this the primary test has insufficient power to conclude")
        )
        self.add_criterion(Criterion("max drawdown", dd, 0.25, "<="))
        return self.section("TRADES", body)

    def regime_section(self, trades: Sequence) -> "ValidationReport":
        if not trades:
            return self
        rows = []
        for k, name in enumerate(REGIME_NAMES):
            sel = [t.ret_sigma for t in trades if t.regime == k]
            if not sel:
                rows.append(f"  {name:16s} no trades")
                continue
            a = np.array(sel)
            rows.append(
                f"  {name:16s} n={a.size:5d}  mean={a.mean():+.4f}  "
                f"hit={100 * float((a > 0).mean()):5.1f}%"
            )
        rows.append("")
        rows.append("  The theory makes a falsifiable prediction here: the profitable")
        rows.append("  action should differ in kind between Trending and Mean-Reverting.")
        rows.append("  If both regimes show the same behaviour, the regime layer is not")
        rows.append("  doing the job the architecture assigns it.")
        return self.section("PER-REGIME BREAKDOWN", "\n".join(rows))

    def calibration_section(self, health: Mapping[str, float]) -> "ValidationReport":
        if not health or health.get("n", 0) < 30:
            return self.section(
                "CALIBRATION",
                "  Too few resolved outcomes to assess calibration.",
            )
        brier = float(health.get("brier", math.nan))
        body = "\n".join(
            [
                f"  outcomes              {health.get('n', 0):.0f}",
                f"  Brier score           {brier:.4f}   (0.25 is a coin flip)",
                f"  reliability           {health.get('reliability', math.nan):.4f}   lower is better",
                f"  resolution            {health.get('resolution', math.nan):.4f}   higher is better",
                f"  skill vs base rate    {health.get('skill', math.nan):+.4f}",
                f"  expected cal. error   {health.get('ece', math.nan):.4f}",
                "",
                "  Reliability and resolution answer different questions. A forecaster",
                "  that always predicts the base rate is perfectly calibrated and",
                "  completely useless; only resolution is worth anything to a trading",
                "  system.",
            ]
        )
        self.add_criterion(Criterion("Brier score", brier, self.cfg.brier_alarm, "<="))
        self.add_criterion(
            Criterion("expected calibration error", float(health.get("ece", math.nan)), 0.05, "<=")
        )
        self.add_criterion(
            Criterion("Brier skill vs base rate", float(health.get("skill", math.nan)), 0.0, ">=")
        )
        return self.section("CALIBRATION", body)

    def breadth_section(self, diagnostics: Mapping[str, float]) -> "ValidationReport":
        if not diagnostics:
            return self
        ebe = float(diagnostics.get("ebe_at_full_reliability", math.nan))
        body = "\n".join(
            [
                f"  mean |engine correlation|   {diagnostics.get('mean_abs_corr', math.nan):.4f}",
                f"  max  |engine correlation|   {diagnostics.get('max_abs_corr', math.nan):.4f}",
                f"  effective breadth (unit rel){ebe:9.2f}",
                f"  correlation condition no.   {diagnostics.get('condition_number', math.nan):.1f}",
                "",
                "  Effective breadth is the number of genuinely independent engine",
                "  opinions the pooled confidence rests on. Rising engine correlation",
                "  means falling breadth, which means the system is becoming",
                "  overconfident -- and the trailing estimator is slowest to see this",
                "  precisely during a crisis. Watch this number above all others.",
            ]
        )
        self.add_criterion(Criterion("effective breadth", ebe, self.cfg.ebe_min, ">="))
        return self.section("EVIDENCE BREADTH", body)

    def surrogate_section(self, pvalues: Mapping[str, float]) -> "ValidationReport":
        if not pvalues:
            return self
        rows = [f"  {k:34s} p = {v:.4f}" for k, v in sorted(pvalues.items())]
        rows += [
            "",
            "  The GARCH surrogate null is the most informative single test here.",
            "  Surrogates preserve volatility clustering and fat tails while",
            "  destroying directional structure, so a strategy that still profits on",
            "  them is harvesting volatility or exploiting a cost-model artifact --",
            "  not predicting direction.",
        ]
        for k, v in pvalues.items():
            self.add_criterion(Criterion(f"surrogate null: {k}", v, 0.05, "<="))
        return self.section("SURROGATE NULLS", "\n".join(rows))

    def sensitivity_section(self, scans: Mapping[str, float]) -> "ValidationReport":
        if not scans:
            return self
        rows = [f"  {k:34s} plateau ratio {v:.3f}" for k, v in sorted(scans.items())]
        rows += [
            "",
            "  A plateau ratio near 1 means the chosen value sits on a flat region.",
            "  A ratio well above 1 means it sits on a spike, which is what a tuned",
            "  parameter looks like. Interiority also matters: the chosen point must",
            "  not be the best point in its own scan.",
        ]
        worst = max(scans.values()) if scans else math.nan
        self.add_criterion(Criterion("worst plateau ratio", worst, 1.35, "<="))
        return self.section("PARAMETER SENSITIVITY", "\n".join(rows))

    def training_section(self, diag: Mapping[str, float]) -> "ValidationReport":
        if not diag:
            return self
        rows = [f"  {k:34s} {v:+.4f}" if v == v else f"  {k:34s} n/a"
                for k, v in sorted(diag.items())]
        return self.section("TRAINING DIAGNOSTICS", "\n".join(rows))

    # -- output --------------------------------------------------------- #
    def verdict(self) -> str:
        if not self.criteria:
            return "  No criteria evaluated."
        lines = [c.render() for c in self.criteria]
        n_fail = sum(1 for c in self.criteria if not c.passed)
        lines.append("")
        if n_fail == 0:
            lines.append("  OVERALL: PASS on every pre-registered criterion.")
        else:
            lines.append(
                f"  OVERALL: FAIL -- {n_fail} of {len(self.criteria)} criteria not met."
            )
            lines.append("")
            lines.append("  Per docs/00-PREREGISTRATION.md, a failure that leads to a redesign")
            lines.append("  increments the trial counter and the deflated Sharpe ratio must be")
            lines.append("  recharged. Redesigning without recording the trial invalidates every")
            lines.append("  subsequent acceptance test.")
        return "\n".join(lines)

    def render(self) -> str:
        return (
            "\n".join(self.sections)
            + f"\n\n{_THIN}\nPRE-REGISTERED ACCEPTANCE VERDICT\n{_THIN}\n"
            + self.verdict()
            + "\n"
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.render()


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return math.nan
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / np.where(peak > 0, peak, 1.0)))


def build_report(
    cfg: Config,
    trades: Sequence,
    training_diag: Mapping[str, float] | None = None,
    systems: Sequence | None = None,
    surrogate_pvalues: Mapping[str, float] | None = None,
    sensitivity: Mapping[str, float] | None = None,
) -> str:
    """Assemble the standard report from whatever results are available."""
    rep = ValidationReport(cfg).header()
    rep.trades_section(trades)
    rep.regime_section(trades)
    if systems:
        s0 = systems[0]
        rep.calibration_section(s0.calibrator.health())
        rep.breadth_section(s0.calop.diagnostics())
    if training_diag:
        rep.training_section(training_diag)
    if surrogate_pvalues:
        rep.surrogate_section(surrogate_pvalues)
    if sensitivity:
        rep.sensitivity_section(sensitivity)
    return rep.render()
