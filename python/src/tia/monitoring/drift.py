"""Live health monitoring and the automatic demotion ladder.

A system that stands aside for weeks at a time cannot be judged by its P&L on any
timescale a human finds satisfying. So the things actually monitored here are
*model* properties, which have far more statistical power per unit time than
returns do: is the probability model still calibrated, do the live features still
look like the ones the model was developed on, are the engines still as
independent as the fusion layer assumes, and is execution costing what the cost
model says.

Each has a pre-committed threshold from :class:`tia.config.Config` and a defined
automatic response. The ladder matters more than any individual rung: degradation
is gradual, and a monitor whose only action is "halt" will be overridden long
before it fires.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Mapping, Sequence

import numpy as np

from ..config import Config
from ..fusion.calibration import brier_decomposition, brier_score, expected_calibration_error

__all__ = ["population_stability_index", "HealthReport", "DriftMonitor", "DemotionLevel"]


def population_stability_index(
    reference: np.ndarray, live: np.ndarray, bins: int = 10
) -> float:
    """PSI between a reference and a live sample of one feature.

    Conventional reading: below 0.10 no meaningful shift, 0.10-0.25 moderate,
    above 0.25 the live population is not the development population. Bin edges
    come from the *reference* quantiles, which is what makes the statistic
    directional -- it asks how the live data sits in the old world's coordinates,
    not how the two look in some shared frame.
    """
    ref = np.asarray(reference, dtype=np.float64)
    liv = np.asarray(live, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    liv = liv[np.isfinite(liv)]
    if ref.size < 50 or liv.size < 20:
        return math.nan
    edges = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0
    r, _ = np.histogram(ref, bins=edges)
    l, _ = np.histogram(liv, bins=edges)
    rp = np.clip(r / max(r.sum(), 1), 1e-6, None)
    lp = np.clip(l / max(l.sum(), 1), 1e-6, None)
    return float(np.sum((lp - rp) * np.log(lp / rp)))


class DemotionLevel:
    """The ladder. Higher is more restricted."""

    NORMAL = 0
    TIGHTENED = 1  #: gate raised, size cut
    RESTRICTED = 2  #: highest-confidence regime only
    NO_TRADE = 3  #: signals suppressed, monitoring continues
    HALTED = 4  #: requires human review to resume

    NAMES = {0: "normal", 1: "tightened", 2: "restricted", 3: "no trade", 4: "halted"}


@dataclass(slots=True)
class HealthReport:
    level: int = DemotionLevel.NORMAL
    brier: float = math.nan
    ece: float = math.nan
    skill: float = math.nan
    psi_max: float = math.nan
    psi_worst_feature: str = ""
    mean_engine_corr: float = math.nan
    ebe_full: float = math.nan
    slippage_ratio: float = math.nan
    n_outcomes: int = 0
    findings: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    @property
    def level_name(self) -> str:
        return DemotionLevel.NAMES.get(self.level, "unknown")

    def render(self) -> str:
        lines = [f"TI-A health: {self.level_name.upper()}"]
        if self.n_outcomes:
            lines.append(
                f"  calibration   Brier {self.brier:.4f}  ECE {self.ece:.4f}  "
                f"skill {self.skill:+.3f}  n={self.n_outcomes}"
            )
        if self.psi_max == self.psi_max:
            lines.append(f"  feature drift PSI max {self.psi_max:.3f} ({self.psi_worst_feature})")
        if self.mean_engine_corr == self.mean_engine_corr:
            lines.append(
                f"  engines       mean|r| {self.mean_engine_corr:.3f}  EBE {self.ebe_full:.2f}"
            )
        if self.slippage_ratio == self.slippage_ratio:
            lines.append(f"  execution     realised/modelled slippage {self.slippage_ratio:.2f}x")
        for f in self.findings:
            lines.append(f"  ! {f}")
        for a in self.actions:
            lines.append(f"  -> {a}")
        return "\n".join(lines)


class DriftMonitor:
    """Rolling model-health monitor with an explicit demotion ladder."""

    def __init__(self, cfg: Config | None = None, reference_window: int = 2000) -> None:
        self.cfg = cfg or Config()
        self.reference_window = int(reference_window)
        self._ref: dict[str, list[float]] = {}
        self._live: dict[str, Deque[float]] = {}
        self._p: Deque[float] = deque(maxlen=2000)
        self._y: Deque[float] = deque(maxlen=2000)
        self._slip: Deque[float] = deque(maxlen=100)
        self._frozen = False

    def reset(self) -> None:
        self._ref.clear()
        self._live.clear()
        self._p.clear()
        self._y.clear()
        self._slip.clear()
        self._frozen = False

    # ------------------------------------------------------------------ #
    def observe_features(self, feats: Mapping[str, float]) -> None:
        """Accumulate the reference distribution, then track the live one.

        The reference is frozen once ``reference_window`` observations exist.
        Letting it keep updating would make drift undetectable by construction:
        the comparison would always be against a distribution that had already
        absorbed the drift.
        """
        for k, v in feats.items():
            if v != v or not math.isfinite(v):
                continue
            if not self._frozen:
                self._ref.setdefault(k, []).append(float(v))
            self._live.setdefault(k, deque(maxlen=500)).append(float(v))
        if not self._frozen and self._ref:
            longest = max(len(v) for v in self._ref.values())
            if longest >= self.reference_window:
                self._frozen = True

    def observe_outcome(self, p: float, success: bool | float) -> None:
        if p == p:
            self._p.append(float(p))
            self._y.append(float(success))

    def observe_slippage(self, realised: float, modelled: float) -> None:
        if modelled > 0.0 and realised == realised:
            self._slip.append(realised / modelled)

    # ------------------------------------------------------------------ #
    def report(self, engine_diagnostics: Mapping[str, float] | None = None) -> HealthReport:
        cfg = self.cfg
        r = HealthReport()
        findings: list[str] = []
        actions: list[str] = []
        level = DemotionLevel.NORMAL

        n = len(self._p)
        r.n_outcomes = n
        if n >= 50:
            p = np.array(self._p)
            y = np.array(self._y)
            r.brier = brier_score(p, y)
            r.ece = expected_calibration_error(p, y)
            d = brier_decomposition(p, y)
            r.skill = d.skill
            if r.brier > cfg.brier_alarm:
                findings.append(
                    f"Brier {r.brier:.3f} above alarm {cfg.brier_alarm:.3f}; "
                    "0.25 is a coin flip"
                )
                level = max(level, DemotionLevel.NO_TRADE)
            elif r.brier > 0.9 * cfg.brier_alarm:
                findings.append(f"Brier {r.brier:.3f} approaching alarm")
                level = max(level, DemotionLevel.TIGHTENED)
            if r.skill == r.skill and r.skill < 0.0:
                if n >= cfg.calibration_min_samples:
                    findings.append(
                        f"Brier skill {r.skill:+.3f} on n={n}: forecasts are worse "
                        "than the base rate"
                    )
                    level = max(level, DemotionLevel.NO_TRADE)
                else:
                    # Below the isotonic map's own sample floor, a negative
                    # skill estimate is sampling noise, not evidence. The final
                    # review board found a demonstrably healthy run demoted to
                    # NO_TRADE at n=62 because gated selection lifts realized
                    # success above the population forecast -- miscalibration in
                    # the SAFE direction (under-confidence). Demoting on that
                    # teaches operators to ignore the ladder, which is how
                    # ladders die. It stays a visible finding; the absolute
                    # Brier alarm still catches genuine breakage regardless of
                    # sample size.
                    findings.append(
                        f"Brier skill {r.skill:+.3f} on n={n} (below "
                        f"{cfg.calibration_min_samples}): noted, not actionable "
                        "at this sample size"
                    )

        if self._frozen:
            worst, worst_k = -math.inf, ""
            for k, ref in self._ref.items():
                live = self._live.get(k)
                if not live or len(live) < 100:
                    continue
                v = population_stability_index(np.array(ref), np.array(live))
                if v == v and v > worst:
                    worst, worst_k = v, k
            if worst > -math.inf:
                r.psi_max, r.psi_worst_feature = worst, worst_k
                if worst > cfg.psi_alarm:
                    findings.append(
                        f"feature '{worst_k}' PSI {worst:.2f} above {cfg.psi_alarm:.2f}: "
                        "live inputs no longer resemble the development distribution"
                    )
                    level = max(level, DemotionLevel.RESTRICTED)

        if engine_diagnostics:
            r.mean_engine_corr = float(engine_diagnostics.get("mean_abs_corr", math.nan))
            r.ebe_full = float(engine_diagnostics.get("ebe_at_full_reliability", math.nan))
            if r.ebe_full == r.ebe_full and r.ebe_full < cfg.ebe_min:
                findings.append(
                    f"effective breadth at full reliability is {r.ebe_full:.2f}, below the "
                    f"gate's {cfg.ebe_min:.2f}: the engines have converged and are no longer "
                    "independent sources"
                )
                level = max(level, DemotionLevel.RESTRICTED)

        if len(self._slip) >= 10:
            med = float(np.median(np.array(self._slip)))
            r.slippage_ratio = med
            if med > cfg.slippage_alarm_mult:
                findings.append(
                    f"realised slippage {med:.1f}x modelled: the cost model is wrong, so "
                    "every expectancy in the gate is overstated"
                )
                level = max(level, DemotionLevel.NO_TRADE)

        actions.append(
            {
                DemotionLevel.NORMAL: "continue",
                DemotionLevel.TIGHTENED: "raise ev_lcb_min_sigma by 50% and halve size",
                DemotionLevel.RESTRICTED: "trade only the dominant regime at full confidence",
                DemotionLevel.NO_TRADE: "suppress signals; keep computing and logging",
                DemotionLevel.HALTED: "stop and require human review before resuming",
            }[level]
        )
        r.level = level
        r.findings = findings
        r.actions = actions
        return r
