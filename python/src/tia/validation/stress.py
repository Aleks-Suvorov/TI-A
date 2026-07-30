"""Robustness stress tests: noise, timing, dropout, latency, ablation, plateaus.

Statistical tests (:mod:`tia.validation.montecarlo`) ask whether the result could
have arisen by chance. The tests here ask a different and equally decisive
question: **is the result fragile?** A genuine edge is a property of the market
and therefore degrades smoothly when the measurement is perturbed. An artifact is
a property of one particular alignment of one particular dataset with one
particular parameter vector, and it collapses -- or, far more tellingly, improves
-- under perturbations that should not matter.

The five diagnostics, and what a failure means:

*   :func:`noise_injection` -- perturb prices inside a plausible spread. A
    strategy whose returns move materially when prices move by half a tick is
    reading noise, or is being filled at prices that do not exist.
*   :func:`timing_shift` -- move the signal ±1..5 bars. **The single most
    informative cheap test in this file**; see its docstring for how to read the
    decay profile.
*   :func:`bar_dropout` -- delete bars at random. Tests dependence on an exact
    bar count or alignment (a "20-bar high" that is really "this specific bar").
*   :func:`latency_injection` -- add execution delay. The realism dial: SPEC §2.3
    fixes ``execution_lag_bars >= 1``, and this measures the cost of the next bar
    of slippage in the decision-to-fill path.
*   :func:`ablation_study` and :func:`parameter_plateau` -- attribution and
    tuning detection. If one engine carries everything, the ensemble is
    decoration; if performance is a spike rather than a plateau in a parameter,
    that parameter was tuned whatever its provenance tag claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from ..config import Config

__all__ = [
    "noise_injection",
    "timing_shift",
    "timing_decay_profile",
    "TimingDecay",
    "bar_dropout",
    "latency_injection",
    "AblationEntry",
    "AblationResult",
    "ablation_study",
    "ParameterScan",
    "parameter_plateau",
]


# ---------------------------------------------------------------------------
# Price and bar perturbations
# ---------------------------------------------------------------------------


def noise_injection(
    bars_array: np.ndarray, spread_frac: float, rng: np.random.Generator
) -> np.ndarray:
    """Perturb O/H/L/C inside a plausible spread, preserving OHLC consistency.

    ``bars_array`` is a 2-D array whose first four columns are open, high, low,
    close; any further columns (volume, trades, delta) are copied unchanged.
    Each of the four prices is multiplied by ``exp(u * spread_frac / 2)`` with
    ``u ~ U(-1, 1)`` independently, i.e. every price is moved by up to half a
    spread in either direction -- the correct order of magnitude for "which side
    of the book did this print come from".

    Consistency is then re-imposed rather than assumed::

        high = max(open, high, low, close)
        low  = min(open, high, low, close)

    so the result always satisfies ``low <= open, close <= high`` and can be fed
    to :class:`~tia.types.Bar` without tripping its validator. Note the
    asymmetry this introduces: independent perturbation followed by
    re-bracketing slightly *widens* the average range, which is conservative for
    any strategy that profits from wide ranges and slightly punitive for one that
    profits from narrow ones -- worth remembering when reading the result of a
    compression-based engine.

    Interpretation: run the full pipeline over many noise draws and compare the
    distribution of performance with the unperturbed value. If the unperturbed
    value sits in the upper tail of that distribution, the strategy is exploiting
    the exact prices of the historical record, which will not repeat.
    """
    a = np.asarray(bars_array, dtype=float)
    if a.ndim != 2 or a.shape[1] < 4:
        raise ValueError("bars_array must be 2-D with at least 4 columns (O, H, L, C)")
    if spread_frac < 0.0:
        raise ValueError("spread_frac must be non-negative")
    out = a.copy()
    if spread_frac == 0.0:
        return out
    u = rng.uniform(-1.0, 1.0, size=(a.shape[0], 4))
    px = a[:, :4] * np.exp(u * (spread_frac / 2.0))
    hi = px.max(axis=1)
    lo = px.min(axis=1)
    out[:, 0] = px[:, 0]
    out[:, 1] = hi
    out[:, 2] = lo
    out[:, 3] = px[:, 3]
    if np.any(out[:, :4] <= 0.0):
        raise ValueError("noise injection produced a non-positive price; spread_frac too large")
    return out


def timing_shift(
    signal_indices: Sequence[int] | np.ndarray, shift: int, n_bars: int | None = None
) -> np.ndarray:
    """Shift signal bars by ``shift`` (typically -5..+5), dropping invalid ones.

    **How to read the diagnostic.** Re-run the strategy with the signal moved by
    -5, -4, ..., +4, +5 bars and plot performance against the shift. Three
    profiles, three verdicts:

    1.  **Smooth, monotone decay away from zero, peaking at zero.** What a real
        edge looks like. The information that made bar ``t`` special is still
        partly present at ``t ± 1`` and fades with distance, because market state
        is persistent. The width of the peak is a genuine estimate of the edge's
        horizon.
    2.  **A discontinuous collapse -- performance falls off a cliff at ±1.** The
        result depends on the exact bar, which for bar data almost always means
        an alignment artifact: a look-ahead of exactly one bar, a label computed
        from the same bar that generated the signal, or an execution price taken
        from the decision bar.
    3.  **Performance *improves* at a non-zero shift.** The most damning outcome
        of all, and the reason to always scan both signs. It means the chosen
        timing was not derived from the mechanism but selected -- and if a
        *negative* shift (acting earlier) improves things, the "signal" is partly
        a function of information that postdates it.

    A positive ``shift`` delays the signal; a negative ``shift`` advances it.
    Advancing is *not* a legal strategy -- it is a look-ahead -- but it is a
    legitimate and necessary diagnostic, so it is permitted here and nowhere else.
    """
    idx = np.asarray(signal_indices, dtype=np.int64).ravel()
    shifted = idx + int(shift)
    ok = shifted >= 0
    if n_bars is not None:
        ok &= shifted < int(n_bars)
    return shifted[ok]


@dataclass(frozen=True)
class TimingDecay:
    """Performance as a function of signal timing shift.

    ``monotone`` is ``True`` when performance is non-increasing as the shift moves
    away from zero on both sides -- the profile a real edge produces.
    ``peak_at_zero`` is ``True`` when no shifted variant beats the intended bar.
    Both must hold to pass acceptance; either failing points at the profiles
    described in :func:`timing_shift`.
    """

    shifts: tuple[int, ...]
    scores: tuple[float, ...]
    baseline: float
    peak_at_zero: bool
    monotone: bool
    half_life_bars: float

    @property
    def verdict(self) -> str:
        if not self.peak_at_zero:
            return "improves-off-bar"
        return "smooth-decay" if self.monotone else "discontinuous"

    def as_dict(self) -> dict[str, float]:
        return {
            "timing_baseline": self.baseline,
            "timing_peak_at_zero": float(self.peak_at_zero),
            "timing_monotone": float(self.monotone),
            "timing_half_life_bars": self.half_life_bars,
        }


def timing_decay_profile(
    scores_by_shift: Mapping[int, float], tolerance: float = 1e-9
) -> TimingDecay:
    """Summarise a timing scan into the two acceptance flags.

    ``scores_by_shift`` must include shift 0. ``tolerance`` absorbs floating-point
    noise when checking monotonicity; it should not be used to excuse a real
    increase.
    """
    if 0 not in scores_by_shift:
        raise ValueError("the timing scan must include shift 0 (the intended bar)")
    shifts = sorted(scores_by_shift)
    scores = [float(scores_by_shift[s]) for s in shifts]
    base = float(scores_by_shift[0])
    peak = all(v <= base + tolerance for v in scores)
    mono = True
    # Walk outward from zero in both directions; scores must not increase.
    neg = [s for s in shifts if s < 0][::-1]
    pos = [s for s in shifts if s > 0]
    prev = base
    for s in neg:
        v = float(scores_by_shift[s])
        if v > prev + tolerance:
            mono = False
            break
        prev = v
    if mono:
        prev = base
        for s in pos:
            v = float(scores_by_shift[s])
            if v > prev + tolerance:
                mono = False
                break
            prev = v
    # Half-life: smallest |shift| at which performance has halved.
    half = math.inf
    if base > 0.0:
        for s in sorted(shifts, key=abs):
            if s == 0:
                continue
            if float(scores_by_shift[s]) <= 0.5 * base:
                half = float(abs(s))
                break
    return TimingDecay(
        shifts=tuple(shifts),
        scores=tuple(scores),
        baseline=base,
        peak_at_zero=peak,
        monotone=mono,
        half_life_bars=half,
    )


def bar_dropout(
    bars: np.ndarray, p: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly delete bars, returning ``(kept_bars, kept_mask)``.

    Simulates a feed that misses prints, and tests whether the strategy depends on
    an exact bar count. Any feature defined as "the highest high of the last 20
    bars" changes meaning when bars go missing; a feature defined over a
    volatility-scaled window degrades gracefully. The first bar is always kept so
    that downstream warmup logic has a starting point.

    Note that dropping bars *shortens* the series, so per-bar statistics stay
    comparable but any horizon expressed in bars (including
    ``Config.max_holding_bars``) now covers more wall-clock time. That is the
    intended stress, not a bug: it is what a data outage actually does.
    """
    a = np.asarray(bars)
    if not (0.0 <= p < 1.0):
        raise ValueError("p must lie in [0, 1)")
    n = a.shape[0]
    keep = rng.random(n) >= p
    if n:
        keep[0] = True
    return a[keep], keep


def latency_injection(decisions: Sequence[object], extra_bars: int) -> list[object]:
    """Delay execution by ``extra_bars`` additional bars.

    Accepts either a sequence of :class:`~tia.types.Decision`-like objects
    carrying ``execute_at_index`` (returned with that field advanced, via
    ``dataclasses.replace`` so frozen dataclasses are respected) or a sequence of
    plain integer execution indices.

    ``decided_at_index`` is never touched: the decision was still made when it
    was made, and the whole point is to widen the gap between decision and fill.
    A strategy whose edge is intact at ``execution_lag_bars = 1`` but gone at 2 is
    living inside one bar of information and is not implementable by a human
    reading a card.
    """
    if extra_bars < 0:
        raise ValueError("extra_bars must be non-negative")
    out: list[object] = []
    import dataclasses

    for d in decisions:
        if hasattr(d, "execute_at_index"):
            new_idx = int(getattr(d, "execute_at_index")) + extra_bars
            try:
                out.append(dataclasses.replace(d, execute_at_index=new_idx))  # type: ignore[type-var]
            except TypeError:  # pragma: no cover - non-dataclass fallback
                out.append(new_idx)
        else:
            out.append(int(d) + extra_bars)  # type: ignore[arg-type]
    return out


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationEntry:
    """One engine's ablation record.

    ``marginal`` is ``full - leave_one_out``: what the ensemble loses by removing
    this engine. ``solo`` is the engine's performance on its own.

    The interesting cases are the mismatches. High ``solo`` with near-zero
    ``marginal`` means the engine is redundant -- its information is already in
    the others, and counting it again is exactly the failure that
    ``Config.ebe_min`` and the correlation-aware pooling of SPEC §6 exist to
    prevent. Low ``solo`` with high ``marginal`` is the *good* case: an engine
    that is useless alone but decorrelates the ensemble. **Negative** ``marginal``
    means the engine actively harms the ensemble and should be switched off, not
    down-weighted.
    """

    name: str
    leave_one_out: float
    solo: float
    marginal: float

    @property
    def marginal_share(self) -> float:
        return math.nan


@dataclass(frozen=True)
class AblationResult:
    """Leave-one-out and one-only ablation over a set of engines."""

    full: float
    entries: tuple[AblationEntry, ...]
    baseline: float = 0.0

    @property
    def total_marginal(self) -> float:
        return float(sum(max(e.marginal, 0.0) for e in self.entries))

    def marginal_share(self, name: str) -> float:
        """Share of total positive marginal contribution attributable to ``name``.

        Concentration here is the number to watch: a single engine holding, say,
        70% of the total marginal contribution means the other engines are
        window dressing and the system's claimed evidence breadth is fictional.
        """
        tot = self.total_marginal
        if tot <= 0.0:
            return math.nan
        for e in self.entries:
            if e.name == name:
                return float(max(e.marginal, 0.0) / tot)
        raise KeyError(name)

    @property
    def max_marginal_share(self) -> float:
        tot = self.total_marginal
        if tot <= 0.0 or not self.entries:
            return math.nan
        return float(max(max(e.marginal, 0.0) for e in self.entries) / tot)

    @property
    def min_marginal(self) -> float:
        if not self.entries:
            return math.nan
        return float(min(e.marginal for e in self.entries))

    def as_dict(self) -> dict[str, float]:
        return {
            "ablation_full": self.full,
            "ablation_max_marginal_share": self.max_marginal_share,
            "ablation_min_marginal": self.min_marginal,
        }

    def __str__(self) -> str:
        w = max([len("engine")] + [len(e.name) for e in self.entries]) if self.entries else 6
        lines = [
            f"ablation: full ensemble = {self.full:.4f}",
            f"{'engine':<{w}}  {'leave-1-out':>12}  {'solo':>10}  {'marginal':>10}  {'share':>7}",
            "-" * (w + 46),
        ]
        for e in self.entries:
            share = self.marginal_share(e.name)
            lines.append(
                f"{e.name:<{w}}  {e.leave_one_out:>12.4f}  {e.solo:>10.4f}  "
                f"{e.marginal:>10.4f}  {share:>7.3f}"
            )
        return "\n".join(lines)


def ablation_study(
    engine_names: Sequence[str], run_fn: Callable[[frozenset[str]], float]
) -> AblationResult:
    """Leave-one-engine-out and one-engine-only performance.

    ``run_fn`` receives the set of *active* engine names and returns a scalar
    objective -- use the same objective throughout an acceptance run, and prefer
    one that is already risk-adjusted and net of costs (a per-observation Sharpe
    or an expectancy in sigma units), because raw return rewards whichever subset
    happens to trade most.

    Cost: ``2 * len(engine_names) + 1`` runs.
    """
    names = list(dict.fromkeys(engine_names))
    if not names:
        raise ValueError("no engines to ablate")
    full = float(run_fn(frozenset(names)))
    entries: list[AblationEntry] = []
    for nm in names:
        loo = float(run_fn(frozenset(n for n in names if n != nm)))
        solo = float(run_fn(frozenset((nm,))))
        entries.append(
            AblationEntry(name=nm, leave_one_out=loo, solo=solo, marginal=full - loo)
        )
    return AblationResult(full=full, entries=tuple(entries))


# ---------------------------------------------------------------------------
# Parameter plateaus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterScan:
    """Performance of one parameter over its declared sensitivity neighbourhood.

    ``plateau_ratio``
        ``score(chosen) / median(scores over the neighbourhood)``. Near 1 means
        the chosen point is unremarkable within its own neighbourhood, which is
        what a structural constant should look like. Far above 1 means the chosen
        point is a **spike**: performance depends on this parameter's exact value,
        which is the signature of tuning regardless of what the parameter's
        provenance tag claims.
    ``left_support`` / ``right_support``
        ``max(score) / score(chosen)`` on the low and high side. Small values mean
        the plateau has a cliff on that side.
    ``interior``
        ``False`` when the chosen point is the best in the scan *and* sits at an
        endpoint of it -- i.e. the optimum lies outside the neighbourhood, so the
        scan cannot demonstrate a plateau at all and the neighbourhood should be
        widened before the result is trusted.
    """

    name: str
    chosen_value: float
    multipliers: tuple[float, ...]
    values: tuple[float, ...]
    scores: tuple[float, ...]
    chosen_score: float
    median_score: float
    plateau_ratio: float
    left_support: float
    right_support: float
    fraction_positive: float
    interior: bool
    verdict: str
    max_plateau_ratio: float = 1.5
    min_support: float = 0.6

    @property
    def passed(self) -> bool:
        return self.verdict == "plateau"

    def as_dict(self) -> dict[str, float]:
        return {
            "plateau_ratio": self.plateau_ratio,
            "left_support": self.left_support,
            "right_support": self.right_support,
            "fraction_positive": self.fraction_positive,
            "interior": float(self.interior),
            "passed": float(self.passed),
        }


def _scan_grid(lo: float, hi: float, n_points: int) -> np.ndarray:
    """Geometric grid over ``[lo, hi]`` that always contains exactly 1.0.

    Geometric rather than linear because the neighbourhoods in
    :func:`~tia.config.P` are *multiplicative*: (0.5, 2.0) means "half to double",
    and a linear grid over that range would put two thirds of its points above
    the chosen value.
    """
    if not (lo > 0.0 and hi > lo):
        raise ValueError("sensitivity neighbourhood must satisfy 0 < lo < hi")
    if n_points < 3:
        raise ValueError("n_points must be at least 3")
    g = np.geomspace(lo, hi, n_points)
    g = np.concatenate([g, [1.0]])
    return np.unique(np.round(g, 12))


def parameter_plateau(
    cfg: Config,
    param_names: Sequence[str],
    run_fn: Callable[[Config], float],
    n_points: int = 7,
    max_plateau_ratio: float = 1.5,
    min_support: float = 0.6,
) -> dict[str, ParameterScan]:
    """Scan each parameter over its ``Config.sensitivity_for`` neighbourhood.

    For every name in ``param_names`` the value is multiplied by each point of a
    geometric grid spanning that parameter's declared neighbourhood (integers are
    rounded and floored at 1), the pipeline is re-run through ``run_fn``, and the
    resulting curve is classified:

    ``"plateau"``
        The chosen point is inside a region of comparable performance:
        ``plateau_ratio <= max_plateau_ratio`` and both sides retain at least
        ``min_support`` of the chosen score. **Pass.**
    ``"spike"``
        The chosen point stands well above the neighbourhood median, or one side
        falls away sharply. The parameter was tuned, or the performance surface is
        too rough for the parameter to be identified from this sample. **Fail.**
    ``"edge"``
        The chosen point is the scan's best *and* lies at an endpoint, so
        interiority cannot be demonstrated. **Fail**, and widen the
        neighbourhood before concluding anything.
    ``"degenerate"``
        The chosen score is not positive, or fewer than three points evaluated.
        Nothing can be concluded; treated as a failure so that a broken run
        cannot pass by accident.

    Requiring interiority is what distinguishes this from a sensitivity table: a
    parameter sitting at the best end of its own scan is telling you the search
    has not finished, and the honest response is either to widen the scan or --
    much better -- to re-derive the parameter from theory so that it is not being
    searched at all. ``Config``'s provenance tags exist to make the second option
    the normal one.

    Cost: roughly ``len(param_names) * n_points`` full pipeline runs.
    """
    out: dict[str, ParameterScan] = {}
    for name in param_names:
        base = getattr(cfg, name)
        if isinstance(base, bool) or not isinstance(base, (int, float)):
            raise TypeError(f"{name} is not a numeric parameter and cannot be scanned")
        lo, hi = cfg.sensitivity_for(name)
        mults = _scan_grid(float(lo), float(hi), int(n_points))
        values: list[float] = []
        scores: list[float] = []
        kept_mults: list[float] = []
        seen: set[float] = set()
        for mlt in mults:
            if isinstance(base, int):
                v: float = float(max(1, int(round(base * mlt))))
                cast_v: int | float = int(v)
            else:
                v = float(base) * float(mlt)
                cast_v = v
            if v in seen:
                continue
            seen.add(v)
            kept_mults.append(float(mlt))
            values.append(v)
            scores.append(float(run_fn(cfg.replace(**{name: cast_v}))))
        out[name] = _classify_scan(
            name=name,
            chosen_value=float(base),
            multipliers=kept_mults,
            values=values,
            scores=scores,
            max_plateau_ratio=max_plateau_ratio,
            min_support=min_support,
        )
    return out


def _classify_scan(
    name: str,
    chosen_value: float,
    multipliers: Sequence[float],
    values: Sequence[float],
    scores: Sequence[float],
    max_plateau_ratio: float,
    min_support: float,
) -> ParameterScan:
    m = np.asarray(multipliers, dtype=float)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(m)
    m, s = m[order], s[order]
    v = np.asarray(values, dtype=float)[order]
    # The chosen point is the multiplier closest to 1.0 (exactly 1.0 by grid
    # construction, unless integer rounding collapsed it).
    ci = int(np.argmin(np.abs(m - 1.0)))
    chosen = float(s[ci])
    med = float(np.median(s)) if s.size else math.nan
    frac_pos = float(np.mean(s > 0.0)) if s.size else math.nan

    def support(sl: slice) -> float:
        part = s[sl]
        if part.size == 0 or chosen <= 0.0:
            return math.nan
        return float(np.max(part) / chosen)

    left = support(slice(0, ci))
    right = support(slice(ci + 1, s.size))
    is_best = bool(np.all(s <= chosen + 1e-12))
    at_edge = ci == 0 or ci == s.size - 1
    interior = not (is_best and at_edge)

    if s.size < 3 or not np.isfinite(chosen) or chosen <= 0.0:
        ratio = math.inf if (np.isfinite(chosen) and chosen > 0.0 and med <= 0.0) else math.nan
        verdict = "degenerate"
    else:
        ratio = math.inf if med <= 0.0 else float(chosen / med)
        if not interior:
            verdict = "edge"
        elif ratio > max_plateau_ratio:
            verdict = "spike"
        elif (np.isfinite(left) and left < min_support) or (
            np.isfinite(right) and right < min_support
        ):
            # A side that falls away means the plateau has a cliff on it.
            verdict = "spike"
        else:
            verdict = "plateau"
    return ParameterScan(
        name=name,
        chosen_value=chosen_value,
        multipliers=tuple(float(x) for x in m),
        values=tuple(float(x) for x in v),
        scores=tuple(float(x) for x in s),
        chosen_score=chosen,
        median_score=med,
        plateau_ratio=ratio,
        left_support=left,
        right_support=right,
        fraction_positive=frac_pos,
        interior=interior,
        verdict=verdict,
        max_plateau_ratio=max_plateau_ratio,
        min_support=min_support,
    )
