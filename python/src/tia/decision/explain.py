"""The explainability card: nine fixed rows, and nothing else.

``SPEC.md`` §8. The card is deliberately a small, closed set of plain-language
rows. A diagnostic surface that grows with the model defeats its own purpose: a
system requiring the user to weigh eight sub-indicators has not reduced their
decision problem, it has renamed it. New engines map into the existing rows.

The confidence row carries its credible interval, and that is not decoration. It
is the visible output of the variance term in ``docs/01-THEORY.md`` §6.3, and a
wide interval is the system saying that its engines agree for what may be a
single reason.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from ..engines.edgebook import EdgeEstimate, SetupFamily
from ..types import (
    Action,
    CostEstimate,
    EngineOutput,
    ExplainCard,
    FeatureSnapshot,
    FusionResult,
    Regime,
    RiskDecision,
    TargetSpec,
)

__all__ = ["build_card"]


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100.0 * x:.0f}%"


def _trend_row(outputs: Mapping[str, EngineOutput]) -> str:
    o = outputs.get("trend")
    if o is None or not o.valid or o.reliability <= 0.02:
        return "n/a"
    # Alignment is the score's magnitude expressed as agreement, weighted by how
    # much the engine should be trusted in this regime. A strong slope in the
    # wrong regime therefore reads low, which is the honest presentation.
    return f"{_pct(abs(o.score) * o.reliability)} {'long' if o.score > 0 else 'short'}"


def _liquidity_row(outputs: Mapping[str, EngineOutput]) -> str:
    o = outputs.get("liquidity")
    if o is None or not o.valid:
        return "n/a"
    age = float(o.features.get("sweep_age", 999.0))
    if age <= 4:
        return "Confirmed sweep"
    ar = float(o.features.get("absorption_rank", float("nan")))
    if ar == ar and ar > 0.65:
        return "Absorbing"
    return "None"


def _volatility_row(f: FeatureSnapshot) -> str:
    if f.vol_rank != f.vol_rank:
        return "n/a"
    if f.compression == f.compression and f.compression > 0.75:
        return "Compressed"
    if f.vol_rank > 0.85:
        return "Elevated"
    if f.vol_rank < 0.15:
        return "Very low"
    return "Favourable"


def _flow_row(outputs: Mapping[str, EngineOutput], f: FeatureSnapshot) -> str:
    """Institutional flow, stated as what was measured rather than who did it.

    "Accumulation" here means volume was transacted without price falling --
    an observable. It is not a claim about the identity or intent of any
    participant, which is unobservable from this data and which the system does
    not pretend to know.
    """
    liq = outputs.get("liquidity")
    if liq is None or not liq.valid:
        return "Neutral"
    ar = float(liq.features.get("absorption_rank", float("nan")))
    if ar != ar or ar < 0.65:
        mo = outputs.get("momentum")
        if mo is not None and mo.valid and abs(mo.score) > 0.4:
            return "Directional flow" if mo.reliability > 0.4 else "Neutral"
        return "Neutral"
    if f.clv_ema == f.clv_ema and f.clv_ema > 0.15:
        return "Accumulation"
    if f.clv_ema == f.clv_ema and f.clv_ema < -0.15:
        return "Distribution"
    return "Absorption, side unclear"


def build_card(
    action: Action,
    f: FeatureSnapshot,
    fusion: FusionResult,
    regime: Regime,
    outputs: Mapping[str, EngineOutput],
    edge: EdgeEstimate | None = None,
    costs: CostEstimate | None = None,
    spec: TargetSpec | None = None,
    risk: RiskDecision | None = None,
    setup: SetupFamily | None = None,
    veto_reasons: Sequence[str] = (),
) -> ExplainCard:
    if action is Action.NO_TRADE:
        headline = "NO TRADE"
    elif action.is_exit:
        headline = f"{action.value}  (close position)"
    else:
        side = "long" if action is Action.BUY else "short"
        headline = f"{action.value}  ({side})"

    conf = fusion.p_success
    conf_txt = (
        f"{_pct(conf)}  [{_pct(fusion.p_low)}-{_pct(fusion.p_high)}]"
        if conf == conf
        else "n/a"
    )

    rr = "n/a"
    if spec is not None and spec.stop_sigma > 0.0:
        rr = f"{spec.reward_risk:.2f} : 1"

    rows: list[tuple[str, str]] = [
        ("Trend Alignment", _trend_row(outputs)),
        ("Liquidity", _liquidity_row(outputs)),
        ("Momentum Expansion", _pct(abs(outputs["momentum"].score)) if "momentum" in outputs and outputs["momentum"].valid else "n/a"),
        ("Volatility", _volatility_row(f)),
        ("Regime", regime.label),
        ("Institutional Flow", _flow_row(outputs, f)),
        ("Evidence Breadth", f"{fusion.ebe:.1f} independent" if fusion.ebe == fusion.ebe else "n/a"),
        ("Confidence", conf_txt),
        ("Expected R:R", rr),
    ]

    if action.is_entry and edge is not None and costs is not None:
        rows.append(
            (
                "Expectancy (net)",
                f"{edge.lcb - costs.round_trip_sigma:+.3f}s at 90% credence",
            )
        )
        if setup is not None:
            rows.append(("Setup", f"{setup.label} (n={edge.n_eff:.0f})"))
        if spec is not None:
            rows.append(("Stop / Target", f"{spec.stop_price:.4g} / {spec.target_price:.4g}"))
        if risk is not None:
            rows.append(("Risk", f"{100.0 * risk.risk_fraction:.2f}% of equity"))

    return ExplainCard(
        headline=headline,
        confidence_pct=100.0 * conf if conf == conf else math.nan,
        rows=tuple(rows),
        veto_reasons=tuple(veto_reasons),
    )
