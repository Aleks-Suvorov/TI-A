"""TI-A -- an adaptive, abstention-first trading signal framework.

The design is documented in ``docs/01-THEORY.md``. The short version:

*   Directional predictability at these horizons has an information coefficient
    of roughly 0.04, so most bars contain nothing actionable and abstention is
    the primary output rather than a defensive afterthought.
*   Volatility is far more forecastable than direction, so the risk denominator
    is estimated first and everything else is denominated in it.
*   The two directional effects with real out-of-sample evidence -- momentum
    continuation and liquidity-provision reversal -- issue opposite instructions,
    so a regime posterior sits at the top of the architecture rather than being
    bolted on as a filter.
*   Correlated engines counted as independent evidence is the dominant failure
    mode of multi-signal designs, so pooling is correlation-aware and reports the
    effective breadth of evidence it actually rests on.
*   The trade gate acts on the *lower credible bound* of expectancy net of
    modelled costs, which produces extreme selectivity as a consequence of the
    decision rule rather than as a hand-set threshold.

Quick start::

    from tia import TIA, Config
    from tia.synthetic import generate_null

    system = TIA(Config())
    for decision in system.stream(generate_null(3000)):
        if decision.is_actionable:
            print(decision.card.render())
    print(system.summary())
"""

from .config import DEFAULT_CONFIG, Config, Provenance
from .pipeline import TIA, TradeRecord
from .types import (
    Action, Bar, BarContext, CostEstimate, Decision, Engine, EngineOutput,
    ExogenousSnapshot, ExplainCard, FeatureSnapshot, FusionResult, Position,
    Regime, RiskDecision, SessionPhase, TargetSpec,
)

__version__ = "1.0.0"

__all__ = [
    "TIA", "TradeRecord", "Config", "DEFAULT_CONFIG", "Provenance",
    "Action", "Position", "Regime", "SessionPhase", "Bar", "BarContext",
    "ExogenousSnapshot", "FeatureSnapshot", "EngineOutput", "FusionResult",
    "TargetSpec", "CostEstimate", "RiskDecision", "Decision", "ExplainCard",
    "Engine", "__version__",
]
