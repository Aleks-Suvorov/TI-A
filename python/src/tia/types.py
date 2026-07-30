"""Core type contract for TI-A (Trading Indicator - Adaptive).

Everything in this package is built on the types defined here. Two rules are
enforced structurally rather than by convention:

1.  **Causality.** Engines are *streaming*: they receive one closed bar at a
    time through :meth:`Engine.update` and hold their own state. An engine
    cannot look ahead because it is never handed the future. Vectorised code
    paths exist only as replays of the same stream.

2.  **Execution realism.** A :class:`Decision` produced from the close of bar
    ``t`` carries ``decided_at_index == t`` and is executed no earlier than
    ``t + execution_lag_bars`` (default 1, i.e. next bar's open). Nothing in
    the library may read a price at or after the execution index when forming
    the decision.

Units and conventions
---------------------
*   Returns are natural logarithms unless a name ends in ``_pct``.
*   ``sigma`` always means a per-bar standard deviation of log returns on the
    execution timeframe, never annualised, unless the name ends ``_ann``.
*   Anything named ``*_rank`` lives in ``[0, 1]`` and is a causal rolling
    empirical quantile. Anything named ``*_z`` is a robust standard score.
*   Engine scores live in ``[-1, +1]``: sign is direction (positive = long),
    magnitude is strength. Exactly ``0.0`` means "no opinion".
*   Reliabilities and probabilities live in ``[0, 1]``.
*   Prices, stops and targets are in instrument price units. Risk distances
    are additionally reported in ``sigma`` units as ``*_sigma``.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

__all__ = [
    "Action",
    "Position",
    "Regime",
    "SessionPhase",
    "Bar",
    "ExogenousSnapshot",
    "FeatureSnapshot",
    "BarContext",
    "EngineOutput",
    "FusionResult",
    "TargetSpec",
    "CostEstimate",
    "RiskDecision",
    "Decision",
    "ExplainCard",
    "Engine",
    "REGIME_NAMES",
    "clip",
    "safe_div",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Action(enum.Enum):
    """The only five things the trader is ever shown.

    The semantics are those of a single-position discretionary book:

    ``BUY``      open a long from flat
    ``SELL``     close an existing long (never opens a short)
    ``SHORT``    open a short from flat
    ``COVER``    close an existing short (never opens a long)
    ``NO_TRADE`` do nothing; this is the default and by far the most common
    """

    NO_TRADE = "NO TRADE"
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"

    @property
    def is_entry(self) -> bool:
        return self in (Action.BUY, Action.SHORT)

    @property
    def is_exit(self) -> bool:
        return self in (Action.SELL, Action.COVER)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.value


class Position(enum.IntEnum):
    """Book state. Only one position may be open at a time."""

    SHORT = -1
    FLAT = 0
    LONG = 1


class Regime(enum.IntEnum):
    """The four regimes the system distinguishes.

    These are deliberately few and deliberately chosen from theory rather than
    discovered by clustering, because the two directional effects with the
    strongest out-of-sample evidence -- time-series momentum and short-horizon
    liquidity-provision reversal -- point in *opposite* directions and are
    separated primarily by return autocorrelation and volatility state.
    """

    TREND = 0  #: persistent, directional; momentum is paid
    REVERT = 1  #: range-bound, absorptive; liquidity provision is paid
    STRESS = 2  #: high volatility, jumpy, correlations converge; tails fat
    QUIET = 3  #: low volatility, low participation, no structure; stand aside

    @property
    def label(self) -> str:
        return REGIME_NAMES[int(self)]


REGIME_NAMES: tuple[str, ...] = ("Trending", "Mean-Reverting", "Stress", "Quiet")


class SessionPhase(enum.IntEnum):
    """Intraday phase. ``ALL_DAY`` is used for 24h instruments and daily bars."""

    ALL_DAY = 0
    PRE_OPEN = 1
    OPENING_AUCTION = 2
    MORNING = 3
    MIDDAY = 4  #: the documented lunchtime liquidity trough
    AFTERNOON = 5
    CLOSING_AUCTION = 6
    POST_CLOSE = 7
    OVERNIGHT = 8

    @property
    def is_auction(self) -> bool:
        return self in (SessionPhase.OPENING_AUCTION, SessionPhase.CLOSING_AUCTION)

    @property
    def is_tradeable(self) -> bool:
        return self not in (
            SessionPhase.PRE_OPEN,
            SessionPhase.POST_CLOSE,
            SessionPhase.OVERNIGHT,
        )


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bar:
    """One *closed* bar.

    ``timestamp`` is the epoch seconds of the bar's **close**. Using close time
    rather than open time removes an entire family of off-by-one look-ahead
    bugs: a bar stamped ``t`` is fully known at ``t`` and at no earlier moment.
    """

    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    #: Number of trades, if the feed provides it. ``None`` degrades gracefully.
    trades: float | None = None
    #: Signed volume imbalance (buy - sell) if a footprint feed is available.
    #: ``None`` everywhere else; no engine may *require* it.
    delta: float | None = None
    #: True when the bar is known to be incomplete. Such bars are dropped.
    partial: bool = False

    def __post_init__(self) -> None:
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError(
                f"OHLC inconsistent at t={self.timestamp}: "
                f"O={self.open} H={self.high} L={self.low} C={self.close}"
            )
        if self.low <= 0.0:
            raise ValueError(f"non-positive price at t={self.timestamp}")
        if self.volume < 0.0:
            raise ValueError(f"negative volume at t={self.timestamp}")

    @property
    def typical(self) -> float:
        """The (H+L+C)/3 proxy for the bar's volume-weighted average price."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def log_range(self) -> float:
        return math.log(self.high / self.low)

    @property
    def clv(self) -> float:
        """Close location value in ``[-1, +1]``.

        +1 means the bar closed on its high, -1 on its low. This is the
        cheapest available proxy for who won the bar's auction.
        """
        span = self.high - self.low
        if span <= 0.0:
            return 0.0
        return (2.0 * self.close - self.high - self.low) / span

    @property
    def dollar_volume(self) -> float:
        return self.typical * self.volume


@dataclass(frozen=True, slots=True)
class ExogenousSnapshot:
    """Optional, strictly contracted outside information for bar ``t``.

    Every field defaults to ``None``. Engines that consume these must report
    ``reliability == 0.0`` when the data is absent, so that the fusion layer
    ignores them rather than silently substituting a neutral value. No field
    here may ever be required for the core edge -- see ``docs/09-ASSUMPTIONS.md``.
    """

    #: Reference-asset log returns keyed by symbol, aligned to this bar's close.
    peers: Mapping[str, float] = field(default_factory=dict)
    #: Implied-volatility index level (e.g. VIX) as of this bar's close.
    implied_vol: float | None = None
    #: Days until the next scheduled high-impact event for this instrument.
    #: 0 means the event lands inside the current bar.
    days_to_event: float | None = None
    #: Coarse dealer-gamma imbalance in ``[-1, +1]``; see the positioning
    #: engine's data contract. Default OFF because public estimates are poor.
    gamma_imbalance: float | None = None
    #: Open interest change, normalised. Futures/options only.
    oi_change: float | None = None


# ---------------------------------------------------------------------------
# Feature kernel output
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeatureSnapshot:
    """Everything the feature kernel knows at the close of one bar.

    This is a flat, named record rather than a dict so that typos are caught at
    import time and so that the Pine Script port has an unambiguous list of
    quantities to reproduce. ``valid`` is ``False`` until warmup completes; no
    engine may emit a non-zero score while ``valid`` is ``False``.
    """

    index: int
    timestamp: float
    close: float

    # -- volatility -------------------------------------------------------
    sigma_bp: float = math.nan  #: bipower (jump-robust) per-bar vol
    sigma_rv: float = math.nan  #: plain realised vol over the same window
    sigma_rs: float = math.nan  #: Rogers-Satchell range vol, drift-free
    sigma_ew: float = math.nan  #: EWMA vol, fast-reacting
    sigma_fcst: float = math.nan  #: HAR-style forecast for the *next* bar
    jump_share: float = math.nan  #: (RV - BPV)/RV clipped to [0,1]
    vol_of_vol: float = math.nan  #: dispersion of log sigma
    vol_rank: float = math.nan
    vol_ratio: float = math.nan  #: sigma_ew / sigma_fcst; >1 means expanding
    compression: float = math.nan  #: how coiled the range is, in [0,1]

    # -- displacement and flow -------------------------------------------
    ret: float = math.nan  #: log close-to-close return of this bar
    ret_norm: float = math.nan  #: ret / sigma_bp
    participation: float = math.nan  #: dollar volume / robust local median
    ladr: float = math.nan  #: liquidity-adjusted displacement ratio (signed)
    ladr_rank: float = math.nan  #: rank of |ladr|, sign carried separately
    absorption: float = math.nan  #: participation per unit displacement
    absorption_rank: float = math.nan
    amihud: float = math.nan  #: |ret| / dollar volume, illiquidity proxy
    kyle_lambda: float = math.nan  #: regression-free Kyle price-impact proxy
    clv_ema: float = math.nan  #: smoothed close location value
    delta_norm: float = math.nan  #: normalised signed volume, NaN if no feed

    # -- trend ------------------------------------------------------------
    kalman_level: float = math.nan
    kalman_slope: float = math.nan
    kalman_slope_t: float = math.nan  #: slope / sqrt(posterior variance)
    vr_z: float = math.nan  #: Lo-MacKinlay variance-ratio test statistic
    hurst_implied: float = math.nan  #: 0.5 + log(VR)/(2 log q); biased, display only
    efficiency: float = math.nan  #: |net move| / path length, in [0,1]
    trend_agree: float = math.nan  #: multi-horizon sign agreement in [-1,1]

    # -- structure --------------------------------------------------------
    pivot_high: float = math.nan  #: last *confirmed* swing high price
    pivot_low: float = math.nan
    pivot_high_age: float = math.nan  #: bars since confirmation
    pivot_low_age: float = math.nan
    range_pos: float = math.nan  #: position inside the confirmed range, [0,1]
    htf_slope_t: float = math.nan  #: higher-timeframe trend t-statistic
    htf_range_pos: float = math.nan
    swept_high: float = 0.0  #: 1.0 when this bar swept and failed a high
    swept_low: float = 0.0

    # -- information ------------------------------------------------------
    perm_entropy: float = math.nan  #: permutation entropy in [0,1]
    ret_kurtosis: float = math.nan
    run_asym: float = math.nan  #: up-run vs down-run length asymmetry

    # -- session / behaviour ----------------------------------------------
    phase: SessionPhase = SessionPhase.ALL_DAY
    tod_frac: float = math.nan  #: fraction through the session, [0,1]
    dow: int = -1  #: 0 = Monday
    overnight_ret: float = math.nan
    bars_since_session_open: int = -1

    # -- bookkeeping -------------------------------------------------------
    valid: bool = False
    warmup_remaining: int = 0

    def as_dict(self) -> dict[str, float]:
        """Flatten to a name -> float mapping for logging and drift monitors."""
        out: dict[str, float] = {}
        for slot in self.__slots__:  # type: ignore[attr-defined]
            v = getattr(self, slot)
            if isinstance(v, bool):
                out[slot] = float(v)
            elif isinstance(v, enum.IntEnum):
                out[slot] = float(int(v))
            elif isinstance(v, (int, float)):
                out[slot] = float(v)
        return out


@dataclass(slots=True)
class BarContext:
    """What an engine sees on one call to :meth:`Engine.update`.

    Holding the bar and the kernel snapshot together, and nothing else, is what
    makes the causality guarantee checkable: there is no handle here through
    which an engine could reach a future bar.
    """

    bar: Bar
    features: FeatureSnapshot
    exog: ExogenousSnapshot = field(default_factory=ExogenousSnapshot)
    #: Set by the pipeline so engines can adapt without reading the clock.
    position: Position = Position.FLAT
    bars_in_position: int = 0


# ---------------------------------------------------------------------------
# Engine output and fusion
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EngineOutput:
    """One engine's opinion about the next ``horizon`` bars.

    ``score`` is a *direction and strength*, not a probability. Converting to
    log-odds is the fusion layer's job, because only the fusion layer knows the
    calibration and the between-engine correlation structure.

    ``reliability`` answers a different question from ``score``: not "which way"
    but "should you listen to me at all right now". An engine with plenty of
    data and a regime it was designed for reports high reliability; the same
    engine mid-warmup, or looking at a regime it has no business judging,
    reports low reliability and is thereby down-weighted rather than muted.
    """

    name: str
    score: float
    reliability: float
    #: Human-facing quantities for the explainability card.
    features: Mapping[str, float] = field(default_factory=dict)
    #: Machine-facing internals for monitoring; never shown to the trader.
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    #: Short strings describing what fired, e.g. "sweep of 12-bar low, failed".
    notes: Sequence[str] = field(default_factory=tuple)
    valid: bool = True

    def __post_init__(self) -> None:
        if not (-1.0 - 1e-9 <= self.score <= 1.0 + 1e-9):
            raise ValueError(f"{self.name}: score {self.score} outside [-1, 1]")
        if not (0.0 - 1e-9 <= self.reliability <= 1.0 + 1e-9):
            raise ValueError(f"{self.name}: reliability {self.reliability} outside [0, 1]")

    @property
    def effective(self) -> float:
        """Score scaled by reliability; convenient for display only."""
        return self.score * self.reliability

    @staticmethod
    def abstain(name: str, reason: str = "warmup") -> "EngineOutput":
        return EngineOutput(
            name=name, score=0.0, reliability=0.0, notes=(reason,), valid=False
        )


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Output of correlation-aware pooling plus calibration."""

    #: Pooled latent log-odds of the long side being right.
    log_odds: float
    #: Standard error of ``log_odds`` implied by the engine covariance.
    log_odds_se: float
    #: Calibrated probability that the *proposed direction* reaches its target
    #: barrier before its stop. Always expressed for the proposed side.
    p_success: float
    #: Credible interval on ``p_success`` from ``log_odds_se``.
    p_low: float
    p_high: float
    #: Effective breadth of evidence: how many genuinely independent engine
    #: votes the pooled number actually rests on. See ``fusion/calop.py``.
    ebe: float
    #: Direction implied by the pooled log-odds: +1 long, -1 short, 0 none.
    direction: int
    #: Per-engine pooling weights, keyed by engine name.
    weights: Mapping[str, float] = field(default_factory=dict)
    #: Per-engine log-odds contributions after weighting.
    contributions: Mapping[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Targets, costs, risk
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """A triple barrier, sized from the volatility forecast.

    Barriers are set in ``sigma`` units first and only then converted to price,
    which is what makes a single parameter set portable across instruments and
    across decades.
    """

    direction: int  #: +1 long, -1 short
    entry_ref: float  #: reference price the barriers are measured from
    stop_sigma: float
    target_sigma: float
    max_holding_bars: int
    sigma: float  #: the per-bar sigma used for the conversion

    @property
    def stop_price(self) -> float:
        return self.entry_ref * math.exp(-self.direction * self.stop_sigma * self.sigma)

    @property
    def target_price(self) -> float:
        return self.entry_ref * math.exp(self.direction * self.target_sigma * self.sigma)

    @property
    def reward_risk(self) -> float:
        return safe_div(self.target_sigma, self.stop_sigma, 0.0)


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Round-trip frictions, in ``sigma`` units of the execution timeframe."""

    half_spread_sigma: float
    impact_sigma: float
    slippage_sigma: float
    fee_sigma: float

    @property
    def round_trip_sigma(self) -> float:
        return 2.0 * (
            self.half_spread_sigma + self.impact_sigma + self.slippage_sigma + self.fee_sigma
        )


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Sizing and the reasons it was throttled."""

    #: Fraction of equity to risk on this trade, already net of every throttle.
    risk_fraction: float
    #: Units of the instrument implied by ``risk_fraction`` and the stop.
    units: float
    #: Multiplicative throttles applied, for the audit trail.
    throttles: Mapping[str, float] = field(default_factory=dict)
    #: Hard blocks that forced ``risk_fraction`` to zero.
    blocks: Sequence[str] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# The user-facing decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExplainCard:
    """The only diagnostic surface the trader ever reads.

    Deliberately a small, fixed set of plain-language rows. A card that grows
    with the model defeats its own purpose, so new engines map into the
    existing rows rather than adding new ones.
    """

    headline: str
    confidence_pct: float
    rows: Sequence[tuple[str, str]]
    #: Why the system stood aside, when it did. Empty for entries.
    veto_reasons: Sequence[str] = field(default_factory=tuple)

    def render(self, width: int = 46) -> str:  # pragma: no cover - display only
        lines = [self.headline, "-" * width]
        for label, value in self.rows:
            pad = max(1, width - len(label) - len(value) - 1)
            lines.append(f"{label}:{' ' * pad}{value}")
        if self.veto_reasons:
            lines.append("-" * width)
            lines.extend(f"  stood aside: {r}" for r in self.veto_reasons)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Decision:
    """Everything the system emits for one bar.

    ``decided_at_index`` is the index of the closed bar that produced this
    decision. ``execute_at_index`` is strictly greater. Backtest and live
    execution both honour the same two fields, which is the whole point: there
    is exactly one notion of when an order may be sent.
    """

    action: Action
    decided_at_index: int
    execute_at_index: int
    timestamp: float
    position_before: Position
    position_after: Position
    #: Populated for entries; ``None`` for exits and ``NO_TRADE``.
    target: TargetSpec | None = None
    fusion: FusionResult | None = None
    risk: RiskDecision | None = None
    costs: CostEstimate | None = None
    #: Expectancy in sigma units, net of ``costs``, at the posterior mean.
    expected_value_sigma: float = 0.0
    #: Lower credible bound of the same quantity. The gate acts on *this*.
    expected_value_lcb: float = 0.0
    regime: Regime = Regime.QUIET
    regime_posterior: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    engine_outputs: Mapping[str, EngineOutput] = field(default_factory=dict)
    card: ExplainCard | None = None

    @property
    def is_actionable(self) -> bool:
        return self.action is not Action.NO_TRADE


# ---------------------------------------------------------------------------
# The engine protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Engine(Protocol):
    """A streaming, stateful estimator of directional edge.

    Implementations must satisfy three properties, all of which are tested:

    *   **Determinism.** Feeding the same bar sequence twice yields identical
        outputs. No wall clock, no unseeded randomness.
    *   **Prefix consistency.** The output at bar ``t`` when fed ``bars[:t+1]``
        equals the output at bar ``t`` when fed the whole history. This is what
        "does not repaint" means operationally, and
        ``tests/test_causality.py`` checks it by brute force.
    *   **Graceful degradation.** Missing optional inputs lower ``reliability``
        toward zero; they never raise and never fabricate a neutral reading.
    """

    name: str
    #: Bars required before ``update`` may return ``valid=True``.
    warmup: int

    def reset(self) -> None:
        """Return to the pre-warmup state, discarding all history."""
        ...

    def update(self, ctx: BarContext) -> EngineOutput:
        """Consume one closed bar and emit this engine's current opinion."""
        ...


# ---------------------------------------------------------------------------
# Tiny numeric helpers used across the package
# ---------------------------------------------------------------------------


def clip(x: float, lo: float, hi: float) -> float:
    """Clamp, propagating NaN as the midpoint so callers never see NaN scores."""
    if x != x:  # NaN
        return 0.5 * (lo + hi)
    return lo if x < lo else (hi if x > hi else x)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` for zero, NaN or infinite denominators."""
    if b == 0.0 or b != b or a != a or math.isinf(b):
        return default
    out = a / b
    return default if (out != out or math.isinf(out)) else out
