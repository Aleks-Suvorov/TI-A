"""Frozen parameter manifest.

The single most effective defence against overfitting in a system this large is
not a clever validation scheme -- it is keeping the number of parameters that
were ever *chosen by looking at outcomes* small, and making that number
auditable. So every parameter here carries a provenance tag:

``Provenance.THEORY``
    Set a priori from published results or from a dimensional argument, never
    tuned against backtest performance. Changing one of these is a research
    decision that must be pre-registered.

``Provenance.DEV``
    Estimated once on the development universe (see ``docs/00-PREREGISTRATION.md``)
    and then frozen. These are the only parameters fit to market outcomes, and
    :meth:`Config.fitted_dof` counts them so the deflated Sharpe ratio can be
    charged correctly.

``Provenance.ONLINE``
    Updated during live operation by a causal recursion only (Bayesian counts,
    calibration maps). These add no in-sample optimism because at every point in
    time they use strictly past data, but they do add estimation variance, which
    :meth:`Config.online_dof` reports.

``Provenance.OPS``
    Deployment knobs -- account size, risk appetite, symbol lists. They change
    the scale of the result, never its statistical validity.

``fitted_dof`` is what goes into the deflated Sharpe ratio trial count, together
with the trials ledger. Anyone who edits a ``DEV`` default without a new
pre-registration entry has invalidated the acceptance test, and the hash in
:meth:`Config.manifest_hash` is what makes that detectable after the fact.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any

__all__ = ["Provenance", "Config", "P", "DEFAULT_CONFIG"]


class Provenance(enum.Enum):
    THEORY = "theory"
    DEV = "dev"
    ONLINE = "online"
    OPS = "ops"


def P(
    default: Any,
    provenance: Provenance,
    doc: str,
    *,
    sensitivity: tuple[float, float] | None = None,
) -> Any:
    """Declare a parameter with provenance and a sensitivity-scan neighbourhood.

    ``sensitivity`` is the multiplicative range over which
    ``validation/stress.py`` will re-run the whole pipeline. A parameter whose
    performance is a sharp peak rather than a plateau over that range fails
    acceptance regardless of its point performance.
    """
    return field(
        default=default,
        metadata={
            "provenance": provenance,
            "doc": doc,
            "sensitivity": sensitivity or (0.7, 1.4),
        },
    )


@dataclass(frozen=True)
class Config:
    """Complete configuration. Immutable; use :meth:`replace` to derive variants."""

    # -- identity ---------------------------------------------------------
    version: str = P("1.0.0", Provenance.OPS, "manifest version string")

    # ------------------------------------------------------------------ #
    # Feature kernel                                                     #
    # ------------------------------------------------------------------ #
    vol_window: int = P(
        22,
        Provenance.THEORY,
        "bars in the bipower/realised variance window; 22 is the conventional "
        "monthly horizon of the HAR literature and is not tuned",
    )
    vol_window_short: int = P(5, Provenance.THEORY, "weekly HAR component")
    vol_window_long: int = P(66, Provenance.THEORY, "quarterly HAR component")
    ewma_halflife: float = P(
        10.0, Provenance.THEORY, "half-life in bars of the fast EWMA volatility"
    )
    har_weights: tuple[float, float, float] = P(
        (0.5, 0.3, 0.2),
        Provenance.THEORY,
        "geometric blend weights on daily/weekly/monthly log volatility. A "
        "parameter-light stand-in for a fitted HAR-RV; the fitted variant lives "
        "in features/scales.py and is optional",
    )
    rank_window: int = P(
        252,
        Provenance.THEORY,
        "lookback for causal rank transforms; one trading year of bars. Long "
        "enough for a stable empirical CDF, short enough to track slow "
        "structural change in market microstructure",
    )
    rank_min_obs: int = P(60, Provenance.THEORY, "minimum observations before a rank is emitted")

    # -- microstructure ---------------------------------------------------
    impact_exponent: float = P(
        0.5,
        Provenance.THEORY,
        "exponent of the price-impact law used to normalise displacement by "
        "participation. 0.5 is the square-root law; the scan covers [0.3, 0.7]",
        sensitivity=(0.6, 1.4),
    )
    participation_window: int = P(
        60, Provenance.THEORY, "bars used for the robust median of dollar volume"
    )
    tod_buckets: int = P(
        13,
        Provenance.THEORY,
        "time-of-day buckets for de-seasonalising volume and volatility. The "
        "intraday U-shape is one of the most robust facts in the literature and "
        "must be removed before any volume feature is comparable across hours",
    )

    # -- structure --------------------------------------------------------
    pivot_atr_mult: float = P(
        1.5,
        Provenance.THEORY,
        "swing must exceed this many sigma to count as a pivot; scale-free by "
        "construction so one value serves every instrument",
    )
    pivot_confirm_bars: int = P(
        3,
        Provenance.THEORY,
        "bars of non-violation before a pivot is *confirmed*. Until confirmed, "
        "a pivot is invisible to every consumer. This delay is precisely the "
        "cost of not repainting",
    )
    htf_multiple: int = P(
        5,
        Provenance.THEORY,
        "higher timeframe as a multiple of the execution timeframe. Derived, "
        "not user-set: the point is that the trader configures nothing",
    )
    htf_multiple_2: int = P(25, Provenance.THEORY, "second, structural timeframe multiple")

    # -- trend ------------------------------------------------------------
    kalman_snr: float = P(
        0.02,
        Provenance.DEV,
        "signal-to-noise ratio of the local-linear-trend filter: process "
        "variance as a fraction of observation variance. The single knob that "
        "sets how fast the trend estimate turns",
        sensitivity=(0.4, 2.5),
    )
    kalman_slope_snr: float = P(
        0.002, Provenance.DEV, "slope-component process variance fraction", sensitivity=(0.4, 2.5)
    )
    vr_lags: tuple[int, ...] = P(
        (2, 4, 8, 16),
        Provenance.THEORY,
        "aggregation horizons for the Lo-MacKinlay variance-ratio test; dyadic "
        "by convention",
    )
    vr_window: int = P(120, Provenance.THEORY, "sample size for the variance-ratio statistic")

    # -- entropy ----------------------------------------------------------
    perm_entropy_order: int = P(3, Provenance.THEORY, "embedding dimension for permutation entropy")
    perm_entropy_window: int = P(60, Provenance.THEORY, "window for permutation entropy")

    # ------------------------------------------------------------------ #
    # Regime engine                                                      #
    # ------------------------------------------------------------------ #
    regime_stickiness: float = P(
        0.985,
        Provenance.DEV,
        "diagonal of the regime transition matrix. Implies a mean dwell time of "
        "1/(1-p) bars. Estimated once from the documented persistence of "
        "volatility regimes on the development universe, then frozen",
        sensitivity=(0.99, 1.005),
    )
    regime_beta_concentration: float = P(
        6.0,
        Provenance.THEORY,
        "concentration of the Beta measurement densities. Higher means each "
        "regime makes sharper claims about its rank features",
        sensitivity=(0.5, 2.0),
    )
    regime_shift_hazard_halflife: float = P(
        20.0, Provenance.THEORY, "half-life of the reference posterior used to detect transitions"
    )
    regime_hazard_max: float = P(
        0.35,
        Provenance.THEORY,
        "maximum tolerated regime-transition hazard before entries are vetoed. "
        "Transitions are where strategies die, so the system stands aside "
        "through them by design",
        sensitivity=(0.6, 1.6),
    )

    # ------------------------------------------------------------------ #
    # Fusion and calibration                                             #
    # ------------------------------------------------------------------ #
    fusion_temperature: float = P(
        1.0,
        Provenance.DEV,
        "temperature applied to pooled log-odds before the sigmoid. Fit by "
        "minimising negative log-likelihood on the development universe",
        sensitivity=(0.5, 2.0),
    )
    fusion_bias: float = P(0.0, Provenance.DEV, "intercept of the calibration link")
    fusion_shrink: float = P(
        0.15,
        Provenance.THEORY,
        "ridge added to the engine correlation matrix diagonal before "
        "inversion. Without it, near-collinear engines produce explosive "
        "weights -- the classic failure of naive confluence scoring",
        sensitivity=(0.4, 3.0),
    )
    ebe_min: float = P(
        2.5,
        Provenance.THEORY,
        "minimum effective breadth of evidence. Below this the apparent "
        "agreement of many engines is really one piece of information counted "
        "several times, and the trade is refused however high its probability",
        sensitivity=(0.7, 1.5),
    )
    correlation_window: int = P(
        500, Provenance.ONLINE, "bars over which engine score correlations are re-estimated"
    )
    calibration_bins: int = P(15, Provenance.THEORY, "bins for the reliability diagram")
    calibration_min_samples: int = P(
        200, Provenance.THEORY, "samples before the isotonic map replaces the parametric link"
    )

    # ------------------------------------------------------------------ #
    # Edge book (hierarchical conditional expectancy)                    #
    # ------------------------------------------------------------------ #
    edge_evidence_buckets: int = P(
        5, Provenance.THEORY, "quantile buckets of pre-fusion evidence per regime/setup cell"
    )
    edge_pooling_strength: float = P(
        25.0,
        Provenance.THEORY,
        "pseudo-observations of the parent node injected into each child. The "
        "shrinkage that makes a thin cell say 'no edge' instead of 'huge edge'",
        sensitivity=(0.4, 3.0),
    )
    edge_prior_mean_sigma: float = P(
        0.0,
        Provenance.THEORY,
        "prior mean outcome in sigma units. Zero: the null hypothesis is that "
        "there is no edge anywhere, and evidence must move us off it",
    )
    edge_prior_strength: float = P(
        40.0, Provenance.THEORY, "pseudo-observations behind the global no-edge prior"
    )
    edge_halflife_bars: float = P(
        5040.0,
        Provenance.ONLINE,
        "exponential forgetting half-life for outcome counts, roughly twenty "
        "years of daily bars or one year of hourly. Long, because the cost of "
        "forgetting a real regime you have not seen recently exceeds the cost "
        "of stale data",
        sensitivity=(0.5, 2.0),
    )
    edge_lcb_quantile: float = P(
        0.10,
        Provenance.THEORY,
        "credible-bound quantile the expectancy gate acts on. The system trades "
        "on the pessimistic end of what it believes, not the mean",
        sensitivity=(0.5, 2.0),
    )

    # ------------------------------------------------------------------ #
    # Targets and costs                                                   #
    # ------------------------------------------------------------------ #
    stop_sigma: float = P(
        1.6,
        Provenance.THEORY,
        "stop distance in forecast sigma. Wide enough that ordinary noise does "
        "not touch it, tight enough that one loss is survivable",
        sensitivity=(0.6, 1.6),
    )
    target_sigma: float = P(
        2.6, Provenance.THEORY, "profit target in forecast sigma", sensitivity=(0.6, 1.7)
    )
    max_holding_bars: int = P(
        30,
        Provenance.THEORY,
        "vertical barrier. Beyond this the conditioning information that "
        "justified the trade has decayed and the position is no longer the one "
        "that was analysed",
        sensitivity=(0.5, 2.0),
    )
    min_holding_bars: int = P(
        2, Provenance.THEORY, "dwell time before a reversal signal may close a position"
    )
    cost_impact_eta: float = P(
        0.6,
        Provenance.DEV,
        "coefficient of the square-root impact law, calibrated to observed "
        "slippage on the development universe",
        sensitivity=(0.5, 2.5),
    )
    cost_slippage_range_frac: float = P(
        0.10,
        Provenance.THEORY,
        "fraction of the execution bar's range charged as adverse slippage. A "
        "deliberately pessimistic fill assumption",
        sensitivity=(0.5, 3.0),
    )
    cost_fee_bps: float = P(1.0, Provenance.OPS, "per-side commission in basis points")
    spread_estimator_window: int = P(
        22, Provenance.THEORY, "window for the Corwin-Schultz effective spread estimator"
    )

    # ------------------------------------------------------------------ #
    # Decision gate                                                       #
    # ------------------------------------------------------------------ #
    p_min: float = P(
        0.58,
        Provenance.THEORY,
        "minimum calibrated probability of reaching target before stop. With a "
        "reward/risk above one this is already a demanding bar; it is a floor, "
        "not the primary gate",
        sensitivity=(0.9, 1.15),
    )
    p_exit: float = P(
        0.55, Provenance.THEORY, "opposite-side probability required to close early"
    )
    ev_lcb_min_sigma: float = P(
        0.05,
        Provenance.THEORY,
        "the primary gate: the lower credible bound of net expectancy, in "
        "sigma units, that a trade must clear. Everything else is a filter on "
        "top of this",
        sensitivity=(0.5, 3.0),
    )
    max_trades_per_100_bars: float = P(
        4.0,
        Provenance.THEORY,
        "a rate limiter, not an optimiser. If the gate somehow starts firing "
        "constantly, something has broken and the correct response is to trade "
        "less, not more",
        sensitivity=(0.5, 3.0),
    )
    execution_lag_bars: int = P(
        1,
        Provenance.THEORY,
        "bars between decision and execution. One means 'next bar open'. This "
        "is never zero",
    )
    allow_shorts: bool = P(True, Provenance.OPS, "whether short entries are permitted")

    # ------------------------------------------------------------------ #
    # Risk                                                                #
    # ------------------------------------------------------------------ #
    risk_per_trade: float = P(
        0.005, Provenance.OPS, "baseline fraction of equity risked to the stop"
    )
    kelly_fraction: float = P(
        0.25,
        Provenance.THEORY,
        "fraction of the full Kelly stake. Quarter-Kelly is the conventional "
        "compromise between growth and the fact that our edge estimate is "
        "itself uncertain",
        sensitivity=(0.4, 2.0),
    )
    max_risk_per_trade: float = P(0.02, Provenance.OPS, "hard cap on per-trade risk fraction")
    vol_target_ann: float = P(0.15, Provenance.OPS, "annualised volatility target of the book")
    dd_throttle_start: float = P(
        0.04, Provenance.THEORY, "drawdown at which sizing begins to shrink"
    )
    dd_throttle_stop: float = P(
        0.15, Provenance.THEORY, "drawdown at which sizing reaches its floor"
    )
    dd_throttle_floor: float = P(0.25, Provenance.THEORY, "minimum size multiplier under drawdown")
    dd_kill: float = P(
        0.25, Provenance.OPS, "drawdown at which the system stops trading and requires review"
    )

    # ------------------------------------------------------------------ #
    # Monitoring                                                          #
    # ------------------------------------------------------------------ #
    monitor_window: int = P(100, Provenance.ONLINE, "trades in the rolling health window")
    brier_alarm: float = P(
        0.27,
        Provenance.THEORY,
        "rolling Brier score above which the probability model is declared "
        "unreliable and the system demotes itself to NO TRADE. 0.25 is the "
        "score of a coin flip",
        sensitivity=(0.9, 1.15),
    )
    psi_alarm: float = P(
        0.25, Provenance.THEORY, "population-stability-index alarm on the feature distribution"
    )
    slippage_alarm_mult: float = P(
        2.0, Provenance.THEORY, "realised/modelled slippage ratio that trips the execution alarm"
    )

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #
    cv_folds: int = P(8, Provenance.THEORY, "groups for combinatorial purged cross-validation")
    cv_test_groups: int = P(2, Provenance.THEORY, "groups held out per CPCV split")
    cv_embargo_frac: float = P(
        0.01,
        Provenance.THEORY,
        "embargo as a fraction of sample length, applied after each test block "
        "to break the serial dependence that overlapping labels create",
    )
    mc_paths: int = P(2000, Provenance.THEORY, "Monte Carlo / bootstrap replications")
    bootstrap_block_bars: int = P(
        60,
        Provenance.THEORY,
        "block length for the stationary block bootstrap; long enough to carry "
        "volatility clustering through the resample",
    )
    trials_ledger_count: int = P(
        1,
        Provenance.OPS,
        "number of distinct configurations ever evaluated against validation "
        "data. Charged to the deflated Sharpe ratio. Increment it honestly",
    )

    # ------------------------------------------------------------------ #
    # Optional modules, off by default                                    #
    # ------------------------------------------------------------------ #
    enable_positioning_engine: bool = P(
        False,
        Provenance.OPS,
        "dealer-gamma / open-interest engine. Off by default because publicly "
        "available estimates of aggregate dealer positioning are of poor and "
        "unverifiable quality; the mechanism is real, the retail data is not",
    )
    enable_crossasset_engine: bool = P(
        True, Provenance.OPS, "peer and implied-volatility confirmation engine"
    )
    enable_delta_features: bool = P(
        False, Provenance.OPS, "use signed volume when a footprint feed is present"
    )
    use_volume_clock: bool = P(
        False,
        Provenance.THEORY,
        "resample to equal-dollar-volume bars before analysis. Improves the "
        "normality and IID-ness of returns; costs interpretability and needs a "
        "tick or fine-grained feed",
    )

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def replace(self, **kw: Any) -> "Config":
        return dataclasses.replace(self, **kw)

    def by_provenance(self, provenance: Provenance) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.metadata.get("provenance") is provenance
        }

    def fitted_dof(self) -> int:
        """Count parameters ever fit against market outcomes.

        This number, not the total parameter count, is what the deflated Sharpe
        ratio must be charged for. Keeping it in single digits is the design
        goal that most of the architecture exists to serve.
        """
        return len(self.by_provenance(Provenance.DEV))

    def online_dof(self) -> int:
        return len(self.by_provenance(Provenance.ONLINE))

    def doc_for(self, name: str) -> str:
        for f in fields(self):
            if f.name == name:
                return str(f.metadata.get("doc", ""))
        raise KeyError(name)

    def sensitivity_for(self, name: str) -> tuple[float, float]:
        for f in fields(self):
            if f.name == name:
                return tuple(f.metadata.get("sensitivity", (0.7, 1.4)))  # type: ignore[return-value]
        raise KeyError(name)

    def scannable(self) -> list[str]:
        """Numeric parameters worth including in a sensitivity scan."""
        out = []
        for f in fields(self):
            if f.metadata.get("provenance") in (Provenance.THEORY, Provenance.DEV):
                v = getattr(self, f.name)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(f.name)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def manifest_hash(self) -> str:
        """Stable hash of every parameter value.

        Recorded with every backtest and every live session. If a reported
        result and a running system disagree on this hash, the report does not
        describe the system.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def manifest_report(self) -> str:
        """Human-readable provenance audit, printed at the top of every report."""
        lines = [
            f"TI-A configuration {self.version}  manifest={self.manifest_hash()}",
            f"  parameters fitted to market outcomes : {self.fitted_dof()}",
            f"  parameters updated online (causal)   : {self.online_dof()}",
            "",
        ]
        for prov in Provenance:
            items = self.by_provenance(prov)
            lines.append(f"[{prov.value.upper()}] {len(items)} parameters")
            for k, v in sorted(items.items()):
                lines.append(f"    {k} = {v!r}")
            lines.append("")
        return "\n".join(lines)


DEFAULT_CONFIG = Config()
