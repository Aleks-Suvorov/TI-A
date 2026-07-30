"""Validation: purged cross-validation, walk-forward, surrogate nulls, stress."""

from .cpcv import CombinatorialPurgedCV, LeakageReport, PurgedKFold
from .montecarlo import (
    GarchFit, bootstrap_ci, fit_garch11, garch_surrogate, iid_bootstrap, mc_pvalue,
    sign_scramble_surrogate, stationary_block_bootstrap, surrogate_pvalue,
    trade_order_permutation,
)
from .stress import (
    AblationResult, ParameterScan, TimingDecay, ablation_study, bar_dropout,
    latency_injection, noise_injection, parameter_plateau, timing_decay_profile,
    timing_shift,
)
from .walkforward import (
    ParameterDrift, WalkForwardResult, anchored_walkforward, parameter_drift,
    rolling_walkforward,
)

__all__ = [
    "PurgedKFold", "CombinatorialPurgedCV", "LeakageReport",
    "anchored_walkforward", "rolling_walkforward", "WalkForwardResult",
    "ParameterDrift", "parameter_drift",
    "stationary_block_bootstrap", "iid_bootstrap", "trade_order_permutation",
    "fit_garch11", "GarchFit", "garch_surrogate", "sign_scramble_surrogate",
    "mc_pvalue", "surrogate_pvalue", "bootstrap_ci",
    "noise_injection", "timing_shift", "timing_decay_profile", "TimingDecay",
    "bar_dropout", "latency_injection", "ablation_study", "AblationResult",
    "parameter_plateau", "ParameterScan",
]
