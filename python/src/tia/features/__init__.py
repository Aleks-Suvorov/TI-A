"""Feature kernel: robust scales, microstructure, structure, trend, information."""

from .entropy import PermutationEntropy, RunAsymmetry
from .jumps import JumpTest
from .kernel import FeatureKernel, HTFView
from .microstructure import CorwinSchultzSpread, MicrostructureKernel
from .rolling import (
    BucketedMedian, CausalRank, EWMA, EWVar, RingBuffer, RobustZ,
    RollingCorrelation, RollingExtreme, RollingMoments, RollingQuantile, RollingSum,
)
from .scales import VolatilityKernel, fit_har_weights
from .structure import HTFAggregator, Pivot, PivotTracker, SweepDetector, SweepEvent
from .trend import KalmanTrend, MultiHorizonAgreement, PathEfficiency, VarianceRatio

__all__ = [
    "FeatureKernel", "HTFView", "VolatilityKernel", "fit_har_weights",
    "MicrostructureKernel", "CorwinSchultzSpread", "JumpTest",
    "KalmanTrend", "VarianceRatio", "PathEfficiency", "MultiHorizonAgreement",
    "PivotTracker", "Pivot", "SweepDetector", "SweepEvent", "HTFAggregator",
    "PermutationEntropy", "RunAsymmetry",
    "RingBuffer", "RollingMoments", "RollingSum", "EWMA", "EWVar",
    "RollingQuantile", "CausalRank", "RollingExtreme", "RobustZ",
    "BucketedMedian", "RollingCorrelation",
]
