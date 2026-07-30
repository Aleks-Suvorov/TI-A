"""Correlation-aware pooling and probability calibration."""

from .calibration import (
    BrierDecomposition, IsotonicCalibrator, OnlineCalibrator, brier_decomposition,
    brier_score, expected_calibration_error, fit_platt, pav, reliability_table,
)
from .calop import CALOP, effective_breadth, evidence_from_score, logistic

__all__ = [
    "CALOP", "effective_breadth", "evidence_from_score", "logistic",
    "IsotonicCalibrator", "OnlineCalibrator", "pav", "fit_platt",
    "brier_score", "brier_decomposition", "BrierDecomposition",
    "reliability_table", "expected_calibration_error",
]
