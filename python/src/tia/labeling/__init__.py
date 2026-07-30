"""Label construction for supervised evaluation of the decision layer.

Two concerns, deliberately separated:

*   :mod:`tia.labeling.triple_barrier` turns a proposed entry into an *outcome*
    -- which barrier was reached first, at what price, after how many bars --
    under the pessimistic fill rules of SPEC §2.7.
*   :mod:`tia.labeling.weights` turns a set of overlapping outcomes into sample
    weights and an honest effective sample size, because overlapping labels are
    not independent observations and every naive t-statistic computed from them
    is inflated.

Neither module imports anything from ``features/``, ``engines/``, ``fusion/``,
``decision/`` or ``risk/``: labelling is offline evaluation machinery and must
stay usable on nothing but numpy arrays.
"""

from __future__ import annotations

from .triple_barrier import (
    OUTCOME_STOP,
    OUTCOME_TARGET,
    OUTCOME_TIMEOUT,
    BarrierOutcome,
    apply_triple_barrier,
    apply_triple_barrier_to_bars,
    opens_from_closes,
)
from .weights import (
    average_uniqueness,
    combined_weights,
    concurrency,
    effective_sample_size,
    linear_decay_weights,
    time_decay_weights,
    tstat_inflation_factor,
    weighted_mean,
    weighted_tstat,
)

__all__ = [
    "OUTCOME_STOP",
    "OUTCOME_TARGET",
    "OUTCOME_TIMEOUT",
    "BarrierOutcome",
    "apply_triple_barrier",
    "apply_triple_barrier_to_bars",
    "opens_from_closes",
    "average_uniqueness",
    "combined_weights",
    "concurrency",
    "effective_sample_size",
    "linear_decay_weights",
    "time_decay_weights",
    "tstat_inflation_factor",
    "weighted_mean",
    "weighted_tstat",
]
