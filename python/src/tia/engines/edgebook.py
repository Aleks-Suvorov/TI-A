"""The Edge Book: hierarchical conditional expectancy with credible bounds.

``docs/01-THEORY.md`` §7. This is what the trade gate actually acts on, and it is
the component that produces selectivity as a *consequence* rather than as a
hand-set threshold.

Layers 1-6 of the system produce a direction and a calibrated probability. They
do not produce an expectancy, because expectancy depends on the distribution of
outcome returns given the setup, not on a win rate. Modelling the realised return
directly -- rather than a win/loss label -- also handles all three triple-barrier
outcomes uniformly; a win-rate model has to special-case timeouts, which are
common and are neither wins nor losses.

The estimator is a Normal-Inverse-Gamma conjugate model per cell, with each cell's
prior taken from its parent in a three-level hierarchy whose root asserts *zero*
expectancy. The null hypothesis is therefore built into the model: absent
evidence, every cell believes there is no edge, and evidence has to move it.

**Architectural note.** Despite living in ``engines/``, this is not an
:class:`tia.types.Engine`. It sits *downstream* of fusion, consuming the pooled
evidence rather than voting alongside it, so it has no ``score`` and is never
pooled. ``SPEC.md`` §4 lists it here for discoverability; the deviation is
deliberate and is the correct topology.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from ..config import Config

__all__ = ["SetupFamily", "EdgeEstimate", "EdgeBook", "student_t_ppf", "norm_ppf"]


class SetupFamily(enum.IntEnum):
    """The qualitatively distinct configurations the engines can produce.

    Named, not learned. Four families keep every cell populated; a learned
    taxonomy would find more of them and would populate none.
    """

    CONTINUATION = 0  #: cheap displacement in a persistent regime
    SWEEP_REVERSAL = 1  #: a confirmed sweep that failed
    COMPRESSION_BREAK = 2  #: expansion out of a compressed range
    RANGE_FADE = 3  #: absorption at a confirmed range extreme

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Quantile functions (no scipy dependency in the live path)
# ---------------------------------------------------------------------------


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF, Acklam's rational approximation.

    Accurate to about 1.15e-9 in absolute value across the domain, which is
    several orders of magnitude better than the estimation error of anything it
    is used on here.
    """
    if not 0.0 < p < 1.0:
        return -math.inf if p <= 0.0 else math.inf
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


def student_t_ppf(p: float, dof: float) -> float:
    """Inverse Student-t CDF via the Cornish-Fisher expansion.

    Exact enough for our purpose: the expansion's error at ``dof >= 5`` is small
    compared with the posterior width it is being applied to, and the direction
    of any residual error is conservative because the tail is slightly
    over-weighted. Below ``dof = 3`` the t distribution's variance does not
    exist, and a cell that thin should not be producing a tradeable bound at
    all -- so the function widens aggressively there rather than pretending.
    """
    if dof <= 2.0:
        return norm_ppf(p) * 4.0
    z = norm_ppf(p)
    z2, z3, z5 = z * z, z**3, z**5
    g1 = (z3 + z) / 4.0
    g2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / 96.0
    g3 = (3.0 * z**7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / 384.0
    return z + g1 / dof + g2 / dof**2 + g3 / dof**3


# ---------------------------------------------------------------------------
# Estimates and cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EdgeEstimate:
    """Posterior summary for one cell, in sigma units."""

    mean: float  #: posterior mean expectancy, mu_n
    lcb: float  #: lower credible bound at the configured quantile
    ucb: float
    var_mean: float  #: posterior variance of mu, the parameter uncertainty
    outcome_var: float  #: posterior mean of the outcome variance
    n_eff: float  #: effective (weighted, decayed) observation count
    dof: float
    cell: str

    @property
    def is_thin(self) -> bool:
        """True when the estimate is dominated by the prior rather than data."""
        return self.n_eff < 10.0


class _Cell:
    """Decayed weighted sufficient statistics. Three floats and a clock."""

    __slots__ = ("n_w", "s1", "s2", "t_last")

    def __init__(self) -> None:
        self.n_w = 0.0
        self.s1 = 0.0
        self.s2 = 0.0
        self.t_last = 0.0

    def decay_to(self, t: float, lam_per_bar: float) -> None:
        """Apply exponential forgetting lazily.

        Decaying every cell on every bar would be O(cells) per bar for no
        benefit. Decaying on access is exact, because the decay factor is
        multiplicative and depends only on elapsed time.
        """
        dt = t - self.t_last
        if dt <= 0.0:
            return
        f = lam_per_bar**dt
        if f < 1e-12:
            self.n_w = self.s1 = self.s2 = 0.0
        else:
            self.n_w *= f
            self.s1 *= f
            self.s2 *= f
        self.t_last = t

    def add(self, r: float, w: float) -> None:
        self.n_w += w
        self.s1 += w * r
        self.s2 += w * r * r


class EdgeBook:
    """Three-level hierarchical expectancy model.

    ``(regime, setup, bucket)`` shrinks toward ``(regime, setup)``, which shrinks
    toward a global root whose prior mean is zero. Cells are addressed by tuple;
    there are at most ``4 x 4 x edge_evidence_buckets`` leaves, so the whole book
    is a few kilobytes and every lookup is O(1).

    Thread-safety is not provided and is not needed: the book is updated from the
    single bar-processing thread.
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        c = self.cfg
        self._lam = 0.5 ** (1.0 / max(c.edge_halflife_bars, 1.0))
        self._leaf: dict[tuple[int, int, int], _Cell] = {}
        self._node: dict[tuple[int, int], _Cell] = {}
        self._root = _Cell()
        self._clock = 0.0
        # Prior on the outcome variance. With barriers at roughly 1.6 and 2.6
        # sigma, a typical outcome standard deviation is near 1.5 sigma, so
        # E[var] = b0/(a0-1) = 2.25. Weakly informative: a0 = 3 is worth about
        # four observations.
        self.a0 = 3.0
        self.b0 = 4.5

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._leaf.clear()
        self._node.clear()
        self._root = _Cell()
        self._clock = 0.0

    def tick(self, bars: float = 1.0) -> None:
        """Advance the forgetting clock by one bar."""
        self._clock += bars

    # ------------------------------------------------------------------ #
    def observe(
        self,
        regime: int,
        setup: int,
        bucket: int,
        ret_sigma: float,
        weight: float = 1.0,
    ) -> None:
        """Record one realised outcome, in sigma units, signed by direction.

        ``weight`` is the sample weight from :mod:`tia.labeling.weights` --
        uniqueness times time decay. Overlapping labels are not independent
        observations and must not be counted as such.
        """
        if ret_sigma != ret_sigma or weight <= 0.0 or not math.isfinite(ret_sigma):
            return
        # Clip pathological outcomes. A stop that gapped ten sigma is real, but
        # letting one such observation dominate a cell's variance estimate makes
        # the cell useless for the next hundred trades.
        r = max(-12.0, min(12.0, float(ret_sigma)))
        for cell in (
            self._leaf.setdefault((regime, setup, bucket), _Cell()),
            self._node.setdefault((regime, setup), _Cell()),
            self._root,
        ):
            cell.decay_to(self._clock, self._lam)
            cell.add(r, float(weight))

    # ------------------------------------------------------------------ #
    def _posterior(self, cell: _Cell, mu0: float, kappa0: float, label: str) -> EdgeEstimate:
        cell.decay_to(self._clock, self._lam)
        n_w, s1, s2 = cell.n_w, cell.s1, cell.s2
        kappa_n = kappa0 + n_w
        a_n = self.a0 + 0.5 * n_w

        if n_w <= 0.0:
            mu_n = mu0
            b_n = self.b0
        else:
            rbar = s1 / n_w
            ss = max(s2 - s1 * s1 / n_w, 0.0)
            mu_n = (kappa0 * mu0 + s1) / kappa_n
            b_n = self.b0 + 0.5 * ss + (kappa0 * n_w * (rbar - mu0) ** 2) / (2.0 * kappa_n)

        dof = 2.0 * a_n
        scale2 = b_n / (a_n * kappa_n)
        scale = math.sqrt(max(scale2, 0.0))
        q = self.cfg.edge_lcb_quantile
        t_lo = student_t_ppf(q, dof)
        t_hi = student_t_ppf(1.0 - q, dof)
        var_mu = scale2 * (dof / (dof - 2.0)) if dof > 2.0 else float("inf")

        return EdgeEstimate(
            mean=mu_n,
            lcb=mu_n + t_lo * scale,
            ucb=mu_n + t_hi * scale,
            var_mean=var_mu,
            outcome_var=b_n / max(a_n - 1.0, 1e-9),
            n_eff=n_w,
            dof=dof,
            cell=label,
        )

    @staticmethod
    def _minus(a: _Cell, b: _Cell) -> _Cell:
        """``a`` with ``b``'s contribution removed. Both must share a clock."""
        out = _Cell()
        out.n_w = max(a.n_w - b.n_w, 0.0)
        out.s1 = a.s1 - b.s1
        out.s2 = max(a.s2 - b.s2, 0.0)
        out.t_last = a.t_last
        if out.n_w <= 0.0:
            out.s1 = out.s2 = 0.0
        return out

    def estimate(self, regime: int, setup: int, bucket: int) -> EdgeEstimate:
        """Posterior expectancy for one cell, shrunk through the hierarchy.

        Shrinkage is **leave-one-out**: a cell's prior comes from its parent's
        evidence *excluding the cell's own*, and the parent's prior likewise
        excludes the parent's. Without that exclusion the same observations set
        the prior and then update against it, so each level re-counts them --
        five observations of +2 sigma produced a posterior mean of 0.77 instead
        of 0.33, and the resulting credible bound cleared a gate it had no
        business clearing. Using data twice is exactly the error the whole
        architecture is built to avoid, and a hierarchy makes it easy to commit
        by accident.
        """
        c = self.cfg
        leaf_cell = self._leaf.setdefault((regime, setup, bucket), _Cell())
        node_cell = self._node.setdefault((regime, setup), _Cell())
        for cell in (leaf_cell, node_cell, self._root):
            cell.decay_to(self._clock, self._lam)

        root_excl = self._minus(self._root, node_cell)
        root = self._posterior(
            root_excl, c.edge_prior_mean_sigma, c.edge_prior_strength, "global"
        )
        node_excl = self._minus(node_cell, leaf_cell)
        node = self._posterior(
            node_excl, root.mean, c.edge_pooling_strength, f"r{regime}/s{setup}"
        )
        return self._posterior(
            leaf_cell, node.mean, c.edge_pooling_strength, f"r{regime}/s{setup}/b{bucket}"
        )

    # ------------------------------------------------------------------ #
    def bucket_of(self, log_odds: float, n_buckets: int | None = None) -> int:
        """Map pooled evidence to an evidence bucket.

        Fixed log-odds boundaries rather than empirical quantiles, so the mapping
        is stable through time and identical between the research and live paths.
        An empirical-quantile mapping would drift, silently reassigning history
        to different cells.
        """
        n = n_buckets or self.cfg.edge_evidence_buckets
        edges = [0.25, 0.6, 1.0, 1.6]
        a = abs(log_odds)
        b = 0
        for e in edges[: max(n - 1, 0)]:
            if a >= e:
                b += 1
        return min(b, n - 1)

    def summary(self) -> str:
        """Human-readable dump of populated cells. For reports and audits."""
        lines = [f"Edge Book  clock={self._clock:.0f} bars  halflife={self.cfg.edge_halflife_bars:.0f}"]
        root = self._posterior(self._root, self.cfg.edge_prior_mean_sigma, self.cfg.edge_prior_strength, "global")
        lines.append(
            f"  global      n_eff={root.n_eff:8.1f}  mean={root.mean:+.4f}  lcb={root.lcb:+.4f}"
        )
        for key in sorted(self._leaf):
            est = self.estimate(*key)
            if est.n_eff < 1.0:
                continue
            lines.append(
                f"  r{key[0]} s{key[1]} b{key[2]}   n_eff={est.n_eff:8.1f}  "
                f"mean={est.mean:+.4f}  lcb={est.lcb:+.4f}  sd={math.sqrt(est.outcome_var):.3f}"
            )
        return "\n".join(lines)

    def state_dict(self) -> dict[str, object]:
        """Serialisable state, for persistence across restarts."""
        return {
            "clock": self._clock,
            "root": [self._root.n_w, self._root.s1, self._root.s2, self._root.t_last],
            "node": {f"{k[0]}|{k[1]}": [v.n_w, v.s1, v.s2, v.t_last] for k, v in self._node.items()},
            "leaf": {
                f"{k[0]}|{k[1]}|{k[2]}": [v.n_w, v.s1, v.s2, v.t_last]
                for k, v in self._leaf.items()
            },
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.reset()
        self._clock = float(state.get("clock", 0.0))  # type: ignore[arg-type]
        r = state.get("root")
        if isinstance(r, list):
            self._root.n_w, self._root.s1, self._root.s2, self._root.t_last = (float(x) for x in r)
        for k, v in dict(state.get("node", {})).items():  # type: ignore[arg-type]
            a, b = (int(x) for x in k.split("|"))
            c = _Cell()
            c.n_w, c.s1, c.s2, c.t_last = (float(x) for x in v)
            self._node[(a, b)] = c
        for k, v in dict(state.get("leaf", {})).items():  # type: ignore[arg-type]
            a, b, d = (int(x) for x in k.split("|"))
            c = _Cell()
            c.n_w, c.s1, c.s2, c.t_last = (float(x) for x in v)
            self._leaf[(a, b, d)] = c
