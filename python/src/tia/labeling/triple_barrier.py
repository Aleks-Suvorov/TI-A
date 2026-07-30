"""Triple-barrier labelling with pessimistic tie-breaks.

The triple-barrier method (López de Prado, *Advances in Financial Machine
Learning*, 2018, ch. 3) replaces the fixed-horizon label -- "was the return over
the next ``h`` bars positive?" -- with the question a trader actually faces:
*starting from a real fill, does price reach the profit target before it reaches
the stop, and if neither, where is it when the clock runs out?* Three barriers
therefore bound every label:

1.  an upper (profit) barrier at ``target_sigma`` volatilities,
2.  a lower (stop) barrier at ``stop_sigma`` volatilities,
3.  a vertical barrier ``max_holding`` bars after the decision.

Why this labelling and not fixed-horizon returns
------------------------------------------------
*   The label matches the position that will actually be held, including the
    path dependence a stop introduces. A fixed-horizon label credits a strategy
    for a favourable close it would never have seen, because it was stopped out
    on the way.
*   Barriers stated in *sigma* units rather than price or ticks make one
    parameter set portable across instruments and volatility regimes
    (SPEC §1, :class:`~tia.types.TargetSpec`).
*   The vertical barrier encodes the fact that the conditioning information
    which justified the trade decays: after ``max_holding`` bars the position is
    no longer the one that was analysed (see ``Config.max_holding_bars``).

Causality and pessimism
-----------------------
Two rules from SPEC §2 are enforced structurally here, because they are the two
places where barrier labelling silently manufactures edge:

*   **Decide at ``t``, fill at ``t + execution_lag``.** The entry price is the
    **open** of bar ``entry_index + execution_lag`` and all three barriers are
    measured from that price. Using the decision bar's close as the fill price
    -- the default in most published implementations -- is a look-ahead worth
    a large fraction of a typical edge, because the decision was made *because*
    of that close.
*   **Pessimistic tie-breaks.** Bar data cannot say whether the high or the low
    came first. When one bar's range contains both barriers the **stop** is
    recorded as hit (:data:`OUTCOME_STOP`). When a bar's *open* has already
    gapped through the stop, the fill is the open, i.e. the worse of the two
    prices. Target fills are never credited with a favourable gap: the exit is
    booked at the target price even when the open leapt beyond it. Every one of
    these choices costs the strategy something, which is the point -- an
    optimistic tie-break rule can convert a coin flip into a 55% hit rate.

What a suspicious result looks like
-----------------------------------
If a strategy's measured hit rate is materially higher when the tie-break is
optimistic (``prefer_stop=False``, provided only as a diagnostic), a large part
of its apparent edge lives inside bars and is unobtainable without tick data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import Bar

__all__ = [
    "OUTCOME_TARGET",
    "OUTCOME_STOP",
    "OUTCOME_TIMEOUT",
    "BarrierOutcome",
    "apply_triple_barrier",
    "apply_triple_barrier_to_bars",
    "opens_from_closes",
]

#: Profit barrier touched first.
OUTCOME_TARGET: int = 1
#: Stop barrier touched first, or assumed first under a same-bar tie.
OUTCOME_STOP: int = -1
#: Vertical barrier reached with neither price barrier touched.
OUTCOME_TIMEOUT: int = 0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarrierOutcome:
    """Dataclass-of-arrays describing the resolution of every label.

    A struct-of-arrays rather than an array-of-structs so that downstream code
    (sample weights, purged cross-validation, calibration) can slice columns
    without a Python loop, and so that a NaN cannot hide inside an opaque record.

    All arrays share length ``len(self)``. ``source_position`` maps each row back
    to its position in the *input* arrays, because labels that cannot be filled
    (no bar at ``entry_index + execution_lag``) or whose sigma is not finite are
    dropped rather than fabricated.
    """

    entry_index: np.ndarray  #: decision bar; the label's information cut-off
    fill_index: np.ndarray  #: bar whose OPEN is the entry price
    exit_index: np.ndarray  #: bar on which the position was closed
    entry_price: np.ndarray
    exit_price: np.ndarray
    outcome: np.ndarray  #: +1 target, -1 stop, 0 timeout
    ret_sigma: np.ndarray  #: signed realised log return in sigma units
    ret_log: np.ndarray  #: signed realised log return, raw
    holding_bars: np.ndarray  #: exit_index - fill_index
    direction: np.ndarray
    sigma: np.ndarray  #: per-bar sigma used to place the barriers
    stop_price: np.ndarray
    target_price: np.ndarray
    truncated: np.ndarray  #: vertical barrier clipped by the end of the sample
    source_position: np.ndarray
    n_dropped: int = 0

    def __len__(self) -> int:
        return int(self.entry_index.size)

    def __post_init__(self) -> None:
        n = len(self)
        for name in (
            "fill_index",
            "exit_index",
            "entry_price",
            "exit_price",
            "outcome",
            "ret_sigma",
            "ret_log",
            "holding_bars",
            "direction",
            "sigma",
            "stop_price",
            "target_price",
            "truncated",
            "source_position",
        ):
            arr = getattr(self, name)
            if arr.size != n:
                raise ValueError(f"BarrierOutcome.{name} has length {arr.size}, expected {n}")

    # -- convenience -------------------------------------------------------

    def select(self, mask: np.ndarray) -> "BarrierOutcome":
        """Row subset, preserving the container type."""
        m = np.asarray(mask)
        return BarrierOutcome(
            entry_index=self.entry_index[m],
            fill_index=self.fill_index[m],
            exit_index=self.exit_index[m],
            entry_price=self.entry_price[m],
            exit_price=self.exit_price[m],
            outcome=self.outcome[m],
            ret_sigma=self.ret_sigma[m],
            ret_log=self.ret_log[m],
            holding_bars=self.holding_bars[m],
            direction=self.direction[m],
            sigma=self.sigma[m],
            stop_price=self.stop_price[m],
            target_price=self.target_price[m],
            truncated=self.truncated[m],
            source_position=self.source_position[m],
            n_dropped=self.n_dropped,
        )

    def spans(self) -> tuple[np.ndarray, np.ndarray]:
        """``(entry_index, exit_index)``: the interval each label *observes*.

        This is the span purged cross-validation must respect. It starts at the
        decision bar, not the fill bar, because the decision consumed
        information dated ``entry_index``.
        """
        return self.entry_index, self.exit_index

    def binary_label(self, timeout_rule: str = "sign") -> np.ndarray:
        """Map outcomes to ``{0, 1}`` for calibration and Brier scoring.

        ``timeout_rule="sign"`` scores a timeout by the sign of its realised
        return; ``"loss"`` scores every timeout as a failure. ``"sign"`` matches
        what the fusion layer's probability actually claims (``p_success`` is
        "reaches target before stop"), but ``"loss"`` is the strictly
        conservative reading and is worth reporting alongside.
        """
        if timeout_rule == "loss":
            return (self.outcome > 0).astype(float)
        if timeout_rule != "sign":
            raise ValueError("timeout_rule must be 'sign' or 'loss'")
        y = (self.outcome > 0).astype(float)
        tmo = self.outcome == OUTCOME_TIMEOUT
        y[tmo] = (self.ret_sigma[tmo] > 0.0).astype(float)
        return y

    def outcome_counts(self) -> dict[str, int]:
        return {
            "target": int(np.sum(self.outcome == OUTCOME_TARGET)),
            "stop": int(np.sum(self.outcome == OUTCOME_STOP)),
            "timeout": int(np.sum(self.outcome == OUTCOME_TIMEOUT)),
            "dropped": int(self.n_dropped),
            "truncated": int(np.sum(self.truncated)),
        }

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "entry_index": self.entry_index,
            "fill_index": self.fill_index,
            "exit_index": self.exit_index,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "outcome": self.outcome,
            "ret_sigma": self.ret_sigma,
            "ret_log": self.ret_log,
            "holding_bars": self.holding_bars,
            "direction": self.direction,
        }


# ---------------------------------------------------------------------------
# Argument coercion
# ---------------------------------------------------------------------------


def opens_from_closes(close: np.ndarray) -> np.ndarray:
    """Degenerate open series ``open[i] = close[i-1]`` for continuous instruments.

    Used only when no open series is supplied. It is not a look-ahead (bar
    ``i-1``'s close is known before bar ``i`` trades) but it does assume gapless
    continuation, so it understates gap risk. Pass real opens whenever they
    exist.
    """
    c = np.asarray(close, dtype=float).ravel()
    o = np.empty_like(c)
    o[0] = c[0]
    o[1:] = c[:-1]
    return o


def _as_label_array(
    value: float | int | Sequence[float] | np.ndarray, m: int, name: str
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(m, float(arr))
    arr = arr.ravel()
    if arr.size != m:
        raise ValueError(f"{name} must be scalar or have one entry per label ({m}), got {arr.size}")
    return arr.astype(float)


def _resolve_sigma(
    sigma: float | Sequence[float] | np.ndarray, entry_indices: np.ndarray, n_bars: int
) -> np.ndarray:
    """Resolve ``sigma`` to one value per label.

    Accepts a scalar, a per-*bar* series (length ``n_bars``, sampled causally at
    the decision bar), or a per-*label* series. When ``n_bars`` equals the label
    count the per-bar interpretation wins; pass a scalar or reshape if that is
    not what you meant.
    """
    arr = np.asarray(sigma, dtype=float)
    if arr.ndim == 0:
        return np.full(entry_indices.size, float(arr))
    arr = arr.ravel()
    if arr.size == n_bars:
        return arr[entry_indices].astype(float)
    if arr.size == entry_indices.size:
        return arr.astype(float)
    raise ValueError(
        f"sigma must be scalar, per-bar (len {n_bars}) or per-label (len {entry_indices.size}); "
        f"got len {arr.size}"
    )


# ---------------------------------------------------------------------------
# The labeller
# ---------------------------------------------------------------------------


def apply_triple_barrier(
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
    entry_indices: Sequence[int] | np.ndarray,
    directions: Sequence[int] | np.ndarray,
    stop_sigma: float | Sequence[float] | np.ndarray,
    target_sigma: float | Sequence[float] | np.ndarray,
    sigma: float | Sequence[float] | np.ndarray,
    max_holding: int | Sequence[int] | np.ndarray,
    execution_lag: int = 1,
    open_: Sequence[float] | np.ndarray | None = None,
    *,
    prefer_stop: bool = True,
    credit_target_gap: bool = False,
) -> BarrierOutcome:
    """Label every proposed entry by which barrier it reaches first.

    Parameters
    ----------
    high, low, close
        Bar arrays of equal length ``n``, in chronological order.
    entry_indices
        Indices of the **decision** bars. Need not be sorted or unique.
    directions
        ``+1`` long, ``-1`` short, one per entry. Zeros are rejected: an
        engine score of exactly 0.0 means *no opinion* (SPEC §1) and must not
        reach the labeller.
    stop_sigma, target_sigma
        Barrier distances in units of ``sigma``, scalar or one per label. Both
        must be strictly positive; a zero-width barrier would resolve on the
        fill bar by construction.
    sigma
        Per-bar log-return standard deviation: scalar, per-bar series (sampled
        at the decision bar) or per-label series. See :func:`_resolve_sigma`.
    max_holding
        Vertical barrier, in bars after the **decision** bar. Must be at least
        ``execution_lag``.
    execution_lag
        Bars between decision and fill; never 0 (SPEC §2.3). The entry price is
        ``open_[entry_index + execution_lag]``.
    open_
        Open series. When ``None``, :func:`opens_from_closes` is used and gap
        risk is understated -- see its docstring.
    prefer_stop
        Diagnostic only. ``True`` (the default and the only setting permitted in
        an acceptance run) applies the pessimistic same-bar tie-break of
        SPEC §2.7. Setting it ``False`` measures how much of the strategy's edge
        depends on winning intrabar coin flips.
    credit_target_gap
        When ``True``, a target that gaps open beyond its level is filled at the
        open (better than the target). Default ``False`` books the target price,
        the conservative choice.

    Returns
    -------
    BarrierOutcome
        One row per *fillable* label. Labels whose fill bar falls past the end
        of the sample, or whose ``sigma`` is not finite and positive, are
        dropped and counted in ``n_dropped``.

    Notes
    -----
    Barriers are geometric, matching :class:`~tia.types.TargetSpec`::

        stop   = fill * exp(-direction * stop_sigma   * sigma)
        target = fill * exp(+direction * target_sigma * sigma)

    Resolution order within a bar, from first observable price onward:

    1.  open gapped through the stop  -> stop, filled at ``worse(open, stop)``
    2.  open gapped through the target -> target (the open genuinely traded
        through it, so no tie-break is involved)
    3.  range touched the stop        -> stop at the stop price
    4.  range touched the target      -> target at the target price

    Steps 3 and 4 are the ambiguous case, and their order is the pessimism
    required by SPEC §2.7.
    """
    hi = np.asarray(high, dtype=float).ravel()
    lo = np.asarray(low, dtype=float).ravel()
    cl = np.asarray(close, dtype=float).ravel()
    n = hi.size
    if not (lo.size == n and cl.size == n):
        raise ValueError("high, low and close must have the same length")
    if n == 0:
        raise ValueError("empty price series")
    op = opens_from_closes(cl) if open_ is None else np.asarray(open_, dtype=float).ravel()
    if op.size != n:
        raise ValueError("open_ must have the same length as close")
    if execution_lag < 1:
        raise ValueError("execution_lag must be >= 1 (SPEC 2.3: it is never zero)")

    e_idx = np.asarray(entry_indices, dtype=np.int64).ravel()
    m = e_idx.size
    if m == 0:
        return _empty_outcome()
    if e_idx.min() < 0 or e_idx.max() >= n:
        raise ValueError("entry_indices out of range")

    d = np.asarray(directions, dtype=np.int64).ravel()
    if d.size != m:
        raise ValueError("directions must have one entry per label")
    if not np.all((d == 1) | (d == -1)):
        raise ValueError("directions must be +1 or -1; 0 means 'no opinion' and is not labellable")

    stop_s = _as_label_array(stop_sigma, m, "stop_sigma")
    tgt_s = _as_label_array(target_sigma, m, "target_sigma")
    if np.any(stop_s <= 0.0) or np.any(tgt_s <= 0.0):
        raise ValueError("stop_sigma and target_sigma must be strictly positive")
    sig = _resolve_sigma(sigma, e_idx, n)
    hold = np.asarray(_as_label_array(max_holding, m, "max_holding")).astype(np.int64)
    if np.any(hold < execution_lag):
        raise ValueError("max_holding must be at least execution_lag")

    fill_idx = e_idx + execution_lag
    keep = (fill_idx < n) & np.isfinite(sig) & (sig > 0.0)
    n_dropped = int(m - keep.sum())
    pos = np.nonzero(keep)[0]
    k = pos.size
    if k == 0:
        return _empty_outcome(n_dropped=n_dropped)

    out_entry = e_idx[pos]
    out_fill = fill_idx[pos]
    out_dir = d[pos]
    out_sig = sig[pos]
    entry_px = op[out_fill]
    if np.any(~np.isfinite(entry_px)) or np.any(entry_px <= 0.0):
        raise ValueError("non-positive or non-finite entry price; clean the bar series first")
    stop_px = entry_px * np.exp(-out_dir * stop_s[pos] * out_sig)
    tgt_px = entry_px * np.exp(out_dir * tgt_s[pos] * out_sig)
    vertical = out_entry + hold[pos]
    truncated = vertical > (n - 1)
    last = np.minimum(vertical, n - 1)

    exit_idx = np.empty(k, dtype=np.int64)
    exit_px = np.empty(k, dtype=float)
    outcome = np.empty(k, dtype=np.int64)

    for i in range(k):
        a, b = int(out_fill[i]), int(last[i])
        di = int(out_dir[i])
        sp, tp = float(stop_px[i]), float(tgt_px[i])
        seg_o, seg_h, seg_l = op[a : b + 1], hi[a : b + 1], lo[a : b + 1]

        if di > 0:
            gap_stop = seg_o <= sp
            gap_tgt = seg_o >= tp
            hit_stop = seg_l <= sp
            hit_tgt = seg_h >= tp
        else:
            gap_stop = seg_o >= sp
            gap_tgt = seg_o <= tp
            hit_stop = seg_h >= sp
            hit_tgt = seg_l <= tp
        # The fill bar's open *is* the entry price; it cannot be a gap.
        gap_stop[0] = False
        gap_tgt[0] = False

        resolved = gap_stop | gap_tgt | hit_stop | hit_tgt
        where = np.nonzero(resolved)[0]
        if where.size == 0:
            exit_idx[i] = b
            exit_px[i] = float(cl[b])
            outcome[i] = OUTCOME_TIMEOUT
            continue
        j = int(where[0])
        exit_idx[i] = a + j
        if gap_stop[j]:
            # Worse of the stop and the gap-through open (SPEC 2.7).
            exit_px[i] = min(sp, float(seg_o[j])) if di > 0 else max(sp, float(seg_o[j]))
            outcome[i] = OUTCOME_STOP
        elif gap_tgt[j]:
            exit_px[i] = float(seg_o[j]) if credit_target_gap else tp
            outcome[i] = OUTCOME_TARGET
        elif hit_stop[j] and hit_tgt[j]:
            # Both barriers inside one bar: the ambiguous case.
            if prefer_stop:
                exit_px[i] = sp
                outcome[i] = OUTCOME_STOP
            else:
                exit_px[i] = tp
                outcome[i] = OUTCOME_TARGET
        elif hit_stop[j]:
            exit_px[i] = sp
            outcome[i] = OUTCOME_STOP
        else:
            exit_px[i] = tp
            outcome[i] = OUTCOME_TARGET

    ret_log = out_dir * np.log(exit_px / entry_px)
    ret_sigma = ret_log / out_sig
    return BarrierOutcome(
        entry_index=out_entry,
        fill_index=out_fill,
        exit_index=exit_idx,
        entry_price=entry_px,
        exit_price=exit_px,
        outcome=outcome,
        ret_sigma=ret_sigma,
        ret_log=ret_log,
        holding_bars=(exit_idx - out_fill).astype(np.int64),
        direction=out_dir,
        sigma=out_sig,
        stop_price=stop_px,
        target_price=tgt_px,
        truncated=truncated,
        source_position=pos.astype(np.int64),
        n_dropped=n_dropped,
    )


def _empty_outcome(n_dropped: int = 0) -> BarrierOutcome:
    zi = np.zeros(0, dtype=np.int64)
    zf = np.zeros(0, dtype=float)
    zb = np.zeros(0, dtype=bool)
    return BarrierOutcome(
        entry_index=zi,
        fill_index=zi.copy(),
        exit_index=zi.copy(),
        entry_price=zf,
        exit_price=zf.copy(),
        outcome=zi.copy(),
        ret_sigma=zf.copy(),
        ret_log=zf.copy(),
        holding_bars=zi.copy(),
        direction=zi.copy(),
        sigma=zf.copy(),
        stop_price=zf.copy(),
        target_price=zf.copy(),
        truncated=zb,
        source_position=zi.copy(),
        n_dropped=n_dropped,
    )


def apply_triple_barrier_to_bars(
    bars: Sequence[Bar],
    entry_indices: Sequence[int] | np.ndarray,
    directions: Sequence[int] | np.ndarray,
    stop_sigma: float | Sequence[float] | np.ndarray,
    target_sigma: float | Sequence[float] | np.ndarray,
    sigma: float | Sequence[float] | np.ndarray,
    max_holding: int | Sequence[int] | np.ndarray,
    execution_lag: int = 1,
    **kwargs: object,
) -> BarrierOutcome:
    """Convenience wrapper over a sequence of :class:`~tia.types.Bar`.

    Rejects partial bars loudly rather than silently mislabelling them: a
    partial bar's high and low are not final, so any barrier decision taken on
    one is a repainting bug (SPEC §2.2).
    """
    if any(b.partial for b in bars):
        raise ValueError("partial bars must be dropped by the data layer before labelling")
    o = np.fromiter((b.open for b in bars), dtype=float, count=len(bars))
    h = np.fromiter((b.high for b in bars), dtype=float, count=len(bars))
    l = np.fromiter((b.low for b in bars), dtype=float, count=len(bars))
    c = np.fromiter((b.close for b in bars), dtype=float, count=len(bars))
    return apply_triple_barrier(
        h,
        l,
        c,
        entry_indices,
        directions,
        stop_sigma,
        target_sigma,
        sigma,
        max_holding,
        execution_lag,
        open_=o,
        **kwargs,  # type: ignore[arg-type]
    )
