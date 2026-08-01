"""Structured decision and trade journal.

A trading system without a journal cannot be operated: a live-vs-backtest
divergence, a broker dispute, or a post-mortem all require knowing exactly what
the system saw and decided, bar by bar. Python's ``logging`` module is the wrong
tool here -- log lines are for humans, and the questions above are answered by
machines. This journal writes one JSON object per line (JSONL), append-only,
flushed per write, using only the standard library.

What gets journalled and why:

``decision``  every *actionable* decision, plus every Nth heartbeat NO_TRADE so
              a silent system is distinguishable from a dead one
``trade``     every booked trade, with gross/net/cost and the cell it updated
``health``    every demotion-level transition, with the findings that drove it
``lifecycle`` start/stop/reset, carrying the config manifest hash so any replay
              can prove which parameter set produced the record

The journal is disabled when constructed with ``path=None``, which is the
research default: backtests keep everything in memory already, and writing a
half-gigabyte of JSONL per sweep would help nobody.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, IO, Mapping

__all__ = ["DecisionJournal"]


def _clean(value: Any) -> Any:
    """Make a value JSON-safe. NaN and infinities become ``None``: a journal
    entry that cannot round-trip through ``json.loads`` is worse than none."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class DecisionJournal:
    """Append-only JSONL journal. One instance per pipeline.

    Failure policy: journalling must never take the trading system down, but a
    silently dead journal defeats its purpose. So the first write failure
    disables the journal and records the error on :attr:`last_error`, which the
    health report surfaces -- the operator learns the journal died without the
    pipeline dying with it.
    """

    def __init__(
        self,
        path: str | Path | None,
        manifest_hash: str = "",
        heartbeat_every: int = 100,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.heartbeat_every = int(heartbeat_every)
        self.manifest_hash = manifest_hash
        self.last_error: str | None = None
        self._fh: IO[str] | None = None
        self._n_since_heartbeat = 0
        self.n_written = 0
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
                self._emit("lifecycle", {"event": "start", "manifest": manifest_hash,
                                         "pid": os.getpid()})
            except OSError as exc:
                self.last_error = f"journal open failed: {exc}"
                self._fh = None

    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._fh is None:
            return
        record = {"t": time.time(), "kind": kind, **_clean(payload)}
        try:
            self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._fh.flush()
            self.n_written += 1
        except OSError as exc:
            self.last_error = f"journal write failed: {exc}"
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    # ------------------------------------------------------------------ #
    def decision(self, dec: Any) -> None:
        """Journal an actionable decision fully; heartbeat the quiet ones.

        The heartbeat is not decoration. A system that stands aside for weeks is
        indistinguishable from a dead one without a periodic proof of life, and
        `docs/13-MONITORING.md` lists exactly that confusion as an operator trap.
        """
        actionable = dec.is_actionable
        self._n_since_heartbeat += 1
        if not actionable and self._n_since_heartbeat < self.heartbeat_every:
            return
        self._n_since_heartbeat = 0
        payload: dict[str, Any] = {
            "bar": dec.decided_at_index,
            "ts": dec.timestamp,
            "action": dec.action.value,
            "position_after": int(dec.position_after),
            "regime": int(dec.regime),
            "ev_lcb": dec.expected_value_lcb,
        }
        if dec.fusion is not None:
            payload.update(
                p=dec.fusion.p_success, p_low=dec.fusion.p_low, p_high=dec.fusion.p_high,
                ebe=dec.fusion.ebe, log_odds=dec.fusion.log_odds,
            )
        if actionable and dec.target is not None:
            payload.update(
                stop=dec.target.stop_price, target=dec.target.target_price,
                sigma=dec.target.sigma,
            )
        if actionable and dec.risk is not None:
            payload["risk_fraction"] = dec.risk.risk_fraction
        if not actionable and dec.card is not None:
            payload["veto"] = list(dec.card.veto_reasons)[:3]
        self._emit("decision", payload)

    def trade(self, trade: Any) -> None:
        self._emit(
            "trade",
            {
                "entry_index": trade.entry_index,
                "exit_index": trade.exit_index,
                "direction": trade.direction,
                "entry": trade.entry_price,
                "exit": trade.exit_price,
                "gross_sigma": trade.ret_sigma,
                "cost_sigma": trade.cost_sigma,
                "net_sigma": trade.net_sigma,
                "outcome": trade.outcome,
                "reason": trade.reason,
                "setup": trade.setup,
                "regime": trade.regime,
                "bucket": trade.bucket,
                "risk_fraction": trade.risk_fraction,
                "equity_after": trade.equity_after,
            },
        )

    def health(self, level_name: str, findings: list[str]) -> None:
        self._emit("health", {"level": level_name, "findings": findings[:6]})

    def lifecycle(self, event: str, **extra: Any) -> None:
        self._emit("lifecycle", {"event": event, "manifest": self.manifest_hash, **extra})

    def close(self) -> None:
        if self._fh is not None:
            self._emit("lifecycle", {"event": "stop", "n_written": self.n_written})
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
