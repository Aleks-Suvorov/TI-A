# Changelog

All notable changes, most recent first. Every entry that touched a
`THEORY`-tagged default is also recorded in the trials ledger of
`docs/00-PREREGISTRATION.md`.

## 1.0.0-rc4 — final review board

- **Wired the safety layer.** `docs/13-MONITORING.md` documented a demotion
  ladder and four kill switches; until this release the ladder was never
  instantiated and only the drawdown switch could ever trip. `TIA` now owns a
  `DriftMonitor`, feeds it every valid feature snapshot and every resolved
  outcome (regardless of learn mode), refreshes health every 50 bars, feeds
  the rolling Brier into the `RiskLimits` calibration switch, and applies the
  ladder through the existing gate so every refusal keeps a named reason:
  TIGHTENED halves size, RESTRICTED requires a committed regime posterior,
  NO_TRADE and above block entries.
- **Demotion requires evidence.** The negative-Brier-skill rung demoted a
  demonstrably healthy run at n=62 outcomes — gated selection lifts realized
  success above the population forecast, i.e. miscalibration in the safe
  direction. Skill-based demotion now requires the isotonic map's own sample
  floor (`calibration_min_samples`); below it the finding is reported, not
  acted on. The absolute Brier alarm is unchanged and catches real breakage
  at any sample size.
- **Added the decision journal.** Append-only JSONL of decisions (with
  NO_TRADE heartbeats), trades (gross/cost/net), health transitions and
  lifecycle events, each tied to the config manifest hash. Stdlib only;
  disabled by default; first write failure disables it without taking the
  pipeline down. Verified to change no decision.
- **`Config.validate()`.** The pipeline refuses to start on a nonsensical
  configuration — negative stops, probability floors above one, zero
  execution lag (look-ahead by construction), inverted drawdown thresholds.
- **The synthetic generator can now express overnight gaps**
  (`overnight_gap_sigma`, default 0, byte-identical to old seeds when off).
  This closes the blind spot that hid the D2 fill-timing bug. Accounting is
  verified exact on gapped markets, including gap-through-stop fills at the
  open.
- **First contact with real data** (`tools/realdata_check.py`): six ETFs,
  15 years daily. 22,632 bars, zero fatal validator rejects, zero untrained
  signals, and 0 out-of-sample trades from 66 pooled training candidates —
  the honest outcome the power arithmetic predicts. SPY's median overnight
  gap is 27.9bp: the magnitude of per-trade optimism the D2 bug would have
  injected on real data.

## 1.0.0-rc3 — adversarial audit (docs/19)

- D1: bar-close exits (vertical barrier, reversal, session close) were never
  booked — no trade record, no equity change, no learning signal.
- D2: entries filled at the decision close while training assumed next-bar
  open; fixed with deferred fills re-anchored to the actual open.
- D3: realized equity charged zero transaction costs; now charged the
  decision-time round trip, with gross kept for the Edge Book.
- D4: `SWEEP_REVERSAL` was unreachable (wrong dict read); a quarter of the
  Edge Book taxonomy did not exist.
- D5: the Pine regime posterior omitted Beta log-normalizers (a permanent
  ~e⁴ inter-regime tilt); the audit's own first fix inverted the sign and was
  caught by cross-check before shipping.

## 1.0.0-rc2 — deep invariant audit

- Effective sample size was measured with Kish's statistic, which is
  scale-invariant and blind to uniform overlap (2.6× t-statistic inflation);
  replaced with the overlap-aware `combined_effective_sample_size`.
- Barrier outcomes were misaligned with candidates after drops
  (`source_position` ignored), silently corrupting Edge Book cells in
  walk-forward.
- Two Pine compile errors (`str` parameter shadowing; comma-assignments);
  stale isotonic constants asserting 0%/100% confidence; NaN exported as
  −1e15; a flat market crashed the volatility kernel; export emits into
  `pine/*.pine` by default so JSON and Pine cannot drift.
- Added `tools/audit.py` (1.19M invariant assertions) and
  `tools/pinelint.py`.

## 1.0.0-rc1 — initial system

- Streaming kernel, nine engines, four-regime filter with zero parameters
  fitted to returns, CALOP fusion, hierarchical NIG Edge Book, triple-barrier
  policy, Bayesian-Kelly sizing, purged CPCV validation suite, Pine port with
  frozen-model export, 19 documentation files.
- Early fixes found by the first test pass: Edge Book leave-one-out shrinkage
  (a cell's own data informed its own prior), spread-estimator aggregation
  (median, CS-only), isotonic endpoint certainty bounds.
