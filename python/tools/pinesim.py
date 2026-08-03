"""A bar-by-bar simulator of the Pine port's arithmetic.

TradingView's compiler is not available in this environment, so "does it
compile" can only be checked statically (``tools/pinelint.py``). This tool
answers the *other* question, which static checking cannot: **when it runs,
does it do anything?**

It transliterates ``pine/TIA.pine``'s computation -- the same feature kernel,
the same regime recursion, the same CALOP pooling, the same twelve gates -- in
plain Python, driven by the same ``pine/frozen_model.json`` the Pine constants
are generated from, so no constant is copied by hand. Then it replays real or
synthetic bars through it and reports:

  * how many bars reach each gate, and which gate is the binding constraint;
  * how many BUY / SHORT / SELL / COVER signals each mode and strictness
    produces, so "you will see roughly N signals" is a measurement rather
    than a guess;
  * whether any quantity ever went non-finite, which is how a Pine script
    ends up silently painting nothing.

What it does NOT do: prove the Pine file compiles, prove the transliteration
is faithful line by line, or say anything whatsoever about profitability. It
reports signal counts and veto reasons. Treat a disagreement between this and
the chart as evidence against the transliteration, not against TradingView.

Run:  cd python && PYTHONPATH=src python3 tools/pinesim.py            # synthetic
      cd python && PYTHONPATH=src python3 tools/pinesim.py --real     # ETFs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Pine primitives, reproduced exactly -- including the parts that exist only to
# stop a denominator or a logarithm from reaching an illegal value.
# --------------------------------------------------------------------------- #
def clip(x: float, lo: float, hi: float) -> float:
    if x is None or x != x:
        return (lo + hi) * 0.5
    return max(lo, min(hi, x))


def safediv(a: float, b: float, d: float) -> float:
    if a is None or b is None or a != a or b != b or b == 0.0:
        return d
    return a / b


def slog(x: float) -> float:
    return math.log(max(x if x == x else 1e-12, 1e-12))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-clip(x, -60.0, 60.0)))


def norm_ppf(p: float) -> float:
    pp = clip(p, 1e-12, 1.0 - 1e-12)
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    if pp < 0.02425:
        q = math.sqrt(-2.0 * math.log(pp))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if pp > 0.97575:
        q = math.sqrt(-2.0 * math.log(1.0 - pp))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = pp - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def student_t_ppf(p: float, dof: float) -> float:
    if dof <= 2.0:
        return norm_ppf(p) * 4.0
    z = norm_ppf(p)
    z2, z3 = z * z, z * z * z
    z5, z7 = z3 * z2, z3 * z2 * z2
    g1 = (z3 + z) / 4.0
    g2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / 96.0
    g3 = (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / 384.0
    return z + g1 / dof + g2 / dof**2 + g3 / dof**3


def interp(x: float, xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return x
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    hit = 1
    for k in range(1, n):
        if xs[k] >= x:
            hit = k
            break
    w = safediv(x - xs[hit - 1], xs[hit] - xs[hit - 1], 0.0)
    return ys[hit - 1] + w * (ys[hit] - ys[hit - 1])


def beta_log_pdf_kernel(x: float, a: float, b: float) -> float:
    xx = clip(x, 1e-6, 1.0 - 1e-6)
    return (a - 1.0) * math.log(xx) + (b - 1.0) * math.log(1.0 - xx)


def ewma_alpha(halflife: float) -> float:
    return 1.0 - 0.5 ** (1.0 / max(halflife, 0.5))


def percentrank(window: list[float], x: float) -> float:
    """ta.percentrank: percent of the previous `length` values below x."""
    if not window:
        return float("nan")
    return 100.0 * sum(1 for v in window if v < x) / len(window)


def median(window: list[float]) -> float:
    s = sorted(window)
    n = len(s)
    return float("nan") if n == 0 else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def variance(window: list[float]) -> float:
    n = len(window)
    if n < 2:
        return float("nan")
    m = sum(window) / n
    return sum((v - m) ** 2 for v in window) / n


class Ring:
    """A fixed-length trailing window, the shape every ta.* call sees."""

    __slots__ = ("n", "buf")

    def __init__(self, n: int) -> None:
        self.n = n
        self.buf: list[float] = []

    def push(self, x: float) -> None:
        self.buf.append(x)
        if len(self.buf) > self.n:
            self.buf.pop(0)

    @property
    def full(self) -> bool:
        return len(self.buf) == self.n

    def total(self) -> float:
        return sum(self.buf)


# --------------------------------------------------------------------------- #
class PineSim:
    """One instance per chart, exactly like one instance of the Pine script."""

    def __init__(self, model: dict, mode_a: bool = False, strict: int = 3,
                 allow_long: bool = True, allow_short: bool = True,
                 use_htf: bool = True, cooldown: int = 10, max_per_100: int = 6) -> None:
        ft, reg, eng = model["features"], model["regime"], model["engines"]
        fus, eb, ba = model["fusion"], model["edge_book"], model["barriers"]
        co, ga = model["costs"], model["gate"]
        self.m = model
        self.mode_a, self.strict = mode_a, strict
        self.allow_long, self.allow_short = allow_long, allow_short
        self.use_htf, self.cooldown, self.max_per_100 = use_htf, cooldown, max_per_100

        self.VOL_N, self.VOL_NS, self.VOL_NL = ft["vol_window"], ft["vol_window_short"], ft["vol_window_long"]
        self.RANK_N, self.PART_N = ft["rank_window"], ft["participation_window"]
        self.VR_N, self.PE_N, self.EFF_N = ft["vr_window"], ft["perm_entropy_window"], ft["efficiency_window"]
        self.PIV, self.HTF = ft["pivot_confirm_bars"], ft["htf_multiple"]
        self.HAR = ft["har_weights"]
        self.EWMA_HL, self.CLV_HL = ft["ewma_halflife"], ft["clv_halflife"]
        self.PSI, self.KAL, self.KAL_S = ft["impact_exponent"], ft["kalman_snr"], ft["kalman_slope_snr"]
        self.SW_MAX, self.SW_PEN, self.SW_PART = (ft["sweep_max_bars_to_reclaim"],
                                                  ft["sweep_min_penetration_sigma"],
                                                  ft["sweep_participation_rank_min"])
        self.WARMUP = self.RANK_N + 48

        self.DA = _flat(reg["design_a"])
        self.DB = _flat(reg["design_b"])
        self.DLN = _flat(reg["design_log_beta_norm"])
        self.LOG_STAY, self.LOG_SWITCH = reg["log_stay"], reg["log_switch"]
        self.TEMPER, self.HAZ_HL = reg["tempering_exponent"], reg["hazard_halflife_bars"]
        self.HAZ_MAX, self.MIN_FEAT = reg["hazard_max"], reg["min_usable_features"]

        self.NENG = eng["count"]
        self.PREC = _flat(eng["precision"])
        self.FUSE_T, self.FUSE_B, self.FUSE_SD = fus["temperature"], fus["bias"], fus["evidence_sd"]
        self.CLAMP, self.ALIGN = fus["score_clamp"], fus["align_eps"]
        self.ISO_ON, self.ISO_X, self.ISO_Y = fus["isotonic_enabled"], fus["knots_x"], fus["knots_y"]

        self.NSETUP, self.NBUCKET = eb["n_setups"], eb["n_buckets"]
        self.EDGES, self.EB_Q = eb["bucket_edges"], eb["lcb_quantile"]
        self.MU, self.SCALE, self.DOF = eb["cells"]["mu_n"], eb["cells"]["scale"], eb["cells"]["dof"]

        self.STOP_S, self.TGT_S = ba["stop_sigma"], ba["target_sigma"]
        self.MAX_HOLD, self.MIN_HOLD = ba["max_holding_bars"], ba["min_holding_bars"]
        self.ETA, self.PHI, self.FEE = co["impact_eta"], co["slippage_range_frac"], co["fee_bps"]
        self.Q, self.MAXHS, self.SFALL = (co["default_participation"], co["max_half_spread_sigma"],
                                          co["spread_fallback_rel"])
        self.EVLCB, self.PMIN, self.PEXIT = ga["ev_lcb_min_sigma"], ga["p_min"], ga["p_exit"]
        self.EBEMIN, self.RATE = ga["ebe_min"], ga["max_trades_per_100_bars"]

        # rolling state
        self.i = -1
        self.prev_close = None
        self.absR = Ring(2)
        self.r_bp = Ring(self.VOL_N)
        self.r_rv = Ring(self.VOL_N)
        self.r_rvS = Ring(self.VOL_NS)
        self.r_rvL = Ring(self.VOL_NL)
        self.r_rs = Ring(self.VOL_N)
        self.r_eff_n = Ring(self.EFF_N)
        self.r_eff_d = Ring(self.EFF_N)
        self.r_ret = Ring(self.VR_N)
        self.r_lag = {q: Ring(self.VR_N) for q in (2, 4, 8, 16)}
        self.r_ret_raw = Ring(16)
        self.pr_vol, self.pr_comp = Ring(self.RANK_N), Ring(self.RANK_N)
        self.pr_part, self.pr_ladr, self.pr_abs, self.pr_vr = (Ring(self.RANK_N), Ring(self.RANK_N),
                                                               Ring(self.RANK_N), Ring(self.RANK_N))
        self.r_dv = Ring(self.PART_N)
        self.ewVar = None
        self.clv_ema = None
        self.kal = [None, 0.0, None, 0.0, None]
        self.pe_counts, self.pe_ring, self.pe_head, self.pe_filled = [0] * 6, [-1] * self.PE_N, 0, 0
        self.closes = Ring(67)
        self.highs, self.lows = Ring(2 * self.PIV + 1), Ring(2 * self.PIV + 1)
        self.pivot_hi = self.pivot_lo = None
        self.pivot_hi_bar = self.pivot_lo_bar = 0
        self.htf_n, self.htf_h, self.htf_l = 0, None, None
        self.htf_prev, self.htf_ewvar = None, None
        self.hkal = [None, 0.0, None, 0.0, None]
        self.htf_slope, self.htf_rangepos = None, None
        self.htf_highs, self.htf_lows = [], []
        self.sweep_dir, self.sweep_age = 0, 999
        self.sw_lo = self.sw_hi = None
        self.pend_lo, self.pend_hi = 999, 999
        self.log_alpha = [math.log(0.25)] * 4
        self.ref_post = [0.25] * 4
        self.posterior = [0.25] * 4
        self.has_volume = False

        self.pos, self.entry, self.stop, self.target = 0, None, None, None
        self.bars_held, self.last_exit = 0, -99999
        self.entry_bars: list[int] = []

        self.signals: list[tuple[int, str, float, float, float, float]] = []
        self.vetoes: Counter[str] = Counter()
        self.gate_fail: Counter[str] = Counter()
        self.nonfinite: Counter[str] = Counter()

    # -- one closed bar ----------------------------------------------------- #
    def on_bar(self, o: float, h: float, l: float, c: float, v: float) -> None:
        self.i += 1
        i = self.i
        m = self.m

        ret = None if self.prev_close is None or self.prev_close <= 0 else math.log(c / self.prev_close)
        aR = abs(ret) if ret is not None else 0.0
        prev_aR = self.absR.buf[-1] if self.absR.buf else 0.0
        self.absR.push(aR)

        self.r_bp.push(aR * prev_aR)
        sq = (ret * ret) if ret is not None else 0.0
        self.r_rv.push(sq)
        self.r_rvS.push(sq)
        self.r_rvL.push(sq)
        self.r_rs.push(slog(h / c) * slog(h / o) + slog(l / c) * slog(l / o))

        sigma_bp = math.sqrt(max(1.5707963267948966 * self.r_bp.total() / (self.VOL_N - 1), 1e-16))
        rv_mid = self.r_rv.total() / self.VOL_N
        sigma_rv = math.sqrt(max(rv_mid, 1e-16))
        sigma_s = math.sqrt(max(self.r_rvS.total() / self.VOL_NS, 1e-16))
        sigma_l = math.sqrt(max(self.r_rvL.total() / self.VOL_NL, 1e-16))
        sigma_rs = math.sqrt(max(self.r_rs.total() / self.VOL_N, 1e-16))
        jump = clip(safediv(sigma_rv**2 - sigma_bp**2, sigma_rv**2, 0.0), 0.0, 1.0)

        a_ew = ewma_alpha(self.EWMA_HL)
        self.ewVar = rv_mid if self.ewVar is None else a_ew * sq + (1 - a_ew) * self.ewVar
        sigma_ew = math.sqrt(max(self.ewVar, 1e-16))

        sigma_f = max(math.exp(self.HAR[0] * slog(sigma_s) + self.HAR[1] * slog(sigma_rv)
                               + self.HAR[2] * slog(sigma_l)), 1e-8)
        if sigma_s <= 1.001e-8 and sigma_rv <= 1.001e-8:
            sigma_f = max(sigma_rs, 1e-6)

        vol_rank = _pr(self.pr_vol, slog(sigma_f))
        comp = 1.0 - _pr(self.pr_comp, safediv(sigma_s, sigma_l, 1.0))
        vol_ratio = safediv(sigma_ew, sigma_f, 1.0)
        range_sigma = clip(safediv(slog(h / l), sigma_f, 1.3), 0.0, 8.0)

        self.has_volume = self.has_volume or v > 0
        dv = (h + l + c) / 3.0 * v if self.has_volume else 1.0
        med = median(self.r_dv.buf) if self.r_dv.buf else float("nan")
        self.r_dv.push(dv)
        part = clip(safediv(dv, max(med if med == med else 1e-12, 1e-12), 1.0), 0.0, 50.0) if self.has_volume else 1.0
        part_rank = _pr(self.pr_part, part) if self.has_volume else 0.5

        d_norm = safediv(ret or 0.0, sigma_bp, 0.0)
        ad = abs(d_norm)
        ladr_abs = safediv(ad, max(part, 1e-12) ** self.PSI, 0.0)
        ladr = ladr_abs if (ret or 0.0) >= 0 else -ladr_abs
        ladr_rank = _pr(self.pr_ladr, ladr_abs)
        absorption = safediv(part, ad + 1e-6, 0.0)
        abs_rank = _pr(self.pr_abs, math.log(1.0 + absorption)) if self.has_volume else 0.5

        clv = 0.0 if h - l <= 0 else (2 * c - h - l) / (h - l)
        k_clv = 2.0 / (self.CLV_HL + 1.0)
        self.clv_ema = clv if self.clv_ema is None else k_clv * clv + (1 - k_clv) * self.clv_ema

        slope_t = self._kalman(self.kal, slog(c), sigma_f * sigma_f)
        if i < 5:
            slope_t = None

        r = ret or 0.0
        self.r_ret.push(r)
        self.r_ret_raw.push(r)
        var1 = variance(self.r_ret.buf) if self.r_ret.full else float("nan")
        vr_terms = []
        for q in (2, 4, 8, 16):
            s = sum(self.r_ret_raw.buf[-q:]) if len(self.r_ret_raw.buf) >= q else 0.0
            self.r_lag[q].push(s)
            vq = variance(self.r_lag[q].buf) if self.r_lag[q].full else float("nan")
            vr = safediv(safediv(vq, q, float("nan")), var1, 1.0)
            vr_terms.append((vr - 1.0) * math.sqrt(self.VR_N / (2.0 * (q - 1))))
        vr_z = sum(vr_terms) / 4.0
        vr_rank = _pr(self.pr_vr, vr_z if vr_z == vr_z else 0.0)

        self.r_eff_n.push(r)
        self.r_eff_d.push(aR)
        eff = clip(abs(safediv(self.r_eff_n.total(), self.r_eff_d.total(), 0.0)), 0.0, 1.0)

        self.closes.push(c)
        cb = self.closes.buf
        c1 = cb[-2] if len(cb) >= 2 else c
        c2 = cb[-3] if len(cb) >= 3 else c
        pat = (0 if c1 < c else 1 if c2 < c else 2) if c2 < c1 else (3 if c2 < c else 4) if c1 < c else 5
        if i >= 2:
            ev = self.pe_ring[self.pe_head]
            if ev >= 0:
                self.pe_counts[ev] = max(self.pe_counts[ev] - 1, 0)
            self.pe_ring[self.pe_head] = pat
            self.pe_counts[pat] += 1
            self.pe_head = (self.pe_head + 1) % self.PE_N
            self.pe_filled = min(self.pe_filled + 1, self.PE_N)
        entropy = 0.5
        if self.pe_filled >= 20:
            s = 0.0
            for k in range(6):
                pk = self.pe_counts[k] / self.pe_filled
                s += -pk * math.log(pk) if pk > 0 else 0.0
            entropy = clip(s / math.log(6.0), 0.0, 1.0)

        self.highs.push(h)
        self.lows.push(l)
        if self.highs.full:
            mid = self.highs.buf[self.PIV]
            if mid == max(self.highs.buf) and self.highs.buf.count(mid) == 1:
                self.pivot_hi, self.pivot_hi_bar = mid, i - self.PIV
            midl = self.lows.buf[self.PIV]
            if midl == min(self.lows.buf) and self.lows.buf.count(midl) == 1:
                self.pivot_lo, self.pivot_lo_bar = midl, i - self.PIV
        range_pos = None
        if self.pivot_hi is not None and self.pivot_lo is not None and self.pivot_hi > self.pivot_lo:
            range_pos = clip((c - self.pivot_lo) / (self.pivot_hi - self.pivot_lo), 0.0, 1.0)

        self._htf(h, l, c)

        self.sweep_age += 1
        self.pend_lo += 1
        self.pend_hi += 1
        heavy = part_rank >= self.SW_PART
        if self.pend_lo <= self.SW_MAX and self.sw_lo is not None and c > self.sw_lo:
            self.sweep_dir, self.sweep_age, self.pend_lo = 1, 0, 999
        if self.pend_hi <= self.SW_MAX and self.sw_hi is not None and c < self.sw_hi:
            self.sweep_dir, self.sweep_age, self.pend_hi = -1, 0, 999
        if heavy and self.pivot_lo is not None and l < self.pivot_lo and \
                safediv(slog(self.pivot_lo / max(l, 1e-12)), sigma_bp, 0.0) >= self.SW_PEN and (clv > 0 or abs_rank >= 0.6):
            self.pend_lo, self.sw_lo = 0, self.pivot_lo
        if heavy and self.pivot_hi is not None and h > self.pivot_hi and \
                safediv(slog(h / self.pivot_hi), sigma_bp, 0.0) >= self.SW_PEN and (clv < 0 or abs_rank >= 0.6):
            self.pend_hi, self.sw_hi = 0, self.pivot_hi

        valid = i >= self.WARMUP
        hazard, dom = 1.0, 3
        if valid:
            hazard, dom = self._regime([vol_rank, vr_rank, ladr_rank, abs_rank,
                                        part_rank, entropy, jump, eff])
        pT, pR, pS, pQ = self.posterior
        commit = max(self.posterior)

        # ---- engines ------------------------------------------------------ #
        sT = slope_t or 0.0
        hT = self.htf_slope or 0.0
        reg_sc = clip(0.5 * (pT * math.tanh(sT / 2.0) + pR * (0.5 - (range_pos if range_pos is not None else 0.5)) * 2.0), -1, 1)
        reg_rel = clip(commit * (1 - hazard), 0, 1)

        def back(n: int) -> float:
            return cb[-1 - n] if len(cb) > n else c
        agree = ((1.0 if c > back(5) else -1.0) + (1.0 if c > back(22) else -1.0)
                 + (1.0 if c > back(66) else -1.0))
        tr_sc = clip(0.45 * math.tanh(sT / 2.0) + 0.25 * agree / 3.0 + 0.30 * math.tanh(hT / 2.0), -1, 1)
        persist = clip(0.5 + (vr_z if vr_z == vr_z else 0.0) / 4.0, 0, 1)
        tf_ok = self.htf_slope is None or slope_t is None or (self.htf_slope >= 0) == (slope_t >= 0)
        tr_rel = clip((0.25 + 0.75 * pT) * (0.15 + 0.85 * persist) * (0.45 + 0.55 * eff)
                      * (1.0 if tf_ok else 0.45) * (1 - 0.6 * pR), 0, 1)

        mo_str = max(0.0, 2 * ladr_rank - 1)
        mo_dir = 1.0 if ladr > 0 else -1.0 if ladr < 0 else 0.0
        mo_sc = clip(0.625 * mo_dir * mo_str + 0.25 * clip(self.clv_ema, -1, 1), -1, 1)
        mo_rel = clip((0.20 + 0.80 * mo_str) * (0.35 + 0.65 * pT) * (1 - 0.5 * pR)
                      * (1 - 0.7 * pQ) * clip(1 - 0.5 * jump, 0.4, 1.0), 0, 1)

        sweep_sc = self.sweep_dir * 0.6 * (1 - self.sweep_age / 5.0) if (self.sweep_age <= 4 and self.sweep_dir) else 0.0
        edge_d = 0.0 if range_pos is None else abs(range_pos - 0.5) * 2
        absorbing = self.has_volume and abs_rank > 0.65 and edge_d > 0.5
        abs_sc = (-clip((abs_rank - 0.65) / 0.35 * edge_d, 0, 1)
                  * (1.0 if (range_pos or 0.5) > 0.5 else -1.0)) if absorbing else 0.0
        liq_sc = 0.7 * sweep_sc + 0.3 * abs_sc if (sweep_sc and abs_sc) else sweep_sc + abs_sc
        liq_rel = (clip((0.30 + 0.70 * abs(liq_sc)) * (0.40 + 0.60 * (pR + 0.6 * pS)) * (1 - 0.35 * pT), 0, 1)
                   if (sweep_sc or absorbing) else (0.35 if self.has_volume else 0.10))

        vfav = 1.0 if 0.25 <= vol_rank <= 0.80 else clip(
            1 - 2 * ((0.25 - vol_rank) / 0.55 if vol_rank < 0.25 else (vol_rank - 0.80) / 0.55), -1, 1)
        vol_sc = clip(0.35 * math.tanh(d_norm / 2.0), -0.4, 0.4) if (vol_ratio > 1.15 and abs(d_norm) > 0.5) else 0.0
        vol_rel = clip((0.35 + 0.45 * clip((vfav + 1) / 2, 0, 1)) * (1 - 0.4 * pS), 0, 1)

        st_pos = 0.0 if range_pos is None else 2 * range_pos - 1
        wT, wR = pT + 1e-9, pR + 1e-9
        st_sc0 = (wT * st_pos + wR * -st_pos) / (wT + wR)
        st_sc = st_sc0 if self.htf_rangepos is None else clip(
            0.7 * st_sc0 + 0.3 * (wT - wR) * (2 * self.htf_rangepos - 1) / (wT + wR), -1, 1)
        st_dec = safediv(abs(pT - pR), pT + pR, 0.0)
        age = max(i - self.pivot_hi_bar, i - self.pivot_lo_bar)
        st_rel = 0.0 if range_pos is None else clip(
            (0.25 + 0.75 * abs(st_pos)) * (0.35 + 0.65 * st_dec) * clip(1 - (age - 20) / 120.0, 0.25, 1.0), 0, 1)

        scores = [reg_sc, tr_sc, mo_sc, liq_sc, vol_sc, st_sc, 0.0]
        rels = [reg_rel, tr_rel, mo_rel, liq_rel, vol_rel, st_rel, 0.9]

        for name, val in (("sigma_f", sigma_f), ("vol_rank", vol_rank), ("ladr", ladr),
                          ("vr_z", vr_z), ("entropy", entropy)):
            if val != val or val in (float("inf"), float("-inf")):
                self.nonfinite[name] += 1

        # ---- CALOP -------------------------------------------------------- #
        log_odds = ebe_total = 0.0
        if valid:
            for a in range(self.NENG):
                da = math.sqrt(max(rels[a], 0.0))
                for b in range(self.NENG):
                    db = math.sqrt(max(rels[b], 0.0))
                    p = self.PREC[a * self.NENG + b]
                    x = clip(scores[b], -self.CLAMP, self.CLAMP)
                    log_odds += da * p * db * math.log((1 + x) / (1 - x))
                    ebe_total += da * p * db
        direction = 1 if log_odds > 0 else -1 if log_odds < 0 else 0

        ebe_al = 0.0
        if valid and direction:
            for a in range(self.NENG):
                if rels[a] > 0 and abs(scores[a]) * rels[a] > self.ALIGN and (1 if scores[a] > 0 else -1) == direction:
                    for b in range(self.NENG):
                        if rels[b] > 0 and abs(scores[b]) * rels[b] > self.ALIGN and (1 if scores[b] > 0 else -1) == direction:
                            ebe_al += math.sqrt(rels[a]) * self.PREC[a * self.NENG + b] * math.sqrt(rels[b])

        var_l = self.FUSE_SD**2 * max(ebe_total, 1e-9)
        kap = 1.0 / math.sqrt(1 + 0.39269908169872414 * self.FUSE_T**2 * var_l)
        p_raw = logistic((self.FUSE_T * log_odds + self.FUSE_B) * kap)
        p_up = interp(p_raw, self.ISO_X, self.ISO_Y) if (self.ISO_ON and len(self.ISO_X) >= 2) else p_raw
        p_succ = p_up if direction >= 0 else 1 - p_up

        # ---- Mode A gate --------------------------------------------------- #
        setup = 1 if (self.sweep_age <= 4 and self.sweep_dir) else 2 if (comp > 0.7 and vol_ratio > 1.1) else 3 if pR > pT else 0
        bucket = min(sum(1 for e in self.EDGES if abs(log_odds) >= e), self.NBUCKET - 1)
        cell = (dom * self.NSETUP + setup) * self.NBUCKET + bucket
        ok = 0 <= cell < len(self.MU)
        lcb = (self.MU[cell] + student_t_ppf(self.EB_Q, max(self.DOF[cell], 2.1)) * max(self.SCALE[cell], 1e-6)) if ok else -1.0
        kstr = 1 + 1.5 * pS
        hs = min(safediv(0.5 * self.SFALL, sigma_f, 1e9), 1e9)
        if hs > self.MAXHS:
            hs = 1e9
        rt = 2 * (kstr * hs + kstr * self.ETA * self.Q ** self.PSI + kstr * self.PHI * min(range_sigma, 5.0)
                  + safediv(self.FEE * 1e-4, sigma_f, 0.0))
        ev_net = lcb - rt

        # ---- Mode B gate --------------------------------------------------- #
        s_min = {1: 0.25, 2: 0.35, 3: 0.45, 4: 0.55, 5: 0.65}[self.strict]
        q_max = 0.70 if self.strict <= 2 else 0.60 if self.strict == 3 else 0.50
        h_max = 0.55 if self.strict <= 2 else 0.45 if self.strict == 3 else 0.40
        b_brd = 0.60 if self.strict <= 2 else 0.90 if self.strict == 3 else 1.20
        wsum = sum(rels[:6]) + 1e-9
        b_score = clip(sum(rels[k] * scores[k] for k in range(6)) / wsum * 2.0, -1, 1)
        b_dir = 1 if b_score > 0 else -1 if b_score < 0 else 0
        b_conf = 50 + 45 * abs(b_score) * commit

        # ---- state machine and gate --------------------------------------- #
        pos_before = self.pos
        if self.pos:
            self.bars_held += 1
        e_stop = e_tgt = False
        if self.pos == 1 and self.stop is not None:
            if l <= self.stop:
                e_stop = True
            elif h >= self.target:
                e_tgt = True
        if self.pos == -1 and self.stop is not None:
            if h >= self.stop:
                e_stop = True
            elif l <= self.target:
                e_tgt = True
        prop = direction if self.mode_a else b_dir
        t_exit = bool(self.pos) and not e_stop and not e_tgt and self.bars_held >= self.MAX_HOLD
        r_exit = (bool(self.pos) and not e_stop and not e_tgt and not t_exit
                  and self.bars_held >= self.MIN_HOLD and prop == -self.pos
                  and (p_succ >= self.PEXIT if self.mode_a else abs(b_score) >= s_min * 0.6))
        if e_stop or e_tgt or t_exit or r_exit:
            self.signals.append((i, "SELL" if pos_before == 1 else "COVER", c,
                                 self.stop or 0.0, self.target or 0.0, 0.0))
            self.pos, self.bars_held, self.last_exit = 0, 0, i
            self.stop = self.target = None

        while self.entry_bars and i - self.entry_bars[0] > 100:
            self.entry_bars.pop(0)
        cap = self.RATE if self.mode_a else self.max_per_100

        conf = p_succ * 100 if self.mode_a else b_conf
        checks = [
            ("warm-up", valid),
            ("no direction", prop != 0),
            ("direction disabled", (self.allow_long if prop > 0 else self.allow_short if prop < 0 else False)),
            ("quiet regime", (dom != 3) if self.mode_a else (pQ < q_max)),
            ("regime hazard", hazard <= (self.HAZ_MAX if self.mode_a else h_max)),
            ("edge/score gate", (ev_net > self.EVLCB) if self.mode_a else (abs(b_score) >= s_min)),
            ("confidence floor", (p_succ >= self.PMIN) if self.mode_a else (b_conf >= 55.0)),
            ("evidence breadth", ebe_al >= (self.EBEMIN if self.mode_a else b_brd)),
            ("htf disagrees", (not self.use_htf or self.htf_slope is None or prop == 0
                               or (self.htf_slope > -0.5 if prop > 0 else self.htf_slope < 0.5))),
            ("session", True),
            ("rate/cooldown", len(self.entry_bars) < cap and i - self.last_exit >= max(self.cooldown, 1)),
            ("in position", self.pos == 0),
        ]
        first_fail = next((n for n, okk in checks if not okk), None)
        if first_fail:
            self.vetoes[first_fail] += 1
        for n, okk in checks:
            if not okk:
                self.gate_fail[n] += 1

        if first_fail is None:
            stop_sig = clip(self.STOP_S * (1 + 0.6 * pS if pS > 0.2 else 1.0)
                            * (1 + 0.4 * (jump - 0.4) / 0.6 if jump > 0.4 else 1.0), 0.4, 8.0)
            sd = c * (1 - math.exp(-stop_sig * sigma_f))
            td = sd * (self.TGT_S / self.STOP_S)
            self.pos = prop
            self.entry = c
            self.stop = c - sd if prop > 0 else c + sd
            self.target = c + td if prop > 0 else c - td
            self.bars_held = 0
            self.entry_bars.append(i)
            self.signals.append((i, "BUY" if prop > 0 else "SHORT", c, self.stop, self.target, conf))

        self.prev_close = c

    # -- helpers ------------------------------------------------------------ #
    def _kalman(self, st: list, obs: float, s2: float) -> float | None:
        lev, slp, p00, p01, p11 = st
        if lev is None:
            st[0], st[2], st[4] = obs, 100.0 * s2, s2
            return None
        lev = lev + slp
        n00 = p00 + 2 * p01 + p11 + self.KAL * s2
        n01 = p01 + p11
        n11 = p11 + self.KAL_S * s2
        iv = n00 + s2
        k0, k1 = safediv(n00, iv, 0.0), safediv(n01, iv, 0.0)
        res = obs - lev
        st[0] = lev + k0 * res
        st[1] = slp + k1 * res
        st[2] = max(n00 - k0 * n00, 1e-18)
        st[3] = n01 - k0 * n01
        st[4] = max(n11 - k1 * n01, 1e-18)
        return safediv(st[1], math.sqrt(st[4]), None)

    def _htf(self, h: float, l: float, c: float) -> None:
        self.htf_h = h if self.htf_h is None else max(self.htf_h, h)
        self.htf_l = l if self.htf_l is None else min(self.htf_l, l)
        self.htf_n += 1
        if self.htf_n < self.HTF:
            return
        hr = None if (self.htf_prev is None or self.htf_prev <= 0) else math.log(c / self.htf_prev)
        if hr is not None:
            self.htf_ewvar = hr * hr if self.htf_ewvar is None else 0.1 * hr * hr + 0.9 * self.htf_ewvar
        hs = math.sqrt(max(self.htf_ewvar if self.htf_ewvar is not None else 1e-8, 1e-10))
        s = self._kalman(self.hkal, slog(c), hs * hs)
        if s is not None:
            self.htf_slope = s
        self.htf_highs.append(self.htf_h)
        self.htf_lows.append(self.htf_l)
        if len(self.htf_highs) > 20:
            self.htf_highs.pop(0)
            self.htf_lows.pop(0)
        if len(self.htf_highs) >= 5:
            hh, ll = max(self.htf_highs), min(self.htf_lows)
            self.htf_rangepos = None if hh <= ll else clip((c - ll) / (hh - ll), 0.0, 1.0)
        self.htf_prev, self.htf_n, self.htf_h, self.htf_l = c, 0, None, None

    def _regime(self, feats: list[float]) -> tuple[float, int]:
        usable = [f for f in feats if f == f]
        if len(usable) < self.MIN_FEAT:
            return 1.0, 3
        loglik = []
        for k in range(4):
            acc = 0.0
            for j, x in enumerate(feats):
                if x == x:
                    acc += beta_log_pdf_kernel(x, self.DA[k * 8 + j], self.DB[k * 8 + j]) - self.DLN[k * 8 + j]
            loglik.append(self.TEMPER * acc)
        new = []
        for k in range(4):
            vs = [self.log_alpha[j] + (self.LOG_STAY if j == k else self.LOG_SWITCH) for j in range(4)]
            mx = max(vs)
            new.append(mx + math.log(max(sum(math.exp(x - mx) for x in vs), 1e-300)) + loglik[k])
        mx2 = max(new)
        ln = mx2 + math.log(max(sum(math.exp(x - mx2) for x in new), 1e-300))
        a_ref = ewma_alpha(self.HAZ_HL)
        drift, mp, dom = 0.0, 0.0, 3
        for k in range(4):
            self.log_alpha[k] = new[k] - ln
            pk = math.exp(self.log_alpha[k])
            self.posterior[k] = pk
            self.ref_post[k] = (1 - a_ref) * self.ref_post[k] + a_ref * pk
            drift += abs(pk - self.ref_post[k])
            if pk > mp:
                mp, dom = pk, k
        return max(0.5 * drift, 1.0 - mp), dom


def _flat(mat) -> list[float]:
    return [float(v) for row in mat for v in row] if mat and isinstance(mat[0], list) else [float(v) for v in mat]


def _pr(ring: Ring, x: float) -> float:
    v = percentrank(ring.buf, x)
    ring.push(x)
    return (50.0 if v != v else v) / 100.0


# --------------------------------------------------------------------------- #
UNIVERSE = ("SPY", "QQQ", "IWM", "GLD", "TLT", "EEM")


#: The vendor caps how far back each interval reaches; asking for 15y of
#: hourly bars returns an error, not a short answer.
MAX_RANGE = {"1d": "15y", "1wk": "15y", "1h": "2y", "60m": "2y",
             "30m": "60d", "15m": "60d", "5m": "60d", "1m": "7d"}


def fetch(symbol: str, rng: str = "15y", interval: str = "1d"):
    rng = MAX_RANGE.get(interval, rng)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?range={rng}&interval={interval}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    q = payload["chart"]["result"][0]["indicators"]["quote"][0]
    out = []
    for i in range(len(payload["chart"]["result"][0]["timestamp"])):
        o, h, l, c, v = (q[k][i] for k in ("open", "high", "low", "close", "volume"))
        if None in (o, h, l, c, v) or min(o, h, l, c) <= 0:
            continue
        out.append((float(o), float(max(h, o, c)), float(min(l, o, c)), float(c), float(v)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", action="store_true", help="fetch ETFs instead of generating bars")
    ap.add_argument("--interval", default="1d")
    args = ap.parse_args()

    model = json.loads((ROOT / "pine" / "frozen_model.json").read_text())
    print(f"TI-A Pine-semantics simulation   model {model['config_manifest_hash']}"
          f"   placeholder={model['is_placeholder']}")

    data: dict[str, list] = {}
    if args.real:
        for s in UNIVERSE:
            try:
                data[s] = fetch(s, interval=args.interval)
            except Exception as exc:  # noqa: BLE001
                print(f"  {s}: fetch failed ({exc})")
    if not data:
        from tia.synthetic import generate_with_regimes
        for seed in (11, 12, 13):
            bars, _ = generate_with_regimes(4000, seed=seed)
            data[f"synthetic-{seed}"] = [(b.open, b.high, b.low, b.close, b.volume) for b in bars]

    print("\nMODE B — signals by strictness (what a chart will actually show)")
    print(f"  {'symbol':<14}{'bars':>7}" + "".join(f"{'s=' + str(s):>8}" for s in (1, 2, 3, 4, 5)))
    totals = {s: 0 for s in (1, 2, 3, 4, 5)}
    for sym, bars in data.items():
        row = ""
        for s in (1, 2, 3, 4, 5):
            sim = PineSim(model, mode_a=False, strict=s)
            for b in bars:
                sim.on_bar(*b)
            n = sum(1 for x in sim.signals if x[1] in ("BUY", "SHORT"))
            totals[s] += n
            row += f"{n:>8}"
        print(f"  {sym:<14}{len(bars):>7}{row}")
    print(f"  {'TOTAL':<14}{'':>7}" + "".join(f"{totals[s]:>8}" for s in (1, 2, 3, 4, 5)))

    print("\nMODE A — the frozen model through its own unrelaxed gate")
    a_total = 0
    veto = Counter()
    for sym, bars in data.items():
        sim = PineSim(model, mode_a=True)
        for b in bars:
            sim.on_bar(*b)
        n = sum(1 for x in sim.signals if x[1] in ("BUY", "SHORT"))
        a_total += n
        veto += sim.vetoes
        print(f"  {sym:<14}{len(bars):>7} bars -> {n} entries")
    print(f"  binding constraint, most common first:")
    for name, cnt in veto.most_common(6):
        print(f"    {name:<22}{cnt:>8} bars")

    print("\nRUNTIME SAFETY (a non-finite value is how a Pine script paints nothing)")
    sim = PineSim(model, mode_a=False, strict=3)
    sym0 = next(iter(data))
    for b in data[sym0]:
        sim.on_bar(*b)
    if sim.nonfinite:
        print(f"  NON-FINITE VALUES on {sym0}: {dict(sim.nonfinite)}")
        return 1
    print(f"  {sym0}: no non-finite feature values across {len(data[sym0])} bars")

    print("\nEXIT ACCOUNTING (every entry must be matched by an exit or an open position)")
    ent = sum(1 for x in sim.signals if x[1] in ("BUY", "SHORT"))
    ex = sum(1 for x in sim.signals if x[1] in ("SELL", "COVER"))
    print(f"  {sym0}: {ent} entries, {ex} exits, {'1 open' if sim.pos else 'flat'} at the end")
    if ent - ex not in (0, 1):
        print("  MISMATCH: entries and exits do not reconcile")
        return 1

    print("\nThis measures signal FREQUENCY and runtime safety only. It says nothing")
    print("about whether the signals are profitable, and no such claim is made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
