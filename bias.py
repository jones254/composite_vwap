"""
Direction-bias engine for the Gold Scalper.

Three independent pillars, each mapped to [-100, +100]:

    VWAP       where price sits vs. volume-weighted fair value
               intraday bars -> session ("daily") VWAP, resets every day
               daily bars    -> rolling-window VWAP (a daily bar cannot reset)
    STRUCTURE  market structure (swing HH/HL vs LH/LL + BOS / CHoCH), read on
               the base timeframe AND on higher timeframes (multi-timeframe)
    TREND      SuperTrend direction + DI spread, scaled by ADX so that trend
               signals fade to zero in a range

The pillars are blended according to `BiasConfig.mode`:

    "hybrid"     VWAP + STRUCTURE + TREND        (default)
    "structure"  VWAP + STRUCTURE                (option A)
    "trend"      VWAP + TREND (SuperTrend / ADX) (option B)

The blend is then multiplied by a CONFLUENCE factor (0.5 .. 1.0): when the
pillars disagree the bias is shrunk, when they agree it is kept in full.

Every calculation is causal (no look-ahead):
  * swing pivots are only "known" `swing_len` bars after they form
  * higher-timeframe values come from the last COMPLETED higher-TF bar
  * session VWAP only uses bars up to the current one

Data robustness (this repo's data has quirks):
  * Missing High/Low  -> Close is used  (signal_backtest used to pass Close-only)
  * Missing / zero Volume (DXY, VIX, EUR/USD) -> equal-weight VWAP (TWAP proxy)
  * Forward-filled rows from data._align (exact duplicate of the previous bar)
    are detected and given zero weight
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MODES = ("hybrid", "structure", "trend")

_PILLAR_WEIGHTS = {
    "hybrid":    {"vwap": 0.35, "structure": 0.40, "trend": 0.25},
    "structure": {"vwap": 0.40, "structure": 0.60, "trend": 0.00},
    "trend":     {"vwap": 0.40, "structure": 0.00, "trend": 0.60},
}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class BiasConfig:
    enabled: bool = True
    mode: str = "hybrid"            # "hybrid" | "structure" | "trend"

    # How the bias enters the 7-market composite
    blend: float = 0.50             # share of each market's asset score that comes from bias (0..1)
    gold_gate: float = 0.25         # damp composite when Gold's own bias opposes it (0 = off)

    # VWAP
    vwap_window: int = 20                       # rolling window for daily bars
    vwap_session_shift_hours: float = 6.0       # 6h => Globex day (18:00 ET) starts a new session

    # Market structure
    swing_len: int = 3                          # pivot strength, higher-TF + daily bars
    swing_len_intraday: int = 5                 # pivot strength, intraday base bars
    htf_count: int = 2                          # how many higher timeframes to add
    mtf_weights: Tuple[float, ...] = (0.25, 0.35, 0.40)   # base, htf1, htf2

    # Trend / volatility
    st_period: int = 10
    st_mult: float = 3.0
    adx_len: int = 14
    adx_trending: float = 25.0
    adx_ranging: float = 20.0

    # Labelling
    neutral_band: float = 20.0                  # |bias| below this => Neutral

    def weights(self) -> Dict[str, float]:
        return dict(_PILLAR_WEIGHTS.get(self.mode, _PILLAR_WEIGHTS["hybrid"]))


@dataclass
class BiasResult:
    frame: pd.DataFrame
    htf: List[str]               # higher-timeframe rules actually used
    intraday: bool
    vwap_is_proxy: bool          # True => no usable volume, equal-weight VWAP
    mode: str


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to float OHLCV on a clean, sorted, unique DatetimeIndex."""
    c = pd.to_numeric(df["Close"], errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["Close"] = c
    for col in ("Open", "High", "Low"):
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else c
    out["Volume"] = (pd.to_numeric(df["Volume"], errors="coerce")
                     if "Volume" in df.columns else np.nan)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Close"])
    for col in ("Open", "High", "Low"):
        out[col] = out[col].fillna(out["Close"])
    out["High"] = out[["High", "Open", "Close"]].max(axis=1)
    out["Low"] = out[["Low", "Open", "Close"]].min(axis=1)
    return out


def _bar_minutes(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 3:
        return 1440.0
    d = pd.Series(idx).diff().dropna().dt.total_seconds() / 60.0
    m = float(d.median())
    return m if m > 0 else 1440.0


def _atr_safe(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    """Wilder ATR with no NaN warm-up (min_periods=1) so early bars stay usable."""
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=1).mean()


def _resample_ohlc(p: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if rule == "1W":
        r = p.resample("W-MON", label="left", closed="left")
    else:
        r = p.resample(rule, label="left", closed="left")
    return r.agg(agg).dropna(subset=["Close"])


def _htf_rules(bar_min: float, count: int) -> List[str]:
    ladder = [("1h", 60), ("4h", 240), ("1D", 1440), ("1W", 10080)]
    return [r for r, m in ladder if m >= 4 * bar_min][:max(0, count)]


# -----------------------------------------------------------------------------
# PILLAR 1 - VWAP
# -----------------------------------------------------------------------------
def _vwap_block(p: pd.DataFrame, cfg: BiasConfig, bar_min: float,
                intraday: bool, atr_s: pd.Series) -> Tuple[pd.DataFrame, bool]:
    o, h, l, c = p["Open"], p["High"], p["Low"], p["Close"]
    tp = (h + l + c) / 3.0

    # forward-filled rows are exact duplicates of the previous bar -> no weight
    stale = (o.eq(o.shift(1)) & h.eq(h.shift(1)) & l.eq(l.shift(1)) & c.eq(c.shift(1)))

    vol = p["Volume"]
    has_vol = bool((vol.fillna(0) > 0).mean() > 0.5)
    w = vol.fillna(0).clip(lower=0) if has_vol else pd.Series(1.0, index=p.index)
    w = w.where(~stale, 0.0)

    pv, p2 = tp * w, tp * tp * w

    if intraday:
        shift = pd.Timedelta(hours=cfg.vwap_session_shift_hours)
        key = (p.index + shift).normalize().to_numpy()
        cw = w.groupby(key).cumsum()
        cpv = pv.groupby(key).cumsum()
        cp2 = p2.groupby(key).cumsum()
        n_in_sess = pd.Series(1.0, index=p.index).groupby(key).cumsum()
    else:
        win = max(5, int(cfg.vwap_window))
        cw = w.rolling(win, min_periods=3).sum()
        cpv = pv.rolling(win, min_periods=3).sum()
        cp2 = p2.rolling(win, min_periods=3).sum()
        n_in_sess = pd.Series(float(win), index=p.index)
        key = None

    cw_ok = cw.where(cw > 0)
    vwap = cpv / cw_ok
    sd = np.sqrt((cp2 / cw_ok - vwap ** 2).clip(lower=0.0)).fillna(0.0)
    band = np.maximum(sd, 0.35 * atr_s)                      # floor: avoid z blow-ups
    band = band.where(band > 0, np.nan)

    z_cur = ((c - vwap) / band).fillna(0.0)
    if intraday:
        # early in the session VWAP ~ price: lean on the previous session's VWAP
        sess_last = vwap.groupby(key).last()
        prev_by_sess = sess_last.shift(1)
        prev_v = pd.Series(prev_by_sess.reindex(key).to_numpy(), index=p.index)
        z_prev = ((c - prev_v) / band).fillna(0.0)
        ramp_bars = max(2.0, round(120.0 / bar_min))
        ramp = (n_in_sess / ramp_bars).clip(0.0, 1.0)
        z = ramp * z_cur + (1.0 - ramp) * z_prev
        k_pers, m_slope = 12, 6
        dv = vwap.groupby(key).diff(m_slope)
    else:
        ramp = pd.Series(1.0, index=p.index)
        z = z_cur
        k_pers, m_slope = 10, 5
        dv = vwap.diff(m_slope)

    pos = 100.0 * np.tanh(z / 1.5)
    side = np.sign(c - vwap)
    pers = (side.rolling(k_pers, min_periods=3).mean() * 100.0).fillna(0.0)
    slope = (100.0 * np.tanh(1.5 * (dv / atr_s.replace(0.0, np.nan)).fillna(0.0))) * ramp

    score = (0.55 * pos + 0.30 * pers + 0.15 * slope).clip(-100.0, 100.0)

    out = pd.DataFrame({
        "vwap": vwap, "vwap_hi": vwap + band, "vwap_lo": vwap - band,
        "vwap_z": z, "s_vwap": score,
    }, index=p.index)
    return out, (not has_vol)


# -----------------------------------------------------------------------------
# PILLAR 2 - market structure (single timeframe + multi-timeframe)
# -----------------------------------------------------------------------------
def _last_two(piv: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Latest confirmed pivot value and the one before it, forward-filled."""
    ev = piv.dropna()
    last = ev.reindex(piv.index).ffill()
    prev = ev.shift(1).reindex(piv.index).ffill()
    return last, prev


def structure_tf(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.DataFrame:
    """
    Swing structure on one timeframe.

    pivot high at i : high[i] is the max of high[i-n .. i+n]   (known at i+n)
    pivot low  at i : low[i]  is the min of low[i-n .. i+n]    (known at i+n)

    dir : +1 / -1 / 0   state machine - a close above the last pivot high flips
          to +1, a close below the last pivot low flips to -1.  A flip against
          the previous direction is a CHoCH, a re-break in the same direction
          is a BOS.
    seq : (HH + HL - LH - LL) / 2   in [-1, +1]
    """
    w = 2 * n + 1
    wmax = h.rolling(w, center=True, min_periods=w).max()
    wmin = l.rolling(w, center=True, min_periods=w).min()
    ph = h.where(h == wmax).shift(n)          # shift => only known n bars later
    pl = l.where(l == wmin).shift(n)

    H1, H0 = _last_two(ph)
    L1, L0 = _last_two(pl)

    seq = ((H1 > H0).astype(float) + (L1 > L0).astype(float)
           - (H1 < H0).astype(float) - (L1 < L0).astype(float)) / 2.0

    up = (c > H1)
    dn = (c < L1)
    ev = pd.Series(np.where(up & ~dn, 1.0, np.where(dn & ~up, -1.0, 0.0)), index=c.index)
    direction = ev.replace(0.0, np.nan).ffill().fillna(0.0)

    cross = (ev != 0.0) & (ev != ev.shift(1).fillna(0.0))
    prior_dir = direction.shift(1).fillna(0.0)
    choch = cross & (prior_dir != 0.0) & (prior_dir != ev)
    bos = cross & ~choch
    evt = pd.Series(0.0, index=c.index)
    evt = evt.mask(bos, ev).mask(choch, 2.0 * ev)      # +/-1 BOS, +/-2 CHoCH

    score = 100.0 * (0.6 * direction + 0.4 * seq)
    return pd.DataFrame({
        "score": score, "dir": direction, "seq": seq,
        "H1": H1, "L1": L1, "evt": evt,
    }, index=c.index)


def _structure_block(p: pd.DataFrame, cfg: BiasConfig, bar_min: float,
                     intraday: bool) -> Tuple[pd.DataFrame, List[str]]:
    n_base = cfg.swing_len_intraday if intraday else cfg.swing_len
    base = structure_tf(p["High"], p["Low"], p["Close"], n_base)

    comps = [base["score"]]
    dirs = {"struct_base": base["dir"]}
    used: List[str] = []
    for i, rule in enumerate(_htf_rules(bar_min, cfg.htf_count), start=1):
        try:
            hp = _resample_ohlc(p, rule)
        except Exception:
            continue
        if len(hp) < 2 * cfg.swing_len + 6:
            continue
        st = structure_tf(hp["High"], hp["Low"], hp["Close"], cfg.swing_len)
        # completed higher-TF bars only: shift(1) then carry forward onto base bars
        sc = st["score"].shift(1).reindex(p.index, method="ffill").fillna(0.0)
        dr = st["dir"].shift(1).reindex(p.index, method="ffill").fillna(0.0)
        comps.append(sc)
        dirs[f"struct_htf{i}"] = dr
        used.append(rule)

    wts = np.array(list(cfg.mtf_weights)[:len(comps)], dtype=float)
    if len(wts) < len(comps) or wts.sum() <= 0:
        wts = np.ones(len(comps))
    wts = wts / wts.sum()
    mtf = sum(wt * s for wt, s in zip(wts, comps))

    out = pd.DataFrame({"s_struct": mtf.clip(-100.0, 100.0)}, index=p.index)
    for k, v in dirs.items():
        out[k] = v
    out["struct_evt"] = base["evt"]
    out["piv_hi"] = base["H1"]
    out["piv_lo"] = base["L1"]
    return out, used


# -----------------------------------------------------------------------------
# PILLAR 3 - trend / volatility  (SuperTrend + ADX)
# -----------------------------------------------------------------------------
def supertrend(h: pd.Series, l: pd.Series, c: pd.Series,
               period: int = 10, mult: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """Classic SuperTrend. Returns (line, dir) with dir in {+1, -1}, 0 during warm-up."""
    a = _atr_safe(h, l, c, period)
    hl2 = ((h + l) / 2.0).to_numpy(dtype=float)
    av = a.to_numpy(dtype=float)
    cv = c.to_numpy(dtype=float)
    ub = hl2 + mult * av
    lb = hl2 - mult * av
    n = len(cv)
    fub, flb = ub.copy(), lb.copy()
    d = np.zeros(n)
    line = np.full(n, np.nan)
    cur = 1.0
    for i in range(1, n):
        if not (np.isfinite(ub[i]) and np.isfinite(cv[i])):
            fub[i], flb[i] = fub[i - 1], flb[i - 1]
            d[i] = d[i - 1]
            continue
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or cv[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or cv[i - 1] < flb[i - 1]) else flb[i - 1]
        if cur == 1.0 and cv[i] < flb[i]:
            cur = -1.0
        elif cur == -1.0 and cv[i] > fub[i]:
            cur = 1.0
        if i >= period:
            d[i] = cur
            line[i] = flb[i] if cur == 1.0 else fub[i]
    return (pd.Series(line, index=c.index), pd.Series(d, index=c.index))


def adx_di(h: pd.Series, l: pd.Series, c: pd.Series,
           n: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder ADX, +DI, -DI."""
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    kw = dict(alpha=1.0 / n, adjust=False, min_periods=n)
    atr_w = tr.ewm(**kw).mean()
    pdi = 100.0 * plus_dm.ewm(**kw).mean() / atr_w
    mdi = 100.0 * minus_dm.ewm(**kw).mean() / atr_w
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    adx = dx.ewm(**kw).mean()
    return adx, pdi, mdi


def _trend_block(p: pd.DataFrame, cfg: BiasConfig, atr_s: pd.Series,
                 need_supertrend: bool) -> pd.DataFrame:
    h, l, c = p["High"], p["Low"], p["Close"]
    adx, pdi, mdi = adx_di(h, l, c, cfg.adx_len)
    adx_f = ((adx - 15.0) / 20.0).clip(0.0, 1.0).fillna(0.0)

    if need_supertrend:
        st_line, st_dir = supertrend(h, l, c, cfg.st_period, cfg.st_mult)
    else:
        st_line = pd.Series(np.nan, index=p.index)
        st_dir = pd.Series(0.0, index=p.index)

    di_part = 100.0 * np.tanh(((pdi - mdi) / 15.0).fillna(0.0))
    raw = 0.6 * (st_dir * 100.0) + 0.4 * di_part
    score = (raw * (0.4 + 0.6 * adx_f)).where(adx.notna(), 0.0).clip(-100.0, 100.0)

    ratio = atr_s / atr_s.rolling(100, min_periods=20).median()
    regime = np.select(
        [adx >= cfg.adx_trending, adx < cfg.adx_ranging],
        ["Trending", "Ranging"], default="Transition")
    regime = pd.Series(np.where(adx.notna(), regime, "Warm-up"), index=p.index)

    return pd.DataFrame({
        "s_trend": score, "supertrend": st_line, "st_dir": st_dir,
        "adx": adx, "plus_di": pdi, "minus_di": mdi,
        "adx_f": adx_f, "vol_ratio": ratio, "regime": regime,
    }, index=p.index)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def _empty(index: pd.DatetimeIndex, cfg: BiasConfig) -> BiasResult:
    cols = ["vwap", "vwap_hi", "vwap_lo", "vwap_z", "s_vwap", "s_struct", "s_trend",
            "bias", "confidence", "consensus", "adx", "st_dir", "supertrend"]
    f = pd.DataFrame(0.0, index=index, columns=cols)
    f["label"] = "Neutral"
    f["regime"] = "Warm-up"
    return BiasResult(f, [], False, True, cfg.mode)


def compute_bias(df: pd.DataFrame, cfg: Optional[BiasConfig] = None) -> BiasResult:
    """Direction bias for every bar of `df` (needs a Close column; OHLCV optional)."""
    cfg = cfg or BiasConfig()
    if df is None or len(df) == 0 or "Close" not in df.columns:
        return _empty(pd.DatetimeIndex([]), cfg)
    p = _prepare(df)
    if len(p) < 30:
        return _empty(p.index, cfg)

    w = cfg.weights()
    bar_min = _bar_minutes(p.index)
    intraday = bar_min < 600.0
    atr_s = _atr_safe(p["High"], p["Low"], p["Close"], 14)

    vw, proxy = _vwap_block(p, cfg, bar_min, intraday, atr_s)
    tr = _trend_block(p, cfg, atr_s, need_supertrend=w["trend"] > 0)
    if w["structure"] > 0:
        st, htf = _structure_block(p, cfg, bar_min, intraday)
    else:
        # structure pillar switched off (mode="trend"): keep the same columns, neutral values
        st = pd.DataFrame({"s_struct": 0.0, "struct_base": 0.0, "struct_evt": 0.0,
                           "piv_hi": np.nan, "piv_lo": np.nan}, index=p.index)
        htf = []

    pillars = {"vwap": vw["s_vwap"].to_numpy(),
               "structure": st["s_struct"].to_numpy(),
               "trend": tr["s_trend"].to_numpy()}
    wsum = sum(v for v in w.values() if v > 0) or 1.0
    raw = sum(w[k] * pillars[k] for k in pillars if w[k] > 0) / wsum
    sgn = sum(w[k] * np.where(np.abs(pillars[k]) >= 10.0, np.sign(pillars[k]), 0.0)
              for k in pillars if w[k] > 0) / wsum
    consensus = np.abs(sgn)
    bias = np.clip(raw * (0.5 + 0.5 * consensus), -100.0, 100.0)

    conf = (45.0 * np.minimum(np.abs(bias) / 60.0, 1.0)
            + 35.0 * consensus
            + 20.0 * tr["adx_f"].to_numpy())
    conf = np.where(tr["vol_ratio"].to_numpy() > 2.0, conf * 0.9, conf)

    nb = cfg.neutral_band
    label = np.select(
        [bias > 60, bias > nb, bias < -60, bias < -nb],
        ["Strong Bullish", "Bullish", "Strong Bearish", "Bearish"], default="Neutral")

    frame = pd.concat([vw, st, tr], axis=1)
    frame["consensus"] = consensus
    frame["bias"] = bias
    frame["confidence"] = np.clip(conf, 0.0, 100.0)
    frame["label"] = label
    return BiasResult(frame, htf, intraday, proxy, cfg.mode)


def bias_series(df: pd.DataFrame, cfg: Optional[BiasConfig] = None) -> pd.Series:
    return compute_bias(df, cfg).frame["bias"]


# -----------------------------------------------------------------------------
# Latest-bar summary (for the UI / Telegram)
# -----------------------------------------------------------------------------
_EVT = {1: "BOS ▲", -1: "BOS ▼", 2: "CHoCH ▲ (bullish reversal)", -2: "CHoCH ▼ (bearish reversal)"}


def latest_summary(res: BiasResult) -> Dict[str, object]:
    f = res.frame
    if len(f) == 0:
        return {}
    last = f.iloc[-1]
    evt = f["struct_evt"] if "struct_evt" in f.columns else pd.Series(0.0, index=f.index)
    nz = evt[evt != 0]
    last_evt, bars_ago = "none", None
    if len(nz):
        last_evt = _EVT.get(int(nz.iloc[-1]), "none")
        bars_ago = int(len(f) - 1 - f.index.get_loc(nz.index[-1]))
    return {
        "bias": float(last["bias"]), "label": str(last["label"]),
        "confidence": float(last["confidence"]), "consensus": float(last["consensus"]),
        "vwap": float(last["vwap"]) if pd.notna(last["vwap"]) else None,
        "vwap_z": float(last["vwap_z"]),
        "s_vwap": float(last["s_vwap"]), "s_struct": float(last["s_struct"]),
        "s_trend": float(last["s_trend"]),
        "regime": str(last["regime"]),
        "adx": float(last["adx"]) if pd.notna(last["adx"]) else None,
        "st_dir": float(last["st_dir"]),
        "last_event": last_evt, "event_bars_ago": bars_ago,
        "htf": list(res.htf), "intraday": res.intraday,
        "vwap_is_proxy": res.vwap_is_proxy, "mode": res.mode,
    }
