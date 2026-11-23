"""
Self-checks for bias.py  (run:  python test_bias.py)

Uses synthetic OHLCV only - it verifies MECHANICS (no look-ahead, graceful
degradation, sane ranges).  It says nothing about predictive edge; judge that
with the app's Backtest / Signal-BT tabs on real data.
"""
import numpy as np
import pandas as pd

from bias import (BiasConfig, compute_bias, supertrend, structure_tf,
                  latest_summary)


def make_ohlcv(n, freq, seed=0, vol=True, start="2026-01-05"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    # regime-switching drift so trends and ranges both occur
    drift = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=n // 60 + 1), 60)[:n] * 0.04
    ret = drift + rng.normal(0, 0.25, n)
    close = 2000 + np.cumsum(ret)
    openp = np.r_[close[0], close[:-1]]
    hi = np.maximum(openp, close) + np.abs(rng.normal(0, 0.15, n))
    lo = np.minimum(openp, close) - np.abs(rng.normal(0, 0.15, n))
    df = pd.DataFrame({"Open": openp, "High": hi, "Low": lo, "Close": close}, index=idx)
    df["Volume"] = rng.integers(100, 5000, n).astype(float) if vol else 0.0
    return df


def test_ranges_and_columns():
    for freq, n in [("15min", 2000), ("1h", 1400), ("1D", 780)]:
        r = compute_bias(make_ohlcv(n, freq), BiasConfig())
        f = r.frame
        assert len(f) == n
        for c in ["bias", "s_vwap", "s_struct", "s_trend", "confidence", "label", "regime"]:
            assert c in f.columns, c
        assert f["bias"].abs().max() <= 100.0 + 1e-9
        assert f["confidence"].between(0, 100).all()
        assert not f["bias"].isna().any()
        print(f"  ok ranges  {freq:>5}  htf={r.htf}  intraday={r.intraday}")


def test_no_lookahead():
    """bias at bar t must be identical whether or not future bars exist.
    Sweeps ~60 cut points per timeframe so partially-formed higher-TF bars
    (the classic leak) are exercised."""
    cfg = BiasConfig()
    for freq, n, seed in [("15min", 1500, 3), ("1h", 900, 3), ("1h", 900, 8), ("1D", 500, 3)]:
        df = make_ohlcv(n, freq, seed=seed)
        full = compute_bias(df, cfg).frame
        cols = ["bias", "s_vwap", "s_struct", "s_trend"]
        step = max(1, (n // 2) // 60)
        for cut in range(n // 2, n, step):
            part = compute_bias(df.iloc[:cut], cfg).frame
            a = part[cols].iloc[-1].to_numpy(float)
            b = full[cols].iloc[cut - 1].to_numpy(float)
            assert np.allclose(a, b, atol=1e-8), (freq, seed, cut, a, b)
    print("  ok causality (15m, 1h x2 seeds, 1d - truncated == full at ~60 cuts each)")


def test_missing_columns():
    df = make_ohlcv(900, "1h", seed=5)
    r1 = compute_bias(df[["Close"]], BiasConfig())            # Close only
    r2 = compute_bias(df.drop(columns=["Volume"]), BiasConfig())
    r3 = compute_bias(make_ohlcv(900, "1h", vol=False), BiasConfig())   # volume all zero
    assert r2.vwap_is_proxy and r3.vwap_is_proxy and not compute_bias(df).vwap_is_proxy
    for r in (r1, r2, r3):
        assert r.frame["bias"].abs().max() <= 100 and not r.frame["bias"].isna().any()
    print("  ok graceful degradation (Close-only / no volume / zero volume)")


def test_ffill_rows_get_no_weight():
    """duplicate (forward-filled) rows with fake volume must not move VWAP."""
    df = make_ohlcv(600, "1h", seed=7)
    ff = df.copy()
    ff.iloc[300:312] = ff.iloc[299].to_numpy()                # stale block, repeated volume
    clean = ff.copy()
    clean.loc[clean.index[300:312], "Volume"] = 0.0
    a = compute_bias(ff).frame["vwap"].iloc[-1]
    b = compute_bias(clean).frame["vwap"].iloc[-1]
    assert np.isclose(a, b, equal_nan=True)
    print("  ok stale forward-filled rows ignored by VWAP")


def test_modes_differ_and_trend_skips_structure():
    df = make_ohlcv(1400, "1h", seed=11)
    out = {m: compute_bias(df, BiasConfig(mode=m)).frame["bias"].iloc[-300:] for m in
           ("hybrid", "structure", "trend")}
    assert not np.allclose(out["hybrid"], out["structure"])
    assert not np.allclose(out["hybrid"], out["trend"])
    print("  ok three modes produce distinct bias")


def test_structure_logic():
    # hand-built zig-zag: clean uptrend (HH/HL) then a break below the last low (CHoCH)
    px = [10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 14, 12, 9, 8, 7, 8, 9, 8]
    s = pd.Series(px, dtype=float)
    r = structure_tf(s + 0.2, s - 0.2, s, n=1)
    assert r["dir"].iloc[10] == 1.0, r["dir"].tolist()
    assert (r["evt"] == -2.0).any(), "expected a bearish CHoCH"
    assert r["dir"].iloc[-1] == -1.0
    print("  ok structure: uptrend -> bearish CHoCH -> downtrend")


def test_supertrend_flips():
    up = pd.Series(np.linspace(100, 140, 60))
    dn = pd.Series(np.linspace(140, 90, 60))
    c = pd.concat([up, dn], ignore_index=True)
    _, d = supertrend(c + 0.5, c - 0.5, c, 10, 3.0)
    assert d.iloc[50] == 1.0 and d.iloc[-1] == -1.0
    print("  ok supertrend: long in rally, short after reversal")


def test_summary():
    s = latest_summary(compute_bias(make_ohlcv(900, "1h", seed=2)))
    assert {"bias", "label", "regime", "htf", "last_event"} <= set(s)
    print("  ok latest_summary")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("ALL BIAS TESTS PASSED")
