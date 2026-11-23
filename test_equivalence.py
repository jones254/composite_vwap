"""
Proves the backtest fast-path (score history once, read values as-of each step)
equals the slow path (re-score a truncated history at every step).
Run:  python test_equivalence.py
"""
import numpy as np
from test_bias import make_ohlcv
from config import Config, MARKET_LABELS
from scoring import composite_score
from signals import evaluate, SignalConfig


def world(n, freq, seed):
    d = {}
    for i, m in enumerate(MARKET_LABELS):
        d[m] = make_ohlcv(n, freq, seed=seed + i, vol=m not in ("dxy", "vix", "eurusd"))
    return d


def run(bias_on, loose=False):
    d15, d1h = world(1800, "15min", 10), world(1300, "1h", 40)
    cfg = Config()
    cfg.bias.enabled = bias_on
    sig = SignalConfig(bias_gate=bias_on)
    if loose:   # make trades fire so entry / stop / target are compared too
        sig = SignalConfig(bias_gate=bias_on, forecast_alignment_required=1,
                           momentum_gap_required=False)
    pre15, pre1h = composite_score(d15, cfg), composite_score(d1h, cfg)
    g1h = d1h["gold"]
    checked = signals = 0
    for i in range(400, len(g1h) - 1, 37):
        ts = g1h.index[i]
        slow = evaluate({m: df.loc[:ts] for m, df in d15.items()},
                        {m: df.loc[:ts] for m, df in d1h.items()}, cfg, sig)
        fast = evaluate({"gold": d15["gold"].loc[:ts]}, {"gold": d1h["gold"].loc[:ts]},
                        cfg, sig, precomputed=(pre15, pre1h, ts))
        assert slow.signal == fast.signal, (ts, slow.signal, fast.signal)
        for f in ("composite_15m", "composite_1h", "bias_15m", "bias_1h", "probability"):
            assert np.isclose(getattr(slow, f), getattr(fast, f), atol=1e-9), (ts, f)
        if slow.signal in ("BUY", "SELL"):
            for f in ("entry", "stop", "target"):
                assert np.isclose(getattr(slow, f), getattr(fast, f), atol=1e-9), (ts, f)
        checked += 1
        signals += slow.signal in ("BUY", "SELL")
    return checked, signals


if __name__ == "__main__":
    for on in (True, False):
        for loose in (False, True):
            n, s = run(on, loose)
            print(f"bias={'ON ' if on else 'OFF'} {'loose' if loose else 'strict'}: "
                  f"{n} timestamps identical (fast == slow); {s} were BUY/SELL")
    print("EQUIVALENCE VERIFIED")
