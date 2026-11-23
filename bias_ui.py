"""
Streamlit panel for the direction-bias engine (kept out of app.py on purpose).

    render_bias_panel(score_result, data, config)

Shows Gold's bias (VWAP / Structure / Trend pillars, confluence, regime), the
per-market legacy-vs-bias table that feeds the 7-market composite, and a Gold
chart with VWAP bands, SuperTrend and the last swing pivots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .bias import latest_summary
    from .config import MARKET_LABELS, NEGATIVE_CORRELATIONS
except ImportError:
    from bias import latest_summary
    from config import MARKET_LABELS, NEGATIVE_CORRELATIONS

_GREEN, _RED, _GOLD, _GREY = "#2E8B57", "#B22222", "#DAA520", "#888888"


def _col(v: float) -> str:
    return _GREEN if v > 20 else (_RED if v < -20 else _GOLD)


def _pillar_card(title: str, value: float, sub: str) -> str:
    c = _col(value)
    return (
        f'<div style="border-left:4px solid {c};background:{c}14;padding:8px 12px;'
        f'border-radius:6px;height:100%">'
        f'<div style="font-size:0.68rem;color:#777;letter-spacing:0.04em">{title}</div>'
        f'<div style="font-size:1.25rem;font-weight:700;color:{c};line-height:1.15">{value:+.0f}</div>'
        f'<div style="font-size:0.72rem;color:#888">{sub}</div></div>'
    )


def render_bias_panel(score_result, data, config, n_bars: int = 150) -> None:
    bcfg = getattr(config, "bias", None)
    if bcfg is None or not bcfg.enabled:
        st.info("Direction bias is turned off (sidebar → Direction bias). "
                "The composite is using the original EMA / RSI / ROC scoring only.")
        return
    gold = score_result.bias_results.get("gold")
    if gold is None or len(gold.frame) == 0:
        st.warning("Direction bias could not be computed for Gold (not enough bars).")
        return

    sm = latest_summary(gold)
    f = gold.frame

    st.subheader("🧭 Direction bias — Gold")
    mode_name = {"hybrid": "VWAP + Structure + Trend",
                 "structure": "VWAP + Market Structure (MTF)",
                 "trend": "VWAP + SuperTrend / ADX"}.get(sm["mode"], sm["mode"])
    htf_txt = ", ".join(sm["htf"]) if sm["htf"] else "—"
    vwap_kind = ("session VWAP (resets daily)" if sm["intraday"]
                 else f"rolling {bcfg.vwap_window}-bar VWAP")
    if sm["vwap_is_proxy"]:
        vwap_kind += " · equal-weight (no volume feed)"
    st.caption(f"Model: **{mode_name}** · {vwap_kind} · structure TFs: base + {htf_txt}")

    c = _col(sm["bias"])
    gate_now = 1.0
    if score_result.gate_factor is not None and len(score_result.gate_factor):
        gate_now = float(score_result.gate_factor.iloc[-1])
    top1, top2, top3, top4 = st.columns([2, 1, 1, 1])
    with top1:
        st.markdown(
            f"""<div style="background:{c}1a;border-left:5px solid {c};padding:10px 14px;
                            border-radius:6px;display:flex;align-items:center;gap:14px">
                  <div>
                    <div style="font-size:0.7rem;color:#666;letter-spacing:0.04em">GOLD DIRECTION BIAS</div>
                    <div style="font-size:1.15rem;font-weight:700;color:{c};line-height:1.1">{sm['label']}</div>
                  </div>
                  <div style="font-size:0.85rem;color:#888">bias {sm['bias']:+.1f} · conf {sm['confidence']:.0f}%</div>
                </div>""",
            unsafe_allow_html=True)
    with top2:
        st.metric("Confluence", f"{sm['consensus'] * 100:.0f}%",
                  help="How much the three pillars agree on direction (100% = all agree).")
    with top3:
        adx_txt = f"ADX {sm['adx']:.0f}" if sm["adx"] is not None else "ADX …"
        st.metric("Regime", sm["regime"], delta=adx_txt, delta_color="off")
    with top4:
        st.metric("Composite gate", "open" if gate_now >= 0.999 else f"×{gate_now:.2f}",
                  help="×<1 means Gold's own bias opposes the macro composite, so the composite was damped.")

    p1, p2, p3 = st.columns(3)
    vwap_sub = (f"price {'above' if sm['vwap_z'] > 0 else 'below'} VWAP ({sm['vwap_z']:+.1f}σ)"
                if sm["vwap"] is not None else "warming up")
    with p1:
        st.markdown(_pillar_card("VWAP", sm["s_vwap"], vwap_sub), unsafe_allow_html=True)
    with p2:
        ago = f" · {sm['event_bars_ago']} bars ago" if sm["event_bars_ago"] is not None else ""
        st.markdown(_pillar_card("MARKET STRUCTURE (MTF)", sm["s_struct"],
                                 f"last break: {sm['last_event']}{ago}"), unsafe_allow_html=True)
    with p3:
        st_txt = "SuperTrend long" if sm["st_dir"] > 0 else ("SuperTrend short" if sm["st_dir"] < 0 else "warming up")
        st.markdown(_pillar_card("TREND (SuperTrend + ADX)", sm["s_trend"], st_txt), unsafe_allow_html=True)

    # ---- per-timeframe structure -----------------------------------------
    tf_names = ["base"] + list(sm["htf"])
    dir_cols = [c_ for c_ in ("struct_base", "struct_htf1", "struct_htf2") if c_ in f.columns]
    if dir_cols and sm["mode"] != "trend":
        sym = {1.0: "▲ bullish", -1.0: "▼ bearish", 0.0: "• undefined"}
        cells = []
        for name, col in zip(tf_names, dir_cols):
            cells.append(f"**{name}** {sym.get(float(f[col].iloc[-1]), '•')}")
        st.markdown("Structure by timeframe: " + " · ".join(cells))

    # ---- chart -----------------------------------------------------------
    gdf = data.get("gold")
    if gdf is not None and len(gdf) > 5 and all(k in gdf.columns for k in ("Open", "High", "Low", "Close")):
        tail = gdf.tail(n_bars)
        ft = f.reindex(tail.index)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=tail.index, open=tail["Open"], high=tail["High"], low=tail["Low"], close=tail["Close"],
            name="Gold", increasing_line_color=_GREEN, decreasing_line_color=_RED, showlegend=False))
        fig.add_trace(go.Scatter(x=tail.index, y=ft["vwap_hi"], mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=tail.index, y=ft["vwap_lo"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(70,130,180,0.12)",
                                 hoverinfo="skip", name="VWAP band"))
        fig.add_trace(go.Scatter(x=tail.index, y=ft["vwap"], mode="lines",
                                 line=dict(color="#4682B4", width=1.6), name="VWAP"))
        if ft["supertrend"].notna().any():
            fig.add_trace(go.Scatter(x=tail.index, y=ft["supertrend"].where(ft["st_dir"] > 0),
                                     mode="lines", line=dict(color=_GREEN, width=1.2, dash="dot"),
                                     name="SuperTrend ▲"))
            fig.add_trace(go.Scatter(x=tail.index, y=ft["supertrend"].where(ft["st_dir"] < 0),
                                     mode="lines", line=dict(color=_RED, width=1.2, dash="dot"),
                                     name="SuperTrend ▼"))
        if ft["piv_hi"].notna().any():
            fig.add_trace(go.Scatter(x=tail.index, y=ft["piv_hi"], mode="lines",
                                     line=dict(color="#999", width=1, dash="dash", shape="hv"),
                                     name="last swing high"))
            fig.add_trace(go.Scatter(x=tail.index, y=ft["piv_lo"], mode="lines",
                                     line=dict(color="#999", width=1, dash="dash", shape="hv"),
                                     name="last swing low"))
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", y=1.04, x=0))
        st.plotly_chart(fig, use_container_width=True)

    # ---- bias history ----------------------------------------------------
    hist = f["bias"].tail(max(n_bars, 200))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist.index, y=hist.clip(lower=0), fill="tozeroy", mode="none",
                              fillcolor="rgba(46,139,87,0.35)", name="bullish"))
    fig2.add_trace(go.Scatter(x=hist.index, y=hist.clip(upper=0), fill="tozeroy", mode="none",
                              fillcolor="rgba(178,34,34,0.35)", name="bearish"))
    fig2.add_trace(go.Scatter(x=hist.index, y=hist, mode="lines", line=dict(color="#555", width=1), name="bias"))
    fig2.update_layout(height=170, margin=dict(l=10, r=10, t=8, b=8), showlegend=False,
                       yaxis=dict(range=[-100, 100], zeroline=True))
    st.plotly_chart(fig2, use_container_width=True)

    # ---- per-market table: how the bias enters the 7-market composite -----
    st.markdown("**How the bias enters the 7-market composite**")
    latest = score_result.composite.index[-1]
    weights = config.normalized_weights()
    rows = []
    for mkt, label in MARKET_LABELS.items():
        if mkt not in score_result.bias_results:
            continue
        legacy = float(score_result.legacy_per_market[mkt].loc[latest])
        b = float(score_result.bias_per_market[mkt].loc[latest])
        final = float(score_result.per_market[mkt].loc[latest])
        flip = mkt in NEGATIVE_CORRELATIONS
        br = score_result.bias_results[mkt].frame.iloc[-1]
        rows.append({
            "Market": f"{label}{' 🔄' if flip else ''}",
            "Legacy (EMA/RSI/ROC)": f"{legacy:+.0f}",
            "Bias": f"{b:+.0f}",
            "VWAP": f"{br['s_vwap']:+.0f}",
            "Structure": f"{br['s_struct']:+.0f}",
            "Trend": f"{br['s_trend']:+.0f}",
            f"Blended ({bcfg.blend:.0%} bias)": f"{final:+.0f}",
            "Weight": f"{weights[mkt] * 100:.0f}%",
            "Regime": str(br["regime"]),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Market"), use_container_width=True)
        st.caption("🔄 = sign-inverted for Gold (final column is already inverted, as in the composite). "
                   "Markets without a volume feed (DXY, VIX, EUR/USD) use an equal-weight VWAP.")
