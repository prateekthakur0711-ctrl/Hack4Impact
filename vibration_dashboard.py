"""
Live Vibration Dashboard — Acoustic Leak-Net
Fixes:
  1. No blinking — uirevision on every Plotly figure prevents re-animation
  2. Full city block network map — 12 sensors, 17 pipe segments
"""

from __future__ import annotations
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import fft as sp_fft
from algorithms import sim as sim_alg
from algorithms import locator as loc_alg

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DISPLAY_WINDOW_S  = 0.05
FFT_MAX_FREQ      = 800
DEFAULT_THRESHOLD = 0.70

# ─────────────────────────────────────────────
# City block layout
# ─────────────────────────────────────────────
CITY_SENSORS = {
    "S01": (1, 5), "S02": (3, 5), "S03": (5, 5), "S04": (7, 5),
    "S05": (1, 3), "S06": (3, 3), "S07": (5, 3), "S08": (7, 3),
    "S09": (1, 1), "S10": (3, 1), "S11": (5, 1), "S12": (7, 1),
}
CITY_PIPES = [
    ("S01","S02"),("S02","S03"),("S03","S04"),
    ("S05","S06"),("S06","S07"),("S07","S08"),
    ("S09","S10"),("S10","S11"),("S11","S12"),
    ("S01","S05"),("S05","S09"),
    ("S02","S06"),("S06","S10"),
    ("S03","S07"),("S07","S11"),
    ("S04","S08"),("S08","S12"),
]
STREET_LABELS = [
    (0.2, 5.4, "Main St"), (0.2, 3.4, "Oak St"), (0.2, 1.4, "Elm St"),
    (1,   6.2, "1st Ave"), (3,   6.2, "2nd Ave"),
    (5,   6.2, "3rd Ave"), (7,   6.2, "4th Ave"),
]
VALID_PAIRS = [f"{a} → {b}" for a, b in CITY_PIPES]
PAIR_MAP    = {f"{a} → {b}": (a, b) for a, b in CITY_PIPES}


def _pipe_len(sa, sb):
    ax, ay = CITY_SENSORS[sa]; bx, by = CITY_SENSORS[sb]
    return round(np.sqrt((bx-ax)**2 + (by-ay)**2) * 5.0, 2)


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "vd_running": False, "vd_result": None, "vd_meta": None,
        "vd_signals": None,  "vd_lags": None,   "vd_corr": None,
        "vd_peak_lag": None, "vd_alert_history": [],
        "vd_run_count": 0,   "vd_active_pair_key": VALID_PAIRS[0],
        "vd_leak_history_map": {}, "vd_pair_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
# Core simulation cycle
# ─────────────────────────────────────────────
def run_cycle(pipe_len_m, leak_pos_m, noise_std, threshold, seed):
    sim_cfg = sim_alg.SimulationConfig(
        fs_hz=44100, duration_s=2.0,
        sensor_spacing_m=float(pipe_len_m),
        leak_frequency_hz=240.0, speed_of_sound_m_s=1480.0,
        leak_distance_from_a_m=float(leak_pos_m),
        noise_std=float(noise_std), seed=seed,
    )
    t, a, b, meta = sim_alg.generate_signals(sim_cfg)
    loc_cfg = loc_alg.LocatorConfig(
        fs_hz=sim_cfg.fs_hz, speed_of_sound_m_s=sim_cfg.speed_of_sound_m_s,
        sensor_spacing_m=sim_cfg.sensor_spacing_m,
        target_frequency_hz=sim_cfg.leak_frequency_hz,
        band_half_width_hz=25.0, confidence_threshold=float(threshold),
    )
    result, _, (lags_w, corr_w, peak_lag) = loc_alg.localize_from_arrays(a, b, loc_cfg, meta=meta)
    pair_key = st.session_state.vd_active_pair_key
    st.session_state.update({
        "vd_signals": (t, a, b), "vd_meta": meta, "vd_result": result,
        "vd_lags": lags_w, "vd_corr": corr_w, "vd_peak_lag": peak_lag,
    })
    st.session_state.vd_run_count += 1
    st.session_state.vd_leak_history_map[pair_key] = result
    entry = {
        "ts": time.strftime("%H:%M:%S"),
        "confidence": round(float(result.confidence_score), 3),
        "dist_m": round(float(result.estimated_distance_m), 2),
        "detected": bool(result.leak_detected),
        "run": st.session_state.vd_run_count,
        "pair": pair_key,
    }
    hist = st.session_state.vd_alert_history
    hist.append(entry)
    if len(hist) > 50:
        hist.pop(0)


# ─────────────────────────────────────────────
# FFT helper
# ─────────────────────────────────────────────
def magnitude_spectrum(x, fs_hz):
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    n = x.size
    X = sp_fft.rfft(x * np.hanning(n))
    freq = sp_fft.rfftfreq(n, d=1.0/float(fs_hz))
    return freq, np.abs(X) / max(n, 1)


# ─────────────────────────────────────────────
# Shared plot layout (uirevision prevents blink)
# ─────────────────────────────────────────────
_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=8, b=0), font=dict(size=12),
    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
)


def fig_timeseries(t, a, b, fs_hz, window_s, thr):
    n = int(min(len(t), window_s * fs_hz))
    t_s, a_s, b_s = t[:n], a[:n], b[:n]
    above = max(float(np.max(np.abs(a_s))), float(np.max(np.abs(b_s)))) > thr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_s, y=a_s, name="Sensor A",
                             line=dict(color="#378ADD", width=1.5), mode="lines"))
    fig.add_trace(go.Scatter(x=t_s, y=b_s, name="Sensor B",
                             line=dict(color="#1D9E75", width=1.5), mode="lines", opacity=0.85))
    fig.add_hline(y=thr,  line=dict(color="#E24B4A", dash="dash", width=1.2),
                  annotation_text="threshold", annotation_position="top right")
    fig.add_hline(y=-thr, line=dict(color="#E24B4A", dash="dash", width=1.2))
    if above:
        fig.add_hrect(y0=thr, y1=max(float(np.max(a_s)), thr+0.1),
                      fillcolor="rgba(226,75,74,0.07)", line_width=0)
    fig.update_layout(height=280, uirevision="ts",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis_title="Time (s)", yaxis_title="Amplitude", **_BASE)
    return fig


def fig_fft(a, b, fs_hz, target_hz):
    fa, ma = magnitude_spectrum(a, fs_hz)
    fb, mb = magnitude_spectrum(b, fs_hz)
    mask = fa <= FFT_MAX_FREQ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fa[mask], y=ma[mask], name="Sensor A",
                             line=dict(color="#378ADD", width=1.5), mode="lines"))
    fig.add_trace(go.Scatter(x=fb[mask], y=mb[mask], name="Sensor B",
                             line=dict(color="#1D9E75", width=1.5), mode="lines", opacity=0.85))
    fig.add_vline(x=target_hz, line=dict(color="#E24B4A", dash="dot", width=1.5),
                  annotation_text=f"{int(target_hz)} Hz", annotation_position="top right")
    fig.update_layout(height=220, uirevision="fft",
                      xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      **_BASE)
    return fig


def fig_xcorr(lags_w, corr_w, fs_hz, peak_lag):
    lag_s = lags_w / float(fs_hz)
    peak_s = peak_lag / float(fs_hz)
    pc = corr_w[lags_w == peak_lag]
    peak_y = float(pc[0]) if len(pc) > 0 else 0.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lag_s, y=corr_w, line=dict(color="#7F77DD", width=2),
                             mode="lines", fill="tozeroy", fillcolor="rgba(127,119,221,0.08)"))
    fig.add_vline(x=peak_s, line=dict(color="#333", dash="dash", width=1.2),
                  annotation_text=f"peak {peak_s*1000:.3f} ms", annotation_position="top left")
    fig.add_trace(go.Scatter(x=[peak_s], y=[peak_y], mode="markers",
                             marker=dict(color="#E24B4A", size=9), showlegend=False))
    fig.update_layout(height=220, uirevision="xc", showlegend=False,
                      xaxis_title="Lag (s)  [+ve → B later than A]",
                      yaxis_title="Correlation", **_BASE)
    return fig


def fig_alert_history(history):
    if not history:
        return go.Figure()
    runs   = [h["run"] for h in history]
    confs  = [h["confidence"] for h in history]
    dists  = [h["dist_m"] for h in history]
    colors = ["#E24B4A" if h["detected"] else "#1D9E75" for h in history]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig.add_trace(go.Bar(x=runs, y=confs, marker_color=colors, showlegend=False), row=1, col=1)
    fig.add_hline(y=st.session_state.get("vd_threshold", DEFAULT_THRESHOLD),
                  line=dict(color="#888", dash="dash", width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=runs, y=dists, mode="lines+markers",
                             line=dict(color="#378ADD", width=1.5),
                             marker=dict(size=5), showlegend=False), row=2, col=1)
    fig.update_layout(
        height=220, uirevision="hist",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis2_title="Run #", yaxis_title="Confidence", yaxis2_title="Dist (m)",
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


# ─────────────────────────────────────────────
# City block network map
# ─────────────────────────────────────────────
def fig_city_map(active_pair_key, result, history_map):
    fig = go.Figure()
    active_a, active_b = PAIR_MAP[active_pair_key]
    leak_detected = result is not None and result.leak_detected

    # pipe segments
    for sa, sb in CITY_PIPES:
        pair_key = f"{sa} → {sb}"
        ax, ay = CITY_SENSORS[sa]; bx, by = CITY_SENSORS[sb]
        is_active = (sa == active_a and sb == active_b)
        hist = history_map.get(pair_key)
        if is_active:
            color, width, dash = ("#D92D20" if leak_detected else "#039855"), 5, "solid"
        elif hist is not None and hist.leak_detected:
            color, width, dash = "#F97316", 3, "solid"
        else:
            color, width, dash = "#CBD5E1", 2, "dot"
        fig.add_trace(go.Scatter(
            x=[ax, bx], y=[ay, by], mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertext=f"{pair_key} | {_pipe_len(sa,sb)} m",
            hoverinfo="text", showlegend=False,
        ))

    # leak dot
    if result is not None:
        ax, ay = CITY_SENSORS[active_a]; bx, by = CITY_SENSORS[active_b]
        pipe_len = _pipe_len(active_a, active_b)
        frac = min(max(float(result.estimated_distance_m) / max(pipe_len, 0.001), 0), 1)
        lx = ax + (bx - ax) * frac
        ly = ay + (by - ay) * frac
        dot_color = "#D92D20" if leak_detected else "#059669"
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="markers+text",
            marker=dict(color=dot_color, size=14, symbol="circle",
                        line=dict(color="white", width=2)),
            text=["💧" if leak_detected else "✓"],
            textposition="top center",
            hovertext=f"Est: {result.estimated_distance_m:.2f} m | Conf: {result.confidence_score:.3f}",
            hoverinfo="text", showlegend=False,
        ))

    # sensor nodes
    active_sensors = {active_a, active_b}
    leaked_sensors = {s for pk, r in history_map.items() if r.leak_detected
                      for s in PAIR_MAP[pk]}
    for sid, (sx, sy) in CITY_SENSORS.items():
        if sid in active_sensors:
            color, size = ("#D92D20" if leak_detected else "#059669"), 14
        elif sid in leaked_sensors:
            color, size = "#F97316", 11
        else:
            color, size = "#64748B", 10
        fig.add_trace(go.Scatter(
            x=[sx], y=[sy], mode="markers+text",
            marker=dict(color=color, size=size, line=dict(color="white", width=1.5)),
            text=[sid], textposition="bottom center",
            textfont=dict(size=9, color="#334155"),
            hovertext=sid, hoverinfo="text", showlegend=False,
        ))

    # street labels
    for lx, ly, label in STREET_LABELS:
        fig.add_annotation(x=lx, y=ly, text=label, showarrow=False,
                           font=dict(size=9, color="#94A3B8"), xanchor="left")

    # legend
    for color, label in [("#D92D20","Active – leak"),("#039855","Active – clear"),
                          ("#F97316","Past leak"),("#CBD5E1","Unmonitored")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color=color, width=3),
                                 name=label, showlegend=True))

    fig.update_layout(
        height=400, uirevision="citymap",
        xaxis=dict(range=[-0.5, 8.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5, 7.2], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0,
                    font=dict(size=10)),
        font=dict(size=11),
    )
    return fig


# ─────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────
def render():
    st.title("Live Vibration Dashboard")
    st.caption("Acoustic leak simulation — threshold alerting and TDOA localization.")

    with st.sidebar:
        st.header("Dashboard Controls")

        pair_idx = st.selectbox(
            "Sensor pair (pipe segment)",
            options=list(range(len(VALID_PAIRS))),
            format_func=lambda i: VALID_PAIRS[i],
            index=st.session_state.vd_pair_index,
            key="vd_pair_select_idx",
        )
        new_pair_key = VALID_PAIRS[pair_idx]
        if new_pair_key != st.session_state.vd_active_pair_key:
            st.session_state.vd_active_pair_key = new_pair_key
            st.session_state.vd_pair_index = pair_idx
            st.session_state.vd_result = None

        sa, sb = PAIR_MAP[st.session_state.vd_active_pair_key]
        pipe_len_m = _pipe_len(sa, sb)
        st.caption(f"Pipe length: **{pipe_len_m} m**")

        leak_pos_m = st.slider("Leak position from Sensor A (m)",
                                0.0, float(pipe_len_m), min(4.2, float(pipe_len_m)), 0.1,
                                key="vd_leak_pos")
        noise_std  = st.slider("Noise level", 0.0, 1.0, 0.25, 0.01, key="vd_noise")
        threshold  = st.slider("Alert threshold (confidence)", 0.3, 1.0,
                                DEFAULT_THRESHOLD, 0.01, key="vd_threshold")
        amp_thresh = st.slider("Amplitude alert level", 0.1, 2.0, 0.8, 0.05,
                                key="vd_amp_thresh")
        rand_seed  = st.checkbox("Randomize seed each run", value=False, key="vd_random_seed")

        st.divider()
        col_run, col_live = st.columns(2)
        with col_run:
            run_once = st.button("Run once", use_container_width=True, type="primary")
        with col_live:
            live_toggle = st.button(
                "Stop live" if st.session_state.vd_running else "Start live",
                use_container_width=True,
            )

        if run_once:
            seed = None if rand_seed else st.session_state.vd_run_count
            run_cycle(pipe_len_m, leak_pos_m, noise_std, threshold, seed)

        if live_toggle:
            st.session_state.vd_running = not st.session_state.vd_running

        if st.session_state.vd_alert_history:
            st.divider()
            if st.button("Clear history", use_container_width=True):
                st.session_state.vd_alert_history    = []
                st.session_state.vd_leak_history_map = {}
                st.rerun()

    # live loop runs data BEFORE rendering (no double rerun)
    if st.session_state.vd_running:
        seed = None if rand_seed else st.session_state.vd_run_count
        run_cycle(pipe_len_m, leak_pos_m, noise_std, threshold, seed)

    # no data yet — just show map
    if st.session_state.vd_result is None:
        st.info("Pick a sensor pair and click **Run once** or **Start live**.")
        st.subheader("City Block Sensor Network")
        st.plotly_chart(
            fig_city_map(st.session_state.vd_active_pair_key, None,
                         st.session_state.vd_leak_history_map),
            use_container_width=True, config={"displayModeBar": False},
        )
        if st.session_state.vd_running:
            time.sleep(0.4)
            st.rerun()
        return

    result   = st.session_state.vd_result
    t, a, b  = st.session_state.vd_signals
    meta     = st.session_state.vd_meta
    lags_w   = st.session_state.vd_lags
    corr_w   = st.session_state.vd_corr
    peak_lag = st.session_state.vd_peak_lag
    fs_hz    = int(meta["fs_hz"])
    peak_amp = float(np.max(np.abs(a[:int(DISPLAY_WINDOW_S * fs_hz)])))
    amp_alert = peak_amp > amp_thresh

    # alert banner
    if result.leak_detected:
        st.markdown(
            f"""<div style="padding:14px 18px;border-radius:10px;border:2px solid #D92D20;
                background:#FFF1F0;margin-bottom:12px;">
              <span style="font-size:20px;font-weight:800;color:#B42318;">⚠ LEAK DETECTED</span>
              <span style="font-size:15px;color:#7A271A;margin-left:12px;">
                Confidence <b>{result.confidence_score:.3f}</b> —
                estimated <b>{result.estimated_distance_m:.2f} m</b> from {sa}
              </span></div>""",
            unsafe_allow_html=True,
        )
    elif amp_alert:
        st.markdown(
            f"""<div style="padding:14px 18px;border-radius:10px;border:2px solid #DC6803;
                background:#FFFAEB;margin-bottom:12px;">
              <span style="font-size:18px;font-weight:700;color:#93370D;">⚡ AMPLITUDE WARNING</span>
              <span style="font-size:14px;color:#7A2E0E;margin-left:12px;">
                Peak <b>{peak_amp:.3f}</b> exceeds <b>{amp_thresh:.2f}</b>
              </span></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="padding:14px 18px;border-radius:10px;border:1.5px solid #039855;
                background:#F6FEF9;margin-bottom:12px;">
              <span style="font-size:16px;font-weight:600;color:#027A48;">✓ System nominal — no leak detected</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Detected freq",  f"{result.detected_frequency_hz:.1f} Hz")
    c2.metric("Confidence",     f"{result.confidence_score:.3f}",
              delta="LEAK" if result.leak_detected else "OK",
              delta_color="inverse" if result.leak_detected else "normal")
    c3.metric("Est. distance",  f"{result.estimated_distance_m:.2f} m")
    c4.metric("TDOA",           f"{result.tdoa_s*1000:.3f} ms")
    c5.metric("Peak amplitude", f"{peak_amp:.3f}",
              delta="ABOVE" if amp_alert else "normal",
              delta_color="inverse" if amp_alert else "off")

    st.markdown("---")

    # row 1: waveform + city map
    col_ts, col_map = st.columns([3, 2])
    with col_ts:
        st.subheader(f"Sensor waveforms — {st.session_state.vd_active_pair_key}")
        st.plotly_chart(fig_timeseries(t, a, b, fs_hz, DISPLAY_WINDOW_S, amp_thresh),
                        use_container_width=True, config={"displayModeBar": False})
    with col_map:
        st.subheader("City Block Network")
        st.plotly_chart(
            fig_city_map(st.session_state.vd_active_pair_key, result,
                         st.session_state.vd_leak_history_map),
            use_container_width=True, config={"displayModeBar": False},
        )

    # row 2: FFT + XCorr
    col_fft, col_xc = st.columns(2)
    with col_fft:
        st.subheader("FFT spectrum (0–800 Hz)")
        st.plotly_chart(fig_fft(a, b, fs_hz, float(meta["leak_frequency_hz"])),
                        use_container_width=True, config={"displayModeBar": False})
    with col_xc:
        st.subheader("Cross-correlation (TDOA)")
        st.plotly_chart(fig_xcorr(lags_w, corr_w, fs_hz, int(peak_lag)),
                        use_container_width=True, config={"displayModeBar": False})

    # row 3: history
    if st.session_state.vd_alert_history:
        st.markdown("---")
        st.subheader(f"Alert history ({len(st.session_state.vd_alert_history)} runs)")
        st.plotly_chart(fig_alert_history(st.session_state.vd_alert_history),
                        use_container_width=True, config={"displayModeBar": False})
        hist_rev = list(reversed(st.session_state.vd_alert_history[-10:]))
        rows_md  = "| Run | Pair | Time | Confidence | Est. dist | Status |\n|---|---|---|---|---|---|\n"
        for h in hist_rev:
            rows_md += (f"| {h['run']} | {h.get('pair','—')} | {h['ts']} "
                        f"| {h['confidence']:.3f} | {h['dist_m']:.2f} m "
                        f"| {'🔴 LEAK' if h['detected'] else '🟢 OK'} |\n")
        st.markdown(rows_md)

    # live indicator + rerun
    if st.session_state.vd_running:
        st.caption(f"🔴 Live — run #{st.session_state.vd_run_count} — refreshing…")
        time.sleep(0.4)
        st.rerun()
    else:
        st.caption(f"Run #{st.session_state.vd_run_count} — click **Start live** for continuous updates.")


if __name__ == "__main__":
    st.set_page_config(page_title="Vibration Dashboard — Acoustic Leak-Net", layout="wide")
    render()