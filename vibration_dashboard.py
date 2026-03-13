"""
Live Vibration Dashboard — Acoustic Leak-Net
Integrates with algorithms/sim.py and algorithms/locator.py.
Add to app.py sidebar or run standalone: streamlit run vibration_dashboard.py
"""

from __future__ import annotations

import time
import json
import threading
from dataclasses import asdict
from pathlib import Path

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
DISPLAY_WINDOW_S = 0.05   # seconds of signal to show in live chart
FFT_MAX_FREQ = 800        # Hz — x-axis cap for spectrum plot
REFRESH_MS = 600          # Streamlit auto-refresh interval
DEFAULT_THRESHOLD = 0.70  # correlation confidence threshold for alert

# ─────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "vd_running": False,
        "vd_result": None,
        "vd_meta": None,
        "vd_signals": None,      # (t, a, b)
        "vd_lags": None,
        "vd_corr": None,
        "vd_peak_lag": None,
        "vd_alert_history": [],  # list of {ts, confidence, dist, detected}
        "vd_run_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
# Core: run one simulation + localization cycle
# ─────────────────────────────────────────────
def run_cycle(
    pipe_len_m: float,
    leak_pos_m: float,
    noise_std: float,
    threshold: float,
    seed: int | None,
) -> None:
    sim_cfg = sim_alg.SimulationConfig(
        fs_hz=44100,
        duration_s=2.0,
        sensor_spacing_m=float(pipe_len_m),
        leak_frequency_hz=240.0,
        speed_of_sound_m_s=1480.0,
        leak_distance_from_a_m=float(leak_pos_m),
        noise_std=float(noise_std),
        seed=seed,
    )
    t, a, b, meta = sim_alg.generate_signals(sim_cfg)

    loc_cfg = loc_alg.LocatorConfig(
        fs_hz=sim_cfg.fs_hz,
        speed_of_sound_m_s=sim_cfg.speed_of_sound_m_s,
        sensor_spacing_m=sim_cfg.sensor_spacing_m,
        target_frequency_hz=sim_cfg.leak_frequency_hz,
        band_half_width_hz=25.0,
        confidence_threshold=float(threshold),
    )
    result, impact, (lags_w, corr_w, peak_lag) = loc_alg.localize_from_arrays(a, b, loc_cfg, meta=meta)

    st.session_state.vd_signals = (t, a, b)
    st.session_state.vd_meta = meta
    st.session_state.vd_result = result
    st.session_state.vd_lags = lags_w
    st.session_state.vd_corr = corr_w
    st.session_state.vd_peak_lag = peak_lag
    st.session_state.vd_run_count += 1

    # append to alert history (keep last 50)
    entry = {
        "ts": time.strftime("%H:%M:%S"),
        "confidence": round(float(result.confidence_score), 3),
        "dist_m": round(float(result.estimated_distance_m), 2),
        "detected": bool(result.leak_detected),
        "run": st.session_state.vd_run_count,
    }
    hist = st.session_state.vd_alert_history
    hist.append(entry)
    if len(hist) > 50:
        hist.pop(0)


# ─────────────────────────────────────────────
# FFT helper (same as main_demo.py)
# ─────────────────────────────────────────────
def magnitude_spectrum(x: np.ndarray, fs_hz: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    n = x.size
    X = sp_fft.rfft(x * np.hanning(n))
    freqs = sp_fft.rfftfreq(n, d=1.0 / float(fs_hz))
    mag = np.abs(X) / max(n, 1)
    return freqs, mag


# ─────────────────────────────────────────────
# Plotly figures
# ─────────────────────────────────────────────
def fig_timeseries(t, a, b, fs_hz: int, window_s: float, threshold_amp: float) -> go.Figure:
    n = int(min(len(t), window_s * fs_hz))
    t_s = t[:n]
    a_s = a[:n]
    b_s = b[:n]

    peak_amp = max(float(np.max(np.abs(a_s))), float(np.max(np.abs(b_s))))
    above = peak_amp > threshold_amp

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_s, y=a_s, name="Sensor A",
                             line=dict(color="#378ADD", width=1.5), mode="lines"))
    fig.add_trace(go.Scatter(x=t_s, y=b_s, name="Sensor B",
                             line=dict(color="#1D9E75", width=1.5), mode="lines", opacity=0.85))
    # threshold bands
    fig.add_hline(y=threshold_amp,  line=dict(color="#E24B4A", dash="dash", width=1.2),
                  annotation_text="threshold", annotation_position="top right")
    fig.add_hline(y=-threshold_amp, line=dict(color="#E24B4A", dash="dash", width=1.2))

    if above:
        fig.add_hrect(y0=threshold_amp, y1=max(float(np.max(a_s)), threshold_amp + 0.1),
                      fillcolor="rgba(226,75,74,0.07)", line_width=0)

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        font=dict(size=12),
    )
    return fig


def fig_fft(a: np.ndarray, b: np.ndarray, fs_hz: int, target_hz: float) -> go.Figure:
    freqs_a, mag_a = magnitude_spectrum(a, fs_hz)
    freqs_b, mag_b = magnitude_spectrum(b, fs_hz)
    mask = freqs_a <= FFT_MAX_FREQ

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs_a[mask], y=mag_a[mask], name="Sensor A",
                             line=dict(color="#378ADD", width=1.5), mode="lines"))
    fig.add_trace(go.Scatter(x=freqs_b[mask], y=mag_b[mask], name="Sensor B",
                             line=dict(color="#1D9E75", width=1.5), mode="lines", opacity=0.85))
    fig.add_vline(x=target_hz, line=dict(color="#E24B4A", dash="dot", width=1.5),
                  annotation_text=f"{int(target_hz)} Hz leak band",
                  annotation_position="top right")
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        font=dict(size=12),
    )
    return fig


def fig_xcorr(lags_w: np.ndarray, corr_w: np.ndarray, fs_hz: int, peak_lag: int) -> go.Figure:
    lag_s = lags_w / float(fs_hz)
    peak_s = peak_lag / float(fs_hz)
    peak_corr = corr_w[lags_w == peak_lag]
    peak_y = float(peak_corr[0]) if len(peak_corr) > 0 else 0.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lag_s, y=corr_w, name="Cross-correlation",
                             line=dict(color="#7F77DD", width=2), mode="lines",
                             fill="tozeroy", fillcolor="rgba(127,119,221,0.08)"))
    fig.add_vline(x=peak_s, line=dict(color="#333", dash="dash", width=1.2),
                  annotation_text=f"peak {peak_s*1000:.3f} ms",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(x=[peak_s], y=[peak_y], mode="markers",
                             marker=dict(color="#E24B4A", size=9), showlegend=False))
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis_title="Lag (s)  [+ve → B later than A]",
        yaxis_title="Correlation",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        font=dict(size=12),
    )
    return fig


def fig_pipe_map(pipe_len_m: float, est_dist_m: float, true_dist_m: float | None) -> go.Figure:
    fig = go.Figure()
    # pipe body
    fig.add_shape(type="rect", x0=0, x1=pipe_len_m, y0=0.35, y1=0.65,
                  fillcolor="#D0D5DD", line=dict(color="#667085", width=2))
    # sensor markers
    for x, label in [(0, "Sensor A"), (pipe_len_m, "Sensor B")]:
        fig.add_shape(type="line", x0=x, x1=x, y0=0.25, y1=0.75,
                      line=dict(color="#667085", width=4))
        fig.add_annotation(x=x, y=0.85, text=label, showarrow=False, font=dict(size=12))

    # estimated leak
    fig.add_trace(go.Scatter(
        x=[est_dist_m], y=[0.5], mode="markers+text",
        marker=dict(color="#D92D20", size=16, symbol="circle"),
        text=[f"Est: {est_dist_m:.2f} m"], textposition="bottom center",
        name="Estimated", textfont=dict(color="#D92D20", size=11)
    ))
    # true leak (if available)
    if true_dist_m is not None:
        fig.add_trace(go.Scatter(
            x=[true_dist_m], y=[0.5], mode="markers+text",
            marker=dict(color="#039855", size=10, symbol="diamond"),
            text=[f"True: {true_dist_m:.2f} m"], textposition="top center",
            name="True", textfont=dict(color="#039855", size=11)
        ))
    fig.update_layout(
        height=140,
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(range=[-0.5, pipe_len_m + 0.5], showgrid=False,
                   zeroline=False, title="Position along pipe (m)"),
        yaxis=dict(range=[0, 1.1], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        font=dict(size=12),
    )
    return fig


# ─────────────────────────────────────────────
# Alert history chart (confidence over runs)
# ─────────────────────────────────────────────
def fig_alert_history(history: list[dict]) -> go.Figure:
    if not history:
        return go.Figure()
    runs = [h["run"] for h in history]
    confs = [h["confidence"] for h in history]
    dists = [h["dist_m"] for h in history]
    colors = ["#E24B4A" if h["detected"] else "#1D9E75" for h in history]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig.add_trace(go.Bar(x=runs, y=confs, marker_color=colors,
                         name="Confidence", showlegend=False), row=1, col=1)
    fig.add_hline(y=st.session_state.get("vd_threshold", DEFAULT_THRESHOLD),
                  line=dict(color="#888", dash="dash", width=1),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=runs, y=dists, mode="lines+markers",
                             line=dict(color="#378ADD", width=1.5),
                             marker=dict(size=5), name="Est. dist (m)", showlegend=False),
                  row=2, col=1)
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis2_title="Run #",
        yaxis_title="Confidence",
        yaxis2_title="Dist (m)",
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


# ─────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────
def render():
    st.title("Live Vibration Dashboard")
    st.caption("Real-time acoustic leak simulation, threshold alerting, and TDOA localization.")

    # ── Sidebar controls ──────────────────────
    with st.sidebar:
        st.header("Vibration Dashboard Controls")
        pipe_len_m = st.slider("Pipe length (m)", 5.0, 50.0, 15.0, 1.0, key="vd_pipe_len")
        leak_pos_m = st.slider(
            "Leak position from Sensor A (m)",
            0.0, float(pipe_len_m), min(4.2, float(pipe_len_m)), 0.1, key="vd_leak_pos"
        )
        noise_std = st.slider("Noise level (std dev)", 0.0, 1.0, 0.25, 0.01, key="vd_noise")
        threshold = st.slider("Alert threshold (confidence)", 0.3, 1.0, DEFAULT_THRESHOLD, 0.01, key="vd_threshold")
        amplitude_threshold = st.slider("Amplitude alert level", 0.1, 2.0, 0.8, 0.05, key="vd_amp_thresh")
        random_seed = st.checkbox("Randomize seed each run", value=False, key="vd_random_seed")

        st.divider()
        col_run, col_live = st.columns(2)
        with col_run:
            run_once = st.button("Run once", use_container_width=True, type="primary")
        with col_live:
            live_toggle = st.button(
                "Stop live" if st.session_state.vd_running else "Start live",
                use_container_width=True
            )

        if run_once:
            seed = None if st.session_state.vd_random_seed else st.session_state.vd_run_count
            run_cycle(pipe_len_m, leak_pos_m, noise_std, threshold, seed)

        if live_toggle:
            st.session_state.vd_running = not st.session_state.vd_running

        if st.session_state.vd_running:
            seed = None if st.session_state.vd_random_seed else st.session_state.vd_run_count
            run_cycle(pipe_len_m, leak_pos_m, noise_std, threshold, seed)
            time.sleep(0.3)
            st.rerun()

        if st.session_state.vd_alert_history:
            st.divider()
            if st.button("Clear history", use_container_width=True):
                st.session_state.vd_alert_history = []
                st.rerun()

    # ── No data yet ───────────────────────────
    if st.session_state.vd_result is None:
        st.info("Use the sidebar to run a simulation. Click **Start live** for continuous updates.")
        return

    result: loc_alg.LocatorResult = st.session_state.vd_result
    t, a, b = st.session_state.vd_signals
    meta = st.session_state.vd_meta
    lags_w = st.session_state.vd_lags
    corr_w = st.session_state.vd_corr
    peak_lag = st.session_state.vd_peak_lag
    fs_hz = int(meta["fs_hz"])

    # ── Alert banner ──────────────────────────
    peak_amp = float(np.max(np.abs(a[:int(DISPLAY_WINDOW_S * fs_hz)])))
    amp_alert = peak_amp > amplitude_threshold

    if result.leak_detected:
        st.markdown(
            f"""<div style="padding:14px 18px;border-radius:10px;border:2px solid #D92D20;
                background:#FFF1F0;margin-bottom:12px;">
              <span style="font-size:20px;font-weight:800;color:#B42318;">⚠ LEAK DETECTED</span>
              <span style="font-size:15px;color:#7A271A;margin-left:12px;">
                Confidence <b>{result.confidence_score:.3f}</b> — estimated
                <b>{result.estimated_distance_m:.2f} m</b> from Sensor A
              </span>
            </div>""",
            unsafe_allow_html=True,
        )
    elif amp_alert:
        st.markdown(
            f"""<div style="padding:14px 18px;border-radius:10px;border:2px solid #DC6803;
                background:#FFFAEB;margin-bottom:12px;">
              <span style="font-size:18px;font-weight:700;color:#93370D;">⚡ AMPLITUDE WARNING</span>
              <span style="font-size:14px;color:#7A2E0E;margin-left:12px;">
                Peak amplitude <b>{peak_amp:.3f}</b> exceeds threshold <b>{amplitude_threshold:.2f}</b>
              </span>
            </div>""",
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

    # ── Metric cards ──────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Detected freq", f"{result.detected_frequency_hz:.1f} Hz")
    c2.metric("Confidence", f"{result.confidence_score:.3f}",
              delta="LEAK" if result.leak_detected else "OK",
              delta_color="inverse" if result.leak_detected else "normal")
    c3.metric("Est. distance (A)", f"{result.estimated_distance_m:.2f} m")
    c4.metric("TDOA", f"{result.tdoa_s * 1000:.3f} ms")
    c5.metric("Peak amplitude", f"{peak_amp:.3f}",
              delta="ABOVE THRESHOLD" if amp_alert else "normal",
              delta_color="inverse" if amp_alert else "off")

    st.markdown("---")

    # ── Row 1: Time series + Pipe map ─────────
    col_ts, col_pipe = st.columns([2, 1])
    with col_ts:
        st.subheader(f"Sensor waveforms (first {int(DISPLAY_WINDOW_S*1000)} ms)")
        st.plotly_chart(
            fig_timeseries(t, a, b, fs_hz, DISPLAY_WINDOW_S, amplitude_threshold),
            use_container_width=True, config={"displayModeBar": False}
        )
    with col_pipe:
        st.subheader("Pipe map")
        st.plotly_chart(
            fig_pipe_map(
                float(meta["sensor_spacing_m"]),
                float(result.estimated_distance_m),
                result.true_distance_m,
            ),
            use_container_width=True, config={"displayModeBar": False}
        )
        if result.true_distance_m is not None and result.error_m is not None:
            st.caption(f"True: {result.true_distance_m:.2f} m — Error: {result.error_m:.3f} m")

    # ── Row 2: FFT + Cross-correlation ────────
    col_fft, col_xc = st.columns(2)
    with col_fft:
        st.subheader("FFT spectrum (0–800 Hz)")
        st.plotly_chart(
            fig_fft(a, b, fs_hz, float(meta["leak_frequency_hz"])),
            use_container_width=True, config={"displayModeBar": False}
        )
    with col_xc:
        st.subheader("Cross-correlation (TDOA)")
        st.plotly_chart(
            fig_xcorr(lags_w, corr_w, fs_hz, int(peak_lag)),
            use_container_width=True, config={"displayModeBar": False}
        )

    # ── Row 3: Alert history ──────────────────
    if st.session_state.vd_alert_history:
        st.markdown("---")
        st.subheader(f"Alert history ({len(st.session_state.vd_alert_history)} runs)")
        st.plotly_chart(
            fig_alert_history(st.session_state.vd_alert_history),
            use_container_width=True, config={"displayModeBar": False}
        )

        # Last 10 events table
        hist_rev = list(reversed(st.session_state.vd_alert_history[-10:]))
        rows_md = "| Run | Time | Confidence | Est. dist (m) | Status |\n|---|---|---|---|---|\n"
        for h in hist_rev:
            status = "🔴 LEAK" if h["detected"] else "🟢 OK"
            rows_md += f"| {h['run']} | {h['ts']} | {h['confidence']:.3f} | {h['dist_m']:.2f} | {status} |\n"
        st.markdown(rows_md)

    # ── Live indicator ────────────────────────
    if st.session_state.vd_running:
        st.caption(f"🔴 Live — run #{st.session_state.vd_run_count} — refreshing…")
    else:
        st.caption(f"Run #{st.session_state.vd_run_count} — click **Start live** in sidebar for continuous updates.")


# ─────────────────────────────────────────────
# Entry: standalone or imported by app.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="Vibration Dashboard — Acoustic Leak-Net", layout="wide")
    render()
