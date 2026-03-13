"""
Live Vibration Dashboard — Acoustic Leak-Net
Integrates with algorithms/sim.py and algorithms/locator.py.
Run standalone: streamlit run vibration_dashboard.py
Or imported by app.py via: from vibration_dashboard import render
"""

from __future__ import annotations

import time
import math
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
DISPLAY_WINDOW_S = 0.05
FFT_MAX_FREQ = 800
DEFAULT_THRESHOLD = 0.70


# ─────────────────────────────────────────────
# City block network — schematic grid
# ─────────────────────────────────────────────
CITY_SENSORS = {
    "S01": {"x": 1.0, "y": 5.0, "street": "Main St / 1st Ave"},
    "S02": {"x": 3.0, "y": 5.0, "street": "Main St / 2nd Ave"},
    "S03": {"x": 5.0, "y": 5.0, "street": "Main St / 3rd Ave"},
    "S04": {"x": 7.0, "y": 5.0, "street": "Main St / 4th Ave"},
    "S05": {"x": 1.0, "y": 3.0, "street": "Oak St / 1st Ave"},
    "S06": {"x": 3.0, "y": 3.0, "street": "Oak St / 2nd Ave"},
    "S07": {"x": 5.0, "y": 3.0, "street": "Oak St / 3rd Ave"},
    "S08": {"x": 7.0, "y": 3.0, "street": "Oak St / 4th Ave"},
    "S09": {"x": 1.0, "y": 1.0, "street": "Elm St / 1st Ave"},
    "S10": {"x": 3.0, "y": 1.0, "street": "Elm St / 2nd Ave"},
    "S11": {"x": 5.0, "y": 1.0, "street": "Elm St / 3rd Ave"},
    "S12": {"x": 7.0, "y": 1.0, "street": "Elm St / 4th Ave"},
}

CITY_PIPES = [
    ("S01", "S02"), ("S02", "S03"), ("S03", "S04"),
    ("S05", "S06"), ("S06", "S07"), ("S07", "S08"),
    ("S09", "S10"), ("S10", "S11"), ("S11", "S12"),
    ("S01", "S05"), ("S05", "S09"),
    ("S02", "S06"), ("S06", "S10"),
    ("S03", "S07"), ("S07", "S11"),
    ("S04", "S08"), ("S08", "S12"),
]

# All valid sensor pairs (only adjacent pipe segments)
VALID_PAIRS = [f"{a} → {b}" for a, b in CITY_PIPES]
PAIR_MAP = {f"{a} → {b}": (a, b) for a, b in CITY_PIPES}


# ─────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "vd_running": False,
        "vd_result": None,
        "vd_meta": None,
        "vd_signals": None,
        "vd_lags": None,
        "vd_corr": None,
        "vd_peak_lag": None,
        "vd_alert_history": [],
        "vd_run_count": 0,
        "vd_active_pair_key": "S02 → S03",
        "vd_leak_history_map": {},  # sensor_pair_key -> last result
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

    # Store result on the map for the active sensor pair
    pair_key = st.session_state.vd_active_pair_key
    st.session_state.vd_leak_history_map[pair_key] = {
        "detected": bool(result.leak_detected),
        "confidence": float(result.confidence_score),
        "dist_m": float(result.estimated_distance_m),
        "run": st.session_state.vd_run_count,
    }

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
    freqs = sp_fft.rfftfreq(n, d=1.0 / float(fs_hz))
    mag = np.abs(X) / max(n, 1)
    return freqs, mag


# ─────────────────────────────────────────────
# Plotly: city block locality map
# ─────────────────────────────────────────────
def fig_city_map(
    active_pair_key: str,
    leak_history: dict,
    leak_detected: bool,
    est_dist_m: float,
    pipe_len_m: float,
) -> go.Figure:
    fig = go.Figure()
    active_a, active_b = PAIR_MAP[active_pair_key]

    # ── Draw building blocks (background rectangles) ──
    block_coords = [
        (1.5, 2.5, 3.5, 4.5), (3.5, 4.5, 5.5, 6.0),  # row 1 blocks between Main & Oak
        (1.5, 2.5, 5.5, 6.0), (5.5, 6.5, 3.5, 4.5),
        (1.5, 2.5, 1.5, 2.5), (3.5, 4.5, 1.5, 2.5),  # row 2 blocks between Oak & Elm
        (5.5, 6.5, 1.5, 2.5), (1.5, 2.5, 3.5, 4.5),
    ]
    for x0, x1, y0, y1 in block_coords:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor="rgba(200,210,220,0.25)",
                      line=dict(color="rgba(150,160,170,0.3)", width=0.5))

    # ── Draw street labels ──
    street_labels = [
        (0.1, 5.0, "Main St", 10),
        (0.1, 3.0, "Oak St", 10),
        (0.1, 1.0, "Elm St", 10),
        (1.0, 5.85, "1st", 9),
        (3.0, 5.85, "2nd", 9),
        (5.0, 5.85, "3rd", 9),
        (7.0, 5.85, "4th", 9),
    ]
    for x, y, text, size in street_labels:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                           font=dict(size=size, color="#8899AA"),
                           xanchor="center", yanchor="middle")

    # ── Draw all pipe segments ──
    for seg_a, seg_b in CITY_PIPES:
        sa = CITY_SENSORS[seg_a]
        sb = CITY_SENSORS[seg_b]
        pair_key = f"{seg_a} → {seg_b}"
        seg_result = leak_history.get(pair_key, None)

        if pair_key == active_pair_key:
            color = "#E24B4A" if leak_detected else "#1D9E75"
            width = 4
            dash = "solid"
        elif seg_result and seg_result["detected"]:
            color = "#E24B4A"
            width = 2.5
            dash = "solid"
        elif seg_result and not seg_result["detected"]:
            color = "#1D9E75"
            width = 2
            dash = "solid"
        else:
            color = "#B0BEC5"
            width = 1.5
            dash = "dot"

        fig.add_trace(go.Scatter(
            x=[sa["x"], sb["x"]], y=[sa["y"], sb["y"]],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="skip",
            showlegend=False,
        ))

    # ── Draw estimated leak position on active pipe ──
    if leak_detected and est_dist_m is not None:
        sa = CITY_SENSORS[active_a]
        sb = CITY_SENSORS[active_b]
        dx = sb["x"] - sa["x"]
        dy = sb["y"] - sa["y"]
        seg_len = math.sqrt(dx*dx + dy*dy)
        frac = min(max(est_dist_m / max(pipe_len_m, 0.01), 0), 1)
        lx = sa["x"] + frac * dx
        ly = sa["y"] + frac * dy

        # Pulsing leak marker — outer ring
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="markers",
            marker=dict(color="rgba(226,75,74,0.25)", size=28, symbol="circle"),
            hoverinfo="skip", showlegend=False,
        ))
        # Inner dot
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="markers+text",
            marker=dict(color="#E24B4A", size=14, symbol="circle",
                        line=dict(color="white", width=2)),
            text=[f"  Leak ~{est_dist_m:.1f}m from {active_a}"],
            textposition="middle right",
            textfont=dict(color="#E24B4A", size=11),
            hovertemplate=f"<b>Leak detected</b><br>~{est_dist_m:.1f} m from {active_a}<extra></extra>",
            showlegend=False,
        ))

    # ── Draw all sensors ──
    for sid, s in CITY_SENSORS.items():
        seg_result = leak_history.get(active_pair_key, None)
        is_active_sensor = (sid == active_a or sid == active_b)
        was_leak = leak_detected and is_active_sensor

        if was_leak:
            color = "#E24B4A"
            size = 18
            border = "white"
            border_w = 2.5
        elif is_active_sensor:
            color = "#1D9E75"
            size = 18
            border = "white"
            border_w = 2.5
        else:
            # Check if this sensor appeared in any historical leak
            involved = any(
                sid in (PAIR_MAP[k][0], PAIR_MAP[k][1])
                for k, v in leak_history.items() if v["detected"]
            )
            color = "#FF8A65" if involved else "#607D8B"
            size = 13
            border = "white"
            border_w = 1.5

        fig.add_trace(go.Scatter(
            x=[s["x"]], y=[s["y"]],
            mode="markers+text",
            marker=dict(
                color=color, size=size, symbol="circle",
                line=dict(color=border, width=border_w)
            ),
            text=[sid],
            textposition="top center",
            textfont=dict(
                size=10,
                color="#E24B4A" if was_leak else ("#1D9E75" if is_active_sensor else "#546E7A")
            ),
            hovertemplate=f"<b>{sid}</b><br>{s['street']}<br>"
                          + ("<b>ACTIVE SENSOR</b>" if is_active_sensor else "Monitoring") + "<extra></extra>",
            showlegend=False,
        ))

    # ── Legend annotations ──
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers+lines",
                             marker=dict(color="#E24B4A", size=10),
                             line=dict(color="#E24B4A", width=3),
                             name="Leak detected"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers+lines",
                             marker=dict(color="#1D9E75", size=10),
                             line=dict(color="#1D9E75", width=3),
                             name="Active / clear"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="#B0BEC5", width=2, dash="dot"),
                             name="Unmonitored pipe"))

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=30, b=20),
        plot_bgcolor="rgba(245,247,250,0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-0.3, 8.5], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[0.0, 6.3], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
        hovermode="closest",
        font=dict(size=11),
    )
    return fig


# ─────────────────────────────────────────────
# Plotly: sensor waveforms
# ─────────────────────────────────────────────
def fig_timeseries(t, a, b, fs_hz, window_s, threshold_amp):
    n = int(min(len(t), window_s * fs_hz))
    t_s, a_s, b_s = t[:n], a[:n], b[:n]
    peak_amp = max(float(np.max(np.abs(a_s))), float(np.max(np.abs(b_s))))
    above = peak_amp > threshold_amp

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_s, y=a_s, name="Sensor A",
                             line=dict(color="#378ADD", width=1.5), mode="lines"))
    fig.add_trace(go.Scatter(x=t_s, y=b_s, name="Sensor B",
                             line=dict(color="#1D9E75", width=1.5), mode="lines", opacity=0.85))
    fig.add_hline(y=threshold_amp, line=dict(color="#E24B4A", dash="dash", width=1.2),
                  annotation_text="threshold", annotation_position="top right")
    fig.add_hline(y=-threshold_amp, line=dict(color="#E24B4A", dash="dash", width=1.2))
    if above:
        fig.add_hrect(y0=threshold_amp, y1=max(float(np.max(a_s)), threshold_amp + 0.1),
                      fillcolor="rgba(226,75,74,0.07)", line_width=0)
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        font=dict(size=12),
    )
    return fig


def fig_fft(a, b, fs_hz, target_hz):
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
        height=220, margin=dict(l=0, r=0, t=8, b=0),
        xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        font=dict(size=12),
    )
    return fig


def fig_xcorr(lags_w, corr_w, fs_hz, peak_lag):
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
        height=220, margin=dict(l=0, r=0, t=8, b=0),
        xaxis_title="Lag (s)  [+ve → B later than A]", yaxis_title="Correlation",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        font=dict(size=12),
    )
    return fig


def fig_pipe_map(pipe_len_m, est_dist_m, true_dist_m):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=pipe_len_m, y0=0.35, y1=0.65,
                  fillcolor="#D0D5DD", line=dict(color="#667085", width=2))
    for x, label in [(0, "Sensor A"), (pipe_len_m, "Sensor B")]:
        fig.add_shape(type="line", x0=x, x1=x, y0=0.25, y1=0.75,
                      line=dict(color="#667085", width=4))
        fig.add_annotation(x=x, y=0.85, text=label, showarrow=False, font=dict(size=12))
    fig.add_trace(go.Scatter(
        x=[est_dist_m], y=[0.5], mode="markers+text",
        marker=dict(color="#D92D20", size=16, symbol="circle"),
        text=[f"Est: {est_dist_m:.2f} m"], textposition="bottom center",
        name="Estimated", textfont=dict(color="#D92D20", size=11)
    ))
    if true_dist_m is not None:
        fig.add_trace(go.Scatter(
            x=[true_dist_m], y=[0.5], mode="markers+text",
            marker=dict(color="#039855", size=10, symbol="diamond"),
            text=[f"True: {true_dist_m:.2f} m"], textposition="top center",
            name="True", textfont=dict(color="#039855", size=11)
        ))
    fig.update_layout(
        height=140, margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(range=[-0.5, pipe_len_m + 0.5], showgrid=False,
                   zeroline=False, title="Position along pipe (m)"),
        yaxis=dict(range=[0, 1.1], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        font=dict(size=12),
    )
    return fig


def fig_alert_history(history):
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
                  line=dict(color="#888", dash="dash", width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=runs, y=dists, mode="lines+markers",
                             line=dict(color="#378ADD", width=1.5),
                             marker=dict(size=5), name="Est. dist (m)", showlegend=False),
                  row=2, col=1)
    fig.update_layout(
        height=220, margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis2_title="Run #", yaxis_title="Confidence", yaxis2_title="Dist (m)",
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


# ─────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────
def render():
    st.title("Live Vibration Dashboard")
    st.caption("Real-time acoustic leak simulation, threshold alerting, TDOA localization, and city-block sensor map.")

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.header("Vibration Dashboard Controls")

        # Sensor pair selector — drives which pipe segment is monitored
        pair_key = st.selectbox(
            "Active sensor pair (pipe segment)",
            options=VALID_PAIRS,
            index=VALID_PAIRS.index(st.session_state.vd_active_pair_key)
            if st.session_state.vd_active_pair_key in VALID_PAIRS else 2,
            key="vd_pair_select",
        )
        st.session_state.vd_active_pair_key = pair_key
        active_a, active_b = PAIR_MAP[pair_key]
        st.caption(f"Sensor A: **{active_a}** ({CITY_SENSORS[active_a]['street']})")
        st.caption(f"Sensor B: **{active_b}** ({CITY_SENSORS[active_b]['street']})")

        st.divider()
        pipe_len_m    = st.slider("Pipe segment length (m)", 5.0, 50.0, 15.0, 1.0, key="vd_pipe_len")
        leak_pos_m    = st.slider("Leak position from Sensor A (m)", 0.0, float(pipe_len_m),
                                   min(4.2, float(pipe_len_m)), 0.1, key="vd_leak_pos")
        noise_std     = st.slider("Noise level (std dev)", 0.0, 1.0, 0.25, 0.01, key="vd_noise")
        threshold     = st.slider("Alert threshold (confidence)", 0.3, 1.0, DEFAULT_THRESHOLD, 0.01, key="vd_threshold")
        amp_threshold = st.slider("Amplitude alert level", 0.1, 2.0, 0.8, 0.05, key="vd_amp_thresh")
        random_seed   = st.checkbox("Randomize seed each run", value=False, key="vd_random_seed")

        st.divider()
        col_run, col_live = st.columns(2)
        with col_run:
            run_once = st.button("Run once", use_container_width=True, type="primary")
        with col_live:
            live_label = "Stop live" if st.session_state.vd_running else "Start live"
            live_toggle = st.button(live_label, use_container_width=True)

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
                st.session_state.vd_leak_history_map = {}
                st.rerun()

    # ── No data yet ───────────────────────────
    if st.session_state.vd_result is None:
        st.info("Select a sensor pair in the sidebar and click **Run once** or **Start live** to begin.")

        # Still show the city map even before first run
        st.markdown("---")
        st.subheader("City block sensor network")
        st.caption("Select an active sensor pair from the sidebar to monitor a pipe segment.")
        st.plotly_chart(
            fig_city_map(
                st.session_state.vd_active_pair_key,
                st.session_state.vd_leak_history_map,
                False, 0.0, 15.0,
            ),
            use_container_width=True, config={"displayModeBar": False}
        )
        return

    result    = st.session_state.vd_result
    t, a, b   = st.session_state.vd_signals
    meta      = st.session_state.vd_meta
    lags_w    = st.session_state.vd_lags
    corr_w    = st.session_state.vd_corr
    peak_lag  = st.session_state.vd_peak_lag
    fs_hz     = int(meta["fs_hz"])
    peak_amp  = float(np.max(np.abs(a[:int(DISPLAY_WINDOW_S * fs_hz)])))
    amp_alert = peak_amp > amp_threshold

    # ── Alert banner ──────────────────────────
    if result.leak_detected:
        st.markdown(
            f"""<div style="padding:14px 18px;border-radius:10px;border:2px solid #D92D20;
                background:#FFF1F0;margin-bottom:12px;">
              <span style="font-size:20px;font-weight:800;color:#B42318;">⚠ LEAK DETECTED</span>
              <span style="font-size:15px;color:#7A271A;margin-left:12px;">
                Segment <b>{st.session_state.vd_active_pair_key}</b> —
                Confidence <b>{result.confidence_score:.3f}</b> —
                estimated <b>{result.estimated_distance_m:.2f} m</b> from {active_a}
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
                Peak amplitude <b>{peak_amp:.3f}</b> exceeds threshold <b>{amp_threshold:.2f}</b>
              </span>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style="padding:14px 18px;border-radius:10px;border:1.5px solid #039855;
                background:#F6FEF9;margin-bottom:12px;">
              <span style="font-size:16px;font-weight:600;color:#027A48;">
                ✓ Segment {st.session_state.vd_active_pair_key} — nominal, no leak detected
              </span>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Metrics ───────────────────────────────
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

    # ── Row 1: City map (left, large) + Pipe map (right) ──
    col_city, col_pipe = st.columns([3, 2])
    with col_city:
        st.subheader("City block sensor network")
        st.plotly_chart(
            fig_city_map(
                st.session_state.vd_active_pair_key,
                st.session_state.vd_leak_history_map,
                bool(result.leak_detected),
                float(result.estimated_distance_m),
                float(meta["sensor_spacing_m"]),
            ),
            use_container_width=True, config={"displayModeBar": False}
        )
        total_leaks = sum(1 for v in st.session_state.vd_leak_history_map.values() if v["detected"])
        total_checked = len(st.session_state.vd_leak_history_map)
        if total_checked > 0:
            st.caption(f"{total_leaks} leak(s) detected across {total_checked} pipe segment(s) monitored so far.")

    with col_pipe:
        st.subheader("Active segment — pipe map")
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

        # Sensor status legend
        st.markdown("**Sensor legend**")
        if result.leak_detected:
            st.markdown(f"🔴 **{active_a}** — leak side (closer)")
            st.markdown(f"🔴 **{active_b}** — leak side (further)")
        else:
            st.markdown(f"🟢 **{active_a}** — active, clear")
            st.markdown(f"🟢 **{active_b}** — active, clear")
        other = [s for s in CITY_SENSORS if s not in (active_a, active_b)]
        st.markdown(f"⚪ {', '.join(other)} — standby")

    st.markdown("---")

    # ── Row 2: Waveforms ──────────────────────
    st.subheader(f"Sensor waveforms — {active_a} & {active_b} (first {int(DISPLAY_WINDOW_S*1000)} ms)")
    st.plotly_chart(
        fig_timeseries(t, a, b, fs_hz, DISPLAY_WINDOW_S, amp_threshold),
        use_container_width=True, config={"displayModeBar": False}
    )

    # ── Row 3: FFT + xcorr ────────────────────
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

    # ── Row 4: Alert history ──────────────────
    if st.session_state.vd_alert_history:
        st.markdown("---")
        st.subheader(f"Alert history ({len(st.session_state.vd_alert_history)} runs)")
        st.plotly_chart(
            fig_alert_history(st.session_state.vd_alert_history),
            use_container_width=True, config={"displayModeBar": False}
        )
        hist_rev = list(reversed(st.session_state.vd_alert_history[-10:]))
        rows_md = "| Run | Time | Segment | Confidence | Est. dist (m) | Status |\n|---|---|---|---|---|---|\n"
        for h in hist_rev:
            status = "🔴 LEAK" if h["detected"] else "🟢 OK"
            rows_md += f"| {h['run']} | {h['ts']} | {h.get('pair','—')} | {h['confidence']:.3f} | {h['dist_m']:.2f} | {status} |\n"
        st.markdown(rows_md)

    # ── Live indicator ────────────────────────
    if st.session_state.vd_running:
        st.caption(f"🔴 Live — run #{st.session_state.vd_run_count} — refreshing…")
    else:
        st.caption(f"Run #{st.session_state.vd_run_count} — click **Start live** in sidebar for continuous updates.")


# ─────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="Vibration Dashboard — Acoustic Leak-Net", layout="wide")
    render()