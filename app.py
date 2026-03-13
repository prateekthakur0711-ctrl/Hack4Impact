from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from algorithms import sim as sim_alg
from algorithms import locator as loc_alg


# ─────────────────────────────────────────────
# Shared helpers (unchanged from original)
# ─────────────────────────────────────────────
def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def pipe_position_figure(pipe_len_m: float, leak_x_m: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 1.5), constrained_layout=True)
    ax.set_xlim(0, pipe_len_m)
    ax.set_ylim(0, 1)
    ax.hlines(0.5, 0, pipe_len_m, linewidth=12, color="#D0D5DD")
    ax.vlines([0, pipe_len_m], 0.35, 0.65, linewidth=4, color="#667085")
    ax.scatter([leak_x_m], [0.5], s=200, color="#D92D20", zorder=3)
    ax.text(0, 0.85, "Sensor A", ha="left", va="center", fontsize=10)
    ax.text(pipe_len_m, 0.85, "Sensor B", ha="right", va="center", fontsize=10)
    ax.text(leak_x_m, 0.15, f"{leak_x_m:.2f} m", ha="center", va="center", fontsize=10, color="#D92D20")
    ax.axis("off")
    return fig


def social_impact_metrics(
    leak_detected: bool,
    leak_rate_l_s: float,
    manual_response_min: float,
    response_improvement_frac: float,
    cost_inr_per_kl: float,
    liters_per_person_per_day: float,
) -> dict:
    if not leak_detected:
        return {
            "water_saved_liters": 0.0,
            "cost_saved_inr": 0.0,
            "people_served_days": 0.0,
        }
    manual_s = float(manual_response_min) * 60.0
    automated_s = manual_s * (1.0 - float(response_improvement_frac))
    automated_s = max(automated_s, 0.0)
    time_saved_s = max(manual_s - automated_s, 0.0)
    water_saved_l = float(leak_rate_l_s) * time_saved_s
    cost_saved_inr = (water_saved_l / 1000.0) * float(cost_inr_per_kl)
    people_served_days = water_saved_l / max(float(liters_per_person_per_day), 1e-6)
    return {
        "water_saved_liters": water_saved_l,
        "cost_saved_inr": cost_saved_inr,
        "people_served_days": people_served_days,
        "assumptions": {
            "leak_rate_l_s": float(leak_rate_l_s),
            "manual_response_min": float(manual_response_min),
            "response_improvement_frac": float(response_improvement_frac),
            "cost_inr_per_kl": float(cost_inr_per_kl),
            "liters_per_person_per_day": float(liters_per_person_per_day),
        },
    }


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Acoustic Leak‑Net — Mission Control", layout="wide")
st.title("Acoustic Leak‑Net — Mission Control")
st.caption("Simulate a leak, detect it, localize it, and monitor live vibration data.")

# ─────────────────────────────────────────────
# Top-level tabs
# ─────────────────────────────────────────────
tab_sim, tab_live = st.tabs(["Simulation & Report", "Live Vibration Dashboard"])


# ══════════════════════════════════════════════
# TAB 1 — original simulation flow (unchanged)
# ══════════════════════════════════════════════
with tab_sim:
    with st.sidebar:
        st.header("Simulation Controls")
        pipe_len_m = st.slider("Pipe Length (m)", min_value=5.0, max_value=50.0, value=15.0, step=1.0)
        leak_pos_m = st.slider("Leak Position from Sensor A (m)", min_value=0.0, max_value=float(pipe_len_m), value=min(4.2, float(pipe_len_m)), step=0.1)
        noise_std = st.slider("Noise Level (std dev)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

        st.divider()
        st.header("Impact Assumptions")
        leak_rate_l_s = st.number_input("Leak rate (liters/sec)", value=2.5, min_value=0.0, step=0.1)
        manual_response_min = st.number_input("Manual response time (minutes)", value=60.0, min_value=1.0, step=5.0)
        response_improvement_frac = st.slider("Response time improvement", min_value=0.0, max_value=0.95, value=0.80, step=0.05, format="%.2f")
        cost_inr_per_kl = st.number_input("Cost of water (INR per kL)", value=20.0, min_value=0.0, step=1.0)
        liters_per_person_per_day = st.number_input("Avg daily consumption (L/person/day)", value=135.0, min_value=1.0, step=5.0)

        st.divider()
        run = st.button("Run Simulation + Localization", type="primary", use_container_width=True)

    if run:
        progress = st.progress(0, text="Scanning pipe network…")
        for i in range(1, 101):
            progress.progress(i, text="Scanning pipe network…")

        sim_cfg = sim_alg.SimulationConfig(
            fs_hz=44100,
            duration_s=2.0,
            sensor_spacing_m=float(pipe_len_m),
            leak_frequency_hz=240.0,
            speed_of_sound_m_s=1480.0,
            leak_distance_from_a_m=float(leak_pos_m),
            noise_std=float(noise_std),
        )
        t, a, b, meta = sim_alg.generate_signals(sim_cfg)

        loc_cfg = loc_alg.LocatorConfig(
            fs_hz=sim_cfg.fs_hz,
            speed_of_sound_m_s=sim_cfg.speed_of_sound_m_s,
            sensor_spacing_m=sim_cfg.sensor_spacing_m,
            target_frequency_hz=sim_cfg.leak_frequency_hz,
            band_half_width_hz=25.0,
            confidence_threshold=0.7,
        )
        result, impact, (lags_w, corr_w, peak_lag) = loc_alg.localize_from_arrays(a, b, loc_cfg, meta=meta)

        impact_social = social_impact_metrics(
            leak_detected=result.leak_detected,
            leak_rate_l_s=float(leak_rate_l_s),
            manual_response_min=float(manual_response_min),
            response_improvement_frac=float(response_improvement_frac),
            cost_inr_per_kl=float(cost_inr_per_kl),
            liters_per_person_per_day=float(liters_per_person_per_day),
        )

        if result.leak_detected:
            st.markdown(
                f"""
                <div style="padding:16px;border-radius:12px;border:2px solid #D92D20;background:#FFFBFA;">
                  <div style="font-size:22px;font-weight:800;color:#B42318;">LEAK ALERT</div>
                  <div style="margin-top:6px;font-size:16px;">
                    Leak detected with confidence <b>{result.confidence_score:.3f}</b>.
                    Estimated location: <b>{result.estimated_distance_m:.2f} m</b> from Sensor A.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No leak detected (below threshold). Adjust noise/leak position and rerun.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detected Frequency", f"{result.detected_frequency_hz:.2f} Hz")
        c2.metric("Confidence", f"{result.confidence_score:.3f}")
        c3.metric("Estimated Distance (from A)", f"{result.estimated_distance_m:.2f} m")
        c4.metric("TDOA", f"{result.tdoa_s*1000:.3f} ms")

        st.subheader("Pipe Map")
        pipe_fig = pipe_position_figure(float(pipe_len_m), float(result.estimated_distance_m))
        st.pyplot(pipe_fig, use_container_width=True)

        st.subheader("Real‑time Sensor Plots (first 0.05 s)")
        n = int(min(len(t), 0.05 * sim_cfg.fs_hz))
        fig_td, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(t[:n], a[:n], label="Sensor A", linewidth=1.0)
        ax.plot(t[:n], b[:n], label="Sensor B", linewidth=1.0, alpha=0.85)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend()
        st.pyplot(fig_td, use_container_width=True)

        st.subheader("Cross‑Correlation (TDOA)")
        corr_fig, axc = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
        lag_s = lags_w / float(loc_cfg.fs_hz)
        axc.plot(lag_s, corr_w, linewidth=1.0)
        axc.axvline(int(peak_lag) / float(loc_cfg.fs_hz), color="k", linestyle="--", linewidth=1.0, alpha=0.7)
        axc.set_xlabel("Lag (s)")
        axc.set_ylabel("Correlation")
        axc.grid(True, alpha=0.25)
        st.pyplot(corr_fig, use_container_width=True)

        st.subheader("Social Impact Calculator")
        i1, i2, i3 = st.columns(3)
        i1.metric("Water Saved", f"{impact_social['water_saved_liters']:.0f} L")
        i2.metric("Cost Saved", f"₹{impact_social['cost_saved_inr']:.0f}")
        i3.metric("People Served (person‑days)", f"{impact_social['people_served_days']:.1f}")

        st.subheader("Export")
        report_md = f"""# Acoustic Leak‑Net — Pitch Report

## Summary
- Leak detected: **{result.leak_detected}**
- Confidence score: **{result.confidence_score:.3f}**
- Detected frequency: **{result.detected_frequency_hz:.2f} Hz**
- Estimated distance from Sensor A: **{result.estimated_distance_m:.2f} m**
- TDOA (B-A): **{result.tdoa_s:.6f} s**

## Social Impact (assumptions in sidebar)
- Water saved: **{impact_social['water_saved_liters']:.0f} L**
- Cost saved: **₹{impact_social['cost_saved_inr']:.0f}**
- People served: **{impact_social['people_served_days']:.1f} person-days**

## Files
This export includes:
- `results.json`
- `sensor_timeseries.png`
- `xcorr.png`
- `pipe_map.png`
"""
        sensor_png = fig_to_png_bytes(fig_td)
        xcorr_png = fig_to_png_bytes(corr_fig)
        pipe_png = fig_to_png_bytes(pipe_position_figure(float(pipe_len_m), float(result.estimated_distance_m)))

        results_payload = {
            **json.loads(json.dumps(asdict(result))),
            "meta": meta,
            "impact_metrics": impact,
            "social_impact": impact_social,
        }

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("report.md", report_md)
            z.writestr("results.json", json.dumps(results_payload, indent=2, sort_keys=True))
            z.writestr("sensor_timeseries.png", sensor_png)
            z.writestr("xcorr.png", xcorr_png)
            z.writestr("pipe_map.png", pipe_png)
        zip_bytes = zip_buf.getvalue()

        st.download_button(
            "Download Pitch Report (ZIP)",
            data=zip_bytes,
            file_name="acoustic_leak_net_pitch_report.zip",
            mime="application/zip",
            use_container_width=True,
        )

    else:
        st.warning("Use the sidebar and click **Run Simulation + Localization** to start the demo.")


# ══════════════════════════════════════════════
# TAB 2 — Live Vibration Dashboard
# ══════════════════════════════════════════════
with tab_live:
    # Import and render the dashboard inline
    from vibration_dashboard import render as render_vibration_dashboard
    render_vibration_dashboard()