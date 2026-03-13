"""
Localization (DSP) engine for Acoustic Leak-Net.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy import fft as sp_fft
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LocatorConfig:
    fs_hz: int = 44100
    speed_of_sound_m_s: float = 1480.0
    sensor_spacing_m: float = 15.0
    target_frequency_hz: float = 240.0
    band_half_width_hz: float = 25.0
    bandpass_order: int = 4
    max_tdoa_s: float | None = None  # if None, derived from spacing/c
    confidence_threshold: float = 0.7
    freq_tolerance_hz: float = 15.0


@dataclass(frozen=True)
class LocatorResult:
    detected_frequency_hz: float
    peak_lag_samples: int
    tdoa_s: float
    estimated_distance_m: float
    confidence_score: float
    leak_detected: bool
    true_distance_m: float | None
    error_m: float | None
    sensor_spacing_m: float
    speed_of_sound_m_s: float


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    t = np.asarray(data["time"], dtype=np.float64)
    a = np.asarray(data["sensor_a_amplitude"], dtype=np.float64)
    b = np.asarray(data["sensor_b_amplitude"], dtype=np.float64)
    if t.size == 0:
        raise ValueError("CSV appears empty.")
    return t, a, b


def load_meta(path: Path) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def estimate_dominant_frequency(x: np.ndarray, fs_hz: int, fmin: float, fmax: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = x.size
    X = sp_fft.rfft(x * np.hanning(n))
    freqs = sp_fft.rfftfreq(n, d=1.0 / float(fs_hz))
    mag = np.abs(X)

    mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    if not np.any(mask):
        raise ValueError("Frequency search band is invalid for this FFT.")
    idx = np.argmax(mag[mask])
    return float(freqs[mask][idx])


def bandpass_filter(x: np.ndarray, fs_hz: int, f_low: float, f_high: float, order: int) -> np.ndarray:
    nyq = 0.5 * float(fs_hz)
    lo = max(float(f_low) / nyq, 1e-6)
    hi = min(float(f_high) / nyq, 0.999999)
    if not (0.0 < lo < hi < 1.0):
        raise ValueError("Bandpass cutoff frequencies are invalid.")
    sos = signal.butter(int(order), [lo, hi], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def fractional_delay_fft(x: np.ndarray, delay_s: float, fs_hz: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or delay_s == 0.0:
        return x.copy()
    X = sp_fft.rfft(x)
    freqs = sp_fft.rfftfreq(n, d=1.0 / float(fs_hz))
    phase = np.exp(-1j * 2.0 * np.pi * freqs * float(delay_s))
    return sp_fft.irfft(X * phase, n=n).astype(np.float64, copy=False)


def estimate_tdoa_from_tone_phase(
    a: np.ndarray,
    b: np.ndarray,
    fs_hz: int,
    target_frequency_hz: float,
    max_tdoa_s: float,
) -> tuple[float, float]:
    a0 = np.asarray(a, dtype=np.float64) - np.mean(a)
    b0 = np.asarray(b, dtype=np.float64) - np.mean(b)
    n = a0.size
    if n != b0.size or n < 16:
        raise ValueError("Signals must have same length and be non-trivial.")

    win = np.hanning(n)
    A = sp_fft.rfft(a0 * win)
    B = sp_fft.rfft(b0 * win)
    freqs = sp_fft.rfftfreq(n, d=1.0 / float(fs_hz))

    idx0 = int(np.argmin(np.abs(freqs - float(target_frequency_hz))))
    f0 = float(freqs[idx0])
    if f0 <= 0.0:
        raise ValueError("Invalid target frequency bin.")

    S0 = B[idx0] * np.conj(A[idx0])
    phi = float(np.angle(S0))
    tau0 = -phi / (2.0 * np.pi * f0)  # modulo 1/f0

    period = 1.0 / f0
    k_max = int(np.ceil(max_tdoa_s / period)) + 2
    candidates: list[float] = []
    for k in range(-k_max, k_max + 1):
        tau = tau0 + k * period
        if -max_tdoa_s <= tau <= max_tdoa_s:
            candidates.append(float(tau))
    if not candidates:
        candidates = [float(np.clip(tau0, -max_tdoa_s, max_tdoa_s))]

    a_norm = np.linalg.norm(a0)
    best_tau = candidates[0]
    best_score = -np.inf
    for tau in candidates:
        b_shift = fractional_delay_fft(b0, -tau, fs_hz)
        denom = a_norm * np.linalg.norm(b_shift)
        if denom == 0.0:
            continue
        score = float(np.dot(a0, b_shift) / denom)
        if score > best_score:
            best_score = score
            best_tau = tau
    return float(best_tau), float(best_score)


def pick_peak_near_expected(lags: np.ndarray, corr: np.ndarray, expected_lag: float, half_window: int) -> tuple[int, float]:
    lo = int(np.floor(expected_lag - half_window))
    hi = int(np.ceil(expected_lag + half_window))
    mask = (lags >= lo) & (lags <= hi)
    if not np.any(mask):
        idx = int(np.argmax(corr))
        return int(lags[idx]), float(corr[idx])
    l = lags[mask]
    c = corr[mask]
    idx = int(np.argmax(c))
    return int(l[idx]), float(c[idx])


def normalized_xcorr(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a0 = np.asarray(a, dtype=np.float64) - np.mean(a)
    b0 = np.asarray(b, dtype=np.float64) - np.mean(b)

    denom = np.linalg.norm(a0) * np.linalg.norm(b0)
    if denom == 0.0:
        raise ValueError("Cannot normalize correlation (zero-energy signal).")

    corr = signal.correlate(b0, a0, mode="full", method="fft") / denom
    lags = signal.correlation_lags(b0.size, a0.size, mode="full")
    return lags.astype(np.int64), corr.astype(np.float64)


def lag_to_distance_from_a(lag_samples_b_minus_a: int, fs_hz: int, speed_of_sound_m_s: float, sensor_spacing_m: float) -> float:
    tdoa_s = float(lag_samples_b_minus_a) / float(fs_hz)
    return float((float(sensor_spacing_m) - float(speed_of_sound_m_s) * tdoa_s) / 2.0)


def plot_correlation(lags: np.ndarray, corr: np.ndarray, fs_hz: int, peak_lag: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lag_s = lags / float(fs_hz)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
    ax.plot(lag_s, corr, linewidth=1.0)
    ax.axvline(peak_lag / float(fs_hz), color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.scatter([peak_lag / float(fs_hz)], [corr[lags == peak_lag][0]], zorder=3)
    ax.set_title("Normalized cross-correlation (bandpassed around leak tone)")
    ax.set_xlabel("Lag (seconds)  [positive => Sensor B later than A]")
    ax.set_ylabel("Normalized correlation")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def export_results_json(out_path: Path, result: LocatorResult, impact_metrics: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "leak_detected": bool(result.leak_detected),
        "confidence_score": float(result.confidence_score),
        "estimated_distance_m": float(result.estimated_distance_m),
        "detected_frequency_hz": float(result.detected_frequency_hz),
        "tdoa_s": float(result.tdoa_s),
        "peak_lag_samples": int(result.peak_lag_samples),
        "impact_metrics": impact_metrics,
    }
    if result.true_distance_m is not None:
        payload["true_distance_m"] = float(result.true_distance_m)
    if result.error_m is not None:
        payload["error_m"] = float(result.error_m)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def compute_impact_metrics(
    estimated_distance_m: float,
    leak_rate_l_s: float = 2.5,
    manual_response_s: float = 3600.0,
    response_time_improvement_frac: float = 0.80,
) -> dict:
    manual = float(manual_response_s)
    improved = manual * (1.0 - float(response_time_improvement_frac))
    improved = max(improved, 0.0)
    saved_time_s = max(manual - improved, 0.0)
    water_saved_l = float(leak_rate_l_s) * saved_time_s
    return {
        "estimated_water_saved_liters": water_saved_l,
        "assumptions": {
            "leak_rate_liters_per_sec": float(leak_rate_l_s),
            "manual_response_time_sec": manual,
            "response_time_improvement_fraction": float(response_time_improvement_frac),
            "automated_response_time_sec": improved,
            "time_saved_sec": saved_time_s,
            "estimated_distance_m": float(estimated_distance_m),
        },
    }


def localize_from_arrays(
    a: np.ndarray,
    b: np.ndarray,
    cfg: LocatorConfig,
    meta: dict | None = None,
) -> tuple[LocatorResult, dict, tuple[np.ndarray, np.ndarray, int]]:
    fmin = max(1.0, cfg.target_frequency_hz - 4.0 * cfg.band_half_width_hz)
    fmax = cfg.target_frequency_hz + 4.0 * cfg.band_half_width_hz
    dom_a = estimate_dominant_frequency(a, cfg.fs_hz, fmin, fmax)
    dom_b = estimate_dominant_frequency(b, cfg.fs_hz, fmin, fmax)
    detected_freq = 0.5 * (dom_a + dom_b)

    f_low = cfg.target_frequency_hz - cfg.band_half_width_hz
    f_high = cfg.target_frequency_hz + cfg.band_half_width_hz
    a_f = bandpass_filter(a, cfg.fs_hz, f_low, f_high, cfg.bandpass_order)
    b_f = bandpass_filter(b, cfg.fs_hz, f_low, f_high, cfg.bandpass_order)

    lags, corr = normalized_xcorr(a_f, b_f)

    max_tdoa_s = float(cfg.sensor_spacing_m) / float(cfg.speed_of_sound_m_s)
    max_lag = int(np.ceil(max_tdoa_s * cfg.fs_hz))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    lags_w = lags[mask]
    corr_w = corr[mask]

    tdoa_phase_s, _ = estimate_tdoa_from_tone_phase(
        a_f, b_f, cfg.fs_hz, cfg.target_frequency_hz, max_tdoa_s=max_tdoa_s
    )
    expected_lag = tdoa_phase_s * cfg.fs_hz

    half_period_samp = int(np.ceil((cfg.fs_hz / cfg.target_frequency_hz) / 2.0))
    peak_lag, peak_corr = pick_peak_near_expected(lags_w, corr_w, expected_lag, half_window=half_period_samp)

    est_distance_a = lag_to_distance_from_a(int(np.round(expected_lag)), cfg.fs_hz, cfg.speed_of_sound_m_s, cfg.sensor_spacing_m)
    est_distance_a = float(np.clip(est_distance_a, 0.0, cfg.sensor_spacing_m))

    freq_ok = abs(float(detected_freq) - float(cfg.target_frequency_hz)) <= float(cfg.freq_tolerance_hz)
    confidence = float(np.clip(peak_corr, 0.0, 1.0))
    leak_detected = bool(freq_ok and (confidence >= float(cfg.confidence_threshold)))

    tdoa_s = peak_lag / float(cfg.fs_hz)

    true_distance = None
    error = None
    if isinstance(meta, dict) and "leak_distance_from_a_m" in meta:
        try:
            true_distance = float(meta["leak_distance_from_a_m"])
            error = abs(est_distance_a - true_distance)
        except Exception:
            true_distance = None
            error = None

    result = LocatorResult(
        detected_frequency_hz=float(detected_freq),
        peak_lag_samples=int(peak_lag),
        tdoa_s=float(tdoa_s),
        estimated_distance_m=float(est_distance_a),
        confidence_score=confidence,
        leak_detected=leak_detected,
        true_distance_m=true_distance,
        error_m=error,
        sensor_spacing_m=float(cfg.sensor_spacing_m),
        speed_of_sound_m_s=float(cfg.speed_of_sound_m_s),
    )
    impact = compute_impact_metrics(est_distance_a)
    return result, impact, (lags_w, corr_w, peak_lag)


def localize_from_files(
    csv_path: Path,
    meta_path: Path | None,
    cfg: LocatorConfig,
) -> tuple[LocatorResult, dict, tuple[np.ndarray, np.ndarray, int]]:
    _t, a, b = load_csv(csv_path)
    meta = load_meta(meta_path) if meta_path is not None else None
    return localize_from_arrays(a, b, cfg, meta=meta)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate leak location from leak_sim.csv using TDOA.")
    p.add_argument("--csv", type=str, default="leak_sim.csv", help="Input CSV (from leak_sim.py).")
    p.add_argument("--meta", type=str, default="leak_sim_meta.json", help="Meta JSON (from leak_sim.py).")
    p.add_argument("--fs", type=int, default=44100, help="Sampling rate (Hz).")
    p.add_argument("--c-water", type=float, default=1480.0, help="Speed of sound in water (m/s).")
    p.add_argument("--sensor-spacing-m", type=float, default=15.0, help="Distance between sensors (m).")
    p.add_argument("--target-freq-hz", type=float, default=240.0, help="Expected leak frequency (Hz).")
    p.add_argument("--band-half-width-hz", type=float, default=25.0, help="Bandpass half-width around target (Hz).")
    p.add_argument("--bandpass-order", type=int, default=4, help="Butterworth bandpass order.")
    p.add_argument("--confidence-threshold", type=float, default=0.7, help="Correlation peak threshold for leak_detected.")
    p.add_argument("--freq-tolerance-hz", type=float, default=15.0, help="Allowed freq error for leak_detected.")
    p.add_argument("--plot-out", type=str, default="xcorr.png", help="Correlation plot output path.")
    p.add_argument("--json-out", type=str, default="results.json", help="Results JSON output path.")
    p.add_argument("--no-plot", action="store_true", help="Disable plotting.")
    p.add_argument("--no-json", action="store_true", help="Disable JSON export.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LocatorConfig(
        fs_hz=int(args.fs),
        speed_of_sound_m_s=float(args.c_water),
        sensor_spacing_m=float(args.sensor_spacing_m),
        target_frequency_hz=float(args.target_freq_hz),
        band_half_width_hz=float(args.band_half_width_hz),
        bandpass_order=int(args.bandpass_order),
        confidence_threshold=float(args.confidence_threshold),
        freq_tolerance_hz=float(args.freq_tolerance_hz),
    )

    csv_path = Path(args.csv)
    meta_path = Path(args.meta) if args.meta else None
    result, impact, (lags_w, corr_w, peak_lag) = localize_from_files(csv_path, meta_path, cfg)

    print("Detected dominant frequency (near target band):")
    print(f"  Reported: {result.detected_frequency_hz:.2f} Hz")
    print()
    print("Cross-correlation result:")
    print(f"  Peak lag (samples, B-A): {result.peak_lag_samples}")
    print(f"  TDOA (seconds, B-A):     {result.tdoa_s:.6f}")
    print(f"  Peak correlation:        {result.confidence_score:.4f}")
    print()
    print("Leak location estimate:")
    print(f"  Distance from Sensor A:  {result.estimated_distance_m:.3f} m  (D={cfg.sensor_spacing_m} m)")
    print(f"  Leak detected?:          {result.leak_detected}")

    if result.true_distance_m is not None and result.error_m is not None:
        print()
        print("Validation vs Phase-1 meta:")
        print(f"  True distance from A:    {result.true_distance_m:.3f} m")
        print(f"  Absolute error:          {result.error_m:.3f} m")
        print(f"  Within 0.5 m?:           {result.error_m <= 0.5}")

    if not bool(args.no_plot):
        plot_path = Path(args.plot_out)
        plot_correlation(lags_w, corr_w, cfg.fs_hz, int(peak_lag), plot_path)
        print("Wrote correlation plot:", plot_path.resolve())

    if not bool(args.no_json):
        json_path = Path(args.json_out)
        export_results_json(json_path, result, impact)
        print("Wrote results JSON:", json_path.resolve())


if __name__ == "__main__":
    main()

