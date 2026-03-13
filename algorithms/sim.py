"""
Simulation engine for Acoustic Leak-Net (two sensors with TDOA).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft as sp_fft
from scipy import signal
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SimulationConfig:
    fs_hz: int = 44100
    duration_s: float = 2.0
    sensor_spacing_m: float = 15.0
    leak_frequency_hz: float = 240.0
    speed_of_sound_m_s: float = 1480.0
    leak_distance_from_a_m: float = 4.2
    leak_amplitude: float = 1.0
    leak_narrowband_noise_std: float = 0.35
    leak_narrowband_half_width_hz: float = 30.0
    noise_std: float = 0.25
    attenuation_power: float = 1.0  # amplitude scale ~ 1 / distance^attenuation_power
    seed: int | None = 0


def time_axis(duration_s: float, fs_hz: int) -> np.ndarray:
    n = int(np.round(duration_s * fs_hz))
    if n <= 0:
        raise ValueError("duration_s must be > 0")
    return np.arange(n, dtype=np.float64) / float(fs_hz)


def validate_geometry(leak_distance_from_a_m: float, sensor_spacing_m: float) -> tuple[float, float]:
    if sensor_spacing_m <= 0:
        raise ValueError("sensor_spacing_m must be > 0")
    if not (0.0 <= leak_distance_from_a_m <= sensor_spacing_m):
        raise ValueError("leak_distance_from_a_m must be within [0, sensor_spacing_m]")
    d_a = float(leak_distance_from_a_m)
    d_b = float(sensor_spacing_m - leak_distance_from_a_m)
    return d_a, d_b


def leak_signal(t: np.ndarray, frequency_hz: float, amplitude: float) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * float(frequency_hz) * t)


def narrowband_noise(
    n: int,
    fs_hz: int,
    center_hz: float,
    half_width_hz: float,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if std <= 0:
        return np.zeros(n, dtype=np.float64)
    nyq = 0.5 * float(fs_hz)
    lo = max((float(center_hz) - float(half_width_hz)) / nyq, 1e-6)
    hi = min((float(center_hz) + float(half_width_hz)) / nyq, 0.999999)
    if not (0.0 < lo < hi < 1.0):
        raise ValueError("Invalid narrowband noise band.")
    x = rng.normal(0.0, 1.0, size=n).astype(np.float64)
    sos = signal.butter(4, [lo, hi], btype="bandpass", output="sos")
    y = signal.sosfiltfilt(sos, x)
    y = y / (np.std(y) + 1e-12)
    return (float(std) * y).astype(np.float64)


def add_white_noise(x: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    if noise_std <= 0:
        return x
    return x + rng.normal(loc=0.0, scale=float(noise_std), size=x.shape)


def scale_by_distance(amplitude: float, distance_m: float, attenuation_power: float) -> float:
    d = max(float(distance_m), 0.05)
    p = max(float(attenuation_power), 0.0)
    return float(amplitude) / (d**p if p != 0.0 else 1.0)


def fractional_delay_fft(x: np.ndarray, delay_s: float, fs_hz: int) -> np.ndarray:
    n = x.size
    if n == 0 or delay_s == 0.0:
        return x.copy()
    x = np.asarray(x, dtype=np.float64)
    X = sp_fft.rfft(x)
    freqs = sp_fft.rfftfreq(n, d=1.0 / float(fs_hz))
    phase = np.exp(-1j * 2.0 * np.pi * freqs * float(delay_s))
    y = sp_fft.irfft(X * phase, n=n)
    return np.asarray(y, dtype=np.float64)


def tdoa_seconds(distance_a_m: float, distance_b_m: float, speed_of_sound_m_s: float) -> float:
    return (float(distance_b_m) - float(distance_a_m)) / float(speed_of_sound_m_s)


def generate_signals(cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(cfg.seed)
    t = time_axis(cfg.duration_s, cfg.fs_hz)

    d_a, d_b = validate_geometry(cfg.leak_distance_from_a_m, cfg.sensor_spacing_m)
    tau_b_minus_a = tdoa_seconds(d_a, d_b, cfg.speed_of_sound_m_s)

    tone = leak_signal(t, cfg.leak_frequency_hz, cfg.leak_amplitude)
    nb = narrowband_noise(
        n=t.size,
        fs_hz=cfg.fs_hz,
        center_hz=cfg.leak_frequency_hz,
        half_width_hz=cfg.leak_narrowband_half_width_hz,
        std=cfg.leak_narrowband_noise_std,
        rng=rng,
    )
    src = tone + nb

    a_amp = scale_by_distance(cfg.leak_amplitude, d_a, cfg.attenuation_power)
    b_amp = scale_by_distance(cfg.leak_amplitude, d_b, cfg.attenuation_power)

    sensor_a = a_amp * src
    sensor_b = b_amp * src

    if tau_b_minus_a > 0:
        sensor_b = fractional_delay_fft(sensor_b, tau_b_minus_a, cfg.fs_hz)
    elif tau_b_minus_a < 0:
        sensor_a = fractional_delay_fft(sensor_a, -tau_b_minus_a, cfg.fs_hz)

    sensor_a = add_white_noise(sensor_a, cfg.noise_std, rng)
    sensor_b = add_white_noise(sensor_b, cfg.noise_std, rng)

    meta = {
        "fs_hz": cfg.fs_hz,
        "duration_s": cfg.duration_s,
        "sensor_spacing_m": cfg.sensor_spacing_m,
        "leak_frequency_hz": cfg.leak_frequency_hz,
        "leak_narrowband_noise_std": cfg.leak_narrowband_noise_std,
        "leak_narrowband_half_width_hz": cfg.leak_narrowband_half_width_hz,
        "speed_of_sound_m_s": cfg.speed_of_sound_m_s,
        "leak_distance_from_a_m": cfg.leak_distance_from_a_m,
        "distance_a_m": d_a,
        "distance_b_m": d_b,
        "tdoa_b_minus_a_s": tau_b_minus_a,
        "tdoa_b_minus_a_samples": tau_b_minus_a * cfg.fs_hz,
        "noise_std": cfg.noise_std,
        "attenuation_power": cfg.attenuation_power,
        "seed": cfg.seed,
    }
    return t, sensor_a, sensor_b, meta


def save_csv(path: Path, t: np.ndarray, sensor_a: np.ndarray, sensor_b: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((t, sensor_a, sensor_b))
    header = "time,sensor_a_amplitude,sensor_b_amplitude"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def magnitude_spectrum(x: np.ndarray, fs_hz: int) -> tuple[np.ndarray, np.ndarray]:
    n = x.size
    X = sp_fft.rfft(x)
    freqs = sp_fft.rfftfreq(n, d=1.0 / float(fs_hz))
    mag = np.abs(X) / max(n, 1)
    return freqs, mag


def plot_verification(
    t: np.ndarray,
    sensor_a: np.ndarray,
    sensor_b: np.ndarray,
    fs_hz: int,
    leak_frequency_hz: float,
    snippet_s: float = 0.01,
) -> plt.Figure:
    n_snip = int(np.clip(np.round(snippet_s * fs_hz), 1, t.size))
    t_snip = t[:n_snip]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    axes[0].plot(t_snip, sensor_a[:n_snip], label="Sensor A", linewidth=1.0)
    axes[0].plot(t_snip, sensor_b[:n_snip], label="Sensor B", linewidth=1.0, alpha=0.85)
    axes[0].set_title(f"First {snippet_s:.3f} s (time domain)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    freqs_a, mag_a = magnitude_spectrum(sensor_a, fs_hz)
    freqs_b, mag_b = magnitude_spectrum(sensor_b, fs_hz)

    axes[1].plot(freqs_a, mag_a, label="Sensor A", linewidth=1.0)
    axes[1].plot(freqs_b, mag_b, label="Sensor B", linewidth=1.0, alpha=0.85)
    axes[1].axvline(float(leak_frequency_hz), color="k", linestyle="--", linewidth=1.0, alpha=0.6, label="Leak freq")
    axes[1].set_xlim(0, 2000)
    axes[1].set_title("Magnitude spectrum (FFT)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic leak signals for two sensors with TDOA.")
    p.add_argument("--leak-distance-m", type=float, default=4.2, help="Leak distance from Sensor A (m).")
    p.add_argument("--sensor-spacing-m", type=float, default=15.0, help="Distance between Sensor A and B (m).")
    p.add_argument("--fs", type=int, default=44100, help="Sampling rate (Hz).")
    p.add_argument("--duration-s", type=float, default=2.0, help="Signal duration (s).")
    p.add_argument("--leak-freq-hz", type=float, default=240.0, help="Leak tone frequency (Hz).")
    p.add_argument("--c-water", type=float, default=1480.0, help="Speed of sound in water (m/s).")
    p.add_argument("--noise-std", type=float, default=0.25, help="Gaussian noise standard deviation.")
    p.add_argument("--leak-amp", type=float, default=1.0, help="Leak tone amplitude (before distance scaling).")
    p.add_argument("--leak-nb-noise-std", type=float, default=0.35, help="Std of narrowband leak component.")
    p.add_argument("--leak-nb-half-width-hz", type=float, default=30.0, help="Half-width of narrowband leak component (Hz).")
    p.add_argument("--attenuation-power", type=float, default=1.0, help="Amplitude scale ~ 1/d^p.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed. Use -1 for non-deterministic.")
    p.add_argument("--out", type=str, default="leak_sim.csv", help="Output CSV path.")
    p.add_argument("--meta-out", type=str, default="leak_sim_meta.json", help="Output meta JSON path.")
    p.add_argument("--no-plot", action="store_true", help="Disable plots.")
    p.add_argument("--plot-out", type=str, default="leak_sim.png", help="Output plot image path (used unless --show-plot).")
    p.add_argument("--show-plot", action="store_true", help="Show interactive plot window.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig(
        fs_hz=int(args.fs),
        duration_s=float(args.duration_s),
        sensor_spacing_m=float(args.sensor_spacing_m),
        leak_frequency_hz=float(args.leak_freq_hz),
        speed_of_sound_m_s=float(args.c_water),
        leak_distance_from_a_m=float(args.leak_distance_m),
        leak_amplitude=float(args.leak_amp),
        leak_narrowband_noise_std=float(args.leak_nb_noise_std),
        leak_narrowband_half_width_hz=float(args.leak_nb_half_width_hz),
        noise_std=float(args.noise_std),
        attenuation_power=float(args.attenuation_power),
        seed=None if int(args.seed) == -1 else int(args.seed),
    )

    t, a, b, meta = generate_signals(cfg)
    out_path = Path(args.out)
    save_csv(out_path, t, a, b)

    meta_path = Path(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("Wrote CSV:", out_path.resolve())
    print("Wrote meta:", meta_path.resolve())

    if not args.no_plot:
        fig = plot_verification(t, a, b, cfg.fs_hz, cfg.leak_frequency_hz, snippet_s=0.01)
        if bool(args.show_plot):
            plt.show()
        else:
            plot_path = Path(args.plot_out)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print("Wrote plot:", plot_path.resolve())


if __name__ == "__main__":
    main()

