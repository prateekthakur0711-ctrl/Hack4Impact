# Acoustic Leak-Net

Low-cost acoustic leak detection + localization using **two ESP32-class sensors** and **Time Difference of Arrival (TDOA)**. The demo simulates a leak signature, estimates delay via signal processing, and converts that delay into a leak distance—enabling **"surgical maintenance"** instead of expensive, city-wide excavation.

---

## Executive Summary (Problem vs. Solution)

Cities lose an estimated **30% of urban water** to leaks—often from aging, buried infrastructure where failures are hard to detect and even harder to pinpoint. For under-funded municipalities, the status quo is reactive: crews respond after visible damage, then spend hours searching, digging, and shutting down service. That delay wastes treated water, increases repair costs, and disproportionately harms high-density communities where outages and road closures have outsized impact.

**Acoustic Leak-Net** is a low-cost, scalable alternative. We use **two synchronized ESP32-based nodes** placed a known distance apart on a pipe. A leak produces a repeatable acoustic signature (in our demo, centered near **240 Hz**). By filtering around the leak band and measuring the **time delay** between sensors, we calculate the leak's position along the pipe segment. This turns leak response into **surgical maintenance**: crews go directly to the most likely location, reducing excavation, downtime, and waste. Our approach is designed to be affordable, deployable, and maintainable—so municipalities can protect water supply with modern sensing even under tight budgets.

---

## Technical Deep-Dive (Cross-Correlation & FFT Pipeline)

Our pipeline has two goals: **confirm a leak-like signature exists** and **localize it**.

- **FFT leak confirmation**: We compute an FFT to verify a dominant spectral component near the target leak band (demo uses **240 Hz**). This is a fast, interpretable check for judges and for automated "leak/no leak" decisions.
- **Bandpass filtering**: We apply a Butterworth bandpass around the leak frequency to suppress broadband environmental noise (traffic, pumps, pipe vibration) and focus on the leak energy.
- **Normalized cross-correlation**: We cross-correlate the filtered signals to find the lag that maximizes similarity. That lag is converted into **TDOA**.
- **Localization math**: With sensor spacing (D) and sound speed in water (c), we estimate distance from Sensor A:

$$\tau = \frac{(D-x)-x}{c} = \frac{D-2x}{c} \quad\Rightarrow\quad x = \frac{D - c\tau}{2}$$

Why **44,100 Hz** sampling? It provides fine time resolution (~22.7 µs per sample), and combined with **narrowband filtering** and fractional-delay estimation, it enables very tight TDOA estimates in simulation. (In real deployments, achievable accuracy depends on coupling, sensor sync, pipe material, and noise conditions.)

---

## Inclusive Innovation (Track 2)

- **Affordability**: Targets a bill of materials that is **~1/100th the cost** of many industrial leak monitoring systems—making city-scale deployment feasible.
- **Infrastructure equity**: Prioritizes protection for **aging pipes in high-density, low-income areas**, where leaks, outages, and road disruption compound existing burdens.
- **Environmental impact**: Faster localization prevents loss of **thousands of liters of treated water per hour**, reducing both water waste and the energy footprint of treatment and pumping.

---

## How to Run

### 1. Install dependencies

```bash
python -m pip install numpy scipy matplotlib rich streamlit plotly
```

### 2. Launch the Web Dashboard (recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with two tabs:

- **Simulation & Report** — run a full leak simulation, view results, and download a pitch report ZIP
- **Live Vibration Dashboard** — real-time sensor waveforms, FFT spectrum, cross-correlation, pipe map, threshold alerting, and alert history

### 3. Run the CLI demo

```bash
python main_demo.py
```

Change leak distance (meters from Sensor A):

```bash
python main_demo.py --leak-distance-m 7.5
```

---

## Live Vibration Dashboard

The dashboard (`vibration_dashboard.py`) provides real-time visualization of simulated acoustic sensor data with full leak detection alerting:

- **Sensor waveforms** — both sensors plotted with adjustable amplitude threshold lines
- **Alert banner** — three states: nominal (green), amplitude warning (amber), leak detected (red)
- **FFT spectrum** — frequency domain view with the 240 Hz leak band highlighted
- **Cross-correlation (TDOA)** — peak lag annotated in milliseconds
- **Pipe map** — estimated vs. true leak position along the pipe
- **Alert history** — confidence and distance plotted over successive runs
- **Live mode** — continuous auto-refresh simulation via "Start live" in the sidebar

---

## Outputs (CLI)

| File | Description |
|---|---|
| `leak_sim.csv` | Time series for Sensor A/B |
| `leak_sim_meta.json` | Ground-truth parameters (including true leak distance) |
| `results.json` | Exportable results for design tools (e.g., Figma) |
| `xcorr.png` | Correlation curve with peak highlighted |
| `demo_report.png` | "Judge's Brief" image combining FFT + correlation + summary |

---

## Project Roadmap (Simulation → Real-World IoT Deployment)

| Phase | Status | Description |
|---|---|---|
| Phase 1 — Simulation & validation | ✅ Done | Generate synthetic leak + noise, inject physically-derived TDOA, validate localization error against ground truth |
| Phase 2 — Bench prototype | ⏳ Planned | Couple ESP32 nodes to a test pipe, collect real audio/vibration data, calibrate bandpass targets per pipe type |
| Phase 3 — Time synchronization | ⏳ Planned | Add precise clock sync (PPS/GPS or wired sync; compensate with periodic calibration pulses) |
| Phase 4 — Edge inference | ⏳ Planned | Run FFT + filtering + correlation on-device, output leak distance and confidence in near real time |
| Phase 5 — Field pilots | ⏳ Planned | Pilot on short municipal segments, measure localization accuracy vs. ground truth, iterate on enclosures |
| Phase 6 — City operations integration | ⏳ Planned | Integrate alerts into work order systems, prioritize by confidence + estimated flow loss + criticality |

---

## Repository Structure

```
Hack4Impact/
├── algorithms/
│   ├── sim.py              # Simulation engine (signal generation, TDOA injection)
│   └── locator.py          # DSP engine (FFT, bandpass, cross-correlation, localization)
├── app.py                  # Streamlit web app (Simulation & Report + Live Dashboard tabs)
├── vibration_dashboard.py  # Live vibration dashboard module
├── leak_sim.py             # CLI wrapper for simulation
├── leak_locator.py         # CLI wrapper for localization
├── main_demo.py            # Full pipeline CLI demo
└── requirements.txt        # Python dependencies
```