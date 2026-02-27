# Project Architecture: AI-Powered Amateur Satellite Ground Station

## 1. Core Objective
**Real-Time Anomaly Detection at the Edge.**
Instead of passive data logging or long-term lifetime prediction, this system acts as a "First Responder." It monitors live telemetry from amateur satellites and flags anomalies (power system faults, tumbling, thermal issues) in real-time.

---

## 2. Target Selection Strategy (The Funnel)
To ensure operational viability, we filtered the entire amateur fleet (338+ satellites) using a rigorous funnel:

1.  **Status Check:** Must be confirmed 'Alive' in SatNOGS DB.
2.  **Band Compatibility:** 433-438 MHz (70cm Amateur Band).
3.  **Modulation:** High-rate 9600 bps GMSK/FSK (Modern Standard).
4.  **Decoder Support:** Must be explicitly supported by `satnogs-decoders` (Kaitai Structs).
5.  **Operational Viability:** Ranked by "Total Observable Time" (Sum of pass durations > 30° elevation over 48h).

### The Golden Cohort (Top Candidates)
Based on our Beni Suef ground station and the `satnogs-decoders` library:

1.  **BugSat-1** - High visibility (~10 mins / 48h), 9600 GMSK.
2.  **UWE-4** - Solid coverage, 9600 FSK.
3.  **INSPIRESat-1** - 9600 GFSK.
4.  **LEDSAT** - 9600 GMSK.
5.  **BDSat** - 9600 GFSK.

---

## 3. High-Level Architecture (The V-Model)
The system is split into two distinct environments that share a common logic core to ensure consistency.

### A. "The Lab" (Offline Training Pipeline)
* **Goal:** Learn "Normal" behavior from historical data.
* **Source:** SatNOGS Database (JSON Archives).

#### 1. Data Ingestion (The Siphon)
*   **Tool:** `scripts/fetch_training_data.py` (Interactive CLI).
*   **Strategy:** "The Lake". We download raw JSON batches (1-day chunks) to `data/raw/`.
*   **Features:**
    *   **Fault Tolerant:** Exponential backoff & Resume capability.
    *   **Rate Limit Aware:** Token bucket delays to respect SatNOGS API limits.
    *   **Streaming:** Appends to `.jsonl` to prevent RAM spikes.

#### 2. Preprocessing & Training
* **Strategy: "Shared Tools, Unique Models"**
    *   **The Problem:** Satellites are physically distinct (different bus voltages, thermal masses). A single "Universal Model" would fail.
    *   **The Solution:** We use the *Shared Core* to normalize data engineering (SI Units), but we train a **separate Autoencoder instance per NORAD ID**.
        *   `models/25397.pkl` (GO-32 Specific Physics).
        *   `models/40908.pkl` (UniSat-6 Specific Physics).
* **Algorithm: Self-Supervised Autoencoder**
    *   **Input:** The current telemetry snapshot (e.g., `[8.2V, 0.15A, 25°C]`).
    *   **Target:** The Input Itself (Reconstruction).
    *   **Goal:** The model learns the **correlations** (physics) of the satellite to compress the data through a bottleneck. It learns rules like "High Voltage usually means Positive Solar Current".
* **Validation Strategy:** **"Injected Physics"**. Since real anomaly labels are rare, we validate the model by synthetically injecting known faults (e.g., voltage drift, sensor noise, stuck values) into clean data to measure detection accuracy.

#### 3. Interpretability: Feature Contribution Analysis
An opaque "Anomaly Score" (e.g., 0.95) is useless to an operator. We provide actionable insights by analyzing the **reconstruction error per feature**.

*   **Logic:** $\text{Contribution} = | \text{Input} - \text{Reconstruction} |$
*   **Example (Heater Stuck ON):**
    *   **Input:** `[Temp: 50°C, Current: 0.1A]` (Hot but low power draw).
    *   **Model Expectation:** `[Temp: 10°C, Current: 0.1A]` (Model knows low power usually means low temp).
    *   **Result:**
        *   `Temp Error`: **40.0** (Critical Contributor -> Flag "Temperature Anomaly").
        *   `Current Error`: 0.0 (Normal).

### B. "The Watchdog" (Online Inference Pipeline)
* **Goal:** Detect anomalies during a 10-minute satellite pass.
* **Source:** Local Antenna -> SDR -> `satnogs-decoders`.
* **Process:**
    1. Demodulate packets.
    2. Decode & Normalize using `satnogs-decoders` (Kaitai Structs).
    3. Run Inference using the pre-trained model.
    4. Alert on high reconstruction error.

---

## 4. Standardization Strategy
To handle multiple disparate satellites without chaos, the system enforces strict standardization layers.

### Transport Layer: SatNOGS Compatible
* **Protocol:** We leverage the **`satnogs-decoders`** ecosystem.
* **Benefit:** Battle-tested Kaitai Structs that already handle a wide variety of amateur satellite formats.

### Semantic Layer: The "Golden Features" (Expanded)
We define a universal target interface (SI Units) that all satellite data must be mapped to.

| Feature | Unit | Description |
| :--- | :--- | :--- |
| `batt_voltage` | Volts (V) | Standardized from mV or ADC counts. |
| `batt_current` | Amps (A) | Charge/Discharge rate. |
| `temp_obc` | Celsius (°C) | Main computer temperature. |
| `solar_current` | Amps (A) | Panel health & eclipse status. |
| `temp_pa` | Celsius (°C) | Power Amp temp (Radio stuck ON detection). |
| `signal_rssi` | dBm | Ground-calculated signal strength (Tumble detection). |

---

## 5. The Shared Python Core (Implementation Details)
This module is imported by both the Training scripts and the Live Receiver.

### The "Adapter" Pattern
We use a **Registry** to map Callsigns to specific Decoders and Adapters.

```python
# Conceptual Architecture

# 1. The Registry (Maps Callsign -> Logic)
REGISTRY = {
    "NJ7P": {
        "decoder": Fox1_Construct_Struct, 
        "adapter": adapt_fox1_to_si
    },
    "UPSAT": {
        "decoder": UPSat_Construct_Struct, 
        "adapter": adapt_upsat_to_si
    }
}

# 2. The Universal Loop (Used in Live & Training)
def process_packet(raw_bytes):
    # Step A: Identify (AX.25 Header)
    callsign = parse_ax25_header(raw_bytes).src_callsign
    
    if callsign in REGISTRY:
        # Step B: Decode (Binary -> Raw Dict)
        raw_data = REGISTRY[callsign]["decoder"].parse(raw_bytes)
        
        # Step C: Normalize (Raw Dict -> Golden Features)
        # This is the crucial step for ML consistency
        ml_vector = REGISTRY[callsign]["adapter"](raw_data)
        
        return ml_vector
```

## 6. Development Progress (Updated Feb 2026)

### Implemented Components

#### 1. The Shared Core (`src/gr_sat/telemetry.py`)
*   **`TelemetryFrame`:** The concrete implementation of "Golden Features" as a Python dataclass. Enforces type safety and SI unit standardization.
*   **`DecoderRegistry`:** A singleton registry that automatically registers decoders via decorators (`@DecoderRegistry.register`).
*   **`process_frame`:** The universal entry point function.

#### 2. Data Refinery (`scripts/process_data.py`)
*   **Pipeline:** `Raw JSONL` -> `Dedup` -> `Decode` -> `Normalize` -> `CSV`.
*   **Status:** Successfully processed GO-32 (TechSat-1B) data with ~92% decode rate.

#### 3. Telemetry Inspector (`notebooks/telemetry_inspector.py`)
*   **Tool:** An interactive Jupyter-based visual debugger.
*   **Goal:** "Ground Truth" verification. Allows humans to visually correlate raw hex bytes with parsed values to ensure the decoder is not hallucinating.
*   **Features:**
    *   **Hex View:** Raw payload inspection.
    *   **Struct View:** Visualization of the intermediate binary parsing steps (ADC counts).
    *   **Telemetry View:** Verification of the final physical values (Volts, Amps).
    *   **Navigation:** Slider-based frame traversal.

---

## 7. Technology Stack

### Hardware & RF
* **Antenna:** Omnidirectional or Yagi.
*   **SDR:** RTL-SDR.
*   **Demodulator:** `gr_satellites` or raw baseband processors.
*   **Decoder:** `satnogs-decoders` (Kaitai Structs).
*   **Interface:** UDP Stream or SatNOGS DB frames.

### Software

*   **Language:** Python 3.11.

*   **Data Engineering:** `requests`, `loguru`, `rich` (Robust CLI pipelines).

*   **ML Core:** `scikit-learn`, `pytorch`, `pandas`.

*   **Physics Engine:** `skyfield` (Orbit prediction & geometry).

*   **Parsing:** `satnogs-decoders` (Kaitai Struct compiler output).

*   **Visualization:** `matplotlib`, `seaborn` (for operational dashboards).


---

## 8. Reviewer Defense Strategy
* **"How do you know it's not hallucinating?"**
    * We use AX.25 Checksums (CRC-16). We only process mathematically valid packets.
    * We use the embedded Callsign for ID, not orbital predictions.
* **"How do you know the model works without real failure data?"**
    * We use **Synthetic Fault Injection**. We prove the model *would* have caught a failure by mathematically superimposing faults onto historical data and measuring recall.