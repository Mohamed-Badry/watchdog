# Gemini Context: Project Watchdog (gr_sat)

## 1. Project Overview
**Goal:** Real-time anomaly detection for amateur satellite telemetry using an Autoencoder.
**Key Concept:** "The Golden Cohort" (Optimal targets) -> "Shared Core" (Normalization) -> "The Lab" (Training) / "The Watchdog" (Inference).

## 2. Agent Mandates & Conventions
*   **Package Manager:** STRICTLY use `pixi`. NEVER use `pip install` or `venv` directly.
*   **Task Runner:** Use `just` for all standard workflows defined in `justfile`.
*   **Decoders:** Use `satnogs-decoders` (Kaitai Structs) for all telemetry parsing. DO NOT write manual `construct` decoders.
*   **Notebooks:** Use `jupytext` to manage scripts as notebooks. Always prefer editing `.py` files and converting/syncing them.
*   **Python:** Version 3.11. Use `loguru` for logging.
*   **Documentation:** ALWAYS update relevant documentation (e.g., `DETAILS.md`, `docs/slides.typ`) to ensure it remains consistent with code changes.
*   **Paths:**
    *   `src/gr_sat/`: Library code (Shared Core, Telemetry Models).
    *   `scripts/`: Executable pipelines (Data ingestion/processing).
    *   `data/`: Data storage (Gitignored).
    *   `docs/`: User-facing documentation.
*   **Commits:** Follow conventional commits (e.g., `feat: add telemetry parser`, `fix: correct orbit calculation`).

## 3. Active Context (Memory)
*   **Current Phase:** "The Refinery" (Data Processing).
*   **Recent Accomplishments:**
    *   Implemented `fetch_training_data.py` (Dynamic Target Loading).
    *   Transitioned to `satnogs-decoders` (Kaitai Structs) for universal coverage.
    *   Updated `sat_analysis.py` & `pass_analysis_viz.py` for new decoder ecosystem.
*   **Current Blockers/Tasks:**
    *   Ingest long-term (180+ days) UWE-4 data to capture Seasonal/Beta Angle thermal variations.
    *   Build `scripts/generate_faults.py` for Synthetic Fault Injection benchmarking.
    *   Train Autoencoder ("The Lab") and benchmark for Edge Deployment (Latency/Memory).

## 4. Key Workflows
*   **Fetch Data:** `just fetch` (Interactive) or `just fetch --all` (Batch).
*   **Analyze Targets:** `just analyze-targets` (Filters candidates to "The Golden Cohort").
*   **Visualize Passes:** `just viz-passes` (Generates Skyplots/Gantt charts).
*   **Regenerate All Analysis:** `just regenerate-all`.
*   **Sync Notebooks:** `just sync-notebooks` (Updates all `.ipynb` from `.py` in `notebooks/`).
*   **Convert Script:** `just convert notebooks/script.py` (Converts a single script to notebook).
*   **Process Data (Manual):** `pixi run python scripts/process_data.py --norad <id>`
*   **Inspect Telemetry:** `pixi run python notebooks/telemetry_inspector.py`

## 6. Technical Specifics
*   **The Shared Core (`src/gr_sat/telemetry.py`):**
    *   **`TelemetryFrame`:** The universal DTO. All decoders MUST return this.
        *   **Key Fields:** `batt_voltage`, `batt_current`, `temp_obc`, `signal_rssi`, `timestamp`.
        *   **Units:** SI Units ONLY (Volts, Amps, Celsius).
    *   **`DecoderRegistry`:**
        *   Use `@DecoderRegistry.register(norad_id)` to register new decoders in `src/gr_sat/decoders/`.
        *   Decoders must implement the `decode(payload: bytes) -> Dict` protocol.
*   **Data Pipeline:**
    *   **Raw:** `data/raw/{norad_id}/*.jsonl` (SatNOGS JSON format).
    *   **Processed:** `data/processed/{norad_id}.csv` (Standardized CSV).
    *   **Flow:** Raw -> Deduplicate (Timestamp) -> Decode -> Normalize -> CSV.

## 8. ML Strategy (The Lab)
*   **Architecture:** Autoencoder (Self-Supervised).
    *   **Input:** Normalized TelemetryFrame (SI Units).
    *   **Output:** Reconstructed Telemetry.
*   **Model Management:** "Shared Tools, Unique Models".
    *   **Training Script:** Generic (`train_model.py`).
    *   **Artifacts:** Specific per NORAD ID (e.g., `models/43880.pkl`).
*   **Interpretability:** Feature Contribution Analysis.
    *   **Metric:** Absolute Error per Feature (`|Input - Reconstruction|`).
    *   **Goal:** Pinpoint the *specific subsystem* causing the anomaly.
*   **Validation & Benchmarking (The Edge):**
    *   **Accuracy Benchmark:** "Synthetic Fault Injection". Since labeled anomaly data is rare, we programmatically inject physical faults (e.g., Sensor Stuck, Tumbling/High Variance, Solar Panel Failure) into a clean test set and measure the model's Recall and False Positive Rate.
    *   **Performance Benchmark:** Measure inference Latency (target < 10ms per frame) and Memory Footprint (model size in MB) to ensure it can run on a Raspberry Pi/Ground Station PC alongside `gr_satellites`.

## 9. EDA Insights & ML Plan (UWE-4)
*   **Data Quality (Zero Variance):** The `temp_obc` feature is perfectly clean but stuck at 17°C (zero variance). It MUST be dropped before training to prevent `StandardScaler` from dividing by zero and crashing.
*   **Time-Series Dynamics (Bursty Data):** The median time between frames is ~18 seconds, but the maximum gap is ~11 hours (due to limited ground station passes). **Conclusion:** Rolling window or sequence models (like LSTM) are inappropriate due to extreme temporal discontinuity. A stateless Feed-Forward Autoencoder processing each frame independently is required.
*   **Physics over Statistics:** Standard statistical tests (IQR) flagged ~40% of the battery current data as "outliers". Deeper physical correlation revealed this is simply the bimodal state of the satellite: Charging during sunlight (Panel > 15°C) and discharging during eclipse (Panel < 15°C). We will NOT clip these "outliers".
*   **Feature Selection:** The final training vector will be 5-dimensional: `[batt_voltage, batt_current, temp_batt_a, temp_batt_b, temp_panel_z]`.
*   **Scaling Strategy:** Use `StandardScaler` (Z-score normalization) to center the bimodal distributions correctly for the Autoencoder.
