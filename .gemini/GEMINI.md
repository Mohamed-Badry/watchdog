# Gemini Context: Project Watchdog (gr_sat)

## 1. Project Overview
**Goal:** Real-time anomaly detection for amateur satellite telemetry using an Autoencoder.
**Key Concept:** "The Golden Cohort" (Top 5 satellites) -> "Shared Core" (Normalization) -> "The Lab" (Training) / "The Watchdog" (Inference).

## 2. Agent Mandates & Conventions
*   **Package Manager:** STRICTLY use `pixi`. NEVER use `pip install` or `venv` directly.
*   **Task Runner:** Use `just` for all standard workflows defined in `justfile`.
*   **Python:** Version 3.11. Use `loguru` for logging.
*   **Paths:**
    *   `src/gr_sat/`: Library code (Shared Core, Telemetry Models).
    *   `scripts/`: Executable pipelines (Data ingestion/processing).
    *   `data/`: Data storage (Gitignored).
    *   `docs/`: User-facing documentation.
*   **Commits:** Follow conventional commits (e.g., `feat: add telemetry parser`, `fix: correct orbit calculation`).

## 3. Active Context (Memory)
*   **Current Phase:** "The Refinery" (Data Processing).
*   **Recent Accomplishments:**
    *   Implemented `fetch_training_data.py` (Ingestion).
    *   Implemented `process_data.py` (Normalization).
    *   Implemented GO-32 Decoder.
*   **Current Blockers/Tasks:**
    *   Train Autoencoder ("The Lab").
    *   Validate with "Injected Physics".

## 4. Key Workflows
*   **Fetch Data:** `just fetch` (Interactive) or `just fetch --all` (Batch).
*   **Analyze Targets:** `just analyze-targets` (Filters candidates to "The Golden Cohort").
*   **Visualize Passes:** `just viz-passes` (Generates Skyplots/Gantt charts).
*   **Regenerate All Analysis:** `just regenerate-all`.
*   **Process Data (Manual):** `pixi run python scripts/process_data.py --norad <id>`
*   **Inspect Telemetry:** `pixi run python scripts/telemetry_inspector.py`

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
    *   **Artifacts:** Specific per NORAD ID (e.g., `models/25397.pkl`).
*   **Interpretability:** Feature Contribution Analysis.
    *   **Metric:** Absolute Error per Feature (`|Input - Reconstruction|`).
    *   **Goal:** Pinpoint the *specific subsystem* causing the anomaly (e.g., "Temp Error: +40C").
*   **Validation:** Synthetic Fault Injection (Drift, Stuck Value, Noise).
