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
    *   Train Autoencoder ("The Lab").
    *   Validate with "Injected Physics".

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
*   **Validation:** Synthetic Fault Injection (Drift, Stuck Value, Noise).

## 9. EDA Insights & ML Plan (UWE-4)
*   **Correlations:** Strong physical correlations exist (e.g., `temp_panel_z` correlates highly with `batt_voltage` and `batt_current`). This proves the Autoencoder will have solid physical rules to learn.
*   **Feature Selection:** Use `batt_voltage`, `batt_current`, `temp_batt_a`, `temp_batt_b`, and `temp_panel_z`.
*   **Limitation/Dropping:** The `temp_obc` feature has a strict zero variance (stuck at 17°C in the current dataset). It MUST be dropped during training to avoid matrix singularity issues during normalization (e.g., `StandardScaler` dividing by zero).
