# Project Watchdog: Operational Context & Memory

## 1. Core Objective
**Real-Time Anomaly Detection at the Edge.**
A system to detect anomalies in amateur satellite telemetry using an Autoencoder trained on historical data.

## 2. Current Status (As of 2026-02-02)
*   **Phase:** Operations Planning Complete.
*   **Next Step:** "The Lab Phase" (Downloading historical training data from SatNOGS).

## 3. The Golden Cohort (Target Satellites)
Selected based on:
1.  **Status:** Active.
2.  **Band:** 70cm (9600 GMSK).
3.  **Decoder:** Explicitly supported by `gr_satellites` (AX.25 Mode Only).
4.  **Metric:** "Total Observable Time" (>30° elevation).

**Primary Targets (Top 5):**
1.  **GO-32 (TechSat-1B)** - Most observable time (~20 mins/48h).
2.  **STRaND-1**
3.  **TigriSat**
4.  **STEP CubeLab-II**
5.  **UniSat-6**

## 4. Technical Constraints & Config

### Ground Station
*   **Location:** Beni Suef, Egypt (Lat: 29.0661, Lon: 31.0994).
*   **Min Elevation:** 30.0° (High quality filter).
*   **Frequency Band:** 433-438 MHz (70cm).

### Visualization Style Guide
(Used in `src/pass_analysis_viz.py` and `src/comprehensive_analysis.py`)
*   **Background Color:** `#fafafa` (Matches Slide Theme).
*   **Theme:** `seaborn-whitegrid` with custom `rc` params.
*   **Font:** `sans-serif`.
*   **DPI:** 120 (Screen), 150 (Saved Figures).

### File Structure
*   `src/comprehensive_analysis.py`: Filters 300+ sats down to the Golden Cohort.
*   `src/pass_analysis_viz.py`: Generates the Skyplot and Gantt Chart for the next 48h.
*   `docs/slides.typ`: The master presentation (Typst).
*   `docs/figures/`: Stores the auto-generated PNGs.

## 5. Key Commands
*   **Regenerate Figures:** `pixi run python src/comprehensive_analysis.py && pixi run python src/pass_analysis_viz.py`
*   **Update Notebooks:** `uv tool run jupytext --to notebook src/pass_analysis_viz.py --output notebooks/pass_visualization.ipynb`
