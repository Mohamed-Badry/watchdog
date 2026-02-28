# DISCLAIMER: This project is still nowhere near complete and heavily AI-assisted use at your own discretion.

# Project Watchdog (gr_sat)

This repository hosts the **Project Watchdog** codebase, an AI-powered system for real-time anomaly detection in amateur satellite telemetry.

## 📚 Key Documentation

*   **[Technical Details & Architecture](DETAILS.md)**: Deep dive into the system's design and "Golden Features".
*   **[.gemini/GEMINI.md](.gemini/GEMINI.md)**: Active project context and agent instructions.

---

## 🚀 Quick Start Guide

### 1. Install Prerequisites
We use **[Pixi](https://pixi.sh/)** for all dependency management (Python, GNU Radio, etc.). You do **not** need to install Python or GNU Radio manually.

**Linux / macOS:**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

### 2. Setup the Project
Clone the repo and install dependencies:
```bash
git clone https://github.com/Mohamed-Badry/watchdog.git
pixi install
```

### 3. Configure API Keys
To download telemetry, you need a **SatNOGS API Token**.

1.  Log in to [SatNOGS Network](https://network.satnogs.org/login/) (create an account if needed).
2.  Go to **[Your Profile / Edit](https://db.satnogs.org/user/edit)**.
3.  Copy the **API Token** at the bottom of the page.
4.  Create your configuration file:
    ```bash
    cp .env.example .env
    ```
5.  Open `.env` and paste your token:
    ```ini
    SATNOGS_API_TOKEN=your_actual_token_here
    ```

---

## 📂 Data Directory Structure

The project organizes telemetry data into three distinct stages to maintain clear boundaries between raw fetches, raw decodes, and clean features:

*   **`data/raw/`**: Contains original, unprocessed JSONL files fetched directly from the SatNOGS DB API.
*   **`data/interim/`**: Contains CSV files with all decoded telemetry fields, extracted exactly as they were parsed by `satnogs-decoders` without modification.
*   **`data/processed/`**: Contains finalized CSV files mapped to our SI-unit "Golden Features", cleaned, scaled, and ready for model training.

---

## 🛠 Usage

We use `just` (installed automatically by Pixi) to run common tasks.

**Recommended:** Enter the environment to use `just` directly:
```bash
pixi shell
```
Now you can run commands simply like `just fetch`.

*(Note: If you prefer not to enter the shell, or don't have `just` installed globally, you can prefix commands with `pixi run`, e.g., `pixi run just fetch`)*

**Common Commands:**
*   **Download Telemetry:**
    ```bash
    just fetch
    ```
*   **Run Analysis:**
    ```bash
    just analyze-targets
    ```
*   **Visualize Passes:**
    ```bash
    just viz-passes
    ```

For a full list of commands, run:
```bash
just --list
```

---

## 🗂️ Full Project Structure

```text
.
├── data/               # Local data storage (ignored by git)
│   ├── interim/        # Decoded raw telemetry CSVs without modifications
│   ├── processed/      # Cleaned, scaled "Golden Features" ready for ML
│   └── raw/            # Raw JSONL fetches from SatNOGS DB API
├── docs/               # Documentation and analysis outputs
│   └── figures/        # Generated plots and architectural diagrams
├── logs/               # Log files from data ingestion pipelines
├── notebooks/          # Jupyter notebooks for EDA and prototyping (Jupytext)
├── scripts/            # Executable scripts for data ingestion and processing
├── src/gr_sat/         # Core library code ("The Shared Core")
│   └── decoders/       # Telemetry parsing logic (satnogs-decoders)
├── .gemini/
│   └── GEMINI.md       # Active project context and agent mandates
├── DETAILS.md          # Technical architecture and system design
├── justfile            # Task runner configuration (project commands)
├── pixi.toml           # Environment and dependency management
└── README.md           # This file
```
