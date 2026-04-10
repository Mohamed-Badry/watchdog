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

## 📂 Data Pipeline

The project processes satellite telemetry through three distinct stages. Each stage has a corresponding directory under `data/` and a clear boundary of responsibility:

*   **`data/raw/`**: Original JSONL files fetched directly from the SatNOGS DB API via `scripts/fetch_training_data.py`. One file per day per satellite.
*   **`data/interim/`**: CSV files with all decoded telemetry fields, extracted exactly as `satnogs-decoders` (Kaitai Structs) parses them — no unit conversion or renaming.
*   **`data/processed/`**: Finalized CSV files mapped to our SI-unit "Golden Features", cleaned, deduplicated, and ready for model training.

---

## 🛠 Usage

We use `just` (installed automatically by Pixi) to run common tasks.

**Recommended:** Enter the Pixi environment first:
```bash
pixi shell
```

**Common Commands:**
```bash
just fetch                  # Download telemetry (interactive)
just fetch --norad 43880    # Download specific satellite
just process                # Run decode + normalize pipeline (interactive)
just process --norad 43880  # Process specific satellite
just analyze-targets        # Regenerate target analysis
just viz-passes             # Generate pass visualizations
just --list                 # Show all available commands
```

*(If you don't have `just` globally, prefix with `pixi run`, e.g. `pixi run just fetch`)*

---

## 🔧 Adding a New Satellite Decoder

The decoder system uses `satnogs-decoders` (Kaitai Structs) for binary parsing and a registry pattern for satellite-specific logic. To add support for a new satellite:

1. Create `src/gr_sat/decoders/<satellite>.py`
2. Subclass `BaseDecoder` and implement `decode()` + `adapt()`
3. Register with `@DecoderRegistry.register(NORAD_ID)`
4. Import in `src/gr_sat/decoders/__init__.py`

See `src/gr_sat/decoders/uwe4.py` for a complete reference implementation.

---

## 🗂️ Full Project Structure

```text
.
├── data/                   # Local data storage (gitignored)
│   ├── raw/                # Raw JSONL fetches from SatNOGS DB API
│   ├── interim/            # Decoded telemetry CSVs (all Kaitai fields)
│   └── processed/          # SI-unit "Golden Features" CSVs (ML-ready)
├── docs/                   # Documentation and analysis outputs
│   ├── figures/            # Generated plots and diagrams
│   └── slides.typ          # Typst presentation slides
├── logs/                   # Log files from data pipelines
├── notebooks/              # Jupyter notebooks for EDA and prototyping (Jupytext)
├── scripts/                # Executable pipeline scripts
│   ├── fetch_training_data.py  # Stage 0: SatNOGS API → data/raw/
│   └── process_data.py         # Stage 1+2: raw → interim → processed
├── src/gr_sat/             # Core library code ("The Shared Core")
│   ├── telemetry.py        # TelemetryFrame, DecoderRegistry, process_frame()
│   └── decoders/           # Satellite-specific decoders (Kaitai Structs)
│       └── uwe4.py         # UWE-4 (NORAD 43880) decoder
├── .gemini/
│   └── GEMINI.md           # Active project context and agent mandates
├── DETAILS.md              # Technical architecture and system design
├── justfile                # Task runner configuration
├── pixi.toml               # Environment and dependency management
└── README.md               # This file
```

### Currently Supported Satellites

| Satellite | NORAD ID | Decoder | Status |
| :--- | :--- | :--- | :--- |
| **UWE-4** | 43880 | `decoders/uwe4.py` | ✅ Primary target, ~7 months of data |
