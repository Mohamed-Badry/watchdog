# DISCLAIMER: This project is still nowhere near complete and heavily AI-assisted use at your own discretion.

# Project Watchdog (gr_sat)

This repository hosts the **Project Watchdog** codebase, an AI-powered system for real-time anomaly detection in amateur satellite telemetry.

## 📚 Key Documentation

*   **[Technical Details & Architecture](DETAILS.md)**: Deep dive into the system's design and "Golden Features".
*   **[Project Context & Status](docs/GEMINI_PROJECT.md)**: Current phase, team objectives, and "Gemini" integration details.

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
iwr -useb https://pixi.sh/install.ps1 | iex
```

### 2. Setup the Project
Clone the repo and install dependencies:
```bash
git clone https://github.com/Mohamed-Badry/watchdog.git
cd gr_sat
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
