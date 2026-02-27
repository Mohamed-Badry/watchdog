# Justfile for Project Watchdog

set shell := ["bash", "-c"]

# List available recipes
default:
    @just --list

# --- Phase 1: The Lab (Data Acquisition) ---

# Fetch telemetry (Interactive or Batch)
# Usage:
#   just fetch              -> Interactive Menu
#   just fetch --all        -> Download all (default 30 days)
#   just fetch --days 7     -> Interactive (default 7 days)
fetch +args='':
    pixi run python scripts/fetch_training_data.py {{args}}

# --- Phase 2: Operations (Analysis & Viz) ---

# Regenerate target analysis and selection
analyze-targets:
    pixi run python notebooks/sat_analysis.py

# Generate operational dashboards (Skyplots, Gantt)
viz-passes:
    pixi run python notebooks/pass_analysis_viz.py

# Run the full visualization pipeline (Select -> Visualize)
regenerate-all: analyze-targets viz-passes

# --- Utilities ---

# Sync Jupyter Notebooks from Scripts
sync-notebooks:
    @for script in notebooks/*.py; do \
        filename=$(basename -- "$script"); \
        name="${filename%.*}"; \
        echo "Syncing notebook-script $script to notebooks/$name.ipynb"; \
        pixi run jupytext --to notebook --update "$script" --output "notebooks/$name.ipynb"; \
    done

# Convert a Python script to a Jupyter Notebook
# Usage: just convert notebooks/script.py
convert script_path:
    pixi run jupytext --to notebook "{{script_path}}"

# Clean temporary files (pycache, etc.)
clean:
    rm -rf __pycache__ .pytest_cache
    find . -name "*.pyc" -delete

# Start zellij with the dev layout
zj:
    zellij --layout dev_zj.kdl
