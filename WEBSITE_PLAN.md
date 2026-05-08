# gr_sat Web Interface & Integration Plan

Based on the analysis of the current `gr_sat` repository, this document outlines the state of the project, major architectural flaws related to online operations, and a comprehensive plan for building a modern web-based frontend (SvelteKit) and API backend (FastAPI), orchestrated via **Docker**. The initial Docker and SvelteKit scaffolding now exists in the repository; this document still describes the broader target UX and service behavior that remain to be built.

## 1. Current State Analysis
The `gr_sat` repository has successfully established a robust **offline machine learning pipeline** for satellite telemetry:
- **Data Pipeline:** Fetching (SatNOGS API) -> Decoding (Kaitai Structs) -> Normalization (Golden Features).
- **Modeling:** Per-satellite Autoencoders/VAEs are trained and evaluated using synthetic-fault benchmarking.
- **Core Logic:** The Python codebase (`src/gr_sat`) is well-structured with dedicated modules for decoding, processing, profiles, and models.
- **Minimal Runtime:** A rudimentary, single-threaded online inference loop (`watchdog_runtime.py`) exists but lacks network ingress, persistence, or a UI.

---

## 2. Containerization Strategy & Tech Stack (Docker)
Dockerizing this project provides a clean separation of concerns and makes deploying to edge hardware (like a ground station PC or Raspberry Pi) trivial. We will use **Docker Compose** to orchestrate 5 microservices:

1. **`broker` (Eclipse Mosquitto - MQTT):** 
   - *Tech:* C (Mosquitto).
   - *Role:* The message bus handling live high-frequency telemetry from the antenna. Extremely lightweight.
2. **`db` (PostgreSQL + TimescaleDB):** 
   - *Tech:* PostgreSQL with the TimescaleDB extension.
   - *Role:* Time-series optimized database to persist telemetry history, ML anomalies, and pass metadata.
3. **`backend` (FastAPI):** 
   - *Tech:* Python, FastAPI, Uvicorn, PyTorch, Paho-MQTT, SQLAlchemy/SQLModel.
   - *Role:* Subscribes to the `broker`, runs the Kaitai decoders & VAE ML inference, writes to the `db`, and serves WebSockets/REST APIs to the frontend.
4. **`frontend` (SvelteKit):** 
   - *Tech:* **Bun** (Runtime/Package Manager - *No Node.js*), Svelte 5, TypeScript, TailwindCSS, Shadcn-Svelte/Skeleton.
   - *Role:* Serves the real-time UI dashboard and orbital tracking visualizations.
5. **`simulator` (Python):** 
   - *Tech:* Python.
   - *Role:* A testing container that replays historical offline data from `data/raw/` into the `broker` to simulate a live antenna.

---

## 3. Database Schema Strategy (Raw vs. Processed)

To ensure the web dashboard is extremely fast, **we must store both raw and processed data.** 

If we only stored the `raw_frame`, the backend would be forced to re-run the Kaitai decoder and the PyTorch ML model every single time the user opens the dashboard or changes the timeframe to view historical charts. This is computationally wasteful.

### Proposed Table Schema: `telemetry_frames` (TimescaleDB Hypertable)
We will use a hypertable partitioned by `timestamp`.

- `id` (UUID or BigSerial): Primary Key.
- `timestamp` (Timestamptz): Exact time of reception (Hypertable index).
- `norad_id` (Integer): Satellite identifier (e.g., 43880).
- `station_id` (String): Which antenna received it.
- `raw_frame` (String): The original hex-encoded payload.
- `features` (JSONB): The decoded "Golden Features" (e.g., `{"batt_voltage": 5.1, "temp_batt_a": 12.0}`). Using JSONB means we don't need to migrate the database schema every time a new satellite profile with different features is added.
- `anomaly_score` (Float): The loss value from the VAE model.
- `is_anomaly` (Boolean): Whether the score exceeded the pre-calibrated threshold.
- `missing_fields` (JSONB/Array): List of fields that could not be parsed.

---

## 4. Telemetry Streaming: The MQTT Antenna Contract

The antenna/demodulator software (GNU Radio, SatNOGS client, etc.) simply needs to act as an MQTT Publisher. It should publish a JSON payload to the topic `telemetry/live/{norad_id}` (e.g., `telemetry/live/43880`).

**Required Output Shape:**
```json
{
  "norad_id": 43880,
  "timestamp": "2026-05-05T14:30:22Z",
  "raw_frame": "8A8A8A8A8A8A...",
  "station_id": "my_local_antenna_1",
  "snr": 12.5 
}
```

---

## 5. Recommended Repository Layout

```text
gr_sat/
├── data/                   # Existing data
├── docs/                   # Diagrams and documentation
├── scripts/                # Existing offline ML scripts
├── docker-compose.yml      # Orchestrates all 5 containers
├── src/
│   ├── gr_sat/             # Existing Core Library
│   ├── api/                # FastAPI Backend
│   │   ├── Dockerfile
│   │   ├── main.py         # REST/WebSocket endpoints
│   │   ├── mqtt_client.py  # Subscribes to broker, triggers ML inference
│   │   ├── database.py     # SQLAlchemy async connection
│   │   └── routers/        # Modular endpoint groups
│   │       ├── status.py
│   │       ├── operations.py
│   │       ├── insights.py
│   │       ├── ml.py
│   │       └── websocket.py
│   └── simulator/          # Antenna mock
│       ├── Dockerfile
│       └── replay.py       # Reads data/raw/ and publishes to MQTT
├── frontend/               # SvelteKit Project
│   ├── Dockerfile          # Uses oven/bun base image
│   ├── bun.lock
│   ├── src/
│   │   ├── app.html        # Main HTML shell
│   │   ├── routes/
│   │   │   ├── +layout.svelte     # Root: theme toggle, app.css
│   │   │   ├── (landing)/         # Route group: antenna bg, top nav
│   │   │   │   ├── +layout.svelte
│   │   │   │   ├── +page.svelte   # Landing hero
│   │   │   │   └── team/+page.svelte
│   │   │   └── (dashboard)/       # Route group: sidebar, footer
│   │   │       ├── +layout.svelte
│   │   │       └── dashboard/
│   │   │           ├── +page.svelte       # Dashboard home
│   │   │           ├── operations/        # Pass prediction, skyplots
│   │   │           ├── live/              # Live packet watcher
│   │   │           ├── insights/          # EDA & telemetry explorer
│   │   │           └── ml/               # VAE vs Z-Score, model health
│   │   └── lib/
│   │       ├── components/    # Shared + page-specific components
│   │       └── shaders/       # WebGL shader sources
│   └── tailwind.config.js     # Dual-color Theme definitions
└── README.md
```

---

## 6. Theme, Styling & Layout Architecture (SvelteKit)

The UI will be designed around a **Dual-Color Light/Dark Theme** driven by CSS variables to ensure strict modularity and easy white-labeling. 

### A. Navigation Architecture
The site uses **two distinct layout shells**:

1. **Landing Shell** (`(landing)` route group): For `/` and `/team`.
   - Top navbar with Overview / Team tabs.
   - Holographic antenna WebGL background.
   - Striking "Enter Dashboard →" CTA button on the landing hero.

2. **Dashboard Shell** (`(dashboard)` route group): For `/dashboard/**`.
   - **No** Overview/Team tabs in the header.
   - Collapsible sidebar with sub-page links: Operations, Live, Insights, ML Lab.
   - Footer with links back to Overview/Team/GitHub.
   - Logo click returns to landing (`/`).
   - No antenna background (dashboard has its own data-focused aesthetic).

### B. Color Palette
Derived from the academic Typst templates, adapting to web conventions:
- **Primary Accent:** Pinkish Red (`#B12142`) - Used for highlights, active states, and H1 banners.
- **Secondary Accent:** Muted Slate Gray (`#6C7A96`) - Used for secondary text, borders, and sub-headers.
- **Light Mode Base:** Near-white (`#F8FAFC`) with dark text (`#111827`).
- **Dark Mode Base:** AMOLED black (`#000000`) with light text (`#F8FAFC`).

### C. WebGL Background Integration
The `(landing)/+layout.svelte` will mount a `<canvas>` element fixed to the background (`z-index: -1`). 
- **Effect:** A grid of antennas that orient themselves to point towards the user's mouse cursor.
- **Animation:** They will shoot a small oscillating signal (using the Primary/Secondary CSS variables for colorization).
- **Performance:** Rendered via WebGL ensuring it does not block the Svelte UI thread.
- **Scope:** Only active on landing/team pages — disabled inside dashboard.

---

## 7. Dashboard Architecture

> **Full specification with API contracts, data shapes, and component hierarchy:**
> See `dashboard_plan.md` (artifact) for the complete dashboard blueprint.

### Summary of Dashboard Sub-Pages:

1. **Dashboard Home** (`/dashboard`) — Service status grid, active satellites, recent anomalies, throughput sparkline.
2. **Operations** (`/dashboard/operations`) — Pass prediction, skyplots, timeline Gantt, satellite rankings. Powered by Skyfield.
3. **Live Watcher** (`/dashboard/live`) — Real-time packet decode visualization, feature gauges, anomaly score timeline. Powered by WebSocket.
4. **EDA & Insights** (`/dashboard/insights`) — Historical telemetry explorer, distributions, eclipse scatter, correlation heatmap, PCA projection.
5. **ML Lab** (`/dashboard/ml`) — VAE vs Z-Score sensitivity curves, ROC comparisons, score distributions, latent space visualization, threshold tuning.

---

## 8. Execution Phases

### Phase 1 — Foundation (Current)
- Restructure frontend routes into `(landing)` and `(dashboard)` groups
- Build DashboardLayout with sidebar + footer
- Create dashboard home with placeholder cards
- Implement `database.py` and `GET /api/status`

### Phase 2 — Live Pipeline
- Implement MQTT subscriber → decode → score → persist
- Implement `WS /api/ws/telemetry`
- Build PipelineVisualizer (animated decode flow)
- Wire simulator → broker → backend → frontend end-to-end

### Phase 3 — Operations
- Port pass prediction logic into backend service
- Build skyplot, schedule table, timeline Gantt

### Phase 4 — Insights & ML Lab
- Implement aggregation queries for EDA
- Port sensitivity sweep into backend
- Build all insight + ML frontend components

### Phase 5 — Polish
- Responsive sidebar, loading skeletons, error boundaries
- Theme verification, performance audit