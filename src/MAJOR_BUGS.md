## [Severity: HIGH] - [CORS Misconfiguration Permitting Universal Cross-Origin Access]
- **File(s):** `src/api/main.py` (Lines 51-57)
- **Category:** Security
- **The Flaw:** The API ignores the predefined `_cors_origins()` helper, which securely loads origins from the environment. Instead, it blindly hardcodes `allow_origins=["*"]` in the `CORSMiddleware`.
- **The Impact:** Cross-Origin Information Disclosure. Malicious external websites can silently make requests to the API and exfiltrate sensitive telemetry, model metadata, and anomaly alerts if the backend is accessed via a user's browser.
- **The Fix:**
```python
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
```

---

## [Severity: CRITICAL] - [Database Connection Pool Leak on MQTT Message]
- **File(s):** `src/api/database.py` (Lines 35-41), `src/api/mqtt_client.py` (Line 47)
- **Category:** Performance / Logic Bug
- **The Flaw:** `get_engine()` in `database.py` calls SQLAlchemy's `create_engine()` on every invocation without caching the instance. The `mqtt_client.py` calls `get_engine()` in its `on_message` callback for every single telemetry packet received.
- **The Impact:** Database Connection Exhaustion and Memory Leak. The application will rapidly spawn thousands of isolated connection pools, quickly exceeding the PostgreSQL `max_connections` limit, causing a complete system crash and dropping all incoming live telemetry.
- **The Fix:**
```python
# src/api/database.py
_engine = None

def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    settings = load_database_settings()
    if settings.configured:
        url = settings.url.replace("postgres://", "postgresql://")
        _engine = create_engine(url, pool_pre_ping=True)
        return _engine
    return None
```

---

## [Severity: CRITICAL] - [Blocking I/O in Async Event Loop for Orbital Predictions]
- **File(s):** `src/api/dashboard_data.py` (Line 418), `src/api/main.py` (Line 151)
- **Category:** Performance
- **The Flaw:** `predict_passes` uses `skyfield` to perform a synchronous HTTP request (`load.tle_file(url)`) to Celestrak. This blocking call is executed directly within the asynchronous FastAPI endpoint `operations_passes` without being delegated to a thread pool.
- **The Impact:** Denial of Service (DoS). A single request to `/api/operations/passes` will block the entire FastAPI async event loop while the TLE data downloads, freezing all other concurrent API requests and the live WebSocket dashboard.
- **The Fix:**
```python
# src/api/main.py (requires: from fastapi.concurrency import run_in_threadpool)
    @app.get("/api/operations/passes")
    async def operations_passes(
        lat: float = Query(..., ge=-90.0, le=90.0),
        lon: float = Query(..., ge=-180.0, le=180.0),
        elevation_m: float = Query(default=0.0, ge=-500.0, le=10000.0),
        station_label: str | None = Query(default=None, max_length=80),
        lookahead_hours: int = Query(default=24, ge=1, le=168),
        min_elevation: float = Query(default=10.0, ge=0.0, le=90.0),
        norad_id: int | None = None,
        include_tracks: bool = Query(default=True),
    ) -> dict:
        try:
            return await run_in_threadpool(
                data.predict_passes,
                lat=lat,
                lon=lon,
                elevation_m=elevation_m,
                station_label=station_label,
                lookahead_hours=lookahead_hours,
                min_elevation=min_elevation,
                norad_id=norad_id,
                include_tracks=include_tracks,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
```

---

## [Severity: HIGH] - [Repeated Disk I/O for PyTorch Model Weights on Every Inference]
- **File(s):** `src/api/dashboard_data.py` (Lines 753-755)
- **Category:** Performance
- **The Flaw:** Inside `_score_frames`, the method invokes `load_model_artifacts()` which loads the model metadata, scaler, and PyTorch VAE weights (`.pt` file) from the filesystem on every call. This method is triggered by the MQTT client for every incoming packet.
- **The Impact:** Severe computational bottleneck. Repeatedly loading PyTorch state dicts from disk per telemetry frame creates massive I/O latency, severely degrading real-time ML anomaly detection performance under normal satellite transmission rates.
- **The Fix:**
```python
# src/api/dashboard_data.py
    def _score_frames(
        self,
        norad_id: int,
        df: pd.DataFrame,
        model_status: ModelStatus,
    ) -> pd.DataFrame:
        assert model_status.metadata is not None
        working = df.copy()
        try:
            if not hasattr(self, "_loaded_models"):
                self._loaded_models = {}
                
            if norad_id not in self._loaded_models:
                self._loaded_models[norad_id] = load_model_artifacts(
                    str(norad_id), self.models_dir
                )
                
            scaler, model, metadata = self._loaded_models[norad_id]
            # ... remainder of the function unchanged
```
