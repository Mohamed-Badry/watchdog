# Adding a New Satellite to GR_SAT

This guide details the complete end-to-end process required to add a new satellite to the GR_SAT platform, process its telemetry, and deploy a machine learning model for live anomaly detection.

## 1. Register the Satellite Profile

The core of the satellite integration lies in the central profile registry. You must add the satellite's configuration to `src/gr_sat/core/satellite_profiles.py`.

*   **File Touched**: `src/gr_sat/core/satellite_profiles.py`
*   **Action**: Create a new `SatelliteProfile` instance with its specific `FeatureContract` and add it to the `_SATELLITE_PROFILES` dictionary.
*   **Details**: Define the exact ML feature keys the satellite will use. These feature keys **must strictly map** to the universal fields defined in the `TelemetryFrame` DTO.

## 2. Implement the Telemetry Decoder

To parse raw Kaitai binary payloads, you need a dedicated decoder class that implements the `BaseDecoder` interface.

*   **Files Touched**: `src/gr_sat/core/decoders/<satellite_name>.py` and `src/gr_sat/core/decoders/__init__.py`
*   **Action**: Create a new python file for the decoder. Create a class (e.g., `CuteDecoder(BaseDecoder)`) and decorate it with `@DecoderRegistry.register(<NORAD_ID>)`.
*   **Details**: 
    *   `decode_with_diagnostics()`: Convert the byte array into a flat dictionary using `satnogsdecoders.decoder.get_fields()`.
    *   `adapt()`: Map the extracted, satellite-specific Kaitai variable names (e.g., `soh_analogs_battery_voltage`) directly to the universal `TelemetryFrame` variables specified in your Feature Contract (e.g., `batt_voltage`). Ensure NaNs are replaced with 0.0 to prevent ML crashes.

## 3. Update the Universal DTO (If Necessary)

If the new satellite requires novel numerical parameters (e.g., a specific solar panel temperature that doesn't fit existing slots), you may need to expand the global DTO.

*   **File Touched**: `src/gr_sat/core/telemetry.py`
*   **Action**: Add the new fields to the `TelemetryFrame` dataclass (e.g., `temp_panel_z: Optional[float] = None`).

## 4. Download and Process Historical Data

Once the code is in place, you need to populate the data directories for ML training.

*   **Action**: Run the fetch and process scripts.
*   **Commands**:
    *   `just fetch --norad <NORAD_ID> --days 30` (Requires `curl` hitting the SatNOGS DB).
    *   `just process --norad <NORAD_ID>`
*   **Details**: The `process_data.py` script will automatically locate your new `Decoder` via the `DecoderRegistry`, adapt the raw JSONL frames from `data/raw/<NORAD_ID>/`, and output a unified, ML-ready CSV file to `data/processed/<NORAD_ID>.csv`.

## 5. Train the Anomaly Detection Model

*   **Action**: Train the PyTorch Variational Autoencoder (VAE) on the processed CSV.
*   **Command**: `just train --norad <NORAD_ID> --epochs 100`
*   **Details**: The script reads the satellite's `FeatureContract` to mask invalid outliers, trains the model, and outputs the model weights to `models/<NORAD_ID>_vae.pt` and its threshold metadata to `models/<NORAD_ID>_metadata.json`.

## 6. Restart Backend and Simulator Environments

Since the live system runs in Docker containers, they must be rebuilt/restarted to pick up the new Python decoder classes and the ML weights.

*   **Command**: `docker compose build backend simulator && docker compose up -d backend simulator`
*   **Details**: The API backend dynamically loads the model weights for any NORAD ID it discovers in the DB. The simulator recursively loops through all `.jsonl` files in `data/raw/` to replay the historical telemetry over MQTT using original timestamps.
