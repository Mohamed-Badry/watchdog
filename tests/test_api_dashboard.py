import tempfile
import unittest
from pathlib import Path

import httpx
import pandas as pd

from api.dashboard_data import DashboardDataRepository
from api.main import create_app
from gr_sat.model_artifacts import ModelArtifactMetadata, model_artifact_paths, save_model_metadata


class DashboardApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.processed_dir = self.root / "data" / "processed"
        self.models_dir = self.root / "models"
        self.processed_dir.mkdir(parents=True)
        self.models_dir.mkdir(parents=True)

        pd.DataFrame(
            [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "batt_voltage": 3.9,
                    "batt_current": -0.1,
                    "temp_batt_a": 10.0,
                    "temp_batt_b": 11.0,
                    "temp_panel_z": 20.0,
                    "frame_is_complete": True,
                    "missing_raw_fields": "",
                    "missing_raw_field_count": 0,
                    "sampling_irregular": False,
                    "dropped_packet_suspect": False,
                    "pass_id": 0,
                    "pass_frame_index": 0,
                    "pass_frame_count": 2,
                    "anomaly_score": 0.10,
                    "is_anomaly": False,
                },
                {
                    "timestamp": "2026-01-01T00:01:00Z",
                    "batt_voltage": 3.8,
                    "batt_current": -0.2,
                    "temp_batt_a": 12.0,
                    "temp_batt_b": 13.0,
                    "temp_panel_z": 18.0,
                    "frame_is_complete": True,
                    "missing_raw_fields": "",
                    "missing_raw_field_count": 0,
                    "sampling_irregular": True,
                    "dropped_packet_suspect": True,
                    "pass_id": 0,
                    "pass_frame_index": 1,
                    "pass_frame_count": 2,
                    "anomaly_score": 0.40,
                    "is_anomaly": True,
                },
                {
                    "timestamp": "2026-01-02T00:00:00Z",
                    "batt_voltage": 4.0,
                    "batt_current": 0.1,
                    "temp_batt_a": 9.0,
                    "temp_batt_b": 10.0,
                    "temp_panel_z": 14.0,
                    "frame_is_complete": False,
                    "missing_raw_fields": "[\"beacon_payload_batt_b_voltage\"]",
                    "missing_raw_field_count": 1,
                    "sampling_irregular": False,
                    "dropped_packet_suspect": False,
                    "pass_id": 1,
                    "pass_frame_index": 0,
                    "pass_frame_count": 2,
                    "anomaly_score": 0.20,
                    "is_anomaly": False,
                },
                {
                    "timestamp": "2026-01-02T00:02:00Z",
                    "batt_voltage": 5.1,
                    "batt_current": 1.2,
                    "temp_batt_a": 30.0,
                    "temp_batt_b": 31.0,
                    "temp_panel_z": 35.0,
                    "frame_is_complete": True,
                    "missing_raw_fields": "",
                    "missing_raw_field_count": 0,
                    "sampling_irregular": False,
                    "dropped_packet_suspect": False,
                    "pass_id": 1,
                    "pass_frame_index": 1,
                    "pass_frame_count": 2,
                    "anomaly_score": 0.50,
                    "is_anomaly": True,
                },
            ]
        ).to_csv(self.processed_dir / "43880.csv", index=False)

        metadata = ModelArtifactMetadata(
            version=2,
            norad_id="43880",
            feature_names=[
                "batt_voltage",
                "batt_current",
                "temp_batt_a",
                "temp_batt_b",
                "temp_panel_z",
            ],
            hidden_dim=12,
            latent_dim=3,
            kld_weight=0.05,
            threshold=0.30,
            threshold_percentile=95.0,
            inference_mode="deterministic",
            train_rows=10,
            validation_rows=2,
            test_rows=2,
            train_start="2026-01-01T00:00:00+00:00",
            train_end="2026-01-01T00:01:00+00:00",
            validation_start="2026-01-02T00:00:00+00:00",
            validation_end="2026-01-02T00:00:00+00:00",
            test_start="2026-01-02T00:02:00+00:00",
            test_end="2026-01-02T00:02:00+00:00",
            feature_contract_version=3,
            diagnosis_feature_names=[
                "batt_voltage",
                "batt_current",
                "temp_batt_a",
                "temp_batt_b",
                "temp_panel_z",
            ],
        )
        save_model_metadata(model_artifact_paths(self.models_dir, "43880").metadata, metadata)

        repository = DashboardDataRepository(root=self.root)
        self.app = create_app(repository)

    def tearDown(self):
        self.tmpdir.cleanup()

    async def _get(self, path: str, params: dict | None = None):
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path, params=params)

    async def test_status_exposes_dashboard_links_and_components(self):
        response = await self._get("/api/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "online")
        self.assertIn("dashboard_summary", payload["links"])
        self.assertEqual(payload["supported_satellites"][0]["norad_id"], 43880)
        component_names = {component["name"] for component in payload["components"]}
        self.assertEqual(
            component_names,
            {"api", "database", "processed_data", "model_artifacts"},
        )

    async def test_dashboard_summary_returns_exact_home_metrics(self):
        response = await self._get("/api/dashboard/summary")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["totals"],
            {
                "satellite_count": 1,
                "frame_count": 4,
                "anomaly_count": 2,
                "partial_frame_count": 1,
                "pass_count": 2,
            },
        )
        self.assertEqual(payload["window"]["start"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(payload["window"]["end"], "2026-01-02T00:02:00+00:00")
        self.assertEqual(payload["active_satellites"][0]["model"]["threshold"], 0.30)
        self.assertEqual(len(payload["recent_anomalies"]), 2)
        self.assertEqual(payload["recent_anomalies"][0]["score"], 0.50)

    async def test_recent_telemetry_contract_preserves_features_quality_and_model(self):
        response = await self._get(
            "/api/telemetry/recent",
            params={"norad_id": 43880, "limit": 3},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["limit"], 3)
        self.assertEqual(len(payload["frames"]), 3)

        latest = payload["frames"][0]
        self.assertEqual(latest["timestamp"], "2026-01-02T00:02:00+00:00")
        self.assertEqual(latest["features"]["batt_voltage"], 5.1)
        self.assertTrue(latest["model"]["is_anomaly"])
        self.assertEqual(latest["model"]["threshold"], 0.30)

        partial = payload["frames"][1]
        self.assertFalse(partial["quality"]["frame_is_complete"])
        self.assertEqual(
            partial["quality"]["missing_raw_fields"],
            ["beacon_payload_batt_b_voltage"],
        )

    async def test_recent_anomalies_are_sorted_by_timestamp_descending(self):
        response = await self._get("/api/anomalies/recent", params={"norad_id": 43880})

        self.assertEqual(response.status_code, 200)
        anomalies = response.json()["anomalies"]
        self.assertEqual(
            [anomaly["timestamp"] for anomaly in anomalies],
            [
                "2026-01-02T00:02:00+00:00",
                "2026-01-01T00:01:00+00:00",
            ],
        )

    async def test_throughput_buckets_counts_frames_and_anomalies(self):
        response = await self._get(
            "/api/telemetry/throughput",
            params={"norad_id": 43880, "bucket": "day", "limit": 10},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_frames"], 4)
        self.assertEqual(payload["returned_frame_count"], 4)
        self.assertEqual(
            payload["buckets"],
            [
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "frame_count": 2,
                    "anomaly_count": 1,
                },
                {
                    "timestamp": "2026-01-02T00:00:00+00:00",
                    "frame_count": 2,
                    "anomaly_count": 1,
                },
            ],
        )

    async def test_unknown_satellite_returns_404(self):
        response = await self._get("/api/telemetry/recent", params={"norad_id": 99999})

        self.assertEqual(response.status_code, 404)
        self.assertIn("No processed telemetry dataset", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
