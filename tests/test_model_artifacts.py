import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from gr_sat.ml_config import ALL_FEATURES
from gr_sat.model_artifacts import (
    ModelArtifactMetadata,
    load_model_artifacts,
    load_model_metadata,
    model_artifact_paths,
    save_model_metadata,
    split_chronological,
    threshold_from_scores,
)
from gr_sat.models import TelemetryVAE


class ModelArtifactTests(unittest.TestCase):
    def test_split_chronological_creates_stable_ordered_partitions(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"),
                "value": list(range(10)),
            }
        )

        split = split_chronological(df, train_fraction=0.6, validation_fraction=0.2)

        self.assertEqual(len(split.train), 6)
        self.assertEqual(len(split.validation), 2)
        self.assertEqual(len(split.test), 2)
        self.assertLess(split.train["timestamp"].max(), split.validation["timestamp"].min())
        self.assertLess(split.validation["timestamp"].max(), split.test["timestamp"].min())

    def test_threshold_round_trip_and_bundle_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            norad_id = "99999"
            paths = model_artifact_paths(models_dir, norad_id)

            scaler = StandardScaler().fit(
                np.array(
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                        [2.0, 3.0, 4.0, 5.0, 6.0],
                    ]
                )
            )
            joblib.dump(scaler, paths.scaler)

            model = TelemetryVAE(input_dim=len(ALL_FEATURES), hidden_dim=4, latent_dim=2)
            torch.save(model.state_dict(), paths.weights)

            metadata = ModelArtifactMetadata(
                version=1,
                norad_id=norad_id,
                feature_names=list(ALL_FEATURES),
                hidden_dim=4,
                latent_dim=2,
                kld_weight=0.05,
                threshold=threshold_from_scores(np.array([0.1, 0.2, 0.3])),
                threshold_percentile=95.0,
                inference_mode="deterministic",
                train_rows=8,
                validation_rows=1,
                test_rows=1,
                train_start="2026-01-01T00:00:00+00:00",
                train_end="2026-01-01T07:00:00+00:00",
                validation_start="2026-01-01T08:00:00+00:00",
                validation_end="2026-01-01T08:00:00+00:00",
                test_start="2026-01-01T09:00:00+00:00",
                test_end="2026-01-01T09:00:00+00:00",
            )
            save_model_metadata(paths.metadata, metadata)

            loaded_metadata = load_model_metadata(paths.metadata)
            loaded_scaler, loaded_model, loaded_bundle_metadata = load_model_artifacts(
                norad_id,
                models_dir,
            )

            self.assertEqual(loaded_metadata.threshold, metadata.threshold)
            self.assertEqual(loaded_bundle_metadata.feature_names, list(ALL_FEATURES))
            self.assertEqual(loaded_model.fc1.in_features, len(ALL_FEATURES))
            self.assertAlmostEqual(
                loaded_scaler.mean_[0],
                scaler.mean_[0],
            )


if __name__ == "__main__":
    unittest.main()
