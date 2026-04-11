import unittest

import pandas as pd

from gr_sat.processing import annotate_pass_and_cadence_metadata, deduplicate_processed_frames


class DeduplicateProcessedFramesTests(unittest.TestCase):
    def test_exact_duplicates_are_removed_but_distinct_same_timestamp_frames_are_kept(self):
        timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
        df = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "observation_id": 101,
                    "batt_voltage": 3.7,
                    "batt_current": 0.1,
                },
                {
                    "timestamp": timestamp,
                    "observation_id": 202,
                    "batt_voltage": 3.7,
                    "batt_current": 0.1,
                },
                {
                    "timestamp": timestamp,
                    "observation_id": 101,
                    "batt_voltage": 3.7,
                    "batt_current": 0.2,
                },
            ]
        )

        deduplicated, stats = deduplicate_processed_frames(df)

        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(stats["exact_duplicates_removed"], 1)
        self.assertEqual(stats["same_timestamp_multi_payload_groups"], 1)
        self.assertEqual(deduplicated["batt_current"].tolist(), [0.1, 0.2])

    def test_same_observation_different_payloads_are_preserved(self):
        timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
        df = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "observation_id": 555,
                    "batt_voltage": 3.9,
                    "batt_current": 0.1,
                },
                {
                    "timestamp": timestamp,
                    "observation_id": 555,
                    "batt_voltage": 3.9,
                    "batt_current": -0.2,
                },
            ]
        )

        deduplicated, stats = deduplicate_processed_frames(df)

        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(stats["exact_duplicates_removed"], 0)
        self.assertEqual(stats["same_observation_multi_payload_rows"], 2)

    def test_pass_and_cadence_metadata_is_explicitly_retained(self):
        df = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
                    "batt_voltage": 1.0,
                    "temp_batt_a": 10.0,
                },
                {
                    "timestamp": pd.Timestamp("2026-01-01T00:01:00Z"),
                    "batt_voltage": 2.0,
                    "temp_batt_a": 10.0,
                },
                {
                    "timestamp": pd.Timestamp("2026-01-01T00:04:10Z"),
                    "batt_voltage": 4.0,
                    "temp_batt_a": 25.0,
                },
            ]
        )

        annotated = annotate_pass_and_cadence_metadata(
            df,
            pass_gap_seconds=300.0,
            cadence_tolerance_ratio=0.5,
            cadence_min_tolerance_seconds=5.0,
            rolling_window=3,
        )

        self.assertEqual(annotated["pass_id"].tolist(), [0, 0, 0])
        self.assertEqual(annotated["pass_frame_count"].tolist(), [3, 3, 3])
        self.assertEqual(annotated["pass_frame_index"].tolist(), [0, 1, 2])
        self.assertEqual(annotated["dropped_packet_suspect"].tolist(), [False, False, True])
        self.assertEqual(annotated["same_timestamp_collision"].tolist(), [False, False, False])
        self.assertEqual(annotated["volt_rolling_std"].iloc[0], 0.0)
        self.assertGreater(annotated["temp_rolling_std"].iloc[2], 0.0)


if __name__ == "__main__":
    unittest.main()
