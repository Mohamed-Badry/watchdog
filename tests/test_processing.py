import unittest

import pandas as pd

from gr_sat.processing import deduplicate_processed_frames


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


if __name__ == "__main__":
    unittest.main()
