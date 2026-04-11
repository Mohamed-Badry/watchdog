import unittest

import pandas as pd

from gr_sat.satellite_profiles import (
    build_baseline_mask,
    feature_completeness_mask,
    get_satellite_profile,
)


class SatelliteProfileTests(unittest.TestCase):
    def test_uwe4_profile_exposes_versioned_feature_contract(self):
        profile = get_satellite_profile(43880)

        self.assertEqual(profile.feature_contract.version, 2)
        self.assertIn("power_consumption", profile.feature_contract.feature_names)
        self.assertIn("volt_rolling_std", profile.feature_contract.feature_names)

    def test_baseline_and_completeness_masks_are_explicit(self):
        profile = get_satellite_profile(43880)
        df = pd.DataFrame(
            [
                {
                    "batt_voltage": 4.0,
                    "batt_current": 0.1,
                    "batt_a_voltage": 4.0,
                    "batt_b_voltage": 4.0,
                    "batt_a_current": 0.1,
                    "batt_b_current": 0.0,
                    "power_consumption": 1.0,
                    "temp_obc": 20.0,
                    "temp_batt_a": 10.0,
                    "temp_batt_b": 11.0,
                    "temp_panel_z": 12.0,
                    "volt_rolling_std": 0.1,
                    "temp_rolling_std": 0.2,
                },
                {
                    "batt_voltage": 5.5,
                    "batt_current": 0.1,
                    "batt_a_voltage": 4.0,
                    "batt_b_voltage": 4.0,
                    "batt_a_current": 0.1,
                    "batt_b_current": 0.0,
                    "power_consumption": 1.0,
                    "temp_obc": 20.0,
                    "temp_batt_a": 10.0,
                    "temp_batt_b": 11.0,
                    "temp_panel_z": 12.0,
                    "volt_rolling_std": 0.1,
                    "temp_rolling_std": 0.2,
                },
                {
                    "batt_voltage": 4.0,
                    "batt_current": 0.1,
                    "batt_a_voltage": None,
                    "batt_b_voltage": 4.0,
                    "batt_a_current": 0.1,
                    "batt_b_current": 0.0,
                    "power_consumption": 1.0,
                    "temp_obc": 20.0,
                    "temp_batt_a": 10.0,
                    "temp_batt_b": 11.0,
                    "temp_panel_z": 12.0,
                    "volt_rolling_std": 0.1,
                    "temp_rolling_std": 0.2,
                },
            ]
        )

        baseline_mask = build_baseline_mask(df, profile)
        complete_mask = feature_completeness_mask(df, profile.feature_contract.feature_names)

        self.assertEqual(baseline_mask.tolist(), [False, True, False])
        self.assertEqual(complete_mask.tolist(), [True, True, False])


if __name__ == "__main__":
    unittest.main()
