import unittest
from datetime import datetime, timezone

from gr_sat.decoders.uwe4 import UWE4Decoder
from gr_sat.telemetry import process_frame_result


class TelemetryDiagnosticsTests(unittest.TestCase):
    def test_adapter_preserves_missingness_instead_of_coercing_to_zero(self):
        decoder = UWE4Decoder()

        outcome = decoder.adapt_with_diagnostics(
            {
                "beacon_payload_batt_a_voltage": 4000,
                "beacon_payload_batt_b_voltage": None,
                "beacon_payload_batt_a_current": 100,
                "beacon_payload_batt_b_current": None,
                "beacon_payload_power_consumption": 500,
                "beacon_payload_obc_temp": 21,
                "beacon_payload_batt_a_temp": 12,
                "beacon_payload_batt_b_temp": 13,
                "beacon_payload_panel_pos_z_temp": 14,
                "beacon_payload_uptime": 123,
            }
        )

        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.data["batt_a_voltage"], 4.0)
        self.assertIsNone(outcome.data["batt_b_voltage"])
        self.assertIsNone(outcome.data["batt_voltage"])
        self.assertFalse(outcome.data["frame_is_complete"])
        self.assertEqual(outcome.data["missing_raw_field_count"], 2)

    def test_adapter_reports_invalid_numeric_values(self):
        decoder = UWE4Decoder()

        outcome = decoder.adapt_with_diagnostics(
            {
                "beacon_payload_batt_a_voltage": "bad",
                "beacon_payload_batt_b_voltage": 4000,
                "beacon_payload_batt_a_current": 100,
                "beacon_payload_batt_b_current": 50,
                "beacon_payload_power_consumption": 500,
                "beacon_payload_obc_temp": 21,
                "beacon_payload_batt_a_temp": 12,
                "beacon_payload_batt_b_temp": 13,
                "beacon_payload_panel_pos_z_temp": 14,
                "beacon_payload_uptime": 123,
            }
        )

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.failure.code, "invalid_numeric_value")

    def test_process_frame_result_reports_missing_decoder(self):
        result = process_frame_result(
            norad_id=99999,
            payload=b"\x00",
            source="test",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.failure.code, "no_decoder")


if __name__ == "__main__":
    unittest.main()
