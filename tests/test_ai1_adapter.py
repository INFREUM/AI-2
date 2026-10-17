"""Tests for the AI① envelope to canonical AI② payload translation."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.ai1_adapter import to_ai2_payload, unix_microseconds_to_utc_iso
from src.api_models import Ai1WindowRequest
from src.baseline_manager import FEATURES
from tests.test_api_models import request_payload


class Ai1AdapterTests(unittest.TestCase):
    def test_utc_microseconds_conversion(self):
        value = 1792924229670000
        self.assertEqual(unix_microseconds_to_utc_iso(value), datetime.fromtimestamp(value / 1_000_000, timezone.utc).isoformat())

    def test_mapping_uses_model_feature_order_and_preserves_metadata(self):
        request = Ai1WindowRequest.model_validate(request_payload())
        canonical = to_ai2_payload(request)
        self.assertEqual(list(canonical["features"]), FEATURES)
        self.assertEqual(canonical["experiment_id"], request.experiment_id)
        self.assertEqual(canonical["environment"]["load_position"], "C")
        self.assertEqual(canonical["event_time"]["window_end_us"], request.window_end_us)
        self.assertNotIn("raw", canonical)
        self.assertEqual(canonical["ai1_result"]["prediction"], "Normal")


if __name__ == "__main__":
    unittest.main()
