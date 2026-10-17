"""Deterministic synthetic-message tests; no actual HW measurements are used."""

from __future__ import annotations

import copy
import json
import unittest
from datetime import datetime, timedelta, timezone

from src.baseline_manager import FEATURES
from src.feature_change import load_baseline_lookup
from src.realtime_processor import RealtimeRiskProcessor


LOOKUP = load_baseline_lookup()
NOW = datetime(2026, 10, 15, 3, 30, tzinfo=timezone.utc)


def fixture(bridge_type: str, condition: str, window_id: str, timestamp: datetime, message_id: str | None = None) -> dict:
    entry = LOOKUP["baselines"][bridge_type][condition]
    load_text, position = condition.split("-", 1)
    payload = {
        "schema_version": "synthetic-realtime-test-v1", "bridge_type": bridge_type,
        "window_id": window_id, "timestamp": timestamp.isoformat(),
        "features": {feature: entry["features"][feature]["mean"] for feature in FEATURES},
        "environment": {"load_level_kg": float(load_text.removesuffix("kg")), "load_position": position, "temperature_c": 24.5},
        "ai1_result": {"anomaly_score": 0.42, "decision_ratio": 0.8, "raw_prediction": "Normal", "prediction": "Normal"},
    }
    if message_id:
        payload["message_id"] = message_id
    return payload


class RealtimeProcessorTests(unittest.TestCase):
    def setUp(self):
        self.processor = RealtimeRiskProcessor(max_age_seconds=60, max_future_seconds=10, baseline_lookup=LOOKUP)

    def test_box_sequential_windows(self):
        self.processor.process(fixture("BOX", "1kg-C", "W001", NOW - timedelta(seconds=5)), now=NOW)
        result = self.processor.process(fixture("BOX", "1kg-C", "W002", NOW - timedelta(seconds=2)), now=NOW)
        self.assertEqual(result["processing_status"], "PROCESSED")
        self.assertEqual(self.processor.bridge_snapshot("BOX")["latest_window_id"], "W002")

    def test_truss_sequential_windows(self):
        result = self.processor.process(fixture("TRUSS", "2kg-E", "W001", NOW - timedelta(seconds=1)), now=NOW)
        self.assertEqual(result["bridge_type"], "TRUSS")
        self.assertEqual(self.processor.bridge_snapshot("TRUSS")["latest_window_id"], "W001")

    def test_bridge_state_is_independent(self):
        self.processor.process(fixture("BOX", "1kg-C", "W001", NOW), now=NOW)
        self.processor.process(fixture("TRUSS", "1kg-C", "W001", NOW), now=NOW)
        self.assertEqual(self.processor.bridge_snapshot("BOX")["processed_window_count"], 1)
        self.assertEqual(self.processor.bridge_snapshot("TRUSS")["processed_window_count"], 1)

    def test_same_window_retransmission_is_cached_duplicate(self):
        payload = fixture("BOX", "1kg-C", "W001", NOW, "m1")
        self.processor.process(payload, now=NOW)
        result = self.processor.process(payload, now=NOW)
        self.assertTrue(result["duplicate"])
        self.assertEqual(result["processing_status"], "DUPLICATE")
        self.assertIn("DUPLICATE_WINDOW", result["data_quality"]["statuses"])

    def test_different_message_id_same_window_is_duplicate(self):
        self.processor.process(fixture("BOX", "1kg-C", "W001", NOW, "m1"), now=NOW)
        result = self.processor.process(fixture("BOX", "1kg-C", "W001", NOW, "m2"), now=NOW)
        self.assertTrue(result["duplicate"])
        self.assertEqual(result["message_id"], "m2")

    def test_same_window_different_content_is_conflict(self):
        payload = fixture("BOX", "1kg-C", "W001", NOW)
        self.processor.process(payload, now=NOW)
        conflict = copy.deepcopy(payload)
        conflict["features"]["acc01_rms_x"] += 0.1
        result = self.processor.process(conflict, now=NOW)
        self.assertEqual(result["processing_status"], "WINDOW_CONFLICT")
        self.assertEqual(result["evidence"]["error"]["code"], "WINDOW_CONFLICT")

    def test_stale_timestamp(self):
        result = self.processor.process(fixture("BOX", "1kg-C", "W001", NOW - timedelta(seconds=61)), now=NOW)
        self.assertEqual(result["processing_status"], "STALE_DATA")
        self.assertIn("STALE_DATA", result["data_quality"]["statuses"])
        self.assertIsNone(self.processor.bridge_snapshot("BOX")["latest_window_id"])

    def test_future_timestamp(self):
        result = self.processor.process(fixture("BOX", "1kg-C", "W001", NOW + timedelta(seconds=11)), now=NOW)
        self.assertEqual(result["processing_status"], "FUTURE_TIMESTAMP")
        self.assertIn("FUTURE_TIMESTAMP", result["data_quality"]["statuses"])

    def test_out_of_order_does_not_replace_latest(self):
        self.processor.process(fixture("BOX", "1kg-C", "W003", NOW - timedelta(seconds=1)), now=NOW)
        result = self.processor.process(fixture("BOX", "1kg-C", "W002", NOW - timedelta(seconds=2)), now=NOW)
        self.assertTrue(result["out_of_order"])
        self.assertEqual(result["processing_status"], "OUT_OF_ORDER")
        self.assertEqual(self.processor.bridge_snapshot("BOX")["latest_window_id"], "W003")

    def test_timezone_free_timestamp_is_error(self):
        payload = fixture("BOX", "1kg-C", "W001", NOW)
        payload["timestamp"] = "2026-10-15T12:30:00"
        result = self.processor.process(payload, now=NOW)
        self.assertEqual(result["evidence"]["error"]["code"], "TIMESTAMP_INVALID")

    def test_json_serialization(self):
        result = self.processor.process(fixture("TRUSS", "1kg-E", "W001", NOW), now=NOW)
        json.dumps(result, allow_nan=False)

    def test_score_and_level_stay_null(self):
        result = self.processor.process(fixture("TRUSS", "1kg-E", "W001", NOW), now=NOW)
        self.assertIsNone(result["risk_score"])
        self.assertIsNone(result["risk_level"])


if __name__ == "__main__":
    unittest.main()
