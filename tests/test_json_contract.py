"""Contract checks against the current in-process AI② implementation."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.baseline_manager import FEATURES
from src.realtime_processor import RealtimeRiskProcessor
from src.risk_evaluator import CLASS_ORDER


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
NOW = datetime(2026, 10, 15, 3, 30, tzinfo=timezone.utc)


def example(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


class JsonContractTests(unittest.TestCase):
    def setUp(self):
        self.processor = RealtimeRiskProcessor(max_age_seconds=3600, max_future_seconds=3600)

    def test_box_input_contract(self):
        payload = example("ai1_to_ai2_box.json")
        self.assertEqual(payload["bridge_type"], "BOX")
        self.assertEqual(len(payload["features"]), 22)

    def test_truss_input_contract(self):
        payload = example("ai1_to_ai2_truss.json")
        self.assertEqual(payload["bridge_type"], "TRUSS")
        self.assertEqual(len(payload["features"]), 22)

    def test_feature_names_and_model_c_order(self):
        for bridge in ("box", "truss"):
            artifact = joblib.load(ROOT / "models" / bridge / "model_c.joblib")
            self.assertEqual(artifact["raw_features"], FEATURES)
            self.assertEqual(artifact["class_order"], CLASS_ORDER)

    def test_missing_required_field(self):
        payload = example("ai1_to_ai2_box.json")
        payload.pop("features")
        result = self.processor.process(payload, now=NOW)
        self.assertEqual(result["risk_status"], "EVALUATION_ERROR")
        self.assertEqual(result["evidence"]["error"]["code"], "FIELD_MISSING")

    def test_invalid_bridge_type(self):
        payload = example("ai1_to_ai2_box.json")
        payload["bridge_type"] = "ARCH"
        self.assertEqual(self.processor.process(payload, now=NOW)["evidence"]["error"]["code"], "BRIDGE_TYPE_INVALID")

    def test_invalid_timestamp(self):
        payload = example("ai1_to_ai2_box.json")
        payload["timestamp"] = "2026-10-15T12:30:00"
        self.assertEqual(self.processor.process(payload, now=NOW)["evidence"]["error"]["code"], "TIMESTAMP_INVALID")

    def test_output_contract_and_null_risk(self):
        result = self.processor.process(example("ai1_to_ai2_box.json"), now=NOW)
        json.dumps(result, allow_nan=False)
        for field in ("schema_version", "bridge_type", "window_id", "timestamp", "damage_class", "damage_probabilities", "risk_score", "risk_level", "risk_status", "data_quality", "evidence", "processing_status", "duplicate", "out_of_order", "received_at", "message_id"):
            self.assertIn(field, result)
        self.assertIsNone(result["risk_score"])
        self.assertIsNone(result["risk_level"])
        self.assertEqual(list(result["damage_probabilities"]), CLASS_ORDER)
        self.assertIn("statuses", result["data_quality"])

    def test_duplicate_window_policy(self):
        payload = example("ai1_to_ai2_truss.json")
        self.processor.process(payload, now=NOW)
        duplicate = self.processor.process(payload, now=NOW)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["processing_status"], "DUPLICATE")

    def test_ai1_optional_fields(self):
        payload = example("ai1_to_ai2_box.json")
        payload["window_id"] = "SYNTHETIC_BOX_OPTIONAL"
        payload.pop("ai1_result")
        result = self.processor.process(payload, now=NOW)
        self.assertEqual(result["evidence"]["ai1_result"]["status"], "MISSING_OPTIONAL")


if __name__ == "__main__":
    unittest.main()
