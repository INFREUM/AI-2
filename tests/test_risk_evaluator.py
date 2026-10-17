"""Synthetic interface tests. Inputs are baseline-derived, not HW measurements."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import joblib

from src.baseline_manager import FEATURES
from src.feature_change import load_baseline_lookup
from src.risk_evaluator import CLASS_ORDER, evaluate_risk_evidence_safe


LOOKUP = load_baseline_lookup()


def fixture(bridge_type: str, condition: str, *, ai1=True) -> dict:
    entry = LOOKUP["baselines"][bridge_type][condition]
    load_text, position = condition.split("-", 1)
    payload = {
        "schema_version": "synthetic-risk-test-v1",
        "bridge_type": bridge_type,
        "window_id": f"SYNTHETIC_{bridge_type}_{condition}",
        "timestamp": "2026-10-15T12:30:00+09:00",
        "features": {feature: entry["features"][feature]["mean"] for feature in FEATURES},
        "environment": {"load_level_kg": float(load_text.removesuffix("kg")), "load_position": position, "temperature_c": 24.5},
    }
    if ai1:
        payload["ai1_result"] = {"anomaly_score": 0.42, "decision_ratio": 0.8, "raw_prediction": "Normal", "prediction": "Normal"}
    return payload


class RiskEvaluatorTests(unittest.TestCase):
    def assert_pending(self, result, bridge_type):
        self.assertEqual(result["bridge_type"], bridge_type)
        self.assertEqual(result["risk_score"], None)
        self.assertEqual(result["risk_level"], None)
        self.assertEqual(result["risk_status"], "PENDING_VALIDATION")
        self.assertEqual(list(result["damage_probabilities"]), CLASS_ORDER)
        self.assertEqual(result["data_quality"]["primary_status"], "INSUFFICIENT_BASELINE")

    def test_box_input(self):
        self.assert_pending(evaluate_risk_evidence_safe(fixture("BOX", "1kg-C"), baseline_lookup=LOOKUP), "BOX")

    def test_truss_input(self):
        self.assert_pending(evaluate_risk_evidence_safe(fixture("TRUSS", "2kg-E"), baseline_lookup=LOOKUP), "TRUSS")

    def test_baseline_shortage_is_preserved(self):
        result = evaluate_risk_evidence_safe(fixture("BOX", "0kg-NONE"), baseline_lookup=LOOKUP)
        self.assertEqual(result["evidence"]["baseline"]["baseline_status"], "INSUFFICIENT_EXPERIMENTS")

    def test_missing_feature_is_not_normal(self):
        payload = fixture("BOX", "1kg-C")
        payload["features"].pop("acc01_rms_x")
        self.assertEqual(evaluate_risk_evidence_safe(payload, baseline_lookup=LOOKUP)["risk_status"], "EVALUATION_ERROR")

    def test_missing_model_file(self):
        with tempfile.TemporaryDirectory() as folder:
            result = evaluate_risk_evidence_safe(fixture("BOX", "1kg-C"), baseline_lookup=LOOKUP, model_root=folder)
        self.assertEqual(result["evidence"]["error"]["code"], "MODEL_ARTIFACT_MISSING")

    def test_invalid_class_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "box"
            path.mkdir()
            joblib.dump({"class_order": ["D1", "NORMAL", "D2", "D3"]}, path / "model_c.joblib")
            result = evaluate_risk_evidence_safe(fixture("BOX", "1kg-C"), baseline_lookup=LOOKUP, model_root=folder)
        self.assertEqual(result["evidence"]["error"]["code"], "MODEL_CLASS_ORDER_INVALID")

    def test_missing_ai1_is_marked_optional(self):
        result = evaluate_risk_evidence_safe(fixture("BOX", "1kg-C", ai1=False), baseline_lookup=LOOKUP)
        self.assertEqual(result["evidence"]["ai1_result"]["status"], "MISSING_OPTIONAL")

    def test_json_serialization(self):
        json.dumps(evaluate_risk_evidence_safe(fixture("TRUSS", "1kg-E"), baseline_lookup=LOOKUP), allow_nan=False)

    def test_score_and_level_are_null(self):
        result = evaluate_risk_evidence_safe(fixture("TRUSS", "1kg-E"), baseline_lookup=LOOKUP)
        self.assertIsNone(result["risk_score"])
        self.assertIsNone(result["risk_level"])


if __name__ == "__main__":
    unittest.main()
