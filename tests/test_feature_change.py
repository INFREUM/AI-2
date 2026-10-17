"""Synthetic-payload tests; values are copied from baseline means, not measurements."""

from __future__ import annotations

import copy
import json
import math
import unittest

from src.baseline_manager import FEATURES
from src.feature_change import calculate_feature_changes, load_baseline_lookup, safe_calculate_feature_changes


LOOKUP = load_baseline_lookup()


def fixture(bridge_type: str, condition: str) -> dict:
    """Create explicitly synthetic input values from a stored baseline."""
    entry = LOOKUP["baselines"][bridge_type][condition]
    load_text, position = condition.split("-", 1)
    return {
        "schema_version": "synthetic-test-v1",
        "bridge_type": bridge_type,
        "window_id": f"SYNTHETIC_{bridge_type}_{condition}",
        "timestamp": "2026-10-15T12:30:00+09:00",
        "features": {feature: entry["features"][feature]["mean"] for feature in FEATURES},
        "environment": {"load_level_kg": float(load_text.removesuffix("kg")), "load_position": position, "temperature_c": 0.0},
        "ai1_result": {"synthetic": True},
    }


class FeatureChangeTests(unittest.TestCase):
    def test_box_normal_input(self):
        result = calculate_feature_changes(fixture("BOX", "1kg-C"), LOOKUP)
        self.assertEqual(result["bridge_type"], "BOX")
        self.assertEqual(result["baseline_status"], "INSUFFICIENT_EXPERIMENTS")
        self.assertEqual(result["feature_changes"]["acc01_rms_x"]["relative_change_percent"], 0.0)

    def test_truss_normal_input(self):
        result = calculate_feature_changes(fixture("TRUSS", "2kg-E"), LOOKUP)
        self.assertEqual(result["load_condition"], "2kg-E")

    def test_zero_load_empty_position_normalizes_to_none(self):
        payload = fixture("BOX", "0kg-NONE")
        payload["environment"]["load_position"] = ""
        self.assertEqual(calculate_feature_changes(payload, LOOKUP)["load_condition"], "0kg-NONE")

    def test_missing_feature(self):
        payload = fixture("BOX", "1kg-C")
        payload["features"].pop("acc01_rms_x")
        self.assertEqual(safe_calculate_feature_changes(payload, LOOKUP)["error"]["code"], "FEATURE_MISSING")

    def test_non_numeric_feature(self):
        payload = fixture("BOX", "1kg-C")
        payload["features"]["acc01_rms_x"] = "not-a-number"
        self.assertEqual(safe_calculate_feature_changes(payload, LOOKUP)["error"]["code"], "FEATURE_NON_NUMERIC")

    def test_non_finite_feature(self):
        payload = fixture("BOX", "1kg-C")
        payload["features"]["acc01_rms_x"] = math.nan
        self.assertEqual(safe_calculate_feature_changes(payload, LOOKUP)["error"]["code"], "FEATURE_NON_FINITE")

    def test_missing_baseline_condition_has_no_fallback(self):
        payload = fixture("BOX", "1kg-C")
        payload["environment"]["load_level_kg"] = 3.0
        self.assertEqual(safe_calculate_feature_changes(payload, LOOKUP)["error"]["code"], "BASELINE_NOT_FOUND")

    def test_zero_baseline_mean_returns_null_relative_change(self):
        synthetic_lookup = copy.deepcopy(LOOKUP)
        synthetic_lookup["baselines"]["BOX"]["1kg-C"]["features"]["acc01_rms_x"]["mean"] = 0.0
        payload = fixture("BOX", "1kg-C")
        payload["features"]["acc01_rms_x"] = 1.0
        change = calculate_feature_changes(payload, synthetic_lookup)["feature_changes"]["acc01_rms_x"]
        self.assertIsNone(change["relative_change_percent"])
        self.assertIsNone(change["absolute_relative_change_percent"])

    def test_output_is_json_serializable(self):
        json.dumps(calculate_feature_changes(fixture("TRUSS", "1kg-E"), LOOKUP), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
