"""End-to-end FastAPI ingress tests with synthetic, non-HW requests."""

from __future__ import annotations

import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import create_app
from src.feature_change import load_baseline_lookup
from src.realtime_processor import RealtimeRiskProcessor
from tests.test_api_models import request_payload


def client_for(processor: RealtimeRiskProcessor | None = None, *, limit: int = 1_048_576) -> TestClient:
    return TestClient(create_app(processor=processor, max_request_bytes=limit), raise_server_exceptions=False)


class ApiWindowTests(unittest.TestCase):
    def setUp(self):
        lookup = load_baseline_lookup()
        self.processor = RealtimeRiskProcessor(max_age_seconds=300, max_future_seconds=30, baseline_lookup=lookup)

    def test_box_normal_and_truss_anomaly_requests(self):
        with client_for(self.processor) as client:
            normal = client.post("/api/ai1/windows", json=request_payload("BOX"))
            anomaly_payload = request_payload("TRUSS", experiment_id="SIM_TRUSS_001")
            anomaly_payload["ai1"]["prediction"] = "Anomaly"
            anomaly = client.post("/api/ai1/windows", json=anomaly_payload)
        for response, bridge in ((normal, "BOX"), (anomaly, "TRUSS")):
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["bridge_type"], bridge)
            self.assertIsNone(body["risk_score"])
            self.assertIsNone(body["risk_level"])

    def test_warming_up_is_explicit_and_does_not_remove_xgboost_evidence(self):
        payload = request_payload()
        payload["ai1"]["warming_up"] = True
        payload["ai1"]["prediction"] = "Normal"
        with client_for(self.processor) as client:
            response = client.post("/api/ai1/windows", json=payload)
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["ai1_status"], "PENDING_WARMUP")
        self.assertIn("AI1_WARMING_UP", body["data_quality"]["statuses"])
        self.assertIsNotNone(body["damage_probabilities"])

    def test_validation_and_raw_size_cases(self):
        missing = request_payload()
        missing["ai1"]["features"].pop("acc01_rms_x")
        too_big = request_payload(experiment_id="SIM_SIZE")
        too_big["raw"]["padding"] = "x" * 2_000
        with client_for(self.processor) as client:
            self.assertEqual(client.post("/api/ai1/windows", json=missing).status_code, 422)
        with client_for(self.processor, limit=1_000) as client:
            self.assertEqual(client.post("/api/ai1/windows", json=too_big).status_code, 413)

    def test_100_kib_raw_request_is_accepted_under_default_limit(self):
        payload = request_payload(experiment_id="SIM_RAW_SIZE")
        payload["raw"]["padding"] = "x" * (100 * 1024)
        with client_for(self.processor) as client:
            response = client.post("/api/ai1/windows", json=payload)
        self.assertEqual(response.status_code, 200)

    def test_deduplication_conflict_and_different_experiment_window_ids(self):
        payload = request_payload(experiment_id="SIM_A", window_id="W001")
        another_experiment = request_payload(experiment_id="SIM_B", window_id="W001")
        conflict = copy.deepcopy(payload)
        conflict["ai1"]["prediction"] = "Anomaly"
        with client_for(self.processor) as client:
            self.assertFalse(client.post("/api/ai1/windows", json=payload).json()["duplicate"])
            self.assertTrue(client.post("/api/ai1/windows", json=payload).json()["duplicate"])
            self.assertEqual(client.post("/api/ai1/windows", json=another_experiment).status_code, 200)
            self.assertEqual(client.post("/api/ai1/windows", json=conflict).status_code, 409)

    def test_bridge_conflict_for_same_experiment_window_is_not_silent(self):
        box = request_payload(experiment_id="SHARED", window_id="W001")
        truss = request_payload("TRUSS", experiment_id="SHARED", window_id="W001")
        with client_for(self.processor) as client:
            self.assertEqual(client.post("/api/ai1/windows", json=box).status_code, 200)
            self.assertEqual(client.post("/api/ai1/windows", json=truss).status_code, 409)

    def test_stale_future_and_out_of_order_are_successful_analysis_results(self):
        stale = request_payload(experiment_id="SIM_TIME", window_id="W001")
        stale["window_end_us"] -= 600_000_000
        stale["window_start_us"] -= 600_000_000
        future = request_payload(experiment_id="SIM_TIME", window_id="W002")
        future["window_end_us"] += 600_000_000
        future["window_start_us"] += 600_000_000
        latest = request_payload(experiment_id="SIM_ORDER", window_id="W003")
        earlier = request_payload(experiment_id="SIM_ORDER", window_id="W002")
        earlier["window_end_us"] -= 1_000_000
        earlier["window_start_us"] -= 1_000_000
        with client_for(self.processor) as client:
            self.assertEqual(client.post("/api/ai1/windows", json=stale).json()["processing_status"], "STALE_DATA")
            self.assertEqual(client.post("/api/ai1/windows", json=future).json()["processing_status"], "FUTURE_TIMESTAMP")
            self.assertEqual(client.post("/api/ai1/windows", json=latest).json()["processing_status"], "PROCESSED")
            self.assertEqual(client.post("/api/ai1/windows", json=earlier).json()["processing_status"], "OUT_OF_ORDER")

    def test_model_missing_returns_sanitized_503(self):
        missing_root = Path(__file__).resolve().parent / "not_a_model_root"
        processor = RealtimeRiskProcessor(baseline_lookup=load_baseline_lookup(), model_root=missing_root)
        with client_for(processor) as client:
            response = client.post("/api/ai1/windows", json=request_payload())
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(str(missing_root), response.text)

    def test_unexpected_analysis_exception_is_500_without_traceback(self):
        class ExplodingProcessor:
            def process(self, _: dict):
                raise RuntimeError("synthetic internal failure")

        with client_for(ExplodingProcessor()) as client:
            response = client.post("/api/ai1/windows", json=request_payload())
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("synthetic internal failure", response.text)

    def test_concurrent_duplicate_requests_do_not_analyze_twice(self):
        payload = request_payload(experiment_id="SIM_CONCURRENT")
        with client_for(self.processor) as client:
            with ThreadPoolExecutor(max_workers=2) as executor:
                responses = list(executor.map(lambda _: client.post("/api/ai1/windows", json=payload), range(2)))
        self.assertEqual(sorted(response.status_code for response in responses), [200, 200])
        self.assertEqual(sum(response.json()["duplicate"] for response in responses), 1)


if __name__ == "__main__":
    unittest.main()
