"""In-memory, bridge-isolated processor for sequential AI① JSON messages.

This module has no HTTP/WebSocket server and never calculates RiskScore or a
risk level. It delegates one-time Window analysis to ``risk_evaluator``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from src.risk_evaluator import DEFAULT_MODEL_ROOT, evaluate_risk_evidence_safe


@dataclass
class WindowRecord:
    fingerprint: str
    message_id: str | None
    timestamp: datetime
    result: dict[str, Any]


@dataclass
class BridgeState:
    windows: dict[tuple[str | None, str], WindowRecord] = field(default_factory=dict)
    latest_timestamp: datetime | None = None
    latest_window_id: str | None = None
    latest_result: dict[str, Any] | None = None


class RealtimeRiskProcessor:
    """Process messages once, retain cached results, and protect latest state.

    ``max_age_seconds`` and ``max_future_seconds`` are transport-quality
    settings only. They are not RiskScore or risk-level thresholds.
    """

    def __init__(
        self,
        *,
        max_age_seconds: float = 300,
        max_future_seconds: float = 30,
        baseline_lookup: dict[str, Any] | None = None,
        model_root: str | Path = DEFAULT_MODEL_ROOT,
    ):
        if max_age_seconds < 0 or max_future_seconds < 0:
            raise ValueError("Time tolerances must be non-negative")
        self.max_age = timedelta(seconds=max_age_seconds)
        self.max_future = timedelta(seconds=max_future_seconds)
        self.baseline_lookup = baseline_lookup
        self.model_root = Path(model_root)
        self._states = {"BOX": BridgeState(), "TRUSS": BridgeState()}
        self._experiment_window_bridges: dict[tuple[str, str], str] = {}
        self._lock = RLock()

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if not isinstance(value, str):
            raise ValueError("timestamp must be an ISO 8601 string with timezone")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("timestamp must be valid ISO 8601") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp must include timezone offset")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _fingerprint(payload: dict[str, Any]) -> str:
        # message_id is transport metadata; a new ID does not make a new Window.
        canonical = {key: value for key, value in payload.items() if key != "message_id"}
        try:
            encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Message cannot be fingerprinted as JSON: {error}") from error
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _base_error(payload: Any, code: str, message: str, processing_status: str, received_at: datetime) -> dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        return {
            "schema_version": "ai2-risk-evidence-v1",
            "bridge_type": payload.get("bridge_type"), "window_id": payload.get("window_id"),
            "experiment_id": payload.get("experiment_id"), "event_time": payload.get("event_time"),
            "timestamp": payload.get("timestamp"), "message_id": payload.get("message_id"),
            "damage_class": None, "damage_probabilities": None,
            "risk_score": None, "risk_level": None, "risk_status": "EVALUATION_ERROR",
            "data_quality": {"statuses": ["MISSING_REQUIRED"], "primary_status": "MISSING_REQUIRED"},
            "evidence": {"error": {"code": code, "message": message}},
            "processing_status": processing_status, "duplicate": False, "out_of_order": False,
            "received_at": received_at.isoformat(),
        }

    @staticmethod
    def _add_quality(result: dict[str, Any], statuses: list[str], evaluated: list[str]) -> dict[str, Any]:
        result = copy.deepcopy(result)
        quality = result.setdefault("data_quality", {})
        old = quality.get("statuses", [])
        quality["statuses"] = list(dict.fromkeys(old + statuses))
        quality["primary_status"] = quality["statuses"][0] if quality["statuses"] else None
        not_evaluated = quality.get("not_evaluated", [])
        quality["not_evaluated"] = [status for status in not_evaluated if status not in evaluated]
        return result

    def _duplicate_result(self, record: WindowRecord, payload: dict[str, Any], received_at: datetime) -> dict[str, Any]:
        result = self._add_quality(record.result, ["DUPLICATE_WINDOW"], ["DUPLICATE_WINDOW"])
        result["data_quality"]["primary_status"] = "DUPLICATE_WINDOW"
        result.update({
            "message_id": payload.get("message_id"), "processing_status": "DUPLICATE",
            "duplicate": True, "out_of_order": False, "received_at": received_at.isoformat(),
        })
        return result

    def _conflict_result(self, payload: dict[str, Any], received_at: datetime) -> dict[str, Any]:
        result = self._base_error(
            payload, "WINDOW_CONFLICT", "Same bridge_type + window_id was received with different content.",
            "WINDOW_CONFLICT", received_at,
        )
        result["duplicate"] = True
        result["data_quality"] = {"statuses": ["DUPLICATE_WINDOW"], "primary_status": "DUPLICATE_WINDOW"}
        return result

    @staticmethod
    def _window_key(payload: dict[str, Any], window_id: str) -> tuple[str | None, str]:
        """Keep legacy canonical payloads working while isolating AI① experiments."""
        experiment_id = payload.get("experiment_id")
        return (experiment_id, window_id) if isinstance(experiment_id, str) and experiment_id else (None, window_id)

    def process(self, payload: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
        """Process exactly one message using an injectable timezone-aware clock."""
        now = datetime.now(timezone.utc) if now is None else now
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Injected now must include timezone")
        now = now.astimezone(timezone.utc)
        if not isinstance(payload, dict):
            return self._base_error(payload, "PAYLOAD_INVALID", "Payload must be an object.", "INPUT_ERROR", now)
        bridge_type, window_id = payload.get("bridge_type"), payload.get("window_id")
        if bridge_type not in self._states:
            return self._base_error(payload, "BRIDGE_TYPE_INVALID", "bridge_type must be BOX or TRUSS.", "INPUT_ERROR", now)
        if not isinstance(window_id, str) or not window_id:
            return self._base_error(payload, "WINDOW_ID_MISSING", "window_id is required for deduplication.", "INPUT_ERROR", now)
        try:
            fingerprint = self._fingerprint(payload)
        except ValueError as error:
            return self._base_error(payload, "FINGERPRINT_INVALID", str(error), "INPUT_ERROR", now)
        with self._lock:
            state = self._states[bridge_type]
            key = self._window_key(payload, window_id)
            experiment_id = key[0]
            if experiment_id is not None:
                prior_bridge = self._experiment_window_bridges.get((experiment_id, window_id))
                if prior_bridge is not None and prior_bridge != bridge_type:
                    return self._base_error(
                        payload,
                        "EXPERIMENT_WINDOW_BRIDGE_CONFLICT",
                        "Same experiment_id + window_id was received with a different bridge_type.",
                        "WINDOW_CONFLICT",
                        now,
                    )
            existing = state.windows.get(key)
            if existing:
                return self._duplicate_result(existing, payload, now) if existing.fingerprint == fingerprint else self._conflict_result(payload, now)
            try:
                timestamp = self._parse_timestamp(payload.get("timestamp"))
            except ValueError as error:
                return self._base_error(payload, "TIMESTAMP_INVALID", str(error), "INPUT_ERROR", now)
            age, future = now - timestamp, timestamp - now
            stale = age > self.max_age
            future_dated = future > self.max_future
            out_of_order = state.latest_timestamp is not None and timestamp < state.latest_timestamp
            result = evaluate_risk_evidence_safe(payload, baseline_lookup=self.baseline_lookup, model_root=self.model_root)
            statuses, evaluated = [], ["STALE_DATA", "DUPLICATE_WINDOW"]
            processing_status = "PROCESSED"
            if stale:
                statuses.append("STALE_DATA")
                processing_status = "STALE_DATA"
            if future_dated:
                statuses.append("FUTURE_TIMESTAMP")
                processing_status = "FUTURE_TIMESTAMP"
            if out_of_order:
                processing_status = "OUT_OF_ORDER" if processing_status == "PROCESSED" else processing_status
            result = self._add_quality(result, statuses, evaluated)
            if statuses:
                result["data_quality"]["primary_status"] = statuses[0]
            result.update({
                "message_id": payload.get("message_id"), "processing_status": processing_status,
                "duplicate": False, "out_of_order": out_of_order, "received_at": now.isoformat(),
            })
            # Store every validly timestamped unique Window for deduplication.
            state.windows[key] = WindowRecord(fingerprint, payload.get("message_id"), timestamp, copy.deepcopy(result))
            if experiment_id is not None:
                self._experiment_window_bridges[(experiment_id, window_id)] = bridge_type
            # Stale, future, and out-of-order messages never overwrite current bridge state.
            if not stale and not future_dated and not out_of_order:
                state.latest_timestamp = timestamp
                state.latest_window_id = window_id
                state.latest_result = copy.deepcopy(result)
            return result

    def bridge_snapshot(self, bridge_type: str) -> dict[str, Any]:
        """Read-only summary for an eventual API/status endpoint."""
        if bridge_type not in self._states:
            raise ValueError("bridge_type must be BOX or TRUSS")
        with self._lock:
            state = self._states[bridge_type]
            return {
                "bridge_type": bridge_type, "processed_window_count": len(state.windows),
                "latest_window_id": state.latest_window_id,
                "latest_timestamp": state.latest_timestamp.isoformat() if state.latest_timestamp else None,
                "latest_data_quality": copy.deepcopy(state.latest_result.get("data_quality")) if state.latest_result else None,
            }
