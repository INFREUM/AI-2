"""Translate the AI① HTTP envelope into the existing AI② canonical payload."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.api_models import Ai1WindowRequest
from src.baseline_manager import FEATURES


def unix_microseconds_to_utc_iso(value: int) -> str:
    """Convert an authoritative Unix-microsecond event time to UTC ISO 8601."""
    return datetime.fromtimestamp(value / 1_000_000, tz=timezone.utc).isoformat()


def to_ai2_payload(request: Ai1WindowRequest) -> dict[str, Any]:
    """Build the payload consumed by ``RealtimeRiskProcessor``.

    Raw samples have already been schema-checked, but are deliberately not
    copied into the analysis cache or XGBoost feature input.
    """
    condition = request.ai1.condition
    return {
        "schema_version": request.schema_version,
        "bridge_type": request.bridge_type,
        "experiment_id": request.experiment_id,
        "window_id": request.window_id,
        # The completed Window time is authoritative for freshness/order checks.
        "timestamp": unix_microseconds_to_utc_iso(request.window_end_us),
        "event_time": {
            "window_start": request.window_start,
            "window_start_us": request.window_start_us,
            "window_end_us": request.window_end_us,
        },
        "features": {feature: request.ai1.features[feature] for feature in FEATURES},
        "environment": {
            "load_level_kg": condition.load_level_kg,
            "load_position": condition.load_position,
            "temperature_c": condition.temperature_c,
        },
        "ai1_result": request.ai1.model_dump(mode="json"),
        "message_id": request.message_id,
    }
