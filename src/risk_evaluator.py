"""Evidence-vector interface for AI② risk evaluation; no RiskScore formula.

The module reads immutable Model C artifacts and immutable normal baselines.
It returns classification and change evidence while leaving RiskScore and its
four-level risk decision pending HW validation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.baseline_manager import FEATURES
from src.feature_change import FeatureChangeInputError, calculate_feature_changes, load_baseline_lookup


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = ROOT / "models"
CLASS_ORDER = ["NORMAL", "D1", "D2", "D3"]
OUTPUT_SCHEMA_VERSION = "ai2-risk-evidence-v1"


class RiskEvaluationError(RuntimeError):
    """Explicit non-normal evaluation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _base_fields(payload: Any) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "bridge_type": payload.get("bridge_type"),
        "window_id": payload.get("window_id"),
        "timestamp": payload.get("timestamp"),
        "experiment_id": payload.get("experiment_id"),
        "event_time": payload.get("event_time"),
    }


def _error_result(payload: Any, error: RiskEvaluationError | FeatureChangeInputError, status: str) -> dict[str, Any]:
    return {
        **_base_fields(payload),
        "damage_class": None,
        "damage_probabilities": None,
        "risk_score": None,
        "risk_level": None,
        "risk_status": "EVALUATION_ERROR",
        "data_quality": {"statuses": [status], "primary_status": status},
        "evidence": {"error": {"code": error.code, "message": error.message}},
    }


@lru_cache(maxsize=16)
def _read_model_artifact(path_text: str) -> dict[str, Any]:
    """Read an immutable model artifact once per process/path."""
    return joblib.load(path_text)


def _load_model(bridge_type: str, model_root: Path) -> dict[str, Any]:
    path = model_root / bridge_type.lower() / "model_c.joblib"
    if not path.is_file():
        raise RiskEvaluationError("MODEL_ARTIFACT_MISSING", f"Model C artifact is missing: {path}")
    try:
        artifact = _read_model_artifact(str(path.resolve()))
    except Exception as error:  # joblib errors are implementation-dependent.
        raise RiskEvaluationError("MODEL_ARTIFACT_INVALID", f"Model C artifact cannot be read: {error}") from error
    if artifact.get("class_order") != CLASS_ORDER:
        raise RiskEvaluationError("MODEL_CLASS_ORDER_INVALID", "Model C class order must be [NORMAL, D1, D2, D3]")
    if artifact.get("raw_features") != FEATURES:
        raise RiskEvaluationError("MODEL_FEATURE_MISMATCH", "Model C feature list differs from the required 22 vibration features")
    if artifact.get("bridge_type") != bridge_type or "pipeline" not in artifact:
        raise RiskEvaluationError("MODEL_ARTIFACT_INVALID", "Model C bridge_type or pipeline is invalid")
    return artifact


def preload_model_artifacts(model_root: str | Path = DEFAULT_MODEL_ROOT) -> None:
    """Fail FastAPI startup early when either immutable Model C artifact is unavailable."""
    root = Path(model_root)
    for bridge_type in ("BOX", "TRUSS"):
        _load_model(bridge_type, root)


def _model_evidence(payload: dict[str, Any], bridge_type: str, model_root: Path) -> dict[str, Any]:
    artifact = _load_model(bridge_type, model_root)
    features = payload["features"]
    frame = pd.DataFrame([{name: features[name] for name in FEATURES}])
    try:
        probabilities = artifact["pipeline"].predict_proba(frame)[0]
    except Exception as error:
        raise RiskEvaluationError("MODEL_INFERENCE_FAILED", f"Model C inference failed: {error}") from error
    if len(probabilities) != len(CLASS_ORDER) or not np.isfinite(probabilities).all():
        raise RiskEvaluationError("MODEL_INFERENCE_INVALID", "Model C returned invalid probabilities")
    if not np.isclose(float(np.sum(probabilities)), 1.0, atol=1e-6):
        raise RiskEvaluationError("MODEL_INFERENCE_INVALID", "Model C probabilities do not sum to one")
    mapped = {label: float(value) for label, value in zip(CLASS_ORDER, probabilities)}
    return {
        "class_order": CLASS_ORDER,
        "predicted_class": CLASS_ORDER[int(np.argmax(probabilities))],
        "probabilities": mapped,
        "max_probability": float(np.max(probabilities)),
        "model_name": artifact.get("model_name"),
        "model_bridge_type": bridge_type,
        "note": "Classification probability is not safety probability or risk severity.",
    }


def _ai1_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    ai1 = payload.get("ai1_result")
    if ai1 is None:
        return {"status": "MISSING_OPTIONAL", "raw": None, "note": "AI① result was not supplied; no AI① evidence is inferred."}
    if not isinstance(ai1, dict):
        return {"status": "INVALID_OPTIONAL", "raw": ai1, "note": "AI① result is not an object; no AI① evidence is interpreted."}
    return {
        "status": "PROVIDED_UNVALIDATED",
        "raw": ai1,
        "anomaly_score": ai1.get("anomaly_score"),
        "decision_ratio": ai1.get("decision_ratio"),
        "raw_prediction": ai1.get("raw_prediction"),
        "prediction": ai1.get("prediction"),
        "note": "Values are preserved only. They are not combined into RiskScore in this version.",
    }


def _baseline_quality(change: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    status = change.get("baseline_status")
    if status == "INSUFFICIENT_EXPERIMENTS":
        statuses = ["INSUFFICIENT_BASELINE"]
    elif status == "SUFFICIENT":
        statuses = ["VALID"]
    else:
        statuses = ["PROVISIONAL_BASELINE"]
    return statuses, {
        "baseline_status": status,
        "baseline_experiment_count": change.get("baseline_experiment_count"),
        "baseline_window_count": change.get("baseline_window_count"),
        "baseline_source": change.get("baseline_source"),
        "load_condition": change.get("load_condition"),
    }


def evaluate_risk_evidence(
    payload: dict[str, Any], *, baseline_lookup: dict[str, Any] | None = None, model_root: str | Path = DEFAULT_MODEL_ROOT
) -> dict[str, Any]:
    """Return evidence vector; never calculate a numeric RiskScore or level."""
    lookup = load_baseline_lookup() if baseline_lookup is None else baseline_lookup
    model_root = Path(model_root)
    try:
        changes = calculate_feature_changes(payload, lookup)
    except FeatureChangeInputError as error:
        status = "OUT_OF_CONDITION" if error.code == "BASELINE_NOT_FOUND" else "MISSING_REQUIRED"
        return _error_result(payload, error, status)
    bridge_type = changes["bridge_type"]
    try:
        model = _model_evidence(payload, bridge_type, model_root)
    except RiskEvaluationError as error:
        return _error_result(payload, error, "VERSION_MISMATCH" if error.code in {"MODEL_CLASS_ORDER_INVALID", "MODEL_FEATURE_MISMATCH"} else "MISSING_REQUIRED")
    statuses, baseline_summary = _baseline_quality(changes)
    ai1 = _ai1_evidence(payload)
    sensor_status = ai1.get("raw", {}).get("sensor_status") if isinstance(ai1.get("raw"), dict) else None
    return {
        **_base_fields(payload),
        "damage_class": model["predicted_class"],
        "damage_probabilities": model["probabilities"],
        "risk_score": None,
        "risk_level": None,
        "risk_status": "PENDING_VALIDATION",
        "data_quality": {
            "statuses": statuses,
            "primary_status": statuses[0],
            "not_evaluated": ["STALE_DATA", "DUPLICATE_WINDOW", "VERSION_MISMATCH"],
            "sensor_status": sensor_status,
            "sensor_status_note": "Sensor status is preserved but not interpreted until the AI① schema is finalized.",
        },
        "evidence": {
            "baseline": baseline_summary,
            "feature_changes": changes["feature_changes"],
            "ai1_result": ai1,
            "xgboost_model_c": model,
        },
    }


def evaluate_risk_evidence_safe(payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Defensive wrapper that also catches unexpected serialization-safe errors."""
    try:
        result = evaluate_risk_evidence(payload, **kwargs)
        json.dumps(result, allow_nan=False)
        return result
    except RiskEvaluationError as error:
        return _error_result(payload, error, "MISSING_REQUIRED")
    except (TypeError, ValueError) as error:
        wrapped = RiskEvaluationError("OUTPUT_SERIALIZATION_FAILED", f"Evaluation output is not JSON serializable: {error}")
        return _error_result(payload, wrapped, "MISSING_REQUIRED")
