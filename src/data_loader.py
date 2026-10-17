"""Read and validate immutable AI① outputs for AI② training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/results/dummy/for_AI2")
SPLIT_PATH = Path("data/processed/ai2_experiment_split.csv")
LABEL_PATH = DATA_DIR / "experiment_labels.csv"
LABEL_MAP = {"D0": "NORMAL", "D1": "D1", "D2": "D2", "D3": "D3"}


class DataValidationError(ValueError):
    """Raised when a source or split invariant is violated."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file is missing: {path}")
    return pd.read_csv(path)


def load_validated_data(bridge_type: str) -> pd.DataFrame:
    """Return one bridge's labelled Window data with its assigned AI② split.

    Source CSVs remain untouched.  This function reads Test assignments only to
    validate split integrity; callers must explicitly select Train/Validation.
    """
    bridge_type = bridge_type.upper()
    if bridge_type not in {"BOX", "TRUSS"}:
        raise ValueError("bridge_type must be BOX or TRUSS")

    raw = _read_csv(DATA_DIR / f"{bridge_type.lower()}_anomaly_results.csv")
    labels = _read_csv(LABEL_PATH)
    split = _read_csv(SPLIT_PATH)

    required_split = {
        "experiment_id", "bridge_type", "condition_code", "damage_type",
        "severity_level", "ai1_split", "ai2_split",
    }
    missing = required_split - set(split.columns)
    if missing:
        raise DataValidationError(f"Split file lacks columns: {sorted(missing)}")
    if split["experiment_id"].isna().any() or split["experiment_id"].duplicated().any():
        raise DataValidationError("Each experiment_id must occur exactly once in split data")
    if (split.groupby("experiment_id")["ai2_split"].nunique() != 1).any():
        raise DataValidationError("An experiment_id is assigned to multiple AI② splits")
    if not set(split["ai2_split"]).issubset({"train", "val", "test"}):
        raise DataValidationError("Unknown AI② split label")

    label_fields = ["experiment_id", "bridge_type", "damage_type", "severity_level", "ai1_split"]
    if labels["experiment_id"].duplicated().any():
        raise DataValidationError("Duplicate experiment_id in experiment_labels.csv")
    joined = split.merge(labels[label_fields], on="experiment_id", how="left", suffixes=("_split", "_label"))
    if joined["damage_type_label"].isna().any():
        raise DataValidationError("AI② split contains experiment_id missing from labels")
    for field in ("bridge_type", "damage_type", "severity_level", "ai1_split"):
        if not (joined[f"{field}_split"] == joined[f"{field}_label"]).all():
            raise DataValidationError(f"Split/label mismatch in {field}")
    if not set(joined["damage_type_split"]).issubset(LABEL_MAP):
        raise DataValidationError("Unsupported damage_type label")
    if not (joined["ai1_split_split"] == "test").all():
        raise DataValidationError("AI① train/validation experiment included in AI② candidate split")

    other_bridge = "TRUSS" if bridge_type == "BOX" else "BOX"
    other_raw = _read_csv(DATA_DIR / f"{other_bridge.lower()}_anomaly_results.csv")
    all_raw_ids = set(raw["experiment_id"]) | set(other_raw["experiment_id"])
    if not set(split["experiment_id"]).issubset(all_raw_ids):
        raise DataValidationError("AI② split contains experiment_id absent from raw source")
    raw = raw.loc[raw["bridge_type"] == bridge_type].copy()
    bridge_split = joined.loc[joined["bridge_type_split"] == bridge_type].copy()
    data = raw.merge(
        bridge_split[["experiment_id", "ai2_split", "damage_type_split", "condition_code"]],
        on="experiment_id", how="inner", validate="many_to_one",
    )
    if data.empty:
        raise DataValidationError(f"No AI② candidate Windows found for {bridge_type}")
    if not (data["bridge_type"] == bridge_type).all():
        raise DataValidationError("Raw bridge_type does not match requested bridge")
    data["target"] = data["damage_type_split"].map(LABEL_MAP)
    if data["target"].isna().any():
        raise DataValidationError("Ground-truth conversion failed")
    return data
