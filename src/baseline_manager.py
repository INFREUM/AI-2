"""Build Train-only NORMAL vibration baselines for future RiskScore lookup.

This module intentionally produces descriptive reference statistics only.  It
does not compute a RiskScore, set risk thresholds, train XGBoost, or use Test
or Validation data in any baseline calculation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_validated_data


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "baselines"
MIN_EXPERIMENTS_PER_CONDITION = 3
NEAR_ZERO_STD = 1e-12
FEATURES = [
    "acc01_rms_x", "acc01_rms_y", "acc01_rms_z", "acc02_rms_x", "acc02_rms_y", "acc02_rms_z",
    "acc01_p2p_x", "acc01_p2p_y", "acc01_p2p_z", "acc02_p2p_x", "acc02_p2p_y", "acc02_p2p_z",
    "acc01_dom_freq_x", "acc01_dom_freq_y", "acc01_dom_freq_z", "acc02_dom_freq_x", "acc02_dom_freq_y", "acc02_dom_freq_z",
    "acc01_damping_z", "acc02_damping_z", "cross_amp_ratio_L4_L2_z", "cross_torsion_y_z_L2",
]
LOAD_ORDER = ["0kg-NONE", "1kg-C", "1kg-E", "2kg-C", "2kg-E"]


def load_condition(frame: pd.DataFrame) -> pd.Series:
    """Return the future JSON-compatible bridge/load lookup key."""
    position = frame["load_position"].fillna("NONE").replace("", "NONE")
    invalid = frame["load_level_kg"].ne(0) & position.eq("NONE")
    if invalid.any():
        raise ValueError("Nonzero load has no load_position; baseline cannot be trusted")
    return frame["load_level_kg"].map(lambda value: f"{value:g}kg") + "-" + position


def feature_statistics(frame: pd.DataFrame, bridge_type: str, scope: str, condition: str | None = None):
    experiments = int(frame["experiment_id"].nunique())
    windows = int(len(frame))
    experiment_means = frame.groupby("experiment_id")[FEATURES].mean()
    output = []
    for feature in FEATURES:
        values = frame[feature]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        output.append({
            "bridge_type": bridge_type, "scope": scope, "load_condition": condition or "ALL",
            "feature": feature, "mean": float(values.mean()), "std": float(values.std()),
            "median": float(values.median()), "min": float(values.min()), "max": float(values.max()),
            "iqr": float(iqr), "experiment_count": experiments, "window_count": windows,
            "inter_experiment_mean_std": float(experiment_means[feature].std()) if experiments > 1 else np.nan,
            "outlier_window_count": int(((values < lower) | (values > upper)).sum()),
            "near_zero_std": bool(values.std() <= NEAR_ZERO_STD),
        })
    return output


def coverage_row(frame: pd.DataFrame, bridge_type: str, condition: str):
    experiment_means = frame.groupby("experiment_id")[FEATURES].mean()
    feature_std = frame[FEATURES].std()
    outlier_windows = pd.Series(False, index=frame.index)
    for feature in FEATURES:
        q1, q3 = frame[feature].quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_windows |= (frame[feature] < q1 - 1.5 * iqr) | (frame[feature] > q3 + 1.5 * iqr)
    inter_std = experiment_means.std() if len(experiment_means) > 1 else pd.Series(dtype=float)
    return {
        "bridge_type": bridge_type, "load_condition": condition,
        "normal_experiment_count": int(frame["experiment_id"].nunique()),
        "normal_window_count": int(len(frame)),
        "minimum_experiment_requirement": MIN_EXPERIMENTS_PER_CONDITION,
        "coverage_status": "SUFFICIENT" if frame["experiment_id"].nunique() >= MIN_EXPERIMENTS_PER_CONDITION else "INSUFFICIENT_EXPERIMENTS",
        "outlier_window_count_any_feature": int(outlier_windows.sum()),
        "near_zero_std_feature_count": int((feature_std <= NEAR_ZERO_STD).sum()),
        "inter_experiment_mean_std_median": float(inter_std.median()) if not inter_std.empty else np.nan,
        "inter_experiment_mean_std_max": float(inter_std.max()) if not inter_std.empty else np.nan,
        "notes": "Window count does not replace independent experiment count.",
    }


def nested_lookup(by_load: pd.DataFrame, coverage: pd.DataFrame):
    lookup = {}
    for bridge_type in ("BOX", "TRUSS"):
        lookup[bridge_type] = {}
        for condition in by_load.loc[by_load["bridge_type"] == bridge_type, "load_condition"].unique():
            stats = by_load.loc[(by_load["bridge_type"] == bridge_type) & (by_load["load_condition"] == condition)]
            cover = coverage.loc[(coverage["bridge_type"] == bridge_type) & (coverage["load_condition"] == condition)].iloc[0].to_dict()
            lookup[bridge_type][condition] = {
                "coverage": cover,
                "features": {row.feature: {
                    key: row[key] for key in ("mean", "std", "median", "min", "max", "iqr", "inter_experiment_mean_std")
                } for _, row in stats.iterrows()},
            }
    return {
        "schema_note": "Lookup by incoming JSON bridge_type and normalized load condition only. No RiskScore or thresholds are included.",
        "baseline_source": "AI② train split, NORMAL only; AI① candidate test experiments only.",
        "minimum_experiment_requirement": MIN_EXPERIMENTS_PER_CONDITION,
        "baselines": lookup,
    }


def json_safe(value):
    """Convert pandas/NumPy missing values to standards-compliant JSON null."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows, by_load_rows, coverage_rows = [], [], []
    for bridge_type in ("BOX", "TRUSS"):
        data = load_validated_data(bridge_type)
        normal_train = data.loc[(data["ai2_split"] == "train") & (data["target"] == "NORMAL")].copy()
        if normal_train.empty:
            raise ValueError(f"{bridge_type} has no AI② Train NORMAL data")
        normal_train["load_condition"] = load_condition(normal_train)
        summary_rows.extend(feature_statistics(normal_train, bridge_type, "normal_train_all_conditions"))
        for condition in LOAD_ORDER:
            subset = normal_train.loc[normal_train["load_condition"] == condition]
            if subset.empty:
                continue
            by_load_rows.extend(feature_statistics(subset, bridge_type, "normal_train_by_load", condition))
            coverage_rows.append(coverage_row(subset, bridge_type, condition))
    summary = pd.DataFrame(summary_rows)
    by_load = pd.DataFrame(by_load_rows)
    coverage = pd.DataFrame(coverage_rows)
    summary.to_csv(OUTPUT_DIR / "normal_baseline_summary.csv", index=False)
    by_load.to_csv(OUTPUT_DIR / "normal_baseline_by_load.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "baseline_coverage_report.csv", index=False)
    with (OUTPUT_DIR / "normal_baseline_lookup.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(nested_lookup(by_load, coverage)), handle, ensure_ascii=False, indent=2, allow_nan=False)
    print("Train-only NORMAL baselines saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
