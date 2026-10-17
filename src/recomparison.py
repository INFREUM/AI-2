"""Validation-only C/D/F/G comparison without modifying saved C/D artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import load_validated_data
from src.evaluate import evaluate_probabilities
from src.preprocessing import make_preprocessor, prepare_features


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "diagnostics"
CLASS_ORDER = ["NORMAL", "D1", "D2", "D3"]
MODEL_C = [
    "acc01_rms_x", "acc01_rms_y", "acc01_rms_z", "acc02_rms_x", "acc02_rms_y", "acc02_rms_z",
    "acc01_p2p_x", "acc01_p2p_y", "acc01_p2p_z", "acc02_p2p_x", "acc02_p2p_y", "acc02_p2p_z",
    "acc01_dom_freq_x", "acc01_dom_freq_y", "acc01_dom_freq_z", "acc02_dom_freq_x", "acc02_dom_freq_y", "acc02_dom_freq_z",
    "acc01_damping_z", "acc02_damping_z", "cross_amp_ratio_L4_L2_z", "cross_torsion_y_z_L2",
]
FEATURES = {
    "C": MODEL_C,
    "F": MODEL_C + ["load_level_kg", "load_position"],
    "G": MODEL_C + ["temperature_c"],
    "D": MODEL_C + ["load_level_kg", "load_position", "temperature_c"],
    "TEMP_ONLY": ["temperature_c"],
}
XGB_PARAMETERS = {
    "objective": "multi:softprob", "num_class": 4, "n_estimators": 200,
    "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9,
    "colsample_bytree": 0.9, "min_child_weight": 1, "reg_lambda": 1.0,
    "tree_method": "hist", "eval_metric": "mlogloss", "n_jobs": 1,
}


def make_new_pipeline(features):
    return Pipeline([
        ("preprocessor", make_preprocessor(features)),
        ("classifier", XGBClassifier(**XGB_PARAMETERS, random_state=20261014)),
    ])


def result_rows(bridge_type, model, data, probabilities):
    result = evaluate_probabilities(data["target"], probabilities, data["experiment_id"], CLASS_ORDER)
    rows = []
    for unit in ("window", "experiment"):
        metrics = result[unit]
        rows.append({
            "bridge_type": bridge_type, "model": model, "unit": unit, "metric_type": "summary",
            "class": "ALL", "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "recall": np.nan, "support": len(data) if unit == "window" else data["experiment_id"].nunique(),
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        })
        for item in metrics["class_metrics"]:
            rows.append({
                "bridge_type": bridge_type, "model": model, "unit": unit, "metric_type": "class_recall",
                "class": item["class"], "accuracy": np.nan, "macro_f1": np.nan,
                "recall": item["recall"], "support": item["support"], "confusion_matrix": "",
            })
    return rows


def feature_group(raw_feature):
    if raw_feature == "temperature_c":
        return "temperature"
    if raw_feature in {"load_level_kg", "load_position"}:
        return "load"
    return "vibration"


def importance_rows(bridge_type, model, pipeline, features, validation):
    x_valid = prepare_features(validation, features)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_names = list(preprocessor.get_feature_names_out())
    booster = classifier.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    encoded_target = validation["target"].map({label: index for index, label in enumerate(CLASS_ORDER)})
    perm = permutation_importance(
        pipeline, x_valid, encoded_target, scoring="f1_macro", n_repeats=10,
        random_state=20261014, n_jobs=1,
    )
    perm_values = dict(zip(features, zip(perm.importances_mean, perm.importances_std)))
    total_gain = sum(gain.values())
    rows = []
    for index, transformed in enumerate(transformed_names):
        raw = "load_position" if transformed.startswith("load_position_") else transformed
        mean, std = perm_values[raw]
        raw_gain = float(gain.get(f"f{index}", 0.0))
        rows.append({
            "bridge_type": bridge_type, "model": model, "xgboost_input_feature": transformed,
            "raw_feature": raw, "feature_group": feature_group(raw), "xgboost_gain": raw_gain,
            "gain_share": raw_gain / total_gain if total_gain else 0.0,
            "xgboost_weight": int(weight.get(f"f{index}", 0)),
            "permutation_macro_f1_mean_decrease": float(mean),
            "permutation_macro_f1_std": float(std),
            "importance_note": "Associative diagnostic only; not causal effect or structural risk.",
        })
    return rows


def temperature_distribution(bridge_type, data):
    rows = []
    for split_name in ("train", "val"):
        grouped = data.loc[data["ai2_split"] == split_name].groupby("target")["temperature_c"]
        for damage_class, values in grouped:
            rows.append({
                "bridge_type": bridge_type, "split": split_name, "damage_class": damage_class,
                "window_count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
                "min": float(values.min()), "median": float(values.median()), "max": float(values.max()),
            })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison, temperature_rows, temperature_only_rows, importance = [], [], [], []
    class_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    for bridge_type in ("BOX", "TRUSS"):
        data = load_validated_data(bridge_type)
        train = data.loc[data["ai2_split"] == "train"].copy()
        valid = data.loc[data["ai2_split"] == "val"].copy()
        temperature_rows.extend(temperature_distribution(bridge_type, data))
        pipelines = {}
        for model in ("C", "D"):
            artifact = joblib.load(ROOT / "models" / bridge_type.lower() / f"model_{model.lower()}.joblib")
            if artifact["raw_features"] != FEATURES[model]:
                raise ValueError(f"Saved {bridge_type} Model {model} has unexpected features")
            pipelines[model] = artifact["pipeline"]
        for model in ("F", "G"):
            pipeline = make_new_pipeline(FEATURES[model])
            pipeline.fit(prepare_features(train, FEATURES[model]), train["target"].map(class_index))
            pipelines[model] = pipeline
        for model in ("C", "D", "F", "G"):
            probabilities = pipelines[model].predict_proba(prepare_features(valid, FEATURES[model]))
            comparison.extend(result_rows(bridge_type, model, valid, probabilities))
        for model in ("F", "G"):
            importance.extend(importance_rows(bridge_type, model, pipelines[model], FEATURES[model], valid))
        temp_pipeline = make_new_pipeline(FEATURES["TEMP_ONLY"])
        temp_pipeline.fit(prepare_features(train, FEATURES["TEMP_ONLY"]), train["target"].map(class_index))
        temp_probabilities = temp_pipeline.predict_proba(prepare_features(valid, FEATURES["TEMP_ONLY"]))
        temperature_only_rows.extend(result_rows(bridge_type, "TEMP_ONLY", valid, temp_probabilities))
    pd.DataFrame(comparison).to_csv(OUTPUT_DIR / "model_recomparison.csv", index=False)
    pd.DataFrame(temperature_rows).to_csv(OUTPUT_DIR / "temperature_damage_distribution.csv", index=False)
    pd.DataFrame(temperature_only_rows).to_csv(OUTPUT_DIR / "temperature_only_diagnostic.csv", index=False)
    pd.DataFrame(importance).to_csv(OUTPUT_DIR / "model_fg_feature_importance.csv", index=False)
    print("Validation-only model re-comparison saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
