"""Train BOX/TRUSS AI② XGBoost candidates using Train and Validation only."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import load_validated_data
from src.evaluate import evaluate_probabilities
from src.preprocessing import make_preprocessor, prepare_features


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/ai2_feature_config.json"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"


def cumulative_features(models, model_name):
    order = ["A", "B", "C", "D", "E"]
    selected = []
    for name in order[: order.index(model_name) + 1]:
        selected.extend(models[name])
    return selected


def train_bridge(bridge_type, config):
    """Fit five fixed candidates and evaluate only Validation rows."""
    data = load_validated_data(bridge_type)
    # Test is loaded solely for integrity validation, then excluded before any fit.
    train_data = data.loc[data["ai2_split"] == "train"].copy()
    validation_data = data.loc[data["ai2_split"] == "val"].copy()
    if train_data.empty or validation_data.empty:
        raise ValueError(f"{bridge_type}: Train and Validation data are both required")
    if set(train_data["experiment_id"]) & set(validation_data["experiment_id"]):
        raise ValueError(f"{bridge_type}: Train/Validation experiment leakage detected")

    classes = config["class_order"]
    class_to_index = {label: index for index, label in enumerate(classes)}
    comparison, class_rows, matrices = [], [], {}
    for model_name in ("A", "B", "C", "D", "E"):
        features = cumulative_features(config["models"], model_name)
        x_train = prepare_features(train_data, features)
        x_validation = prepare_features(validation_data, features)
        y_train = train_data["target"].map(class_to_index)
        if y_train.isna().any():
            raise ValueError(f"{bridge_type} Model {model_name}: target encoding failed")
        preprocessor = make_preprocessor(features)
        classifier = XGBClassifier(
            **config["xgboost_parameters"],
            random_state=config["random_state"],
        )
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        probabilities = pipeline.predict_proba(x_validation)
        validation = evaluate_probabilities(
            validation_data["target"], probabilities, validation_data["experiment_id"], classes
        )
        model_dir = MODELS_DIR / bridge_type.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "pipeline": pipeline,
            "class_order": classes,
            "raw_features": features,
            "bridge_type": bridge_type,
            "model_name": model_name,
            "note": "Model E includes decision_ratio, an AI①-derived score requiring leakage review.",
        }
        joblib.dump(artifact, model_dir / f"model_{model_name.lower()}.joblib")
        comparison.append({
            "bridge_type": bridge_type, "model": model_name, "feature_count": len(features),
            "training_seconds": elapsed,
            "validation_window_accuracy": validation["window"]["accuracy"],
            "validation_window_macro_f1": validation["window"]["macro_f1"],
            "validation_experiment_accuracy": validation["experiment"]["accuracy"],
            "validation_experiment_macro_f1": validation["experiment"]["macro_f1"],
        })
        matrices[f"{bridge_type}_Model_{model_name}"] = {
            "class_order": classes,
            "window": validation["window"]["confusion_matrix"],
            "experiment_probability_mean": validation["experiment"]["confusion_matrix"],
        }
        for unit in ("window", "experiment"):
            for row in validation[unit]["class_metrics"]:
                class_rows.append({"bridge_type": bridge_type, "model": model_name, "unit": unit, **row})
    ranked = sorted(comparison, key=lambda row: (-row["validation_window_macro_f1"], row["feature_count"]))
    for row in comparison:
        row["selected_by_validation_macro_f1"] = row["model"] == ranked[0]["model"]
    return comparison, class_rows, matrices


def main():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    all_comparison, all_class_rows, all_matrices = [], [], {}
    for bridge_type in ("BOX", "TRUSS"):
        comparison, class_rows, matrices = train_bridge(bridge_type, config)
        all_comparison.extend(comparison)
        all_class_rows.extend(class_rows)
        all_matrices.update(matrices)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_comparison).to_csv(RESULTS_DIR / "validation_model_comparison.csv", index=False)
    pd.DataFrame(all_class_rows).to_csv(RESULTS_DIR / "validation_class_metrics.csv", index=False)
    with (RESULTS_DIR / "validation_confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump(all_matrices, handle, ensure_ascii=False, indent=2)
    print(pd.DataFrame(all_comparison).sort_values(["bridge_type", "model"]).to_string(index=False))


if __name__ == "__main__":
    main()
