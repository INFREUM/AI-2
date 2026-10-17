"""Validation-only reliability diagnostics for the frozen AI② Model D artifacts.

The script never scores Test rows.  Importance is associative, not causal, and
must not be interpreted as structural risk or a RiskScore.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import load_validated_data
from src.evaluate import evaluate_probabilities
from src.preprocessing import make_preprocessor, prepare_features


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "diagnostics"
CLASS_ORDER = ["NORMAL", "D1", "D2", "D3"]
SHORTCUT_FEATURES = ["load_level_kg", "load_position", "temperature_c"]
MODEL_D_FEATURES = [
    "acc01_rms_x", "acc01_rms_y", "acc01_rms_z", "acc02_rms_x", "acc02_rms_y", "acc02_rms_z",
    "acc01_p2p_x", "acc01_p2p_y", "acc01_p2p_z", "acc02_p2p_x", "acc02_p2p_y", "acc02_p2p_z",
    "acc01_dom_freq_x", "acc01_dom_freq_y", "acc01_dom_freq_z", "acc02_dom_freq_x", "acc02_dom_freq_y", "acc02_dom_freq_z",
    "acc01_damping_z", "acc02_damping_z", "cross_amp_ratio_L4_L2_z", "cross_torsion_y_z_L2",
    "load_level_kg", "load_position", "temperature_c",
]
XGB_PARAMETERS = {
    "objective": "multi:softprob", "num_class": 4, "n_estimators": 200,
    "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9,
    "colsample_bytree": 0.9, "min_child_weight": 1, "reg_lambda": 1.0,
    "tree_method": "hist", "eval_metric": "mlogloss", "n_jobs": 1,
}


def load_condition(frame: pd.DataFrame) -> pd.Series:
    position = frame["load_position"].fillna("NONE").replace("", "NONE")
    return frame["load_level_kg"].map(lambda value: f"{value:g}kg") + "-" + position


def feature_importance(artifact, validation_data: pd.DataFrame) -> pd.DataFrame:
    pipeline = artifact["pipeline"]
    raw_features = artifact["raw_features"]
    x_validation = prepare_features(validation_data, raw_features)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_names = list(preprocessor.get_feature_names_out())
    booster = classifier.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    encoded_target = validation_data["target"].map({label: index for index, label in enumerate(artifact["class_order"])})
    permutation = permutation_importance(
        pipeline, x_validation, encoded_target, scoring="f1_macro",
        n_repeats=10, random_state=20261014, n_jobs=1,
    )
    permutation_scores = dict(zip(raw_features, zip(permutation.importances_mean, permutation.importances_std)))
    rows = []
    for index, name in enumerate(transformed_names):
        raw_name = "load_position" if name.startswith("load_position_") else name
        perm_mean, perm_std = permutation_scores[raw_name]
        rows.append({
            "xgboost_input_feature": name,
            "raw_feature": raw_name,
            "xgboost_gain": float(gain.get(f"f{index}", 0.0)),
            "xgboost_weight": int(weight.get(f"f{index}", 0)),
            "permutation_macro_f1_mean_decrease": float(perm_mean),
            "permutation_macro_f1_std": float(perm_std),
            "importance_note": "Associative diagnostic only; not causal effect or structural RiskScore.",
        })
    return pd.DataFrame(rows).sort_values("xgboost_gain", ascending=False, ignore_index=True)


def metric_rows(bridge_type, scope, data, probabilities, include_confusion=False):
    evaluation = evaluate_probabilities(data["target"], probabilities, data["experiment_id"], CLASS_ORDER)
    rows = []
    for unit in ("window", "experiment"):
        metrics = evaluation[unit]
        row = {
            "bridge_type": bridge_type, "scope": scope, "unit": unit, "metric_type": "summary",
            "class": "ALL", "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "recall": np.nan, "support": int(len(data) if unit == "window" else data["experiment_id"].nunique()),
        }
        if include_confusion:
            row["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
        rows.append(row)
        for class_metric in metrics["class_metrics"]:
            rows.append({
                "bridge_type": bridge_type, "scope": scope, "unit": unit, "metric_type": "class_recall",
                "class": class_metric["class"], "accuracy": np.nan, "macro_f1": np.nan,
                "recall": class_metric["recall"], "support": class_metric["support"],
                **({"confusion_matrix": ""} if include_confusion else {}),
            })
    return rows


def load_condition_metrics(bridge_type, validation_data, probabilities):
    data = validation_data.copy()
    data["load_condition"] = load_condition(data)
    rows = []
    for condition in ["0kg-NONE", "1kg-C", "1kg-E", "2kg-C", "2kg-E"]:
        subset = data.loc[data["load_condition"] == condition]
        if subset.empty:
            continue
        indices = subset.index.to_numpy()
        # Validation index is reset below, so source positions are stored explicitly.
        probability_subset = probabilities[subset["validation_position"].to_numpy()]
        predicted = np.asarray(CLASS_ORDER)[probability_subset.argmax(axis=1)]
        precision, recall, f1, support = precision_recall_fscore_support(
            subset["target"], predicted, labels=CLASS_ORDER, zero_division=0
        )
        missing_classes = [label for label, count in zip(CLASS_ORDER, support) if count == 0]
        for label, value, count in zip(CLASS_ORDER, recall, support):
            rows.append({
                "bridge_type": bridge_type, "load_condition": condition,
                "window_count": len(subset), "experiment_count": subset["experiment_id"].nunique(),
                "accuracy": float(accuracy_score(subset["target"], predicted)),
                "macro_f1": float(f1_score(subset["target"], predicted, labels=CLASS_ORDER, average="macro", zero_division=0)),
                "class": label, "class_present": bool(count), "recall": float(value),
                "missing_classes": ";".join(missing_classes),
            })
    return rows


def cramers_v(frame: pd.DataFrame) -> float:
    table = pd.crosstab(frame["target"], frame["load_condition"])
    observed = table.to_numpy(dtype=float)
    expected = observed.sum(axis=1, keepdims=True) * observed.sum(axis=0, keepdims=True) / observed.sum()
    chi2 = ((observed - expected) ** 2 / np.where(expected == 0, 1, expected)).sum()
    denominator = observed.sum() * min(observed.shape[0] - 1, observed.shape[1] - 1)
    return float(np.sqrt(chi2 / denominator)) if denominator else 0.0


def crosstab_rows(bridge_type, data):
    output = []
    for split_name in ("train", "val"):
        subset = data.loc[data["ai2_split"] == split_name].copy()
        subset["load_condition"] = load_condition(subset)
        experiments = subset[["experiment_id", "target", "load_condition"]].drop_duplicates()
        association = cramers_v(experiments)
        grouped = experiments.groupby(["target", "load_condition"]).size()
        for (damage_class, condition), count in grouped.items():
            output.append({
                "bridge_type": bridge_type, "split": split_name, "damage_class": damage_class,
                "load_condition": condition, "experiment_count": int(count),
                "cramers_v_damage_vs_load": association,
            })
    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    condition_rows, experiment_rows, shortcut_rows, crosstab = [], [], [], []
    class_to_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    for bridge_type in ("BOX", "TRUSS"):
        data = load_validated_data(bridge_type)
        train = data.loc[data["ai2_split"] == "train"].copy()
        validation = data.loc[data["ai2_split"] == "val"].copy().reset_index(drop=True)
        validation["validation_position"] = np.arange(len(validation))
        artifact = joblib.load(ROOT / "models" / bridge_type.lower() / "model_d.joblib")
        if artifact["raw_features"] != MODEL_D_FEATURES:
            raise ValueError(f"{bridge_type} saved Model D feature set differs from diagnostic specification")
        probabilities = artifact["pipeline"].predict_proba(prepare_features(validation, MODEL_D_FEATURES))
        feature_importance(artifact, validation).to_csv(
            OUTPUT_DIR / f"feature_importance_{bridge_type.lower()}.csv", index=False
        )
        condition_rows.extend(load_condition_metrics(bridge_type, validation, probabilities))
        experiment_rows.extend(metric_rows(bridge_type, "model_d", validation, probabilities, include_confusion=True))

        shortcut = Pipeline([
            ("preprocessor", make_preprocessor(SHORTCUT_FEATURES)),
            ("classifier", XGBClassifier(**XGB_PARAMETERS, random_state=20261014)),
        ])
        shortcut.fit(prepare_features(train, SHORTCUT_FEATURES), train["target"].map(class_to_index))
        shortcut_probabilities = shortcut.predict_proba(prepare_features(validation, SHORTCUT_FEATURES))
        shortcut_rows.extend(metric_rows(bridge_type, "load_temperature_only", validation, shortcut_probabilities))
        crosstab.extend(crosstab_rows(bridge_type, data))

    pd.DataFrame(condition_rows).to_csv(OUTPUT_DIR / "load_condition_metrics.csv", index=False)
    pd.DataFrame(experiment_rows).to_csv(OUTPUT_DIR / "experiment_level_metrics.csv", index=False)
    pd.DataFrame(shortcut_rows).to_csv(OUTPUT_DIR / "shortcut_diagnostic.csv", index=False)
    pd.DataFrame(crosstab).to_csv(OUTPUT_DIR / "damage_load_crosstab.csv", index=False)
    print("Validation-only diagnostics saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
