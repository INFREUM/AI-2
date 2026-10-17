"""Analyze RiskScore candidate inputs without calculating any RiskScore.

All vibration and AI① analyses use AI② Train rows only.  Model C probability
analysis uses only AI② Validation rows and loads the saved artifact unchanged.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.baseline_manager import FEATURES
from src.data_loader import load_validated_data
from src.feature_change import calculate_feature_changes, load_baseline_lookup


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "risk_design"
CLASS_ORDER = ["NORMAL", "D1", "D2", "D3"]
MODEL_C_FEATURES = FEATURES.copy()


def distribution(values: pd.Series) -> dict:
    q1, q3 = values.quantile([0.25, 0.75])
    return {
        "mean": float(values.mean()), "std": float(values.std()), "median": float(values.median()),
        "min": float(values.min()), "max": float(values.max()), "iqr": float(q3 - q1),
    }


def vibration_feature_rows(data: pd.DataFrame, bridge_type: str) -> list[dict]:
    rows = []
    for unit, source in (("window", data), ("experiment_mean", data.groupby(["experiment_id", "target"])[FEATURES].mean().reset_index())):
        normal = source.loc[source["target"] == "NORMAL"]
        for feature in FEATURES:
            normal_values = normal[feature]
            normal_mean = normal_values.mean()
            normal_std = normal_values.std()
            for damage_class, subset in source.groupby("target"):
                values = subset[feature]
                difference = values.mean() - normal_mean
                pooled = np.sqrt((values.std() ** 2 + normal_std ** 2) / 2)
                rows.append({
                    "bridge_type": bridge_type, "source_split": "train", "unit": unit,
                    "damage_class": damage_class, "feature": feature,
                    "experiment_count": int(subset["experiment_id"].nunique()), "window_count": int(len(subset)),
                    **distribution(values), "normal_mean": float(normal_mean),
                    "mean_difference_vs_normal": float(difference),
                    "standardized_mean_difference_vs_normal": float(difference / pooled) if pooled and np.isfinite(pooled) else np.nan,
                    "note": "Window rows are correlated within experiment; experiment_mean is the independent-unit summary.",
                })
    return rows


def payload_from_row(row: pd.Series, bridge_type: str) -> dict:
    position = None if pd.isna(row["load_position"]) else row["load_position"]
    return {
        "schema_version": "risk-design-analysis-v1", "bridge_type": bridge_type,
        "window_id": row["window_id"], "timestamp": str(row["window_start"]),
        "features": {feature: float(row[feature]) for feature in FEATURES},
        "environment": {"load_level_kg": float(row["load_level_kg"]), "load_position": position, "temperature_c": float(row["temperature_c"])},
        "ai1_result": {},
    }


def baseline_change_rows(data: pd.DataFrame, bridge_type: str, lookup: dict) -> list[dict]:
    long_rows = []
    for _, row in data.iterrows():
        change = calculate_feature_changes(payload_from_row(row, bridge_type), lookup)
        for feature, result in change["feature_changes"].items():
            long_rows.append({
                "experiment_id": row["experiment_id"], "damage_class": row["target"], "feature": feature,
                "absolute_difference": result["absolute_difference"],
                "absolute_relative_change_percent": result["absolute_relative_change_percent"],
                "baseline_status": change["baseline_status"],
                "baseline_experiment_count": change["baseline_experiment_count"],
                "baseline_window_count": change["baseline_window_count"],
            })
    changes = pd.DataFrame(long_rows)
    rows = []
    for unit, source in (("window", changes), ("experiment_mean", changes.groupby(["experiment_id", "damage_class", "feature", "baseline_status", "baseline_experiment_count", "baseline_window_count"], dropna=False)[["absolute_difference", "absolute_relative_change_percent"]].mean().reset_index())):
        for (damage_class, feature, status), subset in source.groupby(["damage_class", "feature", "baseline_status"], dropna=False):
            abs_diff = subset["absolute_difference"]
            abs_relative = subset["absolute_relative_change_percent"].dropna()
            rows.append({
                "bridge_type": bridge_type, "source_split": "train", "unit": unit,
                "damage_class": damage_class, "feature": feature, "baseline_status": status,
                "experiment_count": int(subset["experiment_id"].nunique()), "window_count": int(len(subset)),
                "mean_absolute_difference": float(abs_diff.mean()), "median_absolute_difference": float(abs_diff.median()),
                "mean_absolute_relative_change_percent": float(abs_relative.mean()) if not abs_relative.empty else np.nan,
                "median_absolute_relative_change_percent": float(abs_relative.median()) if not abs_relative.empty else np.nan,
                "normal_comparison_note": "NORMAL rows contributed to the same Train baseline; not an independent validation comparison." if damage_class == "NORMAL" else "Damage rows are compared against Train NORMAL baselines with matching load condition.",
            })
    return rows


def ai1_score_rows(data: pd.DataFrame, bridge_type: str) -> list[dict]:
    definitions = {
        "anomaly_score": ("PRESENT", "Isolation Forest bridge-level score; 0~1 and higher is more anomalous.", "AI① direct model output"),
        "decision_ratio": ("PRESENT", "max(anomaly_score/threshold, |z|/z_threshold); >1 indicates anomaly.", "AI① derived per-window decision statistic"),
        "is_anomaly": ("ABSENT", "No is_anomaly column in source CSV.", "Definition confirmation required"),
        "raw_prediction": ("PRESENT", "Per-window Normal/Anomaly decision.", "AI① decision output"),
        "prediction": ("PRESENT", "Final decision using 2 of the latest 3 windows as Anomaly.", "AI① persistence post-processing output"),
    }
    rows = []
    for column, (availability, definition, provenance) in definitions.items():
        if column not in data:
            rows.append({"bridge_type": bridge_type, "source_split": "train", "column": column, "availability": availability, "definition": definition, "provenance": provenance})
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            for damage_class, subset in data.groupby("target"):
                rows.append({
                    "bridge_type": bridge_type, "source_split": "train", "column": column,
                    "availability": availability, "damage_class": damage_class, "window_count": len(subset),
                    **distribution(subset[column]), "missing_count": int(subset[column].isna().sum()),
                    "definition": definition, "provenance": provenance,
                })
        else:
            for damage_class, subset in data.groupby("target"):
                for value, count in subset[column].fillna("<MISSING>").value_counts().items():
                    rows.append({
                        "bridge_type": bridge_type, "source_split": "train", "column": column,
                        "availability": availability, "damage_class": damage_class, "category": value,
                        "window_count": int(count), "missing_count": int(subset[column].isna().sum()),
                        "definition": definition, "provenance": provenance,
                    })
    return rows


def probability_rows(data: pd.DataFrame, bridge_type: str) -> list[dict]:
    artifact = joblib.load(ROOT / "models" / bridge_type.lower() / "model_c.joblib")
    if artifact["class_order"] != CLASS_ORDER or artifact["raw_features"] != MODEL_C_FEATURES:
        raise ValueError(f"Unexpected {bridge_type} Model C artifact schema")
    probabilities = artifact["pipeline"].predict_proba(data[MODEL_C_FEATURES])
    output = data[["experiment_id", "target"]].copy()
    output["predicted_class"] = np.asarray(CLASS_ORDER)[probabilities.argmax(axis=1)]
    output["max_probability"] = probabilities.max(axis=1)
    output["normal_probability"] = probabilities[:, 0]
    output["damage_probability_sum"] = probabilities[:, 1:].sum(axis=1)
    for index, label in enumerate(CLASS_ORDER):
        output[f"prob_{label}"] = probabilities[:, index]
    rows = []
    for actual_class, subset in output.groupby("target"):
        for probability_class in CLASS_ORDER:
            values = subset[f"prob_{probability_class}"]
            rows.append({
                "bridge_type": bridge_type, "source_split": "validation", "row_type": "actual_class_probability",
                "actual_class": actual_class, "probability_class": probability_class, "window_count": len(subset),
                **distribution(values), "note": "Model C probability; probability is classification confidence, not risk severity.",
            })
    for (actual_class, predicted_class), subset in output.groupby(["target", "predicted_class"]):
        rows.append({
            "bridge_type": bridge_type, "source_split": "validation", "row_type": "prediction_outcome",
            "actual_class": actual_class, "predicted_class": predicted_class,
            "is_correct": bool(actual_class == predicted_class), "window_count": len(subset),
            "experiment_count": subset["experiment_id"].nunique(),
            "mean_max_probability": float(subset["max_probability"].mean()),
            "median_max_probability": float(subset["max_probability"].median()),
            "mean_normal_probability": float(subset["normal_probability"].mean()),
            "mean_damage_probability_sum": float(subset["damage_probability_sum"].mean()),
            "note": "Model C probability; probability is classification confidence, not risk severity.",
        })
    return rows


def assessment_markdown() -> str:
    return """# RiskScore 입력 설계 검토\n\n이 문서는 RiskScore 계산식, 가중치, 위험도 임계값을 정의하지 않는다. D1/D2/D3는 손상 유형이며 위험도 순서가 아니다.\n\n## A. 정상 기준 대비 진동 변화량\n\n- 활용 가능한 입력: 22개 진동 Feature의 절대 차이와 상대 변화율, 기준 상태, 기준 실험/Window 수.\n- 장점: 측정값의 방향과 단위를 보존하며, 하중조건 일치 기준과 연결된다.\n- 한계: 현재 모든 조건이 `INSUFFICIENT_EXPERIMENTS`이며 더미 Train NORMAL 기준이다. NORMAL 비교는 동일 기준 구성 데이터이므로 독립 검증이 아니다.\n- 정규화: Feature 단위와 크기가 다르므로 향후 물리적 허용범위 또는 충분한 정상 반복실험 기반의 정규화 정책이 필요하다.\n- HW 주의사항: 센서 교정, 단위, 타임스탬프, 하중위치, 온도, 결측 처리, 조건 미존재 시 행동을 운영 규격으로 확정해야 한다.\n\n## B. XGBoost 구조 상태 예측 확률\n\n- 활용 가능한 입력: Model C의 `NORMAL/D1/D2/D3` 순서 확률과 최대 확률.\n- 장점: 손상 유형 가설의 불확실성을 보존한다.\n- 한계: 확률은 분류 신뢰도이며 위험 심각도가 아니다. 현재 데이터는 시뮬레이션 더미이며 Window 상관성과 분할 규모가 작다.\n- 정규화: 확률 자체는 0~1이지만, 실제 HW 적용 전 calibration 검증이 필요하다.\n- HW 주의사항: Model C만 사용하고 환경 Feature shortcut을 RiskScore 근거로 혼합하지 않는다. 분류 확률을 D1/D2/D3 위험 순위로 해석하지 않는다.\n\n## C. AI① 이상 탐지 결과\n\n- 활용 가능한 입력: `anomaly_score`, `decision_ratio`, `raw_prediction`, `prediction`, 센서별 score/state.\n- 장점: AI①은 동일 하중조건의 정상과 비교한 이상 정도를 제공한다. `decision_ratio`는 문서상 1 초과가 이상이며 교량 간 비교 목적이 명시되어 있다.\n- 한계: AI① 더미 데이터이며 AI① 학습 실험은 AI② 후보에서 제외됐다. `prediction`은 최근 Window 지속성 규칙의 결과라 독립 점수가 아니다.\n- 정규화: `anomaly_score`와 `decision_ratio`의 범위와 기준이 다르므로 공통 가중 합산 전에 측정 단위와 시간 집계 규칙을 확정해야 한다.\n- HW 주의사항: AI① JSON에 점수 정의, threshold, 모델/센서 버전, 기준 조건, 생성 시각을 함께 전달해야 한다.\n\n## 필수 선행 확인\n\n1. 조건별 독립 NORMAL 반복실험을 확보하고 baseline 상태를 `SUFFICIENT`으로 개선한다.\n2. 실제 HW에서 온도·하중·손상 유형이 독립적으로 변하는 검증 데이터를 확보한다.\n3. 실험 단위와 연속 Window 단위의 집계 규칙을 정한다.\n4. RiskScore의 사용 목적, 허용 오경보/미탐, 사람 검토 절차를 합의한 후에만 가중치와 단계 기준을 설계한다.\n"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lookup = load_baseline_lookup()
    vibration, changes, ai1_scores, probabilities = [], [], [], []
    for bridge_type in ("BOX", "TRUSS"):
        data = load_validated_data(bridge_type)
        train = data.loc[data["ai2_split"] == "train"].copy()
        validation = data.loc[data["ai2_split"] == "val"].copy()
        vibration.extend(vibration_feature_rows(train, bridge_type))
        changes.extend(baseline_change_rows(train, bridge_type, lookup))
        ai1_scores.extend(ai1_score_rows(train, bridge_type))
        probabilities.extend(probability_rows(validation, bridge_type))
    pd.DataFrame(vibration).to_csv(OUTPUT_DIR / "vibration_feature_analysis.csv", index=False)
    pd.DataFrame(changes).to_csv(OUTPUT_DIR / "baseline_change_analysis.csv", index=False)
    pd.DataFrame(ai1_scores).to_csv(OUTPUT_DIR / "ai1_anomaly_score_analysis.csv", index=False)
    pd.DataFrame(probabilities).to_csv(OUTPUT_DIR / "xgboost_probability_analysis.csv", index=False)
    (OUTPUT_DIR / "risk_input_assessment.md").write_text(assessment_markdown(), encoding="utf-8")
    print("Risk input analysis saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
