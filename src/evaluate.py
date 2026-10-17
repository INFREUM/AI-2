"""Validation-only metrics for window- and experiment-level classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate_probabilities(y_true, probabilities, experiment_ids, class_order):
    """Return metrics at Window level and probability-averaged experiment level."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predicted = np.asarray(class_order)[probabilities.argmax(axis=1)]
    result = {
        "window": _metrics(y_true, predicted, class_order),
        "experiment": None,
    }
    grouped = pd.DataFrame(probabilities, columns=class_order)
    grouped["experiment_id"] = np.asarray(experiment_ids)
    averaged = grouped.groupby("experiment_id", sort=True)[class_order].mean()
    truth = pd.DataFrame({"experiment_id": experiment_ids, "target": y_true}).groupby("experiment_id")["target"].nunique()
    if not (truth == 1).all():
        raise ValueError("An experiment has inconsistent ground-truth labels")
    experiment_true = pd.DataFrame({"experiment_id": experiment_ids, "target": y_true}).drop_duplicates("experiment_id").set_index("experiment_id").loc[averaged.index, "target"]
    experiment_predicted = np.asarray(class_order)[averaged.to_numpy().argmax(axis=1)]
    result["experiment"] = _metrics(experiment_true.to_numpy(), experiment_predicted, class_order)
    return result


def _metrics(y_true, y_pred, class_order):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_order, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_order).tolist(),
        "class_metrics": [
            {"class": label, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(class_order, precision, recall, f1, support)
        ],
    }
