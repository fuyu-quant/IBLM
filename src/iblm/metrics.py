from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true: np.array, y_pred: np.array, objective: str) -> dict:
    if objective == "regression":
        metric_dict = {
            "mae": np.round(mean_absolute_error(y_true, y_pred), 6),
            "rmse": np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 6),
            "r2": np.round(r2_score(y_true, y_pred), 6),
        }
    elif objective == "binary":
        y_prob = y_pred
        y_pred = np.where(y_prob > 0.5, 1, 0)
        metric_dict = {
            "roc_auc": np.round(roc_auc_score(y_true, y_prob), 6),
            "pr_auc": np.round(average_precision_score(y_true, y_prob), 6),
            "accuracy": np.round(accuracy_score(y_true, y_pred), 6),
            "recall": np.round(recall_score(y_true, y_pred), 6),
            "precision": np.round(precision_score(y_true, y_pred), 6),
            "f1_score": np.round(f1_score(y_true, y_pred), 6),
        }
    elif objective == "multiclass":
        # TODO: add metrics with rounding
        metric_dict = {}

    return metric_dict
