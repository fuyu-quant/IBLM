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
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
        }
    elif objective == "binary":
        y_prob = y_pred
        y_pred = np.where(y_prob > 0.5, 1, 0)
        metric_dict = {
            "roc_auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob),
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }
    elif objective == "multiclass":
        # TODO: add metrics
        metric_dict = {}

    return metric_dict
