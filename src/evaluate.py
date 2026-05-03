"""Evaluation metrics for feature selection experiments.

The metric set depends on the task:
    - classification -> AUC, average precision
    - regression     -> RMSE, R^2

``evaluate_model`` dispatches to the task-specific helper and returns a dict
containing only the metrics relevant to the task. This keeps downstream
dataframes free of all-NaN columns.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score

CLASSIFICATION_METRICS = ("auc", "avg_precision")
REGRESSION_METRICS = ("rmse", "r2")


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
) -> dict[str, float | None]:
    if task == "classification":
        return _evaluate_classification(model, X_test, y_test)
    if task == "regression":
        return _evaluate_regression(model, X_test, y_test)
    raise ValueError(f"Unknown task: {task!r}")


def _evaluate_classification(
    model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray,
) -> dict[str, float | None]:
    auc: float | None = None
    avg_precision: float | None = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        scores = proba[:, 1] if proba.shape[1] == 2 else proba
        try:
            auc = float(
                roc_auc_score(
                    y_test,
                    scores,
                    multi_class="ovr" if proba.shape[1] > 2 else "raise",
                )
            )
        except ValueError:
            auc = None
        try:
            avg_precision = float(average_precision_score(y_test, scores))
        except ValueError:
            avg_precision = None

    return {"auc": auc, "avg_precision": avg_precision}


def _evaluate_regression(
    model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray,
) -> dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
