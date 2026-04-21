"""Evaluation metrics for feature selection experiments.

Reports (per proposal): accuracy, AUC, number of selected features, inference time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score


@dataclass
class EvalResult:
    task: str
    n_features: int
    accuracy: float | None
    auc: float | None
    r2: float | None
    rmse: float | None
    inference_time_s: float

    def to_dict(self) -> dict:
        return self.__dict__


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    selected_idx: list[int],
) -> EvalResult:
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start

    acc = auc = r2 = rmse = None
    if task == "classification":
        acc = float(accuracy_score(y_test, y_pred))
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            try:
                auc = float(
                    roc_auc_score(
                        y_test,
                        proba[:, 1] if proba.shape[1] == 2 else proba,
                        multi_class="ovr" if proba.shape[1] > 2 else "raise",
                    )
                )
            except ValueError:
                auc = None
    else:
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return EvalResult(
        task=task,
        n_features=len(selected_idx),
        accuracy=acc,
        auc=auc,
        r2=r2,
        rmse=rmse,
        inference_time_s=inference_time,
    )
