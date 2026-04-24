"""Evaluation metrics for feature selection experiments.

Reports (per proposal): F1, AUC, number of selected features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score


@dataclass
class EvalResult:
    task: str
    n_features: int
    f1: float | None
    auc: float | None
    r2: float | None
    rmse: float | None

    def to_dict(self) -> dict:
        return self.__dict__


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    selected_idx: list[int],
) -> EvalResult:
    y_pred = model.predict(X_test)

    f1 = auc = r2 = rmse = None
    if task == "classification":
        average = "binary" if len(np.unique(y_test)) == 2 else "macro"
        f1 = float(f1_score(y_test, y_pred, average=average))
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
        f1=f1,
        auc=auc,
        r2=r2,
        rmse=rmse,
    )
