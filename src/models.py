"""Predictive model builders used after feature selection."""

from __future__ import annotations

from typing import Literal

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .data import Task

ModelName = Literal["logreg", "random_forest", "gradient_boosting", "svm"]


def build_model(
    name: ModelName,
    task: Task,
    random_state: int = 0,
    *,
    svm_C: float | None = None,
    svm_gamma: float | str | None = None,
) -> BaseEstimator:
    if name == "logreg":
        return (
            LogisticRegression(max_iter=1000, random_state=random_state)
            if task == "classification"
            else LinearRegression()
        )
    if name == "random_forest":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(n_estimators=200, random_state=random_state, n_jobs=-1)
    if name == "gradient_boosting":
        cls = GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor
        return cls(random_state=random_state)
    if name == "svm":
        # SVMs are scale-sensitive; wrap with StandardScaler. probability=True
        # is required so SVC exposes predict_proba for AUC / avg precision.
        kw: dict = {}
        if svm_C is not None:
            kw["C"] = svm_C
        if svm_gamma is not None:
            kw["gamma"] = svm_gamma
        if task == "classification":
            estimator = SVC(probability=True, random_state=random_state, **kw)
        else:
            # SVR has no random_state on older sklearn; keep deterministic defaults.
            estimator = SVR(**kw)
        return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])
    raise ValueError(f"Unknown model: {name}")
