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

ModelOverrides = dict[str, float | str]


def model_metric_label(model_name: str, overrides: ModelOverrides) -> str:
    """Human-readable model name for metrics tables (SVM includes ``C`` / ``gamma`` when set)."""
    if model_name == "svm" and overrides:
        return f"svm(C={overrides['svm_C']},gamma={overrides['svm_gamma']})"
    return model_name


def build_model(
    name: ModelName,
    task: Task,
    random_state: int = 0,
    *,
    overrides: ModelOverrides | None = None,
) -> BaseEstimator:
    ov = overrides or {}
    if name == "logreg":
        return _build_logreg(task, random_state)
    if name == "random_forest":
        return _build_random_forest(task, random_state)
    if name == "gradient_boosting":
        return _build_gradient_boosting(task, random_state)
    if name == "svm":
        return _build_svm(task, random_state, ov)
    raise ValueError(f"Unknown model: {name}")


def _build_logreg(task: Task, random_state: int) -> BaseEstimator:
    return (
        LogisticRegression(max_iter=1000, random_state=random_state)
        if task == "classification"
        else LinearRegression()
    )


def _build_random_forest(task: Task, random_state: int) -> BaseEstimator:
    cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    return cls(n_estimators=200, random_state=random_state, n_jobs=-1)


def _build_gradient_boosting(task: Task, random_state: int) -> BaseEstimator:
    cls = GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor
    return cls(random_state=random_state)


def _build_svm(task: Task, random_state: int, overrides: ModelOverrides) -> BaseEstimator:
    kw: dict = {}
    c, g = overrides.get("svm_C"), overrides.get("svm_gamma")
    if c is not None:
        kw["C"] = c
    if g is not None:
        kw["gamma"] = g
    if task == "classification":
        estimator = SVC(probability=True, random_state=random_state, **kw)
    else:
        estimator = SVR(**kw)
    return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])
