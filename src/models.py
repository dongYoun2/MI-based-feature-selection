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

# Defaults only for regression (``SVR``). Classification ``SVC`` omits ``C``/``gamma`` when unset,
# matching sklearn; explicit overrides from ``svm_param_search`` / grid expansion apply to both.
SVR_DEFAULT_C = 0.1
SVR_DEFAULT_GAMMA = 0.01


def model_metric_label(model_name: str, overrides: ModelOverrides, *, task: Task) -> str:
    """Human-readable model name for metrics tables; SVM reflects effective ``C`` / ``gamma``."""
    if model_name != "svm":
        return model_name
    ov = overrides or {}
    c, g = ov.get("svm_C"), ov.get("svm_gamma")
    if task == "classification":
        if c is None and g is None:
            return "svm"
        parts = []
        if c is not None:
            parts.append(f"C={c}")
        if g is not None:
            parts.append(f"gamma={g}")
        return "svm(" + ",".join(parts) + ")"
    ceff = SVR_DEFAULT_C if c is None else c
    geff = SVR_DEFAULT_GAMMA if g is None else g
    return f"svm(C={ceff},gamma={geff})"


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
    if task == "classification":
        if c is not None:
            kw["C"] = c
        if g is not None:
            kw["gamma"] = g
        estimator = SVC(probability=True, random_state=random_state, **kw)
    else:
        kw["C"] = SVR_DEFAULT_C if c is None else c
        kw["gamma"] = SVR_DEFAULT_GAMMA if g is None else g
        estimator = SVR(**kw)
    return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])
