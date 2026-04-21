"""Predictive model builders used after feature selection.

Models (per proposal): logistic regression, random forest, gradient boosting.
"""

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

ModelName = Literal["logreg", "random_forest", "gradient_boosting"]


def build_model(
    name: ModelName,
    task: Literal["classification", "regression"] = "classification",
    random_state: int = 0,
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
    raise ValueError(f"Unknown model: {name}")
