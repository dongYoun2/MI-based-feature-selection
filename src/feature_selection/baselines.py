"""Baseline feature selection methods.

Implements:
    - Correlation filtering
    - Marginal mutual information (top-k MI(y; x_j))
    - L1 regularization (Lasso / LogisticRegression-L1)
    - Recursive feature elimination (RFE)
    - SHAP-based feature importance
"""

from __future__ import annotations

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import Lasso, LogisticRegression


def correlation_filter(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
) -> list[int]:
    """Select top-k features with the highest |Pearson correlation| with y."""
    scores = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])])
    scores = np.nan_to_num(scores, nan=0.0)
    return list(np.argsort(scores)[::-1][:k])


def mi_filter(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
    random_state: int = 0,
) -> list[int]:
    """Select top-k features by marginal mutual information MI(y; x_j).

    Pure MI ranking with no redundancy term — a natural baseline against
    methods like mRMR, CMI, and PID.
    """
    mi_fn = mutual_info_classif if task == "classification" else mutual_info_regression
    scores = mi_fn(X, y, random_state=random_state)
    return list(np.argsort(scores)[::-1][:k])


def l1_selection(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
    C: float = 1.0,
    alpha: float = 1e-3,
) -> list[int]:
    """Select top-k features by |coefficient| magnitude of an L1 model."""
    if task == "classification":
        model = LogisticRegression(l1_ratio=1, solver="saga", C=C, max_iter=10000, tol=1e-3)
        model.fit(X, y)
        coef = np.abs(model.coef_).max(axis=0)
    else:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X, y)
        coef = np.abs(model.coef_)
    return list(np.argsort(coef)[::-1][:k])


def rfe_selection(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
) -> list[int]:
    """Recursive feature elimination using a linear estimator.

    Fits with ``n_features_to_select=1`` to obtain the full elimination ranking
    via ``RFE.ranking_``, so a single fit yields a top-k-monotone ordering
    that can be sliced for any k.
    """
    estimator = (
        LogisticRegression(max_iter=1000)
        if task == "classification"
        else Lasso(alpha=1e-3, max_iter=10000)
    )
    selector = RFE(estimator, n_features_to_select=1)
    selector.fit(X, y)
    return list(np.argsort(selector.ranking_)[:k])


def shap_selection(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
    n_estimators: int = 200,
    random_state: int = 0,
) -> list[int]:
    """Top-k features by mean |SHAP value| from a random-forest TreeSHAP model."""
    cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    model = cls(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)

    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = np.abs(values).mean(axis=(0, -1))
    else:
        values = np.abs(values).mean(axis=0)
    return list(np.argsort(values)[::-1][:k])
