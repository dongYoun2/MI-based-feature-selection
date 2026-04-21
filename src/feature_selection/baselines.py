"""Baseline feature selection methods.

Implements:
    - Correlation filtering
    - L1 regularization (Lasso / LogisticRegression-L1)
    - Recursive feature elimination (RFE)
    - SHAP-based feature importance
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import RFE
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
        model = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000)
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
    """Recursive feature elimination using a linear estimator."""
    estimator = (
        LogisticRegression(max_iter=1000)
        if task == "classification"
        else Lasso(alpha=1e-3, max_iter=10000)
    )
    selector = RFE(estimator, n_features_to_select=k)
    selector.fit(X, y)
    return list(np.where(selector.support_)[0])


def shap_selection(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
) -> list[int]:
    """Top-k features by mean |SHAP value| from a tree model."""
    raise NotImplementedError("TODO: implement SHAP-based feature selection")
