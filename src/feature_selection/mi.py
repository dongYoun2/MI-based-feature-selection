"""Mutual information-based feature selection methods.

Implements:
    - mRMR (Peng et al., 2005)
    - PID-based redundancy/relevance (Wollstadt et al., 2023)
    - CMI-based dynamic feature selection (Covert & Lee, 2024)
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def _feature_mi(X: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    if task == "classification":
        return mutual_info_classif(X, y, random_state=0)
    if task == "regression":
        return mutual_info_regression(X, y, random_state=0)
    raise ValueError("task must be 'classification' or 'regression'")


def _pair_feature_mi(x_i: np.ndarray, x_j: np.ndarray) -> float:
    return float(mutual_info_regression(x_i.reshape(-1, 1), x_j, random_state=0)[0])


def _mrmr(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str,
    *,
    redundancy: str,
) -> tuple[list[int], list[float]]:
    if redundancy not in {"correlation", "mi"}:
        raise ValueError("redundancy must be 'correlation' or 'mi'")

    n_features = X.shape[1]
    k = min(k, n_features)
    if k <= 0:
        return [], []

    relevance = _feature_mi(X, y, task)

    selected = [int(np.argmax(relevance))]
    selection_scores = [float(relevance[selected[0]])]

    remaining = set(range(n_features))
    remaining.remove(selected[0])

    redundancy_sum = np.zeros(n_features)

    while len(selected) < k and remaining:
        last = selected[-1]
        x_last = X[:, last]

        for j in remaining:
            if redundancy == "correlation":
                c = np.corrcoef(x_last, X[:, j])[0, 1]
                redundancy_sum[j] += 0.0 if np.isnan(c) else abs(c)
            else:
                redundancy_sum[j] += _pair_feature_mi(X[:, j], x_last)

        best = max(remaining, key=lambda j: relevance[j] - redundancy_sum[j] / len(selected))
        best_score = float(relevance[best] - redundancy_sum[best] / len(selected))

        selected.append(best)
        selection_scores.append(float(best_score))
        remaining.remove(best)

    return selected, selection_scores


def mrmr_heuristic(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
) -> tuple[list[int], list[float]]:
    return _mrmr(X, y, k, task, redundancy="correlation")


def mrmr(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
) -> tuple[list[int], list[float]]:
    return _mrmr(X, y, k, task, redundancy="mi")


def _discretize_1d(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    x = np.asarray(x)

    if np.issubdtype(x.dtype, np.integer) and len(np.unique(x)) <= n_bins:
        _, encoded = np.unique(x, return_inverse=True)
        return encoded

    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    bins = np.unique(np.quantile(x, quantiles))

    return np.digitize(x, bins)


def _entropy_discrete(A: np.ndarray) -> float:
    A = np.asarray(A)

    if A.ndim == 1:
        A = A.reshape(-1, 1)

    _, counts = np.unique(A, axis=0, return_counts=True)

    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _mi_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """I(x; y) = H(x) + H(y) - H(x, y)"""
    xy = np.column_stack([x, y])

    return (
        _entropy_discrete(x)
        + _entropy_discrete(y)
        - _entropy_discrete(xy)
    )


def _cmi_discrete(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """I(x; y | Z) = H(x,Z) + H(y,Z) - H(Z) - H(x,y,Z)"""
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    xZ = np.column_stack([x, Z])
    yZ = np.column_stack([y, Z])
    xyZ = np.column_stack([x, y, Z])

    return (
        _entropy_discrete(xZ)
        + _entropy_discrete(yZ)
        - _entropy_discrete(Z)
        - _entropy_discrete(xyZ)
    )


def pid_selection(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
    *,
    n_bins: int = 10,
) -> tuple[list[int], list[float]]:
    """
    PID-motivated greedy selection using CMI.

    Does NOT estimate PID atoms directly; uses the score:
        score(j) = I(X_j ; y | selected_features)
    """
    n_samples, n_features = X.shape
    k = min(k, n_features)

    if k <= 0:
        return [], []

    X_disc = np.column_stack([
        _discretize_1d(X[:, j], n_bins=n_bins)
        for j in range(n_features)
    ])

    if task == "classification":
        _, y_disc = np.unique(y, return_inverse=True)
    elif task == "regression":
        y_disc = _discretize_1d(y, n_bins=n_bins)
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    selected: list[int] = []
    selection_scores: list[float] = []
    remaining = set(range(n_features))

    while len(selected) < k and remaining:
        best_feature = None
        best_score = -np.inf

        for j in remaining:
            x_j = X_disc[:, j]

            if not selected:
                score = _mi_discrete(x_j, y_disc)
            else:
                score = _cmi_discrete(x_j, y_disc, X_disc[:, selected])

            if score > best_score:
                best_score = score
                best_feature = j

        selected.append(int(best_feature))
        selection_scores.append(float(best_score))
        remaining.remove(best_feature)

    return selected, selection_scores


def dynamic_cmi_selection(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
) -> list[int]:
    raise NotImplementedError("TODO: implement CMI-based dynamic feature selection")

