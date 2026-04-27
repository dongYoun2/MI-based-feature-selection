"""Mutual information-based feature selection methods.

Implements:
    - mRMR (Peng et al., 2005)
    - CMI-based dynamic feature selection (Covert & Lee, 2024)
    - PID-based redundancy/relevance (Wollstadt et al., 2023)
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
) -> list[int]:
    if redundancy not in {"correlation", "mi"}:
        raise ValueError("redundancy must be 'correlation' or 'mi'")

    n_features = X.shape[1]
    k = min(k, n_features)
    if k <= 0:
        return []

    relevance = _feature_mi(X, y, task)

    selected = [int(np.argmax(relevance))]
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

        best = max(
            remaining,
            key=lambda j: relevance[j] - redundancy_sum[j] / len(selected),
        )

        selected.append(best)
        remaining.remove(best)

    return selected


def mrmr_heuristic(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
) -> list[int]:
    return _mrmr(X, y, k, task, redundancy="correlation")


def mrmr(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
) -> list[int]:
    return _mrmr(X, y, k, task, redundancy="mi")


def cmi_selection(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
) -> list[int]:
    """Greedy conditional mutual information selection.

    Placeholder using pairwise MI as a stand-in for CMI; the full method
    trains a value network to estimate CMI(y; x_j | x_S) (Covert & Lee, 2024).
    """
    raise NotImplementedError("TODO: implement CMI-based dynamic feature selection")


def pid_selection(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
) -> list[int]:
    """Partial information decomposition-based selection (Wollstadt et al., 2023)."""
    raise NotImplementedError("TODO: implement PID-based feature selection")
