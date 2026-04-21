"""Mutual information-based feature selection methods.

Implements:
    - mRMR (Peng et al., 2005)
    - CMI-based dynamic feature selection (Covert & Lee, 2024)
    - PID-based redundancy/relevance (Wollstadt et al., 2023)
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def _mi_matrix(X: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    """Marginal MI between each feature and y."""
    mi_fn = mutual_info_classif if task == "classification" else mutual_info_regression
    return mi_fn(X, y, random_state=0)


def mrmr(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
) -> list[int]:
    """Minimum Redundancy Maximum Relevance feature selection.

    Returns the indices of the top-k selected features in selection order.
    """
    n_features = X.shape[1]
    k = min(k, n_features)

    relevance = _mi_matrix(X, y, task)
    selected: list[int] = []
    remaining = set(range(n_features))

    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.discard(first)

    redundancy_sum = np.zeros(n_features)
    while len(selected) < k and remaining:
        last = selected[-1]
        for j in remaining:
            redundancy_sum[j] += abs(np.corrcoef(X[:, last], X[:, j])[0, 1])
        scores = {
            j: relevance[j] - redundancy_sum[j] / len(selected)
            for j in remaining
        }
        best = max(scores, key=scores.get)
        selected.append(best)
        remaining.discard(best)

    return selected


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
