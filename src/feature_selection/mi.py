"""Mutual information-based feature selection methods.

- Implements mRMR (Peng et al., 2005)
- Implements CMIM (Fleuret, 2004)
- Implements a PID-motivated CMI selector (Wollstadt et al., 2023 framing; joint conditioning, not PID atoms)

Greedy order is fixed for a given data split, so ``k = k_max`` then ``[:k]``
matches calling the same function with ``k`` directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def _feature_mi(X: np.ndarray, y: np.ndarray, task: str, random_state: int) -> np.ndarray:
    if task == "classification":
        return mutual_info_classif(X, y, random_state=random_state)
    if task == "regression":
        return mutual_info_regression(X, y, random_state=random_state)
    raise ValueError("task must be 'classification' or 'regression'")


def _pair_feature_mi(x_i: np.ndarray, x_j: np.ndarray, random_state: int) -> float:
    return float(mutual_info_regression(x_i.reshape(-1, 1), x_j, random_state=random_state)[0])


def _mrmr(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str,
    *,
    redundancy: str,
    random_state: int,
) -> list[int]:
    if redundancy not in {"correlation", "mi"}:
        raise ValueError("redundancy must be 'correlation' or 'mi'")

    n_features = X.shape[1]
    k = min(k, n_features)
    if k <= 0:
        return []

    relevance = _feature_mi(X, y, task, random_state=random_state)
    selected = [int(np.argmax(relevance))]
    remaining = set(range(n_features)) - {selected[0]}
    redundancy_sum = np.zeros(n_features)

    while len(selected) < k and remaining:
        x_last = X[:, selected[-1]]
        for j in remaining:
            if redundancy == "correlation":
                c = np.corrcoef(x_last, X[:, j])[0, 1]
                redundancy_sum[j] += 0.0 if np.isnan(c) else abs(c)
            else:
                redundancy_sum[j] += _pair_feature_mi(X[:, j], x_last, random_state=random_state)

        best = max(remaining, key=lambda j: relevance[j] - redundancy_sum[j] / len(selected))
        selected.append(best)
        remaining.remove(best)

    return selected


def mrmr_heuristic(
    X: np.ndarray, y: np.ndarray, k: int, task: str = "classification",
    *, random_state: int = 0,
) -> list[int]:
    return _mrmr(X, y, k, task, redundancy="correlation", random_state=random_state)


def mrmr(
    X: np.ndarray, y: np.ndarray, k: int, task: str = "classification",
    *, random_state: int = 0,
) -> list[int]:
    return _mrmr(X, y, k, task, redundancy="mi", random_state=random_state)


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
    """Mutual information I(x; y) = H(x) + H(y) - H(x, y) on discrete samples."""
    xy = np.column_stack([x, y])
    return _entropy_discrete(x) + _entropy_discrete(y) - _entropy_discrete(xy)


def _cmi_discrete(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """Conditional MI I(x; y | Z) = H(x,Z) + H(y,Z) - H(Z) - H(x,y,Z)."""
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    xZ = np.column_stack([x, Z])
    yZ = np.column_stack([y, Z])
    xyZ = np.column_stack([x, y, Z])

    return _entropy_discrete(xZ) + _entropy_discrete(yZ) - _entropy_discrete(Z) - _entropy_discrete(xyZ)


def _prepare_discrete(
    X: np.ndarray, y: np.ndarray, task: str, n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize X column-wise and y (quantile bins for regression, label encode for classification)."""
    X_disc = np.column_stack([_discretize_1d(X[:, j], n_bins=n_bins) for j in range(X.shape[1])])

    if task == "classification":
        _, y_disc = np.unique(y, return_inverse=True)
    elif task == "regression":
        y_disc = _discretize_1d(y, n_bins=n_bins)
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    return X_disc, y_disc


def _marginal_mi_per_feature(X_disc: np.ndarray, y_disc: np.ndarray) -> np.ndarray:
    """I(X_j; y) for every column j of a pre-discretized X."""
    return np.array([_mi_discrete(X_disc[:, j], y_disc) for j in range(X_disc.shape[1])])


def pid_selection(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
    *,
    n_bins: int = 10,
    random_state: int = 0,  # noqa: ARG001 -- accepted for interface uniformity; method is deterministic
) -> list[int]:
    """
    PID-motivated greedy selection using CMI jointly conditioned on all selected features.

    Does NOT estimate PID atoms directly; uses the score:
        score(j) = I(X_j ; y | X_S)        where S = already-selected features
    """
    n_features = X.shape[1]
    k = min(k, n_features)
    if k <= 0:
        return []

    X_disc, y_disc = _prepare_discrete(X, y, task, n_bins=n_bins)

    relevance = _marginal_mi_per_feature(X_disc, y_disc)
    first = int(np.argmax(relevance))
    selected: list[int] = [first]
    remaining = set(range(n_features)) - {first}

    while len(selected) < k and remaining:
        S = X_disc[:, selected]
        best = max(remaining, key=lambda j: _cmi_discrete(X_disc[:, j], y_disc, S))
        selected.append(int(best))
        remaining.remove(best)

    return selected


def cmim(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
    *,
    n_bins: int = 10,
    random_state: int = 0,  # noqa: ARG001 -- accepted for interface uniformity; method is deterministic
) -> list[int]:
    """Conditional Mutual Information Maximization (Fleuret, 2004).

    First feature: argmax_j I(X_j; y). Later steps: argmax_j min_{s in S} I(X_j; y | X_s),
    where S is the set already chosen (redundancy w.r.t. each picked feature separately).

    Scores are updated incrementally when a new feature enters S (O(k*n) CMI calls total).
    """
    n_features = X.shape[1]
    k = min(k, n_features)
    if k <= 0:
        return []

    X_disc, y_disc = _prepare_discrete(X, y, task, n_bins=n_bins)

    relevance = _marginal_mi_per_feature(X_disc, y_disc)
    first = int(np.argmax(relevance))
    selected: list[int] = [first]
    remaining = set(range(n_features)) - {first}

    # Running min of I(X_j; y | X_s) over s in S; +inf until at least one s is available.
    score = np.full(n_features, np.inf)

    while len(selected) < k and remaining:
        x_last = X_disc[:, selected[-1]]
        for j in remaining:
            cmi_j = _cmi_discrete(X_disc[:, j], y_disc, x_last)
            if cmi_j < score[j]:
                score[j] = cmi_j
        best = max(remaining, key=lambda j: score[j])
        selected.append(int(best))
        remaining.remove(best)

    return selected
