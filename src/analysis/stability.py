"""Selected-feature stability analysis utilities."""

from __future__ import annotations

import ast
from itertools import combinations
from math import isnan
from typing import Iterable

import pandas as pd


def parse_selected_features(x) -> set[str]:
    """Parse a selected_features CSV cell into a set of feature names."""
    if x is None:
        return set()
    if isinstance(x, float) and isnan(x):
        return set()
    if isinstance(x, set):
        return {str(feature) for feature in x}
    if isinstance(x, (list, tuple)):
        return {str(feature) for feature in x}
    if isinstance(x, str):
        if not x.strip():
            return set()
        parsed = ast.literal_eval(x)
        if isinstance(parsed, str):
            return {parsed}
        if isinstance(parsed, (list, tuple, set)):
            return {str(feature) for feature in parsed}
    raise ValueError(f"Cannot parse selected features from value: {x!r}")


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two feature sets."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def mean_pairwise_jaccard(feature_sets: Iterable[set[str]]) -> float:
    """Compute mean Jaccard similarity across all pairs of feature sets."""
    feature_sets = list(feature_sets)
    assert len(feature_sets) >= 2, "Need at least two feature sets"
    similarities = [
        jaccard_similarity(a, b)
        for a, b in combinations(feature_sets, 2)
    ]
    return sum(similarities) / len(similarities)


def compute_stability_table(
    df: pd.DataFrame,
    dataset: str = "nhanes",
    model: str = "logreg",
    k: int = 10,
) -> pd.DataFrame:
    """Compute selector stability across CV folds using selected features."""
    del dataset  # Reserved for a consistent cross-dataset analysis signature.

    required_columns = {"selector", "model", "k", "selected_features"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    stability_col = f"Mean Jacc. Sim. (k={k})"
    filtered = df.loc[(df["model"] == model) & (df["k"] == k)].copy()
    rows = []
    for selector, selector_df in filtered.groupby("selector"):
        feature_sets = [
            parse_selected_features(value)
            for value in selector_df["selected_features"]
        ]
        rows.append(
            {
                "Selector": selector,
                stability_col: mean_pairwise_jaccard(feature_sets),
                "Num folds": len(feature_sets),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(stability_col, ascending=False)
        .reset_index(drop=True)
    )
