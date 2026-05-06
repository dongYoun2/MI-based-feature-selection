"""Build a cross-dataset summary table of top baseline vs MI-based selectors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Mapping

import pandas as pd

MODELS_ALLOWED: tuple[str, ...] = ("gradient_boosting", "logreg")
K_ALLOWED: tuple[int, ...] = (5, 10, 20, 40)

BASELINE_SELECTORS: tuple[str, ...] = ("correlation", "l1", "rfe", "shap")
MI_SELECTORS: tuple[str, ...] = ("mi", "mrmr_heuristic", "cmim", "jmi", "pid")

# Display labels (canonical selector keys -> prose for tables).
SELECTOR_DISPLAY: dict[str, str] = {
    "correlation": "Correlation",
    "l1": "L1",
    "rfe": "RFE",
    "shap": "SHAP",
    "mi": "MI",
    "mrmr_heuristic": "mRMR (heuristic)",
    "cmim": "CMIM",
    "jmi": "JMI",
    "pid": "PID",
}


def _format_selector_labels(selectors: Iterable[str]) -> str:
    parts = []
    for s in selectors:
        parts.append(SELECTOR_DISPLAY.get(s, s.upper() if len(s) <= 4 else s.replace("_", " ").title()))
    return ", ".join(parts)


def _mean_score_by_selector(df: pd.DataFrame, metric_col: str) -> pd.Series:
    sub = df[
        df["model"].isin(MODELS_ALLOWED) & df["k"].isin(K_ALLOWED)
    ].copy()
    if sub.empty:
        raise ValueError(f"No rows after filtering models={MODELS_ALLOWED}, k={K_ALLOWED}")
    return sub.groupby("selector", observed=True)[metric_col].mean()


def _top_two(
    scores: pd.Series,
    candidates: tuple[str, ...],
) -> list[str]:
    rows = [(s, float(scores.loc[s])) for s in candidates if s in scores.index]
    if not rows:
        raise ValueError(f"None of {candidates!r} present in aggregated scores.")
    rows.sort(key=lambda x: (-x[1], x[0]))
    return [s for s, _ in rows[:2]]


DatasetKey = Literal["nhanes", "lending_club", "communities"]


DATASET_LABELS: dict[DatasetKey, str] = {
    "nhanes": "NHANES",
    "lending_club": "Lending Club",
    "communities": "Communities",
}


def cross_dataset_best_selectors(
    csv_paths: Mapping[DatasetKey, str | Path],
) -> pd.DataFrame:
    """
    Aggregate metrics from ``metrics_<dataset>.csv`` files and return top-2 baseline
    vs top-2 MI-based selectors per dataset.

    Parameters
    ----------
    csv_paths
        Map logical dataset keys ``nhanes``, ``lending_club``, ``communities``
        to file paths.

    Notes
    -----
    Classification datasets use mean ``avg_precision_mean`` across models and ``k``.
    Communities uses mean ``r2_mean``.
    """
    rows: list[dict[str, object]] = []
    for key in sorted(csv_paths.keys(), key=str):
        label = DATASET_LABELS[key]
        path = Path(csv_paths[key])
        df = pd.read_csv(path)
        if key == "communities":
            metric_col = "r2_mean"
            main_metric = "R²"
        else:
            metric_col = "avg_precision_mean"
            main_metric = "Avg Precision"

        scores = _mean_score_by_selector(df, metric_col)
        best_bl = _top_two(scores, BASELINE_SELECTORS)
        best_mi = _top_two(scores, MI_SELECTORS)

        rows.append(
            {
                "Dataset": label,
                "Main metric": main_metric,
                "Best baseline": _format_selector_labels(best_bl),
                "Best MI-based": _format_selector_labels(best_mi),
            },
        )

    return pd.DataFrame(rows)
