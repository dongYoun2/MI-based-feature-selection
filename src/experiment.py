"""Main experimental pipeline.

For each (dataset, fold, selector) we run feature selection ONCE at
``k = max(ks)`` and reuse the result for every smaller ``k`` by slicing.
This is correctness-preserving for all selectors here:

    - score-based filters (correlation, mi, l1, shap) and RFE produce a
      fixed importance ranking; top-k is just the prefix.
    - greedy methods (mrmr, mrmr_heuristic, pid) have a selection order
      that is independent of the target k.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .data import DatasetName, arrays_for_fold, load_dataset, prepare_xy
from .evaluate import evaluate_model
from .feature_selection import (
    cmim,
    correlation_filter,
    l1_selection,
    mi_filter,
    mrmr,
    mrmr_heuristic,
    pid_selection,
    rfe_selection,
    shap_selection,
)
from .models import ModelName, build_model

SelectorFn = Callable[..., list[int]]

SELECTORS: dict[str, SelectorFn] = {
    "mrmr": mrmr,
    "mrmr_heuristic": mrmr_heuristic,
    "mi": mi_filter,
    "correlation": correlation_filter,
    "l1": l1_selection,
    "rfe": rfe_selection,
    "shap": shap_selection,
    "cmim": cmim,
    "pid": pid_selection,
}


def _validate_ks(ks: list[int], n_feat: int, dataset: str) -> None:
    """Raise ``ValueError`` if any requested ``k`` exceeds ``n_feat``.

    Called once per dataset with the post-cleaning feature count, so the
    user sees a clear error before any expensive selection/training runs.
    """
    over = [k for k in ks if k > n_feat]
    if over:
        raise ValueError(
            f"[{dataset}] requested k={over} exceed available n_features={n_feat}. "
            f"Reduce --ks (max allowed: {n_feat}) or pick a different dataset."
        )


def _eval_one_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    task: str,
    selectors: Iterable[str],
    models: Iterable[ModelName],
    ks: list[int],
    random_state: int,
    *,
    fold: int,
    feature_names: list[str],
) -> list[dict]:
    """Run all (selector, k, model) combinations on one (train, val) split.

    ``ks`` is assumed to already be sorted/deduped and validated against the
    dataset's feature count by the caller. Selection is performed once per
    selector at ``k_max`` and sliced per ``k``. Each returned record carries
    the ``fold`` index and the names of the selected features for that
    (fold, selector, k).
    """
    k_max = max(ks)

    records: list[dict] = []
    for selector in selectors:
        full_selected = list(SELECTORS[selector](
            X_train, y_train, k=k_max, task=task, random_state=random_state,
        ))
        for k in ks:
            selected = full_selected[:k]
            X_train_sel = X_train[:, selected]
            X_val_sel = X_val[:, selected]
            for model_name in models:
                model = build_model(model_name, task=task, random_state=random_state)
                model.fit(X_train_sel, y_train)
                result = evaluate_model(model, X_val_sel, y_val, task)
                records.append({
                    "fold": fold,
                    "selector": selector,
                    "model": model_name,
                    "k": k,
                    "selected_features": [feature_names[i] for i in selected],
                    **result,
                })
    return records


def _aggregate_cv_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    gcols = ["selector", "model", "k"]

    metric_cols = [c for c in ("auc", "avg_precision", "rmse", "r2") if c in fold_df.columns]
    agg_dict = {f"{m}_{stat}": (m, stat) for m in metric_cols for stat in ("mean", "std")}

    out = fold_df.groupby(gcols, as_index=False).agg(**agg_dict)
    n_fold = fold_df.groupby(gcols, as_index=False).size().rename(columns={"size": "n_cv_folds"})
    out = out.merge(n_fold, on=gcols)
    return out


def run_experiment(
    dataset: DatasetName,
    selectors: Iterable[str],
    models: Iterable[ModelName],
    ks: list[int],
    out_dir: Path | None = None,
    *,
    cv_folds: int = 0,
    random_state: int = 0,
    loader_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the (selector, k, model) grid on a single dataset.

    Uses K-fold CV when ``cv_folds > 1``, otherwise a single train/test split
    (treated as ``fold=0``). In both modes, per-fold preprocessing (median
    imputation + dataset stage-3) is fit on the training part only.

    ``ks`` is expected to be a non-empty, sorted, deduplicated list of
    positive integers (callers should normalize at the input boundary).
    Each ``k`` is validated against the dataset's feature count up-front;
    ``ValueError`` is raised if any ``k`` exceeds it.

    When ``out_dir`` is given, writes two CSV files:

        - ``metrics_<dataset>_per_fold.csv``: one row per
          (fold, selector, k, model), including ``selected_features``.
        - ``metrics_<dataset>.csv``: aggregated summary with mean/std of the
          eval metrics across folds.

    Returns ``(per_fold_df, summary_df)``.
    """
    loader_kwargs = loader_kwargs or {}
    is_cv = cv_folds > 1
    records: list[dict] = []

    if is_cv:
        X_df, y_arr, task = prepare_xy(dataset, **loader_kwargs)
        _validate_ks(ks, X_df.shape[1], dataset)

        splitter_cls = StratifiedKFold if task == "classification" else KFold
        splitter = splitter_cls(n_splits=cv_folds, shuffle=True, random_state=random_state)
        split_y = y_arr if task == "classification" else None

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_df, split_y)):
            X_train, X_val, y_train, y_val, feature_names, fold_task = arrays_for_fold(
                dataset, X_df, y_arr, train_idx, val_idx
            )
            records.extend(_eval_one_fold(
                X_train, X_val, y_train, y_val, fold_task,
                selectors, models, ks, random_state,
                fold=fold_idx, feature_names=feature_names,
            ))
    else:
        ds = load_dataset(dataset, random_state=random_state, **loader_kwargs)
        _validate_ks(ks, ds.X_train.shape[1], dataset)
        records.extend(_eval_one_fold(
            ds.X_train, ds.X_test, ds.y_train, ds.y_test, ds.task,
            selectors, models, ks, random_state,
            fold=0, feature_names=ds.feature_names,
        ))

    per_fold_df = pd.DataFrame(records)
    summary_df = _aggregate_cv_folds(per_fold_df)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_fold_df.to_csv(out_dir / f"metrics_{dataset}_per_fold.csv", index=False)
        summary_df.to_csv(out_dir / f"metrics_{dataset}.csv", index=False)

    return per_fold_df, summary_df
