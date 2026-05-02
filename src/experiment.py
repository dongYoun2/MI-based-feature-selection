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

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .data import DatasetName, arrays_for_fold, load_dataset, prepare_xy
from .evaluate import evaluate_model
from .feature_selection import (
    correlation_filter,
    dynamic_cmi_selection,
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
    "pid": pid_selection,
    "dynamic_cmi": dynamic_cmi_selection,
}


def _cap_ks(ks: Iterable[int], n_feat: int, dataset: str, *, warn: bool) -> list[int]:
    effective = sorted({min(int(k), n_feat) for k in ks if int(k) > 0})
    clipped = sorted({int(k) for k in ks if int(k) > n_feat})
    if warn and clipped:
        warnings.warn(
            f"[{dataset}] requested k={clipped} exceed n_features={n_feat}; "
            f"capped to {n_feat} and deduplicated. Running k={effective}.",
            stacklevel=3,
        )
    return effective


def _eval_one_fold(
    dataset: DatasetName,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    task: str,
    selectors: Iterable[str],
    models: Iterable[ModelName],
    ks: Iterable[int],
    random_state: int,
    *,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """Run all (selector, k, model) combinations on one (train, val) split.

    Selection is performed once per selector at ``k_max`` and sliced per ``k``.
    """
    fold_ks = _cap_ks(ks, X_train.shape[1], dataset, warn=False)
    if not fold_ks:
        return []
    k_max = max(fold_ks)

    records: list[dict] = []
    for selector in selectors:
        full_selected = list(SELECTORS[selector](
            X_train, y_train, k=k_max, task=task, random_state=random_state,
        ))
        for k in fold_ks:
            selected = full_selected[:k]
            X_train_sel = X_train[:, selected]
            X_val_sel = X_val[:, selected]
            for model_name in models:
                model = build_model(model_name, task=task, random_state=random_state)
                model.fit(X_train_sel, y_train)
                result = evaluate_model(model, X_val_sel, y_val, task, selected)
                rec: dict = {
                    "dataset": dataset,
                    "selector": selector,
                    "model": model_name,
                    "k": k,
                    **asdict(result),
                }
                if feature_names is not None:
                    rec["selected_features"] = [feature_names[i] for i in selected]
                records.append(rec)
    return records


def _aggregate_cv_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    gcols = ["dataset", "selector", "model", "k"]
    task_by_ds = fold_df.groupby("dataset")["task"].first()

    metric_cols = [c for c in ("auc", "avg_precision", "rmse", "r2", "n_features") if c in fold_df.columns]
    agg_dict = {f"{m}_{stat}": (m, stat) for m in metric_cols for stat in ("mean", "std")}

    out = fold_df.groupby(gcols, as_index=False).agg(**agg_dict)
    n_fold = fold_df.groupby(gcols, as_index=False).size().rename(columns={"size": "n_cv_folds"})
    out = out.merge(n_fold, on=gcols)
    out["task"] = out["dataset"].map(task_by_ds)
    return out


def run_grid(
    datasets: Iterable[DatasetName],
    selectors: Iterable[str],
    models: Iterable[ModelName],
    ks: Iterable[int],
    out_path: Path | None = None,
    *,
    cv_folds: int = 0,
    random_state: int = 0,
    loader_kwargs: dict | None = None,
) -> pd.DataFrame:
    """K-fold CV (``cv_folds > 1``) or a single train/test split otherwise.

    In both modes, per-fold preprocessing (median imputation + dataset stage-3)
    is fit on the training part only.
    """
    loader_kwargs = loader_kwargs or {}
    is_cv = cv_folds > 1
    records: list[dict] = []

    for dataset in datasets:
        if is_cv:
            X_df, y_arr, task = prepare_xy(dataset, **loader_kwargs)
            _cap_ks(ks, X_df.shape[1], dataset, warn=True)

            splitter_cls = StratifiedKFold if task == "classification" else KFold
            splitter = splitter_cls(n_splits=cv_folds, shuffle=True, random_state=random_state)
            split_y = y_arr if task == "classification" else None

            for train_idx, val_idx in splitter.split(X_df, split_y):
                X_train, X_val, y_train, y_val, _, fold_task = arrays_for_fold(
                    dataset, X_df, y_arr, train_idx, val_idx
                )
                records.extend(_eval_one_fold(
                    dataset, X_train, X_val, y_train, y_val, fold_task,
                    selectors, models, ks, random_state,
                ))
        else:
            ds = load_dataset(dataset, random_state=random_state, **loader_kwargs)
            _cap_ks(ks, ds.X_train.shape[1], dataset, warn=True)
            records.extend(_eval_one_fold(
                dataset, ds.X_train, ds.X_test, ds.y_train, ds.y_test, ds.task,
                selectors, models, ks, random_state,
                feature_names=ds.feature_names,
            ))

    df = pd.DataFrame(records)
    if is_cv:
        df = _aggregate_cv_folds(df)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df
