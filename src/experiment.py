"""Main experimental pipeline.

For each (dataset, feature selection method, model, k) combination:
    1. Load data with train-only preprocessing on each CV fold (or a single
       train/test split when ``cv_folds`` is 0).
    2. Select k features on the training part.
    3. Fit a predictive model on the selected features.
    4. Evaluate on the held-out validation or test part.
    5. Record metrics (with mean ± std across folds when using CV).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .data import Dataset, DatasetName, arrays_for_fold, load_dataset, prepare_xy
from .evaluate import evaluate_model
from .feature_selection import (
    correlation_filter,
    l1_selection,
    mi_filter,
    mrmr,
    rfe_selection,
    shap_selection,
)
from .models import ModelName, build_model

SelectorFn = Callable[..., list[int]]

SELECTORS: dict[str, SelectorFn] = {
    "mrmr": mrmr,
    "mi": mi_filter,
    "correlation": correlation_filter,
    "l1": l1_selection,
    "rfe": rfe_selection,
    "shap": shap_selection,
    # "cmi": cmi_selection,           # TODO
    # "pid": pid_selection,           # TODO
}


@lru_cache(maxsize=None)
def _cached_select(dataset_name: DatasetName, selector: str, k: int):
    ds = load_dataset(dataset_name)
    selector_fn = SELECTORS[selector]
    selected = selector_fn(ds.X_train, ds.y_train, k=k, task=ds.task)
    return tuple(selected)  # make hashable


def run_single(
    ds: Dataset,
    selector: str,
    model_name: ModelName,
    k: int,
    random_state: int = 0,
) -> dict:
    selected = _cached_select(ds.name, selector, int(k))

    X_train_sel = ds.X_train[:, selected]
    X_test_sel = ds.X_test[:, selected]

    model = build_model(model_name, task=ds.task, random_state=random_state)
    model.fit(X_train_sel, ds.y_train)

    result = evaluate_model(model, X_test_sel, ds.y_test, ds.task, selected)

    return {
        "dataset": ds.name,
        "selector": selector,
        "model": model_name,
        "k": k,
        "selected_features": [ds.feature_names[i] for i in selected],
        **asdict(result),
    }


def _run_one_fold(
    dataset: DatasetName,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    task: str,
    selector: str,
    model_name: ModelName,
    k: int,
    random_state: int,
) -> dict:
    selector_fn = SELECTORS[selector]
    selected = selector_fn(X_train, y_train, k=k, task=task)
    X_train_sel = X_train[:, selected]
    X_val_sel = X_val[:, selected]

    model = build_model(model_name, task=task, random_state=random_state)
    model.fit(X_train_sel, y_train)

    result = evaluate_model(model, X_val_sel, y_val, task, selected)

    return {
        "dataset": dataset,
        "selector": selector,
        "model": model_name,
        "k": k,
        **asdict(result),
    }


def _aggregate_cv_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    gcols = ["dataset", "selector", "model", "k"]
    task_by_ds = fold_df.groupby("dataset")["task"].first()

    metric_cols = [c for c in ("auc", "f1", "rmse", "r2", "n_features") if c in fold_df.columns]
    agg_dict = {}
    for m in metric_cols:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")

    out = fold_df.groupby(gcols, as_index=False).agg(**agg_dict)
    n_fold = fold_df.groupby(gcols, as_index=False).size().rename(columns={"size": "n_cv_folds"})
    out = out.merge(n_fold, on=gcols)
    out["task"] = out["dataset"].map(task_by_ds)
    return out


def run_cv(
    datasets: Iterable[DatasetName],
    selectors: Iterable[str],
    models: Iterable[ModelName],
    ks: Iterable[int],
    n_splits: int,
    *,
    random_state: int = 0,
    loader_kwargs: dict | None = None,
) -> pd.DataFrame:
    """K-fold CV: median imputation and stage-3 preprocess are fit on each training fold only."""
    loader_kwargs = loader_kwargs or {}
    fold_records: list[dict] = []

    for dataset in datasets:
        X_df, y_arr, task = prepare_xy(dataset, **loader_kwargs)
        n_feat = X_df.shape[1]

        effective_ks = sorted({min(int(k), n_feat) for k in ks})
        clipped = sorted({int(k) for k in ks if int(k) > n_feat})
        if clipped:
            warnings.warn(
                f"[{dataset}] requested k={clipped} exceed n_features={n_feat}; "
                f"capped to {n_feat} and deduplicated. Running k={effective_ks}.",
                stacklevel=2,
            )

        splitter_cls = StratifiedKFold if task == "classification" else KFold
        splitter = splitter_cls(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        split_y = y_arr if task == "classification" else None

        for train_idx, val_idx in splitter.split(X_df, split_y):
            X_train, X_val, y_train, y_val, _feat_names, _task = arrays_for_fold(
                dataset, X_df, y_arr, train_idx, val_idx
            )
            n_feat_fold = X_train.shape[1]
            fold_ks = sorted({min(int(k), n_feat_fold) for k in effective_ks})

            for selector in selectors:
                for model_name in models:
                    for k in fold_ks:
                        fold_records.append(
                            _run_one_fold(
                                dataset,
                                X_train,
                                X_val,
                                y_train,
                                y_val,
                                _task,
                                selector,
                                model_name,
                                k,
                                random_state,
                            )
                        )

    fold_df = pd.DataFrame(fold_records)
    return _aggregate_cv_folds(fold_df)


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
    if cv_folds > 1:
        df = run_cv(
            datasets,
            selectors,
            models,
            ks,
            n_splits=cv_folds,
            random_state=random_state,
            loader_kwargs=loader_kwargs,
        )
    else:
        records = []
        for dataset in datasets:
            ds = load_dataset(dataset, random_state=random_state, **(loader_kwargs or {}))
            n_feat = ds.X_train.shape[1]

            effective_ks = sorted({min(int(k), n_feat) for k in ks})
            clipped = sorted({int(k) for k in ks if int(k) > n_feat})
            if clipped:
                warnings.warn(
                    f"[{dataset}] requested k={clipped} exceed n_features={n_feat}; "
                    f"capped to {n_feat} and deduplicated. Running k={effective_ks}.",
                    stacklevel=2,
                )

            for selector in selectors:
                for model_name in models:
                    for k in effective_ks:
                        records.append(run_single(ds, selector, model_name, k, random_state))

        df = pd.DataFrame(records)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df
