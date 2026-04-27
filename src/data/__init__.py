"""Dataset loading and preprocessing.

Supported datasets (per project proposal):
    - Communities & Crime (UCI)  -> regression (ViolentCrimesPerPop)
    - Lending Club loans         -> classification (Charged Off vs. Fully Paid)
    - NHANES 2013-2014 (CDC)     -> classification (diabetes diagnosis, DIQ010)

Each dataset lives in its own module under :mod:`src.data` and exposes the
same three-stage API: ``load_<name>_raw`` (stage 1), ``clean_<name>``
(stage 2), ``preprocess_<name>`` (stage 3), plus a ``load_<name>``
convenience wrapper that runs stages 1+2 and returns a clean
``(DataFrame, Series)`` pair. Only NHANES has the staged functions
implemented today; the corresponding stubs for communities and lending_club
raise ``NotImplementedError`` and will be filled in later.

``load_dataset`` is the cross-dataset entrypoint: it runs stages 1+2,
performs the train/test split, median-imputes using the training fold only
(:func:`median_impute_train`), applies the dataset's stage-3 preprocessor
(if any), and packages everything into a :class:`Dataset`. The same order
applies inside each CV fold via :func:`arrays_for_fold`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base import DATA_DIR, RAW_DIR, Dataset, DatasetName, Task
from .imputation import median_impute_train
from .communities import (
    clean_communities,
    load_communities,
    load_communities_raw,
    preprocess_communities,
)
from .lending_club import (
    clean_lending_club,
    load_lending_club,
    load_lending_club_raw,
    preprocess_lending_club,
)
from .nhanes import clean_nhanes, load_nhanes, load_nhanes_raw, preprocess_nhanes

__all__ = [
    "DATA_DIR",
    "RAW_DIR",
    "Dataset",
    "DatasetName",
    "arrays_for_fold",
    "clean_communities",
    "clean_lending_club",
    "clean_nhanes",
    "load_communities",
    "load_communities_raw",
    "load_dataset",
    "load_lending_club",
    "load_lending_club_raw",
    "load_nhanes",
    "load_nhanes_raw",
    "median_impute_train",
    "prepare_xy",
    "preprocess_communities",
    "preprocess_lending_club",
    "preprocess_nhanes",
]

# Stages 1+2 combined (raw -> cleaned DataFrame).
_LOADERS = {
    "communities": load_communities,
    "lending_club": load_lending_club,
    "nhanes": load_nhanes,
}

_TASKS: dict[str, Task] = {
    "communities": "regression",
    "lending_club": "classification",
    "nhanes": "classification",
}

# Stage 3 (train/test-aware feature preprocessing, including scaling).
# Datasets without an entry here are passed through unchanged.
_PREPROCESSORS: dict[str, Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]] = {
    "nhanes": preprocess_nhanes,
    "communities": preprocess_communities,
    "lending_club": preprocess_lending_club,
}


def prepare_xy(name: DatasetName, **loader_kwargs) -> tuple[pd.DataFrame, np.ndarray, Task]:
    """Load cleaned features and labels before train/test split or CV."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(_LOADERS)}.")
    task = _TASKS[name]
    X_df, y = _LOADERS[name](**loader_kwargs)
    X_df = X_df.loc[:, X_df.std(numeric_only=True) > 0]
    y_arr = y.to_numpy(dtype=int if task == "classification" else float)
    return X_df, y_arr, task


def arrays_for_fold(
    name: DatasetName,
    X_df: pd.DataFrame,
    y_arr: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], Task]:
    """One CV fold: train-only median impute, then dataset preprocess, then numpy arrays."""
    task = _TASKS[name]
    X_tr_df = X_df.iloc[train_idx].copy()
    X_va_df = X_df.iloc[val_idx].copy()
    y_tr = y_arr[train_idx]
    y_va = y_arr[val_idx]

    X_tr_df, X_va_df = median_impute_train(X_tr_df, X_va_df)
    preprocess = _PREPROCESSORS.get(name)
    if preprocess is not None:
        X_tr_df, X_va_df = preprocess(X_tr_df, X_va_df)

    feature_names = list(X_tr_df.columns)
    X_train = X_tr_df.to_numpy(dtype=float)
    X_val = X_va_df.to_numpy(dtype=float)
    return X_train, X_val, y_tr, y_va, feature_names, task


def load_dataset(
    name: DatasetName,
    *,
    test_size: float = 0.2,
    random_state: int = 0,
    **loader_kwargs,
) -> Dataset:
    """Load a dataset by name, run the full pipeline, and return train/test splits.

    Scaling is the responsibility of each dataset's stage-3 preprocessor.
    """
    X_df, y_arr, task = prepare_xy(name, **loader_kwargs)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y_arr,
        test_size=test_size,
        random_state=random_state,
        stratify=y_arr if task == "classification" else None,
    )

    X_train_df, X_test_df = median_impute_train(X_train_df, X_test_df)
    preprocess = _PREPROCESSORS.get(name)
    if preprocess is not None:
        X_train_df, X_test_df = preprocess(X_train_df, X_test_df)

    feature_names = list(X_train_df.columns)
    X_train = X_train_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)

    return Dataset(name, X_train, X_test, y_train, y_test, feature_names, task)
