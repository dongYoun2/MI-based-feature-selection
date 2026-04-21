"""Dataset loading and preprocessing.

Supported datasets (per project proposal):
    - Communities & Crime (UCI)
    - Lending Club loans (Kaggle)
    - NHANES (CDC, 2013-2014)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DatasetName = Literal["communities", "lending_club", "nhanes"]


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    task: Literal["classification", "regression"]


def load_communities(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load Communities & Crime dataset. Target: ViolentCrimesPerPop (regression)."""
    raise NotImplementedError("TODO: load Communities & Crime from UCI")


def load_lending_club(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load Lending Club loans. Target: loan default (classification)."""
    raise NotImplementedError("TODO: load Lending Club loan data")


def load_nhanes(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load NHANES 2013-2014. Target: a health outcome (classification)."""
    raise NotImplementedError("TODO: load NHANES")


_LOADERS = {
    "communities": load_communities,
    "lending_club": load_lending_club,
    "nhanes": load_nhanes,
}


def load_dataset(
    name: DatasetName,
    test_size: float = 0.2,
    random_state: int = 0,
    standardize: bool = True,
) -> Dataset:
    """Load a dataset by name, preprocess, and return train/test splits."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(_LOADERS)}.")

    X_df, y = _LOADERS[name]()
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)
    y_arr = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=test_size, random_state=random_state,
        stratify=y_arr if y_arr.dtype.kind in "iuOb" else None,
    )

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    task = "classification" if np.unique(y_arr).size < 20 else "regression"
    return Dataset(X_train, X_test, y_train, y_test, feature_names, task)
