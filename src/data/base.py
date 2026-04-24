"""Shared types and constants for the dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

DatasetName = Literal["communities", "lending_club", "nhanes"]
Task = Literal["classification", "regression"]


@dataclass
class Dataset:
    name: DatasetName
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    task: Task
