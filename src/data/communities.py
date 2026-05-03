"""Communities & Crime (UCI) -- regression on ViolentCrimesPerPop.

Three-stage pipeline:

    1. ``load_communities_raw`` -> ``(X_df, y)``
       Read ``communities.data`` / ``communities.names``; parse ``?`` as NA;
       split target ``ViolentCrimesPerPop`` into ``y``; keep all remaining
       columns on ``X`` (including non-predictive fields removed in stage 2).
    2. ``clean_communities`` -> ``(X_df, y)``
       Drop non-predictive location / fold columns; drop features with missing
       fraction above ``drop_high_missing``. Median imputation is deferred
       until after train/test splits (:mod:`src.data.imputation`).
    3. ``preprocess_communities(X_train, X_test)`` -> ``(X_train_df, X_test_df)``
       ``StandardScaler`` fit on train only (applied after imputation in
       :func:`src.data.load_dataset`).

``load_communities`` runs stages 1+2 and returns a clean ``(DataFrame, Series)``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import RAW_DIR

# Non-predictive columns per the dataset description.
_COMMUNITIES_NON_PREDICTIVE = ["state", "county", "community", "communityname", "fold"]
_COMMUNITIES_TARGET = "ViolentCrimesPerPop"


def _communities_columns(names_path: Path) -> list[str]:
    """Parse @attribute lines from communities.names."""
    cols = []
    pattern = re.compile(r"^@attribute\s+(\S+)\s+")
    for line in names_path.read_text().splitlines():
        m = pattern.match(line)
        if m:
            cols.append(m.group(1))
    return cols


# --- Stage 1: raw loader --------------------------------------------------


def load_communities_raw(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Stage 1 -- read communities files; return ``(X, y)`` with raw features."""
    path = Path(path) if path else RAW_DIR / "communities"
    columns = _communities_columns(path / "communities.names")
    df = pd.read_csv(
        path / "communities.data",
        header=None,
        names=columns,
        na_values="?",
    )
    y = df[_COMMUNITIES_TARGET].astype(float)
    X = df.drop(columns=[_COMMUNITIES_TARGET])
    return X, y


# --- Stage 2: dataset-level cleaning --------------------------------------


def clean_communities(
    X: pd.DataFrame,
    y: pd.Series,
    drop_high_missing: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stage 2 -- drop non-predictive columns and high-missingness columns."""
    X = X.drop(columns=_COMMUNITIES_NON_PREDICTIVE, errors="ignore")

    tmp = X.copy()
    tmp[_COMMUNITIES_TARGET] = y
    missing_frac = tmp.isna().mean()
    tmp = tmp.loc[:, missing_frac <= drop_high_missing]
    y_out = tmp[_COMMUNITIES_TARGET]
    X_out = tmp.drop(columns=[_COMMUNITIES_TARGET])
    return X_out.reset_index(drop=True), y_out.reset_index(drop=True)


# --- Stage 3: model-level feature preprocessing ---------------------------


def preprocess_communities(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 3 -- standardise numeric features (train fit only)."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    cols = list(X_train.columns)
    if not cols:
        return X_train, X_test
    scaler = StandardScaler()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])
    return X_train, X_test


# --- Convenience wrapper (stages 1 + 2) -----------------------------------


def load_communities(
    path: Path | None = None,
    drop_high_missing: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load Communities & Crime. Target: ``ViolentCrimesPerPop`` (regression).

    Args:
        path: directory containing ``communities.data`` and ``communities.names``.
        drop_high_missing: drop columns with a fraction of missing values
            above this threshold (defaults to 0.3, which removes the LEMAS
            block that is missing for most communities).
    """
    X, y = load_communities_raw(path)
    return clean_communities(X, y, drop_high_missing=drop_high_missing)
