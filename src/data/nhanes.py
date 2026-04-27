"""NHANES 2013-2014 (CDC) -- classification on diabetes diagnosis (DIQ010).

Three-stage pipeline:

    1. ``load_nhanes_raw(...)``  -> ``(X_df, y)``
       Pure I/O + minimal munging: read files, merge, filter target rows.
    2. ``clean_nhanes(X, y, ...)`` -> ``(X_df, y)``
       Generic, dataset-level cleaning that is *not* model-specific: drop ID
       columns, drop high-missingness columns, recode sentinels (median impute
       is deferred until after splits; see :mod:`src.data.imputation`).
       Output is still a human-readable DataFrame with original column names.
    3. ``preprocess_nhanes(X_train, X_test)`` -> ``(X_train_df, X_test_df)``
       Model-level feature preprocessing applied *after* the train/test split
       (categorical encoding, log/box-cox transforms, scaling).

``load_nhanes`` is a convenience wrapper that runs stages 1+2 and returns a
``(DataFrame, Series)`` pair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import RAW_DIR

# DIQ010: "Doctor told you have diabetes" (1 = Yes, 2 = No, 3 = Borderline,
# 7 = Refused, 9 = Don't know, NaN = not asked).
_NHANES_TARGET = "DIQ010"

# Participant ID and NHANES survey-design columns: not predictive of diabetes.
_NHANES_DROP_COLS = ["SEQN", "SDDSRVYR", "SDMVPSU", "SDMVSTRA", "WTINT2YR", "WTMEC2YR"]

# True "Refused" / "Don't know" sentinel codes to recode to NaN. Real
# categorical levels (e.g. RIDRETH3 = 7 "Other race") and top-coded counts
# (DMDHHSIZ = 7 "7 or more") are intentionally left alone.
_NHANES_SENTINELS: dict[str, tuple[int, ...]] = {
    "DMDBORN4": (77, 99),
    "DMDCITZN": (7, 9),
    "DMDEDUC2": (7, 9),
    "DMDEDUC3": (77, 99),
    "DMDHRBR4": (77, 99),
    "DMDHREDU": (7, 9),
    "DMDHRMAR": (77, 99),
    "DMDMARTL": (77, 99),
    "INDFMIN2": (77, 99),
    "INDHHIN2": (77, 99),
    "MGQ070":   (9,),
    "MGQ100":   (7, 9),
}

# Nominal categoricals -- one-hot encode in stage 3. Ordinal coded columns
# (DMDHREDU, INDFMIN2, INDHHIN2, ...) are kept as integers.
# DMQMILIZ, DMQADFC, RIDEXPRG, RIDSTATR, AIALANGA are listed defensively:
# under the current loader they are absent or zero-variance after cleaning,
# but if the loader is later extended (e.g. pulling more questionnaire cols)
# they should be one-hot encoded rather than scaled as numerics.
_NHANES_NOMINAL_COLS = [
    "RIAGENDR", "RIDRETH1", "RIDRETH3",
    "DMDMARTL", "DMDHRMAR",
    "DMDBORN4", "DMDHRBR4",
    "DMDHRGND", "DMDCITZN",
    "DMQMILIZ", "DMQADFC",
    "RIDEXMON", "RIDEXPRG", "RIDSTATR",
    "AIALANGA", "FIALANG", "MIALANG",
    "FIAINTRP", "FIAPROXY", "MIAINTRP", "MIAPROXY",
]

# Repeated BP readings to collapse into a single mean per participant.
_NHANES_BP_GROUPS = {
    "BPXSY_MEAN": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"],
    "BPXDI_MEAN": ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
}

# log1p-transform a BMX column if skew(x) on the training split exceeds this.
_NHANES_LOG_SKEW_THRESHOLD = 1.0


# --- Stage 1: raw loader --------------------------------------------------

def load_nhanes_raw(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Stage 1 -- read NHANES files, merge on ``SEQN``, filter target rows.

    Returns a *minimally* processed ``(X, y)`` pair: every original column
    (including ``SEQN``) is kept on ``X``; ``y`` is the recoded diabetes label
    (1 = Yes, 0 = No). Rows with target in ``{3 = Borderline, 7 = Refused,
    9 = Don't know}`` or missing are dropped.
    """
    path = Path(path) if path else RAW_DIR / "nhanes"

    demo = pd.read_csv(path / "demographic.csv")
    exam = pd.read_csv(path / "examination.csv")
    quest = pd.read_csv(path / "questionnaire.csv", usecols=["SEQN", _NHANES_TARGET])

    df = demo.merge(exam, on="SEQN", how="inner").merge(quest, on="SEQN", how="inner")

    df = df[df[_NHANES_TARGET].isin([1, 2])].copy()
    y = (df[_NHANES_TARGET] == 1).astype(int).reset_index(drop=True)
    X = df.drop(columns=[_NHANES_TARGET]).reset_index(drop=True)
    return X, y


# --- Stage 2: dataset-level cleaning --------------------------------------

def clean_nhanes(
    X: pd.DataFrame,
    y: pd.Series,
    drop_high_missing: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stage 2 -- drop ID/survey-design cols, keep numerics, recode sentinels
    to NaN, drop cols with > ``drop_high_missing`` missing, median-impute.
    """
    X = X.drop(columns=_NHANES_DROP_COLS, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    for col, codes in _NHANES_SENTINELS.items():
        if col in X.columns:
            X[col] = X[col].replace(list(codes), np.nan)

    missing_frac = X.isna().mean()
    X = X.loc[:, missing_frac <= drop_high_missing]
    # Median imputation is applied after train/val or train/test splits (see
    # :func:`src.data.median_impute_train`) to avoid leakage.

    return X.reset_index(drop=True), y.reset_index(drop=True)


# --- Stage 3: model-level feature preprocessing ---------------------------

def preprocess_nhanes(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 3 -- average repeated BPs, log1p skewed BMX cols, one-hot
    nominals, then standardise non-dummy columns.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    for new_col, src_cols in _NHANES_BP_GROUPS.items():
        kept = [c for c in src_cols if c in X_train.columns]
        if not kept:
            continue
        X_train[new_col] = X_train[kept].mean(axis=1)
        X_test[new_col] = X_test[kept].mean(axis=1)
        X_train = X_train.drop(columns=kept)
        X_test = X_test.drop(columns=kept)

    bmx_cols = [c for c in X_train.columns if c.startswith("BMX")]
    log_cols = X_train[bmx_cols].skew() > _NHANES_LOG_SKEW_THRESHOLD
    for col in log_cols.index[log_cols]:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])

    nom = [c for c in _NHANES_NOMINAL_COLS if c in X_train.columns]
    if nom:
        X_train = pd.get_dummies(X_train, columns=nom, drop_first=True, dtype=float)
        X_test = pd.get_dummies(X_test, columns=nom, drop_first=True, dtype=float)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    dummy_cols = [c for c in X_train.columns if any(c.startswith(n + "_") for n in nom)]
    scale_cols = [c for c in X_train.columns if c not in dummy_cols]
    if scale_cols:
        scaler = StandardScaler()
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train, X_test


# --- Convenience wrapper (stages 1 + 2) -----------------------------------

def load_nhanes(
    path: Path | None = None,
    drop_high_missing: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run stages 1 + 2 and return a clean ``(DataFrame, Series)`` pair."""
    X, y = load_nhanes_raw(path)
    return clean_nhanes(X, y, drop_high_missing=drop_high_missing)
