"""Lending Club loans -- classification on default vs. fully paid.

Three-stage pipeline mirroring :mod:`src.data.nhanes`:

    1. ``load_lending_club_raw(...)`` -> ``(X_df, y)``
       Read the CSV, filter rows whose ``loan_status`` is *resolved*
       (Charged Off / Default / Fully Paid). All 150 raw feature columns
       are kept on ``X``; ``y`` is the binary default label.
    2. ``clean_lending_club(X, y, ...)`` -> ``(X_df, y)``
       Dataset-level cleaning: drop ID / free-text / date / single-value
       columns, drop *leaky* columns recorded after loan origination,
       parse ``term`` / ``emp_length`` / ``grade`` / ``sub_grade`` to
       numeric, integer-encode the remaining nominal categoricals, drop
       columns with > ``drop_high_missing`` missing.
    3. ``preprocess_lending_club(X_train, X_test)`` -> ``(X_train_df, X_test_df)``
       Model-level preprocessing applied *after* the train/test split:
       ``log1p`` heavily right-skewed continuous columns, one-hot encode
       nominals, standardise the non-dummy columns.

``load_lending_club`` is a convenience wrapper that runs stages 1 + 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import RAW_DIR

# Resolved loan_status values -> binary target. Anything else (Current,
# In Grace Period, Late ...) is dropped because the outcome isn't known yet.
_LC_POSITIVE = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
_LC_NEGATIVE = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}

# Recorded *after* the loan has played out -> trivially leak the outcome.
_LC_LEAKY = [
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low",
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "hardship_start_date", "hardship_end_date", "payment_plan_start_date",
    "hardship_loan_status", "hardship_amount", "hardship_length",
    "hardship_dpd", "deferral_term",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
]

# Free-text / identifier / date / high-cardinality string columns.
_LC_DROP = [
    "id", "member_id", "url", "desc", "emp_title", "title",
    "zip_code", "addr_state",
    "issue_d", "earliest_cr_line",
    "policy_code",
]

# Single-valued columns in the resolved subset (zero variance -> useless).
_LC_SINGLE_VALUE = ["pymnt_plan", "disbursement_method"]

# Nominal categoricals to one-hot in stage 3 (kept as integer codes
# through stage 2 + median imputation, mirroring NHANES).
_LC_NOMINAL_COLS = [
    "home_ownership", "verification_status", "purpose",
    "application_type", "initial_list_status",
]

# Ordinal mappings: keep as numeric so the ordering is preserved.
_LC_GRADE_MAP = {g: i + 1 for i, g in enumerate("ABCDEFG")}
_LC_SUB_GRADE_MAP = {f"{g}{n}": (i * 5 + n) for i, g in enumerate("ABCDEFG") for n in range(1, 6)}
_LC_EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}

# log1p-transform a non-negative continuous column if its training-fold
# skew exceeds this threshold. Lending Club has many heavy-tailed dollar /
# count columns (annual_inc, revol_bal, tot_coll_amt, ...).
_LC_LOG_SKEW_THRESHOLD = 2.0


# --- Stage 1: raw loader --------------------------------------------------

def load_lending_club_raw(
    path: Path | None = None,
    nrows: int | None = 5_000,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stage 1 -- read the Lending Club CSV and return a raw ``(X, y)`` pair.

    Only the *resolved* loan rows are kept (``Charged Off`` / ``Default`` /
    ``Fully Paid`` and their "Does not meet the credit policy" variants),
    then a random sample of size ``nrows`` is drawn (seeded by
    ``random_state``) so the experiment is reproducible *and* spans the
    full 2007-2018 window instead of the first-N rows (which are all
    2015 issues, since the CSV is sorted by ``issue_d``).

    To keep the cost low, we make two passes: a fast single-column read of
    ``loan_status`` to identify resolved rows, then a second read that
    skips everything outside the sampled indices.

    Pass ``nrows=None`` to load every resolved loan (~1.35 M rows, ~3 GB).
    """
    path = Path(path) if path else RAW_DIR / "lending_club" / "accepted_2007_to_2018Q4.csv"
    if path.is_dir():
        path = path / "accepted_2007_to_2018Q4.csv"

    status = pd.read_csv(path, usecols=["loan_status"])["loan_status"]
    resolved_idx = np.where(status.isin(_LC_POSITIVE | _LC_NEGATIVE).to_numpy())[0]

    if nrows is None or nrows >= len(resolved_idx):
        df = pd.read_csv(path, low_memory=False).iloc[resolved_idx]
    else:
        rng = np.random.default_rng(random_state)
        keep = set(rng.choice(resolved_idx, size=nrows, replace=False).tolist())
        # +1 offset accounts for the CSV header row.
        df = pd.read_csv(path, low_memory=False,
                         skiprows=lambda i: i > 0 and (i - 1) not in keep)

    y = df["loan_status"].isin(_LC_POSITIVE).astype(int).reset_index(drop=True)
    X = df.drop(columns=["loan_status"]).reset_index(drop=True)
    return X, y


# --- Stage 2: dataset-level cleaning --------------------------------------

def clean_lending_club(
    X: pd.DataFrame,
    y: pd.Series,
    drop_high_missing: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stage 2 -- drop leaky / id / text / single-value cols, parse string-
    encoded numerics (``term``, ``emp_length``, ``grade``, ``sub_grade``),
    integer-encode nominal categoricals, drop high-missing columns.

    Output is fully numeric so :func:`median_impute_train` and any
    downstream numpy conversion work without further coercion. Median
    imputation itself is deferred until *after* the train/val split (see
    :mod:`src.data.imputation`) to avoid leakage.
    """
    X = X.drop(columns=_LC_LEAKY + _LC_DROP + _LC_SINGLE_VALUE, errors="ignore")

    if "term" in X.columns:
        X["term"] = X["term"].str.extract(r"(\d+)").astype(float)
    if "emp_length" in X.columns:
        X["emp_length"] = X["emp_length"].map(_LC_EMP_LENGTH_MAP).astype(float)
    if "grade" in X.columns:
        X["grade"] = X["grade"].map(_LC_GRADE_MAP).astype(float)
    if "sub_grade" in X.columns:
        X["sub_grade"] = X["sub_grade"].map(_LC_SUB_GRADE_MAP).astype(float)

    for col in _LC_NOMINAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan).astype(float)

    X = X.select_dtypes(include=[np.number])

    missing_frac = X.isna().mean()
    X = X.loc[:, missing_frac <= drop_high_missing]

    return X.reset_index(drop=True), y.reset_index(drop=True)


# --- Stage 3: model-level feature preprocessing ---------------------------

def preprocess_lending_club(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 3 -- ``log1p`` heavily-skewed non-negative columns, one-hot
    nominals, then standardise the non-dummy columns.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    nom = [c for c in _LC_NOMINAL_COLS if c in X_train.columns]
    cont = [c for c in X_train.columns if c not in nom]
    nonneg = [c for c in cont if (X_train[c] >= 0).all()]
    skewed = X_train[nonneg].skew() > _LC_LOG_SKEW_THRESHOLD
    for col in skewed.index[skewed]:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col].clip(lower=0))

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

def load_lending_club(
    path: Path | None = None,
    nrows: int | None = 5_000,
    random_state: int = 0,
    drop_high_missing: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run stages 1 + 2 and return a clean ``(DataFrame, Series)`` pair."""
    X, y = load_lending_club_raw(path, nrows=nrows, random_state=random_state)
    return clean_lending_club(X, y, drop_high_missing=drop_high_missing)
