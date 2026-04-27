"""Lending Club loans -- classification on default vs. fully paid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import RAW_DIR

# Target labels.
_LC_POSITIVE = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
_LC_NEGATIVE = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}

# Columns that leak the outcome (recorded after the loan is issued).
_LC_LEAKY = [
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low",
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
]

# Free-text / identifier columns we drop up front.
_LC_DROP = ["id", "member_id", "url", "desc", "emp_title", "title", "zip_code"]


# --- Stage 1: raw loader --------------------------------------------------

def load_lending_club_raw(
    path: Path | None = None,
    nrows: int | None = 100_000,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stage 1 -- read the Lending Club CSV and return a raw ``(X, y)`` pair."""
    raise NotImplementedError(
        "load_lending_club_raw is not yet split out; use load_lending_club for now."
    )


# --- Stage 2: dataset-level cleaning --------------------------------------

def clean_lending_club(
    X: pd.DataFrame,
    y: pd.Series,
    drop_high_missing: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stage 2 -- dataset-level cleaning for Lending Club."""
    raise NotImplementedError(
        "clean_lending_club is not yet split out; use load_lending_club for now."
    )


# --- Stage 3: model-level feature preprocessing ---------------------------

def preprocess_lending_club(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 3 -- placeholder (scaling etc. can be added later)."""
    return X_train.copy(), X_test.copy()


# --- Convenience wrapper (stages 1 + 2) -----------------------------------

def load_lending_club(
    path: Path | None = None,
    nrows: int | None = 100_000,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load Lending Club loans. Target: default (1) vs. fully paid (0).

    Args:
        path: path to ``accepted_2007_to_2018Q4.csv`` or its parent directory.
        nrows: if set, load only the first ``nrows`` rows (the full file is
            ~2.2M rows). Pass ``None`` to load everything.
    """
    path = Path(path) if path else RAW_DIR / "lending_club" / "accepted_2007_to_2018Q4.csv"
    if path.is_dir():
        path = path / "accepted_2007_to_2018Q4.csv"

    df = pd.read_csv(path, nrows=nrows, low_memory=False)

    mask = df["loan_status"].isin(_LC_POSITIVE | _LC_NEGATIVE)
    df = df.loc[mask].copy()
    y = df["loan_status"].isin(_LC_POSITIVE).astype(int)

    df = df.drop(columns=["loan_status"] + _LC_LEAKY + _LC_DROP, errors="ignore")

    df = df.select_dtypes(include=[np.number])

    missing_frac = df.isna().mean()
    df = df.loc[:, missing_frac <= 0.3]

    if nrows is None and len(df) > 100_000:
        df = df.sample(n=100_000, random_state=random_state)
        y = y.loc[df.index]

    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return df, y
