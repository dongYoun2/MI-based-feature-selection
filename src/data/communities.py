"""Communities & Crime (UCI) -- regression on ViolentCrimesPerPop."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

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
    """Stage 1 -- read communities files and return a raw ``(X, y)`` pair."""
    raise NotImplementedError(
        "load_communities_raw is not yet split out; use load_communities for now."
    )


# --- Stage 2: dataset-level cleaning --------------------------------------

def clean_communities(
    X: pd.DataFrame,
    y: pd.Series,
    drop_high_missing: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stage 2 -- dataset-level cleaning for communities."""
    raise NotImplementedError(
        "clean_communities is not yet split out; use load_communities for now."
    )


# --- Stage 3: model-level feature preprocessing ---------------------------

def preprocess_communities(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 3 -- placeholder (scaling etc. can be added later)."""
    return X_train.copy(), X_test.copy()


# --- Convenience wrapper (stages 1 + 2) -----------------------------------

def load_communities(
    path: Path | None = None,
    drop_high_missing: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load Communities & Crime. Target: ViolentCrimesPerPop (regression).

    Args:
        path: directory containing ``communities.data`` and ``communities.names``.
        drop_high_missing: drop columns with a fraction of missing values
            above this threshold (defaults to 0.3, which removes the LEMAS
            block that is missing for most communities).
    """
    path = Path(path) if path else RAW_DIR / "communities"
    columns = _communities_columns(path / "communities.names")
    df = pd.read_csv(path / "communities.data", header=None, names=columns, na_values="?")

    df = df.drop(columns=_COMMUNITIES_NON_PREDICTIVE, errors="ignore")

    missing_frac = df.isna().mean()
    df = df.loc[:, missing_frac <= drop_high_missing]

    y = df[_COMMUNITIES_TARGET].astype(float)
    X = df.drop(columns=[_COMMUNITIES_TARGET])
    return X, y
