"""Train-only median imputation for tabular splits (avoids leakage across folds)."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer


def median_impute_train(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit ``SimpleImputer(strategy='median')`` on ``X_train``; transform both frames."""
    cols = X_train.columns
    imp = SimpleImputer(strategy="median")
    Xt = pd.DataFrame(
        imp.fit_transform(X_train),
        columns=cols,
        index=X_train.index,
    )
    Xo = pd.DataFrame(
        imp.transform(X_other),
        columns=cols,
        index=X_other.index,
    )
    return Xt, Xo
