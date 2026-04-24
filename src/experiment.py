"""Main experimental pipeline.

For each (dataset, feature selection method, model, k) combination:
    1. Load and split data.
    2. Select k features on the training set.
    3. Fit a predictive model on the selected features.
    4. Evaluate on the held-out test set.
    5. Record metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable
from functools import lru_cache

import pandas as pd

from .data import Dataset, DatasetName, load_dataset
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

    # shapes: (n, d) -> (n, k)
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


def run_grid(
    datasets: Iterable[DatasetName],
    selectors: Iterable[str],
    models: Iterable[ModelName],
    ks: Iterable[int],
    out_path: Path | None = None,
) -> pd.DataFrame:
    records = []
    for dataset in datasets:
        ds = load_dataset(dataset)
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
                    records.append(run_single(ds, selector, model_name, k))

    df = pd.DataFrame(records)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df