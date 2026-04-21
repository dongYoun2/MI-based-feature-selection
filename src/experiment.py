"""Main experimental pipeline.

For each (dataset, feature selection method, model, k) combination:
    1. Load and split data.
    2. Select k features on the training set.
    3. Fit a predictive model on the selected features.
    4. Evaluate on the held-out test set.
    5. Record metrics.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .data import DatasetName, load_dataset
from .evaluate import evaluate_model
from .feature_selection import (
    correlation_filter,
    l1_selection,
    mrmr,
    rfe_selection,
)
from .models import ModelName, build_model

SelectorFn = Callable[..., list[int]]

SELECTORS: dict[str, SelectorFn] = {
    "mrmr": mrmr,
    "correlation": correlation_filter,
    "l1": l1_selection,
    "rfe": rfe_selection,
    # "cmi": cmi_selection,           # TODO
    # "pid": pid_selection,           # TODO
    # "shap": shap_selection,         # TODO
}


def run_single(
    dataset: DatasetName,
    selector: str,
    model_name: ModelName,
    k: int,
    random_state: int = 0,
) -> dict:
    ds = load_dataset(dataset, random_state=random_state)
    selector_fn = SELECTORS[selector]

    selected = selector_fn(ds.X_train, ds.y_train, k=k, task=ds.task)

    X_train_sel = ds.X_train[:, selected]
    X_test_sel = ds.X_test[:, selected]

    model = build_model(model_name, task=ds.task, random_state=random_state)
    model.fit(X_train_sel, ds.y_train)

    result = evaluate_model(model, X_test_sel, ds.y_test, ds.task, selected)
    return {
        "dataset": dataset,
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
        for selector in selectors:
            for model_name in models:
                for k in ks:
                    records.append(run_single(dataset, selector, model_name, k))

    df = pd.DataFrame(records)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df
