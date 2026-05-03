"""Plot metrics from a single-dataset summary dataframe.

Layout: rows = metrics (AUC/Avg Precision for classification, RMSE/R2 for regression),
columns = models, lines within each subplot = feature selectors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ALL_METRICS: list[str] = ["auc", "avg_precision", "rmse", "r2"]


def plot_metrics(df: pd.DataFrame, dataset: str, out_path: Path | None = None) -> None:
    """Save a single ``<dataset>.png`` figure to ``out_path``.

    ``df`` is expected to contain rows for one dataset only (the summary df
    produced by :func:`src.experiment.run_experiment`). ``dataset`` is the
    label used in the figure title.
    """
    metrics = [
        m
        for m in ALL_METRICS
        if m in df.columns or f"{m}_mean" in df.columns
    ]
    if not metrics:
        return None

    models = sorted(df["model"].unique())
    selectors = sorted(df["selector"].unique())

    fig, axes = plt.subplots(
        len(metrics), len(models),
        figsize=(5 * len(models), 3.5 * len(metrics)),
        sharex=True, squeeze=False,
    )
    for r, metric in enumerate(metrics):
        for c, model in enumerate(models):
            ax = axes[r, c]
            sub = df[df["model"] == model]
            for i, selector in enumerate(selectors):
                line = sub.loc[sub["selector"] == selector].sort_values(by="k")
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"
                color = f"C{i % 10}"
                if mean_col in line.columns:
                    y = line[mean_col]
                    kvals = line["k"]
                    if std_col in line.columns and line[std_col].notna().any():
                        std = line[std_col].fillna(0.0)
                        ax.fill_between(
                            kvals,
                            y - std,
                            y + std,
                            color=color,
                            alpha=0.15,
                            linewidth=0,
                        )
                    ax.plot(kvals, y, marker="o", color=color, label=selector)
                else:
                    ax.plot(
                        line["k"],
                        line[metric],
                        marker="o",
                        color=color,
                        label=selector,
                    )
            if r == 0:
                ax.set_title(model)
            if c == 0:
                label = "Avg Precision" if metric == "avg_precision" else metric.upper()
                ylab = f"{label} (mean ± std)" if f"{metric}_mean" in df.columns else label
                ax.set_ylabel(ylab)
            if r == len(metrics) - 1:
                ax.set_xlabel("k")
            ax.grid(True, alpha=0.3)
    axes[0, -1].legend(loc="best", fontsize="small")
    fig.suptitle(dataset)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150)

    # plt.close(fig)
