"""Plot metrics from a run: one figure per dataset.

Layout: rows = metrics (AUC/F1 for classification, RMSE/R2 for regression),
columns = models, lines within each subplot = feature selectors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METRICS_BY_TASK: dict[str, list[str]] = {
    "classification": ["auc", "f1"],
    "regression": ["rmse", "r2"],
}


def plot_metrics(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for dataset, ds_df in df.groupby("dataset"):
        task = ds_df["task"].iloc[0]
        metrics = [
            m
            for m in METRICS_BY_TASK[task]
            if m in ds_df.columns or f"{m}_mean" in ds_df.columns
        ]
        if not metrics:
            continue
        models = sorted(ds_df["model"].unique())
        selectors = sorted(ds_df["selector"].unique())

        fig, axes = plt.subplots(
            len(metrics), len(models),
            figsize=(5 * len(models), 3.5 * len(metrics)),
            sharex=True, squeeze=False,
        )
        for r, metric in enumerate(metrics):
            for c, model in enumerate(models):
                ax = axes[r, c]
                sub = ds_df[ds_df["model"] == model]
                for i, selector in enumerate(selectors):
                    line = sub[sub["selector"] == selector].sort_values("k")
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
                    ylab = f"{metric.upper()} (mean ± std)" if f"{metric}_mean" in ds_df.columns else metric.upper()
                    ax.set_ylabel(ylab)
                if r == len(metrics) - 1:
                    ax.set_xlabel("k")
                ax.grid(True, alpha=0.3)
        axes[0, -1].legend(loc="best", fontsize="small")
        fig.suptitle(f"{dataset} ({task})")
        fig.tight_layout()

        path = out_dir / f"{dataset}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    return saved
