"""Plot metrics from a single-dataset summary dataframe.

Layout: rows = metrics (AUC/Avg Precision for classification, RMSE/R2 for regression),
columns = models, lines within each subplot = feature selectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from src.models import ModelName

ALL_METRICS: list[str] = ["auc", "avg_precision", "rmse", "r2"]

# Stable colors across all figures: index i → f"C{i % 10}"
_SELECTOR_COLOR_ORDER: tuple[str, ...] = (
    "correlation",
    "l1",
    "rfe",
    "shap",
    "mi",
    "mrmr_heuristic",
    "cmim",
    "jmi",
    "pid",
)
SELECTOR_COLOR_INDEX: dict[str, int] = {s: i for i, s in enumerate(_SELECTOR_COLOR_ORDER)}
# Tab10 ``C8`` (olive) is low-contrast; use explicit hues where needed.
_SELECTOR_COLOR_OVERRIDES: dict[str, str] = {
    "pid": "#0d9488",
}

_TWO_PANEL_FIGSIZE = (10.5, 4.2)


def _metric_short_label(metric: str) -> str:
    return "Avg Precision" if metric == "avg_precision" else metric.upper()


def _metric_ylabel_with_band(metric: str) -> str:
    return f"{_metric_short_label(metric)} (mean ± std)"


def _stable_selector_palette(selectors: Sequence[str]) -> dict[str, str]:
    """Consistent selector → tab10 color; unknown names get indices after known ones."""
    sel_set = set(selectors)
    extras = sorted(sel_set - set(SELECTOR_COLOR_INDEX))
    extra_idx = {s: len(_SELECTOR_COLOR_ORDER) + i for i, s in enumerate(extras)}
    palette: dict[str, str] = {}
    for s in sel_set:
        if s in _SELECTOR_COLOR_OVERRIDES:
            palette[s] = _SELECTOR_COLOR_OVERRIDES[s]
            continue
        idx = SELECTOR_COLOR_INDEX[s] if s in SELECTOR_COLOR_INDEX else extra_idx[s]
        palette[s] = f"C{idx % 10}"
    return palette


def _accumulate_band_ylim_bounds(
    ymin: float | None,
    ymax: float | None,
    sub_df: pd.DataFrame,
    selectors: Sequence[str],
    metric: str,
    ks: Sequence[int] | None,
) -> tuple[float | None, float | None]:
    """Grow ``ymin``/``ymax`` to cover mean ± std bands for rows in ``sub_df``."""
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    for selector in selectors:
        line = sub_df.loc[sub_df["selector"] == selector].sort_values(by="k")
        if ks is not None:
            line = line[line["k"].isin(ks)]
        if len(line) == 0 or mean_col not in line.columns:
            continue
        y = line[mean_col].to_numpy(dtype=float)
        lower, upper = y, y
        if std_col in line.columns and line[std_col].notna().any():
            sem = line[std_col].fillna(0.0).to_numpy(dtype=float)
            lower, upper = y - sem, y + sem
        ln, ux = float(np.nanmin(lower)), float(np.nanmax(upper))
        ymin = ln if ymin is None else min(ymin, ln)
        ymax = ux if ymax is None else max(ymax, ux)
    return ymin, ymax


def _accumulate_band_ylim_over_specs(
    specs: Sequence[tuple[pd.DataFrame, Sequence[str]]],
    metric: str,
    ks: Sequence[int] | None,
) -> tuple[float | None, float | None]:
    ymin = ymax = None
    for sub_df, selectors in specs:
        ymin, ymax = _accumulate_band_ylim_bounds(ymin, ymax, sub_df, selectors, metric, ks)
    return ymin, ymax


def _set_primary_panel_padded_ylim(
    ax_primary: Axes, ymin: float | None, ymax: float | None,
) -> None:
    """Apply shared padded y-axis (for ``sharey`` row)."""
    if ymin is None or ymax is None:
        return
    span = max(ymax - ymin, 1e-6)
    pad = 0.05 * span
    ax_primary.set_ylim(ymin - pad, ymax + pad)


def _style_metric_comparison_panel(ax: Axes, metric: str, *, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel(_metric_ylabel_with_band(metric))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small", framealpha=0.9)


def _save_metrics_figure(fig: Figure, save_path: Path | None) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def _plot_metric_curves_on_ax(
    ax: Axes,
    sub_df: pd.DataFrame,
    selectors: Sequence[str],
    metric: str,
    *,
    ks: Sequence[int] | None = None,
    color_offset: int = 0,
    selector_colors: dict[str, str] | None = None,
    selector_labels: dict[str, str] | None = None,
) -> None:
    """Lines + mean±std ribbons for ``selectors`` (used by ``plot_metrics`` / two-panel helper)."""
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    for i, selector in enumerate(selectors):
        line = sub_df.loc[sub_df["selector"] == selector].sort_values(by="k")
        if ks is not None:
            line = line[line["k"].isin(ks)]
        if len(line) == 0:
            continue

        if selector_colors is not None:
            color = selector_colors[selector]
        else:
            color = f"C{(color_offset + i) % 10}"

        legend = (selector_labels or {}).get(selector, selector)

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
                    alpha=0.1,
                    linewidth=0,
                )
            ax.plot(kvals, y, marker="o", color=color, label=legend)
        else:
            ax.plot(
                line["k"],
                line[metric],
                marker="o",
                color=color,
                label=legend,
            )


def plot_metrics(
    df: pd.DataFrame,
    dataset: str,
    save_path: Path | None = None,
    selector_labels: dict[str, str] | None = None,
    models: Sequence[ModelName] | None = None,
    metrics: Sequence[str] | None = None,
) -> Figure | None:
    """Save a single ``<dataset>.png`` figure to ``save_path``.

    ``df`` is expected to contain rows for one dataset only (the summary df
    produced by :func:`src.experiment.run_experiment`). ``dataset`` is the
    label used in the figure title. ``selector_labels`` optionally maps
    canonical selector names to display labels without changing selector colors.
    """
    if not metrics:
        metrics = [
            m
            for m in ALL_METRICS
            if m in df.columns or f"{m}_mean" in df.columns
        ]
        assert metrics

    if not models:
        models = sorted(df["model"].unique())

    selectors = sorted(df["selector"].unique())
    palette = _stable_selector_palette(selectors)

    fig, axes = plt.subplots(
        len(metrics), len(models),
        figsize=(5 * len(models), 3.5 * len(metrics)),
        sharex=True, squeeze=False,
    )
    for r, metric in enumerate(metrics):
        for c, model in enumerate(models):
            ax = axes[r, c]
            sub = df.loc[df["model"] == model]
            _plot_metric_curves_on_ax(
                ax,
                sub,
                selectors,
                metric,
                selector_colors=palette,
                selector_labels=selector_labels,
            )
            if r == 0:
                ax.set_title(model)
            if c == 0:
                label = _metric_short_label(metric)
                ylab = f"{label} (mean ± std)" if f"{metric}_mean" in df.columns else label
                ax.set_ylabel(ylab)
            if r == len(metrics) - 1:
                ax.set_xlabel("k")
            ax.grid(True, alpha=0.3)

    # fig.suptitle(dataset)
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 0.96))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        fontsize="small",
        frameon=True,
        borderaxespad=0,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metric_two_selector_groups(
    df: pd.DataFrame,
    *,
    metric: str,
    selectors_left: Sequence[str],
    selectors_right: Sequence[str],
    title_left: str,
    title_right: str,
    model: str = "logreg",
    ks: Sequence[int] | None = None,
    save_path: Path | None = None,
    selector_labels: dict[str, str] | None = None,
) -> Figure:
    """Two side-by-side line plots comparing selector subsets for one ``model``.

    Shows ``metric`` mean with ± std ribbons (same style as :func:`plot_metrics`).
    Y limits are matched across panels using padded min/max of mean ± std.

    ``selector_labels`` maps canonical ``selector`` column values (e.g. ``"l1"``)
    to legend display strings; keys omitted keep the canonical name.
    """
    sub = df.loc[df["model"] == model]
    selectors_order = list(selectors_left) + [
        s for s in selectors_right if s not in selectors_left
    ]
    palette = _stable_selector_palette(selectors_order)
    ymin, ymax = _accumulate_band_ylim_bounds(
        None, None, sub, selectors_order, metric, ks,
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=_TWO_PANEL_FIGSIZE, sharey=True)
    for ax, sels, tit in (
        (ax0, selectors_left, title_left),
        (ax1, selectors_right, title_right),
    ):
        _plot_metric_curves_on_ax(
            ax,
            sub,
            sels,
            metric,
            ks=ks,
            selector_colors=palette,
            selector_labels=selector_labels,
        )
        _style_metric_comparison_panel(ax, metric, title=tit)

    _set_primary_panel_padded_ylim(ax0, ymin, ymax)
    fig.tight_layout()
    _save_metrics_figure(fig, save_path)
    return fig


def plot_metric_two_models_same_selectors(
    df: pd.DataFrame,
    *,
    metric: str,
    selectors: Sequence[str],
    model_left: str,
    model_right: str,
    title_left: str,
    title_right: str,
    ks: Sequence[int] | None = None,
    save_path: Path | None = None,
    selector_labels: dict[str, str] | None = None,
) -> Figure:
    """Two side-by-side panels: same ``selectors``, different ``model`` (shared y-axis)."""
    sub_l = df.loc[df["model"] == model_left]
    sub_r = df.loc[df["model"] == model_right]
    palette = _stable_selector_palette(selectors)
    ymin, ymax = _accumulate_band_ylim_over_specs(
        [(sub_l, selectors), (sub_r, selectors)], metric, ks,
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=_TWO_PANEL_FIGSIZE, sharey=True)
    _plot_metric_curves_on_ax(
        ax0, sub_l, selectors, metric, ks=ks, selector_colors=palette, selector_labels=selector_labels,
    )
    _plot_metric_curves_on_ax(
        ax1, sub_r, selectors, metric, ks=ks, selector_colors=palette, selector_labels=selector_labels,
    )
    _style_metric_comparison_panel(ax0, metric, title=title_left)
    _style_metric_comparison_panel(ax1, metric, title=title_right)

    _set_primary_panel_padded_ylim(ax0, ymin, ymax)
    fig.tight_layout()
    _save_metrics_figure(fig, save_path)
    return fig
