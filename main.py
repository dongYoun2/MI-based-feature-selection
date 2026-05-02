"""Entry point for running feature-selection experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.experiment import run_grid
from src.plotting import plot_metrics

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUN_DIR = RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MI-based feature selection experiments")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["communities", "lending_club", "nhanes"],
        choices=["communities", "lending_club", "nhanes"],
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["mrmr", "mrmr_heuristic", "mi", "correlation", "l1", "rfe", "shap"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "random_forest", "gradient_boosting"],
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 40])
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Stratified K-fold (classification) or K-fold (regression). "
        "Median imputation and stage-3 preprocess fit on each training fold only. "
        "Use 0 or 1 for a single train/test split (no CV).",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RUN_DIR / "metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_grid(
        datasets=args.datasets,
        selectors=args.selectors,
        models=args.models,
        ks=args.ks,
        out_path=args.out,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    print(df)
    plot_metrics(df, out_dir=args.out.parent)


if __name__ == "__main__":
    main()
