"""Entry point for running feature-selection experiments."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.experiment import run_experiment
from src.plotting import plot_metrics

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUN_DIR = RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MI-based feature selection experiments")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["communities", "lending_club", "nhanes"],
        help="Dataset to run the experiment grid on.",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["mrmr", "mrmr_heuristic", "mi", "correlation", "l1", "rfe", "shap"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "random_forest", "gradient_boosting", "svm"],
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for CSVs and plots. Defaults to results/<timestamp>/.",
    )
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = RUN_DIR
    args.ks = sorted({int(k) for k in args.ks if int(k) > 0})
    if not args.ks:
        parser.error("--ks must include at least one positive integer.")
    return args


def main() -> None:
    args = parse_args()
    try:
        per_fold_df, summary_df = run_experiment(
            dataset=args.dataset,
            selectors=args.selectors,
            models=args.models,
            ks=args.ks,
            out_dir=args.out_dir,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(summary_df)
    plot_metrics(summary_df, dataset=args.dataset, out_path=args.out_dir / f"{args.dataset}.png")


if __name__ == "__main__":
    main()
