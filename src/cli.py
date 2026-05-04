"""Shared argparse helpers for experiment scripts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["communities", "lending_club", "nhanes"],
        help="Dataset to run the experiment grid on.",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["correlation", "l1", "rfe", "shap", "mi", "mrmr_heuristic", "pid", "cmim", "jmi"],
   
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "gradient_boosting", "svm"],
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


def normalize_experiment_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.out_dir is None:
        args.out_dir = RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.ks = sorted({int(k) for k in args.ks if int(k) > 0})
    if not args.ks:
        parser.error("--ks must include at least one positive integer.")
