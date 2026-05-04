"""Run feature-selection experiments over an SVM (C, gamma) grid."""

from __future__ import annotations

import argparse
import sys

from src.cli import add_experiment_args, normalize_experiment_args
from src.experiment import run_experiment
from src.plotting import plot_metrics
from src.svm_grid import expand_models_with_svm, resolve_svm_pairs_from_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MI-based feature selection with SVM hyperparameter grid",
    )
    add_experiment_args(parser)
    parser.add_argument(
        "--svm-pair",
        action="append",
        default=None,
        metavar="C,GAMMA",
        help=(
            "Repeat for each SVM (C, gamma) pair when svm is in --models. "
            "gamma is scale, auto, or a float. Example: --svm-pair 1,scale --svm-pair 10,0.01"
        ),
    )
    parser.add_argument(
        "--svm-c-grid",
        nargs="+",
        type=float,
        default=None,
        metavar="C",
        help="With --svm-gamma-grid: run Cartesian product of C × gamma (mutually exclusive with --svm-pair).",
    )
    parser.add_argument(
        "--svm-gamma-grid",
        nargs="+",
        type=str,
        default=None,
        metavar="GAMMA",
        help="With --svm-c-grid: each token is scale, auto, or a float string (e.g. 0.01).",
    )
    args = parser.parse_args()
    normalize_experiment_args(parser, args)
    try:
        svm_pairs = resolve_svm_pairs_from_cli(args.svm_pair, args.svm_c_grid, args.svm_gamma_grid)
    except ValueError as exc:
        parser.error(str(exc))
    if svm_pairs and "svm" not in args.models:
        print("warning: SVM grid options ignored (svm not in --models)", file=sys.stderr)
        svm_pairs = None
    args.model_specs = expand_models_with_svm(args.models, svm_pairs)
    return args


def main() -> None:
    args = parse_args()
    try:
        per_fold_df, summary_df = run_experiment(
            dataset=args.dataset,
            selectors=args.selectors,
            model_specs=args.model_specs,
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
