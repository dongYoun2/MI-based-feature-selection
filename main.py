"""Entry point for running feature-selection experiments."""

from __future__ import annotations

import argparse
import sys

from src.cli import add_experiment_args, normalize_experiment_args
from src.experiment import run_experiment
from src.plotting import plot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MI-based feature selection experiments")
    add_experiment_args(parser)
    args = parser.parse_args()
    normalize_experiment_args(parser, args)
    return args


def main() -> None:
    args = parse_args()
    try:
        per_fold_df, summary_df = run_experiment(
            dataset=args.dataset,
            selectors=args.selectors,
            model_specs=[(m, {}) for m in args.models],
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
