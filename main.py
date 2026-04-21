"""Entry point for running feature-selection experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment import run_grid

RESULTS_DIR = Path(__file__).resolve().parent / "results"


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
        default=["mrmr", "correlation", "l1", "rfe"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "random_forest", "gradient_boosting"],
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 40])
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_grid(
        datasets=args.datasets,
        selectors=args.selectors,
        models=args.models,
        ks=args.ks,
        out_path=args.out,
    )
    print(df)


if __name__ == "__main__":
    main()
