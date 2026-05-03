"""Entry point for running feature-selection experiments."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.experiment import run_experiment
from src.plotting import plot_metrics


def _svm_cartesian_pairs(
    c_list: list[float],
    gamma_tokens: list[str],
) -> list[tuple[float, float | str]]:
    """All (C, gamma) combinations; gamma token is scale, auto, or float."""
    out: list[tuple[float, float | str]] = []
    for c in c_list:
        for gstr in gamma_tokens:
            gs = gstr.strip()
            if gs in ("scale", "auto"):
                out.append((c, gs))
            else:
                out.append((c, float(gs)))
    return out


def _parse_svm_pairs(raw: list[str] | None) -> list[tuple[float, float | str]] | None:
    """Parse ``--svm-pair C,gamma`` strings (gamma may be scale, auto, or float)."""
    if not raw:
        return None
    out: list[tuple[float, float | str]] = []
    for item in raw:
        parts = item.split(",", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid --svm-pair {item!r}; use C,gamma e.g. 1,scale or 10,0.01"
            )
        c = float(parts[0].strip())
        gr = parts[1].strip()
        if gr in ("scale", "auto"):
            g: float | str = gr
        else:
            g = float(gr)
        out.append((c, g))
    return out

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
    if args.out_dir is None:
        args.out_dir = RUN_DIR
    args.ks = sorted({int(k) for k in args.ks if int(k) > 0})
    if not args.ks:
        parser.error("--ks must include at least one positive integer.")
    if args.svm_c_grid is not None and args.svm_gamma_grid is not None:
        if args.svm_pair:
            parser.error("use either --svm-pair or (--svm-c-grid with --svm-gamma-grid), not both")
        svm_pairs = _svm_cartesian_pairs(args.svm_c_grid, args.svm_gamma_grid)
    elif args.svm_c_grid is not None or args.svm_gamma_grid is not None:
        parser.error("--svm-c-grid and --svm-gamma-grid must be given together")
    else:
        try:
            svm_pairs = _parse_svm_pairs(args.svm_pair)
        except ValueError as exc:
            parser.error(str(exc))
    if svm_pairs and "svm" not in args.models:
        print("warning: --svm-pair ignored (svm not in --models)", file=sys.stderr)
        svm_pairs = None
    args.svm_pairs = svm_pairs
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
            svm_pairs=args.svm_pairs,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(summary_df)
    plot_metrics(summary_df, dataset=args.dataset, out_path=args.out_dir / f"{args.dataset}.png")


if __name__ == "__main__":
    main()
