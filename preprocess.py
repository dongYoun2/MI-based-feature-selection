"""Cache stages 1+2 of a dataset to ``data/processed/<name>/<name>.csv``.

Stage 3 (log1p, one-hot, standard-scale) stays per-fold inside the
experiment to avoid train/val leakage; this script only caches the
expensive raw load + dataset-level cleaning. Once the cache exists,
``main.py`` (and any other caller of :func:`src.data.prepare_xy`) will
load it transparently and ignore ``loader_kwargs``.

Examples:
    python preprocess.py --dataset lending_club --nrows 5000
    python preprocess.py --dataset nhanes
    python preprocess.py --dataset communities
"""

from __future__ import annotations

import argparse

from src.data import make_preprocessed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["communities", "lending_club", "nhanes"],
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="lending_club only: random subsample size (default: loader's default).",
    )
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader_kwargs: dict = {"random_state": args.random_state}
    if args.nrows is not None:
        loader_kwargs["nrows"] = args.nrows
    out_path = make_preprocessed(args.dataset, **loader_kwargs)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
