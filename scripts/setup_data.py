"""CLI helper to download datasets and evaluation tools."""

from __future__ import annotations

import argparse

from cell_tracking.data import ensure_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Download challenge assets")
    parser.add_argument("dataset", help="Dataset identifier (e.g., Fluo-N2DH-GOWT1)")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["training", "test"],
        help="Dataset splits to fetch",
    )
    args = parser.parse_args()
    ensure_all(args.dataset, args.splits)


if __name__ == "__main__":
    main()
