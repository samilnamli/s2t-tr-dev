"""CLI entry: ``python -m src.data.prepare --dataset {ami|voxpopuli|synthetic}``.

Instantiates the matching DataModule with default parameters and calls
``prepare_data()``. Useful when you want to materialize the parquet
explicitly rather than letting the experiment do it on first run.
"""

from __future__ import annotations

import argparse

from src.utils.logging import setup_unified_logging


def main() -> None:
    setup_unified_logging(level="INFO")

    parser = argparse.ArgumentParser(description="Materialize a dataset's parquet.")
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=["ami", "voxpopuli", "synthetic"],
    )
    parser.add_argument("--parquet-path", default=None, help="Override default path.")
    args = parser.parse_args()

    kwargs = {}
    if args.parquet_path:
        kwargs["parquet_path"] = args.parquet_path

    if args.dataset == "ami":
        from src.data.ami import AMIDataModule

        dm = AMIDataModule(**kwargs)
    elif args.dataset == "voxpopuli":
        from src.data.voxpopuli import VoxPopuliDataModule

        dm = VoxPopuliDataModule(**kwargs)
    else:
        from src.data.synthetic import SyntheticDataModule

        dm = SyntheticDataModule(**kwargs)

    dm.prepare_data()


if __name__ == "__main__":
    main()
