"""CLI for the newly rerun formal no-cache baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_no_cache import generate_no_cache
from runner import run_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-lock", default="locks/experiment_v1.lock.json"
    )
    parser.add_argument("--output-dir", default="runs/no_cache")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parent
    run_condition(
        condition="no_cache",
        generator=generate_no_cache,
        experiment_lock_path=Path(args.experiment_lock),
        output_dir=Path(args.output_dir),
        condition_source_paths=[
            source_root / "generate_no_cache.py",
            source_root / "runner.py",
            source_root / "run_no_cache.py",
        ],
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

