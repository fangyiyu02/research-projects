"""CLI for the cache condition after its implementation is frozen."""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_dllm_cache import CACHE_IMPLEMENTED, generate_dllm_cache
from runner import run_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-lock", default="locks/experiment_v1.lock.json"
    )
    parser.add_argument("--cache-lock", default="locks/cache_v1.lock.json")
    parser.add_argument("--output-dir", default="runs/dllm_cache")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not CACHE_IMPLEMENTED:
        raise RuntimeError(
            "dLLM-Cache is not implemented. Complete generate_dllm_cache.py "
            "and freeze cache_v1 before running this condition."
        )
    source_root = Path(__file__).resolve().parent
    run_condition(
        condition="dllm_cache",
        generator=generate_dllm_cache,
        experiment_lock_path=Path(args.experiment_lock),
        condition_lock_path=Path(args.cache_lock),
        output_dir=Path(args.output_dir),
        condition_source_paths=[
            source_root / "generate_dllm_cache.py",
            source_root / "runner.py",
            source_root / "run_dllm_cache.py",
        ],
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

