"""Freeze a completed cache implementation against the existing experiment."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

from common import (
    read_json,
    relative_to_project,
    sha256_file,
    utc_now,
    verify_file_lock,
    write_json_atomic,
)


def contains_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(contains_null(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_null(item) for item in value)
    return False


def cache_flag_is_true(source_path: Path) -> bool:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "CACHE_IMPLEMENTED" in names:
                return isinstance(node.value, ast.Constant) and node.value.value is True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-lock", default="locks/experiment_v1.lock.json"
    )
    parser.add_argument("--cache-config", default="config/cache_v1.json")
    parser.add_argument("--output", default="locks/cache_v1.lock.json")
    args = parser.parse_args()

    experiment_lock_path = Path(args.experiment_lock).resolve()
    project_root, experiment_lock, experiment_lock_sha256 = verify_file_lock(
        experiment_lock_path, "experiment"
    )
    cache_config_path = Path(args.cache_config).resolve()
    try:
        cache_config_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Cache config must remain inside the experiment project") from exc
    cache_config = read_json(cache_config_path)
    if cache_config.get("schema_version") != 1:
        raise ValueError("Unsupported cache config schema_version")
    if cache_config.get("status") != "ready":
        raise ValueError("Cache config status must be 'ready'")
    if cache_config.get("algorithm") != "dllm_cache":
        raise ValueError("Cache algorithm must be explicitly named 'dllm_cache'")
    if contains_null(cache_config.get("parameters")):
        raise ValueError("Cache parameters still contain null placeholders")

    source_paths = [
        project_root / "src" / "generate_dllm_cache.py",
        project_root / "src" / "run_dllm_cache.py",
        project_root / "src" / "freeze_cache_condition.py",
    ]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not cache_flag_is_true(source_paths[0]):
        raise ValueError("CACHE_IMPLEMENTED is not set to literal True")

    output_path = Path(args.output).resolve()
    try:
        output_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Condition lock must remain inside the experiment project") from exc
    if output_path.exists():
        raise FileExistsError("Cache condition lock already exists; create a new version")

    role_paths = [("condition_config", cache_config_path)]
    role_paths.extend(("condition_source", path) for path in source_paths)
    files = [
        {
            "role": role,
            "relative_path": relative_to_project(project_root, path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for role, path in role_paths
    ]
    condition_lock = {
        "schema_version": 1,
        "lock_kind": "condition",
        "condition": "dllm_cache",
        "condition_id": cache_config["condition_id"],
        "frozen_at_utc": utc_now(),
        "experiment_id": experiment_lock["experiment_id"],
        "experiment_lock_relative_path": relative_to_project(
            project_root, experiment_lock_path
        ),
        "experiment_lock_sha256": experiment_lock_sha256,
        "files": files,
    }
    write_json_atomic(output_path, condition_lock)
    print(f"Cache condition frozen: {output_path}")
    print(f"Cache condition lock SHA-256: {sha256_file(output_path)}")


if __name__ == "__main__":
    main()

