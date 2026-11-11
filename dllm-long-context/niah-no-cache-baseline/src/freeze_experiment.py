"""Validate the new benchmark artifacts and create one immutable experiment lock."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    expected_prompt_count,
    iter_jsonl,
    project_root_from_config,
    read_json,
    relative_to_project,
    resolve_project_path,
    sha256_file,
    sha256_input_ids,
    sha256_text,
    utc_now,
    validate_config,
    write_json_atomic,
)


FROZEN_SOURCE_FILES = [
    "src/common.py",
    "src/prepare_experiment.py",
    "src/freeze_experiment.py",
    "src/generate_no_cache.py",
    "src/runner.py",
    "src/run_no_cache.py",
    "src/validate_results.py",
    "src/plot_results.py",
]


def validate_prompt_manifest(
    config: dict[str, Any], manifest_path: Path, haystack_sha256: str
) -> tuple[int, int]:
    seen: set[str] = set()
    run_indices: set[int] = set()
    cell_counts: Counter[tuple[int, float]] = Counter()
    max_shortfall = int(config["grid"]["maximum_length_shortfall_tokens"])
    count = 0
    for record in iter_jsonl(manifest_path):
        sample_id = record["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        run_indices.add(int(record["run_order_index"]))
        if record["experiment_id"] != config["experiment_id"]:
            raise ValueError(f"Experiment ID mismatch for {sample_id}")
        if sha256_input_ids(record["input_ids"]) != record["input_ids_sha256"]:
            raise ValueError(f"Input-ID hash mismatch for {sample_id}")
        if sha256_text(record["prompt_text"]) != record["prompt_text_sha256"]:
            raise ValueError(f"Prompt-text hash mismatch for {sample_id}")
        if record["needle_text"] not in record["prompt_text"]:
            raise ValueError(f"Needle text missing from full prompt for {sample_id}")
        if record["question_text"] not in record["prompt_text"]:
            raise ValueError(f"Question text missing from full prompt for {sample_id}")
        if record["haystack_sha256"] != haystack_sha256:
            raise ValueError(f"Haystack hash mismatch for {sample_id}")
        for record_key, config_section, config_key in (
            ("model_revision", "model", "revision"),
            ("model_code_revision", "model", "code_revision"),
            ("tokenizer_revision", "tokenizer", "revision"),
            ("tokenizer_code_revision", "tokenizer", "code_revision"),
        ):
            if record[record_key] != config[config_section][config_key]:
                raise ValueError(f"Revision mismatch for {sample_id}: {record_key}")
        actual_total = len(record["input_ids"]) + int(config["generation"]["gen_length"])
        if int(record["actual_total_len"]) != actual_total:
            raise ValueError(f"Actual total length mismatch for {sample_id}")
        shortfall = int(record["target_total_len"]) - actual_total
        if int(record["length_shortfall_tokens"]) != shortfall:
            raise ValueError(f"Length shortfall mismatch for {sample_id}")
        if not 0 <= shortfall <= max_shortfall:
            raise ValueError(f"Length shortfall outside frozen tolerance for {sample_id}")
        needle_start = int(record["needle_token_start"])
        needle_end = int(record["needle_token_end_exclusive"])
        question_start = int(record["question_token_start"])
        question_end = int(record["question_token_end_exclusive"])
        prompt_len = len(record["input_ids"])
        if not 0 <= needle_start < needle_end <= question_start < question_end <= prompt_len:
            raise ValueError(f"Invalid exact token spans for {sample_id}")
        if int(record["needle_to_question_start_distance"]) != question_start - needle_start:
            raise ValueError(f"Needle distance mismatch for {sample_id}")
        cell_counts[(int(record["target_total_len"]), float(record["requested_depth"]))] += 1
        count += 1

    expected_count = expected_prompt_count(config)
    if count != expected_count:
        raise ValueError(f"Expected {expected_count} prompts, found {count}")
    if run_indices != set(range(expected_count)):
        raise ValueError("run_order_index is not a complete permutation")
    grid = config["grid"]
    expected_cells = len(grid["target_total_lengths"]) * len(grid["depths"])
    expected_per_cell = int(grid["n_samples_per_cell"])
    if len(cell_counts) != expected_cells or set(cell_counts.values()) != {expected_per_cell}:
        raise ValueError("Prompt manifest is not the complete balanced grid")
    return count, len(cell_counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment_v1.json")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    validate_config(config)
    project_root = project_root_from_config(config_path)
    haystack_path = resolve_project_path(
        project_root, config["fresh_haystack"]["relative_path"]
    )
    haystack_meta_path = resolve_project_path(
        project_root, config["fresh_haystack"]["metadata_relative_path"]
    )
    prompt_path = resolve_project_path(
        project_root, config["artifacts"]["prompt_manifest_relative_path"]
    )
    prompt_meta_path = resolve_project_path(
        project_root, config["artifacts"]["prompt_metadata_relative_path"]
    )
    lock_path = resolve_project_path(
        project_root, config["artifacts"]["experiment_lock_relative_path"]
    )
    if lock_path.exists():
        raise FileExistsError("Experiment lock already exists; create a v2 instead")
    for path in (haystack_path, haystack_meta_path, prompt_path, prompt_meta_path):
        if not path.is_file():
            raise FileNotFoundError(f"Run prepare_experiment.py first: missing {path}")

    haystack_sha256 = sha256_file(haystack_path)
    haystack_meta = read_json(haystack_meta_path)
    if haystack_meta["sha256"] != haystack_sha256:
        raise ValueError("Haystack metadata/hash mismatch")
    if int(haystack_meta["seed"]) != int(config["fresh_haystack"]["seed"]):
        raise ValueError("Haystack seed differs from config")
    if int(haystack_meta["generator_version"]) != int(
        config["fresh_haystack"]["generator_version"]
    ):
        raise ValueError("Haystack generator version differs from config")

    prompt_count, cell_count = validate_prompt_manifest(
        config, prompt_path, haystack_sha256
    )
    prompt_meta = read_json(prompt_meta_path)
    if prompt_meta["manifest_sha256"] != sha256_file(prompt_path):
        raise ValueError("Prompt metadata/manifest hash mismatch")
    if prompt_meta["config_sha256"] != sha256_file(config_path):
        raise ValueError("Prompt metadata/config hash mismatch")
    if prompt_meta["haystack_sha256"] != haystack_sha256:
        raise ValueError("Prompt metadata/haystack hash mismatch")

    role_paths: list[tuple[str, Path]] = [
        ("experiment_config", config_path),
        ("haystack", haystack_path),
        ("haystack_metadata", haystack_meta_path),
        ("prompt_manifest", prompt_path),
        ("prompt_metadata", prompt_meta_path),
        ("environment_spec", project_root / "requirements.txt"),
        ("experiment_protocol", project_root / "docs" / "EXPERIMENT_PROTOCOL.md"),
    ]
    role_paths.extend(
        ("baseline_source", resolve_project_path(project_root, relative_path))
        for relative_path in FROZEN_SOURCE_FILES
    )
    files: list[dict[str, Any]] = []
    for role, path in role_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen file missing: {path}")
        files.append(
            {
                "role": role,
                "relative_path": relative_to_project(project_root, path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    lock = {
        "schema_version": 1,
        "lock_kind": "experiment",
        "experiment_id": config["experiment_id"],
        "frozen_at_utc": utc_now(),
        "prompt_count": prompt_count,
        "cell_count": cell_count,
        "model_revision": config["model"]["revision"],
        "model_code_revision": config["model"]["code_revision"],
        "tokenizer_revision": config["tokenizer"]["revision"],
        "tokenizer_code_revision": config["tokenizer"]["code_revision"],
        "files": files,
    }
    write_json_atomic(lock_path, lock)
    print(f"Fresh experiment frozen: {lock_path}")
    print(f"Experiment lock SHA-256: {sha256_file(lock_path)}")


if __name__ == "__main__":
    main()

