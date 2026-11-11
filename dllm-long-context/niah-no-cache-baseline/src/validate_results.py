"""Validate a complete fresh no-cache or dLLM-Cache result file."""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    iter_jsonl,
    locked_path,
    read_json,
    score_output,
    sha256_file,
    sha256_input_ids,
    sha256_named_files,
    utc_now,
    verify_file_lock,
    write_json_atomic,
)


SHA256_RE = re.compile(r"[0-9a-f]{64}")


def resolve_argument(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["no_cache", "dllm_cache"])
    parser.add_argument(
        "--experiment-lock", default="locks/experiment_v1.lock.json"
    )
    parser.add_argument("--condition-lock", default=None)
    parser.add_argument("--results", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    experiment_lock_path = Path(args.experiment_lock).resolve()
    project_root, experiment_lock, experiment_lock_sha256 = verify_file_lock(
        experiment_lock_path, "experiment"
    )
    config_path = locked_path(experiment_lock, project_root, "experiment_config")
    prompt_path = locked_path(experiment_lock, project_root, "prompt_manifest")
    config = read_json(config_path)
    config_sha256 = sha256_file(config_path)
    prompts = list(iter_jsonl(prompt_path))
    prompt_by_id = {prompt["sample_id"]: prompt for prompt in prompts}

    condition_lock_sha256: str | None = None
    if args.condition == "dllm_cache":
        if not args.condition_lock:
            raise ValueError("dllm_cache validation requires --condition-lock")
        condition_root, condition_lock, condition_lock_sha256 = verify_file_lock(
            Path(args.condition_lock).resolve(), "condition"
        )
        if condition_root != project_root:
            raise ValueError("Condition lock belongs to a different project")
        if condition_lock["experiment_lock_sha256"] != experiment_lock_sha256:
            raise ValueError("Condition lock points to a different experiment lock")
    elif args.condition_lock:
        raise ValueError("no_cache validation must not receive a condition lock")

    source_root = project_root / "src"
    expected_source_bundle_sha256 = sha256_named_files(
        [
            source_root
            / (
                "generate_no_cache.py"
                if args.condition == "no_cache"
                else "generate_dllm_cache.py"
            ),
            source_root / "runner.py",
            source_root
            / ("run_no_cache.py" if args.condition == "no_cache" else "run_dllm_cache.py"),
        ]
    )

    results_path = resolve_argument(project_root, args.results)
    report_path = resolve_argument(project_root, args.report)
    results = list(iter_jsonl(results_path)) if results_path.is_file() else []
    errors: list[str] = []
    seen: set[str] = set()
    valid_results: list[dict[str, Any]] = []

    if len(prompt_by_id) != len(prompts):
        errors.append("Frozen prompt manifest contains duplicate sample IDs")

    for line_index, result in enumerate(results, start=1):
        row_error_start = len(errors)
        sample_id = result.get("sample_id")
        prefix = f"line {line_index} ({sample_id!r})"
        if not isinstance(sample_id, str):
            errors.append(f"{prefix}: sample_id is missing or not a string")
            continue
        if sample_id in seen:
            errors.append(f"{prefix}: duplicate sample_id")
            continue
        seen.add(sample_id)
        prompt = prompt_by_id.get(sample_id)
        if prompt is None:
            errors.append(f"{prefix}: sample_id is not in the frozen manifest")
            continue

        expected_identity = {
            "experiment_id": config["experiment_id"],
            "condition": args.condition,
            "experiment_lock_sha256": experiment_lock_sha256,
            "condition_lock_sha256": condition_lock_sha256,
            "condition_source_bundle_sha256": expected_source_bundle_sha256,
            "config_sha256": config_sha256,
            "input_ids_sha256": prompt["input_ids_sha256"],
            "prompt_text_sha256": prompt["prompt_text_sha256"],
            "model_revision": config["model"]["revision"],
            "model_code_revision": config["model"]["code_revision"],
            "tokenizer_revision": config["tokenizer"]["revision"],
            "tokenizer_code_revision": config["tokenizer"]["code_revision"],
        }
        for key, expected in expected_identity.items():
            if result.get(key) != expected:
                errors.append(f"{prefix}: identity field {key!r} mismatch")
        if sha256_input_ids(prompt["input_ids"]) != prompt["input_ids_sha256"]:
            errors.append(f"{prefix}: frozen prompt input-ID hash is inconsistent")

        prompt_fields = (
            "run_order_index",
            "sample_seed",
            "target_total_len",
            "actual_total_len",
            "prompt_token_length",
            "requested_depth",
            "sample_idx",
            "needle_token_start",
            "needle_token_end_exclusive",
            "question_token_start",
            "needle_to_question_start_distance",
            "needle_end_to_question_gap",
            "needle_prompt_position_ratio",
            "city",
            "needle_number",
        )
        for key in prompt_fields:
            if result.get(key) != prompt.get(key):
                errors.append(f"{prefix}: prompt field {key!r} mismatch")

        expected_scores = score_output(
            str(result.get("model_output", "")), prompt["needle_number"]
        )
        for key, expected in expected_scores.items():
            if result.get(key) != expected:
                errors.append(f"{prefix}: score field {key!r} is inconsistent")

        for key in (
            "latency_seconds",
            "generated_tokens_per_second",
            "peak_memory_allocated_bytes",
            "peak_memory_reserved_bytes",
        ):
            value = result.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"{prefix}: {key!r} must be a positive number")
        source_hash = result.get("condition_source_bundle_sha256")
        if not isinstance(source_hash, str) or not SHA256_RE.fullmatch(source_hash):
            errors.append(f"{prefix}: invalid condition source-bundle hash")

        condition_metrics = result.get("condition_metrics")
        if not isinstance(condition_metrics, dict):
            errors.append(f"{prefix}: condition_metrics must be an object")
        elif args.condition == "no_cache":
            if condition_metrics.get("cache_enabled") is not False:
                errors.append(f"{prefix}: no-cache result does not declare cache_enabled=false")
            if condition_metrics.get("full_sequence_model_forwards") != int(
                config["generation"]["steps"]
            ):
                errors.append(f"{prefix}: no-cache full forward count mismatch")
        elif condition_metrics.get("cache_enabled") is not True:
            errors.append(f"{prefix}: cache result does not declare cache_enabled=true")

        if len(errors) == row_error_start:
            valid_results.append(result)

    missing = sorted(set(prompt_by_id) - seen)
    if missing and not args.allow_partial:
        errors.append(
            f"Run incomplete: missing {len(missing)} of {len(prompts)} frozen samples"
        )
    extra_count = len([sample_id for sample_id in seen if sample_id not in prompt_by_id])
    if extra_count:
        errors.append(f"Run contains {extra_count} extra sample IDs")

    expected_per_cell = int(config["grid"]["n_samples_per_cell"])
    cell_counts = Counter(
        (result["target_total_len"], result["requested_depth"])
        for result in valid_results
    )
    if not args.allow_partial and valid_results:
        expected_cells = (
            len(config["grid"]["target_total_lengths"])
            * len(config["grid"]["depths"])
        )
        if len(cell_counts) != expected_cells or set(cell_counts.values()) != {
            expected_per_cell
        }:
            errors.append("Valid rows do not form the complete balanced grid")

    latencies = [float(row["latency_seconds"]) for row in valid_results]
    peaks = [int(row["peak_memory_allocated_bytes"]) for row in valid_results]
    report = {
        "schema_version": 1,
        "validated_at_utc": utc_now(),
        "status": "valid" if not errors else "invalid",
        "experiment_id": config["experiment_id"],
        "condition": args.condition,
        "experiment_lock_sha256": experiment_lock_sha256,
        "condition_lock_sha256": condition_lock_sha256,
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path) if results_path.is_file() else None,
        "expected_prompt_count": len(prompts),
        "result_count": len(results),
        "valid_result_count": len(valid_results),
        "missing_count": len(missing),
        "observed_cell_count": len(cell_counts),
        "session_count": len({row["session_id"] for row in valid_results}),
        "contains_expected_number_accuracy": (
            sum(bool(row["contains_expected_number"]) for row in valid_results)
            / len(valid_results)
            if valid_results
            else None
        ),
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "median_peak_memory_allocated_bytes": (
            statistics.median(peaks) if peaks else None
        ),
        "error_count": len(errors),
        "errors": errors,
    }
    write_json_atomic(report_path, report)
    print(
        f"status={report['status']} valid={len(valid_results)}/{len(prompts)} "
        f"errors={len(errors)} report={report_path}"
    )
    if errors:
        for error in errors[:20]:
            print(f"ERROR: {error}")
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
