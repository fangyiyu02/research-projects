"""Create fresh baseline, cache, paired quality, speed, and memory figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import (
    iter_jsonl,
    locked_path,
    read_json,
    sha256_file,
    utc_now,
    verify_file_lock,
    write_json_atomic,
)


def resolve_argument(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_condition(
    path: Path,
    *,
    condition: str,
    expected_ids: set[str],
    experiment_lock_sha256: str,
    allow_partial: bool,
) -> pd.DataFrame:
    records = list(iter_jsonl(path))
    if not records:
        raise ValueError(f"No result rows found: {path}")
    frame = pd.DataFrame(records)
    required = {
        "sample_id",
        "condition",
        "target_total_len",
        "requested_depth",
        "contains_expected_number",
        "latency_seconds",
        "peak_memory_allocated_bytes",
        "input_ids_sha256",
        "experiment_lock_sha256",
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} lacks columns: {sorted(missing_columns)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")
    if set(frame["condition"]) != {condition}:
        raise ValueError(f"Unexpected condition values in {path}")
    if set(frame["experiment_lock_sha256"]) != {experiment_lock_sha256}:
        raise ValueError(f"Experiment-lock mismatch in {path}")
    observed_ids = set(frame["sample_id"])
    if not observed_ids <= expected_ids:
        raise ValueError(f"Unknown sample IDs in {path}")
    if not allow_partial and observed_ids != expected_ids:
        raise ValueError(
            f"{condition} results are incomplete: {len(observed_ids)}/{len(expected_ids)}"
        )
    for column in (
        "latency_seconds",
        "peak_memory_allocated_bytes",
        "target_total_len",
        "requested_depth",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["contains_expected_number"] = frame["contains_expected_number"].astype(bool)
    return frame


def require_validation_report(
    report_path: Path,
    *,
    results_path: Path,
    condition: str,
    experiment_lock_sha256: str,
) -> dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Validation report missing: {report_path}. Run validate_results.py first."
        )
    report = read_json(report_path)
    if report.get("status") != "valid":
        raise ValueError(f"Validation report is not valid: {report_path}")
    if report.get("condition") != condition:
        raise ValueError(f"Validation report condition mismatch: {report_path}")
    if report.get("experiment_lock_sha256") != experiment_lock_sha256:
        raise ValueError(f"Validation report experiment-lock mismatch: {report_path}")
    if report.get("results_sha256") != sha256_file(results_path):
        raise ValueError(f"Results changed after validation: {results_path}")
    return report


def cell_matrix(
    frame: pd.DataFrame,
    value_column: str,
    aggregation: str,
    lengths: list[int],
    depths: list[float],
) -> pd.DataFrame:
    grouped = frame.groupby(
        ["requested_depth", "target_total_len"], observed=True
    )[value_column].agg(aggregation)
    return grouped.unstack("target_total_len").reindex(index=depths, columns=lengths)


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    annotation_format: str = ".1f",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(19, 8))
    axis = sns.heatmap(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        annot=True,
        fmt=annotation_format,
        linewidths=0.35,
        cbar_kws={"label": colorbar_label},
    )
    axis.set_xlabel("Target total length (prompt + generated tokens)")
    axis.set_ylabel("Requested needle depth")
    axis.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-lock", default="locks/experiment_v1.lock.json"
    )
    parser.add_argument("--no-cache-results", required=True)
    parser.add_argument(
        "--no-cache-validation",
        default="runs/no_cache/validation_report.json",
    )
    parser.add_argument("--cache-results", default=None)
    parser.add_argument(
        "--cache-validation",
        default="runs/dllm_cache/validation_report.json",
    )
    parser.add_argument("--output-dir", default="analysis")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    project_root, experiment_lock, experiment_lock_sha256 = verify_file_lock(
        Path(args.experiment_lock).resolve(), "experiment"
    )
    config = read_json(
        locked_path(experiment_lock, project_root, "experiment_config")
    )
    prompts = list(
        iter_jsonl(locked_path(experiment_lock, project_root, "prompt_manifest"))
    )
    expected_ids = {record["sample_id"] for record in prompts}
    lengths = [int(value) for value in config["grid"]["target_total_lengths"]]
    depths = [float(value) for value in config["grid"]["depths"]]
    output_dir = resolve_argument(project_root, args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    no_cache_path = resolve_argument(project_root, args.no_cache_results)
    no_cache_validation_path = resolve_argument(
        project_root, args.no_cache_validation
    )
    require_validation_report(
        no_cache_validation_path,
        results_path=no_cache_path,
        condition="no_cache",
        experiment_lock_sha256=experiment_lock_sha256,
    )
    baseline = load_condition(
        no_cache_path,
        condition="no_cache",
        expected_ids=expected_ids,
        experiment_lock_sha256=experiment_lock_sha256,
        allow_partial=args.allow_partial,
    )
    baseline_accuracy = 100 * cell_matrix(
        baseline,
        "contains_expected_number",
        "mean",
        lengths,
        depths,
    )
    no_cache_figure = figures_dir / "no_cache_accuracy_heatmap.png"
    save_heatmap(
        baseline_accuracy,
        no_cache_figure,
        title="Fresh NIAH no-cache accuracy",
        colorbar_label="Accuracy (%)",
        cmap="viridis",
        vmin=0,
        vmax=100,
        annotation_format=".0f",
    )
    baseline_cell = (
        baseline.groupby(["target_total_len", "requested_depth"], observed=True)
        .agg(
            accuracy=("contains_expected_number", "mean"),
            median_latency_seconds=("latency_seconds", "median"),
            median_peak_memory_bytes=("peak_memory_allocated_bytes", "median"),
            n=("sample_id", "size"),
        )
        .reset_index()
    )
    baseline_cell.to_csv(tables_dir / "no_cache_cell_summary.csv", index=False)

    produced_figures = [no_cache_figure]
    produced_tables = [tables_dir / "no_cache_cell_summary.csv"]
    cache_path: Path | None = None
    cache_validation_path: Path | None = None
    if args.cache_results:
        cache_path = resolve_argument(project_root, args.cache_results)
        cache_validation_path = resolve_argument(project_root, args.cache_validation)
        require_validation_report(
            cache_validation_path,
            results_path=cache_path,
            condition="dllm_cache",
            experiment_lock_sha256=experiment_lock_sha256,
        )
        cache = load_condition(
            cache_path,
            condition="dllm_cache",
            expected_ids=expected_ids,
            experiment_lock_sha256=experiment_lock_sha256,
            allow_partial=args.allow_partial,
        )
        cache_ids = set(cache["sample_id"])
        baseline_ids = set(baseline["sample_id"])
        if cache_ids != baseline_ids:
            if not args.allow_partial:
                raise ValueError("Condition result sets are not exactly paired")
            paired_ids = cache_ids & baseline_ids
            baseline = baseline[baseline["sample_id"].isin(paired_ids)].copy()
            cache = cache[cache["sample_id"].isin(paired_ids)].copy()

        cache_accuracy = 100 * cell_matrix(
            cache,
            "contains_expected_number",
            "mean",
            lengths,
            depths,
        )
        cache_figure = figures_dir / "dllm_cache_accuracy_heatmap.png"
        save_heatmap(
            cache_accuracy,
            cache_figure,
            title="Fresh NIAH dLLM-Cache accuracy",
            colorbar_label="Accuracy (%)",
            cmap="viridis",
            vmin=0,
            vmax=100,
            annotation_format=".0f",
        )
        produced_figures.append(cache_figure)

        identity_columns = [
            "sample_id",
            "target_total_len",
            "requested_depth",
            "input_ids_sha256",
        ]
        paired = baseline.merge(
            cache,
            on="sample_id",
            how="inner",
            suffixes=("_no_cache", "_cache"),
            validate="one_to_one",
        )
        if paired.empty:
            raise ValueError("No paired sample IDs are shared by the two conditions")
        for column in identity_columns[1:]:
            left = paired[f"{column}_no_cache"]
            right = paired[f"{column}_cache"]
            if not left.equals(right):
                raise ValueError(f"Paired identity mismatch in {column}")
        paired["target_total_len"] = paired["target_total_len_no_cache"]
        paired["requested_depth"] = paired["requested_depth_no_cache"]
        paired["accuracy_delta_pp"] = 100 * (
            paired["contains_expected_number_cache"].astype(float)
            - paired["contains_expected_number_no_cache"].astype(float)
        )
        paired["latency_speedup"] = (
            paired["latency_seconds_no_cache"] / paired["latency_seconds_cache"]
        )
        paired["peak_memory_no_cache_gib"] = (
            paired["peak_memory_allocated_bytes_no_cache"] / (1024**3)
        )
        paired["peak_memory_cache_gib"] = (
            paired["peak_memory_allocated_bytes_cache"] / (1024**3)
        )

        delta_matrix = cell_matrix(
            paired,
            "accuracy_delta_pp",
            "mean",
            lengths,
            depths,
        )
        finite_delta = np.abs(delta_matrix.to_numpy(dtype=float))
        delta_limit = max(10.0, float(np.nanmax(finite_delta)))
        delta_figure = figures_dir / "accuracy_delta_cache_minus_no_cache.png"
        save_heatmap(
            delta_matrix,
            delta_figure,
            title="Accuracy change: dLLM-Cache minus no-cache",
            colorbar_label="Accuracy difference (percentage points)",
            cmap="RdBu_r",
            vmin=-delta_limit,
            vmax=delta_limit,
            center=0,
            annotation_format=".0f",
        )

        speedup_matrix = cell_matrix(
            paired,
            "latency_speedup",
            "median",
            lengths,
            depths,
        )
        speedup_figure = figures_dir / "paired_latency_speedup_heatmap.png"
        speedup_values = speedup_matrix.to_numpy(dtype=float)
        speedup_max = max(1.5, float(np.nanpercentile(speedup_values, 98)))
        save_heatmap(
            speedup_matrix,
            speedup_figure,
            title="Paired latency speedup (no-cache / dLLM-Cache)",
            colorbar_label="Speedup (×)",
            cmap="RdYlGn",
            vmin=min(0.5, float(np.nanmin(speedup_values))),
            vmax=speedup_max,
            center=1.0,
            annotation_format=".2f",
        )

        memory_by_length = (
            paired.groupby("target_total_len", observed=True)
            .agg(
                no_cache_peak_gib=("peak_memory_no_cache_gib", "median"),
                cache_peak_gib=("peak_memory_cache_gib", "median"),
                n=("sample_id", "size"),
            )
            .reset_index()
        )
        memory_figure = figures_dir / "peak_gpu_memory_comparison.png"
        plt.figure(figsize=(11, 6.5))
        plt.plot(
            memory_by_length["target_total_len"],
            memory_by_length["no_cache_peak_gib"],
            marker="o",
            linewidth=2,
            label="No cache",
        )
        plt.plot(
            memory_by_length["target_total_len"],
            memory_by_length["cache_peak_gib"],
            marker="o",
            linewidth=2,
            label="dLLM-Cache",
        )
        plt.xlabel("Target total length (prompt + generated tokens)")
        plt.ylabel("Median peak allocated GPU memory (GiB)")
        plt.title("Peak GPU-memory comparison")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(memory_figure, dpi=220, bbox_inches="tight")
        plt.close()

        paired_columns = [
            "sample_id",
            "target_total_len",
            "requested_depth",
            "accuracy_delta_pp",
            "latency_seconds_no_cache",
            "latency_seconds_cache",
            "latency_speedup",
            "peak_memory_no_cache_gib",
            "peak_memory_cache_gib",
        ]
        paired_path = tables_dir / "paired_sample_metrics.csv"
        paired[paired_columns].to_csv(paired_path, index=False)
        paired_cell = (
            paired.groupby(["target_total_len", "requested_depth"], observed=True)
            .agg(
                accuracy_delta_pp=("accuracy_delta_pp", "mean"),
                median_latency_speedup=("latency_speedup", "median"),
                median_no_cache_peak_gib=("peak_memory_no_cache_gib", "median"),
                median_cache_peak_gib=("peak_memory_cache_gib", "median"),
                n=("sample_id", "size"),
            )
            .reset_index()
        )
        paired_cell_path = tables_dir / "paired_cell_summary.csv"
        paired_cell.to_csv(paired_cell_path, index=False)
        memory_path = tables_dir / "peak_memory_by_length.csv"
        memory_by_length.to_csv(memory_path, index=False)
        produced_figures.extend([delta_figure, speedup_figure, memory_figure])
        produced_tables.extend([paired_path, paired_cell_path, memory_path])

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "experiment_id": config["experiment_id"],
        "experiment_lock_sha256": experiment_lock_sha256,
        "no_cache_results": {
            "path": str(no_cache_path),
            "sha256": sha256_file(no_cache_path),
            "rows": len(baseline),
            "validation_report_sha256": sha256_file(no_cache_validation_path),
        },
        "cache_results": (
            {
                "path": str(cache_path),
                "sha256": sha256_file(cache_path),
                "validation_report_sha256": sha256_file(cache_validation_path),
            }
            if cache_path
            else None
        ),
        "figures": [str(path) for path in produced_figures],
        "tables": [str(path) for path in produced_tables],
    }
    write_json_atomic(output_dir / "analysis_manifest.json", manifest)
    print(f"Analysis written to {output_dir}")
    for path in produced_figures:
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
