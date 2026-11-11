"""Shared immutable-manifest runner for no-cache and dLLM-Cache conditions."""

from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

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


Generator = Callable[..., tuple[torch.Tensor, dict[str, Any]] | torch.Tensor]


def append_jsonl_durable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def set_runtime_controls(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    seed = int(runtime["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(runtime["cudnn_deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime["allow_tf32"])
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = bool(runtime["allow_tf32"])
    torch.use_deterministic_algorithms(bool(runtime["deterministic_algorithms"]))


def set_sample_seed(sample_seed: int) -> None:
    random.seed(sample_seed)
    np.random.seed(sample_seed % (2**32))
    torch.manual_seed(sample_seed)
    torch.cuda.manual_seed_all(sample_seed)


def resolved_commit_candidates(obj: Any) -> list[str]:
    candidates: list[str] = []
    for source in (obj, getattr(obj, "config", None)):
        value = getattr(source, "_commit_hash", None)
        if isinstance(value, str):
            candidates.append(value)
    init_kwargs = getattr(obj, "init_kwargs", {})
    if isinstance(init_kwargs, dict) and isinstance(init_kwargs.get("_commit_hash"), str):
        candidates.append(init_kwargs["_commit_hash"])
    return sorted(set(candidates))


def require_resolved_commit(obj: Any, expected: str, label: str) -> list[str]:
    observed = resolved_commit_candidates(obj)
    for commit in observed:
        if commit != expected:
            raise ValueError(
                f"{label} resolved to {commit}; frozen revision is {expected}"
            )
    return observed


def nvidia_smi_metadata(device_index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=driver_version,pstate,clocks.current.sm,power.limit",
        "--format=csv,noheader,nounits",
        f"--id={device_index}",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
        values = [value.strip() for value in completed.stdout.strip().split(",")]
        if len(values) == 4:
            return {
                "driver_version": values[0],
                "pstate_at_session_start": values[1],
                "sm_clock_mhz_at_session_start": values[2],
                "power_limit_w_at_session_start": values[3],
            }
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return {"nvidia_smi_query": "unavailable"}


def load_completed(
    results_path: Path,
    prompt_by_id: dict[str, dict[str, Any]],
    *,
    condition: str,
    experiment_lock_sha256: str,
    condition_lock_sha256: str | None,
    condition_source_bundle_sha256: str,
    config_sha256: str,
    config: dict[str, Any],
) -> set[str]:
    completed: set[str] = set()
    if not results_path.exists():
        return completed
    for result in iter_jsonl(results_path):
        sample_id = result.get("sample_id")
        if sample_id not in prompt_by_id:
            raise ValueError(f"Existing result has unknown sample_id: {sample_id!r}")
        if sample_id in completed:
            raise ValueError(f"Existing result has duplicate sample_id: {sample_id}")
        expected_fields = {
            "condition": condition,
            "experiment_lock_sha256": experiment_lock_sha256,
            "condition_lock_sha256": condition_lock_sha256,
            "condition_source_bundle_sha256": condition_source_bundle_sha256,
            "config_sha256": config_sha256,
            "model_revision": config["model"]["revision"],
            "model_code_revision": config["model"]["code_revision"],
            "tokenizer_revision": config["tokenizer"]["revision"],
            "tokenizer_code_revision": config["tokenizer"]["code_revision"],
        }
        for key, expected in expected_fields.items():
            if result.get(key) != expected:
                raise ValueError(
                    f"Existing result {sample_id} has incompatible {key!r}"
                )
        if result.get("input_ids_sha256") != prompt_by_id[sample_id]["input_ids_sha256"]:
            raise ValueError(f"Existing result prompt hash mismatch: {sample_id}")
        completed.add(sample_id)
    return completed


def run_one(
    *,
    model: Any,
    generator: Generator,
    prompt: dict[str, Any],
    generation: dict[str, Any],
    condition_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, int | float], dict[str, Any]]:
    input_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    start = time.perf_counter()
    generated = generator(
        model,
        input_ids,
        attention_mask,
        steps=int(generation["steps"]),
        gen_length=int(generation["gen_length"]),
        block_length=int(generation["block_length"]),
        temperature=float(generation["temperature"]),
        cfg_scale=float(generation["cfg_scale"]),
        remasking=generation["remasking"],
        mask_id=int(generation["mask_id"]),
        logits_eos_inf=bool(generation["logits_eos_inf"]),
        confidence_eos_eot_inf=bool(generation["confidence_eos_eot_inf"]),
        condition_config=condition_config,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if isinstance(generated, tuple):
        output_ids, condition_metrics = generated
    else:
        output_ids, condition_metrics = generated, {}
    efficiency: dict[str, int | float] = {
        "latency_seconds": elapsed,
        "generated_tokens_per_second": int(generation["gen_length"]) / elapsed,
        "memory_allocated_before_bytes": before_allocated,
        "memory_reserved_before_bytes": before_reserved,
        "memory_allocated_after_bytes": torch.cuda.memory_allocated(),
        "memory_reserved_after_bytes": torch.cuda.memory_reserved(),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    return (
        output_ids[:, input_ids.shape[1] :],
        efficiency,
        dict(condition_metrics),
    )


def run_condition(
    *,
    condition: str,
    generator: Generator,
    experiment_lock_path: Path,
    output_dir: Path,
    condition_source_paths: list[Path],
    limit: int | None = None,
    condition_lock_path: Path | None = None,
) -> None:
    if condition not in {"no_cache", "dllm_cache"}:
        raise ValueError(f"Unknown condition: {condition}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal inference requires an available CUDA GPU")

    project_root, experiment_lock, experiment_lock_sha256 = verify_file_lock(
        experiment_lock_path.resolve(), "experiment"
    )
    condition_lock_sha256: str | None = None
    condition_config: dict[str, Any] = {}
    if condition == "dllm_cache":
        if condition_lock_path is None:
            raise ValueError("dllm_cache requires a frozen condition lock")
        condition_root, condition_lock, condition_lock_sha256 = verify_file_lock(
            condition_lock_path.resolve(), "condition"
        )
        if condition_root != project_root:
            raise ValueError("Experiment and condition locks are from different projects")
        if condition_lock["experiment_lock_sha256"] != experiment_lock_sha256:
            raise ValueError("Cache condition was frozen against a different experiment")
        condition_config = read_json(
            locked_path(condition_lock, project_root, "condition_config")
        )
    elif condition_lock_path is not None:
        raise ValueError("no_cache must not receive a cache condition lock")

    config_path = locked_path(experiment_lock, project_root, "experiment_config")
    prompt_path = locked_path(experiment_lock, project_root, "prompt_manifest")
    config = read_json(config_path)
    config_sha256 = sha256_file(config_path)
    prompts = list(iter_jsonl(prompt_path))
    prompt_by_id = {prompt["sample_id"]: prompt for prompt in prompts}
    if len(prompt_by_id) != len(prompts):
        raise ValueError("Frozen prompt manifest contains duplicate sample IDs")
    for prompt in prompts:
        if sha256_input_ids(prompt["input_ids"]) != prompt["input_ids_sha256"]:
            raise ValueError(f"Frozen prompt input IDs changed: {prompt['sample_id']}")

    source_paths = [path.resolve() for path in condition_source_paths]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Condition source missing: {path}")
    condition_source_bundle_sha256 = sha256_named_files(source_paths)

    output_dir = (
        output_dir.resolve()
        if output_dir.is_absolute()
        else (project_root / output_dir).resolve()
    )
    try:
        output_relative = output_dir.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Run output directory must remain inside the project") from exc
    if not output_relative.parts or output_relative.parts[0] != "runs":
        raise ValueError("Run output directory must remain under runs/")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    failures_path = output_dir / "failures.jsonl"
    completed = load_completed(
        results_path,
        prompt_by_id,
        condition=condition,
        experiment_lock_sha256=experiment_lock_sha256,
        condition_lock_sha256=condition_lock_sha256,
        condition_source_bundle_sha256=condition_source_bundle_sha256,
        config_sha256=config_sha256,
        config=config,
    )

    set_runtime_controls(config)
    tokenizer_cfg = config["tokenizer"]
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_cfg["id"],
        revision=tokenizer_cfg["revision"],
        code_revision=tokenizer_cfg["code_revision"],
        trust_remote_code=bool(tokenizer_cfg["trust_remote_code"]),
        use_fast=bool(tokenizer_cfg["use_fast"]),
    )
    model = AutoModel.from_pretrained(
        model_cfg["id"],
        revision=model_cfg["revision"],
        code_revision=model_cfg["code_revision"],
        trust_remote_code=bool(model_cfg["trust_remote_code"]),
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model_commits = require_resolved_commit(model, model_cfg["revision"], "model")
    tokenizer_commits = require_resolved_commit(
        tokenizer, tokenizer_cfg["revision"], "tokenizer"
    )

    session_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    session = {
        "schema_version": 1,
        "session_id": session_id,
        "condition": condition,
        "started_at_utc": utc_now(),
        "experiment_id": config["experiment_id"],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": {
            "index": device_index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            **nvidia_smi_metadata(device_index),
        },
        "experiment_lock_sha256": experiment_lock_sha256,
        "condition_lock_sha256": condition_lock_sha256,
        "config_sha256": config_sha256,
        "prompt_manifest_sha256": sha256_file(prompt_path),
        "condition_source_bundle_sha256": condition_source_bundle_sha256,
        "condition_source_files": [str(path) for path in source_paths],
        "model_revision_requested": model_cfg["revision"],
        "model_code_revision_requested": model_cfg["code_revision"],
        "model_resolved_commit_candidates": model_commits,
        "tokenizer_revision_requested": tokenizer_cfg["revision"],
        "tokenizer_code_revision_requested": tokenizer_cfg["code_revision"],
        "tokenizer_resolved_commit_candidates": tokenizer_commits,
        "generation": config["generation"],
        "runtime": config["runtime"],
        "condition_config": condition_config,
        "timed_region": "generator call only; CUDA synchronized immediately before and after",
        "output_path": str(results_path),
    }
    session_path = output_dir / f"session_{session_id}.json"
    write_json_atomic(session_path, session)

    generation = config["generation"]
    warmup_repetitions = int(config["runtime"]["warmup_repetitions"])
    for warmup_index in range(warmup_repetitions):
        warmup_prompt = prompts[warmup_index % len(prompts)]
        set_sample_seed(int(warmup_prompt["sample_seed"]))
        run_one(
            model=model,
            generator=generator,
            prompt=warmup_prompt,
            generation=generation,
            condition_config=condition_config,
        )
    if warmup_repetitions:
        torch.cuda.synchronize()

    pending = [prompt for prompt in prompts if prompt["sample_id"] not in completed]
    if limit is not None:
        pending = pending[:limit]
    print(
        f"condition={condition} frozen={len(prompts)} completed={len(completed)} "
        f"scheduled_now={len(pending)} session={session_id}",
        flush=True,
    )

    successes = 0
    failures = 0
    for prompt in pending:
        sample_id = prompt["sample_id"]
        try:
            set_sample_seed(int(prompt["sample_seed"]))
            generated_ids, efficiency, condition_metrics = run_one(
                model=model,
                generator=generator,
                prompt=prompt,
                generation=generation,
                condition_config=condition_config,
            )
            output_text = tokenizer.decode(
                generated_ids[0].detach().cpu().tolist(), skip_special_tokens=True
            )
            result = {
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "condition": condition,
                "session_id": session_id,
                "sample_id": sample_id,
                "run_order_index": prompt["run_order_index"],
                "sample_seed": prompt["sample_seed"],
                "target_total_len": prompt["target_total_len"],
                "actual_total_len": prompt["actual_total_len"],
                "prompt_token_length": prompt["prompt_token_length"],
                "requested_depth": prompt["requested_depth"],
                "sample_idx": prompt["sample_idx"],
                "needle_token_start": prompt["needle_token_start"],
                "needle_token_end_exclusive": prompt["needle_token_end_exclusive"],
                "question_token_start": prompt["question_token_start"],
                "needle_to_question_start_distance": prompt[
                    "needle_to_question_start_distance"
                ],
                "needle_end_to_question_gap": prompt["needle_end_to_question_gap"],
                "needle_prompt_position_ratio": prompt["needle_prompt_position_ratio"],
                "city": prompt["city"],
                "needle_number": prompt["needle_number"],
                "model_output": output_text,
                **score_output(output_text, prompt["needle_number"]),
                **efficiency,
                "condition_metrics": condition_metrics,
                "input_ids_sha256": prompt["input_ids_sha256"],
                "prompt_text_sha256": prompt["prompt_text_sha256"],
                "experiment_lock_sha256": experiment_lock_sha256,
                "condition_lock_sha256": condition_lock_sha256,
                "config_sha256": config_sha256,
                "condition_source_bundle_sha256": condition_source_bundle_sha256,
                "model_revision": model_cfg["revision"],
                "model_code_revision": model_cfg["code_revision"],
                "tokenizer_revision": tokenizer_cfg["revision"],
                "tokenizer_code_revision": tokenizer_cfg["code_revision"],
                "completed_at_utc": utc_now(),
            }
            append_jsonl_durable(results_path, result)
            successes += 1
            print(
                f"ok {sample_id} latency={efficiency['latency_seconds']:.6f}s "
                f"peak={efficiency['peak_memory_allocated_bytes']}B",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            append_jsonl_durable(
                failures_path,
                {
                    "schema_version": 1,
                    "experiment_id": config["experiment_id"],
                    "condition": condition,
                    "session_id": session_id,
                    "sample_id": sample_id,
                    "input_ids_sha256": prompt["input_ids_sha256"],
                    "experiment_lock_sha256": experiment_lock_sha256,
                    "condition_lock_sha256": condition_lock_sha256,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at_utc": utc_now(),
                },
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f"failed {sample_id}: {type(exc).__name__}: {exc}", flush=True)

    session["finished_at_utc"] = utc_now()
    session["successes_this_session"] = successes
    session["failures_this_session"] = failures
    session["completed_total_after_session"] = len(completed) + successes
    write_json_atomic(session_path, session)
    print(
        f"Finished {session_id}: successes={successes} failures={failures}",
        flush=True,
    )
    if failures:
        raise SystemExit(2)
