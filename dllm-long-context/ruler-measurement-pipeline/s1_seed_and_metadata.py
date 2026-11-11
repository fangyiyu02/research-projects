#!/usr/bin/env python3
"""S1 seed and experiment metadata for UltraLLaDA experiments.

The module has two experimental responsibilities only:

1. Reset Python, NumPy, and PyTorch RNGs before each measured run.
2. Record the run configuration outside the timed region, including the
   concrete fused backend selected by PyTorch SDPA.

SDPA is allowed to choose between PyTorch FLASH, EFFICIENT, and cuDNN
attention.  MATH is deliberately excluded: if no fused backend can serve the
real input shape/dtype/mask, the preflight fails instead of silently running a
very slow or quadratic-memory MATH fallback.

The SDPA backend detector profiles one real, untimed warm-up forward.  No
profiler, system query, or JSON write belongs inside the timed region.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import socket
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile


SCHEMA_VERSION = 2

_FUSED_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
]
if getattr(SDPBackend, "CUDNN_ATTENTION", None) is not None:
    _FUSED_SDPA_BACKENDS.append(SDPBackend.CUDNN_ATTENTION)

_SDPA_PROFILER_EVENTS = {
    "aten::_scaled_dot_product_flash_attention": "flash",
    "aten::_scaled_dot_product_efficient_attention": "efficient",
    "aten::_scaled_dot_product_cudnn_attention": "cudnn",
    "aten::_scaled_dot_product_attention_math": "math",
}


# --------------------------------------------------------------------------- #
# Goal 1: global seed                                                         #
# --------------------------------------------------------------------------- #


def set_global_seed(seed: int) -> None:
    """Reset the RNGs used by UltraLLaDA generation.

    Call this after all warm-up forwards and immediately before the measured
    run.  Launching Python with ``PYTHONHASHSEED=<seed>`` is still recommended,
    because the current process's hash seed is fixed at interpreter startup.
    """
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Goal 2a: exact workload metadata                                            #
# --------------------------------------------------------------------------- #


def exact_token_metadata(
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    generation_tokens: int,
    diffusion_steps: int | None = None,
    forward_passes: int | None = None,
) -> dict[str, Any]:
    """Count prompt tokens from the exact tensors passed to the model."""
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [sequence] or [batch, sequence]")
    if generation_tokens < 0:
        raise ValueError("generation_tokens must be non-negative")

    batch_size, padded_length = input_ids.shape
    if attention_mask is None:
        prompt_tokens_per_sample = [int(padded_length)] * int(batch_size)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        prompt_tokens_per_sample = [
            int(value)
            for value in attention_mask.to(dtype=torch.int64).sum(dim=-1).cpu().tolist()
        ]

    metadata: dict[str, Any] = {
        "batch_size": int(batch_size),
        "padded_prompt_length": int(padded_length),
        "prompt_tokens_per_sample": prompt_tokens_per_sample,
        "prompt_tokens_total": int(sum(prompt_tokens_per_sample)),
        "generation_tokens_per_sample": int(generation_tokens),
        "generation_tokens_total": int(generation_tokens * batch_size),
        "total_tokens_per_sample": [
            int(prompt_tokens + generation_tokens)
            for prompt_tokens in prompt_tokens_per_sample
        ],
        "total_tokens_all_samples": int(
            sum(prompt_tokens_per_sample) + generation_tokens * batch_size
        ),
        "model_sequence_length": int(padded_length + generation_tokens),
    }
    if diffusion_steps is not None:
        metadata["diffusion_steps"] = int(diffusion_steps)
    if forward_passes is not None:
        metadata["forward_passes"] = int(forward_passes)
    return metadata


# --------------------------------------------------------------------------- #
# Goal 2b: concrete PyTorch SDPA backend                                      #
# --------------------------------------------------------------------------- #


@contextmanager
def fused_sdpa_only() -> Iterator[None]:
    """Allow fused PyTorch SDPA backends while making MATH fallback impossible.

    Enter this context before starting the timer and leave it after the final
    CUDA synchronization.  Context entry/exit occurs outside the timed region.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("fused SDPA requires CUDA")
    with sdpa_kernel(_FUSED_SDPA_BACKENDS):
        yield


def detect_sdpa_backends(one_real_forward: Callable[[], Any]) -> list[str]:
    """Profile one untimed real forward and return the fused SDPA backend(s).

    ``one_real_forward`` must use the same model, dtype, input shape, and
    attention mask as the measured run.  MATH is disabled during this preflight,
    so an unsupported fused configuration raises before formal timing begins.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("cannot detect a CUDA SDPA backend without CUDA")

    with torch.inference_mode(), fused_sdpa_only():
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            output = one_real_forward()
            torch.cuda.synchronize()
        del output

    event_names = {event.key for event in prof.key_averages()}
    detected = sorted(
        {
            backend_name
            for event_name, backend_name in _SDPA_PROFILER_EVENTS.items()
            if event_name in event_names
        }
    )

    if "math" in detected:
        raise RuntimeError("MATH SDPA fallback detected; refusing to benchmark this run")
    fused = [name for name in detected if name in {"flash", "efficient", "cudnn"}]
    if not fused:
        raise RuntimeError(
            "no recognized fused SDPA operator was observed in the real forward; "
            "the model may not be using PyTorch SDPA or profiler visibility may be hidden"
        )
    return fused


# --------------------------------------------------------------------------- #
# Goal 2c: environment and code metadata                                      #
# --------------------------------------------------------------------------- #


def _run_command(command: Sequence[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if result.returncode != 0:
        return {
            "ok": False,
            "returncode": result.returncode,
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    return {"ok": True, "stdout": result.stdout.strip()}


def _git_metadata(code_dir: Path) -> dict[str, Any]:
    root = _run_command(["git", "rev-parse", "--show-toplevel"], code_dir)
    commit = _run_command(["git", "rev-parse", "HEAD"], code_dir)
    status = _run_command(["git", "status", "--porcelain"], code_dir)
    return {
        "requested_code_dir": str(code_dir.resolve()),
        "repository_root": root.get("stdout") if root.get("ok") else None,
        "commit": commit.get("stdout") if commit.get("ok") else None,
        "dirty": bool(status.get("stdout")) if status.get("ok") else None,
        "status_porcelain": status.get("stdout") if status.get("ok") else None,
        "errors": [
            item["error"]
            for item in (root, commit, status)
            if not item.get("ok") and item.get("error")
        ],
    }


def _nvidia_smi_metadata() -> dict[str, Any]:
    fields = [
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "power.limit",
        "temperature.gpu",
        "pstate",
        "clocks.sm",
        "clocks.mem",
        "utilization.gpu",
    ]
    result = _run_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error")}

    rows = []
    for row in csv.reader(result["stdout"].splitlines(), skipinitialspace=True):
        rows.append({key: value.strip() for key, value in zip(fields, row)})
    return {"ok": True, "gpus": rows}


def _torch_device_metadata() -> list[dict[str, Any]]:
    devices = []
    if not torch.cuda.is_available():
        return devices
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        device_uuid = getattr(properties, "uuid", None)
        devices.append(
            {
                "logical_index": index,
                "uuid": str(device_uuid) if device_uuid is not None else None,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": int(properties.total_memory),
                "multiprocessor_count": int(properties.multi_processor_count),
            }
        )
    return devices


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else None,
        "numpy": np.__version__,
    }
    for distribution, field_name in (
        ("transformers", "transformers"),
        ("flash-attn", "third_party_flash_attn"),
    ):
        try:
            versions[field_name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[field_name] = None
    return versions


def collect_run_metadata(
    *,
    seed: int,
    code_dir: str | Path,
    experiment: str,
    workload: Mapping[str, Any],
    actual_sdpa_backends: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect all run metadata before formal timing begins."""
    normalized_backends = sorted(set(actual_sdpa_backends))
    unsupported = set(normalized_backends) - {"flash", "efficient", "cudnn"}
    if not normalized_backends or unsupported:
        raise ValueError(
            f"actual_sdpa_backends must contain only fused backends; got {normalized_backends}"
        )

    now = datetime.now(timezone.utc)
    packages = _package_versions()
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}",
        "status": "prepared",
        "prepared_at_utc": now.isoformat(),
        "experiment": experiment,
        "seed": int(seed),
        "attention": {
            "implementation": "torch.nn.functional.scaled_dot_product_attention",
            "policy": "fused_only_math_disabled",
            "actual_backends": normalized_backends,
            "math_allowed": False,
            "math_fallback_detected": False,
            "third_party_flash_attn_installed": packages["third_party_flash_attn"]
            is not None,
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
        "packages": packages,
        "cuda": {
            "available": torch.cuda.is_available(),
            "visible_device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device()
            if torch.cuda.is_available()
            else None,
            "torch_visible_devices": _torch_device_metadata(),
            "nvidia_smi": _nvidia_smi_metadata(),
        },
        "code": _git_metadata(Path(code_dir)),
        "workload": dict(workload),
        "timing_contract": {
            "sdpa_detection": "one real warm-up forward before timed region",
            "metadata_collection": "before timed region",
            "initial_json_write": "before timed region",
            "seed_reset": "after warm-up, immediately before timed region",
            "profiler_during_timing": False,
            "final_json_write": "after timed region",
        },
    }
    if extra:
        metadata["extra"] = dict(extra)
    return metadata


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass
class RunRecord:
    path: Path
    metadata: dict[str, Any]


def prepare_run(
    *,
    log_dir: str | Path,
    seed: int,
    code_dir: str | Path,
    experiment: str,
    workload: Mapping[str, Any],
    actual_sdpa_backends: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> RunRecord:
    """Collect and write metadata before timing; this function does not seed."""
    metadata = collect_run_metadata(
        seed=seed,
        code_dir=code_dir,
        experiment=experiment,
        workload=workload,
        actual_sdpa_backends=actual_sdpa_backends,
        extra=extra,
    )
    path = Path(log_dir) / f"{metadata['run_id']}.json"
    _write_json_atomic(path, metadata)
    return RunRecord(path=path, metadata=metadata)


def finalize_run(
    record: RunRecord,
    *,
    metrics: Mapping[str, Any],
    status: str = "completed",
) -> None:
    """Append measured results after timing and final CUDA synchronization."""
    record.metadata["status"] = status
    record.metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    record.metadata["metrics"] = dict(metrics)
    _write_json_atomic(record.path, record.metadata)


# --------------------------------------------------------------------------- #
# Standalone mechanism check (not a substitute for the real-model preflight) #
# --------------------------------------------------------------------------- #


def _verify_seed_replay(seed: int) -> dict[str, bool]:
    set_global_seed(seed)
    python_a, numpy_a, torch_cpu_a = random.random(), np.random.random(), torch.rand(4)
    torch_cuda_a = torch.rand(4, device="cuda") if torch.cuda.is_available() else None

    set_global_seed(seed)
    python_b, numpy_b, torch_cpu_b = random.random(), np.random.random(), torch.rand(4)
    torch_cuda_b = torch.rand(4, device="cuda") if torch.cuda.is_available() else None

    results = {
        "python": python_a == python_b,
        "numpy": bool(numpy_a == numpy_b),
        "torch_cpu": bool(torch.equal(torch_cpu_a, torch_cpu_b)),
        "torch_cuda": bool(torch.equal(torch_cuda_a, torch_cuda_b))
        if torch_cuda_a is not None and torch_cuda_b is not None
        else False,
    }
    if not all(results.values()):
        raise RuntimeError(f"seed replay check failed: {results}")
    return results


def _self_test(seed: int, log_dir: Path, code_dir: Path) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("the S1 self-test must run in the CUDA experiment environment")

    seed_results = _verify_seed_replay(seed)
    set_global_seed(seed)
    shape = (1, 32, 1024, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def one_forward() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=False
        )

    actual_backends = detect_sdpa_backends(one_forward)
    dummy_ids = torch.zeros((1, 16), dtype=torch.long)
    workload = exact_token_metadata(
        dummy_ids,
        generation_tokens=8,
        diffusion_steps=8,
        forward_passes=8,
    )
    record = prepare_run(
        log_dir=log_dir,
        seed=seed,
        code_dir=code_dir,
        experiment="s1_seed_and_metadata_self_test",
        workload=workload,
        actual_sdpa_backends=actual_backends,
        extra={"seed_replay": seed_results},
    )

    # This mirrors the formal timing boundary: seed and context entry are both
    # outside the measured call.  The standalone self-test does not time itself.
    set_global_seed(seed)
    with fused_sdpa_only():
        torch.cuda.synchronize()
        output = one_forward()
        torch.cuda.synchronize()
    del output, query, key, value

    finalize_run(
        record,
        metrics={"self_test_passed": True, "actual_sdpa_backends": actual_backends},
    )
    return record.path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify S1 seed control, fused SDPA detection, and run logging"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/s1_self_test"))
    parser.add_argument("--code-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    output = _self_test(args.seed, args.log_dir, args.code_dir)
    payload = json.loads(output.read_text(encoding="utf-8"))
    backends = ", ".join(payload["attention"]["actual_backends"])
    print("S1 seed replay: PASSED")
    print(f"S1 fused SDPA backend: {backends}")
    print(f"S1 metadata log: {output}")


if __name__ == "__main__":
    main()
