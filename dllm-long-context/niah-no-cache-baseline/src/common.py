"""Integrity, path, scoring, and JSON helpers for the fresh experiment."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_input_ids(input_ids: Iterable[int]) -> str:
    payload = json.dumps(
        [int(token_id) for token_id in input_ids],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def sha256_named_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(p).resolve() for p in paths), key=lambda p: str(p)):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def project_root_from_config(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.parent.name != "config":
        raise ValueError("Experiment config must remain directly under config/")
    return config_path.parent.parent


def project_root_from_lock(lock_path: Path) -> Path:
    lock_path = lock_path.resolve()
    if lock_path.parent.name != "locks":
        raise ValueError("Lock must remain directly under locks/")
    return lock_path.parent.parent


def resolve_project_path(project_root: Path, relative_path: str) -> Path:
    candidate = (project_root / relative_path).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {relative_path}") from exc
    return candidate


def relative_to_project(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def require_commit(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value.lower()):
        raise ValueError(f"{field_name} must be a pinned 40-character commit SHA")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported experiment config schema_version")
    for section, fields in {
        "model": ("revision", "code_revision"),
        "tokenizer": ("revision", "code_revision"),
    }.items():
        for field in fields:
            require_commit(config[section][field], f"{section}.{field}")
        if not config[section].get("trust_remote_code"):
            raise ValueError(f"{section}.trust_remote_code must be explicit and true")

    generation = config["generation"]
    gen_length = int(generation["gen_length"])
    block_length = int(generation["block_length"])
    steps = int(generation["steps"])
    if min(gen_length, block_length, steps) <= 0:
        raise ValueError("Generation sizes must be positive")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    if steps % (gen_length // block_length):
        raise ValueError("steps must be divisible by number of blocks")
    if float(generation["temperature"]) != 0.0:
        raise ValueError("Fresh v1 baseline requires temperature=0")

    grid = config["grid"]
    lengths = [int(x) for x in grid["target_total_lengths"]]
    depths = [float(x) for x in grid["depths"]]
    if lengths != sorted(set(lengths)) or min(lengths) <= gen_length:
        raise ValueError("Target lengths must be unique, sorted, and exceed gen_length")
    if depths != sorted(set(depths)) or any(not 0 <= x <= 1 for x in depths):
        raise ValueError("Depths must be unique, sorted, and within [0, 1]")
    if int(grid["n_samples_per_cell"]) <= 0:
        raise ValueError("n_samples_per_cell must be positive")

    runtime = config["runtime"]
    if int(runtime["batch_size"]) != 1:
        raise ValueError("Formal v1 experiment is frozen to batch_size=1")
    if not runtime.get("cuda_synchronize_timing"):
        raise ValueError("CUDA-synchronized timing is mandatory")
    if not runtime.get("record_peak_memory"):
        raise ValueError("Peak CUDA memory recording is mandatory")


def expected_prompt_count(config: dict[str, Any]) -> int:
    grid = config["grid"]
    return (
        len(grid["target_total_lengths"])
        * len(grid["depths"])
        * int(grid["n_samples_per_cell"])
    )


def score_output(output_text: str, expected: str) -> dict[str, bool | float]:
    contains = expected in output_text
    candidates = re.findall(r"\d{5,9}", output_text)
    overlap = 0.0
    for candidate in candidates:
        matches = sum(a == b for a, b in zip(candidate, expected))
        overlap = max(overlap, matches / max(len(candidate), len(expected)))
    if contains:
        overlap = 1.0
    stripped = output_text.strip()
    has_digits = bool(re.search(r"\d", output_text))
    garbage = (not stripped) or (
        not has_digits and bool(output_text) and len(set(output_text.split())) <= 2
    )
    return {
        "contains_expected_number": contains,
        "number_only_exact": stripped == expected,
        "digit_overlap": round(overlap, 6),
        "is_garbage": garbage,
    }


def verify_file_lock(lock_path: Path, expected_kind: str) -> tuple[Path, dict[str, Any], str]:
    project_root = project_root_from_lock(lock_path)
    lock = read_json(lock_path)
    if lock.get("schema_version") != 1 or lock.get("lock_kind") != expected_kind:
        raise ValueError(f"Expected a {expected_kind!r} lock")
    for item in lock.get("files", []):
        path = resolve_project_path(project_root, item["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Locked file missing: {path}")
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Locked file size changed: {item['relative_path']}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Locked file hash changed: {item['relative_path']}")
    return project_root, lock, sha256_file(lock_path)


def locked_path(lock: dict[str, Any], project_root: Path, role: str) -> Path:
    matches = [item for item in lock["files"] if item["role"] == role]
    if len(matches) != 1:
        raise ValueError(f"Expected one locked file with role={role!r}")
    return resolve_project_path(project_root, matches[0]["relative_path"])

