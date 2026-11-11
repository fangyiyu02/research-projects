"""Independently validate the frozen RULER v1 accuracy prompt bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

import s2_generate_niah1 as common


EXPECTED_TASKS = (
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
EXPECTED_LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)
EXPECTED_SAMPLES = 50


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    lock = read_json(root / "generation_complete.lock.json")
    config = read_json(root / "generation_config.json")
    records = read_jsonl(root / "manifests" / "accuracy.jsonl")

    assert tuple(lock["tasks"]) == EXPECTED_TASKS
    assert tuple(lock["lengths"]) == EXPECTED_LENGTHS
    assert lock["samples_per_task_length"] == EXPECTED_SAMPLES
    assert lock["record_count"] == 3600
    assert lock["tensor_file_count"] == 72
    assert "niah_single_1" not in lock["tasks"]
    assert config["excluded_existing_benchmarks"] == ["niah_single_1"]
    assert config["model_loaded"] is False
    assert config["generate_py_imported_or_reimplemented"] is False
    assert len(records) == 3600

    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    counts: Counter[tuple[str, int]] = Counter()
    paired: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    max_length_error = 0
    max_niah_depth_error = 0.0
    max_qa_depth_error = 0.0
    max_vt_depth_error = 0.0
    for record in records:
        task = record["task"]
        target = int(record["target_input_tokens"])
        sample_index = int(record["sample_index"])
        assert task in EXPECTED_TASKS
        assert target in EXPECTED_LENGTHS
        assert 0 <= sample_index < EXPECTED_SAMPLES
        key = (task, target, sample_index)
        assert key not in indexed
        indexed[key] = record
        counts[(task, target)] += 1
        paired[(task, sample_index)].append(record)
        error = abs(int(record["length_error"]))
        assert error == abs(int(record["exact_input_tokens"]) - target)
        assert error <= int(record["allowed_length_error"])
        max_length_error = max(max_length_error, error)
        for position in record.get("needle_positions", []):
            depth_error = abs(
                float(position["actual_depth"]) - float(position["target_depth"])
            )
            assert depth_error <= 0.01 + 1e-12
            max_niah_depth_error = max(max_niah_depth_error, depth_error)
        for position in record.get("chain_positions", []):
            depth_error = abs(
                float(position["actual_depth"]) - float(position["target_depth"])
            )
            assert depth_error <= 0.01 + 1e-12
            max_vt_depth_error = max(max_vt_depth_error, depth_error)
        if "evidence_position" in record:
            position = record["evidence_position"]
            depth_error = abs(
                float(position["actual_depth"]) - float(position["target_depth"])
            )
            assert depth_error <= 0.01 + 1e-12
            max_qa_depth_error = max(max_qa_depth_error, depth_error)

    assert all(counts[(task, target)] == 50 for task in EXPECTED_TASKS for target in EXPECTED_LENGTHS)
    for sample_records in paired.values():
        assert len(sample_records) == 6
        assert len({row["sample_seed"] for row in sample_records}) == 1
        if "query" in sample_records[0]:
            assert len({row["query"] for row in sample_records}) == 1

    tensor_paths = sorted((root / "tokens").rglob("*.pt"))
    assert len(tensor_paths) == 72
    tensor_records_checked = 0
    for path in tensor_paths:
        task = path.parent.name
        target = int(path.stem)
        payload = common.torch_load_cpu(torch, path)
        assert isinstance(payload, list) and len(payload) == EXPECTED_SAMPLES
        for sample_index, tensor in enumerate(payload):
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.int64
            assert tensor.device.type == "cpu"
            assert tensor.ndim == 1
            record = indexed[(task, target, sample_index)]
            assert tensor.numel() == int(record["exact_input_tokens"])
            assert common.sha256_input_ids(tensor.tolist()) == record["input_ids_sha256"]
            for field in ("needle_positions", "chain_positions"):
                for position in record.get(field, []):
                    assert 0 <= int(position["token_start"]) < int(position["token_end_exclusive"]) <= tensor.numel()
            if "evidence_position" in record:
                position = record["evidence_position"]
                assert 0 <= int(position["token_start"]) < int(position["token_end_exclusive"]) <= tensor.numel()
            tensor_records_checked += 1

    checksum_lines = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == 86
    for line in checksum_lines:
        expected, relative = line.split("  ", 1)
        assert sha256_file(root / Path(relative)) == expected
    assert sha256_file(root / "checksums.sha256") == lock["checksums_sha256"]

    return {
        "root": str(root),
        "tasks": len(EXPECTED_TASKS),
        "lengths": len(EXPECTED_LENGTHS),
        "tensor_files": len(tensor_paths),
        "records": len(records),
        "tensor_records_checked": tensor_records_checked,
        "checksum_entries": len(checksum_lines),
        "max_absolute_length_error": max_length_error,
        "max_niah_depth_error": max_niah_depth_error,
        "max_vt_depth_error": max_vt_depth_error,
        "max_qa_depth_error": max_qa_depth_error,
        "status": "PASSED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-bank",
        type=Path,
        default=Path("prompts/ruler_v1/accuracy"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.prompt_bank), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
