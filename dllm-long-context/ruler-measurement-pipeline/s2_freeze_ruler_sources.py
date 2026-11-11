#!/usr/bin/env python3
"""Validate and freeze the external source data used by RULER accuracy tasks.

This script never downloads data.  It validates files already placed in
``prompts/ruler_v1/source_data.building``, records their provenance and hashes,
then atomically renames that staging directory to ``source_data``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPECTED_FILES = {
    "paul_graham_essays": {
        "filename": "PaulGrahamEssays.json",
        "source": "NVIDIA/RULER official download_paulgraham_essay.py",
        "url_list": "PaulGrahamEssays_URLs.txt",
    },
    "squad_v2_dev": {
        "filename": "squad.json",
        "source_url": (
            "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
        ),
    },
    "hotpotqa_distractor_dev": {
        "filename": "hotpotqa.json",
        "source_url": (
            "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
            "7e54db4656209750ff487f6fdf8e39a66dba136b/"
            "hotpot_dev_distractor_v1.json"
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ruler_commit(ruler_dir: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ruler_dir.as_posix()}",
            "-C",
            str(ruler_dir),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_sources(staging_dir: Path) -> dict[str, Any]:
    paths = {
        name: staging_dir / spec["filename"]
        for name, spec in EXPECTED_FILES.items()
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty source file: {path}")

    paul = json.loads(paths["paul_graham_essays"].read_text(encoding="utf-8"))
    paul_text = paul.get("text") if isinstance(paul, dict) else None
    if not isinstance(paul_text, str) or len(paul_text) < 2_000_000:
        raise ValueError("PaulGrahamEssays.json is unexpectedly short")

    squad = json.loads(paths["squad_v2_dev"].read_text(encoding="utf-8"))
    squad_articles = squad.get("data") if isinstance(squad, dict) else None
    if not isinstance(squad_articles, list) or not squad_articles:
        raise ValueError("squad.json does not have the expected SQuAD structure")
    squad_qas = sum(
        len(paragraph.get("qas", []))
        for article in squad_articles
        for paragraph in article.get("paragraphs", [])
    )
    if squad_qas < 10_000:
        raise ValueError("squad.json contains too few questions")

    hotpot = json.loads(paths["hotpotqa_distractor_dev"].read_text(encoding="utf-8"))
    if not isinstance(hotpot, list) or len(hotpot) < 7_000:
        raise ValueError("hotpotqa.json is unexpectedly short")

    url_list_path = staging_dir / "PaulGrahamEssays_URLs.txt"
    url_count = len(
        [
            line
            for line in url_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    if url_count < 200:
        raise ValueError("Paul Graham URL list is unexpectedly short")

    return {
        "paul_graham_characters": len(paul_text),
        "paul_graham_url_count": url_count,
        "squad_article_count": len(squad_articles),
        "squad_question_count": squad_qas,
        "hotpot_example_count": len(hotpot),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("prompts/ruler_v1/source_data.building"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prompts/ruler_v1/source_data"),
    )
    parser.add_argument("--ruler-dir", type=Path, default=Path("RULER"))
    args = parser.parse_args()

    staging_dir = args.staging_dir.resolve()
    output_dir = args.output_dir.resolve()
    ruler_dir = args.ruler_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Frozen source directory already exists: {output_dir}")
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"Source staging directory is missing: {staging_dir}")

    validation = validate_sources(staging_dir)
    created_at = datetime.now(timezone.utc).isoformat()
    file_records: dict[str, Any] = {}
    for name, spec in EXPECTED_FILES.items():
        path = staging_dir / spec["filename"]
        file_records[name] = {
            **spec,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    supporting_files = [
        staging_dir / "download_paulgraham_essay.py",
        staging_dir / "PaulGrahamEssays_URLs.txt",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at,
        "download_encoding": "UTF-8 (PYTHONUTF8=1)",
        "ruler_commit": ruler_commit(ruler_dir),
        "files": file_records,
        "validation": validation,
        "supporting_files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in supporting_files
        },
    }
    manifest_path = staging_dir / "sources_manifest.json"
    write_json_atomic(manifest_path, manifest)

    checksum_inputs = [
        *(staging_dir / spec["filename"] for spec in EXPECTED_FILES.values()),
        *supporting_files,
        manifest_path,
    ]
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}" for path in sorted(checksum_inputs)
    ]
    checksum_path = staging_dir / "checksums.sha256"
    checksum_path.write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n"
    )
    lock = {
        "schema_version": SCHEMA_VERSION,
        "lock_kind": "ruler_v1_external_sources",
        "created_at_utc": created_at,
        "checksums_sha256": sha256_file(checksum_path),
        "ruler_commit": manifest["ruler_commit"],
    }
    write_json_atomic(staging_dir / "source_data_complete.lock.json", lock)
    os.replace(staging_dir, output_dir)

    print(f"Frozen RULER source data: {output_dir}")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
