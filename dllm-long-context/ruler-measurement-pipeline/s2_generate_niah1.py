"""Generate and freeze the RULER niah_single_1 efficiency prompt bank.

This script prepares token IDs only.  It does not load an UltraLLaDA model and
does not import, copy, or reimplement generate.py.

Formal design:
  * 13 exact input-length targets from 1K through 128K.
  * 10 paired semantic samples at every length.
  * Needle depths 10/30/50/70/90 percent, two samples per depth.
  * Final token IDs include the UltraLLaDA chat template and RULER answer prefix.
  * Every file is a List[torch.LongTensor] on CPU.

The clean NVIDIA RULER checkout remains read-only input to this wrapper.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
GENERATOR_VERSION = 1
TASK_NAME = "niah_single_1"
DEFAULT_GLOBAL_SEED = 1234
DEFAULT_TOKENIZER_ID = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_TOKENIZER_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
DEFAULT_LENGTHS = (
    1024,
    2048,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
)
DEPTH_LEVELS = (0.10, 0.30, 0.50, 0.70, 0.90)
SAMPLES_PER_DEPTH = 2
SAMPLE_COUNT = len(DEPTH_LEVELS) * SAMPLES_PER_DEPTH
DEPTH_TOLERANCE = 0.01
LOCAL_SEARCH_RADIUS = 24
EXPECTED_RULER_ORIGIN = "https://github.com/NVIDIA/RULER.git"
NEEDLE_TEMPLATE = "One of the special magic numbers for {key} is: {value}."


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_input_ids(input_ids: Iterable[int]) -> str:
    values = [int(value) for value in input_ids]
    if not values:
        return sha256_bytes(b"")
    return sha256_bytes(struct.pack(f"<{len(values)}q", *values))


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return repr(value)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def derive_seed(global_seed: int, *parts: object) -> int:
    payload = "|".join([str(global_seed), *[str(part) for part in parts]])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def allowed_length_error(target_tokens: int) -> int:
    return min(8, math.floor(target_tokens * 0.002))


def depth_schedule(global_seed: int) -> list[float]:
    schedule = list(DEPTH_LEVELS) * SAMPLES_PER_DEPTH
    random.Random(derive_seed(global_seed, TASK_NAME, "depth_schedule")).shuffle(
        schedule
    )
    return schedule


def run_git(ruler_dir: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ruler_dir), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to verify the RULER checkout") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise RuntimeError(f"RULER git verification failed: {stderr}") from exc
    return result.stdout.strip()


def verify_ruler_checkout(ruler_dir: Path) -> dict[str, Any]:
    ruler_dir = ruler_dir.resolve()
    required_files = {
        "task_config": ruler_dir / "scripts" / "synthetic.yaml",
        "task_constants": ruler_dir
        / "scripts"
        / "data"
        / "synthetic"
        / "constants.py",
        "niah_source": ruler_dir
        / "scripts"
        / "data"
        / "synthetic"
        / "niah.py",
    }
    for role, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"RULER {role} is missing: {path}")

    origin = run_git(ruler_dir, "remote", "get-url", "origin")
    normalized_origin = origin.removesuffix("/")
    accepted_origins = {
        EXPECTED_RULER_ORIGIN,
        "git@github.com:NVIDIA/RULER.git",
    }
    if normalized_origin not in accepted_origins:
        raise ValueError(f"Unexpected RULER origin: {origin}")
    status = run_git(ruler_dir, "status", "--porcelain")
    if status:
        raise RuntimeError(
            "RULER checkout is not clean; keep generated data outside the repository"
        )
    commit = run_git(ruler_dir, "rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError(f"Unexpected RULER commit: {commit}")

    return {
        "directory": str(ruler_dir),
        "origin": origin,
        "branch": run_git(ruler_dir, "branch", "--show-current"),
        "commit": commit,
        "dirty": False,
        "files": {
            role: {
                "path": str(path.relative_to(ruler_dir).as_posix()),
                "sha256": sha256_file(path),
            }
            for role, path in required_files.items()
        },
    }


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install the RULER Python requirements first."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def load_python_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_noise_text(niah_source_path: Path) -> str:
    tree = ast.parse(niah_source_path.read_text(encoding="utf-8"))
    candidates = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "The grass is green." in node.value
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely extract the official RULER noise haystack"
        )
    return candidates[0]


def load_ruler_task_definition(ruler_dir: Path) -> dict[str, Any]:
    synthetic_path = ruler_dir / "scripts" / "synthetic.yaml"
    constants_path = (
        ruler_dir / "scripts" / "data" / "synthetic" / "constants.py"
    )
    niah_source_path = ruler_dir / "scripts" / "data" / "synthetic" / "niah.py"

    synthetic = load_yaml(synthetic_path)
    task_config = synthetic.get(TASK_NAME)
    expected_args = {
        "type_haystack": "noise",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
        "num_needle_k": 1,
        "num_needle_v": 1,
        "num_needle_q": 1,
    }
    if not isinstance(task_config, dict) or task_config.get("task") != "niah":
        raise ValueError("RULER niah_single_1 task definition changed")
    if task_config.get("args") != expected_args:
        raise ValueError(
            f"Unexpected niah_single_1 arguments: {task_config.get('args')}"
        )

    constants_module = load_python_module(constants_path, "ruler_s2_constants")
    task_base = constants_module.TASKS["niah"]
    return {
        "task_config": task_config,
        "task_template": str(task_base["template"]),
        "answer_prefix": str(task_base["answer_prefix"]),
        "official_generation_budget": int(task_base["tokens_to_generate"]),
        "noise_text": extract_noise_text(niah_source_path),
    }


def singularize_ruler_text(text: str) -> str:
    # This intentionally follows NVIDIA/RULER scripts/data/synthetic/niah.py.
    text = text.replace("Some", "A")
    text = text.replace("are all", "is")
    text = text.replace("are", "is")
    text = text.replace("answers", "answer")
    return text


def load_wonderwords_lists() -> tuple[list[str], list[str], str]:
    try:
        from wonderwords import random_word
    except ImportError as exc:
        raise RuntimeError(
            "wonderwords is required because official RULER uses its word lists"
        ) from exc
    adjectives = sorted(
        set(random_word._get_words_from_text_file("adjectivelist.txt"))
    )
    nouns = sorted(set(random_word._get_words_from_text_file("nounlist.txt")))
    if not adjectives or not nouns:
        raise RuntimeError("wonderwords returned empty adjective or noun lists")
    try:
        version = importlib.metadata.version("wonderwords")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return adjectives, nouns, version


def resolved_commit_candidates(tokenizer: Any) -> list[str]:
    candidates: list[str] = []
    for source in (tokenizer, getattr(tokenizer, "config", None)):
        value = getattr(source, "_commit_hash", None)
        if isinstance(value, str):
            candidates.append(value)
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if isinstance(init_kwargs, dict):
        value = init_kwargs.get("_commit_hash")
        if isinstance(value, str):
            candidates.append(value)
    return sorted(set(candidates))


def load_tokenizer(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required. Run this script in the UltraLLaDA environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id,
        revision=args.tokenizer_revision,
        code_revision=args.tokenizer_revision,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for exact needle spans")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("The pinned tokenizer does not expose a chat template")

    resolved_commits = resolved_commit_candidates(tokenizer)
    mismatches = [
        commit for commit in resolved_commits if commit != args.tokenizer_revision
    ]
    if mismatches:
        raise ValueError(f"Tokenizer resolved to unexpected commits: {mismatches}")

    fingerprint_payload = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "special_tokens_map": json_safe(getattr(tokenizer, "special_tokens_map", {})),
        "chat_template": str(tokenizer.chat_template),
    }
    metadata = {
        "id": args.tokenizer_id,
        "requested_revision": args.tokenizer_revision,
        "resolved_commit_candidates": resolved_commits,
        "transformers_version": transformers.__version__,
        "is_fast": True,
        "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
        "fingerprint_sha256": sha256_text(
            json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=True)
        ),
    }
    return tokenizer, metadata


def tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if len(input_ids) != len(offsets):
        raise ValueError("Tokenizer returned inconsistent input IDs and offsets")
    return input_ids, offsets


def char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    indexes = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not indexes:
        raise ValueError("Could not map needle character span to token offsets")
    return min(indexes), max(indexes) + 1


def decode_slice(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def create_noise_pool(
    tokenizer: Any, noise_text: str, required_tokens: int
) -> list[int]:
    one_block_ids = tokenizer(
        noise_text, add_special_tokens=False, truncation=False
    )["input_ids"]
    if not one_block_ids:
        raise RuntimeError("RULER noise haystack tokenized to zero tokens")
    repeats = max(2, math.ceil(required_tokens / len(one_block_ids)) + 4)
    while True:
        repeated_text = (noise_text + "\n") * repeats
        pool = [
            int(value)
            for value in tokenizer(
                repeated_text, add_special_tokens=False, truncation=False
            )["input_ids"]
        ]
        if len(pool) >= required_tokens:
            return pool
        repeats *= 2


def format_prompt_parts(
    task_definition: dict[str, Any], context: str, key: str
) -> tuple[str, str]:
    task_template = singularize_ruler_text(task_definition["task_template"])
    answer_prefix = singularize_ruler_text(task_definition["answer_prefix"])
    user_text = task_template.format(
        type_needle_v="number",
        context=context,
        query=key,
    )
    assistant_prefix = answer_prefix.format(
        type_needle_v="number",
        query=key,
    )
    return user_text, assistant_prefix


def build_candidate(
    *,
    tokenizer: Any,
    task_definition: dict[str, Any],
    noise_pool_ids: list[int],
    filler_token_budget: int,
    requested_depth: float,
    key: str,
    value: str,
    include_details: bool,
) -> dict[str, Any]:
    split = int(round(filler_token_budget * requested_depth))
    before_text = decode_slice(tokenizer, noise_pool_ids[:split])
    after_text = decode_slice(
        tokenizer, noise_pool_ids[split:filler_token_budget]
    )
    needle_text = NEEDLE_TEMPLATE.format(key=key, value=value)
    context_parts = [part for part in (before_text, needle_text, after_text) if part]
    context = "\n".join(context_parts)
    user_text, assistant_prefix = format_prompt_parts(task_definition, context, key)
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    full_prompt = chat_text + assistant_prefix

    if not include_details:
        input_ids = tokenizer(
            full_prompt,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        return {"exact_input_tokens": len(input_ids)}

    input_ids, offsets = tokenize_with_offsets(tokenizer, full_prompt)
    needle_char_start = full_prompt.index(needle_text)
    needle_char_end = needle_char_start + len(needle_text)
    needle_start, needle_end = char_span_to_token_span(
        offsets, needle_char_start, needle_char_end
    )
    before_tokens = len(
        tokenizer(before_text, add_special_tokens=False, truncation=False)["input_ids"]
    )
    after_tokens = len(
        tokenizer(after_text, add_special_tokens=False, truncation=False)["input_ids"]
    )
    actual_depth = before_tokens / max(1, before_tokens + after_tokens)
    return {
        "input_ids": input_ids,
        "exact_input_tokens": len(input_ids),
        "full_prompt": full_prompt,
        "needle_text": needle_text,
        "needle_token_start": needle_start,
        "needle_token_end_exclusive": needle_end,
        "target_needle_depth": requested_depth,
        "actual_needle_depth": actual_depth,
        "haystack_tokens_before_needle": before_tokens,
        "haystack_tokens_after_needle": after_tokens,
        "filler_token_budget": filler_token_budget,
    }


def find_exact_candidate(
    *,
    tokenizer: Any,
    task_definition: dict[str, Any],
    noise_pool_ids: list[int],
    target_tokens: int,
    requested_depth: float,
    key: str,
    value: str,
) -> dict[str, Any]:
    length_cache: dict[int, int] = {}

    def candidate_length(filler_budget: int) -> int:
        if filler_budget not in length_cache:
            length_cache[filler_budget] = int(
                build_candidate(
                    tokenizer=tokenizer,
                    task_definition=task_definition,
                    noise_pool_ids=noise_pool_ids,
                    filler_token_budget=filler_budget,
                    requested_depth=requested_depth,
                    key=key,
                    value=value,
                    include_details=False,
                )["exact_input_tokens"]
            )
        return length_cache[filler_budget]

    minimum_length = candidate_length(0)
    if minimum_length > target_tokens:
        raise ValueError(
            f"Target {target_tokens} is shorter than the fixed prompt overhead "
            f"({minimum_length} tokens)"
        )

    lower = 0
    upper = min(target_tokens, len(noise_pool_ids))
    while candidate_length(upper) < target_tokens:
        if upper >= len(noise_pool_ids):
            raise RuntimeError("Noise pool is too short for the requested length")
        lower = upper + 1
        upper = min(len(noise_pool_ids), max(upper + 1, upper * 2))

    while lower < upper:
        midpoint = (lower + upper) // 2
        if candidate_length(midpoint) < target_tokens:
            lower = midpoint + 1
        else:
            upper = midpoint
    crossing = lower

    budgets = range(
        max(0, crossing - LOCAL_SEARCH_RADIUS),
        min(len(noise_pool_ids), crossing + LOCAL_SEARCH_RADIUS) + 1,
    )
    ranked_budgets = sorted(
        budgets,
        key=lambda budget: (
            abs(candidate_length(budget) - target_tokens),
            candidate_length(budget) > target_tokens,
            budget,
        ),
    )
    tolerance = allowed_length_error(target_tokens)
    inspected: list[dict[str, Any]] = []
    for budget in ranked_budgets:
        if abs(candidate_length(budget) - target_tokens) > tolerance:
            break
        detailed = build_candidate(
            tokenizer=tokenizer,
            task_definition=task_definition,
            noise_pool_ids=noise_pool_ids,
            filler_token_budget=budget,
            requested_depth=requested_depth,
            key=key,
            value=value,
            include_details=True,
        )
        detailed["length_error"] = detailed["exact_input_tokens"] - target_tokens
        detailed["allowed_length_error"] = tolerance
        detailed["needle_depth_error"] = (
            detailed["actual_needle_depth"] - requested_depth
        )
        inspected.append(detailed)
        if (
            detailed["length_error"] == 0
            and abs(detailed["needle_depth_error"]) <= DEPTH_TOLERANCE
        ):
            return detailed

    valid = [
        candidate
        for candidate in inspected
        if abs(candidate["length_error"]) <= tolerance
        and abs(candidate["needle_depth_error"]) <= DEPTH_TOLERANCE
    ]
    if not valid:
        best_length_error = min(
            abs(candidate_length(budget) - target_tokens) for budget in budgets
        )
        raise RuntimeError(
            f"Could not satisfy length/depth tolerances for target={target_tokens}, "
            f"depth={requested_depth:.2f}; best length error={best_length_error}"
        )
    return min(
        valid,
        key=lambda candidate: (
            abs(candidate["length_error"]),
            abs(candidate["needle_depth_error"]),
            candidate["filler_token_budget"],
        ),
    )


def semantic_instances(
    *, global_seed: int, adjectives: list[str], nouns: list[str]
) -> list[dict[str, Any]]:
    schedule = depth_schedule(global_seed)
    instances: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    for sample_index, depth in enumerate(schedule):
        sample_seed = derive_seed(
            global_seed, TASK_NAME, sample_index, "semantic", GENERATOR_VERSION
        )
        rng = random.Random(sample_seed)
        for _ in range(100):
            key = f"{rng.choice(adjectives)}-{rng.choice(nouns)}"
            value = str(rng.randint(1_000_000, 9_999_999))
            if (key, value) not in used_pairs:
                used_pairs.add((key, value))
                break
        else:
            raise RuntimeError("Could not create unique NIAH key/value pairs")
        instances.append(
            {
                "sample_index": sample_index,
                "sample_id": f"S{sample_index:02d}",
                "sample_seed": sample_seed,
                "key": key,
                "value": value,
                "target_needle_depth": depth,
            }
        )
    return instances


def torch_save_atomic(torch_module: Any, path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    torch_module.save(payload, temp_path)
    os.replace(temp_path, path)


def torch_load_cpu(torch_module: Any, path: Path) -> Any:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def verify_tensor_file(
    torch_module: Any,
    path: Path,
    expected_lengths: list[int],
) -> None:
    payload = torch_load_cpu(torch_module, path)
    if not isinstance(payload, list) or len(payload) != SAMPLE_COUNT:
        raise ValueError(f"Unexpected tensor list in {path}")
    actual_lengths: list[int] = []
    for tensor in payload:
        if not isinstance(tensor, torch_module.Tensor):
            raise TypeError(f"Non-tensor item in {path}")
        if tensor.device.type != "cpu" or tensor.dtype != torch_module.int64:
            raise TypeError(f"Tensor dtype/device mismatch in {path}")
        if tensor.ndim != 1:
            raise ValueError(f"Tensor must be one-dimensional in {path}")
        actual_lengths.append(int(tensor.numel()))
    if actual_lengths != expected_lengths:
        raise ValueError(f"Tensor lengths do not match the manifest in {path}")


def write_checksums(root: Path, relative_paths: list[Path]) -> Path:
    checksum_path = root / "checksums.sha256"
    lines = [
        f"{sha256_file(root / relative_path)}  {relative_path.as_posix()}"
        for relative_path in sorted(relative_paths, key=lambda path: path.as_posix())
    ]
    temp_path = checksum_path.with_name(checksum_path.name + ".tmp")
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(temp_path, checksum_path)
    return checksum_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frozen UltraLLaDA token IDs for RULER niah_single_1"
    )
    parser.add_argument("--ruler-dir", type=Path, default=Path("RULER"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prompts/ruler_v1/efficiency/niah_single_1"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_GLOBAL_SEED)
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument(
        "--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact Hugging Face; require the pinned tokenizer in cache",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run dependency-free design checks without generating prompt files",
    )
    return parser.parse_args()


def self_test() -> None:
    assert allowed_length_error(1024) == 2
    assert allowed_length_error(2048) == 4
    assert allowed_length_error(4096) == 8
    assert allowed_length_error(131072) == 8
    first = depth_schedule(DEFAULT_GLOBAL_SEED)
    second = depth_schedule(DEFAULT_GLOBAL_SEED)
    assert first == second
    assert len(first) == SAMPLE_COUNT
    assert Counter(first) == Counter({depth: 2 for depth in DEPTH_LEVELS})
    seeds = {
        derive_seed(DEFAULT_GLOBAL_SEED, TASK_NAME, index, "semantic")
        for index in range(SAMPLE_COUNT)
    }
    assert len(seeds) == SAMPLE_COUNT
    print("S2 design self-test: PASSED")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not 0 <= args.seed < 2**63:
        raise ValueError("--seed must be in [0, 2**63)")

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required. Run this script in the UltraLLaDA environment."
        ) from exc

    ruler_dir = args.ruler_dir.resolve()
    output_dir = args.output_dir.resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists():
        raise FileExistsError(
            f"Frozen output already exists: {output_dir}. Use a new version directory."
        )
    if staging_dir.exists():
        raise FileExistsError(
            f"Incomplete staging directory exists: {staging_dir}. Inspect it first."
        )

    ruler_metadata = verify_ruler_checkout(ruler_dir)
    task_definition = load_ruler_task_definition(ruler_dir)
    tokenizer, tokenizer_metadata = load_tokenizer(args)
    adjectives, nouns, wonderwords_version = load_wonderwords_lists()
    instances = semantic_instances(
        global_seed=args.seed,
        adjectives=adjectives,
        nouns=nouns,
    )
    noise_pool_ids = create_noise_pool(
        tokenizer,
        task_definition["noise_text"],
        required_tokens=max(DEFAULT_LENGTHS) * 2,
    )

    staging_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    try:
        for target_tokens in DEFAULT_LENGTHS:
            tensors: list[Any] = []
            expected_lengths: list[int] = []
            print(f"Generating {TASK_NAME} at {target_tokens} input tokens...")
            for instance in instances:
                built = find_exact_candidate(
                    tokenizer=tokenizer,
                    task_definition=task_definition,
                    noise_pool_ids=noise_pool_ids,
                    target_tokens=target_tokens,
                    requested_depth=float(instance["target_needle_depth"]),
                    key=str(instance["key"]),
                    value=str(instance["value"]),
                )
                input_tensor = torch.tensor(
                    built["input_ids"], dtype=torch.long, device="cpu"
                )
                tensors.append(input_tensor)
                expected_lengths.append(int(input_tensor.numel()))
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "generator_version": GENERATOR_VERSION,
                    "task": TASK_NAME,
                    "sample_id": instance["sample_id"],
                    "sample_index": instance["sample_index"],
                    "global_seed": args.seed,
                    "sample_seed": instance["sample_seed"],
                    "key": instance["key"],
                    "answer": instance["value"],
                    "target_input_tokens": target_tokens,
                    "exact_input_tokens": built["exact_input_tokens"],
                    "length_error": built["length_error"],
                    "allowed_length_error": built["allowed_length_error"],
                    "target_needle_depth": built["target_needle_depth"],
                    "actual_needle_depth": round(
                        float(built["actual_needle_depth"]), 8
                    ),
                    "needle_depth_error": round(
                        float(built["needle_depth_error"]), 8
                    ),
                    "needle_token_start": built["needle_token_start"],
                    "needle_token_end_exclusive": built[
                        "needle_token_end_exclusive"
                    ],
                    "haystack_tokens_before_needle": built[
                        "haystack_tokens_before_needle"
                    ],
                    "haystack_tokens_after_needle": built[
                        "haystack_tokens_after_needle"
                    ],
                    "filler_token_budget": built["filler_token_budget"],
                    "official_generation_budget": task_definition[
                        "official_generation_budget"
                    ],
                    "tensor_file": f"tokens/{target_tokens}.pt",
                    "tensor_index": int(instance["sample_index"]),
                    "input_ids_sha256": sha256_input_ids(built["input_ids"]),
                    "prompt_text_sha256": sha256_text(built["full_prompt"]),
                }
                records.append(record)

            tensor_relative_path = Path("tokens") / f"{target_tokens}.pt"
            tensor_path = staging_dir / tensor_relative_path
            torch_save_atomic(torch, tensor_path, tensors)
            verify_tensor_file(torch, tensor_path, expected_lengths)
            artifact_paths.append(tensor_relative_path)

        run_order = list(range(len(records)))
        run_order_seed = derive_seed(args.seed, TASK_NAME, "run_order")
        random.Random(run_order_seed).shuffle(run_order)
        for run_order_index, record_index in enumerate(run_order):
            records[record_index]["run_order_index"] = run_order_index

        expected_record_count = len(DEFAULT_LENGTHS) * SAMPLE_COUNT
        if len(records) != expected_record_count:
            raise AssertionError(
                f"Expected {expected_record_count} records, found {len(records)}"
            )
        for target_tokens in DEFAULT_LENGTHS:
            cell = [
                record
                for record in records
                if record["target_input_tokens"] == target_tokens
            ]
            if len(cell) != SAMPLE_COUNT:
                raise AssertionError(f"Unbalanced sample count at {target_tokens}")
            depth_counts = Counter(record["target_needle_depth"] for record in cell)
            if depth_counts != Counter({depth: 2 for depth in DEPTH_LEVELS}):
                raise AssertionError(f"Unbalanced depth schedule at {target_tokens}")

        samples_relative_path = Path("manifests") / "samples.jsonl"
        efficiency_relative_path = Path("manifests") / "efficiency.jsonl"
        write_jsonl_atomic(staging_dir / samples_relative_path, records)
        write_jsonl_atomic(staging_dir / efficiency_relative_path, records)
        artifact_paths.extend([samples_relative_path, efficiency_relative_path])

        config_relative_path = Path("generation_config.json")
        generation_config = {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "created_at_utc": utc_now(),
            "task": TASK_NAME,
            "purpose": "latency_and_vram_prompt_bank",
            "global_seed": args.seed,
            "lengths": list(DEFAULT_LENGTHS),
            "sample_count_per_length": SAMPLE_COUNT,
            "depth_levels": list(DEPTH_LEVELS),
            "samples_per_depth": SAMPLES_PER_DEPTH,
            "depth_tolerance": DEPTH_TOLERANCE,
            "length_tolerance_rule": "min(8, floor(target_input_tokens * 0.002))",
            "run_order_seed": run_order_seed,
            "ruler": ruler_metadata,
            "ruler_task": {
                "name": TASK_NAME,
                "config": task_definition["task_config"],
                "official_generation_budget": task_definition[
                    "official_generation_budget"
                ],
                "noise_text_sha256": sha256_text(task_definition["noise_text"]),
            },
            "tokenizer": tokenizer_metadata,
            "wonderwords_version": wonderwords_version,
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "generator_source_sha256": sha256_file(Path(__file__).resolve()),
            "model_loaded": False,
            "generate_py_imported_or_reimplemented": False,
        }
        write_json_atomic(staging_dir / config_relative_path, generation_config)
        artifact_paths.append(config_relative_path)

        checksum_path = write_checksums(staging_dir, artifact_paths)
        lock_relative_path = Path("generation_complete.lock.json")
        lock_payload = {
            "schema_version": SCHEMA_VERSION,
            "lock_kind": "s2_ruler_niah1_efficiency",
            "created_at_utc": utc_now(),
            "task": TASK_NAME,
            "record_count": len(records),
            "tensor_file_count": len(DEFAULT_LENGTHS),
            "checksums_sha256": sha256_file(checksum_path),
            "ruler_commit": ruler_metadata["commit"],
            "tokenizer_revision": args.tokenizer_revision,
        }
        write_json_atomic(staging_dir / lock_relative_path, lock_payload)

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_dir, output_dir)
    except Exception:
        print(
            f"Generation failed. Staging data was left for inspection at: {staging_dir}",
            file=sys.stderr,
        )
        raise

    print(f"Frozen prompt bank: {output_dir}")
    print(f"Records: {len(records)}")
    print(f"RULER commit: {ruler_metadata['commit']}")
    print(f"Tokenizer revision: {args.tokenizer_revision}")
    print("No model or generate.py was loaded.")


if __name__ == "__main__":
    main()
