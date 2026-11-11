#!/usr/bin/env python3
"""Generate and freeze the remaining NVIDIA RULER accuracy prompt bank.

The existing ``niah_single_1`` efficiency bank is deliberately excluded.  This
wrapper generates the other 12 official RULER benchmarks at six exact input
lengths with 50 paired semantic samples per task and length.  It prepares CPU
token IDs only: no model is loaded and model ``generate.py`` is never imported,
copied, or reimplemented.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import struct
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

import s2_generate_niah1 as common


SCHEMA_VERSION = 1
GENERATOR_VERSION = 1
GLOBAL_SEED = 1234
LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)
SAMPLES_PER_TASK = 50
DEPTH_LEVELS = (0.10, 0.30, 0.50, 0.70, 0.90)
DEPTH_TOLERANCE = 0.01
LOCAL_SEARCH_RADIUS = 24
TASK_NAMES = (
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
OFFICIAL_TASK_SOURCES = (
    "niah.py",
    "variable_tracking.py",
    "common_words_extraction.py",
    "freq_words_extraction.py",
    "qa.py",
    "constants.py",
)
NEEDLE_TEMPLATE = "One of the special magic {type_needle_v} for {key} is: {value}."
NOISE_TEXT = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)


@dataclass
class RenderedPrompt:
    full_prompt: str
    metadata: dict[str, Any]


@dataclass
class TaskResources:
    tokenizer: Any
    task_name: str
    config: dict[str, Any]
    task_template: str
    answer_prefix: str
    generation_budget: int
    source_data: Mapping[str, Any]
    words: Mapping[str, list[str]]
    shared_pools: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_seed(global_seed: int, *parts: object) -> int:
    return common.derive_seed(global_seed, *parts)


def allowed_error(target: int) -> int:
    return common.allowed_length_error(target)


def encode(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    ]


def decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def format_chat(tokenizer: Any, user_text: str, assistant_prefix: str) -> str:
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return chat + assistant_prefix


def balanced_depth_schedule(task_name: str) -> list[float]:
    values = list(DEPTH_LEVELS) * (SAMPLES_PER_TASK // len(DEPTH_LEVELS))
    random.Random(derive_seed(GLOBAL_SEED, task_name, "depth_schedule")).shuffle(
        values
    )
    if Counter(values) != Counter({depth: 10 for depth in DEPTH_LEVELS}):
        raise AssertionError(f"Unbalanced depth schedule for {task_name}")
    return values


def ensure_token_pool(
    tokenizer: Any,
    text: str,
    minimum_tokens: int,
    *,
    separator: str = "\n",
) -> list[int]:
    one = encode(tokenizer, text)
    if not one:
        raise ValueError("Cannot build a token pool from empty text")
    repeats = max(1, math.ceil(minimum_tokens / len(one)) + 2)
    while True:
        pool = encode(tokenizer, (text + separator) * repeats)
        if len(pool) >= minimum_tokens:
            return pool
        repeats *= 2


def insert_at_depths(
    tokenizer: Any,
    filler_ids: Sequence[int],
    filler_budget: int,
    insertions: Sequence[tuple[float, str, str]],
) -> tuple[str, list[dict[str, Any]]]:
    if filler_budget < 0 or filler_budget > len(filler_ids):
        raise ValueError("Filler budget is outside the prepared token pool")
    ordered = sorted(insertions, key=lambda item: (item[0], item[2]))
    boundaries = [0]
    for depth, _, _ in ordered:
        boundaries.append(int(round(filler_budget * depth)))
    boundaries.append(filler_budget)
    boundaries = [max(0, min(filler_budget, value)) for value in boundaries]

    parts: list[str] = []
    chunk_token_counts: list[int] = []
    details: list[dict[str, Any]] = []
    for index, (_, insertion_text, insertion_id) in enumerate(ordered):
        chunk = decode(tokenizer, filler_ids[boundaries[index] : boundaries[index + 1]])
        if chunk:
            parts.append(chunk)
        chunk_token_counts.append(len(encode(tokenizer, chunk)) if chunk else 0)
        parts.append(insertion_text)
        details.append({"insertion_id": insertion_id, "text": insertion_text})
    final_chunk = decode(tokenizer, filler_ids[boundaries[-2] : boundaries[-1]])
    if final_chunk:
        parts.append(final_chunk)
    chunk_token_counts.append(len(encode(tokenizer, final_chunk)) if final_chunk else 0)

    total_haystack = sum(chunk_token_counts)
    cumulative = 0
    for index, ((target_depth, _, _), detail) in enumerate(zip(ordered, details)):
        cumulative += chunk_token_counts[index]
        detail["target_depth"] = target_depth
        detail["actual_depth"] = cumulative / max(1, total_haystack)
    return "\n".join(parts), details


def fit_exact_prompt(
    *,
    tokenizer: Any,
    target_tokens: int,
    maximum_budget: int,
    render: Callable[[int], RenderedPrompt],
    validate: Callable[[RenderedPrompt], None] | None = None,
) -> tuple[list[int], RenderedPrompt, int]:
    length_cache: dict[int, int] = {}
    render_cache: dict[int, RenderedPrompt] = {}

    def rendered(budget: int) -> RenderedPrompt:
        budget = max(0, min(maximum_budget, int(budget)))
        if budget not in render_cache:
            render_cache[budget] = render(budget)
        return render_cache[budget]

    def prompt_length(budget: int) -> int:
        budget = max(0, min(maximum_budget, int(budget)))
        if budget not in length_cache:
            length_cache[budget] = len(encode(tokenizer, rendered(budget).full_prompt))
        return length_cache[budget]

    minimum_length = prompt_length(0)
    if minimum_length > target_tokens:
        raise ValueError(
            f"Fixed prompt overhead {minimum_length} exceeds target {target_tokens}"
        )

    tolerance = allowed_error(target_tokens)
    budget = min(maximum_budget, max(0, target_tokens - minimum_length))
    visited: set[int] = set()
    best_budget = budget
    best_error = abs(prompt_length(budget) - target_tokens)
    for _ in range(12):
        budget = max(0, min(maximum_budget, budget))
        if budget in visited:
            break
        visited.add(budget)
        error = target_tokens - prompt_length(budget)
        if abs(error) < best_error:
            best_budget, best_error = budget, abs(error)
        if abs(error) <= tolerance:
            candidate = rendered(budget)
            if validate is not None:
                validate(candidate)
            return encode(tokenizer, candidate.full_prompt), candidate, budget
        budget += error

    candidates = range(
        max(0, best_budget - LOCAL_SEARCH_RADIUS),
        min(maximum_budget, best_budget + LOCAL_SEARCH_RADIUS) + 1,
    )
    ranked = sorted(
        candidates,
        key=lambda value: (
            abs(prompt_length(value) - target_tokens),
            prompt_length(value) > target_tokens,
            value,
        ),
    )
    validation_errors: list[str] = []
    for candidate_budget in ranked:
        error = prompt_length(candidate_budget) - target_tokens
        if abs(error) > tolerance:
            break
        candidate = rendered(candidate_budget)
        try:
            if validate is not None:
                validate(candidate)
        except ValueError as exc:
            validation_errors.append(str(exc))
            continue
        return encode(tokenizer, candidate.full_prompt), candidate, candidate_budget
    raise RuntimeError(
        f"Could not fit target={target_tokens}; best absolute error={best_error}; "
        f"validation_errors={validation_errors[:3]}"
    )


def load_task_definitions(ruler_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    synthetic_path = ruler_dir / "scripts" / "synthetic.yaml"
    constants_path = ruler_dir / "scripts" / "data" / "synthetic" / "constants.py"
    synthetic = common.load_yaml(synthetic_path)
    constants = common.load_python_module(constants_path, "ruler_accuracy_constants").TASKS
    definitions: dict[str, Any] = {}
    for task_name in TASK_NAMES:
        config = synthetic.get(task_name)
        if not isinstance(config, dict) or config.get("task") not in constants:
            raise ValueError(f"Invalid official RULER config for {task_name}")
        base = constants[config["task"]]
        definitions[task_name] = {
            "config": config,
            "task_template": str(base["template"]),
            "answer_prefix": str(base["answer_prefix"]),
            "generation_budget": int(base["tokens_to_generate"]),
        }
    return definitions, synthetic


def validate_source_lock(source_dir: Path) -> dict[str, Any]:
    lock_path = source_dir / "source_data_complete.lock.json"
    manifest_path = source_dir / "sources_manifest.json"
    checksum_path = source_dir / "checksums.sha256"
    for path in (lock_path, manifest_path, checksum_path):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen source-data artifact is missing: {path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("checksums_sha256") != common.sha256_file(checksum_path):
        raise ValueError("Source-data checksum list does not match its lock")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative_name = line.split("  ", 1)
        path = source_dir / relative_name
        if not path.is_file() or common.sha256_file(path) != digest:
            raise ValueError(f"Source-data checksum mismatch: {relative_name}")
    return {
        "directory": str(source_dir),
        "lock": lock,
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "manifest_sha256": common.sha256_file(manifest_path),
    }


def load_source_data(source_dir: Path) -> dict[str, Any]:
    paul = json.loads((source_dir / "PaulGrahamEssays.json").read_text(encoding="utf-8"))
    squad = json.loads((source_dir / "squad.json").read_text(encoding="utf-8"))
    hotpot = json.loads((source_dir / "hotpotqa.json").read_text(encoding="utf-8"))
    return {"paul": paul, "squad": squad, "hotpot": hotpot}


def load_word_resources(ruler_dir: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    try:
        from wonderwords import random_word
    except ImportError as exc:
        raise RuntimeError("wonderwords is required by official RULER") from exc
    nouns = sorted(set(random_word._get_words_from_text_file("nounlist.txt")))
    adjectives = sorted(set(random_word._get_words_from_text_file("adjectivelist.txt")))
    verbs = sorted(set(random_word._get_words_from_text_file("verblist.txt")))
    english_path = (
        ruler_dir / "scripts" / "data" / "synthetic" / "json" / "english_words.json"
    )
    english_payload = json.loads(english_path.read_text(encoding="utf-8"))
    english_words = sorted(set(str(value) for value in english_payload.values()))
    versions = {"wonderwords": importlib.metadata.version("wonderwords")}
    return {
        "nouns": nouns,
        "adjectives": adjectives,
        "verbs": verbs,
        "cwe_words": sorted(set(nouns + adjectives + verbs)),
        "english_words": english_words,
    }, versions


def random_needle_value(
    kind: str,
    rng: random.Random,
    word_resources: Mapping[str, Sequence[str]],
) -> str:
    if kind == "numbers":
        return str(rng.randint(1_000_000, 9_999_999))
    if kind == "words":
        return (
            f"{rng.choice(word_resources['adjectives'])}-"
            f"{rng.choice(word_resources['nouns'])}"
        )
    if kind == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    raise ValueError(f"Unsupported RULER needle type: {kind}")


def query_string(values: Sequence[str]) -> str:
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + ", and " + values[-1]


def niah_semantic(resources: TaskResources, sample_index: int) -> dict[str, Any]:
    args = resources.config["args"]
    seed = derive_seed(GLOBAL_SEED, resources.task_name, sample_index, "semantic")
    rng = random.Random(seed)
    num_keys = max(int(args["num_needle_k"]), int(args["num_needle_q"]))
    num_values = int(args["num_needle_v"])
    num_queries = int(args["num_needle_q"])
    keys: list[str] = []
    values: list[list[str]] = []
    used: set[str] = set()
    for _ in range(num_keys):
        while True:
            key = random_needle_value(args["type_needle_k"], rng, resources.words)
            if key not in used:
                used.add(key)
                break
        keys.append(key)
        row: list[str] = []
        for _ in range(num_values):
            while True:
                value = random_needle_value(args["type_needle_v"], rng, resources.words)
                if value not in used:
                    used.add(value)
                    break
            row.append(value)
        values.append(row)

    query_indices = rng.sample(range(num_keys), num_queries)
    answers = [value for index in query_indices for value in values[index]]
    primary = balanced_depth_schedule(resources.task_name)[sample_index]
    start = DEPTH_LEVELS.index(primary)
    all_depths = [DEPTH_LEVELS[(start + offset) % len(DEPTH_LEVELS)] for offset in range(num_keys * num_values)]

    needle_rows: list[dict[str, Any]] = []
    queried_needle_indices = [index * num_values + value_index for index in query_indices for value_index in range(num_values)]
    remaining_indices = [index for index in range(num_keys * num_values) if index not in queried_needle_indices]
    ordered_indices = queried_needle_indices + remaining_indices
    depths_by_index = {needle_index: all_depths[position] for position, needle_index in enumerate(ordered_indices)}
    for key_index, key in enumerate(keys):
        for value_index, value in enumerate(values[key_index]):
            needle_index = key_index * num_values + value_index
            needle_rows.append(
                {
                    "needle_id": f"N{needle_index:02d}",
                    "key": key,
                    "value": value,
                    "text": NEEDLE_TEMPLATE.format(
                        type_needle_v=args["type_needle_v"], key=key, value=value
                    ),
                    "target_depth": depths_by_index[needle_index],
                    "queried": key_index in query_indices,
                }
            )
    return {
        "sample_seed": seed,
        "keys": keys,
        "values": values,
        "query_indices": query_indices,
        "query": query_string([keys[index] for index in query_indices]),
        "answers": answers,
        "needles": needle_rows,
    }


def niah_renderer(
    resources: TaskResources,
    semantic: Mapping[str, Any],
    filler_ids: Sequence[int],
) -> Callable[[int], RenderedPrompt]:
    args = resources.config["args"]
    single_answer = int(args["num_needle_q"]) * int(args["num_needle_v"]) == 1
    task_template = resources.task_template
    answer_prefix = resources.answer_prefix
    type_name = str(args["type_needle_v"])
    if single_answer:
        task_template = common.singularize_ruler_text(task_template)
        answer_prefix = common.singularize_ruler_text(answer_prefix)
        type_name = type_name[:-1]
    insertions = [
        (float(row["target_depth"]), str(row["text"]), str(row["needle_id"]))
        for row in semantic["needles"]
    ]

    def render(budget: int) -> RenderedPrompt:
        context, insertion_details = insert_at_depths(
            resources.tokenizer, filler_ids, budget, insertions
        )
        user_text = task_template.format(
            type_needle_v=type_name,
            context=context,
            query=semantic["query"],
        )
        assistant = answer_prefix.format(
            type_needle_v=type_name,
            query=semantic["query"],
        )
        return RenderedPrompt(
            full_prompt=format_chat(resources.tokenizer, user_text, assistant),
            metadata={"insertions": insertion_details},
        )

    return render


def validate_niah(rendered: RenderedPrompt) -> None:
    for insertion in rendered.metadata["insertions"]:
        error = float(insertion["actual_depth"]) - float(insertion["target_depth"])
        if abs(error) > DEPTH_TOLERANCE:
            raise ValueError(f"NIAH depth error {error} exceeds tolerance")
        if rendered.full_prompt.count(str(insertion["text"])) != 1:
            raise ValueError("A target NIAH needle is missing or duplicated")


def distractor_needle_pool(resources: TaskResources, task_name: str) -> list[int]:
    args = resources.config["args"]
    rng = random.Random(derive_seed(GLOBAL_SEED, task_name, "distractor_pool"))
    rows: list[str] = []
    for _ in range(12_000):
        key = random_needle_value(args["type_needle_k"], rng, resources.words)
        value = random_needle_value(args["type_needle_v"], rng, resources.words)
        rows.append(
            NEEDLE_TEMPLATE.format(
                type_needle_v=args["type_needle_v"], key=key, value=value
            )
        )
    pool = encode(resources.tokenizer, "\n".join(rows))
    if len(pool) < max(LENGTHS) * 2:
        pool = ensure_token_pool(
            resources.tokenizer,
            "\n".join(rows),
            max(LENGTHS) * 2,
        )
    return pool


def vt_semantic(resources: TaskResources, sample_index: int) -> dict[str, Any]:
    seed = derive_seed(GLOBAL_SEED, resources.task_name, sample_index, "semantic")
    rng = random.Random(seed)

    def make_chain(variable_length: int) -> tuple[list[str], str, list[str]]:
        variables: list[str] = []
        while len(variables) < 5:
            candidate = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=variable_length))
            if candidate not in variables:
                variables.append(candidate)
        value = str(rng.randint(10_000, 99_998))
        chain = [f"VAR {variables[0]} = {value}"]
        chain.extend(f"VAR {variables[index]} = VAR {variables[index - 1]}" for index in range(1, 5))
        return variables, value, chain

    variables, value, chain = make_chain(5)
    fewshot_variables, fewshot_value, fewshot_chain = make_chain(3)
    return {
        "sample_seed": seed,
        "variables": variables,
        "value": value,
        "chain": chain,
        "fewshot_variables": fewshot_variables,
        "fewshot_value": fewshot_value,
        "fewshot_chain": fewshot_chain,
        "answers": variables,
    }


def vt_renderer(
    resources: TaskResources,
    semantic: Mapping[str, Any],
    filler_ids: Sequence[int],
) -> Callable[[int], RenderedPrompt]:
    chain_depths = DEPTH_LEVELS
    insertions = [
        (depth, line, f"hop_{index}")
        for index, (depth, line) in enumerate(zip(chain_depths, semantic["chain"]))
    ]
    fewshot_context = "\n".join(semantic["fewshot_chain"])
    fewshot_user = resources.task_template.format(
        context=fewshot_context,
        query=semantic["fewshot_value"],
        num_v=5,
    )
    fewshot_prefix = resources.answer_prefix.format(
        query=semantic["fewshot_value"], num_v=5
    )
    fewshot = (
        fewshot_user
        + fewshot_prefix
        + " "
        + " ".join(str(value) for value in semantic["fewshot_variables"])
    )

    def render(budget: int) -> RenderedPrompt:
        context, details = insert_at_depths(
            resources.tokenizer, filler_ids, budget, insertions
        )
        actual_user = resources.task_template.format(
            context=context,
            query=semantic["value"],
            num_v=5,
        )
        assistant = resources.answer_prefix.format(query=semantic["value"], num_v=5)
        return RenderedPrompt(
            full_prompt=format_chat(
                resources.tokenizer, fewshot + "\n" + actual_user, assistant
            ),
            metadata={"chain_positions": details},
        )

    return render


def cwe_semantic(resources: TaskResources, sample_index: int) -> dict[str, Any]:
    seed = derive_seed(GLOBAL_SEED, resources.task_name, sample_index, "semantic")
    rng = random.Random(seed)
    official_words = list(resources.words["cwe_words"])
    english = list(resources.words["english_words"])
    common_words = rng.sample(official_words, 10)
    excluded = set(common_words)
    distractors = [word for word in english if word not in excluded]
    rng.shuffle(distractors)
    distractors = distractors[:25_000]

    fewshot_words = rng.sample(
        [word for word in official_words if word not in excluded], 40
    )
    fewshot_common = fewshot_words[:10]
    fewshot_items = fewshot_common * 10 + fewshot_words[10:] * 3
    rng.shuffle(fewshot_items)
    fewshot_context = " ".join(
        f"{index + 1}. {word}" for index, word in enumerate(fewshot_items)
    )

    # Keep the invariant-bearing core comfortably below the smallest 4K
    # target.  The remaining uncommon words are still added through the
    # variable tail, so this does not change the CWE answer distribution.
    core_distractors = distractors[:20]
    core_items = common_words * 30 + core_distractors * 3
    rng.shuffle(core_items)
    core_context = " ".join(
        f"{index + 1}. {word}" for index, word in enumerate(core_items)
    )
    tail_items = [
        word for word in distractors[20:] for _ in range(3)
    ]
    tail_text = " ".join(
        f"{index + len(core_items) + 1}. {word}"
        for index, word in enumerate(tail_items)
    )
    tail_ids = encode(resources.tokenizer, " " + tail_text)
    return {
        "sample_seed": seed,
        "answers": common_words,
        "fewshot_answers": fewshot_common,
        "fewshot_context": fewshot_context,
        "core_context": core_context,
        "tail_ids": tail_ids,
    }


def cwe_renderer(resources: TaskResources, semantic: Mapping[str, Any]) -> Callable[[int], RenderedPrompt]:
    fewshot_user = resources.task_template.format(
        num_cw=10,
        context=semantic["fewshot_context"],
        query="",
    )
    fewshot = fewshot_user + " " + " ".join(
        f"{index + 1}. {word}"
        for index, word in enumerate(semantic["fewshot_answers"])
    )

    def render(budget: int) -> RenderedPrompt:
        tail = decode(resources.tokenizer, semantic["tail_ids"][:budget])
        context = str(semantic["core_context"])
        if tail:
            context += " " + tail
        actual_user = resources.task_template.format(num_cw=10, context=context, query="")
        return RenderedPrompt(
            full_prompt=format_chat(
                resources.tokenizer,
                fewshot + "\n" + actual_user,
                resources.answer_prefix,
            ),
            metadata={},
        )

    return render


def fwe_semantic(
    resources: TaskResources,
    sample_index: int,
    target_tokens: int,
) -> dict[str, Any]:
    seed = derive_seed(GLOBAL_SEED, resources.task_name, sample_index, "semantic")
    vocab_key = f"fwe_vocab:{sample_index}"
    if vocab_key not in resources.shared_pools:
        rng = random.Random(seed)
        unique_vocab: set[str] = set()
        while len(unique_vocab) < 3000:
            unique_vocab.add("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=6)))
        vocab = sorted(unique_vocab)
        random.Random(seed).shuffle(vocab)
        vocab[0] = "..."
        resources.shared_pools[vocab_key] = vocab
    vocab = list(resources.shared_pools[vocab_key])

    # Match RULER's FWE construction: integer Zipf counts are generated first,
    # then the complete word multiset is shuffled.  Build a target-specific
    # pool so exact-length trimming removes only a small suffix rather than
    # taking a tiny, statistically unstable prefix of a 128K-sized sample.
    ranks = np.arange(1, len(vocab) + 1, dtype=np.float64)
    probabilities = ranks ** -2.0
    probabilities /= probabilities.sum()
    requested_words = max(1000, target_tokens // 3)
    pool_ids: list[int] = []
    for _ in range(6):
        counts = np.floor(requested_words * probabilities).astype(np.int64)
        sampled_words = [
            word
            for word, count in zip(vocab, counts)
            for _ in range(int(count))
        ]
        random.Random(derive_seed(seed, target_tokens, "shuffle")).shuffle(sampled_words)
        pool_ids = encode(resources.tokenizer, " ".join(sampled_words))
        if len(pool_ids) >= target_tokens + 512:
            break
        requested_words = math.ceil(
            requested_words * (target_tokens + 1024) / max(1, len(pool_ids))
        )
    if len(pool_ids) < target_tokens:
        raise RuntimeError(f"FWE word pool is too short for target {target_tokens}")
    return {
        "sample_seed": seed,
        "answers": vocab[1:4],
        "pool_ids": pool_ids,
    }


def fwe_renderer(resources: TaskResources, semantic: Mapping[str, Any]) -> Callable[[int], RenderedPrompt]:
    def render(budget: int) -> RenderedPrompt:
        context = decode(resources.tokenizer, semantic["pool_ids"][:budget])
        user_text = resources.task_template.format(context=context, query="")
        return RenderedPrompt(
            full_prompt=format_chat(resources.tokenizer, user_text, resources.answer_prefix),
            metadata={"context": context},
        )

    return render


def validate_fwe(expected: Sequence[str], rendered: RenderedPrompt) -> None:
    words = re.findall(r"[a-z]{6}|\.\.\.", rendered.metadata["context"])
    counts = Counter(word for word in words if word != "...")
    actual = [word for word, _ in counts.most_common(3)]
    if actual != list(expected):
        raise ValueError(f"FWE top-three mismatch: expected={list(expected)}, actual={actual}")


def read_squad(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    docs = sorted(
        {
            paragraph["context"]
            for article in payload["data"]
            for paragraph in article["paragraphs"]
        }
    )
    lookup = {document: index for index, document in enumerate(docs)}
    qas: list[dict[str, Any]] = []
    for article in payload["data"]:
        article_docs = [lookup[paragraph["context"]] for paragraph in article["paragraphs"]]
        for paragraph in article["paragraphs"]:
            context_id = lookup[paragraph["context"]]
            for qa in paragraph["qas"]:
                if qa.get("is_impossible", False):
                    continue
                qas.append(
                    {
                        "query": qa["question"],
                        "outputs": sorted({answer["text"] for answer in qa["answers"]}),
                        "context": [context_id],
                        "more_context": [value for value in article_docs if value != context_id],
                    }
                )
    return qas, docs


def read_hotpot(payload: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    docs = sorted(
        {
            f"{title}\n{''.join(paragraph)}"
            for row in payload
            for title, paragraph in row["context"]
        }
    )
    lookup = {document: index for index, document in enumerate(docs)}
    qas = [
        {
            "query": row["question"],
            "outputs": [row["answer"]],
            "context": [
                lookup[f"{title}\n{''.join(paragraph)}"]
                for title, paragraph in row["context"]
            ],
            "more_context": [],
        }
        for row in payload
    ]
    return qas, docs


def qa_dataset(resources: TaskResources) -> tuple[list[dict[str, Any]], list[str]]:
    dataset = resources.config["args"]["dataset"]
    if dataset == "squad":
        return resources.source_data["squad_parsed"]
    if dataset == "hotpotqa":
        return resources.source_data["hotpot_parsed"]
    raise ValueError(f"Unsupported QA dataset: {dataset}")


def qa_semantic(resources: TaskResources, sample_index: int) -> dict[str, Any]:
    qas, docs = qa_dataset(resources)
    row = qas[sample_index]
    seed = derive_seed(GLOBAL_SEED, resources.task_name, sample_index, "semantic")
    rng = random.Random(seed)
    required_ids = list(dict.fromkeys(int(value) for value in row["context"]))
    more_ids = [int(value) for value in row.get("more_context", []) if int(value) not in required_ids]
    other_ids = [
        index
        for index in range(len(docs))
        if index not in set(required_ids) and index not in set(more_ids)
    ]
    rng.shuffle(more_ids)
    rng.shuffle(other_ids)
    distractor_ids = (more_ids + other_ids)[:3000]
    lines = [
        f"Document D{position + 1}:\n{docs[doc_id]}"
        for position, doc_id in enumerate(distractor_ids)
    ]
    distractor_ids_tokens = encode(resources.tokenizer, "\n\n".join(lines))
    if len(distractor_ids_tokens) < max(LENGTHS) * 2:
        distractor_ids_tokens = ensure_token_pool(
            resources.tokenizer,
            "\n\n".join(lines),
            max(LENGTHS) * 2,
            separator="\n\n",
        )
    evidence = "\n\n".join(
        f"Document E{position + 1}:\n{docs[doc_id]}"
        for position, doc_id in enumerate(required_ids)
    )
    return {
        "sample_seed": seed,
        "query": row["query"],
        "answers": row["outputs"],
        "required_document_ids": required_ids,
        "evidence": evidence,
        "evidence_target_depth": balanced_depth_schedule(resources.task_name)[sample_index],
        "distractor_ids": distractor_ids_tokens,
    }


def qa_renderer(resources: TaskResources, semantic: Mapping[str, Any]) -> Callable[[int], RenderedPrompt]:
    insertion = [
        (
            float(semantic["evidence_target_depth"]),
            str(semantic["evidence"]),
            "evidence",
        )
    ]

    def render(budget: int) -> RenderedPrompt:
        context, details = insert_at_depths(
            resources.tokenizer,
            semantic["distractor_ids"],
            budget,
            insertion,
        )
        user_text = resources.task_template.format(
            context=context,
            query=semantic["query"],
        )
        return RenderedPrompt(
            full_prompt=format_chat(resources.tokenizer, user_text, resources.answer_prefix),
            metadata={"evidence_position": details[0]},
        )

    return render


def validate_qa(rendered: RenderedPrompt) -> None:
    position = rendered.metadata["evidence_position"]
    error = float(position["actual_depth"]) - float(position["target_depth"])
    if abs(error) > DEPTH_TOLERANCE:
        raise ValueError(f"QA evidence depth error {error} exceeds tolerance")
    if rendered.full_prompt.count(str(position["text"])) != 1:
        raise ValueError("QA evidence was not preserved exactly once")


def build_shared_pools(
    tokenizer: Any,
    source_data: Mapping[str, Any],
) -> dict[str, list[int]]:
    essay_text = re.sub(r"\s+", " ", str(source_data["paul"]["text"])).strip()
    return {
        "noise": ensure_token_pool(tokenizer, NOISE_TEXT, max(LENGTHS) * 2),
        "essay": ensure_token_pool(tokenizer, essay_text, max(LENGTHS) * 2, separator=" "),
    }


def source_hashes(ruler_dir: Path) -> dict[str, Any]:
    root = ruler_dir / "scripts" / "data" / "synthetic"
    files = [root / name for name in OFFICIAL_TASK_SOURCES]
    files.extend(
        [
            ruler_dir / "scripts" / "synthetic.yaml",
            root / "json" / "english_words.json",
        ]
    )
    return {
        str(path.relative_to(ruler_dir).as_posix()): common.sha256_file(path)
        for path in files
    }


def verify_ruler_checkout_without_global_git_change(ruler_dir: Path) -> dict[str, Any]:
    """Allow the read-only Codex sandbox account without editing global git config."""
    keys = ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["GIT_CONFIG_COUNT"] = "1"
    os.environ["GIT_CONFIG_KEY_0"] = "safe.directory"
    os.environ["GIT_CONFIG_VALUE_0"] = ruler_dir.as_posix()
    try:
        return common.verify_ruler_checkout(ruler_dir)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def task_generator_inputs(
    resources: TaskResources,
    sample_index: int,
    target_tokens: int | None = None,
) -> tuple[dict[str, Any], Callable[[int], RenderedPrompt], int, Callable[[RenderedPrompt], None] | None]:
    base_task = resources.config["task"]
    if base_task == "niah":
        semantic = niah_semantic(resources, sample_index)
        haystack_type = resources.config["args"]["type_haystack"]
        if haystack_type == "essay":
            pool = resources.shared_pools["essay"]
        elif haystack_type == "needle":
            pool_key = f"needle:{resources.task_name}"
            if pool_key not in resources.shared_pools:
                resources.shared_pools[pool_key] = distractor_needle_pool(
                    resources, resources.task_name
                )
            pool = resources.shared_pools[pool_key]
        else:
            pool = resources.shared_pools["noise"]
        return semantic, niah_renderer(resources, semantic, pool), len(pool), validate_niah
    if base_task == "variable_tracking":
        semantic = vt_semantic(resources, sample_index)
        pool = resources.shared_pools["noise"]
        return semantic, vt_renderer(resources, semantic, pool), len(pool), None
    if base_task == "common_words_extraction":
        semantic = cwe_semantic(resources, sample_index)
        return semantic, cwe_renderer(resources, semantic), len(semantic["tail_ids"]), None
    if base_task == "freq_words_extraction":
        if target_tokens is None:
            raise ValueError("FWE generation requires a target token length")
        semantic = fwe_semantic(resources, sample_index, target_tokens)
        validator = lambda rendered: validate_fwe(semantic["answers"], rendered)
        return semantic, fwe_renderer(resources, semantic), len(semantic["pool_ids"]), validator
    if base_task == "qa":
        semantic = qa_semantic(resources, sample_index)
        return semantic, qa_renderer(resources, semantic), len(semantic["distractor_ids"]), validate_qa
    raise ValueError(f"Unsupported RULER base task: {base_task}")


def record_for_sample(
    *,
    resources: TaskResources,
    semantic: Mapping[str, Any],
    rendered: RenderedPrompt,
    input_ids: Sequence[int],
    target: int,
    budget: int,
    sample_index: int,
) -> dict[str, Any]:
    exact = len(input_ids)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "task": resources.task_name,
        "ruler_base_task": resources.config["task"],
        "sample_id": f"S{sample_index:02d}",
        "sample_index": sample_index,
        "global_seed": GLOBAL_SEED,
        "sample_seed": semantic["sample_seed"],
        "target_input_tokens": target,
        "exact_input_tokens": exact,
        "length_error": exact - target,
        "allowed_length_error": allowed_error(target),
        "answers": list(semantic["answers"]),
        "official_generation_budget": resources.generation_budget,
        "filler_token_budget": budget,
        "tensor_file": f"tokens/{resources.task_name}/{target}.pt",
        "tensor_index": sample_index,
        "input_ids_sha256": common.sha256_input_ids(input_ids),
        "prompt_text_sha256": common.sha256_text(rendered.full_prompt),
    }
    if "query" in semantic:
        record["query"] = semantic["query"]
    if "keys" in semantic:
        record["keys"] = semantic["keys"]
        record["values"] = semantic["values"]
        record["query_indices"] = semantic["query_indices"]
    if "variables" in semantic:
        record["variables"] = semantic["variables"]
        record["tracked_value"] = semantic["value"]

    def positioned(details: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        token_ids_with_offsets, offsets = common.tokenize_with_offsets(
            resources.tokenizer, rendered.full_prompt
        )
        if token_ids_with_offsets != list(input_ids):
            raise ValueError("Offset tokenization changed the final prompt IDs")
        result: list[dict[str, Any]] = []
        for source in details:
            detail = dict(source)
            text = str(detail["text"])
            char_start = rendered.full_prompt.find(text)
            if char_start < 0 or rendered.full_prompt.count(text) != 1:
                raise ValueError("Could not uniquely locate a preserved task span")
            token_start, token_end = common.char_span_to_token_span(
                offsets, char_start, char_start + len(text)
            )
            detail["token_start"] = token_start
            detail["token_end_exclusive"] = token_end
            result.append(detail)
        return result

    if "insertions" in rendered.metadata:
        record["needle_positions"] = positioned(rendered.metadata["insertions"])
    if "chain_positions" in rendered.metadata:
        record["chain_positions"] = positioned(rendered.metadata["chain_positions"])
    if "evidence_position" in rendered.metadata:
        record["evidence_position"] = positioned(
            [rendered.metadata["evidence_position"]]
        )[0]
        record["required_document_ids"] = semantic["required_document_ids"]
    return record


def verify_task_files(
    torch_module: Any,
    staging_dir: Path,
    task_name: str,
    records: Sequence[Mapping[str, Any]],
    *,
    target_lengths: Sequence[int],
    sample_count: int,
) -> None:
    for target in target_lengths:
        path = staging_dir / "tokens" / task_name / f"{target}.pt"
        payload = common.torch_load_cpu(torch_module, path)
        if not isinstance(payload, list) or len(payload) != sample_count:
            raise ValueError(f"Unexpected tensor count in {path}")
        matching = [record for record in records if record["target_input_tokens"] == target]
        for index, tensor in enumerate(payload):
            if (
                not isinstance(tensor, torch_module.Tensor)
                or tensor.dtype != torch_module.int64
                or tensor.device.type != "cpu"
                or tensor.ndim != 1
            ):
                raise TypeError(f"Invalid tensor schema in {path} at index {index}")
            if tensor.numel() != matching[index]["exact_input_tokens"]:
                raise ValueError(f"Manifest/tensor length mismatch in {path}")


def write_task(
    *,
    torch_module: Any,
    resources: TaskResources,
    staging_dir: Path,
    target_lengths: Sequence[int],
    sample_count: int,
) -> tuple[list[dict[str, Any]], list[Path]]:
    tensors_by_length: dict[int, list[Any]] = {target: [] for target in target_lengths}
    records: list[dict[str, Any]] = []
    for sample_index in range(sample_count):
        per_length_semantics = resources.config["task"] == "freq_words_extraction"
        if not per_length_semantics:
            semantic, render, maximum_budget, validator = task_generator_inputs(
                resources, sample_index
            )
        print(
            f"[{resources.task_name}] sample {sample_index + 1:02d}/{sample_count}",
            flush=True,
        )
        for target in target_lengths:
            if per_length_semantics:
                semantic, render, maximum_budget, validator = task_generator_inputs(
                    resources, sample_index, target
                )
            input_ids, rendered, budget = fit_exact_prompt(
                tokenizer=resources.tokenizer,
                target_tokens=target,
                maximum_budget=maximum_budget,
                render=render,
                validate=validator,
            )
            tensor = torch_module.tensor(input_ids, dtype=torch_module.long, device="cpu")
            tensors_by_length[target].append(tensor)
            records.append(
                record_for_sample(
                    resources=resources,
                    semantic=semantic,
                    rendered=rendered,
                    input_ids=input_ids,
                    target=target,
                    budget=budget,
                    sample_index=sample_index,
                )
            )

    artifact_paths: list[Path] = []
    for target, tensors in tensors_by_length.items():
        relative = Path("tokens") / resources.task_name / f"{target}.pt"
        common.torch_save_atomic(torch_module, staging_dir / relative, tensors)
        artifact_paths.append(relative)
    verify_task_files(
        torch_module,
        staging_dir,
        resources.task_name,
        records,
        target_lengths=target_lengths,
        sample_count=sample_count,
    )
    task_manifest = Path("manifests") / "tasks" / f"{resources.task_name}.jsonl"
    common.write_jsonl_atomic(staging_dir / task_manifest, records)
    artifact_paths.append(task_manifest)
    return records, artifact_paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-dir", type=Path, default=Path("RULER"))
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=Path("prompts/ruler_v1/source_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prompts/ruler_v1/accuracy"),
    )
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--tokenizer-id", default=common.DEFAULT_TOKENIZER_ID)
    parser.add_argument(
        "--tokenizer-revision", default=common.DEFAULT_TOKENIZER_REVISION
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a failed formal build from verified per-task checkpoints",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--smoke-task",
        choices=TASK_NAMES,
        help="Generate one sample into a non-formal smoke directory",
    )
    parser.add_argument(
        "--smoke-length",
        type=int,
        choices=LENGTHS,
        default=4096,
        help="Input length used with --smoke-task (default: 4096)",
    )
    return parser.parse_args()


def self_test() -> None:
    assert TASK_NAMES and "niah_single_1" not in TASK_NAMES
    assert len(TASK_NAMES) == 12
    assert len(LENGTHS) == 6
    assert SAMPLES_PER_TASK == 50
    assert len(TASK_NAMES) * len(LENGTHS) * SAMPLES_PER_TASK == 3600
    for task_name in [name for name in TASK_NAMES if name.startswith("niah_")]:
        schedule = balanced_depth_schedule(task_name)
        assert Counter(schedule) == Counter({depth: 10 for depth in DEPTH_LEVELS})
    print("S2 accuracy design self-test: PASSED")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.seed != GLOBAL_SEED:
        raise ValueError(f"Formal RULER v1 accuracy bank requires --seed {GLOBAL_SEED}")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Run this script in the S2 environment with torch") from exc

    ruler_dir = args.ruler_dir.resolve()
    source_dir = args.source_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.smoke_task:
        output_dir = output_dir.with_name(
            f"{output_dir.name}.smoke.{args.smoke_task}.{args.smoke_length}"
        )
    staging_dir = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}"
        )
    if staging_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Staging directory already exists; inspect it or pass --resume: {staging_dir}"
        )
    if args.resume and args.smoke_task:
        raise ValueError("--resume is only supported for the formal prompt bank")

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)

    ruler_metadata = verify_ruler_checkout_without_global_git_change(ruler_dir)
    definitions, synthetic = load_task_definitions(ruler_dir)
    source_metadata = validate_source_lock(source_dir)
    source_data = load_source_data(source_dir)
    source_data["squad_parsed"] = read_squad(source_data["squad"])
    source_data["hotpot_parsed"] = read_hotpot(source_data["hotpot"])
    tokenizer, tokenizer_metadata = common.load_tokenizer(args)
    words, word_versions = load_word_resources(ruler_dir)
    shared_pools = build_shared_pools(tokenizer, source_data)

    selected_tasks = (args.smoke_task,) if args.smoke_task else TASK_NAMES
    sample_count = 1 if args.smoke_task else SAMPLES_PER_TASK
    target_lengths = (args.smoke_length,) if args.smoke_task else LENGTHS
    staging_dir.mkdir(parents=True, exist_ok=args.resume)
    all_records: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    try:
        for task_name in selected_tasks:
            definition = definitions[task_name]
            resources = TaskResources(
                tokenizer=tokenizer,
                task_name=task_name,
                config=definition["config"],
                task_template=definition["task_template"],
                answer_prefix=definition["answer_prefix"],
                generation_budget=definition["generation_budget"],
                source_data=source_data,
                words=words,
                shared_pools=shared_pools,
            )
            task_manifest = staging_dir / "manifests" / "tasks" / f"{task_name}.jsonl"
            expected_tensor_paths = [
                Path("tokens") / task_name / f"{target}.pt"
                for target in target_lengths
            ]
            if args.resume and task_manifest.exists():
                task_records = read_jsonl(task_manifest)
                verify_task_files(
                    torch,
                    staging_dir,
                    task_name,
                    task_records,
                    target_lengths=target_lengths,
                    sample_count=sample_count,
                )
                if len(task_records) != len(target_lengths) * sample_count:
                    raise ValueError(f"Invalid checkpoint record count for {task_name}")
                task_artifacts = expected_tensor_paths + [
                    task_manifest.relative_to(staging_dir)
                ]
                print(f"[{task_name}] resumed verified checkpoint", flush=True)
            else:
                task_records, task_artifacts = write_task(
                    torch_module=torch,
                    resources=resources,
                    staging_dir=staging_dir,
                    target_lengths=target_lengths,
                    sample_count=sample_count,
                )
            all_records.extend(task_records)
            artifact_paths.extend(task_artifacts)

        manifest_relative = Path("manifests") / "accuracy.jsonl"
        common.write_jsonl_atomic(staging_dir / manifest_relative, all_records)
        artifact_paths.append(manifest_relative)

        config_relative = Path("generation_config.json")
        config_payload = {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "created_at_utc": utc_now(),
            "purpose": "ruler_accuracy_prompt_bank",
            "excluded_existing_benchmarks": ["niah_single_1"],
            "tasks": list(selected_tasks),
            "lengths": list(target_lengths),
            "samples_per_task_length": sample_count,
            "record_count": len(all_records),
            "global_seed": GLOBAL_SEED,
            "length_tolerance_rule": "min(8, floor(target_input_tokens * 0.002))",
            "niah_depth_levels": list(DEPTH_LEVELS),
            "niah_samples_per_depth": 10 if not args.smoke_task else None,
            "ruler": ruler_metadata,
            "ruler_task_configs": {name: synthetic[name] for name in selected_tasks},
            "ruler_source_hashes": source_hashes(ruler_dir),
            "external_source_data": source_metadata,
            "tokenizer": tokenizer_metadata,
            "word_resources": word_versions,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "generator_source_sha256": common.sha256_file(Path(__file__).resolve()),
            "model_loaded": False,
            "generate_py_imported_or_reimplemented": False,
            "formal": not bool(args.smoke_task),
        }
        common.write_json_atomic(staging_dir / config_relative, config_payload)
        artifact_paths.append(config_relative)
        checksum_path = common.write_checksums(staging_dir, artifact_paths)

        lock_payload = {
            "schema_version": SCHEMA_VERSION,
            "lock_kind": "ruler_v1_accuracy" if not args.smoke_task else "ruler_v1_accuracy_smoke",
            "created_at_utc": utc_now(),
            "tasks": list(selected_tasks),
            "lengths": list(target_lengths),
            "samples_per_task_length": sample_count,
            "record_count": len(all_records),
            "tensor_file_count": len(selected_tasks) * len(target_lengths),
            "checksums_sha256": common.sha256_file(checksum_path),
            "ruler_commit": ruler_metadata["commit"],
            "tokenizer_revision": args.tokenizer_revision,
        }
        common.write_json_atomic(
            staging_dir / "generation_complete.lock.json", lock_payload
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_dir, output_dir)
    except Exception:
        print(
            f"Accuracy generation failed; inspect staging directory: {staging_dir}",
            file=sys.stderr,
        )
        raise

    print(f"Frozen accuracy prompt bank: {output_dir}")
    print(f"Tasks: {len(selected_tasks)}")
    print(f"Records: {len(all_records)}")
    print("No model or generate.py was loaded.")


if __name__ == "__main__":
    main()
