"""Create a new deterministic haystack and immutable NIAH prompt manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoTokenizer

from common import (
    expected_prompt_count,
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
    write_jsonl_atomic,
    write_text_atomic,
)


CITIES = [
    "Amsterdam", "Barcelona", "Cairo", "Denver", "Edinburgh", "Florence",
    "Geneva", "Helsinki", "Istanbul", "Jakarta", "Kyoto", "Lisbon",
    "Madrid", "Nairobi", "Oslo", "Prague", "Quebec", "Reykjavik",
    "Santiago", "Toronto", "Uppsala", "Vienna", "Warsaw", "Zurich",
]
ADJECTIVES = [
    "amber", "ancient", "calm", "careful", "cedar", "clear", "distant",
    "gentle", "hidden", "ivory", "luminous", "marble", "narrow", "quiet",
    "silver", "steady", "verdant", "warm",
]
SUBJECTS = [
    "archivist", "botanist", "carpenter", "cartographer", "curator",
    "gardener", "geologist", "historian", "librarian", "navigator",
    "painter", "researcher", "scholar", "surveyor", "teacher", "weaver",
]
VERBS = [
    "catalogued", "compared", "described", "examined", "followed", "mapped",
    "measured", "observed", "organized", "recorded", "reviewed", "studied",
    "traced", "visited",
]
OBJECTS = [
    "a collection of coastal plants", "a group of weathered manuscripts",
    "a quiet path beside the river", "an arrangement of colored stones",
    "the changing outline of the valley", "the grain of a wooden panel",
    "the growth of moss beneath the trees", "the history of a small harbor",
    "the migration of birds across the marsh", "the pattern of clouds at dusk",
    "the shape of leaves in the courtyard", "the texture of woven cloth",
]
PLACES = [
    "beside the old observatory", "beyond the northern garden",
    "inside the public reading room", "near the forest boundary",
    "under the covered walkway", "within the stone workshop",
]
ENDINGS = [
    "and wrote a concise note for the archive",
    "before sharing the observation with colleagues",
    "while the afternoon remained unusually quiet",
    "and preserved the account for later comparison",
    "without changing the surrounding materials",
    "as part of a long-running descriptive survey",
]

NEEDLE_TEMPLATE = "The special magic number for {city} is {number}."
QUESTION_TEMPLATE = (
    "What is the special magic number for {city} mentioned in the text above? "
    "Answer with the number only."
)


def hash_index(seed: int, sentence_index: int, slot: str, size: int) -> int:
    payload = f"{seed}|{sentence_index}|{slot}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % size


def generate_fresh_haystack(seed: int, sentence_count: int) -> str:
    """Generate version-1 prose without digits or needle-like statements."""
    sentences: list[str] = []
    for index in range(sentence_count):
        adjective = ADJECTIVES[hash_index(seed, index, "adjective", len(ADJECTIVES))]
        subject = SUBJECTS[hash_index(seed, index, "subject", len(SUBJECTS))]
        verb = VERBS[hash_index(seed, index, "verb", len(VERBS))]
        obj = OBJECTS[hash_index(seed, index, "object", len(OBJECTS))]
        place = PLACES[hash_index(seed, index, "place", len(PLACES))]
        ending = ENDINGS[hash_index(seed, index, "ending", len(ENDINGS))]
        sentence = (
            f"The {adjective} {subject} {verb} {obj} {place}, {ending}."
        )
        sentences.append(sentence)
    text = "\n".join(sentences) + "\n"
    if any(character.isdigit() for character in text):
        raise AssertionError("Fresh haystack generator unexpectedly produced digits")
    if "special magic number" in text.lower():
        raise AssertionError("Fresh haystack contains reserved needle language")
    return text


def derive_sample_seed(
    master_seed: int, target_total_len: int, depth: float, sample_idx: int
) -> int:
    key = (
        f"{master_seed}|{target_total_len}|{depth:.6f}|{sample_idx}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int, label: str
) -> tuple[int, int]:
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not indices:
        raise ValueError(f"Could not map {label} character span to token offsets")
    return min(indices), max(indices) + 1


def tokenize_with_offsets(
    tokenizer: Any, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError) as exc:
        raise RuntimeError(
            "Exact needle positions require a fast tokenizer with offset mappings"
        ) from exc
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if len(input_ids) != len(offsets):
        raise ValueError("Tokenizer returned inconsistent input IDs and offsets")
    return input_ids, offsets


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


def build_prompt(
    *,
    tokenizer: Any,
    haystack_ids: list[int],
    target_total_len: int,
    generation: dict[str, Any],
    requested_depth: float,
    city: str,
    number: str,
    rng: random.Random,
    safety_margin: int,
) -> dict[str, Any]:
    gen_length = int(generation["gen_length"])
    needle_text = NEEDLE_TEMPLATE.format(city=city, number=number)
    question_text = QUESTION_TEMPLATE.format(city=city)
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": needle_text + "\n\n" + question_text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    overhead = len(tokenizer(probe, add_special_tokens=False)["input_ids"])
    budget = target_total_len - gen_length - overhead - safety_margin
    if budget <= 0:
        raise ValueError(f"target_total_len={target_total_len} is too small")
    max_start = len(haystack_ids) - budget - 1
    if max_start <= 0:
        raise ValueError("Fresh haystack is too short for the requested grid")
    haystack_start = rng.randrange(max_start)
    chunk = haystack_ids[haystack_start : haystack_start + budget]

    full_prompt = ""
    input_ids: list[int] = []
    offsets: list[tuple[int, int]] = []
    for _ in range(8):
        split = min(int(round(requested_depth * len(chunk))), len(chunk))
        part_a = tokenizer.decode(chunk[:split], skip_special_tokens=True)
        part_b = tokenizer.decode(chunk[split:], skip_special_tokens=True)
        content = (
            part_a
            + "\n\n"
            + needle_text
            + "\n\n"
            + part_b
            + "\n\n"
            + question_text
        )
        full_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids, offsets = tokenize_with_offsets(tokenizer, full_prompt)
        excess = len(input_ids) + gen_length - target_total_len
        if excess <= 0:
            break
        chunk = chunk[: max(0, len(chunk) - excess - 4)]
    else:
        raise RuntimeError("Failed to fit prompt under target total length")

    needle_char_start = full_prompt.index(needle_text)
    needle_char_end = needle_char_start + len(needle_text)
    question_char_start = full_prompt.rindex(question_text)
    question_char_end = question_char_start + len(question_text)
    needle_start, needle_end = char_span_to_token_span(
        offsets, needle_char_start, needle_char_end, "needle"
    )
    question_start, question_end = char_span_to_token_span(
        offsets, question_char_start, question_char_end, "question"
    )
    actual_total_len = len(input_ids) + gen_length
    return {
        "input_ids": input_ids,
        "prompt_text": full_prompt,
        "actual_total_len": actual_total_len,
        "length_shortfall_tokens": target_total_len - actual_total_len,
        "prompt_token_length": len(input_ids),
        "haystack_start_token": haystack_start,
        "haystack_chunk_token_count": len(chunk),
        "needle_token_start": needle_start,
        "needle_token_end_exclusive": needle_end,
        "needle_token_center": (needle_start + needle_end - 1) / 2,
        "question_token_start": question_start,
        "question_token_end_exclusive": question_end,
        "needle_to_question_start_distance": question_start - needle_start,
        "needle_end_to_question_gap": question_start - needle_end,
        "needle_prompt_position_ratio": needle_start / max(1, len(input_ids)),
        "needle_to_question_position_ratio": needle_start / max(1, question_start),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment_v1.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    validate_config(config)
    project_root = project_root_from_config(config_path)
    paths = {
        "haystack": resolve_project_path(
            project_root, config["fresh_haystack"]["relative_path"]
        ),
        "haystack_meta": resolve_project_path(
            project_root, config["fresh_haystack"]["metadata_relative_path"]
        ),
        "prompts": resolve_project_path(
            project_root, config["artifacts"]["prompt_manifest_relative_path"]
        ),
        "prompts_meta": resolve_project_path(
            project_root, config["artifacts"]["prompt_metadata_relative_path"]
        ),
        "lock": resolve_project_path(
            project_root, config["artifacts"]["experiment_lock_relative_path"]
        ),
    }
    if paths["lock"].exists():
        raise FileExistsError("Experiment is already locked; create a v2 instead")
    existing = [path for name, path in paths.items() if name != "lock" and path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Preparation outputs already exist; inspect them or use a new experiment version"
        )

    haystack_cfg = config["fresh_haystack"]
    haystack_text = generate_fresh_haystack(
        int(haystack_cfg["seed"]), int(haystack_cfg["sentence_count"])
    )
    write_text_atomic(paths["haystack"], haystack_text)
    haystack_sha256 = sha256_file(paths["haystack"])

    tokenizer_cfg = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_cfg["id"],
        revision=tokenizer_cfg["revision"],
        code_revision=tokenizer_cfg["code_revision"],
        trust_remote_code=bool(tokenizer_cfg["trust_remote_code"]),
        use_fast=bool(tokenizer_cfg["use_fast"]),
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for exact token spans")
    resolved_commits = resolved_commit_candidates(tokenizer)
    if any(commit != tokenizer_cfg["revision"] for commit in resolved_commits):
        raise ValueError("Tokenizer resolved to an unexpected commit")

    haystack_ids = [
        int(value)
        for value in tokenizer(haystack_text, add_special_tokens=False)["input_ids"]
    ]
    maximum_target = max(int(x) for x in config["grid"]["target_total_lengths"])
    if len(haystack_ids) < maximum_target * 2:
        raise ValueError("Generated haystack is unexpectedly short")

    haystack_meta = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "generator_version": int(haystack_cfg["generator_version"]),
        "seed": int(haystack_cfg["seed"]),
        "sentence_count": int(haystack_cfg["sentence_count"]),
        "character_count": len(haystack_text),
        "token_count": len(haystack_ids),
        "contains_digits": any(character.isdigit() for character in haystack_text),
        "relative_path": relative_to_project(project_root, paths["haystack"]),
        "sha256": haystack_sha256,
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(paths["haystack_meta"], haystack_meta)

    grid = config["grid"]
    generation = config["generation"]
    master_seed = int(grid["master_prompt_seed"])
    records: list[dict[str, Any]] = []
    for target_len in grid["target_total_lengths"]:
        for depth in grid["depths"]:
            for sample_idx in range(int(grid["n_samples_per_cell"])):
                sample_seed = derive_sample_seed(
                    master_seed, int(target_len), float(depth), sample_idx
                )
                rng = random.Random(sample_seed)
                city = rng.choice(CITIES)
                number = "".join(rng.choice("0123456789") for _ in range(7))
                built = build_prompt(
                    tokenizer=tokenizer,
                    haystack_ids=haystack_ids,
                    target_total_len=int(target_len),
                    generation=generation,
                    requested_depth=float(depth),
                    city=city,
                    number=number,
                    rng=rng,
                    safety_margin=int(grid["safety_margin_tokens"]),
                )
                depth_tag = int(round(float(depth) * 1000))
                sample_id = f"L{int(target_len):05d}_D{depth_tag:04d}_S{sample_idx:02d}"
                record = {
                    "schema_version": 1,
                    "experiment_id": config["experiment_id"],
                    "sample_id": sample_id,
                    "sample_seed": sample_seed,
                    "target_total_len": int(target_len),
                    "requested_depth": float(depth),
                    "sample_idx": sample_idx,
                    "city": city,
                    "needle_number": number,
                    "needle_text": NEEDLE_TEMPLATE.format(city=city, number=number),
                    "question_text": QUESTION_TEMPLATE.format(city=city),
                    "haystack_sha256": haystack_sha256,
                    "model_revision": config["model"]["revision"],
                    "model_code_revision": config["model"]["code_revision"],
                    "tokenizer_revision": tokenizer_cfg["revision"],
                    "tokenizer_code_revision": tokenizer_cfg["code_revision"],
                    **built,
                }
                record["input_ids_sha256"] = sha256_input_ids(record["input_ids"])
                record["prompt_text_sha256"] = sha256_text(record["prompt_text"])
                records.append(record)

    random.Random(int(grid["run_order_seed"])).shuffle(records)
    for run_order_index, record in enumerate(records):
        record["run_order_index"] = run_order_index

    if len(records) != expected_prompt_count(config):
        raise AssertionError("Unexpected prompt count")
    cell_counts = Counter(
        (record["target_total_len"], record["requested_depth"])
        for record in records
    )
    if set(cell_counts.values()) != {int(grid["n_samples_per_cell"])}:
        raise AssertionError("Prompt grid is not balanced")
    max_shortfall = int(grid["maximum_length_shortfall_tokens"])
    bad_shortfalls = [
        record["sample_id"]
        for record in records
        if not 0 <= int(record["length_shortfall_tokens"]) <= max_shortfall
    ]
    if bad_shortfalls:
        raise ValueError(
            f"{len(bad_shortfalls)} prompts exceed the allowed length shortfall"
        )

    write_jsonl_atomic(paths["prompts"], records)
    prompt_meta = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "experiment_id": config["experiment_id"],
        "config_relative_path": relative_to_project(project_root, config_path),
        "config_sha256": sha256_file(config_path),
        "manifest_relative_path": relative_to_project(project_root, paths["prompts"]),
        "manifest_sha256": sha256_file(paths["prompts"]),
        "haystack_sha256": haystack_sha256,
        "prompt_count": len(records),
        "cell_count": len(cell_counts),
        "samples_per_cell": int(grid["n_samples_per_cell"]),
        "model_revision": config["model"]["revision"],
        "model_code_revision": config["model"]["code_revision"],
        "tokenizer_revision": tokenizer_cfg["revision"],
        "tokenizer_code_revision": tokenizer_cfg["code_revision"],
        "tokenizer_resolved_commit_candidates": resolved_commits,
        "transformers_version": transformers.__version__,
    }
    write_json_atomic(paths["prompts_meta"], prompt_meta)
    print(f"Fresh haystack: {paths['haystack']} ({len(haystack_ids)} tokens)")
    print(f"Immutable prompts: {paths['prompts']} ({len(records)} samples)")
    print(f"Prompt manifest SHA-256: {prompt_meta['manifest_sha256']}")


if __name__ == "__main__":
    main()

