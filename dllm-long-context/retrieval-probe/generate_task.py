"""Build a small, token-verified retrieval task; no model weights or GPU needed.

Requires tokenizers. Use the tokenizer.json and tokenizer_config.json from
GSAI-ML/LLaDA-8B-Instruct. The saved input_ids already include the chat template
and one masked answer position. Never serialize the whole row as a model prompt.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from tokenizers import Tokenizer
import tokenizers


MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
POSITIONS = {"early": 0.2, "middle": 0.5, "late": 0.8}
NAMES = ["Maple", "Cedar", "Birch", "Willow", "Pine", "Oak", "Elm", "Ash",
         "Alder", "Hazel", "Laurel", "Olive", "Aspen", "Juniper", "Spruce", "Fir"]
FILLER = [
    "The store opens after sunrise and closes before sunset.",
    "The shelves are arranged in neat rows along the wall.",
    "Staff members check the inventory during quiet hours.",
    "A notice board hangs beside the main entrance.",
    "The delivery boxes are stored near the back door.",
    "Customers can request a paper receipt at the counter.",
    "The display area is cleaned before the store opens.",
    "The windows allow natural light into the room.",
    "A small desk stands next to the storage room.",
    "The clerk keeps a notebook beside the register.",
    "Empty cartons are folded and placed in the recycling bin.",
    "The signs on the shelves are checked regularly.",
    "The front door has a handle made of metal.",
    "Visitors leave their umbrellas near the entrance.",
    "The floor is swept at the end of the day.",
    "A clock hangs above the service counter.",
]

# Exact cached official template. Rendering this single-user special case avoids
# loading transformers/torch. Reject a different template instead of guessing.
EXPECTED_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chat_text(user):
    return ("<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n"
            + user.strip() + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def build_prompt(lines, item):
    context = "\n".join(lines)
    return ("Read the store notes and retrieve the requested price.\n\n"
            "Store notes:\n" + context + "\n\n"
            f"What is the price of item {item}?\n"
            "Reply with only the single digit. Do not include a dollar sign, words, or punctuation.")


def overlap_positions(encoding, start, end):
    return [i for i, (a, b) in enumerate(encoding.offsets) if a < end and b > start]


def price_span(text, record):
    # Locate within the full fact rather than searching a digit globally.
    start = text.index(record) + record.index("$") + 1
    return start, start + 1


def place_fact(tok, background, fact, item, fraction):
    """Choose a sentence boundary closest to the requested context-token depth."""
    best = None
    for insertion in range(len(background) + 1):
        lines = background[:insertion] + [fact] + background[insertion:]
        context = "\n".join(lines)
        enc = tok.encode(context, add_special_tokens=False)
        positions = overlap_positions(enc, *price_span(context, fact))
        depth = (positions[0] + 0.5) / len(enc.ids)
        candidate = (abs(depth - fraction), lines, depth)
        if best is None or candidate[0] < best[0]:
            best = candidate
    _, lines, depth = best
    user = build_prompt(lines, item)
    enc = tok.encode(chat_text(user), add_special_tokens=False)
    return lines, user, enc, depth


def make_sample(tok, rng, sample_id, split, answer, position, max_tokens, mask_id):
    item, *others = rng.sample(NAMES, 4)
    other_prices = rng.sample([str(x) for x in range(1, 10) if str(x) != answer], 3)
    fact = f"The price of item {item} is ${answer}."
    distractors = [f"The price of item {name} is ${price}."
                   for name, price in zip(others, other_prices)]
    background = distractors.copy()
    # Fill toward a token budget. Sentences stay intact; length is recorded,
    # never represented as exactly max_tokens unless it actually is.
    while True:
        filler = rng.choice(FILLER)
        trial = background + [filler]
        user = build_prompt(trial + [fact], item)
        if len(tok.encode(chat_text(user), add_special_tokens=False).ids) + 1 > max_tokens:
            break
        background = trial
    rng.shuffle(background)
    while True:
        lines, user, enc, depth = place_fact(tok, background, fact, item, POSITIONS[position])
        if len(enc.ids) + 1 <= max_tokens:
            break
        # A changed boundary can alter BPE length; remove background prose only.
        remove = next((i for i in range(len(background) - 1, -1, -1)
                       if background[i] not in distractors), None)
        if remove is None:
            raise ValueError("Token budget is too small for the facts and instructions.")
        background.pop(remove)

    full = chat_text(user)
    critical = overlap_positions(enc, *price_span(full, fact))
    control = overlap_positions(enc, *price_span(full, distractors[0]))
    gold = tok.encode(answer, add_special_tokens=False).ids
    if len(gold) != 1 or len(critical) != 1 or enc.ids[critical[0]] != gold[0]:
        raise ValueError("Expected one-token answer and one-token evidence span.")
    if len(control) != 1:
        raise ValueError("Expected one-token control span.")
    context = "\n".join(lines)
    context_start = full.index(context)
    context_positions = overlap_positions(enc, context_start, context_start + len(context))
    actual_depth = (critical[0] - context_positions[0] + 0.5) / len(context_positions)
    row = {
        "id": sample_id,
        "split": split,
        "task": "single_digit_price_retrieval",
        "target_item": item,
        "position_group": position,
        "requested_context_depth": POSITIONS[position],
        "actual_context_depth": actual_depth,
        "prompt": user,
        "messages": [{"role": "user", "content": user}],
        "answer": answer,
        "answer_token_id": gold[0],
        "evidence_text": fact,
        "evidence_char_span_in_prompt": list(price_span(user, fact)),
        "control_text": distractors[0],
        "control_char_span_in_prompt": list(price_span(user, distractors[0])),
        "critical_token_positions": critical,
        "control_token_positions": control,
        "input_ids": enc.ids + [mask_id],
        "prompt_token_count": len(enc.ids),
        "input_token_count": len(enc.ids) + 1,
        "answer_position": len(enc.ids),
        "mask_token_id": mask_id,
    }
    validate_sample(tok, row, max_tokens)
    return row


def validate_sample(tok, row, max_tokens):
    full_enc = tok.encode(chat_text(row["prompt"]), add_special_tokens=False)
    assert row["input_ids"] == full_enc.ids + [row["mask_token_id"]]
    assert row["input_ids"].count(row["mask_token_id"]) == 1
    assert row["input_token_count"] <= max_tokens
    assert row["input_ids"][row["answer_position"]] == row["mask_token_id"]
    assert row["answer_position"] == len(row["input_ids"]) - 1
    assert row["prompt"].count(row["answer"]) == 1
    assert row["input_ids"].count(row["answer_token_id"]) == 1
    assert row["prompt"].count(row["target_item"]) == 2
    assert row["prompt"].count("The price of item ") == 4
    assert row["messages"] == [{"role": "user", "content": row["prompt"]}]
    for stem in ["evidence", "control"]:
        start, end = row[f"{stem}_char_span_in_prompt"]
        assert end - start == 1 and row["prompt"][start:end].isdigit()
    assert len(row["critical_token_positions"]) == len(row["control_token_positions"]) == 1
    assert set(row["critical_token_positions"]).isdisjoint(row["control_token_positions"])
    assert row["critical_token_positions"][0] < row["prompt_token_count"]
    assert row["control_token_positions"][0] < row["prompt_token_count"]
    assert abs(row["actual_context_depth"] - row["requested_context_depth"]) <= 0.05
    assert tok.decode([row["answer_token_id"]]) == row["answer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.max_input_tokens < 256:
        parser.error("Use at least 256 tokens to keep three evidence depths meaningful.")
    tokenizer_path = args.tokenizer_dir / "tokenizer.json"
    config_path = args.tokenizer_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # JSON template may use literal newlines or escaped newline sequences.
    if config["chat_template"].replace("\\n", "\n") != EXPECTED_TEMPLATE.replace("\\n", "\n"):
        raise ValueError("Unexpected chat template; use the original LLaDA-8B-Instruct tokenizer.")
    if config["bos_token"] != "<|startoftext|>" or config.get("add_bos_token") or config.get("add_eos_token"):
        raise ValueError("Unexpected special-token settings.")
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.no_padding()
    tok.no_truncation()
    mask_id = tok.token_to_id("<|mdm_mask|>")
    if mask_id != 126336:
        raise ValueError("Unexpected mask token for LLaDA-8B-Instruct.")
    rng = random.Random(args.seed)
    groups = list(POSITIONS)
    # 81 fully crossed samples plus 9 extras: 90 evaluation rows, 30 per
    # depth and 10 per price. Each price-depth cell has either 3 or 4 rows.
    evaluation = [(str(digit), group) for digit in range(1, 10)
                  for group in groups for _ in range(3)]
    evaluation += [(str(digit), groups[(digit - 1) % 3]) for digit in range(1, 10)]
    pilot = [(str((i % 9) + 1), groups[i % 3]) for i in range(10)]
    rng.shuffle(pilot)
    rng.shuffle(evaluation)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = [args.out / name for name in ["pilot.jsonl", "eval.jsonl", "manifest.json", "example.txt"]]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Output files already exist; select a new --out directory.")
    all_rows = []
    for split, layout in [("pilot", pilot), ("eval", evaluation)]:
        rows = [make_sample(tok, rng, f"{split}_{i:03d}", split, answer, group,
                            args.max_input_tokens, mask_id)
                for i, (answer, group) in enumerate(layout)]
        all_rows.extend(rows)
        (args.out / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    assert len({row["prompt"] for row in all_rows}) == 100
    eval_rows = [row for row in all_rows if row["split"] == "eval"]
    assert Counter(row["position_group"] for row in eval_rows) == {group: 30 for group in groups}
    assert Counter(row["answer"] for row in eval_rows) == {str(d): 10 for d in range(1, 10)}
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "tokenizer_snapshot_directory": args.tokenizer_dir.name,
        "tokenizer_json_sha256": sha256(tokenizer_path),
        "tokenizer_config_sha256": sha256(config_path),
        "generator_sha256": sha256(Path(__file__)),
        "tokenizers_version": tokenizers.__version__,
        "seed": args.seed,
        "counts": {"pilot": 10, "eval": 90},
        "eval_depth_counts": dict(Counter(row["position_group"] for row in eval_rows)),
        "eval_answer_counts": dict(Counter(row["answer"] for row in eval_rows)),
        "max_input_tokens": args.max_input_tokens,
        "actual_input_tokens_min": min(row["input_token_count"] for row in all_rows),
        "actual_input_tokens_max": max(row["input_token_count"] for row in all_rows),
        "max_depth_error": max(abs(row["actual_context_depth"] - row["requested_context_depth"])
                               for row in all_rows),
        "digit_token_ids": {str(d): tok.encode(str(d), add_special_tokens=False).ids for d in range(1, 10)},
        "three_digit_example": {"text": "278", "token_ids": tok.encode("278", add_special_tokens=False).ids},
        "validation": "All 100 rows passed token alignment, answer uniqueness, mask, length, and balance checks.",
        "inference_status": "Not run. No model accuracy or pruning effect has been measured.",
        "coordinate_system": "Token positions are zero-based in input_ids, including chat template; char spans are [start, end) in prompt.",
        "data_sha256": {name: sha256(args.out / name) for name in ["pilot.jsonl", "eval.jsonl"]},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    example = all_rows[0]
    (args.out / "example.txt").write_text(
        "MODEL USER MESSAGE\n\n" + example["prompt"]
        + "\n\n--- METADATA BELOW IS NOT PART OF THE MODEL INPUT ---\n"
        + json.dumps({key: example[key] for key in ["id", "answer", "evidence_text", "position_group",
                                                  "critical_token_positions", "control_token_positions",
                                                  "answer_position", "input_token_count"]}, indent=2)
        + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
