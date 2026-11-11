"""
Score predictions produced by bench_latency.py.

Pipeline:
- parse configuration
- load benchmark results
- validate and normalize results
- match each prediction with its expected answers
- calculate the aggregate score
- save per-sample scored results
- save the summary
"""

import argparse
import json
import re
from pathlib import Path


def parse_configuration():
    """Read scoring settings supplied through the terminal."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", required=True)
    parser.add_argument("--scored-output-path", required=True)
    parser.add_argument("--summary-output-path", required=True)
    parser.add_argument(
        "--task",
        choices=["niah", "vt", "cwe", "fwe", "qa"],
        default="niah",
    )

    return parser.parse_args()


def load_results(input_path):
    """Load the JSONL results produced by bench_latency.py."""
    results = []

    with open(input_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            result = json.loads(line)
            if not isinstance(result, dict):
                raise ValueError(
                    f"Line {line_number} must contain one JSON object."
                )
            results.append(result)

    return results


def postprocess(pred):
    pred = re.sub(r"[\x00-\x1f]", "\n", pred.strip()).strip()
    return pred


def validate_normalize_results(results):
    """Validate required fields and normalize predictions and references."""
    normalized_results = []

    for result_number, result in enumerate(results, start=1):
        if "expected_answer" not in result:
            raise KeyError(
                f"Result {result_number} is missing 'expected_answer'."
            )

        prediction = result.get("prediction", "")
        if prediction is None:
            prediction = ""
        elif not isinstance(prediction, str):
            prediction = str(prediction)

        refs = result["expected_answer"]
        if isinstance(refs, str):
            refs = [refs]
        elif not isinstance(refs, list):
            raise TypeError(
                f"Result {result_number} 'expected_answer' must be "
                "a string or a list."
            )

        refs = [str(ref).strip() for ref in refs]
        if not refs or any(not ref for ref in refs):
            raise ValueError(
                f"Result {result_number} has an empty expected answer."
            )

        normalized_result = dict(result)
        normalized_result["prediction"] = postprocess(prediction)
        normalized_result["expected_answer"] = refs
        normalized_results.append(normalized_result)

    return normalized_results


def match_all(pred, refs):  # niah / vt / cwe / fwe
    return sum(r.lower() in pred.lower() for r in refs) / len(refs)


def match_part(pred, refs):  # qa
    return float(any(r.lower() in pred.lower() for r in refs))


def calculate_aggregate_score(results, task):
    """Score every sample and average the per-sample scores."""
    scored_results = []
    total_score = 0.0

    for result in results:
        if task == "qa":
            score = match_part(
                result["prediction"],
                result["expected_answer"],
            )
        else:
            score = match_all(
                result["prediction"],
                result["expected_answer"],
            )

        scored_result = dict(result)
        scored_result["score"] = score
        scored_results.append(scored_result)
        total_score += score

    aggregate_score = total_score / len(scored_results)
    summary = {
        "task": task,
        "match_type": "part" if task == "qa" else "all",
        "num_samples": len(scored_results),
        "aggregate_score": aggregate_score,
        "aggregate_score_percent": aggregate_score * 100,
    }

    return scored_results, summary


def save_per_sample_scored_results(scored_results, output_path):
    """Save original benchmark fields plus a score for every sample."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for result in scored_results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


def save_summary(summary, output_path):
    """Save the aggregate scoring result as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    args = parse_configuration()

    print("Loading benchmark results...")
    results = load_results(args.input_path)
    if not results:
        raise ValueError("No benchmark results were loaded.")

    print("Validating and normalizing results...")
    results = validate_normalize_results(results)

    print("Scoring predictions...")
    scored_results, summary = calculate_aggregate_score(
        results,
        args.task,
    )

    save_per_sample_scored_results(
        scored_results,
        args.scored_output_path,
    )
    save_summary(
        summary,
        args.summary_output_path,
    )

    print(
        f"Aggregate score: "
        f"{summary['aggregate_score_percent']:.2f}%"
    )
    print(f"Scored results saved to {args.scored_output_path}")
    print(f"Summary saved to {args.summary_output_path}")


if __name__ == "__main__":
    main()
