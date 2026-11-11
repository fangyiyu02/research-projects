"""
UltraLLaDA latency pipeline:
- parse configuration from terminal
- load tokenizer and model
- load RULER-generated JSONL samples
- tokenize prompts
- warm up
- call the official UltraLLaDA/LLaDA generate()
- measure E2E latency and CUDA peak memory
- decode predictions
- save results

Important:
Place the official `generate.py` beside this script, or make it importable through
PYTHONPATH. This benchmark calls that file directly; it does not reimplement the
diffusion generation algorithm.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# Use the official diffusion generation implementation.
from generate import generate


def parse_args():
    """Read settings supplied through the terminal."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)

    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--cfg-scale", type=float, default=0.0)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)

    return parser.parse_args()


def model_loading(args):
    """Load the tokenizer and model once, then place the model on the GPU."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if args.dtype not in dtype_map:
        raise ValueError(
            "--dtype must be bfloat16, float16, or float32."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
    )
    model = model.to(args.device)
    model.eval()

    return tokenizer, model


def load_ruler_samples(data_path, max_samples=None):
    """Read prompts and expected answers from a RULER JSONL file."""
    samples = []

    with open(data_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if not line.strip():
                continue

            sample = json.loads(line)

            # NeMo-Skills RULER uses question/expected_answer.
            # Original RULER commonly uses input/outputs.
            if "question" in sample:
                prompt = sample["question"]
            elif "input" in sample:
                prompt = sample["input"] + sample.get("answer_prefix", "")
            else:
                raise KeyError(
                    "RULER sample needs a 'question' or 'input' field."
                )

            if "expected_answer" in sample:
                expected_answer = sample["expected_answer"]
            elif "outputs" in sample:
                expected_answer = sample["outputs"]
            else:
                raise KeyError(
                    "RULER sample needs an 'expected_answer' or 'outputs' field."
                )

            samples.append(
                {
                    "sample_id": sample.get(
                        "index",
                        sample.get("id", line_number),
                    ),
                    "prompt": prompt,
                    "expected_answer": expected_answer,
                    "target_length": sample.get("length"),
                }
            )

            if max_samples is not None and len(samples) >= max_samples:
                break

    return samples


def prepare_input(prompt, tokenizer, device):
    """Convert one prompt string into token IDs on the GPU."""
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )
    input_ids = encoded["input_ids"].to(device)
    return input_ids


def call_generate(model, input_ids, args):
    """Call the official UltraLLaDA/LLaDA diffusion generate() function."""
    mask_id = getattr(model.config, "mask_token_id", 126336)

    output_ids = generate(
        model=model,
        prompt=input_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        remasking=args.remasking,
        cfg_scale=args.cfg_scale,
        mask_id=mask_id,
    )
    return output_ids


def warmup(model, tokenizer, sample, args):
    """Run unmeasured generation before the real latency measurement."""
    input_ids = prepare_input(
        sample["prompt"],
        tokenizer,
        args.device,
    )

    for _ in range(args.warmup_runs):
        call_generate(model, input_ids, args)
        torch.cuda.synchronize(args.device)


def measure_latency(model, tokenizer, sample, args):
    """
    Generate one answer and directly measure its E2E latency and peak memory.

    Tokenization is performed before timing, so the reported latency measures
    model generation rather than CPU tokenization.
    """
    input_ids = prepare_input(
        sample["prompt"],
        tokenizer,
        args.device,
    )
    input_length = input_ids.shape[1]

    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)

    start_time = time.perf_counter()
    output_ids = call_generate(model, input_ids, args)
    torch.cuda.synchronize(args.device)
    end_time = time.perf_counter()

    e2e_latency_seconds = end_time - start_time
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(args.device) / 1024**2
    )

    # generate() returns prompt tokens followed by generated tokens.
    generated_ids = output_ids[:, input_length:]
    prediction = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()

    return {
        "sample_id": sample["sample_id"],
        "target_context_length": sample["target_length"],
        "actual_input_tokens": input_length,
        "expected_answer": sample["expected_answer"],
        "prediction": prediction,
        "e2e_latency_seconds": e2e_latency_seconds,
        "peak_memory_mb": peak_memory_mb,
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "remasking": args.remasking,
        "cfg_scale": args.cfg_scale,
        "batch_size": 1,
    }


def save_results(results, output_path):
    """Save all predictions and measurements as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    print("Loading RULER samples...")
    samples = load_ruler_samples(
        args.data_path,
        args.max_samples,
    )
    if not samples:
        raise ValueError("No RULER samples were loaded.")

    print("Loading tokenizer and UltraLLaDA model...")
    tokenizer, model = model_loading(args)

    print("Warming up...")
    warmup(model, tokenizer, samples[0], args)

    print("Running benchmark...")
    results = []
    for sample_number, sample in enumerate(samples, start=1):
        result = measure_latency(
            model,
            tokenizer,
            sample,
            args,
        )
        results.append(result)

        print(
            f"[{sample_number}/{len(samples)}] "
            f"tokens={result['actual_input_tokens']}, "
            f"latency={result['e2e_latency_seconds']:.4f}s, "
            f"peak_memory={result['peak_memory_mb']:.1f}MB"
        )

    save_results(results, args.output_path)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
