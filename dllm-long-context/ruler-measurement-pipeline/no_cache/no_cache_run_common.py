#!/usr/bin/env python3
"""Shared implementation for two no-cache memory-policy entry points.

This worker compares exactly two variants:

* ``no_cache_nochunk``: original FFN and original full-sequence lm_head.
* ``no_cache_chunked``: sequence-chunked FFN and sequence-chunked lm_head.

Only memory execution policies change. The script consumes the frozen NIAH CPU
LongTensors directly; it never tokenizes or modifies a prompt at runtime.
This module contains latency measurement and frozen-prompt validation only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import statistics
import struct
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


SCHEMA_VERSION = 1
VARIANTS = ("no_cache_nochunk", "no_cache_chunked")
MIB = 1024**2


def parse_args(variant: str) -> argparse.Namespace:
    """Parse one fresh-process cell while keeping its entry-point variant fixed."""

    if variant not in VARIANTS:
        raise ValueError(f"unsupported fixed variant: {variant}")

    parser = argparse.ArgumentParser(
        description=(
            "Measure no-cache UltraLLaDA with original execution or sequence-"
            "chunked FFN and lm_head. "
            "Invoke once per variant and prompt length."
        )
    )
    parser.add_argument("--model", required=True, help="HF model ID or local model path")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--target-input-tokens", type=int, required=True)
    parser.add_argument(
        "--prompt-bank",
        type=Path,
        default=Path("prompts/ruler_v1/efficiency/niah_single_1"),
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Subset such as S00 S01; default uses every frozen sample at this length",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--ffn-chunk", type=int, default=8192)
    parser.add_argument("--lmhead-chunk", type=int, default=512)
    parser.add_argument(
        "--generate-module",
        default="generate",
        help="Importable module containing the official generate()",
    )
    parser.add_argument(
        "--generate-py",
        type=Path,
        default=None,
        help="Explicit official generate.py; takes precedence over --generate-module",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu-index", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("latency", "validate-prompts"),
        default="latency",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Latency JSONL output path",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Allow latency records to be appended to an existing JSONL",
    )
    args = parser.parse_args()
    args.variant = variant

    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if args.ffn_chunk <= 0:
        parser.error("--ffn-chunk must be positive")
    if args.lmhead_chunk <= 0:
        parser.error("--lmhead-chunk must be positive")
    if args.gen_length <= 0 or args.steps <= 0 or args.block_length <= 0:
        parser.error("generation lengths and steps must be positive")
    if args.gen_length % args.block_length != 0:
        parser.error("--gen-length must be divisible by --block-length")
    num_blocks = args.gen_length // args.block_length
    if args.steps % num_blocks != 0:
        parser.error("--steps must be divisible by gen_length / block_length")
    if args.temperature != 0.0:
        parser.error("this deterministic pilot requires --temperature 0.0")
    if args.mode == "latency" and args.output.exists() and not args.append:
        parser.error(f"output already exists: {args.output}; pass --append or use a new path")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file without accepting blank records."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            records.append(payload)
    return records


def discover_target_lengths(prompt_bank: Path) -> list[int]:
    """Discover the available target lengths solely from the frozen manifest."""

    manifest = prompt_bank / "manifests" / "efficiency.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"frozen efficiency manifest not found: {manifest}")
    lengths = sorted({int(row["target_input_tokens"]) for row in read_jsonl(manifest)})
    if not lengths:
        raise ValueError(f"manifest contains no target lengths: {manifest}")
    return lengths


def sha256_input_ids(input_ids: Iterable[int]) -> str:
    """Hash token IDs exactly like the frozen S2 prompt generator."""

    values = [int(value) for value in input_ids]
    payload = b"" if not values else struct.pack(f"<{len(values)}q", *values)
    return hashlib.sha256(payload).hexdigest()


def load_prompt_records(
    prompt_bank: Path,
    target_input_tokens: int,
    sample_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Select and validate manifest rows for one requested target length."""

    manifest = prompt_bank / "manifests" / "efficiency.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"frozen efficiency manifest not found: {manifest}")
    rows = [
        row
        for row in read_jsonl(manifest)
        if int(row["target_input_tokens"]) == target_input_tokens
    ]
    if sample_ids is not None:
        requested = list(dict.fromkeys(sample_ids))
        by_id = {str(row["sample_id"]): row for row in rows}
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            raise ValueError(f"sample IDs absent at target {target_input_tokens}: {missing}")
        rows = [by_id[sample_id] for sample_id in requested]
    if not rows:
        raise ValueError(f"no prompts selected at target_input_tokens={target_input_tokens}")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate sample_id rows in frozen manifest")
    return rows


def _torch_load_cpu(path: Path) -> Any:
    """Load a trusted frozen tensor file onto CPU across PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_frozen_prompt(prompt_bank: Path, record: Mapping[str, Any]) -> torch.Tensor:
    """Load one frozen CPU LongTensor and verify its manifest length and hash."""

    tensor_path = prompt_bank / str(record["tensor_file"])
    payload = _torch_load_cpu(tensor_path)
    tensor_index = int(record["tensor_index"])
    if not isinstance(payload, list) or not 0 <= tensor_index < len(payload):
        raise ValueError(f"unexpected tensor payload/index in {tensor_path}")
    tensor = payload[tensor_index]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"frozen item is not a Tensor: {tensor_path}[{tensor_index}]")
    if tensor.device.type != "cpu" or tensor.dtype != torch.int64 or tensor.ndim != 1:
        raise TypeError("frozen prompt must be a one-dimensional CPU int64 Tensor")
    expected_length = int(record["exact_input_tokens"])
    if tensor.numel() != expected_length:
        raise ValueError(
            f"prompt length mismatch: tensor={tensor.numel()} manifest={expected_length}"
        )
    observed_hash = sha256_input_ids(tensor.tolist())
    if observed_hash != str(record["input_ids_sha256"]):
        raise ValueError(
            f"prompt hash mismatch for {record['sample_id']}: "
            f"{observed_hash} != {record['input_ids_sha256']}"
        )
    return tensor.contiguous()


def derive_seed(*parts: object) -> int:
    """Derive a stable 63-bit seed from cell metadata."""

    text = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") & (2**63 - 1)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Durably append one completed, OOM, or invalid run record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def set_global_seed(seed: int) -> None:
    """Reset all RNGs used by this deterministic generation worker."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_transformer(model: nn.Module) -> nn.Module:
    """Locate the UltraLLaDA transformer through common wrapper layouts."""

    node: nn.Module = model
    for _ in range(5):
        children = dict(node.named_children())
        if "transformer" in children:
            return children["transformer"]
        if len(children) == 1:
            node = next(iter(children.values()))
            continue
        break
    raise RuntimeError("could not locate the UltraLLaDA transformer module")


@dataclass
class RuntimeCounters:
    """Mutable per-generation counters shared by the chunked modules."""

    ffn_chunk_calls: int = 0
    lmhead_chunks_per_forward: list[int] = field(default_factory=list)
    lmhead_projected_tokens: list[int] = field(default_factory=list)

    def reset(self) -> None:
        """Reset counters immediately before one generation."""

        self.ffn_chunk_calls = 0
        self.lmhead_chunks_per_forward = []
        self.lmhead_projected_tokens = []


class ChunkedFFN(nn.Module):
    """Execute UltraLLaDA SwiGLU independently over sequence chunks."""

    def __init__(
        self,
        ff_proj: nn.Module,
        up_proj: nn.Module,
        act: nn.Module,
        ff_out: nn.Module,
        chunk_size: int,
        counters: RuntimeCounters,
    ) -> None:
        super().__init__()
        self.ff_proj = ff_proj
        self.up_proj = up_proj
        self.act = act
        self.ff_out = ff_out
        self.chunk_size = int(chunk_size)
        self.counters = counters

    def _one(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """Run the unmodified SwiGLU formula on one contiguous token slice."""

        gate = self.act(self.ff_proj(x_chunk))
        up = self.up_proj(x_chunk)
        return self.ff_out(gate * up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split only the sequence axis, then write chunks in original order."""

        if x.ndim != 3:
            raise ValueError(f"ChunkedFFN expected [B,L,D], got {tuple(x.shape)}")
        sequence_length = int(x.shape[1])
        if sequence_length <= self.chunk_size:
            self.counters.ffn_chunk_calls += 1
            return self._one(x)

        out_features = int(getattr(self.ff_out, "out_features"))
        output = torch.empty(
            x.shape[0],
            sequence_length,
            out_features,
            dtype=x.dtype,
            device=x.device,
        )
        for start in range(0, sequence_length, self.chunk_size):
            end = min(start + self.chunk_size, sequence_length)
            self.counters.ffn_chunk_calls += 1
            output[:, start:end].copy_(self._one(x[:, start:end]))
        return output


class _FFNProductMarker:
    """Sentinel used only to preserve the original block's Python expression."""


FFN_PRODUCT_MARKER = _FFNProductMarker()


class _PrecomputedFFNProduct:
    """Unwrap a precomputed FFN result at the original gate-times-up operation."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def __mul__(self, other: Any) -> torch.Tensor:
        if other is not FFN_PRODUCT_MARKER:
            raise RuntimeError(
                "UltraLLaDA FFN expression changed: expected act(ff_proj(x)) * up_proj(x)"
            )
        return self.value


class ChunkedFFNGateAdapter(nn.Module):
    """Expose ChunkedFFN through the original block's ff_proj call site."""

    def __init__(self, chunked_ffn: ChunkedFFN) -> None:
        super().__init__()
        self.chunked_ffn = chunked_ffn

    def forward(self, x: torch.Tensor) -> _PrecomputedFFNProduct:
        return _PrecomputedFFNProduct(self.chunked_ffn(x))


class FFNProductMarkerAdapter(nn.Module):
    """Replace up_proj with a zero-CUDA-work marker for the original multiply."""

    def forward(self, x: torch.Tensor) -> _FFNProductMarker:
        return FFN_PRODUCT_MARKER


def install_chunked_ffn(
    model: nn.Module,
    chunk_size: int,
    counters: RuntimeCounters,
) -> int:
    """Route the original block forward's FFN expression through ChunkedFFN."""

    transformer = _find_transformer(model)
    blocks = transformer.blocks
    installed = 0
    for block in blocks:
        required = ("ff_proj", "up_proj", "act", "ff_out", "ff_norm")
        missing = [name for name in required if not hasattr(block, name)]
        if missing:
            raise RuntimeError(
                f"{type(block).__name__} is incompatible with the frozen FFN patch; "
                f"missing {missing}"
            )
        if isinstance(block.ff_proj, ChunkedFFNGateAdapter):
            raise RuntimeError("chunked FFN was already installed")
        original_ff_proj = block.ff_proj
        original_up_proj = block.up_proj
        original_act = block.act
        original_ff_out = block.ff_out
        block.ff_proj = ChunkedFFNGateAdapter(
            ChunkedFFN(
                original_ff_proj,
                original_up_proj,
                original_act,
                original_ff_out,
                chunk_size,
                counters,
            )
        )
        block.up_proj = FFNProductMarkerAdapter()
        block.act = nn.Identity()
        block.ff_out = nn.Identity()
        installed += 1
    if installed == 0:
        raise RuntimeError("no transformer layers received the FFN patch")
    return installed


class _ChunkedSoftmax:
    """Expose cached per-token confidence to the official gather operation."""

    def __init__(self, confidence: torch.Tensor) -> None:
        self.confidence = confidence

    def gather(
        self,
        dim: int,
        index: torch.Tensor,
        *,
        sparse_grad: bool = False,
    ) -> torch.Tensor:
        if dim not in (-1, 2) or sparse_grad:
            raise RuntimeError("chunked lm_head supports only dense vocab-axis gather")
        expected_shape = (*self.confidence.shape, 1)
        if tuple(index.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected confidence gather index shape {tuple(index.shape)}; "
                f"expected {expected_shape}"
            )
        return self.confidence.unsqueeze(-1)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func is torch.gather:
            return args[0].gather(*args[1:], **dict(kwargs or {}))
        return NotImplemented


class _ChunkedLogits:
    """Present chunked IDs/confidence through the official logits operations."""

    def __init__(
        self,
        token_ids: torch.Tensor,
        confidence: torch.Tensor,
        vocab_size: int,
        logical_dtype: torch.dtype,
    ) -> None:
        self.token_ids = token_ids
        self.confidence = confidence
        self.vocab_size = int(vocab_size)
        self.dtype = logical_dtype
        self.device = token_ids.device
        self.shape = torch.Size((*token_ids.shape, self.vocab_size))
        self.ndim = 3

    def to(self, *args: Any, **kwargs: Any) -> "_ChunkedLogits":
        requested_dtype = kwargs.get("dtype")
        for argument in args:
            if isinstance(argument, torch.dtype):
                requested_dtype = argument
        if requested_dtype not in (None, self.dtype, torch.float64):
            raise RuntimeError(
                f"chunked lm_head cannot materialize logits as {requested_dtype}"
            )
        return self

    def argmax(self, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
        if dim not in (-1, 2):
            raise RuntimeError("chunked lm_head supports argmax only on the vocab axis")
        return self.token_ids.unsqueeze(-1) if keepdim else self.token_ids

    def softmax(
        self,
        dim: int | None = None,
        dtype: torch.dtype | None = None,
        **_: Any,
    ) -> _ChunkedSoftmax:
        if dim not in (-1, 2):
            raise RuntimeError("chunked lm_head supports softmax only on the vocab axis")
        if dtype not in (None, torch.float64):
            raise RuntimeError("low-confidence chunking requires FP64 softmax")
        return _ChunkedSoftmax(self.confidence)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        call_kwargs = dict(kwargs or {})
        call_kwargs.pop("_stacklevel", None)
        if func is torch.argmax:
            return args[0].argmax(*args[1:], **call_kwargs)
        if func in (torch.softmax, F.softmax):
            return args[0].softmax(*args[1:], **call_kwargs)
        return NotImplemented


class ChunkedLMHeadAdapter(nn.Module):
    """Project hidden states and reduce logits one sequence chunk at a time."""

    def __init__(
        self,
        lm_head: nn.Linear,
        chunk_size: int,
        counters: RuntimeCounters,
    ) -> None:
        super().__init__()
        self.lm_head = lm_head
        self.chunk_size = int(chunk_size)
        self.counters = counters

    def forward(self, hidden: torch.Tensor) -> _ChunkedLogits:
        if hidden.ndim != 3 or int(hidden.shape[-1]) != self.lm_head.in_features:
            raise ValueError(
                "chunked lm_head expected [B,L,D] hidden states with "
                f"D={self.lm_head.in_features}; got {tuple(hidden.shape)}"
            )
        batch_size, sequence_length, _ = hidden.shape
        token_ids = torch.empty(
            batch_size,
            sequence_length,
            dtype=torch.long,
            device=hidden.device,
        )
        confidence = torch.empty(
            batch_size,
            sequence_length,
            dtype=torch.float64,
            device=hidden.device,
        )

        chunk_calls = 0
        for start in range(0, sequence_length, self.chunk_size):
            end = min(start + self.chunk_size, sequence_length)
            logits = self.lm_head(hidden[:, start:end])
            chunk_ids = torch.argmax(logits, dim=-1)
            logits_fp64 = logits.to(torch.float64)
            probabilities = F.softmax(logits_fp64, dim=-1)
            chunk_confidence = probabilities.gather(
                -1, chunk_ids.unsqueeze(-1)
            ).squeeze(-1)
            token_ids[:, start:end].copy_(chunk_ids)
            confidence[:, start:end].copy_(chunk_confidence)
            chunk_calls += 1
            del logits, logits_fp64, probabilities, chunk_ids, chunk_confidence

        self.counters.lmhead_chunks_per_forward.append(chunk_calls)
        self.counters.lmhead_projected_tokens.append(int(sequence_length))
        return _ChunkedLogits(
            token_ids,
            confidence,
            self.lm_head.out_features,
            hidden.dtype,
        )


def _chunked_lmhead_model_forward(
    self: nn.Module,
    *forward_args: Any,
    **forward_kwargs: Any,
) -> Any:
    """Call the original model forward and replace only its logits payload."""

    output = self._lmhead_original_forward(*forward_args, **forward_kwargs)
    hidden = getattr(output, "logits", None)
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("model output does not expose Tensor logits/hidden states")
    output.logits = self._chunked_lmhead_adapter(hidden)
    return output


def install_chunked_lmhead(
    model: nn.Module,
    chunk_size: int,
    counters: RuntimeCounters,
) -> int:
    """Detach the original lm_head and route its output through sequence chunks."""

    if hasattr(model, "_chunked_lmhead_adapter"):
        raise RuntimeError("chunked lm_head was already installed")
    transformer = _find_transformer(model)
    lm_head = getattr(transformer, "ff_out", None)
    if not isinstance(lm_head, nn.Linear):
        raise RuntimeError(
            f"expected transformer.ff_out to be Linear, got {type(lm_head).__name__}"
        )
    if bool(getattr(model.config, "weight_tying", False)):
        raise RuntimeError("chunked lm_head requires weight_tying=False")
    if bool(getattr(model.config, "scale_logits", False)):
        raise RuntimeError("chunked lm_head requires scale_logits=False")

    adapter = ChunkedLMHeadAdapter(lm_head, chunk_size, counters)
    transformer.ff_out = nn.Identity()
    model._chunked_lmhead_adapter = adapter
    model._lmhead_original_forward = model.forward
    model.forward = MethodType(_chunked_lmhead_model_forward, model)
    return int(lm_head.out_features)


def load_model(args: argparse.Namespace) -> nn.Module:
    """Load one BF16 eval model on the requested CUDA device with cache disabled."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device")
    if device.index is not None:
        torch.cuda.set_device(device.index)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "local_files_only": bool(args.local_files_only),
    }
    if args.revision is not None:
        kwargs["revision"] = args.revision
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model = model.to(device).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def validate_prompt_for_model(
    prompt: torch.Tensor,
    model: nn.Module,
    mask_id: int,
) -> None:
    """Reject out-of-vocabulary IDs or a mask token inside the frozen prompt."""

    vocab_size = int(getattr(model.config, "vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError("model config does not expose a valid vocab_size")
    minimum = int(prompt.min().item())
    maximum = int(prompt.max().item())
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"prompt token range [{minimum},{maximum}] is outside [0,{vocab_size})"
        )
    if bool((prompt == mask_id).any().item()):
        raise ValueError("frozen prompt already contains the generation mask token")


def load_official_generate(
    args: argparse.Namespace,
) -> tuple[Callable[..., Any], str]:
    """Load generate() from the project's official module without copying its logic."""

    if args.generate_py is not None:
        generate_path = args.generate_py.expanduser().resolve()
        if not generate_path.is_file():
            raise FileNotFoundError(f"official generate.py not found: {generate_path}")
        path_digest = hashlib.sha256(str(generate_path).encode()).hexdigest()[:12]
        module_name = f"_official_generate_{path_digest}"
        spec = importlib.util.spec_from_file_location(module_name, generate_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import official generate.py: {generate_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        parent = str(generate_path.parent)
        inserted_parent = parent not in sys.path
        if inserted_parent:
            sys.path.insert(0, parent)
        try:
            spec.loader.exec_module(module)
        finally:
            if inserted_parent:
                sys.path.remove(parent)
        source = str(generate_path)
    else:
        module = importlib.import_module(args.generate_module)
        source = str(getattr(module, "__file__", args.generate_module))

    generate_fn = getattr(module, "generate", None)
    if not callable(generate_fn):
        raise AttributeError(f"{source} does not expose a callable generate()")
    return generate_fn, source


@torch.no_grad()
def call_official_generate(
    generate_fn: Callable[..., Any],
    model: nn.Module,
    prompt: torch.Tensor,
    args: argparse.Namespace,
    mask_id: int,
) -> Any:
    """Call the unmodified project generate() with the experiment's fixed arguments."""

    if prompt.ndim != 2 or int(prompt.shape[0]) != 1:
        raise ValueError("this pilot requires prompt shape [1,L] and batch_size=1")
    return generate_fn(
        model,
        prompt,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=mask_id,
    )


def extract_generated_ids(
    output: Any,
    *,
    prompt_length: int,
    gen_length: int,
) -> torch.Tensor:
    """Normalize official generate() output to exactly the generated token suffix."""

    if isinstance(output, (tuple, list)):
        if not output:
            raise TypeError("official generate() returned an empty tuple/list")
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise TypeError("official generate() must return a 2-D token Tensor")
    if int(output.shape[0]) != 1:
        raise ValueError("this pilot requires official generate() batch size 1")

    output_length = int(output.shape[1])
    if output_length == gen_length:
        generated = output
    elif output_length == prompt_length + gen_length:
        generated = output[:, prompt_length:]
    else:
        raise ValueError(
            "official generate() returned sequence length "
            f"{output_length}; expected {gen_length} or {prompt_length + gen_length}"
        )
    return generated


class ForwardTimingRecorder:
    """Temporarily time top-level model.forward calls made by official generate()."""

    def __init__(self, model: nn.Module, expected_calls: int) -> None:
        self.model = model
        self.expected_calls = expected_calls
        self.starts = [
            torch.cuda.Event(enable_timing=True) for _ in range(expected_calls)
        ]
        self.ends = [
            torch.cuda.Event(enable_timing=True) for _ in range(expected_calls)
        ]
        self.calls = 0
        self._original_forward: Callable[..., Any] | None = None
        self._had_instance_forward = "forward" in model.__dict__
        self._instance_forward = model.__dict__.get("forward")

    def __enter__(self) -> "ForwardTimingRecorder":
        self._original_forward = self.model.forward
        recorder = self

        def timed_forward(_model: nn.Module, *forward_args: Any, **forward_kwargs: Any) -> Any:
            index = recorder.calls
            if index >= recorder.expected_calls:
                raise RuntimeError(
                    "official generate() made more model.forward calls than --steps"
                )
            recorder.starts[index].record()
            try:
                return recorder._original_forward(*forward_args, **forward_kwargs)
            finally:
                recorder.ends[index].record()
                recorder.calls += 1

        self.model.forward = MethodType(timed_forward, self.model)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._original_forward is None:
            return
        if self._had_instance_forward:
            self.model.forward = self._instance_forward
        else:
            delattr(self.model, "forward")

    def validate_complete(self) -> None:
        """Require one top-level model call for each configured denoising step."""

        if self.calls != self.expected_calls:
            raise RuntimeError(
                f"official generate() made {self.calls} model.forward calls; "
                f"expected {self.expected_calls}"
            )

    def model_elapsed_ms(self) -> list[float]:
        """Return CUDA duration of every observed top-level model forward."""

        return [
            float(self.starts[index].elapsed_time(self.ends[index]))
            for index in range(self.calls)
        ]

    def step_elapsed_ms(self, generation_end: torch.cuda.Event) -> list[float]:
        """Partition generation from each forward start to the next forward start."""

        values: list[float] = []
        for index in range(self.calls):
            end = self.starts[index + 1] if index + 1 < self.calls else generation_end
            values.append(float(self.starts[index].elapsed_time(end)))
        return values


def summarize_ms(values: Sequence[float]) -> tuple[float, float, float]:
    """Return population mean, population standard deviation, and CV."""

    if not values:
        raise ValueError("cannot summarize an empty timing array")
    mean = float(statistics.fmean(values))
    standard_deviation = float(statistics.pstdev(values))
    cv = standard_deviation / mean if mean else math.nan
    return mean, standard_deviation, cv


def _run_command(command: Sequence[str]) -> dict[str, Any]:
    """Run a short system query and preserve failures as structured metadata."""

    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if completed.returncode != 0:
        return {
            "ok": False,
            "returncode": int(completed.returncode),
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    return {"ok": True, "stdout": completed.stdout.strip()}


def _parse_number(value: Any, *, integer: bool = False) -> int | float | None:
    """Normalize an nvidia-smi numeric field while retaining unavailable values."""

    text = str(value).strip()
    if not text or text.lower() in {"n/a", "[not supported]", "not supported"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if integer else number


def query_nvidia_smi() -> dict[str, Any]:
    """Capture GPU telemetry and the current compute-process memory table."""

    gpu_fields = [
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "temperature.gpu",
        "clocks.sm",
        "clocks.mem",
        "power.draw",
        "pstate",
    ]
    process_fields = ["gpu_uuid", "pid", "process_name", "used_gpu_memory"]
    gpu_result = _run_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(gpu_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    process_result = _run_command(
        [
            "nvidia-smi",
            f"--query-compute-apps={','.join(process_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )

    snapshot: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": [],
        "compute_processes": [],
        "errors": [],
    }
    if gpu_result.get("ok"):
        for row in csv.reader(gpu_result["stdout"].splitlines(), skipinitialspace=True):
            if len(row) == len(gpu_fields):
                snapshot["gpus"].append(
                    {name: value.strip() for name, value in zip(gpu_fields, row)}
                )
    else:
        snapshot["errors"].append(gpu_result.get("error", "GPU query failed"))

    if process_result.get("ok"):
        for row in csv.reader(process_result["stdout"].splitlines(), skipinitialspace=True):
            if len(row) == len(process_fields):
                snapshot["compute_processes"].append(
                    {name: value.strip() for name, value in zip(process_fields, row)}
                )
    else:
        error = str(process_result.get("error", "compute-process query failed"))
        if "No running processes found" not in error:
            snapshot["errors"].append(error)
    return snapshot


def _gpu_snapshot(snapshot: Mapping[str, Any], physical_index: int) -> dict[str, Any]:
    """Select one physical GPU row from a multi-GPU nvidia-smi snapshot."""

    for row in snapshot.get("gpus", []):
        if _parse_number(row.get("index"), integer=True) == physical_index:
            return dict(row)
    return {}


def foreign_memory_mib(snapshot: Mapping[str, Any], physical_index: int) -> float | None:
    """Sum compute memory on the target GPU used by PIDs other than this worker."""

    gpu = _gpu_snapshot(snapshot, physical_index)
    gpu_uuid = gpu.get("uuid")
    if not gpu_uuid:
        return None
    total = 0.0
    observed = False
    for process in snapshot.get("compute_processes", []):
        if process.get("gpu_uuid") != gpu_uuid:
            continue
        pid = _parse_number(process.get("pid"), integer=True)
        used = _parse_number(process.get("used_gpu_memory"))
        if pid is None or used is None:
            continue
        observed = True
        if pid != os.getpid():
            total += float(used)
    return total if observed else 0.0


def run_warmups(
    model: nn.Module,
    prompt_gpu: torch.Tensor,
    args: argparse.Namespace,
    mask_id: int,
    runtime_counters: RuntimeCounters,
    generate_fn: Callable[..., Any],
) -> None:
    """Run and discard complete official generations at the exact measured shape."""

    for warmup_index in range(args.warmup_runs):
        set_global_seed(derive_seed(args.seed, int(prompt_gpu.shape[1]), "warmup", warmup_index))
        runtime_counters.reset()
        output = call_official_generate(generate_fn, model, prompt_gpu, args, mask_id)
        torch.cuda.synchronize(torch.device(args.device))
        del output


def _package_version(distribution: str) -> str | None:
    """Return an installed package version without failing the measurement."""

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_measured_generation(
    *,
    model: nn.Module,
    prompt_gpu: torch.Tensor,
    prompt_record: Mapping[str, Any],
    repeat_id: int,
    args: argparse.Namespace,
    mask_id: int,
    generate_fn: Callable[..., Any],
    generate_source: str,
    transformer_layers: int,
    runtime_counters: RuntimeCounters,
    mem_after_model_load_mib: float,
) -> dict[str, Any]:
    """Measure one complete generation and build its schema-complete record."""

    device = torch.device(args.device)
    exact_input_tokens = int(prompt_gpu.shape[1])
    sequence_length = exact_input_tokens + args.gen_length
    run_seed = int(args.seed)
    generation_start = torch.cuda.Event(enable_timing=True)
    generation_end = torch.cuda.Event(enable_timing=True)
    forward_timing = ForwardTimingRecorder(model, args.steps)
    nvidia_before = query_nvidia_smi()
    set_global_seed(run_seed)
    runtime_counters.reset()
    torch.cuda.synchronize(device)
    mem_before_generation_mib = torch.cuda.memory_allocated(device) / MIB
    torch.cuda.reset_peak_memory_stats(device)

    started_at = datetime.now(timezone.utc).isoformat()
    wall_start_ns = time.perf_counter_ns()
    generation_start.record()
    with forward_timing:
        raw_output = call_official_generate(
            generate_fn, model, prompt_gpu, args, mask_id
        )
    generation_end.record()
    torch.cuda.synchronize(device)
    wall_end_ns = time.perf_counter_ns()
    ended_at = datetime.now(timezone.utc).isoformat()

    forward_timing.validate_complete()
    output = extract_generated_ids(
        raw_output,
        prompt_length=exact_input_tokens,
        gen_length=args.gen_length,
    )
    generation_cuda_ms = float(generation_start.elapsed_time(generation_end))
    model_forward_ms = forward_timing.model_elapsed_ms()
    denoising_step_ms = forward_timing.step_elapsed_ms(generation_end)
    model_forward_calls = forward_timing.calls
    if args.variant == "no_cache_chunked":
        lmhead_chunks_per_forward = list(
            runtime_counters.lmhead_chunks_per_forward
        )
        lmhead_projected_tokens = list(runtime_counters.lmhead_projected_tokens)
        if len(lmhead_chunks_per_forward) != model_forward_calls:
            raise RuntimeError(
                "lm_head forward count mismatch: "
                f"observed={len(lmhead_chunks_per_forward)}, "
                f"expected={model_forward_calls}"
            )
        expected_lmhead_chunks = math.ceil(sequence_length / args.lmhead_chunk)
        if any(
            count != expected_lmhead_chunks
            for count in lmhead_chunks_per_forward
        ):
            raise RuntimeError(
                "lm_head chunk count mismatch: "
                f"observed={lmhead_chunks_per_forward}, "
                f"expected_each={expected_lmhead_chunks}"
            )
        if any(
            tokens != sequence_length for tokens in lmhead_projected_tokens
        ):
            raise RuntimeError(
                "lm_head projected-token count does not match forward sequence length"
            )
    else:
        lmhead_chunks_per_forward = [1] * model_forward_calls
        lmhead_projected_tokens = [sequence_length] * model_forward_calls
    model_mean, model_std, model_cv = summarize_ms(model_forward_ms)
    step_mean, step_std, step_cv = summarize_ms(denoising_step_ms)
    generation_wall_ms = (wall_end_ns - wall_start_ns) / 1e6
    derived_model_ms = float(sum(model_forward_ms))
    derived_step_ms = float(sum(denoising_step_ms))
    step_gap_pct = (
        100.0 * (derived_step_ms - generation_cuda_ms) / generation_cuda_ms
        if generation_cuda_ms
        else math.nan
    )

    max_allocated_mib = torch.cuda.max_memory_allocated(device) / MIB
    max_reserved_mib = torch.cuda.max_memory_reserved(device) / MIB
    nvidia_after = query_nvidia_smi()

    chunks_per_layer = (
        math.ceil(sequence_length / args.ffn_chunk)
        if args.variant == "no_cache_chunked"
        else 1
    )
    expected_ffn_calls = (
        chunks_per_layer * transformer_layers * model_forward_calls
    )
    if args.variant == "no_cache_chunked":
        observed_ffn_calls = runtime_counters.ffn_chunk_calls
        if observed_ffn_calls != expected_ffn_calls:
            raise RuntimeError(
                f"FFN chunk count mismatch: observed={observed_ffn_calls}, "
                f"expected={expected_ffn_calls}"
            )
    else:
        observed_ffn_calls = expected_ffn_calls

    output_ids = [int(token) for token in output[0].tolist()]
    output_hash = sha256_input_ids(output_ids)
    del output, raw_output
    gpu_before = _gpu_snapshot(nvidia_before, args.physical_gpu_index)
    gpu_after = _gpu_snapshot(nvidia_after, args.physical_gpu_index)
    frag_gap_pct = (
        100.0 * (max_reserved_mib - max_allocated_mib) / max_allocated_mib
        if max_allocated_mib
        else math.nan
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "official_generate_source": generate_source,
        "run_id": uuid.uuid4().hex,
        "variant": args.variant,
        "sample_id": str(prompt_record["sample_id"]),
        "repeat_id": int(repeat_id),
        "target_input_tokens": int(prompt_record["target_input_tokens"]),
        "exact_input_tokens": exact_input_tokens,
        "input_ids_sha256": str(prompt_record["input_ids_sha256"]),
        "gen_length": int(args.gen_length),
        "forward_sequence_length": sequence_length,
        "block_length": int(args.block_length),
        "steps": int(args.steps),
        "num_blocks": int(args.gen_length // args.block_length),
        "temperature": float(args.temperature),
        "seed": int(args.seed),
        "ffn_chunk": int(args.ffn_chunk)
        if args.variant == "no_cache_chunked"
        else None,
        "lmhead_chunk": int(args.lmhead_chunk)
        if args.variant == "no_cache_chunked"
        else None,
        "ffn_chunks_per_layer_per_forward": int(chunks_per_layer),
        "ffn_chunk_calls_total": int(observed_ffn_calls),
        "transformer_layers": int(transformer_layers),
        "model_forward_calls": int(model_forward_calls),
        "lmhead_chunks_per_forward": lmhead_chunks_per_forward,
        "lmhead_chunk_calls_total": int(sum(lmhead_chunks_per_forward)),
        "lmhead_projected_tokens": lmhead_projected_tokens,
        "model_forward_cuda_ms": model_forward_ms,
        "model_forward_timing_definition": (
            "top-level patched model.forward including chunked lm_head reduction"
            if args.variant == "no_cache_chunked"
            else "top-level original model.forward entry to return"
        ),
        "model_forward_mean_ms": model_mean,
        "model_forward_std_ms": model_std,
        "model_forward_cv": model_cv,
        "denoising_step_cuda_ms": denoising_step_ms,
        "denoising_step_timing_definition": (
            "forward_start[i] to forward_start[i+1]; final forward_start to "
            "generation_end"
        ),
        "denoising_step_mean_ms": step_mean,
        "denoising_step_std_ms": step_std,
        "denoising_step_cv": step_cv,
        "generation_wall_ms": generation_wall_ms,
        "generation_cuda_ms": generation_cuda_ms,
        "cpu_scheduling_gap_ms": generation_wall_ms - generation_cuda_ms,
        "derived_model_ms": derived_model_ms,
        "derived_step_ms": derived_step_ms,
        "step_vs_generation_gap_pct": step_gap_pct,
        "effective_tokens_per_second": (
            args.gen_length / (generation_wall_ms / 1000.0)
            if generation_wall_ms
            else math.nan
        ),
        "mem_after_model_load_MiB": mem_after_model_load_mib,
        "mem_before_generation_MiB": mem_before_generation_mib,
        "max_memory_allocated_MiB": max_allocated_mib,
        "max_memory_reserved_MiB": max_reserved_mib,
        "frag_gap_pct": frag_gap_pct,
        "transient_peak_from_load_MiB": max_allocated_mib - mem_after_model_load_mib,
        "transient_peak_from_prerun_MiB": max_allocated_mib
        - mem_before_generation_mib,
        "output_token_ids": output_ids,
        "output_sha256": output_hash,
        "status": "completed",
        "oom_stage": None,
        "error": None,
        "nvidia_smi_before": nvidia_before,
        "nvidia_smi_after": nvidia_after,
        "foreign_mem_MiB_before": foreign_memory_mib(
            nvidia_before, args.physical_gpu_index
        ),
        "foreign_mem_MiB_after": foreign_memory_mib(
            nvidia_after, args.physical_gpu_index
        ),
        "gpu_clock_MHz_start": _parse_number(gpu_before.get("clocks.sm")),
        "gpu_clock_MHz_end": _parse_number(gpu_after.get("clocks.sm")),
        "gpu_temp_C_start": _parse_number(gpu_before.get("temperature.gpu")),
        "gpu_temp_C_end": _parse_number(gpu_after.get("temperature.gpu")),
        "torch_version": torch.__version__,
        "transformers_version": _package_version("transformers"),
        "cuda_version": torch.version.cuda,
        "driver_version": gpu_before.get("driver_version")
        or gpu_after.get("driver_version"),
        "gpu_name": gpu_before.get("name")
        or gpu_after.get("name")
        or torch.cuda.get_device_name(device),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
    }


def make_failure_record(
    *,
    args: argparse.Namespace,
    status: str,
    stage: str,
    error: BaseException,
    prompt_record: Mapping[str, Any] | None = None,
    repeat_id: int | None = None,
) -> dict[str, Any]:
    """Build a durable OOM/invalid record when a cell cannot complete."""

    snapshot = query_nvidia_smi()
    gpu = _gpu_snapshot(snapshot, args.physical_gpu_index)
    exact_input_tokens = (
        int(prompt_record["exact_input_tokens"])
        if prompt_record is not None
        else int(args.target_input_tokens)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "official_generate_source": getattr(
            args, "official_generate_source", None
        ),
        "run_id": uuid.uuid4().hex,
        "variant": args.variant,
        "sample_id": str(prompt_record["sample_id"]) if prompt_record else None,
        "repeat_id": repeat_id,
        "target_input_tokens": int(args.target_input_tokens),
        "exact_input_tokens": exact_input_tokens,
        "input_ids_sha256": str(prompt_record["input_ids_sha256"])
        if prompt_record
        else None,
        "gen_length": int(args.gen_length),
        "forward_sequence_length": exact_input_tokens + int(args.gen_length),
        "block_length": int(args.block_length),
        "steps": int(args.steps),
        "num_blocks": int(args.gen_length // args.block_length),
        "temperature": float(args.temperature),
        "seed": int(args.seed),
        "ffn_chunk": int(args.ffn_chunk)
        if args.variant == "no_cache_chunked"
        else None,
        "lmhead_chunk": int(args.lmhead_chunk)
        if args.variant == "no_cache_chunked"
        else None,
        "ffn_chunks_per_layer_per_forward": None,
        "ffn_chunk_calls_total": None,
        "transformer_layers": None,
        "model_forward_calls": None,
        "lmhead_chunks_per_forward": [],
        "lmhead_chunk_calls_total": None,
        "lmhead_projected_tokens": [],
        "model_forward_cuda_ms": [],
        "model_forward_timing_definition": (
            "top-level patched model.forward including chunked lm_head reduction"
            if args.variant == "no_cache_chunked"
            else "top-level original model.forward entry to return"
        ),
        "model_forward_mean_ms": None,
        "model_forward_std_ms": None,
        "model_forward_cv": None,
        "denoising_step_cuda_ms": [],
        "denoising_step_timing_definition": (
            "forward_start[i] to forward_start[i+1]; final forward_start to "
            "generation_end"
        ),
        "denoising_step_mean_ms": None,
        "denoising_step_std_ms": None,
        "denoising_step_cv": None,
        "generation_wall_ms": None,
        "generation_cuda_ms": None,
        "cpu_scheduling_gap_ms": None,
        "derived_model_ms": None,
        "derived_step_ms": None,
        "step_vs_generation_gap_pct": None,
        "effective_tokens_per_second": None,
        "mem_after_model_load_MiB": None,
        "mem_before_generation_MiB": None,
        "max_memory_allocated_MiB": None,
        "max_memory_reserved_MiB": None,
        "frag_gap_pct": None,
        "transient_peak_from_load_MiB": None,
        "transient_peak_from_prerun_MiB": None,
        "output_token_ids": [],
        "output_sha256": None,
        "status": status,
        "oom_stage": stage if status == "oom" else None,
        "error": f"{type(error).__name__}: {error}",
        "nvidia_smi_before": snapshot,
        "nvidia_smi_after": snapshot,
        "foreign_mem_MiB_before": foreign_memory_mib(
            snapshot, args.physical_gpu_index
        ),
        "foreign_mem_MiB_after": foreign_memory_mib(
            snapshot, args.physical_gpu_index
        ),
        "gpu_clock_MHz_start": _parse_number(gpu.get("clocks.sm")),
        "gpu_clock_MHz_end": _parse_number(gpu.get("clocks.sm")),
        "gpu_temp_C_start": _parse_number(gpu.get("temperature.gpu")),
        "gpu_temp_C_end": _parse_number(gpu.get("temperature.gpu")),
        "torch_version": torch.__version__,
        "transformers_version": _package_version("transformers"),
        "cuda_version": torch.version.cuda,
        "driver_version": gpu.get("driver_version"),
        "gpu_name": gpu.get("name"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def ordered_records_for_repeat(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    target_input_tokens: int,
    repeat_id: int,
) -> list[dict[str, Any]]:
    """Create the same deterministic per-repeat sample order for both variants."""

    ordered = list(records)
    rng = random.Random(
        derive_seed(seed, target_input_tokens, repeat_id, "sample_order")
    )
    rng.shuffle(ordered)
    return ordered


def validate_prompt_bank(
    prompt_bank: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Validate every selected frozen tensor without loading an inference model."""

    for record in records:
        load_frozen_prompt(prompt_bank, record)
    print(f"validated {len(records)} frozen prompts")


def main(variant: str) -> None:
    """Run one fixed variant/length cell in a fresh process."""

    args = parse_args(variant)
    prompt_bank = args.prompt_bank.resolve()
    available_lengths = discover_target_lengths(prompt_bank)
    if args.target_input_tokens not in available_lengths:
        raise ValueError(
            f"target_input_tokens={args.target_input_tokens} is absent from the frozen "
            f"manifest; available={available_lengths}"
        )
    records = load_prompt_records(
        prompt_bank, args.target_input_tokens, args.sample_ids
    )
    if args.mode == "validate-prompts":
        validate_prompt_bank(prompt_bank, records)
        return

    loaded_prompts: dict[str, torch.Tensor] = {
        str(record["sample_id"]): load_frozen_prompt(prompt_bank, record)
        for record in records
    }
    runtime_counters = RuntimeCounters()
    current_stage = "generate_import"
    current_record: Mapping[str, Any] | None = None
    current_repeat: int | None = None

    try:
        generate_fn, generate_source = load_official_generate(args)
        args.official_generate_source = generate_source
        current_stage = "model_load"
        model = load_model(args)
        mask_id = (
            int(args.mask_id)
            if args.mask_id is not None
            else int(getattr(model.config, "mask_token_id", 126336))
        )
        transformer = _find_transformer(model)
        transformer_layers = len(transformer.blocks)
        if args.variant == "no_cache_chunked":
            installed_layers = install_chunked_ffn(
                model, args.ffn_chunk, runtime_counters
            )
            if installed_layers != transformer_layers:
                raise RuntimeError(
                    f"installed FFN patch on {installed_layers}/{transformer_layers} layers"
                )
            installed_vocab_size = install_chunked_lmhead(
                model, args.lmhead_chunk, runtime_counters
            )
            if installed_vocab_size != int(model.config.vocab_size):
                raise RuntimeError(
                    f"installed lm_head vocab={installed_vocab_size}, "
                    f"config vocab={model.config.vocab_size}"
                )

        for prompt_cpu in loaded_prompts.values():
            validate_prompt_for_model(prompt_cpu, model, mask_id)
        torch.cuda.synchronize(torch.device(args.device))
        mem_after_model_load_mib = (
            torch.cuda.memory_allocated(torch.device(args.device)) / MIB
        )

        current_stage = "warmup"
        warmup_by_exact_length: dict[int, Mapping[str, Any]] = {}
        for record in records:
            warmup_by_exact_length.setdefault(int(record["exact_input_tokens"]), record)
        for warmup_record in warmup_by_exact_length.values():
            current_record = warmup_record
            warmup_gpu = loaded_prompts[str(warmup_record["sample_id"])].unsqueeze(0).to(
                args.device
            )
            run_warmups(
                model,
                warmup_gpu,
                args,
                mask_id,
                runtime_counters,
                generate_fn,
            )
            del warmup_gpu

        current_stage = "generation"
        for repeat_id in range(args.repeats):
            current_repeat = repeat_id
            ordered = ordered_records_for_repeat(
                records,
                seed=args.seed,
                target_input_tokens=args.target_input_tokens,
                repeat_id=repeat_id,
            )
            for prompt_record in ordered:
                current_record = prompt_record
                prompt_gpu = loaded_prompts[str(prompt_record["sample_id"])].unsqueeze(0).to(
                    args.device
                )
                result = run_measured_generation(
                    model=model,
                    prompt_gpu=prompt_gpu,
                    prompt_record=prompt_record,
                    repeat_id=repeat_id,
                    args=args,
                    mask_id=mask_id,
                    generate_fn=generate_fn,
                    generate_source=generate_source,
                    transformer_layers=transformer_layers,
                    runtime_counters=runtime_counters,
                    mem_after_model_load_mib=mem_after_model_load_mib,
                )
                append_jsonl(args.output, result)
                print(
                    f"[{args.variant}] target_L={args.target_input_tokens} "
                    f"repeat={repeat_id} sample={prompt_record['sample_id']} "
                    f"wall={result['generation_wall_ms']:.3f} ms "
                    f"peak={result['max_memory_allocated_MiB']:.1f} MiB"
                )
                del prompt_gpu
    except torch.cuda.OutOfMemoryError as error:
        failure = make_failure_record(
            args=args,
            status="oom",
            stage=current_stage,
            error=error,
            prompt_record=current_record,
            repeat_id=current_repeat,
        )
        append_jsonl(args.output, failure)
        raise SystemExit(2) from error
    except Exception as error:
        failure = make_failure_record(
            args=args,
            status="invalid",
            stage=current_stage,
            error=error,
            prompt_record=current_record,
            repeat_id=current_repeat,
        )
        append_jsonl(args.output, failure)
        raise
