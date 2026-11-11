"""dLLM-Cache adapter contract.

This file is deliberately non-runnable until the cache algorithm is actually
implemented. It must preserve the same output semantics and generation
parameters as `generate_no_cache.py`, while keeping all cache state local to a
single prompt. Never replace this with a call to the no-cache generator merely
to make the cache runner execute.
"""

from __future__ import annotations

from typing import Any

import torch


CACHE_IMPLEMENTED = False


@torch.inference_mode()
def generate_dllm_cache(
    model,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    condition_config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return full output IDs and cache-specific metrics.

    Required implementation rules:

    - Read exact cache parameters from `condition_config`.
    - Do not alter the frozen prompt or generation settings.
    - Initialize and destroy cache state inside this call; never leak state
      across sample IDs.
    - Return `(output_ids, metrics)`, where `output_ids` includes prompt and
      generated positions, matching `generate_no_cache`.
    - Include useful metrics such as refreshed/reused layer-token counts and
      effective cache-hit fraction when the algorithm supports them.
    """
    del (
        model,
        prompt,
        attention_mask,
        steps,
        gen_length,
        block_length,
        temperature,
        cfg_scale,
        remasking,
        mask_id,
        logits_eos_inf,
        confidence_eos_eot_inf,
        condition_config,
    )
    raise NotImplementedError(
        "Implement dLLM-Cache here, set CACHE_IMPLEMENTED=True, finalize "
        "config/cache_v1.json, then run freeze_cache_condition.py"
    )

