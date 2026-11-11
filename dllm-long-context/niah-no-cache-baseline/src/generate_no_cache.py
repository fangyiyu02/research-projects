"""Strict full-recomputation LLaDA generation for the fresh baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def transfer_schedule(mask_token_count: int, steps: int) -> list[int]:
    """Return the exact integer unmask count for each step without GPU sync."""
    base, remainder = divmod(mask_token_count, steps)
    return [base + (1 if index < remainder else 0) for index in range(steps)]


@torch.inference_mode()
def generate_no_cache(
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
    """Run one complete-sequence model forward at every denoising step.

    No `past_key_values`, KV tensor, feature cache, or cross-step model state is
    accepted or retained. `condition_config` is accepted only so this function
    shares the cache adapter's runner interface.
    """
    del condition_config
    if prompt.ndim != 2:
        raise ValueError("prompt must have shape [batch, sequence]")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks:
        raise ValueError("steps must be divisible by number of blocks")
    steps_per_block = steps // num_blocks

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=prompt.device,
    )
    x[:, : prompt.shape[1]] = prompt
    if attention_mask is not None:
        generated_mask = torch.ones(
            (prompt.shape[0], gen_length),
            dtype=attention_mask.dtype,
            device=prompt.device,
        )
        attention_mask = torch.cat([attention_mask, generated_mask], dim=-1)

    prompt_index = x != mask_id
    forward_calls = 0
    for block_index in range(num_blocks):
        block_start = prompt.shape[1] + block_index * block_length
        block_end = prompt.shape[1] + (block_index + 1) * block_length
        counts = transfer_schedule(block_length, steps_per_block)

        for step_index, transfer_count in enumerate(counts):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                unconditional_x = x.clone()
                unconditional_x[prompt_index] = mask_id
                model_input = torch.cat([x, unconditional_x], dim=0)
                model_attention_mask = None
                if attention_mask is not None:
                    model_attention_mask = torch.cat(
                        [attention_mask, attention_mask], dim=0
                    )
                model_output = model(
                    model_input, attention_mask=model_attention_mask
                )
                conditional_logits, unconditional_logits = torch.chunk(
                    model_output.logits, 2, dim=0
                )
                logits = unconditional_logits + (cfg_scale + 1.0) * (
                    conditional_logits - unconditional_logits
                )
            else:
                # Defining no-cache operation: the complete current sequence is
                # recomputed, and no state is passed into the next call.
                model_output = model(x, attention_mask=attention_mask)
                logits = model_output.logits
            forward_calls += 1

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = -torch.inf
                logits[:, :, 126348] = -torch.inf

            if remasking == "low_confidence":
                probabilities = F.softmax(logits, dim=-1)
                x0_probability = torch.gather(
                    probabilities, dim=-1, index=x0.unsqueeze(-1)
                ).squeeze(-1)
            elif remasking == "random":
                x0_probability = torch.rand(
                    x0.shape, device=x0.device, dtype=torch.float32
                )
            else:
                raise NotImplementedError(remasking)

            x0_probability[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_probability, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for batch_index in range(confidence.shape[0]):
                _, selected = torch.topk(
                    confidence[batch_index], k=int(transfer_count)
                )
                transfer_index[batch_index, selected] = True
            x[transfer_index] = x0[transfer_index]

            del model_output, logits, logits_with_noise

    return x, {
        "cache_enabled": False,
        "full_sequence_model_forwards": forward_calls,
        "configured_denoising_steps": steps,
    }

