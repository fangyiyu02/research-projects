"""LossyDSparkEvaluator: DeepSpec's Qwen3DSparkEvaluator + accept_rule + per-sample grading/timing.

Everything model-related (loading, _propose, _update, _post_verify, ConfidenceHeadRecorder) is
inherited untouched. Only run_dataset and generate_one_sample are overridden.
run_dataset mirrors BaseEvaluator.run_dataset step for step (same seed_all calls, same shuffle),
so the strict rule must reproduce the original eval.py token-for-token.
"""
from __future__ import annotations

import random
import time
from types import SimpleNamespace

import torch

from deepspec.eval.base_evaluator import (
    generate_decoding_sample,
    has_stop_token,
    resolve_stop_token_ids,
)
from deepspec.data.parser import encode_chat_messages
from deepspec.eval.dspark import Qwen3DSparkEvaluator
from deepspec.utils import seed_all

from .metrics import grade


class LossyDSparkEvaluator(Qwen3DSparkEvaluator):
    def __init__(self, local_rank: int, args, *, rule, logger, dataset_rows: list[dict]):
        super().__init__(local_rank, args)
        self.rule = rule
        self.logger = logger
        self.dataset_rows = dataset_rows  # rows with id / turns / answer
        self.sample_records: list[dict] = []

    # ---- override: same loop as BaseEvaluator.run_dataset, plus per-sample bookkeeping ----
    def run_dataset(self, *, dataset_name: str, max_samples: int | None):
        seed = int(self.args.seed)
        seed_all(seed)
        dataset = list(self.dataset_rows)
        if max_samples is not None and len(dataset) > max_samples:
            rng = random.Random(seed)
            rng.shuffle(dataset)
            dataset = dataset[:max_samples]

        stop_token_ids = resolve_stop_token_ids(self.target_model, self.tokenizer)
        max_new_tokens = int(self.args.max_new_tokens)
        responses = []
        for idx in range(self.global_rank, len(dataset), self.world_size):
            seed_all(seed + idx)
            instance = dataset[idx]
            messages = [{"role": "user", "content": instance["turns"][0]}]
            input_ids = encode_chat_messages(
                self.tokenizer, messages, add_generation_prompt=True, enable_thinking=False,
            ).to(self.device)

            self.rule.reset()
            self.logger.start_sample(instance["id"])
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            resp = self.generate_one_sample(input_ids=input_ids, stop_token_ids=stop_token_ids)
            torch.cuda.synchronize(self.device)
            wall = time.perf_counter() - t0

            gen_ids = resp.output_ids[0, resp.num_input_tokens:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            g = grade(text, instance["answer"])
            hit_max = (resp.num_output_tokens >= max_new_tokens) and not has_stop_token(
                gen_ids.unsqueeze(0), stop_token_ids
            )
            self.logger.end_sample(g["correct_fallback"])

            self.sample_records.append({
                "id": instance["id"],
                "dataset_idx": idx,
                "gold": instance["answer"],
                "text": text,
                **g,
                "num_output_tokens": int(resp.num_output_tokens),
                "hit_max_new_tokens": bool(hit_max),
                "verify_rounds": int(resp.verify_count),  # = target forwards after prefill
                "acceptance_lengths": [int(x) for x in resp.acceptance_lengths],
                "proposal_lengths": [int(x) for x in resp.proposal_lengths],
                "wall_s": wall,
                "output_ids": [int(x) for x in gen_ids.tolist()],  # for two-run identity checks
            })
            responses.append(resp)
            if (len(responses) % 20) == 0:
                done = len(responses)
                acc = sum(r["correct_fallback"] for r in self.sample_records[-done:]) / done
                print(f"[{dataset_name}] {done}/{len(dataset)} acc_fallback={acc:.3f}", flush=True)
        return responses

    # ---- override: pass the rule into the decoding loop ----
    def generate_one_sample(self, *, input_ids, stop_token_ids) -> SimpleNamespace:
        return generate_decoding_sample(
            target_model=self.target_model,
            input_ids=input_ids,
            max_new_tokens=int(self.args.max_new_tokens),
            max_proposal_tokens=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            stop_token_ids=stop_token_ids,
            init_context=self._init_context,
            propose=self._propose,
            update=self._update,
            post_verify=self._post_verify,
            accept_rule=self.rule,
        )
