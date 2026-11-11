"""Idea B verify rule.

At every mismatch in a verify round, decide whether the draft token E counts as a match:
    E   = draft token at position k
    E*  = target argmax at position k
    z   = target logits at the next position k+1  (already computed by verify, with E as prefix)
    d   = M(E*) - M(E)    M(x) = drafter Markov-head bias for "previous token = x"
    tv  = TV( softmax(z), softmax(z + d) )
    tv <= theta  -> accept E, keep checking later positions the same way
    tv >  theta  -> truncate here (same as the original rule)
Nothing else limits acceptance. r_i is never used in the decision.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from lossy_verify.context import VerifyContext


class IdeaBRule:
    name = "ideab"

    def __init__(self, *, theta: float, logger=None):
        self.theta = float(theta)
        self.logger = logger
        self.markov = None

    def bind_draft_model(self, draft_model) -> None:
        self.markov = getattr(draft_model, "markov_head", None)
        if self.markov is None:
            raise RuntimeError("drafter has no markov_head")

    def reset(self) -> None:
        pass

    def params(self) -> dict:
        return {"theta": self.theta}

    def tv_est(self, z: torch.Tensor, e: torch.Tensor, e_star: torch.Tensor) -> float:
        z = z.float()
        bias = self.markov.compute_step_bias(torch.stack([e_star, e]), None).float()  # [2, V]
        if bias.shape[-1] != z.shape[-1]:
            raise RuntimeError(f"vocab mismatch: target {z.shape[-1]} vs markov {bias.shape[-1]}")
        p = torch.softmax(z, dim=-1)
        q = torch.softmax(z + (bias[0] - bias[1]), dim=-1)
        return float(0.5 * (p - q).abs().sum())

    def __call__(self, ns: SimpleNamespace) -> torch.Tensor:
        logits_all = getattr(ns, "target_logits_all", None)  # [1, K+1, V]
        if logits_all is None:
            raise RuntimeError("target_logits_all missing: add the one-line hook change in DeepSpec (see idea_b/STEPS.md)")
        if self.markov is None:
            raise RuntimeError("call bind_draft_model() before decoding")
        ctx = VerifyContext.from_namespace(ns)  # only for logging and target argmax
        if self.logger is not None:
            self.logger.log_positions(ctx)

        mask = ctx.strict_mask.clone()
        target_argmax = ctx.target_argmax()
        n_accepted = 0
        for k in range(ctx.K):
            if int(mask[0, k]) == 1:
                continue  # match: accepted as usual
            tv = self.tv_est(logits_all[0, k + 1], ctx.draft_tokens[0, k], target_argmax[k])
            ok = tv <= self.theta
            if ok:
                mask[0, k] = 1
                n_accepted += 1
            if self.logger is not None:
                self.logger.log_mismatch_b(ctx, k, accepted=ok, tv_est=tv, n_accepted=n_accepted)
            if not ok:
                break  # cumprod truncates here
        return mask
