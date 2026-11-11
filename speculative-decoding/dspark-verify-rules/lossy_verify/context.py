"""VerifyContext: what the hook in base_evaluator.verify_draft_tokens hands to a rule."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

# q_k definition (ASD single-point gate r_k / q_k <= g).
#   True : q_k = K - k      (k from 0; last position has q = 1, no division by zero)
#   False: q_k = K - k - 1  (q = "tokens strictly after k"; last position has q = 0)
# proposal §2.3 says "该位置之后 block 内剩余 token 数" (= False); 代码方案 §2.1 says K - i (= True).
# Confirm against the ASD paper and set once; config.json records the value used.
Q_INCLUDES_SELF = True

_FIELDS = (
    "target_logits",
    "target_probs",
    "draft_tokens",
    "draft_probs",
    "draft_logits",
    "confidence_logits",
    "strict_mask",
    "start",
)


@dataclass
class VerifyContext:
    target_logits: torch.Tensor  # [1, K, V] raw target logits for the K draft positions
    target_probs: torch.Tensor  # [1, K, V] after logits_to_probs (one-hot at T=0) — not used for r_i
    draft_tokens: torch.Tensor  # [1, K]
    draft_probs: torch.Tensor  # [1, K, V] (one-hot at T=0)
    draft_logits: torch.Tensor | None  # [1, K, V] raw drafter logits (needs the draft_ops patch)
    confidence_logits: torch.Tensor | None  # [1, K]
    strict_mask: torch.Tensor  # [1, K] int64 0/1, 1 = draft token == target argmax (at T=0)
    start: int
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_namespace(cls, ns: SimpleNamespace) -> "VerifyContext":
        return cls(**{name: getattr(ns, name) for name in _FIELDS})

    # ---- sizes -----------------------------------------------------------
    @property
    def K(self) -> int:
        return int(self.draft_tokens.shape[1])

    # ---- derived quantities, computed once ---------------------------------
    def _target_logprobs(self) -> torch.Tensor:  # [K, V] float32
        if "lp" not in self._cache:
            self._cache["lp"] = torch.log_softmax(self.target_logits[0].float(), dim=-1)
        return self._cache["lp"]

    def target_argmax(self) -> torch.Tensor:  # [K]
        if "argmax" not in self._cache:
            self._cache["argmax"] = self.target_logits[0].argmax(dim=-1)
        return self._cache["argmax"]

    def target_logprob_of_token(self) -> torch.Tensor:  # [K] log p_T(x_k)
        if "lp_x" not in self._cache:
            lp = self._target_logprobs()
            self._cache["lp_x"] = lp.gather(-1, self.draft_tokens[0].unsqueeze(-1)).squeeze(-1)
        return self._cache["lp_x"]

    def target_logprob_top1(self) -> torch.Tensor:  # [K] log p_T(y*_k)
        if "lp_top" not in self._cache:
            lp = self._target_logprobs()
            self._cache["lp_top"] = lp.gather(-1, self.target_argmax().unsqueeze(-1)).squeeze(-1)
        return self._cache["lp_top"]

    def regret(self) -> torch.Tensor:  # [K] r_k = log p_T(y*) - log p_T(x_k) >= 0 ; 0 at matches
        if "regret" not in self._cache:
            self._cache["regret"] = (self.target_logprob_top1() - self.target_logprob_of_token()).clamp_min(0.0)
        return self._cache["regret"]

    def remaining(self) -> torch.Tensor:  # [K] q_k
        if "q" not in self._cache:
            k = torch.arange(self.K, device=self.draft_tokens.device)
            q = self.K - k if Q_INCLUDES_SELF else self.K - k - 1
            self._cache["q"] = q.float()
        return self._cache["q"]

    def confidence(self) -> torch.Tensor | None:  # [K] c_k = sigmoid(confidence_logit)
        if self.confidence_logits is None:
            return None
        if "c" not in self._cache:
            self._cache["c"] = torch.sigmoid(self.confidence_logits[0].float())
        return self._cache["c"]

    def draft_prob_of_token(self) -> torch.Tensor | None:  # [K] p_D(x_k) from raw drafter logits
        if self.draft_logits is None:
            return None
        if "pD_x" not in self._cache:
            lp = torch.log_softmax(self.draft_logits[0].float(), dim=-1)
            self._cache["pD_x"] = lp.gather(-1, self.draft_tokens[0].unsqueeze(-1)).squeeze(-1).exp()
        return self._cache["pD_x"]

    def target_prob_of_token(self) -> torch.Tensor:  # [K] p_T(x_k)
        return self.target_logprob_of_token().exp()

    def target_top1_prob(self) -> torch.Tensor:  # [K] p_T(y*_k)
        return self.target_logprob_top1().exp()
