"""Acceptance rules. Base class runs the three ASD gates; subclasses only define the single-point gate.

Rule names are the ones in proposal §6.1: strict, asd, a1, a2, a3, p.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from .context import VerifyContext


class LossyRule:
    name = "base"

    def __init__(self, *, M: int | None, B: float | None, logger=None):
        self.M = M  # per-block cap on accepted mismatches (None = off)
        self.B = B  # per-request regret budget (None = off; `spent` still accumulates for logging)
        self.logger = logger
        self.spent = 0.0

    def reset(self) -> None:
        """Call at the start of every sample."""
        self.spent = 0.0

    def params(self) -> dict:
        return {"M": self.M, "B": self.B}

    def gate(self, ctx: VerifyContext, k: int) -> bool:  # single-point gate at mismatch position k
        raise NotImplementedError

    def __call__(self, ns: SimpleNamespace) -> torch.Tensor:
        ctx = VerifyContext.from_namespace(ns)
        if self.logger is not None:
            self.logger.log_positions(ctx)  # also advances round_id

        mask = ctx.strict_mask.clone()
        regret = ctx.regret()
        n_exc = 0
        for k in range(ctx.K):
            if int(mask[0, k]) == 1:
                continue  # match: accepted as usual
            r = float(regret[k])
            ok = self.gate(ctx, k)
            if ok and self.M is not None and n_exc >= self.M:
                ok = False
            if ok and self.B is not None and self.spent + r > self.B:
                ok = False
            if ok:
                mask[0, k] = 1
                n_exc += 1
                self.spent += r
            if self.logger is not None:
                self.logger.log_mismatch(
                    ctx, k, accepted=ok, rule_name=self.name,
                    cum_regret_after=self.spent, n_exc_after=n_exc,
                )
            if not ok:
                break  # cumprod truncates here; later positions are never examined
        return mask


class StrictRule(LossyRule):
    """Original DSpark verify. Must reproduce accept_rule=None token-for-token."""
    name = "strict"

    def __init__(self, *, logger=None, **_):
        super().__init__(M=None, B=None, logger=logger)

    def gate(self, ctx, k):
        return False


class ASDRule(LossyRule):
    """ASD (arXiv 2608.03447): r_k / q_k <= g, with M, B."""
    name = "asd"

    def __init__(self, *, g: float, M: int | None = 2, B: float | None = 8.0, logger=None):
        super().__init__(M=M, B=B, logger=logger)
        self.g = float(g)

    def params(self):
        return {**super().params(), "g": self.g}

    def gate(self, ctx, k):
        return float(ctx.regret()[k] / ctx.remaining()[k]) <= self.g


class A1ConfidenceRule(LossyRule):
    """A1: c_k >= theta replaces the regret gate; M, B kept."""
    name = "a1"

    def __init__(self, *, theta: float, M: int | None = 2, B: float | None = 8.0, logger=None):
        super().__init__(M=M, B=B, logger=logger)
        self.theta = float(theta)

    def params(self):
        return {**super().params(), "theta": self.theta}

    def gate(self, ctx, k):
        c = ctx.confidence()
        if c is None:
            raise RuntimeError("confidence_logits is None: drafter has no confidence head?")
        return float(c[k]) >= self.theta


class A2ConfAndRegretRule(A1ConfidenceRule):
    """A2: c_k >= theta AND r_k / q_k <= g; M, B kept."""
    name = "a2"

    def __init__(self, *, theta: float, g: float, M: int | None = 2, B: float | None = 8.0, logger=None):
        super().__init__(theta=theta, M=M, B=B, logger=logger)
        self.g = float(g)

    def params(self):
        return {**super().params(), "g": self.g}

    def gate(self, ctx, k):
        return super().gate(ctx, k) and float(ctx.regret()[k] / ctx.remaining()[k]) <= self.g


class A3ConfOnlyRule(A1ConfidenceRule):
    """A3 (ablation): c_k >= theta with M and B removed."""
    name = "a3"

    def __init__(self, *, theta: float, logger=None, **_):
        super().__init__(theta=theta, M=None, B=None, logger=logger)


class PDraftProbRule(LossyRule):
    """P (ablation): p_D(x_k) >= theta_p from the drafter's own softmax; M, B kept."""
    name = "p"

    def __init__(self, *, theta: float, M: int | None = 2, B: float | None = 8.0, logger=None):
        super().__init__(M=M, B=B, logger=logger)
        self.theta = float(theta)

    def params(self):
        return {**super().params(), "theta": self.theta}

    def gate(self, ctx, k):
        p = ctx.draft_prob_of_token()
        if p is None:
            raise RuntimeError(
                "draft_logits is None: the draft_ops.py part of lossy-verify-hook.patch is not applied."
            )
        return float(p[k]) >= self.theta


RULES = {
    "strict": StrictRule,
    "asd": ASDRule,
    "a1": A1ConfidenceRule,
    "a2": A2ConfAndRegretRule,
    "a3": A3ConfOnlyRule,
    "p": PDraftProbRule,
}


def build_rule(name: str, *, theta=None, g=0.25, M=2, B=8.0, logger=None) -> LossyRule:
    if name not in RULES:
        raise ValueError(f"unknown rule {name!r}; choose from {sorted(RULES)}")
    kw = {"logger": logger}
    if name in ("asd", "a2"):
        kw["g"] = g
    if name in ("a1", "a2", "a3", "p"):
        if theta is None:
            raise ValueError(f"rule {name!r} needs --theta")
        kw["theta"] = theta
    if name in ("asd", "a1", "a2", "p"):
        kw["M"] = M
        kw["B"] = B
    return RULES[name](**kw)
