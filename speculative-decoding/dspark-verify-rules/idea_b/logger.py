"""Idea B logger: the lossy_verify logger plus a tv_est column. Subclass only; the original is untouched."""
from __future__ import annotations

from lossy_verify.logger import MismatchLogger


class IdeaBLogger(MismatchLogger):
    def log_mismatch_b(self, ctx, k, *, accepted: bool, tv_est: float, n_accepted: int) -> None:
        self.log_mismatch(ctx, k, accepted=accepted, rule_name="ideab",
                          cum_regret_after=0.0, n_exc_after=n_accepted)
        self._mrows[-1]["tv_est"] = float(tv_est)
