"""Two jsonl logs per run.

mismatches.jsonl — one row per mismatch the verify loop actually examined (up to and including the
                   first rejected one in each round). Fields = proposal §4.
positions.jsonl  — one row per draft position per round, for ALL positions (match and mismatch),
                   so c_k / r_k histograms at match vs mismatch positions can be drawn (卡 0 附带产出).
Rows are buffered per sample and flushed in end_sample() with sample_final_correct filled in.
"""
from __future__ import annotations

import json
from pathlib import Path

from .context import VerifyContext


class MismatchLogger:
    def __init__(self, out_dir: str | Path, *, log_positions: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.mismatch_path = self.out_dir / "mismatches.jsonl"
        self.positions_path = self.out_dir / "positions.jsonl"
        self.log_positions_enabled = log_positions
        self._mismatch_fh = self.mismatch_path.open("w", encoding="utf-8")
        self._positions_fh = self.positions_path.open("w", encoding="utf-8") if log_positions else None
        self.sample_id = None
        self.round_id = 0
        self._mrows: list[dict] = []
        self._prows: list[dict] = []
        # running totals for summary.json
        self.total_examined = 0
        self.total_accepted = 0

    # ---- per sample --------------------------------------------------------
    def start_sample(self, sample_id) -> None:
        self.sample_id = sample_id
        self.round_id = 0
        self._mrows, self._prows = [], []

    def end_sample(self, final_correct: bool) -> None:
        for row in self._mrows:
            row["sample_final_correct"] = bool(final_correct)
            self._mismatch_fh.write(json.dumps(row) + "\n")
        if self._positions_fh is not None:
            for row in self._prows:
                self._positions_fh.write(json.dumps(row) + "\n")
        self._mismatch_fh.flush()
        if self._positions_fh is not None:
            self._positions_fh.flush()
        self._mrows, self._prows = [], []

    def close(self) -> None:
        self._mismatch_fh.close()
        if self._positions_fh is not None:
            self._positions_fh.close()

    # ---- per round ---------------------------------------------------------
    def log_positions(self, ctx: VerifyContext) -> None:
        """Called once per verify round (first thing in LossyRule.__call__). Advances round_id."""
        self.round_id += 1
        if not self.log_positions_enabled:
            return
        c = ctx.confidence()
        pD = ctx.draft_prob_of_token()
        r = ctx.regret().tolist()
        q = ctx.remaining().tolist()
        pT = ctx.target_prob_of_token().tolist()
        strict = ctx.strict_mask[0].tolist()
        c = c.tolist() if c is not None else [None] * ctx.K
        pD = pD.tolist() if pD is not None else [None] * ctx.K
        for k in range(ctx.K):
            self._prows.append({
                "sample_id": self.sample_id,
                "round_id": self.round_id,
                "pos_k": k,
                "strict_match": bool(strict[k]),
                "c_k": c[k],
                "r_i": r[k],
                "q_i": q[k],
                "p_T_xk": pT[k],
                "p_D_xk": pD[k],
            })

    def log_mismatch(self, ctx: VerifyContext, k: int, *, accepted: bool, rule_name: str,
                     cum_regret_after: float, n_exc_after: int) -> None:
        c = ctx.confidence()
        pD = ctx.draft_prob_of_token()
        self.total_examined += 1
        self.total_accepted += int(accepted)
        self._mrows.append({
            "sample_id": self.sample_id,
            "round_id": self.round_id,
            "pos_k": k,
            "draft_token": int(ctx.draft_tokens[0, k]),
            "target_argmax": int(ctx.target_argmax()[k]),
            "r_i": float(ctx.regret()[k]),
            "q_i": float(ctx.remaining()[k]),
            "c_k": float(c[k]) if c is not None else None,
            "p_D_xk": float(pD[k]) if pD is not None else None,
            "p_T_xk": float(ctx.target_prob_of_token()[k]),
            "p_T_top1": float(ctx.target_top1_prob()[k]),
            "accepted": bool(accepted),
            "rule_name": rule_name,
            "cum_regret_after": float(cum_regret_after),
            "n_mismatch_in_block_after": int(n_exc_after),
            "sample_final_correct": None,  # filled in end_sample
        })
