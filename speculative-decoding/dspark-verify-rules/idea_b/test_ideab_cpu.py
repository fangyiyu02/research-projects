"""CPU test for the Idea B rule and logger. No GPU, no model.
    cd /workspace && python -m idea_b.test_ideab_cpu
"""
import json
import tempfile
import types
from pathlib import Path

import torch

from idea_b.logger import IdeaBLogger
from idea_b.rule import IdeaBRule

V, K = 8, 7


class FakeMarkov:
    """Bias row per previous token. Tokens 3 and 4 have identical rows (tv 0); token 5 differs from 6."""
    def __init__(self):
        self.W = torch.zeros(V, V)
        self.W[3] = self.W[4] = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0])
        self.W[5] = torch.tensor([0., 0, 0, 0, 0, 0, 0, 9])
        self.W[6] = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0])

    def compute_step_bias(self, ids, hidden_states):
        return self.W[ids]


fake = types.SimpleNamespace(markov_head=FakeMarkov())
draft = [1, 3, 2, 5, 2, 2, 2]
target = [1, 4, 2, 6, 2, 0, 2]  # mismatches at k=1 (tv 0), k=3 (tv 0.011), k=5 (tv 0)
logits_all = torch.zeros(1, K + 1, V)
for k in range(K):
    logits_all[0, k, target[k]] = 5.0
logits_all[0, K, 1] = 5.0
dt = torch.tensor([draft])
strict = (dt == torch.tensor([target])).long()
ns = types.SimpleNamespace(target_logits=logits_all[:, :-1], target_probs=None, draft_tokens=dt, draft_probs=None,
                           draft_logits=None, confidence_logits=None, strict_mask=strict, start=0,
                           target_logits_all=logits_all)


def rule(theta, logger=None):
    r = IdeaBRule(theta=theta, logger=logger)
    r.bind_draft_model(fake)
    return r


assert rule(-1.0)(ns).tolist() == strict.tolist()          # θ < 0: never accept = original rule
assert rule(0.01)(ns).tolist() == [[1, 1, 1, 0, 1, 0, 1]]  # k=1 accepted, k=3 rejected -> stop there
assert rule(2.0)(ns).tolist() == [[1] * 7]                 # every mismatch accepted
fake.markov_head.W[5] = fake.markov_head.W[6]
assert rule(0.01)(ns).tolist() == [[1] * 7]                # k=1, k=3, k=5 each judged on its own and accepted
fake.markov_head.W[5] = torch.tensor([0., 0, 0, 0, 0, 0, 0, 9])
print("rule OK")

with tempfile.TemporaryDirectory() as d:
    lg = IdeaBLogger(d)
    lg.start_sample(7)
    rule(0.01, lg)(ns)
    lg.end_sample(True)
    lg.close()
    rows = [json.loads(l) for l in open(Path(d) / "mismatches.jsonl")]
    assert [(x["pos_k"], x["accepted"]) for x in rows] == [(1, True), (3, False)], rows
    assert rows[0]["tv_est"] < 1e-6 < 0.01 < rows[1]["tv_est"] and rows[0]["rule_name"] == "ideab"
print("logger OK  tv_est:", [round(x["tv_est"], 4) for x in rows])

try:
    rule(0.1)(types.SimpleNamespace(**{k: v for k, v in vars(ns).items() if k != "target_logits_all"}))
    raise SystemExit("should have failed")
except RuntimeError as e:
    print("missing-hook check OK")
