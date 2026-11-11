"""CPU unit tests for lossy_verify.context / rules / logger / metrics. No DeepSpec import needed."""
import importlib.util, json, math, sys, tempfile, types
from pathlib import Path
import torch

# load modules without triggering lossy_verify/__init__ (which needs DeepSpec on disk)
pkg = types.ModuleType("lossy_verify"); pkg.__path__ = [str(Path("lossy_verify").resolve())]; pkg.DEEPSPEC_DIR = "/nonexistent"
sys.modules["lossy_verify"] = pkg
def load(name):
    spec = importlib.util.spec_from_file_location(f"lossy_verify.{name}", f"lossy_verify/{name}.py")
    m = importlib.util.module_from_spec(spec); sys.modules[f"lossy_verify.{name}"] = m; spec.loader.exec_module(m); return m
context, rules, logger_mod, metrics = load("context"), load("rules"), load("logger"), load("metrics")

V, K = 11, 7
def make_ns(draft_tokens, target_top, regrets, conf, pD=None):
    """Build a hook namespace. regrets[k] = log p_T(top) - log p_T(draft) at position k (0 where match)."""
    target_logits = torch.full((1, K, V), -5.0)
    for k in range(K):
        target_logits[0, k, target_top[k]] = 5.0
        if draft_tokens[k] != target_top[k]:
            target_logits[0, k, draft_tokens[k]] = 5.0 - regrets[k]
    draft_logits = torch.full((1, K, V), -5.0)
    for k in range(K):
        draft_logits[0, k, draft_tokens[k]] = 5.0
        if pD is not None:  # put mass on another token so p_D(x_k) ≈ pD[k]
            other = (draft_tokens[k] + 1) % V
            draft_logits[0, k, other] = 5.0 + math.log((1 - pD[k]) / pD[k]) if pD[k] < 1 else -50
    dt = torch.tensor([draft_tokens])
    strict = (dt == torch.tensor([target_top])).to(torch.int64)
    conf_logits = torch.logit(torch.tensor([conf]).clamp(1e-6, 1-1e-6))
    return types.SimpleNamespace(target_logits=target_logits, target_probs=None, draft_tokens=dt,
                                 draft_probs=None, draft_logits=draft_logits, confidence_logits=conf_logits,
                                 strict_mask=strict, start=0)

# scenario: mismatches at k=2 (regret 1.0), k=4 (regret 3.0), k=6 (regret 0.5); matches elsewhere
draft  = [1, 2, 3, 4, 5, 6, 7]
target = [1, 2, 9, 4, 8, 6, 0]
reg    = [0, 0, 1.0, 0, 3.0, 0, 0.5]
conf   = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
ns = make_ns(draft, target, reg, conf)
ctx = context.VerifyContext.from_namespace(ns)

# --- context derived quantities
assert ctx.K == K
assert ctx.strict_mask.tolist() == [[1,1,0,1,0,1,0]]
r = ctx.regret()
assert all(abs(r[k].item() - reg[k]) < 1e-3 for k in range(K)), r
q = ctx.remaining().tolist()
assert q == ([7,6,5,4,3,2,1] if context.Q_INCLUDES_SELF else [6,5,4,3,2,1,0]), q
assert torch.allclose(ctx.confidence(), torch.tensor(conf), atol=1e-4)
print("context OK; q =", q)

# --- strict: never flips
assert rules.StrictRule()(ns).tolist() == [[1,1,0,1,0,1,0]]
# --- asd: r/q at k=2 is 1.0/5=0.2 <= 0.25 -> accept; k=4: 3/3=1.0 > 0.25 -> reject, stop
m = rules.ASDRule(g=0.25)(ns); assert m.tolist() == [[1,1,1,1,0,1,0]], m
# asd with big g: k=2 accept (spent 1), k=4 accept (spent 4), k=6 -> M=2 reached -> reject
m = rules.ASDRule(g=10.0)(ns); assert m.tolist() == [[1,1,1,1,1,1,0]], m
# asd big g, M=None, B=4.2 : k=2 (1.0) ok, k=4 (3.0, total 4.0) ok, k=6 (0.5 -> 4.5 > 4.2) reject
m = rules.ASDRule(g=10.0, M=None, B=4.2)(ns); assert m.tolist() == [[1,1,1,1,1,1,0]], m
# asd big g, M=None, B=None : all accepted
m = rules.ASDRule(g=10.0, M=None, B=None)(ns); assert m.tolist() == [[1]*7], m
# budget persists across rounds within a sample: first call accepts all three (spent 4.5); second call k=2 -> 5.5 ok, k=4 -> 8.5 > 8 reject
rule = rules.ASDRule(g=10.0, M=None, B=8.0); rule(ns); assert abs(rule.spent-4.5)<1e-3; m = rule(ns); assert m.tolist() == [[1,1,1,1,0,1,0]], m
rule.reset(); assert rule.spent == 0.0
# --- a1: theta 0.75 -> k=2 (0.8) accept, k=4 (0.6) reject
m = rules.A1ConfidenceRule(theta=0.75)(ns); assert m.tolist() == [[1,1,1,1,0,1,0]], m
m = rules.A1ConfidenceRule(theta=1.0)(ns); assert m.tolist() == [[1,1,0,1,0,1,0]], m   # theta=1 == strict
m = rules.A1ConfidenceRule(theta=0.0)(ns); assert m.tolist() == [[1,1,1,1,1,1,0]], m   # theta=0: only M=2 bites
m = rules.A3ConfOnlyRule(theta=0.0)(ns); assert m.tolist() == [[1]*7], m               # a3: no M/B -> all
# --- a2: theta 0.5 passes k=2,4,6 on conf; regret gate g=0.25 passes k=2 only
m = rules.A2ConfAndRegretRule(theta=0.5, g=0.25)(ns); assert m.tolist() == [[1,1,1,1,0,1,0]], m
# --- p: p_D(x_k) thresholds
ns_p = make_ns(draft, target, reg, conf, pD=[1,1,0.9,1,0.3,1,0.7])
pD = context.VerifyContext.from_namespace(ns_p).draft_prob_of_token()
assert abs(pD[2].item()-0.9) < 1e-3 and abs(pD[4].item()-0.3) < 1e-3, pD
m = rules.PDraftProbRule(theta=0.5)(ns_p); assert m.tolist() == [[1,1,1,1,0,1,0]], m
print("rules OK")

# --- logger
with tempfile.TemporaryDirectory() as d:
    lg = logger_mod.MismatchLogger(d)
    rule = rules.ASDRule(g=0.25, logger=lg)
    lg.start_sample(42); rule.reset(); rule(ns); rule(ns); lg.end_sample(True); lg.close()
    mm = [json.loads(l) for l in open(Path(d)/"mismatches.jsonl")]
    pp = [json.loads(l) for l in open(Path(d)/"positions.jsonl")]
    assert len(pp) == 2*K and {p["round_id"] for p in pp} == {1,2}
    assert [(x["round_id"], x["pos_k"], x["accepted"]) for x in mm] == [(1,2,True),(1,4,False),(2,2,True),(2,4,False)], mm
    assert all(x["sample_final_correct"] is True for x in mm)
    assert abs(mm[2]["cum_regret_after"] - 2.0) < 1e-3  # budget carried across rounds
    assert set(mm[0]) >= {"sample_id","round_id","pos_k","draft_token","target_argmax","r_i","q_i","c_k","p_D_xk","p_T_xk","p_T_top1","accepted","rule_name","cum_regret_after","n_mismatch_in_block_after","sample_final_correct"}
    assert lg.total_examined == 4 and lg.total_accepted == 2
print("logger OK")

# --- metrics
g = metrics.grade("so the answer is \\boxed{1,234}.", "1234"); assert g["correct_strict"] and not g["extract_fallback"], g
g = metrics.grade("The answer is 18 dollars", "18"); assert g["extract_fallback"] and g["correct_fallback"] and not g["correct_strict"], g
g = metrics.grade("\\boxed{\\text{18 dollars}}", "18"); assert g["correct_strict"], g
g = metrics.grade("\\boxed{\\frac{1}{2}} then \\boxed{17.0}", "17"); assert g["correct_strict"], g
g = metrics.grade("no numbers here", "3"); assert g["pred_fallback"] is None and not g["correct_fallback"], g
assert metrics.extract_boxed("\\boxed{a{b}c} x \\boxed{d{e}}") == "d{e}"
print("metrics OK")
