"""Shape smoke test: mock blocks, 5 raw forwards (no manual step calls — the
recorder auto-detects step boundaries), save npz, run plots.
Run: python test_smoke.py"""
import numpy as np, torch, torch.nn as nn
from drift_recorder import DriftRecorder

D, N = 64, 48

class FusedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_dims = (D, D, D)
        self.att_proj = nn.Linear(D, 3 * D)
    def forward(self, x): return self.att_proj(x)

class SepBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj, self.v_proj = nn.Linear(D, D), nn.Linear(D, D)
    def forward(self, x): return self.k_proj(x), self.v_proj(x)

def case(blocks, tag):
    L = len(blocks)
    rec = DriftRecorder(blocks, heatmap_token_stride=4)
    rec.attach()
    torch.manual_seed(0)
    base = torch.randn(1, N, D)
    for t in range(5):                       # plain forwards, nothing else
        x = base + 0.01 * t * torch.randn(1, N, D)
        for b in blocks:
            b(x)
    out = rec.save(f"smoke_{tag}.npz"); rec.detach()
    z = np.load(out)
    assert z["sim_K"].shape == (5, L), z["sim_K"].shape
    assert np.isnan(z["sim_K"][0]).all() and not np.isnan(z["sim_K"][1:]).any()
    assert z["total_K"].shape == (5, L)
    assert np.isnan(z["total_K"][0]).all() and not np.isnan(z["total_K"][1:]).any()
    assert np.nanmean(1 - z["total_K"][-1]) >= np.nanmean(1 - z["total_K"][1]) - 1e-4
    assert z["heat_K"].shape == (5, L, int(np.ceil(N / 4)))
    assert z["step_time"].shape == (5,)
    assert 0.9 < np.nanmean(z["sim_K"]) <= 1.0
    print(f"[{tag}] OK  sim_K mean={np.nanmean(z['sim_K']):.4f}  "
          f"steps auto-detected={z['sim_K'].shape[0]}")
    return out

if __name__ == "__main__":
    f1 = case([FusedBlock() for _ in range(3)], "fused")
    f2 = case([SepBlock() for _ in range(3)], "sep")
    import subprocess, os
    subprocess.run(["python", "plot_drift.py", f1, f2, "--outdir", "smoke_figs"], check=True)
    print("figures:", sorted(os.listdir("smoke_figs")))
