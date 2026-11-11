"""plot_drift.py — figures from run_drift npz files.
  python plot_drift.py runs/*.npz --outdir figs/
Per file: <name>_heat_simK.png (steps x layers), <name>_layers.png (drift vs
layer, K and V). Printed: SVD coupling index of the drift field (post-hoc
analysis only; nothing here affects measurement). Multi-file: overlay of
layer curves for cross-length comparison.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def coupling_index(D):
    D = np.nan_to_num(D, nan=np.nanmean(D))
    D = D - D.mean()
    s = np.linalg.svd(D, compute_uv=False)
    return float(1 - s[0] ** 2 / (s ** 2).sum()) if s.sum() > 0 else 0.0


def one(path, outdir):
    z = np.load(path)
    name = os.path.splitext(os.path.basename(path))[0]
    simK, simV = z["sim_K"], z["sim_V"]
    L = simK.shape[1]

    plt.figure(figsize=(7, 4.5))
    plt.imshow(simK, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(label="cos(K^t, K^{t-1})")
    plt.xlabel("layer"); plt.ylabel("decoding step"); plt.title(name)
    plt.tight_layout(); plt.savefig(f"{outdir}/{name}_heat_simK.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(range(L), np.nanmean(1 - simK, 0), "o-", label="K (1-cos)")
    plt.plot(range(L), np.nanmean(1 - simV, 0), "s-", label="V (1-cos)")
    plt.plot(range(L), np.nanmean(z["rel_K"], 0), "^--", alpha=.6, label="K rel-L2")
    plt.xlabel("layer"); plt.ylabel("mean drift"); plt.legend(); plt.title(name)
    plt.tight_layout(); plt.savefig(f"{outdir}/{name}_layers.png", dpi=150)
    plt.close()


    tK = z["total_K"]
    plt.figure(figsize=(6, 4))
    for l in range(0, L, max(1, L // 6)):
        plt.plot(range(tK.shape[0]), 1 - tK[:, l], "-", lw=1, label=f"layer {l}")
    plt.xlabel("decoding step"); plt.ylabel("cumulative drift vs step 0 (1-cos, K)")
    plt.legend(fontsize=7); plt.title(name)
    plt.tight_layout(); plt.savefig(f"{outdir}/{name}_total.png", dpi=150)
    plt.close()
    fin = 1 - tK[-1]
    print(f"{name}: final total K drift  mean={np.nanmean(fin):.4f}  "
          f"max={np.nanmax(fin):.4f} (layer {int(np.nanargmax(fin))})")
    print(f"{name}: coupling index = {coupling_index(1 - simK):.3f}")
    return name, np.nanmean(1 - simK, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+")
    p.add_argument("--outdir", default="workspace/runs/results/figs")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    curves = [one(f, a.outdir) for f in a.npz]
    if len(curves) > 1:
        plt.figure(figsize=(6, 4))
        for n, c in curves:
            plt.plot(range(len(c)), c, "o-", ms=3, label=n)
        plt.xlabel("layer"); plt.ylabel("mean K drift (1-cos)")
        plt.legend(fontsize=7); plt.title("drift vs layer across settings")
        plt.tight_layout(); plt.savefig(f"{a.outdir}/layers_compare.png", dpi=150)


if __name__ == "__main__":
    main()
