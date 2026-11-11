"""
coupling_map.py — visualize the step x layer coupling structure of the drift
field D(t, l) = 1 - sim_K, per length directory.

  python coupling_map.py /workspace/runs/results/4k /workspace/runs/results/32k \
      --outdir /workspace/runs/results/summary

Per directory (samples averaged into one field):
  <name>_coupling_map.png   three panels:
      [1] D          the measured drift field (steps x layers)
      [2] rank-1     best separable approximation f(t) x g(l)
      [3] residual   D - rank-1  ->  THE coupling map: bright regions are
                     (step, layer) cells no product of two curves can explain
  <name>_svd_profiles.png   the f(t) and g(l) profiles themselves
  stdout                    coupling index of the averaged field
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_field(d):
    fs = sorted(glob.glob(os.path.join(d, "*.npz")))
    Ds = [1 - np.load(f)["sim_K"][1:] for f in fs]     # drop step-0 NaN row
    T = min(x.shape[0] for x in Ds)
    return np.nanmean([x[:T] for x in Ds], axis=0), len(fs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    for d in a.dirs:
        name = os.path.basename(d.rstrip("/"))
        D, n = load_field(d)
        mu = D.mean()
        U, S, Vt = np.linalg.svd(D - mu, full_matrices=False)
        rank1 = mu + S[0] * np.outer(U[:, 0], Vt[0])
        resid = D - rank1
        ci = 1 - S[0] ** 2 / (S ** 2).sum()

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, mat, title, cmap, sym in (
                (axes[0], D, f"D = 1-sim_K  ({name}, n={n})", "viridis", False),
                (axes[1], rank1, "rank-1: f(t) x g(l)", "viridis", False),
                (axes[2], resid, f"residual = coupling  (index {ci:.3f})",
                 "RdBu_r", True)):
            if sym:
                v = np.nanmax(np.abs(mat))
                im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap,
                               vmin=-v, vmax=v, interpolation="nearest")
            else:
                im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap,
                               interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("layer")
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes[0].set_ylabel("decoding step")
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, f"{name}_coupling_map.png"), dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        axes[0].plot(U[:, 0] * S[0], lw=1)
        axes[0].set_xlabel("decoding step"); axes[0].set_title("f(t): step profile")
        axes[1].plot(Vt[0], "o-", ms=3)
        axes[1].set_xlabel("layer"); axes[1].set_title("g(l): layer profile")
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, f"{name}_svd_profiles.png"), dpi=150)
        plt.close(fig)

        top3 = (S[:3] ** 2 / (S ** 2).sum()).round(3)
        print(f"{name}: coupling index={ci:.3f}  top-3 singular energy={top3}")


if __name__ == "__main__":
    main()
