# 2+3) 尖刺周期性 & 去掉开场后的耦合指数（存成 check_32k.py 跑一次）
import numpy as np, glob
Ds = [1 - np.load(f)["sim_K"][1:] for f in glob.glob("/workspace/runs/results/32k/*.npz")]
T = min(d.shape[0] for d in Ds); D = np.nanmean([d[:T] for d in Ds], 0)
t = np.nanmean(D, 1)
sp = sorted(np.argsort(t)[::-1][:20])
print("top-20 尖刺步:", sp, "\n对 32 取模:", sorted(s % 32 for s in sp))  # 集中在 0/31 附近 = 块边界
def ci(X):
    X = X - X.mean(); s = np.linalg.svd(X, compute_uv=False)
    return 1 - s[0]**2 / (s**2).sum()
print("耦合指数 全程:", round(ci(D), 3), " 去掉前32步:", round(ci(D[32:]), 3))