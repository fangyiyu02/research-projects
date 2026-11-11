# check_env.py
import sys, torch
from importlib.metadata import PackageNotFoundError, version

def check_env(model=None, target=None):
    print("\n===== DFlash environment check =====")
    print("Python:", sys.executable)
    print("PyTorch CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Compute capability:", torch.cuda.get_device_capability(0))
    for pkg in ("torch", "transformers", "torchao", "triton"):
        try:
            print(f"{pkg}: {version(pkg)}")
        except PackageNotFoundError:
            print(f"{pkg}: not installed")
    if model is not None:
        print("Draft class:", type(model).__name__)
        print("Draft layers:", len(model.layers))
        print("Draft weight dtype:", model.layers[0].self_attn.q_proj.weight.dtype)
    if target is not None:
        print("Target embedding dtype:", target.get_input_embeddings().weight.dtype)
    print("====================================\n")

if __name__ == "__main__":
    check_env()   