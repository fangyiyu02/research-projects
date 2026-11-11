import torch
import torch.nn as nn
from torchao.quantization import (
    quantize_,
    Int8WeightOnlyConfig,
    Int4WeightOnlyConfig,
    IntxWeightOnlyConfig,
)
from torchao.quantization.granularity import PerGroup


def quantize_draft_linear(model, scheme="int8", layers=None,
                          targets=("q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj")):
    """
    对 drafter 的指定 Linear 层做 weight-only 量化。
    scheme: "int8" / "int4" / "int2"
    layers: None = 全部层; (2,) = 只量化第2层; (0, 3) = 量化第0和第3层
    """
    def _filter(module, name):
        if not (isinstance(module, nn.Linear) and name.split(".")[-1] in targets):
            return False
        if layers is None:
            return True
        parts = name.split(".")
        return "layers" in parts and int(parts[parts.index("layers") + 1]) in layers

    if scheme == "int8":
        cfg = Int8WeightOnlyConfig()
    elif scheme == "int4":
        cfg = Int4WeightOnlyConfig(group_size=128,
                                   int4_packing_format="tile_packed_to_4d")
    elif scheme == "int2":
        cfg = IntxWeightOnlyConfig(weight_dtype=torch.int2,
                                   granularity=PerGroup(32))
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    quantize_(model, cfg, filter_fn=_filter)
    return model