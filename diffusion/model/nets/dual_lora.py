import math
import re
import torch
import torch.nn as nn


def _block_id_from_name(name: str):
    m = re.search(r"blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _lora_target_kind(module_name: str):
    if ("attn.qkv" in module_name) or ("attn.proj" in module_name):
        return "attn"
    return None


class DualLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, pixel_rank: int = 16, semantic_rank: int = 16, pixel_alpha: float = 16.0, semantic_alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.pixel_scaling = float(pixel_alpha) / max(1, int(pixel_rank))
        self.semantic_scaling = float(semantic_alpha) / max(1, int(semantic_rank))
        self.lambda_pix = 1.0
        self.lambda_sem = 1.0

        self.pixel_lora_A = nn.Linear(base.in_features, int(pixel_rank), bias=False, dtype=torch.float32)
        self.pixel_lora_B = nn.Linear(int(pixel_rank), base.out_features, bias=False, dtype=torch.float32)
        self.semantic_lora_A = nn.Linear(base.in_features, int(semantic_rank), bias=False, dtype=torch.float32)
        self.semantic_lora_B = nn.Linear(int(semantic_rank), base.out_features, bias=False, dtype=torch.float32)
        for m in (self.pixel_lora_A, self.pixel_lora_B, self.semantic_lora_A, self.semantic_lora_B):
            m.to(base.weight.device)
        nn.init.kaiming_uniform_(self.pixel_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.pixel_lora_B.weight)
        nn.init.kaiming_uniform_(self.semantic_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.semantic_lora_B.weight)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        x32 = x.float()
        delta_pix = self.pixel_lora_B(self.pixel_lora_A(x32)) * (self.pixel_scaling * float(self.lambda_pix))
        delta_sem = self.semantic_lora_B(self.semantic_lora_A(x32)) * (self.semantic_scaling * float(self.lambda_sem))
        return out + (delta_pix + delta_sem).to(out.dtype)


def apply_dual_lora(model, pixel_rank: int = 16, semantic_rank: int = 16, pixel_alpha: float = 16.0, semantic_alpha: float = 16.0):
    cnt = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        block_id = _block_id_from_name(name)
        if block_id is None or not (0 <= block_id <= 27):
            continue
        if _lora_target_kind(name) is None:
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[1]
        setattr(parent, child, DualLoRALinear(module, pixel_rank=pixel_rank, semantic_rank=semantic_rank, pixel_alpha=pixel_alpha, semantic_alpha=semantic_alpha))
        cnt += 1
    return cnt


def set_dual_lora_scales(model, lambda_pix: float = 1.0, lambda_sem: float = 1.0):
    for m in model.modules():
        if isinstance(m, DualLoRALinear):
            m.lambda_pix = float(lambda_pix)
            m.lambda_sem = float(lambda_sem)
