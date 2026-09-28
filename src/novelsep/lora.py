"""LoRA layers compatible with the experiment checkpoints."""
from __future__ import annotations
import math
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def should_wrap_with_lora(module_name: str, target_names: set[str]) -> bool:
    if not target_names:
        return True
    leaf_name = module_name.rsplit(".", 1)[-1]
    return leaf_name in target_names or module_name in target_names


def replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora_into_transformer(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_names: set[str],
) -> list[str]:
    replaced: list[str] = []
    for module_name, module in list(model.transformer.named_modules()):
        if module_name == "":
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not should_wrap_with_lora(module_name, target_names):
            continue
        replace_module(
            model.transformer,
            module_name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        replaced.append(module_name)
    if not replaced:
        raise RuntimeError("No transformer linear layers matched the requested LoRA targets.")
    return replaced


def freeze_all_but_lora(model: nn.Module) -> None:
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.weight.requires_grad = True
            module.lora_b.weight.requires_grad = True


