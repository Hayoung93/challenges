"""LoRA (Low-Rank Adaptation) for DINOv3 backbones.

Provides a manual LoRA implementation that preserves LinearKMaskedBias
forward behavior and checkpoint key stability.
"""

import math

import torch
import torch.nn as nn

_DEFAULT_TARGETS = {
    # ViT family: adapt attention projections
    "dinov3_vits16plus": ["attn.qkv", "attn.proj"],
    "dinov3_vitb16": ["attn.qkv", "attn.proj"],
    "dinov3_vitl16": ["attn.qkv", "attn.proj"],
    # ConvNeXt family: adapt pointwise convolutions
    "dinov3_convnext_tiny": ["pwconv1", "pwconv2"],
    "dinov3_convnext_small": ["pwconv1", "pwconv2"],
    "dinov3_convnext_base": ["pwconv1", "pwconv2"],
    "dinov3_convnext_large": ["pwconv1", "pwconv2"],
}


class LoRALinear(nn.Module):
    """LoRA adapter wrapping an existing ``nn.Linear`` (or subclass).

    ``output = original(x) + lora_up(lora_down(dropout(x))) * scaling``

    The *original* module is kept as a sub-module so that its forward
    (including ``LinearKMaskedBias.forward``) is preserved unchanged.
    ``lora_up`` is zero-initialised so initial output equals the original.

    Args:
        original: The ``nn.Linear`` (or subclass) to wrap.
        rank: LoRA rank *r*.
        alpha: LoRA scaling numerator.  ``scaling = alpha / rank``.
        dropout: Dropout applied before the low-rank branch.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        in_features = original.in_features
        out_features = original.out_features

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Proxy attributes that external code may access on the original layer.
        self.in_features = original.in_features
        self.out_features = original.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.original(x)
            + self.lora_up(self.lora_down(self.lora_dropout(x))) * self.scaling
        )


def apply_lora_to_model(
    model: nn.Module,
    model_name: str,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: list | None = None,
) -> int:
    """Replace matching ``nn.Linear`` layers with :class:`LoRALinear`.

    Args:
        model: The backbone module to modify **in-place**.
        model_name: Used to look up default target suffixes when
            *target_modules* is ``None``.
        rank, alpha, dropout: Forwarded to :class:`LoRALinear`.
        target_modules: Explicit list of module-name suffixes to target.
            If ``None``, architecture defaults from ``_DEFAULT_TARGETS``
            are used.

    Returns:
        Number of modules replaced.
    """
    targets = target_modules or _DEFAULT_TARGETS.get(model_name, [])
    if not targets:
        return 0

    modules_dict = dict(model.named_modules())
    replaced = 0
    for name, module in list(model.named_modules()):
        if not any(name.endswith(t) for t in targets):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = modules_dict[parent_name]
        else:
            parent = model
            attr_name = name
        setattr(parent, attr_name, LoRALinear(module, rank, alpha, dropout))
        replaced += 1

    return replaced


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA parameters from *model*."""
    return {
        k: v
        for k, v in model.state_dict().items()
        if "lora_down" in k or "lora_up" in k
    }


def count_lora_params(model: nn.Module) -> tuple[int, int, int]:
    """Count parameters in *model*.

    Returns:
        ``(total, trainable, lora_only)``
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = 0
    for name, param in model.named_parameters():
        if "lora_down" in name or "lora_up" in name:
            lora += param.numel()
    return total, trainable, lora


def merge_lora_weights(model: nn.Module) -> int:
    """Merge LoRA weights into the original layers for inference.

    Each :class:`LoRALinear` is replaced by its original ``nn.Linear``
    with weight updated as ``W += (B @ A) * scaling``.

    Args:
        model: The model to modify **in-place**.

    Returns:
        Number of modules merged.
    """
    merged = 0
    modules_dict = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        with torch.no_grad():
            delta = (module.lora_up.weight @ module.lora_down.weight) * module.scaling
            module.original.weight.add_(delta)
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = modules_dict[parent_name]
        else:
            parent = model
            attr_name = name
        setattr(parent, attr_name, module.original)
        merged += 1
    return merged
