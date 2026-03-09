"""LoRA-MoE: Mixture of Experts via per-expert LoRA adapters in the backbone.

Each expert is a set of LoRA adapters injected into the backbone layers.
During training, active expert deltas are combined via expert masks and
applied in a single forward pass.  During inference, each expert runs
independently and outputs are aggregated via entropy-weighted combination.

Includes both Linear (LoRAMoELinear) and Conv2d (LoRAMoEConv2d) variants.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# Default target modules (same as lora.py)
# ──────────────────────────────────────────────────────────────────────

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

_DEFAULT_CONVLORA_MOE_TARGETS = {
    # ConvNeXt family: adapt depthwise convolutions
    "dinov3_convnext_tiny": ["dwconv"],
    "dinov3_convnext_small": ["dwconv"],
    "dinov3_convnext_base": ["dwconv"],
    "dinov3_convnext_large": ["dwconv"],
}

# ──────────────────────────────────────────────────────────────────────
# LoRAMoELinear
# ──────────────────────────────────────────────────────────────────────


class LoRAMoELinear(nn.Module):
    """K expert LoRA adapter sets wrapping an existing ``nn.Linear``.

    ``output = original(x) + combined_delta * scaling``

    Modes:
        - ``_expert_masks`` set: training — combined delta from active experts.
        - ``_active_expert`` set: inference — single expert delta.
        - Neither set: fallback — ``original(x)`` only.

    Args:
        original: The ``nn.Linear`` (or subclass) to wrap.
        num_experts: Number of expert adapter sets.
        rank: LoRA rank *r*.
        alpha: LoRA scaling numerator.  ``scaling = alpha / rank``.
        dropout: Dropout applied before the low-rank branch.
    """

    def __init__(
        self,
        original: nn.Linear,
        num_experts: int = 8,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.num_experts = num_experts
        in_features = original.in_features
        out_features = original.out_features

        self.lora_downs = nn.ModuleList(
            [nn.Linear(in_features, rank, bias=False) for _ in range(num_experts)]
        )
        self.lora_ups = nn.ModuleList(
            [nn.Linear(rank, out_features, bias=False) for _ in range(num_experts)]
        )
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        for k in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_downs[k].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_ups[k].weight)

        # Proxy attributes that external code may access on the original layer.
        self.in_features = original.in_features
        self.out_features = original.out_features

        # Runtime state (set externally via helpers).
        self._expert_masks: torch.Tensor | None = None  # (B, K)
        self._active_expert: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)

        if self._active_expert is not None:
            # Inference: single expert — efficient.
            k = self._active_expert
            delta = self.lora_ups[k](self.lora_downs[k](self.lora_dropout(x)))
            return base_out + delta * self.scaling

        if self._expert_masks is not None:
            masks = self._expert_masks  # (B, K)
            dropped_x = self.lora_dropout(x)

            # Compute per-expert deltas.  Even inactive experts run a
            # dummy forward (multiplied by 0) so their parameters remain
            # in the autograd graph — required for DDP with
            # find_unused_parameters=False.
            deltas = []
            for k in range(self.num_experts):
                delta = self.lora_ups[k](self.lora_downs[k](dropped_x))
                if not masks[:, k].any():
                    delta = delta * 0
                deltas.append(delta)

            stacked = torch.stack(deltas, dim=0)  # (K, B, ..., D_out)

            # Mask broadcasting: (B, K) → (K, B) → (K, B, 1, ..., 1)
            mask_t = masks.t()  # (K, B)
            for _ in range(x.dim() - 1):
                mask_t = mask_t.unsqueeze(-1)

            weighted = (stacked * mask_t).sum(dim=0)  # (B, ..., D_out)

            # Normalise by number of active experts per sample.
            num_active = masks.sum(dim=1).clamp(min=1.0)  # (B,)
            for _ in range(x.dim() - 1):
                num_active = num_active.unsqueeze(-1)

            return base_out + (weighted / num_active) * self.scaling

        # Fallback: no LoRA.
        return base_out


# ──────────────────────────────────────────────────────────────────────
# LoRAMoEConv2d
# ──────────────────────────────────────────────────────────────────────


class LoRAMoEConv2d(nn.Module):
    """K expert ConvLoRA adapter sets wrapping an existing ``nn.Conv2d``.

    Each expert stores low-rank factors ``lora_As[k]`` and ``lora_Bs[k]``
    whose product is reshaped to the original weight shape.

    Modes:
        - ``_expert_masks`` set: training — K conv outputs combined via masks.
        - ``_active_expert`` set: inference — single expert weight merge.
        - Neither set: fallback — ``original(x)`` only.

    Args:
        original: The ``nn.Conv2d`` to wrap.
        num_experts: Number of expert adapter sets.
        rank: LoRA rank *r*.
        alpha: LoRA scaling numerator.  ``scaling = alpha / rank``.
        dropout: Dropout probability (applied on the LoRA branch only).
    """

    def __init__(
        self,
        original: nn.Conv2d,
        num_experts: int = 8,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.num_experts = num_experts

        out_channels = original.out_channels
        in_channels_per_group = original.in_channels // original.groups
        kernel_size = original.kernel_size[0]  # assumes square kernel

        self.lora_As = nn.ParameterList([
            nn.Parameter(
                original.weight.new_zeros(
                    (rank * kernel_size, in_channels_per_group * kernel_size)
                )
            )
            for _ in range(num_experts)
        ])
        self.lora_Bs = nn.ParameterList([
            nn.Parameter(
                original.weight.new_zeros(
                    (out_channels * kernel_size, rank * kernel_size)
                )
            )
            for _ in range(num_experts)
        ])
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        for k in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_As[k], a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bs[k])

        # Proxy attributes.
        self.in_channels = original.in_channels
        self.out_channels = original.out_channels
        self.kernel_size = original.kernel_size
        self.groups = original.groups

        # Runtime state.
        self._expert_masks: torch.Tensor | None = None
        self._active_expert: int | None = None

    def _delta_weight(self, k: int) -> torch.Tensor:
        """Compute weight delta for expert *k*."""
        return (self.lora_Bs[k] @ self.lora_As[k]).view(
            self.original.weight.shape
        ) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._active_expert is not None:
            # Inference: merge weight delta → single efficient F.conv2d.
            delta_w = self._delta_weight(self._active_expert)
            return F.conv2d(
                x, self.original.weight + delta_w,
                self.original.bias, self.original.stride,
                self.original.padding, self.original.dilation,
                self.original.groups,
            )

        base_out = self.original(x)

        if self._expert_masks is not None:
            masks = self._expert_masks  # (B, K)
            B = x.shape[0]

            # Compute per-expert deltas.  Inactive experts still run a
            # dummy forward (* 0) to keep parameters in the autograd
            # graph for DDP with find_unused_parameters=False.
            delta_outputs = []
            for k in range(self.num_experts):
                delta_w = self._delta_weight(k)
                if self.training and self.lora_dropout is not None:
                    delta_out = F.conv2d(
                        self.lora_dropout(x), delta_w, None,
                        self.original.stride, self.original.padding,
                        self.original.dilation, self.original.groups,
                    )
                else:
                    delta_out = F.conv2d(
                        x, delta_w, None,
                        self.original.stride, self.original.padding,
                        self.original.dilation, self.original.groups,
                    )
                if not masks[:, k].any():
                    delta_out = delta_out * 0
                delta_outputs.append(delta_out)

            stacked = torch.stack(delta_outputs, dim=0)  # (K, B, C_out, H', W')
            # (K, B, 1, 1, 1)
            mask_t = masks.t().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weighted = (stacked * mask_t).sum(dim=0)  # (B, C_out, H', W')
            num_active = masks.sum(dim=1).clamp(min=1.0).view(B, 1, 1, 1)
            return base_out + weighted / num_active

        # Fallback: no LoRA.
        return base_out


# ──────────────────────────────────────────────────────────────────────
# Apply functions
# ──────────────────────────────────────────────────────────────────────


def apply_lora_moe_to_model(
    model: nn.Module,
    model_name: str,
    num_experts: int = 8,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: list | None = None,
) -> int:
    """Replace matching ``nn.Linear`` layers with :class:`LoRAMoELinear`.

    Args:
        model: The backbone module to modify **in-place**.
        model_name: Used to look up default target suffixes.
        num_experts: Number of expert LoRA adapter sets.
        rank, alpha, dropout: Forwarded to :class:`LoRAMoELinear`.
        target_modules: Override target module suffixes.

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
        setattr(
            parent, attr_name,
            LoRAMoELinear(module, num_experts, rank, alpha, dropout),
        )
        replaced += 1

    return replaced


def apply_convlora_moe_to_model(
    model: nn.Module,
    model_name: str,
    num_experts: int = 8,
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.0,
    target_modules: list | None = None,
) -> int:
    """Replace matching ``nn.Conv2d`` layers with :class:`LoRAMoEConv2d`.

    Args:
        model: The backbone module to modify **in-place**.
        model_name: Used to look up default target suffixes.
        num_experts: Number of expert ConvLoRA adapter sets.
        rank, alpha, dropout: Forwarded to :class:`LoRAMoEConv2d`.
        target_modules: Override target module suffixes.

    Returns:
        Number of modules replaced.
    """
    targets = target_modules or _DEFAULT_CONVLORA_MOE_TARGETS.get(model_name, [])
    if not targets:
        return 0

    modules_dict = dict(model.named_modules())
    replaced = 0
    for name, module in list(model.named_modules()):
        if not any(name.endswith(t) for t in targets):
            continue
        if not isinstance(module, nn.Conv2d):
            continue
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = modules_dict[parent_name]
        else:
            parent = model
            attr_name = name
        setattr(
            parent, attr_name,
            LoRAMoEConv2d(module, num_experts, rank, alpha, dropout),
        )
        replaced += 1

    return replaced


# ──────────────────────────────────────────────────────────────────────
# Mask propagation helpers
# ──────────────────────────────────────────────────────────────────────


def _get_backbone(model: nn.Module) -> nn.Module:
    """Unwrap DDP/DataParallel and GenAIClassifier to get the backbone."""
    raw = model.module if hasattr(model, "module") else model
    return raw.backbone if hasattr(raw, "backbone") else raw


def set_expert_masks(model: nn.Module, expert_masks: torch.Tensor) -> None:
    """Set ``_expert_masks`` on all LoRA-MoE modules in the backbone.

    Also clears ``_active_expert`` to prevent it from taking priority
    in the forward pass.

    Args:
        model: The model (may be DDP-wrapped).
        expert_masks: ``(B, K)`` float tensor.
    """
    backbone = _get_backbone(model)
    for m in backbone.modules():
        if isinstance(m, (LoRAMoELinear, LoRAMoEConv2d)):
            m._expert_masks = expert_masks
            m._active_expert = None


def set_active_expert(model: nn.Module, expert_idx: int) -> None:
    """Set a single active expert for inference.

    Clears ``_expert_masks`` and sets ``_active_expert`` on all
    LoRA-MoE modules.

    Args:
        model: The model (may be DDP-wrapped).
        expert_idx: Index of the expert to activate.
    """
    backbone = _get_backbone(model)
    for m in backbone.modules():
        if isinstance(m, (LoRAMoELinear, LoRAMoEConv2d)):
            m._expert_masks = None
            m._active_expert = expert_idx


def clear_lora_moe_state(model: nn.Module) -> None:
    """Clear all runtime state from LoRA-MoE modules.

    Resets both ``_expert_masks`` and ``_active_expert`` to ``None``.

    Args:
        model: The model (may be DDP-wrapped).
    """
    backbone = _get_backbone(model)
    for m in backbone.modules():
        if isinstance(m, (LoRAMoELinear, LoRAMoEConv2d)):
            m._expert_masks = None
            m._active_expert = None


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────


def _is_lora_moe_key(k: str) -> bool:
    """Return True if *k* is a LoRA-MoE parameter name."""
    return "lora_downs" in k or "lora_ups" in k or "lora_As" in k or "lora_Bs" in k


def get_lora_moe_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA-MoE parameters from *model*."""
    return {
        k: v
        for k, v in model.state_dict().items()
        if _is_lora_moe_key(k)
    }


def count_lora_moe_params(model: nn.Module) -> tuple[int, int, int]:
    """Count parameters in *model*.

    Returns:
        ``(total, trainable, lora_moe_only)``
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_moe = 0
    for name, param in model.named_parameters():
        if _is_lora_moe_key(name):
            lora_moe += param.numel()
    return total, trainable, lora_moe
