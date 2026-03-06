"""WSGM (Weighted Side Gating Module) wrapper for DINOv3 backbones.

Provides a unified adapter that wraps DINOv3 ViT or ConvNeXt backbones
with lightweight WSGM modules for forgery-specific feature learning.
Follows the same integration pattern as ``models/lora.py``.
"""

from typing import List

import torch
import torch.nn as nn

from dinov3.layers.wsgm import WSGM


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def _detect_arch(backbone: nn.Module) -> str:
    """Detect backbone architecture type from its attributes.

    Returns ``"vit"`` or ``"convnext"``.
    """
    if hasattr(backbone, "blocks"):
        return "vit"
    if hasattr(backbone, "stages"):
        return "convnext"
    raise ValueError(
        "Cannot detect backbone architecture: expected 'blocks' (ViT) "
        "or 'stages' (ConvNeXt) attribute"
    )


def _compute_layer_indices(depth: int, num_layers: int) -> List[int]:
    """Evenly distribute *num_layers* WSGM attachment points across *depth* layers.

    Example (depth=12, num_layers=6): [1, 3, 5, 7, 9, 11]
    """
    indices = []
    step = depth / num_layers
    for i in range(num_layers):
        idx = int(i * step + step / 2)
        indices.append(min(idx, depth - 1))
    return indices


# ---------------------------------------------------------------------------
# WSGMWrapper
# ---------------------------------------------------------------------------

class WSGMWrapper(nn.Module):
    """Unified WSGM wrapper for DINOv3 ViT and ConvNeXt backbones.

    Architecture:
        1. Frozen DINOv3 backbone (ViT or ConvNeXt)
        2. WSGM modules attached to selected intermediate layers/stages
        3. Aggregation of WSGM-adapted CLS tokens (average or concat)
        4. LayerNorm + classifier head

    Auto-scaling:
        - **ViT**: ``num_wsgm_layers = depth // 2``, uniform ``embed_dim``
        - **ConvNeXt**: all 4 stages, per-stage projection to ``final_dim``

    Args:
        backbone: Pre-initialised DINOv3 backbone.
        model_name: Identifier (e.g. ``"dinov3_vits16plus"``).
        num_classes: Output classes (default 2 for binary).
        reduction_factor: Bottleneck = ``embed_dim // reduction_factor``.
        dropout: Dropout probability in WSGM and classifier.
        aggregation: ``"average"`` or ``"concat"``.
        freeze_backbone: Freeze backbone weights (default True).
    """

    def __init__(
        self,
        backbone: nn.Module,
        model_name: str,
        num_classes: int = 2,
        reduction_factor: int = 4,
        dropout: float = 0.5,
        aggregation: str = "average",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.arch_type = _detect_arch(backbone)
        self.backbone = backbone
        self.model_name = model_name
        self.num_classes = num_classes
        self.aggregation = aggregation
        self.final_dim = backbone.embed_dim

        if aggregation not in ("average", "concat"):
            raise ValueError(
                f"aggregation must be 'average' or 'concat', got '{aggregation}'"
            )

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Architecture-specific initialisation
        if self.arch_type == "vit":
            self._init_vit_wsgm(reduction_factor, dropout)
        else:
            self._init_convnext_wsgm(reduction_factor, dropout)

        # Post-aggregation norm + classifier
        self.ln_post = nn.LayerNorm(self.final_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.final_dim, num_classes),
        )
        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.02)
        nn.init.constant_(self.classifier[1].bias, 0)

    # ------------------------------------------------------------------
    # ViT initialisation
    # ------------------------------------------------------------------

    def _init_vit_wsgm(self, reduction_factor: int, dropout: float):
        depth = self.backbone.n_blocks
        embed_dim = self.backbone.embed_dim

        self.num_wsgm_layers = max(depth // 2, 1)
        self.wsgm_layer_indices = _compute_layer_indices(depth, self.num_wsgm_layers)

        bottleneck = max(embed_dim // reduction_factor, 1)
        self.wsgm_modules = nn.ModuleList([
            WSGM(embed_dim, bottleneck, dropout_prob=dropout)
            for _ in range(self.num_wsgm_layers)
        ])

        # No per-layer projection needed — uniform embed_dim
        self.stage_projections = None

        if self.aggregation == "concat":
            concat_dim = embed_dim * self.num_wsgm_layers
            self.concat_proj = nn.Sequential(
                nn.Linear(concat_dim, self.final_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            nn.init.normal_(self.concat_proj[0].weight, mean=0.0, std=0.02)
            nn.init.constant_(self.concat_proj[0].bias, 0)

    # ------------------------------------------------------------------
    # ConvNeXt initialisation
    # ------------------------------------------------------------------

    def _init_convnext_wsgm(self, reduction_factor: int, dropout: float):
        embed_dims: List[int] = self.backbone.embed_dims  # e.g. [96,192,384,768]

        self.num_wsgm_layers = self.backbone.n_blocks  # 4
        self.wsgm_layer_indices = list(range(self.num_wsgm_layers))

        # Per-stage projection to final_dim
        self.stage_projections = nn.ModuleList()
        for dim in embed_dims:
            if dim != self.final_dim:
                proj = nn.Linear(dim, self.final_dim)
                nn.init.normal_(proj.weight, mean=0.0, std=0.02)
                nn.init.constant_(proj.bias, 0)
                self.stage_projections.append(proj)
            else:
                self.stage_projections.append(nn.Identity())

        # All WSGM modules operate at final_dim
        bottleneck = max(self.final_dim // reduction_factor, 1)
        self.wsgm_modules = nn.ModuleList([
            WSGM(self.final_dim, bottleneck, dropout_prob=dropout)
            for _ in range(self.num_wsgm_layers)
        ])

        if self.aggregation == "concat":
            concat_dim = self.final_dim * self.num_wsgm_layers
            self.concat_proj = nn.Sequential(
                nn.Linear(concat_dim, self.final_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            nn.init.normal_(self.concat_proj[0].weight, mean=0.0, std=0.02)
            nn.init.constant_(self.concat_proj[0].bias, 0)

    # ------------------------------------------------------------------
    # Forward — ViT
    # ------------------------------------------------------------------

    def _forward_vit(self, x: torch.Tensor) -> List[torch.Tensor]:
        last_layer_idx = self.backbone.n_blocks - 1
        has_last_layer = last_layer_idx in self.wsgm_layer_indices

        if has_last_layer:
            non_last = [i for i in self.wsgm_layer_indices if i != last_layer_idx]

            if non_last:
                outputs_no_norm = self.backbone.get_intermediate_layers(
                    x, n=non_last, reshape=False,
                    return_class_token=True, return_extra_tokens=False,
                    norm=False,
                )
            else:
                outputs_no_norm = []

            outputs_with_norm = self.backbone.get_intermediate_layers(
                x, n=[last_layer_idx], reshape=False,
                return_class_token=True, return_extra_tokens=False,
                norm=True,
            )

            outputs = []
            non_last_iter = iter(outputs_no_norm)
            for idx in self.wsgm_layer_indices:
                if idx == last_layer_idx:
                    outputs.append(outputs_with_norm[0])
                else:
                    outputs.append(next(non_last_iter))
        else:
            outputs = self.backbone.get_intermediate_layers(
                x, n=self.wsgm_layer_indices, reshape=False,
                return_class_token=True, return_extra_tokens=False,
                norm=False,
            )

        wsgm_outputs = []
        for i, (patches, cls_token) in enumerate(outputs):
            wsgm_out = self.wsgm_modules[i](cls_token)
            wsgm_outputs.append(cls_token + wsgm_out)

        return wsgm_outputs

    # ------------------------------------------------------------------
    # Forward — ConvNeXt
    # ------------------------------------------------------------------

    def _forward_convnext(self, x: torch.Tensor) -> List[torch.Tensor]:
        # ConvNeXt get_intermediate_layers does NOT accept return_extra_tokens
        outputs = self.backbone.get_intermediate_layers(
            x, n=self.wsgm_layer_indices,
            reshape=False, return_class_token=True, norm=False,
        )

        wsgm_outputs = []
        for i, (patches, cls_token) in enumerate(outputs):
            projected = self.stage_projections[i](cls_token)
            wsgm_out = self.wsgm_modules[i](projected)
            wsgm_outputs.append(projected + wsgm_out)

        return wsgm_outputs

    # ------------------------------------------------------------------
    # Unified forward
    # ------------------------------------------------------------------

    def _aggregate(self, wsgm_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate WSGM outputs and apply post-norm."""
        if self.aggregation == "average":
            feat = torch.stack(wsgm_outputs, dim=0).mean(dim=0)
        else:  # concat
            feat = torch.cat(wsgm_outputs, dim=-1)
            feat = self.concat_proj(feat)
        return self.ln_post(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch_type == "vit":
            wsgm_outputs = self._forward_vit(x)
        else:
            wsgm_outputs = self._forward_convnext(x)

        feat = self._aggregate(wsgm_outputs)
        return self.classifier(feat)

    def forward_with_embedding(self, x: torch.Tensor) -> tuple:
        """Forward pass returning both logits and pre-classifier embedding.

        Returns:
            ``(logits, embedding)`` where embedding has shape ``(B, final_dim)``.
        """
        if self.arch_type == "vit":
            wsgm_outputs = self._forward_vit(x)
        else:
            wsgm_outputs = self._forward_convnext(x)

        embedding = self._aggregate(wsgm_outputs)
        logits = self.classifier(embedding)
        return logits, embedding

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self.final_dim

    @property
    def head(self) -> nn.Linear:
        """Expose the classifier Linear for compatibility with GenAIClassifier."""
        return self.classifier[1]

    @head.setter
    def head(self, value: nn.Module):
        """Allow GenAIClassifier to replace the head if needed."""
        self.classifier[1] = value

    def print_config(self):
        backbone_frozen = not any(
            p.requires_grad for p in self.backbone.parameters()
        )
        print(f"\nWSGMWrapper Configuration:")
        print(f"  Backbone: {self.model_name} ({self.arch_type})")
        print(f"  Backbone frozen: {backbone_frozen}")
        print(f"  WSGM modules: {self.num_wsgm_layers}")
        print(f"  WSGM attached to layers: {self.wsgm_layer_indices}")
        print(f"  Aggregation: {self.aggregation}")
        print(f"  Final dim: {self.final_dim}")
        print(f"  Classes: {self.num_classes}")
        if self.stage_projections is not None:
            n_proj = sum(
                1 for p in self.stage_projections if not isinstance(p, nn.Identity)
            )
            print(f"  Stage projections: {n_proj} non-identity")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n  Trainable params: {trainable:,}")
        print(f"  Frozen params: {total - trainable:,}")
        print(f"  Trainable ratio: {100 * trainable / total:.2f}%")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_wsgm_params(model: nn.Module) -> tuple[int, int, int]:
    """Count parameters in a model containing a :class:`WSGMWrapper`.

    Returns:
        ``(total, trainable, wsgm_adapter_only)``

    ``wsgm_adapter_only`` includes WSGM modules, projections, ln_post,
    and the classifier — i.e. everything that is trainable when the
    backbone is frozen.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Find the WSGMWrapper inside the model
    wrapper = None
    if isinstance(model, WSGMWrapper):
        wrapper = model
    else:
        for m in model.modules():
            if isinstance(m, WSGMWrapper):
                wrapper = m
                break

    if wrapper is None:
        return total, trainable, 0

    wsgm_params = 0
    # WSGM modules
    wsgm_params += sum(p.numel() for p in wrapper.wsgm_modules.parameters())
    # Stage projections (ConvNeXt)
    if wrapper.stage_projections is not None:
        wsgm_params += sum(p.numel() for p in wrapper.stage_projections.parameters())
    # Concat projection (if concat mode)
    if hasattr(wrapper, "concat_proj"):
        wsgm_params += sum(p.numel() for p in wrapper.concat_proj.parameters())
    # Post LayerNorm
    wsgm_params += sum(p.numel() for p in wrapper.ln_post.parameters())
    # Classifier
    wsgm_params += sum(p.numel() for p in wrapper.classifier.parameters())

    return total, trainable, wsgm_params
