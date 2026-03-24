"""WSGM (Weighted Side Gating Module) wrappers for DINOv3 backbones.

Provides two WSGM integration modes:

- **Post-extraction** (``WSGMWrapper``): Applies WSGM to CLS tokens
  extracted from selected intermediate layers via ``get_intermediate_layers``.
- **Inline injection** (``InlineWSGMWrapper``): Injects WSGM residually
  into *every* transformer block, modifying all tokens (CLS + patches)
  so that adapted features cascade through subsequent blocks.  Based on
  the DFD-NDC / ForgeLens Stage 1 architecture.

Both follow the same integration pattern as ``models/lora.py``.
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
# AttentionPooling
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Cross-attention pooling with a learnable query token.

    A single learnable query attends to all patch tokens via multi-head
    attention, producing one pooled vector.  Used by
    :class:`InlineWSGMWrapper` when ``pooling_type="attn"``.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, attn_drop: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Pool patch tokens into a single vector.

        Args:
            patch_tokens: ``(B, N, D)`` float32 patch token features.

        Returns:
            Pooled vector of shape ``(B, D)``.
        """
        B = patch_tokens.size(0)
        query = self.query.expand(B, -1, -1)  # (B, 1, D)
        pooled, _ = self.attn(
            query=query, key=patch_tokens, value=patch_tokens,
            need_weights=False,
        )
        return self.norm(pooled.squeeze(1))  # (B, D)


# ---------------------------------------------------------------------------
# InlineWSGMWrapper
# ---------------------------------------------------------------------------

class InlineWSGMWrapper(nn.Module):
    """DFD-NDC / ForgeLens-style inline WSGM injection for DINOv3 ViT.

    Unlike :class:`WSGMWrapper` which applies WSGM only to extracted CLS
    tokens, this wrapper manually iterates through every transformer block
    and applies WSGM residually to **all** tokens (CLS + patches + storage).
    Modified features cascade into subsequent blocks, enabling richer
    forgery-specific adaptation.

    Classification uses CLS token concatenated with pooled patch tokens
    (GAP or Attention Pooling), producing a ``2 * embed_dim`` feature
    before the head.

    Args:
        backbone: Pre-initialised DINOv3 ViT backbone.
        model_name: Identifier (e.g. ``"dinov3_vitl16"``).
        num_classes: Output classes (default 2).
        reduction_factor: Bottleneck = ``embed_dim // reduction_factor``.
        dropout: Dropout in WSGM modules and classifier.
        num_wsgm: Number of WSGM modules.  ``0`` = auto (``n_blocks // 2``).
        pooling_type: ``"gap"`` (mean pooling) or ``"attn"`` (attention pooling).
        attn_heads: Number of heads for ``AttentionPooling``.
        attn_drop: Dropout for ``AttentionPooling``.
        use_bfloat16: Permanently cast frozen backbone to bfloat16.
        freeze_backbone: Freeze backbone weights (default True).
    """

    def __init__(
        self,
        backbone: nn.Module,
        model_name: str,
        num_classes: int = 2,
        reduction_factor: int = 4,
        dropout: float = 0.5,
        num_wsgm: int = 0,
        pooling_type: str = "gap",
        attn_heads: int = 8,
        attn_drop: float = 0.1,
        use_bfloat16: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Validate: inline mode requires ViT (needs blocks, prepare_tokens_with_masks)
        if not hasattr(backbone, "blocks"):
            raise ValueError(
                "InlineWSGMWrapper requires a ViT backbone with 'blocks' attribute. "
                "ConvNeXt models are not supported — use WSGMWrapper instead."
            )

        self.backbone = backbone
        self.model_name = model_name
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.use_bfloat16 = use_bfloat16

        self.num_blocks = backbone.n_blocks
        self._backbone_embed_dim = backbone.embed_dim
        self.n_storage_tokens = getattr(backbone, "n_storage_tokens", 0)

        # Auto-determine num_wsgm
        self.num_wsgm = num_wsgm if num_wsgm > 0 else max(self.num_blocks // 2, 1)

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optional bfloat16 cast for frozen backbone
        if use_bfloat16:
            self.backbone = self.backbone.to(torch.bfloat16)

        # Trainable WSGM modules (float32)
        bottleneck = max(self._backbone_embed_dim // reduction_factor, 1)
        self.wsgm_modules = nn.ModuleList([
            WSGM(self._backbone_embed_dim, bottleneck, dropout_prob=dropout)
            for _ in range(self.num_wsgm)
        ])

        # Attention pooling (optional)
        if pooling_type == "attn":
            self.attn_pool = AttentionPooling(
                self._backbone_embed_dim, num_heads=attn_heads, attn_drop=attn_drop,
            )

        # Head: CLS + pooled → 2 * embed_dim
        head_dim = self._backbone_embed_dim * 2
        self.final_dim = head_dim
        self.ln_post = nn.LayerNorm(head_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )
        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.02)
        nn.init.constant_(self.classifier[1].bias, 0)

    # ------------------------------------------------------------------
    # ForgeLens block-to-WSGM mapping
    # ------------------------------------------------------------------

    def _get_wsgm_idx(self, block_idx: int) -> int:
        """Map transformer block index to WSGM module index."""
        return (block_idx * self.num_wsgm) // self.num_blocks

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run backbone blocks with inline WSGM injection.

        When ``use_bfloat16=True``, precision is managed manually
        (backbone bf16, WSGM fp32).  ``torch.amp.autocast`` is disabled
        inside this method to prevent conflicts — the same strategy used
        by the DFD-NDC reference implementation.

        Returns:
            ``(cls_token, pooled)`` both as float32, each ``(B, embed_dim)``.
        """
        # Disable autocast: we manage bf16/fp32 precision manually.
        # autocast + permanent bfloat16 backbone causes NaN gradients
        # (same issue as DFD_NDC with nn.DataParallel).
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if self.use_bfloat16:
                x = x.to(torch.bfloat16)

            x, (H, W) = self.backbone.prepare_tokens_with_masks(x)
            rope = self.backbone.rope_embed(H=H, W=W)

            for i, blk in enumerate(self.backbone.blocks):
                x = blk(x, rope)
                wsgm_idx = self._get_wsgm_idx(i)
                wsgm_out = self.wsgm_modules[wsgm_idx](x.float())
                if self.use_bfloat16:
                    x = x + wsgm_out.to(torch.bfloat16)
                else:
                    x = x + wsgm_out

            # Final norm — handle untie_cls_and_patch_norms
            if getattr(self.backbone, "untie_cls_and_patch_norms", False):
                x_cls_reg = self.backbone.cls_norm(
                    x[:, : self.n_storage_tokens + 1]
                )
                x_patch = self.backbone.norm(
                    x[:, self.n_storage_tokens + 1:]
                )
            else:
                x_norm = self.backbone.norm(x)
                x_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_patch = x_norm[:, self.n_storage_tokens + 1:]

            cls_token = x_cls_reg[:, 0].float()       # (B, D)
            patch_tokens = x_patch.float()             # (B, N_patches, D)

            # Pool patch tokens
            if self.pooling_type == "attn":
                pooled = self.attn_pool(patch_tokens)  # (B, D)
            else:
                pooled = patch_tokens.mean(dim=1)      # (B, D)

        return cls_token, pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token, pooled = self._forward_features(x)
        features = torch.cat([cls_token, pooled], dim=1)  # (B, 2*D)
        features = self.ln_post(features)
        return self.classifier(features)

    def forward_with_embedding(self, x: torch.Tensor) -> tuple:
        """Forward pass returning both logits and pre-classifier embedding.

        Returns:
            ``(logits, embedding)`` where embedding has shape ``(B, 2*embed_dim)``.
        """
        cls_token, pooled = self._forward_features(x)
        embedding = torch.cat([cls_token, pooled], dim=1)
        embedding = self.ln_post(embedding)
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
        self.classifier[1] = value

    def print_config(self):
        backbone_frozen = not any(
            p.requires_grad for p in self.backbone.parameters()
        )
        print(f"\nInlineWSGMWrapper Configuration:")
        print(f"  Backbone: {self.model_name}")
        print(f"  Backbone frozen: {backbone_frozen}")
        print(f"  Backbone dtype: {'bfloat16' if self.use_bfloat16 else 'float32'}")
        print(f"  WSGM modules: {self.num_wsgm} (across {self.num_blocks} blocks)")
        print(f"  Pooling: {self.pooling_type}")
        print(f"  Feature dim: {self.final_dim} (CLS + pooled)")
        print(f"  Classes: {self.num_classes}")

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

    # Find the WSGMWrapper or InlineWSGMWrapper inside the model
    wrapper = None
    if isinstance(model, (WSGMWrapper, InlineWSGMWrapper)):
        wrapper = model
    else:
        for m in model.modules():
            if isinstance(m, (WSGMWrapper, InlineWSGMWrapper)):
                wrapper = m
                break

    if wrapper is None:
        return total, trainable, 0

    wsgm_params = 0
    # WSGM modules
    wsgm_params += sum(p.numel() for p in wrapper.wsgm_modules.parameters())
    # Stage projections (ConvNeXt, WSGMWrapper only)
    if getattr(wrapper, "stage_projections", None) is not None:
        wsgm_params += sum(p.numel() for p in wrapper.stage_projections.parameters())
    # Concat projection (if concat mode)
    if hasattr(wrapper, "concat_proj"):
        wsgm_params += sum(p.numel() for p in wrapper.concat_proj.parameters())
    # Attention pooling (InlineWSGMWrapper)
    if hasattr(wrapper, "attn_pool"):
        wsgm_params += sum(p.numel() for p in wrapper.attn_pool.parameters())
    # Post LayerNorm
    wsgm_params += sum(p.numel() for p in wrapper.ln_post.parameters())
    # Classifier
    wsgm_params += sum(p.numel() for p in wrapper.classifier.parameters())

    return total, trainable, wsgm_params
