"""GenAI image classifier built on MambaVision and DINOv3 backbones."""

import os

import torch
import torch.nn as nn
from mambavision import create_model
from timm.layers import trunc_normal_

VALID_MODELS = [
    # MambaVision
    "mamba_vision_T", "mamba_vision_T2", "mamba_vision_S",
    "mamba_vision_B", "mamba_vision_B_21k",
    "mamba_vision_L", "mamba_vision_L_21k",
    "mamba_vision_L2", "mamba_vision_L2_512_21k",
    "mamba_vision_L3_256_21k", "mamba_vision_L3_512_21k",
    # DINOv3 ViT
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    # DINOv3 ConvNeXt
    "dinov3_convnext_tiny",
    "dinov3_convnext_small",
    "dinov3_convnext_base",
    "dinov3_convnext_large",
]

_DINOV3_WEIGHTS = {
    "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "dinov3_convnext_small": "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "dinov3_convnext_base": "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "dinov3_convnext_large": "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
}

# Native training resolution for each MambaVision variant.
_MAMBA_NATIVE_RESOLUTION = {
    "mamba_vision_T": 224,
    "mamba_vision_T2": 224,
    "mamba_vision_S": 224,
    "mamba_vision_B": 224,
    "mamba_vision_B_21k": 224,
    "mamba_vision_L": 224,
    "mamba_vision_L_21k": 224,
    "mamba_vision_L2": 224,
    "mamba_vision_L2_512_21k": 512,
    "mamba_vision_L3_256_21k": 256,
    "mamba_vision_L3_512_21k": 512,
}


def compute_mambavision_window_size(model_name: str, image_size: int) -> list | None:
    """Return adjusted window_size for non-native resolutions, or None to keep defaults.

    MambaVision stages 2-3 use window-based attention on feature maps of size
    ``image_size // 16`` and ``image_size // 32`` respectively.  Setting
    ``window_size`` equal to the feature map size gives global attention,
    matching the original 224-resolution design.
    """
    native = _MAMBA_NATIVE_RESOLUTION.get(model_name)
    if native is None:  # DINOv3 or unknown model
        return None
    if image_size == native:
        return None
    if image_size % 32 != 0:
        raise ValueError(
            f"image_size must be divisible by 32, got {image_size}"
        )
    return [8, 8, image_size // 16, image_size // 32]


def update_mambavision_window_size(model: nn.Module, image_size: int) -> None:
    """Dynamically update window_size for MambaVision transformer stages.

    MambaVision stages 2-3 use window-based attention.  By default,
    window_size equals the feature map size (global attention).  This
    function updates ``window_size`` on the relevant layers to match a
    new input resolution, preserving the global-attention semantics.

    For non-MambaVision models this function is a no-op.

    Args:
        model: The model (or DDP-wrapped model).  Unwraps ``.module``
            and ``.backbone`` as needed to reach the MambaVision levels.
        image_size: Current input resolution.
    """
    # Unwrap DDP / DataParallel
    raw = model.module if hasattr(model, "module") else model
    # Unwrap GenAIClassifier
    backbone = raw.backbone if hasattr(raw, "backbone") else raw

    if not hasattr(backbone, "levels"):
        return

    # stages 0,1 are conv (no window attention) — skip
    # stages 2,3 are transformer — update window_size
    for stage_idx in (2, 3):
        if stage_idx < len(backbone.levels):
            level = backbone.levels[stage_idx]
            if hasattr(level, "window_size"):
                divisor = 16 * (2 ** (stage_idx - 2))  # 16 for stage 2, 32 for stage 3
                level.window_size = image_size // divisor


def _create_dinov3_backbone(model_name: str, pretrained: bool, dinov3_weights_dir: str):
    """Create a DINOv3 backbone and optionally load pretrained weights."""
    from dinov3.hub.backbones import (
        dinov3_convnext_base,
        dinov3_convnext_large,
        dinov3_convnext_small,
        dinov3_convnext_tiny,
        dinov3_vitb16,
        dinov3_vitl16,
        dinov3_vits16plus,
    )

    factory_map = {
        "dinov3_vits16plus": dinov3_vits16plus,
        "dinov3_vitb16": dinov3_vitb16,
        "dinov3_vitl16": dinov3_vitl16,
        "dinov3_convnext_tiny": dinov3_convnext_tiny,
        "dinov3_convnext_small": dinov3_convnext_small,
        "dinov3_convnext_base": dinov3_convnext_base,
        "dinov3_convnext_large": dinov3_convnext_large,
    }
    factory_fn = factory_map[model_name]

    if pretrained:
        weights_path = os.path.join(dinov3_weights_dir, _DINOV3_WEIGHTS[model_name])
        return factory_fn(pretrained=True, weights=weights_path)
    return factory_fn(pretrained=False)


class GenAIClassifier(nn.Module):
    """Binary classifier for GenAI image detection.

    Wraps a MambaVision or DINOv3 backbone with a configurable classification head.

    Args:
        model_name: Backbone variant (e.g., ``"mamba_vision_T"``,
            ``"dinov3_vits16plus"``, ``"dinov3_convnext_tiny"``).
        pretrained: Load pretrained backbone weights.
        num_classes: Number of output classes (default 2: real/fake).
        freeze_backbone: If True, freeze all backbone parameters.
        drop_rate: Dropout rate passed to the MambaVision backbone
            (ignored for DINOv3).
        image_size: Input image resolution.  When this differs from the
            model's native training resolution, ``window_size`` is
            automatically adjusted so that stages 2-3 use global
            attention (window = full feature map).
        checkpoint_path: Path to a full model checkpoint to load
            (applied *after* head replacement).
        dinov3_weights_dir: Directory containing DINOv3 pretrained weight files.
        lora_enabled: Attach LoRA adapters to the backbone (DINOv3 only).
        lora_rank: LoRA rank *r*.
        lora_alpha: LoRA scaling numerator.
        lora_dropout: Dropout on the LoRA branch.
        lora_target_modules: Override the default target module suffixes.
        convlora_enabled: Attach ConvLoRA adapters to depthwise Conv2d
            layers (DINOv3 ConvNeXt only).  Can be combined with
            ``lora_enabled``.
        convlora_rank: ConvLoRA rank *r*.
        convlora_alpha: ConvLoRA scaling numerator.
        convlora_dropout: Dropout on the ConvLoRA branch.
        wsgm: Attach WSGM adapters to the backbone (DINOv3 only).
            Mutually exclusive with ``lora_enabled`` and ``convlora_enabled``.
        wsgm_reduction_factor: WSGM bottleneck = embed_dim // factor.
        wsgm_dropout: Dropout probability in WSGM modules.
        wsgm_aggregation: ``"average"`` or ``"concat"``.
        moe_enabled: Replace the single classification head with a
            :class:`~models.moe.MoEHead` (8 expert heads).  Mutually
            exclusive with ``wsgm``.
    """

    def __init__(
        self,
        model_name: str = "mamba_vision_T",
        pretrained: bool = False,
        num_classes: int = 2,
        freeze_backbone: bool = False,
        drop_rate: float = 0.0,
        image_size: int = 224,
        checkpoint_path: str = "",
        dinov3_weights_dir: str = "",
        lora_enabled: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        lora_target_modules: list | None = None,
        convlora_enabled: bool = False,
        convlora_rank: int = 4,
        convlora_alpha: float = 4.0,
        convlora_dropout: float = 0.0,
        wsgm: bool = False,
        wsgm_reduction_factor: int = 4,
        wsgm_dropout: float = 0.5,
        wsgm_aggregation: str = "average",
        projection_dim: int = 0,
        moe_enabled: bool = False,
    ):
        super().__init__()
        if model_name not in VALID_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {VALID_MODELS}"
            )
        if num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2 for classification, got {num_classes}"
            )

        self.model_name = model_name
        self.num_classes = num_classes

        # Mutual exclusion
        if wsgm and lora_enabled:
            raise ValueError("--wsgm and --lora_enabled are mutually exclusive")
        if wsgm and convlora_enabled:
            raise ValueError("--wsgm and --convlora_enabled are mutually exclusive")
        if moe_enabled and wsgm:
            raise ValueError("--moe_enabled and --wsgm are mutually exclusive")

        if model_name in _DINOV3_WEIGHTS:
            # --- DINOv3 backbone ---
            self.backbone = _create_dinov3_backbone(
                model_name, pretrained=pretrained,
                dinov3_weights_dir=dinov3_weights_dir,
            )
            # DINOv3 models have head=nn.Identity() and embed_dim attribute.
            num_features = self.backbone.embed_dim
        else:
            # --- MambaVision backbone ---
            # Create with original 1000-class head so pretrained
            # weights load without size mismatch.
            create_kwargs = dict(pretrained=pretrained, drop_rate=drop_rate)
            ws = compute_mambavision_window_size(model_name, image_size)
            if ws is not None:
                create_kwargs["window_size"] = ws
                create_kwargs["resolution"] = image_size
            self.backbone = create_model(model_name, **create_kwargs)
            num_features = self.backbone.head.in_features

        # Store feature dimension for projection head construction below.
        self._num_features = num_features

        # WSGM wraps the backbone with its own classifier head;
        # otherwise replace the head normally.
        self.lora_enabled = lora_enabled
        self.convlora_enabled = convlora_enabled
        self.wsgm_enabled = wsgm
        self.moe_enabled = moe_enabled

        if wsgm:
            if model_name not in _DINOV3_WEIGHTS:
                raise ValueError("WSGM is only supported for DINOv3 models")
            from .wsgm import WSGMWrapper

            self.backbone = WSGMWrapper(
                backbone=self.backbone,
                model_name=model_name,
                num_classes=num_classes,
                reduction_factor=wsgm_reduction_factor,
                dropout=wsgm_dropout,
                aggregation=wsgm_aggregation,
                freeze_backbone=freeze_backbone,
            )
        else:
            # Replace the classification head for our target num_classes.
            if moe_enabled:
                from .moe import MoEHead

                self.backbone.head = nn.Identity()
                self.moe_head = MoEHead(num_features, num_classes)
            else:
                self.backbone.head = nn.Linear(num_features, num_classes)
                trunc_normal_(self.backbone.head.weight, std=0.02)
                nn.init.zeros_(self.backbone.head.bias)

            # LoRA — must be applied before freeze and checkpoint load.
            if lora_enabled:
                if model_name not in _DINOV3_WEIGHTS:
                    raise ValueError("LoRA is only supported for DINOv3 models")
                from .lora import apply_lora_to_model

                apply_lora_to_model(
                    self.backbone,
                    model_name,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    target_modules=lora_target_modules or None,
                )

            # ConvLoRA — adapt depthwise Conv2d layers (ConvNeXt only).
            if convlora_enabled:
                if model_name not in _DINOV3_WEIGHTS:
                    raise ValueError(
                        "ConvLoRA is only supported for DINOv3 models"
                    )
                if not model_name.startswith("dinov3_convnext"):
                    raise ValueError(
                        "ConvLoRA targets dwconv layers available only "
                        "in DINOv3 ConvNeXt models, not ViT"
                    )
                from .lora import apply_convlora_to_model

                apply_convlora_to_model(
                    self.backbone,
                    model_name,
                    rank=convlora_rank,
                    alpha=convlora_alpha,
                    dropout=convlora_dropout,
                )

            if freeze_backbone:
                self._freeze_backbone()

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # Projection head for contrastive learning (used only during training).
        # For WSGM, the embedding dim is the wrapper's final_dim.
        feat_dim = self.backbone.embed_dim if wsgm else self._num_features
        self.projection_dim = projection_dim
        if projection_dim > 0:
            self.projection_head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, projection_dim),
            )
        else:
            self.projection_head = None

    def _freeze_backbone(self):
        """Freeze all parameters except the head, LoRA, and ConvLoRA."""
        for name, param in self.backbone.named_parameters():
            if name.startswith("head."):
                continue
            if self.lora_enabled and ("lora_down" in name or "lora_up" in name):
                continue
            if self.convlora_enabled and ("lora_A" in name or "lora_B" in name):
                continue
            param.requires_grad = False

    def _load_checkpoint(self, path: str):
        """Load a full model checkpoint (backbone + head)."""
        import logging
        logger = logging.getLogger(__name__)

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict):
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        # Strip "module." prefix from DDP checkpoints
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {
                k.removeprefix("module."): v for k, v in state_dict.items()
            }
        # Strip "backbone." prefix if saved from GenAIClassifier wrapper
        if any(k.startswith("backbone.") for k in state_dict.keys()):
            state_dict = {
                k.removeprefix("backbone."): v for k, v in state_dict.items()
            }

        # WSGM checkpoint remapping: when loading a non-WSGM checkpoint
        # into a WSGM model, backbone keys need a "backbone." prefix
        # because WSGMWrapper stores the backbone as self.backbone.
        if self.wsgm_enabled:
            has_wsgm_keys = any("wsgm_modules" in k for k in state_dict)
            if not has_wsgm_keys:
                state_dict = {
                    f"backbone.{k}": v for k, v in state_dict.items()
                }

        # Remap non-LoRA checkpoint keys to LoRA / ConvLoRA model structure.
        # E.g. "blocks.0.attn.qkv.weight" → "blocks.0.attn.qkv.original.weight"
        if self.lora_enabled or self.convlora_enabled:
            from .lora import LoRAConv2d, LoRALinear

            lora_names = {
                n for n, m in self.backbone.named_modules()
                if isinstance(m, (LoRALinear, LoRAConv2d))
            }
            remapped = {}
            for k, v in state_dict.items():
                matched = False
                for ln in lora_names:
                    if k.startswith(ln + ".") and ".original." not in k:
                        suffix = k[len(ln) + 1:]
                        if suffix.startswith("lora_"):
                            remapped[k] = v  # keep LoRA keys as-is
                        else:
                            remapped[f"{ln}.original.{suffix}"] = v
                        matched = True
                        break
                if not matched:
                    remapped[k] = v
            state_dict = remapped

        # Filter out keys with shape mismatches (e.g., head with different num_classes)
        model_state = self.backbone.state_dict()
        filtered_state = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape != v.shape:
                skipped_keys.append(
                    f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(model_state[k].shape)}"
                )
            else:
                filtered_state[k] = v

        if skipped_keys:
            logger.warning(
                "Skipped checkpoint keys with shape mismatch:\n  %s",
                "\n  ".join(skipped_keys),
            )

        result = self.backbone.load_state_dict(filtered_state, strict=False)

        # Load MoE head parameters separately (they live outside backbone).
        if self.moe_enabled:
            moe_state = {
                k.removeprefix("moe_head."): v
                for k, v in filtered_state.items()
                if k.startswith("moe_head.")
            }
            if moe_state:
                self.moe_head.load_state_dict(moe_state, strict=False)
                result = result._replace(
                    unexpected_keys=[
                        k for k in result.unexpected_keys
                        if not k.startswith("moe_head.")
                    ]
                )

        if result.missing_keys:
            logger.warning(
                "Checkpoint missing keys (kept at init values):\n  %s",
                "\n  ".join(result.missing_keys),
            )
        if result.unexpected_keys:
            logger.warning(
                "Checkpoint unexpected keys (ignored):\n  %s",
                "\n  ".join(result.unexpected_keys),
            )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without the classification head."""
        if self.model_name in _DINOV3_WEIGHTS:
            ret = self.backbone.forward_features(x)
            return ret["x_norm_clstoken"]
        else:  # MambaVision
            return self.backbone.forward_features(x)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        moe_expert_masks: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.
            return_embedding: If True, also return the embedding and
                optional projection.
            moe_expert_masks: ``(B, K)`` binary mask indicating which
                experts are active per sample (training only).  When
                ``None`` and MoE is enabled, uses entropy-weighted
                aggregation (inference mode).

        Returns:
            Logits ``(B, num_classes)`` when ``return_embedding=False``
            and MoE is disabled or in inference mode.
            Logits ``(B, K, num_classes)`` when MoE training with masks.
            ``(logits, embedding, projection)`` when ``return_embedding=True``,
            where *projection* is ``None`` if no projection head is configured.
        """
        if self.moe_enabled:
            features = self._extract_features(x)
            if moe_expert_masks is not None:
                logits = self.moe_head.forward_routed(features, moe_expert_masks)
            else:
                logits = self.moe_head.inference_aggregate(features)

            if return_embedding:
                projection = None
                if self.projection_head is not None:
                    projection = self.projection_head(features)
                return logits, features, projection
            return logits

        if not return_embedding:
            return self.backbone(x)

        # Extract embedding and logits separately per backbone type.
        if self.wsgm_enabled:
            logits, embedding = self.backbone.forward_with_embedding(x)
        elif self.model_name in _DINOV3_WEIGHTS:
            ret = self.backbone.forward_features(x)
            embedding = ret["x_norm_clstoken"]
            logits = self.backbone.head(embedding)
        else:  # MambaVision
            embedding = self.backbone.forward_features(x)
            logits = self.backbone.head(embedding)

        projection = None
        if self.projection_head is not None:
            projection = self.projection_head(embedding)

        return logits, embedding, projection


def build_model(args) -> GenAIClassifier:
    """Build a GenAIClassifier from an argparse namespace.

    Args:
        args: Namespace with model config attributes (see ``config.py``).

    Returns:
        Configured ``GenAIClassifier`` instance.
    """
    return GenAIClassifier(
        model_name=getattr(args, "model_name", "mamba_vision_T"),
        pretrained=getattr(args, "pretrained", False),
        num_classes=getattr(args, "num_classes", 2),
        freeze_backbone=getattr(args, "freeze_backbone", False),
        drop_rate=getattr(args, "drop_rate", 0.0),
        image_size=getattr(args, "image_size", 224),
        checkpoint_path=getattr(args, "checkpoint_path", ""),
        dinov3_weights_dir=getattr(args, "dinov3_weights_dir", ""),
        lora_enabled=getattr(args, "lora_enabled", False),
        lora_rank=getattr(args, "lora_rank", 8),
        lora_alpha=getattr(args, "lora_alpha", 8.0),
        lora_dropout=getattr(args, "lora_dropout", 0.0),
        lora_target_modules=getattr(args, "lora_target_modules", None) or None,
        convlora_enabled=getattr(args, "convlora_enabled", False),
        convlora_rank=getattr(args, "convlora_rank", 4),
        convlora_alpha=getattr(args, "convlora_alpha", 4.0),
        convlora_dropout=getattr(args, "convlora_dropout", 0.0),
        wsgm=getattr(args, "wsgm", False),
        wsgm_reduction_factor=getattr(args, "wsgm_reduction_factor", 4),
        wsgm_dropout=getattr(args, "wsgm_dropout", 0.5),
        wsgm_aggregation=getattr(args, "wsgm_aggregation", "average"),
        projection_dim=getattr(args, "projection_dim", 0),
        moe_enabled=getattr(args, "moe_enabled", False),
    )
