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
    # DINOv3
    "dinov3_vits16plus",
    "dinov3_convnext_tiny",
]

_DINOV3_WEIGHTS = {
    "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
}


def _create_dinov3_backbone(model_name: str, pretrained: bool, dinov3_weights_dir: str):
    """Create a DINOv3 backbone and optionally load pretrained weights."""
    from dinov3.hub.backbones import dinov3_convnext_tiny, dinov3_vits16plus

    factory_map = {
        "dinov3_vits16plus": dinov3_vits16plus,
        "dinov3_convnext_tiny": dinov3_convnext_tiny,
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
        checkpoint_path: Path to a full model checkpoint to load
            (applied *after* head replacement).
        dinov3_weights_dir: Directory containing DINOv3 pretrained weight files.
    """

    def __init__(
        self,
        model_name: str = "mamba_vision_T",
        pretrained: bool = False,
        num_classes: int = 2,
        freeze_backbone: bool = False,
        drop_rate: float = 0.0,
        checkpoint_path: str = "",
        dinov3_weights_dir: str = "",
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
            self.backbone = create_model(
                model_name,
                pretrained=pretrained,
                drop_rate=drop_rate,
            )
            num_features = self.backbone.head.in_features

        # Replace the classification head for our target num_classes.
        self.backbone.head = nn.Linear(num_features, num_classes)
        trunc_normal_(self.backbone.head.weight, std=0.02)
        nn.init.zeros_(self.backbone.head.bias)

        if freeze_backbone:
            self._freeze_backbone()

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _freeze_backbone(self):
        """Freeze all parameters except the classification head."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("head."):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        return self.backbone(x)


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
        checkpoint_path=getattr(args, "checkpoint_path", ""),
        dinov3_weights_dir=getattr(args, "dinov3_weights_dir", ""),
    )
