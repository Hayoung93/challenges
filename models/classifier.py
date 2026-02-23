"""GenAI image classifier built on MambaVision backbones."""

import torch
import torch.nn as nn
from mambavision import create_model
from timm.layers import trunc_normal_

VALID_MODELS = [
    "mamba_vision_T", "mamba_vision_T2", "mamba_vision_S",
    "mamba_vision_B", "mamba_vision_B_21k",
    "mamba_vision_L", "mamba_vision_L_21k",
    "mamba_vision_L2", "mamba_vision_L2_512_21k",
    "mamba_vision_L3_256_21k", "mamba_vision_L3_512_21k",
]


class GenAIClassifier(nn.Module):
    """Binary classifier for GenAI image detection.

    Wraps a MambaVision backbone with a configurable classification head.

    Args:
        model_name: MambaVision variant (e.g., ``"mamba_vision_T"``).
        pretrained: Load ImageNet-pretrained backbone weights.
        num_classes: Number of output classes (default 2: real/fake).
        freeze_backbone: If True, freeze all backbone parameters.
        drop_rate: Dropout rate passed to the MambaVision backbone.
        checkpoint_path: Path to a full model checkpoint to load
            (applied *after* head replacement).
    """

    def __init__(
        self,
        model_name: str = "mamba_vision_T",
        pretrained: bool = False,
        num_classes: int = 2,
        freeze_backbone: bool = False,
        drop_rate: float = 0.0,
        checkpoint_path: str = "",
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

        # Create backbone with original 1000-class head so pretrained
        # weights load without size mismatch.
        self.backbone = create_model(
            model_name,
            pretrained=pretrained,
            drop_rate=drop_rate,
        )

        # Replace the classification head for our target num_classes.
        num_features = self.backbone.head.in_features
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
    )
