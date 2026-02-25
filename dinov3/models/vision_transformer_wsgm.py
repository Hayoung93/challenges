# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
DINOv3 Vision Transformer with WSGM modules for DeepFake Detection

This module provides a DINOv3 backbone enhanced with Weighted Side Gating Modules (WSGM)
for forgery-specific feature learning, without requiring HuggingFace transformers dependency.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .vision_transformer import DinoVisionTransformer
from dinov3.layers import WSGM


class DinoVisionTransformerWithWSGM(nn.Module):
    """
    DINOv3 Vision Transformer with WSGM modules (No transformers dependency)

    This model extends the DINOv3 backbone with lightweight WSGM adapter modules
    attached to specific transformer layers, designed for deepfake detection tasks.

    Architecture:
        1. DINOv3 ViT backbone (typically frozen)
        2. WSGM modules on selected transformer layers
        3. Aggregation of WSGM outputs (average or concat)
        4. LayerNorm + Classifier head

    Args:
        backbone_model (DinoVisionTransformer): Pre-initialized DINOv3 backbone
        num_classes (int): Number of output classes (default: 2 for binary classification)
        num_wsgm_layers (int): Number of WSGM modules to attach (default: 12)
        wsgm_layer_indices (Optional[List[int]]): Specific layer indices for WSGM attachment.
            If None, layers are evenly distributed. For ViT-L (24 layers), default is
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        wsgm_reduction_factor (int): Bottleneck reduction factor for WSGM (default: 4)
            e.g., 1024 -> 256 for ViT-L
        dropout_rate (float): Dropout probability (default: 0.5)
        aggregation_mode (str): How to aggregate WSGM outputs (default: 'average')
            - 'average': Simple average of all WSGM outputs (lightweight)
            - 'concat': Concatenate all outputs + learnable projection
        freeze_backbone (bool): Whether to freeze the backbone weights (default: True)

    Example:
        >>> from dinov3.hub.backbones import dinov3_vitl16
        >>> from dinov3.models import DinoVisionTransformerWithWSGM
        >>>
        >>> # Create backbone
        >>> backbone = dinov3_vitl16(pretrained=True)
        >>>
        >>> # Create WSGM model
        >>> model = DinoVisionTransformerWithWSGM(
        ...     backbone_model=backbone,
        ...     num_classes=2,
        ...     num_wsgm_layers=12,
        ...     wsgm_layer_indices=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        ...     aggregation_mode='average',
        ...     freeze_backbone=True
        ... )
        >>>
        >>> # Load converted weights
        >>> checkpoint = torch.load('converted_wsgm_model.pth')
        >>> model.load_state_dict(checkpoint)
        >>>
        >>> # Inference
        >>> x = torch.randn(2, 3, 256, 256)
        >>> model.eval()
        >>> with torch.no_grad():
        ...     logits = model(x)
    """

    def __init__(
        self,
        backbone_model: DinoVisionTransformer,
        num_classes: int = 2,
        num_wsgm_layers: int = 12,
        wsgm_layer_indices: Optional[List[int]] = None,
        wsgm_reduction_factor: int = 4,
        dropout_rate: float = 0.5,
        aggregation_mode: str = "average",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = backbone_model
        self.num_classes = num_classes
        self.num_wsgm_layers = num_wsgm_layers
        self.aggregation_mode = aggregation_mode
        self.dim = backbone_model.embed_dim

        # Validate aggregation mode
        assert aggregation_mode in ["average", "concat"], (
            f"aggregation_mode must be 'average' or 'concat', got '{aggregation_mode}'"
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Determine WSGM layer indices
        if wsgm_layer_indices is None:
            self.wsgm_layer_indices = self._get_wsgm_layer_indices()
        else:
            assert len(wsgm_layer_indices) == num_wsgm_layers, (
                f"wsgm_layer_indices must have {num_wsgm_layers} elements, "
                f"got {len(wsgm_layer_indices)}"
            )
            self.wsgm_layer_indices = wsgm_layer_indices

        # Create WSGM modules
        bottleneck_dim = self.dim // wsgm_reduction_factor
        self.wsgm_modules = nn.ModuleList(
            [WSGM(self.dim, bottleneck_dim, dropout_prob=dropout_rate) for _ in range(num_wsgm_layers)]
        )

        # Concat mode: projection layer to reduce concatenated features back to dim
        if self.aggregation_mode == "concat":
            concat_dim = self.dim * num_wsgm_layers
            self.concat_proj = nn.Sequential(
                nn.Linear(concat_dim, self.dim), nn.GELU(), nn.Dropout(dropout_rate)
            )
            # Initialize projection weights
            nn.init.normal_(self.concat_proj[0].weight, mean=0.0, std=0.02)
            nn.init.constant_(self.concat_proj[0].bias, 0)

        # Layer norm before classifier
        self.ln_post = nn.LayerNorm(self.dim)

        # Classifier head
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(self.dim, num_classes))

        # Initialize classifier
        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.02)
        nn.init.constant_(self.classifier[1].bias, 0)

    def _get_wsgm_layer_indices(self) -> List[int]:
        """
        Calculate evenly distributed layer indices for WSGM attachment

        For ViT-L with 24 layers and 12 WSGM modules:
            step = 24 / 12 = 2.0
            indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

        Returns:
            List[int]: Layer indices where WSGM modules will be attached
        """
        total_layers = self.backbone.n_blocks
        indices = []
        step = total_layers / self.num_wsgm_layers
        for i in range(self.num_wsgm_layers):
            idx = int(i * step + step / 2)
            indices.append(min(idx, total_layers - 1))
        return indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv3 + WSGM model

        Args:
            x (torch.Tensor): Input images of shape (batch, 3, height, width)

        Returns:
            torch.Tensor: Classification logits of shape (batch, num_classes)

        Note:
            During inference, use model.eval() to disable dropout layers.
        """
        # Extract intermediate layer features from backbone
        # get_intermediate_layers returns: List[Tensor] when return_class_token=False
        # or List[Tuple[Tensor, Tensor]] when return_class_token=True
        #   where each tuple is (patch_tokens, cls_token)
        # IMPORTANT: HuggingFace hidden_states[i] for i < total_layers are NOT normalized
        # BUT hidden_states[total_layers] (last layer) IS normalized with final norm!
        # So we need to handle the last layer specially if it's in wsgm_layer_indices

        # Check if last layer (23 for ViT-L) is in wsgm_layer_indices
        last_layer_idx = self.backbone.n_blocks - 1  # 23 for ViT-L
        has_last_layer = last_layer_idx in self.wsgm_layer_indices

        if has_last_layer:
            # Split into non-last and last layers
            non_last_indices = [idx for idx in self.wsgm_layer_indices if idx != last_layer_idx]

            # Extract non-last layers without norm
            if non_last_indices:
                outputs_no_norm = self.backbone.get_intermediate_layers(
                    x,
                    n=non_last_indices,
                    reshape=False,
                    return_class_token=True,
                    return_extra_tokens=False,
                    norm=False,  # No normalization
                )
            else:
                outputs_no_norm = []

            # Extract last layer WITH norm (to match HuggingFace hidden_states[24])
            outputs_with_norm = self.backbone.get_intermediate_layers(
                x,
                n=[last_layer_idx],
                reshape=False,
                return_class_token=True,
                return_extra_tokens=False,
                norm=True,  # Apply normalization for last layer
            )

            # Combine outputs in the correct order
            outputs = []
            non_last_iter = iter(outputs_no_norm)
            for idx in self.wsgm_layer_indices:
                if idx == last_layer_idx:
                    outputs.append(outputs_with_norm[0])
                else:
                    outputs.append(next(non_last_iter))
        else:
            # No last layer, use norm=False for all
            outputs = self.backbone.get_intermediate_layers(
                x,
                n=self.wsgm_layer_indices,
                reshape=False,
                return_class_token=True,
                return_extra_tokens=False,
                norm=False,
            )

        # Apply WSGM to cls tokens from each selected layer
        wsgm_outputs = []
        for i, (patches, cls_token) in enumerate(outputs):
            # cls_token: (batch, dim)
            # Apply WSGM with residual connection
            wsgm_out = self.wsgm_modules[i](cls_token)
            wsgm_outputs.append(cls_token + wsgm_out)

        # Aggregate WSGM outputs based on mode
        if self.aggregation_mode == "average":
            # Stack: (num_wsgm_layers, batch, dim) -> mean over dim 0 -> (batch, dim)
            x = torch.stack(wsgm_outputs, dim=0).mean(dim=0)
        else:  # concat
            # Concat: (batch, num_wsgm_layers * dim) -> project -> (batch, dim)
            x = torch.cat(wsgm_outputs, dim=-1)
            x = self.concat_proj(x)

        # Post-processing
        x = self.ln_post(x)

        # Classification
        logits = self.classifier(x)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classifier (for visualization, t-SNE, etc.)

        Args:
            x (torch.Tensor): Input images of shape (batch, 3, height, width)

        Returns:
            torch.Tensor: Feature vectors of shape (batch, dim)
        """
        # Extract intermediate layer features
        # IMPORTANT: HuggingFace's last hidden_states is normalized, so handle last layer specially
        last_layer_idx = self.backbone.n_blocks - 1
        has_last_layer = last_layer_idx in self.wsgm_layer_indices

        if has_last_layer:
            non_last_indices = [idx for idx in self.wsgm_layer_indices if idx != last_layer_idx]

            if non_last_indices:
                outputs_no_norm = self.backbone.get_intermediate_layers(
                    x,
                    n=non_last_indices,
                    reshape=False,
                    return_class_token=True,
                    return_extra_tokens=False,
                    norm=False,
                )
            else:
                outputs_no_norm = []

            outputs_with_norm = self.backbone.get_intermediate_layers(
                x,
                n=[last_layer_idx],
                reshape=False,
                return_class_token=True,
                return_extra_tokens=False,
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
                x,
                n=self.wsgm_layer_indices,
                reshape=False,
                return_class_token=True,
                return_extra_tokens=False,
                norm=False,
            )

        # Apply WSGM to cls tokens
        wsgm_outputs = []
        for i, (patches, cls_token) in enumerate(outputs):
            wsgm_out = self.wsgm_modules[i](cls_token)
            wsgm_outputs.append(cls_token + wsgm_out)

        # Aggregate WSGM outputs
        if self.aggregation_mode == "average":
            x = torch.stack(wsgm_outputs, dim=0).mean(dim=0)
        else:  # concat
            x = torch.cat(wsgm_outputs, dim=-1)
            x = self.concat_proj(x)

        # Apply layer norm
        x = self.ln_post(x)

        return x

    def get_num_params(self, trainable_only: bool = False) -> int:
        """
        Get the number of parameters in the model

        Args:
            trainable_only (bool): If True, count only trainable parameters

        Returns:
            int: Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def print_config(self):
        """Print model configuration"""
        print(f"\nDINOv3 + WSGM Configuration:")
        print(f"  Backbone: DINOv3 ({self.backbone.n_blocks} layers, dim={self.dim})")
        print(f"  Backbone frozen: {not any(p.requires_grad for p in self.backbone.parameters())}")
        print(f"  WSGM modules: {self.num_wsgm_layers}")
        print(f"  WSGM attached to layers: {self.wsgm_layer_indices}")
        if self.aggregation_mode == "average":
            print(f"  Aggregation: AVERAGE of all {self.num_wsgm_layers} WSGM outputs")
        else:
            print(f"  Aggregation: CONCAT of all {self.num_wsgm_layers} WSGM outputs + projection")
        print(f"  Number of classes: {self.num_classes}")

        # Count parameters
        trainable_params = self.get_num_params(trainable_only=True)
        total_params = self.get_num_params(trainable_only=False)
        frozen_params = total_params - trainable_params

        print(f"\n  Trainable params: {trainable_params:,}")
        print(f"  Frozen params: {frozen_params:,}")
        print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
