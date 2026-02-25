# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Weighted Side Gating Module (WSGM) for DINOv3

WSGM is a lightweight adapter module for forgery-specific feature learning.
Structure: Input -> Down -> Middle -> Up -> Output (with residual connection)
"""

import torch
import torch.nn as nn


class WSGM(nn.Module):
    """
    Weighted Side Gating Module (from ForgeLens)

    A lightweight bottleneck module that adapts features for specific tasks
    while maintaining the original feature distribution through residual connections.

    Structure:
        Input (dim) -> Down (bottleneck_dim) -> Middle (bottleneck_dim) -> Up (dim) -> Output
        Output = Input + WSGM(Input)  # Residual connection

    Args:
        input_dim (int): Input dimension (e.g., 1024 for ViT-L)
        bottleneck_dim (int): Bottleneck dimension (e.g., 256 for reduction_factor=4)
        dropout_prob (float): Dropout probability (default: 0.5)

    Example:
        >>> wsgm = WSGM(input_dim=1024, bottleneck_dim=256, dropout_prob=0.5)
        >>> x = torch.randn(8, 1024)  # (batch, dim)
        >>> out = wsgm(x)  # (batch, dim)
        >>> residual_out = x + out  # Apply residual
    """

    def __init__(self, input_dim: int, bottleneck_dim: int, dropout_prob: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Down projection: input_dim -> bottleneck_dim
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.relu1 = nn.ReLU()

        # Middle projection: bottleneck_dim -> bottleneck_dim
        self.middle = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.relu2 = nn.ReLU()

        # Up projection: bottleneck_dim -> input_dim
        self.up = nn.Linear(bottleneck_dim, input_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with small normal distribution"""
        nn.init.normal_(self.down.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.down.bias, 0)
        nn.init.normal_(self.middle.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.middle.bias, 0)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.up.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through WSGM module

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch, input_dim)
                         Note: Residual connection should be applied externally

        Note:
            This module returns the WSGM transformation only.
            The residual connection (x + wsgm(x)) should be applied by the caller.
        """
        out = self.down(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.middle(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.up(out)
        return out
