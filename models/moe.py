"""Mixture of Experts classification head for GenAI detection.

Shared backbone produces features; K expert Linear heads each produce logits.
No gating network — inference uses entropy-based confidence weighting with
learned per-expert temperature scaling.

Training: Loss is routed to the expert heads whose augmentation groups were
active for each sample.

Inference: All experts score the image; outputs are combined via
entropy-based confidence weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

# Canonical expert group names matching the 7 robust augmentation groups
# plus a dedicated "clean" expert.
EXPERT_GROUPS = [
    "blur",
    "compression",
    "noise",
    "resize",
    "color",
    "spatial",
    "sharpness_brightness",
    "clean",
]
NUM_EXPERTS = len(EXPERT_GROUPS)
EXPERT_GROUP_TO_IDX = {name: i for i, name in enumerate(EXPERT_GROUPS)}


def entropy_weighted_aggregate(
    all_logits: torch.Tensor,
    temperatures: torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """Entropy-weighted combination of expert logits.

    Args:
        all_logits: ``(B, K, C)`` raw logits from each expert.
        temperatures: ``(K,)`` positive per-expert temperature scalars.
        num_classes: Number of output classes (used for max entropy).

    Returns:
        ``(B, C)`` aggregated logits.
    """
    # Temperature scaling: (1, K, 1) broadcasts over batch and classes.
    temps = temperatures.unsqueeze(0).unsqueeze(-1)  # (1, K, 1)
    scaled_logits = all_logits / temps  # (B, K, C)

    # Per-expert softmax and entropy.
    probs = F.softmax(scaled_logits, dim=-1)  # (B, K, C)
    log_probs = F.log_softmax(scaled_logits, dim=-1)  # (B, K, C)
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, K)

    # Maximum entropy for C classes.
    max_entropy = torch.log(
        torch.tensor(
            float(num_classes), device=entropy.device,
            dtype=entropy.dtype,
        )
    )

    # Confidence weight: low entropy → high weight.
    weights = (1.0 - entropy / max_entropy).clamp(min=0.0)  # (B, K)

    # Normalise so weights sum to 1 per sample.
    weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    weights = weights / weight_sum  # (B, K)

    # Weighted combination of probability distributions.
    combined_probs = (weights.unsqueeze(-1) * probs).sum(dim=1)  # (B, C)

    # Convert back to logits for compatibility with argmax / softmax.
    return torch.log(combined_probs.clamp(min=1e-8))


class MoEHead(nn.Module):
    """Mixture of Experts classification head.

    Args:
        in_features: Feature dimension from backbone.
        num_classes: Number of output classes (default 2).
        num_experts: Number of expert heads (default 8).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        num_experts: int = NUM_EXPERTS,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_experts = num_experts

        self.expert_heads = nn.ModuleList(
            [nn.Linear(in_features, num_classes) for _ in range(num_experts)]
        )
        for head in self.expert_heads:
            trunc_normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

        # Per-expert temperature for post-hoc calibration.
        # Stored as log(tau) so exp() is always positive.
        # Frozen during training; optimised on validation set afterwards.
        self.log_temperatures = nn.Parameter(
            torch.zeros(num_experts), requires_grad=False,
        )

    @property
    def temperatures(self) -> torch.Tensor:
        """Per-expert temperature scalars (always positive)."""
        return self.log_temperatures.exp()

    def forward_all_experts(self, features: torch.Tensor) -> torch.Tensor:
        """Run all expert heads on the same features.

        Args:
            features: ``(B, D)`` backbone features.

        Returns:
            ``(B, K, C)`` logits from each expert.
        """
        return torch.stack(
            [head(features) for head in self.expert_heads], dim=1,
        )

    def forward_routed(
        self,
        features: torch.Tensor,
        expert_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward: compute logits for all experts.

        Inactive experts still produce logits (cheap linear ops) but they
        will be masked out in the loss.  Computing all keeps the code simple
        and avoids scattered indexing.

        Args:
            features: ``(B, D)`` backbone features.
            expert_masks: ``(B, K)`` binary mask (unused here but kept for
                API symmetry with the loss).

        Returns:
            ``(B, K, C)`` logits from every expert.
        """
        return self.forward_all_experts(features)

    def inference_aggregate(self, features: torch.Tensor) -> torch.Tensor:
        """Entropy-weighted combination of all expert predictions.

        Delegates to :func:`entropy_weighted_aggregate` after computing
        per-expert logits.

        Args:
            features: ``(B, D)`` backbone features.

        Returns:
            ``(B, C)`` aggregated logits.
        """
        all_logits = self.forward_all_experts(features)  # (B, K, C)
        return entropy_weighted_aggregate(
            all_logits, self.temperatures, self.num_classes,
        )
