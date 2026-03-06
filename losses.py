"""Multi-view consistency and supervised contrastive losses for GenAI detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Positive pairs: (1) two views of the same image, (2) samples with the
    same label within the batch.

    Args:
        temperature: Softmax temperature for similarity scaling.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            features: ``(2*B, D)`` — L2-normalised projected embeddings
                (view1 embeddings concatenated with view2 embeddings).
            labels: ``(2*B,)`` — class labels (view1 labels concat view2 labels,
                same order so ``labels[i] == labels[i + B]``).

        Returns:
            Scalar loss averaged over all positive pairs.
        """
        device = features.device
        n = features.size(0)  # 2*B

        # L2 normalise
        features = F.normalize(features, dim=1)

        # Pairwise cosine similarity / temperature
        sim = torch.mm(features, features.t()) / self.temperature  # (2B, 2B)

        # Mask: positive pairs (same label, excluding self)
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.t()).float().to(device)  # (2B, 2B)

        # Exclude self-contrast (diagonal)
        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        positive_mask = positive_mask.masked_fill(self_mask, 0.0)

        # For numerical stability, subtract max from each row
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Exclude self from denominator
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.masked_fill(self_mask, 0.0)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean of log-prob over positive pairs
        log_prob = logits - log_sum_exp  # (2B, 2B)
        num_positives = positive_mask.sum(dim=1)  # (2B,)

        # Avoid division by zero for samples with no positives
        num_positives = torch.clamp(num_positives, min=1.0)
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / num_positives

        loss = -mean_log_prob.mean()
        return loss


class MultiViewConsistencyLoss(nn.Module):
    """Symmetrised KL divergence between two views' softmax distributions."""

    def forward(
        self, logits1: torch.Tensor, logits2: torch.Tensor
    ) -> torch.Tensor:
        """Compute symmetrised KL divergence.

        Args:
            logits1: ``(B, C)`` — logits from view 1.
            logits2: ``(B, C)`` — logits from view 2.

        Returns:
            Scalar loss: ``0.5 * (KL(p1 || p2) + KL(p2 || p1))``.
        """
        p1 = F.log_softmax(logits1, dim=1)
        p2 = F.log_softmax(logits2, dim=1)
        q1 = F.softmax(logits1, dim=1)
        q2 = F.softmax(logits2, dim=1)

        kl_12 = F.kl_div(p2, q1, reduction="batchmean", log_target=False)
        kl_21 = F.kl_div(p1, q2, reduction="batchmean", log_target=False)

        return 0.5 * (kl_12 + kl_21)


class MultiViewCriterion(nn.Module):
    """Combined loss: CE + Supervised Contrastive + Multi-View Consistency.

    ``L = L_ce + lambda_con * L_supcon + lambda_mvc * L_mvc``

    where ``L_ce`` is the average cross-entropy over both views.

    Args:
        lambda_con: Weight for supervised contrastive loss.
        lambda_mvc: Weight for multi-view consistency loss.
        temperature: Temperature for :class:`SupConLoss`.
        label_smoothing: Label smoothing for cross-entropy.
    """

    def __init__(
        self,
        lambda_con: float = 0.1,
        lambda_mvc: float = 0.05,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_con = lambda_con
        self.lambda_mvc = lambda_mvc
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.supcon = SupConLoss(temperature=temperature)
        self.mvc = MultiViewConsistencyLoss()

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        proj1: torch.Tensor,
        proj2: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """Compute combined loss.

        Args:
            logits1: ``(B, C)`` — classification logits from view 1.
            logits2: ``(B, C)`` — classification logits from view 2.
            proj1: ``(B, D)`` — projected embeddings from view 1.
            proj2: ``(B, D)`` — projected embeddings from view 2.
            labels: ``(B,)`` — ground-truth class labels.

        Returns:
            ``(total_loss, {"ce": float, "supcon": float, "mvc": float})``
        """
        ce_loss = 0.5 * (self.ce(logits1, labels) + self.ce(logits2, labels))

        # SupCon: concat both views' projections and labels
        proj_all = torch.cat([proj1, proj2], dim=0)
        labels_all = torch.cat([labels, labels], dim=0)
        supcon_loss = self.supcon(proj_all, labels_all)

        mvc_loss = self.mvc(logits1, logits2)

        total = (
            ce_loss
            + self.lambda_con * supcon_loss
            + self.lambda_mvc * mvc_loss
        )

        components = {
            "ce": ce_loss.item(),
            "supcon": supcon_loss.item(),
            "mvc": mvc_loss.item(),
        }
        return total, components
