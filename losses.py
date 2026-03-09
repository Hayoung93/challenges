"""Multi-view consistency and supervised contrastive losses for GenAI detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) with optional label smoothing.

    Down-weights well-classified examples so training focuses on hard samples.
    When ``gamma=0`` this reduces to standard cross-entropy.

    Label smoothing is applied *before* computing the focal modulation factor
    (pre-smoothing), so ``p_t`` is measured against the smoothed target
    distribution.  This ensures ``FocalLoss(gamma=0, label_smoothing=s)``
    is numerically identical to ``nn.CrossEntropyLoss(label_smoothing=s)``.

    Args:
        gamma: Focusing parameter (0 = standard CE, 2.0 typical).
        alpha: Optional per-class weight tensor of shape ``(C,)``.
        label_smoothing: Label smoothing factor.
        reduction: ``"none"`` | ``"mean"`` | ``"sum"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: ``(B, C)`` raw class scores.
            targets: ``(B,)`` integer class labels.

        Returns:
            Loss tensor whose shape depends on *reduction*.
        """
        C = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Smoothed one-hot targets
        with torch.no_grad():
            targets_oh = torch.zeros_like(logits)
            targets_oh.scatter_(1, targets.unsqueeze(1), 1.0)
            if self.label_smoothing > 0.0:
                targets_oh = (
                    targets_oh * (1.0 - self.label_smoothing)
                    + self.label_smoothing / C
                )

        # p_t: probability assigned to the (smoothed) target distribution
        p_t = (probs * targets_oh).sum(dim=1)  # (B,)

        # Focal modulating factor
        focal_weight = (1.0 - p_t) ** self.gamma  # (B,)

        # Per-sample CE with smoothed targets
        ce = -(targets_oh * log_probs).sum(dim=1)  # (B,)

        loss = focal_weight * ce  # (B,)

        # Optional per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HardSampleMiningLoss(nn.Module):
    """In-batch online hard sample mining wrapper.

    Computes per-sample loss via the wrapped *base_loss*, then keeps only the
    top-k hardest samples (highest loss) for gradient computation.

    When ``keep_ratio=1.0`` this is a transparent pass-through.

    Args:
        base_loss: Loss module that supports ``reduction="none"``.
        keep_ratio: Fraction of batch to keep ``(0, 1]``.
        min_keep: Minimum samples to keep regardless of *keep_ratio*.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        keep_ratio: float = 1.0,
        min_keep: int = 4,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.keep_ratio = keep_ratio
        self.min_keep = min_keep

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute hard-mined loss.

        Args:
            logits: ``(B, C)`` raw class scores.
            targets: ``(B,)`` integer class labels.

        Returns:
            Scalar loss averaged over the kept (hard) samples.
        """
        # Temporarily switch base_loss to per-sample mode
        old_reduction = self.base_loss.reduction
        self.base_loss.reduction = "none"
        try:
            per_sample = self.base_loss(logits, targets)  # (B,)
        finally:
            self.base_loss.reduction = old_reduction

        B = per_sample.size(0)

        if self.keep_ratio >= 1.0 or B <= self.min_keep:
            return per_sample.mean()

        k = max(int(B * self.keep_ratio + 0.5), self.min_keep)
        k = min(k, B)

        topk_losses, _ = torch.topk(per_sample, k, sorted=False)
        return topk_losses.mean()

    def mine(self, per_sample_losses: torch.Tensor) -> torch.Tensor:
        """Apply top-k selection on pre-computed per-sample losses.

        Useful when the caller needs to combine per-sample losses from
        multiple sources (e.g. multi-view) before mining.
        """
        B = per_sample_losses.size(0)
        if self.keep_ratio >= 1.0 or B <= self.min_keep:
            return per_sample_losses.mean()
        k = max(int(B * self.keep_ratio + 0.5), self.min_keep)
        k = min(k, B)
        topk, _ = torch.topk(per_sample_losses, k, sorted=False)
        return topk.mean()


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
    """KL divergence between two views' softmax distributions.

    Supports two modes:

    * **Symmetric** (default): ``0.5 * (KL(p1 || p2) + KL(p2 || p1))``
    * **Teacher-student** (``teacher_student=True``): ``KL(student || teacher)``,
      where *logits1* is the student and *logits2* is the (detached) teacher.
    """

    def __init__(self, teacher_student: bool = False):
        super().__init__()
        self.teacher_student = teacher_student

    def forward(
        self, logits1: torch.Tensor, logits2: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence loss.

        Args:
            logits1: ``(B, C)`` — student (or view 1) logits.
            logits2: ``(B, C)`` — teacher (or view 2) logits.

        Returns:
            Scalar KL divergence loss.
        """
        if self.teacher_student:
            # One-directional: student learns to match teacher
            log_student = F.log_softmax(logits1, dim=1)
            teacher_prob = F.softmax(logits2.detach(), dim=1)
            return F.kl_div(log_student, teacher_prob, reduction="batchmean", log_target=False)

        # Symmetric mode (original)
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
        mvc_ema: bool = False,
        ce_criterion: nn.Module | None = None,
    ):
        super().__init__()
        self.lambda_con = lambda_con
        self.lambda_mvc = lambda_mvc
        self.mvc_ema = mvc_ema
        if ce_criterion is not None:
            self.ce = ce_criterion
        else:
            self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.supcon = SupConLoss(temperature=temperature)
        self.mvc = MultiViewConsistencyLoss(teacher_student=mvc_ema)

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        proj1: torch.Tensor,
        proj2: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """Compute combined loss.

        When ``mvc_ema=True``, *logits1/proj1* are from the student and
        *logits2/proj2* are from the EMA teacher (clean view).  CE is
        computed on the student only, and SupCon uses student projections
        from both views (proj2 is detached teacher output).

        Args:
            logits1: ``(B, C)`` — student (or view 1) classification logits.
            logits2: ``(B, C)`` — teacher (or view 2) classification logits.
            proj1: ``(B, D)`` — student (or view 1) projected embeddings.
            proj2: ``(B, D)`` — teacher (or view 2) projected embeddings.
            labels: ``(B,)`` — ground-truth class labels.

        Returns:
            ``(total_loss, {"ce": float, "supcon": float, "mvc": float})``
        """
        if self.mvc_ema:
            # EMA mode: CE on student only
            ce_loss = self.ce(logits1, labels)
        elif isinstance(self.ce, HardSampleMiningLoss):
            # Unified mining: combine per-sample CE from both views,
            # then apply a single top-k selection so the same samples
            # are kept for both views.
            base = self.ce.base_loss
            old_reduction = base.reduction
            base.reduction = "none"
            try:
                combined = 0.5 * (base(logits1, labels) + base(logits2, labels))
            finally:
                base.reduction = old_reduction
            ce_loss = self.ce.mine(combined)
        else:
            ce_loss = 0.5 * (self.ce(logits1, labels) + self.ce(logits2, labels))

        # SupCon: concat both views' projections and labels
        proj_all = torch.cat([proj1, proj2.detach() if self.mvc_ema else proj2], dim=0)
        labels_all = torch.cat([labels, labels], dim=0)
        supcon_loss = self.supcon(proj_all, labels_all)

        # MVC: in EMA mode, logits1=student, logits2=teacher (detach inside loss)
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


class MoECriterion(nn.Module):
    """Loss for Mixture of Experts training.

    Routes the classification loss to the expert heads whose augmentation
    groups were active for each sample.

    For each active expert *k*, CE loss is computed on the subset of
    samples where that expert's augmentation group was applied.
    The final loss averages over all active experts.

    Args:
        base_criterion: Underlying per-sample loss (CrossEntropyLoss,
            FocalLoss, etc.).
        num_experts: Number of expert heads (default 8).
    """

    def __init__(self, base_criterion: nn.Module, num_experts: int = 8):
        super().__init__()
        self.base_criterion = base_criterion
        self.num_experts = num_experts

    def forward(
        self,
        all_logits: torch.Tensor,
        labels: torch.Tensor,
        expert_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expert-routed loss.

        Args:
            all_logits: ``(B, K, C)`` logits from all expert heads.
            labels: ``(B,)`` ground-truth class indices.
            expert_masks: ``(B, K)`` binary mask — 1 if expert *k* is
                active for sample *i*.

        Returns:
            Scalar loss (mean across active experts).
        """
        total_loss = torch.tensor(
            0.0, device=all_logits.device, dtype=all_logits.dtype,
        )
        count = 0

        for k in range(self.num_experts):
            mask = expert_masks[:, k].bool()
            if not mask.any():
                continue
            loss_k = self.base_criterion(all_logits[mask, k, :], labels[mask])
            total_loss = total_loss + loss_k
            count += 1

        if count == 0:
            return total_loss.requires_grad_()
        return total_loss / count


def build_criterion(args) -> nn.Module:
    """Build the training loss criterion from config flags.

    Returns one of:

    * ``nn.CrossEntropyLoss`` — default (``focal_gamma=0``, ``ohsm_enabled=False``)
    * ``FocalLoss`` — ``focal_gamma > 0`` only
    * ``HardSampleMiningLoss(CrossEntropyLoss)`` — ``ohsm_enabled``, ``focal_gamma=0``
    * ``HardSampleMiningLoss(FocalLoss)`` — both enabled
    * ``MoECriterion(base_loss)`` — ``moe_enabled=True``
    """
    label_smoothing = getattr(args, "label_smoothing", 0.1)
    focal_gamma = getattr(args, "focal_gamma", 0.0)
    ohsm_enabled = getattr(args, "ohsm_enabled", False)
    ohsm_keep_ratio = getattr(args, "ohsm_keep_ratio", 1.0)
    ohsm_min_keep = getattr(args, "ohsm_min_keep", 4)

    if focal_gamma > 0.0:
        base_loss = FocalLoss(
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
            reduction="mean",
        )
    else:
        base_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if ohsm_enabled and ohsm_keep_ratio < 1.0:
        base_loss = HardSampleMiningLoss(
            base_loss=base_loss,
            keep_ratio=ohsm_keep_ratio,
            min_keep=ohsm_min_keep,
        )

    if getattr(args, "moe_enabled", False):
        from models.moe import NUM_EXPERTS

        return MoECriterion(base_loss, num_experts=NUM_EXPERTS)

    return base_loss
