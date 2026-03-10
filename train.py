"""GenAI image detection — training and validation script."""

import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import add_data_args, add_model_args, add_train_args, merge_config
from data import build_train_val_loaders
from models import build_model


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════


def seed_everything(seed: int, rank: int = 0) -> None:
    """Set seed for reproducibility across random, numpy, and torch.

    In DDP, each rank adds its rank to the seed so augmentations differ
    across workers while remaining reproducible.
    """
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed(args) -> tuple:
    """Initialize the distributed process group.

    Reads RANK, LOCAL_RANK, WORLD_SIZE from environment (set by torchrun).

    Returns:
        (rank, local_rank, world_size)
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=getattr(args, "dist_backend", "nccl"),
        rank=rank,
        world_size=world_size,
    )
    dist.barrier(device_ids=[local_rank])
    return rank, local_rank, world_size


def cleanup_distributed():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args) -> bool:
    """Return True if this is rank 0 or non-distributed training."""
    if not getattr(args, "distributed", False):
        return True
    return getattr(args, "_rank", 0) == 0


def print_rank0(msg, args):
    """Print only on the main process."""
    if is_main_process(args):
        print(msg)


def _find_ohsm(criterion: nn.Module):
    """Find the :class:`HardSampleMiningLoss` inside *criterion*, if any."""
    from losses import HardSampleMiningLoss, MoEMultiViewCriterion

    if isinstance(criterion, HardSampleMiningLoss):
        return criterion
    if hasattr(criterion, "ce") and isinstance(criterion.ce, HardSampleMiningLoss):
        return criterion.ce
    # MoEMultiViewCriterion or MoECriterion: base_criterion may be OHSM
    if hasattr(criterion, "base_criterion") and isinstance(
        criterion.base_criterion, HardSampleMiningLoss
    ):
        return criterion.base_criterion
    return None


def _update_ohsm_ratio(criterion: nn.Module, ratio: float) -> None:
    """Update *keep_ratio* on a :class:`HardSampleMiningLoss`, if present."""
    ohsm = _find_ohsm(criterion)
    if ohsm is not None:
        ohsm.keep_ratio = ratio


def _get_ohsm_ratio(criterion: nn.Module):
    """Return current *keep_ratio* or ``None`` if OHSM is not active."""
    ohsm = _find_ohsm(criterion)
    return ohsm.keep_ratio if ohsm is not None else None


def gather_predictions(local_probs: torch.Tensor, local_labels: torch.Tensor) -> tuple:
    """Gather predictions and labels from all ranks.

    Handles variable sizes across ranks by padding to the max size.

    Args:
        local_probs: Tensor of shape (N_local,) on the current device.
        local_labels: Tensor of shape (N_local,) on the current device.

    Returns:
        (all_probs, all_labels) gathered across all ranks.
    """
    world_size = dist.get_world_size()

    local_size = torch.tensor([local_probs.size(0)], dtype=torch.long, device=local_probs.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_size = max(s.item() for s in all_sizes)

    padded_probs = torch.zeros(max_size, device=local_probs.device, dtype=local_probs.dtype)
    padded_probs[:local_probs.size(0)] = local_probs
    padded_labels = torch.zeros(max_size, device=local_labels.device, dtype=local_labels.dtype)
    padded_labels[:local_labels.size(0)] = local_labels

    gathered_probs = [torch.zeros_like(padded_probs) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(padded_labels) for _ in range(world_size)]
    dist.all_gather(gathered_probs, padded_probs)
    dist.all_gather(gathered_labels, padded_labels)

    all_probs_list = []
    all_labels_list = []
    for i in range(world_size):
        n = all_sizes[i].item()
        all_probs_list.append(gathered_probs[i][:n])
        all_labels_list.append(gathered_labels[i][:n])

    return torch.cat(all_probs_list), torch.cat(all_labels_list)


def _wrap_mamba_fp32(model: nn.Module) -> None:
    """Force MambaVisionMixer layers to run in fp32 under AMP.

    Disables autocast inside each MambaVisionMixer.forward() so that
    selective-scan operations execute in fp32, preventing fp16 overflow.
    Other layers (Conv, Attention) remain in fp16.
    """
    patched = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == "MambaVisionMixer":
            original_forward = module.forward

            def _make_fp32_forward(fwd):
                def _fp32_forward(hidden_states):
                    with torch.amp.autocast(device_type="cuda", enabled=False):
                        return fwd(hidden_states.float())
                return _fp32_forward

            module.forward = _make_fp32_forward(original_forward)
            patched += 1
    if patched:
        print(f"  Wrapped {patched} MambaVisionMixer layer(s) to fp32")


def vram_precheck(
    model: nn.Module,
    image_size: int,
    batch_size: int,
    device: torch.device,
    amp: bool = True,
    args=None,
) -> None:
    """Run a dummy forward pass to verify VRAM is sufficient.

    Uses the largest resolution from the multi-scale pool to ensure
    training won't OOM mid-epoch.  If the check fails, prints a
    diagnostic message with suggested batch sizes and exits.

    Note: This check uses inference mode (no gradients) so actual
    training VRAM usage will be higher due to gradient and optimizer
    state memory.
    """
    if device.type != "cuda":
        return

    print_rank0(
        f"  VRAM pre-check: batch_size={batch_size}, "
        f"image_size={image_size}x{image_size} ...",
        args,
    )

    from models.classifier import update_mambavision_window_size
    update_mambavision_window_size(model, image_size)

    dummy_input = torch.randn(
        batch_size, 3, image_size, image_size, device=device,
    )
    dummy_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    criterion_check = nn.CrossEntropyLoss()

    # LoRA-MoE: use single expert for VRAM check to avoid K× forward.
    # We bypass the classifier's forward (which loops over all experts)
    # and directly test a single-expert forward pass.
    _lora_moe_active = False
    if args is not None and getattr(args, "lora_moe_enabled", False):
        _lora_moe_active = True

    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=amp):
                if _lora_moe_active:
                    from models.lora_moe import (
                        clear_lora_moe_state,
                        set_active_expert,
                    )
                    raw = model.module if hasattr(model, "module") else model
                    set_active_expert(model, 0)
                    features = raw._extract_features(dummy_input)
                    logits = raw.backbone.head(features)
                    clear_lora_moe_state(model)
                else:
                    logits = model(dummy_input)
                _ = criterion_check(logits, dummy_labels)
        model.train(was_training)

        del dummy_input, dummy_labels, logits
        torch.cuda.empty_cache()
        print_rank0("  VRAM pre-check: PASSED", args)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if _lora_moe_active:
            from models.lora_moe import clear_lora_moe_state
            clear_lora_moe_state(model)

        suggested = None
        for try_bs in [batch_size // 2, batch_size // 4, batch_size // 8]:
            if try_bs < 1:
                break
            try:
                dummy = torch.randn(
                    try_bs, 3, image_size, image_size, device=device,
                )
                model.eval()
                with torch.no_grad():
                    with autocast(device_type="cuda", enabled=amp):
                        out = model(dummy)
                del dummy, out
                torch.cuda.empty_cache()
                suggested = try_bs
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

        vram_total = torch.cuda.get_device_properties(device).total_mem / (1024**3)
        msg = (
            f"\n  VRAM pre-check: FAILED\n"
            f"  GPU: {torch.cuda.get_device_name(device)} ({vram_total:.1f} GB)\n"
            f"  Requested: batch_size={batch_size}, "
            f"image_size={image_size}x{image_size}\n"
        )
        if suggested:
            msg += f"  Suggested: --batch_size {suggested}\n"
        else:
            msg += (
                f"  Even batch_size=1 failed. Consider:\n"
                f"    - Removing {image_size} from --multiscale_sizes\n"
                f"    - Using a smaller model\n"
                f"    - Enabling --amp\n"
            )
        print(msg)
        raise SystemExit(1)


def _log_training_images(
    writer: SummaryWriter,
    global_step: int,
    images: torch.Tensor,
    clean_images: torch.Tensor | None,
    count: int,
) -> None:
    """Log a grid of training input images to TensorBoard.

    For multi-view training, ``images`` is the augmented view and
    ``clean_images`` is the clean (geometric-only) view.  The two are
    rendered as a vertically-stacked grid (augmented on top, clean on
    bottom) so that each column forms a visual pair.

    Images are denormalized from ImageNet stats before logging.
    """
    from torchvision.utils import make_grid

    from data.transforms import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def denorm(x: torch.Tensor) -> torch.Tensor:
        return (x.detach().cpu().float() * std + mean).clamp(0, 1)

    if clean_images is not None:
        # Multi-view: augmented on top, clean on bottom
        n = min(count, images.size(0))
        aug_grid = make_grid(denorm(images[:n]), nrow=n, padding=2)
        clean_grid = make_grid(denorm(clean_images[:n]), nrow=n, padding=2)
        paired = torch.cat([aug_grid, clean_grid], dim=1)
        writer.add_image("train/input_paired", paired, global_step)
    else:
        n = min(count, images.size(0))
        nrow = min(n, 4)
        grid = make_grid(denorm(images[:n]), nrow=nrow, padding=2)
        writer.add_image("train/input_images", grid, global_step)


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """Update teacher parameters as EMA of student parameters.

    Automatically unwraps DDP / DataParallel wrappers so that parameter
    names match between the (possibly wrapped) student and the
    (unwrapped) teacher.
    """
    student_raw = student.module if hasattr(student, "module") else student
    student_params = dict(student_raw.named_parameters())
    for name, teacher_param in teacher.named_parameters():
        if name in student_params:
            teacher_param.data.mul_(decay).add_(student_params[name].data, alpha=1.0 - decay)


def build_ema_teacher(model: nn.Module) -> nn.Module:
    """Create an EMA teacher by deep-copying the model and freezing it."""
    import copy
    teacher = copy.deepcopy(model)
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


class AverageMeter:
    """Computes and stores a running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class EarlyStopping:
    """Track validation metric and signal when to stop training.

    Args:
        patience: Epochs without improvement before stopping. 0 = disabled.
    """

    def __init__(self, patience: int = 7):
        self.patience = patience
        self.best_score = float("-inf")
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def step(self, score: float, epoch: int) -> bool:
        """Update with new validation score. Returns True if improved."""
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.enabled and self.counter >= self.patience:
                self.should_stop = True
            return False


# ═══════════════════════════════════════════════════════════════════
#  Optimizer & Scheduler
# ═══════════════════════════════════════════════════════════════════


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Build AdamW optimizer with weight-decay only on non-bias, non-norm params."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=args.lr)


def build_scheduler(optimizer, args, steps_per_epoch: int):
    """Build LR scheduler with linear warmup + cosine/step main schedule.

    Scheduler is stepped per-batch (not per-epoch).
    """
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=min(1e-6 / max(args.lr, 1e-8), 1.0),
        end_factor=1.0,
        total_iters=max(warmup_steps, 1),
    )

    if args.scheduler == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=1e-7,
        )
    elif args.scheduler == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_lr_size * steps_per_epoch,
            gamma=args.step_lr_decay,
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler


# ═══════════════════════════════════════════════════════════════════
#  Training & Validation loops
# ═══════════════════════════════════════════════════════════════════


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args,
    *,
    writer: SummaryWriter | None = None,
    ema_teacher: nn.Module | None = None,
) -> dict:
    """Train for one epoch. Returns dict with 'loss' and 'accuracy'.

    When ``args.multi_view`` is True, the loader yields 4-tuples
    ``(view1, view2, labels, metadata)`` and sub-loss components
    (``loss_ce``, ``loss_supcon``, ``loss_mvc``) are included in
    the returned dict.

    When *writer* is not None, augmented input images are logged to
    TensorBoard ``tb_log_images_per_epoch`` times during the epoch.
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    use_multi_view = getattr(args, "multi_view", False)
    use_moe = getattr(args, "moe_enabled", False)
    use_lora_moe = getattr(args, "lora_moe_enabled", False)
    use_moe_mv = use_multi_view and use_moe
    use_lora_moe_mv = use_multi_view and use_lora_moe
    sub_loss_meters = {}
    if use_lora_moe_mv:
        sub_loss_meters = {k: AverageMeter() for k in ("ce", "supcon", "mvc")}
    elif use_moe_mv:
        sub_loss_meters = {
            k: AverageMeter()
            for k in ("ce_moe", "ce_clean", "ce", "supcon", "mvc")
        }
    elif use_multi_view:
        sub_loss_meters = {k: AverageMeter() for k in ("ce", "supcon", "mvc")}

    # Iteration-level multi-scale setup
    ms_interval = getattr(args, "multiscale_interval", 0)
    use_iter_ms = getattr(args, "multiscale", False) and ms_interval > 0
    ms_sizes = getattr(args, "multiscale_sizes", [])
    ms_base_size = getattr(args, "_multiscale_train_size", getattr(args, "image_size", 224))
    current_ms_size = ms_base_size  # start at max (no downscale on first batch)

    # Same-label CutMix setup
    _ROBUST_AUGS = {"robust", "robust_curriculum", "robust_curriculum_range"}
    _cutmix_active = (
        getattr(args, "cutmix_p", 0.0) > 0.0
        and getattr(args, "augmentation", "default") in _ROBUST_AUGS
    )

    # Image logging schedule: compute which batch indices to log at
    _img_log_steps: set[int] = set()
    if writer is not None and getattr(args, "tb_log_images", True):
        n_logs = getattr(args, "tb_log_images_per_epoch", 5)
        total_batches = len(loader)
        if total_batches > 0 and n_logs > 0:
            interval = max(total_batches // n_logs, 1)
            _img_log_steps = {i * interval for i in range(n_logs)}

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False,
                disable=not is_main_process(args))
    for batch_idx, batch in enumerate(pbar):
        # Iteration-level multi-scale: change resolution every N iterations
        if use_iter_ms and batch_idx % ms_interval == 0:
            current_ms_size = random.choice(ms_sizes)
            from models.classifier import update_mambavision_window_size
            update_mambavision_window_size(model, current_ms_size)
            if ema_teacher is not None:
                update_mambavision_window_size(ema_teacher, current_ms_size)

        if use_moe_mv:
            # ── MoE + Multi-View combined branch ──
            views1, views2, labels, _metadata = batch
            views1 = views1.to(device, non_blocking=True)
            views2 = views2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = views1.size(0)

            # GPU-side nearest downscale for multi-scale
            if use_iter_ms and current_ms_size != ms_base_size:
                views1 = nn.functional.interpolate(
                    views1, size=current_ms_size, mode="nearest",
                )
                views2 = nn.functional.interpolate(
                    views2, size=current_ms_size, mode="nearest",
                )

            # Same-label CutMix (multi-view)
            if _cutmix_active:
                from data.cutmix import same_label_cutmix_multi_view
                views1, views2 = same_label_cutmix_multi_view(
                    views1, views2, labels,
                    p=args.cutmix_p, alpha=args.cutmix_alpha,
                )

            # Log augmented/clean pairs to TensorBoard
            if batch_idx in _img_log_steps:
                global_step = epoch * len(loader) + batch_idx
                _log_training_images(
                    writer, global_step, views1, views2,
                    count=getattr(args, "tb_log_images_pairs", 4),
                )

            # Build expert masks for augmented view from metadata
            from models.moe import EXPERT_GROUP_TO_IDX, NUM_EXPERTS

            expert_masks_v1 = torch.zeros(
                batch_size, NUM_EXPERTS,
                device=device, dtype=torch.float32,
            )
            for i, meta in enumerate(_metadata):
                groups = meta.get("aug_groups", frozenset({"clean"}))
                for g in groups:
                    idx = EXPERT_GROUP_TO_IDX.get(g)
                    if idx is not None:
                        expert_masks_v1[i, idx] = 1.0
                if not groups or "clean" in groups:
                    expert_masks_v1[i, EXPERT_GROUP_TO_IDX["clean"]] = 1.0

            # Clean view: always route to clean expert
            expert_masks_v2 = torch.zeros(
                batch_size, NUM_EXPERTS,
                device=device, dtype=torch.float32,
            )
            expert_masks_v2[:, EXPERT_GROUP_TO_IDX["clean"]] = 1.0

            with autocast(device_type="cuda", enabled=args.amp):
                all_logits_v1, _, proj1 = model(
                    views1, return_embedding=True,
                    moe_expert_masks=expert_masks_v1,
                )

                if ema_teacher is not None:
                    with torch.no_grad():
                        all_logits_v2, _, proj2 = ema_teacher(
                            views2, return_embedding=True,
                            moe_expert_masks=expert_masks_v2,
                        )
                else:
                    all_logits_v2, _, proj2 = model(
                        views2, return_embedding=True,
                        moe_expert_masks=expert_masks_v2,
                    )

                loss, loss_components = criterion(
                    all_logits_v1, all_logits_v2,
                    expert_masks_v1, proj1, proj2, labels,
                )

            # Accuracy: augmented view's active-expert-averaged logits
            with torch.no_grad():
                avg_logits = (
                    all_logits_v1 * expert_masks_v1.unsqueeze(-1)
                ).sum(dim=1)
                avg_logits = avg_logits / expert_masks_v1.sum(
                    dim=1, keepdim=True,
                ).clamp(min=1)
            preds = avg_logits.argmax(dim=1)

            for k, v in loss_components.items():
                sub_loss_meters[k].update(v, batch_size)

        elif use_lora_moe_mv:
            # ── LoRA-MoE + Multi-View combined branch ──
            # LoRA-MoE outputs (B, C) via shared head (not B,K,C),
            # so we use standard MultiViewCriterion.
            views1, views2, labels, _metadata = batch
            views1 = views1.to(device, non_blocking=True)
            views2 = views2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = views1.size(0)

            # GPU-side nearest downscale for multi-scale
            if use_iter_ms and current_ms_size != ms_base_size:
                views1 = nn.functional.interpolate(
                    views1, size=current_ms_size, mode="nearest",
                )
                views2 = nn.functional.interpolate(
                    views2, size=current_ms_size, mode="nearest",
                )

            # Same-label CutMix (multi-view)
            if _cutmix_active:
                from data.cutmix import same_label_cutmix_multi_view
                views1, views2 = same_label_cutmix_multi_view(
                    views1, views2, labels,
                    p=args.cutmix_p, alpha=args.cutmix_alpha,
                )

            # Log augmented/clean pairs to TensorBoard
            if batch_idx in _img_log_steps:
                global_step = epoch * len(loader) + batch_idx
                _log_training_images(
                    writer, global_step, views1, views2,
                    count=getattr(args, "tb_log_images_pairs", 4),
                )

            # Build expert masks for augmented view from metadata
            from models.moe import EXPERT_GROUP_TO_IDX, NUM_EXPERTS

            expert_masks_v1 = torch.zeros(
                batch_size, NUM_EXPERTS,
                device=device, dtype=torch.float32,
            )
            for i, meta in enumerate(_metadata):
                groups = meta.get("aug_groups", frozenset({"clean"}))
                for g in groups:
                    idx = EXPERT_GROUP_TO_IDX.get(g)
                    if idx is not None:
                        expert_masks_v1[i, idx] = 1.0
                if not groups or "clean" in groups:
                    expert_masks_v1[i, EXPERT_GROUP_TO_IDX["clean"]] = 1.0

            # Clean view: always route to clean expert
            expert_masks_v2 = torch.zeros(
                batch_size, NUM_EXPERTS,
                device=device, dtype=torch.float32,
            )
            expert_masks_v2[:, EXPERT_GROUP_TO_IDX["clean"]] = 1.0

            with autocast(device_type="cuda", enabled=args.amp):
                logits1, _, proj1 = model(
                    views1, return_embedding=True,
                    moe_expert_masks=expert_masks_v1,
                )

                if ema_teacher is not None:
                    with torch.no_grad():
                        logits2, _, proj2 = ema_teacher(
                            views2, return_embedding=True,
                            moe_expert_masks=expert_masks_v2,
                        )
                else:
                    logits2, _, proj2 = model(
                        views2, return_embedding=True,
                        moe_expert_masks=expert_masks_v2,
                    )

                loss, loss_components = criterion(
                    logits1, logits2, proj1, proj2, labels,
                )

            preds = logits1.argmax(dim=1)
            for k, v in loss_components.items():
                sub_loss_meters[k].update(v, batch_size)

        elif use_multi_view:
            views1, views2, labels, _metadata = batch
            views1 = views1.to(device, non_blocking=True)
            views2 = views2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = views1.size(0)

            # GPU-side nearest downscale for multi-scale
            if use_iter_ms and current_ms_size != ms_base_size:
                views1 = nn.functional.interpolate(
                    views1, size=current_ms_size, mode="nearest",
                )
                views2 = nn.functional.interpolate(
                    views2, size=current_ms_size, mode="nearest",
                )

            # Same-label CutMix (multi-view)
            if _cutmix_active:
                from data.cutmix import same_label_cutmix_multi_view
                views1, views2 = same_label_cutmix_multi_view(
                    views1, views2, labels,
                    p=args.cutmix_p, alpha=args.cutmix_alpha,
                )

            # Log augmented/clean pairs to TensorBoard
            if batch_idx in _img_log_steps:
                global_step = epoch * len(loader) + batch_idx
                _log_training_images(
                    writer, global_step, views1, views2,
                    count=getattr(args, "tb_log_images_pairs", 4),
                )

            with autocast(device_type="cuda", enabled=args.amp):
                logits1, _, proj1 = model(views1, return_embedding=True)
                if ema_teacher is not None:
                    # Teacher-student: clean view through EMA teacher (no grad)
                    with torch.no_grad():
                        logits2, _, proj2 = ema_teacher(views2, return_embedding=True)
                else:
                    logits2, _, proj2 = model(views2, return_embedding=True)
                loss, loss_components = criterion(
                    logits1, logits2, proj1, proj2, labels,
                )

            preds = logits1.argmax(dim=1)
            for k, v in loss_components.items():
                sub_loss_meters[k].update(v, batch_size)
        else:
            images, labels, _metadata = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)

            # GPU-side nearest downscale for multi-scale
            if use_iter_ms and current_ms_size != ms_base_size:
                images = nn.functional.interpolate(
                    images, size=current_ms_size, mode="nearest",
                )

            # Same-label CutMix (single-view)
            if _cutmix_active:
                from data.cutmix import same_label_cutmix
                images = same_label_cutmix(
                    images, labels,
                    p=args.cutmix_p, alpha=args.cutmix_alpha,
                )

            # Log augmented images to TensorBoard
            if batch_idx in _img_log_steps:
                global_step = epoch * len(loader) + batch_idx
                _log_training_images(
                    writer, global_step, images, None,
                    count=getattr(args, "tb_log_images_count", 8),
                )

            with autocast(device_type="cuda", enabled=args.amp):
                if use_moe:
                    from models.moe import EXPERT_GROUP_TO_IDX, NUM_EXPERTS

                    expert_masks = torch.zeros(
                        batch_size, NUM_EXPERTS,
                        device=device, dtype=torch.float32,
                    )
                    for i, meta in enumerate(_metadata):
                        groups = meta.get("aug_groups", frozenset({"clean"}))
                        for g in groups:
                            idx = EXPERT_GROUP_TO_IDX.get(g)
                            if idx is not None:
                                expert_masks[i, idx] = 1.0
                        if not groups or "clean" in groups:
                            expert_masks[i, EXPERT_GROUP_TO_IDX["clean"]] = 1.0

                    logits = model(images, moe_expert_masks=expert_masks)
                    loss = criterion(logits, labels, expert_masks)

                    # Accuracy: aggregate from already-computed expert logits
                    with torch.no_grad():
                        avg_logits = (logits * expert_masks.unsqueeze(-1)).sum(dim=1)
                        avg_logits = avg_logits / expert_masks.sum(dim=1, keepdim=True).clamp(min=1)
                    preds = avg_logits.argmax(dim=1)
                elif use_lora_moe:
                    from models.moe import EXPERT_GROUP_TO_IDX, NUM_EXPERTS

                    expert_masks = torch.zeros(
                        batch_size, NUM_EXPERTS,
                        device=device, dtype=torch.float32,
                    )
                    for i, meta in enumerate(_metadata):
                        groups = meta.get("aug_groups", frozenset({"clean"}))
                        for g in groups:
                            idx = EXPERT_GROUP_TO_IDX.get(g)
                            if idx is not None:
                                expert_masks[i, idx] = 1.0
                        if not groups or "clean" in groups:
                            expert_masks[i, EXPERT_GROUP_TO_IDX["clean"]] = 1.0

                    # LoRA-MoE returns (B, C) — standard CE loss
                    logits = model(images, moe_expert_masks=expert_masks)
                    loss = criterion(logits, labels)
                    preds = logits.argmax(dim=1)
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)
                    preds = logits.argmax(dim=1)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # EMA teacher update
        if ema_teacher is not None:
            ema_update(model, ema_teacher, args.mvc_ema_decay)

        correct = (preds == labels).sum().item()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(correct / batch_size, batch_size)

        postfix = dict(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        if use_iter_ms:
            postfix["res"] = current_ms_size
        pbar.set_postfix(postfix)

    result = {"loss": loss_meter.avg, "accuracy": acc_meter.avg}
    if sub_loss_meters:
        for k, meter in sub_loss_meters.items():
            result[f"loss_{k}"] = meter.avg
    return result


def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args,
) -> dict:
    """Run validation. Returns dict with 'loss', 'accuracy' (and 'auc', 'f1' if sklearn available)."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Val   Epoch {epoch}", leave=False,
                disable=not is_main_process(args))
    with torch.no_grad():
        for images, labels, _metadata in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)

            with autocast(device_type="cuda", enabled=args.amp):
                logits = model(images)
                loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(correct / batch_size, batch_size)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    # In DDP: gather predictions from all ranks for correct global metrics
    local_probs = torch.cat(all_probs)
    local_labels = torch.cat(all_labels)

    if getattr(args, "distributed", False) and dist.is_initialized():
        global_probs, global_labels = gather_predictions(
            local_probs.to(device), local_labels.to(device)
        )
        global_probs = global_probs.cpu()
        global_labels = global_labels.cpu()

        total_correct = (global_probs >= 0.5).long().eq(global_labels).sum().item()
        total_samples = global_labels.size(0)
        results = {
            "loss": loss_meter.avg,
            "accuracy": total_correct / max(total_samples, 1),
        }
    else:
        global_probs = local_probs
        global_labels = local_labels
        results = {"loss": loss_meter.avg, "accuracy": acc_meter.avg}

    try:
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        all_probs_np = global_probs.numpy()
        all_labels_np = global_labels.numpy()
        all_preds_np = (all_probs_np >= 0.5).astype(int)
        results["auc"] = roc_auc_score(all_labels_np, all_probs_np)
        results["f1"] = f1_score(all_labels_np, all_preds_np)
        results["precision"] = precision_score(all_labels_np, all_preds_np)
        results["recall"] = recall_score(all_labels_np, all_preds_np)
    except (ImportError, ValueError):
        pass

    return results


# ═══════════════════════════════════════════════════════════════════
#  Checkpointing
# ═══════════════════════════════════════════════════════════════════


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_val_auc: float,
    args,
    best_val_acc: float = 0.0,
    ema_teacher: nn.Module | None = None,
) -> None:
    """Save training checkpoint to disk. In DDP, call only on rank 0."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    args_dict["tb_log_dir"] = getattr(args, "_tb_log_dir", args_dict.get("tb_log_dir", ""))
    # Unwrap DDP model for portable checkpoints
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_auc": best_val_auc,
        "best_val_acc": best_val_acc,
        "args": args_dict,
    }
    if ema_teacher is not None:
        teacher_to_save = ema_teacher.module if hasattr(ema_teacher, "module") else ema_teacher
        checkpoint["ema_teacher"] = teacher_to_save.state_dict()
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Train GenAI image classifier")
    add_data_args(parser)
    add_model_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    args = merge_config(args)

    # Validate dataset_mode
    if getattr(args, "dataset_mode", "concat") != "concat":
        raise NotImplementedError(
            "train.py currently requires --dataset_mode concat. "
            "Separate mode training is not yet supported."
        )

    # Auto-detect distributed from torchrun environment
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.distributed = True

    # --dp and --distributed are mutually exclusive
    if getattr(args, "dp", False) and args.distributed:
        raise ValueError("--dp and --distributed (torchrun) are mutually exclusive")

    # Setup distributed
    if args.distributed:
        rank, local_rank, world_size = setup_distributed(args)
        args._rank = rank
        args._local_rank = local_rank
        args._world_size = world_size
    else:
        args._rank = 0
        args._local_rank = 0
        args._world_size = 1

    # Reproducibility (rank offset for per-worker augmentation diversity)
    seed_everything(args.seed, rank=args._rank)

    # Device
    if args.distributed:
        device = torch.device(f"cuda:{args._local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_rank0(f"Device: {device}", args)
    if device.type == "cuda":
        print_rank0(f"  GPU: {torch.cuda.get_device_name(device)}", args)
        if args.distributed:
            print_rank0(f"  World size: {args._world_size}", args)
        elif getattr(args, "dp", False):
            print(f"  DataParallel GPUs: {torch.cuda.device_count()}")
    else:
        print_rank0("WARNING: CUDA not available, disabling AMP", args)
        args.amp = False

    # Data
    print_rank0("Building data loaders...", args)
    train_loader, val_loader = build_train_val_loaders(args)
    steps_per_epoch = len(train_loader)
    print_rank0(f"  Steps per epoch: {steps_per_epoch}", args)

    # LoRA implies frozen backbone
    if getattr(args, "lora_enabled", False) and not args.freeze_backbone:
        args.freeze_backbone = True
        print_rank0("  LoRA enabled: auto-freezing backbone", args)

    # ConvLoRA implies frozen backbone
    if getattr(args, "convlora_enabled", False) and not args.freeze_backbone:
        args.freeze_backbone = True
        print_rank0("  ConvLoRA enabled: auto-freezing backbone", args)

    # WSGM implies frozen backbone
    if getattr(args, "wsgm", False) and not args.freeze_backbone:
        args.freeze_backbone = True
        print_rank0("  WSGM enabled: auto-freezing backbone", args)

    # LoRA-MoE implies frozen backbone
    if getattr(args, "lora_moe_enabled", False) and not args.freeze_backbone:
        args.freeze_backbone = True
        print_rank0("  LoRA-MoE enabled: auto-freezing backbone", args)

    # Auto-enable projection head when multi_view is on
    if getattr(args, "multi_view", False):
        if getattr(args, "projection_dim", 0) == 0:
            args.projection_dim = 128
        print_rank0("  Multi-view enabled: projection_dim="
                    f"{args.projection_dim}", args)

    # Model
    print_rank0("Building model...", args)
    model = build_model(args).to(device)
    if args.amp:
        _wrap_mamba_fp32(model)

    # SyncBatchNorm (optional, before DDP wrap)
    if args.distributed and getattr(args, "sync_bn", False):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print_rank0("  Converted BatchNorm -> SyncBatchNorm", args)

    # Wrap model with DDP or DataParallel
    if args.distributed:
        model = DDP(model, device_ids=[args._local_rank], find_unused_parameters=False)
        print_rank0("  Wrapped model with DistributedDataParallel", args)
    elif getattr(args, "dp", False) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"  Wrapped model with DataParallel ({torch.cuda.device_count()} GPUs)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"  Total params: {total_params:,}", args)
    print_rank0(f"  Trainable params: {trainable_params:,}", args)
    if getattr(args, "lora_enabled", False) or getattr(args, "convlora_enabled", False):
        from models.lora import count_lora_params

        raw_model = model.module if hasattr(model, "module") else model
        _, _, lora_params = count_lora_params(raw_model)
        print_rank0(f"  LoRA + ConvLoRA params: {lora_params:,}", args)
    if getattr(args, "wsgm", False):
        from models.wsgm import count_wsgm_params

        raw_model = model.module if hasattr(model, "module") else model
        _, _, wsgm_params = count_wsgm_params(raw_model)
        print_rank0(f"  WSGM adapter params: {wsgm_params:,}", args)
    if getattr(args, "lora_moe_enabled", False):
        from models.lora_moe import count_lora_moe_params

        raw_model = model.module if hasattr(model, "module") else model
        _, _, lora_moe_params = count_lora_moe_params(raw_model)
        print_rank0(f"  LoRA-MoE expert params: {lora_moe_params:,}", args)

    # VRAM pre-check (multi-scale: verify largest resolution fits in VRAM)
    if getattr(args, "multiscale", False) and device.type == "cuda":
        max_size = max(args.multiscale_sizes)
        vram_precheck(model, max_size, args.batch_size, device, args.amp, args)

    # Linear LR scaling (before optimizer build)
    if args.distributed and getattr(args, "scale_lr", False):
        original_lr = args.lr
        args.lr = args.lr * args._world_size
        print_rank0(f"  LR scaled: {original_lr} -> {args.lr} (x{args._world_size})", args)

    # Loss, optimizer, scheduler, scaler
    val_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    from losses import build_criterion
    ce_criterion = build_criterion(args)

    _use_mv = getattr(args, "multi_view", False)
    _use_moe = getattr(args, "moe_enabled", False)

    if _use_mv and _use_moe:
        from losses import MoEMultiViewCriterion
        from models.moe import EXPERT_GROUP_TO_IDX, NUM_EXPERTS

        # ce_criterion is MoECriterion(base_loss) — extract raw base_loss
        base_loss = ce_criterion.base_criterion

        criterion = MoEMultiViewCriterion(
            base_criterion=base_loss,
            num_experts=NUM_EXPERTS,
            clean_expert_idx=EXPERT_GROUP_TO_IDX["clean"],
            lambda_con=args.lambda_con,
            lambda_mvc=args.lambda_mvc,
            temperature=args.con_temperature,
            mvc_ema=getattr(args, "mvc_ema", False),
        )
        print_rank0(
            f"  MoE+Multi-view criterion: lambda_con={args.lambda_con}, "
            f"lambda_mvc={args.lambda_mvc}, temperature={args.con_temperature}"
            f", num_experts={NUM_EXPERTS}"
            f"{', mvc_ema=True' if args.mvc_ema else ''}",
            args,
        )
    elif _use_mv:
        from losses import MultiViewCriterion

        criterion = MultiViewCriterion(
            lambda_con=args.lambda_con,
            lambda_mvc=args.lambda_mvc,
            temperature=args.con_temperature,
            label_smoothing=args.label_smoothing,
            mvc_ema=getattr(args, "mvc_ema", False),
            ce_criterion=ce_criterion,
        )
        print_rank0(
            f"  Multi-view criterion: lambda_con={args.lambda_con}, "
            f"lambda_mvc={args.lambda_mvc}, temperature={args.con_temperature}"
            f"{', mvc_ema=True' if args.mvc_ema else ''}",
            args,
        )
    else:
        criterion = ce_criterion

    if getattr(args, "focal_gamma", 0.0) > 0 or getattr(args, "ohsm_enabled", False):
        print_rank0(
            f"  OHSM: focal_gamma={getattr(args, 'focal_gamma', 0.0)}, "
            f"mining={'on' if getattr(args, 'ohsm_enabled', False) else 'off'}, "
            f"keep_ratio={getattr(args, 'ohsm_keep_ratio', 1.0)}, "
            f"curriculum={getattr(args, 'ohsm_curriculum', False)}",
            args,
        )
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch)
    scaler = GradScaler("cuda", enabled=args.amp)

    # Resume
    start_epoch = 0
    best_val_auc = 0.0
    best_val_acc = 0.0
    ckpt = None
    if args.resume:
        print_rank0(f"Resuming from: {args.resume}", args)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if hasattr(model, "module"):
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt.get("best_val_auc", ckpt.get("best_val_acc", 0.0))
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print_rank0(
            f"  Resumed at epoch {start_epoch}, "
            f"best_val_auc={best_val_auc:.4f}, best_val_acc={best_val_acc:.4f}",
            args,
        )

    # EMA teacher model for teacher-student MVC
    # Build from the unwrapped model to avoid deep-copying DDP/DP internals.
    ema_teacher = None
    if getattr(args, "mvc_ema", False) and getattr(args, "multi_view", False):
        raw_model = model.module if hasattr(model, "module") else model
        ema_teacher = build_ema_teacher(raw_model)
        ema_teacher.eval()
        # Load EMA teacher state from checkpoint if resuming
        if ckpt is not None and "ema_teacher" in ckpt:
            ema_state = ckpt["ema_teacher"]
            target = ema_teacher.module if hasattr(ema_teacher, "module") else ema_teacher
            target.load_state_dict(ema_state)
            print_rank0("  Loaded EMA teacher from checkpoint", args)
        ema_teacher_params = sum(p.numel() for p in ema_teacher.parameters())
        print_rank0(f"  EMA teacher: {ema_teacher_params:,} params, "
                    f"decay={args.mvc_ema_decay}", args)

    # TensorBoard — only rank 0
    writer = None
    if args.resume and ckpt is not None and ckpt.get("args", {}).get("tb_log_dir"):
        tb_log_dir = ckpt["args"]["tb_log_dir"]
        run_name = os.path.basename(tb_log_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.model_name}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
        tb_log_dir = os.path.join(args.log_dir, run_name)
    args._tb_log_dir = tb_log_dir
    # Mirror run_name into save_dir so checkpoints are per-trial
    args.save_dir = os.path.join(args.save_dir, run_name)

    if is_main_process(args):
        writer = SummaryWriter(log_dir=tb_log_dir)
        hparam_str = "\n".join(f"  {k}: {v}" for k, v in sorted(vars(args).items()))
        writer.add_text("hyperparameters", hparam_str, 0)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    if args.resume and best_val_auc > 0:
        early_stopping.best_score = best_val_auc
        early_stopping.best_epoch = start_epoch - 1

    # Training loop
    if is_main_process(args):
        os.makedirs(args.save_dir, exist_ok=True)
    print_rank0(f"\nStarting training for {args.epochs} epochs...", args)
    print_rank0(f"  Checkpoints: {args.save_dir}", args)
    print_rank0(f"  TensorBoard: {os.path.join(args.log_dir, run_name)}", args)
    if getattr(args, "multiscale", False):
        ms_interval = getattr(args, "multiscale_interval", 0)
        if ms_interval > 0:
            print_rank0(
                f"  Multi-scale: iteration-level (every {ms_interval} iters), "
                f"sizes={args.multiscale_sizes}, base={getattr(args, '_multiscale_train_size', args.image_size)}, "
                f"downscale=nearest",
                args,
            )
        else:
            print_rank0(
                f"  Multi-scale: epoch-level (round-robin), sizes={args.multiscale_sizes}",
                args,
            )

    # Curricular augmentation epoch state (None when not using genai_curriculum)
    _epoch_state = getattr(train_loader, "_epoch_state", None)
    _scale_state = getattr(train_loader, "_scale_state", None)

    epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Update curriculum augmentation epoch
        if _epoch_state is not None:
            _epoch_state.value = epoch

        # Multi-scale: per-epoch resolution change (legacy, multiscale_interval=0)
        # When multiscale_interval > 0, resolution is changed per-iteration
        # inside train_one_epoch via GPU-side F.interpolate(nearest).
        if _scale_state is not None:
            scales = args.multiscale_sizes
            current_scale = scales[epoch % len(scales)]
            _scale_state.value = current_scale

            from models.classifier import update_mambavision_window_size
            update_mambavision_window_size(model, current_scale)
            if ema_teacher is not None:
                update_mambavision_window_size(ema_teacher, current_scale)

            print_rank0(
                f"  Multi-scale: epoch {epoch} -> {current_scale}x{current_scale}",
                args,
            )

        # Set epoch on DistributedSampler for proper shuffling
        if args.distributed and hasattr(train_loader, "sampler"):
            sampler = train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        # OHSM curriculum: ramp keep_ratio from 1.0 -> target
        if getattr(args, "ohsm_enabled", False) and getattr(args, "ohsm_curriculum", False):
            _ohsm_target = getattr(args, "ohsm_keep_ratio", 0.7)
            _ohsm_start = getattr(args, "ohsm_curriculum_start_epoch", 0)
            _ohsm_cr = getattr(args, "ohsm_curriculum_ratio", None)
            if _ohsm_cr is None:
                _ohsm_cr = getattr(args, "curriculum_ratio", 0.5)
            _ohsm_end = int(args.epochs * _ohsm_cr)
            if epoch < _ohsm_start:
                _cur_ratio = 1.0
            elif epoch >= _ohsm_end:
                _cur_ratio = _ohsm_target
            else:
                _progress = (epoch - _ohsm_start) / max(_ohsm_end - _ohsm_start, 1)
                _cur_ratio = 1.0 - _progress * (1.0 - _ohsm_target)
            _update_ohsm_ratio(criterion, _cur_ratio)
            print_rank0(f"  OHSM curriculum: keep_ratio={_cur_ratio:.3f}", args)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, args,
            writer=writer,
            ema_teacher=ema_teacher,
        )

        if writer is not None:
            writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
            if _scale_state is not None:
                writer.add_scalar("train/image_size", _scale_state.value, epoch)
            # Multi-view / MoE+Multi-view sub-loss components
            for key in train_metrics:
                if key.startswith("loss_"):
                    writer.add_scalar(f"train/{key}", train_metrics[key], epoch)
            # OHSM keep_ratio tracking
            _ohsm_r = _get_ohsm_ratio(criterion)
            if _ohsm_r is not None:
                writer.add_scalar("train/ohsm_keep_ratio", _ohsm_r, epoch)

        # Validate
        val_metrics = None
        _use_iter_ms = getattr(args, "multiscale", False) and getattr(args, "multiscale_interval", 0) > 0
        _need_window_restore = _scale_state is not None or _use_iter_ms
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            # MambaVision: restore default resolution window_size for validation
            if _need_window_restore:
                from models.classifier import update_mambavision_window_size
                update_mambavision_window_size(model, args.image_size)

            val_metrics = validate(model, val_loader, val_criterion, device, epoch, args)

            # MambaVision: re-apply scale for next training epoch
            # (per-epoch mode only; iteration-level resets at next batch)
            if _scale_state is not None:
                update_mambavision_window_size(model, _scale_state.value)

            if writer is not None:
                writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
                if "auc" in val_metrics:
                    writer.add_scalar("val/auc", val_metrics["auc"], epoch)
                if "f1" in val_metrics:
                    writer.add_scalar("val/f1", val_metrics["f1"], epoch)
                if "precision" in val_metrics:
                    writer.add_scalar("val/precision", val_metrics["precision"], epoch)
                if "recall" in val_metrics:
                    writer.add_scalar("val/recall", val_metrics["recall"], epoch)

            val_auc = val_metrics.get("auc")
            if val_auc is None:
                print_rank0("  WARNING: AUC not available, falling back to accuracy", args)
                val_auc = val_metrics["accuracy"]
            val_acc = val_metrics["accuracy"]

            # Best AUC tracking (drives early stopping)
            auc_improved = early_stopping.step(val_auc, epoch)
            if auc_improved and is_main_process(args):
                best_val_auc = val_auc
                save_checkpoint(
                    os.path.join(args.save_dir, "best_auc.pth"),
                    model, optimizer, scheduler, scaler, epoch,
                    val_auc, args, best_val_acc=best_val_acc,
                    ema_teacher=ema_teacher,
                )

            # Best accuracy tracking (independent)
            if val_acc > best_val_acc and is_main_process(args):
                best_val_acc = val_acc
                save_checkpoint(
                    os.path.join(args.save_dir, "best_acc.pth"),
                    model, optimizer, scheduler, scaler, epoch,
                    best_val_auc, args, best_val_acc=val_acc,
                    ema_teacher=ema_teacher,
                )

        # Periodic checkpoint (rank 0 only)
        if (epoch + 1) % args.save_every == 0 and is_main_process(args):
            save_checkpoint(
                os.path.join(args.save_dir, f"epoch_{epoch}.pth"),
                model, optimizer, scheduler, scaler, epoch,
                early_stopping.best_score, args, best_val_acc=best_val_acc,
                ema_teacher=ema_teacher,
            )

        # Last checkpoint — overwrite every epoch (rank 0 only)
        if is_main_process(args):
            save_checkpoint(
                os.path.join(args.save_dir, "last.pth"),
                model, optimizer, scheduler, scaler, epoch,
                early_stopping.best_score, args, best_val_acc=best_val_acc,
                ema_teacher=ema_teacher,
            )

        if writer is not None:
            writer.flush()

        # Epoch summary (rank 0 only)
        elapsed = time.time() - epoch_start
        if is_main_process(args):
            summary = f"Epoch {epoch}/{args.epochs - 1} ({elapsed:.1f}s)"
            if _scale_state is not None:
                summary += f" [{_scale_state.value}px]"
            summary += f" | train_loss={train_metrics['loss']:.4f}"
            summary += f" train_acc={train_metrics['accuracy']:.4f}"
            if val_metrics:
                summary += f" | val_loss={val_metrics['loss']:.4f}"
                summary += f" val_acc={val_metrics['accuracy']:.4f}"
                if "auc" in val_metrics:
                    summary += f" val_auc={val_metrics['auc']:.4f}"
                if "precision" in val_metrics:
                    summary += f" val_prec={val_metrics['precision']:.4f}"
                if "recall" in val_metrics:
                    summary += f" val_rec={val_metrics['recall']:.4f}"
            summary += f" | lr={optimizer.param_groups[0]['lr']:.2e}"
            if early_stopping.enabled:
                summary += f" | patience={early_stopping.counter}/{early_stopping.patience}"
            print(summary)

        # Early stopping: broadcast decision from rank 0 to all ranks
        if args.distributed:
            stop_flag = torch.tensor(
                [1 if early_stopping.should_stop else 0],
                dtype=torch.int, device=device,
            )
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                print_rank0(f"\nEarly stopping at epoch {epoch}.", args)
                print_rank0(
                    f"  Best val AUC: {early_stopping.best_score:.4f} "
                    f"(epoch {early_stopping.best_epoch})", args,
                )
                print_rank0(f"  Best val accuracy: {best_val_acc:.4f}", args)
                break
        else:
            if early_stopping.should_stop:
                print(f"\nEarly stopping at epoch {epoch}.")
                print(f"  Best val AUC: {early_stopping.best_score:.4f} (epoch {early_stopping.best_epoch})")
                print(f"  Best val accuracy: {best_val_acc:.4f}")
                break

    # Save final checkpoint (rank 0 only)
    if is_main_process(args):
        save_checkpoint(
            os.path.join(args.save_dir, "last.pth"),
            model, optimizer, scheduler, scaler,
            epoch, early_stopping.best_score, args,
            best_val_acc=best_val_acc,
            ema_teacher=ema_teacher,
        )

    if writer is not None:
        writer.close()

    print_rank0(f"\nTraining complete.", args)
    print_rank0(f"  Best val AUC: {early_stopping.best_score:.4f} (epoch {early_stopping.best_epoch})", args)
    print_rank0(f"  Best val accuracy: {best_val_acc:.4f}", args)
    if is_main_process(args):
        print(f"  Checkpoints: {args.save_dir}")
        print(f"  TensorBoard: {os.path.join(args.log_dir, run_name)}")

    # Cleanup distributed
    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
