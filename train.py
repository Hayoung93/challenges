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
) -> dict:
    """Train for one epoch. Returns dict with 'loss' and 'accuracy'."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False,
                disable=not is_main_process(args))
    for images, labels, _metadata in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        with autocast(device_type="cuda", enabled=args.amp):
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(correct / batch_size, batch_size)

        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


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
    best_val_acc: float,
    args,
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
        "best_val_acc": best_val_acc,
        "args": args_dict,
    }
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

    # Model
    print_rank0("Building model...", args)
    model = build_model(args).to(device)
    if args.amp:
        _wrap_mamba_fp32(model)

    # SyncBatchNorm (optional, before DDP wrap)
    if args.distributed and getattr(args, "sync_bn", False):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print_rank0("  Converted BatchNorm -> SyncBatchNorm", args)

    # Wrap model with DDP
    if args.distributed:
        model = DDP(model, device_ids=[args._local_rank])
        print_rank0("  Wrapped model with DistributedDataParallel", args)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"  Total params: {total_params:,}", args)
    print_rank0(f"  Trainable params: {trainable_params:,}", args)
    if getattr(args, "lora_enabled", False):
        from models.lora import count_lora_params

        raw_model = model.module if hasattr(model, "module") else model
        _, _, lora_params = count_lora_params(raw_model)
        print_rank0(f"  LoRA params: {lora_params:,}", args)

    # Linear LR scaling (before optimizer build)
    if args.distributed and getattr(args, "scale_lr", False):
        original_lr = args.lr
        args.lr = args.lr * args._world_size
        print_rank0(f"  LR scaled: {original_lr} -> {args.lr} (x{args._world_size})", args)

    # Loss, optimizer, scheduler, scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch)
    scaler = GradScaler("cuda", enabled=args.amp)

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    ckpt = None
    if args.resume:
        print_rank0(f"Resuming from: {args.resume}", args)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if args.distributed:
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print_rank0(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}", args)

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
    if args.resume and best_val_acc > 0:
        early_stopping.best_score = best_val_acc
        early_stopping.best_epoch = start_epoch - 1

    # Training loop
    if is_main_process(args):
        os.makedirs(args.save_dir, exist_ok=True)
    print_rank0(f"\nStarting training for {args.epochs} epochs...", args)
    print_rank0(f"  Checkpoints: {args.save_dir}", args)
    print_rank0(f"  TensorBoard: {os.path.join(args.log_dir, run_name)}", args)

    # Curricular augmentation epoch state (None when not using genai_curriculum)
    _epoch_state = getattr(train_loader, "_epoch_state", None)

    epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Update curriculum augmentation epoch
        if _epoch_state is not None:
            _epoch_state.value = epoch

        # Set epoch on DistributedSampler for proper shuffling
        if args.distributed and hasattr(train_loader, "sampler"):
            sampler = train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, args,
        )

        if writer is not None:
            writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # Validate
        val_metrics = None
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_metrics = validate(model, val_loader, criterion, device, epoch, args)

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

            improved = early_stopping.step(val_metrics["accuracy"], epoch)
            if improved and is_main_process(args):
                save_checkpoint(
                    os.path.join(args.save_dir, "best.pth"),
                    model, optimizer, scheduler, scaler, epoch,
                    val_metrics["accuracy"], args,
                )

        # Periodic checkpoint (rank 0 only)
        if (epoch + 1) % args.save_every == 0 and is_main_process(args):
            save_checkpoint(
                os.path.join(args.save_dir, f"epoch_{epoch}.pth"),
                model, optimizer, scheduler, scaler, epoch,
                early_stopping.best_score, args,
            )

        if writer is not None:
            writer.flush()

        # Epoch summary (rank 0 only)
        elapsed = time.time() - epoch_start
        if is_main_process(args):
            summary = f"Epoch {epoch}/{args.epochs - 1} ({elapsed:.1f}s)"
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
                    f"  Best val accuracy: {early_stopping.best_score:.4f} "
                    f"(epoch {early_stopping.best_epoch})", args,
                )
                break
        else:
            if early_stopping.should_stop:
                print(f"\nEarly stopping at epoch {epoch}.")
                print(f"  Best val accuracy: {early_stopping.best_score:.4f} (epoch {early_stopping.best_epoch})")
                break

    # Save final checkpoint (rank 0 only)
    if is_main_process(args):
        save_checkpoint(
            os.path.join(args.save_dir, "last.pth"),
            model, optimizer, scheduler, scaler,
            epoch, early_stopping.best_score, args,
        )

    if writer is not None:
        writer.close()

    print_rank0(f"\nTraining complete.", args)
    print_rank0(f"  Best val accuracy: {early_stopping.best_score:.4f} (epoch {early_stopping.best_epoch})", args)
    if is_main_process(args):
        print(f"  Checkpoints: {args.save_dir}")
        print(f"  TensorBoard: {os.path.join(args.log_dir, run_name)}")

    # Cleanup distributed
    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
