"""GenAI image detection — training and validation script."""

import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import add_data_args, add_model_args, add_train_args, merge_config
from data import build_train_val_loaders
from models import build_model


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════


def seed_everything(seed: int) -> None:
    """Set seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
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

    pbar = tqdm(loader, desc=f"Val   Epoch {epoch}", leave=False)
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

    results = {"loss": loss_meter.avg, "accuracy": acc_meter.avg}

    try:
        from sklearn.metrics import f1_score, roc_auc_score

        all_probs_np = torch.cat(all_probs).numpy()
        all_labels_np = torch.cat(all_labels).numpy()
        results["auc"] = roc_auc_score(all_labels_np, all_probs_np)
        results["f1"] = f1_score(all_labels_np, (all_probs_np >= 0.5).astype(int))
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
    """Save training checkpoint to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    args_dict["tb_log_dir"] = getattr(args, "_tb_log_dir", args_dict.get("tb_log_dir", ""))
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
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

    # Reproducibility
    seed_everything(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, disabling AMP")
        args.amp = False

    # Data
    print("Building data loaders...")
    train_loader, val_loader = build_train_val_loaders(args)
    steps_per_epoch = len(train_loader)
    print(f"  Steps per epoch: {steps_per_epoch}")

    # Model
    print("Building model...")
    model = build_model(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Loss, optimizer, scheduler, scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch)
    scaler = GradScaler("cuda", enabled=args.amp)

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    # TensorBoard
    if args.resume and ckpt.get("args", {}).get("tb_log_dir"):
        tb_log_dir = ckpt["args"]["tb_log_dir"]
        run_name = os.path.basename(tb_log_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.model_name}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
        tb_log_dir = os.path.join(args.log_dir, run_name)
    args._tb_log_dir = tb_log_dir
    writer = SummaryWriter(log_dir=tb_log_dir)
    hparam_str = "\n".join(f"  {k}: {v}" for k, v in sorted(vars(args).items()))
    writer.add_text("hyperparameters", hparam_str, 0)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    if args.resume and best_val_acc > 0:
        early_stopping.best_score = best_val_acc
        early_stopping.best_epoch = start_epoch - 1

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Checkpoints: {args.save_dir}")
    print(f"  TensorBoard: {os.path.join(args.log_dir, run_name)}")

    epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, args,
        )

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # Validate
        val_metrics = None
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_metrics = validate(model, val_loader, criterion, device, epoch, args)

            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            if "auc" in val_metrics:
                writer.add_scalar("val/auc", val_metrics["auc"], epoch)
            if "f1" in val_metrics:
                writer.add_scalar("val/f1", val_metrics["f1"], epoch)

            improved = early_stopping.step(val_metrics["accuracy"], epoch)
            if improved:
                save_checkpoint(
                    os.path.join(args.save_dir, "best.pth"),
                    model, optimizer, scheduler, scaler, epoch,
                    val_metrics["accuracy"], args,
                )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.save_dir, f"epoch_{epoch}.pth"),
                model, optimizer, scheduler, scaler, epoch,
                early_stopping.best_score, args,
            )

        writer.flush()

        # Epoch summary
        elapsed = time.time() - epoch_start
        summary = f"Epoch {epoch}/{args.epochs - 1} ({elapsed:.1f}s)"
        summary += f" | train_loss={train_metrics['loss']:.4f}"
        summary += f" train_acc={train_metrics['accuracy']:.4f}"
        if val_metrics:
            summary += f" | val_loss={val_metrics['loss']:.4f}"
            summary += f" val_acc={val_metrics['accuracy']:.4f}"
            if "auc" in val_metrics:
                summary += f" val_auc={val_metrics['auc']:.4f}"
        summary += f" | lr={optimizer.param_groups[0]['lr']:.2e}"
        if early_stopping.enabled:
            summary += f" | patience={early_stopping.counter}/{early_stopping.patience}"
        print(summary)

        if early_stopping.should_stop:
            print(f"\nEarly stopping at epoch {epoch}.")
            print(f"  Best val accuracy: {early_stopping.best_score:.4f} (epoch {early_stopping.best_epoch})")
            break

    # Save final checkpoint
    save_checkpoint(
        os.path.join(args.save_dir, "last.pth"),
        model, optimizer, scheduler, scaler,
        epoch, early_stopping.best_score, args,
    )

    writer.close()
    print(f"\nTraining complete.")
    print(f"  Best val accuracy: {early_stopping.best_score:.4f} (epoch {early_stopping.best_epoch})")
    print(f"  Checkpoints: {args.save_dir}")
    print(f"  TensorBoard: {os.path.join(args.log_dir, run_name)}")


if __name__ == "__main__":
    main()
