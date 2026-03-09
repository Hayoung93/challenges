"""Post-training temperature calibration for MoE / LoRA-MoE expert heads.

Optimises a per-expert temperature scalar ``tau_k`` on a held-out
validation set by minimising the negative log-likelihood:

    NLL_k = -mean_i log p_k(y_i | x_i, tau_k)

Each expert is calibrated independently via a bounded 1-D search
(``scipy.optimize.minimize_scalar``).  The calibrated temperatures
are written back into the checkpoint so that inference automatically
uses them.

Usage::

    # Head MoE
    python calibrate_moe.py \\
        --checkpoint_path checkpoints/moe_run/best_auc.pth \\
        --model_name dinov3_convnext_base \\
        --moe_enabled \\
        [--batch_size 64] [--num_workers 8]

    # LoRA-MoE
    python calibrate_moe.py \\
        --checkpoint_path checkpoints/lora_moe_run/best_auc.pth \\
        --model_name dinov3_vitb16 \\
        --lora_moe_enabled \\
        [--batch_size 64] [--num_workers 8]
"""

import argparse
import copy
import logging
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DEFAULTS, add_data_args, add_model_args, merge_config
from data import build_train_val_loaders
from models import build_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _extract_features(model, images):
    """Extract backbone features without the MoE head."""
    raw = model.module if hasattr(model, "module") else model
    return raw._extract_features(images)


def _get_moe_head(model):
    raw = model.module if hasattr(model, "module") else model
    return raw.moe_head


@torch.no_grad()
def collect_logits_and_labels(model, val_loader, device, use_amp=True):
    """Run the model on the validation set and collect per-expert logits.

    Returns:
        all_logits: ``(N, K, C)`` tensor of raw (unscaled) expert logits.
        all_labels: ``(N,)`` tensor of ground-truth labels.
    """
    model.eval()
    moe_head = _get_moe_head(model)

    logits_list = []
    labels_list = []

    for batch in tqdm(val_loader, desc="Collecting logits", leave=False):
        images, labels, _ = batch
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            features = _extract_features(model, images)
            expert_logits = moe_head.forward_all_experts(features)  # (B, K, C)

        logits_list.append(expert_logits.float().cpu())
        labels_list.append(labels)

    return torch.cat(logits_list), torch.cat(labels_list)


@torch.no_grad()
def collect_lora_moe_logits(model, val_loader, device, num_experts=8,
                            use_amp=True):
    """Collect per-expert logits for LoRA-MoE via K forward passes.

    Each expert's LoRA adapters are activated individually to extract
    expert-specific features and logits.

    Returns:
        all_logits: ``(N, K, C)`` tensor of raw expert logits.
        all_labels: ``(N,)`` tensor of ground-truth labels.
    """
    from models.lora_moe import clear_lora_moe_state, set_active_expert

    raw = model.module if hasattr(model, "module") else model
    model.eval()

    logits_list = []
    labels_list = []

    for batch in tqdm(val_loader, desc="Collecting LoRA-MoE logits",
                      leave=False):
        images, labels, _ = batch
        images = images.to(device, non_blocking=True)

        expert_logits = []
        for k in range(num_experts):
            set_active_expert(model, k)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                features = raw._extract_features(images)
                logits_k = raw.backbone.head(features)  # (B, C)
            expert_logits.append(logits_k.float().cpu())
        clear_lora_moe_state(model)

        logits_list.append(torch.stack(expert_logits, dim=1))  # (B, K, C)
        labels_list.append(labels)

    return torch.cat(logits_list), torch.cat(labels_list)


def calibrate_temperatures(all_logits, all_labels):
    """Find optimal temperature per expert via 1-D NLL minimisation.

    Args:
        all_logits: ``(N, K, C)`` raw expert logits.
        all_labels: ``(N,)`` ground-truth labels.

    Returns:
        List of optimal temperature values (one per expert).
    """
    from scipy.optimize import minimize_scalar

    N, K, C = all_logits.shape
    optimal_temps = []

    for k in range(K):
        expert_logits = all_logits[:, k, :]  # (N, C)

        def nll_at_log_temp(log_temp):
            temp = math.exp(log_temp)
            scaled = expert_logits / temp
            log_probs = F.log_softmax(scaled, dim=-1)
            nll = F.nll_loss(log_probs, all_labels)
            return nll.item()

        result = minimize_scalar(
            nll_at_log_temp, bounds=(-2.0, 3.0), method="bounded",
        )
        tau = math.exp(result.x)
        nll = result.fun
        logger.info(
            "Expert %d: tau=%.4f  NLL=%.4f", k, tau, nll,
        )
        optimal_temps.append(tau)

    return optimal_temps


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate MoE expert temperatures on a validation set",
    )
    add_data_args(parser)
    add_model_args(parser)
    parser.add_argument("--resume", type=str, default="",
                        help="Checkpoint to resume from (overrides checkpoint_path)")
    args = parser.parse_args()
    args = merge_config(args)

    use_moe = getattr(args, "moe_enabled", False)
    use_lora_moe = getattr(args, "lora_moe_enabled", False)

    if not use_moe and not use_lora_moe:
        logger.error(
            "--moe_enabled or --lora_moe_enabled must be set for "
            "temperature calibration"
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load checkpoint
    checkpoint_path = args.resume or args.checkpoint_path
    if not checkpoint_path:
        logger.error("Must provide --checkpoint_path or --resume")
        sys.exit(1)

    model = build_model(args).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt)
    # Strip DDP prefix
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Build validation loader only
    # Use a temporary args copy with moe/lora_moe disabled for val transforms
    val_args = copy.deepcopy(args)
    val_args.moe_enabled = False
    val_args.lora_moe_enabled = False
    _, val_loader = build_train_val_loaders(val_args)

    logger.info("Collecting expert logits on %d validation samples...",
                len(val_loader.dataset))

    if use_lora_moe:
        raw = model.module if hasattr(model, "module") else model
        num_experts = raw.lora_moe_num_experts
        all_logits, all_labels = collect_lora_moe_logits(
            model, val_loader, device,
            num_experts=num_experts,
            use_amp=getattr(args, "amp", True),
        )
    else:
        all_logits, all_labels = collect_logits_and_labels(
            model, val_loader, device, use_amp=getattr(args, "amp", True),
        )

    logger.info("Optimising per-expert temperatures...")
    optimal_temps = calibrate_temperatures(all_logits, all_labels)

    # Write calibrated temperatures back into checkpoint
    raw = model.module if hasattr(model, "module") else model
    if use_lora_moe:
        raw.lora_moe_log_temperatures.data = torch.log(
            torch.tensor(optimal_temps, dtype=torch.float32),
        )
    else:
        moe_head = _get_moe_head(model)
        moe_head.log_temperatures.data = torch.log(
            torch.tensor(optimal_temps, dtype=torch.float32),
        )

    # Save updated checkpoint
    output_path = checkpoint_path.replace(".pth", "_calibrated.pth")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt["model"] = model.state_dict()
    else:
        ckpt = model.state_dict()
    torch.save(ckpt, output_path)
    logger.info("Saved calibrated checkpoint to %s", output_path)

    # Summary
    from models.moe import EXPERT_GROUPS
    logger.info("--- Calibrated Temperatures ---")
    for name, tau in zip(EXPERT_GROUPS, optimal_temps):
        logger.info("  %-22s tau=%.4f", name, tau)


if __name__ == "__main__":
    main()
