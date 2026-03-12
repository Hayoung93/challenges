"""Evaluate a checkpoint on the distorted validation set (AUC, ACC, F1)."""

import argparse
import sys
import torch
from torch.utils.data import DataLoader

from config import DEFAULTS
from data.ntire import DistortedValDataset
from data.transforms import get_val_transform
from models import build_model
from test import evaluate_val, print_metrics
from tta import tta_forward  # noqa: F401 — needed by evaluate_val


def main():
    parser = argparse.ArgumentParser(description="Evaluate on distorted val set")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--distorted_val_dir", default="/data/data/NTIRE2026_GenAI/distorted_val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=None,
                        help="Override image_size (default: from checkpoint)")
    parser.add_argument("--tta", default="flip", help="TTA mode")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint to get training args
    print(f"Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})

    # Build model args namespace from checkpoint
    model_args = argparse.Namespace(**{**DEFAULTS, **ckpt_args})
    if args.image_size is not None:
        model_args.image_size = args.image_size

    image_size = getattr(model_args, "image_size", 224)
    resize_size = getattr(model_args, "resize_size", 256)

    print(f"Model: {model_args.model_name}, image_size={image_size}")
    print(f"LoRA: {getattr(model_args, 'lora_enabled', False)}, "
          f"ConvLoRA: {getattr(model_args, 'convlora_enabled', False)}, "
          f"MoE: {getattr(model_args, 'moe_enabled', False)}")

    # Build model and load checkpoint weights
    model = build_model(model_args)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()
    print("Checkpoint weights loaded successfully.")

    # Build distorted val dataset
    transform = get_val_transform(image_size=image_size, resize_size=resize_size)
    ds = DistortedValDataset(root=args.distorted_val_dir, transform=transform)
    print(f"Distorted val: {len(ds):,} samples from {args.distorted_val_dir}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Evaluate
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"TTA: {args.tta}, AMP: {use_amp}")

    metrics = evaluate_val(model, loader, device, use_amp=use_amp,
                           tta_mode=args.tta, image_size=image_size)
    ckpt_name = args.checkpoint_path.split("/")[-1]
    print_metrics(metrics, f"distorted_val ({ckpt_name})")


if __name__ == "__main__":
    main()
