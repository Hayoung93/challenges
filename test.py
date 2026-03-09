"""GenAI image detection — test/inference script."""

import argparse
import csv
import gc
import os

import torch
from torch.amp import autocast
from tqdm import tqdm

from config import add_data_args, add_model_args, add_test_args, merge_config
from data import build_dataloader, build_test_dataloader
from models import build_model
from tta import tta_forward


# ═══════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def run_inference(model, dataloader, device, use_amp=True, tta_mode="none",
                  image_size=224, multicrop_stride_ratio=0.75,
                  multicrop_max_crops=36, multicrop_flip=True):
    """Run inference on a single DataLoader.

    Args:
        model: Classifier model.
        dataloader: DataLoader yielding ``(images, labels, metadata)``.
        device: ``torch.device`` to run on.
        use_amp: Enable automatic mixed precision.
        tta_mode: TTA strategy name.
        image_size: Model input spatial size (for multi-scale TTA crops).
        multicrop_stride_ratio: Stride fraction for ``"multicrop"`` TTA.
        multicrop_max_crops: Maximum views for ``"multicrop"`` TTA.
        multicrop_flip: Include flips in ``"multicrop"`` TTA.

    Returns:
        List of ``(image_name, predicted_label, score)`` tuples where
        ``score`` is the softmax probability of class 1 (AI-generated).
    """
    model.eval()
    predictions = []

    for images, _labels, metadata in tqdm(dataloader, desc="Inference", leave=False):
        images = images.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            logits = tta_forward(model, images, tta_mode, image_size=image_size,
                                 multicrop_stride_ratio=multicrop_stride_ratio,
                                 multicrop_max_crops=multicrop_max_crops,
                                 multicrop_flip=multicrop_flip)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        for i, meta in enumerate(metadata):
            predictions.append((meta["source_id"], preds[i].item(), probs[i].item()))

        del images, logits, probs, preds, _labels, metadata

    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def generate_csv(predictions, output_path):
    """Write predictions to CSV with ``image_name,label`` format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "label"])
        for row in predictions:
            writer.writerow([row[0], row[1]])

    print(f"Saved {len(predictions)} predictions to {output_path}")


def generate_score_csv(predictions, output_path):
    """Write predictions to CSV with ``image_name,score`` format.

    Args:
        predictions: List of ``(image_name, label, score)`` tuples.
        output_path: Destination file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "score"])
        for row in predictions:
            image_name, _label, score = row
            writer.writerow([image_name, f"{score:.6f}"])

    print(f"Saved {len(predictions)} score predictions to {output_path}")


# ═══════════════════════════════════════════════════════════════════
#  Validation evaluation (labeled data)
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_val(model, dataloader, device, use_amp=True, tta_mode="none",
                 image_size=224, multicrop_stride_ratio=0.75,
                 multicrop_max_crops=36, multicrop_flip=True):
    """Evaluate on labeled data. Returns dict with metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels, _metadata in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            logits = tta_forward(model, images, tta_mode, image_size=image_size,
                                 multicrop_stride_ratio=multicrop_stride_ratio,
                                 multicrop_max_crops=multicrop_max_crops,
                                 multicrop_flip=multicrop_flip)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

        del images, logits, probs, preds, labels, _metadata

    num_samples = len(all_labels)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(num_samples, 1)
    results = {"accuracy": accuracy, "num_samples": num_samples}

    try:
        from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

        results["auc"] = roc_auc_score(all_labels, all_probs)
        results["f1"] = f1_score(all_labels, all_preds)
        results["confusion_matrix"] = confusion_matrix(all_labels, all_preds)
    except ImportError:
        print("WARNING: scikit-learn not installed, skipping AUC/F1/CM. "
              "Install with: pip install scikit-learn")
    except ValueError:
        pass

    return results


def print_metrics(metrics, subset_name="val"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"  Evaluation Results: {subset_name}")
    print(f"{'=' * 50}")
    print(f"  Samples:    {metrics['num_samples']}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    if "auc" in metrics:
        print(f"  AUC:        {metrics['auc']:.4f}")
    if "f1" in metrics:
        print(f"  F1:         {metrics['f1']:.4f}")
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        print(f"\n  Confusion Matrix:")
        print(f"               Pred Real  Pred Fake")
        print(f"  Actual Real  {cm[0][0]:>8d}  {cm[0][1]:>8d}")
        print(f"  Actual Fake  {cm[1][0]:>8d}  {cm[1][1]:>8d}")
    print(f"{'=' * 50}\n")


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint config verification
# ═══════════════════════════════════════════════════════════════════


# Keys that affect model architecture — mismatch likely causes errors or silent bugs.
_CRITICAL_KEYS = [
    ("model_name",   "Model architecture"),
    ("num_classes",  "Number of classes"),
    ("lora_enabled", "LoRA enabled"),
    ("wsgm",         "WSGM enabled"),
]

# Keys that affect inference quality — mismatch may degrade results.
_WARN_KEYS = [
    ("image_size",            "Image size"),
    ("lora_rank",             "LoRA rank"),
    ("lora_alpha",            "LoRA alpha"),
    ("lora_dropout",          "LoRA dropout"),
    ("lora_target_modules",   "LoRA target modules"),
    ("wsgm_reduction_factor", "WSGM reduction factor"),
    ("wsgm_dropout",          "WSGM dropout"),
    ("wsgm_aggregation",      "WSGM aggregation"),
    ("drop_rate",             "Dropout rate"),
    ("small_pad_p",           "Small image augmentation prob"),
]


def _fmt_val(v):
    """Format a value for display, handling lists and bools."""
    if isinstance(v, list):
        return str(v) if v else "(empty)"
    return str(v)


def verify_checkpoint_config(checkpoint_path, current_args):
    """Load checkpoint metadata and warn if current args differ from training args.

    Prints CRITICAL warnings for architecture mismatches (model_name, LoRA,
    WSGM, num_classes) and regular warnings for other settings (image_size,
    LoRA hyperparams, etc.).  Returns the checkpoint training args dict, or
    None if the checkpoint contains no saved args.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args")
    if ckpt_args is None:
        print("NOTE: Checkpoint does not contain saved training args — "
              "skipping config verification.")
        return None

    critical_mismatches = []
    warn_mismatches = []

    for key, label in _CRITICAL_KEYS:
        ckpt_val = ckpt_args.get(key)
        cur_val = getattr(current_args, key, None)
        if ckpt_val is not None and cur_val is not None and ckpt_val != cur_val:
            critical_mismatches.append((label, key, ckpt_val, cur_val))

    for key, label in _WARN_KEYS:
        ckpt_val = ckpt_args.get(key)
        cur_val = getattr(current_args, key, None)
        if ckpt_val is not None and cur_val is not None and ckpt_val != cur_val:
            warn_mismatches.append((label, key, ckpt_val, cur_val))

    if critical_mismatches or warn_mismatches:
        print(f"\n{'!' * 60}")
        print("  Checkpoint config mismatch detected!")
        print(f"{'!' * 60}")

        if critical_mismatches:
            print("\n  CRITICAL (architecture mismatch — may cause errors):")
            for label, key, ckpt_val, cur_val in critical_mismatches:
                print(f"    {label} ({key}):")
                print(f"      checkpoint = {_fmt_val(ckpt_val)}")
                print(f"      current    = {_fmt_val(cur_val)}")

        if warn_mismatches:
            print("\n  WARNING (may affect inference quality):")
            for label, key, ckpt_val, cur_val in warn_mismatches:
                print(f"    {label} ({key}):")
                print(f"      checkpoint = {_fmt_val(ckpt_val)}")
                print(f"      current    = {_fmt_val(cur_val)}")

        print(f"\n{'!' * 60}\n")

        answer = input("Continue with mismatched config? [y/N]: ").strip().lower()
        if answer != "y":
            raise SystemExit("Aborted by user due to checkpoint config mismatch.")

    return ckpt_args


def check_inference_consistency(ckpt_args, current_args):
    """Warn about logical mismatches between training config and inference settings.

    Unlike :func:`verify_checkpoint_config` (which compares identical keys),
    this checks *cross-key* consistency — e.g. whether the TTA mode is
    appropriate given the training augmentation strategy.

    Two scenarios trigger a blocking confirmation prompt:

    1. Model trained with ``small_pad_p > 0`` but ``--tta none``:
       Small test images will be bilinear-resized instead of reflect-padded,
       which contradicts what the model learned during training.

    2. TTA mode that reflect-pads small images, but model trained with
       ``small_pad_p == 0``: The model never saw reflect-padded patterns,
       so small-image predictions may be unreliable.

    Args:
        ckpt_args: Training args dict from checkpoint, or ``None``.
        current_args: Current inference ``argparse.Namespace``.
    """
    if ckpt_args is None:
        return

    tta_mode = getattr(current_args, "tta", "none")
    ckpt_small_pad_p = ckpt_args.get("small_pad_p", 0.0)

    issues = []

    # Case 1: trained with small-image augmentation but TTA won't reflect-pad
    if ckpt_small_pad_p > 0 and tta_mode == "none":
        issues.append(
            f"Model was trained with small image augmentation "
            f"(small_pad_p={ckpt_small_pad_p}), but --tta is 'none'.\n"
            f"      Small test images will be bilinear-resized instead of "
            f"reflect-padded,\n"
            f"      which mismatches the training distribution.\n"
            f"      Recommendation: use --tta multicrop (or --tta flip at minimum)."
        )

    # Case 2: TTA will reflect-pad, but model never saw padded images
    if tta_mode != "none" and ckpt_small_pad_p == 0:
        issues.append(
            f"Using --tta '{tta_mode}' which reflect-pads small images, "
            f"but the model\n"
            f"      was trained without small image augmentation "
            f"(small_pad_p=0).\n"
            f"      The model may not classify reflect-padded small images "
            f"accurately.\n"
            f"      Recommendation: retrain with --small_pad_p 0.1 "
            f"(or higher)."
        )

    if not issues:
        return

    print(f"\n{'!' * 60}")
    print("  Inference consistency warning")
    print(f"{'!' * 60}")
    for issue in issues:
        print(f"\n  >> {issue}")
    print(f"\n{'!' * 60}\n")

    answer = input("Continue with current settings? [y/N]: ").strip().lower()
    if answer != "y":
        raise SystemExit("Aborted by user due to inference consistency warning.")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════


def get_log_name(checkpoint_path):
    """Extract a log/run name from a checkpoint path.

    If the checkpoint lives in a per-run subdirectory (e.g.
    ``checkpoints/run_name/best.pth``), returns the subdirectory name.
    Otherwise falls back to the checkpoint filename stem (e.g. ``best``).
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path)))
    if parent and parent != "checkpoints":
        return parent
    return os.path.splitext(os.path.basename(checkpoint_path))[0]


def main():
    parser = argparse.ArgumentParser(description="GenAI Image Detection - Inference")
    add_data_args(parser)
    add_model_args(parser)
    add_test_args(parser)
    args = parser.parse_args()
    args = merge_config(args)

    if not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required for inference")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        args.amp = False

    # Resolve output directory: {output_dir}/{log_name}/
    log_name = get_log_name(args.checkpoint_path)
    args.output_dir = os.path.join(args.output_dir, log_name)

    # Verify checkpoint config against current args
    ckpt_args = verify_checkpoint_config(args.checkpoint_path, args)
    check_inference_consistency(ckpt_args, args)

    # Model
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Device: {device}")
    print(f"AMP: {args.amp}, TTA: {args.tta}")
    if args.tta not in ("none",):
        min_ps = getattr(args, "tta_min_prep_size", 512)
        print(f"TTA mode: {args.tta} (min prep size: {min_ps})")
    print(f"Output: {args.output_dir}")

    model = build_model(args).to(device)
    model.eval()

    # Test inference
    print(f"\nTest mode: {args.ntire_test_mode}")
    print("Building test dataloader...")
    test_loader = build_test_dataloader(args)

    os.makedirs(args.output_dir, exist_ok=True)

    mc_kwargs = {
        "multicrop_stride_ratio": getattr(args, "multicrop_stride_ratio", 0.75),
        "multicrop_max_crops": getattr(args, "multicrop_max_crops", 36),
        "multicrop_flip": getattr(args, "multicrop_flip", True),
    }

    if isinstance(test_loader, dict):
        # Mode 3: dict of DataLoaders
        all_predictions = []
        for subset_name, loader in test_loader.items():
            print(f"\nRunning inference on {subset_name}...")
            preds = run_inference(model, loader, device, args.amp,
                                  tta_mode=args.tta, image_size=args.image_size,
                                  **mc_kwargs)
            csv_path = os.path.join(args.output_dir, f"predictions_{subset_name}.csv")
            generate_csv(preds, csv_path)
            if args.output_scores:
                score_path = os.path.join(args.output_dir, f"scores_{subset_name}.csv")
                generate_score_csv(preds, score_path)
            all_predictions.extend(preds)

        combined_csv = os.path.join(args.output_dir, "predictions_all.csv")
        generate_csv(all_predictions, combined_csv)
        if args.output_scores:
            combined_score_csv = os.path.join(args.output_dir, "scores_all.csv")
            generate_score_csv(all_predictions, combined_score_csv)
    else:
        # Mode 1 or 2: single DataLoader
        print("\nRunning inference...")
        preds = run_inference(model, test_loader, device, args.amp,
                              tta_mode=args.tta, image_size=args.image_size,
                              **mc_kwargs)
        mode_names = {1: "val_images", 2: "val_images_hard"}
        subset_name = mode_names.get(args.ntire_test_mode, "test")
        csv_path = os.path.join(args.output_dir, f"predictions_{subset_name}.csv")
        generate_csv(preds, csv_path)
        if args.output_scores:
            score_path = os.path.join(args.output_dir, f"scores_{subset_name}.csv")
            generate_score_csv(preds, score_path)

    # Optional: evaluate on labeled val data
    if args.eval_val:
        print("\nBuilding validation dataloader...")
        val_loader = build_dataloader(args, split="val")

        if isinstance(val_loader, dict):
            for subset_name, loader in val_loader.items():
                metrics = evaluate_val(model, loader, device, args.amp,
                                       tta_mode=args.tta, image_size=args.image_size,
                                       **mc_kwargs)
                print_metrics(metrics, subset_name)
        else:
            metrics = evaluate_val(model, val_loader, device, args.amp,
                                   tta_mode=args.tta, image_size=args.image_size,
                                   **mc_kwargs)
            print_metrics(metrics, "val")

    print("Done.")


if __name__ == "__main__":
    main()
