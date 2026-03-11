"""GenAI image detection — ensemble inference with TTA.

Loads multiple model checkpoints (potentially different architectures),
runs TTA-augmented inference on each, and aggregates predictions via a
configurable ensemble strategy for higher accuracy.

Usage example (3 models, each with its own TTA):

    python ensemble_test.py \
        --ensemble_checkpoints \
            checkpoints/mamba_T/best.pth \
            checkpoints/dinov3_vits/best.pth \
            checkpoints/dinov3_convnext/best.pth \
        --ensemble_models \
            mamba_vision_T \
            dinov3_vits16plus \
            dinov3_convnext_tiny \
        --ensemble_tta flip full multiscale \
        --ensemble_method mean_prob \
        --ensemble_weights 1.0 1.2 1.0 \
        --output_scores \
        --ntire_test_mode 3
"""

import argparse
import csv
import gc
import os
from collections import defaultdict

import torch
from torch.amp import autocast
from tqdm import tqdm

from config import (
    add_data_args,
    add_ensemble_args,
    add_model_args,
    add_test_args,
    merge_config,
)
from data import build_dataloader, build_test_dataloader
from models import build_model
from tta import tta_forward


# ═══════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════


def load_ensemble_models(args, device):
    """Load all ensemble member models onto *device*.

    Returns:
        List of ``(model, tta_mode, weight)`` tuples.
    """
    checkpoints = args.ensemble_checkpoints
    model_names = args.ensemble_models
    weights = args.ensemble_weights
    tta_modes = args.ensemble_tta
    n = len(checkpoints)

    if len(model_names) != n:
        raise ValueError(
            f"--ensemble_checkpoints ({n}) and --ensemble_models "
            f"({len(model_names)}) must have the same length"
        )

    # Default weights: equal
    if not weights:
        weights = [1.0] * n
    elif len(weights) != n:
        raise ValueError(
            f"--ensemble_weights ({len(weights)}) must match "
            f"--ensemble_checkpoints ({n})"
        )

    # Default TTA: use the global --tta for all models
    if not tta_modes:
        tta_modes = [args.tta] * n
    elif len(tta_modes) != n:
        raise ValueError(
            f"--ensemble_tta ({len(tta_modes)}) must match "
            f"--ensemble_checkpoints ({n})"
        )

    models = []
    for i, (ckpt, mname, w, tta) in enumerate(
        zip(checkpoints, model_names, weights, tta_modes)
    ):
        print(f"\n[Ensemble {i+1}/{n}] Loading {mname} from {ckpt} (TTA={tta}, w={w})")
        member_args = argparse.Namespace(
            model_name=mname,
            pretrained=False,
            num_classes=args.num_classes,
            freeze_backbone=False,
            drop_rate=0.0,
            image_size=args.image_size,
            checkpoint_path=ckpt,
            dinov3_weights_dir=args.dinov3_weights_dir,
        )
        model = build_model(member_args).to(device)
        model.eval()
        models.append((model, tta, w))

    return models


# ═══════════════════════════════════════════════════════════════════
#  Ensemble inference
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def run_ensemble_inference(
    models,
    dataloader,
    device,
    use_amp=True,
    image_size=224,
    method="mean_prob",
    multicrop_stride_ratio=0.75,
    multicrop_max_crops=36,
    multicrop_flip=True,
):
    """Run ensemble inference with per-model TTA.

    Args:
        models: List of ``(model, tta_mode, weight)`` from :func:`load_ensemble_models`.
        dataloader: DataLoader yielding ``(images, labels, metadata)``.
        device: Target device.
        use_amp: Enable automatic mixed precision.
        image_size: Model input spatial size (for TTA multi-scale crops).
        method: Aggregation strategy —
            ``"mean_prob"``: weighted average of softmax probabilities,
            ``"mean_logit"``: weighted average of raw logits then softmax,
            ``"majority_vote"``: weighted majority vote on argmax labels.

    Returns:
        List of ``(image_name, predicted_label, score)`` tuples.
    """
    # Accumulate per-image results across batches
    image_names = []
    all_logits = []      # shape will be (n_models, n_images, n_classes)
    all_weights = [w for _, _, w in models]

    for images, _labels, metadata in tqdm(dataloader, desc="Ensemble inference", leave=False):
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)

        batch_logits = []  # (n_models, batch_size, n_classes)
        for model, tta_mode, _w in models:
            with autocast(device_type="cuda", enabled=use_amp):
                logits = tta_forward(model, images, tta_mode, image_size=image_size,
                                     multicrop_stride_ratio=multicrop_stride_ratio,
                                     multicrop_max_crops=multicrop_max_crops,
                                     multicrop_flip=multicrop_flip)
            batch_logits.append(logits)

        # Stack: (n_models, B, C)
        stacked = torch.stack(batch_logits, dim=0)
        all_logits.append(stacked.cpu())

        for meta in metadata:
            image_names.append(meta["source_id"])

        del images, batch_logits, stacked, _labels, metadata

    gc.collect()
    torch.cuda.empty_cache()

    if not all_logits:
        return []

    # Concatenate all batches: (n_models, N, C)
    all_logits = torch.cat(all_logits, dim=1)
    weights_tensor = torch.tensor(all_weights, dtype=torch.float32)
    weights_tensor = weights_tensor / weights_tensor.sum()  # normalize

    predictions = _aggregate(all_logits, weights_tensor, method)

    results = []
    for i, name in enumerate(image_names):
        label = predictions["labels"][i]
        score = predictions["scores"][i]
        results.append((name, label, score))

    return results


def _aggregate(all_logits, weights, method):
    """Aggregate logits from multiple models.

    Args:
        all_logits: ``(n_models, N, C)`` tensor.
        weights: ``(n_models,)`` normalized weight tensor.
        method: ``"mean_prob"``, ``"mean_logit"``, or ``"majority_vote"``.

    Returns:
        Dict with ``"labels"`` (list of int) and ``"scores"`` (list of float).
    """
    n_models, n_images, n_classes = all_logits.shape

    if method == "mean_prob":
        # Softmax per model, then weighted average
        probs = torch.softmax(all_logits, dim=2)  # (M, N, C)
        # Weighted average: (N, C)
        weighted = torch.einsum("m,mnc->nc", weights, probs)
        scores = weighted[:, 1].tolist()
        labels = weighted.argmax(dim=1).tolist()

    elif method == "mean_logit":
        # Weighted average of raw logits, then softmax
        weighted = torch.einsum("m,mnc->nc", weights, all_logits)
        probs = torch.softmax(weighted, dim=1)
        scores = probs[:, 1].tolist()
        labels = weighted.argmax(dim=1).tolist()

    elif method == "majority_vote":
        # Weighted majority vote
        preds = all_logits.argmax(dim=2)  # (M, N)
        probs = torch.softmax(all_logits, dim=2)  # (M, N, C)

        labels = []
        scores = []
        for i in range(n_images):
            vote_weight = {0: 0.0, 1: 0.0}
            for m in range(n_models):
                vote_weight[preds[m, i].item()] += weights[m].item()
            label = 1 if vote_weight[1] > vote_weight[0] else 0
            labels.append(label)
            # Score: weighted average of class-1 probs
            score = sum(
                weights[m].item() * probs[m, i, 1].item() for m in range(n_models)
            )
            scores.append(score)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return {"labels": labels, "scores": scores}


# ═══════════════════════════════════════════════════════════════════
#  CSV output
# ═══════════════════════════════════════════════════════════════════


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
    """Write predictions to CSV with ``image_name,score`` format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "score"])
        for row in predictions:
            writer.writerow([row[0], f"{row[2]:.6f}"])
    print(f"Saved {len(predictions)} score predictions to {output_path}")


# ═══════════════════════════════════════════════════════════════════
#  Validation evaluation
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_ensemble_val(
    models, dataloader, device, use_amp=True, image_size=224, method="mean_prob",
    multicrop_stride_ratio=0.75, multicrop_max_crops=36, multicrop_flip=True,
):
    """Evaluate ensemble on labeled data. Returns dict with metrics."""
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels, _metadata in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)

        batch_logits = []
        for model, tta_mode, _w in models:
            with autocast(device_type="cuda", enabled=use_amp):
                logits = tta_forward(model, images, tta_mode, image_size=image_size,
                                     multicrop_stride_ratio=multicrop_stride_ratio,
                                     multicrop_max_crops=multicrop_max_crops,
                                     multicrop_flip=multicrop_flip)
            batch_logits.append(logits)

        stacked = torch.stack(batch_logits, dim=0)  # (M, B, C)
        weights_tensor = torch.tensor(
            [w for _, _, w in models], dtype=torch.float32
        )
        weights_tensor = weights_tensor / weights_tensor.sum()

        result = _aggregate(stacked.cpu(), weights_tensor, method)

        all_preds.extend(result["labels"])
        all_labels.extend(labels.tolist())
        all_probs.extend(result["scores"])

        del images, batch_logits, stacked, weights_tensor, result, labels, _metadata

    gc.collect()
    torch.cuda.empty_cache()

    num_samples = len(all_labels)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(num_samples, 1)
    results = {"accuracy": accuracy, "num_samples": num_samples}

    try:
        from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

        results["auc"] = roc_auc_score(all_labels, all_probs)
        results["f1"] = f1_score(all_labels, all_preds)
        results["confusion_matrix"] = confusion_matrix(all_labels, all_preds)
    except ImportError:
        print(
            "WARNING: scikit-learn not installed, skipping AUC/F1/CM. "
            "Install with: pip install scikit-learn"
        )
    except ValueError:
        pass

    return results


def print_metrics(metrics, subset_name="val"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"  Ensemble Evaluation: {subset_name}")
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
#  Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="GenAI Image Detection - Ensemble Inference with TTA"
    )
    add_data_args(parser)
    add_model_args(parser)
    add_test_args(parser)
    add_ensemble_args(parser)
    args = parser.parse_args()
    args = merge_config(args)

    if not args.ensemble_checkpoints:
        raise ValueError(
            "--ensemble_checkpoints is required. Provide checkpoint paths "
            "for each ensemble member."
        )
    if not args.ensemble_models:
        raise ValueError(
            "--ensemble_models is required. Provide model names "
            "for each ensemble member."
        )

    n = len(args.ensemble_checkpoints)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        args.amp = False

    # Output directory
    args.output_dir = os.path.join(args.output_dir, "ensemble")

    # Summary
    print("=" * 60)
    print("  Ensemble Inference Configuration")
    print("=" * 60)
    print(f"  Models:  {n}")
    for i in range(n):
        tta_i = (
            args.ensemble_tta[i] if args.ensemble_tta else args.tta
        )
        w_i = args.ensemble_weights[i] if args.ensemble_weights else 1.0
        print(
            f"    [{i+1}] {args.ensemble_models[i]} "
            f"| TTA={tta_i} | w={w_i}"
        )
    print(f"  Method:  {args.ensemble_method}")
    print(f"  Device:  {device}")
    print(f"  AMP:     {args.amp}")
    print(f"  Output:  {args.output_dir}")
    print("=" * 60)

    # Load all models
    models = load_ensemble_models(args, device)

    # Build test dataloader
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
            print(f"\nRunning ensemble inference on {subset_name}...")
            preds = run_ensemble_inference(
                models, loader, device, args.amp,
                image_size=args.image_size, method=args.ensemble_method,
                **mc_kwargs,
            )
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
        print("\nRunning ensemble inference...")
        preds = run_ensemble_inference(
            models, test_loader, device, args.amp,
            image_size=args.image_size, method=args.ensemble_method,
            **mc_kwargs,
        )
        mode_names = {1: "val_images", 2: "val_images_hard", 4: "public_test"}
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
                metrics = evaluate_ensemble_val(
                    models, loader, device, args.amp,
                    image_size=args.image_size, method=args.ensemble_method,
                    **mc_kwargs,
                )
                print_metrics(metrics, subset_name)
        else:
            metrics = evaluate_ensemble_val(
                models, val_loader, device, args.amp,
                image_size=args.image_size, method=args.ensemble_method,
                **mc_kwargs,
            )
            print_metrics(metrics, "val")

    print("Done.")


if __name__ == "__main__":
    main()
