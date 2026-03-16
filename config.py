import argparse
import copy

DEFAULTS = {
    # Dataset selection
    "train_datasets": ["dragon", "ntire"],
    "val_datasets": ["ntire"],

    # Dataset paths
    "dragon_root": "/data/data/dragon_dataset_regular",
    "ntire_root": "/data/data/NTIRE2026_GenAI",
    "ntire_shards": None,  # None = auto-discover all shards
    "ntire_test_mode": 1,  # 1=val_images, 2=val_images_hard, 3=both, 4=public_test, 5=all

    # Train/val split
    "val_split_ratio": 0.1,  # fraction of training data for validation
    "train_sampling": False,       # enable training data sub-sampling
    "train_sample_ratio": 0.1,    # fraction of training data per epoch (when enabled)

    # Distorted validation (pre-generated images for robust model selection)
    "distorted_val_dir": "",  # path to distorted val dir; empty = disabled

    # DataLoader
    "batch_size": 32,
    "num_workers": 8,
    "pin_memory": True,
    "drop_last": True,
    "prefetch_factor": 2,

    # Multi-dataset mode: "concat" or "separate"
    "dataset_mode": "concat",

    # Transforms
    "image_size": 224,
    "resize_size": 256,
    "augmentation": "default",  # "none", "default", "strong", "genai", "genai_curriculum", "augly", "augly_curriculum"
    "curriculum_ratio": 0.5,  # fraction of total epochs for curriculum to reach max (0.5 = halfway)
    "curriculum_n_min": 1,        # fixed lower bound for group count sampling
    "curriculum_n_max_start": 1,  # upper bound at epoch 0
    "curriculum_n_max_end": 7,    # upper bound at curriculum completion

    # Robust geometric resize mode
    "robust_resize_mode": "resize_or_crop",  # "resize_or_crop", "resize", or "crop_only"

    # Small-crop pad augmentation (simulates tiny test images)
    "small_pad_p": 0.0,
    "small_pad_mode": "zero",  # "zero", "reflect", or "resize"
    "small_pad_interpolation": "random",  # "random", "bilinear", "lanczos", "nearest"
    "small_crop_range_min": 48,
    "small_crop_range_max": 192,

    # Multi-scale training
    "multiscale": False,
    "multiscale_sizes": [224, 256, 288, 320, 384, 448, 512],
    "multiscale_interval": 100,  # change resolution every N iterations (0 = per-epoch)

    # Dragon-specific
    "dragon_lru_capacity": 4,
    "dragon_index_cache": "/workspace/challenge_genai/.cache/dragon_index.json",

    # Distributed
    "distributed": False,
    "dp": False,  # nn.DataParallel (single-process multi-GPU, no torchrun needed)
    "dist_backend": "nccl",
    "scale_lr": False,
    "sync_bn": False,
    "seed": 42,

    # Model
    "model_name": "mamba_vision_T",
    "pretrained": True,
    "num_classes": 2,
    "freeze_backbone": False,
    "drop_rate": 0.0,
    "checkpoint_path": "",
    "dinov3_weights_dir": "/data/checkpoints/dinov3",

    # LoRA
    "lora_enabled": False,
    "lora_rank": 8,
    "lora_alpha": 8.0,
    "lora_dropout": 0.0,
    "lora_target_modules": [],  # empty = architecture defaults

    # ConvLoRA (depthwise conv adaptation)
    "convlora_enabled": False,
    "convlora_rank": 4,
    "convlora_alpha": 4.0,
    "convlora_dropout": 0.0,

    # WSGM
    "wsgm": False,
    "wsgm_mode": "post",           # "post" (CLS-only extraction) or "inline" (all-token injection)
    "wsgm_reduction_factor": 4,
    "wsgm_dropout": 0.5,
    "wsgm_aggregation": "average",  # "average" or "concat" (post mode only)
    "wsgm_num_modules": 0,         # 0 = auto (n_blocks // 2)
    "wsgm_pooling": "gap",         # "gap" or "attn" (inline mode only)
    "wsgm_attn_heads": 8,          # AttentionPooling num_heads (inline + attn)
    "wsgm_attn_drop": 0.1,         # AttentionPooling dropout (inline + attn)
    "wsgm_use_bfloat16": True,     # cast frozen backbone to bfloat16 (inline mode)

    # Mixture of Experts
    "moe_enabled": False,

    # LoRA-MoE (expert LoRA adapters in backbone)
    "lora_moe_enabled": False,
    "lora_moe_num_experts": 8,
    "lora_moe_rank": 8,
    "lora_moe_alpha": 8.0,
    "lora_moe_dropout": 0.0,
    "lora_moe_convlora_rank": 4,
    "lora_moe_convlora_alpha": 4.0,
    "lora_moe_convlora_dropout": 0.0,

    # Expert diversity loss (LoRA-MoE regularization)
    "lambda_diversity": 0.0,  # 0.0 = disabled; 0.01-0.1 recommended

    # Training hyperparameters
    "lr": 1e-4,
    "weight_decay": 0.05,
    "epochs": 30,
    "warmup_epochs": 5,
    "scheduler": "cosine",           # "cosine" or "step"
    "step_lr_decay": 0.1,
    "step_lr_size": 10,
    "amp": True,
    "grad_clip_norm": 1.0,           # 0.0 = disabled
    "early_stopping_patience": 7,    # 0 = disabled
    "save_dir": "./checkpoints",
    "save_every": 5,
    "log_dir": "./runs",
    "eval_every": 1,
    "label_smoothing": 0.1,
    "resume": "",

    # Same-label CutMix (robust augmentation only)
    "cutmix_p": 0.0,       # 0.0 = disabled by default
    "cutmix_alpha": 0.4,   # Beta distribution alpha

    # Online Hard Sample Mining (OHSM)
    "ohsm_enabled": False,            # Master switch for in-batch hard mining
    "focal_gamma": 0.0,               # Focal loss gamma (0.0 = standard CE)
    "ohsm_keep_ratio": 0.7,           # Fraction of batch to keep (1.0 = no mining)
    "ohsm_min_keep": 4,               # Minimum samples to keep per batch
    "ohsm_curriculum": False,          # Ramp keep_ratio from 1.0 -> ohsm_keep_ratio
    "ohsm_curriculum_ratio": None,     # OHSM schedule length (None = use curriculum_ratio)
    "ohsm_curriculum_start_epoch": 0,  # Epoch to start OHSM curriculum

    # TensorBoard image logging
    "tb_log_images": True,
    "tb_log_images_per_epoch": 5,  # how many times per epoch to log input images
    "tb_log_images_count": 8,     # images per grid (single-view)
    "tb_log_images_pairs": 4,     # pairs per grid (multi-view)

    # Multi-view consistency training
    "multi_view": False,
    "lambda_con": 0.1,        # Supervised contrastive loss weight
    "lambda_mvc": 0.05,       # Multi-view consistency loss weight
    "con_temperature": 0.07,  # SupCon temperature
    "projection_dim": 0,      # Contrastive projection head dimension (0=disabled, auto-set when multi_view)
    "mvc_ema": False,         # Use EMA teacher for MVC loss (teacher-student mode)
    "mvc_ema_decay": 0.999,   # EMA decay rate for teacher model

    # Test / inference
    "output_dir": "./predictions",
    "tta": "none",
    "tta_min_prep_size": 512,  # Minimum prep size; images >= this kept at native resolution
    "multicrop_stride_ratio": 0.75,  # stride as fraction of crop size (0.75 = 25% overlap)
    "multicrop_max_crops": 36,       # maximum total views for multicrop TTA
    "multicrop_flip": True,          # include flipped versions in multicrop
    "eval_val": False,
    "output_scores": False,

    # Ensemble inference
    "ensemble_checkpoints": [],     # list of checkpoint paths
    "ensemble_models": [],          # list of model names (parallel to checkpoints)
    "ensemble_weights": [],         # optional per-model weights (empty = equal)
    "ensemble_method": "mean_prob", # "mean_prob", "mean_logit", or "majority_vote"
    "ensemble_tta": [],             # per-model TTA modes (empty = use --tta for all)
}


def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add all data-related arguments to *parser*."""
    g = parser.add_argument_group("data")
    g.add_argument("--train_datasets", nargs="+", default=DEFAULTS["train_datasets"],
                    help="Dataset names to use for training")
    g.add_argument("--val_datasets", nargs="+", default=DEFAULTS["val_datasets"],
                    help="Dataset names to use for validation")
    g.add_argument("--dragon_root", type=str, default=DEFAULTS["dragon_root"])
    g.add_argument("--ntire_root", type=str, default=DEFAULTS["ntire_root"])
    g.add_argument("--ntire_shards", nargs="+", type=int, default=DEFAULTS["ntire_shards"],
                    help="Shard indices to load (default: all)")
    g.add_argument("--ntire_test_mode", type=int, default=DEFAULTS["ntire_test_mode"],
                    choices=[1, 2, 3, 4, 5],
                    help="Test subset: 1=val_images, 2=val_images_hard, 3=both, "
                         "4=public_test, 5=all (val_images+val_images_hard+public_test)")
    g.add_argument("--val_split_ratio", type=float, default=DEFAULTS["val_split_ratio"],
                    help="Fraction of training data to hold out for validation")
    g.add_argument("--distorted_val_dir", type=str,
                    default=DEFAULTS["distorted_val_dir"],
                    help="Path to pre-generated distorted val images "
                         "(from scripts/generate_distorted_val.py). "
                         "Empty string = disabled (clean val only)")
    g.add_argument("--train_sampling", action="store_true",
                    default=DEFAULTS["train_sampling"],
                    help="Enable training data sub-sampling (use fraction of data per epoch)")
    g.add_argument("--no_train_sampling", dest="train_sampling", action="store_false")
    g.add_argument("--train_sample_ratio", type=float,
                    default=DEFAULTS["train_sample_ratio"],
                    help="Fraction of training data per epoch when --train_sampling "
                         "is on (default: 0.1 = 10%%)")
    g.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    g.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    g.add_argument("--pin_memory", action="store_true", default=DEFAULTS["pin_memory"])
    g.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    g.add_argument("--drop_last", action="store_true", default=DEFAULTS["drop_last"])
    g.add_argument("--no_drop_last", dest="drop_last", action="store_false")
    g.add_argument("--prefetch_factor", type=int, default=DEFAULTS["prefetch_factor"],
                    help="Number of batches loaded in advance by each worker")
    g.add_argument("--dataset_mode", type=str, default=DEFAULTS["dataset_mode"],
                    choices=["concat", "separate"])
    g.add_argument("--image_size", type=int, default=DEFAULTS["image_size"])
    g.add_argument("--resize_size", type=int, default=DEFAULTS["resize_size"])
    g.add_argument("--augmentation", type=str, default=DEFAULTS["augmentation"],
                    choices=["none", "default", "strong", "genai", "genai_curriculum",
                             "augly", "augly_curriculum", "robust", "robust_curriculum",
                             "robust_curriculum_range"])
    g.add_argument("--curriculum_ratio", type=float, default=DEFAULTS["curriculum_ratio"],
                    help="Fraction of total epochs for curriculum to reach max strength "
                         "(0.5 = reach max at halfway, 1.0 = original behavior)")
    g.add_argument("--curriculum_n_min", type=int,
                    default=DEFAULTS["curriculum_n_min"],
                    help="Fixed lower bound for group count sampling "
                         "(used by robust_curriculum_range)")
    g.add_argument("--curriculum_n_max_start", type=int,
                    default=DEFAULTS["curriculum_n_max_start"],
                    help="Upper bound of group count at epoch 0 "
                         "(used by robust_curriculum_range)")
    g.add_argument("--curriculum_n_max_end", type=int,
                    default=DEFAULTS["curriculum_n_max_end"],
                    help="Upper bound of group count at curriculum completion "
                         "(used by robust_curriculum_range)")
    g.add_argument("--robust_resize_mode", type=str,
                    default=DEFAULTS["robust_resize_mode"],
                    choices=["resize_or_crop", "resize", "crop_only"],
                    help="Geometric resize mode for robust augmentation: "
                         "'resize_or_crop' (RandomResizeOrCrop, preserves pixel artifacts), "
                         "'resize' (Resize+CenterCrop, no random cropping), or "
                         "'crop_only' (random crop only, zero-pad if smaller than input size, no resize)")
    g.add_argument("--small_pad_p", type=float,
                    default=DEFAULTS["small_pad_p"],
                    help="Probability of small-crop+reflect-pad augmentation "
                         "during training (0.0 = disabled, 0.1 recommended)")
    g.add_argument("--small_pad_mode", type=str,
                    default=DEFAULTS["small_pad_mode"],
                    choices=["zero", "reflect", "resize"],
                    help="Padding mode for small-crop augmentation: "
                         "'zero' (constant 0 padding), 'reflect' (mirror padding), "
                         "or 'resize' (interpolate small crop to target size)")
    g.add_argument("--small_pad_interpolation", type=str,
                    default=DEFAULTS["small_pad_interpolation"],
                    choices=["random", "bilinear", "bicubic", "lanczos", "nearest"],
                    help="Interpolation for small_pad_mode='resize': "
                         "'random' (randomly pick each time), or a fixed method")
    g.add_argument("--small_crop_range_min", type=int,
                    default=DEFAULTS["small_crop_range_min"],
                    help="Minimum crop size for small-pad augmentation")
    g.add_argument("--small_crop_range_max", type=int,
                    default=DEFAULTS["small_crop_range_max"],
                    help="Maximum crop size for small-pad augmentation")
    g.add_argument("--multiscale", action="store_true",
                    default=DEFAULTS["multiscale"],
                    help="Enable multi-scale training (resolution changes per epoch)")
    g.add_argument("--no_multiscale", dest="multiscale", action="store_false")
    g.add_argument("--multiscale_sizes", nargs="+", type=int,
                    default=DEFAULTS["multiscale_sizes"],
                    help="Pool of input resolutions for multi-scale training "
                         "(all must be divisible by 32)")
    g.add_argument("--multiscale_interval", type=int,
                    default=DEFAULTS["multiscale_interval"],
                    help="Change resolution every N iterations "
                         "(0 = per-epoch round-robin)")
    g.add_argument("--dragon_lru_capacity", type=int, default=DEFAULTS["dragon_lru_capacity"])
    g.add_argument("--dragon_index_cache", type=str, default=DEFAULTS["dragon_index_cache"])
    g.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    g.add_argument("--distributed", action="store_true", default=DEFAULTS["distributed"],
                    help="Enable distributed data parallel training (requires torchrun)")
    g.add_argument("--no_distributed", dest="distributed", action="store_false")
    g.add_argument("--dp", action="store_true", default=DEFAULTS["dp"],
                    help="Use nn.DataParallel for multi-GPU (no torchrun needed)")
    g.add_argument("--no_dp", dest="dp", action="store_false")
    g.add_argument("--dist_backend", type=str, default=DEFAULTS["dist_backend"],
                    choices=["nccl", "gloo"],
                    help="Backend for torch.distributed (nccl for GPU, gloo for CPU)")
    g.add_argument("--scale_lr", action="store_true", default=DEFAULTS["scale_lr"],
                    help="Scale LR linearly by world_size")
    g.add_argument("--no_scale_lr", dest="scale_lr", action="store_false")
    g.add_argument("--sync_bn", action="store_true", default=DEFAULTS["sync_bn"],
                    help="Convert BatchNorm layers to SyncBatchNorm for DDP")
    g.add_argument("--no_sync_bn", dest="sync_bn", action="store_false")
    g.add_argument("--amp", action="store_true", default=DEFAULTS["amp"],
                    help="Enable automatic mixed precision")
    g.add_argument("--no_amp", dest="amp", action="store_false")
    return parser


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add all model-related arguments to *parser*."""
    from models.classifier import VALID_MODELS
    g = parser.add_argument_group("model")
    g.add_argument("--model_name", type=str, default=DEFAULTS["model_name"],
                    choices=VALID_MODELS,
                    help="Backbone model name (MambaVision or DINOv3 variant)")
    g.add_argument("--pretrained", action="store_true", default=DEFAULTS["pretrained"],
                    help="Load ImageNet-pretrained weights")
    g.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    g.add_argument("--num_classes", type=int, default=DEFAULTS["num_classes"],
                    help="Number of output classes (2 for binary)")
    g.add_argument("--freeze_backbone", action="store_true",
                    default=DEFAULTS["freeze_backbone"],
                    help="Freeze backbone parameters (train head only)")
    g.add_argument("--no_freeze_backbone", dest="freeze_backbone",
                    action="store_false")
    g.add_argument("--drop_rate", type=float, default=DEFAULTS["drop_rate"],
                    help="Dropout rate (MambaVision only, ignored for DINOv3)")
    g.add_argument("--checkpoint_path", type=str, default=DEFAULTS["checkpoint_path"],
                    help="Path to a saved model checkpoint")
    g.add_argument("--dinov3_weights_dir", type=str,
                    default=DEFAULTS["dinov3_weights_dir"],
                    help="Directory containing DINOv3 pretrained weight files")
    g.add_argument("--lora_enabled", action="store_true",
                    default=DEFAULTS["lora_enabled"],
                    help="Enable LoRA adapters (DINOv3 only)")
    g.add_argument("--no_lora_enabled", dest="lora_enabled",
                    action="store_false")
    g.add_argument("--lora_rank", type=int, default=DEFAULTS["lora_rank"],
                    help="LoRA rank r (4, 8, 16 recommended)")
    g.add_argument("--lora_alpha", type=float, default=DEFAULTS["lora_alpha"],
                    help="LoRA scaling factor (scaling = alpha / rank)")
    g.add_argument("--lora_dropout", type=float,
                    default=DEFAULTS["lora_dropout"],
                    help="Dropout on LoRA branch")
    g.add_argument("--lora_target_modules", nargs="+",
                    default=DEFAULTS["lora_target_modules"],
                    help="Override LoRA target module suffixes")
    g.add_argument("--convlora_enabled", action="store_true",
                    default=DEFAULTS["convlora_enabled"],
                    help="Enable ConvLoRA adapters on depthwise Conv2d "
                         "(DINOv3 ConvNeXt only)")
    g.add_argument("--no_convlora_enabled", dest="convlora_enabled",
                    action="store_false")
    g.add_argument("--convlora_rank", type=int,
                    default=DEFAULTS["convlora_rank"],
                    help="ConvLoRA rank r (2, 4 recommended)")
    g.add_argument("--convlora_alpha", type=float,
                    default=DEFAULTS["convlora_alpha"],
                    help="ConvLoRA scaling factor (scaling = alpha / rank)")
    g.add_argument("--convlora_dropout", type=float,
                    default=DEFAULTS["convlora_dropout"],
                    help="Dropout on ConvLoRA branch")
    g.add_argument("--wsgm", action="store_true",
                    default=DEFAULTS["wsgm"],
                    help="Enable WSGM adapters (DINOv3 only)")
    g.add_argument("--no_wsgm", dest="wsgm",
                    action="store_false")
    g.add_argument("--wsgm_reduction_factor", type=int,
                    default=DEFAULTS["wsgm_reduction_factor"],
                    help="WSGM bottleneck reduction factor (embed_dim // factor)")
    g.add_argument("--wsgm_dropout", type=float,
                    default=DEFAULTS["wsgm_dropout"],
                    help="Dropout probability in WSGM modules")
    g.add_argument("--wsgm_aggregation", type=str,
                    default=DEFAULTS["wsgm_aggregation"],
                    choices=["average", "concat"],
                    help="WSGM output aggregation mode (post mode only)")
    g.add_argument("--wsgm_mode", type=str,
                    default=DEFAULTS["wsgm_mode"],
                    choices=["post", "inline"],
                    help="WSGM mode: 'post' (CLS extraction) or 'inline' (all-token injection)")
    g.add_argument("--wsgm_num_modules", type=int,
                    default=DEFAULTS["wsgm_num_modules"],
                    help="Number of WSGM modules (0 = auto: n_blocks // 2)")
    g.add_argument("--wsgm_pooling", type=str,
                    default=DEFAULTS["wsgm_pooling"],
                    choices=["gap", "attn"],
                    help="Patch pooling type for inline WSGM: 'gap' or 'attn'")
    g.add_argument("--wsgm_attn_heads", type=int,
                    default=DEFAULTS["wsgm_attn_heads"],
                    help="Number of attention heads for AttentionPooling")
    g.add_argument("--wsgm_attn_drop", type=float,
                    default=DEFAULTS["wsgm_attn_drop"],
                    help="Dropout for AttentionPooling")
    g.add_argument("--wsgm_use_bfloat16", action="store_true",
                    default=DEFAULTS["wsgm_use_bfloat16"],
                    help="Cast frozen backbone to bfloat16 (inline mode)")
    g.add_argument("--no_wsgm_use_bfloat16", dest="wsgm_use_bfloat16",
                    action="store_false")
    g.add_argument("--moe_enabled", action="store_true",
                    default=DEFAULTS["moe_enabled"],
                    help="Enable Mixture of Experts classification heads "
                         "(requires robust augmentation mode)")
    g.add_argument("--no_moe_enabled", dest="moe_enabled",
                    action="store_false")
    g.add_argument("--lora_moe_enabled", action="store_true",
                    default=DEFAULTS["lora_moe_enabled"],
                    help="Enable LoRA-MoE: K expert LoRA adapter sets in "
                         "backbone (DINOv3 only)")
    g.add_argument("--no_lora_moe_enabled", dest="lora_moe_enabled",
                    action="store_false")
    g.add_argument("--lora_moe_num_experts", type=int,
                    default=DEFAULTS["lora_moe_num_experts"],
                    help="Number of LoRA-MoE experts (default 8)")
    g.add_argument("--lora_moe_rank", type=int,
                    default=DEFAULTS["lora_moe_rank"],
                    help="LoRA-MoE rank r (4, 8, 16 recommended)")
    g.add_argument("--lora_moe_alpha", type=float,
                    default=DEFAULTS["lora_moe_alpha"],
                    help="LoRA-MoE scaling factor (scaling = alpha / rank)")
    g.add_argument("--lora_moe_dropout", type=float,
                    default=DEFAULTS["lora_moe_dropout"],
                    help="Dropout on LoRA-MoE branch")
    g.add_argument("--lora_moe_convlora_rank", type=int,
                    default=DEFAULTS["lora_moe_convlora_rank"],
                    help="ConvLoRA-MoE rank for dwconv (ConvNeXt, auto-enabled)")
    g.add_argument("--lora_moe_convlora_alpha", type=float,
                    default=DEFAULTS["lora_moe_convlora_alpha"],
                    help="ConvLoRA-MoE scaling factor")
    g.add_argument("--lora_moe_convlora_dropout", type=float,
                    default=DEFAULTS["lora_moe_convlora_dropout"],
                    help="Dropout on ConvLoRA-MoE branch")
    return parser


def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add all training-related arguments to *parser*."""
    g = parser.add_argument_group("training")
    g.add_argument("--lr", type=float, default=DEFAULTS["lr"],
                    help="Initial learning rate")
    g.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    g.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    g.add_argument("--warmup_epochs", type=int, default=DEFAULTS["warmup_epochs"],
                    help="Linear warmup epochs before main scheduler")
    g.add_argument("--scheduler", type=str, default=DEFAULTS["scheduler"],
                    choices=["cosine", "step"],
                    help="LR scheduler type")
    g.add_argument("--step_lr_decay", type=float, default=DEFAULTS["step_lr_decay"],
                    help="Gamma for StepLR scheduler")
    g.add_argument("--step_lr_size", type=int, default=DEFAULTS["step_lr_size"],
                    help="Step size (epochs) for StepLR scheduler")
    g.add_argument("--grad_clip_norm", type=float, default=DEFAULTS["grad_clip_norm"],
                    help="Max gradient norm for clipping (0=disabled)")
    g.add_argument("--early_stopping_patience", type=int,
                    default=DEFAULTS["early_stopping_patience"],
                    help="Epochs without improvement before stopping (0=disabled)")
    g.add_argument("--save_dir", type=str, default=DEFAULTS["save_dir"],
                    help="Directory for checkpoint saving")
    g.add_argument("--save_every", type=int, default=DEFAULTS["save_every"],
                    help="Save periodic checkpoint every N epochs")
    g.add_argument("--log_dir", type=str, default=DEFAULTS["log_dir"],
                    help="TensorBoard log directory")
    g.add_argument("--eval_every", type=int, default=DEFAULTS["eval_every"],
                    help="Run validation every N epochs")
    g.add_argument("--label_smoothing", type=float, default=DEFAULTS["label_smoothing"],
                    help="Label smoothing factor for CrossEntropyLoss")
    g.add_argument("--resume", type=str, default=DEFAULTS["resume"],
                    help="Path to checkpoint to resume training from")
    # Same-label CutMix
    g.add_argument("--cutmix_p", type=float, default=DEFAULTS["cutmix_p"],
                    help="Probability of applying same-label CutMix per batch "
                         "(only active for robust augmentation modes, 0=disabled)")
    g.add_argument("--cutmix_alpha", type=float, default=DEFAULTS["cutmix_alpha"],
                    help="Beta distribution alpha for CutMix lambda sampling "
                         "(smaller = smaller cuts on average)")
    # Online Hard Sample Mining (OHSM)
    g.add_argument("--ohsm_enabled", action="store_true",
                    default=DEFAULTS["ohsm_enabled"],
                    help="Enable in-batch online hard sample mining (loss-level)")
    g.add_argument("--no_ohsm_enabled", dest="ohsm_enabled",
                    action="store_false")
    g.add_argument("--focal_gamma", type=float,
                    default=DEFAULTS["focal_gamma"],
                    help="Focal loss gamma (0.0=standard CE, 2.0 typical). "
                         "Can be used independently without --ohsm_enabled")
    g.add_argument("--ohsm_keep_ratio", type=float,
                    default=DEFAULTS["ohsm_keep_ratio"],
                    help="Fraction of batch to keep in hard mining "
                         "(1.0=all, 0.5=top 50%%)")
    g.add_argument("--ohsm_min_keep", type=int,
                    default=DEFAULTS["ohsm_min_keep"],
                    help="Minimum samples to keep per batch regardless of ratio")
    g.add_argument("--ohsm_curriculum", action="store_true",
                    default=DEFAULTS["ohsm_curriculum"],
                    help="Ramp keep_ratio from 1.0 to ohsm_keep_ratio over "
                         "ohsm_curriculum_ratio epochs")
    g.add_argument("--no_ohsm_curriculum", dest="ohsm_curriculum",
                    action="store_false")
    g.add_argument("--ohsm_curriculum_ratio", type=float,
                    default=DEFAULTS["ohsm_curriculum_ratio"],
                    help="Fraction of total epochs for OHSM curriculum "
                         "(default: use --curriculum_ratio)")
    g.add_argument("--ohsm_curriculum_start_epoch", type=int,
                    default=DEFAULTS["ohsm_curriculum_start_epoch"],
                    help="Epoch at which OHSM curriculum begins ramping")
    # Expert diversity loss
    g.add_argument("--lambda_diversity", type=float,
                    default=DEFAULTS["lambda_diversity"],
                    help="Weight for expert diversity loss (LoRA-MoE only). "
                         "Penalizes cosine similarity between expert weight "
                         "deltas to prevent collapse. 0.0=disabled, "
                         "0.01-0.1 recommended.")
    # TensorBoard image logging
    g.add_argument("--tb_log_images", action="store_true",
                    default=DEFAULTS["tb_log_images"],
                    help="Log augmented training images to TensorBoard")
    g.add_argument("--no_tb_log_images", dest="tb_log_images",
                    action="store_false")
    g.add_argument("--tb_log_images_per_epoch", type=int,
                    default=DEFAULTS["tb_log_images_per_epoch"],
                    help="Number of times per epoch to log input images")
    g.add_argument("--tb_log_images_count", type=int,
                    default=DEFAULTS["tb_log_images_count"],
                    help="Number of images per grid (single-view)")
    g.add_argument("--tb_log_images_pairs", type=int,
                    default=DEFAULTS["tb_log_images_pairs"],
                    help="Number of augmented/clean pairs per grid (multi-view)")
    # Multi-view consistency
    g.add_argument("--multi_view", action="store_true",
                    default=DEFAULTS["multi_view"],
                    help="Enable multi-view consistency training with "
                         "supervised contrastive loss")
    g.add_argument("--no_multi_view", dest="multi_view",
                    action="store_false")
    g.add_argument("--lambda_con", type=float,
                    default=DEFAULTS["lambda_con"],
                    help="Weight for supervised contrastive loss")
    g.add_argument("--lambda_mvc", type=float,
                    default=DEFAULTS["lambda_mvc"],
                    help="Weight for multi-view consistency loss")
    g.add_argument("--con_temperature", type=float,
                    default=DEFAULTS["con_temperature"],
                    help="Temperature for supervised contrastive loss")
    g.add_argument("--projection_dim", type=int,
                    default=DEFAULTS["projection_dim"],
                    help="Projection head output dimension for contrastive loss")
    g.add_argument("--mvc_ema", action="store_true",
                    default=DEFAULTS["mvc_ema"],
                    help="Use EMA teacher model for MVC loss (teacher-student mode)")
    g.add_argument("--no_mvc_ema", dest="mvc_ema", action="store_false")
    g.add_argument("--mvc_ema_decay", type=float,
                    default=DEFAULTS["mvc_ema_decay"],
                    help="EMA decay rate for teacher model (0.999 typical)")
    # VRAM pre-check control
    g.add_argument("--no_vram_precheck", action="store_true", default=False,
                    help="Skip VRAM pre-check before training "
                         "(useful when the check itself is too heavy)")
    return parser


def add_test_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add test/inference-specific arguments to *parser*."""
    g = parser.add_argument_group("test")
    g.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"],
                    help="Directory for CSV prediction output")
    g.add_argument("--tta", type=str, nargs="?", const="flip",
                    default=DEFAULTS["tta"],
                    choices=["none", "flip", "multiscale", "full",
                             "full_legacy", "multicrop"],
                    help="TTA mode: none, flip, multiscale, full (pixel-preserving), "
                         "full_legacy (rotation-based), or multicrop (grid coverage). "
                         "(default: none; --tta without value means 'flip')")
    g.add_argument("--tta_min_prep_size", type=int,
                    default=DEFAULTS["tta_min_prep_size"],
                    help="Minimum prep tensor size. Images smaller than this are "
                         "reflect-padded; larger images kept at native resolution. "
                         "(default: 512)")
    g.add_argument("--multicrop_stride_ratio", type=float,
                    default=DEFAULTS["multicrop_stride_ratio"],
                    help="Stride as fraction of crop size for multicrop TTA "
                         "(0.75 = 25%% overlap)")
    g.add_argument("--multicrop_max_crops", type=int,
                    default=DEFAULTS["multicrop_max_crops"],
                    help="Maximum total views for multicrop TTA")
    g.add_argument("--multicrop_flip", action="store_true",
                    default=DEFAULTS["multicrop_flip"],
                    help="Include flipped versions in multicrop TTA")
    g.add_argument("--no_multicrop_flip", dest="multicrop_flip",
                    action="store_false")
    g.add_argument("--eval_val", action="store_true", default=DEFAULTS["eval_val"],
                    help="Run evaluation on labeled validation data")
    g.add_argument("--no_eval_val", dest="eval_val", action="store_false")
    g.add_argument("--output_scores", action="store_true",
                    default=DEFAULTS["output_scores"],
                    help="Write a score CSV with softmax probabilities (image_name,score)")
    g.add_argument("--no_output_scores", dest="output_scores", action="store_false")
    return parser


def add_ensemble_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ensemble-specific arguments to *parser*."""
    g = parser.add_argument_group("ensemble")
    g.add_argument("--ensemble_checkpoints", nargs="+", type=str,
                    default=DEFAULTS["ensemble_checkpoints"],
                    help="Checkpoint paths for each ensemble member")
    g.add_argument("--ensemble_models", nargs="+", type=str,
                    default=DEFAULTS["ensemble_models"],
                    help="Model names for each ensemble member "
                         "(must match --ensemble_checkpoints length)")
    g.add_argument("--ensemble_weights", nargs="+", type=float,
                    default=DEFAULTS["ensemble_weights"],
                    help="Per-model weights for weighted averaging "
                         "(default: equal weights)")
    g.add_argument("--ensemble_method", type=str,
                    default=DEFAULTS["ensemble_method"],
                    choices=["mean_prob", "mean_logit", "majority_vote"],
                    help="Ensemble aggregation strategy")
    g.add_argument("--ensemble_tta", nargs="+", type=str,
                    default=DEFAULTS["ensemble_tta"],
                    choices=["none", "flip", "multiscale", "full",
                             "full_legacy", "multicrop"],
                    help="Per-model TTA modes (default: use --tta for all)")
    return parser


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in any attributes missing from *args* with values from DEFAULTS."""
    # Backward compat: old tta_prep_size -> new tta_min_prep_size
    if hasattr(args, "tta_prep_size") and not hasattr(args, "tta_min_prep_size"):
        args.tta_min_prep_size = args.tta_prep_size
    for key, value in DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, copy.deepcopy(value))

    # Validate multiscale_sizes
    if getattr(args, "multiscale", False):
        # Defensive copy to avoid mutating DEFAULTS when argparse reuses
        # the default list object (user did not pass --multiscale_sizes).
        args.multiscale_sizes = list(args.multiscale_sizes)
        for s in args.multiscale_sizes:
            if s % 32 != 0:
                raise ValueError(
                    f"All multiscale_sizes must be divisible by 32, got {s}"
                )
        if args.image_size not in args.multiscale_sizes:
            args.multiscale_sizes.append(args.image_size)
            args.multiscale_sizes.sort()

    # Auto-adjust resize_size so that CenterCrop never zero-pads.
    if args.resize_size < args.image_size:
        args.resize_size = args.image_size

    # Ensure TTA prep tensors are large enough for crop-based TTA views.
    if args.tta_min_prep_size < args.image_size + 128:
        args.tta_min_prep_size = args.image_size + 128

    # MoE requires grouped augmentation (robust variants).
    if getattr(args, "moe_enabled", False):
        aug = getattr(args, "augmentation", "default")
        if aug not in ("robust", "robust_curriculum", "robust_curriculum_range"):
            raise ValueError(
                f"--moe_enabled requires --augmentation robust/robust_curriculum/"
                f"robust_curriculum_range, got '{aug}'"
            )
    # Validate train_sample_ratio bounds.
    ratio = getattr(args, "train_sample_ratio", 0.1)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(
            f"--train_sample_ratio must be in (0.0, 1.0], got {ratio}"
        )

    # LoRA-MoE requires grouped augmentation (robust variants).
    if getattr(args, "lora_moe_enabled", False):
        aug = getattr(args, "augmentation", "default")
        if aug not in ("robust", "robust_curriculum", "robust_curriculum_range"):
            raise ValueError(
                f"--lora_moe_enabled requires --augmentation robust/"
                f"robust_curriculum/robust_curriculum_range, got '{aug}'"
            )

    # Inline WSGM requires DINOv3 ViT (not ConvNeXt).
    if getattr(args, "wsgm", False) and getattr(args, "wsgm_mode", "post") == "inline":
        model = getattr(args, "model_name", "")
        if not model.startswith("dinov3_vit"):
            raise ValueError(
                f"--wsgm_mode inline requires a DINOv3 ViT model "
                f"(dinov3_vit*), got '{model}'"
            )

    return args
