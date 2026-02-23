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
    "ntire_test_mode": 1,  # 1=val_images, 2=val_images_hard, 3=both

    # Train/val split
    "val_split_ratio": 0.1,  # fraction of training data for validation

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
    "augmentation": "default",  # "none", "default", "strong"

    # Dragon-specific
    "dragon_lru_capacity": 4,
    "dragon_index_cache": "/workspace/challenge_genai/.cache/dragon_index.json",

    # Distributed
    "distributed": False,
    "seed": 42,

    # Model
    "model_name": "mamba_vision_T",
    "pretrained": True,
    "num_classes": 2,
    "freeze_backbone": False,
    "drop_rate": 0.0,
    "checkpoint_path": "",

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

    # Test / inference
    "output_dir": "./predictions",
    "tta": False,
    "eval_val": False,
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
                    choices=[1, 2, 3],
                    help="Test subset: 1=val_images, 2=val_images_hard, 3=both")
    g.add_argument("--val_split_ratio", type=float, default=DEFAULTS["val_split_ratio"],
                    help="Fraction of training data to hold out for validation")
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
                    choices=["none", "default", "strong"])
    g.add_argument("--dragon_lru_capacity", type=int, default=DEFAULTS["dragon_lru_capacity"])
    g.add_argument("--dragon_index_cache", type=str, default=DEFAULTS["dragon_index_cache"])
    g.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    g.add_argument("--distributed", action="store_true", default=DEFAULTS["distributed"],
                    help="Enable distributed data parallel training")
    g.add_argument("--no_distributed", dest="distributed", action="store_false")
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
                    help="MambaVision variant name")
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
                    help="Dropout rate for MambaVision layers")
    g.add_argument("--checkpoint_path", type=str, default=DEFAULTS["checkpoint_path"],
                    help="Path to a saved model checkpoint")
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
    return parser


def add_test_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add test/inference-specific arguments to *parser*."""
    g = parser.add_argument_group("test")
    g.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"],
                    help="Directory for CSV prediction output")
    g.add_argument("--tta", action="store_true", default=DEFAULTS["tta"],
                    help="Enable test-time augmentation (horizontal flip)")
    g.add_argument("--no_tta", dest="tta", action="store_false")
    g.add_argument("--eval_val", action="store_true", default=DEFAULTS["eval_val"],
                    help="Run evaluation on labeled validation data")
    g.add_argument("--no_eval_val", dest="eval_val", action="store_false")
    return parser


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in any attributes missing from *args* with values from DEFAULTS."""
    for key, value in DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, copy.deepcopy(value))
    return args
