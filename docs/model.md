# Model 사용법

## 개요

`GenAIClassifier`는 MambaVision backbone 위에 binary classification head를 얹은 모델이다. ImageNet pretrained backbone을 로드한 뒤 classification head만 교체하여 GenAI 이미지 탐지(real=0, fake=1)에 사용한다.

## 기본 사용법

```python
import argparse
from config import add_model_args, merge_config
from models import build_model

parser = argparse.ArgumentParser()
add_model_args(parser)
args = parser.parse_args()
args = merge_config(args)

model = build_model(args).cuda()
model.eval()

# images: (B, 3, 224, 224) — DataLoader 출력과 동일
logits = model(images)  # (B, 2) — [real_score, fake_score]
preds = logits.argmax(dim=1)  # 0=real, 1=fake
```

## CLI 옵션

```bash
# 모델 선택
python train.py --model_name mamba_vision_T       # 기본값, 가장 작은 variant
python train.py --model_name mamba_vision_B       # Base

# Pretrained weights (ImageNet-1K)
python train.py --pretrained                      # 다운로드 후 로드
python train.py --no_pretrained                   # random init

# Backbone freeze (head만 학습)
python train.py --pretrained --freeze_backbone
python train.py --no_freeze_backbone              # 전체 fine-tune (기본값)

# Dropout
python train.py --drop_rate 0.1

# 학습된 checkpoint 로드
python train.py --checkpoint_path ./checkpoints/best.pth
```

## 모델 아키텍처

```
GenAIClassifier
└── backbone (MambaVision)
    ├── patch_embed          # Conv stem: 3 → in_dim → dim, stride 4x
    ├── levels[0]            # Conv blocks (stage 1)
    ├── levels[1]            # Conv blocks (stage 2)
    ├── levels[2]            # Mamba + Transformer blocks (stage 3)
    ├── levels[3]            # Mamba + Transformer blocks (stage 4)
    ├── norm                 # BatchNorm2d
    ├── avgpool              # AdaptiveAvgPool2d(1)
    └── head                 # Linear(num_features → num_classes) ← 교체됨
```

MambaVision은 초기 stage에서 CNN을 사용하고, 후기 stage에서 Mamba SSM과 Transformer attention을 혼합하여 사용하는 hybrid 아키텍처이다.

## 사용 가능한 모델

| 모델명 | Params | Feature Dim | Input Size | Pretrained Data |
|--------|--------|-------------|------------|-----------------|
| `mamba_vision_T` | ~26.8M | 640 | 224 | ImageNet-1K |
| `mamba_vision_T2` | ~35.1M | 640 | 224 | ImageNet-1K |
| `mamba_vision_S` | ~50.1M | 768 | 224 | ImageNet-1K |
| `mamba_vision_B` | ~97.7M | 1024 | 224 | ImageNet-1K |
| `mamba_vision_B_21k` | ~97.7M | 1024 | 224 | ImageNet-21K |
| `mamba_vision_L` | ~227.9M | 1568 | 224 | ImageNet-1K |
| `mamba_vision_L_21k` | ~227.9M | 1568 | 224 | ImageNet-21K |
| `mamba_vision_L2` | ~241.5M | 1568 | 224 | ImageNet-1K |
| `mamba_vision_L2_512_21k` | ~241.5M | 1568 | 512 | ImageNet-21K |
| `mamba_vision_L3_256_21k` | ~688.2M | 2048 | 256 | ImageNet-21K |
| `mamba_vision_L3_512_21k` | ~688.2M | 2048 | 512 | ImageNet-21K |

권장: 개발/디버깅에는 `mamba_vision_T`, 실험에는 `mamba_vision_B` 또는 `mamba_vision_L`.

## Head 교체 방식

```
1. create_model(name, pretrained=True)  → backbone 생성 (num_classes=1000)
2. ImageNet pretrained weights 로드     → head 포함 전체 weight 정상 로드
3. backbone.head 교체                   → Linear(num_features, 2) + trunc_normal_ 초기화
```

`num_classes=1000`으로 먼저 생성하는 이유: pretrained weights에 `head.weight` shape이 `(1000, num_features)`로 저장되어 있으므로, 처음부터 `num_classes=2`로 생성하면 size mismatch가 발생한다. 정상 로드 후 head만 교체하는 방식이 안전하다.

## Backbone Freeze

`--freeze_backbone` 사용 시 `head.weight`와 `head.bias`만 학습 가능하다.

```python
model = GenAIClassifier(pretrained=True, freeze_backbone=True)

# 확인
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,}")
# mamba_vision_T: Trainable: 1,282 / 26,817,922
```

Freeze 후 unfreeze하려면:

```python
for param in model.backbone.parameters():
    param.requires_grad = True
```

## Checkpoint 로딩

`GenAIClassifier`는 다양한 checkpoint 형식을 자동으로 처리한다:

| 형식 | 예시 |
|------|------|
| raw state_dict | `torch.save(model.backbone.state_dict(), path)` |
| `{"model": state_dict}` | 일반적인 학습 checkpoint |
| `{"state_dict": state_dict}` | Lightning 스타일 |
| DDP prefix | `module.levels.0.blocks...` → 자동 strip |
| Wrapper prefix | `backbone.levels.0.blocks...` → 자동 strip |

```python
# 저장
torch.save({
    "model": model.backbone.state_dict(),
    "epoch": epoch,
    "best_acc": best_acc,
}, "checkpoint.pth")

# 로드
model = GenAIClassifier(checkpoint_path="checkpoint.pth")
```

## DataLoader와의 연결

```python
from data import build_dataloader
from models import build_model

train_loader = build_dataloader(args, split="train")
model = build_model(args).cuda()

for images, labels, metadata in train_loader:
    images = images.cuda()       # (B, 3, 224, 224)
    labels = labels.cuda()       # (B,) — 0 or 1
    logits = model(images)       # (B, 2)
    loss = F.cross_entropy(logits, labels)
```

## Args 필드 레퍼런스

`build_model(args)` 호출 시 `args`에서 참조하는 필드 목록. 모든 필드는 `getattr(args, field, default)` 패턴으로 접근되므로 누락 시 기본값이 적용된다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `model_name` | `str` | `"mamba_vision_T"` | MambaVision variant 이름 |
| `pretrained` | `bool` | `False` | ImageNet pretrained weights 로드 여부 |
| `num_classes` | `int` | `2` | 출력 클래스 수 (binary: 2) |
| `freeze_backbone` | `bool` | `False` | backbone 파라미터 freeze 여부 |
| `drop_rate` | `float` | `0.0` | MambaVision 내부 dropout rate |
| `checkpoint_path` | `str` | `""` | 학습된 checkpoint 경로 (빈 문자열이면 무시) |

### 최소 호출 예시

```python
from types import SimpleNamespace
from models import build_model

args = SimpleNamespace(model_name="mamba_vision_T", pretrained=True)
model = build_model(args).cuda()
```

## 주의사항

### CUDA 필수

MambaVision의 Mamba SSM 레이어(`selective_scan_cuda`)는 CUDA 커널로 구현되어 있어 **GPU에서만 forward pass가 가능**하다. CPU에서는 모델 생성(weight 로드)은 가능하지만, `model(x)` 호출 시 `RuntimeError`가 발생한다.

### GPU 메모리

| 모델 | Params | FP32 메모리 (추정) | AMP 메모리 (추정) |
|------|--------|-------------------|-------------------|
| `mamba_vision_T` | ~26.8M | ~1.5 GB | ~0.8 GB |
| `mamba_vision_B` | ~97.7M | ~4.5 GB | ~2.5 GB |
| `mamba_vision_L` | ~227.9M | ~10 GB | ~5.5 GB |
| `mamba_vision_L2` | ~241.5M | ~11 GB | ~6 GB |

메모리는 batch_size=32, image_size=224 기준 학습 시 대략적인 추정치이다. `--amp` 사용 시 메모리를 약 40-50% 절약할 수 있다.

### ImageNet 정규화

모든 MambaVision variant는 ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`를 사용한다. DataLoader의 transform이 이미 이 정규화를 적용하므로 추가 처리가 불필요하다.
