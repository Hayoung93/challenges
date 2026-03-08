# 학습 (Training) 사용법

## 기본 사용법

```bash
python train.py
```

기본값으로 Dragon + NTIRE 데이터셋을 사용하여 MambaVision-T 모델을 30 epoch 학습한다.

## 학습 흐름

```
1. argparse 파싱 (data + model + training args)
2. seed 고정 (재현성)
3. build_train_val_loaders() → train_loader, val_loader
4. build_model() → GenAIClassifier
5. Epoch 루프:
   ├── train_one_epoch() → loss, accuracy
   ├── validate() (매 eval_every epoch) → loss, accuracy, AUC, F1
   ├── EarlyStopping 체크 → best.pth 저장
   └── 주기적 checkpoint 저장
6. last.pth 저장, TensorBoard writer 종료
```

## CLI 옵션

```bash
# 기본 학습 (기본값 사용)
python train.py

# 모델 선택 및 pretrained 가중치 사용
python train.py --model_name mamba_vision_S --pretrained

# 학습률, 배치 크기, epoch 수 조정
python train.py --lr 5e-5 --batch_size 64 --epochs 50

# 스케줄러 선택
python train.py --scheduler cosine --warmup_epochs 3
python train.py --scheduler step --step_lr_size 10 --step_lr_decay 0.1

# 데이터 augmentation 강도
python train.py --augmentation default             # 기본 (RandomCrop, HFlip, ColorJitter)
python train.py --augmentation strong              # 강한 geometric/color augmentation
python train.py --augmentation genai               # GenAI artifact augmentation (JPEG압축, resize artifact, noise)
python train.py --augmentation genai_curriculum    # GenAI augmentation + 점진적 강도 증가 (curricular)
python train.py --augmentation augly               # AugLy 하이브리드: 11개 pool에서 5개 랜덤 선택·적용 (N-of-K)
python train.py --augmentation augly_curriculum     # AugLy 하이브리드 + curriculum (epoch별 1→5개 점진 증가)
python train.py --augmentation robust              # Robust: 6개 그룹(blur,compression,noise,resize,color,spatial)에서 4개 선택·그룹당 1개 적용
python train.py --augmentation robust_curriculum   # Robust + curriculum (epoch별 2→5개 그룹 점진 증가)
python train.py --augmentation robust_curriculum_range  # Robust curriculum + n_min~n_max 범위 샘플링

# Curriculum 파라미터 조정
python train.py --augmentation genai_curriculum --curriculum_ratio 0.5  # 전체 epoch의 50% 시점에 최대 강도 도달
python train.py --augmentation robust_curriculum_range \
    --curriculum_n_min 2 --curriculum_n_max_start 3 --curriculum_n_max_end 7

# Same-label CutMix (robust augmentation 전용)
python train.py --augmentation robust --cutmix_p 0.5                  # 50% 확률로 CutMix 적용
python train.py --augmentation robust --cutmix_p 0.5 --cutmix_alpha 0.4  # Beta 분포 alpha 조정

# backbone 고정 (head만 학습)
python train.py --freeze_backbone --lr 1e-3 --epochs 10

# AMP 비활성화
python train.py --no_amp

# 특정 데이터셋만 사용
python train.py --train_datasets ntire --ntire_shards 0 1 2

# Checkpoint 저장 위치 및 주기
python train.py --save_dir ./checkpoints/exp01 --save_every 5

# TensorBoard 로그 디렉토리
python train.py --log_dir ./runs/exp01

# 학습 재개
python train.py --resume ./checkpoints/exp01/last.pth
```

## 고해상도 학습 (384, 512 등)

`--image_size`를 변경하면 MambaVision의 `window_size`가 자동 조정된다.

```bash
# 384 해상도로 학습 (window_size=[8,8,24,12] 자동 적용)
python train.py --image_size 384 --resize_size 384 --model_name mamba_vision_T --pretrained

# 512 pretrained 모델을 384에서 fine-tune
python train.py --image_size 384 --resize_size 384 --model_name mamba_vision_L2_512_21k --pretrained
```

224 pretrained 가중치는 384에서 shape 충돌 없이 그대로 로드된다 (position embedding 없음).

## Multi-scale 학습

학습 중 입력 해상도를 동적으로 변경하여 다양한 스케일에서의 robustness를 높인다.

```bash
# 기본 multi-scale (224~512 사이 7단계, 100 iteration마다 변경)
python train.py --multiscale

# 커스텀 해상도 풀 + epoch 단위 변경
python train.py --multiscale --multiscale_sizes 224 384 512 --multiscale_interval 0
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--multiscale` | `False` | Multi-scale 학습 활성화 |
| `--multiscale_sizes` | `[224,256,288,320,384,448,512]` | 해상도 풀 (32의 배수) |
| `--multiscale_interval` | `100` | 해상도 변경 주기 (iterations). `0` = epoch 단위 |

`image_size`가 풀에 없으면 자동 추가된다.

## DINOv3 모델 사용

MambaVision 외에 DINOv3 pretrained backbone(ViT, ConvNeXt)을 사용할 수 있다.

### 사용 가능한 DINOv3 모델

| 모델명 | Backbone | Feature Dim | Pretrained Data |
|--------|----------|-------------|-----------------|
| `dinov3_vits16plus` | ViT-S+/16 | 384 | LVD-1689M |
| `dinov3_convnext_tiny` | ConvNeXt Tiny | 768 | LVD-1689M |

### CLI 사용법

```bash
# DINOv3 ViT-S+/16 학습
python train.py --model_name dinov3_vits16plus --pretrained

# DINOv3 ConvNeXt Tiny 학습
python train.py --model_name dinov3_convnext_tiny --pretrained

# Backbone freeze (head만 학습)
python train.py --model_name dinov3_vits16plus --pretrained --freeze_backbone

# 커스텀 가중치 경로
python train.py --model_name dinov3_vits16plus --pretrained \
    --dinov3_weights_dir /path/to/custom/weights
```

### Pretrained 가중치

DINOv3 pretrained 가중치는 기본값 `/data/checkpoints/dinov3/`에 위치한다:
- `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` (ViT-S+/16)
- `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` (ConvNeXt Tiny)

`--dinov3_weights_dir` 옵션으로 다른 디렉토리를 지정할 수 있다.

### MambaVision과의 차이점

- `--drop_rate`는 MambaVision 전용이며, DINOv3에서는 무시된다.
- DINOv3는 CPU에서도 forward pass가 가능하다 (MambaVision은 CUDA 필수).
- DINOv3는 ImageNet이 아닌 LVD-1689M 데이터로 pretrained되었다.

## LoRA (Low-Rank Adaptation)

DINOv3 모델의 frozen backbone에 경량 LoRA 어댑터를 부착하여 feature를 미세 조정한다.
Backbone은 자동으로 freeze되며, LoRA 파라미터 + head만 학습된다.

### 기본 사용법

```bash
# ViT-S+ with LoRA (rank=8)
python train.py --model_name dinov3_vits16plus --pretrained \
    --lora_enabled --lora_rank 8 --lr 1e-4

# ConvNeXt-tiny with LoRA
python train.py --model_name dinov3_convnext_tiny --pretrained \
    --lora_enabled --lora_rank 8 --lr 1e-4
```

### LoRA 하이퍼파라미터

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--lora_enabled` | `False` | LoRA 활성화 (DINOv3 전용) |
| `--lora_rank` | `8` | LoRA rank (r). 4, 8, 16 권장 |
| `--lora_alpha` | `8.0` | LoRA 스케일링. scaling = alpha/rank |
| `--lora_dropout` | `0.0` | LoRA 브랜치 dropout |
| `--lora_target_modules` | (자동) | LoRA 대상 모듈 오버라이드 |

### 추론

```bash
python test.py --model_name dinov3_vits16plus \
    --lora_enabled --lora_rank 8 \
    --checkpoint_path checkpoints/run/best.pth
```

### LoRA vs Freeze-only 비교

| 전략 | Trainable params (ViT-S+) | 설명 |
|------|---------------------------|------|
| `--freeze_backbone` | ~770 (head만) | Linear probing |
| `--lora_enabled --lora_rank 8` | ~222K | LoRA + head |
| (no freeze) | ~22M | Full fine-tuning |

## WSGM (Weighted Side Gating Module)

DINOv3 모델의 frozen backbone에 경량 WSGM 어댑터를 부착하여 forgery-specific feature를 학습한다.
LoRA와 상호 배제이며, backbone은 자동으로 freeze된다.

### 기본 사용법

```bash
# ViT-S+ with WSGM
python train.py --model_name dinov3_vits16plus --pretrained --wsgm

# ConvNeXt-tiny with WSGM (concat 모드)
python train.py --model_name dinov3_convnext_tiny --pretrained --wsgm --wsgm_aggregation concat

# 보틀넥 비율 변경
python train.py --model_name dinov3_vits16plus --pretrained --wsgm --wsgm_reduction_factor 8
```

### WSGM 하이퍼파라미터

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--wsgm` | `False` | WSGM 활성화 (DINOv3 전용) |
| `--wsgm_reduction_factor` | `4` | 보틀넥 축소 비율 (embed_dim // factor) |
| `--wsgm_dropout` | `0.5` | WSGM 모듈 내 dropout |
| `--wsgm_aggregation` | `"average"` | `"average"` 또는 `"concat"` |

### 자동 스케일링

모델 크기에 따라 WSGM 파라미터가 자동 조정된다:

| 아키텍처 | num_wsgm_layers | bottleneck_dim | 비고 |
|----------|-----------------|----------------|------|
| ViT (depth=D) | D // 2 | embed_dim // factor | 균등 분배된 레이어 인덱스 |
| ConvNeXt (4 stages) | 4 | final_dim // factor | per-stage projection 포함 |

### 추론

```bash
python test.py --model_name dinov3_vits16plus --wsgm \
    --checkpoint_path checkpoints/run/best.pth
```

## Multi-view Consistency Training

동일 이미지의 서로 다른 augmentation 뷰 간 일관성을 학습하여 robustness를 높인다. CE loss에 supervised contrastive loss + view consistency loss를 추가한다.

```bash
# 기본 multi-view
python train.py --multi_view

# loss 가중치 조정
python train.py --multi_view --lambda_con 0.1 --lambda_mvc 0.05

# contrastive projection head 차원 지정
python train.py --multi_view --projection_dim 128
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--multi_view` | `False` | Multi-view consistency 학습 활성화 |
| `--lambda_con` | `0.1` | Supervised contrastive loss 가중치 |
| `--lambda_mvc` | `0.05` | Multi-view consistency loss 가중치 |
| `--con_temperature` | `0.07` | SupCon 온도 파라미터 |
| `--projection_dim` | `0` | Contrastive projection head 차원 (`0` = multi_view 시 자동 설정) |

## 데이터 분할

`build_train_val_loaders()`는 `--train_datasets`에 지정된 학습 데이터를 `--val_split_ratio` 비율로 train/val로 분할한다. `--seed`를 고정하면 동일한 분할이 재현된다.

```bash
python train.py --val_split_ratio 0.1 --seed 42   # 기본값: 10% validation
python train.py --val_split_ratio 0.2              # 20% validation
```

## Optimizer

AdamW를 사용한다. bias, normalization 파라미터에는 weight decay를 적용하지 않는다.

```bash
python train.py --lr 1e-4 --weight_decay 0.05
```

## LR Scheduler

Linear warmup 후 main schedule로 전환한다. 스케줄러는 배치(step) 단위로 업데이트된다.

### Cosine Annealing (기본값)

```
lr
 ↑  ╱‾‾‾‾╲
 │ ╱      ╲
 │╱ warmup ╲───────── eta_min (1e-7)
 └──────────────────→ step
```

```bash
python train.py --scheduler cosine --warmup_epochs 5
```

### StepLR

```bash
python train.py --scheduler step --step_lr_size 10 --step_lr_decay 0.1
# warmup 5 epoch 후, 매 10 epoch마다 lr × 0.1
```

## AMP / Gradient Clipping / Early Stopping / Label Smoothing

기본 활성화 옵션들. 각 기본값과 비활성화 방법은 아래와 같다:

```bash
python train.py --no_amp                      # AMP 비활성화 (기본: 활성, CUDA 없으면 자동 비활성)
python train.py --grad_clip_norm 0            # Gradient clipping 비활성화 (기본: 1.0)
python train.py --early_stopping_patience 0   # Early stopping 비활성화 (기본: 7 epoch)
python train.py --label_smoothing 0.0         # Label smoothing 비활성화 (기본: 0.1)
```

## Checkpoint

### 저장 파일

| 파일 | 조건 | 설명 |
|------|------|------|
| `best.pth` | val accuracy 갱신 시 | 최고 성능 모델 |
| `epoch_{N}.pth` | 매 `save_every` epoch | 주기적 백업 |
| `last.pth` | 학습 종료 시 | 최종 상태 (early stop 포함) |

### Checkpoint 포맷

```python
{
    "epoch": int,              # 저장 시점 epoch
    "model": OrderedDict,      # model.state_dict()
    "optimizer": dict,         # optimizer.state_dict()
    "scheduler": dict,         # scheduler.state_dict()
    "scaler": dict,            # GradScaler.state_dict()
    "best_val_auc": float,     # 최고 validation AUC (이전 checkpoint의 "best_val_acc"도 호환)
    "args": dict,              # 전체 하이퍼파라미터 스냅샷
}
```

`"model"` 키는 `GenAIClassifier._load_checkpoint()`와 호환된다. 따라서 저장된 checkpoint를 `test.py`에서 `--checkpoint_path`로 직접 로드할 수 있다.

### 학습 재개 (Resume)

`--resume`으로 checkpoint에서 model, optimizer, scheduler, scaler, epoch 전체를 복원한다.

```bash
python train.py --resume ./checkpoints/last.pth --epochs 50
# epoch 30에서 중단된 경우, epoch 30부터 50까지 이어서 학습
```

## TensorBoard 로깅

```bash
tensorboard --logdir=./runs
```

### 기록되는 스칼라

| 태그 | 주기 | 설명 |
|------|------|------|
| `train/loss` | 매 epoch | 학습 평균 loss |
| `train/accuracy` | 매 epoch | 학습 평균 accuracy |
| `train/lr` | 매 epoch | 현재 learning rate |
| `val/loss` | 매 `eval_every` epoch | 검증 평균 loss |
| `val/accuracy` | 매 `eval_every` epoch | 검증 평균 accuracy |
| `val/auc` | 매 `eval_every` epoch | 검증 ROC-AUC (sklearn 설치 시) |
| `val/f1` | 매 `eval_every` epoch | 검증 F1 score (sklearn 설치 시) |

로그 디렉토리는 `{log_dir}/{model_name}_lr{lr}_bs{batch_size}_{timestamp}/` 형태로 자동 생성된다.

## Validation 메트릭

기본 메트릭: loss, accuracy. `scikit-learn`이 설치되어 있으면 AUC, F1도 자동 계산된다.

```bash
pip install scikit-learn   # 선택 사항
```

## Args 레퍼런스

### Training 하이퍼파라미터

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `lr` | `float` | `1e-4` | 초기 learning rate |
| `weight_decay` | `float` | `0.05` | AdamW weight decay |
| `epochs` | `int` | `30` | 총 학습 epoch 수 |
| `warmup_epochs` | `int` | `5` | Linear warmup epoch 수 |
| `scheduler` | `str` | `"cosine"` | `"cosine"` 또는 `"step"` |
| `step_lr_decay` | `float` | `0.1` | StepLR gamma (scheduler=step 시) |
| `step_lr_size` | `int` | `10` | StepLR step size in epochs (scheduler=step 시) |
| `amp` | `bool` | `True` | Automatic Mixed Precision 활성화 |
| `grad_clip_norm` | `float` | `1.0` | Gradient norm clipping. `0` = 비활성화 |
| `early_stopping_patience` | `int` | `7` | Early stopping patience. `0` = 비활성화 |
| `label_smoothing` | `float` | `0.1` | CrossEntropyLoss label smoothing |
| `cutmix_p` | `float` | `0.0` | Same-label CutMix 적용 확률 (robust augmentation 전용, `0` = 비활성화) |
| `cutmix_alpha` | `float` | `0.4` | CutMix lambda 샘플링용 Beta 분포 alpha |

### 저장 및 로깅

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `save_dir` | `str` | `"./checkpoints"` | Checkpoint 저장 디렉토리 |
| `save_every` | `int` | `5` | 주기적 checkpoint 저장 간격 (epoch) |
| `log_dir` | `str` | `"./runs"` | TensorBoard 로그 디렉토리 |
| `eval_every` | `int` | `1` | Validation 실행 간격 (epoch) |
| `resume` | `str` | `""` | 학습 재개용 checkpoint 경로 |
| `tb_log_images` | `bool` | `True` | 학습 이미지를 TensorBoard에 기록 |
| `tb_log_images_per_epoch` | `int` | `5` | Epoch당 이미지 기록 횟수 |
| `tb_log_images_count` | `int` | `8` | 그리드당 이미지 수 (single-view) |
| `tb_log_images_pairs` | `int` | `4` | 그리드당 augmented/clean 쌍 수 (multi-view) |

### 분산 학습

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `distributed` | `bool` | `False` | DDP 활성화 (`torchrun` 시 자동) |
| `dp` | `bool` | `False` | `nn.DataParallel` 멀티 GPU (`torchrun` 불필요) |
| `dist_backend` | `str` | `"nccl"` | `"nccl"` (GPU) 또는 `"gloo"` (CPU) |
| `scale_lr` | `bool` | `False` | Linear LR scaling (lr × world_size) |
| `sync_bn` | `bool` | `False` | SyncBatchNorm 변환 |
| `seed` | `int` | `42` | 재현성을 위한 random seed |

데이터, 모델 관련 옵션은 [dataloader.md](dataloader.md) 참조.

## 분산 학습 (Multi-GPU DDP)

`torchrun`으로 2개 이상의 GPU에서 DistributedDataParallel 학습을 수행한다.

### 실행 방법

```bash
# 2 GPU (DDP, torchrun 필요)
torchrun --nproc_per_node=2 train.py --batch_size 32

# 4 GPU + LR scaling (lr × world_size)
torchrun --nproc_per_node=4 train.py --batch_size 32 --lr 1e-4 --scale_lr

# DataParallel (torchrun 없이 멀티 GPU)
python train.py --dp --batch_size 64

# 특정 GPU 지정
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py

# 단일 GPU (기존과 동일)
python train.py
```

### DDP 동작 방식

- `batch_size`는 GPU당 크기. 4 GPU × batch_size 32 = 유효 배치 128
- `torchrun`이 `RANK/LOCAL_RANK/WORLD_SIZE` 환경변수를 설정하면 자동으로 DDP 모드 활성화
- Rank 0만 checkpoint 저장, TensorBoard 로깅, 콘솔 출력 수행
- Checkpoint는 DDP prefix 없이 저장되어 단일 GPU에서도 로드 가능
- `--scale_lr` 사용 시 learning rate가 world_size 배로 자동 스케일링 (Linear Scaling Rule)
- `--sync_bn` 사용 시 BatchNorm이 SyncBatchNorm으로 변환되어 GPU 간 통계 동기화

### Multi-node 학습

```bash
# Node 0:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 \
    train.py --batch_size 32

# Node 1:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=29500 \
    train.py --batch_size 32
```

## 주의사항

### dataset_mode 제한

현재 `train.py`는 `--dataset_mode concat`만 지원한다. `--dataset_mode separate`를 사용하면 `NotImplementedError`가 발생한다.

### CPU fallback

CUDA가 사용 불가능하면 AMP가 자동으로 비활성화되고 CPU에서 학습이 진행된다.

### 재현성

`--seed` 옵션으로 random, numpy, torch seed를 고정한다. `cudnn.deterministic=True`, `cudnn.benchmark=False`가 설정되어 동일 seed에서 동일 결과를 보장한다.
