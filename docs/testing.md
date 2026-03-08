# 테스트 (Inference) 사용법

## 기본 사용법

```bash
python test.py --checkpoint_path ./checkpoints/best.pth
```

학습된 모델로 NTIRE 테스트 이미지에 대한 예측을 수행하고 CSV 파일을 생성한다.

## 추론 흐름

```
1. argparse 파싱 (data + model + test args)
2. checkpoint_path 검증 (필수)
3. build_model() → GenAIClassifier (checkpoint 자동 로드)
4. build_test_dataloader() → DataLoader (unlabeled 테스트 데이터)
5. run_inference() → predictions
6. generate_csv() → CSV 파일 저장
7. (선택) evaluate_val() → labeled val 데이터 평가
```

## CLI 옵션

```bash
# 기본 추론
python test.py --checkpoint_path ./checkpoints/best.pth

# Test mode 선택
python test.py --checkpoint_path ./checkpoints/best.pth --ntire_test_mode 1   # val_images만
python test.py --checkpoint_path ./checkpoints/best.pth --ntire_test_mode 2   # val_images_hard만
python test.py --checkpoint_path ./checkpoints/best.pth --ntire_test_mode 3   # 둘 다

# TTA (Test-Time Augmentation) 활성화
python test.py --checkpoint_path ./checkpoints/best.pth --tta

# AMP 비활성화
python test.py --checkpoint_path ./checkpoints/best.pth --no_amp

# 출력 디렉토리 지정
python test.py --checkpoint_path ./checkpoints/best.pth --output_dir ./predictions/exp01

# Validation 데이터 평가 추가
python test.py --checkpoint_path ./checkpoints/best.pth --eval_val --val_datasets ntire

# 배치 크기, worker 수 조정
python test.py --checkpoint_path ./checkpoints/best.pth --batch_size 64 --num_workers 4

# Softmax score CSV 출력
python test.py --checkpoint_path ./checkpoints/best.pth --output_scores
```

## Test Mode

NTIRE 대회 테스트 데이터는 두 개의 subset으로 구성된다.

| 모드 | `--ntire_test_mode` | 대상 | 반환 타입 |
|------|---------------------|------|-----------|
| 1 | `val_images` | 일반 테스트 이미지 | 단일 DataLoader |
| 2 | `val_images_hard` | 어려운 테스트 이미지 | 단일 DataLoader |
| 3 | 둘 다 | 양쪽 모두 | `dict[str, DataLoader]` |

### Mode 1, 2 출력

```
predictions/
  predictions_val_images.csv          # mode 1
  predictions_val_images_hard.csv     # mode 2
```

### Mode 3 출력

```
predictions/
  predictions_val_images.csv          # subset별 CSV
  predictions_val_images_hard.csv
  predictions_all.csv                 # 합산 CSV
```

## CSV 출력 포맷

NTIRE 대회 제출 형식에 맞춘 CSV 파일이 생성된다.

```csv
image_name,label
test_0000.jpg,0
test_0001.jpg,1
test_0002.jpg,0
...
```

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `image_name` | `str` | 이미지 파일명 (metadata의 `source_id`) |
| `label` | `int` | 예측 라벨. `0` = real, `1` = fake |

### Score CSV (`--output_scores`)

`--output_scores` 사용 시 softmax 확률이 포함된 별도 CSV도 생성된다.

```csv
image_name,score
test_0000.jpg,0.1234
test_0001.jpg,0.9876
```

`score`는 fake 클래스(class 1)의 softmax 확률이다.

## TTA (Test-Time Augmentation)

`--tta` 옵션으로 다양한 TTA 전략을 선택할 수 있다. 각 augmented view에 대한 logit을 평균하여 최종 예측을 수행한다.

```
logits_final = mean(logits_view_1, logits_view_2, ..., logits_view_N)
prediction = argmax(logits_final)
```

### TTA 모드

| 모드 | 설명 | Views |
|------|------|-------|
| `none` | TTA 없음 (원본만) | 1 |
| `flip` | 원본 + horizontal flip | 2 |
| `multiscale` | 원본 + 3 multi-scale crops | 4 |
| `full` | Pixel-preserving: center + flip + 4 corner crops + 3 multi-scale | 9 |
| `full_legacy` | Rotation 기반: 원본 + flip + 3 rotations + 3 multi-scale | 8 |

### Pixel-Preserving TTA (`full` 모드)

`full` 모드는 prep tensor(기본 512)에서 resize 없이 직접 crop하여 원본 pixel을 보존한다. GenAI 아티팩트 검출에 적합하다.

| 뷰 유형 | 구성 | 비고 |
|----------|------|------|
| Crop-only | center + flip + 4 corners (6) | Interpolation 없이 원본 pixel 보존 |
| Resize | 3 scales (3) | 원본 해상도에서 출발하여 resize 후 center crop |

- `--tta_min_prep_size`: prep tensor 최소 크기 (기본 512). 이 크기 이상 이미지는 원본 해상도 유지
- `full_legacy`: 기존 rotation 기반 TTA가 필요한 경우 사용

### 사용법

```bash
# Pixel-preserving TTA (권장)
python test.py --checkpoint_path ./checkpoints/best.pth --tta full

# Flip만
python test.py --checkpoint_path ./checkpoints/best.pth --tta flip

# 기존 rotation 기반 TTA
python test.py --checkpoint_path ./checkpoints/best.pth --tta full_legacy

# prep tensor 크기 조정
python test.py --checkpoint_path ./checkpoints/best.pth --tta full --tta_min_prep_size 768

# 메모리 부족 시 batch_size 조정
python test.py --checkpoint_path ./checkpoints/best.pth --tta full --batch_size 16

# TTA 없이 추론
python test.py --checkpoint_path ./checkpoints/best.pth   # 기본값 = none
```

## Validation 평가 (--eval_val)

`--eval_val` 플래그로 라벨이 있는 validation 데이터에 대한 정량 평가를 수행할 수 있다.

```bash
python test.py --checkpoint_path ./checkpoints/best.pth \
  --eval_val --val_datasets ntire
```

### 출력 예시

```
==================================================
  Evaluation Results: val
==================================================
  Samples:    5000
  Accuracy:   0.8745
  AUC:        0.9312
  F1:         0.8690

  Confusion Matrix:
               Pred Real  Pred Fake
  Actual Real      2312       188
  Actual Fake       440      2060
==================================================
```

### 메트릭

| 메트릭 | 설명 | 필요 패키지 |
|--------|------|-------------|
| Accuracy | 정확도 | 내장 |
| AUC | ROC-AUC score | scikit-learn |
| F1 | F1 score (binary) | scikit-learn |
| Confusion Matrix | 혼동 행렬 | scikit-learn |

`scikit-learn`이 설치되어 있지 않으면 accuracy만 출력되고 나머지는 건너뛴다.

```bash
pip install scikit-learn   # 선택 사항
```

## Checkpoint 호환성

`test.py`는 `--checkpoint_path`를 통해 checkpoint를 로드한다. 내부적으로 `GenAIClassifier(checkpoint_path=...)` → `_load_checkpoint()`가 호출되어 다음 형식을 모두 지원한다:

- `train.py`가 저장한 checkpoint (`"model"` 키)
- `"state_dict"` 키를 가진 checkpoint
- DDP `"module."` prefix가 있는 checkpoint
- `"backbone."` prefix가 있는 checkpoint
- Raw `state_dict`

### DDP 학습 후 추론

DDP로 학습된 checkpoint는 `"module."` prefix 없이 저장되므로, 단일 GPU에서 바로 추론 가능하다.

```bash
# DDP 학습
torchrun --nproc_per_node=4 train.py --batch_size 32

# 단일 GPU 추론 (변경 없이 사용)
python test.py --checkpoint_path ./checkpoints/best.pth
```

## Ensemble 추론

여러 모델의 예측을 결합하여 성능을 높인다.

```bash
# 2개 모델 앙상블 (mean_prob)
python test.py \
    --ensemble_checkpoints ./ckpt/model_a.pth ./ckpt/model_b.pth \
    --ensemble_models dinov3_vits16plus mamba_vision_T

# 가중 평균 + 모델별 TTA
python test.py \
    --ensemble_checkpoints ./ckpt/a.pth ./ckpt/b.pth \
    --ensemble_models dinov3_vits16plus mamba_vision_T \
    --ensemble_weights 0.7 0.3 \
    --ensemble_tta full flip

# Majority vote
python test.py \
    --ensemble_checkpoints ./ckpt/a.pth ./ckpt/b.pth ./ckpt/c.pth \
    --ensemble_models dinov3_vits16plus dinov3_vits16plus mamba_vision_T \
    --ensemble_method majority_vote
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--ensemble_checkpoints` | `[]` | 각 모델의 checkpoint 경로 리스트 |
| `--ensemble_models` | `[]` | 각 checkpoint에 대응하는 모델명 리스트 |
| `--ensemble_weights` | `[]` (균등) | 모델별 가중치 (비어있으면 균등 배분) |
| `--ensemble_method` | `"mean_prob"` | `mean_prob`, `mean_logit`, `majority_vote` |
| `--ensemble_tta` | `[]` | 모델별 TTA 모드 (비어있으면 `--tta` 값 공통 적용) |

## Args 레퍼런스

### Test 전용 옵션

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `checkpoint_path` | `str` | `""` | 모델 checkpoint 경로 **(필수)** |
| `output_dir` | `str` | `"./predictions"` | CSV 출력 디렉토리 |
| `tta` | `str` | `"none"` | TTA 모드: `none`, `flip`, `multiscale`, `full`, `full_legacy` |
| `tta_min_prep_size` | `int` | `512` | Prep tensor 최소 크기. 이 크기 이상 이미지는 원본 해상도 유지 |
| `eval_val` | `bool` | `False` | Labeled validation 데이터 평가 수행 |
| `output_scores` | `bool` | `False` | Softmax score CSV 출력 (`image_name,score`) |
| `amp` | `bool` | `True` | Automatic Mixed Precision |
| `ntire_test_mode` | `int` | `1` | 테스트 subset: `1`=val_images, `2`=val_images_hard, `3`=둘 다 |

데이터, 모델 관련 옵션은 [dataloader.md](dataloader.md) 참조.

## 주의사항

### checkpoint_path 필수

`--checkpoint_path`를 지정하지 않으면 `ValueError`가 발생한다. 랜덤 가중치로 추론하는 것을 방지하기 위함이다.

### CPU fallback

CUDA가 사용 불가능하면 AMP가 자동으로 비활성화되고 CPU에서 추론이 진행된다.

### 메모리

추론 시 `@torch.no_grad()`로 gradient 계산이 없어 메모리 효율적이다. `--tta full` 사용 시 prep tensor 크기 증가로 `--batch_size` 조정이 필요할 수 있다.
