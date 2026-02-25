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

## TTA (Test-Time Augmentation)

`--tta` 옵션으로 다양한 TTA 전략을 선택할 수 있다. 각 augmented view에 대한 logit을 평균하여 최종 예측을 수행한다.

```
logits_final = mean(logits_view_1, logits_view_2, ..., logits_view_N)
prediction = argmax(logits_final)
```

### TTA 모드 (Legacy: `tta_prep_size` 미지정 또는 `== image_size`)

| 모드 | Views | Forward passes | 구성 |
|------|-------|---------------|------|
| `none` | 1 | 1x | 원본만 (TTA 없음) |
| `flip` | 2 | 2x | 원본 + horizontal flip |
| `multiscale` | 4 | 4x | 원본 + 3 multi-scale crops (256, 288, 320 → CenterCrop 224) |
| `full` | 8 | 8x | 원본 + flip + 3 rotations (90°/180°/270°) + 3 multi-scale crops |

### Pixel-Preserving TTA (`--tta_prep_size 512`)

`--tta_prep_size`를 `image_size`보다 크게 설정하면 pixel-preserving TTA가 활성화된다. 이미지를 큰 prep tensor로 로드한 뒤, resize 없이 직접 crop하는 뷰(pixel 보존)와 resize 기반 뷰를 함께 생성한다. GenAI 아티팩트 검출에 적합하다.

| 모드 | Crop-only 뷰 | Resize 뷰 | 총 Views | Crop:Resize 비율 |
|------|-------------|-----------|----------|-----------------|
| `none` | center crop (1) | - | 1 | 1:0 |
| `flip` | center + flip (2) | - | 2 | 2:0 |
| `multiscale` | center (1) | 3 scales (3) | 4 | 1:3 |
| `full` | center + flip + 4 corners (6) | 3 scales (3) | 9 | **2:1** |

- **Crop-only 뷰**: prep tensor에서 직접 crop (tensor slicing). Interpolation 없이 원본 pixel 그대로 보존.
- **Resize 뷰**: prep tensor를 resize 후 center crop. 기존 multi-scale과 동일하지만 원본 해상도에서 출발.
- `ResizeIfSmaller`: 이미지의 shortest edge가 `prep_size` 이상이면 resize를 하지 않음 → 대부분의 GenAI 이미지(512+)에서 interpolation 0회.

### 사용법

```bash
# Legacy TTA (기존 동작)
python test.py --checkpoint_path ./checkpoints/best.pth --tta flip
python test.py --checkpoint_path ./checkpoints/best.pth --tta full

# Pixel-preserving TTA (권장)
python test.py --checkpoint_path ./checkpoints/best.pth --tta full --tta_prep_size 512

# 메모리 부족 시 batch_size 조정
python test.py --checkpoint_path ./checkpoints/best.pth --tta full --tta_prep_size 512 --batch_size 16

# TTA 없이 추론
python test.py --checkpoint_path ./checkpoints/best.pth --tta none
python test.py --checkpoint_path ./checkpoints/best.pth   # 기본값 = none
```

### 설계 원리

- **Pixel-preserving crop**: prep tensor에서 center crop, 4 corner crop을 tensor slicing으로 수행. Interpolation이 없으므로 GenAI 생성 아티팩트가 완전히 보존됨.
- **Multi-scale crop**: normalized tensor에 `F.interpolate` (bilinear) → `center_crop(224)`. Normalization은 per-channel affine transform이므로 interpolation 후 normalize와 수학적으로 동치.
- **Legacy 호환**: `tta_prep_size` 미지정 시(기본값 224) rotation 기반 legacy 경로가 그대로 사용됨.
- **메모리 효율**: view별 순차 forward로 batch를 N배로 concat하지 않음. `tta_prep_size=512` 사용 시 tensor 크기가 ~5.2x 증가하므로 `batch_size`를 줄여야 할 수 있음.

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

## Args 레퍼런스

### Test 전용 옵션

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `checkpoint_path` | `str` | `""` | 모델 checkpoint 경로 **(필수)** |
| `output_dir` | `str` | `"./predictions"` | CSV 출력 디렉토리 |
| `tta` | `str` | `"none"` | TTA 모드: `none`, `flip`, `multiscale`, `full` (값 없이 `--tta`만 쓰면 `flip`) |
| `tta_prep_size` | `int` | `224` | TTA prep tensor 크기. `image_size`보다 크면 pixel-preserving crop 활성화 (권장: `512`) |
| `eval_val` | `bool` | `False` | Labeled validation 데이터 평가 수행 |
| `amp` | `bool` | `True` | Automatic Mixed Precision |
| `ntire_test_mode` | `int` | `1` | 테스트 subset: `1`=val_images, `2`=val_images_hard, `3`=둘 다 |

데이터, 모델 관련 옵션은 [dataloader.md](dataloader.md) 참조.

## 주의사항

### checkpoint_path 필수

`--checkpoint_path`를 지정하지 않으면 `ValueError`가 발생한다. 랜덤 가중치로 추론하는 것을 방지하기 위함이다.

### CPU fallback

CUDA가 사용 불가능하면 AMP가 자동으로 비활성화되고 CPU에서 추론이 진행된다.

### 메모리

추론 시 gradient 계산이 없으므로 (`@torch.no_grad()`) 학습 대비 메모리 사용량이 적다. Legacy TTA 사용 시에도 추가 메모리는 거의 없다. `--tta_prep_size 512` 사용 시 batch tensor 크기가 ~5.2x 증가하므로 `--batch_size`를 줄여야 할 수 있다.
