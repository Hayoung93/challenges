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

`--tta` 플래그로 horizontal flip TTA를 활성화할 수 있다. 원본과 좌우 반전 이미지에 대한 logit을 평균하여 최종 예측을 수행한다.

```
logits_final = (logits_original + logits_flipped) / 2
prediction = argmax(logits_final)
```

```bash
python test.py --checkpoint_path ./checkpoints/best.pth --tta
```

TTA는 forward pass를 2배로 증가시키지만, gradient 계산이 없으므로 메모리 사용량은 거의 동일하다.

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

## Args 레퍼런스

### Test 전용 옵션

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `checkpoint_path` | `str` | `""` | 모델 checkpoint 경로 **(필수)** |
| `output_dir` | `str` | `"./predictions"` | CSV 출력 디렉토리 |
| `tta` | `bool` | `False` | Test-Time Augmentation (horizontal flip) |
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

추론 시 gradient 계산이 없으므로 (`@torch.no_grad()`) 학습 대비 메모리 사용량이 적다. TTA 사용 시에도 추가 메모리는 거의 없다.
