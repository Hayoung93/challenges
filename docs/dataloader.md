# Dataloader 사용법

## 기본 사용법

```python
import argparse
from config import add_data_args, merge_config
from data import build_dataloader

parser = argparse.ArgumentParser()
add_data_args(parser)
args = parser.parse_args()
args = merge_config(args)

loader = build_dataloader(args, split="train")
for images, labels, metadata in loader:
    # images: (B, 3, 224, 224)  — ImageNet 정규화 적용된 텐서
    # labels: (B,)              — 0=real, 1=fake
    # metadata: list[dict]      — 데이터셋별 메타정보
    ...
```

## CLI 옵션

```bash
# 데이터셋 선택 (nargs="+")
python train.py --train_datasets dragon ntire   # 둘 다 사용 (기본값)
python train.py --train_datasets dragon         # Dragon만
python train.py --train_datasets ntire          # NTIRE만

# 멀티 데이터셋 모드
python train.py --dataset_mode concat           # ConcatDataset (기본값)
python train.py --dataset_mode separate         # 데이터셋별 별도 DataLoader

# NTIRE 특정 shard만 로드
python train.py --ntire_shards 0 1 2

# Transform 설정
python train.py --image_size 224 --augmentation default
# augmentation 옵션: "none", "default", "strong"

# DataLoader 설정
python train.py --batch_size 32 --num_workers 8

# Dragon LRU 캐시 조절
python train.py --dragon_lru_capacity 2
```

## Concat vs Separate 모드

### Concat (기본값)

모든 데이터셋을 `ConcatDataset`으로 합쳐 단일 `DataLoader` 반환.

```python
args.dataset_mode = "concat"
loader = build_dataloader(args, split="train")  # DataLoader 1개
for images, labels, metadata in loader:
    ...
```

### Separate

데이터셋별 `DataLoader`를 `dict`로 반환. 교차 학습이나 데이터셋별 평가에 유용.

```python
args.dataset_mode = "separate"
loaders = build_dataloader(args, split="train")  # dict[str, DataLoader]
for name, loader in loaders.items():
    batch = next(iter(loader))
```

## 반환 형식

`__getitem__`은 항상 `(image, label, metadata)` 3-tuple을 반환.

| 필드 | 타입 | 설명 |
|------|------|------|
| `image` | `torch.Tensor (3, H, W)` | ImageNet 정규화 적용 |
| `label` | `int` | 0=real, 1=fake |
| `metadata` | `dict` | `dataset`, `source_id` 등 |

Dragon metadata 추가 필드: `model` (생성 모델명), `prompt_cls` (프롬프트 카테고리).

## Args 필드 레퍼런스

`build_dataloader(args, split)` 호출 시 `args`에서 참조하는 필드 목록. 모든 필드는 `getattr(args, field, default)` 패턴으로 접근되므로 누락 시 기본값이 적용된다.

### 데이터셋 선택 (필수)

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `train_datasets` | `list[str]` | `["dragon", "ntire"]` | 학습에 사용할 데이터셋. 빈 리스트이면 `ValueError` 발생 |
| `val_datasets` | `list[str]` | `[]` | 검증에 사용할 데이터셋. 빈 리스트이면 `ValueError` 발생 |
| `dataset_mode` | `str` | `"concat"` | `"concat"` (단일 DataLoader) 또는 `"separate"` (데이터셋별 DataLoader dict) |

### 데이터 경로

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `dragon_root` | `str` | `/data/data/dragon_dataset_regular` | Dragon arrow 파일 디렉토리 |
| `ntire_root` | `str` | `/data/data/NTIRE2026_GenAI` | NTIRE shard 디렉토리 |
| `ntire_shards` | `list[int]` \| `None` | `None` | 로드할 shard 인덱스. `None`이면 전체 자동 탐색 |

### Transform

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `image_size` | `int` | `224` | Transform 출력 크기 (crop) |
| `resize_size` | `int` | `256` | val transform에서 crop 전 resize 크기 |
| `augmentation` | `str` | `"default"` | train augmentation: `"none"`, `"default"`, `"strong"` |

### DataLoader

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `batch_size` | `int` | `32` | 배치 크기 |
| `num_workers` | `int` | `8` | DataLoader worker 수 |
| `pin_memory` | `bool` | `True` | CUDA pinned memory 사용 |
| `drop_last` | `bool` | `True` | train에만 적용. val은 항상 `False` |
| `distributed` | `bool` | `False` | `True`이면 `DistributedSampler` 사용 |

### Dragon 전용

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `dragon_lru_capacity` | `int` | `4` | 메모리에 유지할 arrow Table 최대 수 |
| `dragon_index_cache` | `str` | `.cache/dragon_index.json` | 파일별 row count 캐시 경로 |

### 최소 호출 예시

```python
from types import SimpleNamespace
from data import build_dataloader

# 필수 필드만 지정 — 나머지는 모두 getattr default 적용
args = SimpleNamespace(
    train_datasets=["dragon"],
    dragon_root="/data/data/dragon_dataset_regular",
)
loader = build_dataloader(args, split="train")
```

## 새 데이터셋 추가

1. `data/` 아래에 `BaseGenAIDataset`을 상속하는 클래스 작성
2. `data/__init__.py`의 `DATASET_REGISTRY`에 등록
3. `build_dataset()`에 생성 로직 추가

## 주의사항

### Dragon LRU 메모리

Dragon 데이터셋은 62개 arrow 파일(각 ~475MB, 총 ~29GB)로 구성되며, LRU 캐시로 필요한 파일만 메모리에 유지한다. 메모리 사용량 계산:

```
num_workers × dragon_lru_capacity × 475MB
```

| `num_workers` | `lru_capacity` | Worst case |
|---------------|----------------|------------|
| 8 | 4 (기본값) | ~15.2 GB |
| 8 | 2 | ~7.6 GB |
| 4 | 2 | ~3.8 GB |

메모리 부족 시 `--dragon_lru_capacity 2` 또는 `--num_workers 4`로 조절.

### Dragon 인덱스 캐시

첫 실행 시 62개 arrow 파일을 스캔하여 row count를 `.cache/dragon_index.json`에 저장한다. 이후 실행은 캐시에서 즉시 로드되어 초기화 시간이 단축된다. 캐시 경로는 `--dragon_index_cache`로 변경 가능.

### ImageNet 정규화

모든 transform은 ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`로 정규화한다. MambaVision 등 ImageNet pretrained backbone과 호환.
