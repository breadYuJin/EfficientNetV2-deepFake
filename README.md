# EfficientNetV2-deepFake
=======
# Deepfake 이진 분류 (EfficientNetV2-L)

`2.DataSet/new/dataSet`을 사용해 EfficientNetV2-L 기반 이진(real=0 / fake=1) 분류기를 학습한다.
이 디렉토리는 **다른 서버에서 인계받아 실행**하는 것을 전제로 작성되었다.

---

## 1. 데이터 가정

학습/평가용 데이터는 다음과 같은 구조여야 한다 (이후 `<DATA_ROOT>`로 표기).

```
<DATA_ROOT>/
├── real/      # *.jpg, 256x256
├── fake/      # *.jpg, 256x256
├── test/      # 무시 — 사용하지 않음
└── labels.csv # 무시 — 사용하지 않음 (split을 새로 만들기 때문)
```

각 이미지 파일명은 `<source>_<id>.jpg` 형식이며, `<source>`는 다음 중 하나다.

`140K · DF40 · forgeryNet · keggle · wild`

스크립트는 폴더(real|fake)로 라벨을, 파일명 prefix로 소스를 식별한다.
예상 규모 (2026-05-28 기준):

| 폴더 | 장수 |
|---|---|
| real | 1,771,748 |
| fake | 2,110,782 |
| **합계** | **3,882,530** |

---

## 2. 환경 설정

- Python 3.10+ 권장
- CUDA GPU 필수 (V2-L + ~3.9M 이미지)

```bash
cd 3.model

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 1) 서버 CUDA 버전에 맞게 PyTorch 먼저 설치 (예시: CUDA 12.1)
pip install torch==2.4.* torchvision==0.19.* \
    --index-url https://download.pytorch.org/whl/cu121

# 2) 나머지 의존성
pip install -r requirements.txt
```

설치 확인:
```bash
python -c "import torch, timm; print(torch.__version__, torch.cuda.is_available(), timm.__version__)"
```

---

## 3. 디렉토리 구조

```
3.model/
├── README.md
├── requirements.txt
├── src/
│   ├── build_splits.py   # 데이터 분할 (1회 실행)
│   ├── dataset.py        # PyTorch Dataset
│   ├── model.py          # timm 모델 팩토리
│   ├── train.py          # 학습 루프
│   └── evaluate.py       # 테스트셋 평가
├── splits/               # build_splits.py 실행 후 생성
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── split_summary.json
└── runs/<run_name>/      # train.py 실행 후 생성
    ├── config.json
    ├── train_log.jsonl
    ├── checkpoints/{best.pt, last.pt}
    └── reports/          # evaluate.py 결과
```

---

## 4. 실행 순서

### Step 1 — 데이터 분할 (한 번만)

소스(140K/DF40/forgeryNet/keggle/wild) × 클래스(real/fake) 10개 그룹 각각을 **7:2:1로 stratified 분할**한다.
**seed=42 고정** — 어느 서버에서 다시 돌려도 같은 split이 나온다.

```bash
python src/build_splits.py \
    --data_root <DATA_ROOT> \
    --out_dir splits
```

검증:
- `splits/split_summary.json`의 `totals`가 train ≈ 2.72M, val ≈ 0.78M, test ≈ 0.39M 근처면 OK.
- 다른 환경에서 split을 다시 만들고 싶지 않다면 `splits/` 폴더 자체를 복사해 가져가도 동일하다.

### Step 2 — 학습

```bash
python src/train.py \
    --data_root <DATA_ROOT> \
    --splits_dir splits \
    --output_dir runs/v1 \
    --batch_size 64 \
    --epochs 8 \
    --num_workers 8
```

주요 인자:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model_name` | `tf_efficientnetv2_l` | timm 모델 이름 |
| `--image_size` | 256 | 입력 해상도 (원본이 256이므로 그대로 사용) |
| `--batch_size` | 64 | V2-L 기준. VRAM에 맞게 조정 |
| `--epochs` | 8 | 데이터가 많아 적게 잡음 |
| `--lr` | 1e-4 | AdamW + Cosine schedule |
| `--weight_decay` | 1e-4 | |
| `--num_workers` | 8 | DataLoader 워커 수 |
| `--seed` | 42 | |
| `--no_amp` | off | 끄면 FP32. 기본은 mixed precision |
| `--no_pretrained` | off | 끄면 ImageNet 사전학습 가중치 미사용 |
| `--resume` | None | 체크포인트 경로로 이어서 학습 |
| `--max_train_samples` | None | 디버그용 — 학습셋을 N개로 제한 |

산출물 (`runs/v1/`):
- `config.json` — 사용된 인자 전체 (재현용)
- `train_log.jsonl` — 매 epoch loss / val_acc / val_auroc / lr / time
- `checkpoints/last.pt` — 매 epoch 갱신
- `checkpoints/best.pt` — val AUROC 최고 시점

### Step 3 — 평가

테스트셋 전체 + 소스별 metric:

```bash
python src/evaluate.py \
    --data_root <DATA_ROOT> \
    --splits_dir splits \
    --checkpoint runs/v1/checkpoints/best.pt \
    --output_dir runs/v1/reports \
    --split test
```

산출물:
- `reports/test_metrics.json` — overall + 소스별 acc / precision / recall / f1 / AUROC / confusion matrix
- `reports/test_confusion.png`
- `reports/test_roc.png`

### Step 4 — 학습 이어가기 (선택)

서버 재시작이나 중단된 경우:

```bash
python src/train.py \
    --data_root <DATA_ROOT> \
    --splits_dir splits \
    --output_dir runs/v1 \
    --resume runs/v1/checkpoints/last.pt \
    --epochs 8
```

`--resume`는 epoch / optimizer / scheduler / scaler / best AUROC를 모두 복원한다.
`--epochs`는 **총 학습 epoch**이므로, 이미 5 epoch까지 했고 3 epoch 더 돌리려면 그대로 `--epochs 8`로 두면 된다.

---

## 5. 재현성

- 데이터 분할: `build_splits.py`에서 `random.Random(42)` 고정 + 파일명 정렬 후 셔플
- 학습 seed: `train.py --seed 42` 기본 (Python/NumPy/PyTorch 모두 시드)
- DataLoader 워커 셔플은 PyTorch generator 미고정이므로 step-level까지 완전 동일 재현은 아님 — epoch 단위 metric은 거의 일치

---

## 6. 결정사항 / 가정

| 항목 | 선택 | 이유 |
|---|---|---|
| Backbone | EfficientNetV2-L (`tf_efficientnetv2_l`) | 최대 성능 우선 |
| 입력 크기 | 256 | 데이터셋 원본 해상도. V2-L 기본 480은 256 데이터 업스케일 효과 없음 |
| Split | 소스별 stratified 7:2:1 | 5개 소스 도메인 분포를 train/val/test에 균일 분배 |
| 클래스 불균형 | 처리 안 함 | fake:real ≈ 1.19:1, baseline에서 이슈 적음 |
| Loss | BCEWithLogitsLoss (1-logit) | |
| Optim | AdamW + Cosine | |
| Mixed precision | 기본 ON | V2-L + 3.9M 이미지 시간 단축 |
| 증강 | HFlip + 약한 ColorJitter + RandomErasing | 얼굴 위주이므로 보수적 증강 |

---

## 7. 알려진 caveat

1. **프레임 누수**: forgeryNet / wild는 같은 영상에서 추출된 프레임이 다수 존재한다. 소스별 stratified split은 도메인 누수는 막지만 **영상 단위 누수까지는 막지 못한다**. 보고된 test metric은 실 세계 일반화 성능보다 다소 낙관적일 수 있다.
2. **`dataset_report.txt`의 수치(172K)는 outdated**다. 실제 데이터는 ~3.88M장.
3. **`labels.csv`의 split 컬럼은 사용하지 않는다**. forgeryNet에만 채워져 있고 그 split 비율도 본 작업과 다르다.
4. **`test/` 폴더 (8,985장, forgeryNet)는 사용하지 않는다**. 본 split의 test는 별도로 새로 만든다.
5. 파일명 prefix가 5개 소스 중 어디에도 매치되지 않으면 split에서 제외된다. `split_summary.json`의 `skipped_files` 값으로 확인 가능.

---

## 8. 인계 체크리스트

새 환경에서 시작할 때:

- [ ] `<DATA_ROOT>` 위치와 `real/`, `fake/` 폴더 접근 가능 확인
- [ ] `nvidia-smi`로 GPU / VRAM 확인 → batch_size 조정 기준
- [ ] PyTorch + CUDA 버전 호환 설치
- [ ] `pip install -r requirements.txt`
- [ ] `python src/build_splits.py ...` → `split_summary.json` totals 확인
- [ ] (선택) `--max_train_samples 10000`로 train.py 한 번 돌려 파이프라인 검증
- [ ] 본 학습 실행
- [ ] `evaluate.py`로 test metric 산출
>>>>>>> a422c91 (Initial commit: deepfake EfficientNetV2-L training scripts)
