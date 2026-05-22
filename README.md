# AEC 신호 기반 근감소증 예측 연구

**연세대학교 의과대학 강남세브란스병원 의료기기산업학과**

CT 촬영 시 스캐너가 자동으로 조절하는 관전류(mA) 시계열(AEC 신호)을 활용해 근감소증(Sarcopenia)을 예측하는 연구.

---

## 연구 개요

**핵심 가설**: AEC 신호는 성별·나이가 설명하지 못하는 추가적인 체형 정보를 담고 있으며, 다변량 맥락에서 SMI 이진 분류 예측력을 유의미하게 향상시킨다.

**실험 축**:
- **데이터셋**: `ok` (AEC 보유 환자만) vs `ok+missing` (AEC missing 환자 포함)
- **Loss**: BCEWithLogitsLoss (`bce`) vs FocalLoss (`focal`)
- **AEC 해상도**: 128pt vs 256pt
- **AEC 변환**: 5종 민감도 분석

---

## 디렉토리 구조

```text
2026-1_Study/
├── 연구코드/
│   ├── code/0522/
│   │   ├── data/                        # 데이터 전처리 파이프라인
│   │   │   ├── metadata/
│   │   │   │   ├── 0_unique.py          # 중복 PatientID 확인
│   │   │   │   └── 1_verify_metadata.py # metadata 검증
│   │   │   ├── aec/
│   │   │   │   ├── 3_extract_z_bounds.py # z-range 추출
│   │   │   │   └── 4_crop_aec.py         # AEC 신호 크롭·보간
│   │   │   └── merged_features.py        # metadata + AEC 병합 → xlsx
│   │   └── model/                        # [Project 2] AEC curve → SMI 이진 분류
│   │       ├── main.py                   # 진입점 — run_all_cases()
│   │       ├── config.py                 # 전역 하이퍼파라미터·경로·상수
│   │       ├── data.py                   # 데이터 로드·분할
│   │       ├── cross_val.py              # Stratified K-Fold CV (M1/M2/M3)
│   │       ├── evaluate.py               # Test set 최종 평가
│   │       ├── models.py                 # ResNet1D / CrossAttn / CrossAttn3
│   │       ├── metrics.py                # CV 요약·DeLong 검정
│   │       ├── visualize.py              # PNG + results.md 저장
│   │       └── eda_distribution.py       # 데이터 분포 EDA
│   ├── data/강남/
│   │   ├── 강남_merged_features.xlsx     # metadata + aec_128 / aec_256 시트
│   │   ├── metadata/                     # 원본 SMI 결과
│   │   └── aec/                          # 원본·크롭 AEC 데이터
│   ├── results/
│   │   ├── ok/                           # AEC 보유 환자만 포함한 실험 결과
│   │   │   ├── model_1/
│   │   │   ├── model_2/{loss}/aec{N}/{variant}/
│   │   │   ├── model_2_2/{loss}/aec{N}/{variant}/
│   │   │   ├── model_3/{loss}/aec{N}/{variant}/
│   │   │   └── comparison/{loss}/aec{N}/
│   │   ├── ok+missing/                   # AEC missing 환자 포함 실험 결과 (동일 구조)
│   │   └── scaling_comparison_{loss}_aec{N}.md  # 전 모델 비교 테이블
│   └── generate_ppt.py                   # PPTX 보고서 자동 생성
└── 연구자료/                              # 발표자료, 논문 참고자료 등
```

---

## 데이터

| 구분 | 최종 인원 | 남             | 여              | 스캐너 종 | 주요 kVp        |
|------|-----------|----------------|-----------------|-----------|-----------------|
| 강남 | 1,673명   | 665명 (39.7%)  | 1,008명 (60.3%) | 31종      | 100 kVp (93.3%) |

**입력 파일 시트 구성** (`강남_merged_features.xlsx`):
- `metadata`: PatientID, PatientAge, PatientSex, BMI, SMI, kVp, ManufacturerModelName
- `aec_128` / `aec_256`: PatientID + 보간된 AEC 시퀀스 128/256pt

**Sarcopenia 기준 (SMI, cm²/m², AWGS 2019)**:
- 남성: SMI ≤ 40.96 → sarcopenia=1
- 여성: SMI ≤ 30.6 → sarcopenia=1

**데이터셋 구분**:
- `ok`: AEC 파일이 존재하는 환자만
- `ok+missing`: AEC 없는 환자도 포함 (M1 Clinic Only 실험에 활용)

---

## 데이터 전처리 파이프라인

```text
[DICOM] → [TotalSegmentator → SMI 계산] → 강남_DLO_Results_SMI.xlsx
              ↓
[AEC mA 시계열 추출] → 강남_aec_raw.xlsx
              ↓ 3_extract_z_bounds.py (L3 슬라이스 z-range 추출)
              ↓ 4_crop_aec.py         (L3 구간 크롭 → 128/256pt 보간)
          강남_aec_cropped_{ok|ok+missing}.xlsx
              ↓ merged_features.py    (metadata + AEC 병합, kVp 필터, 소수 제조사 제거)
          강남_merged_features.xlsx
```

---

## Project 2 — AEC Curve 기반 SMI 이진 분류 (딥러닝)

### 모델 구성

| 모델   | 입력                                  | 서브모델    | 목적                            |
|--------|---------------------------------------|-------------|---------------------------------|
| M1     | Clinic (Age, Sex, BMI)                | LR          | 임상 기준선 성능                |
| M2     | Clinic + AEC (Matched)                | CrossAttn   | AEC 추가 예측력 검증            |
| M2_2   | Clinic + AEC (Unmatched, 음성 대조군) | CrossAttn   | M2 > M2_2 → Clinic-AEC 대응 의미 있음 |
| M3     | Clinic + Scanner (MFR Embedding) + AEC| CrossAttn3  | 스캐너 보정 효과 검증           |

### 실행 방법

```bash
cd 연구코드/code/0522/model

python main.py
```

- Model 1은 1회 실행 후 Model 2/2_2/3을 `AEC_SIZES × LOSS_TYPES` 조합으로 반복 실행
- M2/M2_2/M3는 `ProcessPoolExecutor(max_workers=3)`로 병렬 실행
- 결과는 `연구코드/results/{ok|ok+missing}/`에 저장

### 실험 조합

| 축          | 값                                              |
|-------------|-------------------------------------------------|
| AEC 해상도  | 128pt, 256pt                                    |
| Loss 유형   | `bce` (BCEWithLogitsLoss), `focal` (FocalLoss)  |
| AEC 변환    | `norm`, `excl_extreme`, `len128`, `crop80`, `crop60` |
| 스케일링    | `scale_clinic` (Age·BMI StandardScaler, AEC raw) |

### AEC 변환 5종

| 변환           | 설명                                              |
|----------------|---------------------------------------------------|
| `norm`         | 곡선 내 z-score 정규화 (스캐너 간 절대값 차이 제거) |
| `excl_extreme` | scan-length 상하위 5% 극단 샘플 제외              |
| `len128`       | 원본 길이와 무관하게 128pt로 보간                  |
| `crop80`       | 중앙 80% 구간 (양끝 10% 제거)                     |
| `crop60`       | 중앙 60% 구간 (양끝 20% 제거)                     |

### 모델 아키텍처

| 모델        | 입력                     | 아키텍처                          |
|-------------|--------------------------|-----------------------------------|
| LR (M1)     | Age, sex_enc, BMI        | LogisticRegression                |
| CrossAttn (M2/M2_2) | Clinic / AEC 분리 | Bidirectional Cross-Attention   |
| CrossAttn3 (M3) | Clinic / MFR Emb / AEC | Bidirectional Cross-Attention  |

**CrossAttn 구조** (`ClinAECCrossAttn`):
- `ScalarFeatureTokenizer`: 각 scalar feature → d_model 독립 토큰
- `ResNet1DEncoder`: AEC 시퀀스 → (B, n_tokens, d_model) 토큰
- `CrossAttentionBlock`: Pre-norm Cross-Attention + FFN + Dropout (양방향)

**CrossAttn3 구조** (`ClinAECScanCrossAttn`):
- `MfrTokenizer`: ManufacturerModelName 정수 → Embedding 토큰 추가

---

## Code Flow — `python main.py` 실행 워크플로우

### 1단계 — 데이터 로드 및 분할

```
run_all_cases()
│
├─ load_data()                    → X (N,3), y, sex            # M1용 (Age, sex_enc, BMI)
├─ load_data_with_aec()           → X_clin, X_aec, y, sex      # M2용 (Clinic + AEC Matched)
├─ load_data_with_aec_unmatched() → X_clin, X_aec, y, sex      # M2_2용 (AEC 행 순서 셔플)
└─ load_data_with_aec_meta()      → X_clin, X_aec, X_mfr, ...  # M3용 (+ ManufacturerModelName)
```

**공통 전처리**:
- kVp·소수 제조사 필터는 `merged_features.py`에서 사전 적용 (xlsx에 저장)
- SMI 이진 레이블 생성 (M: ≤40.96, F: ≤30.6)
- `stratify=label×sex×age_bin×bmi_bin`, `TEST_SIZE=0.2`, `SEED=42`

### 2단계 — Loss × AEC 크기별 반복, M2/M2_2/M3 병렬 실행

```
for aec_size in [128, 256]:
    for loss_type in ["bce", "focal"]:
        ProcessPoolExecutor(max_workers=3)
        ├─ _run_model2(...)    # 5 AEC variants × 1 case
        ├─ _run_model2_2(...)  # 5 AEC variants × 1 case
        └─ _run_model3(...)    # 5 AEC variants × 1 case
```

### 3단계 — 각 모델 내부 (CrossAttn 기준)

```
for aec_var in AEC_VARIANTS:      # 5가지 변환
    for case_name, sc in CASES:   # 1가지 스케일링
        run_cross_validation_cross(...)    # 5-Fold CV
        evaluate_test_cross(...)           # Test set 평가
        save_all_cross(...)                # PNG + results.md
        plot_attention_maps(...)           # Attention Map 시각화
```

### 4단계 — Cross-Validation (M2/M2_2/M3)

```
StratifiedKFold(n_splits=5)
└─ for fold in 5 folds:
    ├─ StandardScaler: Age·BMI만 적용 (sex_enc, AEC는 미적용)
    └─ CrossAttn / CrossAttn3
        ├─ BCEWithLogitsLoss(pos_weight) 또는 FocalLoss(gamma=2)
        ├─ Adam(lr=1e-3) + CosineAnnealingLR
        └─ best val AUC epoch에서 가중치 스냅샷 저장
```

### 5단계 — Test Set 최종 평가

CV fold best epoch의 **중앙값(median)**으로 전체 CV 세트 재학습 → test set 예측

```
evaluate_test_cross / evaluate_test_cross3
└─ 지표: AUC, AUPRC, Brier, Accuracy, F1
   ├─ Bootstrap 95% CI (n_boot=2000)
   └─ 성별 분리 AUC
```

### 6단계 — 시각화 및 보고서 저장

```
save_all / save_all_cross
├─ cv_roc_curves.png          # fold별 ROC
├─ cv_metric_distribution.png # CV 지표 분포
├─ training_curves.png        # 학습 커브
├─ confusion_matrices.png     # 혼동행렬
├─ test_roc_curves.png        # Test ROC (baseline 비교 포함)
├─ test_roc_by_sex.png        # 성별 분리 ROC
├─ calibration.png            # Calibration plot
└─ results.md                 # 수치 보고서

plot_attention_maps
├─ attention_map_c2a.png      # 클래스별 토큰 평균 bar + AEC 신호 오버레이
└─ attention_heatmap.png      # 샘플별 heatmap (Sarco→Normal 순 정렬)
```

### 7단계 — 모델 간 비교 결과 저장

```
_save_comparison_md(results_m1, results_m2, results_m2_2, results_m3, aec_size, loss_type)
├─ Best Cases 요약 테이블 (Test AUC 기준)
├─ 모델별 전체 케이스 성능 테이블 (AUC, AUPRC, Brier, Acc, F1)
├─ Fold-level 통계 검정 (paired t-test + Wilcoxon, n=5)
│   ├─ M1 LR vs M2 CrossAttn     (AEC variant별)
│   ├─ M2 CrossAttn vs M3 CrossAttn3
│   └─ M1 LR vs M3 CrossAttn3
├─ Bootstrap 95% CI (n_boot=2000, 전 모델·전 지표)
└─ DeLong Test (Test-set ROC AUC 쌍별 비교)
    ├─ M1 LR vs M2 CrossAttn
    ├─ M1 LR vs M3 CrossAttn3
    ├─ M2 Matched vs M2_2 Unmatched
    └─ M2 CrossAttn vs M3 CrossAttn3

→ 저장: results/scaling_comparison_{loss_type}_aec{N}.md
→ 저장: results/{ok|ok+missing}/comparison/{loss}/{aec{N}}/roc_all_models_{aec_var}.png
```

---

## 출력 디렉토리 구조

```
results/ok/ (또는 ok+missing/)
├─ model_1/
│   ├─ *.png (7종)
│   ├─ results.md
│   └─ run.log
├─ model_2/
│   └─ {loss_type}/            # bce / focal
│       └─ aec{N}/             # aec128 / aec256
│           ├─ run.log
│           └─ {aec_var}/      # norm, excl_extreme, len128, crop80, crop60
│               ├─ *.png (8종)
│               ├─ attention_map_c2a.png
│               ├─ attention_heatmap.png
│               └─ results.md
├─ model_2_2/  (동일 구조)
├─ model_3/    (동일 구조)
└─ comparison/
    └─ {loss_type}/aec{N}/
        └─ roc_all_models_{aec_var}.png

results/
└─ scaling_comparison_{loss_type}_aec{N}.md
```

---

## 주요 하이퍼파라미터

| 파라미터   | 값   |
|------------|------|
| N_FOLDS    | 5    |
| EPOCHS     | 200  |
| BATCH_SIZE | 32   |
| LR_RATE    | 1e-3 |
| HIDDEN     | 64   |
| N_BLOCKS   | 4    |
| N_HEADS    | 4    |
| TEST_SIZE  | 0.2  |
| SEED       | 42   |

---

## 보고서 생성

```bash
cd 연구코드

python generate_ppt.py
```

`results/ok/` 하위 결과 이미지를 수집해 PPTX 슬라이드를 자동 생성한다.

---

## 의존성

```text
torch, scikit-learn, numpy, pandas
scipy, matplotlib, seaborn
openpyxl, python-pptx
tqdm
```
