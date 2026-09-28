# AEC 신호 기반 근감소증 예측 연구

**연세대학교 의과대학 강남세브란스병원 의료기기산업학과**

CT 촬영 시 스캐너가 자동으로 조절하는 관전류(mA) 시계열(AEC 신호)을 활용해 근감소증(Sarcopenia)을 예측하는 연구.

---

## 연구 개요

**핵심 가설**: AEC 신호는 성별·나이가 설명하지 못하는 추가적인 체형 정보를 담고 있으며, 다변량 맥락에서 SMI 이진 분류 예측력을 유의미하게 향상시킨다.

**실험 축**:

- **AEC 모드**: `raw128` (128pt 원본) vs `crop80` (중앙 103pt 크롭)
- **AEC 변환**: 4종 민감도 분석
- **스테이지 실험**: 하이퍼파라미터 단계별 독립 비교 (`EXPERIMENT_STAGE` 환경변수)

---

## 디렉토리 구조

```text
2026-1_Study/
├── 연구코드/
│   ├── code/
│   │   ├── data/                        # 데이터 전처리 파이프라인
│   │   │   ├── metadata/
│   │   │   │   ├── 0_unique.py          # 중복 PatientID 확인
│   │   │   │   └── 1_verify_metadata.py # metadata 검증
│   │   │   └── aec/
│   │   │       ├── 3_extract_z_bounds_liver.py # z-range 추출
│   │   │       └── 4_crop_aec.py               # AEC 신호 크롭·보간
│   │   ├── model/                        # AEC curve → SMI 이진 분류
│   │   │   ├── main.py                   # 진입점 — run_all_cases()
│   │   │   ├── config.py                 # 전역 하이퍼파라미터·경로·스테이지 정의
│   │   │   ├── data.py                   # 데이터 로드·분할·AEC 변환
│   │   │   ├── cross_val.py              # Stratified K-Fold CV (M1/M2/M3)
│   │   │   ├── evaluate.py               # Test set 최종 평가
│   │   │   ├── models.py                 # FocalLossWithLogits / ResNet1D / CrossAttn / CrossAttn3
│   │   │   ├── metrics.py                # CV 요약·DeLong 검정
│   │   │   ├── visualize.py              # PNG + results.md 저장 + Grad-CAM
│   │   │   ├── experiments.py            # 다중 스테이지 자동 실행
│   │   │   └── eda_distribution.py       # 데이터 분포 EDA
│   │   └── generate_ppt.py              # PPTX 보고서 자동 생성
│   ├── data/강남/
│   │   ├── 강남_liver_merged_features_ok.xlsx  # metadata + aec_128 시트
│   │   └── metadata/                           # 원본 SMI 결과
│   └── results/
│       └── {date}/stage{N}_{lr}_epoch{EPOCHS}/ # 스테이지별 결과 디렉토리
│           ├── model_1/
│           ├── model_2/{mode}/{aec_var}/
│           ├── model_2_2/{mode}/{aec_var}/
│           ├── model_3/{mode}/{aec_var}/
│           └── comparison/{mode}/
└── 연구자료/                              # 발표자료, 논문 참고자료 등
```

---

## 데이터

| 구분 | 최종 인원 | 남             | 여              | 스캐너 종 | 주요 kVp        |
|------|-----------|----------------|-----------------|-----------|-----------------|
| 강남 | 1,673명   | 665명 (39.7%)  | 1,008명 (60.3%) | 31종      | 100 kVp (93.3%) |

**입력 파일** (`강남_liver_merged_features_ok.xlsx`):
- `metadata`: PatientID, PatientAge, PatientSex, BMI, SMI, kVp, ManufacturerModelName
- `aec_128`: PatientID + 보간된 AEC 시퀀스 128pt

**Sarcopenia 기준 (SMI, cm²/m², AWGS 2019)**:
- 남성: SMI ≤ 40.96 → sarcopenia=1
- 여성: SMI ≤ 30.6 → sarcopenia=1

---

## 데이터 전처리 파이프라인

```text
[DICOM] → [TotalSegmentator → SMI 계산] → 강남_DLO_Results_SMI.xlsx
              ↓
[AEC mA 시계열 추출] → 강남_aec_raw.xlsx
              ↓ 3_extract_z_bounds_liver.py  (L3 슬라이스 z-range 추출)
              ↓ 4_crop_aec.py                (L3 구간 크롭 → 128pt 보간)
              ↓ 5_merged_features.py         (metadata + AEC 병합, kVp 필터, 소수 제조사 제거)
          강남_liver_merged_features_ok.xlsx
```

---

## 모델 구성

| 모델 | 입력 | 서브모델 | 목적 |
| --- | --- | --- | --- |
| M1 | Clinic (Age, Sex, BMI) | LR | 임상 기준선 성능 |
| M2 | Clinic + AEC (Matched) | CrossAttn | AEC 추가 예측력 검증 |
| M2_2 | Clinic + AEC (Unmatched, 음성 대조군) | CrossAttn | M2 > M2_2 → Clinic-AEC 대응 의미 있음 |
| M3 | Clinic + Scanner (MFR Embedding) + AEC | CrossAttn3 | 스캐너 보정 효과 검증 |

---

## 실행 방법

```bash
cd 연구코드/code/model

# 기본 실행 (Stage 0)
python main.py

# 특정 스테이지 실행
EXPERIMENT_STAGE=1 python main.py

# 다중 스테이지 자동 실행
python experiments.py --stages 0,1,2
```

- Model 1은 1회 실행 후 Model 2/2_2/3을 `AEC_MODES × AEC_VARIANTS` 조합으로 반복 실행
- M2/M2_2/M3는 `ProcessPoolExecutor(max_workers=3)`로 병렬 실행
- 결과는 `연구코드/results/{date}/stage{N}_{lr}_epoch{EPOCHS}/`에 저장

---

## 실험 조합

| 축         | 값                                               |
|------------|--------------------------------------------------|
| AEC 모드   | `raw128` (128pt 원본), `crop80` (중앙 103pt 크롭) |
| AEC 변환   | `raw`, `std_scaled`, `norm`, `global_zscore`     |

### AEC 변환 4종

| 변환            | 스케일 모드    | 설명                                              |
|-----------------|---------------|---------------------------------------------------|
| `raw`           | `none`        | 전처리 없음                                        |
| `std_scaled`    | `column`      | 열 방향 StandardScaler (시점별 정규화)             |
| `norm`          | `none`        | 행 방향 z-score (환자별 정규화, 사전 적용)         |
| `global_zscore` | `global`      | Train set 전체 단일 μ/σ로 정규화                  |

---

## 하이퍼파라미터 스테이지 시스템

`config.py`의 `EXPERIMENT_STAGES`에 정의. 각 스테이지는 Stage 0 대비 변수 하나만 변경해 독립 비교한다.

| 파라미터    | Stage 0 (기본) |
|-------------|---------------|
| N_FOLDS     | 5             |
| EPOCHS      | 500           |
| BATCH_SIZE  | 32            |
| LR_RATE     | 1e-5          |
| HIDDEN      | 16            |
| N_HEADS     | 1             |
| N_BLOCKS    | 2             |
| N_CA_LAYERS | 2             |
| GRAD_CLIP   | 0.0           |
| FOCAL_GAMMA | 2.0           |
| TEST_SIZE   | 0.2           |
| SEED        | 42            |

환경변수 `EXPERIMENT_STAGE`(기본값 0)로 스테이지 선택.

---

## 모델 아키텍처

### 손실함수

전 모델 공통으로 **FocalLossWithLogits** (Focal Loss, gamma=`FOCAL_GAMMA`) + `pos_weight` (클래스 불균형 보정) 사용.

### CrossAttn 구조 (`ClinAECCrossAttn`, M2/M2_2)

- `ScalarFeatureTokenizer`: 각 scalar feature → `d_model` 독립 토큰 (B, 3, d_model)
- `ResNet1DEncoder`: AEC 시퀀스 → (B, n_aec_tokens=32, d_model) 토큰
- `CrossAttentionBlock` × `N_CA_LAYERS`: Pre-norm Cross-Attention + FFN + Dropout (양방향)
  - 방향 1: Clinical → Query, AEC → Key/Value
  - 방향 2: AEC → Query, Clinical → Key/Value
- Classifier: `[c_tokens.mean | a_tokens.mean]` → Linear(d_model×2 → 1)

### CrossAttn3 구조 (`ClinAECScanCrossAttn`, M3)

- `MfrTokenizer`: ManufacturerModelName 정수 → Embedding 토큰 (B, 1, d_model)
- Clinical(3) + Scanner(1) 토큰 concat → (B, 4, d_model)
- AEC 인코딩·Cross-Attention 구조는 CrossAttn과 동일

---

## Code Flow — `python main.py` 실행 워크플로우

### 1단계 — 데이터 로드 및 분할

```
run_all_cases()
│
├─ load_data()                    → X (N,3), y, sex             # M1용 (Age, sex_enc, BMI)
├─ load_data_with_aec()           → X_clin, X_aec, y, sex       # M2용 (Clinic + AEC Matched)
├─ load_data_with_aec_unmatched() → X_clin, X_aec, y, sex       # M2_2용 (AEC 행 순서 셔플)
└─ load_data_with_aec_meta()      → X_clin, X_aec, X_mfr, ...   # M3용 (+ ManufacturerModelName)
```

**공통 전처리**:
- kVp·소수 제조사 필터는 `merged_features.py`에서 사전 적용 (xlsx에 저장)
- SMI 이진 레이블 생성 (M: ≤40.96, F: ≤30.6)
- `stratify=label×sex×age_bin×bmi_bin`, `TEST_SIZE=0.2`, `SEED=42`

### 2단계 — AEC 모드 × AEC 변환, M2/M2_2/M3 병렬 실행

```
for mode_name, mode_cfg in AEC_MODES.items():    # raw128, crop80
    ProcessPoolExecutor(max_workers=3)
    ├─ _run_model2(...)    # 4 AEC variants
    ├─ _run_model2_2(...)  # 4 AEC variants
    └─ _run_model3(...)    # 4 AEC variants
```

### 3단계 — 각 모델 내부 (CrossAttn 기준)

```
for aec_var in AEC_VARIANTS:      # 4가지 변환
    run_cross_validation_cross(...)    # 5-Fold CV
    evaluate_test_cross(...)           # Test set 평가
    save_all_cross(...)                # PNG + results.md
    plot_attention_maps(...)           # Attention Map 시각화 (4종)
    plot_cam_aec(...)                  # Grad-CAM 시각화 (3종)
```

### 4단계 — Cross-Validation (M2/M2_2/M3)

```
StratifiedKFold(n_splits=5)
└─ for fold in 5 folds:
    ├─ StandardScaler: Age·BMI만 적용 (sex_enc, AEC는 variant별 scale_mode 결정)
    └─ CrossAttn / CrossAttn3
        ├─ FocalLossWithLogits(gamma=FOCAL_GAMMA, pos_weight)
        ├─ AdamW(lr=LR_RATE, weight_decay=1e-4) + CosineAnnealingLR
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
├─ calibration.png            # Calibration plot + Precision-Recall curve
└─ results.md                 # 수치 보고서

plot_attention_maps                   # Attention Map 4종
├─ attention_map_c2a.png              # Clinical→AEC: 클래스별 토큰 평균 bar + AEC 오버레이
├─ attention_heatmap_c2a.png          # Clinical→AEC: 샘플별 heatmap (Sarco→Normal 정렬)
├─ attention_map_a2c.png              # AEC→Clinical: 클래스별 clinical 토큰별 attention bar
└─ attention_heatmap_a2c.png          # AEC→Clinical: 샘플별 heatmap

plot_cam_aec                          # Grad-CAM 3종 (ResNet1DEncoder 마지막 블록)
├─ cam_aec_mean.png                   # 클래스별 평균 AEC ± std + 평균 CAM 배경 heatmap
├─ cam_aec_lines.png                  # 클래스별 10개 샘플 AEC 곡선 (tab10 색상)
└─ cam_aec_heatmap.png                # 전체 샘플 × AEC position heatmap (Sarco→Normal 정렬)
```

### 7단계 — 모델 간 비교 결과 저장

```
_save_comparison_md(results_m1, results_m2, results_m2_2, results_m3, aec_size, mode_name)
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

→ 저장: results/{date}/stage.../comparison/{mode_name}/scaling_comparison_{mode_name}.md
→ 저장: results/{date}/stage.../comparison/{mode_name}/roc_all_models_{aec_var}.png
```

---

## 출력 디렉토리 구조

```
results/{date}/stage{N}_{lr}_epoch{EPOCHS}/
├─ model_1/
│   ├─ *.png (7종)
│   ├─ results.md
│   └─ run.log
├─ model_2/
│   └─ {mode_name}/            # raw128 / crop80
│       ├─ run.log
│       └─ {aec_var}/          # raw, std_scaled, norm, global_zscore
│           ├─ *.png (7종 기본)
│           ├─ attention_map_c2a.png / attention_heatmap_c2a.png
│           ├─ attention_map_a2c.png / attention_heatmap_a2c.png
│           ├─ cam_aec_mean.png / cam_aec_lines.png / cam_aec_heatmap.png
│           └─ results.md
├─ model_2_2/  (동일 구조)
├─ model_3/    (동일 구조)
└─ comparison/
    └─ {mode_name}/
        ├─ aec_individual_normalization_compare.png
        ├─ roc_all_models_{aec_var}.png
        └─ scaling_comparison_{mode_name}.md
```

---

## 보고서 생성

```bash
cd 연구코드

python generate_ppt.py
```

결과 이미지를 수집해 PPTX 슬라이드를 자동 생성한다.

---

## 의존성

```text
torch, scikit-learn, numpy, pandas
scipy, matplotlib, seaborn
openpyxl, python-pptx
tqdm
```
