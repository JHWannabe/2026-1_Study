# AEC 신호 기반 근감소증 예측 연구

**연세대학교 의과대학 강남세브란스병원 의료기기공학 및 관리학과**

CT 촬영 시 스캐너가 자동으로 조절하는 관전류(mA) 시계열(AEC 신호)을 활용해 근감소증(Sarcopenia) 및 TAMA를 예측하는 연구. 두 가지 접근법을 비교한다.

---

## 연구 개요

**핵심 가설**: AEC 신호는 성별·나이가 설명하지 못하는 추가적인 체형 정보를 담고 있으며, 다변량 맥락에서 근육량 예측력을 유의미하게 향상시킨다.

| 구분 | 대상 | 방법 | 코드 위치 |
|------|------|------|-----------|
| Project 1 | TAMA 예측 (회귀) | 선형·로지스틱 회귀 | `연구코드/aec/` |
| Project 2 | SMI 이진 분류 (근감소증) | LR / ResNet1D / CrossAttn | `연구코드/code/0515/model/` |

---

## 디렉토리 구조

```text
2026-1_Study/
├── 연구코드/
│   ├── aec/                            # [Project 1] AEC feature → TAMA 회귀분석
│   │   ├── code/
│   │   │   ├── config.py
│   │   │   ├── run_analysis.py
│   │   │   ├── data_loader.py
│   │   │   ├── feature_selection.py
│   │   │   ├── linear_regression.py
│   │   │   ├── logistic_regression.py
│   │   │   ├── multivariable_analysis.py
│   │   │   ├── generate_plots.py
│   │   │   ├── generate_report.py
│   │   │   └── generate_ppt.py
│   │   ├── data/
│   │   │   ├── 강남_merged_features.xlsx
│   │   │   └── 신촌_merged_features.xlsx
│   │   └── results/
│   │       ├── 강남/
│   │       └── 신촌/
│   ├── code/0515/model/                # [Project 2] AEC curve → SMI 이진 분류
│   │   ├── main.py                     # 진입점 — run_all_cases()
│   │   ├── config.py                   # 전역 하이퍼파라미터·경로·상수
│   │   ├── data.py                     # 데이터 로드·전처리·AEC 변환
│   │   ├── cross_val.py                # Stratified K-Fold CV (M1/M2/M3)
│   │   ├── evaluate.py                 # Test set 최종 평가
│   │   ├── models.py                   # ResNet1D / CrossAttn / CrossAttn3
│   │   ├── metrics.py                  # CV 요약·fold 통계 비교
│   │   ├── visualize.py                # 8종 PNG + results.md 저장
│   │   └── code_flow.md                # 실행 워크플로우 상세 문서
│   ├── data/강남/
│   │   └── 강남_merged_features.xlsx   # metadata + meta_aec256 시트
│   └── results/0515/
│       ├── model_1/                    # M1 Clinic Only 결과
│       ├── model_2/                    # M2 Clinic+AEC 결과
│       ├── model_2_2/                  # M2_2 Unmatched 결과
│       ├── model_3/                    # M3 Clinic+Scanner+AEC 결과
│       └── scaling_comparison.md       # 전 모델 비교 테이블
└── 연구자료/                           # 발표자료, 논문 참고자료 등
```

---

## 데이터

| 구분 | 최종 인원 | 남             | 여              | 스캐너 종 | 주요 kVp        |
|------|-----------|----------------|-----------------|-----------|-----------------|
| 강남 | 1,673명   | 665명 (39.7%)  | 1,008명 (60.3%) | 31종      | 100 kVp (93.3%) |
| 신촌 | 1,269명   | 637명 (50.2%)  | 632명 (49.8%)   | 46종      | 100 kVp (75.3%) |

**입력 파일 시트 구성** (`{SITE}_merged_features.xlsx`):
- `metadata-value` / `meta_aec256`: PatientID, PatientSex, PatientAge, TAMA/SMI, ManufacturerModelName, KVP, aec_1~aec_256

**Sarcopenia 기준 (SMI, cm²/m², AWGS 2019)**:
- 남성: SMI ≤ 40.96 → sarcopenia=1
- 여성: SMI ≤ 30.6 → sarcopenia=1

---

## Project 1 — AEC Feature 기반 TAMA 회귀분석

### 분석 파이프라인

```text
[DICOM/RAW CT]
      ↓ mA 시계열 추출
[feature_selection.py]      ← 65개 AEC 피처 × Pearson r + VIF 필터링
      ↓ 4개 선택 피처: mean, CV, skewness, slope_abs_mean
[data_loader.py]            ← metadata 병합 / Z-score 표준화 / One-hot 인코딩
      ↓
[linear_regression.py]      ← 단변량·다변량 OLS, 5-Fold CV, 잔차진단
[logistic_regression.py]    ← 단변량·다변량 Logit, Bootstrap AUC, HL 검정
[multivariable_analysis.py] ← Case 0→1→2→3 점진적 모델 비교
      ↓
[generate_plots.py]         ← 15개 PNG
[generate_report.py]        ← Markdown 보고서
[generate_ppt.py]           ← PPTX 보고서 자동 생성
```

### Case 구성

| Case   | 투입 변수                                            |
|--------|------------------------------------------------------|
| Case 0 | AEC 피처만 (Sex·Age 없음)                            |
| Case 1 | Sex + Age                                            |
| Case 2 | Sex + Age + AEC 피처 4개                             |
| Case 3 | Sex + Age + AEC 피처 + KVP + ManufacturerModelName   |

### 실행 방법

```bash
cd 연구코드/aec/code

# 전체 파이프라인 실행 (Step 1~7)
python run_analysis.py

# Feature Selection 건너뛰고 실행 (config.py 이미 설정된 경우)
python run_analysis.py --skip-fs
```

**사이트 전환**: `config.py`의 `SITE` 변수를 `"강남"` 또는 `"신촌"`으로 변경 후 재실행.

### 선택 AEC 피처 및 근거

| 피처             | 의미                              | 선택 이유                          |
|------------------|-----------------------------------|------------------------------------|
| `mean`           | AEC 신호 평균 (전반적 체격 크기)  | 강남 r=0.297, amplitude 그룹 대표  |
| `CV`             | 변동계수 (체형 불균일성)          | 체형 이질성 독립 정보              |
| `skewness`       | 신호 비대칭성 (체지방 분포)       | 분포 편향 정보                     |
| `slope_abs_mean` | 평균 절대 기울기 (공간적 변화율)  | 다변량 시 독립 기여 확인           |

제외 피처: `p25`, `AUC_normalized`, `peak_max_height` → mean과 VIF > 50,000 (amplitude 중복)

### 주요 결과 요약

#### 강남 (n=1,673)

| 지표         | Case 1    | Case 2         | Case 3 |
|--------------|-----------|----------------|--------|
| Linear R²    | 0.551     | 0.636 (+0.085) | 0.660  |
| Linear RMSE  | 20.43 cm² | 18.40 cm²      | —      |
| Logistic AUC | 0.624     | 0.720 (+0.096) | 0.751  |
| HL p-value   | —         | —              | 0.601  |
| NPV          | —         | —              | 0.887  |

#### 신촌 (n=1,269)

| 지표         | Case 1 | Case 2        | Case 3 |
|--------------|--------|---------------|--------|
| Linear R²    | 0.520  | 0.548 (+0.028)| 0.590  |
| Logistic AUC | 0.610  | 0.650 (+0.040)| 0.728  |
| HL p-value   | —      | —             | 0.731  |

신촌은 스캐너 이질성(46종)이 커서 AEC 기여(+0.040)보다 스캐너 보정(+0.078) 효과가 더 크게 나타남.

---

## Project 2 — AEC Curve 기반 SMI 이진 분류 (딥러닝)

### 모델 구성

| 모델 | 입력 | 서브모델 |
|------|------|---------|
| M1 | Clinic (Age, Sex, BMI) | LR / ResNet1D |
| M2 | Clinic + AEC Matched | LR / CrossAttn / ResNet1D |
| M2_2 | Clinic + AEC Unmatched (음성 대조군) | LR / CrossAttn / ResNet1D |
| M3 | Clinic + Scanner (MFR Embedding) + AEC | LR / CrossAttn3 / ResNet1D |

M2 > M2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 가진다는 증거.

### 실행 방법

```bash
cd 연구코드/code/0515/model

python main.py
```

Model 1/2/2_2/3을 `ProcessPoolExecutor(max_workers=4)`로 병렬 실행. 결과는 `연구코드/results/0515/`에 저장되고, 로그는 각 모델 디렉토리의 `run.log`에 기록된다.

### 총 실험 수

| 모델 | AEC variants | 스케일링 케이스 | 서브모델 | 총 실험 |
|------|:-----------:|:-----------:|:------:|:------:|
| M1   | 1           | 1           | LR + ResNet1D | 1 |
| M2   | 7           | 2           | LR + CrossAttn + ResNet1D | 14 |
| M2_2 | 7           | 2           | LR + CrossAttn + ResNet1D | 14 |
| M3   | 7           | 2           | LR + CrossAttn3 + ResNet1D | 14 |

> 각 실험마다 5-Fold CV + Test 평가 수행

---

## Code Flow — `python main.py` 실행 워크플로우

### 1단계 — 데이터 로드 및 분할

```
run_all_cases()
│
├─ load_data()                    → X (N,3), y, sex            # M1용 (Age, sex_enc, BMI)
├─ load_data_with_aec()           → X_clin, X_aec, y2, sex2   # M2용 (Clinic + AEC Matched)
├─ load_data_with_aec_unmatched() → X_clin_u, X_aec_u, ...    # M2_2용 (AEC 행 순서 셔플)
└─ load_data_with_aec_meta()      → X_clin3, X_aec3, X_mfr,   # M3용 (+ ManufacturerModelName)
                                     y3, sex3, n_mfr
```

**공통 전처리** (4개 로드 함수 모두 적용):
1. kVp == 100 필터 (다른 kVp는 AEC 신호 특성 상이)
2. 소수 제조사 제거 (비율 < `MIN_MFR_RATIO` = 5%)
3. SMI 이진 레이블 생성 (M: ≤40.96 → sarcopenia=1, F: ≤30.6 → sarcopenia=1)

**분할** (`stratify=y`, `TEST_SIZE=0.2`, `SEED=42`):
```
split_data()      → X_cv / X_te              (M1)
split_data_dual() → X_clin_cv/te, X_aec_cv/te (M2, M2_2)
split_data_quad() → X_clin_cv/te, X_aec_cv/te, X_mfr_cv/te (M3)
```

### 2단계 — 4개 모델 병렬 실행

```
ProcessPoolExecutor(max_workers=4)
├─ fut1   = _run_model1(...)     # 별도 프로세스
├─ fut2   = _run_model2(...)     # 별도 프로세스
├─ fut2_2 = _run_model2_2(...)   # 별도 프로세스
└─ fut3   = _run_model3(...)     # 별도 프로세스
```

각 워커는 stdout을 `io.StringIO`로 캡처 → 완료 후 `run.log`에 저장

### 3단계 — 각 모델 워커 내부 루프

#### Model 1

```
for case_name, sc in CASES_M1:          # 1가지 케이스
    run_cross_validation(X_cv, y_cv, scale_X=sc)
    evaluate_test(...)
    save_all(...)
```

#### Model 2 / Model 2_2

```
for aec_var in AEC_VARIANTS:            # 7가지 AEC 변환
    aec_variant(X_aec_cv, aec_var)      → X_aec_cv_v, mask_cv
    for case_name, sc, sa in CASES_M2:  # 2가지 스케일링 케이스
        run_cross_validation_cross(...)
        evaluate_test_cross(...)
        save_all_cross(...)
```

> Model 2_2는 AEC 행 순서가 셔플된 `X_aec_u`를 입력으로 사용 (음성 대조군)

#### Model 3

```
for aec_var in AEC_VARIANTS:            # 7가지 AEC 변환
    aec_variant(X_aec3_cv, aec_var)     → X_aec3_cv_v, mask_cv
    for case_name, sc, sa in CASES_M3:  # 2가지 스케일링 케이스
        run_cross_validation_cross3(...)
        evaluate_test_cross3(...)
        save_all_cross(...)
```

### 4단계 — Cross-Validation 내부

#### `run_cross_validation` (M1)

```
StratifiedKFold(n_splits=5)
└─ for fold in 5 folds:
    ├─ _maybe_scale_clin(X_tr, X_val, scale_X)   # Age·BMI만 표준화 (sex_enc 제외)
    ├─ [LR] LogisticRegression.fit / predict
    └─ [ResNet1D]
        ├─ build_resnet(y_tr)      # BCEWithLogitsLoss(pos_weight) + Adam + CosineAnnealingLR
        └─ for ep in 1..EPOCHS(200):
            ├─ train_one_epoch → val_auc
            └─ if val_auc > best: 가중치 스냅샷 저장
```

#### `run_cross_validation_cross` (M2 / M2_2)

```
StratifiedKFold(n_splits=5)
└─ for fold in 5 folds:
    ├─ _maybe_scale_clin(X_clin, scale_clin)   # Age·BMI만 표준화
    ├─ _maybe_scale(X_aec, scale_aec)          # AEC 전 컬럼 표준화
    ├─ X_lr = hstack([X_clin_s, X_aec_s])
    ├─ [LR] LogisticRegression
    ├─ [CrossAttn] ClinAECCrossAttn (Bidirectional Cross-Attention)
    └─ [ResNet1D] ResNet1D (X_lr 입력)
```

#### `run_cross_validation_cross3` (M3)

```
StratifiedKFold(n_splits=5)
└─ for fold in 5 folds:
    ├─ X_lr = hstack([X_clin_s, X_mfr.reshape(-1,1), X_aec_s])
    ├─ [LR] LogisticRegression
    ├─ [CrossAttn3] ClinAECScanCrossAttn
    │   └─ MfrTokenizer: ManufacturerModelName 정수 → Embedding 토큰
    └─ [ResNet1D] ResNet1D (X_lr 입력)
```

### 5단계 — Test Set 최종 평가

CV fold best epoch의 **중앙값(median)**을 사용해 전체 CV 세트로 재학습 → test set 예측

```
evaluate_test / evaluate_test_cross / evaluate_test_cross3
└─ _scale_clin_te / _scale_or_copy
   ├─ LR: fit(X_cv_s) → predict(X_te_s)
   ├─ CrossAttn: for _ in range(med_epoch): train → eval
   └─ ResNet1D:  for _ in range(rn_med_epoch): train → eval
```

### 6단계 — 시각화 및 보고서 저장

```
save_all / save_all_cross
├─ plot_data_distribution(...)  → data_distribution.png
├─ plot_roc_curves(...)         → cv_roc_curves.png
├─ plot_metric_distribution(...)→ cv_metric_distribution.png
├─ plot_confusion_matrices(...) → confusion_matrices.png
├─ plot_training_curves(...)    → training_curves.png
├─ plot_test_roc(...)           → test_roc_curves.png
├─ plot_test_roc_by_sex(...)    → test_roc_by_sex.png
├─ plot_calibration(...)        → calibration.png
└─ save_report_md(...)          → results.md
```

### 7단계 — 모델 간 비교 결과 저장

```
_save_comparison_md(results_m1, results_m2, results_m2_2, results_m3)
├─ Best Cases 요약 테이블
├─ 모델별 전체 케이스 성능 테이블
├─ Fold-level 통계 검정 (paired t-test + Wilcoxon)
│   ├─ M1: LR vs ResNet1D
│   ├─ M2: LR vs CrossAttn / LR vs ResNet1D
│   ├─ M2_2: LR vs CrossAttn (Unmatched)
│   └─ M3: LR vs CrossAttn3 / LR vs ResNet1D
└─ Cross-model 비교 (M1→M2, M1→M3, M2→M2_2, M2→M3)

→ 저장: results/0515/scaling_comparison.md
```

### AEC 변환 7종

| 변환 | 설명 |
|------|------|
| `len064` | 256→64점 보간 (해상도 저하 영향) |
| `len128` | 256→128점 보간 |
| `len256` | 원본 256점 full curve (baseline) |
| `crop80` | 중앙 80% 구간 (양끝 10% 제거) |
| `crop60` | 중앙 60% 구간 (양끝 20% 제거) |
| `norm`   | 곡선 내 z-score 정규화 (스캐너 간 절대값 차이 제거) |
| `excl_extreme` | scan-length 상하위 5% 극단 샘플 제외 |

### 모델 아키텍처 참고

| 모델 | 입력 | 아키텍처 | 파일 |
|------|------|---------|------|
| LR (M1) | Age, sex_enc, BMI | LogisticRegression | `cross_val.py` |
| ResNet1D (M1) | Age, sex_enc, BMI | Conv1D ResNet → FC | `models.py` |
| LR (M2) | Clinic + AEC (hstack) | LogisticRegression | `cross_val.py` |
| CrossAttn (M2) | Clinic / AEC 분리 | Bidirectional Cross-Attention | `models.py` |
| ResNet1D (M2) | Clinic + AEC (hstack) | Conv1D ResNet → FC | `models.py` |
| LR (M3) | Clinic + MFR + AEC | LogisticRegression | `cross_val.py` |
| CrossAttn3 (M3) | Clinic / MFR Emb / AEC | Bidirectional Cross-Attention | `models.py` |
| ResNet1D (M3) | Clinic + MFR + AEC | Conv1D ResNet → FC | `models.py` |

### 출력 디렉토리 구조

```
results/0515/
├─ model_1/
│   └─ scale_clinic/
│       ├─ *.png (8종)
│       ├─ results.md
│       └─ ../run.log
├─ model_2/
│   └─ {aec_var}/               # len064, len128, ..., excl_extreme
│       └─ {case}/              # scale_clinic, scale_both
│           ├─ *.png + results.md
│           └─ ../../run.log
├─ model_2_2/   (동일 구조)
├─ model_3/     (동일 구조)
└─ scaling_comparison.md        # 전 모델 비교 테이블
```

---

## 향후 계획

1. **외부 검증**: 강남 → 신촌 교차 검증 (스캐너 이질성 영향 규명)
2. **Center-stratified 분석**: skewness 방향이 강남(−)·신촌(+) 반전된 원인 규명
3. **BMI 확장 데이터셋**: 신장·체중·BMI 추가 (신촌 결측 0.13% / 강남 결측 18.9% → 처리 전략 필요)
4. **ML 앙상블 모델**: Random Forest, XGBoost와 딥러닝 모델 성능 비교

---

## 의존성

```text
# Project 1
pandas, numpy, scipy, statsmodels, sklearn
openpyxl, python-pptx, matplotlib, pywt

# Project 2 (추가)
torch, scikit-learn
```
