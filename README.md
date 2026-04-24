# AEC Feature 기반 TAMA 예측 회귀분석 연구

**연세대학교 의과대학 강남세브란스병원 의료기기공학 및 관리학과**
AEC(Automatic Exposure Control) 신호에서 추출한 피처로 TAMA(Total Abdominal Muscle Area)를 예측하고, Low Muscle 위험을 분류하는 회귀분석 연구.

---

## 연구 개요

CT 촬영 시 스캐너가 자동으로 조절하는 관전류(mA) 시계열(AEC 신호)에서 피처를 추출하여, 성별·나이와 독립적으로 TAMA 예측에 기여하는지 검증한다.

**핵심 가설**: AEC feature는 성별·나이가 설명하지 못하는 추가적인 체형 정보를 담고 있으며, 다변량 맥락에서 TAMA 예측력을 유의미하게 향상시킨다.

---

## 디렉토리 구조

```text
2026-1_Study/
├── 연구코드/aec/
│   ├── code/
│   │   ├── config.py                  # 전체 설정값 (사이트, 경로, 피처, 임계값)
│   │   ├── run_analysis.py            # 전체 파이프라인 일괄 실행
│   │   ├── data_loader.py             # 데이터 로드·전처리 공통 유틸리티
│   │   ├── feature_selection.py       # AEC 피처 상관분석 및 VIF 검사
│   │   ├── linear_regression.py       # 선형 회귀 (단변량 + 다변량)
│   │   ├── logistic_regression.py     # 로지스틱 회귀 (단변량 + 다변량)
│   │   ├── multivariable_analysis.py  # Case 1/2/3 모델 비교
│   │   ├── generate_plots.py          # 결과 시각화 (15개 PNG)
│   │   ├── generate_report.py         # Markdown 연구보고서 생성
│   │   └── generate_ppt.py            # PPTX 보고서 자동 생성
│   ├── data/
│   │   ├── 강남_merged_features.xlsx  # 강남 데이터 (metadata + features 시트)
│   │   └── 신촌_merged_features.xlsx  # 신촌 데이터
│   └── results/
│       ├── 강남/                      # 강남 분석 결과
│       │   ├── feature_selection_report.xlsx
│       │   ├── linear_results.xlsx
│       │   ├── logistic_results.xlsx
│       │   ├── multivariable_results.xlsx
│       │   ├── research_report.md
│       │   ├── 강남_TAMA_연구보고서.pptx
│       │   └── figures/               # 15개 PNG 그래프
│       └── 신촌/                      # 신촌 분석 결과 (동일 구조)
└── 연구자료/                          # 발표자료, 논문 참고자료 등
```

---

## 데이터

| 구분 | 최종 인원 | 남             | 여              | 스캐너 종 | 주요 kVp        |
|------|-----------|----------------|-----------------|-----------|-----------------|
| 강남 | 1,673명   | 665명 (39.7%)  | 1,008명 (60.3%) | 31종      | 100 kVp (93.3%) |
| 신촌 | 1,269명   | 637명 (50.2%)  | 632명 (49.8%)   | 46종      | 100 kVp (75.3%) |

**입력 파일 형식** (`{SITE}_merged_features.xlsx`):
- `metadata-value` 시트: PatientID, PatientSex, PatientAge, TAMA, ManufacturerModelName, KVP 등
- `features` 시트: PatientID + 65개 AEC 피처 (mean, std, skewness 등)

**이진 분류 기준 (Low Muscle)**:
- 남성: TAMA < 데이터의 P25 (동적 계산, config.py)
- 여성: TAMA < 데이터의 P25 (동적 계산, config.py)

---

## 분석 파이프라인

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

---

## 실행 방법

```bash
cd 연구코드/aec/code

# 전체 파이프라인 실행 (Step 1~7)
python run_analysis.py

# Feature Selection 건너뛰고 실행 (config.py 이미 설정된 경우)
python run_analysis.py --skip-fs

# 개별 스크립트 실행
python feature_selection.py
python linear_regression.py
python logistic_regression.py
python multivariable_analysis.py
```

**사이트 전환**: `config.py`의 `SITE` 변수를 `"강남"` 또는 `"신촌"`으로 변경 후 재실행.

---

## 주요 모듈 설명

### `config.py`
전체 설정 중앙 관리. `SITE` 변수 하나로 모든 경로·임계값이 자동 전환된다. TAMA 이진화 임계값은 데이터의 P25를 동적으로 계산한다.

### `data_loader.py`
- `load_raw_data()`: metadata + features 시트 inner join, 중복 PatientID·결측치 제거
- `prepare_full(mode)`: 전체 전처리 (Z-score 표준화, 성별 인코딩, One-hot 인코딩)
- `prepare_cv_fold()`: 5-Fold CV 전처리 (Data Leakage 방지 — scaler를 train fold에서만 fit)
- `get_feature_cols(case, df)`: Case 번호에 따른 예측변수 목록 반환

### `feature_selection.py`
65개 AEC 피처와 TAMA의 Pearson/Spearman 상관계수 계산 및 VIF 검사. `|r|` 상위 피처 중 VIF > 10인 중복 피처를 제거하여 최종 4개를 선택한다. 결과는 `feature_selection_report.xlsx`에 저장.

### `linear_regression.py`
- 단변량 OLS: 변수별 β, 95%CI, p-value, R²
- 다변량 OLS: 전체 변수 동시 투입, 5-Fold CV R²
- 잔차 진단: Shapiro-Wilk(정규성), Breusch-Pagan(등분산), Durbin-Watson(자기상관), Condition Number(다중공선성)

### `logistic_regression.py`
- 단변량 Logit: Crude OR, 95%CI, p-value, AUC
- 다변량 Logit: Adjusted OR, BFGS 최적화 (maxiter=1000)
- 성능 평가: AUC-ROC (Bootstrap 1,000회 95%CI), Sensitivity/Specificity/PPV/NPV
- 보정도: Hosmer-Lemeshow 검정, Brier Score, Nagelkerke R²

### `multivariable_analysis.py`
Case 0→1→2→3 순의 점진적 모델 비교. 각 Case에서 선형(R², Adj R², RMSE, AIC, BIC)과 로지스틱(AUC, Nagelkerke R², HL-test, AIC, BIC) 지표를 동시 산출한다.

### `generate_ppt.py`
`research_report.md`를 파싱하여 결과값을 자동으로 슬라이드에 삽입하고 PPTX를 생성한다.

---

## 선택 AEC 피처 및 근거

| 피처             | 의미                              | 선택 이유                          |
|------------------|-----------------------------------|------------------------------------|
| `mean`           | AEC 신호 평균 (전반적 체격 크기)  | 강남 r=0.297, amplitude 그룹 대표  |
| `CV`             | 변동계수 (체형 불균일성)          | 체형 이질성 독립 정보              |
| `skewness`       | 신호 비대칭성 (체지방 분포)       | 분포 편향 정보                     |
| `slope_abs_mean` | 평균 절대 기울기 (공간적 변화율)  | 다변량 시 독립 기여 확인           |

제외 피처: `p25`, `AUC_normalized`, `peak_max_height` → mean과 VIF > 50,000 (amplitude 중복)

---

## 주요 결과 요약

### 강남 (n=1,673)

| 지표         | Case 1    | Case 2         | Case 3 |
|--------------|-----------|----------------|--------|
| Linear R²    | 0.551     | 0.636 (+0.085) | 0.660  |
| Linear RMSE  | 20.43 cm² | 18.40 cm²      | —      |
| Logistic AUC | 0.624     | 0.720 (+0.096) | 0.751  |
| HL p-value   | —         | —              | 0.601  |
| NPV          | —         | —              | 0.887  |

### 신촌 (n=1,269)

| 지표         | Case 1 | Case 2        | Case 3 |
|--------------|--------|---------------|--------|
| Linear R²    | 0.520  | 0.548 (+0.028)| 0.590  |
| Logistic AUC | 0.610  | 0.650 (+0.040)| 0.728  |
| HL p-value   | —      | —             | 0.731  |

신촌은 스캐너 이질성(46종)이 커서 AEC 기여(+0.040)보다 스캐너 보정(+0.078) 효과가 더 크게 나타남.

---

## 향후 계획

1. **ML 앙상블 모델**: Random Forest, XGBoost와 선형 모델 성능 비교
2. **BMI 확장 데이터셋**: 신장·체중·BMI를 추가 변수로 포함한 분석 (신촌 결측 0.13% / 강남 결측 18.9% → 처리 전략 필요)
3. **Center-stratified 분석**: skewness 방향이 강남(−)·신촌(+) 반전된 원인 규명
4. **외부 검증**: 강남 → 신촌 교차 검증

---

## 의존성

```text
pandas, numpy, scipy, statsmodels, sklearn
openpyxl, python-pptx, matplotlib, pywt
```
