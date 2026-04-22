# TAMA 예측 회귀분석 연구 보고서

> **데이터셋:** 신촌 CT 데이터
> **분석 도구:** Python (statsmodels, scikit-learn, scipy)
> **AEC 사용 변수:** p25, CV, skewness, slope_abs_mean, mean
> **Logistic 임계값:** 남성 < 132 cm², 여성 < 95 cm²

---

## 1. 연구 개요

### 1.1 연구 목적

CT 촬영 시 자동 노출 제어(AEC, Automatic Exposure Control) 곡선과 환자 인구통계학적 특성(성별, 나이)이 복부 근육량 지표인 **TAMA(Total Abdominal Muscle Area)**를 예측하는 데 얼마나 기여하는지 정량화한다.

### 1.2 연구 설계

| 파트 | 분석 방법 | 목적 |
|------|-----------|------|
| **Part 1** | 선형 회귀 (Linear Regression) | TAMA(연속형) 예측 |
| **Part 2** | 로지스틱 회귀 (Logistic Regression) | Low TAMA / Sarcopenia 위험 예측 |
| **Part 3** | Multivariable Analysis (Case 1-3) | AEC·CT모델 추가에 따른 점진적 성능 향상 정량화 |

### 1.3 Feature Set (Case 구성)

| Case | 예측변수 | 목적 |
|------|----------|------|
| Case 0 (AEC 단독) | **AEC 특징** (p25, CV, skewness, slope_abs_mean, mean) | AEC만의 독립 진단력 |
| Case 1 | 성별 (Sex), 나이 (Age) | 인구통계 기반선 |
| Case 2 | 성별, 나이 + **AEC 특징** (p25, CV, skewness, slope_abs_mean, mean) | AEC 기여도 정량화 |
| Case 3 | 성별, 나이, AEC 특징 + **CT 모델명** (ManufacturerModelName), KVP | 스캐너·전압 보정 |

> **주의:** 성별·나이는 근육량 예측의 필수 공변량으로, Case 1·2·3에 항상 포함된다.
> Case 0은 AEC 단독 진단력 확인을 위한 추가 참조 모델이다.

---

## 2. 데이터

### 2.0 데이터 정제 단계

| 단계 | 환자 수 |
|------|---------|
| 원시 데이터 (metadata 시트) | 2,257명 |
| 원시 데이터 (features 시트) | 2,257명 |
| Inner join (공통 PatientID) | 2,257명 |
| 중복 PatientID 제거 후 | 2,257명 |
| 결측치 행 제거 후 (최종) | **1,269명** |

### 2.1 데이터셋 기본 정보

| 항목 | 값 |
|------|----|
| 총 환자 수 (분석 포함) | **1269명** |
| 남성 | 637명 (50.2%) |
| 여성 | 632명 (49.8%) |
| TAMA 범위 | 1 ~ 279 cm² |
| TAMA 평균 (SD) | 126.21 (30.95) cm² |
| CT 스캐너 모델 수 | 46종 |
| kVp 종류 수 | 6종 / 주요 kVp: **100 kVp** (956.0명, 75.3%) |

**CT 스캐너 분포 (상위 5종):**

| CT 스캐너 모델 | N | 비율 |
|---|---|---|
| iCT 256 | 278 | 21.9% |
| SOMATOM Definition Flash | 271 | 21.4% |
| SOMATOM Definition AS+ | 202 | 15.9% |
| SOMATOM Force | 177 | 13.9% |
| Revolution EVO | 70 | 5.5% |

**kVp 분포:**

| kVp | N | 비율 |
|---|---|---|
| 80 kVp | 13.0 | 1.0% |
| 90 kVp | 13.0 | 1.0% |
| 100 kVp | 956.0 | 75.3% |
| 110 kVp | 26.0 | 2.0% |
| 120 kVp | 250.0 | 19.7% |
| 130 kVp | 11.0 | 0.9% |

### 2.2 성별 TAMA 분포

| 성별 | N | 평균 | SD | 중앙값 | P25 | P75 |
|------|---|------|----|--------|-----|-----|
| 남성 (M) | 637 | 147.92 | 26.92 | 147.00 | 131.00 | 165.00 |
| 여성 (F) | 632 | 104.33 | 15.46 | 103.00 | 95.00 | 114.00 |

### 2.3 Logistic Regression 이진화

- **이진화 기준:** 성별별 TAMA **하위 25% (P25)** 기반 동적 계산 (`config.py`에서 자동 산출)
  - 남성 P25 = **132 cm²** → TAMA < 132 cm²: 1 (low muscle), 이상: 0 (normal)
  - 여성 P25 = **95 cm²** → TAMA < 95 cm²: 1 (low muscle), 이상: 0 (normal)
- **양성 비율:** 317/1269 = 25.0%

> **근거 (문헌):** Prado CM et al. (2008) *J Clin Oncol* 26:5572–5579에서 낮은 근육량(low skeletal muscle)을
> 성별별 하위 사분위(P25)로 정의하는 방법이 제안되었다. Martin L et al. (2013) *J Clin Oncol* 31:1539–1547에서도
> 복부 근육 단면적의 성별별 P25 컷오프가 임상 예후와 유의하게 연관됨을 보고하였다.
> 참고 문헌 기준 범위: 남성 170–180 cm², 여성 110–130 cm² (단일기관 P25는 데이터에 따라 다를 수 있음).

### 2.4 AEC Feature 선택 근거

`feature_selection.py`를 통해 TAMA와의 Pearson/Spearman 상관계수를 계산하고, VIF 검사로 다중공선성을 확인하여 아래 4개 feature를 선택하였다.

| Feature | Pearson r | 해석 | VIF |
|---------|-----------|------|-----|
| **p25** | +0.365 | AEC 하위 25%값 → 저선량 구간 tube current | ~1.9 |
| **CV** | -0.349 | 변동계수 (std/mean) → 체형 불균일성 반영 | ~1.8 |
| **skewness** | -0.344 | AEC 곡선 비대칭성 → 체형 분포 특성 | ~1.6 |
| **slope_abs_mean** | ~+0.09 | 평균 절대 기울기 → 곡선 동역학 | ~1.4 |

> **주의:** `mean`과 `AUC_normalized`는 VIF > 50,000으로 심각한 다중공선성이 확인되어 제외하였다.

---

## 3. 분석 방법론

### 3.1 전처리

| 변수 | 처리 방법 |
|------|-----------|
| PatientSex | 이진 인코딩 (M=1, F=0) |
| PatientAge | Z-score 표준화 (StandardScaler) |
| AEC features | Z-score 표준화 |
| ManufacturerModelName | One-hot encoding (drop_first=True) → 45개 더미 변수 |

### 3.2 모델

- **선형 회귀:** statsmodels OLS (최소자승법)
- **로지스틱 회귀:** statsmodels Logit (BFGS 최적화, maxiter=1000)

### 3.3 모델 검증

- **교차검증:** 5-Fold CV (data leakage 방지 위해 각 fold에서 scaler 재적합)
- **AUC 신뢰구간:** Bootstrap n=1000, 95%CI

---

## 4. 결과

### 4.1 Part 1 — 선형 회귀 (Linear Regression)

#### 4.1.1 단변량 분석 (Univariate)

| 변수 | β | 95%CI | p-value | R² |
|------|---|-------|---------|-----|
| Sex (M=1, F=0) | 43.5908 * | [41.1706, 46.0111] | 0.00e+00 | 0.4963 |
| Age (표준화) | -2.3588 * | [-4.0589, -0.6586] | 0.0066 | 0.0058 |
| AEC: p25 (표준화) | 7.5579 * | [5.9045, 9.2114] | 0.00e+00 | 0.0597 |
| AEC: CV (표준화) | -3.8596 * | [-5.5514, -2.1678] | 8.00e-06 | 0.0156 |
| AEC: skewness (표준화) | -1.3831 | [-3.0865, 0.3203] | 0.111 | 0.002 |
| AEC: slope_abs_mean (표준화) | 3.1567 * | [1.4605, 4.8529] | 2.72e-04 | 0.0104 |
| AEC: mean (표준화) | 7.2371 * | [5.5793, 8.8949] | 0.00e+00 | 0.0547 |
| ManufacturerModelName (F-test) | N/A (범주형) * | N/A | 0.0041 | 0.0578 |

#### 4.1.2 다변량 분석 (Multivariate — 전체 변수 투입)

**모델 성능:**

| 지표 | 값 | 근거 |
|------|----|------|
| N | 1269 | 총 환자 수 |
| R² | **0.5833** | TAMA 분산의 58.3% 설명 |
| Adjusted R² | **0.5654** | 변수 수 페널티 적용 설명력 |
| F-statistic | 32.728 (p=4.76e-193) | 전체 모델 유의성 |
| RMSE | **19.972 cm²** | 예측 오차 (임상 단위) |
| MAE | 14.676 cm² | 이상치 강건 오차 |
| AIC | 11306.82 | 모델 복잡도 대비 적합도 |
| BIC | 11579.56 | 강화된 복잡도 페널티 |
| 5-Fold CV R² | N/A ± N/A | 일반화 성능 |

**잔차 진단:**

| 검정 | 통계량 | p-value | 결과 | 해석 |
|------|--------|---------|------|------|
| Shapiro-Wilk | 0.9668 | 0.00e+00 | 비정규 | 잔차 정규성 |
| Breusch-Pagan | 98.8788 | 9.60e-05 | 이분산 | 등분산성 |
| Durbin-Watson | 1.96 | — | 정상 | 자기상관 (1.5-2.5 정상) |
| Condition number (κ) | 367.8 | — | 주의(κ<1000) | 다중공선성 |

> **해석:** 잔차가 정규분포를 따르지 않고(SW p<0.05) 이분산성이 존재하므로, 계수 추정치 자체는 BLUE이나 t-검정 p-value 해석 시 주의가 필요하다. Robust standard errors 사용을 고려할 수 있다.

---

### 4.2 Part 2 — 로지스틱 회귀 (Logistic Regression)

#### 4.2.1 단변량 분석 (Crude OR)

| 변수 | Crude OR | 95%CI | p-value | AUC |
|------|----------|-------|---------|-----|
| Sex (M=1, F=0) | 1.0496 | [0.814, 1.3535] | 0.709 | 0.506 |
| Age (표준화) | 1.4085 * | [1.2243, 1.6205] | 2.00e-06 | 0.61 |
| AEC: p25 (표준화) | 0.7147 * | [0.6147, 0.831] | 1.30e-05 | 0.5911 |
| AEC: CV (표준화) | 1.0977 | [0.969, 1.2435] | 0.143 | 0.5354 |
| AEC: skewness (표준화) | 0.9337 | [0.8235, 1.0587] | 0.285 | 0.5195 |
| AEC: slope_abs_mean (표준화) | 0.7146 | [0.4977, 1.026] | 0.069 | 0.5587 |
| AEC: mean (표준화) | 0.7052 * | [0.6083, 0.8176] | 4.00e-06 | 0.595 |
| ManufacturerModelName (LR χ²-test) | LR_χ²=69.278 * | N/A | 0.012 | 0.601 |

#### 4.2.2 다변량 분석 (전체 변수 투입)

**모델 성능:**

| 지표 | 값 | 근거 |
|------|----|------|
| N | 1269 (양성: 317, 25.0%) | 총 환자 수 |
| AUC-ROC | **0.7170** [0.6847–0.7485] | 판별 능력 (Bootstrap 95%CI) |
| Sensitivity | 0.6530 | TP/(TP+FN) |
| Specificity | 0.6807 | TN/(TN+FP) |
| PPV | 0.4051 | 양성 예측도 |
| NPV | 0.8549 | 음성 예측도 |
| Hosmer-Lemeshow | χ²=7.571, p=0.477 | 보정도 (p>0.05 = 보정 양호) |
| Nagelkerke R² | **0.1698** | 로지스틱 설명력 (pseudo-R²) |
| Brier Score | 0.1640 | 확률 예측 정밀도 (0=완벽) |
| AIC | 1378.14 | 모델 복잡도 대비 적합도 |

---

### 4.3 Part 3 — Multivariable Analysis (Case 비교)

> Case 0 (AEC 단독)은 성별·나이 없이 AEC 특징만으로의 독립 진단력 확인용 참조 모델이다.

#### 4.3.1 선형 회귀 성능 비교

| 지표 | Case 0 (AEC 단독) | Case 1 (Sex+Age) | Case 2 (+AEC) | Case 3 (+CT/kVp) | Case 1→2 향상 | Case 2→3 향상 |
|------|-------------------|------------------|---------------|------------------|--------------|--------------|
| R² | 0.0741 | 0.5146 | 0.5437 | 0.5879 | **+0.0291** | +0.0442 |
| Adj R² | 0.0704 | 0.5139 | 0.5411 | 0.5699 | +0.0272 | +0.0288 |
| RMSE (cm²) | 29.7692 | 21.5534 | 20.8984 | 19.8599 | -0.6550 | -1.0385 |
| AIC | 12225.91 | 11400.28 | 11331.95 | 11294.59 | -68.33 | -37.36 |

#### 4.3.2 로지스틱 회귀 성능 비교

| 지표 | Case 0 (AEC 단독) | Case 1 (Sex+Age) | Case 2 (+AEC) | Case 3 (+CT/kVp) | Case 1→2 향상 | Case 2→3 향상 |
|------|-------------------|------------------|---------------|------------------|--------------|--------------|
| AUC | 0.609 | 0.6105 | 0.6506 | 0.7285 | **+0.0401** | +0.0779 |
| Nagelkerke R² | 0.0394 | 0.0286 | 0.0741 | 0.1811 | +0.0455 | +0.1070 |
| AIC | 1404.41 | 1407.95 | 1377.54 | 1369.13 | -30.41 | -8.41 |
| HL p | 1.5686e-01 | 6.2300e-04 | 1.7321e-01 | 4.4624e-01 | — | — |

---

## 5. 성능 평가지표 근거

### 5.1 선형 회귀 지표

| 지표 | 통계적 근거 |
|------|------------|
| **R²** | 종속변수(TAMA) 분산 중 모델이 설명하는 비율. 0~1 사이이며 1에 가까울수록 설명력 높음 |
| **Adjusted R²** | 예측변수 수 증가 시 R²가 무조건 증가하는 편향을 보정. Case 간 공정한 비교를 위해 필수 |
| **RMSE** | 예측 오차의 제곱 평균의 제곱근. 단위가 cm²로 임상적 해석이 직관적 |
| **MAE** | 이상치에 강건한 평균 절대 오차. RMSE와 함께 보고 |
| **F-statistic** | 전체 회귀 모델의 유의성 검정 (H₀: 모든 계수=0) |
| **AIC/BIC** | 로그우도에서 모델 복잡도 페널티를 차감한 정보기준. Case 간 모델 선택에 사용 (낮을수록 선호) |
| **5-Fold CV R²** | 훈련/테스트 분리 교차검증으로 과적합 탐지 및 일반화 성능 추정 |
| **Durbin-Watson** | 잔차의 1차 자기상관 검정. 2.0에 가까울수록 독립 가정 충족 |
| **Breusch-Pagan** | 등분산성 검정. p>0.05이면 동분산 가정 충족 |
| **Shapiro-Wilk** | 잔차 정규성 검정 (n>500 시 서브샘플 500개 사용) |

### 5.2 로지스틱 회귀 지표

| 지표 | 통계적 근거 |
|------|------------|
| **AUC-ROC** | 임계값 독립적인 판별 능력 지표. 불균형 데이터에 robust. 0.5=무작위, 1=완벽 |
| **Bootstrap 95%CI** | 비모수적 방법으로 AUC 불확실성 정량화 (n=1000) |
| **Sensitivity** | TP/(TP+FN) — 실제 low TAMA를 얼마나 잘 탐지하는가 |
| **Specificity** | TN/(TN+FP) — 정상 TAMA를 얼마나 잘 식별하는가 |
| **PPV/NPV** | 실제 임상 환경에서의 예측값 해석. 유병률에 영향을 받음 |
| **Hosmer-Lemeshow** | 예측 확률을 10개 동등 구간(decile)으로 나눠 각 구간의 관측 사건 수와 예측 사건 수를 χ² 검정으로 비교. p > 0.05이면 예측 확률이 실제 발생률과 잘 일치(보정 양호). p ≤ 0.05이면 특정 위험 구간에서 모델이 체계적으로 과대·과소 추정 중임을 의미. HL 검정은 표본 크기에 민감하므로 큰 n에서는 작은 불일치도 유의해질 수 있음 |
| **Nagelkerke R²** | Cox-Snell R²를 최대값으로 정규화한 pseudo-R². 0~1 사이의 설명력 지표 |
| **Brier Score** | 확률 예측의 Mean Squared Error. 0=완벽, 0.25=무작위 (50:50 레이블 기준) |
| **Odds Ratio (OR)** | exp(β). 예측변수 1 SD 증가 시 결과(low TAMA) 발생 오즈의 배수 |

---

## 6. 시각화 자료 목록

| 파일명 | 내용 |
|--------|------|
| `figures/01_feature_correlation.png` | AEC feature-TAMA 상관계수 Top 20 |
| `figures/02_vif_comparison.png` | 선택 AEC feature VIF 비교 |
| `figures/03_tama_distribution.png` | 성별 TAMA 분포 + 이진화 임계값 |
| `figures/04_linear_actual_vs_pred.png` | 선형 회귀: 실제 vs 예측 산점도 |
| `figures/05_linear_residuals.png` | 선형 회귀: 잔차 진단 4-panel |
| `figures/06_linear_forest.png` | 선형 회귀: 유의한 계수 Forest plot |
| `figures/07_linear_univariate_r2.png` | 선형 회귀: 단변량 R² 비교 |
| `figures/08_logistic_roc.png` | 로지스틱: ROC 곡선 (AUC + Bootstrap CI) |
| `figures/09_logistic_calibration.png` | 로지스틱: Calibration plot (HL 검정) |
| `figures/10_logistic_confusion.png` | 로지스틱: Confusion matrix |
| `figures/11_logistic_forest.png` | 로지스틱: Crude OR Forest plot |
| `figures/12_case_metrics_bar.png` | Case 1-3: 선형 R²/RMSE 비교 |
| `figures/13_case_auc_bar.png` | Case 1-3: AUC/Nagelkerke R² 비교 |
| `figures/14_case_aic_bar.png` | Case 1-3: AIC/BIC 비교 |
| `figures/15_case_progression.png` | Case 1-3: 다중 지표 추이 |
| `figures/16_scanner_distribution.png` | CT 스캐너 모델 분포 (환자 수·비율) |
| `figures/17_kvp_distribution.png` | kVp 값 분포 (주요 전압 확인) |

---

## 7. 결론

1. **성별의 압도적 기여:** 선형 단변량 분석에서 Sex만으로 R²=0.518을 달성. 남성의 TAMA가 여성 대비 평균 44.8 cm² 높음.

2. **AEC 단독 진단력 (Case 0):** 성별·나이 없이 AEC 특징만으로 선형 R²=0.074, 로지스틱 AUC=0.609. AEC 신호가 독립적인 근육량 정보를 내포함을 시사.

3. **AEC 특징의 추가 기여 (Case 1→2):** AEC feature 추가 시 선형 R²가 **0.515 → 0.544** (Δ+0.029)으로 향상됨. 로지스틱 AUC는 **0.611 → 0.651** (Δ+0.040).

4. **CT 모델명·kVp의 추가 기여 (Case 2→3):** 모델명/kVp 추가 시 선형 R²가 추가로 +0.044, 로지스틱 AUC +0.078 향상. AIC 감소로 모델 적합도 개선 확인.

5. **최종 모델 (Case 3):** 선형 R²=0.5879, RMSE=19.8599 cm², 로지스틱 AUC=0.7285 [0.6975-0.7592].

6. **모델 보정도:** Case 2 이상에서 Hosmer-Lemeshow p>0.05로 로지스틱 모델의 보정도가 양호함.

---

## 8. 한계 및 고려사항

- 잔차 정규성 및 등분산성 가정 위반 → Robust SE 또는 비모수 검정 검토 필요
- Logistic regression 이진화 임계값(M<132, F<95 cm²)은 데이터 내 P25로 설정; 외부 문헌(Prado 2008, Martin 2013) 기준값과 비교 필요
- 단일 기관(신촌) 데이터 → 타 기관 데이터로 외부 검증 필요
- AEC feature는 현재 통계적 요약값(mean, std, percentile 등) 기반 — raw AEC 시계열(~200포인트)을 1D CNN 또는 LSTM 입력으로 직접 사용하면 feature engineering 없이 잠재 패턴 학습 가능 (딥러닝 접근 권장)
- AEC feature 선택은 Pearson 상관분석 기반 — 임상적 의미와 교호작용 효과 추가 검토 권장

---

*생성일: 자동 생성 by generate_report.py*
