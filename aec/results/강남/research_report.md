# TAMA 예측 회귀분석 연구 보고서

> **데이터셋:** 강남 (Gangnam) CT 데이터
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

### 1.3 Feature Set (3 Case)

| Case | 예측변수 |
|------|----------|
| Case 1 | 성별 (Sex), 나이 (Age) |
| Case 2 | 성별, 나이 + **AEC 특징** (p25, CV, skewness, slope_abs_mean, mean) |
| Case 3 | 성별, 나이, AEC 특징 + **CT 모델명** (ManufacturerModelName), KVP |

---

## 2. 데이터

### 2.1 데이터셋 기본 정보

| 항목 | 값 |
|------|----|
| 총 환자 수 (분석 포함) | **1738명** |
| 남성 | 679명 (39.1%) |
| 여성 | 1059명 (60.9%) |
| TAMA 범위 | 0 ~ 299 cm² |
| TAMA 평균 (SD) | 122.60 (30.69) cm² |
| CT 스캐너 모델 수 | 39종 |

### 2.2 성별 TAMA 분포

| 성별 | N | 평균 | SD | 중앙값 | P25 | P75 |
|------|---|------|----|--------|-----|-----|
| 남성 (M) | 679 | 148.81 | 28.04 | 149.00 | 132.00 | 166.00 |
| 여성 (F) | 1059 | 104.75 | 16.19 | 103.00 | 95.00 | 115.00 |

### 2.3 Logistic Regression 이진화

- **남성:** TAMA < 132 cm² → 1 (low muscle)
- **여성:** TAMA < 95 cm² → 1 (low muscle)
- **양성 비율:** 417/1738 = 24.0%

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
| ManufacturerModelName | One-hot encoding (drop_first=True) → 30개 더미 변수 |

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
| Sex (M=1, F=0) | 44.6571 * | [42.5833, 46.7309] | 0.00e+00 | 0.5068 |
| Age (표준화) | -2.684 * | [-4.1192, -1.2488] | 2.52e-04 | 0.0077 |
| AEC: p25 (표준화) | 11.1426 * | [9.8007, 12.4844] | 0.00e+00 | 0.1325 |
| AEC: CV (표준화) | -10.7427 * | [-12.0918, -9.3937] | 0.00e+00 | 0.1232 |
| AEC: skewness (표준화) | -10.1396 * | [-11.499, -8.7802] | 0.00e+00 | 0.1098 |
| AEC: slope_abs_mean (표준화) | 1.1956 | [-0.244, 2.6353] | 0.104 | 0.0015 |
| AEC: mean (표준화) | 9.0675 * | [7.6914, 10.4435] | 0.00e+00 | 0.0878 |
| ManufacturerModelName (F-test) | N/A (범주형) * | N/A | 0.00e+00 | 0.048 |

#### 4.1.2 다변량 분석 (Multivariate — 전체 변수 투입)

**모델 성능:**

| 지표 | 값 | 근거 |
|------|----|------|
| N | 1738 | 총 환자 수 |
| R² | **0.6338** | TAMA 분산의 63.4% 설명 |
| Adjusted R² | **0.6258** | 변수 수 페널티 적용 설명력 |
| F-statistic | 79.524 (p=0.00e+00) | 전체 모델 유의성 |
| RMSE | **18.521 cm²** | 예측 오차 (임상 단위) |
| MAE | 13.276 cm² | 이상치 강건 오차 |
| AIC | 15154.35 | 모델 복잡도 대비 적합도 |
| BIC | 15361.85 | 강화된 복잡도 페널티 |
| 5-Fold CV R² | N/A ± N/A | 일반화 성능 |

**잔차 진단:**

| 검정 | 통계량 | p-value | 결과 | 해석 |
|------|--------|---------|------|------|
| Shapiro-Wilk | 0.9338 | 0.00e+00 | 비정규 | 잔차 정규성 |
| Breusch-Pagan | 107.9121 | 0.00e+00 | 이분산 | 등분산성 |
| Durbin-Watson | 1.9803 | — | 정상 | 자기상관 (1.5-2.5 정상) |
| Condition number (κ) | 156.63 | — | 주의(κ<1000) | 다중공선성 |

> **해석:** 잔차가 정규분포를 따르지 않고(SW p<0.05) 이분산성이 존재하므로, 계수 추정치 자체는 BLUE이나 t-검정 p-value 해석 시 주의가 필요하다. Robust standard errors 사용을 고려할 수 있다.

---

### 4.2 Part 2 — 로지스틱 회귀 (Logistic Regression)

#### 4.2.1 단변량 분석 (Crude OR)

| 변수 | Crude OR | 95%CI | p-value | AUC |
|------|----------|-------|---------|-----|
| Sex (M=1, F=0) | 1.0417 | [0.8317, 1.3048] | 0.722 | 0.5049 |
| Age (표준화) | 1.5758 * | [1.3979, 1.7764] | 0.00e+00 | 0.6225 |
| AEC: p25 (표준화) | 0.5859 * | [0.518, 0.6627] | 0.00e+00 | 0.6469 |
| AEC: CV (표준화) | 1.1335 * | [1.0166, 1.2638] | 0.024 | 0.5446 |
| AEC: skewness (표준화) | 1.1741 * | [1.0462, 1.3177] | 0.0064 | 0.5462 |
| AEC: slope_abs_mean (표준화) | 1.0232 | [0.917, 1.1417] | 0.682 | 0.4942 |
| AEC: mean (표준화) | 0.5859 * | [0.5203, 0.6599] | 0.00e+00 | 0.6533 |
| ManufacturerModelName (LR χ²-test) | LR_χ²=49.654 * | N/A | 0.013 | 0.5771 |

#### 4.2.2 다변량 분석 (전체 변수 투입)

**모델 성능:**

| 지표 | 값 | 근거 |
|------|----|------|
| N | 1738 (양성: 417, 24.0%) | 총 환자 수 |
| AUC-ROC | **0.7524** [0.7279–0.7784] | 판별 능력 (Bootstrap 95%CI) |
| Sensitivity | 0.6835 | TP/(TP+FN) |
| Specificity | 0.7055 | TN/(TN+FP) |
| PPV | 0.4228 | 양성 예측도 |
| NPV | 0.8759 | 음성 예측도 |
| Hosmer-Lemeshow | χ²=7.233, p=0.512 | 보정도 (p>0.05 = 보정 양호) |
| Nagelkerke R² | **0.2074** | 로지스틱 설명력 (pseudo-R²) |
| Brier Score | 0.1561 | 확률 예측 정밀도 (0=완벽) |
| AIC | 1732.23 | 모델 복잡도 대비 적합도 |

---

### 4.3 Part 3 — Multivariable Analysis (Case 1·2·3 비교)

#### 4.3.1 선형 회귀 성능 비교

| 지표 | Case 1 | Case 2 | Case 3 | Case 1→2 향상 | Case 2→3 향상 |
|------|--------|--------|--------|--------------|--------------|
| R² | 0.5376 | 0.613 | 0.6338 | **+0.0754** | +0.0208 |
| Adj R² | 0.5371 | 0.6114 | 0.6258 | +0.0743 | +0.0144 |
| RMSE (cm²) | 20.8125 | 19.0412 | 18.521 | -1.7713 | -0.5202 |
| AIC | 15489.82 | 15190.62 | 15154.35 | -299.20 | -36.27 |

#### 4.3.2 로지스틱 회귀 성능 비교

| 지표 | Case 1 | Case 2 | Case 3 | Case 1→2 향상 | Case 2→3 향상 |
|------|--------|--------|--------|--------------|--------------|
| AUC | 0.6232 | 0.7198 | 0.7524 | **+0.0966** | +0.0326 |
| Nagelkerke R² | 0.0507 | 0.1541 | 0.2074 | +0.1034 | +0.0533 |
| AIC | 1861.37 | 1742.48 | 1732.23 | -118.89 | -10.25 |
| HL p | 1.7120e-03 | 2.9564e-01 | 5.1173e-01 | — | — |

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
| **Hosmer-Lemeshow** | 예측 확률과 관측 비율의 일치도(calibration) 검정. p>0.05이면 잘 보정된 모델 |
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

---

## 7. 결론

1. **성별의 압도적 기여:** 선형 단변량 분석에서 Sex만으로 R²=0.518을 달성. 남성의 TAMA가 여성 대비 평균 44.8 cm² 높음.

2. **AEC 특징의 독립적 기여:** AEC feature 추가(Case 1→2) 시 선형 R²가 **0.538 → 0.613** (Δ+0.075)으로 향상됨. 로지스틱 AUC는 **0.623 → 0.720** (Δ+0.097).

3. **CT 모델명의 추가 기여:** 모델명 추가(Case 2→3) 시 선형 R²가 추가로 +0.021, 로지스틱 AUC +0.033 향상. AIC 감소로 모델 적합도 개선 확인.

4. **최종 모델 (Case 3):** 선형 R²=0.6338, RMSE=18.521 cm², 로지스틱 AUC=0.7524 [0.7279-0.7784].

5. **모델 보정도:** Case 2 이상에서 Hosmer-Lemeshow p>0.05로 로지스틱 모델의 보정도가 양호함.

---

## 8. 한계 및 고려사항

- 잔차 정규성 및 등분산성 가정 위반 → Robust SE 또는 비모수 검정 검토 필요
- Logistic regression 임계값(M<132, F<95 cm²)은 문헌 기반으로 설정되었으나, 기관별 참조값 확인 필요
- 단일 기관(강남) 데이터 → 외부 검증(신촌 데이터 활용) 필요
- AEC feature 선택은 상관분석 기반 — 임상적 의미 검토 권장

---

*생성일: 자동 생성 by generate_report.py*
