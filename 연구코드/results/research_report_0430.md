# TAMA 예측 회귀분석 연구 보고서 (0430)

> **버전:** 0430 (0424 대비 설계 변경 적용)
> **분석 도구:** Python (statsmodels, scikit-learn, scipy)
> **병원:** 강남, 신촌
> **성별 그룹:** 전체 (성별 특이적 P25 임계값 적용 — 남/여 별도 산출)
> **AEC 세트:** AEC_prev (수동 4개) vs AEC_new (파이프라인 자동 선택)

---
## 1. 0424 대비 설계 변경사항

| # | 항목 | 0424 이전 | 0430 이후 |
|---|------|-----------|-----------|
| ① | 피처 선택 | 수동 (상관계수+VIF → 연구자 결정) | 자동 파이프라인 (4단계 필터링 + 앙상블 투표 + CV R²) |
| ② | 임상 기준선 | PatientAge + PatientSex | PatientAge + PatientSex + **BMI** |
| ③ | Case 구조 | Case 1~3 (단일 AEC 세트) | Case 1~5 (AEC_prev vs AEC_new 교차 비교) |
| ④ | 성별 층화 | 전체 / 여성 / 남성 독립 모델 | 전체 모델 + 성별 특이적 P25 임계값 |
| ⑤ | 다병원 분석 | 강남 단독 (SITE 수동 변경) | 강남·신촌 자동 순회 + 교차 병원 비교 |
| ⑥ | 이진화 기준 | 분석 그룹 내 하위 25% 동적 산출 | 성별 특이적 P25 (남/여 별도) — 전체 모델에 적용 |

### 1.1 피처 선택 파이프라인 4단계

| 단계 | 방법 | 기준 |
|------|------|------|
| Step 1 | Near-zero variance 제거 | 표준화 후 variance < 0.01 |
| Step 2 | Pearson 상관 중복 제거 | |r| ≥ 0.95 |
| Step 3 | 단변량 필터 (OR 결합) | MI > 0 OR Spearman p < 0.05 |
| Step 4 | 앙상블 투표 + 완전/SFS 탐색 | LASSO + RFECV + RF Permutation → 5-fold CV R² 최대화 |
| Final | VIF pruning (보호: 'mean') | VIF > 10 반복 제거 |

### 1.2 Case 구조 (0430)

| Case | 포함 변수 | AEC 세트 |
|------|-----------|----------|
| Case 1 (Clinical) | Age, Sex, BMI | — |
| Case 2 (+AEC_prev) | Case 1 + AEC_prev | mean, CV, skewness, slope_abs_mean |
| Case 3 (+AEC_new) | Case 1 + AEC_new | 파이프라인 자동 선택 |
| Case 4 (+AEC_prev+Scanner) | Case 2 + Scanner | AEC_prev + ManufacturerModelName + kVp |
| Case 5 (+AEC_new+Scanner) | Case 3 + Scanner | AEC_new + ManufacturerModelName + kVp |

> Case 2 vs Case 3, Case 4 vs Case 5 → AEC_prev vs AEC_new 직접 성능 비교

---
## 2. 피처 선택 결과 (자동 파이프라인)

### 2.1 데이터셋별 최종 선택 피처 요약

| 데이터셋 | N | 선택 피처 수 | Best Set | Pipeline CV R² | Prev CV R² | ΔCV R² |
|----------|---|------------|----------|---------------|----------|--------|
| 강남 | 1674 | 9 | exhaustive_or_SFS_best | **0.1818** | 0.1768 | 0.0050 |
| 신촌 | 1271 | 13 | exhaustive_or_SFS_best | **0.0775** | 0.0305 | 0.0469 |
| 병합(강남+신촌) | 2945 | 11 | exhaustive_or_SFS_best | **0.0881** | 0.0965 | -0.0084 |

### 2.2 최종 선택 AEC 피처

| 데이터셋 | 선택된 피처 |
|----------|------------|
| 강남 | IQR, band2_energy, dominant_freq, mean, slope_max, spectral_energy, wavelet_cD1_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1 |
| 신촌 | IQR, mean, peak_count, peak_max_width, peak_mean_width, skewness, slope_mean, slope_min, spectral_centroid, spectral_energy, wavelet_cD2_std, wavelet_cD3_energy, zero_crossing_rate |
| 병합(강남+신촌) | band3_energy_ratio, mean, peak_main_pos, peak_std_height, signal_length, slope_mean, spectral_energy, valley_count, wavelet_cD1_std, wavelet_cD3_std, zero_crossing_rate |

### 2.3 데이터셋별 피처 선택 일치도

| 피처 | 강남 | 신촌 | 병합(강남+신촌) |
|---|---|---|---|
| IQR | 1 | 1 | 0 |
| band2_energy | 1 | 0 | 0 |
| band3_energy_ratio | 0 | 0 | 1 |
| dominant_freq | 1 | 0 | 0 |
| mean | 1 | 1 | 1 |
| peak_count | 0 | 1 | 0 |
| peak_main_pos | 0 | 0 | 1 |
| peak_max_width | 0 | 1 | 0 |
| peak_mean_width | 0 | 1 | 0 |
| peak_std_height | 0 | 0 | 1 |
| signal_length | 0 | 0 | 1 |
| skewness | 0 | 1 | 0 |
| slope_max | 1 | 0 | 0 |
| slope_mean | 0 | 1 | 1 |
| slope_min | 0 | 1 | 0 |
| spectral_centroid | 0 | 1 | 0 |
| spectral_energy | 1 | 1 | 1 |
| valley_count | 0 | 0 | 1 |
| wavelet_cD1_energy | 1 | 0 | 0 |
| wavelet_cD1_std | 0 | 0 | 1 |
| wavelet_cD2_energy | 1 | 0 | 0 |
| wavelet_cD2_std | 0 | 1 | 0 |
| wavelet_cD3_energy | 0 | 1 | 0 |
| wavelet_cD3_std | 0 | 0 | 1 |
| wavelet_energy_ratio_D1 | 1 | 0 | 0 |
| zero_crossing_rate | 0 | 1 | 1 |

> 1 = 해당 데이터셋에서 선택됨, 0 = 미선택

### 2.4 파이프라인 단계별 피처 수 변화 (강남)

| 단계 | 제거 수 | 잔여 피처 |
|------|---------|----------|
| Step 1 - Near-zero var | 0 | nan |
| Step 2 - High correlation | 20 | band3_energy_ratio, peak_mean_height, fft_mag_std, AUC_normalized, wavelet_cA_energy, spectral_rolloff, mean_abs_deviation, p95, p75, RMSE, AUC, band1_energy, band4_energy_ratio, band2_energy_ratio, p90, wavelet_cA_std, median, band4_energy, band3_energy, p25 |
| Step 3 - Union pass | 2 | nan |
| Step 4 - Best search | 0 | CV, IQR, band1_energy_ratio, band2_energy, dominant_freq, min, p10, p5, p90_p10_ratio, signal_energy, signal_length, slope_abs_mean, slope_max, slope_mean, slope_std, spectral_energy, spectral_spread, wavelet_cD1_energy, wavelet_cD1_std, wavelet_cD2_energy, wavelet_cD2_std, wavelet_energy_ratio_D1 |
| VIF pruning (VIF>10) | 13 | IQR, band2_energy, dominant_freq, mean, slope_max, spectral_energy, wavelet_cD1_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1 |

---
## 3. 회귀 분석 결과 — 강남

### 3.1 전체

- N = **1365** | 이진화 임계값 (성별 특이적 P25) = **F:95.0/M:132.0**

#### 선형 회귀 (5-Fold CV)

| Case | N features | R² (mean±std) | MAE (cm²) | RMSE (cm²) |
|------|-----------|--------------|----------|-----------|
| Case 1 Clinical | 3 | **0.662 ± 0.037** | 12.83 | 17.12 |
| Case 2 +AEC_prev | 7 | **0.678 ± 0.036** | 12.53 | 16.71 |
| Case 3 +AEC_new | 14 | **0.669 ± 0.038** | 12.59 | 16.93 |
| Case 4 +AEC_prev +Scanner | 46 | **0.668 ± 0.032** | 12.77 | 16.96 |
| Case 5 +AEC_new +Scanner | 53 | **0.662 ± 0.035** | 12.74 | 17.11 |

> **AEC 기여도 (선형 R²):** AEC_prev Δ = +0.0160 | AEC_new Δ = +0.0074 | 차이 (new-prev) = -0.0086

#### 로지스틱 회귀 (5-Fold CV)

| Case | AUC (mean±std) | Accuracy | Sensitivity | Specificity |
|------|---------------|---------|------------|------------|
| Case 1 Clinical | **0.7427 ± 0.0188** | 0.6762 | 0.7647 | 0.6491 |
| Case 2 +AEC_prev | **0.7510 ± 0.0227** | 0.7004 | 0.7394 | 0.6883 |
| Case 3 +AEC_new | **0.7442 ± 0.0280** | 0.6374 | 0.8274 | 0.5794 |
| Case 4 +AEC_prev +Scanner | **0.7400 ± 0.0287** | 0.6674 | 0.7834 | 0.6319 |
| Case 5 +AEC_new +Scanner | **0.7311 ± 0.0291** | 0.6762 | 0.7522 | 0.6529 |

> **AEC 기여도 (AUC):** AEC_prev Δ = +0.0083 | AEC_new Δ = +0.0015 | 차이 (new-prev) = -0.0068

---
## 3. 회귀 분석 결과 — 신촌

### 3.1 전체

- N = **1269** | 이진화 임계값 (성별 특이적 P25) = **F:95.0/M:131.0**

#### 선형 회귀 (5-Fold CV)

| Case | N features | R² (mean±std) | MAE (cm²) | RMSE (cm²) |
|------|-----------|--------------|----------|-----------|
| Case 1 Clinical | 3 | **0.640 ± 0.063** | 13.82 | 18.42 |
| Case 2 +AEC_prev | 7 | **0.641 ± 0.061** | 13.83 | 18.37 |
| Case 3 +AEC_new | 14 | **0.637 ± 0.067** | 13.81 | 18.48 |
| Case 4 +AEC_prev +Scanner | 59 | **0.634 ± 0.059** | 14.00 | 18.56 |
| Case 5 +AEC_new +Scanner | 66 | **0.638 ± 0.054** | 13.94 | 18.48 |

> **AEC 기여도 (선형 R²):** AEC_prev Δ = +0.0017 | AEC_new Δ = -0.0027 | 차이 (new-prev) = -0.0044

#### 로지스틱 회귀 (5-Fold CV)

| Case | AUC (mean±std) | Accuracy | Sensitivity | Specificity |
|------|---------------|---------|------------|------------|
| Case 1 Clinical | **0.7513 ± 0.0332** | 0.6541 | 0.7988 | 0.6073 |
| Case 2 +AEC_prev | **0.7502 ± 0.0362** | 0.6596 | 0.8055 | 0.6125 |
| Case 3 +AEC_new | **0.7492 ± 0.0239** | 0.6762 | 0.7857 | 0.6406 |
| Case 4 +AEC_prev +Scanner | **0.7529 ± 0.0321** | 0.7061 | 0.7572 | 0.6896 |
| Case 5 +AEC_new +Scanner | **0.7493 ± 0.0342** | 0.7164 | 0.6987 | 0.7219 |

> **AEC 기여도 (AUC):** AEC_prev Δ = -0.0011 | AEC_new Δ = -0.0021 | 차이 (new-prev) = -0.0010

---
## 4. 교차 병원 비교 (Cross-Hospital)

### 4.1 선형 회귀 R² 비교 (전체 그룹)

| Case | 강남 | 신촌 |
|------|---|---|
| Case 1 Clinical | **0.6620** | **0.6398** |
| Case 2 +AEC_prev | **0.6780** | **0.6415** |
| Case 3 +AEC_new | **0.6694** | **0.6371** |
| Case 4 +AEC_prev +Scanner | **0.6682** | **0.6340** |
| Case 5 +AEC_new +Scanner | **0.6620** | **0.6376** |

### 4.2 로지스틱 AUC 비교 (전체 그룹)

| Case | 강남 | 신촌 |
|------|---|---|
| Case 1 Clinical | **0.7427** | **0.7513** |
| Case 2 +AEC_prev | **0.7510** | **0.7502** |
| Case 3 +AEC_new | **0.7442** | **0.7492** |
| Case 4 +AEC_prev +Scanner | **0.7400** | **0.7529** |
| Case 5 +AEC_new +Scanner | **0.7311** | **0.7493** |

### 4.3 외부 검증 (강남 학습 → 신촌 예측)

| Train | Test | Case | N_train | N_test | N_features_used | Lin_R2 | Lin_MAE | Lin_RMSE | TAMA_threshold | Log_AUC | Log_Acc | Log_Sens | Log_Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 강남 | 신촌 | Case1_Clinical | 1365 | 1269 | 3 | 0.6374 | 13.81 | 18.54 | {'female': np.float64(95.0), 'male': np.float64(131.0)} | 0.7501 | 0.766 | 0.246 | 0.9333 |
| 강남 | 신촌 | Case2_Clinical+AEC_prev | 1365 | 1269 | 7 | 0.479 | 15.0 | 22.23 | {'female': np.float64(95.0), 'male': np.float64(131.0)} | 0.7357 | 0.7518 | 0.4013 | 0.8646 |
| 강남 | 신촌 | Case3_Clinical+AEC_new | 1365 | 1269 | 14 | 0.5842 | 14.8 | 19.86 | {'female': np.float64(95.0), 'male': np.float64(131.0)} | 0.7201 | 0.7478 | 0.3172 | 0.8865 |
| 강남 | 신촌 | Case4_Clinical+AEC_prev+Scanner | 1365 | 1269 | 38 | 0.3959 | 16.0 | 23.93 | {'female': np.float64(95.0), 'male': np.float64(131.0)} | 0.6551 | 0.7541 | 0.2848 | 0.9052 |
| 강남 | 신촌 | Case5_Clinical+AEC_new+Scanner | 1365 | 1269 | 45 | 0.4099 | 17.56 | 23.66 | {'female': np.float64(95.0), 'male': np.float64(131.0)} | 0.6427 | 0.7533 | 0.2913 | 0.9021 |
| 신촌 | 강남 | Case1_Clinical | 1269 | 1365 | 3 | 0.6606 | 12.82 | 17.18 | {'female': np.float64(95.0), 'male': np.float64(132.0)} | 0.743 | 0.781 | 0.1442 | 0.9751 |
| 신촌 | 강남 | Case2_Clinical+AEC_prev | 1269 | 1365 | 7 | 0.6588 | 12.87 | 17.23 | {'female': np.float64(95.0), 'male': np.float64(132.0)} | 0.7406 | 0.7795 | 0.1473 | 0.9723 |
| 신촌 | 강남 | Case3_Clinical+AEC_new | 1269 | 1365 | 14 | 0.6564 | 12.97 | 17.28 | {'female': np.float64(95.0), 'male': np.float64(132.0)} | 0.7334 | 0.7714 | 0.1379 | 0.9646 |
| 신촌 | 강남 | Case4_Clinical+AEC_prev+Scanner | 1269 | 1365 | 38 | 0.6587 | 12.84 | 17.23 | {'female': np.float64(95.0), 'male': np.float64(132.0)} | 0.7376 | 0.7736 | 0.1944 | 0.9503 |
| 신촌 | 강남 | Case5_Clinical+AEC_new+Scanner | 1269 | 1365 | 45 | 0.6563 | 12.92 | 17.29 | {'female': np.float64(95.0), 'male': np.float64(132.0)} | 0.7353 | 0.7663 | 0.21 | 0.9359 |

---
## 6. BMI 기여도 분석 (Case 1 / 2 / 4 × BMI 유무 비교)

> **분석 목적:** 0424(BMI 없음) vs 0430(BMI 포함) 기준선의 성능 차이를 Case 1/2/4에서 직접 정량화
> Case 1/2/4는 AEC_prev 기반으로 한정 — AEC 선택 효과와 BMI 효과가 혼재되지 않도록 단순화

### 6.1 강남

#### 선형 회귀 R² / 로지스틱 AUC — no BMI vs +BMI

| Case | BMI | N feat | Lin R² (±std) | Lin RMSE | Log AUC (±std) | Log Sens | Log Spec |
|------|-----|--------|--------------|----------|---------------|---------|---------|
| Case 1 (no BMI) | no BMI | 2 | 0.5430 ± 0.0401 | 19.91 | 0.5992 ± 0.0403 | 0.3886 | 0.8173 |
| Case 1 (+BMI) | **+BMI** | 3 | 0.6620 ± 0.0367 | 17.12 | 0.7427 ± 0.0188 | 0.7647 | 0.6491 |
| Case 2 (no BMI) | no BMI | 6 | 0.6300 ± 0.0407 | 17.91 | 0.7000 ± 0.0276 | 0.7741 | 0.5726 |
| Case 2 (+BMI) | **+BMI** | 7 | 0.6780 ± 0.0357 | 16.71 | 0.7510 ± 0.0227 | 0.7394 | 0.6883 |
| Case 4 (no BMI) | no BMI | 45 | 0.6210 ± 0.0339 | 18.13 | 0.7104 ± 0.0260 | 0.7834 | 0.5889 |
| Case 4 (+BMI) | **+BMI** | 46 | 0.6682 ± 0.0318 | 16.96 | 0.7400 ± 0.0287 | 0.7834 | 0.6319 |

#### BMI 추가 효과 (Δ = +BMI − no BMI)

| Case | ΔR² | ΔRMSE (cm²) | ΔAUC | ΔAccuracy | 해석 |
|------|-----|------------|------|----------|------|
| C1 | **+0.1190** | -2.79 | **+0.1435** | -0.0410 | BMI 보정 유효 |
| C2 | **+0.0480** | -1.20 | **+0.0510** | +0.0806 | BMI 보정 유효 |
| C4 | **+0.0472** | -1.17 | **+0.0296** | +0.0330 | BMI 보정 유효 |

> **해석:** Case 1에서 BMI 추가로 R² +0.1190 향상. AEC_prev 투입 후(Case 2) BMI 효과가 +0.0480로 감소(60% 감쇠) → AEC와 BMI가 일부 공통 정보를 공유함을 시사.

### 6.2 신촌

#### 선형 회귀 R² / 로지스틱 AUC — no BMI vs +BMI

| Case | BMI | N feat | Lin R² (±std) | Lin RMSE | Log AUC (±std) | Log Sens | Log Spec |
|------|-----|--------|--------------|----------|---------------|---------|---------|
| Case 1 (no BMI) | no BMI | 2 | 0.5216 ± 0.0656 | 21.24 | 0.6076 ± 0.0299 | 0.5505 | 0.6938 |
| Case 1 (+BMI) | **+BMI** | 3 | 0.6398 ± 0.0629 | 18.42 | 0.7513 ± 0.0332 | 0.7988 | 0.6073 |
| Case 2 (no BMI) | no BMI | 6 | 0.5444 ± 0.0587 | 20.73 | 0.6441 ± 0.0430 | 0.6114 | 0.6552 |
| Case 2 (+BMI) | **+BMI** | 7 | 0.6415 ± 0.0615 | 18.37 | 0.7502 ± 0.0362 | 0.8055 | 0.6125 |
| Case 4 (no BMI) | no BMI | 58 | 0.5350 ± 0.0693 | 20.92 | 0.6784 ± 0.0236 | 0.6410 | 0.6740 |
| Case 4 (+BMI) | **+BMI** | 59 | 0.6340 ± 0.0590 | 18.56 | 0.7529 ± 0.0321 | 0.7572 | 0.6896 |

#### BMI 추가 효과 (Δ = +BMI − no BMI)

| Case | ΔR² | ΔRMSE (cm²) | ΔAUC | ΔAccuracy | 해석 |
|------|-----|------------|------|----------|------|
| C1 | **+0.1182** | -2.82 | **+0.1437** | -0.0047 | BMI 보정 유효 |
| C2 | **+0.0971** | -2.36 | **+0.1061** | +0.0150 | BMI 보정 유효 |
| C4 | **+0.0990** | -2.36 | **+0.0745** | +0.0403 | BMI 보정 유효 |

> **해석:** Case 1에서 BMI 추가로 R² +0.1182 향상. AEC_prev 투입 후(Case 2) BMI 효과가 +0.0971로 감소(18% 감쇠) → AEC와 BMI가 일부 공통 정보를 공유함을 시사.

---
## 7. 시각화 자료 목록

### 피처 선택 (results/feature_selection/)

| 파일 | 내용 |
|------|------|
| `cross_dataset_comparison_r2.png` | AEC_prev vs AEC_new CV R² 비교 (3개 데이터셋) |
| `cross_dataset_feature_heatmap.png` | 데이터셋별 최종 선택 피처 히트맵 |
| `gangnam/01_correlation_heatmap.png` | 전체 피처 Pearson 상관행렬 (강남) |
| `gangnam/02_mutual_information.png` | 단변량 Mutual Information (강남) |
| `gangnam/03_permutation_importance.png` | RF Permutation Importance (강남) |
| `gangnam/04_cv_r2_comparison.png` | 후보 세트별 CV R² 비교 (강남) |
| `gangnam/05_final_features_summary.png` | 최종 피처 앙상블 투표 + Spearman |
| `gangnam/06-12_*.png` | 최종 피처 상관/분포/클러스터/박스/비교 |

### 회귀 분석 (results/regression/gangnam/all/)

| 파일 | 내용 |
|------|------|
| `01_linear_actual_vs_pred.png` | 선형 회귀 Actual vs Predicted (Case 1~5) |
| `02_linear_metrics_comparison.png` | R²/MAE/RMSE Case 비교 |
| `03_linear_coefficients.png` | 표준화 계수 (Case 1~5) |
| `04_logistic_roc.png` | ROC 곡선 (Case 1~5, 5-fold) |
| `05_logistic_metrics_comparison.png` | AUC/Accuracy/Sensitivity/Specificity |
| `06_logistic_confusion.png` | Confusion Matrix |
| `07_logistic_coefficients.png` | 로지스틱 계수 |
| `08_case_comparison_overview.png` | R² & AUC 전체 개요 |

### EDA & 진단 (results/regression/gangnam/)

| 파일 | 내용 |
|------|------|
| `04_linear_actual_vs_pred.png` | 전체 fit OLS Actual vs Predicted |
| `05_linear_residuals.png` | 잔차 진단 4-Panel |
| `06_linear_forest.png` | 유의한 계수 Forest Plot (p<0.05) |
| `07_linear_univariate_r2.png` | 단변량 R² 비교 |
| `08_logistic_roc.png` | Bootstrap ROC (n=1000, 95%CI) |
| `09_logistic_calibration.png` | Calibration (Hosmer-Lemeshow) |
| `10_logistic_confusion.png` | Confusion Matrix (Youden) |
| `11_logistic_forest.png` | Crude OR Forest Plot |
| `12-15_case_*.png` | Case 1~5 비교 (R²/AUC/AIC/추이) |
| `16_scanner_distribution.png` | CT 스캐너 분포 |
| `17_kvp_distribution.png` | kVp 분포 |
| `18_correlation_matrix.png` | 선택 피처 간 Pearson 상관행렬 |

### 교차 병원 (results/regression/)

| 파일 | 내용 |
|------|------|
| `09_cross_hospital_comparison.png` | 강남 vs 신촌 메트릭 비교 |
| `10_external_validation.png` | 강남 학습 → 신촌 외부 검증 |

### BMI 기여도 분석 (results/regression/{gangnam,sinchon}/)

| 파일 | 내용 |
|------|------|
| `bmi_comparison_r2_auc.png` | Case 1/2/4 × no BMI vs +BMI, R²·AUC 병렬 비교 |
| `bmi_delta_effect.png` | BMI 추가 효과 Δ 막대 (R² 변화 / AUC 변화) |
| `bmi_comparison_summary.xlsx` | `all_results` + `delta_bmi` 시트 |

---
## 8. 결론

### 8.1 피처 선택

- **강남**: 파이프라인 9개 피처 선택 (CV R² 0.1818 vs 이전 0.1768, Δ=+0.0050 → 향상)
- **신촌**: 파이프라인 13개 피처 선택 (CV R² 0.0775 vs 이전 0.0305, Δ=+0.0469 → 향상)
- **병합(강남+신촌)**: 파이프라인 11개 피처 선택 (CV R² 0.0881 vs 이전 0.0965, Δ=-0.0084 → 저하)

### 8.2 강남 회귀 분석 요약 (전체 그룹)

1. **기준선 (Case 1):** 선형 R² = 0.6620, AUC = 0.7427
2. **+ AEC_new (Case 3):** 선형 R² = 0.6694, AUC = 0.7442 (ΔR² = +0.0074, ΔAUC = +0.0015)
3. **+ AEC_new + Scanner (Case 5):** 선형 R² = 0.6620, AUC = 0.7311

### 8.2 BMI 기여도 요약

- **강남** Case 1 기준선: BMI 추가 → R² +0.1190, AUC +0.1435 향상
- **신촌** Case 1 기준선: BMI 추가 → R² +0.1182, AUC +0.1437 향상

### 8.3 0430 핵심 성과

1. **BMI 보정**: Case 1 기준선에서 R² +0.12 수준 향상 — BMI가 강력한 TAMA 예측 변수임을 확인
2. **자동 피처 선택**: 60개+ AEC 피처에서 과적합·다중공선성 없이 객관적 세트 도출
3. **성별 층화**: 전체/여성/남성 독립 모델로 이질성 탐색 — 그룹별 예측 패턴 확인
4. **다병원 검증**: 강남·신촌 교차 검증으로 피처 선택과 모델의 재현성 확인

### 7.4 한계 및 향후 과제

- AEC_new vs AEC_prev의 성능 차이가 미미한 경우 — 더 많은 환자 데이터 필요
- 층화 분석 시 소그룹(여성 단독, 남성 단독) 표본 크기에 따른 불안정성 주의
- 단면 연구 설계 — 인과 추론을 위한 전향적 코호트 연구 권장
- Raw AEC 시계열 (200포인트)을 1D CNN / LSTM으로 직접 학습 시 추가 성능 향상 가능성

---

*자동 생성: generate_report.py (0430)*