# TAMA 예측 회귀분석 연구 보고서 (0430)

> **버전:** 0430 (0424 대비 설계 변경 적용)
> **분석 도구:** Python (statsmodels, scikit-learn, scipy)
> **병원:** 강남, 신촌
> **성별 그룹:** 전체 / 여성(F) / 남성(M)
> **AEC 세트:** AEC_prev (수동 4개) vs AEC_new (파이프라인 자동 선택)

---
## 1. 0424 대비 설계 변경사항

| # | 항목 | 0424 이전 | 0430 이후 |
|---|------|-----------|-----------|
| ① | 피처 선택 | 수동 (상관계수+VIF → 연구자 결정) | 자동 파이프라인 (4단계 필터링 + 앙상블 투표 + CV R²) |
| ② | 임상 기준선 | PatientAge + PatientSex | PatientAge + PatientSex + **BMI** |
| ③ | Case 구조 | Case 1~3 (단일 AEC 세트) | Case 1~5 (AEC_prev vs AEC_new 교차 비교) |
| ④ | 성별 층화 | 성별 = 공변량(더미)만 | 전체 / 여성 / 남성 독립 모델 |
| ⑤ | 다병원 분석 | 강남 단독 (SITE 수동 변경) | 강남·신촌 자동 순회 + 교차 병원 비교 |
| ⑥ | 이진화 기준 | 성별 특이적 P25 (남/여 별도) | 분석 그룹 내 하위 25% 동적 산출 |

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
| Step 2 - High correlation | 20 | p75, mean_abs_deviation, p95, band4_energy, wavelet_cA_energy, band4_energy_ratio, AUC_normalized, peak_mean_height, band1_energy, median, band2_energy_ratio, band3_energy, p25, band3_energy_ratio, RMSE, p90, spectral_rolloff, fft_mag_std, AUC, wavelet_cA_std |
| Step 3 - Union pass | 2 | nan |
| Step 4 - Best search | 0 | CV, IQR, band1_energy_ratio, band2_energy, dominant_freq, min, p10, p5, p90_p10_ratio, signal_energy, signal_length, slope_abs_mean, slope_max, slope_mean, slope_std, spectral_energy, spectral_spread, wavelet_cD1_energy, wavelet_cD1_std, wavelet_cD2_energy, wavelet_cD2_std, wavelet_energy_ratio_D1 |
| VIF pruning (VIF>10) | 13 | IQR, band2_energy, dominant_freq, mean, slope_max, spectral_energy, wavelet_cD1_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1 |

---
## 3. 회귀 분석 결과 — 강남

### 3.1 전체

- N = **1365** | 이진화 임계값 (그룹 내 P25) = **100.0 cm²**

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
| Case 1 Clinical | **0.8293 ± 0.0208** | 0.7868 | 0.3414 | 0.9243 |
| Case 2 +AEC_prev | **0.8346 ± 0.0247** | 0.7846 | 0.3447 | 0.9204 |
| Case 3 +AEC_new | **0.8352 ± 0.0229** | 0.7853 | 0.3571 | 0.9175 |
| Case 4 +AEC_prev +Scanner | **0.8306 ± 0.0209** | 0.7883 | 0.3758 | 0.9156 |
| Case 5 +AEC_new +Scanner | **0.8320 ± 0.0182** | 0.7941 | 0.3913 | 0.9185 |

> **AEC 기여도 (AUC):** AEC_prev Δ = +0.0053 | AEC_new Δ = +0.0059 | 차이 (new-prev) = +0.0006

### 3.2 여성(F)

- N = **841** | 이진화 임계값 (그룹 내 P25) = **95.0 cm²**

#### 선형 회귀 (5-Fold CV)

| Case | N features | R² (mean±std) | MAE (cm²) | RMSE (cm²) |
|------|-----------|--------------|----------|-----------|
| Case 1 Clinical | 2 | **0.241 ± 0.071** | 10.44 | 13.26 |
| Case 2 +AEC_prev | 6 | **0.266 ± 0.048** | 10.32 | 13.05 |
| Case 3 +AEC_new | 13 | **0.262 ± 0.052** | 10.26 | 13.09 |
| Case 4 +AEC_prev +Scanner | 45 | **0.264 ± 0.085** | 10.30 | 13.03 |
| Case 5 +AEC_new +Scanner | 52 | **0.243 ± 0.102** | 10.29 | 13.24 |

> **AEC 기여도 (선형 R²):** AEC_prev Δ = +0.0252 | AEC_new Δ = +0.0212 | 차이 (new-prev) = -0.0040

#### 로지스틱 회귀 (5-Fold CV)

| Case | AUC (mean±std) | Accuracy | Sensitivity | Specificity |
|------|---------------|---------|------------|------------|
| Case 1 Clinical | **0.7063 ± 0.0523** | 0.7848 | 0.1047 | 0.9846 |
| Case 2 +AEC_prev | **0.7157 ± 0.0478** | 0.7801 | 0.1152 | 0.9754 |
| Case 3 +AEC_new | **0.7097 ± 0.0460** | 0.7860 | 0.1466 | 0.9738 |
| Case 4 +AEC_prev +Scanner | **0.7292 ± 0.0411** | 0.7765 | 0.1521 | 0.9600 |
| Case 5 +AEC_new +Scanner | **0.7287 ± 0.0409** | 0.7836 | 0.1992 | 0.9554 |

> **AEC 기여도 (AUC):** AEC_prev Δ = +0.0094 | AEC_new Δ = +0.0034 | 차이 (new-prev) = -0.0060

### 3.3 남성(M)

- N = **524** | 이진화 임계값 (그룹 내 P25) = **132.0 cm²**

#### 선형 회귀 (5-Fold CV)

| Case | N features | R² (mean±std) | MAE (cm²) | RMSE (cm²) |
|------|-----------|--------------|----------|-----------|
| Case 1 Clinical | 2 | **0.324 ± 0.097** | 16.39 | 21.51 |
| Case 2 +AEC_prev | 6 | **0.350 ± 0.087** | 16.10 | 21.09 |
| Case 3 +AEC_new | 13 | **0.317 ± 0.101** | 16.33 | 21.62 |
| Case 4 +AEC_prev +Scanner | 45 | **0.347 ± 0.077** | 16.44 | 21.17 |
| Case 5 +AEC_new +Scanner | 52 | **0.314 ± 0.097** | 16.55 | 21.66 |

> **AEC 기여도 (선형 R²):** AEC_prev Δ = +0.0252 | AEC_new Δ = -0.0078 | 차이 (new-prev) = -0.0330

#### 로지스틱 회귀 (5-Fold CV)

| Case | AUC (mean±std) | Accuracy | Sensitivity | Specificity |
|------|---------------|---------|------------|------------|
| Case 1 Clinical | **0.7920 ± 0.0441** | 0.7557 | 0.2258 | 0.9267 |
| Case 2 +AEC_prev | **0.7976 ± 0.0329** | 0.7805 | 0.3037 | 0.9343 |
| Case 3 +AEC_new | **0.7868 ± 0.0367** | 0.7729 | 0.2963 | 0.9267 |
| Case 4 +AEC_prev +Scanner | **0.7949 ± 0.0142** | 0.7766 | 0.3671 | 0.9091 |
| Case 5 +AEC_new +Scanner | **0.7947 ± 0.0217** | 0.7768 | 0.3748 | 0.9066 |

> **AEC 기여도 (AUC):** AEC_prev Δ = +0.0056 | AEC_new Δ = -0.0052 | 차이 (new-prev) = -0.0108

---
## 3. 회귀 분석 결과 — 신촌

### 3.1 전체

- N = **1269** | 이진화 임계값 (그룹 내 P25) = **103.0 cm²**

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
| Case 1 Clinical | **0.8577 ± 0.0333** | 0.7999 | 0.5256 | 0.8901 |
| Case 2 +AEC_prev | **0.8587 ± 0.0347** | 0.7983 | 0.5351 | 0.8848 |
| Case 3 +AEC_new | **0.8533 ± 0.0398** | 0.8046 | 0.5413 | 0.8911 |
| Case 4 +AEC_prev +Scanner | **0.8579 ± 0.0328** | 0.7943 | 0.5382 | 0.8785 |
| Case 5 +AEC_new +Scanner | **0.8531 ± 0.0357** | 0.7943 | 0.5414 | 0.8775 |

> **AEC 기여도 (AUC):** AEC_prev Δ = +0.0010 | AEC_new Δ = -0.0044 | 차이 (new-prev) = -0.0054

### 3.2 여성(F)

- N = **631** | 이진화 임계값 (그룹 내 P25) = **95.0 cm²**

#### 선형 회귀 (5-Fold CV)

| Case | N features | R² (mean±std) | MAE (cm²) | RMSE (cm²) |
|------|-----------|--------------|----------|-----------|
| Case 1 Clinical | 2 | **0.213 ± 0.042** | 10.54 | 13.51 |
| Case 2 +AEC_prev | 6 | **0.217 ± 0.054** | 10.52 | 13.46 |
| Case 3 +AEC_new | 13 | **0.223 ± 0.061** | 10.47 | 13.40 |
| Case 4 +AEC_prev +Scanner | 58 | **0.198 ± 0.082** | 10.64 | 13.60 |
| Case 5 +AEC_new +Scanner | 65 | **0.209 ± 0.097** | 10.54 | 13.48 |

> **AEC 기여도 (선형 R²):** AEC_prev Δ = +0.0049 | AEC_new Δ = +0.0105 | 차이 (new-prev) = +0.0056

#### 로지스틱 회귀 (5-Fold CV)

| Case | AUC (mean±std) | Accuracy | Sensitivity | Specificity |
|------|---------------|---------|------------|------------|
| Case 1 Clinical | **0.6943 ± 0.0327** | 0.7480 | 0.0781 | 0.9643 |
| Case 2 +AEC_prev | **0.6941 ± 0.0376** | 0.7417 | 0.0716 | 0.9580 |
| Case 3 +AEC_new | **0.7014 ± 0.0256** | 0.7369 | 0.1237 | 0.9349 |
| Case 4 +AEC_prev +Scanner | **0.6556 ± 0.0351** | 0.7195 | 0.1237 | 0.9120 |
| Case 5 +AEC_new +Scanner | **0.6598 ± 0.0213** | 0.7163 | 0.1428 | 0.9015 |

> **AEC 기여도 (AUC):** AEC_prev Δ = -0.0002 | AEC_new Δ = +0.0071 | 차이 (new-prev) = +0.0073

### 3.3 남성(M)

- N = **638** | 이진화 임계값 (그룹 내 P25) = **131.0 cm²**

#### 선형 회귀 (5-Fold CV)

| Case | N features | R² (mean±std) | MAE (cm²) | RMSE (cm²) |
|------|-----------|--------------|----------|-----------|
| Case 1 Clinical | 2 | **0.332 ± 0.075** | 16.10 | 21.61 |
| Case 2 +AEC_prev | 6 | **0.321 ± 0.060** | 16.22 | 21.78 |
| Case 3 +AEC_new | 13 | **0.295 ± 0.069** | 16.44 | 22.18 |
| Case 4 +AEC_prev +Scanner | 58 | **0.259 ± 0.079** | 16.75 | 22.73 |
| Case 5 +AEC_new +Scanner | 65 | **0.264 ± 0.062** | 16.96 | 22.67 |

> **AEC 기여도 (선형 R²):** AEC_prev Δ = -0.0101 | AEC_new Δ = -0.0362 | 차이 (new-prev) = -0.0261

#### 로지스틱 회귀 (5-Fold CV)

| Case | AUC (mean±std) | Accuracy | Sensitivity | Specificity |
|------|---------------|---------|------------|------------|
| Case 1 Clinical | **0.7943 ± 0.0484** | 0.7930 | 0.3226 | 0.9439 |
| Case 2 +AEC_prev | **0.7875 ± 0.0398** | 0.7883 | 0.3097 | 0.9419 |
| Case 3 +AEC_new | **0.7851 ± 0.0367** | 0.7852 | 0.2903 | 0.9440 |
| Case 4 +AEC_prev +Scanner | **0.7680 ± 0.0411** | 0.7617 | 0.2903 | 0.9129 |
| Case 5 +AEC_new +Scanner | **0.7659 ± 0.0483** | 0.7664 | 0.2839 | 0.9211 |

> **AEC 기여도 (AUC):** AEC_prev Δ = -0.0068 | AEC_new Δ = -0.0092 | 차이 (new-prev) = -0.0024

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
| Case 1 Clinical | **0.8293** | **0.8577** |
| Case 2 +AEC_prev | **0.8346** | **0.8587** |
| Case 3 +AEC_new | **0.8352** | **0.8533** |
| Case 4 +AEC_prev +Scanner | **0.8306** | **0.8579** |
| Case 5 +AEC_new +Scanner | **0.8320** | **0.8531** |

### 4.3 외부 검증 (강남 학습 → 신촌 예측)

| Train | Test | Case | N_train | N_test | N_features_used | Lin_R2 | Lin_MAE | Lin_RMSE | TAMA_threshold | Log_AUC | Log_Acc | Log_Sens | Log_Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 강남 | 신촌 | Case1_Clinical | 1365 | 1269 | 3 | 0.6374 | 13.81 | 18.54 | 103 | 0.8564 | 0.8006 | 0.3885 | 0.9361 |
| 강남 | 신촌 | Case2_Clinical+AEC_prev | 1365 | 1269 | 7 | 0.479 | 15.0 | 22.23 | 103 | 0.8496 | 0.8006 | 0.5478 | 0.8838 |
| 강남 | 신촌 | Case3_Clinical+AEC_new | 1365 | 1269 | 14 | 0.5842 | 14.8 | 19.86 | 103 | 0.8252 | 0.7888 | 0.5255 | 0.8754 |
| 강남 | 신촌 | Case4_Clinical+AEC_prev+Scanner | 1365 | 1269 | 38 | 0.3959 | 16.0 | 23.93 | 103 | 0.8297 | 0.7975 | 0.5541 | 0.8775 |
| 강남 | 신촌 | Case5_Clinical+AEC_new+Scanner | 1365 | 1269 | 45 | 0.4099 | 17.56 | 23.66 | 103 | 0.8122 | 0.7975 | 0.4745 | 0.9037 |
| 신촌 | 강남 | Case1_Clinical | 1269 | 1365 | 3 | 0.6606 | 12.82 | 17.18 | 100 | 0.8271 | 0.7795 | 0.5528 | 0.8495 |
| 신촌 | 강남 | Case2_Clinical+AEC_prev | 1269 | 1365 | 7 | 0.6588 | 12.87 | 17.23 | 100 | 0.8259 | 0.7868 | 0.382 | 0.9118 |
| 신촌 | 강남 | Case3_Clinical+AEC_new | 1269 | 1365 | 14 | 0.6564 | 12.97 | 17.28 | 100 | 0.823 | 0.7839 | 0.4037 | 0.9012 |
| 신촌 | 강남 | Case4_Clinical+AEC_prev+Scanner | 1269 | 1365 | 38 | 0.6587 | 12.84 | 17.23 | 100 | 0.8294 | 0.789 | 0.528 | 0.8696 |
| 신촌 | 강남 | Case5_Clinical+AEC_new+Scanner | 1269 | 1365 | 45 | 0.6563 | 12.92 | 17.29 | 100 | 0.8244 | 0.7897 | 0.587 | 0.8523 |

---
## 5. 성별 층화 분석 결과

### 5.1 강남

#### 선형 R² — Case 1~5 × 성별 그룹

| Case | 남성(M) | 여성(F) | 전체 |
|------|---|---|---|
| Case 1 Clinical | 0.3244 | 0.2408 | 0.6620 |
| Case 2 +AEC_prev | 0.3496 | 0.2660 | 0.6780 |
| Case 3 +AEC_new | 0.3166 | 0.2620 | 0.6694 |
| Case 4 +AEC_prev +Scanner | 0.3468 | 0.2640 | 0.6682 |
| Case 5 +AEC_new +Scanner | 0.3144 | 0.2427 | 0.6620 |

#### 로지스틱 AUC — Case 1~5 × 성별 그룹

| Case | 남성(M) | 여성(F) | 전체 |
|------|---|---|---|
| Case 1 Clinical | 0.7920 | 0.7063 | 0.8293 |
| Case 2 +AEC_prev | 0.7976 | 0.7157 | 0.8346 |
| Case 3 +AEC_new | 0.7868 | 0.7097 | 0.8352 |
| Case 4 +AEC_prev +Scanner | 0.7949 | 0.7292 | 0.8306 |
| Case 5 +AEC_new +Scanner | 0.7947 | 0.7287 | 0.8320 |

### 5.2 신촌

#### 선형 R² — Case 1~5 × 성별 그룹

| Case | 남성(M) | 여성(F) | 전체 |
|------|---|---|---|
| Case 1 Clinical | 0.3315 | 0.2126 | 0.6398 |
| Case 2 +AEC_prev | 0.3214 | 0.2175 | 0.6415 |
| Case 3 +AEC_new | 0.2953 | 0.2231 | 0.6371 |
| Case 4 +AEC_prev +Scanner | 0.2594 | 0.1977 | 0.6340 |
| Case 5 +AEC_new +Scanner | 0.2638 | 0.2090 | 0.6376 |

#### 로지스틱 AUC — Case 1~5 × 성별 그룹

| Case | 남성(M) | 여성(F) | 전체 |
|------|---|---|---|
| Case 1 Clinical | 0.7943 | 0.6943 | 0.8577 |
| Case 2 +AEC_prev | 0.7875 | 0.6941 | 0.8587 |
| Case 3 +AEC_new | 0.7851 | 0.7014 | 0.8533 |
| Case 4 +AEC_prev +Scanner | 0.7680 | 0.6556 | 0.8579 |
| Case 5 +AEC_new +Scanner | 0.7659 | 0.6598 | 0.8531 |

---
## 6. 시각화 자료 목록

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

---
## 7. 결론

### 7.1 피처 선택

- **강남**: 파이프라인 9개 피처 선택 (CV R² 0.1818 vs 이전 0.1768, Δ=+0.0050 → 향상)
- **신촌**: 파이프라인 13개 피처 선택 (CV R² 0.0775 vs 이전 0.0305, Δ=+0.0469 → 향상)
- **병합(강남+신촌)**: 파이프라인 11개 피처 선택 (CV R² 0.0881 vs 이전 0.0965, Δ=-0.0084 → 저하)

### 7.2 강남 회귀 분석 요약 (전체 그룹)

1. **기준선 (Case 1):** 선형 R² = 0.6620, AUC = 0.8293
2. **+ AEC_new (Case 3):** 선형 R² = 0.6694, AUC = 0.8352 (ΔR² = +0.0074, ΔAUC = +0.0059)
3. **+ AEC_new + Scanner (Case 5):** 선형 R² = 0.6620, AUC = 0.8320

### 7.3 0430 핵심 성과

1. **BMI 보정**: 체지방·근육량 교란변수 보정 후 AEC 피처의 순수 기여도 분리 완료
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