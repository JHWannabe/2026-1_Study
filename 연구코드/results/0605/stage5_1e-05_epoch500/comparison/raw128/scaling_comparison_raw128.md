# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8207 | 0.3676 | 0.2712 | 0.6725 | 0.3697 |
| M2_2 | CrossAttn | norm | 0.8378 | 0.3501 | 0.2633 | 0.6725 | 0.3697 |
| M3 | CrossAttn3 | norm | 0.8311 | 0.3368 | 0.2537 | 0.7642 | 0.4255 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_all** | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |

---

## Model 2 — Clinic + AEC (Matched)  (4 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7837 | 0.2708 | 0.2573 | 0.6550 | 0.3471 |
| std_scaled | 0.7909 | 0.2800 | 0.2559 | 0.7642 | 0.3571 |
| **norm** | 0.8207 | 0.3676 | 0.2712 | 0.6725 | 0.3697 |
| global_zscore | 0.8203 | 0.2909 | 0.2587 | 0.7031 | 0.3929 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7701 | 0.2803 | 0.2738 | 0.6550 | 0.3361 |
| std_scaled | 0.7783 | 0.3174 | 0.2655 | 0.6594 | 0.3276 |
| **norm** | 0.8378 | 0.3501 | 0.2633 | 0.6725 | 0.3697 |
| global_zscore | 0.7872 | 0.3191 | 0.2711 | 0.6463 | 0.3415 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7890 | 0.2688 | 0.2619 | 0.7293 | 0.3542 |
| std_scaled | 0.7587 | 0.2599 | 0.2823 | 0.7249 | 0.3505 |
| **norm** | 0.8311 | 0.3368 | 0.2537 | 0.7642 | 0.4255 |
| global_zscore | 0.8144 | 0.3308 | 0.2589 | 0.7205 | 0.3725 |

---

# Cross-Model Comparison — Fold-level Statistical Tests

> Paired t-test + Wilcoxon signed-rank (n=5 folds).
> p-value는 지수표현. Δ Mean = B − A (양수 → B 우세).
> M1·M2·M3 간 pairwise 비교 (M2_2 음성 대조군 제외).
> M1은 단일 case로 M2/M3 각 AEC variant와 개별 비교.
> M1↔M2/M3는 데이터셋이 다를 수 있으므로 해석 시 주의.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10

## M1 (LR) vs M2 (CrossAttn)

> A = M1 LR, B = M2 CrossAttn.

### raw  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8073 | +0.0011 | -0.079 | 9.41e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3851 | -0.0242 | 0.837 | 4.50e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2459 | +0.0651 | -15.328 | 1.06e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7166 | -0.0395 | 1.543 | 1.98e-01 | 1.88e-01 |
| F1  | 0.4163 | 0.3844 | -0.0319 | 1.148 | 3.15e-01 | 4.38e-01 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8098 | +0.0036 | -0.166 | 8.76e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3739 | -0.0353 | 1.075 | 3.43e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2498 | +0.0690 | -13.735 | 1.63e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7593 | +0.0033 | -0.050 | 9.62e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4218 | +0.0054 | -0.109 | 9.18e-01 | 8.12e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8094 | +0.0033 | -0.237 | 8.24e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4011 | -0.0082 | 0.342 | 7.50e-01 | 8.12e-01 |
| Brier ** | 0.1808 | 0.2643 | +0.0835 | -7.886 | 1.40e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7516 | -0.0045 | 0.190 | 8.59e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4122 | -0.0041 | 0.170 | 8.74e-01 | 1.00e+00 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8138 | +0.0077 | -0.462 | 6.68e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.4069 | -0.0023 | 0.073 | 9.45e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2532 | +0.0724 | -11.039 | 3.83e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7287 | -0.0274 | 0.718 | 5.12e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4081 | -0.0083 | 0.239 | 8.23e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8073 | 0.8032 | -0.0041 | 0.371 | 7.30e-01 | 1.00e+00 |
| AUPRC  | 0.3851 | 0.3683 | -0.0167 | 0.919 | 4.10e-01 | 4.38e-01 |
| Brier * | 0.2459 | 0.2537 | +0.0078 | -3.007 | 3.97e-02 | 1.25e-01 |
| Accuracy  | 0.7166 | 0.7123 | -0.0043 | 0.114 | 9.15e-01 | 1.00e+00 |
| F1  | 0.3844 | 0.3818 | -0.0026 | 0.099 | 9.26e-01 | 1.00e+00 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8098 | 0.7992 | -0.0106 | 0.516 | 6.33e-01 | 1.00e+00 |
| AUPRC  | 0.3739 | 0.3530 | -0.0209 | 0.704 | 5.20e-01 | 8.12e-01 |
| Brier  | 0.2498 | 0.2519 | +0.0021 | -0.322 | 7.64e-01 | 1.00e+00 |
| Accuracy  | 0.7593 | 0.7878 | +0.0284 | -1.047 | 3.54e-01 | 4.38e-01 |
| F1  | 0.4218 | 0.4257 | +0.0040 | -0.119 | 9.11e-01 | 6.25e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8094 | 0.8005 | -0.0090 | 0.834 | 4.51e-01 | 6.25e-01 |
| AUPRC  | 0.4011 | 0.3852 | -0.0159 | 0.569 | 6.00e-01 | 6.25e-01 |
| Brier  | 0.2643 | 0.2456 | -0.0186 | 1.557 | 1.94e-01 | 3.12e-01 |
| Accuracy  | 0.7516 | 0.7605 | +0.0088 | -0.224 | 8.34e-01 | 8.12e-01 |
| F1  | 0.4122 | 0.4067 | -0.0055 | 0.258 | 8.09e-01 | 8.12e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8138 | 0.7971 | -0.0167 | 1.535 | 2.00e-01 | 3.12e-01 |
| AUPRC  | 0.4069 | 0.3785 | -0.0284 | 1.037 | 3.58e-01 | 4.38e-01 |
| Brier  | 0.2532 | 0.2519 | -0.0013 | 0.256 | 8.10e-01 | 8.12e-01 |
| Accuracy  | 0.7287 | 0.7572 | +0.0285 | -0.441 | 6.82e-01 | 6.25e-01 |
| F1  | 0.4081 | 0.4156 | +0.0075 | -0.149 | 8.89e-01 | 8.75e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8032 | -0.0030 | 0.130 | 9.02e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3683 | -0.0409 | 1.038 | 3.58e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2537 | +0.0729 | -16.873 | 7.23e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7123 | -0.0437 | 1.631 | 1.78e-01 | 1.88e-01 |
| F1  | 0.4163 | 0.3818 | -0.0346 | 1.136 | 3.19e-01 | 6.25e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7992 | -0.0069 | 0.373 | 7.28e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3530 | -0.0562 | 1.319 | 2.58e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2519 | +0.0711 | -10.350 | 4.92e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7878 | +0.0317 | -0.538 | 6.19e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4257 | +0.0094 | -0.211 | 8.43e-01 | 1.00e+00 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8005 | -0.0057 | 0.363 | 7.35e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3852 | -0.0241 | 0.548 | 6.13e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2456 | +0.0649 | -34.800 | 4.07e-06 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7605 | +0.0044 | -0.099 | 9.26e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4067 | -0.0096 | 0.343 | 7.49e-01 | 1.00e+00 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7971 | -0.0090 | 0.439 | 6.83e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3785 | -0.0307 | 0.570 | 5.99e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2519 | +0.0711 | -12.499 | 2.36e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7572 | +0.0011 | -0.019 | 9.86e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4156 | -0.0008 | 0.016 | 9.88e-01 | 8.12e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_all | AUC-ROC | 0.8030 | 0.7200 | 0.8724 |
| M1 | LR | scale_all | AUPRC | 0.3123 | 0.1806 | 0.4955 |
| M1 | LR | scale_all | Brier | 0.1913 | 0.1647 | 0.2175 |
| M1 | LR | scale_all | Accuracy | 0.7205 | 0.6638 | 0.7817 |
| M1 | LR | scale_all | F1 | 0.3725 | 0.2500 | 0.4884 |
| M2 | CrossAttn | raw | AUC-ROC | 0.7837 | 0.6995 | 0.8588 |
| M2 | CrossAttn | raw | AUPRC | 0.2708 | 0.1610 | 0.4276 |
| M2 | CrossAttn | raw | Brier | 0.2573 | 0.2306 | 0.2820 |
| M2 | CrossAttn | raw | Accuracy | 0.6550 | 0.5895 | 0.7205 |
| M2 | CrossAttn | raw | F1 | 0.3471 | 0.2301 | 0.4553 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7909 | 0.6937 | 0.8708 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2800 | 0.1643 | 0.4404 |
| M2 | CrossAttn | std_scaled | Brier | 0.2559 | 0.2286 | 0.2811 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7642 | 0.7074 | 0.8166 |
| M2 | CrossAttn | std_scaled | F1 | 0.3571 | 0.2258 | 0.4828 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8207 | 0.7446 | 0.8867 |
| M2 | CrossAttn | norm | AUPRC | 0.3676 | 0.2176 | 0.5433 |
| M2 | CrossAttn | norm | Brier | 0.2712 | 0.2468 | 0.2944 |
| M2 | CrossAttn | norm | Accuracy | 0.6725 | 0.6114 | 0.7380 |
| M2 | CrossAttn | norm | F1 | 0.3697 | 0.2500 | 0.4837 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.8203 | 0.7416 | 0.8917 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2909 | 0.1916 | 0.4729 |
| M2 | CrossAttn | global_zscore | Brier | 0.2587 | 0.2324 | 0.2828 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.7031 | 0.6419 | 0.7642 |
| M2 | CrossAttn | global_zscore | F1 | 0.3929 | 0.2689 | 0.5079 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7701 | 0.6817 | 0.8481 |
| M2_2 | CrossAttn | raw | AUPRC | 0.2803 | 0.1603 | 0.4703 |
| M2_2 | CrossAttn | raw | Brier | 0.2738 | 0.2512 | 0.2949 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6550 | 0.5939 | 0.7162 |
| M2_2 | CrossAttn | raw | F1 | 0.3361 | 0.2243 | 0.4444 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7783 | 0.6901 | 0.8554 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3174 | 0.1792 | 0.4992 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2655 | 0.2402 | 0.2902 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6594 | 0.5983 | 0.7205 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3276 | 0.2151 | 0.4348 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8378 | 0.7647 | 0.8990 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3501 | 0.2179 | 0.5584 |
| M2_2 | CrossAttn | norm | Brier | 0.2633 | 0.2369 | 0.2899 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6725 | 0.6114 | 0.7380 |
| M2_2 | CrossAttn | norm | F1 | 0.3697 | 0.2500 | 0.4844 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7872 | 0.6935 | 0.8657 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3191 | 0.1804 | 0.5027 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2711 | 0.2451 | 0.2971 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6463 | 0.5852 | 0.7075 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3415 | 0.2280 | 0.4516 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7890 | 0.6956 | 0.8737 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2688 | 0.1760 | 0.4459 |
| M3 | CrossAttn3 | raw | Brier | 0.2619 | 0.2357 | 0.2872 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7293 | 0.6725 | 0.7860 |
| M3 | CrossAttn3 | raw | F1 | 0.3542 | 0.2299 | 0.4783 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7587 | 0.6601 | 0.8426 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2599 | 0.1577 | 0.4475 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2823 | 0.2554 | 0.3068 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.7249 | 0.6681 | 0.7817 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3505 | 0.2273 | 0.4696 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8311 | 0.7506 | 0.8950 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3368 | 0.2007 | 0.5227 |
| M3 | CrossAttn3 | norm | Brier | 0.2537 | 0.2269 | 0.2785 |
| M3 | CrossAttn3 | norm | Accuracy | 0.7642 | 0.7118 | 0.8210 |
| M3 | CrossAttn3 | norm | F1 | 0.4255 | 0.2898 | 0.5512 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.8144 | 0.7341 | 0.8920 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.3308 | 0.2025 | 0.5219 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2589 | 0.2332 | 0.2837 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7205 | 0.6638 | 0.7817 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3725 | 0.2526 | 0.4951 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7837 | -0.0193 | 0.826 | 4.091e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7909 | -0.0122 | 0.586 | 5.576e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8207 | +0.0177 | -0.814 | 4.155e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.8203 | +0.0173 | -0.678 | 4.975e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7890 | -0.0140 | 0.433 | 6.648e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7587 | -0.0443 | 1.458 | 1.447e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8311 | +0.0280 | -1.374 | 1.695e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.8144 | +0.0114 | -0.378 | 7.052e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7837 | 0.7701 | -0.0136 | 0.634 | 5.264e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7909 | 0.7783 | -0.0126 | 0.685 | 4.935e-01 | ns |
| M2-norm vs M2_2-norm | 0.8207 | 0.8378 | +0.0171 | -0.966 | 3.342e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.8203 | 0.7872 | -0.0331 | 1.311 | 1.899e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7837 | 0.7890 | +0.0053 | -0.221 | 8.248e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7909 | 0.7587 | -0.0321 | 1.566 | 1.174e-01 | ns |
| M2-norm vs M3-norm | 0.8207 | 0.8311 | +0.0104 | -0.759 | 4.478e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.8203 | 0.8144 | -0.0059 | 0.382 | 7.027e-01 | ns |

