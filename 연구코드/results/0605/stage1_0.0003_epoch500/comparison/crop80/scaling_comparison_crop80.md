# Scaling Comparison — Test Set Performance (AEC 103pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8236 | 0.3108 | 0.2975 | 0.6376 | 0.3566 |
| M2_2 | CrossAttn | norm | 0.8409 | 0.3889 | 0.2813 | 0.6856 | 0.3793 |
| M3 | CrossAttn3 | norm | 0.8272 | 0.3088 | 0.2740 | 0.6769 | 0.3833 |

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
| raw | 0.8051 | 0.3011 | 0.2603 | 0.6681 | 0.3667 |
| std_scaled | 0.7945 | 0.2877 | 0.2828 | 0.6681 | 0.3333 |
| **norm** | 0.8236 | 0.3108 | 0.2975 | 0.6376 | 0.3566 |
| global_zscore | 0.7909 | 0.2773 | 0.2606 | 0.6332 | 0.3333 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.8120 | 0.3480 | 0.2829 | 0.6157 | 0.3231 |
| std_scaled | 0.7770 | 0.3586 | 0.3073 | 0.5459 | 0.2778 |
| **norm** | 0.8409 | 0.3889 | 0.2813 | 0.6856 | 0.3793 |
| global_zscore | 0.7500 | 0.2811 | 0.2813 | 0.6114 | 0.2764 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.8010 | 0.2639 | 0.2561 | 0.6332 | 0.3333 |
| std_scaled | 0.8022 | 0.2988 | 0.2567 | 0.7467 | 0.3958 |
| **norm** | 0.8272 | 0.3088 | 0.2740 | 0.6769 | 0.3833 |
| global_zscore | 0.7970 | 0.2747 | 0.2990 | 0.7118 | 0.3400 |

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
| AUC-ROC  | 0.8061 | 0.8216 | +0.0155 | -1.019 | 3.66e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4048 | -0.0044 | 0.116 | 9.13e-01 | 6.25e-01 |
| Brier ** | 0.1808 | 0.2466 | +0.0658 | -7.212 | 1.96e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7036 | -0.0525 | 1.984 | 1.18e-01 | 1.25e-01 |
| F1  | 0.4163 | 0.3899 | -0.0265 | 0.991 | 3.78e-01 | 4.38e-01 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8221 | +0.0159 | -1.246 | 2.81e-01 | 3.12e-01 |
| AUPRC  | 0.4092 | 0.4073 | -0.0020 | 0.071 | 9.47e-01 | 8.12e-01 |
| Brier ** | 0.1808 | 0.2494 | +0.0687 | -6.470 | 2.94e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7550 | -0.0011 | 0.033 | 9.75e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4304 | +0.0141 | -0.391 | 7.16e-01 | 6.25e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8148 | +0.0087 | -0.927 | 4.07e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3882 | -0.0210 | 0.633 | 5.61e-01 | 1.00e+00 |
| Brier ** | 0.1808 | 0.2414 | +0.0606 | -5.970 | 3.95e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7714 | +0.0153 | -0.688 | 5.29e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4311 | +0.0148 | -1.012 | 3.69e-01 | 3.12e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8107 | +0.0046 | -0.270 | 8.01e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3665 | -0.0427 | 0.903 | 4.17e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2416 | +0.0608 | -9.559 | 6.69e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6893 | -0.0668 | 1.491 | 2.10e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.3852 | -0.0312 | 0.900 | 4.19e-01 | 4.38e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8216 | 0.8157 | -0.0059 | 1.099 | 3.33e-01 | 3.12e-01 |
| AUPRC  | 0.4048 | 0.3825 | -0.0223 | 1.054 | 3.51e-01 | 6.25e-01 |
| Brier  | 0.2466 | 0.2277 | -0.0189 | 1.075 | 3.43e-01 | 3.12e-01 |
| Accuracy  | 0.7036 | 0.7550 | +0.0514 | -1.773 | 1.51e-01 | 1.88e-01 |
| F1  | 0.3899 | 0.4111 | +0.0212 | -1.030 | 3.61e-01 | 4.38e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8221 | 0.8131 | -0.0089 | 1.099 | 3.33e-01 | 4.38e-01 |
| AUPRC  | 0.4073 | 0.4196 | +0.0124 | -0.370 | 7.30e-01 | 8.12e-01 |
| Brier  | 0.2494 | 0.2393 | -0.0101 | 0.942 | 3.99e-01 | 6.25e-01 |
| Accuracy  | 0.7550 | 0.7921 | +0.0371 | -1.119 | 3.26e-01 | 3.12e-01 |
| F1  | 0.4304 | 0.4310 | +0.0006 | -0.021 | 9.84e-01 | 8.12e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8148 | 0.8153 | +0.0005 | -0.069 | 9.48e-01 | 6.25e-01 |
| AUPRC  | 0.3882 | 0.3774 | -0.0108 | 0.627 | 5.65e-01 | 6.25e-01 |
| Brier  | 0.2414 | 0.2364 | -0.0050 | 0.747 | 4.96e-01 | 6.25e-01 |
| Accuracy  | 0.7714 | 0.7604 | -0.0110 | 0.775 | 4.82e-01 | 6.25e-01 |
| F1  | 0.4311 | 0.4271 | -0.0040 | 0.305 | 7.76e-01 | 1.00e+00 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8107 | 0.8141 | +0.0033 | -0.973 | 3.86e-01 | 4.38e-01 |
| AUPRC † | 0.3665 | 0.3903 | +0.0238 | -2.253 | 8.74e-02 | 1.25e-01 |
| Brier  | 0.2416 | 0.2341 | -0.0075 | 0.589 | 5.87e-01 | 6.25e-01 |
| Accuracy † | 0.6893 | 0.8205 | +0.1312 | -2.370 | 7.68e-02 | 1.25e-01 |
| F1  | 0.3852 | 0.4633 | +0.0781 | -2.030 | 1.12e-01 | 1.88e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8157 | +0.0096 | -0.543 | 6.16e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3825 | -0.0267 | 0.484 | 6.54e-01 | 1.00e+00 |
| Brier * | 0.1808 | 0.2277 | +0.0469 | -3.238 | 3.17e-02 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7550 | -0.0011 | 0.042 | 9.69e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4111 | -0.0052 | 0.207 | 8.46e-01 | 8.12e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8131 | +0.0070 | -0.618 | 5.70e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4196 | +0.0104 | -0.222 | 8.35e-01 | 8.12e-01 |
| Brier ** | 0.1808 | 0.2393 | +0.0585 | -6.041 | 3.79e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7921 | +0.0360 | -0.993 | 3.77e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4310 | +0.0147 | -0.510 | 6.37e-01 | 8.12e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8153 | +0.0092 | -0.733 | 5.04e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3774 | -0.0319 | 0.931 | 4.05e-01 | 4.38e-01 |
| Brier * | 0.1808 | 0.2364 | +0.0557 | -3.893 | 1.77e-02 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7604 | +0.0043 | -0.152 | 8.86e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4271 | +0.0108 | -0.480 | 6.56e-01 | 4.38e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8141 | +0.0079 | -0.415 | 6.99e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3903 | -0.0190 | 0.458 | 6.71e-01 | 1.00e+00 |
| Brier * | 0.1808 | 0.2341 | +0.0533 | -3.977 | 1.64e-02 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.8205 | +0.0645 | -1.299 | 2.64e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4633 | +0.0469 | -1.001 | 3.74e-01 | 6.25e-01 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.8051 | 0.7127 | 0.8824 |
| M2 | CrossAttn | raw | AUPRC | 0.3011 | 0.1832 | 0.4693 |
| M2 | CrossAttn | raw | Brier | 0.2603 | 0.2326 | 0.2858 |
| M2 | CrossAttn | raw | Accuracy | 0.6681 | 0.6070 | 0.7336 |
| M2 | CrossAttn | raw | F1 | 0.3667 | 0.2500 | 0.4786 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7945 | 0.7120 | 0.8695 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2877 | 0.1686 | 0.4577 |
| M2 | CrossAttn | std_scaled | Brier | 0.2828 | 0.2553 | 0.3091 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.6681 | 0.6070 | 0.7293 |
| M2 | CrossAttn | std_scaled | F1 | 0.3333 | 0.2222 | 0.4444 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8236 | 0.7452 | 0.8866 |
| M2 | CrossAttn | norm | AUPRC | 0.3108 | 0.1901 | 0.4811 |
| M2 | CrossAttn | norm | Brier | 0.2975 | 0.2689 | 0.3247 |
| M2 | CrossAttn | norm | Accuracy | 0.6376 | 0.5763 | 0.7031 |
| M2 | CrossAttn | norm | F1 | 0.3566 | 0.2477 | 0.4658 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7909 | 0.7120 | 0.8657 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2773 | 0.1654 | 0.4395 |
| M2 | CrossAttn | global_zscore | Brier | 0.2606 | 0.2330 | 0.2868 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.6332 | 0.5721 | 0.6987 |
| M2 | CrossAttn | global_zscore | F1 | 0.3333 | 0.2222 | 0.4427 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.8120 | 0.7211 | 0.8858 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3480 | 0.2035 | 0.5378 |
| M2_2 | CrossAttn | raw | Brier | 0.2829 | 0.2549 | 0.3096 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6157 | 0.5546 | 0.6812 |
| M2_2 | CrossAttn | raw | F1 | 0.3231 | 0.2143 | 0.4297 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7770 | 0.6794 | 0.8621 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3586 | 0.2022 | 0.5533 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.3073 | 0.2802 | 0.3329 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.5459 | 0.4803 | 0.6114 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.2778 | 0.1806 | 0.3758 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8409 | 0.7650 | 0.9048 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3889 | 0.2343 | 0.5806 |
| M2_2 | CrossAttn | norm | Brier | 0.2813 | 0.2529 | 0.3082 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6856 | 0.6245 | 0.7467 |
| M2_2 | CrossAttn | norm | F1 | 0.3793 | 0.2653 | 0.4921 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7500 | 0.6501 | 0.8318 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.2811 | 0.1533 | 0.4584 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2813 | 0.2559 | 0.3049 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6114 | 0.5502 | 0.6769 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.2764 | 0.1739 | 0.3810 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.8010 | 0.7203 | 0.8763 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2639 | 0.1700 | 0.4304 |
| M3 | CrossAttn3 | raw | Brier | 0.2561 | 0.2291 | 0.2817 |
| M3 | CrossAttn3 | raw | Accuracy | 0.6332 | 0.5677 | 0.6987 |
| M3 | CrossAttn3 | raw | F1 | 0.3333 | 0.2222 | 0.4370 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.8022 | 0.7080 | 0.8802 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2988 | 0.1890 | 0.4907 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2567 | 0.2287 | 0.2833 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.7467 | 0.6900 | 0.8035 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3958 | 0.2680 | 0.5185 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8272 | 0.7579 | 0.8888 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3088 | 0.1922 | 0.5078 |
| M3 | CrossAttn3 | norm | Brier | 0.2740 | 0.2456 | 0.3012 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6769 | 0.6157 | 0.7380 |
| M3 | CrossAttn3 | norm | F1 | 0.3833 | 0.2645 | 0.4964 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7970 | 0.7075 | 0.8758 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2747 | 0.1776 | 0.4627 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2990 | 0.2702 | 0.3256 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7118 | 0.6550 | 0.7729 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3400 | 0.2169 | 0.4600 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.8051 | +0.0020 | -0.084 | 9.332e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7945 | -0.0085 | 0.341 | 7.331e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8236 | +0.0205 | -1.120 | 2.628e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7909 | -0.0122 | 0.489 | 6.248e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.8010 | -0.0020 | 0.078 | 9.382e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.8022 | -0.0008 | 0.028 | 9.779e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8272 | +0.0242 | -1.165 | 2.439e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7970 | -0.0061 | 0.189 | 8.498e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.8051 | 0.8120 | +0.0069 | -0.458 | 6.471e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7945 | 0.7770 | -0.0175 | 0.690 | 4.902e-01 | ns |
| M2-norm vs M2_2-norm | 0.8236 | 0.8409 | +0.0173 | -1.142 | 2.535e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.7909 | 0.7500 | -0.0409 | 1.558 | 1.192e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.8051 | 0.8010 | -0.0041 | 0.260 | 7.949e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7945 | 0.8022 | +0.0077 | -0.386 | 6.998e-01 | ns |
| M2-norm vs M3-norm | 0.8236 | 0.8272 | +0.0037 | -0.295 | 7.676e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.7909 | 0.7970 | +0.0061 | -0.378 | 7.053e-01 | ns |

