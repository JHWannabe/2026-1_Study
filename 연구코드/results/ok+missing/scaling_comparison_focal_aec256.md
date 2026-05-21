# Scaling Comparison — Test Set Performance (AEC 256pt, FocalLoss)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8119 | 0.4103 | 0.1878 | 0.7124 | 0.3853 |
| M2 | CrossAttn | norm/scale_clinic | 0.8360 | 0.4775 | 0.1570 | 0.8927 | 0.4681 |
| M2_2 | CrossAttn | crop60/scale_clinic | 0.8171 | 0.3927 | 0.1992 | 0.7554 | 0.3871 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | 0.8252 | 0.4469 | 0.2510 | 0.5598 | 0.3134 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_clinic** | 0.8119 | 0.4103 | 0.1878 | 0.7124 | 0.3853 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_clinic | 0.7960 | 0.3245 | 0.2047 | 0.5966 | 0.3088 |
| crop80/scale_clinic | 0.8237 | 0.4218 | 0.1932 | 0.7339 | 0.3800 |
| crop60/scale_clinic | 0.7894 | 0.3315 | 0.2561 | 0.4592 | 0.2674 |
| **norm/scale_clinic** | 0.8360 | 0.4775 | 0.1570 | 0.8927 | 0.4681 |
| excl_extreme/scale_clinic | 0.8264 | 0.4610 | 0.1747 | 0.7321 | 0.3488 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_clinic | 0.8033 | 0.4035 | 0.2241 | 0.6137 | 0.3077 |
| crop80/scale_clinic | 0.8037 | 0.3606 | 0.2004 | 0.7811 | 0.3377 |
| **crop60/scale_clinic** | 0.8171 | 0.3927 | 0.1992 | 0.7554 | 0.3871 |
| norm/scale_clinic | 0.8117 | 0.3860 | 0.2487 | 0.4979 | 0.2909 |
| excl_extreme/scale_clinic | 0.7863 | 0.3017 | 0.2021 | 0.7033 | 0.3261 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_clinic | 0.8140 | 0.4305 | 0.1878 | 0.7983 | 0.4337 |
| crop80/scale_clinic | 0.8119 | 0.4076 | 0.2486 | 0.4850 | 0.2771 |
| crop60/scale_clinic | 0.8248 | 0.4172 | 0.2678 | 0.5322 | 0.2876 |
| norm/scale_clinic | 0.8063 | 0.4561 | 0.1891 | 0.8927 | 0.3902 |
| **excl_extreme/scale_clinic** | 0.8252 | 0.4469 | 0.2510 | 0.5598 | 0.3134 |

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

### len256/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8188 | +0.0137 | -1.032 | 3.60e-01 | 3.12e-01 |
| AUPRC  | 0.3857 | 0.3708 | -0.0149 | 0.656 | 5.48e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1731 | -0.0069 | 1.449 | 2.21e-01 | 6.25e-01 |
| Accuracy  | 0.7491 | 0.7512 | +0.0021 | -0.044 | 9.67e-01 | 8.75e-01 |
| F1  | 0.4187 | 0.4243 | +0.0056 | -0.223 | 8.35e-01 | 8.75e-01 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8108 | +0.0057 | -0.467 | 6.65e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3815 | -0.0042 | 0.118 | 9.12e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1915 | +0.0115 | -0.822 | 4.57e-01 | 1.00e+00 |
| Accuracy  | 0.7491 | 0.7545 | +0.0055 | -0.126 | 9.05e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4163 | -0.0025 | 0.075 | 9.44e-01 | 1.00e+00 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8206 | +0.0156 | -1.073 | 3.44e-01 | 4.38e-01 |
| AUPRC  | 0.3857 | 0.3850 | -0.0007 | 0.021 | 9.85e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.2008 | +0.0208 | -1.142 | 3.17e-01 | 3.12e-01 |
| Accuracy  | 0.7491 | 0.7448 | -0.0043 | 0.152 | 8.87e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4236 | +0.0048 | -0.346 | 7.47e-01 | 6.25e-01 |

### norm/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8099 | +0.0049 | -0.336 | 7.54e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3604 | -0.0253 | 1.038 | 3.58e-01 | 6.25e-01 |
| Brier † | 0.1800 | 0.2256 | +0.0456 | -2.350 | 7.85e-02 | 1.25e-01 |
| Accuracy  | 0.7491 | 0.7309 | -0.0182 | 0.471 | 6.62e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4021 | -0.0167 | 0.724 | 5.09e-01 | 8.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8175 | +0.0124 | -0.358 | 7.38e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.4056 | +0.0200 | -0.348 | 7.45e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.1933 | +0.0133 | -1.378 | 2.40e-01 | 4.38e-01 |
| Accuracy  | 0.7491 | 0.7246 | -0.0245 | 0.342 | 7.50e-01 | 6.25e-01 |
| F1  | 0.4187 | 0.4112 | -0.0075 | 0.137 | 8.98e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len256/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8188 | 0.8023 | -0.0165 | 0.983 | 3.81e-01 | 4.38e-01 |
| AUPRC  | 0.3708 | 0.3776 | +0.0068 | -0.194 | 8.56e-01 | 8.12e-01 |
| Brier † | 0.1731 | 0.1989 | +0.0258 | -2.146 | 9.84e-02 | 1.25e-01 |
| Accuracy  | 0.7512 | 0.7783 | +0.0271 | -0.555 | 6.08e-01 | 1.00e+00 |
| F1  | 0.4243 | 0.4302 | +0.0059 | -0.158 | 8.82e-01 | 1.00e+00 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8108 | 0.8115 | +0.0007 | -0.086 | 9.36e-01 | 1.00e+00 |
| AUPRC  | 0.3815 | 0.3800 | -0.0015 | 0.060 | 9.55e-01 | 8.12e-01 |
| Brier  | 0.1915 | 0.1773 | -0.0142 | 1.571 | 1.91e-01 | 3.12e-01 |
| Accuracy  | 0.7545 | 0.7513 | -0.0033 | 0.126 | 9.06e-01 | 8.75e-01 |
| F1  | 0.4163 | 0.4244 | +0.0081 | -0.286 | 7.89e-01 | 1.00e+00 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8206 | 0.8088 | -0.0118 | 1.743 | 1.56e-01 | 1.88e-01 |
| AUPRC  | 0.3850 | 0.4020 | +0.0171 | -0.718 | 5.13e-01 | 4.38e-01 |
| Brier  | 0.2008 | 0.2033 | +0.0025 | -0.149 | 8.89e-01 | 1.00e+00 |
| Accuracy  | 0.7448 | 0.7438 | -0.0010 | 0.072 | 9.46e-01 | 1.00e+00 |
| F1  | 0.4236 | 0.4064 | -0.0172 | 1.462 | 2.17e-01 | 3.12e-01 |

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8099 | 0.8170 | +0.0071 | -0.603 | 5.79e-01 | 8.12e-01 |
| AUPRC * | 0.3604 | 0.4082 | +0.0478 | -3.198 | 3.30e-02 | 6.25e-02 |
| Brier  | 0.2256 | 0.2152 | -0.0104 | 0.428 | 6.91e-01 | 8.12e-01 |
| Accuracy * | 0.7309 | 0.7858 | +0.0549 | -3.771 | 1.96e-02 | 6.25e-02 |
| F1 * | 0.4021 | 0.4494 | +0.0473 | -3.807 | 1.90e-02 | 6.25e-02 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8175 | 0.8136 | -0.0039 | 0.506 | 6.39e-01 | 6.25e-01 |
| AUPRC  | 0.4056 | 0.3616 | -0.0441 | 1.195 | 2.98e-01 | 3.12e-01 |
| Brier  | 0.1933 | 0.2080 | +0.0146 | -0.874 | 4.32e-01 | 6.25e-01 |
| Accuracy  | 0.7246 | 0.7281 | +0.0036 | -0.063 | 9.53e-01 | 1.00e+00 |
| F1  | 0.4112 | 0.3986 | -0.0126 | 0.314 | 7.69e-01 | 1.00e+00 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len256/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8023 | -0.0028 | 0.124 | 9.07e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3776 | -0.0081 | 0.169 | 8.74e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1989 | +0.0189 | -1.807 | 1.45e-01 | 1.88e-01 |
| Accuracy  | 0.7491 | 0.7783 | +0.0292 | -0.575 | 5.96e-01 | 6.25e-01 |
| F1  | 0.4187 | 0.4302 | +0.0115 | -0.321 | 7.64e-01 | 1.00e+00 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8115 | +0.0065 | -0.353 | 7.42e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3800 | -0.0057 | 0.121 | 9.09e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1773 | -0.0027 | 0.219 | 8.37e-01 | 1.00e+00 |
| Accuracy  | 0.7491 | 0.7513 | +0.0022 | -0.101 | 9.25e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4244 | +0.0057 | -0.425 | 6.93e-01 | 6.25e-01 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8088 | +0.0037 | -0.196 | 8.54e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.4020 | +0.0164 | -0.318 | 7.66e-01 | 8.12e-01 |
| Brier * | 0.1800 | 0.2033 | +0.0234 | -2.853 | 4.63e-02 | 6.25e-02 |
| Accuracy  | 0.7491 | 0.7438 | -0.0053 | 0.138 | 8.97e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4064 | -0.0123 | 0.495 | 6.47e-01 | 6.25e-01 |

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8170 | +0.0119 | -0.601 | 5.80e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.4082 | +0.0225 | -0.732 | 5.05e-01 | 4.38e-01 |
| Brier  | 0.1800 | 0.2152 | +0.0352 | -1.588 | 1.88e-01 | 3.12e-01 |
| Accuracy  | 0.7491 | 0.7858 | +0.0367 | -1.305 | 2.62e-01 | 3.12e-01 |
| F1  | 0.4187 | 0.4494 | +0.0307 | -1.526 | 2.02e-01 | 3.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8136 | +0.0085 | -0.260 | 8.08e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3616 | -0.0241 | 0.399 | 7.10e-01 | 6.25e-01 |
| Brier † | 0.1800 | 0.2080 | +0.0280 | -2.552 | 6.32e-02 | 1.25e-01 |
| Accuracy  | 0.7491 | 0.7281 | -0.0209 | 0.444 | 6.80e-01 | 6.25e-01 |
| F1  | 0.4187 | 0.3986 | -0.0201 | 0.514 | 6.35e-01 | 8.12e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_clinic | AUC-ROC | 0.8119 | 0.7251 | 0.8882 |
| M1 | LR | scale_clinic | AUPRC | 0.4103 | 0.2449 | 0.5885 |
| M1 | LR | scale_clinic | Brier | 0.1878 | 0.1629 | 0.2143 |
| M1 | LR | scale_clinic | Accuracy | 0.7124 | 0.6524 | 0.7682 |
| M1 | LR | scale_clinic | F1 | 0.3853 | 0.2727 | 0.5079 |
| M2 | CrossAttn | len256/scale_clinic | AUC-ROC | 0.7960 | 0.7102 | 0.8735 |
| M2 | CrossAttn | len256/scale_clinic | AUPRC | 0.3245 | 0.1878 | 0.5069 |
| M2 | CrossAttn | len256/scale_clinic | Brier | 0.2047 | 0.1904 | 0.2198 |
| M2 | CrossAttn | len256/scale_clinic | Accuracy | 0.5966 | 0.5322 | 0.6609 |
| M2 | CrossAttn | len256/scale_clinic | F1 | 0.3088 | 0.2097 | 0.4114 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8237 | 0.7343 | 0.9027 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.4218 | 0.2557 | 0.6171 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.1932 | 0.1787 | 0.2079 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.7339 | 0.6738 | 0.7897 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.3800 | 0.2580 | 0.5001 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.7894 | 0.6924 | 0.8733 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3315 | 0.1973 | 0.5414 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.2561 | 0.2405 | 0.2712 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.4592 | 0.3906 | 0.5193 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.2674 | 0.1830 | 0.3563 |
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8360 | 0.7546 | 0.9103 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4775 | 0.2942 | 0.6697 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.1570 | 0.1456 | 0.1697 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.8927 | 0.8498 | 0.9313 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.4681 | 0.2727 | 0.6383 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8264 | 0.7358 | 0.9057 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.4610 | 0.2690 | 0.6632 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1747 | 0.1625 | 0.1877 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7321 | 0.6745 | 0.7895 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3488 | 0.2133 | 0.4776 |
| M2_2 | CrossAttn | len256/scale_clinic | AUC-ROC | 0.8033 | 0.7101 | 0.8857 |
| M2_2 | CrossAttn | len256/scale_clinic | AUPRC | 0.4035 | 0.2390 | 0.5886 |
| M2_2 | CrossAttn | len256/scale_clinic | Brier | 0.2241 | 0.2103 | 0.2373 |
| M2_2 | CrossAttn | len256/scale_clinic | Accuracy | 0.6137 | 0.5494 | 0.6781 |
| M2_2 | CrossAttn | len256/scale_clinic | F1 | 0.3077 | 0.2037 | 0.4133 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8037 | 0.7169 | 0.8817 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.3606 | 0.2059 | 0.5605 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.2004 | 0.1866 | 0.2145 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.7811 | 0.7296 | 0.8284 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.3377 | 0.1904 | 0.4706 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8171 | 0.7319 | 0.8932 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3927 | 0.2279 | 0.5855 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.1992 | 0.1860 | 0.2120 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.7554 | 0.6996 | 0.8112 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.3871 | 0.2597 | 0.5138 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8117 | 0.7334 | 0.8824 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.3860 | 0.2253 | 0.5653 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.2487 | 0.2342 | 0.2633 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.4979 | 0.4335 | 0.5622 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.2909 | 0.2036 | 0.3820 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.7863 | 0.6837 | 0.8790 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3017 | 0.1840 | 0.5213 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.2021 | 0.1839 | 0.2202 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7033 | 0.6411 | 0.7656 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3261 | 0.2000 | 0.4528 |
| M3 | CrossAttn3 | len256/scale_clinic | AUC-ROC | 0.8140 | 0.7227 | 0.8992 |
| M3 | CrossAttn3 | len256/scale_clinic | AUPRC | 0.4305 | 0.2518 | 0.6161 |
| M3 | CrossAttn3 | len256/scale_clinic | Brier | 0.1878 | 0.1726 | 0.2030 |
| M3 | CrossAttn3 | len256/scale_clinic | Accuracy | 0.7983 | 0.7468 | 0.8498 |
| M3 | CrossAttn3 | len256/scale_clinic | F1 | 0.4337 | 0.2898 | 0.5657 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8119 | 0.7255 | 0.8920 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.4076 | 0.2423 | 0.6055 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.2486 | 0.2315 | 0.2650 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.4850 | 0.4206 | 0.5451 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.2771 | 0.1899 | 0.3659 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.8248 | 0.7342 | 0.9050 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.4172 | 0.2531 | 0.6240 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.2678 | 0.2536 | 0.2820 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.5322 | 0.4635 | 0.5966 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.2876 | 0.1958 | 0.3815 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8063 | 0.7140 | 0.8885 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.4561 | 0.2719 | 0.6482 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.1891 | 0.1796 | 0.1987 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.8927 | 0.8498 | 0.9313 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.3902 | 0.1935 | 0.5661 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8252 | 0.7374 | 0.9018 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.4469 | 0.2586 | 0.6461 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.2510 | 0.2351 | 0.2668 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.5598 | 0.4928 | 0.6268 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.3134 | 0.2114 | 0.4189 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-len256 | 0.8119 | 0.7960 | -0.0160 | 0.670 | 5.028e-01 | ns |
| M1-LR vs M2-crop80 | 0.8119 | 0.8237 | +0.0117 | -0.567 | 5.709e-01 | ns |
| M1-LR vs M2-crop60 | 0.8119 | 0.7894 | -0.0225 | 0.554 | 5.793e-01 | ns |
| M1-LR vs M2-norm | 0.8119 | 0.8360 | +0.0240 | -0.849 | 3.961e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-len256 | 0.8119 | 0.8140 | +0.0021 | -0.100 | 9.205e-01 | ns |
| M1-LR vs M3-crop80 | 0.8119 | 0.8119 | -0.0000 | 0.000 | 1.000e+00 | ns |
| M1-LR vs M3-crop60 | 0.8119 | 0.8248 | +0.0129 | -0.431 | 6.665e-01 | ns |
| M1-LR vs M3-norm | 0.8119 | 0.8063 | -0.0056 | 0.173 | 8.630e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len256 vs M2_2-len256 | 0.7960 | 0.8033 | +0.0073 | -0.220 | 8.258e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.8237 | 0.8037 | -0.0200 | 0.829 | 4.071e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.7894 | 0.8171 | +0.0277 | -0.726 | 4.680e-01 | ns |
| M2-norm vs M2_2-norm | 0.8360 | 0.8117 | -0.0242 | 0.978 | 3.283e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8264 | 0.5124 | -0.3140 | 5.256 | 1.470e-07 | *** |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len256 vs M3-len256 | 0.7960 | 0.8140 | +0.0181 | -0.827 | 4.085e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8237 | 0.8119 | -0.0117 | 0.707 | 4.795e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.7894 | 0.8248 | +0.0354 | -1.380 | 1.677e-01 | ns |
| M2-norm vs M3-norm | 0.8360 | 0.8063 | -0.0296 | 2.103 | 3.546e-02 | * |
| M2-excl_extreme vs M3-excl_extreme | 0.8264 | 0.8252 | -0.0012 | 0.060 | 9.521e-01 | ns |

