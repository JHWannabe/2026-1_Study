# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8435 | 0.3433 | 0.1397 | 0.8472 | 0.3396 |
| M2_2 | CrossAttn | norm | 0.8264 | 0.3204 | 0.1887 | 0.7380 | 0.3750 |
| M3 | CrossAttn3 | norm | 0.8455 | 0.3475 | 0.1902 | 0.6856 | 0.3793 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_all** | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.7913 | 0.3367 | 0.1847 | 0.7904 | 0.3684 |
| **norm** | 0.8435 | 0.3433 | 0.1397 | 0.8472 | 0.3396 |
| crop80 | 0.7689 | 0.2327 | 0.2108 | 0.6245 | 0.3281 |
| crop60 | 0.7933 | 0.3147 | 0.2389 | 0.6725 | 0.3478 |
| excl_extreme | 0.8095 | 0.3140 | 0.2065 | 0.7073 | 0.3878 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.7457 | 0.3618 | 0.1836 | 0.7074 | 0.3093 |
| **norm** | 0.8264 | 0.3204 | 0.1887 | 0.7380 | 0.3750 |
| crop80 | 0.8039 | 0.3706 | 0.1968 | 0.7074 | 0.3495 |
| crop60 | 0.8213 | 0.3396 | 0.2167 | 0.7162 | 0.3689 |
| excl_extreme | 0.8060 | 0.3160 | 0.1929 | 0.6244 | 0.3419 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.8033 | 0.2909 | 0.1701 | 0.6681 | 0.3559 |
| **norm** | 0.8455 | 0.3475 | 0.1902 | 0.6856 | 0.3793 |
| crop80 | 0.7835 | 0.2703 | 0.2055 | 0.5895 | 0.3188 |
| crop60 | 0.7886 | 0.2878 | 0.1830 | 0.7642 | 0.4000 |
| excl_extreme | 0.7688 | 0.2594 | 0.1770 | 0.7756 | 0.3235 |

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

### len128  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8215 | +0.0153 | -1.139 | 3.18e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4248 | +0.0156 | -0.532 | 6.23e-01 | 6.25e-01 |
| Brier  | 0.1808 | 0.1947 | +0.0140 | -0.865 | 4.36e-01 | 8.12e-01 |
| Accuracy  | 0.7561 | 0.8019 | +0.0458 | -1.283 | 2.69e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.4621 | +0.0458 | -1.305 | 2.62e-01 | 3.12e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8122 | +0.0060 | -0.361 | 7.36e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4002 | -0.0090 | 0.432 | 6.88e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.2223 | +0.0415 | -1.277 | 2.71e-01 | 4.38e-01 |
| Accuracy  | 0.7561 | 0.7539 | -0.0022 | 0.085 | 9.36e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4235 | +0.0072 | -0.239 | 8.23e-01 | 8.12e-01 |

### crop80  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8084 | +0.0023 | -0.190 | 8.58e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3985 | -0.0107 | 0.254 | 8.12e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1588 | -0.0220 | 1.958 | 1.22e-01 | 3.12e-01 |
| Accuracy  | 0.7561 | 0.7604 | +0.0043 | -0.085 | 9.37e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4199 | +0.0036 | -0.081 | 9.39e-01 | 1.00e+00 |

### crop60  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8164 | +0.0102 | -0.874 | 4.32e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4184 | +0.0092 | -0.257 | 8.10e-01 | 6.25e-01 |
| Brier  | 0.1808 | 0.1896 | +0.0089 | -0.338 | 7.53e-01 | 6.25e-01 |
| Accuracy  | 0.7561 | 0.7462 | -0.0099 | 0.299 | 7.80e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4079 | -0.0084 | 0.300 | 7.79e-01 | 8.12e-01 |

### excl_extreme  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8305 | +0.0244 | -0.585 | 5.90e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4268 | +0.0176 | -0.213 | 8.41e-01 | 8.12e-01 |
| Brier  | 0.1808 | 0.2047 | +0.0240 | -0.571 | 5.99e-01 | 6.25e-01 |
| Accuracy  | 0.7561 | 0.7457 | -0.0104 | 0.213 | 8.42e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4011 | -0.0152 | 0.349 | 7.44e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: len128  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8215 | 0.8200 | -0.0014 | 0.527 | 6.26e-01 | 8.12e-01 |
| AUPRC  | 0.4248 | 0.4236 | -0.0012 | 0.046 | 9.66e-01 | 1.00e+00 |
| Brier  | 0.1947 | 0.1618 | -0.0330 | 1.500 | 2.08e-01 | 3.12e-01 |
| Accuracy  | 0.8019 | 0.7877 | -0.0142 | 0.376 | 7.26e-01 | 1.00e+00 |
| F1  | 0.4621 | 0.4426 | -0.0195 | 0.591 | 5.86e-01 | 1.00e+00 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8122 | 0.8215 | +0.0093 | -0.951 | 3.95e-01 | 4.38e-01 |
| AUPRC  | 0.4002 | 0.3998 | -0.0004 | 0.017 | 9.87e-01 | 1.00e+00 |
| Brier  | 0.2223 | 0.1556 | -0.0667 | 1.604 | 1.84e-01 | 1.88e-01 |
| Accuracy  | 0.7539 | 0.7430 | -0.0109 | 0.573 | 5.97e-01 | 1.00e+00 |
| F1  | 0.4235 | 0.4167 | -0.0068 | 0.285 | 7.90e-01 | 6.25e-01 |

#### Case: crop80  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8084 | 0.8146 | +0.0062 | -0.463 | 6.67e-01 | 4.38e-01 |
| AUPRC  | 0.3985 | 0.4014 | +0.0029 | -0.238 | 8.24e-01 | 1.00e+00 |
| Brier  | 0.1588 | 0.1817 | +0.0229 | -0.843 | 4.47e-01 | 6.25e-01 |
| Accuracy  | 0.7604 | 0.7550 | -0.0054 | 0.136 | 8.98e-01 | 1.00e+00 |
| F1  | 0.4199 | 0.4145 | -0.0054 | 0.153 | 8.86e-01 | 1.00e+00 |

#### Case: crop60  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8164 | 0.8165 | +0.0001 | -0.010 | 9.93e-01 | 1.00e+00 |
| AUPRC  | 0.4184 | 0.4069 | -0.0116 | 0.357 | 7.39e-01 | 1.00e+00 |
| Brier  | 0.1896 | 0.1800 | -0.0096 | 0.517 | 6.33e-01 | 1.00e+00 |
| Accuracy  | 0.7462 | 0.7823 | +0.0361 | -1.155 | 3.12e-01 | 3.12e-01 |
| F1  | 0.4079 | 0.4311 | +0.0232 | -1.572 | 1.91e-01 | 3.12e-01 |

#### Case: excl_extreme  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8305 | 0.8114 | -0.0191 | 1.735 | 1.58e-01 | 1.88e-01 |
| AUPRC * | 0.4268 | 0.3725 | -0.0544 | 3.022 | 3.91e-02 | 6.25e-02 |
| Brier  | 0.2047 | 0.1958 | -0.0089 | 0.251 | 8.14e-01 | 8.12e-01 |
| Accuracy † | 0.7457 | 0.8114 | +0.0657 | -2.502 | 6.66e-02 | 6.25e-02 |
| F1 † | 0.4011 | 0.4351 | +0.0339 | -2.256 | 8.71e-02 | 1.25e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8200 | +0.0139 | -1.179 | 3.04e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4236 | +0.0144 | -0.442 | 6.81e-01 | 8.12e-01 |
| Brier  | 0.1808 | 0.1618 | -0.0190 | 1.789 | 1.48e-01 | 1.88e-01 |
| Accuracy  | 0.7561 | 0.7877 | +0.0317 | -0.546 | 6.14e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4426 | +0.0263 | -0.524 | 6.28e-01 | 8.12e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8215 | +0.0154 | -0.958 | 3.92e-01 | 3.12e-01 |
| AUPRC  | 0.4092 | 0.3998 | -0.0094 | 0.308 | 7.74e-01 | 8.12e-01 |
| Brier † | 0.1808 | 0.1556 | -0.0251 | 2.493 | 6.73e-02 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7430 | -0.0131 | 0.619 | 5.69e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4167 | +0.0004 | -0.016 | 9.88e-01 | 1.00e+00 |

### crop80  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8146 | +0.0085 | -0.435 | 6.86e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4014 | -0.0078 | 0.185 | 8.62e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1817 | +0.0009 | -0.030 | 9.77e-01 | 6.25e-01 |
| Accuracy  | 0.7561 | 0.7550 | -0.0011 | 0.021 | 9.84e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4145 | -0.0018 | 0.051 | 9.61e-01 | 1.00e+00 |

### crop60  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8165 | +0.0103 | -0.568 | 6.00e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4069 | -0.0023 | 0.040 | 9.70e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1800 | -0.0008 | 0.070 | 9.47e-01 | 8.12e-01 |
| Accuracy  | 0.7561 | 0.7823 | +0.0262 | -0.508 | 6.38e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4311 | +0.0148 | -0.376 | 7.26e-01 | 8.12e-01 |

### excl_extreme  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8114 | +0.0053 | -0.121 | 9.09e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3725 | -0.0367 | 0.448 | 6.78e-01 | 8.12e-01 |
| Brier  | 0.1808 | 0.1958 | +0.0151 | -0.507 | 6.39e-01 | 6.25e-01 |
| Accuracy  | 0.7561 | 0.8114 | +0.0553 | -1.864 | 1.36e-01 | 1.25e-01 |
| F1  | 0.4163 | 0.4351 | +0.0187 | -0.516 | 6.33e-01 | 1.00e+00 |

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
| M2 | CrossAttn | len128 | AUC-ROC | 0.7913 | 0.7037 | 0.8686 |
| M2 | CrossAttn | len128 | AUPRC | 0.3367 | 0.1963 | 0.5250 |
| M2 | CrossAttn | len128 | Brier | 0.1847 | 0.1540 | 0.2138 |
| M2 | CrossAttn | len128 | Accuracy | 0.7904 | 0.7380 | 0.8428 |
| M2 | CrossAttn | len128 | F1 | 0.3684 | 0.2353 | 0.5063 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8435 | 0.7770 | 0.9007 |
| M2 | CrossAttn | norm | AUPRC | 0.3433 | 0.2073 | 0.5282 |
| M2 | CrossAttn | norm | Brier | 0.1397 | 0.1161 | 0.1622 |
| M2 | CrossAttn | norm | Accuracy | 0.8472 | 0.8034 | 0.8908 |
| M2 | CrossAttn | norm | F1 | 0.3396 | 0.1702 | 0.4912 |
| M2 | CrossAttn | crop80 | AUC-ROC | 0.7689 | 0.6786 | 0.8516 |
| M2 | CrossAttn | crop80 | AUPRC | 0.2327 | 0.1463 | 0.4018 |
| M2 | CrossAttn | crop80 | Brier | 0.2108 | 0.1801 | 0.2409 |
| M2 | CrossAttn | crop80 | Accuracy | 0.6245 | 0.5633 | 0.6900 |
| M2 | CrossAttn | crop80 | F1 | 0.3281 | 0.2202 | 0.4361 |
| M2 | CrossAttn | crop60 | AUC-ROC | 0.7933 | 0.7079 | 0.8692 |
| M2 | CrossAttn | crop60 | AUPRC | 0.3147 | 0.1894 | 0.5020 |
| M2 | CrossAttn | crop60 | Brier | 0.2389 | 0.2054 | 0.2708 |
| M2 | CrossAttn | crop60 | Accuracy | 0.6725 | 0.6114 | 0.7336 |
| M2 | CrossAttn | crop60 | F1 | 0.3478 | 0.2385 | 0.4587 |
| M2 | CrossAttn | excl_extreme | AUC-ROC | 0.8095 | 0.7202 | 0.8861 |
| M2 | CrossAttn | excl_extreme | AUPRC | 0.3140 | 0.1821 | 0.4990 |
| M2 | CrossAttn | excl_extreme | Brier | 0.2065 | 0.1730 | 0.2433 |
| M2 | CrossAttn | excl_extreme | Accuracy | 0.7073 | 0.6439 | 0.7659 |
| M2 | CrossAttn | excl_extreme | F1 | 0.3878 | 0.2529 | 0.5049 |
| M2_2 | CrossAttn | len128 | AUC-ROC | 0.7457 | 0.6394 | 0.8441 |
| M2_2 | CrossAttn | len128 | AUPRC | 0.3618 | 0.2011 | 0.5583 |
| M2_2 | CrossAttn | len128 | Brier | 0.1836 | 0.1534 | 0.2140 |
| M2_2 | CrossAttn | len128 | Accuracy | 0.7074 | 0.6507 | 0.7686 |
| M2_2 | CrossAttn | len128 | F1 | 0.3093 | 0.1927 | 0.4243 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8264 | 0.7440 | 0.8914 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3204 | 0.2008 | 0.5166 |
| M2_2 | CrossAttn | norm | Brier | 0.1887 | 0.1595 | 0.2180 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7380 | 0.6856 | 0.7948 |
| M2_2 | CrossAttn | norm | F1 | 0.3750 | 0.2465 | 0.4952 |
| M2_2 | CrossAttn | crop80 | AUC-ROC | 0.8039 | 0.7098 | 0.8874 |
| M2_2 | CrossAttn | crop80 | AUPRC | 0.3706 | 0.2200 | 0.5675 |
| M2_2 | CrossAttn | crop80 | Brier | 0.1968 | 0.1657 | 0.2287 |
| M2_2 | CrossAttn | crop80 | Accuracy | 0.7074 | 0.6507 | 0.7686 |
| M2_2 | CrossAttn | crop80 | F1 | 0.3495 | 0.2307 | 0.4615 |
| M2_2 | CrossAttn | crop60 | AUC-ROC | 0.8213 | 0.7412 | 0.8902 |
| M2_2 | CrossAttn | crop60 | AUPRC | 0.3396 | 0.2048 | 0.5371 |
| M2_2 | CrossAttn | crop60 | Brier | 0.2167 | 0.1846 | 0.2493 |
| M2_2 | CrossAttn | crop60 | Accuracy | 0.7162 | 0.6594 | 0.7773 |
| M2_2 | CrossAttn | crop60 | F1 | 0.3689 | 0.2453 | 0.4894 |
| M2_2 | CrossAttn | excl_extreme | AUC-ROC | 0.8060 | 0.7069 | 0.8833 |
| M2_2 | CrossAttn | excl_extreme | AUPRC | 0.3160 | 0.1989 | 0.5112 |
| M2_2 | CrossAttn | excl_extreme | Brier | 0.1929 | 0.1641 | 0.2238 |
| M2_2 | CrossAttn | excl_extreme | Accuracy | 0.6244 | 0.5561 | 0.6878 |
| M2_2 | CrossAttn | excl_extreme | F1 | 0.3419 | 0.2330 | 0.4465 |
| M3 | CrossAttn3 | len128 | AUC-ROC | 0.8033 | 0.7183 | 0.8826 |
| M3 | CrossAttn3 | len128 | AUPRC | 0.2909 | 0.1872 | 0.4823 |
| M3 | CrossAttn3 | len128 | Brier | 0.1701 | 0.1443 | 0.1953 |
| M3 | CrossAttn3 | len128 | Accuracy | 0.6681 | 0.6070 | 0.7293 |
| M3 | CrossAttn3 | len128 | F1 | 0.3559 | 0.2393 | 0.4660 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8455 | 0.7591 | 0.9136 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3475 | 0.2265 | 0.5520 |
| M3 | CrossAttn3 | norm | Brier | 0.1902 | 0.1609 | 0.2181 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6856 | 0.6288 | 0.7467 |
| M3 | CrossAttn3 | norm | F1 | 0.3793 | 0.2626 | 0.4885 |
| M3 | CrossAttn3 | crop80 | AUC-ROC | 0.7835 | 0.6948 | 0.8616 |
| M3 | CrossAttn3 | crop80 | AUPRC | 0.2703 | 0.1632 | 0.4607 |
| M3 | CrossAttn3 | crop80 | Brier | 0.2055 | 0.1756 | 0.2342 |
| M3 | CrossAttn3 | crop80 | Accuracy | 0.5895 | 0.5240 | 0.6550 |
| M3 | CrossAttn3 | crop80 | F1 | 0.3188 | 0.2105 | 0.4224 |
| M3 | CrossAttn3 | crop60 | AUC-ROC | 0.7886 | 0.6883 | 0.8732 |
| M3 | CrossAttn3 | crop60 | AUPRC | 0.2878 | 0.1745 | 0.4547 |
| M3 | CrossAttn3 | crop60 | Brier | 0.1830 | 0.1550 | 0.2097 |
| M3 | CrossAttn3 | crop60 | Accuracy | 0.7642 | 0.7118 | 0.8210 |
| M3 | CrossAttn3 | crop60 | F1 | 0.4000 | 0.2703 | 0.5288 |
| M3 | CrossAttn3 | excl_extreme | AUC-ROC | 0.7688 | 0.6702 | 0.8580 |
| M3 | CrossAttn3 | excl_extreme | AUPRC | 0.2594 | 0.1565 | 0.4520 |
| M3 | CrossAttn3 | excl_extreme | Brier | 0.1770 | 0.1475 | 0.2081 |
| M3 | CrossAttn3 | excl_extreme | Accuracy | 0.7756 | 0.7171 | 0.8294 |
| M3 | CrossAttn3 | excl_extreme | F1 | 0.3235 | 0.1791 | 0.4688 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-len128 | 0.8030 | 0.7913 | -0.0118 | 0.409 | 6.822e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8435 | +0.0404 | -1.951 | 5.105e-02 | † |
| M1-LR vs M2-crop80 | 0.8030 | 0.7689 | -0.0341 | 1.030 | 3.032e-01 | ns |
| M1-LR vs M2-crop60 | 0.8030 | 0.7933 | -0.0098 | 0.279 | 7.799e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-len128 | 0.8030 | 0.8033 | +0.0002 | -0.006 | 9.953e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8455 | +0.0425 | -1.837 | 6.619e-02 | † |
| M1-LR vs M3-crop80 | 0.8030 | 0.7835 | -0.0195 | 0.832 | 4.057e-01 | ns |
| M1-LR vs M3-crop60 | 0.8030 | 0.7886 | -0.0144 | 0.459 | 6.461e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len128 vs M2_2-len128 | 0.7913 | 0.7457 | -0.0455 | 1.340 | 1.804e-01 | ns |
| M2-norm vs M2_2-norm | 0.8435 | 0.8264 | -0.0171 | 1.057 | 2.905e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.7689 | 0.8039 | +0.0350 | -0.886 | 3.759e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.7933 | 0.8213 | +0.0280 | -0.929 | 3.527e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8095 | 0.4995 | -0.3100 | 3.747 | 1.790e-04 | *** |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len128 vs M3-len128 | 0.7913 | 0.8033 | +0.0120 | -0.441 | 6.589e-01 | ns |
| M2-norm vs M3-norm | 0.8435 | 0.8455 | +0.0020 | -0.097 | 9.226e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.7689 | 0.7835 | +0.0146 | -0.734 | 4.629e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.7933 | 0.7886 | -0.0047 | 0.163 | 8.703e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.8095 | 0.7688 | -0.0407 | 1.986 | 4.701e-02 | * |

