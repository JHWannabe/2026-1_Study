# Scaling Comparison — Test Set Performance (AEC 128pt, BCEWithLogitsLoss)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8119 | 0.4103 | 0.1878 | 0.7124 | 0.3853 |
| M2 | CrossAttn | excl_extreme/scale_clinic | 0.8328 | 0.4096 | 0.1709 | 0.7225 | 0.3696 |
| M2_2 | CrossAttn | len128/scale_clinic | 0.8269 | 0.3141 | 0.2184 | 0.6352 | 0.3411 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | 0.8498 | 0.4389 | 0.1474 | 0.8182 | 0.4571 |

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
| norm/scale_clinic | 0.8254 | 0.4451 | 0.1924 | 0.6738 | 0.3333 |
| **excl_extreme/scale_clinic** | 0.8328 | 0.4096 | 0.1709 | 0.7225 | 0.3696 |
| len128/scale_clinic | 0.8256 | 0.4174 | 0.2528 | 0.6052 | 0.3030 |
| crop80/scale_clinic | 0.8067 | 0.3789 | 0.2020 | 0.7253 | 0.3725 |
| crop60/scale_clinic | 0.7808 | 0.3826 | 0.1892 | 0.7511 | 0.3958 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm/scale_clinic | 0.8246 | 0.3619 | 0.1848 | 0.7554 | 0.3871 |
| excl_extreme/scale_clinic | 0.8230 | 0.3938 | 0.2283 | 0.6507 | 0.3423 |
| **len128/scale_clinic** | 0.8269 | 0.3141 | 0.2184 | 0.6352 | 0.3411 |
| crop80/scale_clinic | 0.8031 | 0.3909 | 0.2170 | 0.6824 | 0.3393 |
| crop60/scale_clinic | 0.8169 | 0.3845 | 0.2108 | 0.6609 | 0.3361 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm/scale_clinic | 0.8200 | 0.4517 | 0.2070 | 0.6781 | 0.3363 |
| **excl_extreme/scale_clinic** | 0.8498 | 0.4389 | 0.1474 | 0.8182 | 0.4571 |
| len128/scale_clinic | 0.7904 | 0.3316 | 0.2568 | 0.6266 | 0.3256 |
| crop80/scale_clinic | 0.8013 | 0.4071 | 0.1800 | 0.6910 | 0.3455 |
| crop60/scale_clinic | 0.7871 | 0.3316 | 0.1777 | 0.7554 | 0.3736 |

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

### norm/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8115 | +0.0064 | -0.600 | 5.81e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3594 | -0.0263 | 0.856 | 4.40e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1812 | +0.0012 | -0.081 | 9.39e-01 | 6.25e-01 |
| Accuracy  | 0.7491 | 0.7447 | -0.0043 | 0.092 | 9.31e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4183 | -0.0005 | 0.017 | 9.87e-01 | 1.00e+00 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8261 | +0.0211 | -0.943 | 3.99e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3801 | -0.0055 | 0.134 | 9.00e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1891 | +0.0091 | -0.753 | 4.94e-01 | 6.25e-01 |
| Accuracy  | 0.7491 | 0.7605 | +0.0114 | -0.215 | 8.41e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4367 | +0.0180 | -0.466 | 6.65e-01 | 6.25e-01 |

### len128/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8201 | +0.0150 | -0.982 | 3.82e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3662 | -0.0195 | 0.448 | 6.77e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.2047 | +0.0247 | -1.053 | 3.52e-01 | 3.12e-01 |
| Accuracy  | 0.7491 | 0.7545 | +0.0054 | -0.161 | 8.80e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4289 | +0.0102 | -0.546 | 6.14e-01 | 6.25e-01 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8344 | +0.0293 | -1.989 | 1.18e-01 | 1.25e-01 |
| AUPRC  | 0.3857 | 0.4376 | +0.0519 | -1.065 | 3.47e-01 | 4.38e-01 |
| Brier  | 0.1800 | 0.1987 | +0.0187 | -0.882 | 4.27e-01 | 6.25e-01 |
| Accuracy  | 0.7491 | 0.7718 | +0.0228 | -0.490 | 6.50e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4375 | +0.0188 | -0.468 | 6.64e-01 | 1.00e+00 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8257 | +0.0207 | -1.513 | 2.05e-01 | 3.12e-01 |
| AUPRC  | 0.3857 | 0.4036 | +0.0179 | -0.406 | 7.06e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.1806 | +0.0006 | -0.078 | 9.42e-01 | 1.00e+00 |
| Accuracy  | 0.7491 | 0.7546 | +0.0055 | -0.085 | 9.36e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4261 | +0.0074 | -0.185 | 8.62e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8115 | 0.8037 | -0.0078 | 0.760 | 4.89e-01 | 6.25e-01 |
| AUPRC  | 0.3594 | 0.3810 | +0.0216 | -1.295 | 2.65e-01 | 4.38e-01 |
| Brier  | 0.1812 | 0.1831 | +0.0019 | -0.126 | 9.06e-01 | 8.12e-01 |
| Accuracy  | 0.7447 | 0.7524 | +0.0077 | -0.138 | 8.97e-01 | 8.12e-01 |
| F1  | 0.4183 | 0.4065 | -0.0117 | 0.321 | 7.64e-01 | 1.00e+00 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8261 | 0.8122 | -0.0140 | 2.373 | 7.66e-02 | 1.25e-01 |
| AUPRC  | 0.3801 | 0.3464 | -0.0338 | 0.866 | 4.35e-01 | 6.25e-01 |
| Brier  | 0.1891 | 0.1768 | -0.0124 | 1.026 | 3.63e-01 | 6.25e-01 |
| Accuracy  | 0.7605 | 0.7629 | +0.0024 | -0.061 | 9.54e-01 | 1.00e+00 |
| F1  | 0.4367 | 0.4266 | -0.0101 | 0.307 | 7.74e-01 | 8.12e-01 |

#### Case: len128/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8201 | 0.8137 | -0.0063 | 0.726 | 5.08e-01 | 4.38e-01 |
| AUPRC  | 0.3662 | 0.4098 | +0.0436 | -0.911 | 4.14e-01 | 1.00e+00 |
| Brier  | 0.2047 | 0.1797 | -0.0250 | 0.856 | 4.40e-01 | 6.25e-01 |
| Accuracy  | 0.7545 | 0.7524 | -0.0021 | 0.095 | 9.29e-01 | 1.00e+00 |
| F1  | 0.4289 | 0.4202 | -0.0087 | 0.384 | 7.21e-01 | 1.00e+00 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8344 | 0.8092 | -0.0252 | 3.046 | 3.82e-02 | 1.25e-01 |
| AUPRC  | 0.4376 | 0.3778 | -0.0598 | 1.498 | 2.08e-01 | 3.12e-01 |
| Brier  | 0.1987 | 0.1909 | -0.0078 | 0.416 | 6.98e-01 | 1.00e+00 |
| Accuracy  | 0.7718 | 0.7309 | -0.0410 | 1.411 | 2.31e-01 | 4.38e-01 |
| F1  | 0.4375 | 0.4091 | -0.0284 | 1.255 | 2.78e-01 | 4.38e-01 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8257 | 0.8136 | -0.0121 | 2.324 | 8.08e-02 | 6.25e-02 |
| AUPRC  | 0.4036 | 0.4032 | -0.0004 | 0.012 | 9.91e-01 | 1.00e+00 |
| Brier  | 0.1806 | 0.1650 | -0.0156 | 1.431 | 2.26e-01 | 1.88e-01 |
| Accuracy  | 0.7546 | 0.7879 | +0.0334 | -1.811 | 1.44e-01 | 1.25e-01 |
| F1  | 0.4261 | 0.4533 | +0.0271 | -2.054 | 1.09e-01 | 1.25e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8037 | -0.0014 | 0.074 | 9.45e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3810 | -0.0047 | 0.135 | 8.99e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1831 | +0.0031 | -0.456 | 6.72e-01 | 8.12e-01 |
| Accuracy  | 0.7491 | 0.7524 | +0.0034 | -0.063 | 9.53e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4065 | -0.0122 | 0.395 | 7.13e-01 | 8.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8122 | +0.0071 | -0.419 | 6.97e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3464 | -0.0393 | 0.815 | 4.61e-01 | 4.38e-01 |
| Brier  | 0.1800 | 0.1768 | -0.0032 | 0.184 | 8.63e-01 | 1.00e+00 |
| Accuracy  | 0.7491 | 0.7629 | +0.0138 | -0.347 | 7.46e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4266 | +0.0079 | -0.339 | 7.51e-01 | 8.12e-01 |

### len128/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8137 | +0.0086 | -0.386 | 7.19e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.4098 | +0.0241 | -0.412 | 7.02e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.1797 | -0.0003 | 0.029 | 9.78e-01 | 1.00e+00 |
| Accuracy  | 0.7491 | 0.7524 | +0.0033 | -0.077 | 9.42e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4202 | +0.0015 | -0.054 | 9.60e-01 | 8.12e-01 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8092 | +0.0041 | -0.251 | 8.14e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3778 | -0.0079 | 0.245 | 8.18e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1909 | +0.0109 | -0.778 | 4.80e-01 | 8.12e-01 |
| Accuracy  | 0.7491 | 0.7309 | -0.0182 | 0.419 | 6.97e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4091 | -0.0096 | 0.349 | 7.45e-01 | 8.12e-01 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8136 | +0.0085 | -0.500 | 6.43e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.4032 | +0.0175 | -0.366 | 7.33e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1650 | -0.0150 | 1.166 | 3.08e-01 | 3.12e-01 |
| Accuracy  | 0.7491 | 0.7879 | +0.0388 | -0.665 | 5.43e-01 | 6.25e-01 |
| F1  | 0.4187 | 0.4533 | +0.0345 | -0.860 | 4.38e-01 | 4.38e-01 |

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
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8254 | 0.7436 | 0.9018 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4451 | 0.2673 | 0.6316 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.1924 | 0.1632 | 0.2238 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.6738 | 0.6137 | 0.7339 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.3333 | 0.2222 | 0.4496 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8328 | 0.7509 | 0.9033 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.4096 | 0.2359 | 0.6152 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1709 | 0.1421 | 0.2015 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7225 | 0.6555 | 0.7799 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3696 | 0.2353 | 0.4951 |
| M2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.8256 | 0.7433 | 0.9017 |
| M2 | CrossAttn | len128/scale_clinic | AUPRC | 0.4174 | 0.2476 | 0.6181 |
| M2 | CrossAttn | len128/scale_clinic | Brier | 0.2528 | 0.2214 | 0.2849 |
| M2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6052 | 0.5408 | 0.6695 |
| M2 | CrossAttn | len128/scale_clinic | F1 | 0.3030 | 0.2047 | 0.4068 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8067 | 0.7191 | 0.8869 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.3789 | 0.2203 | 0.5697 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.2020 | 0.1705 | 0.2358 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.7253 | 0.6652 | 0.7811 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.3725 | 0.2500 | 0.4909 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.7808 | 0.6706 | 0.8786 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3826 | 0.2271 | 0.5770 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.1892 | 0.1586 | 0.2217 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.7511 | 0.6953 | 0.8069 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.3958 | 0.2697 | 0.5209 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8246 | 0.7419 | 0.8994 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.3619 | 0.2221 | 0.5669 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.1848 | 0.1545 | 0.2168 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7554 | 0.6996 | 0.8112 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.3871 | 0.2608 | 0.5161 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8230 | 0.7263 | 0.9081 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3938 | 0.2209 | 0.5874 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.2283 | 0.1963 | 0.2620 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.6507 | 0.5885 | 0.7129 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3423 | 0.2291 | 0.4538 |
| M2_2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.8269 | 0.7459 | 0.8993 |
| M2_2 | CrossAttn | len128/scale_clinic | AUPRC | 0.3141 | 0.2027 | 0.5086 |
| M2_2 | CrossAttn | len128/scale_clinic | Brier | 0.2184 | 0.1897 | 0.2477 |
| M2_2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6352 | 0.5708 | 0.6953 |
| M2_2 | CrossAttn | len128/scale_clinic | F1 | 0.3411 | 0.2385 | 0.4445 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8031 | 0.7056 | 0.8880 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.3909 | 0.2321 | 0.5757 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.2170 | 0.1834 | 0.2513 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6824 | 0.6223 | 0.7425 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.3393 | 0.2301 | 0.4500 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8169 | 0.7443 | 0.8898 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3845 | 0.2236 | 0.5667 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.2108 | 0.1857 | 0.2376 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6609 | 0.5966 | 0.7210 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.3361 | 0.2301 | 0.4483 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8200 | 0.7329 | 0.9025 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.4517 | 0.2711 | 0.6536 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.2070 | 0.1808 | 0.2334 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.6781 | 0.6180 | 0.7382 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.3363 | 0.2243 | 0.4496 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8498 | 0.7690 | 0.9161 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.4389 | 0.2652 | 0.6493 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.1474 | 0.1209 | 0.1765 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.8182 | 0.7656 | 0.8660 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.4571 | 0.3000 | 0.5926 |
| M3 | CrossAttn3 | len128/scale_clinic | AUC-ROC | 0.7904 | 0.6909 | 0.8817 |
| M3 | CrossAttn3 | len128/scale_clinic | AUPRC | 0.3316 | 0.2027 | 0.5359 |
| M3 | CrossAttn3 | len128/scale_clinic | Brier | 0.2568 | 0.2243 | 0.2918 |
| M3 | CrossAttn3 | len128/scale_clinic | Accuracy | 0.6266 | 0.5622 | 0.6867 |
| M3 | CrossAttn3 | len128/scale_clinic | F1 | 0.3256 | 0.2202 | 0.4297 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8013 | 0.7062 | 0.8887 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.4071 | 0.2394 | 0.6094 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.1800 | 0.1531 | 0.2109 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.6910 | 0.6309 | 0.7511 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.3455 | 0.2292 | 0.4630 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.7871 | 0.6974 | 0.8677 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.3316 | 0.1840 | 0.5108 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.1777 | 0.1476 | 0.2105 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.7554 | 0.6953 | 0.8069 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.3736 | 0.2409 | 0.4910 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-norm | 0.8119 | 0.8254 | +0.0135 | -0.556 | 5.784e-01 | ns |
| M1-LR vs M2-len128 | 0.8119 | 0.8256 | +0.0137 | -0.624 | 5.326e-01 | ns |
| M1-LR vs M2-crop80 | 0.8119 | 0.8067 | -0.0052 | 0.245 | 8.064e-01 | ns |
| M1-LR vs M2-crop60 | 0.8119 | 0.7808 | -0.0312 | 1.057 | 2.905e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-norm | 0.8119 | 0.8200 | +0.0081 | -0.288 | 7.731e-01 | ns |
| M1-LR vs M3-len128 | 0.8119 | 0.7904 | -0.0215 | 0.744 | 4.569e-01 | ns |
| M1-LR vs M3-crop80 | 0.8119 | 0.8013 | -0.0106 | 0.365 | 7.149e-01 | ns |
| M1-LR vs M3-crop60 | 0.8119 | 0.7871 | -0.0248 | 0.782 | 4.341e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M2_2-norm | 0.8254 | 0.8246 | -0.0008 | 0.037 | 9.704e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8328 | 0.5246 | -0.3082 | 4.650 | 3.326e-06 | *** |
| M2-len128 vs M2_2-len128 | 0.8256 | 0.8269 | +0.0013 | -0.069 | 9.453e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.8067 | 0.8031 | -0.0037 | 0.158 | 8.746e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.7808 | 0.8169 | +0.0362 | -1.101 | 2.711e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M3-norm | 0.8254 | 0.8200 | -0.0054 | 0.523 | 6.010e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.8328 | 0.8498 | +0.0170 | -0.729 | 4.659e-01 | ns |
| M2-len128 vs M3-len128 | 0.8256 | 0.7904 | -0.0352 | 1.409 | 1.589e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8067 | 0.8013 | -0.0054 | 0.213 | 8.316e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.7808 | 0.7871 | +0.0063 | -0.230 | 8.180e-01 | ns |

