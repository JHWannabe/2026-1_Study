# Scaling Comparison — Test Set Performance (AEC 256pt, FocalLoss)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8119 | 0.4103 | 0.1878 | 0.7124 | 0.3853 |
| M2 | CrossAttn | norm/scale_clinic | 0.8267 | 0.4512 | 0.1993 | 0.7082 | 0.3585 |
| M2_2 | CrossAttn | norm/scale_clinic | 0.8213 | 0.4058 | 0.2166 | 0.5579 | 0.2993 |
| M3 | CrossAttn3 | norm/scale_clinic | 0.8342 | 0.4352 | 0.1932 | 0.6266 | 0.3256 |

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
| **norm/scale_clinic** | 0.8267 | 0.4512 | 0.1993 | 0.7082 | 0.3585 |
| excl_extreme/scale_clinic | 0.8223 | 0.3828 | 0.2003 | 0.7608 | 0.3902 |
| len128/scale_clinic | 0.8069 | 0.4020 | 0.2183 | 0.6223 | 0.3333 |
| crop80/scale_clinic | 0.8013 | 0.4345 | 0.2016 | 0.7167 | 0.3400 |
| crop60/scale_clinic | 0.8162 | 0.4138 | 0.1932 | 0.6867 | 0.3540 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm/scale_clinic** | 0.8213 | 0.4058 | 0.2166 | 0.5579 | 0.2993 |
| excl_extreme/scale_clinic | 0.8140 | 0.3672 | 0.1711 | 0.7943 | 0.4416 |
| len128/scale_clinic | 0.7863 | 0.2929 | 0.1851 | 0.6738 | 0.3214 |
| crop80/scale_clinic | 0.7708 | 0.2990 | 0.1901 | 0.6738 | 0.3214 |
| crop60/scale_clinic | 0.7902 | 0.3997 | 0.1775 | 0.7682 | 0.3571 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm/scale_clinic** | 0.8342 | 0.4352 | 0.1932 | 0.6266 | 0.3256 |
| excl_extreme/scale_clinic | 0.8201 | 0.3926 | 0.1890 | 0.7368 | 0.3678 |
| len128/scale_clinic | 0.7992 | 0.4002 | 0.1730 | 0.8112 | 0.4500 |
| crop80/scale_clinic | 0.8294 | 0.3975 | 0.2223 | 0.6094 | 0.3259 |
| crop60/scale_clinic | 0.7627 | 0.3258 | 0.1949 | 0.6481 | 0.3279 |

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
| AUC-ROC  | 0.8051 | 0.8109 | +0.0059 | -0.404 | 7.07e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3851 | -0.0005 | 0.022 | 9.84e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.2086 | +0.0286 | -1.898 | 1.31e-01 | 1.25e-01 |
| Accuracy  | 0.7491 | 0.7631 | +0.0140 | -0.297 | 7.81e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4257 | +0.0070 | -0.319 | 7.66e-01 | 8.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8307 | +0.0257 | -0.644 | 5.55e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3864 | +0.0007 | -0.010 | 9.92e-01 | 1.00e+00 |
| Brier * | 0.1800 | 0.2026 | +0.0226 | -4.082 | 1.51e-02 | 6.25e-02 |
| Accuracy  | 0.7491 | 0.8072 | +0.0581 | -1.551 | 1.96e-01 | 3.12e-01 |
| F1  | 0.4187 | 0.4552 | +0.0364 | -1.076 | 3.42e-01 | 3.12e-01 |

### len128/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8220 | +0.0170 | -1.137 | 3.19e-01 | 3.12e-01 |
| AUPRC  | 0.3857 | 0.3808 | -0.0049 | 0.216 | 8.40e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1816 | +0.0016 | -0.117 | 9.12e-01 | 1.00e+00 |
| Accuracy  | 0.7491 | 0.7643 | +0.0152 | -0.189 | 8.60e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4418 | +0.0230 | -0.436 | 6.85e-01 | 8.12e-01 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8175 | +0.0124 | -0.958 | 3.93e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.4109 | +0.0252 | -0.626 | 5.66e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.2073 | +0.0273 | -1.792 | 1.48e-01 | 1.88e-01 |
| Accuracy  | 0.7491 | 0.7620 | +0.0129 | -0.642 | 5.56e-01 | 6.25e-01 |
| F1  | 0.4187 | 0.4288 | +0.0100 | -0.656 | 5.48e-01 | 4.38e-01 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8184 | +0.0134 | -0.953 | 3.94e-01 | 4.38e-01 |
| AUPRC  | 0.3857 | 0.4181 | +0.0324 | -0.797 | 4.70e-01 | 4.38e-01 |
| Brier  | 0.1800 | 0.1871 | +0.0071 | -0.474 | 6.60e-01 | 6.25e-01 |
| Accuracy  | 0.7491 | 0.7556 | +0.0066 | -0.174 | 8.71e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4209 | +0.0022 | -0.083 | 9.38e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8109 | 0.7977 | -0.0132 | 1.606 | 1.83e-01 | 1.88e-01 |
| AUPRC * | 0.3851 | 0.3648 | -0.0203 | 3.503 | 2.48e-02 | 6.25e-02 |
| Brier  | 0.2086 | 0.1974 | -0.0111 | 0.973 | 3.86e-01 | 4.38e-01 |
| Accuracy  | 0.7631 | 0.7050 | -0.0581 | 1.573 | 1.91e-01 | 2.50e-01 |
| F1  | 0.4257 | 0.3917 | -0.0340 | 1.797 | 1.47e-01 | 1.88e-01 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8307 | 0.8105 | -0.0202 | 1.841 | 1.39e-01 | 3.12e-01 |
| AUPRC  | 0.3864 | 0.3766 | -0.0098 | 0.405 | 7.06e-01 | 6.25e-01 |
| Brier  | 0.2026 | 0.1836 | -0.0189 | 1.020 | 3.65e-01 | 6.25e-01 |
| Accuracy  | 0.8072 | 0.7030 | -0.1042 | 1.494 | 2.10e-01 | 3.12e-01 |
| F1  | 0.4552 | 0.4066 | -0.0486 | 1.494 | 2.09e-01 | 3.12e-01 |

#### Case: len128/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8220 | 0.8069 | -0.0151 | 1.268 | 2.74e-01 | 4.38e-01 |
| AUPRC  | 0.3808 | 0.3745 | -0.0064 | 0.285 | 7.90e-01 | 8.12e-01 |
| Brier  | 0.1816 | 0.1901 | +0.0085 | -0.399 | 7.11e-01 | 4.38e-01 |
| Accuracy  | 0.7643 | 0.7631 | -0.0012 | 0.020 | 9.85e-01 | 8.12e-01 |
| F1  | 0.4418 | 0.4320 | -0.0097 | 0.227 | 8.32e-01 | 8.12e-01 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8175 | 0.8030 | -0.0145 | 1.501 | 2.08e-01 | 3.12e-01 |
| AUPRC * | 0.4109 | 0.3574 | -0.0535 | 3.010 | 3.95e-02 | 6.25e-02 |
| Brier  | 0.2073 | 0.1884 | -0.0189 | 1.488 | 2.11e-01 | 3.12e-01 |
| Accuracy  | 0.7620 | 0.7750 | +0.0130 | -0.373 | 7.28e-01 | 8.75e-01 |
| F1  | 0.4288 | 0.4314 | +0.0026 | -0.114 | 9.15e-01 | 6.25e-01 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8184 | 0.8030 | -0.0154 | 1.483 | 2.12e-01 | 1.88e-01 |
| AUPRC  | 0.4181 | 0.4069 | -0.0112 | 0.428 | 6.91e-01 | 4.38e-01 |
| Brier  | 0.1871 | 0.1680 | -0.0191 | 1.493 | 2.10e-01 | 1.88e-01 |
| Accuracy  | 0.7556 | 0.7363 | -0.0194 | 0.967 | 3.88e-01 | 3.75e-01 |
| F1  | 0.4209 | 0.4061 | -0.0148 | 0.870 | 4.33e-01 | 6.25e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.7977 | -0.0073 | 0.381 | 7.22e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3648 | -0.0208 | 0.781 | 4.78e-01 | 6.25e-01 |
| Brier † | 0.1800 | 0.1974 | +0.0174 | -2.151 | 9.79e-02 | 3.12e-01 |
| Accuracy * | 0.7491 | 0.7050 | -0.0441 | 3.185 | 3.34e-02 | 6.25e-02 |
| F1 † | 0.4187 | 0.3917 | -0.0270 | 2.164 | 9.64e-02 | 1.88e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8105 | +0.0054 | -0.160 | 8.80e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3766 | -0.0091 | 0.170 | 8.73e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1836 | +0.0036 | -0.203 | 8.49e-01 | 1.00e+00 |
| Accuracy  | 0.7491 | 0.7030 | -0.0461 | 0.630 | 5.63e-01 | 4.38e-01 |
| F1  | 0.4187 | 0.4066 | -0.0121 | 0.306 | 7.75e-01 | 1.00e+00 |

### len128/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8069 | +0.0019 | -0.113 | 9.16e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3745 | -0.0112 | 0.282 | 7.92e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1901 | +0.0101 | -1.006 | 3.71e-01 | 4.38e-01 |
| Accuracy  | 0.7491 | 0.7631 | +0.0141 | -0.286 | 7.89e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4320 | +0.0133 | -0.414 | 7.00e-01 | 8.12e-01 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8030 | -0.0021 | 0.181 | 8.65e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3574 | -0.0283 | 0.699 | 5.23e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.1884 | +0.0084 | -0.733 | 5.04e-01 | 8.12e-01 |
| Accuracy  | 0.7491 | 0.7750 | +0.0259 | -0.476 | 6.59e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4314 | +0.0126 | -0.369 | 7.31e-01 | 1.00e+00 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8030 | -0.0020 | 0.117 | 9.12e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.4069 | +0.0212 | -0.498 | 6.45e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.1680 | -0.0120 | 1.546 | 1.97e-01 | 3.12e-01 |
| Accuracy  | 0.7491 | 0.7363 | -0.0128 | 0.256 | 8.11e-01 | 6.25e-01 |
| F1  | 0.4187 | 0.4061 | -0.0126 | 0.374 | 7.27e-01 | 8.12e-01 |

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
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8267 | 0.7447 | 0.8998 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4512 | 0.2670 | 0.6394 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.1993 | 0.1839 | 0.2154 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7082 | 0.6481 | 0.7639 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.3585 | 0.2430 | 0.4779 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8223 | 0.7378 | 0.8931 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3828 | 0.2136 | 0.5731 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.2003 | 0.1838 | 0.2185 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7608 | 0.7033 | 0.8182 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3902 | 0.2531 | 0.5205 |
| M2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.8069 | 0.7125 | 0.8907 |
| M2 | CrossAttn | len128/scale_clinic | AUPRC | 0.4020 | 0.2357 | 0.5963 |
| M2 | CrossAttn | len128/scale_clinic | Brier | 0.2183 | 0.2030 | 0.2340 |
| M2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6223 | 0.5622 | 0.6824 |
| M2 | CrossAttn | len128/scale_clinic | F1 | 0.3333 | 0.2314 | 0.4387 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8013 | 0.7046 | 0.8904 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.4345 | 0.2537 | 0.6388 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.2016 | 0.1882 | 0.2151 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.7167 | 0.6567 | 0.7768 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.3400 | 0.2200 | 0.4577 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8162 | 0.7248 | 0.8987 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.4138 | 0.2496 | 0.6128 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.1932 | 0.1756 | 0.2109 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6867 | 0.6266 | 0.7468 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.3540 | 0.2449 | 0.4727 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8213 | 0.7371 | 0.8975 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4058 | 0.2401 | 0.5909 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.2166 | 0.2023 | 0.2318 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.5579 | 0.4936 | 0.6223 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.2993 | 0.2074 | 0.3971 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8140 | 0.7127 | 0.8998 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3672 | 0.2106 | 0.5701 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1711 | 0.1550 | 0.1883 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7943 | 0.7368 | 0.8469 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4416 | 0.2909 | 0.5714 |
| M2_2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.7863 | 0.6937 | 0.8675 |
| M2_2 | CrossAttn | len128/scale_clinic | AUPRC | 0.2929 | 0.1793 | 0.4902 |
| M2_2 | CrossAttn | len128/scale_clinic | Brier | 0.1851 | 0.1686 | 0.2009 |
| M2_2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6738 | 0.6137 | 0.7339 |
| M2_2 | CrossAttn | len128/scale_clinic | F1 | 0.3214 | 0.2083 | 0.4370 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.7708 | 0.6677 | 0.8597 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.2990 | 0.1712 | 0.4789 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.1901 | 0.1727 | 0.2075 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6738 | 0.6137 | 0.7297 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.3214 | 0.2124 | 0.4348 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.7902 | 0.6893 | 0.8775 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3997 | 0.2304 | 0.5758 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.1775 | 0.1658 | 0.1894 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.7682 | 0.7124 | 0.8240 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.3571 | 0.2222 | 0.4855 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8342 | 0.7508 | 0.9073 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.4352 | 0.2651 | 0.6195 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.1932 | 0.1760 | 0.2096 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.6266 | 0.5622 | 0.6910 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.3256 | 0.2238 | 0.4306 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8201 | 0.7263 | 0.8973 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.3926 | 0.2258 | 0.6030 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.1890 | 0.1743 | 0.2041 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.7368 | 0.6746 | 0.7943 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.3678 | 0.2307 | 0.4948 |
| M3 | CrossAttn3 | len128/scale_clinic | AUC-ROC | 0.7992 | 0.6928 | 0.8908 |
| M3 | CrossAttn3 | len128/scale_clinic | AUPRC | 0.4002 | 0.2376 | 0.5920 |
| M3 | CrossAttn3 | len128/scale_clinic | Brier | 0.1730 | 0.1591 | 0.1871 |
| M3 | CrossAttn3 | len128/scale_clinic | Accuracy | 0.8112 | 0.7597 | 0.8627 |
| M3 | CrossAttn3 | len128/scale_clinic | F1 | 0.4500 | 0.3117 | 0.5800 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8294 | 0.7519 | 0.9007 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.3975 | 0.2368 | 0.5931 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.2223 | 0.2077 | 0.2372 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.6094 | 0.5494 | 0.6695 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.3259 | 0.2258 | 0.4306 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.7627 | 0.6566 | 0.8581 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.3258 | 0.1855 | 0.4975 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.1949 | 0.1770 | 0.2131 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.6481 | 0.5880 | 0.7124 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.3279 | 0.2222 | 0.4384 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-norm | 0.8119 | 0.8267 | +0.0148 | -0.627 | 5.307e-01 | ns |
| M1-LR vs M2-len128 | 0.8119 | 0.8069 | -0.0050 | 0.155 | 8.772e-01 | ns |
| M1-LR vs M2-crop80 | 0.8119 | 0.8013 | -0.0106 | 0.384 | 7.013e-01 | ns |
| M1-LR vs M2-crop60 | 0.8119 | 0.8162 | +0.0042 | -0.173 | 8.628e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-norm | 0.8119 | 0.8342 | +0.0223 | -0.725 | 4.684e-01 | ns |
| M1-LR vs M3-len128 | 0.8119 | 0.7992 | -0.0127 | 0.327 | 7.438e-01 | ns |
| M1-LR vs M3-crop80 | 0.8119 | 0.8294 | +0.0175 | -0.807 | 4.197e-01 | ns |
| M1-LR vs M3-crop60 | 0.8119 | 0.7627 | -0.0492 | 1.576 | 1.150e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M2_2-norm | 0.8267 | 0.8213 | -0.0054 | 0.307 | 7.586e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8223 | 0.5046 | -0.3177 | 4.815 | 1.474e-06 | *** |
| M2-len128 vs M2_2-len128 | 0.8069 | 0.7863 | -0.0206 | 0.734 | 4.628e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.8013 | 0.7708 | -0.0306 | 0.898 | 3.692e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.8162 | 0.7902 | -0.0260 | 0.786 | 4.318e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M3-norm | 0.8267 | 0.8342 | +0.0075 | -0.408 | 6.832e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.8223 | 0.8201 | -0.0022 | 0.093 | 9.259e-01 | ns |
| M2-len128 vs M3-len128 | 0.8069 | 0.7992 | -0.0077 | 0.332 | 7.400e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8013 | 0.8294 | +0.0281 | -1.166 | 2.434e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.8162 | 0.7627 | -0.0535 | 2.238 | 2.521e-02 | * |

