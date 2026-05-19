# Scaling Comparison — Test Set Performance

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.7222 | 0.2451 | 0.2145 | 0.6812 | 0.2826 |
| M2 | CrossAttn | norm/scale_both | 0.7547 | 0.3221 | 0.2554 | 0.5894 | 0.2857 |
| M2_2 | CrossAttn | norm/scale_both | 0.7678 | 0.3539 | 0.2509 | 0.5894 | 0.2857 |
| M3 | CrossAttn3 | crop80/scale_both | 0.7645 | 0.3198 | 0.2162 | 0.6473 | 0.3178 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_clinic** | 0.7222 | 0.2451 | 0.2145 | 0.6812 | 0.2826 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.7348 | 0.3347 | 0.2212 | 0.6715 | 0.3061 |
| crop80/scale_both | 0.7427 | 0.2650 | 0.2037 | 0.6860 | 0.3299 |
| crop60/scale_both | 0.7307 | 0.2853 | 0.2389 | 0.5894 | 0.2609 |
| **norm/scale_both** | 0.7547 | 0.3221 | 0.2554 | 0.5894 | 0.2857 |
| excl_extreme/scale_both | 0.7242 | 0.2936 | 0.2012 | 0.6757 | 0.2857 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.6790 | 0.2149 | 0.2195 | 0.6667 | 0.2887 |
| crop80/scale_both | 0.6882 | 0.2126 | 0.2491 | 0.6232 | 0.2642 |
| crop60/scale_both | 0.6969 | 0.2517 | 0.2253 | 0.7101 | 0.3333 |
| **norm/scale_both** | 0.7678 | 0.3539 | 0.2509 | 0.5894 | 0.2857 |
| excl_extreme/scale_both | 0.6933 | 0.1908 | 0.2366 | 0.6865 | 0.3256 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.7627 | 0.3332 | 0.1793 | 0.7053 | 0.3441 |
| **crop80/scale_both** | 0.7645 | 0.3198 | 0.2162 | 0.6473 | 0.3178 |
| crop60/scale_both | 0.7353 | 0.3354 | 0.1709 | 0.7343 | 0.3210 |
| norm/scale_both | 0.7616 | 0.3255 | 0.2547 | 0.6039 | 0.2807 |
| excl_extreme/scale_both | 0.7189 | 0.2721 | 0.2222 | 0.6595 | 0.2759 |

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

### len128/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8163 | 0.8305 | +0.0142 | -0.692 | 5.27e-01 | 6.25e-01 |
| AUPRC  | 0.4083 | 0.4102 | +0.0019 | -0.052 | 9.61e-01 | 1.00e+00 |
| Brier  | 0.1782 | 0.1518 | -0.0263 | 1.606 | 1.83e-01 | 1.88e-01 |
| Accuracy  | 0.7285 | 0.7527 | +0.0242 | -0.773 | 4.83e-01 | 4.38e-01 |
| F1  | 0.3441 | 0.3732 | +0.0291 | -1.366 | 2.44e-01 | 3.12e-01 |

### crop80/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8163 | 0.8324 | +0.0161 | -0.913 | 4.13e-01 | 6.25e-01 |
| AUPRC  | 0.4083 | 0.3678 | -0.0405 | 1.850 | 1.38e-01 | 1.25e-01 |
| Brier  | 0.1782 | 0.2102 | +0.0321 | -1.106 | 3.31e-01 | 3.12e-01 |
| Accuracy  | 0.7285 | 0.6109 | -0.1176 | 1.329 | 2.55e-01 | 2.50e-01 |
| F1  | 0.3441 | 0.3303 | -0.0137 | 0.411 | 7.02e-01 | 6.25e-01 |

### crop60/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8163 | 0.8263 | +0.0100 | -0.522 | 6.29e-01 | 8.12e-01 |
| AUPRC * | 0.4083 | 0.3672 | -0.0411 | 3.081 | 3.69e-02 | 1.25e-01 |
| Brier  | 0.1782 | 0.1751 | -0.0031 | 0.300 | 7.79e-01 | 1.00e+00 |
| Accuracy  | 0.7285 | 0.7200 | -0.0085 | 0.283 | 7.91e-01 | 8.12e-01 |
| F1  | 0.3441 | 0.3664 | +0.0224 | -0.697 | 5.24e-01 | 6.25e-01 |

### norm/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8163 | 0.8458 | +0.0295 | -1.570 | 1.92e-01 | 3.12e-01 |
| AUPRC  | 0.4083 | 0.3690 | -0.0393 | 1.302 | 2.63e-01 | 3.12e-01 |
| Brier  | 0.1782 | 0.1776 | -0.0005 | 0.062 | 9.53e-01 | 1.00e+00 |
| Accuracy  | 0.7285 | 0.7224 | -0.0061 | 0.587 | 5.89e-01 | 7.50e-01 |
| F1  | 0.3441 | 0.3800 | +0.0360 | -1.371 | 2.42e-01 | 3.75e-01 |

### excl_extreme/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8163 | 0.8351 | +0.0189 | -0.512 | 6.35e-01 | 6.25e-01 |
| AUPRC  | 0.4083 | 0.3892 | -0.0192 | 0.310 | 7.72e-01 | 4.38e-01 |
| Brier  | 0.1782 | 0.1934 | +0.0153 | -0.477 | 6.59e-01 | 6.25e-01 |
| Accuracy  | 0.7285 | 0.6791 | -0.0493 | 0.625 | 5.66e-01 | 1.00e+00 |
| F1  | 0.3441 | 0.3618 | +0.0177 | -0.333 | 7.56e-01 | 6.25e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len128/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8305 | 0.8247 | -0.0058 | 0.667 | 5.41e-01 | 8.12e-01 |
| AUPRC  | 0.4102 | 0.3391 | -0.0711 | 1.500 | 2.08e-01 | 3.12e-01 |
| Brier  | 0.1518 | 0.1868 | +0.0350 | -1.588 | 1.87e-01 | 1.25e-01 |
| Accuracy  | 0.7527 | 0.7079 | -0.0448 | 0.914 | 4.12e-01 | 3.75e-01 |
| F1  | 0.3732 | 0.3726 | -0.0006 | 0.028 | 9.79e-01 | 1.00e+00 |

#### Case: crop80/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8324 | 0.8175 | -0.0149 | 1.275 | 2.71e-01 | 3.12e-01 |
| AUPRC  | 0.3678 | 0.3385 | -0.0293 | 1.199 | 2.97e-01 | 4.38e-01 |
| Brier † | 0.2102 | 0.1733 | -0.0370 | 2.474 | 6.87e-02 | 1.25e-01 |
| Accuracy  | 0.6109 | 0.7406 | +0.1297 | -2.032 | 1.12e-01 | 1.25e-01 |
| F1 † | 0.3303 | 0.3914 | +0.0611 | -2.646 | 5.72e-02 | 6.25e-02 |

#### Case: crop60/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8263 | 0.8189 | -0.0074 | 0.628 | 5.64e-01 | 6.25e-01 |
| AUPRC  | 0.3672 | 0.3588 | -0.0084 | 0.459 | 6.70e-01 | 8.12e-01 |
| Brier  | 0.1751 | 0.1772 | +0.0022 | -0.148 | 8.89e-01 | 1.00e+00 |
| Accuracy  | 0.7200 | 0.7212 | +0.0012 | -0.032 | 9.76e-01 | 6.25e-01 |
| F1  | 0.3664 | 0.3659 | -0.0006 | 0.020 | 9.85e-01 | 1.00e+00 |

#### Case: norm/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8458 | 0.8221 | -0.0237 | 3.021 | 3.91e-02 | 6.25e-02 |
| AUPRC  | 0.3690 | 0.3511 | -0.0179 | 0.637 | 5.59e-01 | 6.25e-01 |
| Brier † | 0.1776 | 0.2268 | +0.0492 | -2.732 | 5.23e-02 | 1.25e-01 |
| Accuracy † | 0.7224 | 0.6473 | -0.0752 | 2.197 | 9.30e-02 | 1.25e-01 |
| F1 * | 0.3800 | 0.3320 | -0.0481 | 2.860 | 4.59e-02 | 1.25e-01 |

#### Case: excl_extreme/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8351 | 0.8134 | -0.0217 | 2.017 | 1.14e-01 | 1.25e-01 |
| AUPRC  | 0.3892 | 0.3996 | +0.0104 | -0.206 | 8.47e-01 | 8.12e-01 |
| Brier  | 0.1934 | 0.1764 | -0.0171 | 0.564 | 6.03e-01 | 1.00e+00 |
| Accuracy  | 0.6791 | 0.7542 | +0.0751 | -0.762 | 4.88e-01 | 8.12e-01 |
| F1  | 0.3618 | 0.3622 | +0.0005 | -0.008 | 9.94e-01 | 6.25e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8163 | 0.8247 | +0.0084 | -0.595 | 5.84e-01 | 8.12e-01 |
| AUPRC † | 0.4083 | 0.3391 | -0.0692 | 2.724 | 5.28e-02 | 1.25e-01 |
| Brier  | 0.1782 | 0.1868 | +0.0087 | -0.675 | 5.37e-01 | 4.38e-01 |
| Accuracy  | 0.7285 | 0.7079 | -0.0206 | 0.645 | 5.54e-01 | 6.25e-01 |
| F1  | 0.3441 | 0.3726 | +0.0285 | -1.162 | 3.10e-01 | 3.12e-01 |

### crop80/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8163 | 0.8175 | +0.0012 | -0.101 | 9.25e-01 | 8.12e-01 |
| AUPRC † | 0.4083 | 0.3385 | -0.0699 | 2.164 | 9.64e-02 | 3.12e-01 |
| Brier  | 0.1782 | 0.1733 | -0.0049 | 0.246 | 8.18e-01 | 8.12e-01 |
| Accuracy  | 0.7285 | 0.7406 | +0.0121 | -0.369 | 7.30e-01 | 8.12e-01 |
| F1 † | 0.3441 | 0.3914 | +0.0474 | -2.500 | 6.68e-02 | 6.25e-02 |

### crop60/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8163 | 0.8189 | +0.0026 | -0.204 | 8.48e-01 | 8.12e-01 |
| AUPRC  | 0.4083 | 0.3588 | -0.0495 | 1.908 | 1.29e-01 | 3.12e-01 |
| Brier  | 0.1782 | 0.1772 | -0.0009 | 0.088 | 9.34e-01 | 1.00e+00 |
| Accuracy  | 0.7285 | 0.7212 | -0.0073 | 0.406 | 7.06e-01 | 6.25e-01 |
| F1  | 0.3441 | 0.3659 | +0.0218 | -1.419 | 2.29e-01 | 4.38e-01 |

### norm/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8163 | 0.8221 | +0.0059 | -0.382 | 7.22e-01 | 8.12e-01 |
| AUPRC  | 0.4083 | 0.3511 | -0.0572 | 1.345 | 2.50e-01 | 3.12e-01 |
| Brier * | 0.1782 | 0.2268 | +0.0487 | -3.422 | 2.67e-02 | 6.25e-02 |
| Accuracy † | 0.7285 | 0.6473 | -0.0812 | 2.401 | 7.43e-02 | 1.25e-01 |
| F1  | 0.3441 | 0.3320 | -0.0121 | 0.343 | 7.49e-01 | 8.12e-01 |

### excl_extreme/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8163 | 0.8134 | -0.0028 | 0.062 | 9.53e-01 | 6.25e-01 |
| AUPRC  | 0.4083 | 0.3996 | -0.0088 | 0.246 | 8.18e-01 | 1.00e+00 |
| Brier  | 0.1782 | 0.1764 | -0.0018 | 0.093 | 9.30e-01 | 8.12e-01 |
| Accuracy  | 0.7285 | 0.7542 | +0.0257 | -0.508 | 6.38e-01 | 6.25e-01 |
| F1  | 0.3441 | 0.3622 | +0.0182 | -1.271 | 2.73e-01 | 3.12e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_clinic | AUC-ROC | 0.7222 | 0.6094 | 0.8161 |
| M1 | LR | scale_clinic | AUPRC | 0.2451 | 0.1191 | 0.4226 |
| M1 | LR | scale_clinic | Brier | 0.2145 | 0.1839 | 0.2454 |
| M1 | LR | scale_clinic | Accuracy | 0.6812 | 0.6184 | 0.7393 |
| M1 | LR | scale_clinic | F1 | 0.2826 | 0.1521 | 0.4001 |
| M2 | CrossAttn | len128/scale_both | AUC-ROC | 0.7348 | 0.6034 | 0.8483 |
| M2 | CrossAttn | len128/scale_both | AUPRC | 0.3347 | 0.1678 | 0.5165 |
| M2 | CrossAttn | len128/scale_both | Brier | 0.2212 | 0.1838 | 0.2608 |
| M2 | CrossAttn | len128/scale_both | Accuracy | 0.6715 | 0.6087 | 0.7343 |
| M2 | CrossAttn | len128/scale_both | F1 | 0.3061 | 0.1818 | 0.4200 |
| M2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.7427 | 0.6205 | 0.8523 |
| M2 | CrossAttn | crop80/scale_both | AUPRC | 0.2650 | 0.1506 | 0.4701 |
| M2 | CrossAttn | crop80/scale_both | Brier | 0.2037 | 0.1685 | 0.2412 |
| M2 | CrossAttn | crop80/scale_both | Accuracy | 0.6860 | 0.6232 | 0.7488 |
| M2 | CrossAttn | crop80/scale_both | F1 | 0.3299 | 0.2022 | 0.4464 |
| M2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.7307 | 0.5911 | 0.8551 |
| M2 | CrossAttn | crop60/scale_both | AUPRC | 0.2853 | 0.1518 | 0.5054 |
| M2 | CrossAttn | crop60/scale_both | Brier | 0.2389 | 0.2045 | 0.2749 |
| M2 | CrossAttn | crop60/scale_both | Accuracy | 0.5894 | 0.5216 | 0.6522 |
| M2 | CrossAttn | crop60/scale_both | F1 | 0.2609 | 0.1538 | 0.3609 |
| M2 | CrossAttn | norm/scale_both | AUC-ROC | 0.7547 | 0.6380 | 0.8520 |
| M2 | CrossAttn | norm/scale_both | AUPRC | 0.3221 | 0.1580 | 0.5104 |
| M2 | CrossAttn | norm/scale_both | Brier | 0.2554 | 0.2207 | 0.2928 |
| M2 | CrossAttn | norm/scale_both | Accuracy | 0.5894 | 0.5169 | 0.6570 |
| M2 | CrossAttn | norm/scale_both | F1 | 0.2857 | 0.1709 | 0.3885 |
| M2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.7242 | 0.5862 | 0.8491 |
| M2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.2936 | 0.1414 | 0.5118 |
| M2 | CrossAttn | excl_extreme/scale_both | Brier | 0.2012 | 0.1657 | 0.2399 |
| M2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.6757 | 0.6054 | 0.7405 |
| M2 | CrossAttn | excl_extreme/scale_both | F1 | 0.2857 | 0.1621 | 0.4086 |
| M2_2 | CrossAttn | len128/scale_both | AUC-ROC | 0.6790 | 0.5291 | 0.8075 |
| M2_2 | CrossAttn | len128/scale_both | AUPRC | 0.2149 | 0.1170 | 0.4020 |
| M2_2 | CrossAttn | len128/scale_both | Brier | 0.2195 | 0.1862 | 0.2536 |
| M2_2 | CrossAttn | len128/scale_both | Accuracy | 0.6667 | 0.6039 | 0.7295 |
| M2_2 | CrossAttn | len128/scale_both | F1 | 0.2887 | 0.1649 | 0.4001 |
| M2_2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.6882 | 0.5551 | 0.8039 |
| M2_2 | CrossAttn | crop80/scale_both | AUPRC | 0.2126 | 0.1122 | 0.3946 |
| M2_2 | CrossAttn | crop80/scale_both | Brier | 0.2491 | 0.2114 | 0.2884 |
| M2_2 | CrossAttn | crop80/scale_both | Accuracy | 0.6232 | 0.5556 | 0.6908 |
| M2_2 | CrossAttn | crop80/scale_both | F1 | 0.2642 | 0.1509 | 0.3726 |
| M2_2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.6969 | 0.5537 | 0.8167 |
| M2_2 | CrossAttn | crop60/scale_both | AUPRC | 0.2517 | 0.1248 | 0.4351 |
| M2_2 | CrossAttn | crop60/scale_both | Brier | 0.2253 | 0.1827 | 0.2690 |
| M2_2 | CrossAttn | crop60/scale_both | Accuracy | 0.7101 | 0.6473 | 0.7729 |
| M2_2 | CrossAttn | crop60/scale_both | F1 | 0.3333 | 0.1999 | 0.4554 |
| M2_2 | CrossAttn | norm/scale_both | AUC-ROC | 0.7678 | 0.6509 | 0.8657 |
| M2_2 | CrossAttn | norm/scale_both | AUPRC | 0.3539 | 0.1871 | 0.5557 |
| M2_2 | CrossAttn | norm/scale_both | Brier | 0.2509 | 0.2196 | 0.2864 |
| M2_2 | CrossAttn | norm/scale_both | Accuracy | 0.5894 | 0.5217 | 0.6522 |
| M2_2 | CrossAttn | norm/scale_both | F1 | 0.2857 | 0.1724 | 0.3802 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.6933 | 0.5377 | 0.8361 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.1908 | 0.1049 | 0.3420 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Brier | 0.2366 | 0.1946 | 0.2837 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.6865 | 0.6162 | 0.7514 |
| M2_2 | CrossAttn | excl_extreme/scale_both | F1 | 0.3256 | 0.1917 | 0.4495 |
| M3 | CrossAttn3 | len128/scale_both | AUC-ROC | 0.7627 | 0.6377 | 0.8746 |
| M3 | CrossAttn3 | len128/scale_both | AUPRC | 0.3332 | 0.1784 | 0.5443 |
| M3 | CrossAttn3 | len128/scale_both | Brier | 0.1793 | 0.1493 | 0.2110 |
| M3 | CrossAttn3 | len128/scale_both | Accuracy | 0.7053 | 0.6425 | 0.7681 |
| M3 | CrossAttn3 | len128/scale_both | F1 | 0.3441 | 0.2121 | 0.4632 |
| M3 | CrossAttn3 | crop80/scale_both | AUC-ROC | 0.7645 | 0.6475 | 0.8653 |
| M3 | CrossAttn3 | crop80/scale_both | AUPRC | 0.3198 | 0.1612 | 0.5251 |
| M3 | CrossAttn3 | crop80/scale_both | Brier | 0.2162 | 0.1860 | 0.2488 |
| M3 | CrossAttn3 | crop80/scale_both | Accuracy | 0.6473 | 0.5845 | 0.7101 |
| M3 | CrossAttn3 | crop80/scale_both | F1 | 0.3178 | 0.1923 | 0.4274 |
| M3 | CrossAttn3 | crop60/scale_both | AUC-ROC | 0.7353 | 0.5954 | 0.8592 |
| M3 | CrossAttn3 | crop60/scale_both | AUPRC | 0.3354 | 0.1755 | 0.5499 |
| M3 | CrossAttn3 | crop60/scale_both | Brier | 0.1709 | 0.1385 | 0.2057 |
| M3 | CrossAttn3 | crop60/scale_both | Accuracy | 0.7343 | 0.6715 | 0.7923 |
| M3 | CrossAttn3 | crop60/scale_both | F1 | 0.3210 | 0.1818 | 0.4516 |
| M3 | CrossAttn3 | norm/scale_both | AUC-ROC | 0.7616 | 0.6418 | 0.8624 |
| M3 | CrossAttn3 | norm/scale_both | AUPRC | 0.3255 | 0.1614 | 0.5370 |
| M3 | CrossAttn3 | norm/scale_both | Brier | 0.2547 | 0.2164 | 0.2941 |
| M3 | CrossAttn3 | norm/scale_both | Accuracy | 0.6039 | 0.5362 | 0.6715 |
| M3 | CrossAttn3 | norm/scale_both | F1 | 0.2807 | 0.1651 | 0.3833 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUC-ROC | 0.7189 | 0.5784 | 0.8453 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUPRC | 0.2721 | 0.1433 | 0.5116 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Brier | 0.2222 | 0.1839 | 0.2628 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Accuracy | 0.6595 | 0.5892 | 0.7297 |
| M3 | CrossAttn3 | excl_extreme/scale_both | F1 | 0.2759 | 0.1522 | 0.4000 |

