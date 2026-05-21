# Scaling Comparison — Test Set Performance (AEC 256pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8187 | 0.2970 | 0.1763 | 0.7326 | 0.3947 |
| M2 | CrossAttn | excl_extreme/scale_both | 0.8109 | 0.3325 | 0.1720 | 0.7143 | 0.3714 |
| M2_2 | CrossAttn | crop60/scale_both | 0.8155 | 0.3119 | 0.2067 | 0.6860 | 0.3864 |
| M3 | CrossAttn3 | crop60/scale_both | 0.8175 | 0.4347 | 0.1823 | 0.7048 | 0.3951 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_clinic** | 0.8187 | 0.2970 | 0.1763 | 0.7326 | 0.3947 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_both | 0.7849 | 0.3117 | 0.2089 | 0.6744 | 0.3778 |
| crop80/scale_both | 0.8057 | 0.3178 | 0.1826 | 0.7326 | 0.4250 |
| crop60/scale_both | 0.7925 | 0.3743 | 0.1899 | 0.7035 | 0.3855 |
| norm/scale_both | 0.7717 | 0.2958 | 0.2351 | 0.6802 | 0.3956 |
| **excl_extreme/scale_both** | 0.8109 | 0.3325 | 0.1720 | 0.7143 | 0.3714 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_both | 0.7597 | 0.2671 | 0.1853 | 0.7500 | 0.4110 |
| crop80/scale_both | 0.7805 | 0.2894 | 0.1904 | 0.6977 | 0.3810 |
| **crop60/scale_both** | 0.8155 | 0.3119 | 0.2067 | 0.6860 | 0.3864 |
| norm/scale_both | 0.7985 | 0.2754 | 0.1913 | 0.7442 | 0.4359 |
| excl_extreme/scale_both | 0.7965 | 0.3465 | 0.1885 | 0.7078 | 0.3478 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_both | 0.8027 | 0.4470 | 0.1829 | 0.7048 | 0.3951 |
| crop80/scale_both | 0.7627 | 0.3384 | 0.1390 | 0.7771 | 0.3729 |
| **crop60/scale_both** | 0.8175 | 0.4347 | 0.1823 | 0.7048 | 0.3951 |
| norm/scale_both | 0.7863 | 0.2941 | 0.1945 | 0.7048 | 0.4096 |
| excl_extreme/scale_both | 0.7434 | 0.2911 | 0.1462 | 0.7770 | 0.3529 |

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

### len256/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8282 | +0.0210 | -0.728 | 5.07e-01 | 6.25e-01 |
| AUPRC  | 0.4392 | 0.4414 | +0.0021 | -0.049 | 9.63e-01 | 1.00e+00 |
| Brier  | 0.1781 | 0.1684 | -0.0098 | 0.439 | 6.83e-01 | 8.12e-01 |
| Accuracy  | 0.7263 | 0.7453 | +0.0190 | -0.454 | 6.74e-01 | 6.88e-01 |
| F1  | 0.3851 | 0.4134 | +0.0283 | -0.732 | 5.05e-01 | 4.38e-01 |

### crop80/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8296 | +0.0224 | -0.596 | 5.84e-01 | 6.25e-01 |
| AUPRC  | 0.4392 | 0.4264 | -0.0129 | 0.181 | 8.65e-01 | 1.00e+00 |
| Brier  | 0.1781 | 0.1942 | +0.0160 | -1.463 | 2.17e-01 | 1.88e-01 |
| Accuracy  | 0.7263 | 0.7279 | +0.0016 | -0.071 | 9.46e-01 | 8.12e-01 |
| F1  | 0.3851 | 0.4221 | +0.0370 | -0.992 | 3.78e-01 | 6.25e-01 |

### crop60/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8345 | +0.0273 | -0.799 | 4.69e-01 | 6.25e-01 |
| AUPRC  | 0.4392 | 0.4584 | +0.0191 | -0.390 | 7.16e-01 | 8.12e-01 |
| Brier  | 0.1781 | 0.1789 | +0.0008 | -0.052 | 9.61e-01 | 1.00e+00 |
| Accuracy  | 0.7263 | 0.7235 | -0.0028 | 0.068 | 9.49e-01 | 1.00e+00 |
| F1  | 0.3851 | 0.4074 | +0.0223 | -0.571 | 5.98e-01 | 6.25e-01 |

### norm/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8384 | +0.0312 | -0.847 | 4.45e-01 | 4.38e-01 |
| AUPRC  | 0.4392 | 0.4411 | +0.0019 | -0.025 | 9.81e-01 | 1.00e+00 |
| Brier  | 0.1781 | 0.1605 | -0.0176 | 0.636 | 5.59e-01 | 6.25e-01 |
| Accuracy  | 0.7263 | 0.7612 | +0.0349 | -0.649 | 5.52e-01 | 6.25e-01 |
| F1  | 0.3851 | 0.4139 | +0.0288 | -0.885 | 4.26e-01 | 8.12e-01 |

### excl_extreme/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8154 | +0.0082 | -0.170 | 8.73e-01 | 8.12e-01 |
| AUPRC  | 0.4392 | 0.4058 | -0.0335 | 0.298 | 7.80e-01 | 8.12e-01 |
| Brier  | 0.1781 | 0.1695 | -0.0086 | 0.664 | 5.43e-01 | 8.12e-01 |
| Accuracy  | 0.7263 | 0.7440 | +0.0177 | -0.649 | 5.52e-01 | 6.25e-01 |
| F1  | 0.3851 | 0.4020 | +0.0169 | -0.375 | 7.27e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len256/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8282 | 0.8252 | -0.0030 | 0.078 | 9.42e-01 | 1.00e+00 |
| AUPRC  | 0.4414 | 0.4418 | +0.0005 | -0.006 | 9.95e-01 | 1.00e+00 |
| Brier  | 0.1684 | 0.1737 | +0.0053 | -0.290 | 7.86e-01 | 1.00e+00 |
| Accuracy  | 0.7453 | 0.7152 | -0.0301 | 0.933 | 4.04e-01 | 6.25e-01 |
| F1  | 0.4134 | 0.3964 | -0.0170 | 0.618 | 5.70e-01 | 8.12e-01 |

#### Case: crop80/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8296 | 0.8249 | -0.0047 | 0.150 | 8.88e-01 | 1.00e+00 |
| AUPRC  | 0.4264 | 0.4412 | +0.0148 | -0.173 | 8.71e-01 | 1.00e+00 |
| Brier  | 0.1942 | 0.1763 | -0.0179 | 1.267 | 2.74e-01 | 3.12e-01 |
| Accuracy  | 0.7279 | 0.7288 | +0.0009 | -0.030 | 9.77e-01 | 1.00e+00 |
| F1  | 0.4221 | 0.4092 | -0.0129 | 0.557 | 6.07e-01 | 1.00e+00 |

#### Case: crop60/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8345 | 0.8211 | -0.0134 | 0.381 | 7.22e-01 | 8.12e-01 |
| AUPRC  | 0.4584 | 0.4539 | -0.0045 | 0.045 | 9.67e-01 | 1.00e+00 |
| Brier * | 0.1789 | 0.2136 | +0.0347 | -3.956 | 1.67e-02 | 6.25e-02 |
| Accuracy † | 0.7235 | 0.6621 | -0.0613 | 2.411 | 7.34e-02 | 1.25e-01 |
| F1 † | 0.4074 | 0.3622 | -0.0451 | 2.386 | 7.55e-02 | 6.25e-02 |

#### Case: norm/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8384 | 0.8415 | +0.0031 | -0.078 | 9.42e-01 | 1.00e+00 |
| AUPRC  | 0.4411 | 0.4559 | +0.0148 | -0.200 | 8.51e-01 | 1.00e+00 |
| Brier  | 0.1605 | 0.1718 | +0.0113 | -0.389 | 7.17e-01 | 1.00e+00 |
| Accuracy  | 0.7612 | 0.7515 | -0.0097 | 0.162 | 8.79e-01 | 6.25e-01 |
| F1  | 0.4139 | 0.4190 | +0.0051 | -0.195 | 8.55e-01 | 1.00e+00 |

#### Case: excl_extreme/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8154 | 0.8357 | +0.0203 | -0.497 | 6.45e-01 | 6.25e-01 |
| AUPRC  | 0.4058 | 0.4340 | +0.0282 | -0.312 | 7.70e-01 | 1.00e+00 |
| Brier  | 0.1695 | 0.1624 | -0.0071 | 0.387 | 7.19e-01 | 6.25e-01 |
| Accuracy  | 0.7440 | 0.7661 | +0.0221 | -0.566 | 6.02e-01 | 6.25e-01 |
| F1  | 0.4020 | 0.4653 | +0.0633 | -1.360 | 2.46e-01 | 3.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len256/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8252 | +0.0180 | -0.480 | 6.56e-01 | 1.00e+00 |
| AUPRC  | 0.4392 | 0.4418 | +0.0026 | -0.032 | 9.76e-01 | 8.12e-01 |
| Brier  | 0.1781 | 0.1737 | -0.0045 | 0.166 | 8.76e-01 | 6.25e-01 |
| Accuracy  | 0.7263 | 0.7152 | -0.0111 | 0.228 | 8.31e-01 | 8.12e-01 |
| F1  | 0.3851 | 0.3964 | +0.0113 | -0.250 | 8.15e-01 | 8.12e-01 |

### crop80/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8249 | +0.0177 | -0.554 | 6.09e-01 | 8.12e-01 |
| AUPRC  | 0.4392 | 0.4412 | +0.0020 | -0.019 | 9.85e-01 | 8.12e-01 |
| Brier  | 0.1781 | 0.1763 | -0.0018 | 0.084 | 9.37e-01 | 8.12e-01 |
| Accuracy  | 0.7263 | 0.7288 | +0.0025 | -0.056 | 9.58e-01 | 1.00e+00 |
| F1  | 0.3851 | 0.4092 | +0.0241 | -0.716 | 5.13e-01 | 4.38e-01 |

### crop60/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8211 | +0.0139 | -0.416 | 6.99e-01 | 1.00e+00 |
| AUPRC  | 0.4392 | 0.4539 | +0.0147 | -0.156 | 8.84e-01 | 8.12e-01 |
| Brier † | 0.1781 | 0.2136 | +0.0355 | -2.398 | 7.45e-02 | 1.25e-01 |
| Accuracy  | 0.7263 | 0.6621 | -0.0642 | 1.551 | 1.96e-01 | 1.88e-01 |
| F1  | 0.3851 | 0.3622 | -0.0229 | 0.719 | 5.12e-01 | 4.38e-01 |

### norm/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8415 | +0.0343 | -1.041 | 3.57e-01 | 4.38e-01 |
| AUPRC  | 0.4392 | 0.4559 | +0.0167 | -0.189 | 8.59e-01 | 1.00e+00 |
| Brier  | 0.1781 | 0.1718 | -0.0063 | 0.345 | 7.48e-01 | 8.12e-01 |
| Accuracy  | 0.7263 | 0.7515 | +0.0252 | -0.614 | 5.72e-01 | 8.12e-01 |
| F1  | 0.3851 | 0.4190 | +0.0339 | -0.822 | 4.57e-01 | 4.38e-01 |

### excl_extreme/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8357 | +0.0285 | -0.924 | 4.08e-01 | 1.00e+00 |
| AUPRC  | 0.4392 | 0.4340 | -0.0052 | 0.178 | 8.68e-01 | 6.25e-01 |
| Brier  | 0.1781 | 0.1624 | -0.0157 | 0.805 | 4.66e-01 | 4.38e-01 |
| Accuracy  | 0.7263 | 0.7661 | +0.0398 | -0.805 | 4.66e-01 | 3.12e-01 |
| F1  | 0.3851 | 0.4653 | +0.0802 | -1.363 | 2.44e-01 | 3.12e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_clinic | AUC-ROC | 0.8187 | 0.7452 | 0.8849 |
| M1 | LR | scale_clinic | AUPRC | 0.2970 | 0.1870 | 0.4879 |
| M1 | LR | scale_clinic | Brier | 0.1763 | 0.1481 | 0.2066 |
| M1 | LR | scale_clinic | Accuracy | 0.7326 | 0.6686 | 0.7965 |
| M1 | LR | scale_clinic | F1 | 0.3947 | 0.2424 | 0.5278 |
| M2 | CrossAttn | len256/scale_both | AUC-ROC | 0.7849 | 0.6911 | 0.8716 |
| M2 | CrossAttn | len256/scale_both | AUPRC | 0.3117 | 0.1888 | 0.5175 |
| M2 | CrossAttn | len256/scale_both | Brier | 0.2089 | 0.1724 | 0.2442 |
| M2 | CrossAttn | len256/scale_both | Accuracy | 0.6744 | 0.6105 | 0.7442 |
| M2 | CrossAttn | len256/scale_both | F1 | 0.3778 | 0.2499 | 0.5000 |
| M2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.8057 | 0.7142 | 0.8906 |
| M2 | CrossAttn | crop80/scale_both | AUPRC | 0.3178 | 0.2001 | 0.5213 |
| M2 | CrossAttn | crop80/scale_both | Brier | 0.1826 | 0.1479 | 0.2179 |
| M2 | CrossAttn | crop80/scale_both | Accuracy | 0.7326 | 0.6685 | 0.7965 |
| M2 | CrossAttn | crop80/scale_both | F1 | 0.4250 | 0.2820 | 0.5542 |
| M2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.7925 | 0.6951 | 0.8810 |
| M2 | CrossAttn | crop60/scale_both | AUPRC | 0.3743 | 0.2162 | 0.5994 |
| M2 | CrossAttn | crop60/scale_both | Brier | 0.1899 | 0.1529 | 0.2275 |
| M2 | CrossAttn | crop60/scale_both | Accuracy | 0.7035 | 0.6337 | 0.7733 |
| M2 | CrossAttn | crop60/scale_both | F1 | 0.3855 | 0.2500 | 0.5155 |
| M2 | CrossAttn | norm/scale_both | AUC-ROC | 0.7717 | 0.6846 | 0.8538 |
| M2 | CrossAttn | norm/scale_both | AUPRC | 0.2958 | 0.1750 | 0.4693 |
| M2 | CrossAttn | norm/scale_both | Brier | 0.2351 | 0.1953 | 0.2732 |
| M2 | CrossAttn | norm/scale_both | Accuracy | 0.6802 | 0.6105 | 0.7442 |
| M2 | CrossAttn | norm/scale_both | F1 | 0.3956 | 0.2637 | 0.5192 |
| M2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8109 | 0.7337 | 0.8813 |
| M2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.3325 | 0.1882 | 0.5296 |
| M2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1720 | 0.1410 | 0.2020 |
| M2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.7143 | 0.6429 | 0.7857 |
| M2 | CrossAttn | excl_extreme/scale_both | F1 | 0.3714 | 0.2154 | 0.5080 |
| M2_2 | CrossAttn | len256/scale_both | AUC-ROC | 0.7597 | 0.6658 | 0.8533 |
| M2_2 | CrossAttn | len256/scale_both | AUPRC | 0.2671 | 0.1633 | 0.4483 |
| M2_2 | CrossAttn | len256/scale_both | Brier | 0.1853 | 0.1478 | 0.2248 |
| M2_2 | CrossAttn | len256/scale_both | Accuracy | 0.7500 | 0.6859 | 0.8140 |
| M2_2 | CrossAttn | len256/scale_both | F1 | 0.4110 | 0.2564 | 0.5435 |
| M2_2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.7805 | 0.7006 | 0.8576 |
| M2_2 | CrossAttn | crop80/scale_both | AUPRC | 0.2894 | 0.1692 | 0.4623 |
| M2_2 | CrossAttn | crop80/scale_both | Brier | 0.1904 | 0.1605 | 0.2211 |
| M2_2 | CrossAttn | crop80/scale_both | Accuracy | 0.6977 | 0.6279 | 0.7616 |
| M2_2 | CrossAttn | crop80/scale_both | F1 | 0.3810 | 0.2499 | 0.5071 |
| M2_2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.8155 | 0.7379 | 0.8921 |
| M2_2 | CrossAttn | crop60/scale_both | AUPRC | 0.3119 | 0.2006 | 0.5238 |
| M2_2 | CrossAttn | crop60/scale_both | Brier | 0.2067 | 0.1695 | 0.2451 |
| M2_2 | CrossAttn | crop60/scale_both | Accuracy | 0.6860 | 0.6163 | 0.7558 |
| M2_2 | CrossAttn | crop60/scale_both | F1 | 0.3864 | 0.2564 | 0.5111 |
| M2_2 | CrossAttn | norm/scale_both | AUC-ROC | 0.7985 | 0.7205 | 0.8730 |
| M2_2 | CrossAttn | norm/scale_both | AUPRC | 0.2754 | 0.1784 | 0.4586 |
| M2_2 | CrossAttn | norm/scale_both | Brier | 0.1913 | 0.1578 | 0.2253 |
| M2_2 | CrossAttn | norm/scale_both | Accuracy | 0.7442 | 0.6802 | 0.8081 |
| M2_2 | CrossAttn | norm/scale_both | F1 | 0.4359 | 0.2909 | 0.5647 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.7965 | 0.7070 | 0.8777 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.3465 | 0.2024 | 0.5749 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1885 | 0.1481 | 0.2261 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.7078 | 0.6364 | 0.7792 |
| M2_2 | CrossAttn | excl_extreme/scale_both | F1 | 0.3478 | 0.1999 | 0.4828 |
| M3 | CrossAttn3 | len256/scale_both | AUC-ROC | 0.8027 | 0.6897 | 0.9013 |
| M3 | CrossAttn3 | len256/scale_both | AUPRC | 0.4470 | 0.2535 | 0.6552 |
| M3 | CrossAttn3 | len256/scale_both | Brier | 0.1829 | 0.1479 | 0.2174 |
| M3 | CrossAttn3 | len256/scale_both | Accuracy | 0.7048 | 0.6325 | 0.7711 |
| M3 | CrossAttn3 | len256/scale_both | F1 | 0.3951 | 0.2564 | 0.5275 |
| M3 | CrossAttn3 | crop80/scale_both | AUC-ROC | 0.7627 | 0.6344 | 0.8749 |
| M3 | CrossAttn3 | crop80/scale_both | AUPRC | 0.3384 | 0.1908 | 0.5458 |
| M3 | CrossAttn3 | crop80/scale_both | Brier | 0.1390 | 0.1047 | 0.1719 |
| M3 | CrossAttn3 | crop80/scale_both | Accuracy | 0.7771 | 0.7169 | 0.8373 |
| M3 | CrossAttn3 | crop80/scale_both | F1 | 0.3729 | 0.2083 | 0.5312 |
| M3 | CrossAttn3 | crop60/scale_both | AUC-ROC | 0.8175 | 0.7085 | 0.9108 |
| M3 | CrossAttn3 | crop60/scale_both | AUPRC | 0.4347 | 0.2484 | 0.6398 |
| M3 | CrossAttn3 | crop60/scale_both | Brier | 0.1823 | 0.1471 | 0.2191 |
| M3 | CrossAttn3 | crop60/scale_both | Accuracy | 0.7048 | 0.6325 | 0.7712 |
| M3 | CrossAttn3 | crop60/scale_both | F1 | 0.3951 | 0.2531 | 0.5287 |
| M3 | CrossAttn3 | norm/scale_both | AUC-ROC | 0.7863 | 0.6772 | 0.8793 |
| M3 | CrossAttn3 | norm/scale_both | AUPRC | 0.2941 | 0.1777 | 0.4969 |
| M3 | CrossAttn3 | norm/scale_both | Brier | 0.1945 | 0.1554 | 0.2355 |
| M3 | CrossAttn3 | norm/scale_both | Accuracy | 0.7048 | 0.6325 | 0.7711 |
| M3 | CrossAttn3 | norm/scale_both | F1 | 0.4096 | 0.2683 | 0.5349 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUC-ROC | 0.7434 | 0.6043 | 0.8621 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUPRC | 0.2911 | 0.1402 | 0.4990 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Brier | 0.1462 | 0.1069 | 0.1871 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Accuracy | 0.7770 | 0.7095 | 0.8380 |
| M3 | CrossAttn3 | excl_extreme/scale_both | F1 | 0.3529 | 0.1714 | 0.5117 |

