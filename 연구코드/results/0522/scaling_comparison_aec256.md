# Scaling Comparison — Test Set Performance (AEC 256pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.7223 | 0.2458 | 0.1633 | 0.7768 | 0.3500 |
| M2 | CrossAttn | excl_extreme/scale_both | 0.7151 | 0.3089 | 0.1729 | 0.7225 | 0.3095 |
| M2_2 | CrossAttn | norm/scale_both | 0.7085 | 0.2671 | 0.1800 | 0.7382 | 0.3146 |
| M3 | CrossAttn3 | excl_extreme/scale_both | 0.7042 | 0.3752 | 0.1431 | 0.7799 | 0.3235 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_clinic** | 0.7223 | 0.2458 | 0.1633 | 0.7768 | 0.3500 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_both | 0.7000 | 0.3003 | 0.1501 | 0.7725 | 0.3117 |
| crop80/scale_both | 0.6687 | 0.2738 | 0.2256 | 0.6524 | 0.2703 |
| crop60/scale_both | 0.6867 | 0.2492 | 0.1702 | 0.7597 | 0.2821 |
| norm/scale_both | 0.6888 | 0.2641 | 0.1775 | 0.7339 | 0.3111 |
| **excl_extreme/scale_both** | 0.7151 | 0.3089 | 0.1729 | 0.7225 | 0.3095 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_both | 0.6815 | 0.1845 | 0.2062 | 0.6867 | 0.2913 |
| crop80/scale_both | 0.6842 | 0.2098 | 0.2315 | 0.6567 | 0.2727 |
| crop60/scale_both | 0.6831 | 0.2175 | 0.2246 | 0.6094 | 0.2479 |
| **norm/scale_both** | 0.7085 | 0.2671 | 0.1800 | 0.7382 | 0.3146 |
| excl_extreme/scale_both | 0.6847 | 0.1834 | 0.2172 | 0.6459 | 0.2128 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_both | 0.6925 | 0.2459 | 0.2182 | 0.6652 | 0.2778 |
| crop80/scale_both | 0.6913 | 0.3122 | 0.1525 | 0.7983 | 0.3380 |
| crop60/scale_both | 0.6844 | 0.3180 | 0.1632 | 0.7554 | 0.2597 |
| norm/scale_both | 0.6967 | 0.2892 | 0.2122 | 0.6695 | 0.2804 |
| **excl_extreme/scale_both** | 0.7042 | 0.3752 | 0.1431 | 0.7799 | 0.3235 |

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
| AUC-ROC † | 0.8378 | 0.8612 | +0.0234 | -2.136 | 9.96e-02 | 1.25e-01 |
| AUPRC  | 0.4431 | 0.4592 | +0.0161 | -0.393 | 7.14e-01 | 1.00e+00 |
| Brier  | 0.1696 | 0.1724 | +0.0027 | -0.566 | 6.01e-01 | 6.25e-01 |
| Accuracy † | 0.7481 | 0.6996 | -0.0485 | 2.670 | 5.58e-02 | 6.25e-02 |
| F1  | 0.3985 | 0.3837 | -0.0148 | 0.673 | 5.38e-01 | 6.25e-01 |

### crop80/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8621 | +0.0244 | -2.097 | 1.04e-01 | 1.25e-01 |
| AUPRC  | 0.4431 | 0.4487 | +0.0056 | -0.235 | 8.26e-01 | 6.25e-01 |
| Brier  | 0.1696 | 0.1586 | -0.0110 | 0.549 | 6.12e-01 | 6.25e-01 |
| Accuracy  | 0.7481 | 0.7372 | -0.0109 | 0.216 | 8.40e-01 | 1.00e+00 |
| F1  | 0.3985 | 0.4161 | +0.0175 | -0.467 | 6.65e-01 | 1.00e+00 |

### crop60/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8562 | +0.0184 | -1.791 | 1.48e-01 | 1.88e-01 |
| AUPRC  | 0.4431 | 0.4784 | +0.0353 | -0.909 | 4.15e-01 | 8.12e-01 |
| Brier  | 0.1696 | 0.1747 | +0.0051 | -0.250 | 8.15e-01 | 1.00e+00 |
| Accuracy  | 0.7481 | 0.7244 | -0.0238 | 0.525 | 6.27e-01 | 6.88e-01 |
| F1  | 0.3985 | 0.3992 | +0.0007 | -0.023 | 9.83e-01 | 8.12e-01 |

### norm/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8591 | +0.0213 | -1.593 | 1.86e-01 | 1.88e-01 |
| AUPRC  | 0.4431 | 0.4080 | -0.0351 | 1.196 | 2.98e-01 | 3.12e-01 |
| Brier  | 0.1696 | 0.1710 | +0.0014 | -0.114 | 9.15e-01 | 1.00e+00 |
| Accuracy  | 0.7481 | 0.7395 | -0.0086 | 0.355 | 7.40e-01 | 1.00e+00 |
| F1  | 0.3985 | 0.4294 | +0.0308 | -1.443 | 2.23e-01 | 4.38e-01 |

### excl_extreme/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8544 | +0.0166 | -1.112 | 3.29e-01 | 3.12e-01 |
| AUPRC  | 0.4431 | 0.4382 | -0.0050 | 0.089 | 9.34e-01 | 1.00e+00 |
| Brier  | 0.1696 | 0.1728 | +0.0032 | -0.196 | 8.54e-01 | 1.00e+00 |
| Accuracy  | 0.7481 | 0.7042 | -0.0440 | 1.622 | 1.80e-01 | 3.12e-01 |
| F1 ** | 0.3985 | 0.3492 | -0.0493 | 4.680 | 9.44e-03 | 6.25e-02 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len256/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8612 | 0.8507 | -0.0105 | 2.631 | 5.81e-02 | 1.25e-01 |
| AUPRC * | 0.4592 | 0.4031 | -0.0561 | 2.958 | 4.16e-02 | 6.25e-02 |
| Brier * | 0.1724 | 0.1469 | -0.0255 | 3.018 | 3.92e-02 | 1.25e-01 |
| Accuracy ** | 0.6996 | 0.7728 | +0.0732 | -6.445 | 2.98e-03 | 6.25e-02 |
| F1 * | 0.3837 | 0.4273 | +0.0436 | -4.485 | 1.09e-02 | 6.25e-02 |

#### Case: crop80/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8621 | 0.8512 | -0.0110 | 1.219 | 2.90e-01 | 3.75e-01 |
| AUPRC * | 0.4487 | 0.4116 | -0.0371 | 3.884 | 1.78e-02 | 6.25e-02 |
| Brier † | 0.1586 | 0.1826 | +0.0240 | -2.739 | 5.19e-02 | 1.25e-01 |
| Accuracy  | 0.7372 | 0.7189 | -0.0183 | 1.118 | 3.26e-01 | 4.38e-01 |
| F1  | 0.4161 | 0.3912 | -0.0248 | 1.809 | 1.45e-01 | 6.25e-02 |

#### Case: crop60/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8562 | 0.8490 | -0.0072 | 1.213 | 2.92e-01 | 5.00e-01 |
| AUPRC * | 0.4784 | 0.4224 | -0.0561 | 2.864 | 4.57e-02 | 1.25e-01 |
| Brier  | 0.1747 | 0.1694 | -0.0053 | 0.838 | 4.49e-01 | 8.12e-01 |
| Accuracy  | 0.7244 | 0.7437 | +0.0194 | -1.021 | 3.65e-01 | 4.38e-01 |
| F1  | 0.3992 | 0.4085 | +0.0093 | -0.874 | 4.31e-01 | 6.25e-01 |

#### Case: norm/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8591 | 0.8612 | +0.0021 | -0.268 | 8.02e-01 | 1.00e+00 |
| AUPRC  | 0.4080 | 0.4106 | +0.0026 | -0.107 | 9.20e-01 | 1.00e+00 |
| Brier  | 0.1710 | 0.1764 | +0.0054 | -0.344 | 7.48e-01 | 6.25e-01 |
| Accuracy  | 0.7395 | 0.7288 | -0.0108 | 0.552 | 6.10e-01 | 1.00e+00 |
| F1  | 0.4294 | 0.4064 | -0.0229 | 1.781 | 1.49e-01 | 3.12e-01 |

#### Case: excl_extreme/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8544 | 0.8512 | -0.0032 | 0.223 | 8.34e-01 | 8.12e-01 |
| AUPRC  | 0.4382 | 0.4184 | -0.0198 | 0.562 | 6.04e-01 | 6.25e-01 |
| Brier  | 0.1728 | 0.1926 | +0.0198 | -0.782 | 4.78e-01 | 6.25e-01 |
| Accuracy  | 0.7042 | 0.6838 | -0.0204 | 0.323 | 7.63e-01 | 1.00e+00 |
| F1  | 0.3492 | 0.3693 | +0.0201 | -0.682 | 5.33e-01 | 4.38e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len256/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8507 | +0.0129 | -1.759 | 1.53e-01 | 1.88e-01 |
| AUPRC  | 0.4431 | 0.4031 | -0.0401 | 1.481 | 2.13e-01 | 3.12e-01 |
| Brier † | 0.1696 | 0.1469 | -0.0227 | 2.132 | 1.00e-01 | 1.88e-01 |
| Accuracy  | 0.7481 | 0.7728 | +0.0247 | -0.950 | 3.96e-01 | 6.25e-01 |
| F1  | 0.3985 | 0.4273 | +0.0288 | -1.005 | 3.72e-01 | 6.25e-01 |

### crop80/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8512 | +0.0134 | -1.735 | 1.58e-01 | 1.88e-01 |
| AUPRC  | 0.4431 | 0.4116 | -0.0315 | 1.400 | 2.34e-01 | 3.12e-01 |
| Brier  | 0.1696 | 0.1826 | +0.0130 | -0.688 | 5.29e-01 | 6.25e-01 |
| Accuracy  | 0.7481 | 0.7189 | -0.0292 | 0.687 | 5.30e-01 | 6.25e-01 |
| F1  | 0.3985 | 0.3912 | -0.0073 | 0.272 | 7.99e-01 | 8.12e-01 |

### crop60/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8490 | +0.0113 | -1.199 | 2.97e-01 | 4.38e-01 |
| AUPRC  | 0.4431 | 0.4224 | -0.0208 | 0.530 | 6.24e-01 | 6.25e-01 |
| Brier  | 0.1696 | 0.1694 | -0.0002 | 0.014 | 9.89e-01 | 1.00e+00 |
| Accuracy  | 0.7481 | 0.7437 | -0.0044 | 0.100 | 9.25e-01 | 1.00e+00 |
| F1  | 0.3985 | 0.4085 | +0.0100 | -0.279 | 7.94e-01 | 1.00e+00 |

### norm/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8378 | 0.8612 | +0.0235 | -3.380 | 2.78e-02 | 6.25e-02 |
| AUPRC  | 0.4431 | 0.4106 | -0.0326 | 1.850 | 1.38e-01 | 3.12e-01 |
| Brier  | 0.1696 | 0.1764 | +0.0068 | -0.508 | 6.38e-01 | 6.25e-01 |
| Accuracy  | 0.7481 | 0.7288 | -0.0194 | 0.795 | 4.71e-01 | 4.38e-01 |
| F1  | 0.3985 | 0.4064 | +0.0079 | -0.393 | 7.14e-01 | 8.12e-01 |

### excl_extreme/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8512 | +0.0134 | -2.065 | 1.08e-01 | 1.88e-01 |
| AUPRC  | 0.4431 | 0.4184 | -0.0248 | 0.388 | 7.18e-01 | 1.00e+00 |
| Brier  | 0.1696 | 0.1926 | +0.0229 | -0.593 | 5.85e-01 | 8.12e-01 |
| Accuracy  | 0.7481 | 0.6838 | -0.0643 | 0.752 | 4.94e-01 | 1.00e+00 |
| F1  | 0.3985 | 0.3693 | -0.0292 | 0.751 | 4.94e-01 | 6.25e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_clinic | AUC-ROC | 0.7223 | 0.5979 | 0.8260 |
| M1 | LR | scale_clinic | AUPRC | 0.2458 | 0.1433 | 0.4183 |
| M1 | LR | scale_clinic | Brier | 0.1633 | 0.1369 | 0.1926 |
| M1 | LR | scale_clinic | Accuracy | 0.7768 | 0.7210 | 0.8283 |
| M1 | LR | scale_clinic | F1 | 0.3500 | 0.2051 | 0.4750 |
| M2 | CrossAttn | len256/scale_both | AUC-ROC | 0.7000 | 0.5731 | 0.8134 |
| M2 | CrossAttn | len256/scale_both | AUPRC | 0.3003 | 0.1497 | 0.4845 |
| M2 | CrossAttn | len256/scale_both | Brier | 0.1501 | 0.1185 | 0.1835 |
| M2 | CrossAttn | len256/scale_both | Accuracy | 0.7725 | 0.7167 | 0.8240 |
| M2 | CrossAttn | len256/scale_both | F1 | 0.3117 | 0.1690 | 0.4390 |
| M2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.6687 | 0.5383 | 0.7822 |
| M2 | CrossAttn | crop80/scale_both | AUPRC | 0.2738 | 0.1292 | 0.4530 |
| M2 | CrossAttn | crop80/scale_both | Brier | 0.2256 | 0.1912 | 0.2634 |
| M2 | CrossAttn | crop80/scale_both | Accuracy | 0.6524 | 0.5880 | 0.7124 |
| M2 | CrossAttn | crop80/scale_both | F1 | 0.2703 | 0.1553 | 0.3726 |
| M2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.6867 | 0.5681 | 0.7911 |
| M2 | CrossAttn | crop60/scale_both | AUPRC | 0.2492 | 0.1227 | 0.4273 |
| M2 | CrossAttn | crop60/scale_both | Brier | 0.1702 | 0.1374 | 0.2047 |
| M2 | CrossAttn | crop60/scale_both | Accuracy | 0.7597 | 0.7039 | 0.8113 |
| M2 | CrossAttn | crop60/scale_both | F1 | 0.2821 | 0.1463 | 0.4063 |
| M2 | CrossAttn | norm/scale_both | AUC-ROC | 0.6888 | 0.5609 | 0.7997 |
| M2 | CrossAttn | norm/scale_both | AUPRC | 0.2641 | 0.1378 | 0.4405 |
| M2 | CrossAttn | norm/scale_both | Brier | 0.1775 | 0.1455 | 0.2152 |
| M2 | CrossAttn | norm/scale_both | Accuracy | 0.7339 | 0.6738 | 0.7897 |
| M2 | CrossAttn | norm/scale_both | F1 | 0.3111 | 0.1795 | 0.4286 |
| M2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.7151 | 0.5849 | 0.8320 |
| M2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.3089 | 0.1523 | 0.5043 |
| M2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1729 | 0.1389 | 0.2063 |
| M2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.7225 | 0.6555 | 0.7847 |
| M2 | CrossAttn | excl_extreme/scale_both | F1 | 0.3095 | 0.1791 | 0.4396 |
| M2_2 | CrossAttn | len256/scale_both | AUC-ROC | 0.6815 | 0.5672 | 0.7825 |
| M2_2 | CrossAttn | len256/scale_both | AUPRC | 0.1845 | 0.1104 | 0.3096 |
| M2_2 | CrossAttn | len256/scale_both | Brier | 0.2062 | 0.1744 | 0.2396 |
| M2_2 | CrossAttn | len256/scale_both | Accuracy | 0.6867 | 0.6223 | 0.7425 |
| M2_2 | CrossAttn | len256/scale_both | F1 | 0.2913 | 0.1702 | 0.4000 |
| M2_2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.6842 | 0.5608 | 0.7874 |
| M2_2 | CrossAttn | crop80/scale_both | AUPRC | 0.2098 | 0.1186 | 0.3613 |
| M2_2 | CrossAttn | crop80/scale_both | Brier | 0.2315 | 0.1942 | 0.2708 |
| M2_2 | CrossAttn | crop80/scale_both | Accuracy | 0.6567 | 0.5923 | 0.7167 |
| M2_2 | CrossAttn | crop80/scale_both | F1 | 0.2727 | 0.1584 | 0.3750 |
| M2_2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.6831 | 0.5719 | 0.7850 |
| M2_2 | CrossAttn | crop60/scale_both | AUPRC | 0.2175 | 0.1149 | 0.3820 |
| M2_2 | CrossAttn | crop60/scale_both | Brier | 0.2246 | 0.1913 | 0.2572 |
| M2_2 | CrossAttn | crop60/scale_both | Accuracy | 0.6094 | 0.5451 | 0.6695 |
| M2_2 | CrossAttn | crop60/scale_both | F1 | 0.2479 | 0.1416 | 0.3408 |
| M2_2 | CrossAttn | norm/scale_both | AUC-ROC | 0.7085 | 0.5802 | 0.8177 |
| M2_2 | CrossAttn | norm/scale_both | AUPRC | 0.2671 | 0.1379 | 0.4358 |
| M2_2 | CrossAttn | norm/scale_both | Brier | 0.1800 | 0.1497 | 0.2152 |
| M2_2 | CrossAttn | norm/scale_both | Accuracy | 0.7382 | 0.6738 | 0.7940 |
| M2_2 | CrossAttn | norm/scale_both | F1 | 0.3146 | 0.1794 | 0.4318 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.6847 | 0.5754 | 0.7880 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.1834 | 0.1042 | 0.3399 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Brier | 0.2172 | 0.1816 | 0.2534 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.6459 | 0.5837 | 0.7081 |
| M2_2 | CrossAttn | excl_extreme/scale_both | F1 | 0.2128 | 0.0988 | 0.3186 |
| M3 | CrossAttn3 | len256/scale_both | AUC-ROC | 0.6925 | 0.5604 | 0.8069 |
| M3 | CrossAttn3 | len256/scale_both | AUPRC | 0.2459 | 0.1356 | 0.4257 |
| M3 | CrossAttn3 | len256/scale_both | Brier | 0.2182 | 0.1831 | 0.2549 |
| M3 | CrossAttn3 | len256/scale_both | Accuracy | 0.6652 | 0.6009 | 0.7253 |
| M3 | CrossAttn3 | len256/scale_both | F1 | 0.2778 | 0.1616 | 0.3802 |
| M3 | CrossAttn3 | crop80/scale_both | AUC-ROC | 0.6913 | 0.5645 | 0.8033 |
| M3 | CrossAttn3 | crop80/scale_both | AUPRC | 0.3122 | 0.1513 | 0.4867 |
| M3 | CrossAttn3 | crop80/scale_both | Brier | 0.1525 | 0.1229 | 0.1856 |
| M3 | CrossAttn3 | crop80/scale_both | Accuracy | 0.7983 | 0.7468 | 0.8455 |
| M3 | CrossAttn3 | crop80/scale_both | F1 | 0.3380 | 0.1818 | 0.4691 |
| M3 | CrossAttn3 | crop60/scale_both | AUC-ROC | 0.6844 | 0.5626 | 0.7956 |
| M3 | CrossAttn3 | crop60/scale_both | AUPRC | 0.3180 | 0.1555 | 0.4891 |
| M3 | CrossAttn3 | crop60/scale_both | Brier | 0.1632 | 0.1312 | 0.1975 |
| M3 | CrossAttn3 | crop60/scale_both | Accuracy | 0.7554 | 0.6995 | 0.8069 |
| M3 | CrossAttn3 | crop60/scale_both | F1 | 0.2597 | 0.1290 | 0.3871 |
| M3 | CrossAttn3 | norm/scale_both | AUC-ROC | 0.6967 | 0.5659 | 0.8129 |
| M3 | CrossAttn3 | norm/scale_both | AUPRC | 0.2892 | 0.1507 | 0.4656 |
| M3 | CrossAttn3 | norm/scale_both | Brier | 0.2122 | 0.1804 | 0.2469 |
| M3 | CrossAttn3 | norm/scale_both | Accuracy | 0.6695 | 0.6093 | 0.7254 |
| M3 | CrossAttn3 | norm/scale_both | F1 | 0.2804 | 0.1633 | 0.3846 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUC-ROC | 0.7042 | 0.5615 | 0.8318 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUPRC | 0.3752 | 0.1947 | 0.5869 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Brier | 0.1431 | 0.1109 | 0.1752 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Accuracy | 0.7799 | 0.7225 | 0.8373 |
| M3 | CrossAttn3 | excl_extreme/scale_both | F1 | 0.3235 | 0.1724 | 0.4706 |

