# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.7223 | 0.2458 | 0.1633 | 0.7768 | 0.3500 |
| M2 | CrossAttn | excl_extreme/scale_both | 0.7205 | 0.3372 | 0.1493 | 0.7847 | 0.3662 |
| M2_2 | CrossAttn | len128/scale_both | 0.6983 | 0.1939 | 0.1849 | 0.7039 | 0.3030 |
| M3 | CrossAttn3 | excl_extreme/scale_both | 0.7149 | 0.3065 | 0.1742 | 0.7368 | 0.3373 |

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
| len128/scale_both | 0.6896 | 0.2479 | 0.1784 | 0.7382 | 0.3146 |
| crop80/scale_both | 0.6794 | 0.2995 | 0.1990 | 0.7167 | 0.3125 |
| crop60/scale_both | 0.6871 | 0.3192 | 0.1877 | 0.7210 | 0.3011 |
| norm/scale_both | 0.6804 | 0.2180 | 0.1836 | 0.7210 | 0.2697 |
| **excl_extreme/scale_both** | 0.7205 | 0.3372 | 0.1493 | 0.7847 | 0.3662 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **len128/scale_both** | 0.6983 | 0.1939 | 0.1849 | 0.7039 | 0.3030 |
| crop80/scale_both | 0.6929 | 0.2118 | 0.1816 | 0.7382 | 0.2824 |
| crop60/scale_both | 0.6885 | 0.2125 | 0.2090 | 0.6695 | 0.2667 |
| norm/scale_both | 0.6635 | 0.2158 | 0.1674 | 0.7811 | 0.3014 |
| excl_extreme/scale_both | 0.6925 | 0.2091 | 0.2056 | 0.6507 | 0.2474 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.6773 | 0.2978 | 0.1781 | 0.7382 | 0.2989 |
| crop80/scale_both | 0.6937 | 0.3190 | 0.2070 | 0.6824 | 0.3019 |
| crop60/scale_both | 0.6940 | 0.3102 | 0.1597 | 0.7639 | 0.3038 |
| norm/scale_both | 0.6971 | 0.2715 | 0.1784 | 0.7425 | 0.3182 |
| **excl_extreme/scale_both** | 0.7149 | 0.3065 | 0.1742 | 0.7368 | 0.3373 |

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
| AUC-ROC † | 0.8378 | 0.8652 | +0.0274 | -2.599 | 6.01e-02 | 1.25e-01 |
| AUPRC  | 0.4431 | 0.4572 | +0.0140 | -0.629 | 5.64e-01 | 4.38e-01 |
| Brier  | 0.1696 | 0.1700 | +0.0004 | -0.047 | 9.65e-01 | 1.00e+00 |
| Accuracy  | 0.7481 | 0.7351 | -0.0130 | 0.459 | 6.70e-01 | 8.75e-01 |
| F1  | 0.3985 | 0.4211 | +0.0226 | -1.242 | 2.82e-01 | 4.38e-01 |

### crop80/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8556 | +0.0179 | -1.840 | 1.40e-01 | 1.88e-01 |
| AUPRC  | 0.4431 | 0.4230 | -0.0202 | 0.816 | 4.60e-01 | 4.38e-01 |
| Brier  | 0.1696 | 0.1737 | +0.0041 | -0.934 | 4.03e-01 | 8.12e-01 |
| Accuracy  | 0.7481 | 0.7223 | -0.0259 | 1.797 | 1.47e-01 | 1.88e-01 |
| F1  | 0.3985 | 0.3902 | -0.0083 | 0.449 | 6.77e-01 | 8.12e-01 |

### crop60/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8378 | 0.8624 | +0.0246 | -2.501 | 6.67e-02 | 1.25e-01 |
| AUPRC  | 0.4431 | 0.4334 | -0.0097 | 0.369 | 7.31e-01 | 8.12e-01 |
| Brier  | 0.1696 | 0.1822 | +0.0126 | -0.770 | 4.84e-01 | 8.12e-01 |
| Accuracy  | 0.7481 | 0.7114 | -0.0367 | 1.055 | 3.51e-01 | 6.25e-01 |
| F1  | 0.3985 | 0.3955 | -0.0030 | 0.116 | 9.13e-01 | 1.00e+00 |

### norm/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8593 | +0.0215 | -1.836 | 1.40e-01 | 1.88e-01 |
| AUPRC  | 0.4431 | 0.4177 | -0.0254 | 1.871 | 1.35e-01 | 1.88e-01 |
| Brier  | 0.1696 | 0.1844 | +0.0148 | -0.905 | 4.17e-01 | 6.25e-01 |
| Accuracy  | 0.7481 | 0.7028 | -0.0453 | 1.364 | 2.44e-01 | 4.38e-01 |
| F1  | 0.3985 | 0.3930 | -0.0055 | 0.258 | 8.09e-01 | 1.00e+00 |

### excl_extreme/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8547 | +0.0170 | -1.187 | 3.01e-01 | 3.12e-01 |
| AUPRC  | 0.4431 | 0.4195 | -0.0236 | 0.423 | 6.94e-01 | 8.12e-01 |
| Brier  | 0.1696 | 0.1881 | +0.0184 | -1.387 | 2.38e-01 | 3.12e-01 |
| Accuracy † | 0.7481 | 0.6934 | -0.0547 | 2.593 | 6.05e-02 | 1.25e-01 |
| F1 † | 0.3985 | 0.3667 | -0.0318 | 2.519 | 6.54e-02 | 1.25e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len128/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8652 | 0.8491 | -0.0160 | 1.810 | 1.44e-01 | 1.88e-01 |
| AUPRC † | 0.4572 | 0.4100 | -0.0472 | 2.395 | 7.48e-02 | 1.25e-01 |
| Brier  | 0.1700 | 0.1583 | -0.0117 | 1.062 | 3.48e-01 | 4.38e-01 |
| Accuracy  | 0.7351 | 0.7599 | +0.0248 | -1.986 | 1.18e-01 | 1.88e-01 |
| F1  | 0.4211 | 0.4102 | -0.0109 | 0.743 | 4.99e-01 | 6.25e-01 |

#### Case: crop80/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8556 | 0.8498 | -0.0059 | 1.028 | 3.62e-01 | 4.38e-01 |
| AUPRC  | 0.4230 | 0.3979 | -0.0251 | 1.264 | 2.75e-01 | 4.38e-01 |
| Brier  | 0.1737 | 0.1556 | -0.0181 | 1.407 | 2.32e-01 | 4.38e-01 |
| Accuracy † | 0.7223 | 0.7686 | +0.0463 | -2.162 | 9.67e-02 | 1.25e-01 |
| F1  | 0.3902 | 0.4089 | +0.0187 | -0.935 | 4.03e-01 | 4.38e-01 |

#### Case: crop60/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8624 | 0.8418 | -0.0206 | 2.698 | 5.42e-02 | 1.25e-01 |
| AUPRC  | 0.4334 | 0.4234 | -0.0100 | 0.344 | 7.48e-01 | 1.00e+00 |
| Brier  | 0.1822 | 0.1526 | -0.0296 | 1.628 | 1.79e-01 | 1.88e-01 |
| Accuracy  | 0.7114 | 0.7697 | +0.0582 | -1.579 | 1.89e-01 | 1.88e-01 |
| F1  | 0.3955 | 0.4090 | +0.0134 | -0.455 | 6.73e-01 | 4.38e-01 |

#### Case: norm/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8593 | 0.8590 | -0.0002 | 0.038 | 9.72e-01 | 1.00e+00 |
| AUPRC  | 0.4177 | 0.4129 | -0.0048 | 0.419 | 6.97e-01 | 8.12e-01 |
| Brier  | 0.1844 | 0.1607 | -0.0237 | 1.228 | 2.87e-01 | 3.12e-01 |
| Accuracy  | 0.7028 | 0.7536 | +0.0507 | -1.253 | 2.79e-01 | 3.75e-01 |
| F1  | 0.3930 | 0.4238 | +0.0308 | -1.542 | 1.98e-01 | 3.12e-01 |

#### Case: excl_extreme/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8547 | 0.8466 | -0.0081 | 1.596 | 1.86e-01 | 3.12e-01 |
| AUPRC  | 0.4195 | 0.4217 | +0.0022 | -0.060 | 9.55e-01 | 1.00e+00 |
| Brier  | 0.1881 | 0.1748 | -0.0133 | 0.566 | 6.02e-01 | 6.25e-01 |
| Accuracy  | 0.6934 | 0.7257 | +0.0323 | -0.724 | 5.09e-01 | 6.25e-01 |
| F1  | 0.3667 | 0.3725 | +0.0058 | -0.189 | 8.60e-01 | 1.00e+00 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8491 | +0.0114 | -1.710 | 1.63e-01 | 1.25e-01 |
| AUPRC  | 0.4431 | 0.4100 | -0.0332 | 1.659 | 1.72e-01 | 3.12e-01 |
| Brier  | 0.1696 | 0.1583 | -0.0113 | 0.898 | 4.20e-01 | 6.25e-01 |
| Accuracy  | 0.7481 | 0.7599 | +0.0118 | -0.542 | 6.16e-01 | 6.25e-01 |
| F1  | 0.3985 | 0.4102 | +0.0117 | -0.804 | 4.67e-01 | 4.38e-01 |

### crop80/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8498 | +0.0120 | -1.722 | 1.60e-01 | 1.88e-01 |
| AUPRC  | 0.4431 | 0.3979 | -0.0452 | 1.263 | 2.75e-01 | 3.12e-01 |
| Brier  | 0.1696 | 0.1556 | -0.0141 | 1.044 | 3.55e-01 | 3.12e-01 |
| Accuracy  | 0.7481 | 0.7686 | +0.0204 | -1.080 | 3.41e-01 | 4.38e-01 |
| F1  | 0.3985 | 0.4089 | +0.0104 | -0.567 | 6.01e-01 | 4.38e-01 |

### crop60/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8418 | +0.0041 | -0.451 | 6.75e-01 | 1.00e+00 |
| AUPRC  | 0.4431 | 0.4234 | -0.0198 | 0.569 | 6.00e-01 | 8.12e-01 |
| Brier † | 0.1696 | 0.1526 | -0.0170 | 2.543 | 6.38e-02 | 1.25e-01 |
| Accuracy  | 0.7481 | 0.7697 | +0.0215 | -0.935 | 4.03e-01 | 6.25e-01 |
| F1  | 0.3985 | 0.4090 | +0.0104 | -0.847 | 4.45e-01 | 4.38e-01 |

### norm/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8378 | 0.8590 | +0.0213 | -2.681 | 5.52e-02 | 1.25e-01 |
| AUPRC  | 0.4431 | 0.4129 | -0.0303 | 1.541 | 1.98e-01 | 3.12e-01 |
| Brier  | 0.1696 | 0.1607 | -0.0089 | 0.512 | 6.35e-01 | 8.12e-01 |
| Accuracy  | 0.7481 | 0.7536 | +0.0054 | -0.148 | 8.89e-01 | 1.00e+00 |
| F1  | 0.3985 | 0.4238 | +0.0253 | -0.892 | 4.23e-01 | 4.38e-01 |

### excl_extreme/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8378 | 0.8466 | +0.0089 | -0.839 | 4.48e-01 | 6.25e-01 |
| AUPRC  | 0.4431 | 0.4217 | -0.0215 | 0.944 | 3.99e-01 | 3.12e-01 |
| Brier  | 0.1696 | 0.1748 | +0.0052 | -0.233 | 8.27e-01 | 6.25e-01 |
| Accuracy  | 0.7481 | 0.7257 | -0.0224 | 0.421 | 6.95e-01 | 6.25e-01 |
| F1  | 0.3985 | 0.3725 | -0.0260 | 0.702 | 5.21e-01 | 4.38e-01 |

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
| M2 | CrossAttn | len128/scale_both | AUC-ROC | 0.6896 | 0.5652 | 0.8005 |
| M2 | CrossAttn | len128/scale_both | AUPRC | 0.2479 | 0.1303 | 0.4324 |
| M2 | CrossAttn | len128/scale_both | Brier | 0.1784 | 0.1473 | 0.2135 |
| M2 | CrossAttn | len128/scale_both | Accuracy | 0.7382 | 0.6781 | 0.7897 |
| M2 | CrossAttn | len128/scale_both | F1 | 0.3146 | 0.1818 | 0.4330 |
| M2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.6794 | 0.5554 | 0.7852 |
| M2 | CrossAttn | crop80/scale_both | AUPRC | 0.2995 | 0.1399 | 0.4763 |
| M2 | CrossAttn | crop80/scale_both | Brier | 0.1990 | 0.1667 | 0.2354 |
| M2 | CrossAttn | crop80/scale_both | Accuracy | 0.7167 | 0.6567 | 0.7725 |
| M2 | CrossAttn | crop80/scale_both | F1 | 0.3125 | 0.1882 | 0.4237 |
| M2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.6871 | 0.5535 | 0.7993 |
| M2 | CrossAttn | crop60/scale_both | AUPRC | 0.3192 | 0.1497 | 0.5025 |
| M2 | CrossAttn | crop60/scale_both | Brier | 0.1877 | 0.1539 | 0.2251 |
| M2 | CrossAttn | crop60/scale_both | Accuracy | 0.7210 | 0.6609 | 0.7768 |
| M2 | CrossAttn | crop60/scale_both | F1 | 0.3011 | 0.1628 | 0.4130 |
| M2 | CrossAttn | norm/scale_both | AUC-ROC | 0.6804 | 0.5607 | 0.7853 |
| M2 | CrossAttn | norm/scale_both | AUPRC | 0.2180 | 0.1193 | 0.3898 |
| M2 | CrossAttn | norm/scale_both | Brier | 0.1836 | 0.1512 | 0.2200 |
| M2 | CrossAttn | norm/scale_both | Accuracy | 0.7210 | 0.6609 | 0.7768 |
| M2 | CrossAttn | norm/scale_both | F1 | 0.2697 | 0.1386 | 0.3838 |
| M2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.7205 | 0.5852 | 0.8427 |
| M2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.3372 | 0.1743 | 0.5324 |
| M2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1493 | 0.1181 | 0.1823 |
| M2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.7847 | 0.7273 | 0.8373 |
| M2 | CrossAttn | excl_extreme/scale_both | F1 | 0.3662 | 0.2181 | 0.5000 |
| M2_2 | CrossAttn | len128/scale_both | AUC-ROC | 0.6983 | 0.5834 | 0.7962 |
| M2_2 | CrossAttn | len128/scale_both | AUPRC | 0.1939 | 0.1199 | 0.3159 |
| M2_2 | CrossAttn | len128/scale_both | Brier | 0.1849 | 0.1547 | 0.2174 |
| M2_2 | CrossAttn | len128/scale_both | Accuracy | 0.7039 | 0.6438 | 0.7639 |
| M2_2 | CrossAttn | len128/scale_both | F1 | 0.3030 | 0.1778 | 0.4138 |
| M2_2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.6929 | 0.5698 | 0.7943 |
| M2_2 | CrossAttn | crop80/scale_both | AUPRC | 0.2118 | 0.1212 | 0.3674 |
| M2_2 | CrossAttn | crop80/scale_both | Brier | 0.1816 | 0.1478 | 0.2185 |
| M2_2 | CrossAttn | crop80/scale_both | Accuracy | 0.7382 | 0.6781 | 0.7940 |
| M2_2 | CrossAttn | crop80/scale_both | F1 | 0.2824 | 0.1463 | 0.4000 |
| M2_2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.6885 | 0.5782 | 0.7809 |
| M2_2 | CrossAttn | crop60/scale_both | AUPRC | 0.2125 | 0.1200 | 0.3801 |
| M2_2 | CrossAttn | crop60/scale_both | Brier | 0.2090 | 0.1766 | 0.2436 |
| M2_2 | CrossAttn | crop60/scale_both | Accuracy | 0.6695 | 0.6052 | 0.7296 |
| M2_2 | CrossAttn | crop60/scale_both | F1 | 0.2667 | 0.1468 | 0.3704 |
| M2_2 | CrossAttn | norm/scale_both | AUC-ROC | 0.6635 | 0.5383 | 0.7767 |
| M2_2 | CrossAttn | norm/scale_both | AUPRC | 0.2158 | 0.1180 | 0.3946 |
| M2_2 | CrossAttn | norm/scale_both | Brier | 0.1674 | 0.1316 | 0.2058 |
| M2_2 | CrossAttn | norm/scale_both | Accuracy | 0.7811 | 0.7253 | 0.8369 |
| M2_2 | CrossAttn | norm/scale_both | F1 | 0.3014 | 0.1519 | 0.4270 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.6925 | 0.5754 | 0.7989 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.2091 | 0.1167 | 0.3848 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Brier | 0.2056 | 0.1722 | 0.2416 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.6507 | 0.5837 | 0.7129 |
| M2_2 | CrossAttn | excl_extreme/scale_both | F1 | 0.2474 | 0.1235 | 0.3619 |
| M3 | CrossAttn3 | len128/scale_both | AUC-ROC | 0.6773 | 0.5434 | 0.7947 |
| M3 | CrossAttn3 | len128/scale_both | AUPRC | 0.2978 | 0.1402 | 0.4725 |
| M3 | CrossAttn3 | len128/scale_both | Brier | 0.1781 | 0.1428 | 0.2140 |
| M3 | CrossAttn3 | len128/scale_both | Accuracy | 0.7382 | 0.6781 | 0.7897 |
| M3 | CrossAttn3 | len128/scale_both | F1 | 0.2989 | 0.1643 | 0.4158 |
| M3 | CrossAttn3 | crop80/scale_both | AUC-ROC | 0.6937 | 0.5617 | 0.8082 |
| M3 | CrossAttn3 | crop80/scale_both | AUPRC | 0.3190 | 0.1550 | 0.5017 |
| M3 | CrossAttn3 | crop80/scale_both | Brier | 0.2070 | 0.1729 | 0.2444 |
| M3 | CrossAttn3 | crop80/scale_both | Accuracy | 0.6824 | 0.6223 | 0.7382 |
| M3 | CrossAttn3 | crop80/scale_both | F1 | 0.3019 | 0.1818 | 0.4039 |
| M3 | CrossAttn3 | crop60/scale_both | AUC-ROC | 0.6940 | 0.5688 | 0.8034 |
| M3 | CrossAttn3 | crop60/scale_both | AUPRC | 0.3102 | 0.1495 | 0.5039 |
| M3 | CrossAttn3 | crop60/scale_both | Brier | 0.1597 | 0.1280 | 0.1936 |
| M3 | CrossAttn3 | crop60/scale_both | Accuracy | 0.7639 | 0.7082 | 0.8155 |
| M3 | CrossAttn3 | crop60/scale_both | F1 | 0.3038 | 0.1647 | 0.4304 |
| M3 | CrossAttn3 | norm/scale_both | AUC-ROC | 0.6971 | 0.5704 | 0.8098 |
| M3 | CrossAttn3 | norm/scale_both | AUPRC | 0.2715 | 0.1480 | 0.4652 |
| M3 | CrossAttn3 | norm/scale_both | Brier | 0.1784 | 0.1478 | 0.2116 |
| M3 | CrossAttn3 | norm/scale_both | Accuracy | 0.7425 | 0.6824 | 0.7983 |
| M3 | CrossAttn3 | norm/scale_both | F1 | 0.3182 | 0.1842 | 0.4390 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUC-ROC | 0.7149 | 0.5717 | 0.8429 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUPRC | 0.3065 | 0.1664 | 0.5208 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Brier | 0.1742 | 0.1429 | 0.2078 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Accuracy | 0.7368 | 0.6746 | 0.7943 |
| M3 | CrossAttn3 | excl_extreme/scale_both | F1 | 0.3373 | 0.2025 | 0.4634 |

