# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8149 | 0.3784 | 0.1934 | 0.6966 | 0.3604 |
| M2 | CrossAttn | norm/scale_both | 0.7922 | 0.3032 | 0.1676 | 0.7564 | 0.3736 |
| M2_2 | CrossAttn | excl_extreme/scale_both | 0.8056 | 0.3097 | 0.1778 | 0.7143 | 0.3750 |
| M3 | CrossAttn3 | crop80/scale_both | 0.6954 | 0.2613 | 0.1687 | 0.7425 | 0.3182 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_clinic** | 0.8149 | 0.3784 | 0.1934 | 0.6966 | 0.3604 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.7478 | 0.2466 | 0.2187 | 0.6709 | 0.3186 |
| crop80/scale_both | 0.7439 | 0.2295 | 0.2209 | 0.6709 | 0.3186 |
| crop60/scale_both | 0.7430 | 0.2322 | 0.2284 | 0.6923 | 0.3455 |
| **norm/scale_both** | 0.7922 | 0.3032 | 0.1676 | 0.7564 | 0.3736 |
| excl_extreme/scale_both | 0.7724 | 0.2209 | 0.1864 | 0.6952 | 0.2727 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.7889 | 0.3063 | 0.2246 | 0.6325 | 0.3175 |
| crop80/scale_both | 0.7971 | 0.3305 | 0.2983 | 0.5556 | 0.3067 |
| crop60/scale_both | 0.7604 | 0.2548 | 0.2007 | 0.6581 | 0.3333 |
| norm/scale_both | 0.7489 | 0.2484 | 0.2474 | 0.6239 | 0.3125 |
| **excl_extreme/scale_both** | 0.8056 | 0.3097 | 0.1778 | 0.7143 | 0.3750 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.6512 | 0.2389 | 0.2236 | 0.6652 | 0.2200 |
| **crop80/scale_both** | 0.6954 | 0.2613 | 0.1687 | 0.7425 | 0.3182 |
| crop60/scale_both | 0.6663 | 0.2402 | 0.2143 | 0.6781 | 0.2424 |
| norm/scale_both | 0.6867 | 0.2584 | 0.1968 | 0.6910 | 0.2653 |
| excl_extreme/scale_both | 0.6738 | 0.2630 | 0.2047 | 0.7129 | 0.2857 |

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
| AUC-ROC  | 0.8122 | 0.8335 | +0.0213 | -1.339 | 2.52e-01 | 4.38e-01 |
| AUPRC  | 0.4046 | 0.4371 | +0.0325 | -1.540 | 1.98e-01 | 3.12e-01 |
| Brier  | 0.1795 | 0.1584 | -0.0211 | 0.995 | 3.76e-01 | 4.38e-01 |
| Accuracy  | 0.7350 | 0.7598 | +0.0248 | -0.600 | 5.81e-01 | 6.25e-01 |
| F1  | 0.3670 | 0.4044 | +0.0374 | -1.251 | 2.79e-01 | 3.12e-01 |

### crop80/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8122 | 0.8287 | +0.0165 | -0.938 | 4.01e-01 | 4.38e-01 |
| AUPRC  | 0.4046 | 0.4081 | +0.0035 | -0.253 | 8.13e-01 | 1.00e+00 |
| Brier  | 0.1795 | 0.1821 | +0.0026 | -0.105 | 9.21e-01 | 1.00e+00 |
| Accuracy  | 0.7350 | 0.7170 | -0.0180 | 0.394 | 7.14e-01 | 1.00e+00 |
| F1  | 0.3670 | 0.3795 | +0.0125 | -0.366 | 7.33e-01 | 8.12e-01 |

### crop60/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8122 | 0.8395 | +0.0272 | -1.525 | 2.02e-01 | 1.88e-01 |
| AUPRC  | 0.4046 | 0.4190 | +0.0144 | -0.526 | 6.27e-01 | 6.25e-01 |
| Brier  | 0.1795 | 0.1760 | -0.0035 | 0.430 | 6.89e-01 | 8.12e-01 |
| Accuracy  | 0.7350 | 0.7307 | -0.0043 | 0.254 | 8.12e-01 | 6.25e-01 |
| F1  | 0.3670 | 0.3955 | +0.0285 | -1.133 | 3.20e-01 | 8.12e-01 |

### norm/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8122 | 0.8228 | +0.0105 | -0.826 | 4.55e-01 | 4.38e-01 |
| AUPRC  | 0.4046 | 0.3987 | -0.0059 | 0.298 | 7.81e-01 | 8.12e-01 |
| Brier  | 0.1795 | 0.1837 | +0.0042 | -0.305 | 7.76e-01 | 1.00e+00 |
| Accuracy  | 0.7350 | 0.7392 | +0.0042 | -0.230 | 8.30e-01 | 8.75e-01 |
| F1  | 0.3670 | 0.3901 | +0.0231 | -1.676 | 1.69e-01 | 1.25e-01 |

### excl_extreme/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8122 | 0.8383 | +0.0261 | -1.539 | 1.99e-01 | 3.12e-01 |
| AUPRC  | 0.4046 | 0.4010 | -0.0036 | 0.242 | 8.20e-01 | 1.00e+00 |
| Brier  | 0.1795 | 0.1658 | -0.0137 | 0.787 | 4.75e-01 | 8.12e-01 |
| Accuracy  | 0.7350 | 0.7517 | +0.0168 | -0.673 | 5.38e-01 | 4.38e-01 |
| F1  | 0.3670 | 0.4060 | +0.0390 | -1.813 | 1.44e-01 | 1.88e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len128/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8335 | 0.8449 | +0.0113 | -1.467 | 2.16e-01 | 3.12e-01 |
| AUPRC  | 0.4371 | 0.4210 | -0.0161 | 0.633 | 5.61e-01 | 6.25e-01 |
| Brier  | 0.1584 | 0.1459 | -0.0125 | 0.629 | 5.64e-01 | 8.12e-01 |
| Accuracy  | 0.7598 | 0.7880 | +0.0282 | -0.845 | 4.46e-01 | 8.12e-01 |
| F1  | 0.4044 | 0.4385 | +0.0340 | -1.526 | 2.02e-01 | 3.12e-01 |

#### Case: crop80/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8287 | 0.8482 | +0.0195 | -1.351 | 2.48e-01 | 3.12e-01 |
| AUPRC  | 0.4081 | 0.4115 | +0.0035 | -0.112 | 9.16e-01 | 1.00e+00 |
| Brier  | 0.1821 | 0.1658 | -0.0163 | 0.547 | 6.13e-01 | 6.25e-01 |
| Accuracy  | 0.7170 | 0.7557 | +0.0387 | -0.774 | 4.82e-01 | 3.12e-01 |
| F1  | 0.3795 | 0.4234 | +0.0440 | -1.359 | 2.46e-01 | 3.12e-01 |

#### Case: crop60/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8395 | 0.8473 | +0.0078 | -0.596 | 5.83e-01 | 6.25e-01 |
| AUPRC  | 0.4190 | 0.4124 | -0.0066 | 0.174 | 8.71e-01 | 8.12e-01 |
| Brier  | 0.1760 | 0.1686 | -0.0074 | 0.578 | 5.94e-01 | 8.12e-01 |
| Accuracy  | 0.7307 | 0.7557 | +0.0250 | -0.724 | 5.09e-01 | 6.25e-01 |
| F1  | 0.3955 | 0.4143 | +0.0189 | -0.575 | 5.96e-01 | 6.25e-01 |

#### Case: norm/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8228 | 0.8469 | +0.0241 | -2.300 | 8.29e-02 | 6.25e-02 |
| AUPRC  | 0.3987 | 0.4226 | +0.0239 | -0.793 | 4.72e-01 | 6.25e-01 |
| Brier  | 0.1837 | 0.1775 | -0.0062 | 0.511 | 6.37e-01 | 1.00e+00 |
| Accuracy  | 0.7392 | 0.7245 | -0.0147 | 0.489 | 6.50e-01 | 8.12e-01 |
| F1  | 0.3901 | 0.3911 | +0.0010 | -0.058 | 9.56e-01 | 6.25e-01 |

#### Case: excl_extreme/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8383 | 0.8574 | +0.0191 | -0.603 | 5.79e-01 | 6.25e-01 |
| AUPRC  | 0.4010 | 0.4556 | +0.0546 | -1.123 | 3.24e-01 | 3.12e-01 |
| Brier  | 0.1658 | 0.1517 | -0.0141 | 1.168 | 3.08e-01 | 6.25e-01 |
| Accuracy † | 0.7517 | 0.7832 | +0.0315 | -2.580 | 6.14e-02 | 1.25e-01 |
| F1  | 0.4060 | 0.4307 | +0.0247 | -0.807 | 4.65e-01 | 8.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8122 | 0.8449 | +0.0326 | -2.347 | 7.87e-02 | 1.25e-01 |
| AUPRC  | 0.4046 | 0.4210 | +0.0164 | -0.534 | 6.22e-01 | 8.12e-01 |
| Brier ** | 0.1795 | 0.1459 | -0.0336 | 5.184 | 6.59e-03 | 6.25e-02 |
| Accuracy * | 0.7350 | 0.7880 | +0.0530 | -4.125 | 1.46e-02 | 6.25e-02 |
| F1 * | 0.3670 | 0.4385 | +0.0715 | -3.165 | 3.40e-02 | 6.25e-02 |

### crop80/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8122 | 0.8482 | +0.0360 | -3.668 | 2.14e-02 | 1.25e-01 |
| AUPRC  | 0.4046 | 0.4115 | +0.0069 | -0.162 | 8.79e-01 | 1.00e+00 |
| Brier  | 0.1795 | 0.1658 | -0.0137 | 0.856 | 4.40e-01 | 6.25e-01 |
| Accuracy  | 0.7350 | 0.7557 | +0.0207 | -0.811 | 4.63e-01 | 3.12e-01 |
| F1 † | 0.3670 | 0.4234 | +0.0564 | -2.270 | 8.57e-02 | 1.25e-01 |

### crop60/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8122 | 0.8473 | +0.0351 | -3.506 | 2.48e-02 | 1.25e-01 |
| AUPRC  | 0.4046 | 0.4124 | +0.0078 | -0.159 | 8.81e-01 | 1.00e+00 |
| Brier  | 0.1795 | 0.1686 | -0.0109 | 1.009 | 3.70e-01 | 4.38e-01 |
| Accuracy  | 0.7350 | 0.7557 | +0.0208 | -0.783 | 4.78e-01 | 4.38e-01 |
| F1 † | 0.3670 | 0.4143 | +0.0473 | -2.132 | 9.99e-02 | 1.25e-01 |

### norm/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8122 | 0.8469 | +0.0346 | -2.115 | 1.02e-01 | 1.25e-01 |
| AUPRC  | 0.4046 | 0.4226 | +0.0180 | -0.473 | 6.61e-01 | 1.00e+00 |
| Brier  | 0.1795 | 0.1775 | -0.0020 | 0.138 | 8.97e-01 | 8.12e-01 |
| Accuracy  | 0.7350 | 0.7245 | -0.0105 | 0.314 | 7.69e-01 | 1.00e+00 |
| F1  | 0.3670 | 0.3911 | +0.0241 | -1.200 | 2.96e-01 | 4.38e-01 |

### excl_extreme/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8122 | 0.8574 | +0.0452 | -1.414 | 2.30e-01 | 1.88e-01 |
| AUPRC  | 0.4046 | 0.4556 | +0.0510 | -0.930 | 4.05e-01 | 4.38e-01 |
| Brier  | 0.1795 | 0.1517 | -0.0279 | 1.410 | 2.31e-01 | 3.12e-01 |
| Accuracy  | 0.7350 | 0.7832 | +0.0483 | -1.987 | 1.18e-01 | 1.88e-01 |
| F1  | 0.3670 | 0.4307 | +0.0637 | -2.064 | 1.08e-01 | 1.25e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_clinic | AUC-ROC | 0.8149 | 0.7374 | 0.8853 |
| M1 | LR | scale_clinic | AUPRC | 0.3784 | 0.2187 | 0.5629 |
| M1 | LR | scale_clinic | Brier | 0.1934 | 0.1676 | 0.2210 |
| M1 | LR | scale_clinic | Accuracy | 0.6966 | 0.6368 | 0.7521 |
| M1 | LR | scale_clinic | F1 | 0.3604 | 0.2444 | 0.4685 |
| M2 | CrossAttn | len128/scale_both | AUC-ROC | 0.7478 | 0.6575 | 0.8306 |
| M2 | CrossAttn | len128/scale_both | AUPRC | 0.2466 | 0.1469 | 0.4209 |
| M2 | CrossAttn | len128/scale_both | Brier | 0.2187 | 0.1873 | 0.2531 |
| M2 | CrossAttn | len128/scale_both | Accuracy | 0.6709 | 0.6111 | 0.7266 |
| M2 | CrossAttn | len128/scale_both | F1 | 0.3186 | 0.2069 | 0.4259 |
| M2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.7439 | 0.6548 | 0.8286 |
| M2 | CrossAttn | crop80/scale_both | AUPRC | 0.2295 | 0.1417 | 0.3927 |
| M2 | CrossAttn | crop80/scale_both | Brier | 0.2209 | 0.1890 | 0.2562 |
| M2 | CrossAttn | crop80/scale_both | Accuracy | 0.6709 | 0.6068 | 0.7265 |
| M2 | CrossAttn | crop80/scale_both | F1 | 0.3186 | 0.2063 | 0.4274 |
| M2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.7430 | 0.6480 | 0.8303 |
| M2 | CrossAttn | crop60/scale_both | AUPRC | 0.2322 | 0.1447 | 0.4001 |
| M2 | CrossAttn | crop60/scale_both | Brier | 0.2284 | 0.1942 | 0.2665 |
| M2 | CrossAttn | crop60/scale_both | Accuracy | 0.6923 | 0.6325 | 0.7479 |
| M2 | CrossAttn | crop60/scale_both | F1 | 0.3455 | 0.2301 | 0.4553 |
| M2 | CrossAttn | norm/scale_both | AUC-ROC | 0.7922 | 0.7069 | 0.8743 |
| M2 | CrossAttn | norm/scale_both | AUPRC | 0.3032 | 0.1846 | 0.4975 |
| M2 | CrossAttn | norm/scale_both | Brier | 0.1676 | 0.1414 | 0.1945 |
| M2 | CrossAttn | norm/scale_both | Accuracy | 0.7564 | 0.7009 | 0.8120 |
| M2 | CrossAttn | norm/scale_both | F1 | 0.3736 | 0.2500 | 0.4938 |
| M2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.7724 | 0.6676 | 0.8563 |
| M2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.2209 | 0.1250 | 0.4078 |
| M2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1864 | 0.1558 | 0.2213 |
| M2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.6952 | 0.6286 | 0.7524 |
| M2 | CrossAttn | excl_extreme/scale_both | F1 | 0.2727 | 0.1481 | 0.3918 |
| M2_2 | CrossAttn | len128/scale_both | AUC-ROC | 0.7889 | 0.6996 | 0.8710 |
| M2_2 | CrossAttn | len128/scale_both | AUPRC | 0.3063 | 0.1790 | 0.4759 |
| M2_2 | CrossAttn | len128/scale_both | Brier | 0.2246 | 0.1917 | 0.2596 |
| M2_2 | CrossAttn | len128/scale_both | Accuracy | 0.6325 | 0.5684 | 0.6923 |
| M2_2 | CrossAttn | len128/scale_both | F1 | 0.3175 | 0.2097 | 0.4211 |
| M2_2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.7971 | 0.7029 | 0.8779 |
| M2_2 | CrossAttn | crop80/scale_both | AUPRC | 0.3305 | 0.1901 | 0.5161 |
| M2_2 | CrossAttn | crop80/scale_both | Brier | 0.2983 | 0.2653 | 0.3344 |
| M2_2 | CrossAttn | crop80/scale_both | Accuracy | 0.5556 | 0.4915 | 0.6154 |
| M2_2 | CrossAttn | crop80/scale_both | F1 | 0.3067 | 0.2080 | 0.3977 |
| M2_2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.7604 | 0.6631 | 0.8514 |
| M2_2 | CrossAttn | crop60/scale_both | AUPRC | 0.2548 | 0.1589 | 0.4299 |
| M2_2 | CrossAttn | crop60/scale_both | Brier | 0.2007 | 0.1715 | 0.2304 |
| M2_2 | CrossAttn | crop60/scale_both | Accuracy | 0.6581 | 0.5983 | 0.7179 |
| M2_2 | CrossAttn | crop60/scale_both | F1 | 0.3333 | 0.2258 | 0.4370 |
| M2_2 | CrossAttn | norm/scale_both | AUC-ROC | 0.7489 | 0.6617 | 0.8290 |
| M2_2 | CrossAttn | norm/scale_both | AUPRC | 0.2484 | 0.1509 | 0.4289 |
| M2_2 | CrossAttn | norm/scale_both | Brier | 0.2474 | 0.2140 | 0.2834 |
| M2_2 | CrossAttn | norm/scale_both | Accuracy | 0.6239 | 0.5598 | 0.6838 |
| M2_2 | CrossAttn | norm/scale_both | F1 | 0.3125 | 0.2080 | 0.4134 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8056 | 0.7236 | 0.8811 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.3097 | 0.1884 | 0.5049 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1778 | 0.1462 | 0.2110 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.7143 | 0.6524 | 0.7714 |
| M2_2 | CrossAttn | excl_extreme/scale_both | F1 | 0.3750 | 0.2424 | 0.4902 |
| M3 | CrossAttn3 | len128/scale_both | AUC-ROC | 0.6512 | 0.5323 | 0.7589 |
| M3 | CrossAttn3 | len128/scale_both | AUPRC | 0.2389 | 0.1202 | 0.3974 |
| M3 | CrossAttn3 | len128/scale_both | Brier | 0.2236 | 0.1854 | 0.2637 |
| M3 | CrossAttn3 | len128/scale_both | Accuracy | 0.6652 | 0.6009 | 0.7296 |
| M3 | CrossAttn3 | len128/scale_both | F1 | 0.2200 | 0.1121 | 0.3261 |
| M3 | CrossAttn3 | crop80/scale_both | AUC-ROC | 0.6954 | 0.5652 | 0.8070 |
| M3 | CrossAttn3 | crop80/scale_both | AUPRC | 0.2613 | 0.1403 | 0.4326 |
| M3 | CrossAttn3 | crop80/scale_both | Brier | 0.1687 | 0.1395 | 0.1993 |
| M3 | CrossAttn3 | crop80/scale_both | Accuracy | 0.7425 | 0.6867 | 0.7983 |
| M3 | CrossAttn3 | crop80/scale_both | F1 | 0.3182 | 0.1882 | 0.4381 |
| M3 | CrossAttn3 | crop60/scale_both | AUC-ROC | 0.6663 | 0.5456 | 0.7769 |
| M3 | CrossAttn3 | crop60/scale_both | AUPRC | 0.2402 | 0.1250 | 0.4079 |
| M3 | CrossAttn3 | crop60/scale_both | Brier | 0.2143 | 0.1780 | 0.2529 |
| M3 | CrossAttn3 | crop60/scale_both | Accuracy | 0.6781 | 0.6180 | 0.7382 |
| M3 | CrossAttn3 | crop60/scale_both | F1 | 0.2424 | 0.1276 | 0.3529 |
| M3 | CrossAttn3 | norm/scale_both | AUC-ROC | 0.6867 | 0.5665 | 0.8004 |
| M3 | CrossAttn3 | norm/scale_both | AUPRC | 0.2584 | 0.1381 | 0.4379 |
| M3 | CrossAttn3 | norm/scale_both | Brier | 0.1968 | 0.1626 | 0.2312 |
| M3 | CrossAttn3 | norm/scale_both | Accuracy | 0.6910 | 0.6309 | 0.7511 |
| M3 | CrossAttn3 | norm/scale_both | F1 | 0.2653 | 0.1443 | 0.3774 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUC-ROC | 0.6738 | 0.5413 | 0.7979 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUPRC | 0.2630 | 0.1301 | 0.4570 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Brier | 0.2047 | 0.1651 | 0.2433 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Accuracy | 0.7129 | 0.6554 | 0.7751 |
| M3 | CrossAttn3 | excl_extreme/scale_both | F1 | 0.2857 | 0.1515 | 0.4045 |

