# Scaling Comparison — Test Set Performance (AEC 256pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8325 | 0.5008 | 0.1804 | 0.6919 | 0.3765 |
| M2 | CrossAttn | len256/scale_both | 0.8682 | 0.5197 | 0.1740 | 0.7093 | 0.4444 |
| M2_2 | CrossAttn | excl_extreme/scale_both | 0.8696 | 0.4237 | 0.1211 | 0.8117 | 0.4727 |
| M3 | CrossAttn3 | norm/scale_both | 0.7967 | 0.3945 | 0.1577 | 0.7530 | 0.4384 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_clinic** | 0.8325 | 0.5008 | 0.1804 | 0.6919 | 0.3765 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **len256/scale_both** | 0.8682 | 0.5197 | 0.1740 | 0.7093 | 0.4444 |
| crop80/scale_both | 0.8366 | 0.4337 | 0.2173 | 0.6919 | 0.4301 |
| crop60/scale_both | 0.8578 | 0.5058 | 0.2277 | 0.6395 | 0.3922 |
| norm/scale_both | 0.8395 | 0.3150 | 0.2330 | 0.6802 | 0.4211 |
| excl_extreme/scale_both | 0.8324 | 0.4601 | 0.1944 | 0.6753 | 0.4444 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_both | 0.8363 | 0.4063 | 0.1715 | 0.7209 | 0.4419 |
| crop80/scale_both | 0.7572 | 0.2956 | 0.1663 | 0.7442 | 0.3333 |
| crop60/scale_both | 0.8051 | 0.3340 | 0.1825 | 0.7674 | 0.4872 |
| norm/scale_both | 0.8395 | 0.5064 | 0.1780 | 0.7326 | 0.4390 |
| **excl_extreme/scale_both** | 0.8696 | 0.4237 | 0.1211 | 0.8117 | 0.4727 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_both | 0.7524 | 0.4133 | 0.2082 | 0.6807 | 0.3457 |
| crop80/scale_both | 0.7300 | 0.2975 | 0.1763 | 0.7349 | 0.3529 |
| crop60/scale_both | 0.7373 | 0.3893 | 0.1893 | 0.7349 | 0.3529 |
| **norm/scale_both** | 0.7967 | 0.3945 | 0.1577 | 0.7530 | 0.4384 |
| excl_extreme/scale_both | 0.7442 | 0.3137 | 0.1742 | 0.7162 | 0.3226 |

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
| AUC-ROC  | 0.8113 | 0.8048 | -0.0065 | 1.209 | 2.93e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3619 | -0.0228 | 1.333 | 2.53e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1844 | +0.0024 | -0.195 | 8.55e-01 | 1.00e+00 |
| Accuracy  | 0.7292 | 0.6972 | -0.0321 | 1.845 | 1.39e-01 | 1.88e-01 |
| F1  | 0.3950 | 0.3794 | -0.0156 | 1.242 | 2.82e-01 | 4.38e-01 |

### crop80/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8133 | +0.0019 | -0.342 | 7.49e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3739 | -0.0107 | 0.332 | 7.56e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.1717 | -0.0102 | 0.911 | 4.14e-01 | 3.12e-01 |
| Accuracy  | 0.7292 | 0.7131 | -0.0162 | 0.764 | 4.88e-01 | 4.38e-01 |
| F1  | 0.3950 | 0.4005 | +0.0055 | -1.000 | 3.74e-01 | 4.38e-01 |

### crop60/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8171 | +0.0058 | -0.842 | 4.47e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3732 | -0.0115 | 0.370 | 7.30e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1734 | -0.0085 | 1.174 | 3.05e-01 | 3.12e-01 |
| Accuracy  | 0.7292 | 0.7306 | +0.0014 | -0.069 | 9.49e-01 | 1.00e+00 |
| F1  | 0.3950 | 0.4012 | +0.0062 | -0.342 | 7.49e-01 | 1.00e+00 |

### norm/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8221 | +0.0108 | -0.897 | 4.21e-01 | 4.38e-01 |
| AUPRC † | 0.3847 | 0.4236 | +0.0390 | -2.243 | 8.83e-02 | 6.25e-02 |
| Brier  | 0.1819 | 0.1704 | -0.0115 | 0.788 | 4.75e-01 | 6.25e-01 |
| Accuracy  | 0.7292 | 0.7437 | +0.0145 | -0.610 | 5.75e-01 | 4.38e-01 |
| F1  | 0.3950 | 0.4125 | +0.0175 | -0.831 | 4.53e-01 | 8.12e-01 |

### excl_extreme/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8062 | -0.0051 | 0.398 | 7.11e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.3557 | -0.0289 | 0.578 | 5.94e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1848 | +0.0029 | -0.297 | 7.81e-01 | 8.12e-01 |
| Accuracy  | 0.7292 | 0.7098 | -0.0195 | 0.733 | 5.04e-01 | 8.12e-01 |
| F1  | 0.3950 | 0.3896 | -0.0054 | 0.217 | 8.39e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len256/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8048 | 0.8309 | +0.0260 | -1.441 | 2.23e-01 | 1.88e-01 |
| AUPRC  | 0.3619 | 0.4240 | +0.0621 | -1.556 | 1.95e-01 | 1.88e-01 |
| Brier  | 0.1844 | 0.2015 | +0.0172 | -0.588 | 5.88e-01 | 1.00e+00 |
| Accuracy  | 0.6972 | 0.6803 | -0.0169 | 0.331 | 7.57e-01 | 1.00e+00 |
| F1  | 0.3794 | 0.4042 | +0.0248 | -0.482 | 6.55e-01 | 8.12e-01 |

#### Case: crop80/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8133 | 0.8327 | +0.0194 | -1.056 | 3.50e-01 | 4.38e-01 |
| AUPRC  | 0.3739 | 0.3430 | -0.0309 | 0.670 | 5.40e-01 | 6.25e-01 |
| Brier  | 0.1717 | 0.1594 | -0.0124 | 0.469 | 6.64e-01 | 8.12e-01 |
| Accuracy  | 0.7131 | 0.7455 | +0.0324 | -0.622 | 5.68e-01 | 8.12e-01 |
| F1  | 0.4005 | 0.4179 | +0.0174 | -0.582 | 5.92e-01 | 6.25e-01 |

#### Case: crop60/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8171 | 0.8281 | +0.0110 | -0.619 | 5.69e-01 | 6.25e-01 |
| AUPRC  | 0.3732 | 0.3748 | +0.0016 | -0.043 | 9.68e-01 | 6.25e-01 |
| Brier  | 0.1734 | 0.1611 | -0.0123 | 0.758 | 4.90e-01 | 8.12e-01 |
| Accuracy  | 0.7306 | 0.7470 | +0.0163 | -0.468 | 6.64e-01 | 1.00e+00 |
| F1  | 0.4012 | 0.4465 | +0.0453 | -2.094 | 1.04e-01 | 1.25e-01 |

#### Case: norm/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8221 | 0.8433 | +0.0212 | -2.986 | 4.05e-02 | 6.25e-02 |
| AUPRC  | 0.4236 | 0.4115 | -0.0121 | 0.565 | 6.02e-01 | 8.12e-01 |
| Brier  | 0.1704 | 0.1701 | -0.0003 | 0.028 | 9.79e-01 | 1.00e+00 |
| Accuracy  | 0.7437 | 0.7455 | +0.0017 | -0.063 | 9.53e-01 | 1.00e+00 |
| F1  | 0.4125 | 0.4213 | +0.0087 | -0.322 | 7.63e-01 | 8.12e-01 |

#### Case: excl_extreme/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8062 | 0.8586 | +0.0524 | -4.386 | 1.18e-02 | 6.25e-02 |
| AUPRC  | 0.3557 | 0.4446 | +0.0889 | -0.987 | 3.79e-01 | 4.38e-01 |
| Brier  | 0.1848 | 0.1633 | -0.0215 | 1.207 | 2.94e-01 | 4.38e-01 |
| Accuracy  | 0.7098 | 0.7729 | +0.0631 | -1.430 | 2.26e-01 | 3.12e-01 |
| F1  | 0.3896 | 0.4473 | +0.0577 | -1.339 | 2.52e-01 | 3.75e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len256/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8309 | +0.0195 | -1.031 | 3.61e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.4240 | +0.0393 | -0.853 | 4.42e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.2015 | +0.0196 | -0.606 | 5.77e-01 | 6.25e-01 |
| Accuracy  | 0.7292 | 0.6803 | -0.0489 | 0.858 | 4.39e-01 | 6.25e-01 |
| F1  | 0.3950 | 0.4042 | +0.0092 | -0.178 | 8.67e-01 | 1.00e+00 |

### crop80/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8327 | +0.0213 | -1.126 | 3.23e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3430 | -0.0417 | 1.643 | 1.76e-01 | 1.88e-01 |
| Brier  | 0.1819 | 0.1594 | -0.0225 | 1.245 | 2.81e-01 | 6.25e-01 |
| Accuracy  | 0.7292 | 0.7455 | +0.0162 | -0.482 | 6.55e-01 | 1.00e+00 |
| F1  | 0.3950 | 0.4179 | +0.0229 | -0.873 | 4.32e-01 | 4.38e-01 |

### crop60/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8281 | +0.0168 | -1.182 | 3.03e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3748 | -0.0098 | 0.981 | 3.82e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1611 | -0.0208 | 1.388 | 2.37e-01 | 3.12e-01 |
| Accuracy  | 0.7292 | 0.7470 | +0.0178 | -0.837 | 4.50e-01 | 6.25e-01 |
| F1  | 0.3950 | 0.4465 | +0.0515 | -2.116 | 1.02e-01 | 1.25e-01 |

### norm/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8113 | 0.8433 | +0.0320 | -4.031 | 1.57e-02 | 6.25e-02 |
| AUPRC  | 0.3847 | 0.4115 | +0.0268 | -0.976 | 3.84e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1701 | -0.0118 | 0.618 | 5.70e-01 | 6.25e-01 |
| Accuracy  | 0.7292 | 0.7455 | +0.0162 | -0.634 | 5.61e-01 | 8.12e-01 |
| F1  | 0.3950 | 0.4213 | +0.0263 | -0.712 | 5.16e-01 | 4.38e-01 |

### excl_extreme/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8113 | 0.8586 | +0.0473 | -2.982 | 4.06e-02 | 6.25e-02 |
| AUPRC  | 0.3847 | 0.4446 | +0.0599 | -0.758 | 4.91e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1633 | -0.0186 | 0.949 | 3.96e-01 | 4.38e-01 |
| Accuracy  | 0.7292 | 0.7729 | +0.0436 | -1.034 | 3.60e-01 | 4.38e-01 |
| F1  | 0.3950 | 0.4473 | +0.0523 | -1.509 | 2.06e-01 | 3.12e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_clinic | AUC-ROC | 0.8325 | 0.7386 | 0.9091 |
| M1 | LR | scale_clinic | AUPRC | 0.5008 | 0.2965 | 0.6937 |
| M1 | LR | scale_clinic | Brier | 0.1804 | 0.1527 | 0.2118 |
| M1 | LR | scale_clinic | Accuracy | 0.6919 | 0.6221 | 0.7616 |
| M1 | LR | scale_clinic | F1 | 0.3765 | 0.2353 | 0.5060 |
| M2 | CrossAttn | len256/scale_both | AUC-ROC | 0.8682 | 0.7768 | 0.9367 |
| M2 | CrossAttn | len256/scale_both | AUPRC | 0.5197 | 0.3206 | 0.7156 |
| M2 | CrossAttn | len256/scale_both | Brier | 0.1740 | 0.1453 | 0.2028 |
| M2 | CrossAttn | len256/scale_both | Accuracy | 0.7093 | 0.6395 | 0.7791 |
| M2 | CrossAttn | len256/scale_both | F1 | 0.4444 | 0.3077 | 0.5715 |
| M2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.8366 | 0.7522 | 0.9099 |
| M2 | CrossAttn | crop80/scale_both | AUPRC | 0.4337 | 0.2572 | 0.6344 |
| M2 | CrossAttn | crop80/scale_both | Brier | 0.2173 | 0.1747 | 0.2614 |
| M2 | CrossAttn | crop80/scale_both | Accuracy | 0.6919 | 0.6221 | 0.7616 |
| M2 | CrossAttn | crop80/scale_both | F1 | 0.4301 | 0.2963 | 0.5545 |
| M2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.8578 | 0.7686 | 0.9302 |
| M2 | CrossAttn | crop60/scale_both | AUPRC | 0.5058 | 0.3076 | 0.7125 |
| M2 | CrossAttn | crop60/scale_both | Brier | 0.2277 | 0.1890 | 0.2662 |
| M2 | CrossAttn | crop60/scale_both | Accuracy | 0.6395 | 0.5640 | 0.7093 |
| M2 | CrossAttn | crop60/scale_both | F1 | 0.3922 | 0.2637 | 0.5102 |
| M2 | CrossAttn | norm/scale_both | AUC-ROC | 0.8395 | 0.7649 | 0.9006 |
| M2 | CrossAttn | norm/scale_both | AUPRC | 0.3150 | 0.2035 | 0.5078 |
| M2 | CrossAttn | norm/scale_both | Brier | 0.2330 | 0.1968 | 0.2703 |
| M2 | CrossAttn | norm/scale_both | Accuracy | 0.6802 | 0.6105 | 0.7443 |
| M2 | CrossAttn | norm/scale_both | F1 | 0.4211 | 0.2892 | 0.5472 |
| M2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8324 | 0.7551 | 0.9023 |
| M2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.4601 | 0.2687 | 0.6611 |
| M2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1944 | 0.1627 | 0.2274 |
| M2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.6753 | 0.5974 | 0.7468 |
| M2 | CrossAttn | excl_extreme/scale_both | F1 | 0.4444 | 0.3095 | 0.5647 |
| M2_2 | CrossAttn | len256/scale_both | AUC-ROC | 0.8363 | 0.7428 | 0.9093 |
| M2_2 | CrossAttn | len256/scale_both | AUPRC | 0.4063 | 0.2342 | 0.6040 |
| M2_2 | CrossAttn | len256/scale_both | Brier | 0.1715 | 0.1334 | 0.2123 |
| M2_2 | CrossAttn | len256/scale_both | Accuracy | 0.7209 | 0.6512 | 0.7849 |
| M2_2 | CrossAttn | len256/scale_both | F1 | 0.4419 | 0.3030 | 0.5652 |
| M2_2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.7572 | 0.6461 | 0.8581 |
| M2_2 | CrossAttn | crop80/scale_both | AUPRC | 0.2956 | 0.1721 | 0.5068 |
| M2_2 | CrossAttn | crop80/scale_both | Brier | 0.1663 | 0.1298 | 0.2067 |
| M2_2 | CrossAttn | crop80/scale_both | Accuracy | 0.7442 | 0.6744 | 0.8023 |
| M2_2 | CrossAttn | crop80/scale_both | F1 | 0.3333 | 0.1935 | 0.4706 |
| M2_2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.8051 | 0.6886 | 0.8982 |
| M2_2 | CrossAttn | crop60/scale_both | AUPRC | 0.3340 | 0.2066 | 0.5495 |
| M2_2 | CrossAttn | crop60/scale_both | Brier | 0.1825 | 0.1453 | 0.2236 |
| M2_2 | CrossAttn | crop60/scale_both | Accuracy | 0.7674 | 0.7035 | 0.8257 |
| M2_2 | CrossAttn | crop60/scale_both | F1 | 0.4872 | 0.3380 | 0.6207 |
| M2_2 | CrossAttn | norm/scale_both | AUC-ROC | 0.8395 | 0.7188 | 0.9306 |
| M2_2 | CrossAttn | norm/scale_both | AUPRC | 0.5064 | 0.3063 | 0.7087 |
| M2_2 | CrossAttn | norm/scale_both | Brier | 0.1780 | 0.1412 | 0.2167 |
| M2_2 | CrossAttn | norm/scale_both | Accuracy | 0.7326 | 0.6686 | 0.7965 |
| M2_2 | CrossAttn | norm/scale_both | F1 | 0.4390 | 0.2909 | 0.5714 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8696 | 0.7711 | 0.9351 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.4237 | 0.2320 | 0.6425 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1211 | 0.0947 | 0.1499 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.8117 | 0.7468 | 0.8701 |
| M2_2 | CrossAttn | excl_extreme/scale_both | F1 | 0.4727 | 0.2800 | 0.6154 |
| M3 | CrossAttn3 | len256/scale_both | AUC-ROC | 0.7524 | 0.6256 | 0.8698 |
| M3 | CrossAttn3 | len256/scale_both | AUPRC | 0.4133 | 0.2159 | 0.6389 |
| M3 | CrossAttn3 | len256/scale_both | Brier | 0.2082 | 0.1610 | 0.2513 |
| M3 | CrossAttn3 | len256/scale_both | Accuracy | 0.6807 | 0.6145 | 0.7530 |
| M3 | CrossAttn3 | len256/scale_both | F1 | 0.3457 | 0.2051 | 0.4750 |
| M3 | CrossAttn3 | crop80/scale_both | AUC-ROC | 0.7300 | 0.6051 | 0.8470 |
| M3 | CrossAttn3 | crop80/scale_both | AUPRC | 0.2975 | 0.1657 | 0.5149 |
| M3 | CrossAttn3 | crop80/scale_both | Brier | 0.1763 | 0.1318 | 0.2219 |
| M3 | CrossAttn3 | crop80/scale_both | Accuracy | 0.7349 | 0.6687 | 0.7953 |
| M3 | CrossAttn3 | crop80/scale_both | F1 | 0.3529 | 0.1967 | 0.4928 |
| M3 | CrossAttn3 | crop60/scale_both | AUC-ROC | 0.7373 | 0.6029 | 0.8616 |
| M3 | CrossAttn3 | crop60/scale_both | AUPRC | 0.3893 | 0.2075 | 0.6159 |
| M3 | CrossAttn3 | crop60/scale_both | Brier | 0.1893 | 0.1457 | 0.2347 |
| M3 | CrossAttn3 | crop60/scale_both | Accuracy | 0.7349 | 0.6685 | 0.8012 |
| M3 | CrossAttn3 | crop60/scale_both | F1 | 0.3529 | 0.1967 | 0.4938 |
| M3 | CrossAttn3 | norm/scale_both | AUC-ROC | 0.7967 | 0.6835 | 0.8962 |
| M3 | CrossAttn3 | norm/scale_both | AUPRC | 0.3945 | 0.2259 | 0.6109 |
| M3 | CrossAttn3 | norm/scale_both | Brier | 0.1577 | 0.1177 | 0.1995 |
| M3 | CrossAttn3 | norm/scale_both | Accuracy | 0.7530 | 0.6867 | 0.8193 |
| M3 | CrossAttn3 | norm/scale_both | F1 | 0.4384 | 0.2807 | 0.5714 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUC-ROC | 0.7442 | 0.6123 | 0.8585 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUPRC | 0.3137 | 0.1770 | 0.5505 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Brier | 0.1742 | 0.1323 | 0.2172 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Accuracy | 0.7162 | 0.6419 | 0.7905 |
| M3 | CrossAttn3 | excl_extreme/scale_both | F1 | 0.3226 | 0.1667 | 0.4706 |

