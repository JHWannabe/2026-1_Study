# Scaling Comparison — Test Set Performance (AEC 256pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |
| M2 | CrossAttn | crop60 | 0.8934 | 0.6140 | 0.1703 | 0.7616 | 0.4938 |
| M2_2 | CrossAttn | excl_extreme | 0.8764 | 0.4707 | 0.1693 | 0.8052 | 0.4444 |
| M3 | CrossAttn3 | len128 | 0.8798 | 0.5875 | 0.2240 | 0.5581 | 0.3448 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_all** | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.8404 | 0.3494 | 0.2137 | 0.7093 | 0.4318 |
| norm | 0.8786 | 0.5160 | 0.1679 | 0.7326 | 0.4651 |
| crop80 | 0.8307 | 0.4144 | 0.1790 | 0.7616 | 0.4533 |
| **crop60** | 0.8934 | 0.6140 | 0.1703 | 0.7616 | 0.4938 |
| excl_extreme | 0.8790 | 0.5129 | 0.2094 | 0.6753 | 0.4444 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.8114 | 0.4036 | 0.2024 | 0.6686 | 0.3871 |
| norm | 0.7827 | 0.3356 | 0.1686 | 0.7267 | 0.4051 |
| crop80 | 0.8329 | 0.4252 | 0.2043 | 0.7558 | 0.4474 |
| crop60 | 0.8127 | 0.3303 | 0.1642 | 0.8081 | 0.4762 |
| **excl_extreme** | 0.8764 | 0.4707 | 0.1693 | 0.8052 | 0.4444 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **len128** | 0.8798 | 0.5875 | 0.2240 | 0.5581 | 0.3448 |
| norm | 0.8767 | 0.5148 | 0.1528 | 0.7791 | 0.4722 |
| crop80 | 0.8625 | 0.4930 | 0.2493 | 0.7442 | 0.4500 |
| crop60 | 0.7893 | 0.3070 | 0.2062 | 0.7267 | 0.4337 |
| excl_extreme | 0.8367 | 0.4310 | 0.2130 | 0.6494 | 0.4255 |

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
| AUC-ROC  | 0.8113 | 0.8170 | +0.0056 | -0.535 | 6.21e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3499 | -0.0347 | 1.779 | 1.50e-01 | 1.88e-01 |
| Brier  | 0.1819 | 0.1867 | +0.0048 | -2.039 | 1.11e-01 | 1.88e-01 |
| Accuracy  | 0.7555 | 0.7744 | +0.0189 | -0.591 | 5.87e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4673 | +0.0159 | -0.644 | 5.54e-01 | 4.38e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8374 | +0.0261 | -1.921 | 1.27e-01 | 1.88e-01 |
| AUPRC  | 0.3847 | 0.4280 | +0.0433 | -1.172 | 3.06e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1692 | -0.0127 | 0.798 | 4.70e-01 | 4.38e-01 |
| Accuracy  | 0.7555 | 0.7817 | +0.0262 | -0.873 | 4.32e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4792 | +0.0278 | -1.633 | 1.78e-01 | 1.88e-01 |

### crop80  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8152 | +0.0038 | -0.330 | 7.58e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3713 | -0.0134 | 0.811 | 4.63e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1872 | +0.0053 | -0.287 | 7.89e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7321 | -0.0233 | 0.963 | 3.90e-01 | 3.12e-01 |
| F1  | 0.4514 | 0.4424 | -0.0090 | 0.379 | 7.24e-01 | 8.12e-01 |

### crop60  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8154 | +0.0040 | -0.555 | 6.09e-01 | 8.12e-01 |
| AUPRC * | 0.3847 | 0.3509 | -0.0337 | 3.021 | 3.91e-02 | 6.25e-02 |
| Brier  | 0.1819 | 0.1876 | +0.0057 | -0.603 | 5.79e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7584 | +0.0030 | -0.084 | 9.37e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4584 | +0.0070 | -0.269 | 8.01e-01 | 8.12e-01 |

### excl_extreme  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8099 | -0.0015 | 0.111 | 9.17e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3768 | -0.0078 | 0.118 | 9.12e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.2011 | +0.0192 | -0.773 | 4.83e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7582 | +0.0028 | -0.045 | 9.66e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4697 | +0.0183 | -0.441 | 6.82e-01 | 6.25e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: len128  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8170 | 0.8195 | +0.0026 | -0.433 | 6.87e-01 | 6.25e-01 |
| AUPRC  | 0.3499 | 0.4216 | +0.0717 | -2.041 | 1.11e-01 | 6.25e-02 |
| Brier  | 0.1867 | 0.1645 | -0.0223 | 1.862 | 1.36e-01 | 3.12e-01 |
| Accuracy  | 0.7744 | 0.7350 | -0.0394 | 1.098 | 3.34e-01 | 3.12e-01 |
| F1  | 0.4673 | 0.4558 | -0.0115 | 0.364 | 7.34e-01 | 8.12e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8374 | 0.8101 | -0.0273 | 1.832 | 1.41e-01 | 1.88e-01 |
| AUPRC  | 0.4280 | 0.4162 | -0.0118 | 0.262 | 8.06e-01 | 8.12e-01 |
| Brier  | 0.1692 | 0.1941 | +0.0249 | -0.937 | 4.02e-01 | 4.38e-01 |
| Accuracy  | 0.7817 | 0.7484 | -0.0333 | 0.596 | 5.83e-01 | 8.12e-01 |
| F1  | 0.4792 | 0.4471 | -0.0322 | 0.924 | 4.08e-01 | 6.25e-01 |

#### Case: crop80  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8152 | 0.8062 | -0.0089 | 0.988 | 3.79e-01 | 6.25e-01 |
| AUPRC  | 0.3713 | 0.3829 | +0.0116 | -0.434 | 6.87e-01 | 6.25e-01 |
| Brier  | 0.1872 | 0.2000 | +0.0129 | -0.488 | 6.51e-01 | 6.25e-01 |
| Accuracy  | 0.7321 | 0.7713 | +0.0392 | -0.986 | 3.80e-01 | 4.38e-01 |
| F1  | 0.4424 | 0.4663 | +0.0238 | -0.540 | 6.18e-01 | 6.25e-01 |

#### Case: crop60  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8154 | 0.8166 | +0.0013 | -0.486 | 6.53e-01 | 8.12e-01 |
| AUPRC  | 0.3509 | 0.3825 | +0.0315 | -1.044 | 3.55e-01 | 3.12e-01 |
| Brier  | 0.1876 | 0.1760 | -0.0116 | 0.662 | 5.44e-01 | 6.25e-01 |
| Accuracy  | 0.7584 | 0.7788 | +0.0204 | -1.310 | 2.60e-01 | 3.12e-01 |
| F1  | 0.4584 | 0.4727 | +0.0143 | -1.227 | 2.87e-01 | 3.12e-01 |

#### Case: excl_extreme  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8099 | 0.8021 | -0.0078 | 0.762 | 4.88e-01 | 8.12e-01 |
| AUPRC  | 0.3768 | 0.3793 | +0.0025 | -0.065 | 9.51e-01 | 1.00e+00 |
| Brier  | 0.2011 | 0.1784 | -0.0227 | 0.723 | 5.09e-01 | 8.12e-01 |
| Accuracy  | 0.7582 | 0.7180 | -0.0403 | 0.690 | 5.28e-01 | 6.25e-01 |
| F1  | 0.4697 | 0.4153 | -0.0544 | 1.169 | 3.07e-01 | 4.38e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8195 | +0.0082 | -0.890 | 4.24e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.4216 | +0.0369 | -1.326 | 2.56e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1645 | -0.0174 | 1.509 | 2.06e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7350 | -0.0205 | 0.476 | 6.59e-01 | 7.50e-01 |
| F1  | 0.4514 | 0.4558 | +0.0043 | -0.222 | 8.35e-01 | 6.25e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8101 | -0.0012 | 0.087 | 9.35e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.4162 | +0.0315 | -1.261 | 2.76e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1941 | +0.0122 | -0.764 | 4.88e-01 | 4.38e-01 |
| Accuracy  | 0.7555 | 0.7484 | -0.0071 | 0.124 | 9.07e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4471 | -0.0044 | 0.099 | 9.26e-01 | 8.12e-01 |

### crop80  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8062 | -0.0051 | 0.562 | 6.04e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.3829 | -0.0018 | 0.108 | 9.20e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.2000 | +0.0181 | -1.169 | 3.07e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7713 | +0.0159 | -0.342 | 7.50e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4663 | +0.0148 | -0.418 | 6.97e-01 | 6.25e-01 |

### crop60  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8166 | +0.0053 | -0.619 | 5.70e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3825 | -0.0022 | 0.057 | 9.58e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1760 | -0.0060 | 0.519 | 6.31e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7788 | +0.0233 | -0.644 | 5.54e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4727 | +0.0213 | -0.744 | 4.98e-01 | 4.38e-01 |

### excl_extreme  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8021 | -0.0092 | 0.681 | 5.33e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3793 | -0.0053 | 0.133 | 9.01e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1784 | -0.0035 | 0.215 | 8.40e-01 | 8.12e-01 |
| Accuracy † | 0.7555 | 0.7180 | -0.0375 | 2.640 | 5.76e-02 | 1.25e-01 |
| F1 † | 0.4514 | 0.4153 | -0.0361 | 2.353 | 7.83e-02 | 1.25e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_all | AUC-ROC | 0.8325 | 0.7386 | 0.9091 |
| M1 | LR | scale_all | AUPRC | 0.5008 | 0.2965 | 0.6937 |
| M1 | LR | scale_all | Brier | 0.1804 | 0.1527 | 0.2118 |
| M1 | LR | scale_all | Accuracy | 0.6977 | 0.6279 | 0.7674 |
| M1 | LR | scale_all | F1 | 0.3953 | 0.2535 | 0.5228 |
| M2 | CrossAttn | len128 | AUC-ROC | 0.8404 | 0.7620 | 0.9072 |
| M2 | CrossAttn | len128 | AUPRC | 0.3494 | 0.2129 | 0.5748 |
| M2 | CrossAttn | len128 | Brier | 0.2137 | 0.1738 | 0.2541 |
| M2 | CrossAttn | len128 | Accuracy | 0.7093 | 0.6395 | 0.7791 |
| M2 | CrossAttn | len128 | F1 | 0.4318 | 0.2941 | 0.5532 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8786 | 0.8034 | 0.9368 |
| M2 | CrossAttn | norm | AUPRC | 0.5160 | 0.3195 | 0.7199 |
| M2 | CrossAttn | norm | Brier | 0.1679 | 0.1404 | 0.1973 |
| M2 | CrossAttn | norm | Accuracy | 0.7326 | 0.6628 | 0.7965 |
| M2 | CrossAttn | norm | F1 | 0.4651 | 0.3256 | 0.5926 |
| M2 | CrossAttn | crop80 | AUC-ROC | 0.8307 | 0.7342 | 0.9077 |
| M2 | CrossAttn | crop80 | AUPRC | 0.4144 | 0.2437 | 0.6268 |
| M2 | CrossAttn | crop80 | Brier | 0.1790 | 0.1440 | 0.2182 |
| M2 | CrossAttn | crop80 | Accuracy | 0.7616 | 0.6919 | 0.8256 |
| M2 | CrossAttn | crop80 | F1 | 0.4533 | 0.3030 | 0.5882 |
| M2 | CrossAttn | crop60 | AUC-ROC | 0.8934 | 0.8310 | 0.9463 |
| M2 | CrossAttn | crop60 | AUPRC | 0.6140 | 0.4082 | 0.7894 |
| M2 | CrossAttn | crop60 | Brier | 0.1703 | 0.1374 | 0.2045 |
| M2 | CrossAttn | crop60 | Accuracy | 0.7616 | 0.6977 | 0.8256 |
| M2 | CrossAttn | crop60 | F1 | 0.4938 | 0.3478 | 0.6214 |
| M2 | CrossAttn | excl_extreme | AUC-ROC | 0.8790 | 0.8080 | 0.9353 |
| M2 | CrossAttn | excl_extreme | AUPRC | 0.5129 | 0.3128 | 0.7107 |
| M2 | CrossAttn | excl_extreme | Brier | 0.2094 | 0.1694 | 0.2520 |
| M2 | CrossAttn | excl_extreme | Accuracy | 0.6753 | 0.5974 | 0.7468 |
| M2 | CrossAttn | excl_extreme | F1 | 0.4444 | 0.3059 | 0.5636 |
| M2_2 | CrossAttn | len128 | AUC-ROC | 0.8114 | 0.7102 | 0.8929 |
| M2_2 | CrossAttn | len128 | AUPRC | 0.4036 | 0.2277 | 0.6068 |
| M2_2 | CrossAttn | len128 | Brier | 0.2024 | 0.1584 | 0.2502 |
| M2_2 | CrossAttn | len128 | Accuracy | 0.6686 | 0.5930 | 0.7384 |
| M2_2 | CrossAttn | len128 | F1 | 0.3871 | 0.2563 | 0.5143 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.7827 | 0.6695 | 0.8816 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3356 | 0.1941 | 0.5708 |
| M2_2 | CrossAttn | norm | Brier | 0.1686 | 0.1308 | 0.2122 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7267 | 0.6570 | 0.7907 |
| M2_2 | CrossAttn | norm | F1 | 0.4051 | 0.2597 | 0.5432 |
| M2_2 | CrossAttn | crop80 | AUC-ROC | 0.8329 | 0.7429 | 0.9059 |
| M2_2 | CrossAttn | crop80 | AUPRC | 0.4252 | 0.2478 | 0.6208 |
| M2_2 | CrossAttn | crop80 | Brier | 0.2043 | 0.1669 | 0.2451 |
| M2_2 | CrossAttn | crop80 | Accuracy | 0.7558 | 0.6919 | 0.8140 |
| M2_2 | CrossAttn | crop80 | F1 | 0.4474 | 0.2973 | 0.5773 |
| M2_2 | CrossAttn | crop60 | AUC-ROC | 0.8127 | 0.7006 | 0.9009 |
| M2_2 | CrossAttn | crop60 | AUPRC | 0.3303 | 0.2108 | 0.5370 |
| M2_2 | CrossAttn | crop60 | Brier | 0.1642 | 0.1259 | 0.2061 |
| M2_2 | CrossAttn | crop60 | Accuracy | 0.8081 | 0.7442 | 0.8663 |
| M2_2 | CrossAttn | crop60 | F1 | 0.4762 | 0.3188 | 0.6182 |
| M2_2 | CrossAttn | excl_extreme | AUC-ROC | 0.8764 | 0.7933 | 0.9396 |
| M2_2 | CrossAttn | excl_extreme | AUPRC | 0.4707 | 0.2543 | 0.6848 |
| M2_2 | CrossAttn | excl_extreme | Brier | 0.1693 | 0.1388 | 0.2028 |
| M2_2 | CrossAttn | excl_extreme | Accuracy | 0.8052 | 0.7403 | 0.8636 |
| M2_2 | CrossAttn | excl_extreme | F1 | 0.4444 | 0.2580 | 0.5965 |
| M3 | CrossAttn3 | len128 | AUC-ROC | 0.8798 | 0.7970 | 0.9516 |
| M3 | CrossAttn3 | len128 | AUPRC | 0.5875 | 0.3804 | 0.7919 |
| M3 | CrossAttn3 | len128 | Brier | 0.2240 | 0.1865 | 0.2606 |
| M3 | CrossAttn3 | len128 | Accuracy | 0.5581 | 0.4826 | 0.6337 |
| M3 | CrossAttn3 | len128 | F1 | 0.3448 | 0.2286 | 0.4565 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8767 | 0.8036 | 0.9375 |
| M3 | CrossAttn3 | norm | AUPRC | 0.5148 | 0.3246 | 0.7257 |
| M3 | CrossAttn3 | norm | Brier | 0.1528 | 0.1254 | 0.1830 |
| M3 | CrossAttn3 | norm | Accuracy | 0.7791 | 0.7151 | 0.8372 |
| M3 | CrossAttn3 | norm | F1 | 0.4722 | 0.3200 | 0.6001 |
| M3 | CrossAttn3 | crop80 | AUC-ROC | 0.8625 | 0.7673 | 0.9351 |
| M3 | CrossAttn3 | crop80 | AUPRC | 0.4930 | 0.2941 | 0.7099 |
| M3 | CrossAttn3 | crop80 | Brier | 0.2493 | 0.2077 | 0.2901 |
| M3 | CrossAttn3 | crop80 | Accuracy | 0.7442 | 0.6802 | 0.8081 |
| M3 | CrossAttn3 | crop80 | F1 | 0.4500 | 0.3014 | 0.5859 |
| M3 | CrossAttn3 | crop60 | AUC-ROC | 0.7893 | 0.6831 | 0.8795 |
| M3 | CrossAttn3 | crop60 | AUPRC | 0.3070 | 0.1871 | 0.5152 |
| M3 | CrossAttn3 | crop60 | Brier | 0.2062 | 0.1640 | 0.2501 |
| M3 | CrossAttn3 | crop60 | Accuracy | 0.7267 | 0.6570 | 0.7965 |
| M3 | CrossAttn3 | crop60 | F1 | 0.4337 | 0.2941 | 0.5647 |
| M3 | CrossAttn3 | excl_extreme | AUC-ROC | 0.8367 | 0.7519 | 0.9076 |
| M3 | CrossAttn3 | excl_extreme | AUPRC | 0.4310 | 0.2491 | 0.6443 |
| M3 | CrossAttn3 | excl_extreme | Brier | 0.2130 | 0.1754 | 0.2527 |
| M3 | CrossAttn3 | excl_extreme | Accuracy | 0.6494 | 0.5714 | 0.7208 |
| M3 | CrossAttn3 | excl_extreme | F1 | 0.4255 | 0.2963 | 0.5455 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-len128 | 0.8325 | 0.8404 | +0.0079 | -0.209 | 8.342e-01 | ns |
| M1-LR vs M2-norm | 0.8325 | 0.8786 | +0.0460 | -1.659 | 9.718e-02 | † |
| M1-LR vs M2-crop80 | 0.8325 | 0.8307 | -0.0019 | 0.068 | 9.460e-01 | ns |
| M1-LR vs M2-crop60 | 0.8325 | 0.8934 | +0.0609 | -2.148 | 3.174e-02 | * |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-len128 | 0.8325 | 0.8798 | +0.0473 | -1.393 | 1.637e-01 | ns |
| M1-LR vs M3-norm | 0.8325 | 0.8767 | +0.0442 | -1.371 | 1.702e-01 | ns |
| M1-LR vs M3-crop80 | 0.8325 | 0.8625 | +0.0300 | -0.916 | 3.594e-01 | ns |
| M1-LR vs M3-crop60 | 0.8325 | 0.7893 | -0.0432 | 0.965 | 3.347e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len128 vs M2_2-len128 | 0.8404 | 0.8114 | -0.0290 | 0.773 | 4.398e-01 | ns |
| M2-norm vs M2_2-norm | 0.8786 | 0.7827 | -0.0959 | 2.298 | 2.154e-02 | * |
| M2-crop80 vs M2_2-crop80 | 0.8307 | 0.8329 | +0.0022 | -0.100 | 9.207e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.8934 | 0.8127 | -0.0807 | 2.235 | 2.541e-02 | * |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8790 | 0.5525 | -0.3265 | 4.292 | 1.771e-05 | *** |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len128 vs M3-len128 | 0.8404 | 0.8798 | +0.0394 | -1.406 | 1.598e-01 | ns |
| M2-norm vs M3-norm | 0.8786 | 0.8767 | -0.0019 | 0.129 | 8.977e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8307 | 0.8625 | +0.0319 | -1.931 | 5.351e-02 | † |
| M2-crop60 vs M3-crop60 | 0.8934 | 0.7893 | -0.1041 | 3.130 | 1.746e-03 | ** |
| M2-excl_extreme vs M3-excl_extreme | 0.8790 | 0.8367 | -0.0422 | 1.897 | 5.788e-02 | † |

