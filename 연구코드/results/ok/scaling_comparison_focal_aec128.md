# Scaling Comparison — Test Set Performance (AEC 128pt, FocalLoss)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |
| M2 | CrossAttn | excl_extreme/scale_clinic | 0.8854 | 0.5782 | 0.1842 | 0.7013 | 0.4390 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | 0.8364 | 0.3969 | 0.1513 | 0.8247 | 0.4000 |
| M3 | CrossAttn3 | crop80/scale_clinic | 0.8584 | 0.4908 | 0.1783 | 0.7209 | 0.4146 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_clinic** | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm/scale_clinic | 0.8739 | 0.5089 | 0.2213 | 0.6221 | 0.3810 |
| **excl_extreme/scale_clinic** | 0.8854 | 0.5782 | 0.1842 | 0.7013 | 0.4390 |
| len128/scale_clinic | 0.7909 | 0.3427 | 0.1905 | 0.7035 | 0.4000 |
| crop80/scale_clinic | 0.8754 | 0.5892 | 0.2240 | 0.5640 | 0.3478 |
| crop60/scale_clinic | 0.8764 | 0.5913 | 0.1884 | 0.6163 | 0.3774 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm/scale_clinic | 0.7975 | 0.3547 | 0.1538 | 0.7384 | 0.4304 |
| **excl_extreme/scale_clinic** | 0.8364 | 0.3969 | 0.1513 | 0.8247 | 0.4000 |
| len128/scale_clinic | 0.7733 | 0.3328 | 0.1941 | 0.7093 | 0.4048 |
| crop80/scale_clinic | 0.8124 | 0.3756 | 0.1649 | 0.7965 | 0.4928 |
| crop60/scale_clinic | 0.8117 | 0.3289 | 0.1953 | 0.7035 | 0.4138 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm/scale_clinic | 0.8448 | 0.4568 | 0.1983 | 0.6686 | 0.3871 |
| excl_extreme/scale_clinic | 0.8443 | 0.5622 | 0.2145 | 0.6104 | 0.3878 |
| len128/scale_clinic | 0.8335 | 0.4313 | 0.2075 | 0.6279 | 0.3846 |
| **crop80/scale_clinic** | 0.8584 | 0.4908 | 0.1783 | 0.7209 | 0.4146 |
| crop60/scale_clinic | 0.8556 | 0.5727 | 0.1985 | 0.6395 | 0.3800 |

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
| AUC-ROC  | 0.8113 | 0.8203 | +0.0090 | -1.896 | 1.31e-01 | 1.25e-01 |
| AUPRC  | 0.3847 | 0.4121 | +0.0275 | -1.328 | 2.55e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1832 | +0.0013 | -0.078 | 9.42e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7801 | +0.0246 | -0.489 | 6.50e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4762 | +0.0248 | -0.603 | 5.79e-01 | 8.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.7927 | -0.0186 | 1.040 | 3.57e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3523 | -0.0323 | 0.631 | 5.62e-01 | 6.25e-01 |
| Brier * | 0.1819 | 0.2128 | +0.0309 | -2.785 | 4.96e-02 | 1.25e-01 |
| Accuracy * | 0.7555 | 0.7002 | -0.0552 | 3.218 | 3.23e-02 | 6.25e-02 |
| F1 * | 0.4514 | 0.3965 | -0.0549 | 3.053 | 3.79e-02 | 1.25e-01 |

### len128/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8240 | +0.0126 | -1.145 | 3.16e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3897 | +0.0050 | -0.183 | 8.64e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1771 | -0.0048 | 1.021 | 3.65e-01 | 4.38e-01 |
| Accuracy  | 0.7555 | 0.7366 | -0.0189 | 0.359 | 7.38e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4405 | -0.0109 | 0.293 | 7.84e-01 | 8.12e-01 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8154 | +0.0041 | -0.414 | 7.00e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3889 | +0.0042 | -0.696 | 5.25e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1819 | +0.0000 | -0.002 | 9.99e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7570 | +0.0015 | -0.045 | 9.66e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4524 | +0.0010 | -0.032 | 9.76e-01 | 1.00e+00 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8062 | -0.0051 | 0.411 | 7.02e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3664 | -0.0183 | 1.627 | 1.79e-01 | 1.88e-01 |
| Brier  | 0.1819 | 0.1826 | +0.0007 | -0.112 | 9.16e-01 | 8.12e-01 |
| Accuracy  | 0.7555 | 0.6752 | -0.0803 | 2.046 | 1.10e-01 | 1.25e-01 |
| F1  | 0.4514 | 0.4048 | -0.0466 | 1.641 | 1.76e-01 | 1.88e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8203 | 0.8084 | -0.0119 | 1.747 | 1.56e-01 | 3.12e-01 |
| AUPRC  | 0.4121 | 0.3961 | -0.0160 | 0.696 | 5.25e-01 | 4.38e-01 |
| Brier  | 0.1832 | 0.1867 | +0.0036 | -0.229 | 8.30e-01 | 1.00e+00 |
| Accuracy  | 0.7801 | 0.7817 | +0.0016 | -0.035 | 9.74e-01 | 1.00e+00 |
| F1  | 0.4762 | 0.4651 | -0.0112 | 0.267 | 8.03e-01 | 1.00e+00 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.7927 | 0.8077 | +0.0150 | -1.993 | 1.17e-01 | 3.12e-01 |
| AUPRC  | 0.3523 | 0.3887 | +0.0363 | -1.828 | 1.42e-01 | 3.12e-01 |
| Brier  | 0.2128 | 0.1927 | -0.0201 | 1.077 | 3.42e-01 | 6.25e-01 |
| Accuracy  | 0.7002 | 0.7196 | +0.0194 | -1.046 | 3.55e-01 | 5.00e-01 |
| F1  | 0.3965 | 0.4187 | +0.0221 | -2.001 | 1.16e-01 | 2.50e-01 |

#### Case: len128/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8240 | 0.8172 | -0.0068 | 0.624 | 5.67e-01 | 6.25e-01 |
| AUPRC  | 0.3897 | 0.4169 | +0.0272 | -0.965 | 3.89e-01 | 4.38e-01 |
| Brier  | 0.1771 | 0.1663 | -0.0108 | 1.150 | 3.14e-01 | 3.12e-01 |
| Accuracy  | 0.7366 | 0.7597 | +0.0232 | -0.899 | 4.19e-01 | 4.38e-01 |
| F1  | 0.4405 | 0.4507 | +0.0102 | -0.509 | 6.38e-01 | 1.00e+00 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8154 | 0.8054 | -0.0101 | 1.008 | 3.70e-01 | 6.25e-01 |
| AUPRC  | 0.3889 | 0.3813 | -0.0076 | 0.298 | 7.81e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1743 | -0.0076 | 0.489 | 6.50e-01 | 8.12e-01 |
| Accuracy  | 0.7570 | 0.7248 | -0.0322 | 0.879 | 4.29e-01 | 6.25e-01 |
| F1  | 0.4524 | 0.4354 | -0.0170 | 0.631 | 5.62e-01 | 6.25e-01 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8062 | 0.8011 | -0.0051 | 2.295 | 8.34e-02 | 1.88e-01 |
| AUPRC  | 0.3664 | 0.3484 | -0.0180 | 0.550 | 6.12e-01 | 8.12e-01 |
| Brier  | 0.1826 | 0.1963 | +0.0137 | -0.778 | 4.80e-01 | 4.38e-01 |
| Accuracy † | 0.6752 | 0.7540 | +0.0788 | -2.554 | 6.31e-02 | 1.25e-01 |
| F1 * | 0.4048 | 0.4508 | +0.0460 | -3.360 | 2.83e-02 | 1.25e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8084 | -0.0029 | 0.282 | 7.92e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3961 | +0.0114 | -0.509 | 6.37e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1867 | +0.0048 | -1.165 | 3.09e-01 | 4.38e-01 |
| Accuracy  | 0.7555 | 0.7817 | +0.0263 | -0.867 | 4.35e-01 | 5.00e-01 |
| F1  | 0.4514 | 0.4651 | +0.0136 | -0.413 | 7.01e-01 | 6.25e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8077 | -0.0036 | 0.207 | 8.46e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3887 | +0.0040 | -0.071 | 9.47e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.1927 | +0.0108 | -0.613 | 5.73e-01 | 8.12e-01 |
| Accuracy * | 0.7555 | 0.7196 | -0.0359 | 4.152 | 1.42e-02 | 6.25e-02 |
| F1 † | 0.4514 | 0.4187 | -0.0328 | 2.528 | 6.48e-02 | 1.25e-01 |

### len128/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8172 | +0.0058 | -0.478 | 6.57e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.4169 | +0.0322 | -1.943 | 1.24e-01 | 1.25e-01 |
| Brier † | 0.1819 | 0.1663 | -0.0157 | 2.210 | 9.16e-02 | 1.25e-01 |
| Accuracy  | 0.7555 | 0.7597 | +0.0043 | -0.078 | 9.42e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4507 | -0.0008 | 0.019 | 9.86e-01 | 1.00e+00 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8054 | -0.0059 | 0.360 | 7.37e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.3813 | -0.0034 | 0.149 | 8.89e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1743 | -0.0076 | 0.858 | 4.39e-01 | 4.38e-01 |
| Accuracy  | 0.7555 | 0.7248 | -0.0307 | 1.026 | 3.63e-01 | 3.12e-01 |
| F1  | 0.4514 | 0.4354 | -0.0160 | 0.690 | 5.28e-01 | 1.00e+00 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8011 | -0.0103 | 0.883 | 4.27e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3484 | -0.0362 | 1.371 | 2.42e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1963 | +0.0144 | -1.007 | 3.71e-01 | 4.38e-01 |
| Accuracy  | 0.7555 | 0.7540 | -0.0015 | 0.077 | 9.42e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4508 | -0.0006 | 0.036 | 9.73e-01 | 8.12e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_clinic | AUC-ROC | 0.8325 | 0.7386 | 0.9091 |
| M1 | LR | scale_clinic | AUPRC | 0.5008 | 0.2965 | 0.6937 |
| M1 | LR | scale_clinic | Brier | 0.1804 | 0.1527 | 0.2118 |
| M1 | LR | scale_clinic | Accuracy | 0.6977 | 0.6279 | 0.7674 |
| M1 | LR | scale_clinic | F1 | 0.3953 | 0.2535 | 0.5228 |
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8739 | 0.7991 | 0.9358 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.5089 | 0.3128 | 0.7135 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.2213 | 0.2061 | 0.2357 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.6221 | 0.5523 | 0.6919 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.3810 | 0.2574 | 0.4957 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8854 | 0.8086 | 0.9477 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.5782 | 0.3750 | 0.7937 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1842 | 0.1663 | 0.2028 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7013 | 0.6234 | 0.7727 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4390 | 0.2933 | 0.5679 |
| M2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.7909 | 0.7029 | 0.8676 |
| M2 | CrossAttn | len128/scale_clinic | AUPRC | 0.3427 | 0.1885 | 0.5323 |
| M2 | CrossAttn | len128/scale_clinic | Brier | 0.1905 | 0.1636 | 0.2180 |
| M2 | CrossAttn | len128/scale_clinic | Accuracy | 0.7035 | 0.6337 | 0.7733 |
| M2 | CrossAttn | len128/scale_clinic | F1 | 0.4000 | 0.2564 | 0.5317 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8754 | 0.7900 | 0.9408 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.5892 | 0.3742 | 0.7982 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.2240 | 0.2077 | 0.2399 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.5640 | 0.4884 | 0.6395 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.3478 | 0.2342 | 0.4602 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8764 | 0.7949 | 0.9426 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.5913 | 0.3815 | 0.7724 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.1884 | 0.1712 | 0.2060 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6163 | 0.5407 | 0.6919 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.3774 | 0.2529 | 0.4912 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.7975 | 0.6971 | 0.8862 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.3547 | 0.2051 | 0.5569 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.1538 | 0.1301 | 0.1786 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7384 | 0.6686 | 0.8023 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.4304 | 0.2817 | 0.5648 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8364 | 0.7377 | 0.9167 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3969 | 0.2105 | 0.6202 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1513 | 0.1324 | 0.1715 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.8247 | 0.7597 | 0.8831 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4000 | 0.2000 | 0.5818 |
| M2_2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.7733 | 0.6636 | 0.8648 |
| M2_2 | CrossAttn | len128/scale_clinic | AUPRC | 0.3328 | 0.1884 | 0.5468 |
| M2_2 | CrossAttn | len128/scale_clinic | Brier | 0.1941 | 0.1707 | 0.2189 |
| M2_2 | CrossAttn | len128/scale_clinic | Accuracy | 0.7093 | 0.6395 | 0.7733 |
| M2_2 | CrossAttn | len128/scale_clinic | F1 | 0.4048 | 0.2597 | 0.5349 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8124 | 0.6936 | 0.9109 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.3756 | 0.2266 | 0.6106 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.1649 | 0.1390 | 0.1947 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.7965 | 0.7326 | 0.8547 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.4928 | 0.3385 | 0.6364 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8117 | 0.7078 | 0.8968 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3289 | 0.2011 | 0.5396 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.1953 | 0.1711 | 0.2213 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.7035 | 0.6337 | 0.7676 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.4138 | 0.2769 | 0.5421 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8448 | 0.7507 | 0.9238 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.4568 | 0.2790 | 0.6898 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.1983 | 0.1762 | 0.2190 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.6686 | 0.5988 | 0.7384 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.3871 | 0.2580 | 0.5094 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8443 | 0.7554 | 0.9217 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.5622 | 0.3550 | 0.7756 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.2145 | 0.1973 | 0.2329 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.6104 | 0.5325 | 0.6818 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.3878 | 0.2637 | 0.5049 |
| M3 | CrossAttn3 | len128/scale_clinic | AUC-ROC | 0.8335 | 0.7464 | 0.9057 |
| M3 | CrossAttn3 | len128/scale_clinic | AUPRC | 0.4313 | 0.2483 | 0.6365 |
| M3 | CrossAttn3 | len128/scale_clinic | Brier | 0.2075 | 0.1866 | 0.2284 |
| M3 | CrossAttn3 | len128/scale_clinic | Accuracy | 0.6279 | 0.5523 | 0.7035 |
| M3 | CrossAttn3 | len128/scale_clinic | F1 | 0.3846 | 0.2609 | 0.5000 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8584 | 0.7773 | 0.9291 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.4908 | 0.2961 | 0.7125 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.1783 | 0.1615 | 0.1953 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.7209 | 0.6512 | 0.7849 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.4146 | 0.2667 | 0.5432 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.8556 | 0.7645 | 0.9345 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.5727 | 0.3625 | 0.7839 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.1985 | 0.1792 | 0.2182 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.6395 | 0.5640 | 0.7151 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.3800 | 0.2532 | 0.5000 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-norm | 0.8325 | 0.8739 | +0.0413 | -1.758 | 7.872e-02 | † |
| M1-LR vs M2-len128 | 0.8325 | 0.7909 | -0.0416 | 1.087 | 2.770e-01 | ns |
| M1-LR vs M2-crop80 | 0.8325 | 0.8754 | +0.0429 | -1.473 | 1.407e-01 | ns |
| M1-LR vs M2-crop60 | 0.8325 | 0.8764 | +0.0438 | -1.473 | 1.407e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-norm | 0.8325 | 0.8448 | +0.0123 | -0.284 | 7.761e-01 | ns |
| M1-LR vs M3-len128 | 0.8325 | 0.8335 | +0.0009 | -0.030 | 9.760e-01 | ns |
| M1-LR vs M3-crop80 | 0.8325 | 0.8584 | +0.0259 | -0.654 | 5.129e-01 | ns |
| M1-LR vs M3-crop60 | 0.8325 | 0.8556 | +0.0230 | -0.579 | 5.626e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M2_2-norm | 0.8739 | 0.7975 | -0.0763 | 2.672 | 7.549e-03 | ** |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8854 | 0.5503 | -0.3351 | 5.058 | 4.245e-07 | *** |
| M2-len128 vs M2_2-len128 | 0.7909 | 0.7733 | -0.0177 | 0.356 | 7.219e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.8754 | 0.8124 | -0.0631 | 1.561 | 1.185e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.8764 | 0.8117 | -0.0646 | 1.780 | 7.515e-02 | † |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M3-norm | 0.8739 | 0.8448 | -0.0290 | 1.046 | 2.956e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.8854 | 0.8443 | -0.0412 | 1.412 | 1.580e-01 | ns |
| M2-len128 vs M3-len128 | 0.7909 | 0.8335 | +0.0426 | -1.173 | 2.407e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8754 | 0.8584 | -0.0170 | 0.761 | 4.465e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.8764 | 0.8556 | -0.0208 | 1.050 | 2.939e-01 | ns |

