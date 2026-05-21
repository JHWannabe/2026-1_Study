# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8325 | 0.5008 | 0.1804 | 0.6919 | 0.3765 |
| M2 | CrossAttn | crop60/scale_both | 0.8600 | 0.5287 | 0.2170 | 0.6744 | 0.4167 |
| M2_2 | CrossAttn | norm/scale_both | 0.8691 | 0.5369 | 0.1938 | 0.6919 | 0.4301 |
| M3 | CrossAttn3 | norm/scale_both | 0.8046 | 0.4923 | 0.1819 | 0.7229 | 0.4250 |

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
| len128/scale_both | 0.8559 | 0.5381 | 0.1747 | 0.6919 | 0.4301 |
| crop80/scale_both | 0.8587 | 0.4936 | 0.1753 | 0.7151 | 0.4235 |
| **crop60/scale_both** | 0.8600 | 0.5287 | 0.2170 | 0.6744 | 0.4167 |
| norm/scale_both | 0.8240 | 0.3251 | 0.1986 | 0.7093 | 0.4186 |
| excl_extreme/scale_both | 0.8453 | 0.4630 | 0.1545 | 0.7597 | 0.4932 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.7969 | 0.3137 | 0.1615 | 0.7733 | 0.4507 |
| crop80/scale_both | 0.8146 | 0.3235 | 0.1877 | 0.7384 | 0.4578 |
| crop60/scale_both | 0.7997 | 0.3676 | 0.2047 | 0.6512 | 0.3617 |
| **norm/scale_both** | 0.8691 | 0.5369 | 0.1938 | 0.6919 | 0.4301 |
| excl_extreme/scale_both | 0.8652 | 0.4839 | 0.1417 | 0.8052 | 0.4828 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.7550 | 0.4024 | 0.1833 | 0.7048 | 0.3797 |
| crop80/scale_both | 0.6949 | 0.2852 | 0.1955 | 0.7410 | 0.3385 |
| crop60/scale_both | 0.7353 | 0.2926 | 0.2090 | 0.6747 | 0.3721 |
| **norm/scale_both** | 0.8046 | 0.4923 | 0.1819 | 0.7229 | 0.4250 |
| excl_extreme/scale_both | 0.7634 | 0.3976 | 0.1525 | 0.7162 | 0.3438 |

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
| AUC-ROC  | 0.8113 | 0.8125 | +0.0011 | -0.245 | 8.19e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3775 | -0.0071 | 0.525 | 6.27e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1850 | +0.0031 | -0.274 | 7.98e-01 | 8.12e-01 |
| Accuracy  | 0.7292 | 0.6970 | -0.0322 | 1.315 | 2.59e-01 | 4.38e-01 |
| F1  | 0.3950 | 0.3933 | -0.0017 | 0.127 | 9.05e-01 | 1.00e+00 |

### crop80/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8115 | +0.0002 | -0.019 | 9.85e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3543 | -0.0304 | 1.000 | 3.74e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1711 | -0.0108 | 1.095 | 3.35e-01 | 4.38e-01 |
| Accuracy  | 0.7292 | 0.7291 | -0.0001 | 0.003 | 9.98e-01 | 1.00e+00 |
| F1  | 0.3950 | 0.3974 | +0.0024 | -0.090 | 9.33e-01 | 1.00e+00 |

### crop60/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8128 | +0.0015 | -0.154 | 8.85e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3867 | +0.0020 | -0.067 | 9.50e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1746 | -0.0074 | 1.152 | 3.14e-01 | 4.38e-01 |
| Accuracy  | 0.7292 | 0.7132 | -0.0160 | 0.724 | 5.09e-01 | 6.25e-01 |
| F1  | 0.3950 | 0.3939 | -0.0011 | 0.053 | 9.60e-01 | 1.00e+00 |

### norm/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8253 | +0.0140 | -1.454 | 2.20e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.4304 | +0.0457 | -1.062 | 3.48e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1877 | +0.0058 | -0.461 | 6.69e-01 | 8.12e-01 |
| Accuracy  | 0.7292 | 0.7175 | -0.0117 | 0.476 | 6.59e-01 | 6.25e-01 |
| F1  | 0.3950 | 0.4122 | +0.0172 | -1.662 | 1.72e-01 | 1.88e-01 |

### excl_extreme/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8048 | -0.0065 | 1.036 | 3.59e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3562 | -0.0284 | 0.462 | 6.68e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1861 | +0.0042 | -0.474 | 6.60e-01 | 8.12e-01 |
| Accuracy  | 0.7292 | 0.7163 | -0.0129 | 0.639 | 5.58e-01 | 6.25e-01 |
| F1  | 0.3950 | 0.3909 | -0.0041 | 0.315 | 7.68e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len128/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8125 | 0.8304 | +0.0179 | -1.120 | 3.25e-01 | 3.12e-01 |
| AUPRC  | 0.3775 | 0.4462 | +0.0687 | -2.028 | 1.12e-01 | 1.88e-01 |
| Brier  | 0.1850 | 0.1891 | +0.0040 | -0.110 | 9.18e-01 | 1.00e+00 |
| Accuracy  | 0.6970 | 0.7030 | +0.0060 | -0.097 | 9.27e-01 | 1.00e+00 |
| F1  | 0.3933 | 0.4198 | +0.0265 | -0.543 | 6.16e-01 | 6.25e-01 |

#### Case: crop80/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8115 | 0.8384 | +0.0269 | -2.045 | 1.10e-01 | 1.25e-01 |
| AUPRC  | 0.3543 | 0.4211 | +0.0668 | -1.117 | 3.27e-01 | 4.38e-01 |
| Brier  | 0.1711 | 0.1615 | -0.0096 | 0.549 | 6.12e-01 | 6.25e-01 |
| Accuracy  | 0.7291 | 0.7636 | +0.0345 | -0.913 | 4.13e-01 | 3.12e-01 |
| F1  | 0.3974 | 0.4086 | +0.0112 | -0.327 | 7.60e-01 | 1.00e+00 |

#### Case: crop60/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8128 | 0.8308 | +0.0180 | -1.276 | 2.71e-01 | 3.12e-01 |
| AUPRC  | 0.3867 | 0.4047 | +0.0181 | -0.442 | 6.81e-01 | 8.12e-01 |
| Brier  | 0.1746 | 0.1632 | -0.0113 | 0.688 | 5.29e-01 | 8.12e-01 |
| Accuracy  | 0.7132 | 0.7424 | +0.0292 | -1.393 | 2.36e-01 | 3.12e-01 |
| F1  | 0.3939 | 0.4321 | +0.0382 | -1.291 | 2.66e-01 | 3.12e-01 |

#### Case: norm/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8253 | 0.8354 | +0.0101 | -0.682 | 5.33e-01 | 6.25e-01 |
| AUPRC  | 0.4304 | 0.4189 | -0.0114 | 0.247 | 8.17e-01 | 1.00e+00 |
| Brier  | 0.1877 | 0.1554 | -0.0323 | 1.405 | 2.33e-01 | 1.88e-01 |
| Accuracy  | 0.7175 | 0.7712 | +0.0537 | -1.446 | 2.22e-01 | 3.12e-01 |
| F1  | 0.4122 | 0.4275 | +0.0154 | -0.415 | 6.99e-01 | 1.00e+00 |

#### Case: excl_extreme/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8048 | 0.8458 | +0.0410 | -2.393 | 7.49e-02 | 6.25e-02 |
| AUPRC  | 0.3562 | 0.4240 | +0.0677 | -0.820 | 4.58e-01 | 6.25e-01 |
| Brier  | 0.1861 | 0.1693 | -0.0168 | 1.272 | 2.72e-01 | 3.12e-01 |
| Accuracy  | 0.7163 | 0.7274 | +0.0111 | -0.433 | 6.87e-01 | 8.12e-01 |
| F1  | 0.3909 | 0.3998 | +0.0089 | -0.275 | 7.97e-01 | 1.00e+00 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8304 | +0.0190 | -1.102 | 3.32e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.4462 | +0.0615 | -1.649 | 1.75e-01 | 1.88e-01 |
| Brier  | 0.1819 | 0.1891 | +0.0072 | -0.216 | 8.40e-01 | 8.12e-01 |
| Accuracy  | 0.7292 | 0.7030 | -0.0262 | 0.453 | 6.74e-01 | 8.12e-01 |
| F1  | 0.3950 | 0.4198 | +0.0248 | -0.477 | 6.58e-01 | 6.25e-01 |

### crop80/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8384 | +0.0271 | -1.347 | 2.49e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.4211 | +0.0365 | -0.754 | 4.93e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1615 | -0.0204 | 1.407 | 2.32e-01 | 4.38e-01 |
| Accuracy  | 0.7292 | 0.7636 | +0.0344 | -0.974 | 3.85e-01 | 4.38e-01 |
| F1  | 0.3950 | 0.4086 | +0.0136 | -0.461 | 6.69e-01 | 6.25e-01 |

### crop60/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8308 | +0.0194 | -1.354 | 2.47e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.4047 | +0.0201 | -0.648 | 5.52e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1632 | -0.0187 | 0.877 | 4.30e-01 | 6.25e-01 |
| Accuracy  | 0.7292 | 0.7424 | +0.0132 | -0.418 | 6.97e-01 | 8.12e-01 |
| F1  | 0.3950 | 0.4321 | +0.0372 | -1.005 | 3.72e-01 | 4.38e-01 |

### norm/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8354 | +0.0241 | -1.575 | 1.90e-01 | 1.88e-01 |
| AUPRC  | 0.3847 | 0.4189 | +0.0343 | -1.147 | 3.15e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1554 | -0.0265 | 1.710 | 1.62e-01 | 3.12e-01 |
| Accuracy  | 0.7292 | 0.7712 | +0.0420 | -1.170 | 3.07e-01 | 4.38e-01 |
| F1  | 0.3950 | 0.4275 | +0.0326 | -0.820 | 4.58e-01 | 6.25e-01 |

### excl_extreme/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8458 | +0.0344 | -2.102 | 1.03e-01 | 1.25e-01 |
| AUPRC  | 0.3847 | 0.4240 | +0.0393 | -0.550 | 6.12e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1693 | -0.0126 | 0.798 | 4.69e-01 | 6.25e-01 |
| Accuracy  | 0.7292 | 0.7274 | -0.0018 | 0.048 | 9.64e-01 | 1.00e+00 |
| F1  | 0.3950 | 0.3998 | +0.0049 | -0.119 | 9.11e-01 | 1.00e+00 |

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
| M2 | CrossAttn | len128/scale_both | AUC-ROC | 0.8559 | 0.7598 | 0.9319 |
| M2 | CrossAttn | len128/scale_both | AUPRC | 0.5381 | 0.3284 | 0.7334 |
| M2 | CrossAttn | len128/scale_both | Brier | 0.1747 | 0.1458 | 0.2037 |
| M2 | CrossAttn | len128/scale_both | Accuracy | 0.6919 | 0.6221 | 0.7558 |
| M2 | CrossAttn | len128/scale_both | F1 | 0.4301 | 0.2963 | 0.5532 |
| M2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.8587 | 0.7729 | 0.9273 |
| M2 | CrossAttn | crop80/scale_both | AUPRC | 0.4936 | 0.2912 | 0.6929 |
| M2 | CrossAttn | crop80/scale_both | Brier | 0.1753 | 0.1438 | 0.2090 |
| M2 | CrossAttn | crop80/scale_both | Accuracy | 0.7151 | 0.6453 | 0.7791 |
| M2 | CrossAttn | crop80/scale_both | F1 | 0.4235 | 0.2820 | 0.5455 |
| M2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.8600 | 0.7636 | 0.9325 |
| M2 | CrossAttn | crop60/scale_both | AUPRC | 0.5287 | 0.3291 | 0.7136 |
| M2 | CrossAttn | crop60/scale_both | Brier | 0.2170 | 0.1761 | 0.2590 |
| M2 | CrossAttn | crop60/scale_both | Accuracy | 0.6744 | 0.5988 | 0.7442 |
| M2 | CrossAttn | crop60/scale_both | F1 | 0.4167 | 0.2830 | 0.5371 |
| M2 | CrossAttn | norm/scale_both | AUC-ROC | 0.8240 | 0.7448 | 0.8921 |
| M2 | CrossAttn | norm/scale_both | AUPRC | 0.3251 | 0.1998 | 0.5302 |
| M2 | CrossAttn | norm/scale_both | Brier | 0.1986 | 0.1576 | 0.2414 |
| M2 | CrossAttn | norm/scale_both | Accuracy | 0.7093 | 0.6395 | 0.7791 |
| M2 | CrossAttn | norm/scale_both | F1 | 0.4186 | 0.2784 | 0.5437 |
| M2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8453 | 0.7613 | 0.9122 |
| M2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.4630 | 0.2807 | 0.6734 |
| M2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1545 | 0.1318 | 0.1788 |
| M2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.7597 | 0.6883 | 0.8247 |
| M2 | CrossAttn | excl_extreme/scale_both | F1 | 0.4932 | 0.3428 | 0.6269 |
| M2_2 | CrossAttn | len128/scale_both | AUC-ROC | 0.7969 | 0.6897 | 0.8874 |
| M2_2 | CrossAttn | len128/scale_both | AUPRC | 0.3137 | 0.1926 | 0.5216 |
| M2_2 | CrossAttn | len128/scale_both | Brier | 0.1615 | 0.1243 | 0.2016 |
| M2_2 | CrossAttn | len128/scale_both | Accuracy | 0.7733 | 0.7035 | 0.8372 |
| M2_2 | CrossAttn | len128/scale_both | F1 | 0.4507 | 0.2899 | 0.5883 |
| M2_2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.8146 | 0.7060 | 0.8999 |
| M2_2 | CrossAttn | crop80/scale_both | AUPRC | 0.3235 | 0.1992 | 0.5311 |
| M2_2 | CrossAttn | crop80/scale_both | Brier | 0.1877 | 0.1501 | 0.2294 |
| M2_2 | CrossAttn | crop80/scale_both | Accuracy | 0.7384 | 0.6686 | 0.8023 |
| M2_2 | CrossAttn | crop80/scale_both | F1 | 0.4578 | 0.3132 | 0.5854 |
| M2_2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.7997 | 0.6857 | 0.8922 |
| M2_2 | CrossAttn | crop60/scale_both | AUPRC | 0.3676 | 0.2129 | 0.5897 |
| M2_2 | CrossAttn | crop60/scale_both | Brier | 0.2047 | 0.1655 | 0.2478 |
| M2_2 | CrossAttn | crop60/scale_both | Accuracy | 0.6512 | 0.5756 | 0.7209 |
| M2_2 | CrossAttn | crop60/scale_both | F1 | 0.3617 | 0.2326 | 0.4842 |
| M2_2 | CrossAttn | norm/scale_both | AUC-ROC | 0.8691 | 0.7783 | 0.9401 |
| M2_2 | CrossAttn | norm/scale_both | AUPRC | 0.5369 | 0.3418 | 0.7260 |
| M2_2 | CrossAttn | norm/scale_both | Brier | 0.1938 | 0.1592 | 0.2301 |
| M2_2 | CrossAttn | norm/scale_both | Accuracy | 0.6919 | 0.6221 | 0.7560 |
| M2_2 | CrossAttn | norm/scale_both | F1 | 0.4301 | 0.2973 | 0.5524 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8652 | 0.7622 | 0.9425 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.4839 | 0.2699 | 0.6997 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1417 | 0.1080 | 0.1789 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.8052 | 0.7403 | 0.8636 |
| M2_2 | CrossAttn | excl_extreme/scale_both | F1 | 0.4828 | 0.3103 | 0.6250 |
| M3 | CrossAttn3 | len128/scale_both | AUC-ROC | 0.7550 | 0.6274 | 0.8762 |
| M3 | CrossAttn3 | len128/scale_both | AUPRC | 0.4024 | 0.2107 | 0.6262 |
| M3 | CrossAttn3 | len128/scale_both | Brier | 0.1833 | 0.1409 | 0.2238 |
| M3 | CrossAttn3 | len128/scale_both | Accuracy | 0.7048 | 0.6386 | 0.7771 |
| M3 | CrossAttn3 | len128/scale_both | F1 | 0.3797 | 0.2352 | 0.5122 |
| M3 | CrossAttn3 | crop80/scale_both | AUC-ROC | 0.6949 | 0.5668 | 0.8168 |
| M3 | CrossAttn3 | crop80/scale_both | AUPRC | 0.2852 | 0.1518 | 0.5069 |
| M3 | CrossAttn3 | crop80/scale_both | Brier | 0.1955 | 0.1464 | 0.2470 |
| M3 | CrossAttn3 | crop80/scale_both | Accuracy | 0.7410 | 0.6747 | 0.8072 |
| M3 | CrossAttn3 | crop80/scale_both | F1 | 0.3385 | 0.1754 | 0.4865 |
| M3 | CrossAttn3 | crop60/scale_both | AUC-ROC | 0.7353 | 0.6051 | 0.8497 |
| M3 | CrossAttn3 | crop60/scale_both | AUPRC | 0.2926 | 0.1648 | 0.5047 |
| M3 | CrossAttn3 | crop60/scale_both | Brier | 0.2090 | 0.1637 | 0.2546 |
| M3 | CrossAttn3 | crop60/scale_both | Accuracy | 0.6747 | 0.6083 | 0.7470 |
| M3 | CrossAttn3 | crop60/scale_both | F1 | 0.3721 | 0.2338 | 0.4938 |
| M3 | CrossAttn3 | norm/scale_both | AUC-ROC | 0.8046 | 0.6887 | 0.9061 |
| M3 | CrossAttn3 | norm/scale_both | AUPRC | 0.4923 | 0.2969 | 0.6961 |
| M3 | CrossAttn3 | norm/scale_both | Brier | 0.1819 | 0.1382 | 0.2269 |
| M3 | CrossAttn3 | norm/scale_both | Accuracy | 0.7229 | 0.6566 | 0.7892 |
| M3 | CrossAttn3 | norm/scale_both | F1 | 0.4250 | 0.2740 | 0.5480 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUC-ROC | 0.7634 | 0.6421 | 0.8683 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUPRC | 0.3976 | 0.2069 | 0.6213 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Brier | 0.1525 | 0.1198 | 0.1854 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Accuracy | 0.7162 | 0.6419 | 0.7905 |
| M3 | CrossAttn3 | excl_extreme/scale_both | F1 | 0.3438 | 0.1818 | 0.4918 |

