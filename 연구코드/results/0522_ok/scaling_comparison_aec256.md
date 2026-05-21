# Scaling Comparison — Test Set Performance (AEC 256pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8325 | 0.5008 | 0.1804 | 0.6919 | 0.3765 |
| M2 | CrossAttn | excl_extreme/scale_clinic | 0.8833 | 0.4842 | 0.1748 | 0.7143 | 0.4762 |
| M2_2 | CrossAttn | norm/scale_clinic | 0.8776 | 0.5306 | 0.1794 | 0.6860 | 0.4255 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | 0.8783 | 0.5899 | 0.2247 | 0.6169 | 0.4040 |

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
| len256/scale_clinic | 0.8426 | 0.4346 | 0.1933 | 0.6744 | 0.4043 |
| crop80/scale_clinic | 0.8360 | 0.4939 | 0.2305 | 0.6453 | 0.3960 |
| crop60/scale_clinic | 0.8625 | 0.4912 | 0.1865 | 0.6860 | 0.4130 |
| norm/scale_clinic | 0.8666 | 0.4995 | 0.1829 | 0.6977 | 0.4091 |
| **excl_extreme/scale_clinic** | 0.8833 | 0.4842 | 0.1748 | 0.7143 | 0.4762 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_clinic | 0.8332 | 0.4324 | 0.1694 | 0.7209 | 0.4146 |
| crop80/scale_clinic | 0.8272 | 0.4165 | 0.2015 | 0.7209 | 0.4419 |
| crop60/scale_clinic | 0.8086 | 0.3455 | 0.2272 | 0.6802 | 0.4086 |
| **norm/scale_clinic** | 0.8776 | 0.5306 | 0.1794 | 0.6860 | 0.4255 |
| excl_extreme/scale_clinic | 0.8311 | 0.3516 | 0.1845 | 0.7532 | 0.4242 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_clinic | 0.8581 | 0.5266 | 0.1898 | 0.6628 | 0.3696 |
| crop80/scale_clinic | 0.8575 | 0.5037 | 0.2305 | 0.6453 | 0.3960 |
| crop60/scale_clinic | 0.8518 | 0.4873 | 0.1843 | 0.7093 | 0.4318 |
| norm/scale_clinic | 0.8423 | 0.4509 | 0.1925 | 0.6860 | 0.4255 |
| **excl_extreme/scale_clinic** | 0.8783 | 0.5899 | 0.2247 | 0.6169 | 0.4040 |

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

### len256/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8159 | +0.0046 | -0.516 | 6.33e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3749 | -0.0098 | 0.697 | 5.24e-01 | 6.25e-01 |
| Brier *** | 0.1819 | 0.1521 | -0.0298 | 12.633 | 2.26e-04 | 6.25e-02 |
| Accuracy † | 0.7292 | 0.7817 | +0.0525 | -2.657 | 5.65e-02 | 1.25e-01 |
| F1  | 0.3950 | 0.4172 | +0.0222 | -1.137 | 3.19e-01 | 4.38e-01 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8072 | -0.0042 | 0.378 | 7.24e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3789 | -0.0058 | 0.443 | 6.80e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.2013 | +0.0194 | -0.982 | 3.82e-01 | 6.25e-01 |
| Accuracy  | 0.7292 | 0.6726 | -0.0566 | 1.633 | 1.78e-01 | 3.12e-01 |
| F1  | 0.3950 | 0.3868 | -0.0082 | 0.946 | 3.98e-01 | 6.25e-01 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8154 | +0.0041 | -0.399 | 7.10e-01 | 8.12e-01 |
| AUPRC † | 0.3847 | 0.3554 | -0.0292 | 2.604 | 5.98e-02 | 1.25e-01 |
| Brier  | 0.1819 | 0.1940 | +0.0121 | -1.281 | 2.70e-01 | 4.38e-01 |
| Accuracy  | 0.7292 | 0.6870 | -0.0422 | 1.970 | 1.20e-01 | 1.88e-01 |
| F1  | 0.3950 | 0.3947 | -0.0003 | 0.014 | 9.89e-01 | 1.00e+00 |

### norm/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8219 | +0.0106 | -1.346 | 2.50e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.4315 | +0.0468 | -1.562 | 1.93e-01 | 1.88e-01 |
| Brier  | 0.1819 | 0.1945 | +0.0126 | -0.680 | 5.34e-01 | 6.25e-01 |
| Accuracy  | 0.7292 | 0.6941 | -0.0352 | 1.015 | 3.67e-01 | 4.38e-01 |
| F1  | 0.3950 | 0.3851 | -0.0099 | 0.600 | 5.81e-01 | 4.38e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8092 | -0.0021 | 0.141 | 8.94e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3507 | -0.0339 | 0.715 | 5.14e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.2018 | +0.0199 | -1.040 | 3.57e-01 | 3.12e-01 |
| Accuracy  | 0.7292 | 0.6612 | -0.0680 | 1.755 | 1.54e-01 | 1.88e-01 |
| F1 † | 0.3950 | 0.3412 | -0.0538 | 2.774 | 5.01e-02 | 1.25e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len256/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8159 | 0.8116 | -0.0042 | 1.176 | 3.05e-01 | 4.38e-01 |
| AUPRC * | 0.3749 | 0.4057 | +0.0308 | -3.228 | 3.20e-02 | 6.25e-02 |
| Brier † | 0.1521 | 0.1823 | +0.0302 | -2.739 | 5.19e-02 | 6.25e-02 |
| Accuracy † | 0.7817 | 0.7117 | -0.0700 | 2.683 | 5.50e-02 | 1.25e-01 |
| F1  | 0.4172 | 0.3929 | -0.0243 | 1.223 | 2.88e-01 | 3.12e-01 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.7988 | -0.0084 | 1.673 | 1.70e-01 | 1.88e-01 |
| AUPRC * | 0.3789 | 0.3556 | -0.0233 | 2.778 | 4.99e-02 | 6.25e-02 |
| Brier  | 0.2013 | 0.1775 | -0.0238 | 1.157 | 3.12e-01 | 4.38e-01 |
| Accuracy  | 0.6726 | 0.7189 | +0.0463 | -1.080 | 3.41e-01 | 4.38e-01 |
| F1  | 0.3868 | 0.3869 | +0.0001 | -0.003 | 9.98e-01 | 1.00e+00 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8154 | 0.8004 | -0.0150 | 1.491 | 2.10e-01 | 3.12e-01 |
| AUPRC  | 0.3554 | 0.3970 | +0.0416 | -1.711 | 1.62e-01 | 1.25e-01 |
| Brier  | 0.1940 | 0.1965 | +0.0025 | -0.078 | 9.42e-01 | 8.12e-01 |
| Accuracy  | 0.6870 | 0.6942 | +0.0072 | -0.137 | 8.98e-01 | 8.75e-01 |
| F1  | 0.3947 | 0.3825 | -0.0122 | 0.401 | 7.09e-01 | 8.12e-01 |

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8219 | 0.8102 | -0.0117 | 1.448 | 2.21e-01 | 3.12e-01 |
| AUPRC  | 0.4315 | 0.3900 | -0.0415 | 1.385 | 2.38e-01 | 3.12e-01 |
| Brier  | 0.1945 | 0.1783 | -0.0162 | 0.706 | 5.19e-01 | 8.12e-01 |
| Accuracy  | 0.6941 | 0.7364 | +0.0424 | -0.904 | 4.17e-01 | 4.38e-01 |
| F1  | 0.3851 | 0.3977 | +0.0126 | -0.475 | 6.59e-01 | 6.25e-01 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8092 | 0.7954 | -0.0138 | 0.994 | 3.77e-01 | 4.38e-01 |
| AUPRC  | 0.3507 | 0.3394 | -0.0113 | 0.216 | 8.39e-01 | 1.00e+00 |
| Brier  | 0.2018 | 0.1650 | -0.0368 | 2.077 | 1.06e-01 | 1.25e-01 |
| Accuracy  | 0.6612 | 0.7343 | +0.0731 | -2.106 | 1.03e-01 | 1.25e-01 |
| F1  | 0.3412 | 0.3799 | +0.0387 | -1.030 | 3.61e-01 | 3.75e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len256/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8116 | +0.0003 | -0.030 | 9.77e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.4057 | +0.0211 | -1.213 | 2.92e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1823 | +0.0004 | -0.040 | 9.70e-01 | 8.12e-01 |
| Accuracy  | 0.7292 | 0.7117 | -0.0175 | 0.885 | 4.26e-01 | 8.12e-01 |
| F1  | 0.3950 | 0.3929 | -0.0021 | 0.163 | 8.79e-01 | 1.00e+00 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.7988 | -0.0126 | 1.232 | 2.85e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3556 | -0.0290 | 2.017 | 1.14e-01 | 1.88e-01 |
| Brier  | 0.1819 | 0.1775 | -0.0044 | 0.328 | 7.59e-01 | 6.25e-01 |
| Accuracy  | 0.7292 | 0.7189 | -0.0103 | 0.355 | 7.41e-01 | 8.12e-01 |
| F1  | 0.3950 | 0.3869 | -0.0081 | 0.379 | 7.24e-01 | 1.00e+00 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8004 | -0.0109 | 1.005 | 3.72e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3970 | +0.0124 | -0.487 | 6.51e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.1965 | +0.0145 | -0.587 | 5.89e-01 | 1.00e+00 |
| Accuracy  | 0.7292 | 0.6942 | -0.0351 | 1.023 | 3.64e-01 | 6.25e-01 |
| F1  | 0.3950 | 0.3825 | -0.0125 | 0.970 | 3.87e-01 | 8.12e-01 |

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8102 | -0.0011 | 0.167 | 8.76e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.3900 | +0.0053 | -0.294 | 7.84e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1783 | -0.0036 | 0.214 | 8.41e-01 | 8.12e-01 |
| Accuracy  | 0.7292 | 0.7364 | +0.0072 | -0.192 | 8.57e-01 | 6.25e-01 |
| F1  | 0.3950 | 0.3977 | +0.0027 | -0.087 | 9.35e-01 | 1.00e+00 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.7954 | -0.0159 | 1.127 | 3.23e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3394 | -0.0453 | 0.825 | 4.56e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1650 | -0.0169 | 1.422 | 2.28e-01 | 3.12e-01 |
| Accuracy  | 0.7292 | 0.7343 | +0.0051 | -0.150 | 8.88e-01 | 8.12e-01 |
| F1  | 0.3950 | 0.3799 | -0.0150 | 0.345 | 7.48e-01 | 6.25e-01 |

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
| M2 | CrossAttn | len256/scale_clinic | AUC-ROC | 0.8426 | 0.7626 | 0.9098 |
| M2 | CrossAttn | len256/scale_clinic | AUPRC | 0.4346 | 0.2570 | 0.6326 |
| M2 | CrossAttn | len256/scale_clinic | Brier | 0.1933 | 0.1597 | 0.2294 |
| M2 | CrossAttn | len256/scale_clinic | Accuracy | 0.6744 | 0.6047 | 0.7442 |
| M2 | CrossAttn | len256/scale_clinic | F1 | 0.4043 | 0.2708 | 0.5273 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8360 | 0.7541 | 0.9070 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.4939 | 0.2894 | 0.6875 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.2305 | 0.1980 | 0.2645 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6453 | 0.5756 | 0.7151 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.3960 | 0.2696 | 0.5192 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8625 | 0.7836 | 0.9274 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.4912 | 0.2942 | 0.6887 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.1865 | 0.1551 | 0.2202 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6860 | 0.6163 | 0.7558 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.4130 | 0.2791 | 0.5334 |
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8666 | 0.7840 | 0.9316 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4995 | 0.3121 | 0.7033 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.1829 | 0.1547 | 0.2133 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.6977 | 0.6279 | 0.7618 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.4091 | 0.2740 | 0.5377 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8833 | 0.8133 | 0.9440 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.4842 | 0.3111 | 0.7351 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1748 | 0.1401 | 0.2121 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7143 | 0.6429 | 0.7857 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4762 | 0.3333 | 0.6000 |
| M2_2 | CrossAttn | len256/scale_clinic | AUC-ROC | 0.8332 | 0.7384 | 0.9107 |
| M2_2 | CrossAttn | len256/scale_clinic | AUPRC | 0.4324 | 0.2511 | 0.6326 |
| M2_2 | CrossAttn | len256/scale_clinic | Brier | 0.1694 | 0.1415 | 0.1988 |
| M2_2 | CrossAttn | len256/scale_clinic | Accuracy | 0.7209 | 0.6512 | 0.7849 |
| M2_2 | CrossAttn | len256/scale_clinic | F1 | 0.4146 | 0.2703 | 0.5435 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8272 | 0.7160 | 0.9146 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.4165 | 0.2378 | 0.6182 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.2015 | 0.1670 | 0.2367 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.7209 | 0.6512 | 0.7849 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.4419 | 0.3055 | 0.5715 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8086 | 0.6947 | 0.8981 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3455 | 0.1986 | 0.5449 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.2272 | 0.1857 | 0.2717 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6802 | 0.6047 | 0.7500 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.4086 | 0.2785 | 0.5334 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8776 | 0.8018 | 0.9397 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.5306 | 0.3255 | 0.7212 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.1794 | 0.1504 | 0.2120 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.6860 | 0.6163 | 0.7558 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.4255 | 0.2916 | 0.5455 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8311 | 0.7296 | 0.9077 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3516 | 0.1748 | 0.5766 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1845 | 0.1509 | 0.2218 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7532 | 0.6818 | 0.8182 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4242 | 0.2608 | 0.5600 |
| M3 | CrossAttn3 | len256/scale_clinic | AUC-ROC | 0.8581 | 0.7656 | 0.9349 |
| M3 | CrossAttn3 | len256/scale_clinic | AUPRC | 0.5266 | 0.3196 | 0.7334 |
| M3 | CrossAttn3 | len256/scale_clinic | Brier | 0.1898 | 0.1550 | 0.2261 |
| M3 | CrossAttn3 | len256/scale_clinic | Accuracy | 0.6628 | 0.5872 | 0.7326 |
| M3 | CrossAttn3 | len256/scale_clinic | F1 | 0.3696 | 0.2368 | 0.4902 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8575 | 0.7681 | 0.9336 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.5037 | 0.3052 | 0.7421 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.2305 | 0.1896 | 0.2711 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.6453 | 0.5756 | 0.7151 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.3960 | 0.2680 | 0.5155 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.8518 | 0.7690 | 0.9244 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.4873 | 0.2886 | 0.7188 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.1843 | 0.1502 | 0.2201 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.7093 | 0.6395 | 0.7791 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.4318 | 0.2927 | 0.5581 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8423 | 0.7464 | 0.9199 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.4509 | 0.2734 | 0.6561 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.1925 | 0.1582 | 0.2268 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.6860 | 0.6163 | 0.7558 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.4255 | 0.2947 | 0.5505 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8783 | 0.7960 | 0.9436 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.5899 | 0.3874 | 0.7980 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.2247 | 0.1854 | 0.2657 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.6169 | 0.5390 | 0.6883 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.4040 | 0.2759 | 0.5225 |

