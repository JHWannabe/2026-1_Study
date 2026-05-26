# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8127 | 0.4105 | 0.1869 | 0.6481 | 0.3387 |
| M2 | CrossAttn | norm | 0.8258 | 0.4338 | 0.1923 | 0.6395 | 0.3115 |
| M2_2 | CrossAttn | norm | 0.8456 | 0.4099 | 0.1897 | 0.6738 | 0.3559 |
| M3 | CrossAttn3 | len128 | 0.8246 | 0.4136 | 0.2177 | 0.6180 | 0.3206 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_all** | 0.8127 | 0.4105 | 0.1869 | 0.6481 | 0.3387 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm** | 0.8258 | 0.4338 | 0.1923 | 0.6395 | 0.3115 |
| excl_extreme | 0.8211 | 0.4092 | 0.1584 | 0.5933 | 0.2975 |
| len128 | 0.8200 | 0.4028 | 0.2088 | 0.6309 | 0.3175 |
| crop80 | 0.8085 | 0.3953 | 0.1675 | 0.7854 | 0.3421 |
| crop60 | 0.7935 | 0.3055 | 0.1525 | 0.8155 | 0.4110 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm** | 0.8456 | 0.4099 | 0.1897 | 0.6738 | 0.3559 |
| excl_extreme | 0.8155 | 0.3762 | 0.1473 | 0.8325 | 0.4068 |
| len128 | 0.8171 | 0.3574 | 0.1738 | 0.7511 | 0.3696 |
| crop80 | 0.7998 | 0.3460 | 0.2175 | 0.6953 | 0.3486 |
| crop60 | 0.8042 | 0.3428 | 0.1937 | 0.7253 | 0.3600 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm | 0.8042 | 0.3970 | 0.2506 | 0.5880 | 0.3043 |
| excl_extreme | 0.8119 | 0.4300 | 0.1805 | 0.6890 | 0.3434 |
| **len128** | 0.8246 | 0.4136 | 0.2177 | 0.6180 | 0.3206 |
| crop80 | 0.7935 | 0.3747 | 0.1685 | 0.7725 | 0.3908 |
| crop60 | 0.8106 | 0.3950 | 0.1766 | 0.7339 | 0.3673 |

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

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8160 | +0.0070 | -0.478 | 6.58e-01 | 8.12e-01 |
| AUPRC  | 0.3931 | 0.3859 | -0.0072 | 0.196 | 8.54e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1740 | -0.0060 | 1.002 | 3.73e-01 | 4.38e-01 |
| Accuracy  | 0.7086 | 0.7237 | +0.0151 | -0.297 | 7.81e-01 | 6.25e-01 |
| F1  | 0.4075 | 0.4001 | -0.0074 | 0.179 | 8.66e-01 | 6.25e-01 |

### excl_extreme  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8339 | +0.0248 | -0.905 | 4.17e-01 | 8.12e-01 |
| AUPRC  | 0.3931 | 0.4161 | +0.0230 | -0.383 | 7.21e-01 | 8.12e-01 |
| Brier † | 0.1800 | 0.1410 | -0.0391 | 2.488 | 6.76e-02 | 1.25e-01 |
| Accuracy  | 0.7086 | 0.7297 | +0.0211 | -0.427 | 6.91e-01 | 8.12e-01 |
| F1  | 0.4075 | 0.4032 | -0.0043 | 0.117 | 9.12e-01 | 8.12e-01 |

### len128  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8253 | +0.0163 | -1.508 | 2.06e-01 | 1.88e-01 |
| AUPRC  | 0.3931 | 0.4042 | +0.0111 | -0.694 | 5.26e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1666 | -0.0135 | 0.799 | 4.69e-01 | 6.25e-01 |
| Accuracy  | 0.7086 | 0.7538 | +0.0452 | -1.321 | 2.57e-01 | 3.75e-01 |
| F1  | 0.4075 | 0.4329 | +0.0254 | -1.519 | 2.03e-01 | 3.12e-01 |

### crop80  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8209 | +0.0119 | -1.375 | 2.41e-01 | 3.12e-01 |
| AUPRC  | 0.3931 | 0.3962 | +0.0032 | -0.101 | 9.24e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1837 | +0.0036 | -0.233 | 8.27e-01 | 1.00e+00 |
| Accuracy  | 0.7086 | 0.7720 | +0.0634 | -1.257 | 2.77e-01 | 2.50e-01 |
| F1  | 0.4075 | 0.4361 | +0.0286 | -0.752 | 4.94e-01 | 4.38e-01 |

### crop60  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8264 | +0.0174 | -1.783 | 1.49e-01 | 1.88e-01 |
| AUPRC  | 0.3931 | 0.3963 | +0.0032 | -0.094 | 9.30e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1852 | +0.0052 | -0.575 | 5.96e-01 | 6.25e-01 |
| Accuracy  | 0.7086 | 0.7602 | +0.0516 | -1.569 | 1.92e-01 | 2.50e-01 |
| F1  | 0.4075 | 0.4271 | +0.0196 | -0.892 | 4.23e-01 | 6.25e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8160 | 0.8167 | +0.0007 | -0.053 | 9.61e-01 | 1.00e+00 |
| AUPRC  | 0.3859 | 0.3966 | +0.0107 | -0.307 | 7.74e-01 | 6.25e-01 |
| Brier  | 0.1740 | 0.1572 | -0.0168 | 1.327 | 2.55e-01 | 3.12e-01 |
| Accuracy  | 0.7237 | 0.7484 | +0.0247 | -1.012 | 3.69e-01 | 4.38e-01 |
| F1  | 0.4001 | 0.4188 | +0.0187 | -1.086 | 3.39e-01 | 4.38e-01 |

#### Case: excl_extreme  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8339 | 0.8310 | -0.0028 | 0.541 | 6.17e-01 | 6.25e-01 |
| AUPRC  | 0.4161 | 0.4198 | +0.0037 | -0.162 | 8.79e-01 | 1.00e+00 |
| Brier  | 0.1410 | 0.1582 | +0.0172 | -0.960 | 3.91e-01 | 8.12e-01 |
| Accuracy * | 0.7297 | 0.7847 | +0.0551 | -3.596 | 2.28e-02 | 6.25e-02 |
| F1 † | 0.4032 | 0.4489 | +0.0457 | -2.770 | 5.04e-02 | 1.25e-01 |

#### Case: len128  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8253 | 0.8206 | -0.0047 | 0.502 | 6.42e-01 | 8.12e-01 |
| AUPRC  | 0.4042 | 0.3816 | -0.0227 | 0.795 | 4.71e-01 | 8.12e-01 |
| Brier  | 0.1666 | 0.1580 | -0.0086 | 0.529 | 6.25e-01 | 8.12e-01 |
| Accuracy  | 0.7538 | 0.7699 | +0.0161 | -0.673 | 5.38e-01 | 8.75e-01 |
| F1  | 0.4329 | 0.4446 | +0.0117 | -0.486 | 6.53e-01 | 1.00e+00 |

#### Case: crop80  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8209 | 0.8246 | +0.0037 | -0.448 | 6.77e-01 | 6.25e-01 |
| AUPRC  | 0.3962 | 0.3975 | +0.0013 | -0.055 | 9.59e-01 | 1.00e+00 |
| Brier  | 0.1837 | 0.1838 | +0.0002 | -0.007 | 9.95e-01 | 1.00e+00 |
| Accuracy  | 0.7720 | 0.7699 | -0.0022 | 0.129 | 9.03e-01 | 1.00e+00 |
| F1  | 0.4361 | 0.4376 | +0.0015 | -0.101 | 9.25e-01 | 8.12e-01 |

#### Case: crop60  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8264 | 0.8204 | -0.0061 | 0.613 | 5.73e-01 | 6.25e-01 |
| AUPRC  | 0.3963 | 0.3842 | -0.0121 | 0.472 | 6.61e-01 | 6.25e-01 |
| Brier  | 0.1852 | 0.1944 | +0.0092 | -0.707 | 5.18e-01 | 4.38e-01 |
| Accuracy  | 0.7602 | 0.7538 | -0.0065 | 0.239 | 8.23e-01 | 7.50e-01 |
| F1  | 0.4271 | 0.4289 | +0.0019 | -0.086 | 9.36e-01 | 8.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8167 | +0.0077 | -0.288 | 7.88e-01 | 8.12e-01 |
| AUPRC  | 0.3931 | 0.3966 | +0.0035 | -0.051 | 9.61e-01 | 8.12e-01 |
| Brier † | 0.1800 | 0.1572 | -0.0228 | 2.519 | 6.54e-02 | 1.25e-01 |
| Accuracy  | 0.7086 | 0.7484 | +0.0398 | -0.763 | 4.88e-01 | 6.25e-01 |
| F1  | 0.4075 | 0.4188 | +0.0113 | -0.222 | 8.35e-01 | 6.25e-01 |

### excl_extreme  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8310 | +0.0220 | -0.713 | 5.15e-01 | 8.12e-01 |
| AUPRC  | 0.3931 | 0.4198 | +0.0267 | -0.369 | 7.31e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1582 | -0.0219 | 1.017 | 3.67e-01 | 4.38e-01 |
| Accuracy  | 0.7086 | 0.7847 | +0.0761 | -1.674 | 1.70e-01 | 1.88e-01 |
| F1  | 0.4075 | 0.4489 | +0.0414 | -1.142 | 3.17e-01 | 6.25e-01 |

### len128  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8206 | +0.0116 | -0.650 | 5.51e-01 | 6.25e-01 |
| AUPRC  | 0.3931 | 0.3816 | -0.0115 | 0.294 | 7.83e-01 | 1.00e+00 |
| Brier * | 0.1800 | 0.1580 | -0.0220 | 3.555 | 2.37e-02 | 6.25e-02 |
| Accuracy  | 0.7086 | 0.7699 | +0.0613 | -1.708 | 1.63e-01 | 3.12e-01 |
| F1  | 0.4075 | 0.4446 | +0.0371 | -1.460 | 2.18e-01 | 3.12e-01 |

### crop80  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8246 | +0.0156 | -1.033 | 3.60e-01 | 4.38e-01 |
| AUPRC  | 0.3931 | 0.3975 | +0.0045 | -0.082 | 9.38e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1838 | +0.0038 | -0.223 | 8.34e-01 | 1.00e+00 |
| Accuracy  | 0.7086 | 0.7699 | +0.0613 | -1.300 | 2.63e-01 | 3.75e-01 |
| F1  | 0.4075 | 0.4376 | +0.0301 | -0.787 | 4.75e-01 | 4.38e-01 |

### crop60  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8204 | +0.0113 | -0.768 | 4.85e-01 | 4.38e-01 |
| AUPRC  | 0.3931 | 0.3842 | -0.0089 | 0.249 | 8.15e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1944 | +0.0143 | -1.598 | 1.85e-01 | 1.88e-01 |
| Accuracy  | 0.7086 | 0.7538 | +0.0452 | -1.070 | 3.45e-01 | 6.25e-01 |
| F1  | 0.4075 | 0.4289 | +0.0215 | -0.657 | 5.47e-01 | 3.75e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_all | AUC-ROC | 0.8127 | 0.7287 | 0.8883 |
| M1 | LR | scale_all | AUPRC | 0.4105 | 0.2428 | 0.5883 |
| M1 | LR | scale_all | Brier | 0.1869 | 0.1618 | 0.2127 |
| M1 | LR | scale_all | Accuracy | 0.6481 | 0.5880 | 0.7082 |
| M1 | LR | scale_all | F1 | 0.3387 | 0.2376 | 0.4480 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8258 | 0.7434 | 0.9025 |
| M2 | CrossAttn | norm | AUPRC | 0.4338 | 0.2598 | 0.6332 |
| M2 | CrossAttn | norm | Brier | 0.1923 | 0.1654 | 0.2208 |
| M2 | CrossAttn | norm | Accuracy | 0.6395 | 0.5751 | 0.6996 |
| M2 | CrossAttn | norm | F1 | 0.3115 | 0.2069 | 0.4242 |
| M2 | CrossAttn | excl_extreme | AUC-ROC | 0.8211 | 0.7285 | 0.8984 |
| M2 | CrossAttn | excl_extreme | AUPRC | 0.4092 | 0.2265 | 0.6106 |
| M2 | CrossAttn | excl_extreme | Brier | 0.1584 | 0.1342 | 0.1849 |
| M2 | CrossAttn | excl_extreme | Accuracy | 0.5933 | 0.5263 | 0.6555 |
| M2 | CrossAttn | excl_extreme | F1 | 0.2975 | 0.1896 | 0.3968 |
| M2 | CrossAttn | len128 | AUC-ROC | 0.8200 | 0.7360 | 0.8942 |
| M2 | CrossAttn | len128 | AUPRC | 0.4028 | 0.2404 | 0.5948 |
| M2 | CrossAttn | len128 | Brier | 0.2088 | 0.1793 | 0.2394 |
| M2 | CrossAttn | len128 | Accuracy | 0.6309 | 0.5708 | 0.6910 |
| M2 | CrossAttn | len128 | F1 | 0.3175 | 0.2154 | 0.4253 |
| M2 | CrossAttn | crop80 | AUC-ROC | 0.8085 | 0.7214 | 0.8871 |
| M2 | CrossAttn | crop80 | AUPRC | 0.3953 | 0.2339 | 0.5914 |
| M2 | CrossAttn | crop80 | Brier | 0.1675 | 0.1417 | 0.1935 |
| M2 | CrossAttn | crop80 | Accuracy | 0.7854 | 0.7339 | 0.8369 |
| M2 | CrossAttn | crop80 | F1 | 0.3421 | 0.2000 | 0.4737 |
| M2 | CrossAttn | crop60 | AUC-ROC | 0.7935 | 0.7002 | 0.8786 |
| M2 | CrossAttn | crop60 | AUPRC | 0.3055 | 0.1945 | 0.4982 |
| M2 | CrossAttn | crop60 | Brier | 0.1525 | 0.1270 | 0.1771 |
| M2 | CrossAttn | crop60 | Accuracy | 0.8155 | 0.7682 | 0.8670 |
| M2 | CrossAttn | crop60 | F1 | 0.4110 | 0.2687 | 0.5480 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8456 | 0.7663 | 0.9112 |
| M2_2 | CrossAttn | norm | AUPRC | 0.4099 | 0.2510 | 0.6101 |
| M2_2 | CrossAttn | norm | Brier | 0.1897 | 0.1619 | 0.2191 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6738 | 0.6137 | 0.7339 |
| M2_2 | CrossAttn | norm | F1 | 0.3559 | 0.2453 | 0.4697 |
| M2_2 | CrossAttn | excl_extreme | AUC-ROC | 0.8155 | 0.7255 | 0.8919 |
| M2_2 | CrossAttn | excl_extreme | AUPRC | 0.3762 | 0.2043 | 0.5739 |
| M2_2 | CrossAttn | excl_extreme | Brier | 0.1473 | 0.1242 | 0.1698 |
| M2_2 | CrossAttn | excl_extreme | Accuracy | 0.8325 | 0.7846 | 0.8804 |
| M2_2 | CrossAttn | excl_extreme | F1 | 0.4068 | 0.2456 | 0.5574 |
| M2_2 | CrossAttn | len128 | AUC-ROC | 0.8171 | 0.7437 | 0.8853 |
| M2_2 | CrossAttn | len128 | AUPRC | 0.3574 | 0.2063 | 0.5415 |
| M2_2 | CrossAttn | len128 | Brier | 0.1738 | 0.1441 | 0.2039 |
| M2_2 | CrossAttn | len128 | Accuracy | 0.7511 | 0.6953 | 0.8069 |
| M2_2 | CrossAttn | len128 | F1 | 0.3696 | 0.2432 | 0.4956 |
| M2_2 | CrossAttn | crop80 | AUC-ROC | 0.7998 | 0.7181 | 0.8763 |
| M2_2 | CrossAttn | crop80 | AUPRC | 0.3460 | 0.1968 | 0.5294 |
| M2_2 | CrossAttn | crop80 | Brier | 0.2175 | 0.1845 | 0.2508 |
| M2_2 | CrossAttn | crop80 | Accuracy | 0.6953 | 0.6352 | 0.7554 |
| M2_2 | CrossAttn | crop80 | F1 | 0.3486 | 0.2353 | 0.4660 |
| M2_2 | CrossAttn | crop60 | AUC-ROC | 0.8042 | 0.7176 | 0.8861 |
| M2_2 | CrossAttn | crop60 | AUPRC | 0.3428 | 0.2050 | 0.5389 |
| M2_2 | CrossAttn | crop60 | Brier | 0.1937 | 0.1620 | 0.2264 |
| M2_2 | CrossAttn | crop60 | Accuracy | 0.7253 | 0.6695 | 0.7811 |
| M2_2 | CrossAttn | crop60 | F1 | 0.3600 | 0.2400 | 0.4815 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8042 | 0.7162 | 0.8879 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3970 | 0.2306 | 0.5933 |
| M3 | CrossAttn3 | norm | Brier | 0.2506 | 0.2174 | 0.2851 |
| M3 | CrossAttn3 | norm | Accuracy | 0.5880 | 0.5236 | 0.6524 |
| M3 | CrossAttn3 | norm | F1 | 0.3043 | 0.2097 | 0.4028 |
| M3 | CrossAttn3 | excl_extreme | AUC-ROC | 0.8119 | 0.7095 | 0.8961 |
| M3 | CrossAttn3 | excl_extreme | AUPRC | 0.4300 | 0.2410 | 0.6243 |
| M3 | CrossAttn3 | excl_extreme | Brier | 0.1805 | 0.1538 | 0.2104 |
| M3 | CrossAttn3 | excl_extreme | Accuracy | 0.6890 | 0.6268 | 0.7464 |
| M3 | CrossAttn3 | excl_extreme | F1 | 0.3434 | 0.2200 | 0.4572 |
| M3 | CrossAttn3 | len128 | AUC-ROC | 0.8246 | 0.7348 | 0.9049 |
| M3 | CrossAttn3 | len128 | AUPRC | 0.4136 | 0.2505 | 0.6098 |
| M3 | CrossAttn3 | len128 | Brier | 0.2177 | 0.1860 | 0.2506 |
| M3 | CrossAttn3 | len128 | Accuracy | 0.6180 | 0.5536 | 0.6781 |
| M3 | CrossAttn3 | len128 | F1 | 0.3206 | 0.2206 | 0.4219 |
| M3 | CrossAttn3 | crop80 | AUC-ROC | 0.7935 | 0.6947 | 0.8814 |
| M3 | CrossAttn3 | crop80 | AUPRC | 0.3747 | 0.2194 | 0.5670 |
| M3 | CrossAttn3 | crop80 | Brier | 0.1685 | 0.1386 | 0.1985 |
| M3 | CrossAttn3 | crop80 | Accuracy | 0.7725 | 0.7210 | 0.8283 |
| M3 | CrossAttn3 | crop80 | F1 | 0.3908 | 0.2609 | 0.5193 |
| M3 | CrossAttn3 | crop60 | AUC-ROC | 0.8106 | 0.7214 | 0.8915 |
| M3 | CrossAttn3 | crop60 | AUPRC | 0.3950 | 0.2309 | 0.5921 |
| M3 | CrossAttn3 | crop60 | Brier | 0.1766 | 0.1510 | 0.2038 |
| M3 | CrossAttn3 | crop60 | Accuracy | 0.7339 | 0.6781 | 0.7897 |
| M3 | CrossAttn3 | crop60 | F1 | 0.3673 | 0.2472 | 0.4874 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-norm | 0.8127 | 0.8258 | +0.0131 | -0.532 | 5.948e-01 | ns |
| M1-LR vs M2-len128 | 0.8127 | 0.8200 | +0.0073 | -0.325 | 7.454e-01 | ns |
| M1-LR vs M2-crop80 | 0.8127 | 0.8085 | -0.0042 | 0.203 | 8.394e-01 | ns |
| M1-LR vs M2-crop60 | 0.8127 | 0.7935 | -0.0192 | 0.626 | 5.316e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-norm | 0.8127 | 0.8042 | -0.0085 | 0.338 | 7.354e-01 | ns |
| M1-LR vs M3-len128 | 0.8127 | 0.8246 | +0.0119 | -0.607 | 5.438e-01 | ns |
| M1-LR vs M3-crop80 | 0.8127 | 0.7935 | -0.0192 | 0.671 | 5.023e-01 | ns |
| M1-LR vs M3-crop60 | 0.8127 | 0.8106 | -0.0021 | 0.094 | 9.255e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M2_2-norm | 0.8258 | 0.8456 | +0.0198 | -0.975 | 3.295e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8211 | 0.4993 | -0.3218 | 5.001 | 5.716e-07 | *** |
| M2-len128 vs M2_2-len128 | 0.8200 | 0.8171 | -0.0029 | 0.146 | 8.842e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.8085 | 0.7998 | -0.0087 | 0.465 | 6.419e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.7935 | 0.8042 | +0.0108 | -0.416 | 6.771e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M3-norm | 0.8258 | 0.8042 | -0.0215 | 1.700 | 8.915e-02 | † |
| M2-excl_extreme vs M3-excl_extreme | 0.8211 | 0.8119 | -0.0092 | 0.414 | 6.791e-01 | ns |
| M2-len128 vs M3-len128 | 0.8200 | 0.8246 | +0.0046 | -0.217 | 8.281e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8085 | 0.7935 | -0.0150 | 0.624 | 5.324e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.7935 | 0.8106 | +0.0171 | -0.582 | 5.607e-01 | ns |

