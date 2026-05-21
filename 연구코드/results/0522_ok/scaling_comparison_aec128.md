# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8187 | 0.2970 | 0.1763 | 0.7326 | 0.3947 |
| M2 | CrossAttn | excl_extreme/scale_both | 0.8226 | 0.3189 | 0.1954 | 0.7273 | 0.4474 |
| M2_2 | CrossAttn | norm/scale_both | 0.7928 | 0.2749 | 0.1886 | 0.7500 | 0.4267 |
| M3 | CrossAttn3 | norm/scale_both | 0.8014 | 0.4032 | 0.1655 | 0.7048 | 0.3797 |

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
| len128/scale_both | 0.8061 | 0.4007 | 0.1500 | 0.7442 | 0.3529 |
| crop80/scale_both | 0.8190 | 0.3671 | 0.2085 | 0.6919 | 0.4176 |
| crop60/scale_both | 0.8218 | 0.3835 | 0.1947 | 0.7209 | 0.4419 |
| norm/scale_both | 0.7805 | 0.2575 | 0.1553 | 0.8023 | 0.4688 |
| **excl_extreme/scale_both** | 0.8226 | 0.3189 | 0.1954 | 0.7273 | 0.4474 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.7408 | 0.2606 | 0.1886 | 0.7267 | 0.3562 |
| crop80/scale_both | 0.7329 | 0.2727 | 0.1834 | 0.7616 | 0.3881 |
| crop60/scale_both | 0.7654 | 0.2788 | 0.2013 | 0.6919 | 0.3457 |
| **norm/scale_both** | 0.7928 | 0.2749 | 0.1886 | 0.7500 | 0.4267 |
| excl_extreme/scale_both | 0.7485 | 0.2616 | 0.1825 | 0.7468 | 0.4000 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.7952 | 0.3892 | 0.1686 | 0.7048 | 0.3951 |
| crop80/scale_both | 0.7634 | 0.3470 | 0.1812 | 0.7229 | 0.3947 |
| crop60/scale_both | 0.7726 | 0.3652 | 0.1850 | 0.7289 | 0.4156 |
| **norm/scale_both** | 0.8014 | 0.4032 | 0.1655 | 0.7048 | 0.3797 |
| excl_extreme/scale_both | 0.7656 | 0.3427 | 0.1675 | 0.7635 | 0.3860 |

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
| AUC-ROC  | 0.8072 | 0.8284 | +0.0212 | -1.774 | 1.51e-01 | 1.25e-01 |
| AUPRC  | 0.4392 | 0.4631 | +0.0239 | -0.782 | 4.78e-01 | 6.25e-01 |
| Brier † | 0.1781 | 0.1614 | -0.0167 | 2.204 | 9.22e-02 | 6.25e-02 |
| Accuracy * | 0.7263 | 0.7700 | +0.0437 | -2.903 | 4.40e-02 | 1.25e-01 |
| F1  | 0.3851 | 0.4447 | +0.0596 | -1.840 | 1.40e-01 | 3.12e-01 |

### crop80/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8283 | +0.0211 | -0.981 | 3.82e-01 | 4.38e-01 |
| AUPRC  | 0.4392 | 0.4566 | +0.0173 | -0.506 | 6.40e-01 | 8.12e-01 |
| Brier  | 0.1781 | 0.1868 | +0.0087 | -1.896 | 1.31e-01 | 1.25e-01 |
| Accuracy † | 0.7263 | 0.7118 | -0.0145 | 2.232 | 8.94e-02 | 1.88e-01 |
| F1  | 0.3851 | 0.3928 | +0.0078 | -0.265 | 8.04e-01 | 6.25e-01 |

### crop60/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8217 | +0.0145 | -1.006 | 3.71e-01 | 6.25e-01 |
| AUPRC  | 0.4392 | 0.4528 | +0.0135 | -0.452 | 6.75e-01 | 8.12e-01 |
| Brier  | 0.1781 | 0.1812 | +0.0030 | -0.177 | 8.68e-01 | 6.25e-01 |
| Accuracy  | 0.7263 | 0.7393 | +0.0130 | -0.517 | 6.33e-01 | 6.25e-01 |
| F1  | 0.3851 | 0.4296 | +0.0445 | -1.890 | 1.32e-01 | 1.88e-01 |

### norm/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8287 | +0.0214 | -1.422 | 2.28e-01 | 1.88e-01 |
| AUPRC  | 0.4392 | 0.4503 | +0.0110 | -0.414 | 7.00e-01 | 8.12e-01 |
| Brier  | 0.1781 | 0.1810 | +0.0029 | -0.157 | 8.83e-01 | 8.12e-01 |
| Accuracy  | 0.7263 | 0.7495 | +0.0232 | -1.277 | 2.71e-01 | 4.38e-01 |
| F1 * | 0.3851 | 0.4271 | +0.0421 | -3.492 | 2.51e-02 | 6.25e-02 |

### excl_extreme/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8194 | +0.0122 | -0.612 | 5.73e-01 | 1.00e+00 |
| AUPRC  | 0.4392 | 0.4583 | +0.0190 | -0.448 | 6.77e-01 | 8.12e-01 |
| Brier  | 0.1781 | 0.2088 | +0.0307 | -0.765 | 4.87e-01 | 8.12e-01 |
| Accuracy  | 0.7263 | 0.6956 | -0.0307 | 0.558 | 6.06e-01 | 6.25e-01 |
| F1  | 0.3851 | 0.3825 | -0.0026 | 0.060 | 9.55e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len128/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8284 | 0.8218 | -0.0067 | 0.280 | 7.94e-01 | 8.12e-01 |
| AUPRC  | 0.4631 | 0.4377 | -0.0254 | 0.710 | 5.17e-01 | 6.25e-01 |
| Brier  | 0.1614 | 0.1592 | -0.0022 | 0.065 | 9.51e-01 | 8.12e-01 |
| Accuracy  | 0.7700 | 0.7530 | -0.0169 | 0.279 | 7.94e-01 | 1.00e+00 |
| F1  | 0.4447 | 0.4154 | -0.0292 | 0.513 | 6.35e-01 | 6.25e-01 |

#### Case: crop80/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8283 | 0.8228 | -0.0055 | 0.131 | 9.02e-01 | 1.00e+00 |
| AUPRC  | 0.4566 | 0.4189 | -0.0377 | 1.584 | 1.88e-01 | 3.12e-01 |
| Brier  | 0.1868 | 0.1886 | +0.0017 | -0.063 | 9.53e-01 | 1.00e+00 |
| Accuracy  | 0.7118 | 0.6833 | -0.0284 | 0.572 | 5.98e-01 | 8.12e-01 |
| F1  | 0.3928 | 0.3952 | +0.0024 | -0.058 | 9.57e-01 | 1.00e+00 |

#### Case: crop60/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8217 | 0.8230 | +0.0013 | -0.048 | 9.64e-01 | 1.00e+00 |
| AUPRC  | 0.4528 | 0.4495 | -0.0033 | 0.116 | 9.13e-01 | 1.00e+00 |
| Brier  | 0.1812 | 0.1832 | +0.0020 | -0.064 | 9.52e-01 | 1.00e+00 |
| Accuracy  | 0.7393 | 0.7136 | -0.0257 | 0.468 | 6.64e-01 | 8.12e-01 |
| F1  | 0.4296 | 0.4186 | -0.0110 | 0.302 | 7.77e-01 | 1.00e+00 |

#### Case: norm/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8287 | 0.8359 | +0.0072 | -0.187 | 8.61e-01 | 1.00e+00 |
| AUPRC  | 0.4503 | 0.4473 | -0.0030 | 0.087 | 9.35e-01 | 8.12e-01 |
| Brier  | 0.1810 | 0.1877 | +0.0067 | -0.135 | 8.99e-01 | 8.12e-01 |
| Accuracy  | 0.7495 | 0.7121 | -0.0374 | 0.489 | 6.51e-01 | 1.00e+00 |
| F1  | 0.4271 | 0.4034 | -0.0237 | 0.363 | 7.35e-01 | 8.12e-01 |

#### Case: excl_extreme/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8194 | 0.8137 | -0.0057 | 0.225 | 8.33e-01 | 1.00e+00 |
| AUPRC  | 0.4583 | 0.4254 | -0.0329 | 0.363 | 7.35e-01 | 8.12e-01 |
| Brier  | 0.2088 | 0.1618 | -0.0470 | 0.830 | 4.53e-01 | 8.12e-01 |
| Accuracy  | 0.6956 | 0.7559 | +0.0603 | -0.748 | 4.96e-01 | 6.25e-01 |
| F1  | 0.3825 | 0.4301 | +0.0476 | -1.040 | 3.57e-01 | 4.38e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8218 | +0.0146 | -0.428 | 6.90e-01 | 1.00e+00 |
| AUPRC  | 0.4392 | 0.4377 | -0.0015 | 0.029 | 9.78e-01 | 1.00e+00 |
| Brier  | 0.1781 | 0.1592 | -0.0189 | 0.649 | 5.52e-01 | 8.12e-01 |
| Accuracy  | 0.7263 | 0.7530 | +0.0267 | -0.413 | 7.01e-01 | 8.12e-01 |
| F1  | 0.3851 | 0.4154 | +0.0304 | -0.486 | 6.52e-01 | 8.12e-01 |

### crop80/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8228 | +0.0156 | -0.400 | 7.09e-01 | 1.00e+00 |
| AUPRC  | 0.4392 | 0.4189 | -0.0203 | 0.481 | 6.56e-01 | 8.12e-01 |
| Brier  | 0.1781 | 0.1886 | +0.0104 | -0.423 | 6.94e-01 | 6.25e-01 |
| Accuracy  | 0.7263 | 0.6833 | -0.0430 | 0.810 | 4.63e-01 | 6.25e-01 |
| F1  | 0.3851 | 0.3952 | +0.0102 | -0.193 | 8.56e-01 | 1.00e+00 |

### crop60/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8230 | +0.0157 | -0.534 | 6.22e-01 | 8.12e-01 |
| AUPRC  | 0.4392 | 0.4495 | +0.0103 | -0.196 | 8.54e-01 | 1.00e+00 |
| Brier  | 0.1781 | 0.1832 | +0.0051 | -0.223 | 8.34e-01 | 8.12e-01 |
| Accuracy  | 0.7263 | 0.7136 | -0.0127 | 0.277 | 7.95e-01 | 1.00e+00 |
| F1  | 0.3851 | 0.4186 | +0.0335 | -0.677 | 5.35e-01 | 4.38e-01 |

### norm/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8359 | +0.0287 | -0.663 | 5.43e-01 | 6.25e-01 |
| AUPRC  | 0.4392 | 0.4473 | +0.0081 | -0.138 | 8.97e-01 | 8.12e-01 |
| Brier  | 0.1781 | 0.1877 | +0.0095 | -0.281 | 7.93e-01 | 1.00e+00 |
| Accuracy  | 0.7263 | 0.7121 | -0.0142 | 0.217 | 8.39e-01 | 1.00e+00 |
| F1  | 0.3851 | 0.4034 | +0.0183 | -0.296 | 7.82e-01 | 8.12e-01 |

### excl_extreme/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.8137 | +0.0065 | -0.185 | 8.62e-01 | 1.00e+00 |
| AUPRC  | 0.4392 | 0.4254 | -0.0139 | 0.161 | 8.80e-01 | 6.25e-01 |
| Brier  | 0.1781 | 0.1618 | -0.0163 | 0.662 | 5.44e-01 | 8.12e-01 |
| Accuracy  | 0.7263 | 0.7559 | +0.0296 | -0.680 | 5.34e-01 | 6.25e-01 |
| F1  | 0.3851 | 0.4301 | +0.0450 | -0.820 | 4.58e-01 | 6.25e-01 |

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
| M2 | CrossAttn | len128/scale_both | AUC-ROC | 0.8061 | 0.7180 | 0.8874 |
| M2 | CrossAttn | len128/scale_both | AUPRC | 0.4007 | 0.2427 | 0.6302 |
| M2 | CrossAttn | len128/scale_both | Brier | 0.1500 | 0.1208 | 0.1799 |
| M2 | CrossAttn | len128/scale_both | Accuracy | 0.7442 | 0.6802 | 0.8081 |
| M2 | CrossAttn | len128/scale_both | F1 | 0.3529 | 0.2105 | 0.4872 |
| M2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.8190 | 0.7371 | 0.8914 |
| M2 | CrossAttn | crop80/scale_both | AUPRC | 0.3671 | 0.2305 | 0.5876 |
| M2 | CrossAttn | crop80/scale_both | Brier | 0.2085 | 0.1721 | 0.2449 |
| M2 | CrossAttn | crop80/scale_both | Accuracy | 0.6919 | 0.6279 | 0.7616 |
| M2 | CrossAttn | crop80/scale_both | F1 | 0.4176 | 0.2892 | 0.5455 |
| M2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.8218 | 0.7323 | 0.8988 |
| M2 | CrossAttn | crop60/scale_both | AUPRC | 0.3835 | 0.2269 | 0.5920 |
| M2 | CrossAttn | crop60/scale_both | Brier | 0.1947 | 0.1581 | 0.2327 |
| M2 | CrossAttn | crop60/scale_both | Accuracy | 0.7209 | 0.6512 | 0.7849 |
| M2 | CrossAttn | crop60/scale_both | F1 | 0.4419 | 0.3056 | 0.5682 |
| M2 | CrossAttn | norm/scale_both | AUC-ROC | 0.7805 | 0.6976 | 0.8597 |
| M2 | CrossAttn | norm/scale_both | AUPRC | 0.2575 | 0.1662 | 0.4253 |
| M2 | CrossAttn | norm/scale_both | Brier | 0.1553 | 0.1266 | 0.1844 |
| M2 | CrossAttn | norm/scale_both | Accuracy | 0.8023 | 0.7384 | 0.8605 |
| M2 | CrossAttn | norm/scale_both | F1 | 0.4688 | 0.3137 | 0.6087 |
| M2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8226 | 0.7383 | 0.8898 |
| M2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.3189 | 0.1919 | 0.5323 |
| M2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1954 | 0.1556 | 0.2368 |
| M2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.7273 | 0.6558 | 0.7987 |
| M2 | CrossAttn | excl_extreme/scale_both | F1 | 0.4474 | 0.2951 | 0.5798 |
| M2_2 | CrossAttn | len128/scale_both | AUC-ROC | 0.7408 | 0.6401 | 0.8363 |
| M2_2 | CrossAttn | len128/scale_both | AUPRC | 0.2606 | 0.1580 | 0.4493 |
| M2_2 | CrossAttn | len128/scale_both | Brier | 0.1886 | 0.1533 | 0.2278 |
| M2_2 | CrossAttn | len128/scale_both | Accuracy | 0.7267 | 0.6628 | 0.7907 |
| M2_2 | CrossAttn | len128/scale_both | F1 | 0.3562 | 0.2121 | 0.4902 |
| M2_2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.7329 | 0.6226 | 0.8297 |
| M2_2 | CrossAttn | crop80/scale_both | AUPRC | 0.2727 | 0.1611 | 0.4701 |
| M2_2 | CrossAttn | crop80/scale_both | Brier | 0.1834 | 0.1426 | 0.2279 |
| M2_2 | CrossAttn | crop80/scale_both | Accuracy | 0.7616 | 0.6977 | 0.8198 |
| M2_2 | CrossAttn | crop80/scale_both | F1 | 0.3881 | 0.2353 | 0.5232 |
| M2_2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.7654 | 0.6752 | 0.8487 |
| M2_2 | CrossAttn | crop60/scale_both | AUPRC | 0.2788 | 0.1654 | 0.4777 |
| M2_2 | CrossAttn | crop60/scale_both | Brier | 0.2013 | 0.1622 | 0.2417 |
| M2_2 | CrossAttn | crop60/scale_both | Accuracy | 0.6919 | 0.6221 | 0.7616 |
| M2_2 | CrossAttn | crop60/scale_both | F1 | 0.3457 | 0.2059 | 0.4719 |
| M2_2 | CrossAttn | norm/scale_both | AUC-ROC | 0.7928 | 0.7042 | 0.8758 |
| M2_2 | CrossAttn | norm/scale_both | AUPRC | 0.2749 | 0.1789 | 0.4592 |
| M2_2 | CrossAttn | norm/scale_both | Brier | 0.1886 | 0.1498 | 0.2291 |
| M2_2 | CrossAttn | norm/scale_both | Accuracy | 0.7500 | 0.6860 | 0.8140 |
| M2_2 | CrossAttn | norm/scale_both | F1 | 0.4267 | 0.2817 | 0.5582 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.7485 | 0.6386 | 0.8496 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.2616 | 0.1585 | 0.4642 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1825 | 0.1441 | 0.2186 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.7468 | 0.6753 | 0.8182 |
| M2_2 | CrossAttn | excl_extreme/scale_both | F1 | 0.4000 | 0.2414 | 0.5385 |
| M3 | CrossAttn3 | len128/scale_both | AUC-ROC | 0.7952 | 0.6737 | 0.8982 |
| M3 | CrossAttn3 | len128/scale_both | AUPRC | 0.3892 | 0.2264 | 0.6140 |
| M3 | CrossAttn3 | len128/scale_both | Brier | 0.1686 | 0.1357 | 0.2033 |
| M3 | CrossAttn3 | len128/scale_both | Accuracy | 0.7048 | 0.6325 | 0.7711 |
| M3 | CrossAttn3 | len128/scale_both | F1 | 0.3951 | 0.2535 | 0.5239 |
| M3 | CrossAttn3 | crop80/scale_both | AUC-ROC | 0.7634 | 0.6437 | 0.8706 |
| M3 | CrossAttn3 | crop80/scale_both | AUPRC | 0.3470 | 0.1955 | 0.5689 |
| M3 | CrossAttn3 | crop80/scale_both | Brier | 0.1812 | 0.1451 | 0.2185 |
| M3 | CrossAttn3 | crop80/scale_both | Accuracy | 0.7229 | 0.6566 | 0.7892 |
| M3 | CrossAttn3 | crop80/scale_both | F1 | 0.3947 | 0.2500 | 0.5302 |
| M3 | CrossAttn3 | crop60/scale_both | AUC-ROC | 0.7726 | 0.6511 | 0.8793 |
| M3 | CrossAttn3 | crop60/scale_both | AUPRC | 0.3652 | 0.2054 | 0.5641 |
| M3 | CrossAttn3 | crop60/scale_both | Brier | 0.1850 | 0.1466 | 0.2257 |
| M3 | CrossAttn3 | crop60/scale_both | Accuracy | 0.7289 | 0.6627 | 0.7952 |
| M3 | CrossAttn3 | crop60/scale_both | F1 | 0.4156 | 0.2683 | 0.5527 |
| M3 | CrossAttn3 | norm/scale_both | AUC-ROC | 0.8014 | 0.6933 | 0.8943 |
| M3 | CrossAttn3 | norm/scale_both | AUPRC | 0.4032 | 0.2276 | 0.6045 |
| M3 | CrossAttn3 | norm/scale_both | Brier | 0.1655 | 0.1345 | 0.1982 |
| M3 | CrossAttn3 | norm/scale_both | Accuracy | 0.7048 | 0.6325 | 0.7711 |
| M3 | CrossAttn3 | norm/scale_both | F1 | 0.3797 | 0.2388 | 0.5116 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUC-ROC | 0.7656 | 0.6284 | 0.8797 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUPRC | 0.3427 | 0.1679 | 0.5733 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Brier | 0.1675 | 0.1272 | 0.2116 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Accuracy | 0.7635 | 0.6959 | 0.8311 |
| M3 | CrossAttn3 | excl_extreme/scale_both | F1 | 0.3860 | 0.2105 | 0.5397 |

