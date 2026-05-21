# Scaling Comparison — Test Set Performance (AEC 256pt, BCEWithLogitsLoss)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8119 | 0.4103 | 0.1878 | 0.7124 | 0.3853 |
| M2 | CrossAttn | norm/scale_clinic | 0.8277 | 0.4404 | 0.1565 | 0.7682 | 0.4000 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | 0.8357 | 0.4007 | 0.2085 | 0.6986 | 0.3762 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | 0.8230 | 0.4074 | 0.1681 | 0.6842 | 0.3400 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_clinic** | 0.8119 | 0.4103 | 0.1878 | 0.7124 | 0.3853 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm/scale_clinic** | 0.8277 | 0.4404 | 0.1565 | 0.7682 | 0.4000 |
| excl_extreme/scale_clinic | 0.8262 | 0.4254 | 0.1720 | 0.6890 | 0.3564 |
| len128/scale_clinic | 0.8017 | 0.4105 | 0.2302 | 0.6567 | 0.3443 |
| crop80/scale_clinic | 0.8050 | 0.3764 | 0.1997 | 0.6996 | 0.3636 |
| crop60/scale_clinic | 0.7667 | 0.3475 | 0.2070 | 0.7725 | 0.3908 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm/scale_clinic | 0.8294 | 0.3881 | 0.2231 | 0.6953 | 0.3717 |
| **excl_extreme/scale_clinic** | 0.8357 | 0.4007 | 0.2085 | 0.6986 | 0.3762 |
| len128/scale_clinic | 0.8065 | 0.3770 | 0.1728 | 0.6266 | 0.3256 |
| crop80/scale_clinic | 0.8146 | 0.3850 | 0.1983 | 0.6781 | 0.3478 |
| crop60/scale_clinic | 0.8204 | 0.4127 | 0.2369 | 0.6094 | 0.3158 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm/scale_clinic | 0.8154 | 0.4239 | 0.1953 | 0.6524 | 0.3306 |
| **excl_extreme/scale_clinic** | 0.8230 | 0.4074 | 0.1681 | 0.6842 | 0.3400 |
| len128/scale_clinic | 0.8077 | 0.4203 | 0.1613 | 0.7639 | 0.4086 |
| crop80/scale_clinic | 0.8058 | 0.4190 | 0.2086 | 0.7425 | 0.3617 |
| crop60/scale_clinic | 0.8040 | 0.3966 | 0.1773 | 0.7296 | 0.3226 |

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
| AUC-ROC  | 0.8051 | 0.8161 | +0.0111 | -0.979 | 3.83e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3907 | +0.0050 | -0.181 | 8.65e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1765 | -0.0035 | 0.233 | 8.27e-01 | 6.25e-01 |
| Accuracy  | 0.7491 | 0.7459 | -0.0032 | 0.078 | 9.42e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4113 | -0.0074 | 0.306 | 7.75e-01 | 8.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8240 | +0.0189 | -0.511 | 6.36e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3994 | +0.0137 | -0.251 | 8.14e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1685 | -0.0115 | 0.532 | 6.23e-01 | 6.25e-01 |
| Accuracy  | 0.7491 | 0.7593 | +0.0102 | -0.182 | 8.65e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4090 | -0.0097 | 0.253 | 8.13e-01 | 1.00e+00 |

### len128/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8163 | +0.0112 | -0.667 | 5.41e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3732 | -0.0124 | 0.407 | 7.05e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1767 | -0.0033 | 0.328 | 7.60e-01 | 1.00e+00 |
| Accuracy  | 0.7491 | 0.7546 | +0.0055 | -0.115 | 9.14e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4209 | +0.0022 | -0.068 | 9.49e-01 | 1.00e+00 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8179 | +0.0128 | -1.000 | 3.74e-01 | 4.38e-01 |
| AUPRC  | 0.3857 | 0.4175 | +0.0318 | -0.776 | 4.81e-01 | 4.38e-01 |
| Brier  | 0.1800 | 0.1753 | -0.0047 | 0.878 | 4.30e-01 | 4.38e-01 |
| Accuracy  | 0.7491 | 0.7363 | -0.0128 | 0.198 | 8.53e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4088 | -0.0099 | 0.254 | 8.12e-01 | 8.12e-01 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8236 | +0.0185 | -1.572 | 1.91e-01 | 3.12e-01 |
| AUPRC  | 0.3857 | 0.4406 | +0.0549 | -0.887 | 4.25e-01 | 4.38e-01 |
| Brier † | 0.1800 | 0.1660 | -0.0140 | 2.204 | 9.23e-02 | 1.25e-01 |
| Accuracy  | 0.7491 | 0.7792 | +0.0301 | -0.546 | 6.14e-01 | 6.25e-01 |
| F1  | 0.4187 | 0.4510 | +0.0323 | -0.874 | 4.31e-01 | 4.38e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8161 | 0.7991 | -0.0170 | 1.820 | 1.43e-01 | 3.12e-01 |
| AUPRC  | 0.3907 | 0.3754 | -0.0153 | 0.743 | 4.99e-01 | 6.25e-01 |
| Brier  | 0.1765 | 0.1695 | -0.0070 | 0.469 | 6.64e-01 | 8.12e-01 |
| Accuracy  | 0.7459 | 0.7363 | -0.0096 | 0.329 | 7.58e-01 | 1.00e+00 |
| F1  | 0.4113 | 0.4022 | -0.0091 | 0.505 | 6.40e-01 | 8.12e-01 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8240 | 0.8136 | -0.0104 | 1.447 | 2.22e-01 | 3.12e-01 |
| AUPRC  | 0.3994 | 0.3601 | -0.0393 | 1.260 | 2.76e-01 | 3.12e-01 |
| Brier  | 0.1685 | 0.1826 | +0.0141 | -1.204 | 2.95e-01 | 6.25e-01 |
| Accuracy  | 0.7593 | 0.7581 | -0.0012 | 0.044 | 9.67e-01 | 8.75e-01 |
| F1  | 0.4090 | 0.3992 | -0.0099 | 0.503 | 6.41e-01 | 6.25e-01 |

#### Case: len128/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8163 | 0.8058 | -0.0105 | 3.559 | 2.36e-02 | 6.25e-02 |
| AUPRC  | 0.3732 | 0.3730 | -0.0003 | 0.014 | 9.90e-01 | 1.00e+00 |
| Brier  | 0.1767 | 0.1918 | +0.0151 | -1.622 | 1.80e-01 | 1.88e-01 |
| Accuracy  | 0.7546 | 0.7330 | -0.0215 | 1.045 | 3.55e-01 | 3.12e-01 |
| F1  | 0.4209 | 0.4068 | -0.0141 | 0.964 | 3.90e-01 | 3.12e-01 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8179 | 0.8166 | -0.0012 | 0.146 | 8.91e-01 | 1.00e+00 |
| AUPRC  | 0.4175 | 0.3858 | -0.0317 | 1.676 | 1.69e-01 | 3.12e-01 |
| Brier  | 0.1753 | 0.1907 | +0.0154 | -0.832 | 4.52e-01 | 8.12e-01 |
| Accuracy  | 0.7363 | 0.7869 | +0.0506 | -1.298 | 2.64e-01 | 5.00e-01 |
| F1  | 0.4088 | 0.4448 | +0.0360 | -1.678 | 1.69e-01 | 3.12e-01 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.8236 | 0.8017 | -0.0219 | 5.250 | 6.30e-03 | 6.25e-02 |
| AUPRC † | 0.4406 | 0.3641 | -0.0765 | 2.196 | 9.31e-02 | 1.25e-01 |
| Brier  | 0.1660 | 0.1891 | +0.0231 | -1.861 | 1.36e-01 | 6.25e-02 |
| Accuracy  | 0.7792 | 0.7589 | -0.0203 | 0.405 | 7.06e-01 | 6.25e-01 |
| F1  | 0.4510 | 0.4183 | -0.0327 | 0.860 | 4.38e-01 | 4.38e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.7991 | -0.0059 | 0.360 | 7.37e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3754 | -0.0103 | 0.320 | 7.65e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1695 | -0.0105 | 1.498 | 2.09e-01 | 1.88e-01 |
| Accuracy  | 0.7491 | 0.7363 | -0.0128 | 0.339 | 7.52e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4022 | -0.0166 | 0.684 | 5.31e-01 | 6.25e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8136 | +0.0085 | -0.229 | 8.30e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3601 | -0.0256 | 0.447 | 6.78e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1826 | +0.0026 | -0.145 | 8.91e-01 | 8.12e-01 |
| Accuracy  | 0.7491 | 0.7581 | +0.0090 | -0.198 | 8.53e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.3992 | -0.0196 | 0.542 | 6.17e-01 | 8.12e-01 |

### len128/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8058 | +0.0007 | -0.045 | 9.66e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3730 | -0.0127 | 0.275 | 7.97e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1918 | +0.0118 | -1.304 | 2.62e-01 | 4.38e-01 |
| Accuracy  | 0.7491 | 0.7330 | -0.0161 | 0.498 | 6.45e-01 | 6.88e-01 |
| F1  | 0.4187 | 0.4068 | -0.0119 | 0.586 | 5.90e-01 | 6.25e-01 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8166 | +0.0116 | -0.909 | 4.15e-01 | 4.38e-01 |
| AUPRC  | 0.3857 | 0.3858 | +0.0001 | -0.002 | 9.98e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1907 | +0.0107 | -0.527 | 6.26e-01 | 1.00e+00 |
| Accuracy  | 0.7491 | 0.7869 | +0.0378 | -0.707 | 5.19e-01 | 5.00e-01 |
| F1  | 0.4187 | 0.4448 | +0.0261 | -0.781 | 4.79e-01 | 6.25e-01 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8017 | -0.0034 | 0.252 | 8.14e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3641 | -0.0216 | 0.495 | 6.47e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1891 | +0.0091 | -0.858 | 4.39e-01 | 4.38e-01 |
| Accuracy  | 0.7491 | 0.7589 | +0.0098 | -0.228 | 8.31e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4183 | -0.0004 | 0.017 | 9.87e-01 | 1.00e+00 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_clinic | AUC-ROC | 0.8119 | 0.7251 | 0.8882 |
| M1 | LR | scale_clinic | AUPRC | 0.4103 | 0.2449 | 0.5885 |
| M1 | LR | scale_clinic | Brier | 0.1878 | 0.1629 | 0.2143 |
| M1 | LR | scale_clinic | Accuracy | 0.7124 | 0.6524 | 0.7682 |
| M1 | LR | scale_clinic | F1 | 0.3853 | 0.2727 | 0.5079 |
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8277 | 0.7443 | 0.9053 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4404 | 0.2677 | 0.6336 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.1565 | 0.1331 | 0.1812 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7682 | 0.7124 | 0.8240 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.4000 | 0.2727 | 0.5263 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8262 | 0.7366 | 0.9007 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.4254 | 0.2384 | 0.6160 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1720 | 0.1449 | 0.2015 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.6890 | 0.6268 | 0.7464 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3564 | 0.2381 | 0.4696 |
| M2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.8017 | 0.7160 | 0.8812 |
| M2 | CrossAttn | len128/scale_clinic | AUPRC | 0.4105 | 0.2384 | 0.5998 |
| M2 | CrossAttn | len128/scale_clinic | Brier | 0.2302 | 0.1969 | 0.2636 |
| M2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6567 | 0.5966 | 0.7167 |
| M2 | CrossAttn | len128/scale_clinic | F1 | 0.3443 | 0.2373 | 0.4516 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8050 | 0.7190 | 0.8840 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.3764 | 0.2209 | 0.5692 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.1997 | 0.1685 | 0.2332 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6996 | 0.6395 | 0.7555 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.3636 | 0.2500 | 0.4812 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.7667 | 0.6545 | 0.8677 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3475 | 0.1990 | 0.5545 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.2070 | 0.1723 | 0.2431 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.7725 | 0.7167 | 0.8283 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.3908 | 0.2532 | 0.5208 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8294 | 0.7458 | 0.9045 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.3881 | 0.2308 | 0.5769 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.2231 | 0.1948 | 0.2531 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.6953 | 0.6352 | 0.7512 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.3717 | 0.2616 | 0.4844 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8357 | 0.7526 | 0.9136 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.4007 | 0.2360 | 0.6014 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.2085 | 0.1782 | 0.2399 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.6986 | 0.6364 | 0.7608 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3762 | 0.2529 | 0.4956 |
| M2_2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.8065 | 0.7213 | 0.8860 |
| M2_2 | CrossAttn | len128/scale_clinic | AUPRC | 0.3770 | 0.2138 | 0.5657 |
| M2_2 | CrossAttn | len128/scale_clinic | Brier | 0.1728 | 0.1526 | 0.1931 |
| M2_2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6266 | 0.5665 | 0.6910 |
| M2_2 | CrossAttn | len128/scale_clinic | F1 | 0.3256 | 0.2261 | 0.4297 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8146 | 0.7241 | 0.8927 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.3850 | 0.2278 | 0.5719 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.1983 | 0.1710 | 0.2256 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6781 | 0.6180 | 0.7339 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.3478 | 0.2376 | 0.4651 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8204 | 0.7349 | 0.8967 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.4127 | 0.2486 | 0.5983 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.2369 | 0.2059 | 0.2676 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6094 | 0.5494 | 0.6738 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.3158 | 0.2174 | 0.4225 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8154 | 0.7312 | 0.8952 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.4239 | 0.2474 | 0.6096 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.1953 | 0.1656 | 0.2269 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.6524 | 0.5923 | 0.7124 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.3306 | 0.2264 | 0.4414 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8230 | 0.7347 | 0.8957 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.4074 | 0.2275 | 0.6160 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.1681 | 0.1435 | 0.1937 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.6842 | 0.6220 | 0.7464 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.3400 | 0.2162 | 0.4583 |
| M3 | CrossAttn3 | len128/scale_clinic | AUC-ROC | 0.8077 | 0.7095 | 0.8958 |
| M3 | CrossAttn3 | len128/scale_clinic | AUPRC | 0.4203 | 0.2571 | 0.6227 |
| M3 | CrossAttn3 | len128/scale_clinic | Brier | 0.1613 | 0.1330 | 0.1931 |
| M3 | CrossAttn3 | len128/scale_clinic | Accuracy | 0.7639 | 0.7082 | 0.8197 |
| M3 | CrossAttn3 | len128/scale_clinic | F1 | 0.4086 | 0.2821 | 0.5307 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8058 | 0.7146 | 0.8874 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.4190 | 0.2421 | 0.6139 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.2086 | 0.1843 | 0.2346 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.7425 | 0.6866 | 0.7940 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.3617 | 0.2353 | 0.4828 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.8040 | 0.7137 | 0.8869 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.3966 | 0.2285 | 0.5910 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.1773 | 0.1522 | 0.2052 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.7296 | 0.6695 | 0.7854 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.3226 | 0.1951 | 0.4445 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-norm | 0.8119 | 0.8277 | +0.0158 | -0.687 | 4.920e-01 | ns |
| M1-LR vs M2-len128 | 0.8119 | 0.8017 | -0.0102 | 0.431 | 6.667e-01 | ns |
| M1-LR vs M2-crop80 | 0.8119 | 0.8050 | -0.0069 | 0.317 | 7.509e-01 | ns |
| M1-LR vs M2-crop60 | 0.8119 | 0.7667 | -0.0452 | 1.349 | 1.773e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-norm | 0.8119 | 0.8154 | +0.0035 | -0.150 | 8.807e-01 | ns |
| M1-LR vs M3-len128 | 0.8119 | 0.8077 | -0.0042 | 0.127 | 8.987e-01 | ns |
| M1-LR vs M3-crop80 | 0.8119 | 0.8058 | -0.0062 | 0.204 | 8.383e-01 | ns |
| M1-LR vs M3-crop60 | 0.8119 | 0.8040 | -0.0079 | 0.328 | 7.430e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M2_2-norm | 0.8277 | 0.8294 | +0.0017 | -0.134 | 8.931e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8262 | 0.4888 | -0.3374 | 5.284 | 1.261e-07 | *** |
| M2-len128 vs M2_2-len128 | 0.8017 | 0.8065 | +0.0048 | -0.189 | 8.504e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.8050 | 0.8146 | +0.0096 | -0.339 | 7.345e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.7667 | 0.8204 | +0.0537 | -1.620 | 1.053e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M3-norm | 0.8277 | 0.8154 | -0.0123 | 0.801 | 4.231e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.8262 | 0.8230 | -0.0032 | 0.147 | 8.834e-01 | ns |
| M2-len128 vs M3-len128 | 0.8017 | 0.8077 | +0.0060 | -0.224 | 8.230e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8050 | 0.8058 | +0.0008 | -0.031 | 9.755e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.7667 | 0.8040 | +0.0373 | -1.275 | 2.025e-01 | ns |

