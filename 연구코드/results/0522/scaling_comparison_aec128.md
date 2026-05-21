# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8119 | 0.4103 | 0.1878 | 0.7167 | 0.3889 |
| M2 | CrossAttn | excl_extreme/scale_both | 0.8223 | 0.3961 | 0.2589 | 0.6077 | 0.3279 |
| M2_2 | CrossAttn | crop60/scale_both | 0.8229 | 0.4278 | 0.1361 | 0.7811 | 0.3855 |
| M3 | CrossAttn3 | norm/scale_both | 0.8329 | 0.4338 | 0.2288 | 0.6824 | 0.3621 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_clinic** | 0.8119 | 0.4103 | 0.1878 | 0.7167 | 0.3889 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.8073 | 0.4147 | 0.1923 | 0.7167 | 0.3654 |
| crop80/scale_both | 0.8142 | 0.4009 | 0.2676 | 0.5837 | 0.3121 |
| crop60/scale_both | 0.8035 | 0.3821 | 0.2048 | 0.6738 | 0.3448 |
| norm/scale_both | 0.8044 | 0.3875 | 0.2422 | 0.6180 | 0.3407 |
| **excl_extreme/scale_both** | 0.8223 | 0.3961 | 0.2589 | 0.6077 | 0.3279 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.7754 | 0.3095 | 0.1701 | 0.7382 | 0.3838 |
| crop80/scale_both | 0.8073 | 0.3985 | 0.2005 | 0.6481 | 0.3279 |
| **crop60/scale_both** | 0.8229 | 0.4278 | 0.1361 | 0.7811 | 0.3855 |
| norm/scale_both | 0.8198 | 0.3463 | 0.1654 | 0.7768 | 0.4348 |
| excl_extreme/scale_both | 0.7910 | 0.2751 | 0.2496 | 0.6029 | 0.3252 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.8156 | 0.4169 | 0.1823 | 0.7382 | 0.3960 |
| crop80/scale_both | 0.7862 | 0.3661 | 0.1923 | 0.7167 | 0.3400 |
| crop60/scale_both | 0.7821 | 0.3833 | 0.1655 | 0.7382 | 0.3441 |
| **norm/scale_both** | 0.8329 | 0.4338 | 0.2288 | 0.6824 | 0.3621 |
| excl_extreme/scale_both | 0.8264 | 0.4025 | 0.1934 | 0.6986 | 0.3226 |

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
| AUC-ROC  | 0.8051 | 0.8256 | +0.0206 | -1.810 | 1.45e-01 | 1.88e-01 |
| AUPRC  | 0.3857 | 0.4273 | +0.0416 | -0.955 | 3.93e-01 | 4.38e-01 |
| Brier  | 0.1800 | 0.1758 | -0.0042 | 0.431 | 6.89e-01 | 6.25e-01 |
| Accuracy  | 0.7395 | 0.7180 | -0.0215 | 2.077 | 1.06e-01 | 1.88e-01 |
| F1  | 0.3714 | 0.3825 | +0.0111 | -1.256 | 2.78e-01 | 4.38e-01 |

### crop80/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8178 | +0.0127 | -1.142 | 3.17e-01 | 3.12e-01 |
| AUPRC  | 0.3857 | 0.4271 | +0.0414 | -1.145 | 3.16e-01 | 3.12e-01 |
| Brier  | 0.1800 | 0.1867 | +0.0067 | -0.445 | 6.79e-01 | 8.12e-01 |
| Accuracy  | 0.7395 | 0.7104 | -0.0291 | 0.991 | 3.78e-01 | 6.25e-01 |
| F1  | 0.3714 | 0.3848 | +0.0134 | -0.618 | 5.70e-01 | 4.38e-01 |

### crop60/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8140 | +0.0089 | -0.713 | 5.15e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3804 | -0.0053 | 0.170 | 8.74e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1911 | +0.0111 | -0.693 | 5.26e-01 | 8.12e-01 |
| Accuracy  | 0.7395 | 0.7007 | -0.0388 | 1.130 | 3.22e-01 | 3.75e-01 |
| F1  | 0.3714 | 0.3690 | -0.0024 | 0.081 | 9.40e-01 | 8.12e-01 |

### norm/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8244 | +0.0193 | -1.275 | 2.71e-01 | 4.38e-01 |
| AUPRC  | 0.3857 | 0.3807 | -0.0049 | 0.158 | 8.82e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1983 | +0.0183 | -1.083 | 3.40e-01 | 3.12e-01 |
| Accuracy  | 0.7395 | 0.6857 | -0.0537 | 1.446 | 2.22e-01 | 3.12e-01 |
| F1  | 0.3714 | 0.3548 | -0.0166 | 0.679 | 5.34e-01 | 6.25e-01 |

### excl_extreme/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8197 | +0.0146 | -0.675 | 5.37e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3745 | -0.0112 | 0.343 | 7.49e-01 | 1.00e+00 |
| Brier † | 0.1800 | 0.2059 | +0.0259 | -2.298 | 8.31e-02 | 1.25e-01 |
| Accuracy  | 0.7395 | 0.6982 | -0.0413 | 1.735 | 1.58e-01 | 1.25e-01 |
| F1  | 0.3714 | 0.3621 | -0.0093 | 0.325 | 7.62e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len128/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8256 | 0.8051 | -0.0206 | 1.976 | 1.19e-01 | 6.25e-02 |
| AUPRC  | 0.4273 | 0.3755 | -0.0518 | 1.573 | 1.91e-01 | 3.12e-01 |
| Brier  | 0.1758 | 0.1746 | -0.0012 | 0.088 | 9.34e-01 | 1.00e+00 |
| Accuracy  | 0.7180 | 0.7276 | +0.0096 | -0.365 | 7.33e-01 | 8.75e-01 |
| F1  | 0.3825 | 0.3661 | -0.0164 | 1.715 | 1.62e-01 | 3.12e-01 |

#### Case: crop80/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8178 | 0.8127 | -0.0051 | 0.809 | 4.64e-01 | 6.25e-01 |
| AUPRC  | 0.4271 | 0.3850 | -0.0422 | 1.409 | 2.32e-01 | 4.38e-01 |
| Brier  | 0.1867 | 0.2064 | +0.0197 | -1.622 | 1.80e-01 | 3.12e-01 |
| Accuracy  | 0.7104 | 0.6878 | -0.0225 | 0.891 | 4.23e-01 | 6.25e-01 |
| F1  | 0.3848 | 0.3744 | -0.0104 | 0.573 | 5.97e-01 | 6.25e-01 |

#### Case: crop60/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8140 | 0.8114 | -0.0026 | 0.626 | 5.65e-01 | 1.00e+00 |
| AUPRC  | 0.3804 | 0.3804 | -0.0000 | 0.002 | 9.99e-01 | 8.12e-01 |
| Brier  | 0.1911 | 0.1751 | -0.0160 | 0.747 | 4.97e-01 | 6.25e-01 |
| Accuracy  | 0.7007 | 0.7373 | +0.0367 | -0.892 | 4.23e-01 | 5.00e-01 |
| F1  | 0.3690 | 0.3854 | +0.0164 | -0.755 | 4.92e-01 | 6.25e-01 |

#### Case: norm/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8244 | 0.8119 | -0.0125 | 1.310 | 2.60e-01 | 3.12e-01 |
| AUPRC  | 0.3807 | 0.3962 | +0.0155 | -0.678 | 5.35e-01 | 8.12e-01 |
| Brier  | 0.1983 | 0.1982 | -0.0001 | 0.005 | 9.97e-01 | 1.00e+00 |
| Accuracy  | 0.6857 | 0.7007 | +0.0150 | -0.362 | 7.36e-01 | 8.12e-01 |
| F1  | 0.3548 | 0.3627 | +0.0079 | -0.375 | 7.27e-01 | 8.12e-01 |

#### Case: excl_extreme/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8197 | 0.8159 | -0.0038 | 0.395 | 7.13e-01 | 1.00e+00 |
| AUPRC † | 0.3745 | 0.4060 | +0.0315 | -2.235 | 8.91e-02 | 1.25e-01 |
| Brier  | 0.2059 | 0.1815 | -0.0243 | 1.455 | 2.19e-01 | 3.12e-01 |
| Accuracy  | 0.6982 | 0.7210 | +0.0228 | -0.716 | 5.13e-01 | 8.12e-01 |
| F1  | 0.3621 | 0.3816 | +0.0195 | -0.800 | 4.69e-01 | 4.38e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8051 | +0.0000 | -0.000 | 1.00e+00 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3755 | -0.0102 | 0.426 | 6.92e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1746 | -0.0054 | 0.683 | 5.32e-01 | 6.25e-01 |
| Accuracy  | 0.7395 | 0.7276 | -0.0119 | 0.600 | 5.81e-01 | 7.50e-01 |
| F1  | 0.3714 | 0.3661 | -0.0053 | 0.387 | 7.18e-01 | 1.00e+00 |

### crop80/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8127 | +0.0076 | -0.658 | 5.47e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3850 | -0.0007 | 0.020 | 9.85e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.2064 | +0.0264 | -1.734 | 1.58e-01 | 3.12e-01 |
| Accuracy  | 0.7395 | 0.6878 | -0.0516 | 1.336 | 2.52e-01 | 3.12e-01 |
| F1  | 0.3714 | 0.3744 | +0.0031 | -0.098 | 9.27e-01 | 6.25e-01 |

### crop60/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8114 | +0.0063 | -0.435 | 6.86e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3804 | -0.0053 | 0.128 | 9.04e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1751 | -0.0049 | 0.403 | 7.07e-01 | 8.12e-01 |
| Accuracy  | 0.7395 | 0.7373 | -0.0021 | 0.101 | 9.24e-01 | 8.75e-01 |
| F1  | 0.3714 | 0.3854 | +0.0141 | -0.960 | 3.91e-01 | 3.12e-01 |

### norm/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8119 | +0.0069 | -0.538 | 6.19e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3962 | +0.0105 | -0.606 | 5.77e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.1982 | +0.0182 | -1.578 | 1.90e-01 | 3.12e-01 |
| Accuracy  | 0.7395 | 0.7007 | -0.0388 | 1.817 | 1.43e-01 | 1.25e-01 |
| F1  | 0.3714 | 0.3627 | -0.0087 | 0.975 | 3.85e-01 | 4.38e-01 |

### excl_extreme/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8159 | +0.0108 | -0.732 | 5.05e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.4060 | +0.0203 | -0.595 | 5.84e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.1815 | +0.0016 | -0.066 | 9.51e-01 | 1.00e+00 |
| Accuracy  | 0.7395 | 0.7210 | -0.0185 | 0.387 | 7.19e-01 | 6.25e-01 |
| F1  | 0.3714 | 0.3816 | +0.0102 | -0.277 | 7.96e-01 | 1.00e+00 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_clinic | AUC-ROC | 0.8119 | 0.7251 | 0.8882 |
| M1 | LR | scale_clinic | AUPRC | 0.4103 | 0.2449 | 0.5885 |
| M1 | LR | scale_clinic | Brier | 0.1878 | 0.1629 | 0.2143 |
| M1 | LR | scale_clinic | Accuracy | 0.7167 | 0.6567 | 0.7725 |
| M1 | LR | scale_clinic | F1 | 0.3889 | 0.2752 | 0.5098 |
| M2 | CrossAttn | len128/scale_both | AUC-ROC | 0.8073 | 0.7054 | 0.8948 |
| M2 | CrossAttn | len128/scale_both | AUPRC | 0.4147 | 0.2495 | 0.6238 |
| M2 | CrossAttn | len128/scale_both | Brier | 0.1923 | 0.1589 | 0.2261 |
| M2 | CrossAttn | len128/scale_both | Accuracy | 0.7167 | 0.6567 | 0.7768 |
| M2 | CrossAttn | len128/scale_both | F1 | 0.3654 | 0.2472 | 0.4886 |
| M2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.8142 | 0.7240 | 0.8936 |
| M2 | CrossAttn | crop80/scale_both | AUPRC | 0.4009 | 0.2363 | 0.5953 |
| M2 | CrossAttn | crop80/scale_both | Brier | 0.2676 | 0.2308 | 0.3055 |
| M2 | CrossAttn | crop80/scale_both | Accuracy | 0.5837 | 0.5193 | 0.6481 |
| M2 | CrossAttn | crop80/scale_both | F1 | 0.3121 | 0.2170 | 0.4138 |
| M2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.8035 | 0.7017 | 0.8915 |
| M2 | CrossAttn | crop60/scale_both | AUPRC | 0.3821 | 0.2286 | 0.5724 |
| M2 | CrossAttn | crop60/scale_both | Brier | 0.2048 | 0.1754 | 0.2354 |
| M2 | CrossAttn | crop60/scale_both | Accuracy | 0.6738 | 0.6137 | 0.7339 |
| M2 | CrossAttn | crop60/scale_both | F1 | 0.3448 | 0.2364 | 0.4615 |
| M2 | CrossAttn | norm/scale_both | AUC-ROC | 0.8044 | 0.7199 | 0.8765 |
| M2 | CrossAttn | norm/scale_both | AUPRC | 0.3875 | 0.2168 | 0.5671 |
| M2 | CrossAttn | norm/scale_both | Brier | 0.2422 | 0.2104 | 0.2768 |
| M2 | CrossAttn | norm/scale_both | Accuracy | 0.6180 | 0.5536 | 0.6781 |
| M2 | CrossAttn | norm/scale_both | F1 | 0.3407 | 0.2424 | 0.4463 |
| M2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8223 | 0.7319 | 0.9000 |
| M2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.3961 | 0.2351 | 0.6018 |
| M2 | CrossAttn | excl_extreme/scale_both | Brier | 0.2589 | 0.2225 | 0.2970 |
| M2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.6077 | 0.5407 | 0.6746 |
| M2 | CrossAttn | excl_extreme/scale_both | F1 | 0.3279 | 0.2185 | 0.4380 |
| M2_2 | CrossAttn | len128/scale_both | AUC-ROC | 0.7754 | 0.6798 | 0.8608 |
| M2_2 | CrossAttn | len128/scale_both | AUPRC | 0.3095 | 0.1779 | 0.4744 |
| M2_2 | CrossAttn | len128/scale_both | Brier | 0.1701 | 0.1428 | 0.1977 |
| M2_2 | CrossAttn | len128/scale_both | Accuracy | 0.7382 | 0.6781 | 0.7897 |
| M2_2 | CrossAttn | len128/scale_both | F1 | 0.3838 | 0.2637 | 0.5051 |
| M2_2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.8073 | 0.7223 | 0.8846 |
| M2_2 | CrossAttn | crop80/scale_both | AUPRC | 0.3985 | 0.2308 | 0.5880 |
| M2_2 | CrossAttn | crop80/scale_both | Brier | 0.2005 | 0.1772 | 0.2244 |
| M2_2 | CrossAttn | crop80/scale_both | Accuracy | 0.6481 | 0.5880 | 0.7082 |
| M2_2 | CrossAttn | crop80/scale_both | F1 | 0.3279 | 0.2185 | 0.4366 |
| M2_2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.8229 | 0.7425 | 0.8943 |
| M2_2 | CrossAttn | crop60/scale_both | AUPRC | 0.4278 | 0.2546 | 0.6082 |
| M2_2 | CrossAttn | crop60/scale_both | Brier | 0.1361 | 0.1158 | 0.1566 |
| M2_2 | CrossAttn | crop60/scale_both | Accuracy | 0.7811 | 0.7253 | 0.8326 |
| M2_2 | CrossAttn | crop60/scale_both | F1 | 0.3855 | 0.2466 | 0.5149 |
| M2_2 | CrossAttn | norm/scale_both | AUC-ROC | 0.8198 | 0.7300 | 0.8985 |
| M2_2 | CrossAttn | norm/scale_both | AUPRC | 0.3463 | 0.2188 | 0.5531 |
| M2_2 | CrossAttn | norm/scale_both | Brier | 0.1654 | 0.1430 | 0.1883 |
| M2_2 | CrossAttn | norm/scale_both | Accuracy | 0.7768 | 0.7210 | 0.8241 |
| M2_2 | CrossAttn | norm/scale_both | F1 | 0.4348 | 0.3077 | 0.5567 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.7910 | 0.6961 | 0.8743 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.2751 | 0.1670 | 0.4750 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Brier | 0.2496 | 0.2129 | 0.2865 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.6029 | 0.5359 | 0.6699 |
| M2_2 | CrossAttn | excl_extreme/scale_both | F1 | 0.3252 | 0.2170 | 0.4329 |
| M3 | CrossAttn3 | len128/scale_both | AUC-ROC | 0.8156 | 0.7177 | 0.9015 |
| M3 | CrossAttn3 | len128/scale_both | AUPRC | 0.4169 | 0.2509 | 0.6123 |
| M3 | CrossAttn3 | len128/scale_both | Brier | 0.1823 | 0.1532 | 0.2143 |
| M3 | CrossAttn3 | len128/scale_both | Accuracy | 0.7382 | 0.6781 | 0.7940 |
| M3 | CrossAttn3 | len128/scale_both | F1 | 0.3960 | 0.2745 | 0.5167 |
| M3 | CrossAttn3 | crop80/scale_both | AUC-ROC | 0.7862 | 0.6918 | 0.8733 |
| M3 | CrossAttn3 | crop80/scale_both | AUPRC | 0.3661 | 0.2091 | 0.5536 |
| M3 | CrossAttn3 | crop80/scale_both | Brier | 0.1923 | 0.1605 | 0.2272 |
| M3 | CrossAttn3 | crop80/scale_both | Accuracy | 0.7167 | 0.6567 | 0.7725 |
| M3 | CrossAttn3 | crop80/scale_both | F1 | 0.3400 | 0.2157 | 0.4602 |
| M3 | CrossAttn3 | crop60/scale_both | AUC-ROC | 0.7821 | 0.6739 | 0.8761 |
| M3 | CrossAttn3 | crop60/scale_both | AUPRC | 0.3833 | 0.2234 | 0.5747 |
| M3 | CrossAttn3 | crop60/scale_both | Brier | 0.1655 | 0.1379 | 0.1961 |
| M3 | CrossAttn3 | crop60/scale_both | Accuracy | 0.7382 | 0.6824 | 0.7940 |
| M3 | CrossAttn3 | crop60/scale_both | F1 | 0.3441 | 0.2192 | 0.4660 |
| M3 | CrossAttn3 | norm/scale_both | AUC-ROC | 0.8329 | 0.7499 | 0.9088 |
| M3 | CrossAttn3 | norm/scale_both | AUPRC | 0.4338 | 0.2589 | 0.6309 |
| M3 | CrossAttn3 | norm/scale_both | Brier | 0.2288 | 0.1944 | 0.2651 |
| M3 | CrossAttn3 | norm/scale_both | Accuracy | 0.6824 | 0.6223 | 0.7468 |
| M3 | CrossAttn3 | norm/scale_both | F1 | 0.3621 | 0.2524 | 0.4783 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUC-ROC | 0.8264 | 0.7437 | 0.8966 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUPRC | 0.4025 | 0.2234 | 0.6035 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Brier | 0.1934 | 0.1628 | 0.2279 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Accuracy | 0.6986 | 0.6364 | 0.7560 |
| M3 | CrossAttn3 | excl_extreme/scale_both | F1 | 0.3226 | 0.1927 | 0.4425 |

