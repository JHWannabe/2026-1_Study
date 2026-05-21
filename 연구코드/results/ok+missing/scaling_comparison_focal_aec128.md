# Scaling Comparison — Test Set Performance (AEC 128pt, FocalLoss)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8119 | 0.4103 | 0.1878 | 0.7124 | 0.3853 |
| M2 | CrossAttn | norm/scale_clinic | 0.8246 | 0.4301 | 0.1971 | 0.6481 | 0.3387 |
| M2_2 | CrossAttn | norm/scale_clinic | 0.8417 | 0.4241 | 0.1975 | 0.7425 | 0.4000 |
| M3 | CrossAttn3 | len128/scale_clinic | 0.8200 | 0.3430 | 0.2334 | 0.6609 | 0.3577 |

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
| **norm/scale_clinic** | 0.8246 | 0.4301 | 0.1971 | 0.6481 | 0.3387 |
| excl_extreme/scale_clinic | 0.7965 | 0.4256 | 0.2241 | 0.5502 | 0.2879 |
| len128/scale_clinic | 0.7973 | 0.3916 | 0.2093 | 0.6009 | 0.3111 |
| crop80/scale_clinic | 0.7685 | 0.3411 | 0.2191 | 0.6481 | 0.3279 |
| crop60/scale_clinic | 0.8042 | 0.3681 | 0.1995 | 0.6867 | 0.3540 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm/scale_clinic** | 0.8417 | 0.4241 | 0.1975 | 0.7425 | 0.4000 |
| excl_extreme/scale_clinic | 0.8029 | 0.3045 | 0.1645 | 0.7703 | 0.4000 |
| len128/scale_clinic | 0.7952 | 0.2848 | 0.1934 | 0.6567 | 0.3220 |
| crop80/scale_clinic | 0.7931 | 0.4111 | 0.1477 | 0.9013 | 0.3030 |
| crop60/scale_clinic | 0.8162 | 0.3766 | 0.2317 | 0.8069 | 0.3662 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm/scale_clinic | 0.8117 | 0.4054 | 0.2025 | 0.6738 | 0.3559 |
| excl_extreme/scale_clinic | 0.8194 | 0.4467 | 0.1996 | 0.6986 | 0.3368 |
| **len128/scale_clinic** | 0.8200 | 0.3430 | 0.2334 | 0.6609 | 0.3577 |
| crop80/scale_clinic | 0.8108 | 0.3455 | 0.1920 | 0.7296 | 0.3883 |
| crop60/scale_clinic | 0.8081 | 0.3739 | 0.1812 | 0.8455 | 0.4375 |

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
| AUC-ROC  | 0.8051 | 0.8181 | +0.0131 | -1.205 | 2.95e-01 | 3.12e-01 |
| AUPRC  | 0.3857 | 0.3782 | -0.0075 | 0.296 | 7.82e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.2093 | +0.0293 | -1.572 | 1.91e-01 | 3.12e-01 |
| Accuracy  | 0.7491 | 0.7006 | -0.0484 | 1.483 | 2.12e-01 | 1.88e-01 |
| F1  | 0.4187 | 0.4023 | -0.0164 | 0.812 | 4.63e-01 | 6.25e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8214 | +0.0163 | -0.935 | 4.03e-01 | 3.12e-01 |
| AUPRC  | 0.3857 | 0.3880 | +0.0024 | -0.056 | 9.58e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1933 | +0.0133 | -0.840 | 4.48e-01 | 6.25e-01 |
| Accuracy  | 0.7491 | 0.6850 | -0.0641 | 1.647 | 1.75e-01 | 1.88e-01 |
| F1  | 0.4187 | 0.3913 | -0.0274 | 0.997 | 3.75e-01 | 6.25e-01 |

### len128/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8251 | +0.0200 | -1.214 | 2.92e-01 | 3.12e-01 |
| AUPRC  | 0.3857 | 0.3912 | +0.0055 | -0.259 | 8.09e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1734 | -0.0066 | 0.499 | 6.44e-01 | 8.12e-01 |
| Accuracy  | 0.7491 | 0.7373 | -0.0118 | 0.259 | 8.09e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4271 | +0.0083 | -0.256 | 8.11e-01 | 8.75e-01 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8230 | +0.0179 | -1.603 | 1.84e-01 | 3.12e-01 |
| AUPRC  | 0.3857 | 0.4189 | +0.0333 | -0.712 | 5.16e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1952 | +0.0152 | -1.539 | 1.99e-01 | 1.88e-01 |
| Accuracy  | 0.7491 | 0.7427 | -0.0064 | 0.155 | 8.85e-01 | 1.00e+00 |
| F1  | 0.4187 | 0.4116 | -0.0071 | 0.247 | 8.17e-01 | 8.12e-01 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8116 | +0.0066 | -0.521 | 6.30e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.4105 | +0.0249 | -0.981 | 3.82e-01 | 3.12e-01 |
| Brier  | 0.1800 | 0.1819 | +0.0019 | -0.218 | 8.38e-01 | 8.12e-01 |
| Accuracy  | 0.7491 | 0.7653 | +0.0162 | -0.397 | 7.12e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4244 | +0.0057 | -0.238 | 8.24e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8181 | 0.7962 | -0.0219 | 2.508 | 6.62e-02 | 1.25e-01 |
| AUPRC  | 0.3782 | 0.3801 | +0.0019 | -0.171 | 8.73e-01 | 6.25e-01 |
| Brier  | 0.2093 | 0.1930 | -0.0163 | 1.185 | 3.01e-01 | 4.38e-01 |
| Accuracy  | 0.7006 | 0.7653 | +0.0647 | -1.600 | 1.85e-01 | 2.50e-01 |
| F1  | 0.4023 | 0.4197 | +0.0174 | -0.759 | 4.90e-01 | 6.25e-01 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8214 | 0.8169 | -0.0045 | 0.784 | 4.77e-01 | 4.38e-01 |
| AUPRC  | 0.3880 | 0.3772 | -0.0108 | 0.510 | 6.37e-01 | 6.25e-01 |
| Brier  | 0.1933 | 0.1872 | -0.0061 | 0.444 | 6.80e-01 | 8.12e-01 |
| Accuracy  | 0.6850 | 0.7174 | +0.0323 | -0.826 | 4.55e-01 | 3.75e-01 |
| F1  | 0.3913 | 0.4014 | +0.0101 | -0.331 | 7.57e-01 | 8.75e-01 |

#### Case: len128/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8251 | 0.8100 | -0.0150 | 1.327 | 2.55e-01 | 3.12e-01 |
| AUPRC  | 0.3912 | 0.3679 | -0.0233 | 0.688 | 5.30e-01 | 8.12e-01 |
| Brier * | 0.1734 | 0.2121 | +0.0387 | -3.704 | 2.08e-02 | 6.25e-02 |
| Accuracy  | 0.7373 | 0.7761 | +0.0388 | -0.691 | 5.28e-01 | 6.25e-01 |
| F1  | 0.4271 | 0.4339 | +0.0068 | -0.172 | 8.72e-01 | 1.00e+00 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8230 | 0.8068 | -0.0162 | 1.829 | 1.41e-01 | 1.88e-01 |
| AUPRC  | 0.4189 | 0.3996 | -0.0193 | 0.502 | 6.42e-01 | 6.25e-01 |
| Brier  | 0.1952 | 0.1767 | -0.0185 | 1.614 | 1.82e-01 | 3.12e-01 |
| Accuracy * | 0.7427 | 0.8019 | +0.0593 | -3.128 | 3.52e-02 | 6.25e-02 |
| F1 * | 0.4116 | 0.4489 | +0.0372 | -3.546 | 2.39e-02 | 6.25e-02 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8116 | 0.8054 | -0.0062 | 0.642 | 5.56e-01 | 8.12e-01 |
| AUPRC  | 0.4105 | 0.3871 | -0.0234 | 0.820 | 4.58e-01 | 4.38e-01 |
| Brier † | 0.1819 | 0.2173 | +0.0354 | -2.506 | 6.63e-02 | 6.25e-02 |
| Accuracy  | 0.7653 | 0.7815 | +0.0162 | -0.789 | 4.74e-01 | 5.00e-01 |
| F1  | 0.4244 | 0.4270 | +0.0026 | -0.144 | 8.92e-01 | 1.00e+00 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.7962 | -0.0088 | 0.471 | 6.62e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3801 | -0.0056 | 0.194 | 8.56e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1930 | +0.0130 | -1.918 | 1.28e-01 | 1.88e-01 |
| Accuracy  | 0.7491 | 0.7653 | +0.0162 | -0.460 | 6.69e-01 | 6.25e-01 |
| F1  | 0.4187 | 0.4197 | +0.0010 | -0.057 | 9.57e-01 | 8.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8169 | +0.0119 | -0.811 | 4.63e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3772 | -0.0084 | 0.312 | 7.71e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1872 | +0.0072 | -0.781 | 4.78e-01 | 8.12e-01 |
| Accuracy  | 0.7491 | 0.7174 | -0.0317 | 0.483 | 6.54e-01 | 6.25e-01 |
| F1  | 0.4187 | 0.4014 | -0.0174 | 0.455 | 6.73e-01 | 6.25e-01 |

### len128/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8100 | +0.0050 | -0.244 | 8.19e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3679 | -0.0178 | 0.359 | 7.38e-01 | 8.12e-01 |
| Brier † | 0.1800 | 0.2121 | +0.0321 | -2.142 | 9.89e-02 | 3.12e-01 |
| Accuracy  | 0.7491 | 0.7761 | +0.0271 | -0.449 | 6.77e-01 | 8.12e-01 |
| F1  | 0.4187 | 0.4339 | +0.0151 | -0.445 | 6.79e-01 | 1.00e+00 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8068 | +0.0017 | -0.127 | 9.05e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3996 | +0.0139 | -0.318 | 7.67e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1767 | -0.0033 | 0.538 | 6.19e-01 | 1.00e+00 |
| Accuracy  | 0.7491 | 0.8019 | +0.0529 | -1.326 | 2.55e-01 | 3.12e-01 |
| F1  | 0.4187 | 0.4489 | +0.0301 | -1.402 | 2.34e-01 | 3.12e-01 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8054 | +0.0004 | -0.023 | 9.83e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3871 | +0.0014 | -0.031 | 9.77e-01 | 1.00e+00 |
| Brier † | 0.1800 | 0.2173 | +0.0373 | -2.639 | 5.77e-02 | 1.25e-01 |
| Accuracy  | 0.7491 | 0.7815 | +0.0324 | -0.700 | 5.23e-01 | 7.50e-01 |
| F1  | 0.4187 | 0.4270 | +0.0082 | -0.320 | 7.65e-01 | 8.12e-01 |

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
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8246 | 0.7402 | 0.9017 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4301 | 0.2585 | 0.6136 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.1971 | 0.1852 | 0.2089 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.6481 | 0.5837 | 0.7082 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.3387 | 0.2314 | 0.4516 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.7965 | 0.6759 | 0.8914 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.4256 | 0.2470 | 0.6327 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.2241 | 0.2088 | 0.2404 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.5502 | 0.4833 | 0.6172 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.2879 | 0.1846 | 0.3857 |
| M2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.7973 | 0.7012 | 0.8826 |
| M2 | CrossAttn | len128/scale_clinic | AUPRC | 0.3916 | 0.2294 | 0.5928 |
| M2 | CrossAttn | len128/scale_clinic | Brier | 0.2093 | 0.1932 | 0.2262 |
| M2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6009 | 0.5365 | 0.6609 |
| M2 | CrossAttn | len128/scale_clinic | F1 | 0.3111 | 0.2121 | 0.4118 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.7685 | 0.6667 | 0.8637 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.3411 | 0.1938 | 0.5287 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.2191 | 0.1998 | 0.2383 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6481 | 0.5880 | 0.7040 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.3279 | 0.2222 | 0.4386 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8042 | 0.7155 | 0.8846 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3681 | 0.2154 | 0.5584 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.1995 | 0.1859 | 0.2133 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6867 | 0.6266 | 0.7425 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.3540 | 0.2419 | 0.4673 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8417 | 0.7606 | 0.9144 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4241 | 0.2629 | 0.6184 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.1975 | 0.1807 | 0.2143 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7425 | 0.6867 | 0.7983 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.4000 | 0.2784 | 0.5243 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8029 | 0.7045 | 0.8899 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3045 | 0.1837 | 0.5203 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1645 | 0.1464 | 0.1826 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7703 | 0.7129 | 0.8278 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4000 | 0.2534 | 0.5287 |
| M2_2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.7952 | 0.7110 | 0.8717 |
| M2_2 | CrossAttn | len128/scale_clinic | AUPRC | 0.2848 | 0.1785 | 0.4743 |
| M2_2 | CrossAttn | len128/scale_clinic | Brier | 0.1934 | 0.1795 | 0.2072 |
| M2_2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6567 | 0.5923 | 0.7167 |
| M2_2 | CrossAttn | len128/scale_clinic | F1 | 0.3220 | 0.2150 | 0.4333 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.7931 | 0.6895 | 0.8828 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.4111 | 0.2450 | 0.5960 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.1477 | 0.1356 | 0.1612 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.9013 | 0.8584 | 0.9399 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.3030 | 0.0833 | 0.5000 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8162 | 0.7307 | 0.8927 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3766 | 0.2187 | 0.5659 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.2317 | 0.2168 | 0.2469 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.8069 | 0.7554 | 0.8584 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.3662 | 0.2104 | 0.5063 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8117 | 0.7179 | 0.8942 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.4054 | 0.2397 | 0.5936 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.2025 | 0.1866 | 0.2185 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.6738 | 0.6137 | 0.7297 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.3559 | 0.2474 | 0.4672 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8194 | 0.7292 | 0.8928 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.4467 | 0.2539 | 0.6447 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.1996 | 0.1852 | 0.2147 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.6986 | 0.6364 | 0.7560 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.3368 | 0.2078 | 0.4571 |
| M3 | CrossAttn3 | len128/scale_clinic | AUC-ROC | 0.8200 | 0.7408 | 0.8921 |
| M3 | CrossAttn3 | len128/scale_clinic | AUPRC | 0.3430 | 0.2107 | 0.5517 |
| M3 | CrossAttn3 | len128/scale_clinic | Brier | 0.2334 | 0.2193 | 0.2479 |
| M3 | CrossAttn3 | len128/scale_clinic | Accuracy | 0.6609 | 0.5966 | 0.7210 |
| M3 | CrossAttn3 | len128/scale_clinic | F1 | 0.3577 | 0.2523 | 0.4685 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8108 | 0.7260 | 0.8862 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.3455 | 0.2034 | 0.5395 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.1920 | 0.1766 | 0.2082 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.7296 | 0.6695 | 0.7897 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.3883 | 0.2697 | 0.5079 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.8081 | 0.7192 | 0.8871 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.3739 | 0.2249 | 0.6005 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.1812 | 0.1688 | 0.1939 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.8455 | 0.7983 | 0.8927 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.4375 | 0.2711 | 0.5797 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-norm | 0.8119 | 0.8246 | +0.0127 | -0.510 | 6.098e-01 | ns |
| M1-LR vs M2-len128 | 0.8119 | 0.7973 | -0.0146 | 0.496 | 6.201e-01 | ns |
| M1-LR vs M2-crop80 | 0.8119 | 0.7685 | -0.0435 | 1.541 | 1.233e-01 | ns |
| M1-LR vs M2-crop60 | 0.8119 | 0.8042 | -0.0077 | 0.325 | 7.453e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-norm | 0.8119 | 0.8117 | -0.0002 | 0.006 | 9.949e-01 | ns |
| M1-LR vs M3-len128 | 0.8119 | 0.8200 | +0.0081 | -0.405 | 6.857e-01 | ns |
| M1-LR vs M3-crop80 | 0.8119 | 0.8108 | -0.0012 | 0.042 | 9.668e-01 | ns |
| M1-LR vs M3-crop60 | 0.8119 | 0.8081 | -0.0038 | 0.145 | 8.848e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M2_2-norm | 0.8246 | 0.8417 | +0.0171 | -0.996 | 3.193e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.7965 | 0.5095 | -0.2871 | 3.679 | 2.342e-04 | *** |
| M2-len128 vs M2_2-len128 | 0.7973 | 0.7952 | -0.0021 | 0.057 | 9.549e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.7685 | 0.7931 | +0.0246 | -0.705 | 4.811e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.8042 | 0.8162 | +0.0119 | -0.460 | 6.457e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M3-norm | 0.8246 | 0.8117 | -0.0129 | 0.614 | 5.391e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.7965 | 0.8194 | +0.0228 | -0.705 | 4.809e-01 | ns |
| M2-len128 vs M3-len128 | 0.7973 | 0.8200 | +0.0227 | -0.981 | 3.266e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.7685 | 0.8108 | +0.0423 | -1.636 | 1.019e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.8042 | 0.8081 | +0.0038 | -0.134 | 8.936e-01 | ns |

