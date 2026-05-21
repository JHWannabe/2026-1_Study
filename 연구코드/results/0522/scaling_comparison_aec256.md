# Scaling Comparison — Test Set Performance (AEC 256pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8119 | 0.4103 | 0.1878 | 0.7167 | 0.3889 |
| M2 | CrossAttn | excl_extreme/scale_both | 0.8189 | 0.4083 | 0.2040 | 0.6316 | 0.3304 |
| M2_2 | CrossAttn | crop60/scale_both | 0.8265 | 0.3845 | 0.2078 | 0.6567 | 0.3443 |
| M3 | CrossAttn3 | norm/scale_both | 0.8352 | 0.3975 | 0.1739 | 0.7468 | 0.4040 |

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
| len256/scale_both | 0.7887 | 0.4100 | 0.2169 | 0.6609 | 0.3248 |
| crop80/scale_both | 0.8010 | 0.3900 | 0.2547 | 0.5837 | 0.3022 |
| crop60/scale_both | 0.8156 | 0.4163 | 0.2130 | 0.6824 | 0.3509 |
| norm/scale_both | 0.7873 | 0.3616 | 0.2584 | 0.5536 | 0.3067 |
| **excl_extreme/scale_both** | 0.8189 | 0.4083 | 0.2040 | 0.6316 | 0.3304 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_both | 0.8025 | 0.3403 | 0.1796 | 0.7039 | 0.3551 |
| crop80/scale_both | 0.7881 | 0.2942 | 0.2082 | 0.6781 | 0.3590 |
| **crop60/scale_both** | 0.8265 | 0.3845 | 0.2078 | 0.6567 | 0.3443 |
| norm/scale_both | 0.8038 | 0.3559 | 0.1815 | 0.7339 | 0.3922 |
| excl_extreme/scale_both | 0.8175 | 0.4052 | 0.1691 | 0.7321 | 0.3778 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_both | 0.8019 | 0.3396 | 0.2312 | 0.6695 | 0.3419 |
| crop80/scale_both | 0.7952 | 0.3873 | 0.1818 | 0.7210 | 0.3434 |
| crop60/scale_both | 0.8098 | 0.4021 | 0.1245 | 0.7897 | 0.3467 |
| **norm/scale_both** | 0.8352 | 0.3975 | 0.1739 | 0.7468 | 0.4040 |
| excl_extreme/scale_both | 0.8116 | 0.4055 | 0.1772 | 0.7273 | 0.3736 |

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
| AUC-ROC  | 0.8051 | 0.8220 | +0.0169 | -1.091 | 3.37e-01 | 4.38e-01 |
| AUPRC  | 0.3857 | 0.4081 | +0.0224 | -0.536 | 6.21e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1762 | -0.0038 | 0.257 | 8.10e-01 | 6.25e-01 |
| Accuracy  | 0.7395 | 0.7201 | -0.0194 | 1.038 | 3.58e-01 | 6.25e-01 |
| F1  | 0.3714 | 0.3895 | +0.0181 | -1.100 | 3.33e-01 | 4.38e-01 |

### crop80/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8112 | +0.0061 | -0.386 | 7.19e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3849 | -0.0008 | 0.024 | 9.82e-01 | 1.00e+00 |
| Brier † | 0.1800 | 0.1544 | -0.0256 | 2.605 | 5.97e-02 | 1.25e-01 |
| Accuracy  | 0.7395 | 0.7652 | +0.0258 | -1.299 | 2.64e-01 | 3.12e-01 |
| F1  | 0.3714 | 0.3993 | +0.0279 | -1.746 | 1.56e-01 | 2.50e-01 |

### crop60/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8262 | +0.0212 | -1.521 | 2.03e-01 | 4.38e-01 |
| AUPRC  | 0.3857 | 0.4337 | +0.0480 | -0.835 | 4.51e-01 | 4.38e-01 |
| Brier  | 0.1800 | 0.1748 | -0.0052 | 0.478 | 6.58e-01 | 8.12e-01 |
| Accuracy  | 0.7395 | 0.7276 | -0.0118 | 0.423 | 6.94e-01 | 8.12e-01 |
| F1  | 0.3714 | 0.3842 | +0.0129 | -0.528 | 6.26e-01 | 6.25e-01 |

### norm/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8178 | +0.0127 | -0.886 | 4.26e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3368 | -0.0489 | 1.260 | 2.76e-01 | 4.38e-01 |
| Brier  | 0.1800 | 0.1724 | -0.0076 | 0.277 | 7.95e-01 | 6.25e-01 |
| Accuracy  | 0.7395 | 0.7040 | -0.0355 | 0.515 | 6.34e-01 | 1.00e+00 |
| F1  | 0.3714 | 0.3503 | -0.0211 | 0.639 | 5.58e-01 | 8.12e-01 |

### excl_extreme/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8259 | +0.0208 | -0.542 | 6.17e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3908 | +0.0051 | -0.097 | 9.28e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1855 | +0.0055 | -0.373 | 7.28e-01 | 8.12e-01 |
| Accuracy  | 0.7395 | 0.6982 | -0.0413 | 0.985 | 3.80e-01 | 4.38e-01 |
| F1  | 0.3714 | 0.3568 | -0.0146 | 0.354 | 7.41e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len256/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8220 | 0.8047 | -0.0173 | 1.561 | 1.94e-01 | 1.88e-01 |
| AUPRC ** | 0.4081 | 0.3782 | -0.0300 | 5.172 | 6.64e-03 | 6.25e-02 |
| Brier  | 0.1762 | 0.1801 | +0.0040 | -0.335 | 7.54e-01 | 6.25e-01 |
| Accuracy  | 0.7201 | 0.7427 | +0.0226 | -0.867 | 4.35e-01 | 6.25e-01 |
| F1  | 0.3895 | 0.3981 | +0.0086 | -0.483 | 6.54e-01 | 8.12e-01 |

#### Case: crop80/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8112 | 0.7991 | -0.0121 | 1.817 | 1.43e-01 | 1.88e-01 |
| AUPRC  | 0.3849 | 0.3761 | -0.0088 | 0.286 | 7.89e-01 | 6.25e-01 |
| Brier * | 0.1544 | 0.2072 | +0.0528 | -3.195 | 3.31e-02 | 6.25e-02 |
| Accuracy † | 0.7652 | 0.6652 | -0.1000 | 2.271 | 8.56e-02 | 1.25e-01 |
| F1 † | 0.3993 | 0.3493 | -0.0500 | 2.507 | 6.63e-02 | 6.25e-02 |

#### Case: crop60/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8262 | 0.8152 | -0.0110 | 1.463 | 2.17e-01 | 3.12e-01 |
| AUPRC  | 0.4337 | 0.3900 | -0.0437 | 1.669 | 1.71e-01 | 1.88e-01 |
| Brier  | 0.1748 | 0.1897 | +0.0149 | -1.178 | 3.04e-01 | 4.38e-01 |
| Accuracy  | 0.7276 | 0.7093 | -0.0183 | 0.950 | 3.96e-01 | 4.38e-01 |
| F1  | 0.3842 | 0.3832 | -0.0010 | 0.098 | 9.26e-01 | 1.00e+00 |

#### Case: norm/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8178 | 0.8119 | -0.0059 | 0.670 | 5.40e-01 | 6.25e-01 |
| AUPRC  | 0.3368 | 0.3832 | +0.0465 | -1.614 | 1.82e-01 | 3.12e-01 |
| Brier  | 0.1724 | 0.1807 | +0.0083 | -0.246 | 8.18e-01 | 8.12e-01 |
| Accuracy  | 0.7040 | 0.7244 | +0.0204 | -0.269 | 8.01e-01 | 8.75e-01 |
| F1  | 0.3503 | 0.3699 | +0.0196 | -0.601 | 5.80e-01 | 8.12e-01 |

#### Case: excl_extreme/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8259 | 0.8165 | -0.0094 | 1.225 | 2.88e-01 | 3.12e-01 |
| AUPRC  | 0.3908 | 0.3967 | +0.0058 | -0.208 | 8.45e-01 | 8.12e-01 |
| Brier  | 0.1855 | 0.1986 | +0.0131 | -0.556 | 6.08e-01 | 1.00e+00 |
| Accuracy  | 0.6982 | 0.7078 | +0.0096 | -0.161 | 8.80e-01 | 8.12e-01 |
| F1  | 0.3568 | 0.3606 | +0.0039 | -0.098 | 9.27e-01 | 8.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len256/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8047 | -0.0004 | 0.020 | 9.85e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3782 | -0.0075 | 0.193 | 8.57e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1801 | +0.0001 | -0.032 | 9.76e-01 | 1.00e+00 |
| Accuracy  | 0.7395 | 0.7427 | +0.0032 | -0.390 | 7.17e-01 | 1.00e+00 |
| F1  | 0.3714 | 0.3981 | +0.0267 | -1.967 | 1.21e-01 | 1.25e-01 |

### crop80/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.7991 | -0.0059 | 0.308 | 7.74e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3761 | -0.0096 | 0.204 | 8.48e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.2072 | +0.0272 | -1.822 | 1.42e-01 | 1.25e-01 |
| Accuracy  | 0.7395 | 0.6652 | -0.0742 | 1.679 | 1.68e-01 | 1.25e-01 |
| F1  | 0.3714 | 0.3493 | -0.0221 | 0.788 | 4.75e-01 | 8.12e-01 |

### crop60/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8152 | +0.0102 | -0.492 | 6.48e-01 | 7.50e-01 |
| AUPRC  | 0.3857 | 0.3900 | +0.0043 | -0.095 | 9.29e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1897 | +0.0097 | -0.955 | 3.93e-01 | 4.38e-01 |
| Accuracy  | 0.7395 | 0.7093 | -0.0302 | 1.986 | 1.18e-01 | 1.25e-01 |
| F1  | 0.3714 | 0.3832 | +0.0118 | -0.678 | 5.35e-01 | 6.25e-01 |

### norm/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8119 | +0.0069 | -0.305 | 7.76e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3832 | -0.0025 | 0.068 | 9.49e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1807 | +0.0007 | -0.039 | 9.71e-01 | 6.25e-01 |
| Accuracy  | 0.7395 | 0.7244 | -0.0150 | 0.440 | 6.83e-01 | 6.25e-01 |
| F1  | 0.3714 | 0.3699 | -0.0015 | 0.115 | 9.14e-01 | 1.00e+00 |

### excl_extreme/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8165 | +0.0114 | -0.306 | 7.75e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3967 | +0.0110 | -0.182 | 8.64e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1986 | +0.0186 | -0.974 | 3.85e-01 | 4.38e-01 |
| Accuracy  | 0.7395 | 0.7078 | -0.0317 | 0.790 | 4.74e-01 | 6.25e-01 |
| F1  | 0.3714 | 0.3606 | -0.0107 | 0.237 | 8.24e-01 | 8.12e-01 |

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
| M2 | CrossAttn | len256/scale_both | AUC-ROC | 0.7887 | 0.6810 | 0.8816 |
| M2 | CrossAttn | len256/scale_both | AUPRC | 0.4100 | 0.2402 | 0.6066 |
| M2 | CrossAttn | len256/scale_both | Brier | 0.2169 | 0.1846 | 0.2496 |
| M2 | CrossAttn | len256/scale_both | Accuracy | 0.6609 | 0.6009 | 0.7210 |
| M2 | CrossAttn | len256/scale_both | F1 | 0.3248 | 0.2174 | 0.4386 |
| M2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.8010 | 0.7155 | 0.8780 |
| M2 | CrossAttn | crop80/scale_both | AUPRC | 0.3900 | 0.2285 | 0.5759 |
| M2 | CrossAttn | crop80/scale_both | Brier | 0.2547 | 0.2204 | 0.2877 |
| M2 | CrossAttn | crop80/scale_both | Accuracy | 0.5837 | 0.5193 | 0.6438 |
| M2 | CrossAttn | crop80/scale_both | F1 | 0.3022 | 0.2059 | 0.4029 |
| M2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.8156 | 0.7227 | 0.8971 |
| M2 | CrossAttn | crop60/scale_both | AUPRC | 0.4163 | 0.2518 | 0.6094 |
| M2 | CrossAttn | crop60/scale_both | Brier | 0.2130 | 0.1778 | 0.2496 |
| M2 | CrossAttn | crop60/scale_both | Accuracy | 0.6824 | 0.6223 | 0.7425 |
| M2 | CrossAttn | crop60/scale_both | F1 | 0.3509 | 0.2435 | 0.4696 |
| M2 | CrossAttn | norm/scale_both | AUC-ROC | 0.7873 | 0.7056 | 0.8621 |
| M2 | CrossAttn | norm/scale_both | AUPRC | 0.3616 | 0.2012 | 0.5371 |
| M2 | CrossAttn | norm/scale_both | Brier | 0.2584 | 0.2268 | 0.2900 |
| M2 | CrossAttn | norm/scale_both | Accuracy | 0.5536 | 0.4893 | 0.6180 |
| M2 | CrossAttn | norm/scale_both | F1 | 0.3067 | 0.2154 | 0.4024 |
| M2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8189 | 0.7261 | 0.8966 |
| M2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.4083 | 0.2329 | 0.6053 |
| M2 | CrossAttn | excl_extreme/scale_both | Brier | 0.2040 | 0.1768 | 0.2327 |
| M2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.6316 | 0.5646 | 0.6938 |
| M2 | CrossAttn | excl_extreme/scale_both | F1 | 0.3304 | 0.2182 | 0.4409 |
| M2_2 | CrossAttn | len256/scale_both | AUC-ROC | 0.8025 | 0.7214 | 0.8754 |
| M2_2 | CrossAttn | len256/scale_both | AUPRC | 0.3403 | 0.1970 | 0.5308 |
| M2_2 | CrossAttn | len256/scale_both | Brier | 0.1796 | 0.1523 | 0.2069 |
| M2_2 | CrossAttn | len256/scale_both | Accuracy | 0.7039 | 0.6438 | 0.7639 |
| M2_2 | CrossAttn | len256/scale_both | F1 | 0.3551 | 0.2400 | 0.4706 |
| M2_2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.7881 | 0.6861 | 0.8768 |
| M2_2 | CrossAttn | crop80/scale_both | AUPRC | 0.2942 | 0.1837 | 0.4791 |
| M2_2 | CrossAttn | crop80/scale_both | Brier | 0.2082 | 0.1750 | 0.2424 |
| M2_2 | CrossAttn | crop80/scale_both | Accuracy | 0.6781 | 0.6180 | 0.7339 |
| M2_2 | CrossAttn | crop80/scale_both | F1 | 0.3590 | 0.2476 | 0.4688 |
| M2_2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.8265 | 0.7456 | 0.9001 |
| M2_2 | CrossAttn | crop60/scale_both | AUPRC | 0.3845 | 0.2265 | 0.5719 |
| M2_2 | CrossAttn | crop60/scale_both | Brier | 0.2078 | 0.1772 | 0.2377 |
| M2_2 | CrossAttn | crop60/scale_both | Accuracy | 0.6567 | 0.5923 | 0.7167 |
| M2_2 | CrossAttn | crop60/scale_both | F1 | 0.3443 | 0.2414 | 0.4533 |
| M2_2 | CrossAttn | norm/scale_both | AUC-ROC | 0.8038 | 0.7177 | 0.8826 |
| M2_2 | CrossAttn | norm/scale_both | AUPRC | 0.3559 | 0.2167 | 0.5644 |
| M2_2 | CrossAttn | norm/scale_both | Brier | 0.1815 | 0.1532 | 0.2098 |
| M2_2 | CrossAttn | norm/scale_both | Accuracy | 0.7339 | 0.6780 | 0.7897 |
| M2_2 | CrossAttn | norm/scale_both | F1 | 0.3922 | 0.2758 | 0.5082 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8175 | 0.7230 | 0.9030 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.4052 | 0.2372 | 0.5973 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1691 | 0.1378 | 0.2031 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.7321 | 0.6699 | 0.7895 |
| M2_2 | CrossAttn | excl_extreme/scale_both | F1 | 0.3778 | 0.2500 | 0.5047 |
| M3 | CrossAttn3 | len256/scale_both | AUC-ROC | 0.8019 | 0.7169 | 0.8808 |
| M3 | CrossAttn3 | len256/scale_both | AUPRC | 0.3396 | 0.2018 | 0.5273 |
| M3 | CrossAttn3 | len256/scale_both | Brier | 0.2312 | 0.1963 | 0.2690 |
| M3 | CrossAttn3 | len256/scale_both | Accuracy | 0.6695 | 0.6094 | 0.7296 |
| M3 | CrossAttn3 | len256/scale_both | F1 | 0.3419 | 0.2342 | 0.4580 |
| M3 | CrossAttn3 | crop80/scale_both | AUC-ROC | 0.7952 | 0.6989 | 0.8837 |
| M3 | CrossAttn3 | crop80/scale_both | AUPRC | 0.3873 | 0.2261 | 0.5927 |
| M3 | CrossAttn3 | crop80/scale_both | Brier | 0.1818 | 0.1541 | 0.2125 |
| M3 | CrossAttn3 | crop80/scale_both | Accuracy | 0.7210 | 0.6609 | 0.7768 |
| M3 | CrossAttn3 | crop80/scale_both | F1 | 0.3434 | 0.2195 | 0.4632 |
| M3 | CrossAttn3 | crop60/scale_both | AUC-ROC | 0.8098 | 0.7232 | 0.8891 |
| M3 | CrossAttn3 | crop60/scale_both | AUPRC | 0.4021 | 0.2297 | 0.6079 |
| M3 | CrossAttn3 | crop60/scale_both | Brier | 0.1245 | 0.1027 | 0.1465 |
| M3 | CrossAttn3 | crop60/scale_both | Accuracy | 0.7897 | 0.7382 | 0.8412 |
| M3 | CrossAttn3 | crop60/scale_both | F1 | 0.3467 | 0.2051 | 0.4789 |
| M3 | CrossAttn3 | norm/scale_both | AUC-ROC | 0.8352 | 0.7519 | 0.9120 |
| M3 | CrossAttn3 | norm/scale_both | AUPRC | 0.3975 | 0.2470 | 0.5899 |
| M3 | CrossAttn3 | norm/scale_both | Brier | 0.1739 | 0.1471 | 0.2020 |
| M3 | CrossAttn3 | norm/scale_both | Accuracy | 0.7468 | 0.6910 | 0.8026 |
| M3 | CrossAttn3 | norm/scale_both | F1 | 0.4040 | 0.2820 | 0.5310 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUC-ROC | 0.8116 | 0.7086 | 0.8942 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUPRC | 0.4055 | 0.2366 | 0.6103 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Brier | 0.1772 | 0.1469 | 0.2116 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Accuracy | 0.7273 | 0.6651 | 0.7847 |
| M3 | CrossAttn3 | excl_extreme/scale_both | F1 | 0.3736 | 0.2409 | 0.5000 |

