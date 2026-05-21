# Scaling Comparison — Test Set Performance (AEC 128pt, BCEWithLogitsLoss)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |
| M2 | CrossAttn | norm/scale_clinic | 0.8868 | 0.5551 | 0.2124 | 0.7384 | 0.4444 |
| M2_2 | CrossAttn | norm/scale_clinic | 0.8420 | 0.5177 | 0.1964 | 0.6105 | 0.3619 |
| M3 | CrossAttn3 | norm/scale_clinic | 0.8944 | 0.5636 | 0.1952 | 0.7791 | 0.4571 |

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
| **norm/scale_clinic** | 0.8868 | 0.5551 | 0.2124 | 0.7384 | 0.4444 |
| excl_extreme/scale_clinic | 0.8339 | 0.3987 | 0.2687 | 0.6104 | 0.4118 |
| len128/scale_clinic | 0.8578 | 0.4132 | 0.2150 | 0.6395 | 0.3922 |
| crop80/scale_clinic | 0.8612 | 0.5462 | 0.2017 | 0.6860 | 0.4130 |
| crop60/scale_clinic | 0.8635 | 0.5408 | 0.1630 | 0.6686 | 0.4000 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm/scale_clinic** | 0.8420 | 0.5177 | 0.1964 | 0.6105 | 0.3619 |
| excl_extreme/scale_clinic | 0.7965 | 0.2934 | 0.2056 | 0.7727 | 0.4262 |
| len128/scale_clinic | 0.8161 | 0.3932 | 0.2176 | 0.6977 | 0.4222 |
| crop80/scale_clinic | 0.8354 | 0.4306 | 0.2297 | 0.6686 | 0.4000 |
| crop60/scale_clinic | 0.8221 | 0.3693 | 0.1956 | 0.6977 | 0.4222 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm/scale_clinic** | 0.8944 | 0.5636 | 0.1952 | 0.7791 | 0.4571 |
| excl_extreme/scale_clinic | 0.8485 | 0.4976 | 0.2141 | 0.6883 | 0.4286 |
| len128/scale_clinic | 0.8357 | 0.4362 | 0.2023 | 0.6744 | 0.3913 |
| crop80/scale_clinic | 0.8575 | 0.4595 | 0.2196 | 0.6977 | 0.4091 |
| crop60/scale_clinic | 0.8505 | 0.5395 | 0.1946 | 0.7035 | 0.4000 |

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
| AUC-ROC  | 0.8113 | 0.8206 | +0.0093 | -0.991 | 3.78e-01 | 4.38e-01 |
| AUPRC † | 0.3847 | 0.4258 | +0.0411 | -2.157 | 9.72e-02 | 1.25e-01 |
| Brier  | 0.1819 | 0.2097 | +0.0278 | -1.283 | 2.69e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.8005 | +0.0451 | -1.193 | 2.99e-01 | 4.38e-01 |
| F1  | 0.4514 | 0.4870 | +0.0355 | -1.073 | 3.44e-01 | 6.25e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8056 | -0.0058 | 0.359 | 7.38e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3686 | -0.0161 | 0.353 | 7.42e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.1999 | +0.0180 | -1.590 | 1.87e-01 | 1.88e-01 |
| Accuracy  | 0.7555 | 0.7372 | -0.0183 | 0.305 | 7.75e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4297 | -0.0217 | 0.700 | 5.23e-01 | 8.12e-01 |

### len128/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8142 | +0.0029 | -0.286 | 7.89e-01 | 1.00e+00 |
| AUPRC * | 0.3847 | 0.4042 | +0.0195 | -4.316 | 1.25e-02 | 6.25e-02 |
| Brier * | 0.1819 | 0.1611 | -0.0208 | 3.274 | 3.07e-02 | 6.25e-02 |
| Accuracy  | 0.7555 | 0.7307 | -0.0248 | 0.381 | 7.23e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4330 | -0.0184 | 0.388 | 7.18e-01 | 6.25e-01 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8106 | -0.0008 | 0.103 | 9.23e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3687 | -0.0159 | 1.127 | 3.23e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1956 | +0.0136 | -1.183 | 3.02e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7220 | -0.0335 | 1.864 | 1.36e-01 | 1.88e-01 |
| F1  | 0.4514 | 0.4207 | -0.0308 | 2.127 | 1.01e-01 | 1.88e-01 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8063 | -0.0051 | 0.456 | 6.72e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3801 | -0.0046 | 0.183 | 8.63e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.1706 | -0.0113 | 1.511 | 2.05e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7014 | -0.0541 | 1.205 | 2.94e-01 | 4.38e-01 |
| F1  | 0.4514 | 0.4148 | -0.0366 | 1.430 | 2.26e-01 | 4.38e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8206 | 0.8070 | -0.0137 | 1.793 | 1.47e-01 | 1.88e-01 |
| AUPRC  | 0.4258 | 0.4002 | -0.0256 | 1.149 | 3.14e-01 | 3.12e-01 |
| Brier  | 0.2097 | 0.1999 | -0.0098 | 1.349 | 2.49e-01 | 3.12e-01 |
| Accuracy † | 0.8005 | 0.7539 | -0.0466 | 2.134 | 9.98e-02 | 1.25e-01 |
| F1  | 0.4870 | 0.4395 | -0.0475 | 2.008 | 1.15e-01 | 1.88e-01 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8056 | 0.7922 | -0.0134 | 1.279 | 2.70e-01 | 3.75e-01 |
| AUPRC  | 0.3686 | 0.3694 | +0.0008 | -0.061 | 9.54e-01 | 1.00e+00 |
| Brier  | 0.1999 | 0.1769 | -0.0230 | 1.130 | 3.22e-01 | 3.12e-01 |
| Accuracy  | 0.7372 | 0.7538 | +0.0166 | -0.224 | 8.34e-01 | 8.12e-01 |
| F1  | 0.4297 | 0.4411 | +0.0114 | -0.255 | 8.11e-01 | 1.00e+00 |

#### Case: len128/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8142 | 0.8122 | -0.0020 | 0.319 | 7.66e-01 | 1.00e+00 |
| AUPRC  | 0.4042 | 0.3952 | -0.0090 | 0.279 | 7.94e-01 | 8.12e-01 |
| Brier * | 0.1611 | 0.2005 | +0.0395 | -3.194 | 3.31e-02 | 6.25e-02 |
| Accuracy  | 0.7307 | 0.7131 | -0.0176 | 0.285 | 7.89e-01 | 8.12e-01 |
| F1  | 0.4330 | 0.4325 | -0.0005 | 0.012 | 9.91e-01 | 1.00e+00 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8106 | 0.8018 | -0.0088 | 0.735 | 5.03e-01 | 8.12e-01 |
| AUPRC  | 0.3687 | 0.3855 | +0.0168 | -0.509 | 6.38e-01 | 8.12e-01 |
| Brier  | 0.1956 | 0.2125 | +0.0169 | -0.746 | 4.97e-01 | 6.25e-01 |
| Accuracy  | 0.7220 | 0.7380 | +0.0160 | -0.353 | 7.42e-01 | 1.00e+00 |
| F1  | 0.4207 | 0.4387 | +0.0181 | -0.500 | 6.43e-01 | 8.12e-01 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8063 | 0.8039 | -0.0024 | 0.246 | 8.18e-01 | 1.00e+00 |
| AUPRC  | 0.3801 | 0.3886 | +0.0085 | -0.192 | 8.57e-01 | 8.12e-01 |
| Brier  | 0.1706 | 0.2205 | +0.0499 | -2.071 | 1.07e-01 | 1.88e-01 |
| Accuracy  | 0.7014 | 0.7408 | +0.0394 | -1.198 | 2.97e-01 | 6.25e-01 |
| F1  | 0.4148 | 0.4423 | +0.0275 | -1.993 | 1.17e-01 | 1.25e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8070 | -0.0044 | 0.585 | 5.90e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.4002 | +0.0155 | -1.310 | 2.61e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1999 | +0.0180 | -0.834 | 4.51e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7539 | -0.0016 | 0.074 | 9.44e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4395 | -0.0119 | 1.224 | 2.88e-01 | 3.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.7922 | -0.0191 | 1.282 | 2.69e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3694 | -0.0153 | 0.267 | 8.03e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.1769 | -0.0050 | 0.444 | 6.80e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7538 | -0.0017 | 0.056 | 9.58e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4411 | -0.0103 | 0.302 | 7.78e-01 | 8.12e-01 |

### len128/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8122 | +0.0009 | -0.087 | 9.35e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3952 | +0.0105 | -0.332 | 7.56e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.2005 | +0.0186 | -1.430 | 2.26e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7131 | -0.0424 | 0.947 | 3.97e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4325 | -0.0190 | 0.892 | 4.23e-01 | 4.38e-01 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8018 | -0.0096 | 0.710 | 5.17e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.3855 | +0.0009 | -0.031 | 9.77e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.2125 | +0.0306 | -2.127 | 1.01e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7380 | -0.0175 | 0.338 | 7.52e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4387 | -0.0127 | 0.282 | 7.92e-01 | 8.12e-01 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8039 | -0.0075 | 0.376 | 7.26e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.3886 | +0.0039 | -0.108 | 9.19e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.2205 | +0.0386 | -1.938 | 1.25e-01 | 1.88e-01 |
| Accuracy  | 0.7555 | 0.7408 | -0.0147 | 0.465 | 6.66e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4423 | -0.0092 | 0.397 | 7.12e-01 | 6.25e-01 |

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
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8868 | 0.8101 | 0.9494 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.5551 | 0.3607 | 0.7635 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.2124 | 0.1773 | 0.2487 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7384 | 0.6744 | 0.8023 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.4444 | 0.3030 | 0.5783 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8339 | 0.7534 | 0.9018 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3987 | 0.2342 | 0.6115 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.2687 | 0.2250 | 0.3143 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.6104 | 0.5325 | 0.6883 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4118 | 0.2857 | 0.5253 |
| M2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.8578 | 0.7829 | 0.9229 |
| M2 | CrossAttn | len128/scale_clinic | AUPRC | 0.4132 | 0.2549 | 0.6437 |
| M2 | CrossAttn | len128/scale_clinic | Brier | 0.2150 | 0.1786 | 0.2528 |
| M2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6395 | 0.5640 | 0.7093 |
| M2 | CrossAttn | len128/scale_clinic | F1 | 0.3922 | 0.2666 | 0.5133 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8612 | 0.7760 | 0.9329 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.5462 | 0.3330 | 0.7336 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.2017 | 0.1656 | 0.2362 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6860 | 0.6163 | 0.7558 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.4130 | 0.2823 | 0.5377 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8635 | 0.7832 | 0.9319 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.5408 | 0.3356 | 0.7472 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.1630 | 0.1336 | 0.1919 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6686 | 0.5930 | 0.7385 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.4000 | 0.2696 | 0.5234 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8420 | 0.7499 | 0.9193 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.5177 | 0.3100 | 0.7088 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.1964 | 0.1593 | 0.2359 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.6105 | 0.5349 | 0.6802 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.3619 | 0.2400 | 0.4786 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.7965 | 0.6862 | 0.8881 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.2934 | 0.1669 | 0.5106 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.2056 | 0.1666 | 0.2459 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7727 | 0.7013 | 0.8377 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4262 | 0.2666 | 0.5672 |
| M2_2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.8161 | 0.7082 | 0.8985 |
| M2_2 | CrossAttn | len128/scale_clinic | AUPRC | 0.3932 | 0.2235 | 0.5983 |
| M2_2 | CrossAttn | len128/scale_clinic | Brier | 0.2176 | 0.1791 | 0.2593 |
| M2_2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6977 | 0.6279 | 0.7616 |
| M2_2 | CrossAttn | len128/scale_clinic | F1 | 0.4222 | 0.2857 | 0.5435 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8354 | 0.7450 | 0.9059 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.4306 | 0.2460 | 0.6263 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.2297 | 0.1881 | 0.2729 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6686 | 0.5930 | 0.7384 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.4000 | 0.2667 | 0.5193 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8221 | 0.7199 | 0.9048 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3693 | 0.2162 | 0.6021 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.1956 | 0.1622 | 0.2332 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6977 | 0.6221 | 0.7674 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.4222 | 0.2888 | 0.5455 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8944 | 0.8250 | 0.9502 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.5636 | 0.3664 | 0.7655 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.1952 | 0.1650 | 0.2269 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.7791 | 0.7151 | 0.8372 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.4571 | 0.3051 | 0.5953 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8485 | 0.7687 | 0.9190 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.4976 | 0.3065 | 0.7150 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.2141 | 0.1757 | 0.2532 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.6883 | 0.6104 | 0.7597 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.4286 | 0.2888 | 0.5567 |
| M3 | CrossAttn3 | len128/scale_clinic | AUC-ROC | 0.8357 | 0.7424 | 0.9130 |
| M3 | CrossAttn3 | len128/scale_clinic | AUPRC | 0.4362 | 0.2521 | 0.6396 |
| M3 | CrossAttn3 | len128/scale_clinic | Brier | 0.2023 | 0.1665 | 0.2396 |
| M3 | CrossAttn3 | len128/scale_clinic | Accuracy | 0.6744 | 0.6047 | 0.7442 |
| M3 | CrossAttn3 | len128/scale_clinic | F1 | 0.3913 | 0.2580 | 0.5149 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8575 | 0.7716 | 0.9273 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.4595 | 0.2808 | 0.6848 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.2196 | 0.1797 | 0.2612 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.6977 | 0.6279 | 0.7674 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.4091 | 0.2716 | 0.5334 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.8505 | 0.7606 | 0.9267 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.5395 | 0.3304 | 0.7436 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.1946 | 0.1600 | 0.2300 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.7035 | 0.6337 | 0.7733 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.4000 | 0.2580 | 0.5253 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-norm | 0.8325 | 0.8868 | +0.0542 | -1.808 | 7.058e-02 | † |
| M1-LR vs M2-len128 | 0.8325 | 0.8578 | +0.0252 | -0.831 | 4.061e-01 | ns |
| M1-LR vs M2-crop80 | 0.8325 | 0.8612 | +0.0287 | -1.007 | 3.139e-01 | ns |
| M1-LR vs M2-crop60 | 0.8325 | 0.8635 | +0.0309 | -0.808 | 4.190e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-norm | 0.8325 | 0.8944 | +0.0618 | -2.049 | 4.046e-02 | * |
| M1-LR vs M3-len128 | 0.8325 | 0.8357 | +0.0032 | -0.100 | 9.200e-01 | ns |
| M1-LR vs M3-crop80 | 0.8325 | 0.8575 | +0.0249 | -0.888 | 3.746e-01 | ns |
| M1-LR vs M3-crop60 | 0.8325 | 0.8505 | +0.0180 | -0.545 | 5.858e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M2_2-norm | 0.8868 | 0.8420 | -0.0448 | 1.485 | 1.377e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8339 | 0.5596 | -0.2743 | 4.214 | 2.510e-05 | *** |
| M2-len128 vs M2_2-len128 | 0.8578 | 0.8161 | -0.0416 | 1.308 | 1.909e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.8612 | 0.8354 | -0.0259 | 0.673 | 5.007e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.8635 | 0.8221 | -0.0413 | 1.115 | 2.647e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M3-norm | 0.8868 | 0.8944 | +0.0076 | -0.777 | 4.371e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.8339 | 0.8485 | +0.0147 | -0.685 | 4.936e-01 | ns |
| M2-len128 vs M3-len128 | 0.8578 | 0.8357 | -0.0221 | 1.139 | 2.548e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8612 | 0.8575 | -0.0038 | 0.263 | 7.927e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.8635 | 0.8505 | -0.0129 | 0.655 | 5.126e-01 | ns |

