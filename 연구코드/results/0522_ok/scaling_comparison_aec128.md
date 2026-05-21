# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8325 | 0.5008 | 0.1804 | 0.6919 | 0.3765 |
| M2 | CrossAttn | excl_extreme/scale_clinic | 0.8786 | 0.5451 | 0.1726 | 0.7013 | 0.4651 |
| M2_2 | CrossAttn | norm/scale_clinic | 0.8562 | 0.5560 | 0.1839 | 0.7267 | 0.4337 |
| M3 | CrossAttn3 | norm/scale_clinic | 0.8805 | 0.5415 | 0.1520 | 0.7674 | 0.4595 |

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
| len128/scale_clinic | 0.8660 | 0.5339 | 0.2383 | 0.5872 | 0.3604 |
| crop80/scale_clinic | 0.8660 | 0.5394 | 0.1626 | 0.7267 | 0.4337 |
| crop60/scale_clinic | 0.8310 | 0.4444 | 0.2049 | 0.7093 | 0.4318 |
| norm/scale_clinic | 0.8663 | 0.5017 | 0.2554 | 0.5872 | 0.3717 |
| **excl_extreme/scale_clinic** | 0.8786 | 0.5451 | 0.1726 | 0.7013 | 0.4651 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_clinic | 0.8193 | 0.3624 | 0.2210 | 0.6570 | 0.3918 |
| crop80/scale_clinic | 0.7786 | 0.3292 | 0.1986 | 0.7093 | 0.4048 |
| crop60/scale_clinic | 0.8146 | 0.3357 | 0.2034 | 0.6744 | 0.4043 |
| **norm/scale_clinic** | 0.8562 | 0.5560 | 0.1839 | 0.7267 | 0.4337 |
| excl_extreme/scale_clinic | 0.8192 | 0.3180 | 0.2003 | 0.7143 | 0.4054 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_clinic | 0.8073 | 0.4252 | 0.1779 | 0.7035 | 0.4000 |
| crop80/scale_clinic | 0.8436 | 0.5163 | 0.2238 | 0.6337 | 0.3883 |
| crop60/scale_clinic | 0.8641 | 0.5292 | 0.1968 | 0.6802 | 0.4086 |
| **norm/scale_clinic** | 0.8805 | 0.5415 | 0.1520 | 0.7674 | 0.4595 |
| excl_extreme/scale_clinic | 0.8611 | 0.5373 | 0.1357 | 0.7662 | 0.4706 |

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

### len128/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8208 | +0.0095 | -0.576 | 5.95e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3812 | -0.0035 | 0.163 | 8.78e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.1874 | +0.0055 | -0.328 | 7.59e-01 | 8.12e-01 |
| Accuracy  | 0.7292 | 0.7221 | -0.0071 | 0.192 | 8.57e-01 | 8.75e-01 |
| F1  | 0.3950 | 0.3976 | +0.0026 | -0.073 | 9.45e-01 | 1.00e+00 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8113 | -0.0000 | 0.004 | 9.97e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3810 | -0.0037 | 0.173 | 8.71e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.1932 | +0.0113 | -0.503 | 6.42e-01 | 8.12e-01 |
| Accuracy  | 0.7292 | 0.6974 | -0.0319 | 0.631 | 5.62e-01 | 6.25e-01 |
| F1  | 0.3950 | 0.3769 | -0.0181 | 1.116 | 3.27e-01 | 3.12e-01 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8115 | +0.0002 | -0.016 | 9.88e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3621 | -0.0226 | 1.026 | 3.63e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1835 | +0.0016 | -0.190 | 8.59e-01 | 6.25e-01 |
| Accuracy  | 0.7292 | 0.7204 | -0.0088 | 0.370 | 7.30e-01 | 1.00e+00 |
| F1  | 0.3950 | 0.4141 | +0.0191 | -0.989 | 3.79e-01 | 4.38e-01 |

### norm/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.8113 | 0.8316 | +0.0203 | -4.748 | 8.98e-03 | 6.25e-02 |
| AUPRC  | 0.3847 | 0.4170 | +0.0323 | -1.100 | 3.33e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1710 | -0.0109 | 1.121 | 3.25e-01 | 4.38e-01 |
| Accuracy  | 0.7292 | 0.7480 | +0.0188 | -0.793 | 4.72e-01 | 6.25e-01 |
| F1  | 0.3950 | 0.4286 | +0.0336 | -1.856 | 1.37e-01 | 1.88e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8075 | -0.0039 | 0.308 | 7.73e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3686 | -0.0161 | 0.705 | 5.19e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.2015 | +0.0196 | -0.850 | 4.43e-01 | 6.25e-01 |
| Accuracy  | 0.7292 | 0.6903 | -0.0389 | 0.917 | 4.11e-01 | 4.38e-01 |
| F1 * | 0.3950 | 0.3510 | -0.0440 | 3.442 | 2.63e-02 | 6.25e-02 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len128/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8208 | 0.8195 | -0.0013 | 0.127 | 9.05e-01 | 1.00e+00 |
| AUPRC † | 0.3812 | 0.4353 | +0.0541 | -2.335 | 7.98e-02 | 6.25e-02 |
| Brier  | 0.1874 | 0.1639 | -0.0235 | 1.052 | 3.52e-01 | 8.12e-01 |
| Accuracy  | 0.7221 | 0.7350 | +0.0129 | -0.281 | 7.93e-01 | 1.00e+00 |
| F1  | 0.3976 | 0.4035 | +0.0059 | -0.336 | 7.53e-01 | 6.25e-01 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8113 | 0.8004 | -0.0109 | 2.859 | 4.60e-02 | 6.25e-02 |
| AUPRC  | 0.3810 | 0.3838 | +0.0029 | -0.111 | 9.17e-01 | 8.12e-01 |
| Brier  | 0.1932 | 0.2076 | +0.0143 | -0.380 | 7.23e-01 | 6.25e-01 |
| Accuracy  | 0.6974 | 0.6810 | -0.0163 | 0.182 | 8.65e-01 | 8.12e-01 |
| F1  | 0.3769 | 0.3854 | +0.0086 | -0.223 | 8.35e-01 | 1.00e+00 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8115 | 0.8071 | -0.0044 | 0.899 | 4.19e-01 | 6.25e-01 |
| AUPRC  | 0.3621 | 0.3525 | -0.0096 | 0.385 | 7.20e-01 | 6.25e-01 |
| Brier  | 0.1835 | 0.2022 | +0.0188 | -1.669 | 1.71e-01 | 1.25e-01 |
| Accuracy  | 0.7204 | 0.6959 | -0.0246 | 0.615 | 5.72e-01 | 6.25e-01 |
| F1  | 0.4141 | 0.3704 | -0.0437 | 1.733 | 1.58e-01 | 1.88e-01 |

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.8316 | 0.8095 | -0.0222 | 5.566 | 5.10e-03 | 6.25e-02 |
| AUPRC  | 0.4170 | 0.4256 | +0.0086 | -0.215 | 8.40e-01 | 8.12e-01 |
| Brier * | 0.1710 | 0.2030 | +0.0320 | -4.579 | 1.02e-02 | 6.25e-02 |
| Accuracy * | 0.7480 | 0.6737 | -0.0743 | 3.290 | 3.02e-02 | 6.25e-02 |
| F1 * | 0.4286 | 0.3713 | -0.0573 | 3.064 | 3.75e-02 | 6.25e-02 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8075 | 0.8028 | -0.0047 | 0.412 | 7.01e-01 | 1.00e+00 |
| AUPRC  | 0.3686 | 0.4186 | +0.0500 | -1.945 | 1.24e-01 | 1.25e-01 |
| Brier  | 0.2015 | 0.1867 | -0.0148 | 1.341 | 2.51e-01 | 3.12e-01 |
| Accuracy  | 0.6903 | 0.7066 | +0.0163 | -0.652 | 5.50e-01 | 6.25e-01 |
| F1  | 0.3510 | 0.3622 | +0.0112 | -0.380 | 7.23e-01 | 1.00e+00 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8195 | +0.0082 | -0.499 | 6.44e-01 | 1.00e+00 |
| AUPRC * | 0.3847 | 0.4353 | +0.0506 | -3.373 | 2.80e-02 | 1.25e-01 |
| Brier † | 0.1819 | 0.1639 | -0.0180 | 2.713 | 5.34e-02 | 6.25e-02 |
| Accuracy  | 0.7292 | 0.7350 | +0.0058 | -0.433 | 6.88e-01 | 7.50e-01 |
| F1  | 0.3950 | 0.4035 | +0.0085 | -0.334 | 7.55e-01 | 8.12e-01 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8004 | -0.0109 | 1.141 | 3.18e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3838 | -0.0009 | 0.049 | 9.63e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.2076 | +0.0256 | -1.321 | 2.57e-01 | 3.12e-01 |
| Accuracy  | 0.7292 | 0.6810 | -0.0482 | 1.180 | 3.04e-01 | 4.38e-01 |
| F1  | 0.3950 | 0.3854 | -0.0095 | 0.377 | 7.25e-01 | 6.25e-01 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8071 | -0.0042 | 0.330 | 7.58e-01 | 8.12e-01 |
| AUPRC * | 0.3847 | 0.3525 | -0.0321 | 3.310 | 2.97e-02 | 6.25e-02 |
| Brier † | 0.1819 | 0.2022 | +0.0203 | -2.287 | 8.41e-02 | 6.25e-02 |
| Accuracy  | 0.7292 | 0.6959 | -0.0333 | 1.096 | 3.34e-01 | 4.38e-01 |
| F1  | 0.3950 | 0.3704 | -0.0246 | 1.172 | 3.06e-01 | 6.25e-01 |

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8095 | -0.0019 | 0.241 | 8.22e-01 | 1.00e+00 |
| AUPRC * | 0.3847 | 0.4256 | +0.0409 | -2.783 | 4.97e-02 | 6.25e-02 |
| Brier  | 0.1819 | 0.2030 | +0.0211 | -1.605 | 1.84e-01 | 1.88e-01 |
| Accuracy  | 0.7292 | 0.6737 | -0.0555 | 2.015 | 1.14e-01 | 1.88e-01 |
| F1  | 0.3950 | 0.3713 | -0.0237 | 1.211 | 2.93e-01 | 3.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8028 | -0.0085 | 0.383 | 7.22e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.4186 | +0.0339 | -0.723 | 5.10e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1867 | +0.0048 | -0.220 | 8.37e-01 | 8.12e-01 |
| Accuracy  | 0.7292 | 0.7066 | -0.0226 | 0.448 | 6.77e-01 | 8.12e-01 |
| F1  | 0.3950 | 0.3622 | -0.0328 | 1.131 | 3.21e-01 | 3.12e-01 |

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
| M2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.8660 | 0.7930 | 0.9299 |
| M2 | CrossAttn | len128/scale_clinic | AUPRC | 0.5339 | 0.3252 | 0.7372 |
| M2 | CrossAttn | len128/scale_clinic | Brier | 0.2383 | 0.2010 | 0.2751 |
| M2 | CrossAttn | len128/scale_clinic | Accuracy | 0.5872 | 0.5116 | 0.6628 |
| M2 | CrossAttn | len128/scale_clinic | F1 | 0.3604 | 0.2424 | 0.4771 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8660 | 0.7783 | 0.9359 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.5394 | 0.3413 | 0.7402 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.1626 | 0.1345 | 0.1909 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.7267 | 0.6570 | 0.7907 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.4337 | 0.2891 | 0.5600 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8310 | 0.7336 | 0.9100 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.4444 | 0.2587 | 0.6490 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.2049 | 0.1636 | 0.2464 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.7093 | 0.6395 | 0.7791 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.4318 | 0.2927 | 0.5600 |
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8663 | 0.7894 | 0.9268 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.5017 | 0.3023 | 0.7186 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.2554 | 0.2185 | 0.2917 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.5872 | 0.5116 | 0.6628 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.3717 | 0.2499 | 0.4828 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8786 | 0.8106 | 0.9384 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.5451 | 0.3402 | 0.7377 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1726 | 0.1386 | 0.2086 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7013 | 0.6299 | 0.7727 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4651 | 0.3283 | 0.5870 |
| M2_2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.8193 | 0.7110 | 0.9057 |
| M2_2 | CrossAttn | len128/scale_clinic | AUPRC | 0.3624 | 0.2074 | 0.5624 |
| M2_2 | CrossAttn | len128/scale_clinic | Brier | 0.2210 | 0.1811 | 0.2609 |
| M2_2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6570 | 0.5872 | 0.7267 |
| M2_2 | CrossAttn | len128/scale_clinic | F1 | 0.3918 | 0.2680 | 0.5094 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.7786 | 0.6641 | 0.8733 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.3292 | 0.1814 | 0.5228 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.1986 | 0.1586 | 0.2399 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.7093 | 0.6395 | 0.7733 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.4048 | 0.2632 | 0.5333 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8146 | 0.7056 | 0.8997 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3357 | 0.2047 | 0.5471 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.2034 | 0.1627 | 0.2466 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6744 | 0.6047 | 0.7384 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.4043 | 0.2758 | 0.5253 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8562 | 0.7639 | 0.9314 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.5560 | 0.3377 | 0.7402 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.1839 | 0.1492 | 0.2206 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7267 | 0.6570 | 0.7907 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.4337 | 0.2927 | 0.5610 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8192 | 0.7067 | 0.9107 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3180 | 0.1880 | 0.5419 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.2003 | 0.1634 | 0.2382 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7143 | 0.6429 | 0.7857 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4054 | 0.2571 | 0.5317 |
| M3 | CrossAttn3 | len128/scale_clinic | AUC-ROC | 0.8073 | 0.7116 | 0.8902 |
| M3 | CrossAttn3 | len128/scale_clinic | AUPRC | 0.4252 | 0.2385 | 0.6299 |
| M3 | CrossAttn3 | len128/scale_clinic | Brier | 0.1779 | 0.1439 | 0.2119 |
| M3 | CrossAttn3 | len128/scale_clinic | Accuracy | 0.7035 | 0.6337 | 0.7733 |
| M3 | CrossAttn3 | len128/scale_clinic | F1 | 0.4000 | 0.2571 | 0.5263 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8436 | 0.7604 | 0.9180 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.5163 | 0.3163 | 0.7213 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.2238 | 0.1847 | 0.2634 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.6337 | 0.5581 | 0.7035 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.3883 | 0.2626 | 0.5051 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.8641 | 0.7827 | 0.9342 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.5292 | 0.3275 | 0.7502 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.1968 | 0.1611 | 0.2313 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.6802 | 0.6105 | 0.7500 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.4086 | 0.2785 | 0.5361 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8805 | 0.8089 | 0.9370 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.5415 | 0.3284 | 0.7394 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.1520 | 0.1251 | 0.1810 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.7674 | 0.7035 | 0.8314 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.4595 | 0.3077 | 0.5883 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8611 | 0.7790 | 0.9273 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.5373 | 0.3398 | 0.7259 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.1357 | 0.1083 | 0.1654 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.7662 | 0.6948 | 0.8312 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.4706 | 0.3077 | 0.6134 |

