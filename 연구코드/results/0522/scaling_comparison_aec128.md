# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8119 | 0.4103 | 0.1878 | 0.7167 | 0.3889 |
| M2 | CrossAttn | excl_extreme/scale_clinic | 0.8272 | 0.3703 | 0.1898 | 0.6890 | 0.3434 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | 0.8233 | 0.3481 | 0.1866 | 0.6794 | 0.3619 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | 0.8296 | 0.4607 | 0.1904 | 0.6603 | 0.3364 |

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
| len128/scale_clinic | 0.8254 | 0.4031 | 0.1956 | 0.6781 | 0.3478 |
| crop80/scale_clinic | 0.8160 | 0.3724 | 0.2684 | 0.5751 | 0.3077 |
| crop60/scale_clinic | 0.8019 | 0.3744 | 0.2122 | 0.6395 | 0.3333 |
| norm/scale_clinic | 0.8181 | 0.4241 | 0.1261 | 0.8155 | 0.3944 |
| **excl_extreme/scale_clinic** | 0.8272 | 0.3703 | 0.1898 | 0.6890 | 0.3434 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_clinic | 0.7948 | 0.3162 | 0.2245 | 0.6223 | 0.3231 |
| crop80/scale_clinic | 0.7696 | 0.2855 | 0.1997 | 0.6824 | 0.3273 |
| crop60/scale_clinic | 0.8150 | 0.3904 | 0.1901 | 0.6867 | 0.3423 |
| norm/scale_clinic | 0.7956 | 0.3795 | 0.1494 | 0.7725 | 0.3908 |
| **excl_extreme/scale_clinic** | 0.8233 | 0.3481 | 0.1866 | 0.6794 | 0.3619 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_clinic | 0.8142 | 0.3342 | 0.2303 | 0.6738 | 0.3559 |
| crop80/scale_clinic | 0.8079 | 0.3629 | 0.1849 | 0.6953 | 0.3486 |
| crop60/scale_clinic | 0.8110 | 0.4042 | 0.2364 | 0.6223 | 0.3231 |
| norm/scale_clinic | 0.8250 | 0.4441 | 0.2295 | 0.6052 | 0.3235 |
| **excl_extreme/scale_clinic** | 0.8296 | 0.4607 | 0.1904 | 0.6603 | 0.3364 |

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
| AUC-ROC  | 0.8051 | 0.8192 | +0.0142 | -1.089 | 3.37e-01 | 4.38e-01 |
| AUPRC  | 0.3857 | 0.4023 | +0.0166 | -0.479 | 6.57e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1699 | -0.0101 | 0.965 | 3.89e-01 | 6.25e-01 |
| Accuracy  | 0.7395 | 0.7308 | -0.0086 | 0.470 | 6.63e-01 | 8.12e-01 |
| F1  | 0.3714 | 0.3854 | +0.0140 | -1.227 | 2.87e-01 | 3.12e-01 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8275 | +0.0224 | -1.898 | 1.31e-01 | 1.88e-01 |
| AUPRC  | 0.3857 | 0.4085 | +0.0228 | -0.928 | 4.06e-01 | 4.38e-01 |
| Brier  | 0.1800 | 0.1932 | +0.0132 | -0.615 | 5.72e-01 | 1.00e+00 |
| Accuracy  | 0.7395 | 0.6717 | -0.0678 | 1.048 | 3.54e-01 | 4.38e-01 |
| F1  | 0.3714 | 0.3670 | -0.0044 | 0.120 | 9.10e-01 | 8.12e-01 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8286 | +0.0235 | -1.521 | 2.03e-01 | 3.12e-01 |
| AUPRC  | 0.3857 | 0.4071 | +0.0214 | -0.518 | 6.32e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1912 | +0.0112 | -0.638 | 5.58e-01 | 8.12e-01 |
| Accuracy  | 0.7395 | 0.7190 | -0.0204 | 0.617 | 5.70e-01 | 8.12e-01 |
| F1  | 0.3714 | 0.3927 | +0.0214 | -0.788 | 4.75e-01 | 6.25e-01 |

### norm/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8174 | +0.0123 | -1.125 | 3.24e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3909 | +0.0052 | -0.181 | 8.65e-01 | 8.12e-01 |
| Brier * | 0.1800 | 0.1472 | -0.0327 | 3.602 | 2.27e-02 | 6.25e-02 |
| Accuracy * | 0.7395 | 0.7933 | +0.0538 | -3.284 | 3.04e-02 | 1.25e-01 |
| F1  | 0.3714 | 0.3800 | +0.0087 | -0.859 | 4.39e-01 | 4.38e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8278 | +0.0227 | -1.235 | 2.84e-01 | 1.88e-01 |
| AUPRC  | 0.3857 | 0.3842 | -0.0015 | 0.042 | 9.69e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1742 | -0.0058 | 0.191 | 8.58e-01 | 8.12e-01 |
| Accuracy  | 0.7395 | 0.7473 | +0.0078 | -0.172 | 8.72e-01 | 8.12e-01 |
| F1  | 0.3714 | 0.3756 | +0.0042 | -0.114 | 9.15e-01 | 6.25e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len128/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8192 | 0.7912 | -0.0281 | 1.613 | 1.82e-01 | 1.88e-01 |
| AUPRC  | 0.4023 | 0.3754 | -0.0269 | 1.486 | 2.12e-01 | 3.12e-01 |
| Brier  | 0.1699 | 0.2011 | +0.0313 | -1.591 | 1.87e-01 | 3.12e-01 |
| Accuracy  | 0.7308 | 0.6963 | -0.0346 | 0.959 | 3.92e-01 | 4.38e-01 |
| F1  | 0.3854 | 0.3476 | -0.0378 | 1.017 | 3.66e-01 | 4.38e-01 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8275 | 0.8036 | -0.0239 | 2.220 | 9.06e-02 | 6.25e-02 |
| AUPRC  | 0.4085 | 0.3655 | -0.0430 | 1.630 | 1.78e-01 | 3.12e-01 |
| Brier  | 0.1932 | 0.1747 | -0.0185 | 0.866 | 4.35e-01 | 6.25e-01 |
| Accuracy  | 0.6717 | 0.7631 | +0.0914 | -1.536 | 1.99e-01 | 6.25e-02 |
| F1  | 0.3670 | 0.3841 | +0.0171 | -1.832 | 1.41e-01 | 1.25e-01 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8286 | 0.8052 | -0.0234 | 2.519 | 6.54e-02 | 1.25e-01 |
| AUPRC  | 0.4071 | 0.3947 | -0.0124 | 0.297 | 7.81e-01 | 6.25e-01 |
| Brier  | 0.1912 | 0.1811 | -0.0101 | 0.493 | 6.48e-01 | 1.00e+00 |
| Accuracy  | 0.7190 | 0.7384 | +0.0193 | -0.615 | 5.72e-01 | 1.00e+00 |
| F1  | 0.3927 | 0.3900 | -0.0028 | 0.140 | 8.96e-01 | 1.00e+00 |

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8174 | 0.8120 | -0.0054 | 0.439 | 6.83e-01 | 8.12e-01 |
| AUPRC  | 0.3909 | 0.3938 | +0.0029 | -0.162 | 8.79e-01 | 1.00e+00 |
| Brier † | 0.1472 | 0.2107 | +0.0634 | -2.659 | 5.65e-02 | 1.25e-01 |
| Accuracy † | 0.7933 | 0.6760 | -0.1173 | 2.171 | 9.57e-02 | 1.25e-01 |
| F1  | 0.3800 | 0.3668 | -0.0132 | 0.432 | 6.88e-01 | 1.00e+00 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8278 | 0.8161 | -0.0117 | 1.137 | 3.19e-01 | 3.12e-01 |
| AUPRC  | 0.3842 | 0.4073 | +0.0231 | -1.200 | 2.96e-01 | 8.12e-01 |
| Brier  | 0.1742 | 0.1935 | +0.0193 | -0.654 | 5.49e-01 | 8.12e-01 |
| Accuracy  | 0.7473 | 0.7078 | -0.0395 | 1.084 | 3.39e-01 | 5.62e-01 |
| F1  | 0.3756 | 0.3628 | -0.0128 | 0.687 | 5.30e-01 | 8.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.7912 | -0.0139 | 0.551 | 6.11e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3754 | -0.0103 | 0.233 | 8.27e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.2011 | +0.0211 | -0.898 | 4.20e-01 | 4.38e-01 |
| Accuracy  | 0.7395 | 0.6963 | -0.0432 | 0.985 | 3.80e-01 | 6.25e-01 |
| F1  | 0.3714 | 0.3476 | -0.0238 | 0.652 | 5.50e-01 | 4.38e-01 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8036 | -0.0015 | 0.101 | 9.25e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3655 | -0.0202 | 0.489 | 6.51e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1747 | -0.0053 | 0.763 | 4.88e-01 | 8.12e-01 |
| Accuracy  | 0.7395 | 0.7631 | +0.0237 | -1.720 | 1.61e-01 | 2.50e-01 |
| F1  | 0.3714 | 0.3841 | +0.0127 | -0.452 | 6.75e-01 | 6.25e-01 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8052 | +0.0001 | -0.008 | 9.94e-01 | 1.00e+00 |
| AUPRC  | 0.3857 | 0.3947 | +0.0090 | -0.203 | 8.49e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1811 | +0.0011 | -0.122 | 9.09e-01 | 1.00e+00 |
| Accuracy  | 0.7395 | 0.7384 | -0.0011 | 0.062 | 9.54e-01 | 1.00e+00 |
| F1  | 0.3714 | 0.3900 | +0.0186 | -1.924 | 1.27e-01 | 1.25e-01 |

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8120 | +0.0069 | -0.326 | 7.61e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3938 | +0.0081 | -0.206 | 8.47e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.2107 | +0.0307 | -1.691 | 1.66e-01 | 1.88e-01 |
| Accuracy  | 0.7395 | 0.6760 | -0.0635 | 1.402 | 2.33e-01 | 3.75e-01 |
| F1  | 0.3714 | 0.3668 | -0.0046 | 0.143 | 8.93e-01 | 1.00e+00 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8161 | +0.0110 | -0.750 | 4.95e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.4073 | +0.0216 | -0.721 | 5.11e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1935 | +0.0135 | -0.666 | 5.42e-01 | 6.25e-01 |
| Accuracy  | 0.7395 | 0.7078 | -0.0317 | 0.787 | 4.75e-01 | 8.12e-01 |
| F1  | 0.3714 | 0.3628 | -0.0086 | 0.265 | 8.04e-01 | 1.00e+00 |

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
| M2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.8254 | 0.7419 | 0.9011 |
| M2 | CrossAttn | len128/scale_clinic | AUPRC | 0.4031 | 0.2434 | 0.6162 |
| M2 | CrossAttn | len128/scale_clinic | Brier | 0.1956 | 0.1629 | 0.2286 |
| M2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6781 | 0.6180 | 0.7382 |
| M2 | CrossAttn | len128/scale_clinic | F1 | 0.3478 | 0.2393 | 0.4672 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8160 | 0.7316 | 0.8929 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.3724 | 0.2236 | 0.5700 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.2684 | 0.2341 | 0.3038 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.5751 | 0.5064 | 0.6395 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.3077 | 0.2127 | 0.4079 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8019 | 0.7062 | 0.8875 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3744 | 0.2223 | 0.5734 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.2122 | 0.1837 | 0.2421 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6395 | 0.5794 | 0.6996 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.3333 | 0.2281 | 0.4384 |
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8181 | 0.7332 | 0.8959 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4241 | 0.2497 | 0.6134 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.1261 | 0.1018 | 0.1510 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.8155 | 0.7639 | 0.8627 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.3944 | 0.2400 | 0.5366 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8272 | 0.7407 | 0.9033 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3703 | 0.2203 | 0.6013 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1898 | 0.1575 | 0.2251 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.6890 | 0.6220 | 0.7465 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3434 | 0.2151 | 0.4646 |
| M2_2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.7948 | 0.7023 | 0.8775 |
| M2_2 | CrossAttn | len128/scale_clinic | AUPRC | 0.3162 | 0.1861 | 0.4953 |
| M2_2 | CrossAttn | len128/scale_clinic | Brier | 0.2245 | 0.1936 | 0.2567 |
| M2_2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6223 | 0.5579 | 0.6824 |
| M2_2 | CrossAttn | len128/scale_clinic | F1 | 0.3231 | 0.2240 | 0.4265 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.7696 | 0.6774 | 0.8547 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.2855 | 0.1666 | 0.4785 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.1997 | 0.1660 | 0.2333 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6824 | 0.6223 | 0.7425 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.3273 | 0.2174 | 0.4407 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8150 | 0.7289 | 0.8937 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3904 | 0.2255 | 0.5803 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.1901 | 0.1625 | 0.2181 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6867 | 0.6266 | 0.7468 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.3423 | 0.2280 | 0.4603 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.7956 | 0.7094 | 0.8772 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.3795 | 0.2145 | 0.5706 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.1494 | 0.1271 | 0.1718 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7725 | 0.7167 | 0.8240 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.3908 | 0.2571 | 0.5161 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8233 | 0.7327 | 0.9032 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3481 | 0.2091 | 0.5634 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1866 | 0.1586 | 0.2163 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.6794 | 0.6172 | 0.7416 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3619 | 0.2400 | 0.4808 |
| M3 | CrossAttn3 | len128/scale_clinic | AUC-ROC | 0.8142 | 0.7204 | 0.8981 |
| M3 | CrossAttn3 | len128/scale_clinic | AUPRC | 0.3342 | 0.2099 | 0.5361 |
| M3 | CrossAttn3 | len128/scale_clinic | Brier | 0.2303 | 0.1906 | 0.2709 |
| M3 | CrossAttn3 | len128/scale_clinic | Accuracy | 0.6738 | 0.6137 | 0.7339 |
| M3 | CrossAttn3 | len128/scale_clinic | F1 | 0.3559 | 0.2479 | 0.4715 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8079 | 0.7281 | 0.8803 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.3629 | 0.2152 | 0.5523 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.1849 | 0.1624 | 0.2073 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.6953 | 0.6352 | 0.7511 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.3486 | 0.2268 | 0.4640 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.8110 | 0.7265 | 0.8850 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.4042 | 0.2394 | 0.5894 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.2364 | 0.2027 | 0.2710 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.6223 | 0.5622 | 0.6824 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.3231 | 0.2187 | 0.4252 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8250 | 0.7430 | 0.9011 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.4441 | 0.2676 | 0.6383 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.2295 | 0.2003 | 0.2594 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.6052 | 0.5408 | 0.6652 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.3235 | 0.2241 | 0.4267 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8296 | 0.7309 | 0.9127 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.4607 | 0.2644 | 0.6641 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.1904 | 0.1637 | 0.2193 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.6603 | 0.5933 | 0.7225 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.3364 | 0.2200 | 0.4465 |

