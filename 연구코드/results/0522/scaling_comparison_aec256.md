# Scaling Comparison — Test Set Performance (AEC 256pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8119 | 0.4103 | 0.1878 | 0.7167 | 0.3889 |
| M2 | CrossAttn | excl_extreme/scale_clinic | 0.8376 | 0.4609 | 0.1635 | 0.7560 | 0.3855 |
| M2_2 | CrossAttn | norm/scale_clinic | 0.8227 | 0.3991 | 0.2416 | 0.5665 | 0.3034 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | 0.8391 | 0.4466 | 0.1907 | 0.6794 | 0.3366 |

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
| len256/scale_clinic | 0.8117 | 0.3536 | 0.2190 | 0.6309 | 0.3175 |
| crop80/scale_clinic | 0.8071 | 0.4123 | 0.1856 | 0.6910 | 0.3571 |
| crop60/scale_clinic | 0.7758 | 0.3937 | 0.1955 | 0.6910 | 0.3455 |
| norm/scale_clinic | 0.8221 | 0.4439 | 0.1542 | 0.7682 | 0.3864 |
| **excl_extreme/scale_clinic** | 0.8376 | 0.4609 | 0.1635 | 0.7560 | 0.3855 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_clinic | 0.8081 | 0.3326 | 0.1944 | 0.7124 | 0.3964 |
| crop80/scale_clinic | 0.7996 | 0.3977 | 0.3130 | 0.5622 | 0.3014 |
| crop60/scale_clinic | 0.8027 | 0.3769 | 0.2058 | 0.6652 | 0.3390 |
| **norm/scale_clinic** | 0.8227 | 0.3991 | 0.2416 | 0.5665 | 0.3034 |
| excl_extreme/scale_clinic | 0.8196 | 0.3534 | 0.1961 | 0.6603 | 0.3364 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_clinic | 0.8048 | 0.3785 | 0.2339 | 0.6352 | 0.3200 |
| crop80/scale_clinic | 0.8175 | 0.3956 | 0.1807 | 0.7296 | 0.3883 |
| crop60/scale_clinic | 0.8113 | 0.4211 | 0.1789 | 0.7339 | 0.3800 |
| norm/scale_clinic | 0.8296 | 0.4521 | 0.2089 | 0.6781 | 0.3478 |
| **excl_extreme/scale_clinic** | 0.8391 | 0.4466 | 0.1907 | 0.6794 | 0.3366 |

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
| AUC-ROC  | 0.8051 | 0.8322 | +0.0271 | -2.032 | 1.12e-01 | 1.25e-01 |
| AUPRC  | 0.3857 | 0.3767 | -0.0090 | 0.252 | 8.13e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1896 | +0.0096 | -0.419 | 6.97e-01 | 8.12e-01 |
| Accuracy  | 0.7395 | 0.7255 | -0.0140 | 0.527 | 6.26e-01 | 8.12e-01 |
| F1  | 0.3714 | 0.4019 | +0.0305 | -1.794 | 1.47e-01 | 1.88e-01 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8221 | +0.0171 | -0.961 | 3.91e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3930 | +0.0073 | -0.224 | 8.34e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1725 | -0.0075 | 0.353 | 7.42e-01 | 6.25e-01 |
| Accuracy  | 0.7395 | 0.7277 | -0.0118 | 0.292 | 7.85e-01 | 1.00e+00 |
| F1  | 0.3714 | 0.3699 | -0.0014 | 0.053 | 9.60e-01 | 8.12e-01 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8051 | 0.8291 | +0.0240 | -2.261 | 8.66e-02 | 1.25e-01 |
| AUPRC  | 0.3857 | 0.4109 | +0.0252 | -0.490 | 6.50e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.2074 | +0.0274 | -1.339 | 2.52e-01 | 4.38e-01 |
| Accuracy  | 0.7395 | 0.6480 | -0.0915 | 1.646 | 1.75e-01 | 1.25e-01 |
| F1  | 0.3714 | 0.3567 | -0.0147 | 0.434 | 6.86e-01 | 1.00e+00 |

### norm/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8140 | +0.0090 | -0.730 | 5.06e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3817 | -0.0039 | 0.132 | 9.02e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1726 | -0.0074 | 0.664 | 5.43e-01 | 1.00e+00 |
| Accuracy  | 0.7395 | 0.7417 | +0.0022 | -0.095 | 9.29e-01 | 1.00e+00 |
| F1  | 0.3714 | 0.3771 | +0.0058 | -0.445 | 6.79e-01 | 8.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8177 | +0.0127 | -0.362 | 7.36e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3675 | -0.0182 | 0.316 | 7.68e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.2086 | +0.0286 | -0.942 | 3.99e-01 | 6.25e-01 |
| Accuracy  | 0.7395 | 0.6587 | -0.0808 | 1.438 | 2.24e-01 | 3.12e-01 |
| F1  | 0.3714 | 0.3340 | -0.0374 | 0.921 | 4.09e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len256/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8322 | 0.7957 | -0.0365 | 2.536 | 6.42e-02 | 6.25e-02 |
| AUPRC  | 0.3767 | 0.3608 | -0.0159 | 0.644 | 5.55e-01 | 6.25e-01 |
| Brier  | 0.1896 | 0.1757 | -0.0139 | 0.823 | 4.57e-01 | 6.25e-01 |
| Accuracy  | 0.7255 | 0.7416 | +0.0161 | -0.789 | 4.74e-01 | 6.25e-01 |
| F1  | 0.4019 | 0.3683 | -0.0336 | 1.031 | 3.61e-01 | 4.38e-01 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8221 | 0.8122 | -0.0100 | 2.426 | 7.23e-02 | 6.25e-02 |
| AUPRC  | 0.3930 | 0.3934 | +0.0004 | -0.025 | 9.81e-01 | 1.00e+00 |
| Brier  | 0.1725 | 0.1899 | +0.0174 | -1.033 | 3.60e-01 | 4.38e-01 |
| Accuracy  | 0.7277 | 0.7212 | -0.0064 | 0.406 | 7.06e-01 | 1.00e+00 |
| F1  | 0.3699 | 0.3751 | +0.0051 | -0.378 | 7.25e-01 | 8.12e-01 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8291 | 0.8137 | -0.0153 | 2.111 | 1.02e-01 | 1.25e-01 |
| AUPRC  | 0.4109 | 0.3894 | -0.0216 | 0.674 | 5.37e-01 | 1.00e+00 |
| Brier  | 0.2074 | 0.1770 | -0.0305 | 1.342 | 2.51e-01 | 4.38e-01 |
| Accuracy  | 0.6480 | 0.7351 | +0.0872 | -1.471 | 2.15e-01 | 1.25e-01 |
| F1  | 0.3567 | 0.3959 | +0.0392 | -1.029 | 3.61e-01 | 3.12e-01 |

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.8140 | 0.7998 | -0.0142 | 4.750 | 8.97e-03 | 6.25e-02 |
| AUPRC  | 0.3817 | 0.3570 | -0.0247 | 1.496 | 2.09e-01 | 3.12e-01 |
| Brier  | 0.1726 | 0.1785 | +0.0059 | -0.278 | 7.95e-01 | 8.12e-01 |
| Accuracy  | 0.7417 | 0.7254 | -0.0163 | 0.452 | 6.75e-01 | 8.12e-01 |
| F1  | 0.3771 | 0.3575 | -0.0196 | 0.779 | 4.80e-01 | 6.25e-01 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8177 | 0.8143 | -0.0035 | 0.645 | 5.54e-01 | 8.12e-01 |
| AUPRC  | 0.3675 | 0.3604 | -0.0071 | 0.221 | 8.36e-01 | 8.12e-01 |
| Brier  | 0.2086 | 0.1944 | -0.0142 | 0.545 | 6.15e-01 | 1.00e+00 |
| Accuracy  | 0.6587 | 0.7162 | +0.0575 | -1.092 | 3.36e-01 | 3.12e-01 |
| F1  | 0.3340 | 0.3731 | +0.0391 | -1.475 | 2.14e-01 | 4.38e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len256/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.7957 | -0.0093 | 0.357 | 7.39e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3608 | -0.0249 | 0.530 | 6.24e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1757 | -0.0043 | 0.283 | 7.92e-01 | 1.00e+00 |
| Accuracy  | 0.7395 | 0.7416 | +0.0021 | -0.109 | 9.18e-01 | 1.00e+00 |
| F1  | 0.3714 | 0.3683 | -0.0031 | 0.122 | 9.09e-01 | 6.25e-01 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8122 | +0.0071 | -0.374 | 7.27e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3934 | +0.0077 | -0.198 | 8.53e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1899 | +0.0099 | -0.740 | 5.01e-01 | 4.38e-01 |
| Accuracy  | 0.7395 | 0.7212 | -0.0183 | 0.502 | 6.42e-01 | 1.00e+00 |
| F1  | 0.3714 | 0.3751 | +0.0037 | -0.120 | 9.10e-01 | 6.25e-01 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8137 | +0.0087 | -0.601 | 5.80e-01 | 6.25e-01 |
| AUPRC  | 0.3857 | 0.3894 | +0.0037 | -0.083 | 9.38e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1770 | -0.0030 | 0.453 | 6.74e-01 | 8.12e-01 |
| Accuracy  | 0.7395 | 0.7351 | -0.0043 | 0.272 | 7.99e-01 | 8.12e-01 |
| F1 * | 0.3714 | 0.3959 | +0.0246 | -3.350 | 2.86e-02 | 1.25e-01 |

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.7998 | -0.0053 | 0.368 | 7.32e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3570 | -0.0286 | 1.001 | 3.74e-01 | 4.38e-01 |
| Brier  | 0.1800 | 0.1785 | -0.0015 | 0.093 | 9.31e-01 | 1.00e+00 |
| Accuracy  | 0.7395 | 0.7254 | -0.0140 | 0.466 | 6.65e-01 | 1.00e+00 |
| F1  | 0.3714 | 0.3575 | -0.0139 | 0.510 | 6.37e-01 | 8.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8051 | 0.8143 | +0.0092 | -0.245 | 8.18e-01 | 8.12e-01 |
| AUPRC  | 0.3857 | 0.3604 | -0.0253 | 0.532 | 6.23e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.1944 | +0.0144 | -0.736 | 5.03e-01 | 8.12e-01 |
| Accuracy  | 0.7395 | 0.7162 | -0.0233 | 0.626 | 5.65e-01 | 8.12e-01 |
| F1  | 0.3714 | 0.3731 | +0.0018 | -0.041 | 9.70e-01 | 6.25e-01 |

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
| M2 | CrossAttn | len256/scale_clinic | AUC-ROC | 0.8117 | 0.7285 | 0.8868 |
| M2 | CrossAttn | len256/scale_clinic | AUPRC | 0.3536 | 0.2081 | 0.5483 |
| M2 | CrossAttn | len256/scale_clinic | Brier | 0.2190 | 0.1867 | 0.2527 |
| M2 | CrossAttn | len256/scale_clinic | Accuracy | 0.6309 | 0.5665 | 0.6910 |
| M2 | CrossAttn | len256/scale_clinic | F1 | 0.3175 | 0.2170 | 0.4275 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8071 | 0.7236 | 0.8852 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.4123 | 0.2431 | 0.6118 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.1856 | 0.1587 | 0.2155 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6910 | 0.6309 | 0.7554 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.3571 | 0.2435 | 0.4746 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.7758 | 0.6621 | 0.8746 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3937 | 0.2318 | 0.5942 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.1955 | 0.1651 | 0.2278 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6910 | 0.6352 | 0.7511 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.3455 | 0.2353 | 0.4615 |
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8221 | 0.7363 | 0.9003 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4439 | 0.2681 | 0.6347 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.1542 | 0.1306 | 0.1782 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7682 | 0.7124 | 0.8240 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.3864 | 0.2500 | 0.5117 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8376 | 0.7521 | 0.9114 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.4609 | 0.2661 | 0.6553 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1635 | 0.1376 | 0.1916 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7560 | 0.6984 | 0.8134 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3855 | 0.2424 | 0.5117 |
| M2_2 | CrossAttn | len256/scale_clinic | AUC-ROC | 0.8081 | 0.7099 | 0.8949 |
| M2_2 | CrossAttn | len256/scale_clinic | AUPRC | 0.3326 | 0.2048 | 0.5365 |
| M2_2 | CrossAttn | len256/scale_clinic | Brier | 0.1944 | 0.1632 | 0.2253 |
| M2_2 | CrossAttn | len256/scale_clinic | Accuracy | 0.7124 | 0.6567 | 0.7725 |
| M2_2 | CrossAttn | len256/scale_clinic | F1 | 0.3964 | 0.2857 | 0.5156 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.7996 | 0.7073 | 0.8828 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.3977 | 0.2308 | 0.5819 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.3130 | 0.2742 | 0.3532 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.5622 | 0.4936 | 0.6266 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.3014 | 0.2078 | 0.4000 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8027 | 0.7073 | 0.8869 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.3769 | 0.2163 | 0.5553 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.2058 | 0.1747 | 0.2361 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6652 | 0.6052 | 0.7253 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.3390 | 0.2321 | 0.4526 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8227 | 0.7426 | 0.8934 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.3991 | 0.2430 | 0.5760 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.2416 | 0.2125 | 0.2704 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.5665 | 0.5021 | 0.6309 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.3034 | 0.2083 | 0.4001 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8196 | 0.7209 | 0.9023 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.3534 | 0.2085 | 0.5495 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1961 | 0.1659 | 0.2279 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.6603 | 0.5933 | 0.7225 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3364 | 0.2182 | 0.4510 |
| M3 | CrossAttn3 | len256/scale_clinic | AUC-ROC | 0.8048 | 0.7132 | 0.8941 |
| M3 | CrossAttn3 | len256/scale_clinic | AUPRC | 0.3785 | 0.2237 | 0.5684 |
| M3 | CrossAttn3 | len256/scale_clinic | Brier | 0.2339 | 0.1997 | 0.2682 |
| M3 | CrossAttn3 | len256/scale_clinic | Accuracy | 0.6352 | 0.5708 | 0.6996 |
| M3 | CrossAttn3 | len256/scale_clinic | F1 | 0.3200 | 0.2185 | 0.4308 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8175 | 0.7277 | 0.8994 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.3956 | 0.2409 | 0.5880 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.1807 | 0.1528 | 0.2112 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.7296 | 0.6695 | 0.7854 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.3883 | 0.2718 | 0.5102 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.8113 | 0.7157 | 0.8955 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.4211 | 0.2521 | 0.6211 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.1789 | 0.1499 | 0.2106 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.7339 | 0.6738 | 0.7897 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.3800 | 0.2581 | 0.5047 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8296 | 0.7513 | 0.9009 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.4521 | 0.2708 | 0.6394 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.2089 | 0.1783 | 0.2403 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.6781 | 0.6179 | 0.7339 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.3478 | 0.2373 | 0.4602 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8391 | 0.7509 | 0.9133 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.4466 | 0.2624 | 0.6510 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.1907 | 0.1649 | 0.2176 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.6794 | 0.6172 | 0.7416 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.3366 | 0.2150 | 0.4510 |

