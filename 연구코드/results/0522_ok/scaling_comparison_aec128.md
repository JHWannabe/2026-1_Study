# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8409 | 0.4704 | 0.1673 | 0.7341 | 0.3947 |
| M2 | CrossAttn | excl_extreme/scale_both | 0.8603 | 0.4688 | 0.1614 | 0.7355 | 0.4384 |
| M2_2 | CrossAttn | norm/scale_both | 0.8380 | 0.5394 | 0.1292 | 0.8035 | 0.4333 |
| M3 | CrossAttn3 | len128/scale_both | 0.8130 | 0.4177 | 0.1997 | 0.6627 | 0.3636 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_clinic** | 0.8409 | 0.4704 | 0.1673 | 0.7341 | 0.3947 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.8308 | 0.3529 | 0.1567 | 0.7746 | 0.4800 |
| crop80/scale_both | 0.8446 | 0.4631 | 0.1760 | 0.6879 | 0.4000 |
| crop60/scale_both | 0.8180 | 0.3276 | 0.1920 | 0.6763 | 0.3913 |
| norm/scale_both | 0.8315 | 0.3753 | 0.2024 | 0.6590 | 0.3789 |
| **excl_extreme/scale_both** | 0.8603 | 0.4688 | 0.1614 | 0.7355 | 0.4384 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128/scale_both | 0.8095 | 0.3507 | 0.1972 | 0.6705 | 0.3736 |
| crop80/scale_both | 0.8236 | 0.3752 | 0.1782 | 0.7283 | 0.4198 |
| crop60/scale_both | 0.8246 | 0.3719 | 0.1793 | 0.7168 | 0.3951 |
| **norm/scale_both** | 0.8380 | 0.5394 | 0.1292 | 0.8035 | 0.4333 |
| excl_extreme/scale_both | 0.8009 | 0.3988 | 0.2235 | 0.6516 | 0.3571 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **len128/scale_both** | 0.8130 | 0.4177 | 0.1997 | 0.6627 | 0.3636 |
| crop80/scale_both | 0.8072 | 0.3906 | 0.2506 | 0.5843 | 0.3429 |
| crop60/scale_both | 0.8120 | 0.4430 | 0.1794 | 0.7108 | 0.4146 |
| norm/scale_both | 0.8123 | 0.4596 | 0.1656 | 0.7229 | 0.4250 |
| excl_extreme/scale_both | 0.7881 | 0.3681 | 0.1826 | 0.6959 | 0.3662 |

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
| AUC-ROC  | 0.8090 | 0.8073 | -0.0017 | 0.098 | 9.27e-01 | 8.12e-01 |
| AUPRC  | 0.4142 | 0.4202 | +0.0060 | -0.166 | 8.76e-01 | 1.00e+00 |
| Brier  | 0.1818 | 0.2178 | +0.0360 | -1.067 | 3.46e-01 | 4.38e-01 |
| Accuracy  | 0.7203 | 0.6638 | -0.0565 | 1.219 | 2.90e-01 | 2.50e-01 |
| F1  | 0.3909 | 0.3929 | +0.0021 | -0.067 | 9.50e-01 | 1.00e+00 |

### crop80/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8107 | +0.0016 | -0.112 | 9.16e-01 | 1.00e+00 |
| AUPRC  | 0.4142 | 0.3900 | -0.0242 | 0.867 | 4.35e-01 | 4.38e-01 |
| Brier  | 0.1818 | 0.2014 | +0.0196 | -1.106 | 3.31e-01 | 4.38e-01 |
| Accuracy  | 0.7203 | 0.6725 | -0.0478 | 1.253 | 2.79e-01 | 3.12e-01 |
| F1  | 0.3909 | 0.3937 | +0.0029 | -0.134 | 9.00e-01 | 8.12e-01 |

### crop60/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8078 | -0.0013 | 0.118 | 9.11e-01 | 1.00e+00 |
| AUPRC  | 0.4142 | 0.4040 | -0.0103 | 0.258 | 8.09e-01 | 8.12e-01 |
| Brier  | 0.1818 | 0.1947 | +0.0129 | -1.247 | 2.80e-01 | 4.38e-01 |
| Accuracy  | 0.7203 | 0.6739 | -0.0464 | 1.828 | 1.42e-01 | 2.50e-01 |
| F1  | 0.3909 | 0.3823 | -0.0085 | 0.414 | 7.00e-01 | 6.25e-01 |

### norm/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8074 | -0.0017 | 0.110 | 9.18e-01 | 8.12e-01 |
| AUPRC  | 0.4142 | 0.4127 | -0.0015 | 0.042 | 9.68e-01 | 1.00e+00 |
| Brier  | 0.1818 | 0.1921 | +0.0103 | -0.566 | 6.01e-01 | 6.25e-01 |
| Accuracy  | 0.7203 | 0.6971 | -0.0232 | 0.664 | 5.43e-01 | 6.88e-01 |
| F1  | 0.3909 | 0.3798 | -0.0111 | 0.547 | 6.14e-01 | 6.25e-01 |

### excl_extreme/scale_both  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8136 | +0.0046 | -0.126 | 9.06e-01 | 6.25e-01 |
| AUPRC  | 0.4142 | 0.3936 | -0.0207 | 0.586 | 5.89e-01 | 1.00e+00 |
| Brier  | 0.1818 | 0.1716 | -0.0102 | 0.338 | 7.53e-01 | 6.25e-01 |
| Accuracy  | 0.7203 | 0.7258 | +0.0055 | -0.104 | 9.22e-01 | 8.12e-01 |
| F1  | 0.3909 | 0.4173 | +0.0265 | -0.477 | 6.59e-01 | 6.25e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len128/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8073 | 0.8048 | -0.0024 | 0.083 | 9.38e-01 | 1.00e+00 |
| AUPRC  | 0.4202 | 0.3842 | -0.0360 | 0.988 | 3.79e-01 | 4.38e-01 |
| Brier  | 0.2178 | 0.2076 | -0.0102 | 0.267 | 8.03e-01 | 1.00e+00 |
| Accuracy  | 0.6638 | 0.6697 | +0.0059 | -0.097 | 9.28e-01 | 1.00e+00 |
| F1  | 0.3929 | 0.3597 | -0.0333 | 0.683 | 5.32e-01 | 6.25e-01 |

#### Case: crop80/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8107 | 0.8091 | -0.0016 | 0.073 | 9.46e-01 | 1.00e+00 |
| AUPRC  | 0.3900 | 0.3901 | +0.0000 | -0.001 | 9.99e-01 | 1.00e+00 |
| Brier  | 0.2014 | 0.1704 | -0.0310 | 1.268 | 2.74e-01 | 4.38e-01 |
| Accuracy  | 0.6725 | 0.7424 | +0.0700 | -1.438 | 2.24e-01 | 3.12e-01 |
| F1  | 0.3937 | 0.3951 | +0.0014 | -0.031 | 9.77e-01 | 1.00e+00 |

#### Case: crop60/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8078 | 0.8258 | +0.0180 | -0.724 | 5.09e-01 | 8.12e-01 |
| AUPRC  | 0.4040 | 0.3983 | -0.0057 | 0.100 | 9.25e-01 | 1.00e+00 |
| Brier  | 0.1947 | 0.1916 | -0.0031 | 0.179 | 8.66e-01 | 1.00e+00 |
| Accuracy  | 0.6739 | 0.7030 | +0.0291 | -2.117 | 1.02e-01 | 1.25e-01 |
| F1  | 0.3823 | 0.3962 | +0.0138 | -0.383 | 7.21e-01 | 6.25e-01 |

#### Case: norm/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8074 | 0.8223 | +0.0149 | -0.383 | 7.21e-01 | 1.00e+00 |
| AUPRC  | 0.4127 | 0.4140 | +0.0013 | -0.016 | 9.88e-01 | 1.00e+00 |
| Brier  | 0.1921 | 0.1879 | -0.0042 | 0.174 | 8.70e-01 | 1.00e+00 |
| Accuracy  | 0.6971 | 0.7061 | +0.0090 | -0.167 | 8.75e-01 | 1.00e+00 |
| F1  | 0.3798 | 0.3940 | +0.0142 | -0.528 | 6.25e-01 | 6.25e-01 |

#### Case: excl_extreme/scale_both  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8136 | 0.8185 | +0.0048 | -0.141 | 8.95e-01 | 8.12e-01 |
| AUPRC  | 0.3936 | 0.4374 | +0.0439 | -0.880 | 4.29e-01 | 6.25e-01 |
| Brier  | 0.1716 | 0.1798 | +0.0082 | -0.234 | 8.26e-01 | 8.12e-01 |
| Accuracy  | 0.7258 | 0.7322 | +0.0064 | -0.105 | 9.22e-01 | 1.00e+00 |
| F1  | 0.4173 | 0.3953 | -0.0220 | 0.378 | 7.24e-01 | 8.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8048 | -0.0042 | 0.191 | 8.58e-01 | 1.00e+00 |
| AUPRC  | 0.4142 | 0.3842 | -0.0300 | 0.569 | 6.00e-01 | 6.25e-01 |
| Brier  | 0.1818 | 0.2076 | +0.0258 | -1.746 | 1.56e-01 | 1.25e-01 |
| Accuracy  | 0.7203 | 0.6697 | -0.0506 | 1.414 | 2.30e-01 | 1.88e-01 |
| F1  | 0.3909 | 0.3597 | -0.0312 | 1.045 | 3.55e-01 | 3.75e-01 |

### crop80/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8091 | +0.0000 | -0.001 | 9.99e-01 | 1.00e+00 |
| AUPRC  | 0.4142 | 0.3901 | -0.0242 | 0.615 | 5.72e-01 | 6.25e-01 |
| Brier  | 0.1818 | 0.1704 | -0.0114 | 1.454 | 2.20e-01 | 4.38e-01 |
| Accuracy  | 0.7203 | 0.7424 | +0.0221 | -1.202 | 2.96e-01 | 3.12e-01 |
| F1  | 0.3909 | 0.3951 | +0.0043 | -0.165 | 8.77e-01 | 8.12e-01 |

### crop60/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8258 | +0.0167 | -0.738 | 5.01e-01 | 8.12e-01 |
| AUPRC  | 0.4142 | 0.3983 | -0.0160 | 0.327 | 7.60e-01 | 1.00e+00 |
| Brier  | 0.1818 | 0.1916 | +0.0098 | -0.695 | 5.25e-01 | 4.38e-01 |
| Accuracy  | 0.7203 | 0.7030 | -0.0173 | 0.892 | 4.23e-01 | 8.12e-01 |
| F1  | 0.3909 | 0.3962 | +0.0053 | -0.181 | 8.66e-01 | 1.00e+00 |

### norm/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8223 | +0.0133 | -0.489 | 6.50e-01 | 1.00e+00 |
| AUPRC  | 0.4142 | 0.4140 | -0.0003 | 0.004 | 9.97e-01 | 1.00e+00 |
| Brier  | 0.1818 | 0.1879 | +0.0061 | -0.885 | 4.26e-01 | 4.38e-01 |
| Accuracy  | 0.7203 | 0.7061 | -0.0142 | 0.724 | 5.09e-01 | 6.25e-01 |
| F1  | 0.3909 | 0.3940 | +0.0031 | -0.117 | 9.13e-01 | 6.25e-01 |

### excl_extreme/scale_both  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8185 | +0.0094 | -0.229 | 8.30e-01 | 1.00e+00 |
| AUPRC  | 0.4142 | 0.4374 | +0.0232 | -0.380 | 7.23e-01 | 8.12e-01 |
| Brier  | 0.1818 | 0.1798 | -0.0021 | 0.116 | 9.13e-01 | 1.00e+00 |
| Accuracy  | 0.7203 | 0.7322 | +0.0119 | -0.466 | 6.65e-01 | 8.12e-01 |
| F1  | 0.3909 | 0.3953 | +0.0045 | -0.133 | 9.01e-01 | 1.00e+00 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_clinic | AUC-ROC | 0.8409 | 0.7489 | 0.9165 |
| M1 | LR | scale_clinic | AUPRC | 0.4704 | 0.2806 | 0.6714 |
| M1 | LR | scale_clinic | Brier | 0.1673 | 0.1401 | 0.1967 |
| M1 | LR | scale_clinic | Accuracy | 0.7341 | 0.6647 | 0.7977 |
| M1 | LR | scale_clinic | F1 | 0.3947 | 0.2535 | 0.5250 |
| M2 | CrossAttn | len128/scale_both | AUC-ROC | 0.8308 | 0.7423 | 0.9072 |
| M2 | CrossAttn | len128/scale_both | AUPRC | 0.3529 | 0.2163 | 0.5582 |
| M2 | CrossAttn | len128/scale_both | Brier | 0.1567 | 0.1277 | 0.1869 |
| M2 | CrossAttn | len128/scale_both | Accuracy | 0.7746 | 0.7110 | 0.8324 |
| M2 | CrossAttn | len128/scale_both | F1 | 0.4800 | 0.3333 | 0.6076 |
| M2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.8446 | 0.7520 | 0.9218 |
| M2 | CrossAttn | crop80/scale_both | AUPRC | 0.4631 | 0.2741 | 0.6778 |
| M2 | CrossAttn | crop80/scale_both | Brier | 0.1760 | 0.1471 | 0.2059 |
| M2 | CrossAttn | crop80/scale_both | Accuracy | 0.6879 | 0.6185 | 0.7572 |
| M2 | CrossAttn | crop80/scale_both | F1 | 0.4000 | 0.2653 | 0.5227 |
| M2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.8180 | 0.7234 | 0.8974 |
| M2 | CrossAttn | crop60/scale_both | AUPRC | 0.3276 | 0.2061 | 0.5234 |
| M2 | CrossAttn | crop60/scale_both | Brier | 0.1920 | 0.1567 | 0.2281 |
| M2 | CrossAttn | crop60/scale_both | Accuracy | 0.6763 | 0.6069 | 0.7457 |
| M2 | CrossAttn | crop60/scale_both | F1 | 0.3913 | 0.2609 | 0.5122 |
| M2 | CrossAttn | norm/scale_both | AUC-ROC | 0.8315 | 0.7413 | 0.9093 |
| M2 | CrossAttn | norm/scale_both | AUPRC | 0.3753 | 0.2328 | 0.5977 |
| M2 | CrossAttn | norm/scale_both | Brier | 0.2024 | 0.1701 | 0.2338 |
| M2 | CrossAttn | norm/scale_both | Accuracy | 0.6590 | 0.5896 | 0.7283 |
| M2 | CrossAttn | norm/scale_both | F1 | 0.3789 | 0.2528 | 0.5000 |
| M2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8603 | 0.7614 | 0.9386 |
| M2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.4688 | 0.2872 | 0.7189 |
| M2 | CrossAttn | excl_extreme/scale_both | Brier | 0.1614 | 0.1297 | 0.1954 |
| M2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.7355 | 0.6645 | 0.8000 |
| M2 | CrossAttn | excl_extreme/scale_both | F1 | 0.4384 | 0.2817 | 0.5715 |
| M2_2 | CrossAttn | len128/scale_both | AUC-ROC | 0.8095 | 0.7071 | 0.8951 |
| M2_2 | CrossAttn | len128/scale_both | AUPRC | 0.3507 | 0.2118 | 0.5675 |
| M2_2 | CrossAttn | len128/scale_both | Brier | 0.1972 | 0.1632 | 0.2306 |
| M2_2 | CrossAttn | len128/scale_both | Accuracy | 0.6705 | 0.6069 | 0.7399 |
| M2_2 | CrossAttn | len128/scale_both | F1 | 0.3736 | 0.2444 | 0.4949 |
| M2_2 | CrossAttn | crop80/scale_both | AUC-ROC | 0.8236 | 0.7234 | 0.9068 |
| M2_2 | CrossAttn | crop80/scale_both | AUPRC | 0.3752 | 0.2341 | 0.5928 |
| M2_2 | CrossAttn | crop80/scale_both | Brier | 0.1782 | 0.1440 | 0.2149 |
| M2_2 | CrossAttn | crop80/scale_both | Accuracy | 0.7283 | 0.6647 | 0.7977 |
| M2_2 | CrossAttn | crop80/scale_both | F1 | 0.4198 | 0.2778 | 0.5474 |
| M2_2 | CrossAttn | crop60/scale_both | AUC-ROC | 0.8246 | 0.7293 | 0.9045 |
| M2_2 | CrossAttn | crop60/scale_both | AUPRC | 0.3719 | 0.2282 | 0.5899 |
| M2_2 | CrossAttn | crop60/scale_both | Brier | 0.1793 | 0.1474 | 0.2127 |
| M2_2 | CrossAttn | crop60/scale_both | Accuracy | 0.7168 | 0.6474 | 0.7803 |
| M2_2 | CrossAttn | crop60/scale_both | F1 | 0.3951 | 0.2535 | 0.5200 |
| M2_2 | CrossAttn | norm/scale_both | AUC-ROC | 0.8380 | 0.7395 | 0.9193 |
| M2_2 | CrossAttn | norm/scale_both | AUPRC | 0.5394 | 0.3388 | 0.7317 |
| M2_2 | CrossAttn | norm/scale_both | Brier | 0.1292 | 0.1047 | 0.1564 |
| M2_2 | CrossAttn | norm/scale_both | Accuracy | 0.8035 | 0.7457 | 0.8613 |
| M2_2 | CrossAttn | norm/scale_both | F1 | 0.4333 | 0.2667 | 0.5763 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUC-ROC | 0.8009 | 0.6849 | 0.8996 |
| M2_2 | CrossAttn | excl_extreme/scale_both | AUPRC | 0.3988 | 0.2126 | 0.6162 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Brier | 0.2235 | 0.1805 | 0.2663 |
| M2_2 | CrossAttn | excl_extreme/scale_both | Accuracy | 0.6516 | 0.5742 | 0.7290 |
| M2_2 | CrossAttn | excl_extreme/scale_both | F1 | 0.3571 | 0.2250 | 0.4952 |
| M3 | CrossAttn3 | len128/scale_both | AUC-ROC | 0.8130 | 0.7183 | 0.8923 |
| M3 | CrossAttn3 | len128/scale_both | AUPRC | 0.4177 | 0.2364 | 0.6204 |
| M3 | CrossAttn3 | len128/scale_both | Brier | 0.1997 | 0.1612 | 0.2367 |
| M3 | CrossAttn3 | len128/scale_both | Accuracy | 0.6627 | 0.5904 | 0.7349 |
| M3 | CrossAttn3 | len128/scale_both | F1 | 0.3636 | 0.2254 | 0.4898 |
| M3 | CrossAttn3 | crop80/scale_both | AUC-ROC | 0.8072 | 0.7109 | 0.8906 |
| M3 | CrossAttn3 | crop80/scale_both | AUPRC | 0.3906 | 0.2225 | 0.5987 |
| M3 | CrossAttn3 | crop80/scale_both | Brier | 0.2506 | 0.2083 | 0.2924 |
| M3 | CrossAttn3 | crop80/scale_both | Accuracy | 0.5843 | 0.5119 | 0.6566 |
| M3 | CrossAttn3 | crop80/scale_both | F1 | 0.3429 | 0.2222 | 0.4545 |
| M3 | CrossAttn3 | crop60/scale_both | AUC-ROC | 0.8120 | 0.7072 | 0.8968 |
| M3 | CrossAttn3 | crop60/scale_both | AUPRC | 0.4430 | 0.2473 | 0.6399 |
| M3 | CrossAttn3 | crop60/scale_both | Brier | 0.1794 | 0.1432 | 0.2151 |
| M3 | CrossAttn3 | crop60/scale_both | Accuracy | 0.7108 | 0.6446 | 0.7771 |
| M3 | CrossAttn3 | crop60/scale_both | F1 | 0.4146 | 0.2750 | 0.5393 |
| M3 | CrossAttn3 | norm/scale_both | AUC-ROC | 0.8123 | 0.6935 | 0.9038 |
| M3 | CrossAttn3 | norm/scale_both | AUPRC | 0.4596 | 0.2574 | 0.6649 |
| M3 | CrossAttn3 | norm/scale_both | Brier | 0.1656 | 0.1344 | 0.1994 |
| M3 | CrossAttn3 | norm/scale_both | Accuracy | 0.7229 | 0.6566 | 0.7892 |
| M3 | CrossAttn3 | norm/scale_both | F1 | 0.4250 | 0.2820 | 0.5517 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUC-ROC | 0.7881 | 0.6865 | 0.8774 |
| M3 | CrossAttn3 | excl_extreme/scale_both | AUPRC | 0.3681 | 0.1877 | 0.5782 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Brier | 0.1826 | 0.1422 | 0.2225 |
| M3 | CrossAttn3 | excl_extreme/scale_both | Accuracy | 0.6959 | 0.6216 | 0.7703 |
| M3 | CrossAttn3 | excl_extreme/scale_both | F1 | 0.3662 | 0.2154 | 0.5060 |

