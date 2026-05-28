# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |
| M2 | CrossAttn | crop60 | 0.8635 | 0.5071 | 0.1630 | 0.7500 | 0.4557 |
| M2_2 | CrossAttn | norm | 0.8496 | 0.5209 | 0.1879 | 0.6221 | 0.3689 |
| M3 | CrossAttn3 | len128 | 0.8445 | 0.5060 | 0.2303 | 0.5581 | 0.3333 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_all** | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.8467 | 0.4688 | 0.1647 | 0.7151 | 0.4235 |
| norm | 0.7874 | 0.3089 | 0.2024 | 0.7733 | 0.4179 |
| crop80 | 0.8158 | 0.3229 | 0.2109 | 0.7151 | 0.3951 |
| **crop60** | 0.8635 | 0.5071 | 0.1630 | 0.7500 | 0.4557 |
| excl_extreme | 0.8561 | 0.4645 | 0.1520 | 0.7597 | 0.5067 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.8013 | 0.4094 | 0.1397 | 0.7558 | 0.3824 |
| **norm** | 0.8496 | 0.5209 | 0.1879 | 0.6221 | 0.3689 |
| crop80 | 0.8155 | 0.3587 | 0.1997 | 0.7616 | 0.4384 |
| crop60 | 0.8187 | 0.4115 | 0.2304 | 0.7093 | 0.4318 |
| excl_extreme | 0.8184 | 0.3779 | 0.1690 | 0.8182 | 0.4167 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **len128** | 0.8445 | 0.5060 | 0.2303 | 0.5581 | 0.3333 |
| norm | 0.7600 | 0.2375 | 0.2316 | 0.6802 | 0.3678 |
| crop80 | 0.8357 | 0.5083 | 0.1663 | 0.7616 | 0.4225 |
| crop60 | 0.7833 | 0.3481 | 0.2417 | 0.7267 | 0.3896 |
| excl_extreme | 0.7852 | 0.3351 | 0.1616 | 0.7922 | 0.4286 |

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

### len128  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8204 | +0.0091 | -1.300 | 2.63e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3497 | -0.0350 | 0.946 | 3.98e-01 | 4.38e-01 |
| Brier † | 0.1819 | 0.1661 | -0.0158 | 2.355 | 7.81e-02 | 1.25e-01 |
| Accuracy  | 0.7555 | 0.7643 | +0.0088 | -0.186 | 8.61e-01 | 8.75e-01 |
| F1  | 0.4514 | 0.4787 | +0.0273 | -0.679 | 5.35e-01 | 6.25e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8113 | 0.8328 | +0.0215 | -2.843 | 4.67e-02 | 6.25e-02 |
| AUPRC † | 0.3847 | 0.4619 | +0.0772 | -2.716 | 5.32e-02 | 6.25e-02 |
| Brier  | 0.1819 | 0.1922 | +0.0103 | -0.472 | 6.61e-01 | 8.12e-01 |
| Accuracy  | 0.7555 | 0.7644 | +0.0089 | -0.224 | 8.34e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4638 | +0.0124 | -0.377 | 7.25e-01 | 6.25e-01 |

### crop80  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8180 | +0.0066 | -1.043 | 3.56e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3491 | -0.0356 | 1.089 | 3.37e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1810 | -0.0010 | 0.098 | 9.27e-01 | 8.12e-01 |
| Accuracy  | 0.7555 | 0.7379 | -0.0176 | 0.832 | 4.52e-01 | 8.75e-01 |
| F1  | 0.4514 | 0.4435 | -0.0080 | 0.594 | 5.84e-01 | 6.25e-01 |

### crop60  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8113 | 0.8187 | +0.0074 | -2.560 | 6.27e-02 | 1.25e-01 |
| AUPRC  | 0.3847 | 0.3782 | -0.0065 | 0.349 | 7.44e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1905 | +0.0086 | -0.544 | 6.15e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7320 | -0.0235 | 0.589 | 5.88e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4493 | -0.0021 | 0.081 | 9.39e-01 | 6.25e-01 |

### excl_extreme  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8106 | -0.0008 | 0.089 | 9.33e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3398 | -0.0449 | 1.949 | 1.23e-01 | 1.25e-01 |
| Brier  | 0.1819 | 0.2053 | +0.0234 | -0.961 | 3.91e-01 | 8.12e-01 |
| Accuracy  | 0.7555 | 0.7244 | -0.0311 | 0.880 | 4.28e-01 | 4.38e-01 |
| F1  | 0.4514 | 0.4146 | -0.0368 | 1.645 | 1.75e-01 | 3.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: len128  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8204 | 0.8201 | -0.0003 | 0.042 | 9.69e-01 | 1.00e+00 |
| AUPRC  | 0.3497 | 0.4369 | +0.0872 | -1.623 | 1.80e-01 | 1.88e-01 |
| Brier  | 0.1661 | 0.1672 | +0.0011 | -0.156 | 8.83e-01 | 8.12e-01 |
| Accuracy  | 0.7643 | 0.7307 | -0.0336 | 1.276 | 2.71e-01 | 3.12e-01 |
| F1  | 0.4787 | 0.4310 | -0.0478 | 1.443 | 2.23e-01 | 4.38e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8328 | 0.8223 | -0.0105 | 1.411 | 2.31e-01 | 3.12e-01 |
| AUPRC  | 0.4619 | 0.4445 | -0.0174 | 0.375 | 7.27e-01 | 8.12e-01 |
| Brier  | 0.1922 | 0.1829 | -0.0093 | 0.263 | 8.06e-01 | 6.25e-01 |
| Accuracy  | 0.7644 | 0.7351 | -0.0293 | 0.770 | 4.84e-01 | 8.12e-01 |
| F1  | 0.4638 | 0.4392 | -0.0246 | 0.914 | 4.12e-01 | 6.25e-01 |

#### Case: crop80  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8180 | 0.8214 | +0.0034 | -0.936 | 4.02e-01 | 4.38e-01 |
| AUPRC  | 0.3491 | 0.3820 | +0.0329 | -1.302 | 2.63e-01 | 1.88e-01 |
| Brier  | 0.1810 | 0.1855 | +0.0046 | -0.271 | 8.00e-01 | 6.25e-01 |
| Accuracy  | 0.7379 | 0.7802 | +0.0423 | -1.216 | 2.91e-01 | 3.75e-01 |
| F1  | 0.4435 | 0.4749 | +0.0315 | -1.621 | 1.80e-01 | 1.88e-01 |

#### Case: crop60  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8187 | 0.8164 | -0.0023 | 0.289 | 7.87e-01 | 8.12e-01 |
| AUPRC  | 0.3782 | 0.3769 | -0.0013 | 0.038 | 9.72e-01 | 1.00e+00 |
| Brier † | 0.1905 | 0.2002 | +0.0097 | -2.148 | 9.82e-02 | 1.88e-01 |
| Accuracy  | 0.7320 | 0.7803 | +0.0483 | -1.096 | 3.35e-01 | 4.38e-01 |
| F1  | 0.4493 | 0.4749 | +0.0255 | -0.714 | 5.15e-01 | 8.12e-01 |

#### Case: excl_extreme  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8106 | 0.7926 | -0.0179 | 0.996 | 3.76e-01 | 4.38e-01 |
| AUPRC  | 0.3398 | 0.3633 | +0.0235 | -0.414 | 7.00e-01 | 1.00e+00 |
| Brier  | 0.2053 | 0.2018 | -0.0035 | 0.200 | 8.52e-01 | 6.25e-01 |
| Accuracy  | 0.7244 | 0.7406 | +0.0163 | -0.478 | 6.58e-01 | 6.88e-01 |
| F1  | 0.4146 | 0.4195 | +0.0049 | -0.178 | 8.67e-01 | 1.00e+00 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8201 | +0.0088 | -1.308 | 2.61e-01 | 3.12e-01 |
| AUPRC † | 0.3847 | 0.4369 | +0.0522 | -2.230 | 8.96e-02 | 1.25e-01 |
| Brier  | 0.1819 | 0.1672 | -0.0147 | 1.997 | 1.16e-01 | 1.88e-01 |
| Accuracy  | 0.7555 | 0.7307 | -0.0248 | 0.812 | 4.63e-01 | 4.38e-01 |
| F1  | 0.4514 | 0.4310 | -0.0205 | 0.815 | 4.61e-01 | 4.38e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8223 | +0.0110 | -0.979 | 3.83e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.4445 | +0.0598 | -1.124 | 3.24e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1829 | +0.0010 | -0.044 | 9.67e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7351 | -0.0204 | 0.652 | 5.50e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4392 | -0.0122 | 0.501 | 6.43e-01 | 8.12e-01 |

### crop80  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8214 | +0.0101 | -1.139 | 3.18e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3820 | -0.0027 | 0.092 | 9.31e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.1855 | +0.0036 | -0.261 | 8.07e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7802 | +0.0247 | -0.672 | 5.39e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4749 | +0.0235 | -1.106 | 3.31e-01 | 3.12e-01 |

### crop60  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8164 | +0.0051 | -0.719 | 5.12e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3769 | -0.0077 | 0.354 | 7.41e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.2002 | +0.0183 | -1.010 | 3.70e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7803 | +0.0248 | -0.576 | 5.96e-01 | 4.38e-01 |
| F1  | 0.4514 | 0.4749 | +0.0235 | -0.683 | 5.32e-01 | 6.25e-01 |

### excl_extreme  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.7926 | -0.0187 | 1.489 | 2.11e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3633 | -0.0213 | 0.344 | 7.48e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.2018 | +0.0199 | -1.227 | 2.87e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7406 | -0.0148 | 0.290 | 7.86e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4195 | -0.0319 | 1.077 | 3.42e-01 | 4.38e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_all | AUC-ROC | 0.8325 | 0.7386 | 0.9091 |
| M1 | LR | scale_all | AUPRC | 0.5008 | 0.2965 | 0.6937 |
| M1 | LR | scale_all | Brier | 0.1804 | 0.1527 | 0.2118 |
| M1 | LR | scale_all | Accuracy | 0.6977 | 0.6279 | 0.7674 |
| M1 | LR | scale_all | F1 | 0.3953 | 0.2535 | 0.5228 |
| M2 | CrossAttn | len128 | AUC-ROC | 0.8467 | 0.7624 | 0.9152 |
| M2 | CrossAttn | len128 | AUPRC | 0.4688 | 0.2754 | 0.6737 |
| M2 | CrossAttn | len128 | Brier | 0.1647 | 0.1282 | 0.2021 |
| M2 | CrossAttn | len128 | Accuracy | 0.7151 | 0.6453 | 0.7791 |
| M2 | CrossAttn | len128 | F1 | 0.4235 | 0.2821 | 0.5517 |
| M2 | CrossAttn | norm | AUC-ROC | 0.7874 | 0.6917 | 0.8746 |
| M2 | CrossAttn | norm | AUPRC | 0.3089 | 0.1874 | 0.5143 |
| M2 | CrossAttn | norm | Brier | 0.2024 | 0.1626 | 0.2451 |
| M2 | CrossAttn | norm | Accuracy | 0.7733 | 0.7093 | 0.8372 |
| M2 | CrossAttn | norm | F1 | 0.4179 | 0.2631 | 0.5588 |
| M2 | CrossAttn | crop80 | AUC-ROC | 0.8158 | 0.7309 | 0.8884 |
| M2 | CrossAttn | crop80 | AUPRC | 0.3229 | 0.1921 | 0.5296 |
| M2 | CrossAttn | crop80 | Brier | 0.2109 | 0.1622 | 0.2604 |
| M2 | CrossAttn | crop80 | Accuracy | 0.7151 | 0.6453 | 0.7849 |
| M2 | CrossAttn | crop80 | F1 | 0.3951 | 0.2535 | 0.5250 |
| M2 | CrossAttn | crop60 | AUC-ROC | 0.8635 | 0.7848 | 0.9281 |
| M2 | CrossAttn | crop60 | AUPRC | 0.5071 | 0.3120 | 0.7056 |
| M2 | CrossAttn | crop60 | Brier | 0.1630 | 0.1296 | 0.1969 |
| M2 | CrossAttn | crop60 | Accuracy | 0.7500 | 0.6860 | 0.8140 |
| M2 | CrossAttn | crop60 | F1 | 0.4557 | 0.3125 | 0.5862 |
| M2 | CrossAttn | excl_extreme | AUC-ROC | 0.8561 | 0.7739 | 0.9198 |
| M2 | CrossAttn | excl_extreme | AUPRC | 0.4645 | 0.2781 | 0.6648 |
| M2 | CrossAttn | excl_extreme | Brier | 0.1520 | 0.1198 | 0.1878 |
| M2 | CrossAttn | excl_extreme | Accuracy | 0.7597 | 0.6883 | 0.8247 |
| M2 | CrossAttn | excl_extreme | F1 | 0.5067 | 0.3529 | 0.6400 |
| M2_2 | CrossAttn | len128 | AUC-ROC | 0.8013 | 0.6931 | 0.8955 |
| M2_2 | CrossAttn | len128 | AUPRC | 0.4094 | 0.2229 | 0.6181 |
| M2_2 | CrossAttn | len128 | Brier | 0.1397 | 0.1092 | 0.1721 |
| M2_2 | CrossAttn | len128 | Accuracy | 0.7558 | 0.6860 | 0.8198 |
| M2_2 | CrossAttn | len128 | F1 | 0.3824 | 0.2222 | 0.5231 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8496 | 0.7428 | 0.9346 |
| M2_2 | CrossAttn | norm | AUPRC | 0.5209 | 0.3260 | 0.7236 |
| M2_2 | CrossAttn | norm | Brier | 0.1879 | 0.1529 | 0.2245 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6221 | 0.5465 | 0.6919 |
| M2_2 | CrossAttn | norm | F1 | 0.3689 | 0.2418 | 0.4828 |
| M2_2 | CrossAttn | crop80 | AUC-ROC | 0.8155 | 0.7286 | 0.8915 |
| M2_2 | CrossAttn | crop80 | AUPRC | 0.3587 | 0.2004 | 0.5581 |
| M2_2 | CrossAttn | crop80 | Brier | 0.1997 | 0.1621 | 0.2413 |
| M2_2 | CrossAttn | crop80 | Accuracy | 0.7616 | 0.6977 | 0.8256 |
| M2_2 | CrossAttn | crop80 | F1 | 0.4384 | 0.2899 | 0.5753 |
| M2_2 | CrossAttn | crop60 | AUC-ROC | 0.8187 | 0.7078 | 0.9059 |
| M2_2 | CrossAttn | crop60 | AUPRC | 0.4115 | 0.2359 | 0.6134 |
| M2_2 | CrossAttn | crop60 | Brier | 0.2304 | 0.1899 | 0.2741 |
| M2_2 | CrossAttn | crop60 | Accuracy | 0.7093 | 0.6394 | 0.7733 |
| M2_2 | CrossAttn | crop60 | F1 | 0.4318 | 0.2963 | 0.5582 |
| M2_2 | CrossAttn | excl_extreme | AUC-ROC | 0.8184 | 0.7036 | 0.9126 |
| M2_2 | CrossAttn | excl_extreme | AUPRC | 0.3779 | 0.2068 | 0.6230 |
| M2_2 | CrossAttn | excl_extreme | Brier | 0.1690 | 0.1345 | 0.2067 |
| M2_2 | CrossAttn | excl_extreme | Accuracy | 0.8182 | 0.7532 | 0.8766 |
| M2_2 | CrossAttn | excl_extreme | F1 | 0.4167 | 0.2222 | 0.5819 |
| M3 | CrossAttn3 | len128 | AUC-ROC | 0.8445 | 0.7579 | 0.9217 |
| M3 | CrossAttn3 | len128 | AUPRC | 0.5060 | 0.3029 | 0.7159 |
| M3 | CrossAttn3 | len128 | Brier | 0.2303 | 0.1995 | 0.2618 |
| M3 | CrossAttn3 | len128 | Accuracy | 0.5581 | 0.4826 | 0.6337 |
| M3 | CrossAttn3 | len128 | F1 | 0.3333 | 0.2178 | 0.4445 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.7600 | 0.6763 | 0.8366 |
| M3 | CrossAttn3 | norm | AUPRC | 0.2375 | 0.1489 | 0.3873 |
| M3 | CrossAttn3 | norm | Brier | 0.2316 | 0.1859 | 0.2815 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6802 | 0.6047 | 0.7500 |
| M3 | CrossAttn3 | norm | F1 | 0.3678 | 0.2368 | 0.5000 |
| M3 | CrossAttn3 | crop80 | AUC-ROC | 0.8357 | 0.7332 | 0.9227 |
| M3 | CrossAttn3 | crop80 | AUPRC | 0.5083 | 0.3096 | 0.7342 |
| M3 | CrossAttn3 | crop80 | Brier | 0.1663 | 0.1306 | 0.2044 |
| M3 | CrossAttn3 | crop80 | Accuracy | 0.7616 | 0.6977 | 0.8256 |
| M3 | CrossAttn3 | crop80 | F1 | 0.4225 | 0.2667 | 0.5614 |
| M3 | CrossAttn3 | crop60 | AUC-ROC | 0.7833 | 0.6703 | 0.8806 |
| M3 | CrossAttn3 | crop60 | AUPRC | 0.3481 | 0.2025 | 0.5874 |
| M3 | CrossAttn3 | crop60 | Brier | 0.2417 | 0.1908 | 0.2953 |
| M3 | CrossAttn3 | crop60 | Accuracy | 0.7267 | 0.6570 | 0.7907 |
| M3 | CrossAttn3 | crop60 | F1 | 0.3896 | 0.2400 | 0.5294 |
| M3 | CrossAttn3 | excl_extreme | AUC-ROC | 0.7852 | 0.6834 | 0.8713 |
| M3 | CrossAttn3 | excl_extreme | AUPRC | 0.3351 | 0.1933 | 0.5286 |
| M3 | CrossAttn3 | excl_extreme | Brier | 0.1616 | 0.1294 | 0.1956 |
| M3 | CrossAttn3 | excl_extreme | Accuracy | 0.7922 | 0.7208 | 0.8571 |
| M3 | CrossAttn3 | excl_extreme | F1 | 0.4286 | 0.2500 | 0.5818 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-len128 | 0.8325 | 0.8467 | +0.0142 | -0.408 | 6.829e-01 | ns |
| M1-LR vs M2-norm | 0.8325 | 0.7874 | -0.0451 | 1.326 | 1.847e-01 | ns |
| M1-LR vs M2-crop80 | 0.8325 | 0.8158 | -0.0167 | 0.422 | 6.728e-01 | ns |
| M1-LR vs M2-crop60 | 0.8325 | 0.8635 | +0.0309 | -1.102 | 2.706e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-len128 | 0.8325 | 0.8445 | +0.0120 | -0.278 | 7.813e-01 | ns |
| M1-LR vs M3-norm | 0.8325 | 0.7600 | -0.0725 | 1.916 | 5.537e-02 | † |
| M1-LR vs M3-crop80 | 0.8325 | 0.8357 | +0.0032 | -0.075 | 9.399e-01 | ns |
| M1-LR vs M3-crop60 | 0.8325 | 0.7833 | -0.0492 | 1.075 | 2.823e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len128 vs M2_2-len128 | 0.8467 | 0.8013 | -0.0454 | 1.030 | 3.030e-01 | ns |
| M2-norm vs M2_2-norm | 0.7874 | 0.8496 | +0.0621 | -1.726 | 8.439e-02 | † |
| M2-crop80 vs M2_2-crop80 | 0.8158 | 0.8155 | -0.0003 | 0.010 | 9.924e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.8635 | 0.8187 | -0.0448 | 1.382 | 1.671e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8561 | 0.5170 | -0.3391 | 5.571 | 2.526e-08 | *** |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len128 vs M3-len128 | 0.8467 | 0.8445 | -0.0022 | 0.073 | 9.417e-01 | ns |
| M2-norm vs M3-norm | 0.7874 | 0.7600 | -0.0274 | 0.934 | 3.502e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8158 | 0.8357 | +0.0199 | -0.576 | 5.647e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.8635 | 0.7833 | -0.0801 | 2.325 | 2.008e-02 | * |
| M2-excl_extreme vs M3-excl_extreme | 0.8561 | 0.7852 | -0.0709 | 2.743 | 6.079e-03 | ** |

