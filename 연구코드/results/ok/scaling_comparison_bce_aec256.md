# Scaling Comparison — Test Set Performance (AEC 256pt, BCEWithLogitsLoss)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |
| M2 | CrossAttn | norm/scale_clinic | 0.8884 | 0.5643 | 0.1620 | 0.7849 | 0.4789 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | 0.8546 | 0.4350 | 0.1935 | 0.7922 | 0.4483 |
| M3 | CrossAttn3 | norm/scale_clinic | 0.8858 | 0.5225 | 0.1819 | 0.7093 | 0.4186 |

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
| **norm/scale_clinic** | 0.8884 | 0.5643 | 0.1620 | 0.7849 | 0.4789 |
| excl_extreme/scale_clinic | 0.8507 | 0.4750 | 0.1614 | 0.6948 | 0.4337 |
| len128/scale_clinic | 0.8694 | 0.4530 | 0.2276 | 0.6047 | 0.3818 |
| crop80/scale_clinic | 0.8521 | 0.4949 | 0.2016 | 0.6977 | 0.4091 |
| crop60/scale_clinic | 0.8553 | 0.4906 | 0.1833 | 0.6163 | 0.3774 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm/scale_clinic | 0.7931 | 0.4422 | 0.1567 | 0.7442 | 0.3714 |
| **excl_extreme/scale_clinic** | 0.8546 | 0.4350 | 0.1935 | 0.7922 | 0.4483 |
| len128/scale_clinic | 0.7887 | 0.3915 | 0.1946 | 0.6686 | 0.3736 |
| crop80/scale_clinic | 0.8051 | 0.3982 | 0.1821 | 0.7151 | 0.4096 |
| crop60/scale_clinic | 0.8098 | 0.4034 | 0.1933 | 0.7442 | 0.4500 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm/scale_clinic** | 0.8858 | 0.5225 | 0.1819 | 0.7093 | 0.4186 |
| excl_extreme/scale_clinic | 0.8801 | 0.6350 | 0.2347 | 0.6104 | 0.4000 |
| len128/scale_clinic | 0.8559 | 0.4901 | 0.1890 | 0.7151 | 0.4368 |
| crop80/scale_clinic | 0.8720 | 0.5352 | 0.1940 | 0.6919 | 0.4176 |
| crop60/scale_clinic | 0.8644 | 0.5628 | 0.1982 | 0.7326 | 0.4250 |

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
| AUC-ROC  | 0.8113 | 0.8113 | -0.0001 | 0.010 | 9.93e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.4274 | +0.0427 | -1.785 | 1.49e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1671 | -0.0148 | 2.066 | 1.08e-01 | 1.25e-01 |
| Accuracy  | 0.7555 | 0.7700 | +0.0145 | -0.404 | 7.07e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4559 | +0.0044 | -0.170 | 8.73e-01 | 8.75e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8003 | -0.0110 | 0.700 | 5.22e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3446 | -0.0400 | 0.795 | 4.71e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1795 | -0.0024 | 0.123 | 9.08e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7519 | -0.0035 | 0.115 | 9.14e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4361 | -0.0153 | 0.548 | 6.13e-01 | 6.25e-01 |

### len128/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8154 | +0.0041 | -0.241 | 8.22e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3930 | +0.0084 | -0.208 | 8.45e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1763 | -0.0056 | 0.357 | 7.39e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7308 | -0.0247 | 0.487 | 6.52e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4358 | -0.0157 | 0.433 | 6.87e-01 | 8.12e-01 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8097 | -0.0017 | 0.175 | 8.70e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3598 | -0.0249 | 1.641 | 1.76e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1857 | +0.0038 | -0.377 | 7.25e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7744 | +0.0189 | -0.745 | 4.97e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4656 | +0.0141 | -0.824 | 4.56e-01 | 8.12e-01 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8222 | +0.0108 | -1.804 | 1.46e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.4102 | +0.0255 | -1.063 | 3.48e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1659 | -0.0160 | 1.643 | 1.76e-01 | 1.88e-01 |
| Accuracy  | 0.7555 | 0.7205 | -0.0350 | 1.149 | 3.15e-01 | 3.12e-01 |
| F1  | 0.4514 | 0.4301 | -0.0213 | 1.479 | 2.13e-01 | 3.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8175 | +0.0063 | -0.933 | 4.04e-01 | 6.25e-01 |
| AUPRC  | 0.4274 | 0.4161 | -0.0113 | 0.685 | 5.31e-01 | 1.00e+00 |
| Brier  | 0.1671 | 0.1672 | +0.0002 | -0.010 | 9.93e-01 | 1.00e+00 |
| Accuracy  | 0.7700 | 0.7540 | -0.0160 | 0.293 | 7.84e-01 | 6.25e-01 |
| F1  | 0.4559 | 0.4535 | -0.0023 | 0.080 | 9.40e-01 | 1.00e+00 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8003 | 0.7981 | -0.0022 | 0.418 | 6.97e-01 | 6.25e-01 |
| AUPRC  | 0.3446 | 0.3696 | +0.0250 | -1.102 | 3.32e-01 | 4.38e-01 |
| Brier  | 0.1795 | 0.1964 | +0.0170 | -1.344 | 2.50e-01 | 3.12e-01 |
| Accuracy  | 0.7519 | 0.7534 | +0.0015 | -0.045 | 9.66e-01 | 1.00e+00 |
| F1  | 0.4361 | 0.4380 | +0.0019 | -0.125 | 9.06e-01 | 1.00e+00 |

#### Case: len128/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8154 | 0.8189 | +0.0035 | -0.328 | 7.60e-01 | 8.12e-01 |
| AUPRC  | 0.3930 | 0.4015 | +0.0085 | -0.230 | 8.29e-01 | 1.00e+00 |
| Brier  | 0.1763 | 0.2064 | +0.0301 | -1.296 | 2.65e-01 | 3.12e-01 |
| Accuracy  | 0.7308 | 0.7465 | +0.0158 | -0.460 | 6.70e-01 | 6.25e-01 |
| F1  | 0.4358 | 0.4525 | +0.0167 | -0.810 | 4.64e-01 | 6.25e-01 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8097 | 0.7952 | -0.0145 | 1.948 | 1.23e-01 | 1.88e-01 |
| AUPRC  | 0.3598 | 0.3552 | -0.0045 | 0.198 | 8.53e-01 | 6.25e-01 |
| Brier  | 0.1857 | 0.1983 | +0.0126 | -0.755 | 4.92e-01 | 1.00e+00 |
| Accuracy  | 0.7744 | 0.7248 | -0.0496 | 0.916 | 4.12e-01 | 6.25e-01 |
| F1  | 0.4656 | 0.4279 | -0.0376 | 1.003 | 3.72e-01 | 6.25e-01 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8222 | 0.8039 | -0.0183 | 2.652 | 5.69e-02 | 1.25e-01 |
| AUPRC  | 0.4102 | 0.3872 | -0.0230 | 1.335 | 2.53e-01 | 3.12e-01 |
| Brier * | 0.1659 | 0.2252 | +0.0592 | -3.676 | 2.13e-02 | 6.25e-02 |
| Accuracy  | 0.7205 | 0.7278 | +0.0074 | -0.219 | 8.38e-01 | 8.12e-01 |
| F1  | 0.4301 | 0.4387 | +0.0086 | -0.373 | 7.28e-01 | 1.00e+00 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8175 | +0.0062 | -1.093 | 3.36e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.4161 | +0.0314 | -1.468 | 2.16e-01 | 1.88e-01 |
| Brier  | 0.1819 | 0.1672 | -0.0147 | 1.310 | 2.60e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7540 | -0.0015 | 0.025 | 9.81e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4535 | +0.0021 | -0.058 | 9.57e-01 | 8.12e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.7981 | -0.0133 | 0.785 | 4.76e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3696 | -0.0150 | 0.330 | 7.58e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1964 | +0.0145 | -0.860 | 4.38e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7534 | -0.0020 | 0.034 | 9.75e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4380 | -0.0134 | 0.434 | 6.87e-01 | 8.12e-01 |

### len128/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8189 | +0.0076 | -0.651 | 5.50e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.4015 | +0.0169 | -0.647 | 5.53e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.2064 | +0.0245 | -1.161 | 3.10e-01 | 4.38e-01 |
| Accuracy  | 0.7555 | 0.7465 | -0.0089 | 0.149 | 8.88e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4525 | +0.0010 | -0.031 | 9.77e-01 | 1.00e+00 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.7952 | -0.0161 | 1.843 | 1.39e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3552 | -0.0294 | 0.865 | 4.36e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1983 | +0.0164 | -0.879 | 4.29e-01 | 8.12e-01 |
| Accuracy  | 0.7555 | 0.7248 | -0.0306 | 0.567 | 6.01e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4279 | -0.0235 | 0.676 | 5.36e-01 | 6.25e-01 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8039 | -0.0075 | 0.856 | 4.40e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3872 | +0.0025 | -0.081 | 9.39e-01 | 1.00e+00 |
| Brier † | 0.1819 | 0.2252 | +0.0433 | -2.374 | 7.65e-02 | 1.25e-01 |
| Accuracy  | 0.7555 | 0.7278 | -0.0276 | 0.798 | 4.70e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4387 | -0.0127 | 0.548 | 6.13e-01 | 4.38e-01 |

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
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8884 | 0.8084 | 0.9502 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.5643 | 0.3613 | 0.7682 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.1620 | 0.1335 | 0.1924 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7849 | 0.7209 | 0.8430 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.4789 | 0.3235 | 0.6197 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8507 | 0.7701 | 0.9187 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.4750 | 0.2766 | 0.6800 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1614 | 0.1324 | 0.1921 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.6948 | 0.6234 | 0.7662 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4337 | 0.2933 | 0.5618 |
| M2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.8694 | 0.7966 | 0.9297 |
| M2 | CrossAttn | len128/scale_clinic | AUPRC | 0.4530 | 0.2837 | 0.6868 |
| M2 | CrossAttn | len128/scale_clinic | Brier | 0.2276 | 0.1927 | 0.2645 |
| M2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6047 | 0.5291 | 0.6802 |
| M2 | CrossAttn | len128/scale_clinic | F1 | 0.3818 | 0.2592 | 0.4964 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8521 | 0.7649 | 0.9230 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.4949 | 0.2983 | 0.6952 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.2016 | 0.1646 | 0.2379 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6977 | 0.6279 | 0.7674 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.4091 | 0.2703 | 0.5333 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8553 | 0.7676 | 0.9261 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.4906 | 0.2938 | 0.7058 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.1833 | 0.1504 | 0.2169 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.6163 | 0.5407 | 0.6860 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.3774 | 0.2548 | 0.4912 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.7931 | 0.6781 | 0.8878 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4422 | 0.2443 | 0.6472 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.1567 | 0.1229 | 0.1926 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7442 | 0.6744 | 0.8081 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.3714 | 0.2154 | 0.5075 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8546 | 0.7505 | 0.9333 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.4350 | 0.2286 | 0.6550 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.1935 | 0.1598 | 0.2307 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.7922 | 0.7208 | 0.8571 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4483 | 0.2711 | 0.5926 |
| M2_2 | CrossAttn | len128/scale_clinic | AUC-ROC | 0.7887 | 0.6709 | 0.8868 |
| M2_2 | CrossAttn | len128/scale_clinic | AUPRC | 0.3915 | 0.2063 | 0.5895 |
| M2_2 | CrossAttn | len128/scale_clinic | Brier | 0.1946 | 0.1566 | 0.2357 |
| M2_2 | CrossAttn | len128/scale_clinic | Accuracy | 0.6686 | 0.5930 | 0.7384 |
| M2_2 | CrossAttn | len128/scale_clinic | F1 | 0.3736 | 0.2353 | 0.5000 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8051 | 0.6848 | 0.9031 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.3982 | 0.2251 | 0.6092 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.1821 | 0.1479 | 0.2186 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.7151 | 0.6453 | 0.7849 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.4096 | 0.2692 | 0.5429 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8098 | 0.6875 | 0.9054 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.4034 | 0.2319 | 0.6208 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.1933 | 0.1506 | 0.2392 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.7442 | 0.6744 | 0.8081 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.4500 | 0.3077 | 0.5814 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8858 | 0.8102 | 0.9444 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.5225 | 0.3303 | 0.7305 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.1819 | 0.1505 | 0.2145 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.7093 | 0.6395 | 0.7733 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.4186 | 0.2750 | 0.5455 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8801 | 0.7981 | 0.9487 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.6350 | 0.4252 | 0.8316 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.2347 | 0.2016 | 0.2703 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.6104 | 0.5325 | 0.6883 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.4000 | 0.2727 | 0.5149 |
| M3 | CrossAttn3 | len128/scale_clinic | AUC-ROC | 0.8559 | 0.7660 | 0.9297 |
| M3 | CrossAttn3 | len128/scale_clinic | AUPRC | 0.4901 | 0.2995 | 0.7017 |
| M3 | CrossAttn3 | len128/scale_clinic | Brier | 0.1890 | 0.1545 | 0.2258 |
| M3 | CrossAttn3 | len128/scale_clinic | Accuracy | 0.7151 | 0.6453 | 0.7849 |
| M3 | CrossAttn3 | len128/scale_clinic | F1 | 0.4368 | 0.2955 | 0.5634 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8720 | 0.7878 | 0.9400 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.5352 | 0.3319 | 0.7515 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.1940 | 0.1573 | 0.2323 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.6919 | 0.6221 | 0.7616 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.4176 | 0.2826 | 0.5400 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.8644 | 0.7740 | 0.9364 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.5628 | 0.3516 | 0.7669 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.1982 | 0.1621 | 0.2359 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.7326 | 0.6685 | 0.7965 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.4250 | 0.2820 | 0.5600 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-norm | 0.8325 | 0.8884 | +0.0558 | -2.490 | 1.279e-02 | * |
| M1-LR vs M2-len128 | 0.8325 | 0.8694 | +0.0369 | -1.009 | 3.128e-01 | ns |
| M1-LR vs M2-crop80 | 0.8325 | 0.8521 | +0.0196 | -0.697 | 4.859e-01 | ns |
| M1-LR vs M2-crop60 | 0.8325 | 0.8553 | +0.0227 | -0.794 | 4.270e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-norm | 0.8325 | 0.8858 | +0.0533 | -1.864 | 6.235e-02 | † |
| M1-LR vs M3-len128 | 0.8325 | 0.8559 | +0.0233 | -0.851 | 3.946e-01 | ns |
| M1-LR vs M3-crop80 | 0.8325 | 0.8720 | +0.0394 | -1.220 | 2.226e-01 | ns |
| M1-LR vs M3-crop60 | 0.8325 | 0.8644 | +0.0319 | -1.009 | 3.129e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M2_2-norm | 0.8884 | 0.7931 | -0.0952 | 2.825 | 4.732e-03 | ** |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8507 | 0.5474 | -0.3033 | 3.828 | 1.291e-04 | *** |
| M2-len128 vs M2_2-len128 | 0.8694 | 0.7887 | -0.0807 | 2.199 | 2.786e-02 | * |
| M2-crop80 vs M2_2-crop80 | 0.8521 | 0.8051 | -0.0470 | 1.378 | 1.681e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.8553 | 0.8098 | -0.0454 | 1.213 | 2.249e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M3-norm | 0.8884 | 0.8858 | -0.0025 | 0.188 | 8.513e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.8507 | 0.8801 | +0.0294 | -0.976 | 3.290e-01 | ns |
| M2-len128 vs M3-len128 | 0.8694 | 0.8559 | -0.0136 | 0.627 | 5.304e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8521 | 0.8720 | +0.0199 | -1.195 | 2.319e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.8553 | 0.8644 | +0.0091 | -0.528 | 5.975e-01 | ns |

