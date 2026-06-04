# Scaling Comparison — Test Set Performance (AEC 103pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8132 | 0.3347 | 0.2519 | 0.6987 | 0.3551 |
| M2_2 | CrossAttn | norm | 0.8230 | 0.3126 | 0.2615 | 0.6725 | 0.3697 |
| M3 | CrossAttn3 | norm | 0.8006 | 0.2856 | 0.2545 | 0.6943 | 0.3396 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_all** | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |

---

## Model 2 — Clinic + AEC (Matched)  (4 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7868 | 0.2729 | 0.2590 | 0.7031 | 0.3333 |
| std_scaled | 0.7917 | 0.2919 | 0.2601 | 0.7118 | 0.3529 |
| **norm** | 0.8132 | 0.3347 | 0.2519 | 0.6987 | 0.3551 |
| global_zscore | 0.7715 | 0.2457 | 0.2663 | 0.6681 | 0.3333 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7898 | 0.2966 | 0.2625 | 0.6812 | 0.3540 |
| std_scaled | 0.7951 | 0.3353 | 0.2668 | 0.6245 | 0.3175 |
| **norm** | 0.8230 | 0.3126 | 0.2615 | 0.6725 | 0.3697 |
| global_zscore | 0.8041 | 0.3407 | 0.2682 | 0.6332 | 0.3333 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7856 | 0.2728 | 0.2648 | 0.6725 | 0.3243 |
| std_scaled | 0.7917 | 0.3131 | 0.2685 | 0.6725 | 0.3478 |
| **norm** | 0.8006 | 0.2856 | 0.2545 | 0.6943 | 0.3396 |
| global_zscore | 0.7780 | 0.2508 | 0.2621 | 0.7205 | 0.3469 |

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

### raw  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8066 | +0.0005 | -0.029 | 9.78e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3904 | -0.0188 | 0.584 | 5.91e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2491 | +0.0684 | -13.558 | 1.71e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7255 | -0.0306 | 1.215 | 2.91e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.3853 | -0.0310 | 1.870 | 1.35e-01 | 1.88e-01 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8067 | +0.0005 | -0.039 | 9.71e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3562 | -0.0530 | 1.305 | 2.62e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2473 | +0.0665 | -14.816 | 1.21e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7485 | -0.0076 | 0.192 | 8.57e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4090 | -0.0074 | 0.238 | 8.24e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8042 | -0.0020 | 0.259 | 8.08e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3751 | -0.0341 | 1.055 | 3.51e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2525 | +0.0717 | -13.206 | 1.90e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7462 | -0.0099 | 0.585 | 5.90e-01 | 5.62e-01 |
| F1  | 0.4163 | 0.4034 | -0.0129 | 0.748 | 4.96e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8026 | -0.0035 | 0.232 | 8.28e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3681 | -0.0411 | 1.240 | 2.83e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2500 | +0.0692 | -19.618 | 3.98e-05 | 6.25e-02 |
| Accuracy † | 0.7561 | 0.6762 | -0.0799 | 2.285 | 8.43e-02 | 6.25e-02 |
| F1  | 0.4163 | 0.3600 | -0.0564 | 1.787 | 1.49e-01 | 1.88e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8066 | 0.8020 | -0.0046 | 0.480 | 6.56e-01 | 8.12e-01 |
| AUPRC  | 0.3904 | 0.3672 | -0.0232 | 1.526 | 2.02e-01 | 3.12e-01 |
| Brier  | 0.2491 | 0.2591 | +0.0099 | -1.503 | 2.07e-01 | 1.88e-01 |
| Accuracy  | 0.7255 | 0.7167 | -0.0087 | 0.283 | 7.91e-01 | 1.00e+00 |
| F1  | 0.3853 | 0.3842 | -0.0011 | 0.059 | 9.56e-01 | 6.25e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8067 | 0.7997 | -0.0070 | 1.294 | 2.65e-01 | 3.12e-01 |
| AUPRC  | 0.3562 | 0.3575 | +0.0013 | -0.081 | 9.40e-01 | 1.00e+00 |
| Brier  | 0.2473 | 0.2523 | +0.0051 | -1.510 | 2.06e-01 | 1.88e-01 |
| Accuracy  | 0.7485 | 0.6926 | -0.0559 | 1.697 | 1.65e-01 | 3.12e-01 |
| F1  | 0.4090 | 0.3709 | -0.0380 | 1.679 | 1.68e-01 | 3.12e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8042 | 0.8007 | -0.0035 | 1.215 | 2.91e-01 | 3.12e-01 |
| AUPRC  | 0.3751 | 0.3855 | +0.0104 | -0.807 | 4.65e-01 | 8.12e-01 |
| Brier  | 0.2525 | 0.2507 | -0.0018 | 0.553 | 6.10e-01 | 4.38e-01 |
| Accuracy  | 0.7462 | 0.7407 | -0.0055 | 0.219 | 8.38e-01 | 6.25e-01 |
| F1  | 0.4034 | 0.4045 | +0.0011 | -0.060 | 9.55e-01 | 8.12e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8026 | 0.7985 | -0.0042 | 0.532 | 6.23e-01 | 6.25e-01 |
| AUPRC  | 0.3681 | 0.3760 | +0.0079 | -0.531 | 6.23e-01 | 6.25e-01 |
| Brier  | 0.2500 | 0.2482 | -0.0018 | 0.238 | 8.24e-01 | 8.12e-01 |
| Accuracy † | 0.6762 | 0.7713 | +0.0951 | -2.662 | 5.62e-02 | 6.25e-02 |
| F1  | 0.3600 | 0.4275 | +0.0676 | -1.953 | 1.23e-01 | 6.25e-02 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8020 | -0.0042 | 0.196 | 8.54e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3672 | -0.0420 | 0.911 | 4.14e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2591 | +0.0783 | -10.526 | 4.61e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7167 | -0.0393 | 1.069 | 3.45e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3842 | -0.0321 | 1.108 | 3.30e-01 | 4.38e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7997 | -0.0064 | 0.447 | 6.78e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3575 | -0.0517 | 1.243 | 2.82e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2523 | +0.0716 | -16.294 | 8.30e-05 | 6.25e-02 |
| Accuracy † | 0.7561 | 0.6926 | -0.0635 | 2.464 | 6.94e-02 | 1.25e-01 |
| F1  | 0.4163 | 0.3709 | -0.0454 | 1.655 | 1.73e-01 | 2.50e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8007 | -0.0054 | 0.650 | 5.51e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3855 | -0.0237 | 0.641 | 5.56e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2507 | +0.0699 | -21.700 | 2.67e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7407 | -0.0154 | 0.565 | 6.02e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4045 | -0.0119 | 0.454 | 6.74e-01 | 1.00e+00 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7985 | -0.0077 | 0.551 | 6.11e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3760 | -0.0332 | 0.885 | 4.26e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2482 | +0.0674 | -11.011 | 3.87e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7713 | +0.0152 | -0.220 | 8.37e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4275 | +0.0112 | -0.184 | 8.63e-01 | 6.25e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_all | AUC-ROC | 0.8030 | 0.7200 | 0.8724 |
| M1 | LR | scale_all | AUPRC | 0.3123 | 0.1806 | 0.4955 |
| M1 | LR | scale_all | Brier | 0.1913 | 0.1647 | 0.2175 |
| M1 | LR | scale_all | Accuracy | 0.7205 | 0.6638 | 0.7817 |
| M1 | LR | scale_all | F1 | 0.3725 | 0.2500 | 0.4884 |
| M2 | CrossAttn | raw | AUC-ROC | 0.7868 | 0.7042 | 0.8610 |
| M2 | CrossAttn | raw | AUPRC | 0.2729 | 0.1619 | 0.4270 |
| M2 | CrossAttn | raw | Brier | 0.2590 | 0.2318 | 0.2837 |
| M2 | CrossAttn | raw | Accuracy | 0.7031 | 0.6419 | 0.7642 |
| M2 | CrossAttn | raw | F1 | 0.3333 | 0.2157 | 0.4516 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7917 | 0.7047 | 0.8688 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2919 | 0.1706 | 0.4688 |
| M2 | CrossAttn | std_scaled | Brier | 0.2601 | 0.2328 | 0.2859 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7118 | 0.6550 | 0.7687 |
| M2 | CrossAttn | std_scaled | F1 | 0.3529 | 0.2308 | 0.4685 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8132 | 0.7354 | 0.8786 |
| M2 | CrossAttn | norm | AUPRC | 0.3347 | 0.1951 | 0.5090 |
| M2 | CrossAttn | norm | Brier | 0.2519 | 0.2248 | 0.2768 |
| M2 | CrossAttn | norm | Accuracy | 0.6987 | 0.6376 | 0.7598 |
| M2 | CrossAttn | norm | F1 | 0.3551 | 0.2325 | 0.4696 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7715 | 0.6862 | 0.8539 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2457 | 0.1538 | 0.4162 |
| M2 | CrossAttn | global_zscore | Brier | 0.2663 | 0.2403 | 0.2907 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.6681 | 0.6070 | 0.7336 |
| M2 | CrossAttn | global_zscore | F1 | 0.3333 | 0.2157 | 0.4465 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7898 | 0.7086 | 0.8637 |
| M2_2 | CrossAttn | raw | AUPRC | 0.2966 | 0.1756 | 0.4896 |
| M2_2 | CrossAttn | raw | Brier | 0.2625 | 0.2369 | 0.2868 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6812 | 0.6201 | 0.7425 |
| M2_2 | CrossAttn | raw | F1 | 0.3540 | 0.2373 | 0.4678 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7951 | 0.6978 | 0.8747 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3353 | 0.1990 | 0.5226 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2668 | 0.2397 | 0.2938 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6245 | 0.5590 | 0.6900 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3175 | 0.2097 | 0.4252 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8230 | 0.7438 | 0.8886 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3126 | 0.1955 | 0.5093 |
| M2_2 | CrossAttn | norm | Brier | 0.2615 | 0.2352 | 0.2872 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6725 | 0.6112 | 0.7380 |
| M2_2 | CrossAttn | norm | F1 | 0.3697 | 0.2549 | 0.4793 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.8041 | 0.7163 | 0.8795 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3407 | 0.1993 | 0.5282 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2682 | 0.2420 | 0.2931 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6332 | 0.5721 | 0.6944 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3333 | 0.2222 | 0.4429 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7856 | 0.6921 | 0.8720 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2728 | 0.1776 | 0.4546 |
| M3 | CrossAttn3 | raw | Brier | 0.2648 | 0.2398 | 0.2884 |
| M3 | CrossAttn3 | raw | Accuracy | 0.6725 | 0.6157 | 0.7336 |
| M3 | CrossAttn3 | raw | F1 | 0.3243 | 0.2135 | 0.4355 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7917 | 0.7043 | 0.8707 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.3131 | 0.1810 | 0.4923 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2685 | 0.2423 | 0.2936 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.6725 | 0.6114 | 0.7336 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3478 | 0.2342 | 0.4640 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8006 | 0.7137 | 0.8712 |
| M3 | CrossAttn3 | norm | AUPRC | 0.2856 | 0.1759 | 0.4780 |
| M3 | CrossAttn3 | norm | Brier | 0.2545 | 0.2276 | 0.2793 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6943 | 0.6332 | 0.7555 |
| M3 | CrossAttn3 | norm | F1 | 0.3396 | 0.2197 | 0.4500 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7780 | 0.6898 | 0.8593 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2508 | 0.1581 | 0.4225 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2621 | 0.2361 | 0.2872 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7205 | 0.6638 | 0.7817 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3469 | 0.2273 | 0.4681 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7868 | -0.0163 | 0.693 | 4.882e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7917 | -0.0114 | 0.476 | 6.338e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8132 | +0.0102 | -0.685 | 4.933e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7715 | -0.0315 | 1.152 | 2.492e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7856 | -0.0175 | 0.519 | 6.040e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7917 | -0.0114 | 0.404 | 6.863e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8006 | -0.0024 | 0.131 | 8.956e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7780 | -0.0250 | 0.848 | 3.962e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7868 | 0.7898 | +0.0030 | -0.165 | 8.692e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7917 | 0.7951 | +0.0035 | -0.133 | 8.941e-01 | ns |
| M2-norm vs M2_2-norm | 0.8132 | 0.8230 | +0.0098 | -0.734 | 4.630e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.7715 | 0.8041 | +0.0325 | -1.298 | 1.942e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7868 | 0.7856 | -0.0012 | 0.049 | 9.610e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7917 | 0.7917 | +0.0000 | 0.000 | 1.000e+00 | ns |
| M2-norm vs M3-norm | 0.8132 | 0.8006 | -0.0126 | 1.357 | 1.746e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.7715 | 0.7780 | +0.0065 | -0.365 | 7.151e-01 | ns |

