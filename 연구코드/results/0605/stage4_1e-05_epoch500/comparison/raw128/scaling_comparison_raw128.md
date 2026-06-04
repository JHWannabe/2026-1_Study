# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | global_zscore | 0.8148 | 0.3054 | 0.2411 | 0.6638 | 0.3529 |
| M2_2 | CrossAttn | norm | 0.8382 | 0.3448 | 0.2491 | 0.6681 | 0.3667 |
| M3 | CrossAttn3 | global_zscore | 0.7671 | 0.2641 | 0.2696 | 0.6943 | 0.3269 |

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
| raw | 0.7921 | 0.2919 | 0.2424 | 0.7467 | 0.3830 |
| std_scaled | 0.7799 | 0.2706 | 0.2529 | 0.6769 | 0.3393 |
| norm | 0.8033 | 0.2864 | 0.2462 | 0.6900 | 0.3717 |
| **global_zscore** | 0.8148 | 0.3054 | 0.2411 | 0.6638 | 0.3529 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7988 | 0.3352 | 0.2483 | 0.6856 | 0.3684 |
| std_scaled | 0.7703 | 0.2841 | 0.2516 | 0.6201 | 0.3150 |
| **norm** | 0.8382 | 0.3448 | 0.2491 | 0.6681 | 0.3667 |
| global_zscore | 0.7762 | 0.3181 | 0.2607 | 0.5895 | 0.3188 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7496 | 0.2333 | 0.2621 | 0.7467 | 0.3409 |
| std_scaled | 0.7439 | 0.2251 | 0.2687 | 0.6987 | 0.3168 |
| norm | 0.7508 | 0.2714 | 0.2678 | 0.6900 | 0.3238 |
| **global_zscore** | 0.7671 | 0.2641 | 0.2696 | 0.6943 | 0.3269 |

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
| AUC-ROC  | 0.8061 | 0.8111 | +0.0049 | -0.263 | 8.05e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3881 | -0.0212 | 0.594 | 5.84e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2410 | +0.0603 | -8.863 | 8.95e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7385 | -0.0176 | 0.743 | 4.99e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4083 | -0.0080 | 0.332 | 7.57e-01 | 1.00e+00 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8020 | -0.0041 | 0.271 | 8.00e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3618 | -0.0475 | 1.243 | 2.82e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2416 | +0.0609 | -11.690 | 3.06e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7681 | +0.0121 | -0.189 | 8.59e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4291 | +0.0127 | -0.263 | 8.05e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8003 | -0.0058 | 0.277 | 7.95e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3912 | -0.0180 | 0.438 | 6.84e-01 | 8.12e-01 |
| Brier ** | 0.1808 | 0.2450 | +0.0642 | -5.836 | 4.30e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7143 | -0.0418 | 0.700 | 5.22e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.3925 | -0.0238 | 0.607 | 5.76e-01 | 4.38e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8019 | -0.0042 | 0.238 | 8.23e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3840 | -0.0253 | 0.627 | 5.65e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2417 | +0.0610 | -31.288 | 6.22e-06 | 6.25e-02 |
| Accuracy † | 0.7561 | 0.6796 | -0.0765 | 2.487 | 6.77e-02 | 6.25e-02 |
| F1  | 0.4163 | 0.3658 | -0.0506 | 1.991 | 1.17e-01 | 1.88e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8111 | 0.7639 | -0.0472 | 2.448 | 7.06e-02 | 1.25e-01 |
| AUPRC  | 0.3881 | 0.3381 | -0.0499 | 0.984 | 3.81e-01 | 4.38e-01 |
| Brier  | 0.2410 | 0.2548 | +0.0137 | -2.099 | 1.04e-01 | 1.25e-01 |
| Accuracy  | 0.7385 | 0.7233 | -0.0152 | 0.152 | 8.86e-01 | 8.12e-01 |
| F1  | 0.4083 | 0.3765 | -0.0318 | 0.699 | 5.23e-01 | 6.25e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8020 | 0.7940 | -0.0080 | 0.381 | 7.23e-01 | 8.12e-01 |
| AUPRC  | 0.3618 | 0.3888 | +0.0270 | -1.669 | 1.71e-01 | 3.12e-01 |
| Brier  | 0.2416 | 0.2492 | +0.0076 | -1.606 | 1.84e-01 | 3.12e-01 |
| Accuracy  | 0.7681 | 0.7692 | +0.0010 | -0.025 | 9.81e-01 | 7.50e-01 |
| F1  | 0.4291 | 0.4016 | -0.0275 | 0.857 | 4.40e-01 | 6.25e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8003 | 0.8041 | +0.0038 | -0.332 | 7.57e-01 | 6.25e-01 |
| AUPRC  | 0.3912 | 0.3826 | -0.0086 | 0.364 | 7.34e-01 | 1.00e+00 |
| Brier  | 0.2450 | 0.2459 | +0.0010 | -0.084 | 9.37e-01 | 8.12e-01 |
| Accuracy  | 0.7143 | 0.7222 | +0.0079 | -0.093 | 9.30e-01 | 1.00e+00 |
| F1  | 0.3925 | 0.4034 | +0.0108 | -0.277 | 7.96e-01 | 1.00e+00 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8019 | 0.7922 | -0.0097 | 0.867 | 4.35e-01 | 4.38e-01 |
| AUPRC  | 0.3840 | 0.3564 | -0.0276 | 0.843 | 4.47e-01 | 8.12e-01 |
| Brier † | 0.2417 | 0.2501 | +0.0084 | -2.339 | 7.94e-02 | 1.25e-01 |
| Accuracy † | 0.6796 | 0.7638 | +0.0842 | -2.164 | 9.65e-02 | 6.25e-02 |
| F1 † | 0.3658 | 0.4145 | +0.0487 | -2.543 | 6.37e-02 | 6.25e-02 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7639 | -0.0423 | 1.124 | 3.24e-01 | 3.75e-01 |
| AUPRC  | 0.4092 | 0.3381 | -0.0711 | 0.963 | 3.90e-01 | 4.38e-01 |
| Brier ** | 0.1808 | 0.2548 | +0.0740 | -6.984 | 2.21e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7233 | -0.0328 | 0.366 | 7.33e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3765 | -0.0399 | 0.753 | 4.93e-01 | 6.25e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7940 | -0.0121 | 0.558 | 6.07e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3888 | -0.0204 | 0.418 | 6.97e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2492 | +0.0684 | -18.802 | 4.71e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7692 | +0.0131 | -0.213 | 8.42e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4016 | -0.0148 | 0.303 | 7.77e-01 | 1.00e+00 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8041 | -0.0020 | 0.138 | 8.97e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3826 | -0.0266 | 0.740 | 5.01e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2459 | +0.0652 | -12.074 | 2.70e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7222 | -0.0339 | 1.168 | 3.08e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4034 | -0.0130 | 0.943 | 3.99e-01 | 4.38e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7922 | -0.0139 | 0.607 | 5.77e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3564 | -0.0528 | 0.845 | 4.46e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2501 | +0.0694 | -13.087 | 1.97e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7638 | +0.0077 | -0.152 | 8.87e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4145 | -0.0018 | 0.050 | 9.62e-01 | 1.00e+00 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7921 | 0.6948 | 0.8741 |
| M2 | CrossAttn | raw | AUPRC | 0.2919 | 0.1749 | 0.4538 |
| M2 | CrossAttn | raw | Brier | 0.2424 | 0.2157 | 0.2678 |
| M2 | CrossAttn | raw | Accuracy | 0.7467 | 0.6900 | 0.8035 |
| M2 | CrossAttn | raw | F1 | 0.3830 | 0.2558 | 0.5051 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7799 | 0.6880 | 0.8605 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2706 | 0.1593 | 0.4307 |
| M2 | CrossAttn | std_scaled | Brier | 0.2529 | 0.2285 | 0.2759 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.6769 | 0.6157 | 0.7380 |
| M2 | CrossAttn | std_scaled | F1 | 0.3393 | 0.2245 | 0.4500 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8033 | 0.7152 | 0.8759 |
| M2 | CrossAttn | norm | AUPRC | 0.2864 | 0.1813 | 0.4743 |
| M2 | CrossAttn | norm | Brier | 0.2462 | 0.2206 | 0.2705 |
| M2 | CrossAttn | norm | Accuracy | 0.6900 | 0.6332 | 0.7511 |
| M2 | CrossAttn | norm | F1 | 0.3717 | 0.2523 | 0.4874 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.8148 | 0.7367 | 0.8863 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.3054 | 0.1883 | 0.4781 |
| M2 | CrossAttn | global_zscore | Brier | 0.2411 | 0.2161 | 0.2646 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.6638 | 0.6026 | 0.7293 |
| M2 | CrossAttn | global_zscore | F1 | 0.3529 | 0.2373 | 0.4615 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7988 | 0.6992 | 0.8807 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3352 | 0.1937 | 0.5253 |
| M2_2 | CrossAttn | raw | Brier | 0.2483 | 0.2217 | 0.2734 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6856 | 0.6245 | 0.7467 |
| M2_2 | CrossAttn | raw | F1 | 0.3684 | 0.2456 | 0.4822 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7703 | 0.6832 | 0.8491 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.2841 | 0.1609 | 0.4615 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2516 | 0.2262 | 0.2764 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6201 | 0.5588 | 0.6856 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3150 | 0.2051 | 0.4218 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8382 | 0.7560 | 0.9025 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3448 | 0.2154 | 0.5455 |
| M2_2 | CrossAttn | norm | Brier | 0.2491 | 0.2224 | 0.2749 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6681 | 0.6070 | 0.7293 |
| M2_2 | CrossAttn | norm | F1 | 0.3667 | 0.2478 | 0.4818 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7762 | 0.6727 | 0.8590 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3181 | 0.1812 | 0.5019 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2607 | 0.2356 | 0.2847 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.5895 | 0.5240 | 0.6550 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3188 | 0.2131 | 0.4255 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7496 | 0.6605 | 0.8390 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2333 | 0.1460 | 0.4037 |
| M3 | CrossAttn3 | raw | Brier | 0.2621 | 0.2428 | 0.2804 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7467 | 0.6856 | 0.8035 |
| M3 | CrossAttn3 | raw | F1 | 0.3409 | 0.2118 | 0.4694 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7439 | 0.6390 | 0.8401 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2251 | 0.1412 | 0.3716 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2687 | 0.2443 | 0.2920 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.6987 | 0.6376 | 0.7556 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3168 | 0.1975 | 0.4400 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.7508 | 0.6568 | 0.8396 |
| M3 | CrossAttn3 | norm | AUPRC | 0.2714 | 0.1512 | 0.4495 |
| M3 | CrossAttn3 | norm | Brier | 0.2678 | 0.2484 | 0.2874 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6900 | 0.6288 | 0.7468 |
| M3 | CrossAttn3 | norm | F1 | 0.3238 | 0.2083 | 0.4381 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7671 | 0.6777 | 0.8566 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2641 | 0.1625 | 0.4526 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2696 | 0.2486 | 0.2891 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.6943 | 0.6374 | 0.7555 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3269 | 0.2105 | 0.4425 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7921 | -0.0110 | 0.416 | 6.773e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7799 | -0.0232 | 0.881 | 3.782e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8033 | +0.0002 | -0.011 | 9.915e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.8148 | +0.0118 | -0.431 | 6.667e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7496 | -0.0535 | 1.290 | 1.970e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7439 | -0.0591 | 1.473 | 1.407e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.7508 | -0.0522 | 1.487 | 1.371e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7671 | -0.0360 | 0.975 | 3.296e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7921 | 0.7988 | +0.0067 | -0.391 | 6.960e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7799 | 0.7703 | -0.0096 | 0.450 | 6.524e-01 | ns |
| M2-norm vs M2_2-norm | 0.8033 | 0.8382 | +0.0350 | -1.926 | 5.405e-02 | † |
| M2-global_zscore vs M2_2-global_zscore | 0.8148 | 0.7762 | -0.0386 | 1.284 | 1.990e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7921 | 0.7496 | -0.0425 | 1.109 | 2.675e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7799 | 0.7439 | -0.0360 | 1.549 | 1.215e-01 | ns |
| M2-norm vs M3-norm | 0.8033 | 0.7508 | -0.0524 | 1.863 | 6.246e-02 | † |
| M2-global_zscore vs M3-global_zscore | 0.8148 | 0.7671 | -0.0478 | 2.142 | 3.216e-02 | * |

