# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8307 | 0.3323 | 0.2533 | 0.6812 | 0.3652 |
| M2_2 | CrossAttn | norm | 0.8380 | 0.3518 | 0.2575 | 0.7380 | 0.3878 |
| M3 | CrossAttn3 | norm | 0.7990 | 0.2755 | 0.2543 | 0.6507 | 0.3651 |

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
| raw | 0.7921 | 0.2848 | 0.2625 | 0.6987 | 0.3551 |
| std_scaled | 0.7896 | 0.3028 | 0.2580 | 0.7293 | 0.3922 |
| **norm** | 0.8307 | 0.3323 | 0.2533 | 0.6812 | 0.3652 |
| global_zscore | 0.7831 | 0.2817 | 0.2679 | 0.6638 | 0.3529 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7770 | 0.2836 | 0.2689 | 0.6245 | 0.3281 |
| std_scaled | 0.7703 | 0.2969 | 0.2716 | 0.7118 | 0.3400 |
| **norm** | 0.8380 | 0.3518 | 0.2575 | 0.7380 | 0.3878 |
| global_zscore | 0.7947 | 0.3003 | 0.2714 | 0.6725 | 0.3590 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7791 | 0.2658 | 0.2589 | 0.6812 | 0.3178 |
| std_scaled | 0.7388 | 0.2204 | 0.2759 | 0.6987 | 0.3168 |
| **norm** | 0.7990 | 0.2755 | 0.2543 | 0.6507 | 0.3651 |
| global_zscore | 0.7589 | 0.2431 | 0.2758 | 0.6812 | 0.3178 |

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
| AUC-ROC  | 0.8061 | 0.7985 | -0.0077 | 0.359 | 7.38e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3720 | -0.0372 | 0.770 | 4.84e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2580 | +0.0772 | -11.792 | 2.96e-04 | 6.25e-02 |
| Accuracy * | 0.7561 | 0.6773 | -0.0788 | 4.458 | 1.12e-02 | 6.25e-02 |
| F1  | 0.4163 | 0.3661 | -0.0503 | 2.002 | 1.16e-01 | 6.25e-02 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8107 | +0.0046 | -0.244 | 8.20e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3962 | -0.0130 | 0.360 | 7.37e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2570 | +0.0763 | -13.322 | 1.84e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7507 | -0.0054 | 0.101 | 9.25e-01 | 8.75e-01 |
| F1  | 0.4163 | 0.4279 | +0.0116 | -0.242 | 8.20e-01 | 6.25e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7993 | -0.0068 | 0.403 | 7.07e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3789 | -0.0303 | 1.470 | 2.15e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2510 | +0.0702 | -34.198 | 4.36e-06 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7275 | -0.0285 | 0.722 | 5.10e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3904 | -0.0259 | 0.718 | 5.13e-01 | 8.12e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8065 | +0.0004 | -0.022 | 9.84e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3965 | -0.0127 | 0.361 | 7.37e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2534 | +0.0726 | -24.558 | 1.63e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7035 | -0.0526 | 0.960 | 3.91e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3847 | -0.0316 | 0.692 | 5.27e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.7985 | 0.7875 | -0.0110 | 1.301 | 2.63e-01 | 3.12e-01 |
| AUPRC  | 0.3720 | 0.3423 | -0.0298 | 1.132 | 3.21e-01 | 4.38e-01 |
| Brier  | 0.2580 | 0.2586 | +0.0007 | -0.171 | 8.72e-01 | 1.00e+00 |
| Accuracy  | 0.6773 | 0.7047 | +0.0274 | -0.304 | 7.76e-01 | 1.00e+00 |
| F1  | 0.3661 | 0.3806 | +0.0145 | -0.322 | 7.63e-01 | 6.25e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8107 | 0.7853 | -0.0254 | 1.807 | 1.45e-01 | 1.25e-01 |
| AUPRC  | 0.3962 | 0.3943 | -0.0019 | 0.110 | 9.18e-01 | 6.25e-01 |
| Brier  | 0.2570 | 0.2607 | +0.0037 | -1.287 | 2.67e-01 | 3.12e-01 |
| Accuracy  | 0.7507 | 0.7649 | +0.0142 | -0.395 | 7.13e-01 | 1.00e+00 |
| F1  | 0.4279 | 0.4103 | -0.0176 | 0.472 | 6.62e-01 | 1.00e+00 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.7993 | 0.8000 | +0.0007 | -0.130 | 9.03e-01 | 1.00e+00 |
| AUPRC  | 0.3789 | 0.3551 | -0.0238 | 1.119 | 3.26e-01 | 3.12e-01 |
| Brier  | 0.2510 | 0.2576 | +0.0067 | -0.753 | 4.93e-01 | 6.25e-01 |
| Accuracy  | 0.7275 | 0.6873 | -0.0403 | 0.809 | 4.64e-01 | 6.25e-01 |
| F1  | 0.3904 | 0.3786 | -0.0118 | 0.319 | 7.66e-01 | 6.25e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8065 | 0.7850 | -0.0215 | 3.514 | 2.46e-02 | 6.25e-02 |
| AUPRC  | 0.3965 | 0.3707 | -0.0258 | 0.680 | 5.34e-01 | 6.25e-01 |
| Brier  | 0.2534 | 0.2607 | +0.0073 | -1.137 | 3.19e-01 | 4.38e-01 |
| Accuracy  | 0.7035 | 0.7462 | +0.0427 | -1.290 | 2.66e-01 | 3.12e-01 |
| F1  | 0.3847 | 0.3954 | +0.0107 | -0.507 | 6.39e-01 | 1.00e+00 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7875 | -0.0186 | 0.881 | 4.28e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3423 | -0.0670 | 1.560 | 1.94e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2586 | +0.0779 | -14.388 | 1.36e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7047 | -0.0513 | 0.587 | 5.89e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3806 | -0.0358 | 0.699 | 5.23e-01 | 6.25e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7853 | -0.0208 | 1.179 | 3.04e-01 | 3.12e-01 |
| AUPRC  | 0.4092 | 0.3943 | -0.0149 | 0.385 | 7.20e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2607 | +0.0800 | -13.802 | 1.60e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7649 | +0.0088 | -0.147 | 8.90e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4103 | -0.0060 | 0.144 | 8.92e-01 | 1.00e+00 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8000 | -0.0061 | 0.407 | 7.05e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3551 | -0.0542 | 1.597 | 1.85e-01 | 1.25e-01 |
| Brier *** | 0.1808 | 0.2576 | +0.0769 | -8.970 | 8.55e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6873 | -0.0688 | 1.498 | 2.09e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.3786 | -0.0377 | 1.219 | 2.90e-01 | 4.38e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7850 | -0.0212 | 0.919 | 4.10e-01 | 3.12e-01 |
| AUPRC  | 0.4092 | 0.3707 | -0.0385 | 0.608 | 5.76e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2607 | +0.0799 | -10.520 | 4.62e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7462 | -0.0099 | 0.172 | 8.72e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.3954 | -0.0210 | 0.407 | 7.05e-01 | 8.12e-01 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7921 | 0.7068 | 0.8705 |
| M2 | CrossAttn | raw | AUPRC | 0.2848 | 0.1711 | 0.4469 |
| M2 | CrossAttn | raw | Brier | 0.2625 | 0.2354 | 0.2884 |
| M2 | CrossAttn | raw | Accuracy | 0.6987 | 0.6419 | 0.7598 |
| M2 | CrossAttn | raw | F1 | 0.3551 | 0.2342 | 0.4706 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7896 | 0.6921 | 0.8698 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.3028 | 0.1765 | 0.4862 |
| M2 | CrossAttn | std_scaled | Brier | 0.2580 | 0.2313 | 0.2845 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7293 | 0.6681 | 0.7860 |
| M2 | CrossAttn | std_scaled | F1 | 0.3922 | 0.2653 | 0.5149 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8307 | 0.7600 | 0.8938 |
| M2 | CrossAttn | norm | AUPRC | 0.3323 | 0.2096 | 0.5373 |
| M2 | CrossAttn | norm | Brier | 0.2533 | 0.2271 | 0.2780 |
| M2 | CrossAttn | norm | Accuracy | 0.6812 | 0.6243 | 0.7424 |
| M2 | CrossAttn | norm | F1 | 0.3652 | 0.2453 | 0.4762 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7831 | 0.6967 | 0.8629 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2817 | 0.1658 | 0.4528 |
| M2 | CrossAttn | global_zscore | Brier | 0.2679 | 0.2413 | 0.2923 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.6638 | 0.5983 | 0.7293 |
| M2 | CrossAttn | global_zscore | F1 | 0.3529 | 0.2376 | 0.4681 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7770 | 0.6861 | 0.8525 |
| M2_2 | CrossAttn | raw | AUPRC | 0.2836 | 0.1660 | 0.4717 |
| M2_2 | CrossAttn | raw | Brier | 0.2689 | 0.2417 | 0.2951 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6245 | 0.5633 | 0.6900 |
| M2_2 | CrossAttn | raw | F1 | 0.3281 | 0.2185 | 0.4386 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7703 | 0.6761 | 0.8526 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.2969 | 0.1687 | 0.4798 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2716 | 0.2474 | 0.2946 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.7118 | 0.6507 | 0.7686 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3400 | 0.2222 | 0.4565 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8380 | 0.7643 | 0.9003 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3518 | 0.2168 | 0.5561 |
| M2_2 | CrossAttn | norm | Brier | 0.2575 | 0.2317 | 0.2823 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7380 | 0.6812 | 0.7948 |
| M2_2 | CrossAttn | norm | F1 | 0.3878 | 0.2637 | 0.5102 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7947 | 0.7039 | 0.8687 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3003 | 0.1787 | 0.4950 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2714 | 0.2442 | 0.2979 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6725 | 0.6114 | 0.7380 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3590 | 0.2430 | 0.4724 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7791 | 0.6821 | 0.8666 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2658 | 0.1724 | 0.4461 |
| M3 | CrossAttn3 | raw | Brier | 0.2589 | 0.2329 | 0.2836 |
| M3 | CrossAttn3 | raw | Accuracy | 0.6812 | 0.6243 | 0.7467 |
| M3 | CrossAttn3 | raw | F1 | 0.3178 | 0.2020 | 0.4340 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7388 | 0.6309 | 0.8365 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2204 | 0.1409 | 0.3675 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2759 | 0.2542 | 0.2970 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.6987 | 0.6376 | 0.7598 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3168 | 0.2000 | 0.4444 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.7990 | 0.7196 | 0.8700 |
| M3 | CrossAttn3 | norm | AUPRC | 0.2755 | 0.1720 | 0.4709 |
| M3 | CrossAttn3 | norm | Brier | 0.2543 | 0.2284 | 0.2787 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6507 | 0.5895 | 0.7162 |
| M3 | CrossAttn3 | norm | F1 | 0.3651 | 0.2500 | 0.4769 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7589 | 0.6643 | 0.8457 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2431 | 0.1537 | 0.4223 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2758 | 0.2524 | 0.2973 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.6812 | 0.6245 | 0.7425 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3178 | 0.2041 | 0.4407 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7921 | -0.0110 | 0.472 | 6.371e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7896 | -0.0134 | 0.498 | 6.186e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8307 | +0.0276 | -1.120 | 2.629e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7831 | -0.0199 | 0.754 | 4.509e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7791 | -0.0240 | 0.752 | 4.518e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7388 | -0.0642 | 1.382 | 1.671e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.7990 | -0.0041 | 0.184 | 8.541e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7589 | -0.0441 | 1.364 | 1.724e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7921 | 0.7770 | -0.0150 | 0.746 | 4.557e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7896 | 0.7703 | -0.0193 | 0.746 | 4.559e-01 | ns |
| M2-norm vs M2_2-norm | 0.8307 | 0.8380 | +0.0073 | -0.377 | 7.060e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.7831 | 0.7947 | +0.0116 | -0.573 | 5.663e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7921 | 0.7791 | -0.0130 | 0.676 | 4.988e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7896 | 0.7388 | -0.0508 | 1.573 | 1.158e-01 | ns |
| M2-norm vs M3-norm | 0.8307 | 0.7990 | -0.0317 | 1.857 | 6.327e-02 | † |
| M2-global_zscore vs M3-global_zscore | 0.7831 | 0.7589 | -0.0242 | 1.343 | 1.793e-01 | ns |

