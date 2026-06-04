# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8195 | 0.3314 | 0.2622 | 0.7074 | 0.3495 |
| M2_2 | CrossAttn | norm | 0.8358 | 0.3398 | 0.2620 | 0.6638 | 0.3636 |
| M3 | CrossAttn3 | norm | 0.8108 | 0.2975 | 0.2597 | 0.7380 | 0.3478 |

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
| raw | 0.7882 | 0.2876 | 0.2569 | 0.7380 | 0.3617 |
| std_scaled | 0.7807 | 0.2923 | 0.2656 | 0.6507 | 0.3103 |
| **norm** | 0.8195 | 0.3314 | 0.2622 | 0.7074 | 0.3495 |
| global_zscore | 0.7839 | 0.2879 | 0.2647 | 0.6507 | 0.3443 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7939 | 0.3332 | 0.2661 | 0.6419 | 0.3279 |
| std_scaled | 0.7709 | 0.3528 | 0.2755 | 0.5459 | 0.2973 |
| **norm** | 0.8358 | 0.3398 | 0.2620 | 0.6638 | 0.3636 |
| global_zscore | 0.7951 | 0.3235 | 0.2641 | 0.6812 | 0.3303 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7705 | 0.2409 | 0.2792 | 0.6638 | 0.3304 |
| std_scaled | 0.7555 | 0.2462 | 0.2693 | 0.7118 | 0.3529 |
| **norm** | 0.8108 | 0.2975 | 0.2597 | 0.7380 | 0.3478 |
| global_zscore | 0.7732 | 0.2660 | 0.2628 | 0.7031 | 0.3704 |

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
| AUC-ROC  | 0.8061 | 0.8085 | +0.0024 | -0.142 | 8.94e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3934 | -0.0159 | 0.453 | 6.74e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2557 | +0.0750 | -12.271 | 2.53e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7669 | +0.0108 | -0.301 | 7.79e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4189 | +0.0026 | -0.080 | 9.40e-01 | 1.00e+00 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8071 | +0.0010 | -0.067 | 9.50e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3917 | -0.0175 | 0.395 | 7.13e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2556 | +0.0749 | -20.080 | 3.63e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7133 | -0.0427 | 0.704 | 5.20e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.3901 | -0.0262 | 0.598 | 5.82e-01 | 6.25e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8152 | +0.0091 | -0.815 | 4.61e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4018 | -0.0074 | 0.236 | 8.25e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2476 | +0.0668 | -19.031 | 4.49e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7943 | +0.0382 | -0.824 | 4.56e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4490 | +0.0327 | -0.726 | 5.08e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8004 | -0.0057 | 0.395 | 7.13e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3772 | -0.0320 | 0.823 | 4.57e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2543 | +0.0736 | -16.404 | 8.08e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6904 | -0.0657 | 1.687 | 1.67e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.3803 | -0.0360 | 0.952 | 3.95e-01 | 6.25e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8085 | 0.7765 | -0.0320 | 1.731 | 1.59e-01 | 1.25e-01 |
| AUPRC  | 0.3934 | 0.3382 | -0.0552 | 1.226 | 2.88e-01 | 1.88e-01 |
| Brier  | 0.2557 | 0.2658 | +0.0101 | -1.901 | 1.30e-01 | 1.88e-01 |
| Accuracy  | 0.7669 | 0.6729 | -0.0940 | 1.129 | 3.22e-01 | 3.12e-01 |
| F1  | 0.4189 | 0.3639 | -0.0550 | 1.593 | 1.86e-01 | 1.88e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8071 | 0.7995 | -0.0076 | 0.820 | 4.58e-01 | 6.25e-01 |
| AUPRC  | 0.3917 | 0.4029 | +0.0112 | -0.886 | 4.25e-01 | 4.38e-01 |
| Brier  | 0.2556 | 0.2598 | +0.0042 | -1.057 | 3.50e-01 | 4.38e-01 |
| Accuracy * | 0.7133 | 0.7790 | +0.0657 | -3.219 | 3.23e-02 | 1.25e-01 |
| F1  | 0.3901 | 0.4268 | +0.0366 | -1.583 | 1.89e-01 | 1.88e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8152 | 0.7923 | -0.0229 | 4.069 | 1.52e-02 | 6.25e-02 |
| AUPRC  | 0.4018 | 0.4169 | +0.0151 | -0.747 | 4.96e-01 | 1.00e+00 |
| Brier  | 0.2476 | 0.2581 | +0.0106 | -1.669 | 1.70e-01 | 1.25e-01 |
| Accuracy  | 0.7943 | 0.7649 | -0.0294 | 0.675 | 5.37e-01 | 8.12e-01 |
| F1  | 0.4490 | 0.4162 | -0.0328 | 0.787 | 4.76e-01 | 4.38e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8004 | 0.7974 | -0.0030 | 0.335 | 7.54e-01 | 1.00e+00 |
| AUPRC  | 0.3772 | 0.4053 | +0.0281 | -1.540 | 1.98e-01 | 1.88e-01 |
| Brier † | 0.2543 | 0.2632 | +0.0089 | -2.510 | 6.60e-02 | 1.25e-01 |
| Accuracy  | 0.6904 | 0.6727 | -0.0177 | 0.405 | 7.06e-01 | 7.50e-01 |
| F1  | 0.3803 | 0.3600 | -0.0203 | 0.798 | 4.70e-01 | 8.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7765 | -0.0297 | 0.888 | 4.25e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3382 | -0.0711 | 1.067 | 3.46e-01 | 4.38e-01 |
| Brier ** | 0.1808 | 0.2658 | +0.0851 | -8.188 | 1.21e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6729 | -0.0832 | 1.025 | 3.63e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.3639 | -0.0524 | 1.057 | 3.50e-01 | 4.38e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7995 | -0.0066 | 0.322 | 7.63e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4029 | -0.0063 | 0.138 | 8.97e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2598 | +0.0791 | -17.307 | 6.54e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7790 | +0.0229 | -0.425 | 6.92e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4268 | +0.0104 | -0.204 | 8.48e-01 | 1.00e+00 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7923 | -0.0139 | 1.081 | 3.41e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.4169 | +0.0077 | -0.213 | 8.41e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2581 | +0.0774 | -10.468 | 4.71e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7649 | +0.0088 | -0.180 | 8.66e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4162 | -0.0002 | 0.006 | 9.96e-01 | 1.00e+00 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7974 | -0.0087 | 0.436 | 6.85e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.4053 | -0.0039 | 0.077 | 9.42e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2632 | +0.0824 | -14.530 | 1.30e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6727 | -0.0833 | 1.630 | 1.79e-01 | 1.25e-01 |
| F1  | 0.4163 | 0.3600 | -0.0563 | 1.403 | 2.33e-01 | 3.12e-01 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7882 | 0.6872 | 0.8709 |
| M2 | CrossAttn | raw | AUPRC | 0.2876 | 0.1705 | 0.4440 |
| M2 | CrossAttn | raw | Brier | 0.2569 | 0.2294 | 0.2827 |
| M2 | CrossAttn | raw | Accuracy | 0.7380 | 0.6812 | 0.7948 |
| M2 | CrossAttn | raw | F1 | 0.3617 | 0.2326 | 0.4842 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7807 | 0.6961 | 0.8601 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2923 | 0.1703 | 0.4743 |
| M2 | CrossAttn | std_scaled | Brier | 0.2656 | 0.2408 | 0.2883 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.6507 | 0.5895 | 0.7162 |
| M2 | CrossAttn | std_scaled | F1 | 0.3103 | 0.2018 | 0.4194 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8195 | 0.7402 | 0.8871 |
| M2 | CrossAttn | norm | AUPRC | 0.3314 | 0.1974 | 0.5151 |
| M2 | CrossAttn | norm | Brier | 0.2622 | 0.2356 | 0.2873 |
| M2 | CrossAttn | norm | Accuracy | 0.7074 | 0.6549 | 0.7686 |
| M2 | CrossAttn | norm | F1 | 0.3495 | 0.2292 | 0.4688 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7839 | 0.6946 | 0.8647 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2879 | 0.1694 | 0.4586 |
| M2 | CrossAttn | global_zscore | Brier | 0.2647 | 0.2397 | 0.2878 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.6507 | 0.5895 | 0.7162 |
| M2 | CrossAttn | global_zscore | F1 | 0.3443 | 0.2281 | 0.4496 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7939 | 0.6916 | 0.8762 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3332 | 0.1906 | 0.5221 |
| M2_2 | CrossAttn | raw | Brier | 0.2661 | 0.2396 | 0.2910 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6419 | 0.5808 | 0.7074 |
| M2_2 | CrossAttn | raw | F1 | 0.3279 | 0.2162 | 0.4370 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7709 | 0.6802 | 0.8490 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3528 | 0.1955 | 0.5338 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2755 | 0.2523 | 0.2975 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.5459 | 0.4847 | 0.6115 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.2973 | 0.1969 | 0.3951 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8358 | 0.7709 | 0.8948 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3398 | 0.2116 | 0.5429 |
| M2_2 | CrossAttn | norm | Brier | 0.2620 | 0.2364 | 0.2859 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6638 | 0.6025 | 0.7293 |
| M2_2 | CrossAttn | norm | F1 | 0.3636 | 0.2520 | 0.4746 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7951 | 0.7107 | 0.8687 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3235 | 0.1858 | 0.5069 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2641 | 0.2380 | 0.2892 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6812 | 0.6245 | 0.7424 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3303 | 0.2151 | 0.4423 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7705 | 0.6840 | 0.8560 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2409 | 0.1575 | 0.4059 |
| M3 | CrossAttn3 | raw | Brier | 0.2792 | 0.2570 | 0.3006 |
| M3 | CrossAttn3 | raw | Accuracy | 0.6638 | 0.5983 | 0.7293 |
| M3 | CrossAttn3 | raw | F1 | 0.3304 | 0.2185 | 0.4444 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7555 | 0.6519 | 0.8445 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2462 | 0.1551 | 0.4226 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2693 | 0.2465 | 0.2905 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.7118 | 0.6507 | 0.7729 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3529 | 0.2307 | 0.4717 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8108 | 0.7342 | 0.8763 |
| M3 | CrossAttn3 | norm | AUPRC | 0.2975 | 0.1775 | 0.4696 |
| M3 | CrossAttn3 | norm | Brier | 0.2597 | 0.2343 | 0.2832 |
| M3 | CrossAttn3 | norm | Accuracy | 0.7380 | 0.6812 | 0.7948 |
| M3 | CrossAttn3 | norm | F1 | 0.3478 | 0.2222 | 0.4681 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7732 | 0.6838 | 0.8564 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2660 | 0.1569 | 0.4267 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2628 | 0.2381 | 0.2848 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7031 | 0.6462 | 0.7642 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3704 | 0.2474 | 0.4848 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7882 | -0.0148 | 0.575 | 5.655e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7807 | -0.0224 | 0.808 | 4.189e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8195 | +0.0165 | -0.944 | 3.449e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7839 | -0.0191 | 0.740 | 4.595e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7705 | -0.0325 | 0.887 | 3.750e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7555 | -0.0476 | 1.226 | 2.201e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8108 | +0.0077 | -0.339 | 7.346e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7732 | -0.0299 | 1.108 | 2.680e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7882 | 0.7939 | +0.0057 | -0.334 | 7.384e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7807 | 0.7709 | -0.0098 | 0.370 | 7.113e-01 | ns |
| M2-norm vs M2_2-norm | 0.8195 | 0.8358 | +0.0163 | -0.911 | 3.622e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.7839 | 0.7951 | +0.0112 | -0.533 | 5.939e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7882 | 0.7705 | -0.0177 | 0.533 | 5.939e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7807 | 0.7555 | -0.0252 | 0.823 | 4.105e-01 | ns |
| M2-norm vs M3-norm | 0.8195 | 0.8108 | -0.0087 | 0.505 | 6.137e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.7839 | 0.7732 | -0.0108 | 0.854 | 3.931e-01 | ns |

