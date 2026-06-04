# Scaling Comparison — Test Set Performance (AEC 103pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8244 | 0.3209 | 0.2521 | 0.7380 | 0.3478 |
| M2_2 | CrossAttn | norm | 0.8504 | 0.3988 | 0.2544 | 0.6725 | 0.3802 |
| M3 | CrossAttn3 | norm | 0.8378 | 0.3088 | 0.2508 | 0.6769 | 0.3729 |

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
| raw | 0.8010 | 0.2675 | 0.2536 | 0.7118 | 0.3654 |
| std_scaled | 0.8057 | 0.2999 | 0.2627 | 0.7118 | 0.3654 |
| **norm** | 0.8244 | 0.3209 | 0.2521 | 0.7380 | 0.3478 |
| global_zscore | 0.8083 | 0.2747 | 0.2603 | 0.6638 | 0.3636 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7809 | 0.2938 | 0.2656 | 0.6507 | 0.3333 |
| std_scaled | 0.7892 | 0.3120 | 0.2613 | 0.6900 | 0.3486 |
| **norm** | 0.8504 | 0.3988 | 0.2544 | 0.6725 | 0.3802 |
| global_zscore | 0.7789 | 0.2973 | 0.2648 | 0.6114 | 0.3206 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7852 | 0.2859 | 0.2659 | 0.7249 | 0.3636 |
| std_scaled | 0.7868 | 0.2657 | 0.2595 | 0.7773 | 0.4000 |
| **norm** | 0.8378 | 0.3088 | 0.2508 | 0.6769 | 0.3729 |
| global_zscore | 0.7996 | 0.2517 | 0.2587 | 0.6463 | 0.3415 |

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
| AUC-ROC  | 0.8061 | 0.8076 | +0.0014 | -0.076 | 9.43e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3809 | -0.0283 | 0.785 | 4.76e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2410 | +0.0603 | -16.666 | 7.59e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7079 | -0.0482 | 1.279 | 2.70e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.3861 | -0.0302 | 0.930 | 4.05e-01 | 8.12e-01 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8148 | +0.0086 | -0.529 | 6.24e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4005 | -0.0087 | 0.252 | 8.13e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2430 | +0.0622 | -9.048 | 8.27e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7517 | -0.0044 | 0.226 | 8.33e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4160 | -0.0004 | 0.016 | 9.88e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7980 | -0.0082 | 0.556 | 6.08e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3717 | -0.0376 | 1.288 | 2.67e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2436 | +0.0629 | -9.438 | 7.03e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7537 | -0.0024 | 0.042 | 9.69e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.3993 | -0.0171 | 0.374 | 7.28e-01 | 1.00e+00 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8100 | +0.0039 | -0.201 | 8.51e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3810 | -0.0282 | 0.759 | 4.90e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2397 | +0.0590 | -15.730 | 9.54e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7210 | -0.0351 | 0.743 | 4.99e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3952 | -0.0212 | 0.529 | 6.25e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8076 | 0.8077 | +0.0001 | -0.020 | 9.85e-01 | 8.12e-01 |
| AUPRC  | 0.3809 | 0.3775 | -0.0034 | 0.185 | 8.62e-01 | 8.12e-01 |
| Brier  | 0.2410 | 0.2411 | +0.0001 | -0.034 | 9.74e-01 | 1.00e+00 |
| Accuracy † | 0.7079 | 0.7878 | +0.0799 | -2.273 | 8.55e-02 | 1.25e-01 |
| F1  | 0.3861 | 0.4304 | +0.0443 | -1.630 | 1.79e-01 | 3.12e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8148 | 0.8084 | -0.0064 | 0.634 | 5.60e-01 | 6.25e-01 |
| AUPRC  | 0.4005 | 0.3886 | -0.0120 | 0.465 | 6.66e-01 | 6.25e-01 |
| Brier  | 0.2430 | 0.2432 | +0.0003 | -0.038 | 9.72e-01 | 1.00e+00 |
| Accuracy  | 0.7517 | 0.7964 | +0.0447 | -1.092 | 3.36e-01 | 4.38e-01 |
| F1  | 0.4160 | 0.4387 | +0.0227 | -0.735 | 5.03e-01 | 8.12e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.7980 | 0.8144 | +0.0165 | -2.099 | 1.04e-01 | 1.88e-01 |
| AUPRC  | 0.3717 | 0.3842 | +0.0125 | -0.414 | 7.00e-01 | 8.12e-01 |
| Brier  | 0.2436 | 0.2414 | -0.0022 | 0.266 | 8.03e-01 | 1.00e+00 |
| Accuracy  | 0.7537 | 0.7571 | +0.0034 | -0.083 | 9.38e-01 | 8.12e-01 |
| F1  | 0.3993 | 0.4184 | +0.0192 | -0.800 | 4.68e-01 | 4.38e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8100 | 0.8154 | +0.0054 | -1.551 | 1.96e-01 | 1.88e-01 |
| AUPRC * | 0.3810 | 0.4011 | +0.0201 | -3.513 | 2.46e-02 | 6.25e-02 |
| Brier  | 0.2397 | 0.2319 | -0.0078 | 0.829 | 4.54e-01 | 6.25e-01 |
| Accuracy  | 0.7210 | 0.7297 | +0.0087 | -0.254 | 8.12e-01 | 1.00e+00 |
| F1  | 0.3952 | 0.4024 | +0.0072 | -0.345 | 7.47e-01 | 1.00e+00 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8077 | +0.0015 | -0.078 | 9.41e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3775 | -0.0317 | 0.645 | 5.54e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2411 | +0.0604 | -22.588 | 2.27e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7878 | +0.0317 | -0.768 | 4.86e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4304 | +0.0141 | -0.385 | 7.20e-01 | 1.00e+00 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8084 | +0.0022 | -0.223 | 8.35e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3886 | -0.0207 | 0.501 | 6.43e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2432 | +0.0625 | -11.950 | 2.81e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7964 | +0.0403 | -0.890 | 4.24e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4387 | +0.0224 | -0.644 | 5.55e-01 | 4.38e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8144 | +0.0083 | -0.463 | 6.67e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3842 | -0.0251 | 0.627 | 5.65e-01 | 8.12e-01 |
| Brier ** | 0.1808 | 0.2414 | +0.0606 | -6.322 | 3.20e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7571 | +0.0010 | -0.036 | 9.73e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4184 | +0.0021 | -0.068 | 9.49e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8154 | +0.0093 | -0.482 | 6.55e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.4011 | -0.0081 | 0.198 | 8.53e-01 | 1.00e+00 |
| Brier ** | 0.1808 | 0.2319 | +0.0512 | -5.216 | 6.44e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7297 | -0.0264 | 0.459 | 6.70e-01 | 8.75e-01 |
| F1  | 0.4163 | 0.4024 | -0.0140 | 0.283 | 7.91e-01 | 6.25e-01 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.8010 | 0.7212 | 0.8757 |
| M2 | CrossAttn | raw | AUPRC | 0.2675 | 0.1736 | 0.4422 |
| M2 | CrossAttn | raw | Brier | 0.2536 | 0.2262 | 0.2798 |
| M2 | CrossAttn | raw | Accuracy | 0.7118 | 0.6507 | 0.7729 |
| M2 | CrossAttn | raw | F1 | 0.3654 | 0.2424 | 0.4822 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.8057 | 0.7197 | 0.8787 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2999 | 0.1789 | 0.4746 |
| M2 | CrossAttn | std_scaled | Brier | 0.2627 | 0.2336 | 0.2894 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7118 | 0.6550 | 0.7729 |
| M2 | CrossAttn | std_scaled | F1 | 0.3654 | 0.2449 | 0.4865 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8244 | 0.7571 | 0.8852 |
| M2 | CrossAttn | norm | AUPRC | 0.3209 | 0.1945 | 0.5018 |
| M2 | CrossAttn | norm | Brier | 0.2521 | 0.2252 | 0.2784 |
| M2 | CrossAttn | norm | Accuracy | 0.7380 | 0.6856 | 0.7948 |
| M2 | CrossAttn | norm | F1 | 0.3478 | 0.2253 | 0.4727 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.8083 | 0.7257 | 0.8823 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2747 | 0.1790 | 0.4454 |
| M2 | CrossAttn | global_zscore | Brier | 0.2603 | 0.2323 | 0.2869 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.6638 | 0.6026 | 0.7250 |
| M2 | CrossAttn | global_zscore | F1 | 0.3636 | 0.2472 | 0.4762 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7809 | 0.6946 | 0.8575 |
| M2_2 | CrossAttn | raw | AUPRC | 0.2938 | 0.1674 | 0.4692 |
| M2_2 | CrossAttn | raw | Brier | 0.2656 | 0.2382 | 0.2922 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6507 | 0.5895 | 0.7118 |
| M2_2 | CrossAttn | raw | F1 | 0.3333 | 0.2222 | 0.4429 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7892 | 0.6967 | 0.8689 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3120 | 0.1868 | 0.4930 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2613 | 0.2347 | 0.2880 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6900 | 0.6288 | 0.7511 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3486 | 0.2316 | 0.4603 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8504 | 0.7760 | 0.9116 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3988 | 0.2449 | 0.5949 |
| M2_2 | CrossAttn | norm | Brier | 0.2544 | 0.2284 | 0.2801 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6725 | 0.6070 | 0.7380 |
| M2_2 | CrossAttn | norm | F1 | 0.3802 | 0.2636 | 0.4918 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7789 | 0.6849 | 0.8588 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.2973 | 0.1718 | 0.4782 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2648 | 0.2387 | 0.2903 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6114 | 0.5502 | 0.6725 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3206 | 0.2121 | 0.4314 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7852 | 0.6969 | 0.8660 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2859 | 0.1707 | 0.4583 |
| M3 | CrossAttn3 | raw | Brier | 0.2659 | 0.2388 | 0.2908 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7249 | 0.6681 | 0.7860 |
| M3 | CrossAttn3 | raw | F1 | 0.3636 | 0.2391 | 0.4828 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7868 | 0.6914 | 0.8682 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2657 | 0.1704 | 0.4434 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2595 | 0.2302 | 0.2871 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.7773 | 0.7249 | 0.8297 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.4000 | 0.2632 | 0.5306 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8378 | 0.7695 | 0.8974 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3088 | 0.1998 | 0.4998 |
| M3 | CrossAttn3 | norm | Brier | 0.2508 | 0.2241 | 0.2756 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6769 | 0.6157 | 0.7380 |
| M3 | CrossAttn3 | norm | F1 | 0.3729 | 0.2560 | 0.4849 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7996 | 0.7054 | 0.8774 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2517 | 0.1671 | 0.3996 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2587 | 0.2318 | 0.2838 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.6463 | 0.5852 | 0.7074 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3415 | 0.2264 | 0.4496 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.8010 | -0.0020 | 0.086 | 9.314e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.8057 | +0.0026 | -0.112 | 9.111e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8244 | +0.0213 | -1.163 | 2.447e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.8083 | +0.0053 | -0.222 | 8.240e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7852 | -0.0179 | 0.604 | 5.459e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7868 | -0.0163 | 0.518 | 6.042e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8378 | +0.0348 | -1.765 | 7.754e-02 | † |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7996 | -0.0035 | 0.133 | 8.943e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.8010 | 0.7809 | -0.0201 | 1.010 | 3.125e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.8057 | 0.7892 | -0.0165 | 0.655 | 5.127e-01 | ns |
| M2-norm vs M2_2-norm | 0.8244 | 0.8504 | +0.0260 | -1.272 | 2.034e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.8083 | 0.7789 | -0.0295 | 1.330 | 1.836e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.8010 | 0.7852 | -0.0159 | 1.059 | 2.897e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.8057 | 0.7868 | -0.0189 | 1.339 | 1.806e-01 | ns |
| M2-norm vs M3-norm | 0.8244 | 0.8378 | +0.0134 | -0.896 | 3.700e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.8083 | 0.7996 | -0.0087 | 0.870 | 3.843e-01 | ns |

