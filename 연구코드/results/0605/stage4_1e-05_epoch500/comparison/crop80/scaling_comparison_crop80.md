# Scaling Comparison — Test Set Performance (AEC 103pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8108 | 0.2962 | 0.2535 | 0.6594 | 0.3607 |
| M2_2 | CrossAttn | global_zscore | 0.8124 | 0.3579 | 0.2512 | 0.6332 | 0.3438 |
| M3 | CrossAttn3 | norm | 0.8435 | 0.3738 | 0.2361 | 0.7249 | 0.3762 |

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
| raw | 0.7880 | 0.2853 | 0.2399 | 0.7424 | 0.3516 |
| std_scaled | 0.7907 | 0.2800 | 0.2516 | 0.7074 | 0.3738 |
| **norm** | 0.8108 | 0.2962 | 0.2535 | 0.6594 | 0.3607 |
| global_zscore | 0.7848 | 0.2793 | 0.2470 | 0.7511 | 0.3596 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.8008 | 0.3401 | 0.2459 | 0.6332 | 0.3333 |
| std_scaled | 0.8049 | 0.3236 | 0.2531 | 0.6288 | 0.3411 |
| norm | 0.8016 | 0.3314 | 0.2695 | 0.6376 | 0.3360 |
| **global_zscore** | 0.8124 | 0.3579 | 0.2512 | 0.6332 | 0.3438 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7409 | 0.2323 | 0.2675 | 0.5109 | 0.2821 |
| std_scaled | 0.7384 | 0.2137 | 0.2655 | 0.6900 | 0.3107 |
| **norm** | 0.8435 | 0.3738 | 0.2361 | 0.7249 | 0.3762 |
| global_zscore | 0.7866 | 0.2818 | 0.2555 | 0.7162 | 0.3564 |

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
| AUC-ROC  | 0.8061 | 0.8082 | +0.0021 | -0.106 | 9.21e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3813 | -0.0280 | 0.729 | 5.06e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2428 | +0.0621 | -9.254 | 7.58e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7659 | +0.0098 | -0.395 | 7.13e-01 | 5.62e-01 |
| F1  | 0.4163 | 0.4191 | +0.0027 | -0.108 | 9.19e-01 | 1.00e+00 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8050 | -0.0012 | 0.073 | 9.45e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3676 | -0.0417 | 0.916 | 4.11e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2397 | +0.0590 | -10.090 | 5.43e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7320 | -0.0240 | 0.574 | 5.97e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4049 | -0.0115 | 0.396 | 7.12e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8086 | +0.0024 | -0.230 | 8.29e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3881 | -0.0212 | 0.564 | 6.03e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2383 | +0.0576 | -26.682 | 1.17e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7188 | -0.0373 | 1.281 | 2.69e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.3877 | -0.0286 | 0.963 | 3.90e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8050 | -0.0011 | 0.061 | 9.55e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.4000 | -0.0092 | 0.224 | 8.34e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2374 | +0.0567 | -12.374 | 2.45e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7418 | -0.0143 | 0.395 | 7.13e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4016 | -0.0147 | 0.462 | 6.68e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8082 | 0.7603 | -0.0479 | 2.500 | 6.68e-02 | 1.25e-01 |
| AUPRC  | 0.3813 | 0.3078 | -0.0734 | 1.375 | 2.41e-01 | 3.12e-01 |
| Brier  | 0.2428 | 0.2447 | +0.0019 | -0.214 | 8.41e-01 | 6.25e-01 |
| Accuracy  | 0.7659 | 0.6707 | -0.0951 | 1.092 | 3.36e-01 | 4.38e-01 |
| F1  | 0.4191 | 0.3629 | -0.0561 | 1.373 | 2.42e-01 | 3.12e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8050 | 0.7857 | -0.0193 | 1.602 | 1.84e-01 | 3.12e-01 |
| AUPRC  | 0.3676 | 0.3584 | -0.0092 | 0.535 | 6.21e-01 | 4.38e-01 |
| Brier  | 0.2397 | 0.2457 | +0.0060 | -1.144 | 3.16e-01 | 4.38e-01 |
| Accuracy  | 0.7320 | 0.7812 | +0.0492 | -0.705 | 5.20e-01 | 6.25e-01 |
| F1  | 0.4049 | 0.4250 | +0.0201 | -0.406 | 7.05e-01 | 1.00e+00 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8086 | 0.7975 | -0.0111 | 1.740 | 1.57e-01 | 1.88e-01 |
| AUPRC  | 0.3881 | 0.3601 | -0.0279 | 1.495 | 2.09e-01 | 4.38e-01 |
| Brier  | 0.2383 | 0.2452 | +0.0068 | -1.775 | 1.51e-01 | 1.88e-01 |
| Accuracy  | 0.7188 | 0.7386 | +0.0198 | -0.642 | 5.56e-01 | 6.25e-01 |
| F1  | 0.3877 | 0.3943 | +0.0065 | -0.336 | 7.54e-01 | 8.12e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8050 | 0.7951 | -0.0099 | 1.285 | 2.68e-01 | 3.12e-01 |
| AUPRC  | 0.4000 | 0.3677 | -0.0323 | 2.076 | 1.06e-01 | 1.25e-01 |
| Brier  | 0.2374 | 0.2445 | +0.0071 | -1.772 | 1.51e-01 | 1.88e-01 |
| Accuracy  | 0.7418 | 0.7834 | +0.0416 | -0.690 | 5.28e-01 | 1.00e+00 |
| F1  | 0.4016 | 0.4154 | +0.0138 | -0.412 | 7.01e-01 | 1.00e+00 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7603 | -0.0459 | 1.203 | 2.95e-01 | 3.12e-01 |
| AUPRC  | 0.4092 | 0.3078 | -0.1014 | 1.254 | 2.78e-01 | 4.38e-01 |
| Brier ** | 0.1808 | 0.2447 | +0.0639 | -5.139 | 6.80e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6707 | -0.0854 | 1.017 | 3.67e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.3629 | -0.0534 | 1.059 | 3.49e-01 | 4.38e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7857 | -0.0205 | 1.313 | 2.59e-01 | 1.88e-01 |
| AUPRC  | 0.4092 | 0.3584 | -0.0508 | 0.990 | 3.78e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2457 | +0.0650 | -12.288 | 2.52e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7812 | +0.0252 | -0.402 | 7.08e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4250 | +0.0086 | -0.150 | 8.88e-01 | 1.00e+00 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7975 | -0.0086 | 0.871 | 4.33e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3601 | -0.0491 | 1.722 | 1.60e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2452 | +0.0644 | -11.274 | 3.53e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7386 | -0.0175 | 0.426 | 6.92e-01 | 8.75e-01 |
| F1  | 0.4163 | 0.3943 | -0.0221 | 0.831 | 4.53e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7951 | -0.0110 | 0.611 | 5.74e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3677 | -0.0416 | 0.847 | 4.45e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2445 | +0.0637 | -21.220 | 2.92e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7834 | +0.0273 | -0.549 | 6.12e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4154 | -0.0009 | 0.024 | 9.82e-01 | 1.00e+00 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7880 | 0.6897 | 0.8706 |
| M2 | CrossAttn | raw | AUPRC | 0.2853 | 0.1702 | 0.4423 |
| M2 | CrossAttn | raw | Brier | 0.2399 | 0.2131 | 0.2651 |
| M2 | CrossAttn | raw | Accuracy | 0.7424 | 0.6856 | 0.7991 |
| M2 | CrossAttn | raw | F1 | 0.3516 | 0.2222 | 0.4773 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7907 | 0.6976 | 0.8661 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2800 | 0.1674 | 0.4427 |
| M2 | CrossAttn | std_scaled | Brier | 0.2516 | 0.2278 | 0.2747 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7074 | 0.6463 | 0.7643 |
| M2 | CrossAttn | std_scaled | F1 | 0.3738 | 0.2553 | 0.4909 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8108 | 0.7251 | 0.8807 |
| M2 | CrossAttn | norm | AUPRC | 0.2962 | 0.1886 | 0.4812 |
| M2 | CrossAttn | norm | Brier | 0.2535 | 0.2276 | 0.2774 |
| M2 | CrossAttn | norm | Accuracy | 0.6594 | 0.5983 | 0.7249 |
| M2 | CrossAttn | norm | F1 | 0.3607 | 0.2453 | 0.4727 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7848 | 0.6985 | 0.8616 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2793 | 0.1647 | 0.4463 |
| M2 | CrossAttn | global_zscore | Brier | 0.2470 | 0.2230 | 0.2691 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.7511 | 0.6943 | 0.8079 |
| M2 | CrossAttn | global_zscore | F1 | 0.3596 | 0.2308 | 0.4848 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.8008 | 0.7022 | 0.8807 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3401 | 0.1964 | 0.5299 |
| M2_2 | CrossAttn | raw | Brier | 0.2459 | 0.2200 | 0.2705 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6332 | 0.5721 | 0.6987 |
| M2_2 | CrossAttn | raw | F1 | 0.3333 | 0.2222 | 0.4428 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.8049 | 0.7256 | 0.8718 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3236 | 0.1881 | 0.5093 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2531 | 0.2335 | 0.2721 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6288 | 0.5677 | 0.6900 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3411 | 0.2342 | 0.4444 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8016 | 0.7237 | 0.8691 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3314 | 0.1883 | 0.5193 |
| M2_2 | CrossAttn | norm | Brier | 0.2695 | 0.2537 | 0.2849 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6376 | 0.5764 | 0.7031 |
| M2_2 | CrossAttn | norm | F1 | 0.3360 | 0.2276 | 0.4417 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.8124 | 0.7228 | 0.8878 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3579 | 0.2118 | 0.5511 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2512 | 0.2263 | 0.2753 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6332 | 0.5677 | 0.6943 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3438 | 0.2321 | 0.4567 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7409 | 0.6508 | 0.8331 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2323 | 0.1444 | 0.4122 |
| M3 | CrossAttn3 | raw | Brier | 0.2675 | 0.2487 | 0.2861 |
| M3 | CrossAttn3 | raw | Accuracy | 0.5109 | 0.4454 | 0.5764 |
| M3 | CrossAttn3 | raw | F1 | 0.2821 | 0.1923 | 0.3759 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7384 | 0.6374 | 0.8312 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2137 | 0.1394 | 0.3595 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2655 | 0.2458 | 0.2858 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.6900 | 0.6288 | 0.7511 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3107 | 0.1978 | 0.4260 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8435 | 0.7686 | 0.9073 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3738 | 0.2274 | 0.5571 |
| M3 | CrossAttn3 | norm | Brier | 0.2361 | 0.2115 | 0.2582 |
| M3 | CrossAttn3 | norm | Accuracy | 0.7249 | 0.6681 | 0.7817 |
| M3 | CrossAttn3 | norm | F1 | 0.3762 | 0.2500 | 0.5000 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7866 | 0.7016 | 0.8663 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2818 | 0.1701 | 0.4489 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2555 | 0.2294 | 0.2788 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7162 | 0.6594 | 0.7773 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3564 | 0.2366 | 0.4742 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7880 | -0.0150 | 0.572 | 5.675e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7907 | -0.0124 | 0.434 | 6.644e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8108 | +0.0077 | -0.457 | 6.474e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7848 | -0.0183 | 0.671 | 5.020e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7409 | -0.0622 | 1.426 | 1.539e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7384 | -0.0646 | 1.513 | 1.302e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8435 | +0.0404 | -1.733 | 8.307e-02 | † |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7866 | -0.0165 | 0.594 | 5.528e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7880 | 0.8008 | +0.0128 | -0.724 | 4.688e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7907 | 0.8049 | +0.0142 | -0.476 | 6.339e-01 | ns |
| M2-norm vs M2_2-norm | 0.8108 | 0.8016 | -0.0091 | 0.325 | 7.448e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.7848 | 0.8124 | +0.0276 | -0.986 | 3.240e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7880 | 0.7409 | -0.0472 | 1.195 | 2.321e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7907 | 0.7384 | -0.0522 | 1.767 | 7.721e-02 | † |
| M2-norm vs M3-norm | 0.8108 | 0.8435 | +0.0327 | -1.812 | 7.002e-02 | † |
| M2-global_zscore vs M3-global_zscore | 0.7848 | 0.7866 | +0.0018 | -0.106 | 9.153e-01 | ns |

