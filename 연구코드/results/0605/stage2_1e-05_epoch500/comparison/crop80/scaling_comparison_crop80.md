# Scaling Comparison — Test Set Performance (AEC 103pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8220 | 0.3229 | 0.2543 | 0.6681 | 0.3667 |
| M2_2 | CrossAttn | norm | 0.8272 | 0.3406 | 0.2663 | 0.7293 | 0.3922 |
| M3 | CrossAttn3 | std_scaled | 0.8002 | 0.3007 | 0.2588 | 0.7031 | 0.3333 |

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
| raw | 0.7878 | 0.2886 | 0.2582 | 0.7336 | 0.3579 |
| std_scaled | 0.8018 | 0.2925 | 0.2599 | 0.7729 | 0.3659 |
| **norm** | 0.8220 | 0.3229 | 0.2543 | 0.6681 | 0.3667 |
| global_zscore | 0.7925 | 0.2758 | 0.2702 | 0.7031 | 0.3585 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7974 | 0.3315 | 0.2634 | 0.6725 | 0.3590 |
| std_scaled | 0.8144 | 0.3497 | 0.2674 | 0.6812 | 0.3652 |
| **norm** | 0.8272 | 0.3406 | 0.2663 | 0.7293 | 0.3922 |
| global_zscore | 0.7868 | 0.3200 | 0.2756 | 0.6114 | 0.3407 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7575 | 0.2367 | 0.2827 | 0.5590 | 0.2628 |
| **std_scaled** | 0.8002 | 0.3007 | 0.2588 | 0.7031 | 0.3333 |
| norm | 0.7943 | 0.2640 | 0.2676 | 0.6943 | 0.3269 |
| global_zscore | 0.7646 | 0.2318 | 0.2661 | 0.7074 | 0.3495 |

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
| AUC-ROC  | 0.8061 | 0.8043 | -0.0018 | 0.104 | 9.22e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3849 | -0.0243 | 0.645 | 5.54e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2583 | +0.0776 | -13.544 | 1.72e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7691 | +0.0130 | -0.381 | 7.22e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4179 | +0.0015 | -0.054 | 9.59e-01 | 1.00e+00 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8006 | -0.0056 | 0.567 | 6.01e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3799 | -0.0294 | 0.663 | 5.44e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2558 | +0.0750 | -13.290 | 1.85e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7889 | +0.0328 | -0.755 | 4.92e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4279 | +0.0115 | -0.366 | 7.33e-01 | 6.25e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8100 | +0.0039 | -0.398 | 7.11e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3875 | -0.0217 | 0.757 | 4.91e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2509 | +0.0701 | -18.775 | 4.74e-05 | 6.25e-02 |
| Accuracy ** | 0.7561 | 0.7222 | -0.0339 | 4.836 | 8.42e-03 | 6.25e-02 |
| F1  | 0.4163 | 0.4004 | -0.0159 | 1.351 | 2.48e-01 | 3.12e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7942 | -0.0119 | 0.543 | 6.16e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3738 | -0.0354 | 1.043 | 3.56e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2624 | +0.0816 | -17.817 | 5.83e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7321 | -0.0240 | 0.599 | 5.82e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3879 | -0.0284 | 1.001 | 3.73e-01 | 3.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8043 | 0.7706 | -0.0337 | 1.712 | 1.62e-01 | 1.25e-01 |
| AUPRC  | 0.3849 | 0.3297 | -0.0552 | 1.288 | 2.67e-01 | 3.12e-01 |
| Brier  | 0.2583 | 0.2660 | +0.0077 | -1.397 | 2.35e-01 | 3.12e-01 |
| Accuracy  | 0.7691 | 0.6608 | -0.1083 | 1.261 | 2.76e-01 | 4.38e-01 |
| F1  | 0.4179 | 0.3547 | -0.0632 | 1.688 | 1.67e-01 | 1.88e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8006 | 0.7691 | -0.0315 | 1.771 | 1.51e-01 | 1.88e-01 |
| AUPRC  | 0.3799 | 0.3416 | -0.0382 | 1.516 | 2.04e-01 | 3.12e-01 |
| Brier  | 0.2558 | 0.2652 | +0.0095 | -2.104 | 1.03e-01 | 6.25e-02 |
| Accuracy † | 0.7889 | 0.7200 | -0.0689 | 2.328 | 8.05e-02 | 1.25e-01 |
| F1 † | 0.4279 | 0.3793 | -0.0486 | 2.189 | 9.38e-02 | 1.25e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8100 | 0.7924 | -0.0176 | 2.393 | 7.49e-02 | 1.25e-01 |
| AUPRC  | 0.3875 | 0.3595 | -0.0280 | 1.502 | 2.07e-01 | 6.25e-02 |
| Brier † | 0.2509 | 0.2744 | +0.0235 | -2.506 | 6.63e-02 | 6.25e-02 |
| Accuracy  | 0.7222 | 0.7452 | +0.0230 | -0.541 | 6.17e-01 | 7.50e-01 |
| F1  | 0.4004 | 0.3919 | -0.0085 | 0.375 | 7.27e-01 | 1.00e+00 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.7942 | 0.7846 | -0.0096 | 0.701 | 5.22e-01 | 6.25e-01 |
| AUPRC  | 0.3738 | 0.3545 | -0.0193 | 0.471 | 6.62e-01 | 1.00e+00 |
| Brier  | 0.2624 | 0.2593 | -0.0031 | 0.825 | 4.56e-01 | 6.25e-01 |
| Accuracy  | 0.7321 | 0.7364 | +0.0043 | -0.161 | 8.80e-01 | 1.00e+00 |
| F1  | 0.3879 | 0.3733 | -0.0146 | 0.853 | 4.42e-01 | 4.38e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7706 | -0.0355 | 1.018 | 3.66e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3297 | -0.0795 | 1.196 | 2.98e-01 | 4.38e-01 |
| Brier ** | 0.1808 | 0.2660 | +0.0852 | -8.496 | 1.05e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6608 | -0.0953 | 1.114 | 3.28e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.3547 | -0.0617 | 1.176 | 3.05e-01 | 6.25e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7691 | -0.0370 | 1.543 | 1.98e-01 | 1.88e-01 |
| AUPRC  | 0.4092 | 0.3416 | -0.0676 | 1.370 | 2.42e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2652 | +0.0845 | -11.002 | 3.88e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7200 | -0.0361 | 0.506 | 6.40e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.3793 | -0.0370 | 0.774 | 4.82e-01 | 4.38e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7924 | -0.0137 | 1.078 | 3.42e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3595 | -0.0497 | 1.384 | 2.39e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2744 | +0.0937 | -12.397 | 2.43e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7452 | -0.0109 | 0.261 | 8.07e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.3919 | -0.0245 | 0.828 | 4.54e-01 | 4.38e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7846 | -0.0215 | 1.028 | 3.62e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3545 | -0.0547 | 0.970 | 3.87e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2593 | +0.0785 | -16.347 | 8.20e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7364 | -0.0197 | 0.342 | 7.49e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.3733 | -0.0430 | 0.990 | 3.78e-01 | 4.38e-01 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7878 | 0.6868 | 0.8716 |
| M2 | CrossAttn | raw | AUPRC | 0.2886 | 0.1723 | 0.4468 |
| M2 | CrossAttn | raw | Brier | 0.2582 | 0.2306 | 0.2834 |
| M2 | CrossAttn | raw | Accuracy | 0.7336 | 0.6769 | 0.7904 |
| M2 | CrossAttn | raw | F1 | 0.3579 | 0.2326 | 0.4808 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.8018 | 0.7281 | 0.8724 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2925 | 0.1754 | 0.4673 |
| M2 | CrossAttn | std_scaled | Brier | 0.2599 | 0.2331 | 0.2850 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7729 | 0.7205 | 0.8253 |
| M2 | CrossAttn | std_scaled | F1 | 0.3659 | 0.2338 | 0.5000 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8220 | 0.7478 | 0.8856 |
| M2 | CrossAttn | norm | AUPRC | 0.3229 | 0.1959 | 0.4979 |
| M2 | CrossAttn | norm | Brier | 0.2543 | 0.2287 | 0.2785 |
| M2 | CrossAttn | norm | Accuracy | 0.6681 | 0.6070 | 0.7293 |
| M2 | CrossAttn | norm | F1 | 0.3667 | 0.2476 | 0.4762 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7925 | 0.7055 | 0.8729 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2758 | 0.1777 | 0.4553 |
| M2 | CrossAttn | global_zscore | Brier | 0.2702 | 0.2467 | 0.2914 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.7031 | 0.6463 | 0.7642 |
| M2 | CrossAttn | global_zscore | F1 | 0.3585 | 0.2330 | 0.4742 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7974 | 0.6973 | 0.8781 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3315 | 0.1897 | 0.5184 |
| M2_2 | CrossAttn | raw | Brier | 0.2634 | 0.2369 | 0.2883 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6725 | 0.6114 | 0.7336 |
| M2_2 | CrossAttn | raw | F1 | 0.3590 | 0.2407 | 0.4714 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.8144 | 0.7339 | 0.8813 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3497 | 0.2028 | 0.5421 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2674 | 0.2410 | 0.2921 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6812 | 0.6201 | 0.7424 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3652 | 0.2478 | 0.4787 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8272 | 0.7481 | 0.8916 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3406 | 0.2038 | 0.5252 |
| M2_2 | CrossAttn | norm | Brier | 0.2663 | 0.2413 | 0.2901 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7293 | 0.6725 | 0.7860 |
| M2_2 | CrossAttn | norm | F1 | 0.3922 | 0.2653 | 0.5128 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7868 | 0.6994 | 0.8627 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3200 | 0.1771 | 0.5054 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2756 | 0.2539 | 0.2963 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6114 | 0.5459 | 0.6769 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3407 | 0.2361 | 0.4445 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7575 | 0.6675 | 0.8488 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2367 | 0.1512 | 0.4021 |
| M3 | CrossAttn3 | raw | Brier | 0.2827 | 0.2625 | 0.3021 |
| M3 | CrossAttn3 | raw | Accuracy | 0.5590 | 0.4934 | 0.6245 |
| M3 | CrossAttn3 | raw | F1 | 0.2628 | 0.1654 | 0.3586 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.8002 | 0.7066 | 0.8785 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.3007 | 0.1763 | 0.4745 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2588 | 0.2336 | 0.2833 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.7031 | 0.6419 | 0.7598 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3333 | 0.2157 | 0.4486 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.7943 | 0.7127 | 0.8686 |
| M3 | CrossAttn3 | norm | AUPRC | 0.2640 | 0.1703 | 0.4414 |
| M3 | CrossAttn3 | norm | Brier | 0.2676 | 0.2465 | 0.2871 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6943 | 0.6332 | 0.7511 |
| M3 | CrossAttn3 | norm | F1 | 0.3269 | 0.2105 | 0.4370 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7646 | 0.6696 | 0.8504 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2318 | 0.1478 | 0.3837 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2661 | 0.2412 | 0.2896 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7074 | 0.6507 | 0.7686 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3495 | 0.2316 | 0.4673 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7878 | -0.0152 | 0.586 | 5.578e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.8018 | -0.0012 | 0.051 | 9.590e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8220 | +0.0189 | -1.013 | 3.112e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7925 | -0.0106 | 0.395 | 6.928e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7575 | -0.0455 | 1.127 | 2.596e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.8002 | -0.0028 | 0.120 | 9.045e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.7943 | -0.0087 | 0.297 | 7.668e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7646 | -0.0384 | 1.262 | 2.068e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7878 | 0.7974 | +0.0096 | -0.556 | 5.785e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.8018 | 0.8144 | +0.0126 | -0.609 | 5.423e-01 | ns |
| M2-norm vs M2_2-norm | 0.8220 | 0.8272 | +0.0053 | -0.300 | 7.640e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.7925 | 0.7868 | -0.0057 | 0.229 | 8.191e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7878 | 0.7575 | -0.0303 | 0.856 | 3.920e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.8018 | 0.8002 | -0.0016 | 0.106 | 9.153e-01 | ns |
| M2-norm vs M3-norm | 0.8220 | 0.7943 | -0.0276 | 1.308 | 1.909e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.7925 | 0.7646 | -0.0278 | 1.526 | 1.269e-01 | ns |

