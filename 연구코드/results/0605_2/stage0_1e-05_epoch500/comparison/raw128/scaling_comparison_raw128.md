# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.7904 | 0.2691 | 0.2589 | 0.6943 | 0.2857 |
| M2_2 | CrossAttn | norm | 0.8280 | 0.3213 | 0.2575 | 0.6987 | 0.3894 |
| M3 | CrossAttn3 | global_zscore | 0.8203 | 0.3219 | 0.2589 | 0.7380 | 0.3750 |

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
| raw | 0.7803 | 0.2685 | 0.2602 | 0.7118 | 0.3400 |
| std_scaled | 0.7850 | 0.2999 | 0.2589 | 0.7729 | 0.3659 |
| **norm** | 0.7904 | 0.2691 | 0.2589 | 0.6943 | 0.2857 |
| global_zscore | 0.7646 | 0.2553 | 0.2742 | 0.6550 | 0.3361 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7896 | 0.2750 | 0.2674 | 0.6812 | 0.3652 |
| std_scaled | 0.7833 | 0.2909 | 0.2678 | 0.6725 | 0.3243 |
| **norm** | 0.8280 | 0.3213 | 0.2575 | 0.6987 | 0.3894 |
| global_zscore | 0.7911 | 0.3086 | 0.2738 | 0.6332 | 0.3333 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7894 | 0.2908 | 0.2578 | 0.7555 | 0.3636 |
| std_scaled | 0.7624 | 0.2473 | 0.2616 | 0.7424 | 0.3059 |
| norm | 0.7913 | 0.2996 | 0.2540 | 0.7860 | 0.3636 |
| **global_zscore** | 0.8203 | 0.3219 | 0.2589 | 0.7380 | 0.3750 |

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
| AUC-ROC  | 0.8061 | 0.8087 | +0.0026 | -0.169 | 8.74e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3780 | -0.0313 | 0.702 | 5.21e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2524 | +0.0716 | -23.237 | 2.03e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7528 | -0.0033 | 0.136 | 8.99e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4080 | -0.0084 | 0.307 | 7.74e-01 | 8.12e-01 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8178 | +0.0116 | -0.968 | 3.88e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.4033 | -0.0059 | 0.212 | 8.42e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2405 | +0.0597 | -12.911 | 2.08e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.8063 | +0.0502 | -1.117 | 3.26e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.4587 | +0.0424 | -1.061 | 3.49e-01 | 3.12e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7970 | -0.0092 | 0.784 | 4.77e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3807 | -0.0285 | 0.783 | 4.77e-01 | 4.38e-01 |
| Brier ** | 0.1808 | 0.2588 | +0.0780 | -8.062 | 1.29e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7189 | -0.0372 | 0.832 | 4.52e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.3901 | -0.0263 | 0.823 | 4.57e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8064 | +0.0003 | -0.015 | 9.89e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3916 | -0.0176 | 0.433 | 6.88e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2468 | +0.0660 | -15.583 | 9.90e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7254 | -0.0306 | 0.785 | 4.77e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3945 | -0.0219 | 0.625 | 5.66e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8087 | 0.8006 | -0.0081 | 0.933 | 4.04e-01 | 6.25e-01 |
| AUPRC  | 0.3780 | 0.3719 | -0.0061 | 0.250 | 8.15e-01 | 8.12e-01 |
| Brier  | 0.2524 | 0.2529 | +0.0005 | -0.144 | 8.93e-01 | 1.00e+00 |
| Accuracy  | 0.7528 | 0.7364 | -0.0164 | 0.388 | 7.18e-01 | 8.12e-01 |
| F1  | 0.4080 | 0.3907 | -0.0173 | 0.580 | 5.93e-01 | 8.12e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8178 | 0.7954 | -0.0224 | 2.365 | 7.72e-02 | 6.25e-02 |
| AUPRC  | 0.4033 | 0.3659 | -0.0374 | 0.852 | 4.42e-01 | 6.25e-01 |
| Brier  | 0.2405 | 0.2476 | +0.0071 | -1.666 | 1.71e-01 | 6.25e-02 |
| Accuracy  | 0.8063 | 0.8173 | +0.0110 | -0.456 | 6.72e-01 | 8.12e-01 |
| F1  | 0.4587 | 0.4450 | -0.0137 | 0.431 | 6.88e-01 | 1.00e+00 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.7970 | 0.7979 | +0.0009 | -0.131 | 9.02e-01 | 6.25e-01 |
| AUPRC  | 0.3807 | 0.3700 | -0.0107 | 0.692 | 5.27e-01 | 8.12e-01 |
| Brier  | 0.2588 | 0.2487 | -0.0101 | 1.215 | 2.91e-01 | 4.38e-01 |
| Accuracy  | 0.7189 | 0.7517 | +0.0328 | -0.642 | 5.56e-01 | 4.38e-01 |
| F1  | 0.3901 | 0.4108 | +0.0208 | -0.541 | 6.17e-01 | 4.38e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8064 | 0.8024 | -0.0041 | 0.852 | 4.42e-01 | 8.12e-01 |
| AUPRC  | 0.3916 | 0.3664 | -0.0252 | 0.602 | 5.80e-01 | 4.38e-01 |
| Brier  | 0.2468 | 0.2533 | +0.0066 | -1.086 | 3.39e-01 | 4.38e-01 |
| Accuracy † | 0.7254 | 0.7835 | +0.0580 | -2.218 | 9.08e-02 | 1.88e-01 |
| F1  | 0.3945 | 0.4258 | +0.0314 | -1.671 | 1.70e-01 | 2.50e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8006 | -0.0056 | 0.284 | 7.91e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3719 | -0.0374 | 0.983 | 3.81e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2529 | +0.0721 | -12.242 | 2.56e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7364 | -0.0197 | 0.369 | 7.31e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3907 | -0.0257 | 0.558 | 6.06e-01 | 6.25e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7954 | -0.0107 | 0.646 | 5.54e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3659 | -0.0433 | 0.934 | 4.03e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2476 | +0.0669 | -9.314 | 7.40e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.8173 | +0.0612 | -1.507 | 2.06e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.4450 | +0.0287 | -0.860 | 4.38e-01 | 8.12e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7979 | -0.0082 | 0.481 | 6.56e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3700 | -0.0392 | 0.951 | 3.95e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2487 | +0.0680 | -11.320 | 3.47e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7517 | -0.0044 | 0.150 | 8.88e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4108 | -0.0055 | 0.271 | 8.00e-01 | 8.12e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8024 | -0.0038 | 0.164 | 8.78e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3664 | -0.0428 | 0.668 | 5.41e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2533 | +0.0726 | -27.395 | 1.06e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7835 | +0.0274 | -0.689 | 5.29e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4258 | +0.0095 | -0.317 | 7.67e-01 | 1.00e+00 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7803 | 0.6986 | 0.8569 |
| M2 | CrossAttn | raw | AUPRC | 0.2685 | 0.1582 | 0.4287 |
| M2 | CrossAttn | raw | Brier | 0.2602 | 0.2335 | 0.2850 |
| M2 | CrossAttn | raw | Accuracy | 0.7118 | 0.6550 | 0.7729 |
| M2 | CrossAttn | raw | F1 | 0.3400 | 0.2198 | 0.4554 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7850 | 0.6872 | 0.8671 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2999 | 0.1736 | 0.4829 |
| M2 | CrossAttn | std_scaled | Brier | 0.2589 | 0.2315 | 0.2854 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7729 | 0.7205 | 0.8253 |
| M2 | CrossAttn | std_scaled | F1 | 0.3659 | 0.2337 | 0.4941 |
| M2 | CrossAttn | norm | AUC-ROC | 0.7904 | 0.7078 | 0.8622 |
| M2 | CrossAttn | norm | AUPRC | 0.2691 | 0.1702 | 0.4573 |
| M2 | CrossAttn | norm | Brier | 0.2589 | 0.2347 | 0.2824 |
| M2 | CrossAttn | norm | Accuracy | 0.6943 | 0.6374 | 0.7555 |
| M2 | CrossAttn | norm | F1 | 0.2857 | 0.1728 | 0.3929 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7646 | 0.6740 | 0.8451 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2553 | 0.1456 | 0.4098 |
| M2 | CrossAttn | global_zscore | Brier | 0.2742 | 0.2489 | 0.2970 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.6550 | 0.5939 | 0.7205 |
| M2 | CrossAttn | global_zscore | F1 | 0.3361 | 0.2222 | 0.4480 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7896 | 0.7090 | 0.8606 |
| M2_2 | CrossAttn | raw | AUPRC | 0.2750 | 0.1664 | 0.4645 |
| M2_2 | CrossAttn | raw | Brier | 0.2674 | 0.2406 | 0.2939 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6812 | 0.6243 | 0.7424 |
| M2_2 | CrossAttn | raw | F1 | 0.3652 | 0.2430 | 0.4793 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7833 | 0.6991 | 0.8579 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.2909 | 0.1674 | 0.4682 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2678 | 0.2426 | 0.2918 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6725 | 0.6114 | 0.7336 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3243 | 0.2087 | 0.4355 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8280 | 0.7492 | 0.8927 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3213 | 0.2027 | 0.5153 |
| M2_2 | CrossAttn | norm | Brier | 0.2575 | 0.2301 | 0.2842 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6987 | 0.6376 | 0.7555 |
| M2_2 | CrossAttn | norm | F1 | 0.3894 | 0.2680 | 0.5001 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7911 | 0.7083 | 0.8611 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3086 | 0.1735 | 0.4854 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2738 | 0.2488 | 0.2982 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6332 | 0.5721 | 0.6987 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3333 | 0.2222 | 0.4412 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7894 | 0.6917 | 0.8781 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2908 | 0.1891 | 0.4888 |
| M3 | CrossAttn3 | raw | Brier | 0.2578 | 0.2325 | 0.2809 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7555 | 0.6987 | 0.8122 |
| M3 | CrossAttn3 | raw | F1 | 0.3636 | 0.2353 | 0.4865 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7624 | 0.6646 | 0.8488 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2473 | 0.1577 | 0.4189 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2616 | 0.2361 | 0.2854 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.7424 | 0.6900 | 0.7991 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3059 | 0.1818 | 0.4330 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.7913 | 0.7047 | 0.8709 |
| M3 | CrossAttn3 | norm | AUPRC | 0.2996 | 0.1836 | 0.5061 |
| M3 | CrossAttn3 | norm | Brier | 0.2540 | 0.2294 | 0.2774 |
| M3 | CrossAttn3 | norm | Accuracy | 0.7860 | 0.7336 | 0.8341 |
| M3 | CrossAttn3 | norm | F1 | 0.3636 | 0.2254 | 0.4884 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.8203 | 0.7408 | 0.8924 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.3219 | 0.1961 | 0.5056 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2589 | 0.2331 | 0.2826 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7380 | 0.6812 | 0.7948 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3750 | 0.2449 | 0.5000 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7803 | -0.0228 | 0.901 | 3.674e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7850 | -0.0181 | 0.777 | 4.374e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.7904 | -0.0126 | 0.591 | 5.547e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7646 | -0.0384 | 1.333 | 1.824e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7894 | -0.0136 | 0.338 | 7.354e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7624 | -0.0407 | 1.695 | 9.000e-02 | † |
| M1-LR vs M3-norm | 0.8030 | 0.7913 | -0.0118 | 0.416 | 6.777e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.8203 | +0.0173 | -0.675 | 4.996e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7803 | 0.7896 | +0.0093 | -0.508 | 6.114e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7850 | 0.7833 | -0.0016 | 0.068 | 9.455e-01 | ns |
| M2-norm vs M2_2-norm | 0.7904 | 0.8280 | +0.0376 | -2.089 | 3.674e-02 | * |
| M2-global_zscore vs M2_2-global_zscore | 0.7646 | 0.7911 | +0.0264 | -1.232 | 2.180e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7803 | 0.7894 | +0.0091 | -0.363 | 7.168e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7850 | 0.7624 | -0.0226 | 1.480 | 1.390e-01 | ns |
| M2-norm vs M3-norm | 0.7904 | 0.7913 | +0.0008 | -0.049 | 9.611e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.7646 | 0.8203 | +0.0557 | -2.701 | 6.904e-03 | ** |

