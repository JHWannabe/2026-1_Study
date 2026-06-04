# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8120 | 0.2949 | 0.2684 | 0.6856 | 0.3571 |
| M2_2 | CrossAttn | norm | 0.8472 | 0.3767 | 0.2526 | 0.7293 | 0.3800 |
| M3 | CrossAttn3 | global_zscore | 0.8006 | 0.2996 | 0.2655 | 0.7817 | 0.3421 |

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
| raw | 0.7896 | 0.2883 | 0.2574 | 0.7380 | 0.3617 |
| std_scaled | 0.7892 | 0.2895 | 0.2624 | 0.7162 | 0.3434 |
| **norm** | 0.8120 | 0.2949 | 0.2684 | 0.6856 | 0.3571 |
| global_zscore | 0.7945 | 0.2938 | 0.2624 | 0.7817 | 0.3750 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7963 | 0.3325 | 0.2649 | 0.6419 | 0.3387 |
| std_scaled | 0.7803 | 0.3070 | 0.2742 | 0.6681 | 0.3333 |
| **norm** | 0.8472 | 0.3767 | 0.2526 | 0.7293 | 0.3800 |
| global_zscore | 0.7606 | 0.2736 | 0.2684 | 0.5459 | 0.2877 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7705 | 0.2507 | 0.2732 | 0.7118 | 0.3265 |
| std_scaled | 0.7382 | 0.2212 | 0.2842 | 0.7031 | 0.2917 |
| norm | 0.7482 | 0.2681 | 0.2806 | 0.6681 | 0.3214 |
| **global_zscore** | 0.8006 | 0.2996 | 0.2655 | 0.7817 | 0.3421 |

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
| AUC-ROC  | 0.8061 | 0.8111 | +0.0050 | -0.288 | 7.88e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3902 | -0.0190 | 0.521 | 6.30e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2565 | +0.0757 | -11.182 | 3.64e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7702 | +0.0141 | -0.371 | 7.30e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4261 | +0.0097 | -0.305 | 7.76e-01 | 8.12e-01 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8072 | +0.0010 | -0.084 | 9.37e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3963 | -0.0130 | 0.480 | 6.56e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2531 | +0.0724 | -10.045 | 5.52e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7713 | +0.0152 | -0.318 | 7.66e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4350 | +0.0187 | -0.408 | 7.04e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8133 | +0.0072 | -0.763 | 4.88e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.4029 | -0.0063 | 0.166 | 8.76e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2531 | +0.0724 | -23.819 | 1.84e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7528 | -0.0033 | 0.297 | 7.81e-01 | 7.50e-01 |
| F1  | 0.4163 | 0.4152 | -0.0012 | 0.072 | 9.46e-01 | 8.12e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8033 | -0.0028 | 0.125 | 9.07e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3981 | -0.0111 | 0.297 | 7.81e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2551 | +0.0744 | -10.119 | 5.37e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7605 | +0.0044 | -0.073 | 9.46e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4184 | +0.0020 | -0.049 | 9.63e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8111 | 0.7766 | -0.0345 | 1.695 | 1.65e-01 | 1.88e-01 |
| AUPRC  | 0.3902 | 0.3403 | -0.0500 | 1.067 | 3.46e-01 | 3.12e-01 |
| Brier  | 0.2565 | 0.2646 | +0.0081 | -1.501 | 2.08e-01 | 3.12e-01 |
| Accuracy  | 0.7702 | 0.6916 | -0.0786 | 0.816 | 4.60e-01 | 6.25e-01 |
| F1  | 0.4261 | 0.3785 | -0.0476 | 1.002 | 3.73e-01 | 4.38e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8072 | 0.7942 | -0.0130 | 1.549 | 1.96e-01 | 1.88e-01 |
| AUPRC  | 0.3963 | 0.3923 | -0.0039 | 0.197 | 8.54e-01 | 1.00e+00 |
| Brier  | 0.2531 | 0.2638 | +0.0107 | -1.891 | 1.32e-01 | 1.88e-01 |
| Accuracy † | 0.7713 | 0.8052 | +0.0339 | -2.391 | 7.51e-02 | 1.25e-01 |
| F1  | 0.4350 | 0.4347 | -0.0004 | 0.021 | 9.84e-01 | 6.25e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8133 | 0.8053 | -0.0080 | 1.000 | 3.74e-01 | 4.38e-01 |
| AUPRC  | 0.4029 | 0.3848 | -0.0181 | 1.151 | 3.14e-01 | 3.12e-01 |
| Brier  | 0.2531 | 0.2606 | +0.0075 | -1.528 | 2.01e-01 | 1.88e-01 |
| Accuracy  | 0.7528 | 0.6993 | -0.0535 | 1.583 | 1.89e-01 | 1.88e-01 |
| F1  | 0.4152 | 0.3911 | -0.0241 | 1.096 | 3.35e-01 | 3.12e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8033 | 0.7905 | -0.0129 | 1.061 | 3.49e-01 | 6.25e-01 |
| AUPRC  | 0.3981 | 0.3817 | -0.0164 | 0.764 | 4.88e-01 | 4.38e-01 |
| Brier  | 0.2551 | 0.2613 | +0.0062 | -1.983 | 1.18e-01 | 1.88e-01 |
| Accuracy  | 0.7605 | 0.7539 | -0.0065 | 0.236 | 8.25e-01 | 1.00e+00 |
| F1  | 0.4184 | 0.4192 | +0.0009 | -0.045 | 9.66e-01 | 8.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7766 | -0.0295 | 0.838 | 4.49e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3403 | -0.0690 | 1.070 | 3.45e-01 | 4.38e-01 |
| Brier ** | 0.1808 | 0.2646 | +0.0838 | -8.160 | 1.23e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6916 | -0.0645 | 0.764 | 4.87e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.3785 | -0.0379 | 0.777 | 4.80e-01 | 8.12e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7942 | -0.0119 | 0.611 | 5.74e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3923 | -0.0169 | 0.363 | 7.35e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2638 | +0.0831 | -26.262 | 1.25e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.8052 | +0.0491 | -0.980 | 3.82e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4347 | +0.0183 | -0.387 | 7.18e-01 | 6.25e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8053 | -0.0008 | 0.054 | 9.60e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3848 | -0.0244 | 0.680 | 5.34e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2606 | +0.0799 | -10.423 | 4.79e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6993 | -0.0568 | 1.765 | 1.52e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.3911 | -0.0252 | 1.286 | 2.68e-01 | 4.38e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7905 | -0.0157 | 0.808 | 4.64e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3817 | -0.0276 | 0.483 | 6.54e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2613 | +0.0805 | -13.022 | 2.01e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7539 | -0.0022 | 0.032 | 9.76e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4192 | +0.0029 | -0.060 | 9.55e-01 | 1.00e+00 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7896 | 0.6918 | 0.8733 |
| M2 | CrossAttn | raw | AUPRC | 0.2883 | 0.1714 | 0.4458 |
| M2 | CrossAttn | raw | Brier | 0.2574 | 0.2303 | 0.2831 |
| M2 | CrossAttn | raw | Accuracy | 0.7380 | 0.6856 | 0.7948 |
| M2 | CrossAttn | raw | F1 | 0.3617 | 0.2340 | 0.4848 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7892 | 0.7025 | 0.8662 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2895 | 0.1707 | 0.4630 |
| M2 | CrossAttn | std_scaled | Brier | 0.2624 | 0.2361 | 0.2866 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7162 | 0.6594 | 0.7773 |
| M2 | CrossAttn | std_scaled | F1 | 0.3434 | 0.2222 | 0.4615 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8120 | 0.7393 | 0.8770 |
| M2 | CrossAttn | norm | AUPRC | 0.2949 | 0.1765 | 0.4563 |
| M2 | CrossAttn | norm | Brier | 0.2684 | 0.2412 | 0.2947 |
| M2 | CrossAttn | norm | Accuracy | 0.6856 | 0.6288 | 0.7467 |
| M2 | CrossAttn | norm | F1 | 0.3571 | 0.2353 | 0.4715 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7945 | 0.7024 | 0.8742 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2938 | 0.1750 | 0.4644 |
| M2 | CrossAttn | global_zscore | Brier | 0.2624 | 0.2364 | 0.2863 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.7817 | 0.7293 | 0.8341 |
| M2 | CrossAttn | global_zscore | F1 | 0.3750 | 0.2424 | 0.5067 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7963 | 0.6960 | 0.8783 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3325 | 0.1908 | 0.5206 |
| M2_2 | CrossAttn | raw | Brier | 0.2649 | 0.2377 | 0.2904 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6419 | 0.5808 | 0.7074 |
| M2_2 | CrossAttn | raw | F1 | 0.3387 | 0.2264 | 0.4478 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7803 | 0.6955 | 0.8547 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3070 | 0.1721 | 0.4896 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2742 | 0.2506 | 0.2973 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6681 | 0.6070 | 0.7293 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3333 | 0.2182 | 0.4445 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8472 | 0.7711 | 0.9089 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3767 | 0.2341 | 0.5937 |
| M2_2 | CrossAttn | norm | Brier | 0.2526 | 0.2288 | 0.2754 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7293 | 0.6767 | 0.7860 |
| M2_2 | CrossAttn | norm | F1 | 0.3800 | 0.2588 | 0.5000 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7606 | 0.6720 | 0.8402 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.2736 | 0.1562 | 0.4631 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2684 | 0.2416 | 0.2939 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.5459 | 0.4803 | 0.6114 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.2877 | 0.1890 | 0.3851 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7705 | 0.6860 | 0.8539 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2507 | 0.1594 | 0.4322 |
| M3 | CrossAttn3 | raw | Brier | 0.2732 | 0.2518 | 0.2938 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7118 | 0.6507 | 0.7729 |
| M3 | CrossAttn3 | raw | F1 | 0.3265 | 0.2069 | 0.4424 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7382 | 0.6342 | 0.8344 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2212 | 0.1382 | 0.3706 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2842 | 0.2589 | 0.3080 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.7031 | 0.6463 | 0.7642 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.2917 | 0.1707 | 0.4087 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.7482 | 0.6536 | 0.8382 |
| M3 | CrossAttn3 | norm | AUPRC | 0.2681 | 0.1509 | 0.4425 |
| M3 | CrossAttn3 | norm | Brier | 0.2806 | 0.2606 | 0.3002 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6681 | 0.6114 | 0.7293 |
| M3 | CrossAttn3 | norm | F1 | 0.3214 | 0.2075 | 0.4333 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.8006 | 0.7177 | 0.8815 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2996 | 0.1904 | 0.4979 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2655 | 0.2428 | 0.2857 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7817 | 0.7293 | 0.8341 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3421 | 0.2105 | 0.4777 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7896 | -0.0134 | 0.523 | 6.010e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7892 | -0.0138 | 0.524 | 6.002e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8120 | +0.0089 | -0.424 | 6.719e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7945 | -0.0085 | 0.380 | 7.039e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7705 | -0.0325 | 0.904 | 3.658e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7382 | -0.0648 | 1.613 | 1.068e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.7482 | -0.0549 | 1.564 | 1.179e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.8006 | -0.0024 | 0.060 | 9.525e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7896 | 0.7963 | +0.0067 | -0.431 | 6.666e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7892 | 0.7803 | -0.0089 | 0.376 | 7.072e-01 | ns |
| M2-norm vs M2_2-norm | 0.8120 | 0.8472 | +0.0352 | -1.628 | 1.036e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.7945 | 0.7606 | -0.0339 | 1.433 | 1.517e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7896 | 0.7705 | -0.0191 | 0.543 | 5.870e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7892 | 0.7382 | -0.0510 | 2.233 | 2.554e-02 | * |
| M2-norm vs M3-norm | 0.8120 | 0.7482 | -0.0638 | 2.235 | 2.540e-02 | * |
| M2-global_zscore vs M3-global_zscore | 0.7945 | 0.8006 | +0.0061 | -0.172 | 8.633e-01 | ns |

