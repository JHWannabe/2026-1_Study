# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8022 | 0.3459 | 0.2661 | 0.7511 | 0.3448 |
| M2_2 | CrossAttn | norm | 0.8411 | 0.3541 | 0.2593 | 0.7118 | 0.3889 |
| M3 | CrossAttn3 | norm | 0.8303 | 0.3374 | 0.2577 | 0.7293 | 0.3922 |
| M4 | AECOnly | global_zscore | 0.6043 | 0.2076 | 0.3008 | 0.4236 | 0.2048 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_all** | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |

---

## Model 2 — Clinic + AEC (Matched)  (3 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7780 | 0.2655 | 0.2621 | 0.7598 | 0.3678 |
| **norm** | 0.8022 | 0.3459 | 0.2661 | 0.7511 | 0.3448 |
| global_zscore | 0.7994 | 0.2924 | 0.2523 | 0.7249 | 0.3762 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (3 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7911 | 0.2722 | 0.2634 | 0.6812 | 0.3652 |
| **norm** | 0.8411 | 0.3541 | 0.2593 | 0.7118 | 0.3889 |
| global_zscore | 0.8140 | 0.3373 | 0.2612 | 0.6288 | 0.3200 |

---

## Model 3 — Clinic + Scanner + AEC  (3 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7872 | 0.2709 | 0.2634 | 0.7380 | 0.3750 |
| **norm** | 0.8303 | 0.3374 | 0.2577 | 0.7293 | 0.3922 |
| global_zscore | 0.8163 | 0.3452 | 0.2643 | 0.6812 | 0.3652 |

---

## Model 4 — AEC Only  (3 AEC variants)

> 임상 특징 없이 AEC 시퀀스만으로 분류.

### AECOnly

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.6014 | 0.2251 | 0.3066 | 0.4498 | 0.2125 |
| norm | 0.5213 | 0.1126 | 0.3037 | 0.7293 | 0.1143 |
| **global_zscore** | 0.6043 | 0.2076 | 0.3008 | 0.4236 | 0.2048 |

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
| AUC-ROC  | 0.8061 | 0.8045 | -0.0016 | 0.111 | 9.17e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3702 | -0.0390 | 0.861 | 4.38e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2513 | +0.0705 | -24.783 | 1.57e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7834 | +0.0273 | -0.775 | 4.82e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4273 | +0.0109 | -0.309 | 7.73e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8089 | +0.0027 | -0.249 | 8.16e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3815 | -0.0278 | 0.985 | 3.81e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2416 | +0.0608 | -24.410 | 1.67e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7779 | +0.0219 | -0.843 | 4.47e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4446 | +0.0282 | -1.385 | 2.38e-01 | 3.75e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8134 | +0.0073 | -0.389 | 7.17e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.4116 | +0.0024 | -0.066 | 9.51e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2465 | +0.0658 | -21.423 | 2.81e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7549 | -0.0012 | 0.049 | 9.63e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4183 | +0.0020 | -0.078 | 9.42e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8045 | 0.7916 | -0.0129 | 1.232 | 2.85e-01 | 8.12e-01 |
| AUPRC  | 0.3702 | 0.3586 | -0.0116 | 0.384 | 7.21e-01 | 8.12e-01 |
| Brier  | 0.2513 | 0.2526 | +0.0014 | -0.244 | 8.19e-01 | 1.00e+00 |
| Accuracy  | 0.7834 | 0.7266 | -0.0568 | 0.899 | 4.20e-01 | 8.12e-01 |
| F1  | 0.4273 | 0.3908 | -0.0365 | 0.800 | 4.69e-01 | 8.12e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8089 | 0.8074 | -0.0014 | 0.232 | 8.28e-01 | 6.25e-01 |
| AUPRC  | 0.3815 | 0.4064 | +0.0249 | -1.206 | 2.94e-01 | 3.12e-01 |
| Brier  | 0.2416 | 0.2402 | -0.0014 | 0.240 | 8.22e-01 | 8.12e-01 |
| Accuracy  | 0.7779 | 0.7735 | -0.0044 | 0.128 | 9.04e-01 | 1.00e+00 |
| F1  | 0.4446 | 0.4233 | -0.0212 | 0.965 | 3.89e-01 | 6.25e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8134 | 0.8005 | -0.0129 | 1.205 | 2.95e-01 | 4.38e-01 |
| AUPRC  | 0.4116 | 0.3651 | -0.0465 | 1.946 | 1.23e-01 | 6.25e-02 |
| Brier  | 0.2465 | 0.2490 | +0.0025 | -0.375 | 7.27e-01 | 1.00e+00 |
| Accuracy  | 0.7549 | 0.7222 | -0.0327 | 0.573 | 5.97e-01 | 6.25e-01 |
| F1  | 0.4183 | 0.3838 | -0.0345 | 0.893 | 4.22e-01 | 8.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7916 | -0.0145 | 0.663 | 5.44e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3586 | -0.0507 | 1.119 | 3.26e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2526 | +0.0719 | -9.364 | 7.24e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7266 | -0.0295 | 0.482 | 6.55e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3908 | -0.0255 | 0.552 | 6.10e-01 | 8.12e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8074 | +0.0013 | -0.131 | 9.02e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.4064 | -0.0028 | 0.067 | 9.50e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2402 | +0.0594 | -13.295 | 1.85e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7735 | +0.0174 | -0.510 | 6.37e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4233 | +0.0070 | -0.229 | 8.30e-01 | 1.00e+00 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8005 | -0.0056 | 0.246 | 8.18e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3651 | -0.0441 | 1.002 | 3.73e-01 | 8.12e-01 |
| Brier ** | 0.1808 | 0.2490 | +0.0683 | -7.605 | 1.60e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7222 | -0.0339 | 0.580 | 5.93e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.3838 | -0.0325 | 0.735 | 5.03e-01 | 6.25e-01 |

## M1 (LR) vs M4 (AECOnly)

> A = M1 LR, B = M4 AECOnly.

### raw  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8061 | 0.6624 | -0.1437 | 3.708 | 2.07e-02 | 6.25e-02 |
| AUPRC * | 0.4092 | 0.1972 | -0.2120 | 3.789 | 1.93e-02 | 6.25e-02 |
| Brier *** | 0.1808 | 0.2937 | +0.1129 | -12.468 | 2.38e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6183 | -0.1378 | 1.857 | 1.37e-01 | 1.25e-01 |
| F1 † | 0.4163 | 0.2961 | -0.1202 | 2.517 | 6.55e-02 | 1.25e-01 |

### norm  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.8061 | 0.5583 | -0.2478 | 7.563 | 1.64e-03 | 6.25e-02 |
| AUPRC * | 0.4092 | 0.1554 | -0.2539 | 4.283 | 1.28e-02 | 6.25e-02 |
| Brier *** | 0.1808 | 0.3056 | +0.1248 | -14.230 | 1.42e-04 | 6.25e-02 |
| Accuracy † | 0.7561 | 0.5842 | -0.1719 | 2.512 | 6.59e-02 | 1.25e-01 |
| F1 ** | 0.4163 | 0.2511 | -0.1653 | 5.113 | 6.92e-03 | 6.25e-02 |

### global_zscore  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8061 | 0.6773 | -0.1289 | 3.436 | 2.64e-02 | 6.25e-02 |
| AUPRC † | 0.4092 | 0.2454 | -0.1638 | 2.146 | 9.84e-02 | 1.25e-01 |
| Brier ** | 0.1808 | 0.2742 | +0.0935 | -5.029 | 7.34e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6533 | -0.1028 | 1.097 | 3.34e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.3239 | -0.0925 | 1.718 | 1.61e-01 | 1.88e-01 |

## M4 (AECOnly) vs M2 (CrossAttn)

> A = M4 AECOnly, B = M2 CrossAttn. aec_var 키로 매칭.

#### Case: raw  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.6624 | 0.8045 | +0.1421 | -5.634 | 4.88e-03 | 6.25e-02 |
| AUPRC ** | 0.1972 | 0.3702 | +0.1730 | -8.003 | 1.32e-03 | 6.25e-02 |
| Brier ** | 0.2937 | 0.2513 | -0.0424 | 6.358 | 3.14e-03 | 6.25e-02 |
| Accuracy † | 0.6183 | 0.7834 | +0.1650 | -2.256 | 8.70e-02 | 6.25e-02 |
| F1 * | 0.2961 | 0.4273 | +0.1311 | -3.561 | 2.36e-02 | 6.25e-02 |

#### Case: norm  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC *** | 0.5583 | 0.8089 | +0.2505 | -10.818 | 4.14e-04 | 6.25e-02 |
| AUPRC ** | 0.1554 | 0.3815 | +0.2261 | -4.878 | 8.17e-03 | 6.25e-02 |
| Brier ** | 0.3056 | 0.2416 | -0.0640 | 6.127 | 3.60e-03 | 6.25e-02 |
| Accuracy † | 0.5842 | 0.7779 | +0.1937 | -2.443 | 7.10e-02 | 1.25e-01 |
| F1 ** | 0.2511 | 0.4446 | +0.1935 | -5.302 | 6.08e-03 | 6.25e-02 |

#### Case: global_zscore  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.6773 | 0.8134 | +0.1361 | -5.968 | 3.96e-03 | 6.25e-02 |
| AUPRC * | 0.2454 | 0.4116 | +0.1662 | -3.280 | 3.05e-02 | 6.25e-02 |
| Brier  | 0.2742 | 0.2465 | -0.0277 | 1.491 | 2.10e-01 | 1.88e-01 |
| Accuracy  | 0.6533 | 0.7549 | +0.1016 | -1.132 | 3.21e-01 | 4.38e-01 |
| F1 † | 0.3239 | 0.4183 | +0.0944 | -2.257 | 8.70e-02 | 1.25e-01 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7780 | 0.6923 | 0.8558 |
| M2 | CrossAttn | raw | AUPRC | 0.2655 | 0.1555 | 0.4229 |
| M2 | CrossAttn | raw | Brier | 0.2621 | 0.2351 | 0.2876 |
| M2 | CrossAttn | raw | Accuracy | 0.7598 | 0.7031 | 0.8166 |
| M2 | CrossAttn | raw | F1 | 0.3678 | 0.2381 | 0.4941 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8022 | 0.7184 | 0.8732 |
| M2 | CrossAttn | norm | AUPRC | 0.3459 | 0.1947 | 0.5269 |
| M2 | CrossAttn | norm | Brier | 0.2661 | 0.2442 | 0.2868 |
| M2 | CrossAttn | norm | Accuracy | 0.7511 | 0.6943 | 0.8035 |
| M2 | CrossAttn | norm | F1 | 0.3448 | 0.2133 | 0.4652 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7994 | 0.7159 | 0.8714 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2924 | 0.1736 | 0.4672 |
| M2 | CrossAttn | global_zscore | Brier | 0.2523 | 0.2254 | 0.2764 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.7249 | 0.6681 | 0.7818 |
| M2 | CrossAttn | global_zscore | F1 | 0.3762 | 0.2524 | 0.4952 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7911 | 0.7145 | 0.8599 |
| M2_2 | CrossAttn | raw | AUPRC | 0.2722 | 0.1664 | 0.4579 |
| M2_2 | CrossAttn | raw | Brier | 0.2634 | 0.2369 | 0.2896 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6812 | 0.6201 | 0.7424 |
| M2_2 | CrossAttn | raw | F1 | 0.3652 | 0.2439 | 0.4746 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8411 | 0.7736 | 0.8990 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3541 | 0.2145 | 0.5426 |
| M2_2 | CrossAttn | norm | Brier | 0.2593 | 0.2347 | 0.2833 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7118 | 0.6550 | 0.7729 |
| M2_2 | CrossAttn | norm | F1 | 0.3889 | 0.2617 | 0.5047 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.8140 | 0.7356 | 0.8816 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3373 | 0.1987 | 0.5177 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2612 | 0.2349 | 0.2856 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6288 | 0.5677 | 0.6900 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3200 | 0.2069 | 0.4243 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7872 | 0.6943 | 0.8709 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2709 | 0.1768 | 0.4567 |
| M3 | CrossAttn3 | raw | Brier | 0.2634 | 0.2373 | 0.2877 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7380 | 0.6812 | 0.7991 |
| M3 | CrossAttn3 | raw | F1 | 0.3750 | 0.2500 | 0.4956 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8303 | 0.7535 | 0.8933 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3374 | 0.2070 | 0.5395 |
| M3 | CrossAttn3 | norm | Brier | 0.2577 | 0.2318 | 0.2817 |
| M3 | CrossAttn3 | norm | Accuracy | 0.7293 | 0.6725 | 0.7860 |
| M3 | CrossAttn3 | norm | F1 | 0.3922 | 0.2653 | 0.5094 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.8163 | 0.7336 | 0.8924 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.3452 | 0.2108 | 0.5428 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2643 | 0.2380 | 0.2890 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.6812 | 0.6201 | 0.7424 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3652 | 0.2453 | 0.4769 |
| M4 | AECOnly | raw | AUC-ROC | 0.6014 | 0.4736 | 0.7282 |
| M4 | AECOnly | raw | AUPRC | 0.2251 | 0.1097 | 0.4009 |
| M4 | AECOnly | raw | Brier | 0.3066 | 0.2945 | 0.3188 |
| M4 | AECOnly | raw | Accuracy | 0.4498 | 0.3885 | 0.5153 |
| M4 | AECOnly | raw | F1 | 0.2125 | 0.1282 | 0.2994 |
| M4 | AECOnly | norm | AUC-ROC | 0.5213 | 0.3974 | 0.6417 |
| M4 | AECOnly | norm | AUPRC | 0.1126 | 0.0724 | 0.1808 |
| M4 | AECOnly | norm | Brier | 0.3037 | 0.2983 | 0.3084 |
| M4 | AECOnly | norm | Accuracy | 0.7293 | 0.6725 | 0.7860 |
| M4 | AECOnly | norm | F1 | 0.1143 | 0.0282 | 0.2222 |
| M4 | AECOnly | global_zscore | AUC-ROC | 0.6043 | 0.4819 | 0.7260 |
| M4 | AECOnly | global_zscore | AUPRC | 0.2076 | 0.1023 | 0.3578 |
| M4 | AECOnly | global_zscore | Brier | 0.3008 | 0.2894 | 0.3124 |
| M4 | AECOnly | global_zscore | Accuracy | 0.4236 | 0.3624 | 0.4891 |
| M4 | AECOnly | global_zscore | F1 | 0.2048 | 0.1234 | 0.2892 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7780 | -0.0250 | 1.016 | 3.098e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8022 | -0.0008 | 0.044 | 9.651e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7994 | -0.0037 | 0.171 | 8.641e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7872 | -0.0159 | 0.416 | 6.771e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8303 | +0.0272 | -1.243 | 2.139e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.8163 | +0.0132 | -0.434 | 6.643e-01 | ns |

## M1 LR vs M4 AECOnly

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M4-raw | 0.8030 | 0.6014 | -0.2016 | 2.595 | 9.457e-03 | ** |
| M1-LR vs M4-norm | 0.8030 | 0.5213 | -0.2817 | 4.276 | 1.899e-05 | *** |
| M1-LR vs M4-global_zscore | 0.8030 | 0.6043 | -0.1988 | 2.659 | 7.846e-03 | ** |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7780 | 0.7911 | +0.0130 | -0.710 | 4.775e-01 | ns |
| M2-norm vs M2_2-norm | 0.8022 | 0.8411 | +0.0388 | -2.071 | 3.837e-02 | * |
| M2-global_zscore vs M2_2-global_zscore | 0.7994 | 0.8140 | +0.0146 | -0.806 | 4.202e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7780 | 0.7872 | +0.0091 | -0.395 | 6.928e-01 | ns |
| M2-norm vs M3-norm | 0.8022 | 0.8303 | +0.0280 | -1.931 | 5.345e-02 | † |
| M2-global_zscore vs M3-global_zscore | 0.7994 | 0.8163 | +0.0169 | -0.729 | 4.659e-01 | ns |

## M4 AECOnly vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M4-raw vs M2-raw | 0.6014 | 0.7780 | +0.1766 | -2.736 | 6.227e-03 | ** |
| M4-norm vs M2-norm | 0.5213 | 0.8022 | +0.2809 | -4.114 | 3.889e-05 | *** |
| M4-global_zscore vs M2-global_zscore | 0.6043 | 0.7994 | +0.1951 | -2.980 | 2.881e-03 | ** |

