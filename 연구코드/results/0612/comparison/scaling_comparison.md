# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8246 | 0.3614 | 0.2606 | 0.6681 | 0.3559 |
| M2_2 | CrossAttn | norm | 0.8398 | 0.3816 | 0.2614 | 0.7074 | 0.3738 |
| M3 | CrossAttn3 | norm | 0.8382 | 0.3496 | 0.2629 | 0.6769 | 0.3833 |
| M4 | AECOnly | global_zscore | 0.6030 | 0.2116 | 0.3163 | 0.5328 | 0.2190 |

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
| raw | 0.7807 | 0.2724 | 0.2630 | 0.7293 | 0.3404 |
| **norm** | 0.8246 | 0.3614 | 0.2606 | 0.6681 | 0.3559 |
| global_zscore | 0.8112 | 0.2934 | 0.2544 | 0.6943 | 0.3396 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (3 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7925 | 0.2738 | 0.2653 | 0.6463 | 0.3520 |
| **norm** | 0.8398 | 0.3816 | 0.2614 | 0.7074 | 0.3738 |
| global_zscore | 0.7986 | 0.3084 | 0.2645 | 0.6769 | 0.3729 |

---

## Model 3 — Clinic + Scanner + AEC  (3 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7896 | 0.2690 | 0.2643 | 0.7467 | 0.3830 |
| **norm** | 0.8382 | 0.3496 | 0.2629 | 0.6769 | 0.3833 |
| global_zscore | 0.7797 | 0.2652 | 0.2568 | 0.6943 | 0.3396 |

---

## Model 4 — AEC Only  (3 AEC variants)

> 임상 특징 없이 AEC 시퀀스만으로 분류.

### AECOnly

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.6006 | 0.2167 | 0.3071 | 0.4672 | 0.2179 |
| norm | 0.5392 | 0.1199 | 0.3005 | 0.8297 | 0.0488 |
| **global_zscore** | 0.6030 | 0.2116 | 0.3163 | 0.5328 | 0.2190 |

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
| AUC-ROC  | 0.8061 | 0.8069 | +0.0008 | -0.049 | 9.63e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3771 | -0.0322 | 0.742 | 4.99e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2520 | +0.0713 | -19.865 | 3.79e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7681 | +0.0120 | -0.348 | 7.45e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4160 | -0.0004 | 0.011 | 9.92e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8018 | -0.0043 | 0.413 | 7.01e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3664 | -0.0428 | 1.268 | 2.73e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2494 | +0.0686 | -10.501 | 4.65e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7516 | -0.0044 | 0.098 | 9.27e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4049 | -0.0114 | 0.348 | 7.45e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8063 | +0.0001 | -0.006 | 9.95e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3526 | -0.0567 | 1.194 | 2.98e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2415 | +0.0607 | -11.020 | 3.85e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7232 | -0.0329 | 0.546 | 6.14e-01 | 8.75e-01 |
| F1  | 0.4163 | 0.4059 | -0.0104 | 0.220 | 8.37e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8069 | 0.7975 | -0.0094 | 1.034 | 3.59e-01 | 8.12e-01 |
| AUPRC  | 0.3771 | 0.3785 | +0.0015 | -0.063 | 9.53e-01 | 1.00e+00 |
| Brier  | 0.2520 | 0.2516 | -0.0004 | 0.085 | 9.36e-01 | 1.00e+00 |
| Accuracy  | 0.7681 | 0.7463 | -0.0218 | 0.713 | 5.15e-01 | 5.00e-01 |
| F1  | 0.4160 | 0.4055 | -0.0104 | 0.415 | 6.99e-01 | 1.00e+00 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8018 | 0.7946 | -0.0072 | 1.275 | 2.71e-01 | 3.75e-01 |
| AUPRC  | 0.3664 | 0.3944 | +0.0280 | -1.398 | 2.35e-01 | 3.12e-01 |
| Brier  | 0.2494 | 0.2571 | +0.0077 | -1.127 | 3.23e-01 | 4.38e-01 |
| Accuracy  | 0.7516 | 0.7113 | -0.0404 | 1.227 | 2.87e-01 | 4.38e-01 |
| F1  | 0.4049 | 0.3803 | -0.0246 | 1.051 | 3.52e-01 | 4.38e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8063 | 0.7896 | -0.0167 | 1.853 | 1.38e-01 | 1.25e-01 |
| AUPRC  | 0.3526 | 0.3618 | +0.0092 | -0.523 | 6.29e-01 | 4.38e-01 |
| Brier  | 0.2415 | 0.2493 | +0.0078 | -1.589 | 1.87e-01 | 1.88e-01 |
| Accuracy * | 0.7232 | 0.6816 | -0.0416 | 2.971 | 4.11e-02 | 1.25e-01 |
| F1 ** | 0.4059 | 0.3756 | -0.0303 | 5.794 | 4.41e-03 | 6.25e-02 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7975 | -0.0086 | 0.434 | 6.87e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3785 | -0.0307 | 0.727 | 5.08e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2516 | +0.0709 | -11.097 | 3.75e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7463 | -0.0098 | 0.183 | 8.64e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4055 | -0.0108 | 0.237 | 8.24e-01 | 8.12e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7946 | -0.0115 | 0.789 | 4.74e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3944 | -0.0149 | 0.354 | 7.41e-01 | 8.12e-01 |
| Brier ** | 0.1808 | 0.2571 | +0.0763 | -7.779 | 1.47e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7113 | -0.0448 | 0.947 | 3.97e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.3803 | -0.0361 | 1.120 | 3.25e-01 | 3.12e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7896 | -0.0166 | 0.781 | 4.78e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3618 | -0.0475 | 0.994 | 3.77e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2493 | +0.0685 | -14.091 | 1.47e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6816 | -0.0744 | 1.098 | 3.34e-01 | 5.00e-01 |
| F1  | 0.4163 | 0.3756 | -0.0408 | 0.840 | 4.48e-01 | 8.12e-01 |

## M1 (LR) vs M4 (AECOnly)

> A = M1 LR, B = M4 AECOnly.

### raw  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8061 | 0.6712 | -0.1349 | 3.554 | 2.37e-02 | 6.25e-02 |
| AUPRC * | 0.4092 | 0.2131 | -0.1961 | 3.656 | 2.17e-02 | 6.25e-02 |
| Brier *** | 0.1808 | 0.2955 | +0.1147 | -12.391 | 2.44e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6216 | -0.1345 | 1.811 | 1.44e-01 | 1.88e-01 |
| F1 † | 0.4163 | 0.2982 | -0.1181 | 2.471 | 6.88e-02 | 1.25e-01 |

### norm  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.8061 | 0.5655 | -0.2406 | 5.726 | 4.61e-03 | 6.25e-02 |
| AUPRC * | 0.4092 | 0.1421 | -0.2672 | 4.381 | 1.19e-02 | 6.25e-02 |
| Brier *** | 0.1808 | 0.3001 | +0.1193 | -9.885 | 5.88e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6676 | -0.0885 | 1.583 | 1.89e-01 | 1.88e-01 |
| F1 * | 0.4163 | 0.2427 | -0.1736 | 4.372 | 1.19e-02 | 6.25e-02 |

### global_zscore  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8061 | 0.6693 | -0.1369 | 3.197 | 3.30e-02 | 1.25e-01 |
| AUPRC * | 0.4092 | 0.2298 | -0.1795 | 3.153 | 3.44e-02 | 6.25e-02 |
| Brier *** | 0.1808 | 0.2942 | +0.1135 | -17.637 | 6.07e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6423 | -0.1138 | 1.135 | 3.20e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.3080 | -0.1083 | 1.814 | 1.44e-01 | 1.88e-01 |

## M4 (AECOnly) vs M2 (CrossAttn)

> A = M4 AECOnly, B = M2 CrossAttn. aec_var 키로 매칭.

#### Case: raw  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.6712 | 0.8069 | +0.1357 | -5.388 | 5.74e-03 | 6.25e-02 |
| AUPRC ** | 0.2131 | 0.3771 | +0.1639 | -7.271 | 1.90e-03 | 6.25e-02 |
| Brier ** | 0.2955 | 0.2520 | -0.0434 | 6.469 | 2.94e-03 | 6.25e-02 |
| Accuracy † | 0.6216 | 0.7681 | +0.1465 | -2.559 | 6.27e-02 | 6.25e-02 |
| F1 ** | 0.2982 | 0.4160 | +0.1178 | -5.458 | 5.48e-03 | 6.25e-02 |

#### Case: norm  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.5655 | 0.8018 | +0.2363 | -7.173 | 2.00e-03 | 6.25e-02 |
| AUPRC ** | 0.1421 | 0.3664 | +0.2244 | -5.224 | 6.41e-03 | 6.25e-02 |
| Brier * | 0.3001 | 0.2494 | -0.0507 | 3.167 | 3.39e-02 | 1.25e-01 |
| Accuracy  | 0.6676 | 0.7516 | +0.0840 | -1.133 | 3.21e-01 | 4.38e-01 |
| F1 ** | 0.2427 | 0.4049 | +0.1622 | -8.455 | 1.07e-03 | 6.25e-02 |

#### Case: global_zscore  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.6693 | 0.8063 | +0.1370 | -5.759 | 4.51e-03 | 6.25e-02 |
| AUPRC ** | 0.2298 | 0.3526 | +0.1228 | -5.391 | 5.72e-03 | 6.25e-02 |
| Brier ** | 0.2942 | 0.2415 | -0.0527 | 6.766 | 2.49e-03 | 6.25e-02 |
| Accuracy  | 0.6423 | 0.7232 | +0.0809 | -1.201 | 2.96e-01 | 3.12e-01 |
| F1 † | 0.3080 | 0.4059 | +0.0979 | -2.407 | 7.38e-02 | 1.25e-01 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7807 | 0.6956 | 0.8571 |
| M2 | CrossAttn | raw | AUPRC | 0.2724 | 0.1604 | 0.4349 |
| M2 | CrossAttn | raw | Brier | 0.2630 | 0.2361 | 0.2883 |
| M2 | CrossAttn | raw | Accuracy | 0.7293 | 0.6725 | 0.7860 |
| M2 | CrossAttn | raw | F1 | 0.3404 | 0.2174 | 0.4600 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8246 | 0.7489 | 0.8888 |
| M2 | CrossAttn | norm | AUPRC | 0.3614 | 0.2100 | 0.5450 |
| M2 | CrossAttn | norm | Brier | 0.2606 | 0.2326 | 0.2870 |
| M2 | CrossAttn | norm | Accuracy | 0.6681 | 0.6070 | 0.7293 |
| M2 | CrossAttn | norm | F1 | 0.3559 | 0.2385 | 0.4697 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.8112 | 0.7269 | 0.8877 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2934 | 0.1904 | 0.4779 |
| M2 | CrossAttn | global_zscore | Brier | 0.2544 | 0.2282 | 0.2790 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.6943 | 0.6376 | 0.7555 |
| M2 | CrossAttn | global_zscore | F1 | 0.3396 | 0.2200 | 0.4538 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7925 | 0.7164 | 0.8610 |
| M2_2 | CrossAttn | raw | AUPRC | 0.2738 | 0.1683 | 0.4600 |
| M2_2 | CrossAttn | raw | Brier | 0.2653 | 0.2388 | 0.2917 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6463 | 0.5852 | 0.7118 |
| M2_2 | CrossAttn | raw | F1 | 0.3520 | 0.2364 | 0.4638 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8398 | 0.7633 | 0.9023 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3816 | 0.2286 | 0.5688 |
| M2_2 | CrossAttn | norm | Brier | 0.2614 | 0.2369 | 0.2850 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7074 | 0.6507 | 0.7642 |
| M2_2 | CrossAttn | norm | F1 | 0.3738 | 0.2523 | 0.4874 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7986 | 0.7120 | 0.8716 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3084 | 0.1778 | 0.4893 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2645 | 0.2394 | 0.2884 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6769 | 0.6201 | 0.7380 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3729 | 0.2521 | 0.4892 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7896 | 0.6963 | 0.8735 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2690 | 0.1763 | 0.4520 |
| M3 | CrossAttn3 | raw | Brier | 0.2643 | 0.2382 | 0.2882 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7467 | 0.6900 | 0.8035 |
| M3 | CrossAttn3 | raw | F1 | 0.3830 | 0.2564 | 0.5091 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8382 | 0.7698 | 0.8985 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3496 | 0.2133 | 0.5350 |
| M3 | CrossAttn3 | norm | Brier | 0.2629 | 0.2388 | 0.2851 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6769 | 0.6157 | 0.7380 |
| M3 | CrossAttn3 | norm | F1 | 0.3833 | 0.2653 | 0.4885 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7797 | 0.6862 | 0.8665 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2652 | 0.1750 | 0.4396 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2568 | 0.2315 | 0.2813 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.6943 | 0.6376 | 0.7555 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3396 | 0.2245 | 0.4553 |
| M4 | AECOnly | raw | AUC-ROC | 0.6006 | 0.4708 | 0.7290 |
| M4 | AECOnly | raw | AUPRC | 0.2167 | 0.1082 | 0.3825 |
| M4 | AECOnly | raw | Brier | 0.3071 | 0.2950 | 0.3197 |
| M4 | AECOnly | raw | Accuracy | 0.4672 | 0.4017 | 0.5328 |
| M4 | AECOnly | raw | F1 | 0.2179 | 0.1333 | 0.3086 |
| M4 | AECOnly | norm | AUC-ROC | 0.5392 | 0.4215 | 0.6526 |
| M4 | AECOnly | norm | AUPRC | 0.1199 | 0.0765 | 0.2001 |
| M4 | AECOnly | norm | Brier | 0.3005 | 0.2954 | 0.3049 |
| M4 | AECOnly | norm | Accuracy | 0.8297 | 0.7773 | 0.8777 |
| M4 | AECOnly | norm | F1 | 0.0488 | 0.0000 | 0.1538 |
| M4 | AECOnly | global_zscore | AUC-ROC | 0.6030 | 0.4728 | 0.7287 |
| M4 | AECOnly | global_zscore | AUPRC | 0.2116 | 0.1129 | 0.3971 |
| M4 | AECOnly | global_zscore | Brier | 0.3163 | 0.3029 | 0.3290 |
| M4 | AECOnly | global_zscore | Accuracy | 0.5328 | 0.4671 | 0.5983 |
| M4 | AECOnly | global_zscore | F1 | 0.2190 | 0.1304 | 0.3158 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7807 | -0.0224 | 0.887 | 3.750e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8246 | +0.0215 | -1.057 | 2.906e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.8112 | +0.0081 | -0.322 | 7.478e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7896 | -0.0134 | 0.359 | 7.199e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8382 | +0.0352 | -1.490 | 1.362e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7797 | -0.0234 | 0.692 | 4.892e-01 | ns |

## M1 LR vs M4 AECOnly

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M4-raw | 0.8030 | 0.6006 | -0.2024 | 2.620 | 8.792e-03 | ** |
| M1-LR vs M4-norm | 0.8030 | 0.5392 | -0.2638 | 4.075 | 4.595e-05 | *** |
| M1-LR vs M4-global_zscore | 0.8030 | 0.6030 | -0.2000 | 2.562 | 1.041e-02 | * |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7807 | 0.7925 | +0.0118 | -0.632 | 5.275e-01 | ns |
| M2-norm vs M2_2-norm | 0.8246 | 0.8398 | +0.0152 | -1.127 | 2.596e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.8112 | 0.7986 | -0.0126 | 0.523 | 6.011e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7807 | 0.7896 | +0.0089 | -0.389 | 6.973e-01 | ns |
| M2-norm vs M3-norm | 0.8246 | 0.8382 | +0.0136 | -0.939 | 3.476e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.8112 | 0.7797 | -0.0315 | 1.231 | 2.182e-01 | ns |

## M4 AECOnly vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M4-raw vs M2-raw | 0.6006 | 0.7807 | +0.1801 | -2.815 | 4.881e-03 | ** |
| M4-norm vs M2-norm | 0.5392 | 0.8246 | +0.2854 | -4.853 | 1.218e-06 | *** |
| M4-global_zscore vs M2-global_zscore | 0.6030 | 0.8112 | +0.2081 | -3.219 | 1.288e-03 | ** |

