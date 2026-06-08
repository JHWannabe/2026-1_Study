# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | global_zscore | 0.8142 | 0.3038 | 0.2552 | 0.7511 | 0.3871 |
| M2_2 | CrossAttn | norm | 0.8120 | 0.3099 | 0.2574 | 0.6943 | 0.3636 |
| M3 | CrossAttn3 | norm | 0.8382 | 0.3498 | 0.2629 | 0.6769 | 0.3833 |
| M4 | AECOnly | raw | 0.6006 | 0.2166 | 0.3071 | 0.4760 | 0.2105 |

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
| raw | 0.7783 | 0.2656 | 0.2621 | 0.7336 | 0.3579 |
| norm | 0.8028 | 0.3466 | 0.2661 | 0.7249 | 0.3226 |
| **global_zscore** | 0.8142 | 0.3038 | 0.2552 | 0.7511 | 0.3871 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (3 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7911 | 0.2750 | 0.2647 | 0.6419 | 0.3492 |
| **norm** | 0.8120 | 0.3099 | 0.2574 | 0.6943 | 0.3636 |
| global_zscore | 0.7661 | 0.2717 | 0.2636 | 0.6507 | 0.3333 |

---

## Model 3 — Clinic + Scanner + AEC  (3 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7888 | 0.2673 | 0.2644 | 0.7467 | 0.3830 |
| **norm** | 0.8382 | 0.3498 | 0.2629 | 0.6769 | 0.3833 |
| global_zscore | 0.7825 | 0.2675 | 0.2577 | 0.6900 | 0.3364 |

---

## Model 4 — AEC Only  (3 AEC variants)

> 임상 특징 없이 AEC 시퀀스만으로 분류.

### AECOnly

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **raw** | 0.6006 | 0.2166 | 0.3071 | 0.4760 | 0.2105 |
| norm | 0.5386 | 0.1193 | 0.3015 | 0.8210 | 0.0465 |
| global_zscore | 0.5852 | 0.1928 | 0.2956 | 0.4978 | 0.2069 |

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
| AUC-ROC  | 0.8061 | 0.8071 | +0.0009 | -0.057 | 9.57e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3751 | -0.0342 | 0.745 | 4.98e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2523 | +0.0715 | -20.304 | 3.47e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7681 | +0.0120 | -0.348 | 7.45e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4160 | -0.0004 | 0.011 | 9.92e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8097 | +0.0035 | -0.326 | 7.61e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3766 | -0.0326 | 1.095 | 3.35e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2474 | +0.0666 | -12.936 | 2.06e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7943 | +0.0383 | -1.892 | 1.31e-01 | 1.88e-01 |
| F1  | 0.4163 | 0.4547 | +0.0384 | -1.714 | 1.62e-01 | 3.12e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8107 | +0.0045 | -0.229 | 8.30e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3837 | -0.0255 | 0.630 | 5.63e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2443 | +0.0636 | -15.595 | 9.87e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7528 | -0.0033 | 0.063 | 9.53e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4290 | +0.0127 | -0.284 | 7.91e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8071 | 0.7975 | -0.0095 | 1.045 | 3.55e-01 | 6.25e-01 |
| AUPRC  | 0.3751 | 0.3785 | +0.0035 | -0.141 | 8.95e-01 | 1.00e+00 |
| Brier  | 0.2523 | 0.2516 | -0.0006 | 0.153 | 8.86e-01 | 1.00e+00 |
| Accuracy  | 0.7681 | 0.7463 | -0.0218 | 0.713 | 5.15e-01 | 5.00e-01 |
| F1  | 0.4160 | 0.4055 | -0.0104 | 0.415 | 6.99e-01 | 1.00e+00 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8097 | 0.7947 | -0.0150 | 2.102 | 1.03e-01 | 1.88e-01 |
| AUPRC  | 0.3766 | 0.3862 | +0.0096 | -0.444 | 6.80e-01 | 6.25e-01 |
| Brier  | 0.2474 | 0.2541 | +0.0067 | -0.550 | 6.12e-01 | 8.12e-01 |
| Accuracy  | 0.7943 | 0.7211 | -0.0732 | 1.570 | 1.92e-01 | 2.50e-01 |
| F1  | 0.4547 | 0.3867 | -0.0680 | 2.007 | 1.15e-01 | 1.25e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8107 | 0.7896 | -0.0210 | 1.424 | 2.28e-01 | 1.88e-01 |
| AUPRC  | 0.3837 | 0.3616 | -0.0221 | 0.826 | 4.55e-01 | 4.38e-01 |
| Brier  | 0.2443 | 0.2491 | +0.0048 | -1.414 | 2.30e-01 | 3.12e-01 |
| Accuracy * | 0.7528 | 0.6827 | -0.0700 | 2.928 | 4.29e-02 | 1.25e-01 |
| F1 * | 0.4290 | 0.3765 | -0.0525 | 3.365 | 2.82e-02 | 6.25e-02 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7975 | -0.0086 | 0.434 | 6.87e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3785 | -0.0307 | 0.727 | 5.08e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2516 | +0.0709 | -11.094 | 3.76e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7463 | -0.0098 | 0.183 | 8.64e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4055 | -0.0108 | 0.237 | 8.24e-01 | 8.12e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7947 | -0.0114 | 0.786 | 4.76e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3862 | -0.0231 | 0.569 | 5.99e-01 | 4.38e-01 |
| Brier ** | 0.1808 | 0.2541 | +0.0733 | -7.353 | 1.82e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7211 | -0.0350 | 0.637 | 5.59e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.3867 | -0.0296 | 0.800 | 4.68e-01 | 4.38e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7896 | -0.0165 | 0.779 | 4.79e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3616 | -0.0476 | 0.997 | 3.75e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2491 | +0.0683 | -14.156 | 1.45e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6827 | -0.0733 | 1.084 | 3.39e-01 | 5.00e-01 |
| F1  | 0.4163 | 0.3765 | -0.0399 | 0.829 | 4.54e-01 | 8.12e-01 |

## M1 (LR) vs M4 (AECOnly)

> A = M1 LR, B = M4 AECOnly.

### raw  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8061 | 0.6711 | -0.1351 | 3.568 | 2.34e-02 | 6.25e-02 |
| AUPRC * | 0.4092 | 0.2121 | -0.1971 | 3.674 | 2.13e-02 | 6.25e-02 |
| Brier *** | 0.1808 | 0.2965 | +0.1157 | -13.004 | 2.02e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6227 | -0.1334 | 1.809 | 1.45e-01 | 1.88e-01 |
| F1 † | 0.4163 | 0.2991 | -0.1173 | 2.481 | 6.81e-02 | 1.25e-01 |

### norm  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.8061 | 0.5657 | -0.2404 | 5.710 | 4.65e-03 | 6.25e-02 |
| AUPRC * | 0.4092 | 0.1419 | -0.2673 | 4.389 | 1.18e-02 | 6.25e-02 |
| Brier *** | 0.1808 | 0.3001 | +0.1193 | -9.871 | 5.91e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6676 | -0.0885 | 1.547 | 1.97e-01 | 1.88e-01 |
| F1 * | 0.4163 | 0.2415 | -0.1748 | 4.335 | 1.23e-02 | 6.25e-02 |

### global_zscore  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8061 | 0.6580 | -0.1482 | 3.936 | 1.70e-02 | 6.25e-02 |
| AUPRC * | 0.4092 | 0.2040 | -0.2053 | 3.874 | 1.79e-02 | 6.25e-02 |
| Brier ** | 0.1808 | 0.2738 | +0.0930 | -4.856 | 8.30e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6227 | -0.1334 | 1.342 | 2.51e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.3065 | -0.1098 | 2.044 | 1.10e-01 | 1.88e-01 |

## M4 (AECOnly) vs M2 (CrossAttn)

> A = M4 AECOnly, B = M2 CrossAttn. aec_var 키로 매칭.

#### Case: raw  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.6711 | 0.8071 | +0.1360 | -5.426 | 5.59e-03 | 6.25e-02 |
| AUPRC ** | 0.2121 | 0.3751 | +0.1630 | -7.556 | 1.64e-03 | 6.25e-02 |
| Brier ** | 0.2965 | 0.2523 | -0.0442 | 7.240 | 1.93e-03 | 6.25e-02 |
| Accuracy † | 0.6227 | 0.7681 | +0.1454 | -2.559 | 6.27e-02 | 6.25e-02 |
| F1 ** | 0.2991 | 0.4160 | +0.1169 | -5.524 | 5.25e-03 | 6.25e-02 |

#### Case: norm  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.5657 | 0.8097 | +0.2439 | -7.208 | 1.96e-03 | 6.25e-02 |
| AUPRC ** | 0.1419 | 0.3766 | +0.2347 | -4.787 | 8.73e-03 | 6.25e-02 |
| Brier * | 0.3001 | 0.2474 | -0.0527 | 3.511 | 2.46e-02 | 6.25e-02 |
| Accuracy  | 0.6676 | 0.7943 | +0.1267 | -2.122 | 1.01e-01 | 1.25e-01 |
| F1 ** | 0.2415 | 0.4547 | +0.2132 | -5.860 | 4.23e-03 | 6.25e-02 |

#### Case: global_zscore  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.6580 | 0.8107 | +0.1527 | -5.814 | 4.36e-03 | 6.25e-02 |
| AUPRC ** | 0.2040 | 0.3837 | +0.1797 | -6.763 | 2.49e-03 | 6.25e-02 |
| Brier  | 0.2738 | 0.2443 | -0.0294 | 1.442 | 2.23e-01 | 3.12e-01 |
| Accuracy  | 0.6227 | 0.7528 | +0.1301 | -1.635 | 1.77e-01 | 2.50e-01 |
| F1 † | 0.3065 | 0.4290 | +0.1225 | -2.611 | 5.94e-02 | 6.25e-02 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7783 | 0.6934 | 0.8560 |
| M2 | CrossAttn | raw | AUPRC | 0.2656 | 0.1558 | 0.4229 |
| M2 | CrossAttn | raw | Brier | 0.2621 | 0.2351 | 0.2877 |
| M2 | CrossAttn | raw | Accuracy | 0.7336 | 0.6769 | 0.7904 |
| M2 | CrossAttn | raw | F1 | 0.3579 | 0.2319 | 0.4753 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8028 | 0.7195 | 0.8733 |
| M2 | CrossAttn | norm | AUPRC | 0.3466 | 0.1976 | 0.5284 |
| M2 | CrossAttn | norm | Brier | 0.2661 | 0.2438 | 0.2872 |
| M2 | CrossAttn | norm | Accuracy | 0.7249 | 0.6681 | 0.7817 |
| M2 | CrossAttn | norm | F1 | 0.3226 | 0.1975 | 0.4400 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.8142 | 0.7418 | 0.8817 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.3038 | 0.1848 | 0.4758 |
| M2 | CrossAttn | global_zscore | Brier | 0.2552 | 0.2277 | 0.2809 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.7511 | 0.6943 | 0.8079 |
| M2 | CrossAttn | global_zscore | F1 | 0.3871 | 0.2558 | 0.5111 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7911 | 0.7126 | 0.8596 |
| M2_2 | CrossAttn | raw | AUPRC | 0.2750 | 0.1666 | 0.4609 |
| M2_2 | CrossAttn | raw | Brier | 0.2647 | 0.2379 | 0.2910 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6419 | 0.5852 | 0.7032 |
| M2_2 | CrossAttn | raw | F1 | 0.3492 | 0.2342 | 0.4604 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8120 | 0.7337 | 0.8776 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3099 | 0.1829 | 0.4826 |
| M2_2 | CrossAttn | norm | Brier | 0.2574 | 0.2322 | 0.2815 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6943 | 0.6376 | 0.7555 |
| M2_2 | CrossAttn | norm | F1 | 0.3636 | 0.2449 | 0.4741 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7661 | 0.6752 | 0.8449 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.2717 | 0.1520 | 0.4443 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2636 | 0.2395 | 0.2874 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6507 | 0.5895 | 0.7118 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3333 | 0.2167 | 0.4397 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7888 | 0.6947 | 0.8731 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2673 | 0.1752 | 0.4494 |
| M3 | CrossAttn3 | raw | Brier | 0.2644 | 0.2383 | 0.2882 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7467 | 0.6900 | 0.8035 |
| M3 | CrossAttn3 | raw | F1 | 0.3830 | 0.2564 | 0.5091 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8382 | 0.7697 | 0.8988 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3498 | 0.2131 | 0.5355 |
| M3 | CrossAttn3 | norm | Brier | 0.2629 | 0.2389 | 0.2852 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6769 | 0.6157 | 0.7380 |
| M3 | CrossAttn3 | norm | F1 | 0.3833 | 0.2653 | 0.4885 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7825 | 0.6895 | 0.8693 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2675 | 0.1746 | 0.4509 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2577 | 0.2319 | 0.2822 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.6900 | 0.6332 | 0.7511 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3364 | 0.2222 | 0.4510 |
| M4 | AECOnly | raw | AUC-ROC | 0.6006 | 0.4708 | 0.7290 |
| M4 | AECOnly | raw | AUPRC | 0.2166 | 0.1081 | 0.3825 |
| M4 | AECOnly | raw | Brier | 0.3071 | 0.2950 | 0.3197 |
| M4 | AECOnly | raw | Accuracy | 0.4760 | 0.4105 | 0.5416 |
| M4 | AECOnly | raw | F1 | 0.2105 | 0.1259 | 0.2987 |
| M4 | AECOnly | norm | AUC-ROC | 0.5386 | 0.4215 | 0.6518 |
| M4 | AECOnly | norm | AUPRC | 0.1193 | 0.0756 | 0.1978 |
| M4 | AECOnly | norm | Brier | 0.3015 | 0.2964 | 0.3060 |
| M4 | AECOnly | norm | Accuracy | 0.8210 | 0.7686 | 0.8690 |
| M4 | AECOnly | norm | F1 | 0.0465 | 0.0000 | 0.1463 |
| M4 | AECOnly | global_zscore | AUC-ROC | 0.5852 | 0.4565 | 0.7081 |
| M4 | AECOnly | global_zscore | AUPRC | 0.1928 | 0.0933 | 0.3357 |
| M4 | AECOnly | global_zscore | Brier | 0.2956 | 0.2868 | 0.3042 |
| M4 | AECOnly | global_zscore | Accuracy | 0.4978 | 0.4323 | 0.5633 |
| M4 | AECOnly | global_zscore | F1 | 0.2069 | 0.1231 | 0.2968 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7783 | -0.0248 | 1.009 | 3.130e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8028 | -0.0002 | 0.011 | 9.912e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.8142 | +0.0112 | -0.390 | 6.965e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7888 | -0.0142 | 0.379 | 7.047e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8382 | +0.0352 | -1.487 | 1.369e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7825 | -0.0205 | 0.605 | 5.453e-01 | ns |

## M1 LR vs M4 AECOnly

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M4-raw | 0.8030 | 0.6006 | -0.2024 | 2.621 | 8.772e-03 | ** |
| M1-LR vs M4-norm | 0.8030 | 0.5386 | -0.2644 | 4.108 | 3.993e-05 | *** |
| M1-LR vs M4-global_zscore | 0.8030 | 0.5852 | -0.2179 | 2.850 | 4.377e-03 | ** |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7783 | 0.7911 | +0.0128 | -0.701 | 4.830e-01 | ns |
| M2-norm vs M2_2-norm | 0.8028 | 0.8120 | +0.0091 | -0.562 | 5.741e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.8142 | 0.7661 | -0.0482 | 1.663 | 9.623e-02 | † |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7783 | 0.7888 | +0.0106 | -0.466 | 6.415e-01 | ns |
| M2-norm vs M3-norm | 0.8028 | 0.8382 | +0.0354 | -1.694 | 9.017e-02 | † |
| M2-global_zscore vs M3-global_zscore | 0.8142 | 0.7825 | -0.0317 | 1.095 | 2.736e-01 | ns |

## M4 AECOnly vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M4-raw vs M2-raw | 0.6006 | 0.7783 | +0.1776 | -2.758 | 5.811e-03 | ** |
| M4-norm vs M2-norm | 0.5386 | 0.8028 | +0.2642 | -4.463 | 8.077e-06 | *** |
| M4-global_zscore vs M2-global_zscore | 0.5852 | 0.8142 | +0.2291 | -3.452 | 5.559e-04 | *** |

