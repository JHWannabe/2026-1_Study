# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8110 | 0.2844 | 0.2537 | 0.7118 | 0.3654 |
| M2_2 | CrossAttn | norm | 0.8411 | 0.3541 | 0.2612 | 0.6376 | 0.3566 |
| M3 | CrossAttn3 | norm | 0.8087 | 0.2814 | 0.2519 | 0.6419 | 0.3492 |

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
| raw | 0.8004 | 0.2908 | 0.2576 | 0.7336 | 0.3579 |
| std_scaled | 0.8037 | 0.2979 | 0.2577 | 0.7293 | 0.3673 |
| **norm** | 0.8110 | 0.2844 | 0.2537 | 0.7118 | 0.3654 |
| global_zscore | 0.8028 | 0.3005 | 0.2508 | 0.7380 | 0.3750 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7839 | 0.3094 | 0.2689 | 0.6550 | 0.3361 |
| std_scaled | 0.7943 | 0.3313 | 0.2615 | 0.7074 | 0.3495 |
| **norm** | 0.8411 | 0.3541 | 0.2612 | 0.6376 | 0.3566 |
| global_zscore | 0.7996 | 0.3247 | 0.2642 | 0.5852 | 0.3165 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7921 | 0.2884 | 0.2632 | 0.7729 | 0.3953 |
| std_scaled | 0.8075 | 0.3034 | 0.2481 | 0.7686 | 0.4045 |
| **norm** | 0.8087 | 0.2814 | 0.2519 | 0.6419 | 0.3492 |
| global_zscore | 0.8000 | 0.2660 | 0.2661 | 0.6856 | 0.3455 |

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
| AUC-ROC  | 0.8061 | 0.8099 | +0.0038 | -0.212 | 8.43e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3718 | -0.0375 | 1.297 | 2.65e-01 | 1.88e-01 |
| Brier *** | 0.1808 | 0.2403 | +0.0595 | -12.684 | 2.23e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7561 | -0.0000 | 0.001 | 1.00e+00 | 1.00e+00 |
| F1  | 0.4163 | 0.4203 | +0.0040 | -0.110 | 9.17e-01 | 1.00e+00 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8203 | +0.0141 | -0.974 | 3.85e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4146 | +0.0053 | -0.204 | 8.48e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2397 | +0.0589 | -12.521 | 2.34e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7647 | +0.0086 | -0.213 | 8.41e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4307 | +0.0143 | -0.349 | 7.45e-01 | 8.12e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8079 | +0.0018 | -0.137 | 8.97e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3796 | -0.0297 | 0.727 | 5.07e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2437 | +0.0630 | -18.148 | 5.42e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7473 | -0.0088 | 0.777 | 4.81e-01 | 7.50e-01 |
| F1  | 0.4163 | 0.4119 | -0.0044 | 0.259 | 8.08e-01 | 8.12e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8135 | +0.0074 | -0.426 | 6.92e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3773 | -0.0320 | 0.784 | 4.77e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2390 | +0.0582 | -14.804 | 1.21e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7604 | +0.0043 | -0.109 | 9.18e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4231 | +0.0067 | -0.206 | 8.47e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8099 | 0.8095 | -0.0004 | 0.064 | 9.52e-01 | 8.12e-01 |
| AUPRC  | 0.3718 | 0.3738 | +0.0020 | -0.093 | 9.31e-01 | 1.00e+00 |
| Brier  | 0.2403 | 0.2382 | -0.0021 | 0.378 | 7.24e-01 | 8.12e-01 |
| Accuracy  | 0.7561 | 0.7965 | +0.0405 | -1.020 | 3.66e-01 | 3.75e-01 |
| F1  | 0.4203 | 0.4435 | +0.0232 | -0.776 | 4.81e-01 | 8.12e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8203 | 0.8168 | -0.0035 | 0.897 | 4.20e-01 | 6.25e-01 |
| AUPRC  | 0.4146 | 0.4114 | -0.0032 | 0.220 | 8.36e-01 | 1.00e+00 |
| Brier  | 0.2397 | 0.2406 | +0.0009 | -0.110 | 9.18e-01 | 1.00e+00 |
| Accuracy  | 0.7647 | 0.7911 | +0.0264 | -0.724 | 5.09e-01 | 6.25e-01 |
| F1  | 0.4307 | 0.4427 | +0.0121 | -0.401 | 7.09e-01 | 1.00e+00 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8079 | 0.7987 | -0.0093 | 1.187 | 3.01e-01 | 4.38e-01 |
| AUPRC  | 0.3796 | 0.3735 | -0.0060 | 0.282 | 7.92e-01 | 6.25e-01 |
| Brier  | 0.2437 | 0.2466 | +0.0029 | -0.726 | 5.08e-01 | 8.12e-01 |
| Accuracy  | 0.7473 | 0.7353 | -0.0120 | 0.409 | 7.04e-01 | 8.12e-01 |
| F1  | 0.4119 | 0.3961 | -0.0158 | 0.817 | 4.60e-01 | 6.25e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8135 | 0.8150 | +0.0015 | -0.343 | 7.49e-01 | 1.00e+00 |
| AUPRC  | 0.3773 | 0.3900 | +0.0128 | -0.523 | 6.29e-01 | 4.38e-01 |
| Brier  | 0.2390 | 0.2387 | -0.0003 | 0.076 | 9.43e-01 | 8.12e-01 |
| Accuracy  | 0.7604 | 0.7702 | +0.0098 | -0.245 | 8.19e-01 | 8.75e-01 |
| F1  | 0.4231 | 0.4207 | -0.0023 | 0.074 | 9.45e-01 | 8.75e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8095 | +0.0033 | -0.182 | 8.65e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3738 | -0.0354 | 0.829 | 4.54e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2382 | +0.0575 | -16.248 | 8.39e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7965 | +0.0404 | -1.484 | 2.12e-01 | 2.50e-01 |
| F1  | 0.4163 | 0.4435 | +0.0272 | -1.165 | 3.09e-01 | 3.12e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8168 | +0.0106 | -0.706 | 5.19e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4114 | +0.0022 | -0.056 | 9.58e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2406 | +0.0598 | -10.700 | 4.32e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7911 | +0.0350 | -0.823 | 4.57e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4427 | +0.0264 | -0.812 | 4.62e-01 | 6.25e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7987 | -0.0075 | 0.516 | 6.33e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3735 | -0.0357 | 1.265 | 2.75e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2466 | +0.0659 | -9.744 | 6.21e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7353 | -0.0208 | 0.626 | 5.65e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3961 | -0.0202 | 0.774 | 4.82e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8150 | +0.0089 | -0.559 | 6.06e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3900 | -0.0192 | 0.384 | 7.21e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2387 | +0.0579 | -10.761 | 4.23e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7702 | +0.0141 | -0.269 | 8.01e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4207 | +0.0044 | -0.101 | 9.24e-01 | 1.00e+00 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.8004 | 0.7174 | 0.8770 |
| M2 | CrossAttn | raw | AUPRC | 0.2908 | 0.1740 | 0.4576 |
| M2 | CrossAttn | raw | Brier | 0.2576 | 0.2299 | 0.2843 |
| M2 | CrossAttn | raw | Accuracy | 0.7336 | 0.6812 | 0.7904 |
| M2 | CrossAttn | raw | F1 | 0.3579 | 0.2352 | 0.4773 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.8037 | 0.7161 | 0.8772 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2979 | 0.1787 | 0.4677 |
| M2 | CrossAttn | std_scaled | Brier | 0.2577 | 0.2300 | 0.2839 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7293 | 0.6725 | 0.7861 |
| M2 | CrossAttn | std_scaled | F1 | 0.3673 | 0.2418 | 0.4870 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8110 | 0.7373 | 0.8781 |
| M2 | CrossAttn | norm | AUPRC | 0.2844 | 0.1803 | 0.4758 |
| M2 | CrossAttn | norm | Brier | 0.2537 | 0.2270 | 0.2801 |
| M2 | CrossAttn | norm | Accuracy | 0.7118 | 0.6550 | 0.7729 |
| M2 | CrossAttn | norm | F1 | 0.3654 | 0.2400 | 0.4844 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.8028 | 0.7157 | 0.8804 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.3005 | 0.1807 | 0.4748 |
| M2 | CrossAttn | global_zscore | Brier | 0.2508 | 0.2231 | 0.2771 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.7380 | 0.6812 | 0.7948 |
| M2 | CrossAttn | global_zscore | F1 | 0.3750 | 0.2500 | 0.5000 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7839 | 0.6987 | 0.8613 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3094 | 0.1725 | 0.4918 |
| M2_2 | CrossAttn | raw | Brier | 0.2689 | 0.2417 | 0.2950 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6550 | 0.5939 | 0.7162 |
| M2_2 | CrossAttn | raw | F1 | 0.3361 | 0.2222 | 0.4483 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7943 | 0.7000 | 0.8720 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3313 | 0.1933 | 0.5188 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2615 | 0.2346 | 0.2871 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.7074 | 0.6463 | 0.7686 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3495 | 0.2278 | 0.4667 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8411 | 0.7654 | 0.9030 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3541 | 0.2206 | 0.5662 |
| M2_2 | CrossAttn | norm | Brier | 0.2612 | 0.2362 | 0.2854 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6376 | 0.5721 | 0.7031 |
| M2_2 | CrossAttn | norm | F1 | 0.3566 | 0.2435 | 0.4655 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7996 | 0.7104 | 0.8744 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3247 | 0.1860 | 0.5089 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2642 | 0.2375 | 0.2898 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.5852 | 0.5197 | 0.6507 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3165 | 0.2148 | 0.4173 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7921 | 0.7037 | 0.8723 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2884 | 0.1741 | 0.4585 |
| M3 | CrossAttn3 | raw | Brier | 0.2632 | 0.2361 | 0.2890 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7729 | 0.7205 | 0.8254 |
| M3 | CrossAttn3 | raw | F1 | 0.3953 | 0.2653 | 0.5208 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.8075 | 0.7105 | 0.8834 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.3034 | 0.1813 | 0.4777 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2481 | 0.2199 | 0.2752 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.7686 | 0.7162 | 0.8210 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.4045 | 0.2750 | 0.5319 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8087 | 0.7253 | 0.8807 |
| M3 | CrossAttn3 | norm | AUPRC | 0.2814 | 0.1811 | 0.4650 |
| M3 | CrossAttn3 | norm | Brier | 0.2519 | 0.2255 | 0.2765 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6419 | 0.5764 | 0.7074 |
| M3 | CrossAttn3 | norm | F1 | 0.3492 | 0.2342 | 0.4595 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.8000 | 0.7152 | 0.8776 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2660 | 0.1745 | 0.4385 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2661 | 0.2387 | 0.2915 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.6856 | 0.6287 | 0.7467 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3455 | 0.2301 | 0.4576 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.8004 | -0.0026 | 0.107 | 9.149e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.8037 | +0.0006 | -0.025 | 9.804e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8110 | +0.0079 | -0.337 | 7.361e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.8028 | -0.0002 | 0.008 | 9.934e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7921 | -0.0110 | 0.391 | 6.957e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.8075 | +0.0045 | -0.176 | 8.604e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8087 | +0.0057 | -0.225 | 8.223e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.8000 | -0.0030 | 0.110 | 9.122e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.8004 | 0.7839 | -0.0165 | 0.744 | 4.572e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.8037 | 0.7943 | -0.0093 | 0.416 | 6.777e-01 | ns |
| M2-norm vs M2_2-norm | 0.8110 | 0.8411 | +0.0301 | -1.582 | 1.137e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.8028 | 0.7996 | -0.0033 | 0.134 | 8.937e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.8004 | 0.7921 | -0.0083 | 0.599 | 5.491e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.8037 | 0.8075 | +0.0039 | -0.333 | 7.394e-01 | ns |
| M2-norm vs M3-norm | 0.8110 | 0.8087 | -0.0022 | 0.118 | 9.060e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.8028 | 0.8000 | -0.0028 | 0.288 | 7.732e-01 | ns |

