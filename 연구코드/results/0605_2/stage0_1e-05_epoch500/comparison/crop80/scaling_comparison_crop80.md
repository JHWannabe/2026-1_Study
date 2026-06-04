# Scaling Comparison — Test Set Performance (AEC 103pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8043 | 0.2900 | 0.2533 | 0.6769 | 0.3509 |
| M2_2 | CrossAttn | global_zscore | 0.8014 | 0.3212 | 0.2674 | 0.6332 | 0.3333 |
| M3 | CrossAttn3 | std_scaled | 0.8254 | 0.3490 | 0.2652 | 0.6987 | 0.3670 |

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
| raw | 0.7799 | 0.2513 | 0.2635 | 0.7118 | 0.3529 |
| std_scaled | 0.7978 | 0.3181 | 0.2640 | 0.7118 | 0.3529 |
| **norm** | 0.8043 | 0.2900 | 0.2533 | 0.6769 | 0.3509 |
| global_zscore | 0.7717 | 0.2590 | 0.2573 | 0.7293 | 0.3404 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7963 | 0.3005 | 0.2642 | 0.6376 | 0.3465 |
| std_scaled | 0.7654 | 0.2952 | 0.2611 | 0.6856 | 0.3333 |
| norm | 0.7963 | 0.3421 | 0.2653 | 0.6769 | 0.3509 |
| **global_zscore** | 0.8014 | 0.3212 | 0.2674 | 0.6332 | 0.3333 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7874 | 0.2886 | 0.2635 | 0.7380 | 0.3750 |
| **std_scaled** | 0.8254 | 0.3490 | 0.2652 | 0.6987 | 0.3670 |
| norm | 0.8173 | 0.2801 | 0.2533 | 0.7074 | 0.3619 |
| global_zscore | 0.7894 | 0.2934 | 0.2742 | 0.7205 | 0.3333 |

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
| AUC-ROC  | 0.8061 | 0.8066 | +0.0005 | -0.031 | 9.77e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3791 | -0.0302 | 0.704 | 5.20e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2529 | +0.0722 | -25.600 | 1.38e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7375 | -0.0186 | 0.799 | 4.69e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.3989 | -0.0175 | 0.685 | 5.31e-01 | 6.25e-01 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8142 | +0.0080 | -0.588 | 5.88e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3737 | -0.0355 | 0.817 | 4.60e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2486 | +0.0678 | -11.665 | 3.09e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7450 | -0.0110 | 0.235 | 8.26e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4124 | -0.0040 | 0.087 | 9.35e-01 | 8.12e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8036 | -0.0025 | 0.222 | 8.35e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3711 | -0.0381 | 1.238 | 2.83e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2478 | +0.0671 | -11.019 | 3.86e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7517 | -0.0044 | 0.356 | 7.40e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4123 | -0.0041 | 0.255 | 8.11e-01 | 1.00e+00 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8088 | +0.0027 | -0.166 | 8.76e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3896 | -0.0196 | 0.567 | 6.01e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2531 | +0.0724 | -32.629 | 5.26e-06 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7265 | -0.0296 | 0.856 | 4.40e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.3886 | -0.0278 | 0.904 | 4.17e-01 | 4.38e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8066 | 0.7984 | -0.0082 | 0.885 | 4.26e-01 | 8.12e-01 |
| AUPRC  | 0.3791 | 0.3772 | -0.0019 | 0.097 | 9.28e-01 | 1.00e+00 |
| Brier  | 0.2529 | 0.2530 | +0.0000 | -0.009 | 9.93e-01 | 1.00e+00 |
| Accuracy  | 0.7375 | 0.7288 | -0.0087 | 0.225 | 8.33e-01 | 8.12e-01 |
| F1  | 0.3989 | 0.3978 | -0.0011 | 0.040 | 9.70e-01 | 1.00e+00 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8142 | 0.7998 | -0.0143 | 1.149 | 3.15e-01 | 4.38e-01 |
| AUPRC  | 0.3737 | 0.3668 | -0.0070 | 0.195 | 8.55e-01 | 8.12e-01 |
| Brier  | 0.2486 | 0.2552 | +0.0066 | -0.877 | 4.30e-01 | 4.38e-01 |
| Accuracy  | 0.7450 | 0.7058 | -0.0392 | 0.707 | 5.19e-01 | 8.75e-01 |
| F1  | 0.4124 | 0.3783 | -0.0341 | 0.825 | 4.56e-01 | 8.12e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8036 | 0.8003 | -0.0033 | 0.473 | 6.61e-01 | 8.12e-01 |
| AUPRC  | 0.3711 | 0.3588 | -0.0123 | 0.450 | 6.76e-01 | 8.12e-01 |
| Brier  | 0.2478 | 0.2471 | -0.0007 | 0.198 | 8.53e-01 | 6.25e-01 |
| Accuracy  | 0.7517 | 0.7582 | +0.0065 | -0.237 | 8.24e-01 | 8.75e-01 |
| F1  | 0.4123 | 0.4043 | -0.0080 | 0.486 | 6.52e-01 | 8.12e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.8088 | 0.7910 | -0.0178 | 5.397 | 5.70e-03 | 6.25e-02 |
| AUPRC  | 0.3896 | 0.3540 | -0.0356 | 1.957 | 1.22e-01 | 1.88e-01 |
| Brier  | 0.2531 | 0.2516 | -0.0015 | 0.316 | 7.68e-01 | 6.25e-01 |
| Accuracy  | 0.7265 | 0.7594 | +0.0329 | -0.801 | 4.68e-01 | 6.25e-01 |
| F1  | 0.3886 | 0.4160 | +0.0275 | -1.363 | 2.45e-01 | 3.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7984 | -0.0077 | 0.369 | 7.31e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3772 | -0.0320 | 0.832 | 4.52e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2530 | +0.0722 | -12.258 | 2.54e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7288 | -0.0273 | 0.467 | 6.65e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3978 | -0.0186 | 0.387 | 7.18e-01 | 6.25e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7998 | -0.0063 | 0.381 | 7.22e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3668 | -0.0425 | 0.971 | 3.86e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2552 | +0.0745 | -14.076 | 1.48e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7058 | -0.0503 | 1.060 | 3.49e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.3783 | -0.0381 | 1.158 | 3.11e-01 | 3.12e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8003 | -0.0058 | 0.646 | 5.53e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3588 | -0.0504 | 1.295 | 2.65e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2471 | +0.0664 | -9.809 | 6.06e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7582 | +0.0021 | -0.074 | 9.44e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4043 | -0.0121 | 0.585 | 5.90e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7910 | -0.0152 | 0.873 | 4.32e-01 | 3.12e-01 |
| AUPRC  | 0.4092 | 0.3540 | -0.0552 | 1.192 | 2.99e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2516 | +0.0709 | -12.153 | 2.63e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7594 | +0.0034 | -0.078 | 9.42e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4160 | -0.0003 | 0.012 | 9.91e-01 | 1.00e+00 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7799 | 0.6961 | 0.8567 |
| M2 | CrossAttn | raw | AUPRC | 0.2513 | 0.1595 | 0.4208 |
| M2 | CrossAttn | raw | Brier | 0.2635 | 0.2366 | 0.2887 |
| M2 | CrossAttn | raw | Accuracy | 0.7118 | 0.6550 | 0.7686 |
| M2 | CrossAttn | raw | F1 | 0.3529 | 0.2307 | 0.4737 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7978 | 0.7135 | 0.8683 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.3181 | 0.1788 | 0.4830 |
| M2 | CrossAttn | std_scaled | Brier | 0.2640 | 0.2364 | 0.2902 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7118 | 0.6507 | 0.7686 |
| M2 | CrossAttn | std_scaled | F1 | 0.3529 | 0.2299 | 0.4667 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8043 | 0.7236 | 0.8729 |
| M2 | CrossAttn | norm | AUPRC | 0.2900 | 0.1829 | 0.4772 |
| M2 | CrossAttn | norm | Brier | 0.2533 | 0.2281 | 0.2771 |
| M2 | CrossAttn | norm | Accuracy | 0.6769 | 0.6157 | 0.7380 |
| M2 | CrossAttn | norm | F1 | 0.3509 | 0.2340 | 0.4640 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7717 | 0.6879 | 0.8464 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2590 | 0.1531 | 0.4162 |
| M2 | CrossAttn | global_zscore | Brier | 0.2573 | 0.2303 | 0.2834 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.7293 | 0.6725 | 0.7860 |
| M2 | CrossAttn | global_zscore | F1 | 0.3404 | 0.2117 | 0.4578 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7963 | 0.7179 | 0.8647 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3005 | 0.1750 | 0.4778 |
| M2_2 | CrossAttn | raw | Brier | 0.2642 | 0.2381 | 0.2892 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6376 | 0.5764 | 0.6987 |
| M2_2 | CrossAttn | raw | F1 | 0.3465 | 0.2314 | 0.4561 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7654 | 0.6780 | 0.8491 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.2952 | 0.1634 | 0.4826 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2611 | 0.2347 | 0.2870 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6856 | 0.6245 | 0.7467 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3333 | 0.2173 | 0.4500 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.7963 | 0.7084 | 0.8685 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3421 | 0.1928 | 0.5357 |
| M2_2 | CrossAttn | norm | Brier | 0.2653 | 0.2426 | 0.2873 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6769 | 0.6157 | 0.7380 |
| M2_2 | CrossAttn | norm | F1 | 0.3509 | 0.2330 | 0.4616 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.8014 | 0.7162 | 0.8711 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3212 | 0.1839 | 0.5031 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2674 | 0.2412 | 0.2937 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6332 | 0.5721 | 0.6943 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3333 | 0.2203 | 0.4462 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7874 | 0.6896 | 0.8750 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2886 | 0.1831 | 0.4815 |
| M3 | CrossAttn3 | raw | Brier | 0.2635 | 0.2383 | 0.2866 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7380 | 0.6812 | 0.7948 |
| M3 | CrossAttn3 | raw | F1 | 0.3750 | 0.2500 | 0.5000 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.8254 | 0.7521 | 0.8947 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.3490 | 0.2097 | 0.5427 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2652 | 0.2394 | 0.2899 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.6987 | 0.6376 | 0.7598 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3670 | 0.2453 | 0.4800 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8173 | 0.7505 | 0.8797 |
| M3 | CrossAttn3 | norm | AUPRC | 0.2801 | 0.1810 | 0.4651 |
| M3 | CrossAttn3 | norm | Brier | 0.2533 | 0.2273 | 0.2768 |
| M3 | CrossAttn3 | norm | Accuracy | 0.7074 | 0.6507 | 0.7686 |
| M3 | CrossAttn3 | norm | F1 | 0.3619 | 0.2400 | 0.4762 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7894 | 0.6950 | 0.8768 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2934 | 0.1814 | 0.4890 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2742 | 0.2510 | 0.2951 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7205 | 0.6638 | 0.7773 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3333 | 0.2168 | 0.4555 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7799 | -0.0232 | 0.925 | 3.547e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7978 | -0.0053 | 0.253 | 8.000e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8043 | +0.0012 | -0.054 | 9.569e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7717 | -0.0313 | 1.394 | 1.633e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7874 | -0.0157 | 0.386 | 6.998e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.8254 | +0.0224 | -0.772 | 4.401e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8173 | +0.0142 | -0.615 | 5.386e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7894 | -0.0136 | 0.370 | 7.111e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7799 | 0.7963 | +0.0165 | -0.890 | 3.736e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7978 | 0.7654 | -0.0323 | 1.073 | 2.834e-01 | ns |
| M2-norm vs M2_2-norm | 0.8043 | 0.7963 | -0.0079 | 0.281 | 7.785e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.7717 | 0.8014 | +0.0297 | -1.188 | 2.347e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7799 | 0.7874 | +0.0075 | -0.292 | 7.705e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7978 | 0.8254 | +0.0276 | -1.471 | 1.413e-01 | ns |
| M2-norm vs M3-norm | 0.8043 | 0.8173 | +0.0130 | -1.000 | 3.172e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.7717 | 0.7894 | +0.0177 | -0.580 | 5.617e-01 | ns |

