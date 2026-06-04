# Scaling Comparison — Test Set Performance (AEC 103pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | std_scaled | 0.7927 | 0.2990 | 0.2667 | 0.7336 | 0.3711 |
| M2_2 | CrossAttn | norm | 0.8325 | 0.3568 | 0.2581 | 0.7424 | 0.4158 |
| M3 | CrossAttn3 | norm | 0.8224 | 0.3163 | 0.2592 | 0.6900 | 0.3717 |

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
| raw | 0.7801 | 0.2540 | 0.2637 | 0.7467 | 0.3696 |
| **std_scaled** | 0.7927 | 0.2990 | 0.2667 | 0.7336 | 0.3711 |
| norm | 0.7894 | 0.2787 | 0.2648 | 0.6725 | 0.3802 |
| global_zscore | 0.7795 | 0.2810 | 0.2611 | 0.6900 | 0.3238 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7929 | 0.3139 | 0.2709 | 0.6681 | 0.3559 |
| std_scaled | 0.7795 | 0.3121 | 0.2711 | 0.6419 | 0.3167 |
| **norm** | 0.8325 | 0.3568 | 0.2581 | 0.7424 | 0.4158 |
| global_zscore | 0.7929 | 0.2978 | 0.2608 | 0.6638 | 0.3636 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7762 | 0.2569 | 0.2606 | 0.6201 | 0.3256 |
| std_scaled | 0.7348 | 0.2198 | 0.2823 | 0.6201 | 0.2810 |
| **norm** | 0.8224 | 0.3163 | 0.2592 | 0.6900 | 0.3717 |
| global_zscore | 0.7224 | 0.2333 | 0.2765 | 0.7511 | 0.3294 |

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
| AUC-ROC  | 0.8061 | 0.7949 | -0.0113 | 0.519 | 6.31e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3787 | -0.0306 | 0.694 | 5.26e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2617 | +0.0810 | -13.043 | 1.99e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7517 | -0.0044 | 0.107 | 9.20e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3957 | -0.0206 | 0.678 | 5.35e-01 | 4.38e-01 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8116 | +0.0055 | -0.417 | 6.98e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3822 | -0.0270 | 0.803 | 4.67e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2582 | +0.0774 | -16.564 | 7.78e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7452 | -0.0109 | 0.232 | 8.28e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4118 | -0.0046 | 0.143 | 8.93e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8003 | -0.0059 | 0.675 | 5.37e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3662 | -0.0430 | 0.802 | 4.67e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2526 | +0.0719 | -12.196 | 2.59e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7133 | -0.0428 | 1.822 | 1.43e-01 | 6.25e-02 |
| F1  | 0.4163 | 0.3873 | -0.0290 | 1.212 | 2.92e-01 | 8.12e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7923 | -0.0139 | 0.616 | 5.71e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3894 | -0.0198 | 0.411 | 7.02e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2589 | +0.0782 | -19.661 | 3.95e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7429 | -0.0132 | 0.242 | 8.21e-01 | 8.75e-01 |
| F1  | 0.4163 | 0.3983 | -0.0181 | 0.350 | 7.44e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.7949 | 0.7816 | -0.0133 | 1.073 | 3.44e-01 | 3.12e-01 |
| AUPRC  | 0.3787 | 0.3324 | -0.0462 | 1.789 | 1.48e-01 | 1.88e-01 |
| Brier  | 0.2617 | 0.2603 | -0.0014 | 0.229 | 8.30e-01 | 8.12e-01 |
| Accuracy  | 0.7517 | 0.6675 | -0.0842 | 1.518 | 2.04e-01 | 3.12e-01 |
| F1  | 0.3957 | 0.3585 | -0.0372 | 1.335 | 2.53e-01 | 4.38e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8116 | 0.7836 | -0.0280 | 2.547 | 6.35e-02 | 6.25e-02 |
| AUPRC  | 0.3822 | 0.3889 | +0.0067 | -0.291 | 7.85e-01 | 8.12e-01 |
| Brier  | 0.2582 | 0.2626 | +0.0044 | -1.042 | 3.56e-01 | 6.25e-01 |
| Accuracy  | 0.7452 | 0.7473 | +0.0021 | -0.057 | 9.58e-01 | 1.00e+00 |
| F1  | 0.4118 | 0.3903 | -0.0215 | 1.002 | 3.73e-01 | 4.38e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8003 | 0.7996 | -0.0007 | 0.075 | 9.44e-01 | 1.00e+00 |
| AUPRC  | 0.3662 | 0.3915 | +0.0253 | -0.874 | 4.31e-01 | 6.25e-01 |
| Brier  | 0.2526 | 0.2625 | +0.0099 | -1.072 | 3.44e-01 | 4.38e-01 |
| Accuracy  | 0.7133 | 0.7332 | +0.0199 | -0.487 | 6.52e-01 | 8.12e-01 |
| F1  | 0.3873 | 0.3966 | +0.0092 | -0.334 | 7.55e-01 | 1.00e+00 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.7923 | 0.7783 | -0.0140 | 2.427 | 7.22e-02 | 6.25e-02 |
| AUPRC  | 0.3894 | 0.3449 | -0.0445 | 1.156 | 3.12e-01 | 6.25e-01 |
| Brier † | 0.2589 | 0.2692 | +0.0103 | -2.511 | 6.60e-02 | 1.25e-01 |
| Accuracy  | 0.7429 | 0.7342 | -0.0086 | 0.221 | 8.36e-01 | 1.00e+00 |
| F1  | 0.3983 | 0.3814 | -0.0168 | 1.032 | 3.60e-01 | 3.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7816 | -0.0246 | 0.994 | 3.76e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3324 | -0.0768 | 1.565 | 1.93e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2603 | +0.0796 | -13.063 | 1.98e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6675 | -0.0885 | 1.019 | 3.66e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.3585 | -0.0578 | 1.119 | 3.26e-01 | 3.12e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7836 | -0.0225 | 1.213 | 2.92e-01 | 3.12e-01 |
| AUPRC  | 0.4092 | 0.3889 | -0.0203 | 0.512 | 6.36e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2626 | +0.0819 | -12.942 | 2.06e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7473 | -0.0088 | 0.171 | 8.72e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.3903 | -0.0261 | 0.631 | 5.62e-01 | 6.25e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7996 | -0.0066 | 0.612 | 5.74e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3915 | -0.0178 | 0.471 | 6.62e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2625 | +0.0818 | -10.221 | 5.16e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7332 | -0.0229 | 0.851 | 4.43e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.3966 | -0.0198 | 1.069 | 3.45e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7783 | -0.0278 | 1.027 | 3.63e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3449 | -0.0643 | 1.124 | 3.24e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2692 | +0.0884 | -11.687 | 3.06e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7342 | -0.0218 | 0.371 | 7.29e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.3814 | -0.0349 | 0.745 | 4.98e-01 | 6.25e-01 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7801 | 0.6860 | 0.8646 |
| M2 | CrossAttn | raw | AUPRC | 0.2540 | 0.1604 | 0.4226 |
| M2 | CrossAttn | raw | Brier | 0.2637 | 0.2368 | 0.2886 |
| M2 | CrossAttn | raw | Accuracy | 0.7467 | 0.6943 | 0.8035 |
| M2 | CrossAttn | raw | F1 | 0.3696 | 0.2469 | 0.4894 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7927 | 0.7041 | 0.8700 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2990 | 0.1749 | 0.4702 |
| M2 | CrossAttn | std_scaled | Brier | 0.2667 | 0.2403 | 0.2918 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7336 | 0.6769 | 0.7904 |
| M2 | CrossAttn | std_scaled | F1 | 0.3711 | 0.2453 | 0.4909 |
| M2 | CrossAttn | norm | AUC-ROC | 0.7894 | 0.7148 | 0.8576 |
| M2 | CrossAttn | norm | AUPRC | 0.2787 | 0.1687 | 0.4645 |
| M2 | CrossAttn | norm | Brier | 0.2648 | 0.2403 | 0.2883 |
| M2 | CrossAttn | norm | Accuracy | 0.6725 | 0.6114 | 0.7380 |
| M2 | CrossAttn | norm | F1 | 0.3802 | 0.2641 | 0.4952 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7795 | 0.6941 | 0.8604 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2810 | 0.1658 | 0.4544 |
| M2 | CrossAttn | global_zscore | Brier | 0.2611 | 0.2351 | 0.2854 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.6900 | 0.6332 | 0.7511 |
| M2 | CrossAttn | global_zscore | F1 | 0.3238 | 0.2083 | 0.4386 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7929 | 0.6996 | 0.8690 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3139 | 0.1796 | 0.4944 |
| M2_2 | CrossAttn | raw | Brier | 0.2709 | 0.2438 | 0.2967 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6681 | 0.6070 | 0.7293 |
| M2_2 | CrossAttn | raw | F1 | 0.3559 | 0.2406 | 0.4715 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7795 | 0.6940 | 0.8548 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3121 | 0.1752 | 0.4943 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2711 | 0.2478 | 0.2935 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6419 | 0.5808 | 0.7074 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3167 | 0.2087 | 0.4259 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8325 | 0.7568 | 0.8952 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3568 | 0.2106 | 0.5511 |
| M2_2 | CrossAttn | norm | Brier | 0.2581 | 0.2314 | 0.2836 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7424 | 0.6856 | 0.7991 |
| M2_2 | CrossAttn | norm | F1 | 0.4158 | 0.2857 | 0.5334 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7929 | 0.7093 | 0.8667 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.2978 | 0.1714 | 0.4759 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2608 | 0.2340 | 0.2861 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6638 | 0.6070 | 0.7249 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3636 | 0.2459 | 0.4776 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7762 | 0.6772 | 0.8643 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2569 | 0.1666 | 0.4269 |
| M3 | CrossAttn3 | raw | Brier | 0.2606 | 0.2345 | 0.2851 |
| M3 | CrossAttn3 | raw | Accuracy | 0.6201 | 0.5546 | 0.6856 |
| M3 | CrossAttn3 | raw | F1 | 0.3256 | 0.2162 | 0.4341 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7348 | 0.6308 | 0.8306 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2198 | 0.1387 | 0.3758 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2823 | 0.2609 | 0.3027 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.6201 | 0.5546 | 0.6856 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.2810 | 0.1786 | 0.3934 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8224 | 0.7421 | 0.8894 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3163 | 0.1976 | 0.5225 |
| M3 | CrossAttn3 | norm | Brier | 0.2592 | 0.2326 | 0.2834 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6900 | 0.6288 | 0.7511 |
| M3 | CrossAttn3 | norm | F1 | 0.3717 | 0.2476 | 0.4848 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7224 | 0.6154 | 0.8281 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2333 | 0.1447 | 0.4183 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2765 | 0.2579 | 0.2936 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7511 | 0.6943 | 0.8079 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3294 | 0.1999 | 0.4615 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7801 | -0.0230 | 0.862 | 3.888e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7927 | -0.0104 | 0.443 | 6.574e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.7894 | -0.0136 | 0.689 | 4.907e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7795 | -0.0236 | 0.844 | 3.985e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7762 | -0.0268 | 0.845 | 3.982e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7348 | -0.0683 | 1.466 | 1.426e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8224 | +0.0193 | -0.811 | 4.176e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7224 | -0.0807 | 1.728 | 8.394e-02 | † |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7801 | 0.7929 | +0.0128 | -0.536 | 5.916e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7927 | 0.7795 | -0.0132 | 0.521 | 6.026e-01 | ns |
| M2-norm vs M2_2-norm | 0.7894 | 0.8325 | +0.0431 | -1.965 | 4.940e-02 | * |
| M2-global_zscore vs M2_2-global_zscore | 0.7795 | 0.7929 | +0.0134 | -0.599 | 5.494e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7801 | 0.7762 | -0.0039 | 0.217 | 8.280e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7927 | 0.7348 | -0.0579 | 1.840 | 6.571e-02 | † |
| M2-norm vs M3-norm | 0.7894 | 0.8224 | +0.0329 | -1.722 | 8.511e-02 | † |
| M2-global_zscore vs M3-global_zscore | 0.7795 | 0.7224 | -0.0571 | 1.879 | 6.030e-02 | † |

