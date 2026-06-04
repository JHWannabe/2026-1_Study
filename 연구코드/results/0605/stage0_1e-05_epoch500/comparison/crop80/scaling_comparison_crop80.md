# Scaling Comparison — Test Set Performance (AEC 103pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8189 | 0.3443 | 0.2576 | 0.7424 | 0.3656 |
| M2_2 | CrossAttn | norm | 0.8301 | 0.3580 | 0.2594 | 0.6812 | 0.3652 |
| M3 | CrossAttn3 | std_scaled | 0.7827 | 0.2583 | 0.2671 | 0.6812 | 0.3303 |

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
| raw | 0.7898 | 0.2881 | 0.2568 | 0.7031 | 0.3704 |
| std_scaled | 0.7620 | 0.2578 | 0.2704 | 0.7860 | 0.3951 |
| **norm** | 0.8189 | 0.3443 | 0.2576 | 0.7424 | 0.3656 |
| global_zscore | 0.8024 | 0.2779 | 0.2649 | 0.7205 | 0.3846 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7978 | 0.3343 | 0.2636 | 0.6288 | 0.3411 |
| std_scaled | 0.7805 | 0.2861 | 0.2679 | 0.7118 | 0.3654 |
| **norm** | 0.8301 | 0.3580 | 0.2594 | 0.6812 | 0.3652 |
| global_zscore | 0.8236 | 0.3680 | 0.2724 | 0.6812 | 0.3652 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7520 | 0.2399 | 0.2795 | 0.6332 | 0.2759 |
| **std_scaled** | 0.7827 | 0.2583 | 0.2671 | 0.6812 | 0.3303 |
| norm | 0.7740 | 0.2889 | 0.2661 | 0.6507 | 0.3443 |
| global_zscore | 0.7382 | 0.2082 | 0.2829 | 0.6725 | 0.3478 |

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
| AUC-ROC  | 0.8061 | 0.8087 | +0.0026 | -0.141 | 8.95e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3841 | -0.0251 | 0.630 | 5.63e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2588 | +0.0781 | -11.955 | 2.81e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7352 | -0.0209 | 0.799 | 4.69e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4088 | -0.0075 | 0.311 | 7.71e-01 | 1.00e+00 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8071 | +0.0009 | -0.077 | 9.42e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3840 | -0.0252 | 0.665 | 5.43e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2542 | +0.0735 | -18.196 | 5.36e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7572 | +0.0011 | -0.028 | 9.79e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4165 | +0.0002 | -0.005 | 9.96e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8045 | -0.0017 | 0.218 | 8.38e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3774 | -0.0318 | 1.094 | 3.35e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2513 | +0.0706 | -19.986 | 3.70e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7747 | +0.0186 | -0.874 | 4.32e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4266 | +0.0102 | -0.824 | 4.56e-01 | 4.38e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7928 | -0.0134 | 0.632 | 5.62e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3512 | -0.0580 | 1.278 | 2.70e-01 | 3.12e-01 |
| Brier ** | 0.1808 | 0.2589 | +0.0781 | -8.048 | 1.29e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7047 | -0.0514 | 1.691 | 1.66e-01 | 1.88e-01 |
| F1  | 0.4163 | 0.3761 | -0.0403 | 1.258 | 2.77e-01 | 3.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8087 | 0.7716 | -0.0371 | 1.824 | 1.42e-01 | 1.25e-01 |
| AUPRC  | 0.3841 | 0.3231 | -0.0610 | 1.258 | 2.77e-01 | 3.12e-01 |
| Brier  | 0.2588 | 0.2665 | +0.0077 | -1.654 | 1.74e-01 | 1.88e-01 |
| Accuracy  | 0.7352 | 0.6818 | -0.0535 | 0.602 | 5.80e-01 | 1.00e+00 |
| F1  | 0.4088 | 0.3701 | -0.0387 | 0.933 | 4.04e-01 | 8.12e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8071 | 0.7939 | -0.0131 | 1.195 | 2.98e-01 | 3.12e-01 |
| AUPRC  | 0.3840 | 0.3635 | -0.0205 | 0.781 | 4.78e-01 | 6.25e-01 |
| Brier † | 0.2542 | 0.2620 | +0.0078 | -2.213 | 9.13e-02 | 1.25e-01 |
| Accuracy * | 0.7572 | 0.6915 | -0.0657 | 3.039 | 3.84e-02 | 6.25e-02 |
| F1 * | 0.4165 | 0.3649 | -0.0516 | 2.787 | 4.94e-02 | 6.25e-02 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8045 | 0.7991 | -0.0054 | 0.842 | 4.47e-01 | 8.12e-01 |
| AUPRC  | 0.3774 | 0.3652 | -0.0122 | 0.780 | 4.79e-01 | 6.25e-01 |
| Brier † | 0.2513 | 0.2764 | +0.0251 | -2.450 | 7.05e-02 | 6.25e-02 |
| Accuracy * | 0.7747 | 0.6861 | -0.0886 | 2.926 | 4.30e-02 | 6.25e-02 |
| F1 ** | 0.4266 | 0.3770 | -0.0496 | 4.770 | 8.84e-03 | 6.25e-02 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.7928 | 0.7904 | -0.0023 | 0.110 | 9.18e-01 | 1.00e+00 |
| AUPRC  | 0.3512 | 0.3977 | +0.0465 | -1.161 | 3.10e-01 | 3.12e-01 |
| Brier  | 0.2589 | 0.2603 | +0.0014 | -0.113 | 9.16e-01 | 8.12e-01 |
| Accuracy  | 0.7047 | 0.7507 | +0.0460 | -1.120 | 3.26e-01 | 3.75e-01 |
| F1  | 0.3761 | 0.4074 | +0.0313 | -1.373 | 2.42e-01 | 2.50e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7716 | -0.0345 | 0.968 | 3.88e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3231 | -0.0861 | 1.319 | 2.58e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2665 | +0.0857 | -8.718 | 9.53e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6818 | -0.0743 | 0.927 | 4.06e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.3701 | -0.0462 | 1.069 | 3.45e-01 | 4.38e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7939 | -0.0122 | 0.594 | 5.84e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3635 | -0.0457 | 1.100 | 3.33e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2620 | +0.0813 | -11.629 | 3.13e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6915 | -0.0646 | 1.645 | 1.75e-01 | 1.88e-01 |
| F1  | 0.4163 | 0.3649 | -0.0515 | 1.564 | 1.93e-01 | 3.12e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7991 | -0.0071 | 1.505 | 2.07e-01 | 3.12e-01 |
| AUPRC  | 0.4092 | 0.3652 | -0.0440 | 1.578 | 1.90e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2764 | +0.0957 | -9.010 | 8.40e-04 | 6.25e-02 |
| Accuracy * | 0.7561 | 0.6861 | -0.0700 | 3.338 | 2.89e-02 | 6.25e-02 |
| F1 † | 0.4163 | 0.3770 | -0.0394 | 2.213 | 9.13e-02 | 6.25e-02 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7904 | -0.0157 | 0.572 | 5.98e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3977 | -0.0115 | 0.219 | 8.38e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2603 | +0.0795 | -14.703 | 1.25e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7507 | -0.0054 | 0.090 | 9.32e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4074 | -0.0090 | 0.195 | 8.55e-01 | 6.25e-01 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7898 | 0.6916 | 0.8737 |
| M2 | CrossAttn | raw | AUPRC | 0.2881 | 0.1715 | 0.4445 |
| M2 | CrossAttn | raw | Brier | 0.2568 | 0.2291 | 0.2826 |
| M2 | CrossAttn | raw | Accuracy | 0.7031 | 0.6419 | 0.7642 |
| M2 | CrossAttn | raw | F1 | 0.3704 | 0.2472 | 0.4865 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.7620 | 0.6667 | 0.8452 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2578 | 0.1503 | 0.4150 |
| M2 | CrossAttn | std_scaled | Brier | 0.2704 | 0.2453 | 0.2936 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7860 | 0.7336 | 0.8384 |
| M2 | CrossAttn | std_scaled | F1 | 0.3951 | 0.2571 | 0.5209 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8189 | 0.7461 | 0.8833 |
| M2 | CrossAttn | norm | AUPRC | 0.3443 | 0.2004 | 0.5213 |
| M2 | CrossAttn | norm | Brier | 0.2576 | 0.2317 | 0.2819 |
| M2 | CrossAttn | norm | Accuracy | 0.7424 | 0.6856 | 0.7991 |
| M2 | CrossAttn | norm | F1 | 0.3656 | 0.2391 | 0.4860 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.8024 | 0.7156 | 0.8800 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2779 | 0.1791 | 0.4562 |
| M2 | CrossAttn | global_zscore | Brier | 0.2649 | 0.2390 | 0.2886 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.7205 | 0.6638 | 0.7773 |
| M2 | CrossAttn | global_zscore | F1 | 0.3846 | 0.2529 | 0.5047 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.7978 | 0.6983 | 0.8786 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3343 | 0.1934 | 0.5236 |
| M2_2 | CrossAttn | raw | Brier | 0.2636 | 0.2366 | 0.2889 |
| M2_2 | CrossAttn | raw | Accuracy | 0.6288 | 0.5677 | 0.6943 |
| M2_2 | CrossAttn | raw | F1 | 0.3411 | 0.2295 | 0.4511 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7805 | 0.6977 | 0.8547 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.2861 | 0.1643 | 0.4611 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2679 | 0.2435 | 0.2911 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.7118 | 0.6550 | 0.7729 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3654 | 0.2444 | 0.4786 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8301 | 0.7484 | 0.8967 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3580 | 0.2105 | 0.5495 |
| M2_2 | CrossAttn | norm | Brier | 0.2594 | 0.2330 | 0.2849 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6812 | 0.6201 | 0.7467 |
| M2_2 | CrossAttn | norm | F1 | 0.3652 | 0.2453 | 0.4779 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.8236 | 0.7363 | 0.8971 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3680 | 0.2206 | 0.5606 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2724 | 0.2481 | 0.2965 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6812 | 0.6201 | 0.7381 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3652 | 0.2476 | 0.4794 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7520 | 0.6638 | 0.8404 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2399 | 0.1469 | 0.4235 |
| M3 | CrossAttn3 | raw | Brier | 0.2795 | 0.2590 | 0.2987 |
| M3 | CrossAttn3 | raw | Accuracy | 0.6332 | 0.5677 | 0.6987 |
| M3 | CrossAttn3 | raw | F1 | 0.2759 | 0.1714 | 0.3803 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7827 | 0.6910 | 0.8647 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2583 | 0.1649 | 0.4287 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2671 | 0.2408 | 0.2916 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.6812 | 0.6201 | 0.7424 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3303 | 0.2150 | 0.4444 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.7740 | 0.6871 | 0.8507 |
| M3 | CrossAttn3 | norm | AUPRC | 0.2889 | 0.1613 | 0.4657 |
| M3 | CrossAttn3 | norm | Brier | 0.2661 | 0.2433 | 0.2882 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6507 | 0.5895 | 0.7119 |
| M3 | CrossAttn3 | norm | F1 | 0.3443 | 0.2308 | 0.4559 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7382 | 0.6387 | 0.8286 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2082 | 0.1350 | 0.3415 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2829 | 0.2592 | 0.3048 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.6725 | 0.6114 | 0.7336 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3478 | 0.2321 | 0.4643 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7898 | -0.0132 | 0.515 | 6.067e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.7620 | -0.0411 | 1.644 | 1.002e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8189 | +0.0159 | -0.881 | 3.783e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.8024 | -0.0006 | 0.024 | 9.811e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7520 | -0.0510 | 1.313 | 1.890e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7827 | -0.0203 | 0.729 | 4.658e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.7740 | -0.0291 | 1.019 | 3.080e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7382 | -0.0648 | 1.721 | 8.530e-02 | † |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7898 | 0.7978 | +0.0079 | -0.501 | 6.166e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.7620 | 0.7805 | +0.0185 | -0.908 | 3.641e-01 | ns |
| M2-norm vs M2_2-norm | 0.8189 | 0.8301 | +0.0112 | -0.657 | 5.111e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.8024 | 0.8236 | +0.0211 | -1.154 | 2.483e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7898 | 0.7520 | -0.0378 | 1.068 | 2.857e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.7620 | 0.7827 | +0.0207 | -1.038 | 2.991e-01 | ns |
| M2-norm vs M3-norm | 0.8189 | 0.7740 | -0.0449 | 1.976 | 4.817e-02 | * |
| M2-global_zscore vs M3-global_zscore | 0.8024 | 0.7382 | -0.0642 | 2.711 | 6.711e-03 | ** |

