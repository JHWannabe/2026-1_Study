# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8118 | 0.3599 | 0.2581 | 0.6943 | 0.3519 |
| M2_2 | CrossAttn | norm | 0.8059 | 0.2782 | 0.2627 | 0.7031 | 0.3704 |
| M3 | CrossAttn3 | norm | 0.8209 | 0.3141 | 0.2645 | 0.7424 | 0.4040 |
| M4 | AECOnly | raw | 0.5856 | 0.1882 | 0.3041 | 0.6507 | 0.2157 |
| M5 | CrossAttn-Feat | norm | 0.8173 | 0.3304 | 0.2624 | 0.7074 | 0.3366 |

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
| raw | 0.7980 | 0.2828 | 0.2672 | 0.7162 | 0.3434 |
| **norm** | 0.8118 | 0.3599 | 0.2581 | 0.6943 | 0.3519 |
| global_zscore | 0.7894 | 0.2837 | 0.2751 | 0.6812 | 0.3423 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (3 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.8022 | 0.3040 | 0.2645 | 0.5764 | 0.3121 |
| **norm** | 0.8059 | 0.2782 | 0.2627 | 0.7031 | 0.3704 |
| global_zscore | 0.8041 | 0.3194 | 0.2634 | 0.6026 | 0.3259 |

---

## Model 3 — Clinic + Scanner + AEC  (3 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7864 | 0.2858 | 0.2564 | 0.7729 | 0.3659 |
| **norm** | 0.8209 | 0.3141 | 0.2645 | 0.7424 | 0.4040 |
| global_zscore | 0.7530 | 0.2342 | 0.2704 | 0.6725 | 0.3363 |

---

## Model 4 — AEC Only  (3 AEC variants)

> 임상 특징 없이 AEC 시퀀스만으로 분류.

### AECOnly

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **raw** | 0.5856 | 0.1882 | 0.3041 | 0.6507 | 0.2157 |
| norm | 0.5386 | 0.1490 | 0.2529 | 0.8952 | 0.0000 |
| global_zscore | 0.5785 | 0.1369 | 0.2795 | 0.8952 | 0.0000 |

---

## Model 5 — Clinic + AEC Hand-crafted Features CrossAttn  (3 AEC variants)

> Age/Sex/BMI + AEC 통계 피처 11개(mean·std·max·min·peak_pos·auc·skew·kurt·early/mid/late mean) → Cross Attention.

### CrossAttn-Feat

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.8124 | 0.3041 | 0.2588 | 0.6463 | 0.3520 |
| **norm** | 0.8173 | 0.3304 | 0.2624 | 0.7074 | 0.3366 |
| global_zscore | 0.8144 | 0.3103 | 0.2599 | 0.6769 | 0.3621 |

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
| AUC-ROC  | 0.8061 | 0.8170 | +0.0108 | -0.835 | 4.51e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4038 | -0.0054 | 0.143 | 8.93e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2462 | +0.0655 | -19.202 | 4.33e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7364 | -0.0197 | 0.661 | 5.44e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4137 | -0.0027 | 0.103 | 9.23e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8045 | -0.0016 | 0.128 | 9.04e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3677 | -0.0416 | 1.217 | 2.90e-01 | 4.38e-01 |
| Brier *** | 0.1808 | 0.2470 | +0.0663 | -12.563 | 2.31e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7692 | +0.0131 | -0.521 | 6.30e-01 | 8.75e-01 |
| F1  | 0.4163 | 0.4273 | +0.0110 | -0.429 | 6.90e-01 | 8.12e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8142 | +0.0080 | -0.562 | 6.04e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4044 | -0.0049 | 0.177 | 8.68e-01 | 8.12e-01 |
| Brier *** | 0.1808 | 0.2479 | +0.0672 | -11.408 | 3.37e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7539 | -0.0022 | 0.052 | 9.61e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4314 | +0.0151 | -0.387 | 7.18e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8170 | 0.8059 | -0.0110 | 1.267 | 2.74e-01 | 3.12e-01 |
| AUPRC  | 0.4038 | 0.3806 | -0.0232 | 1.067 | 3.46e-01 | 6.25e-01 |
| Brier  | 0.2462 | 0.2434 | -0.0029 | 1.405 | 2.33e-01 | 3.12e-01 |
| Accuracy  | 0.7364 | 0.7528 | +0.0164 | -0.272 | 7.99e-01 | 6.25e-01 |
| F1  | 0.4137 | 0.4120 | -0.0017 | 0.039 | 9.71e-01 | 6.25e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8045 | 0.7996 | -0.0049 | 0.699 | 5.23e-01 | 4.38e-01 |
| AUPRC  | 0.3677 | 0.3769 | +0.0092 | -0.886 | 4.26e-01 | 6.25e-01 |
| Brier  | 0.2470 | 0.2398 | -0.0072 | 1.623 | 1.80e-01 | 1.88e-01 |
| Accuracy  | 0.7692 | 0.7670 | -0.0022 | 0.097 | 9.27e-01 | 1.00e+00 |
| F1  | 0.4273 | 0.4272 | -0.0002 | 0.011 | 9.92e-01 | 8.12e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8142 | 0.8109 | -0.0033 | 0.477 | 6.58e-01 | 8.12e-01 |
| AUPRC  | 0.4044 | 0.3892 | -0.0151 | 0.461 | 6.69e-01 | 8.12e-01 |
| Brier  | 0.2479 | 0.2455 | -0.0025 | 0.445 | 6.79e-01 | 6.25e-01 |
| Accuracy  | 0.7539 | 0.7331 | -0.0208 | 0.402 | 7.08e-01 | 1.00e+00 |
| F1  | 0.4314 | 0.3983 | -0.0331 | 0.772 | 4.83e-01 | 4.38e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8059 | -0.0002 | 0.013 | 9.90e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3806 | -0.0286 | 0.785 | 4.76e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2434 | +0.0626 | -13.934 | 1.54e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7528 | -0.0032 | 0.080 | 9.40e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4120 | -0.0043 | 0.127 | 9.05e-01 | 1.00e+00 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.7996 | -0.0065 | 0.559 | 6.06e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3769 | -0.0323 | 0.989 | 3.79e-01 | 3.12e-01 |
| Brier *** | 0.1808 | 0.2398 | +0.0590 | -24.911 | 1.54e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7670 | +0.0109 | -0.291 | 7.85e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4272 | +0.0108 | -0.372 | 7.29e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8109 | +0.0047 | -0.280 | 7.93e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3892 | -0.0200 | 0.350 | 7.44e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2455 | +0.0647 | -9.864 | 5.93e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7331 | -0.0230 | 1.004 | 3.72e-01 | 5.00e-01 |
| F1  | 0.4163 | 0.3983 | -0.0181 | 0.734 | 5.04e-01 | 4.38e-01 |

## M1 (LR) vs M4 (AECOnly)

> A = M1 LR, B = M4 AECOnly.

### raw  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8061 | 0.6792 | -0.1269 | 3.573 | 2.33e-02 | 6.25e-02 |
| AUPRC * | 0.4092 | 0.2397 | -0.1695 | 3.440 | 2.63e-02 | 6.25e-02 |
| Brier *** | 0.1808 | 0.3044 | +0.1236 | -14.913 | 1.18e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6369 | -0.1192 | 1.581 | 1.89e-01 | 3.12e-01 |
| F1 † | 0.4163 | 0.2983 | -0.1181 | 2.455 | 7.01e-02 | 1.25e-01 |

### norm  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.8061 | 0.5322 | -0.2740 | 7.964 | 1.35e-03 | 6.25e-02 |
| AUPRC ** | 0.4092 | 0.1402 | -0.2690 | 4.741 | 9.03e-03 | 6.25e-02 |
| Brier *** | 0.1808 | 0.2985 | +0.1177 | -9.057 | 8.24e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6491 | -0.1070 | 2.120 | 1.01e-01 | 1.25e-01 |
| F1 ** | 0.4163 | 0.2334 | -0.1829 | 6.128 | 3.59e-03 | 6.25e-02 |

### global_zscore  (M1-LR vs M4-AECOnly)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.8061 | 0.6685 | -0.1377 | 2.933 | 4.27e-02 | 1.25e-01 |
| AUPRC * | 0.4092 | 0.2118 | -0.1974 | 2.881 | 4.50e-02 | 1.25e-01 |
| Brier ** | 0.1808 | 0.2995 | +0.1187 | -6.397 | 3.07e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.6477 | -0.1083 | 1.009 | 3.70e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.3174 | -0.0989 | 1.454 | 2.20e-01 | 3.12e-01 |

## M4 (AECOnly) vs M2 (CrossAttn)

> A = M4 AECOnly, B = M2 CrossAttn. aec_var 키로 매칭.

#### Case: raw  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.6792 | 0.8170 | +0.1377 | -5.878 | 4.19e-03 | 6.25e-02 |
| AUPRC * | 0.2397 | 0.4038 | +0.1641 | -3.468 | 2.56e-02 | 6.25e-02 |
| Brier ** | 0.3044 | 0.2462 | -0.0582 | 7.561 | 1.64e-03 | 6.25e-02 |
| Accuracy  | 0.6369 | 0.7364 | +0.0995 | -1.187 | 3.01e-01 | 6.25e-01 |
| F1 † | 0.2983 | 0.4137 | +0.1154 | -2.694 | 5.44e-02 | 6.25e-02 |

#### Case: norm  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC *** | 0.5322 | 0.8045 | +0.2723 | -9.977 | 5.67e-04 | 6.25e-02 |
| AUPRC ** | 0.1402 | 0.3677 | +0.2274 | -6.301 | 3.24e-03 | 6.25e-02 |
| Brier ** | 0.2985 | 0.2470 | -0.0515 | 5.435 | 5.56e-03 | 6.25e-02 |
| Accuracy  | 0.6491 | 0.7692 | +0.1201 | -1.911 | 1.29e-01 | 1.88e-01 |
| F1 ** | 0.2334 | 0.4273 | +0.1939 | -5.278 | 6.18e-03 | 6.25e-02 |

#### Case: global_zscore  (M4-AECOnly vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC * | 0.6685 | 0.8142 | +0.1457 | -4.384 | 1.18e-02 | 6.25e-02 |
| AUPRC * | 0.2118 | 0.4044 | +0.1926 | -3.731 | 2.03e-02 | 6.25e-02 |
| Brier † | 0.2995 | 0.2479 | -0.0515 | 2.699 | 5.42e-02 | 6.25e-02 |
| Accuracy  | 0.6477 | 0.7539 | +0.1061 | -1.300 | 2.64e-01 | 4.38e-01 |
| F1  | 0.3174 | 0.4314 | +0.1140 | -1.930 | 1.26e-01 | 1.25e-01 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.7980 | 0.7186 | 0.8689 |
| M2 | CrossAttn | raw | AUPRC | 0.2828 | 0.1680 | 0.4437 |
| M2 | CrossAttn | raw | Brier | 0.2672 | 0.2397 | 0.2925 |
| M2 | CrossAttn | raw | Accuracy | 0.7162 | 0.6594 | 0.7773 |
| M2 | CrossAttn | raw | F1 | 0.3434 | 0.2222 | 0.4598 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8118 | 0.7413 | 0.8771 |
| M2 | CrossAttn | norm | AUPRC | 0.3599 | 0.2106 | 0.5411 |
| M2 | CrossAttn | norm | Brier | 0.2581 | 0.2308 | 0.2843 |
| M2 | CrossAttn | norm | Accuracy | 0.6943 | 0.6376 | 0.7555 |
| M2 | CrossAttn | norm | F1 | 0.3519 | 0.2268 | 0.4685 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7894 | 0.7076 | 0.8666 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2837 | 0.1678 | 0.4481 |
| M2 | CrossAttn | global_zscore | Brier | 0.2751 | 0.2495 | 0.2988 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.6812 | 0.6201 | 0.7424 |
| M2 | CrossAttn | global_zscore | F1 | 0.3423 | 0.2222 | 0.4567 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.8022 | 0.7266 | 0.8695 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3040 | 0.1776 | 0.4829 |
| M2_2 | CrossAttn | raw | Brier | 0.2645 | 0.2375 | 0.2907 |
| M2_2 | CrossAttn | raw | Accuracy | 0.5764 | 0.5153 | 0.6419 |
| M2_2 | CrossAttn | raw | F1 | 0.3121 | 0.2069 | 0.4173 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8059 | 0.7201 | 0.8763 |
| M2_2 | CrossAttn | norm | AUPRC | 0.2782 | 0.1800 | 0.4604 |
| M2_2 | CrossAttn | norm | Brier | 0.2627 | 0.2375 | 0.2872 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7031 | 0.6462 | 0.7642 |
| M2_2 | CrossAttn | norm | F1 | 0.3704 | 0.2478 | 0.4860 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.8041 | 0.7212 | 0.8727 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3194 | 0.1835 | 0.4974 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2634 | 0.2368 | 0.2895 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6026 | 0.5371 | 0.6681 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3259 | 0.2185 | 0.4348 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7864 | 0.6857 | 0.8768 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2858 | 0.1854 | 0.4762 |
| M3 | CrossAttn3 | raw | Brier | 0.2564 | 0.2305 | 0.2808 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7729 | 0.7205 | 0.8253 |
| M3 | CrossAttn3 | raw | F1 | 0.3659 | 0.2353 | 0.5000 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8209 | 0.7398 | 0.8889 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3141 | 0.1952 | 0.5211 |
| M3 | CrossAttn3 | norm | Brier | 0.2645 | 0.2407 | 0.2862 |
| M3 | CrossAttn3 | norm | Accuracy | 0.7424 | 0.6856 | 0.7991 |
| M3 | CrossAttn3 | norm | F1 | 0.4040 | 0.2784 | 0.5243 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.7530 | 0.6564 | 0.8415 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2342 | 0.1496 | 0.4083 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2704 | 0.2439 | 0.2945 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.6725 | 0.6114 | 0.7380 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3363 | 0.2203 | 0.4522 |
| M4 | AECOnly | raw | AUC-ROC | 0.5856 | 0.4544 | 0.7105 |
| M4 | AECOnly | raw | AUPRC | 0.1882 | 0.1021 | 0.3675 |
| M4 | AECOnly | raw | Brier | 0.3041 | 0.2917 | 0.3164 |
| M4 | AECOnly | raw | Accuracy | 0.6507 | 0.5852 | 0.7118 |
| M4 | AECOnly | raw | F1 | 0.2157 | 0.1111 | 0.3273 |
| M4 | AECOnly | norm | AUC-ROC | 0.5386 | 0.4010 | 0.6694 |
| M4 | AECOnly | norm | AUPRC | 0.1490 | 0.0862 | 0.2802 |
| M4 | AECOnly | norm | Brier | 0.2529 | 0.2523 | 0.2535 |
| M4 | AECOnly | norm | Accuracy | 0.8952 | 0.8515 | 0.9301 |
| M4 | AECOnly | norm | F1 | 0.0000 | 0.0000 | 0.0000 |
| M4 | AECOnly | global_zscore | AUC-ROC | 0.5785 | 0.4567 | 0.6939 |
| M4 | AECOnly | global_zscore | AUPRC | 0.1369 | 0.0846 | 0.2462 |
| M4 | AECOnly | global_zscore | Brier | 0.2795 | 0.2720 | 0.2871 |
| M4 | AECOnly | global_zscore | Accuracy | 0.8952 | 0.8515 | 0.9301 |
| M4 | AECOnly | global_zscore | F1 | 0.0000 | 0.0000 | 0.0000 |
| M5 | CrossAttn-Feat | raw | AUC-ROC | 0.8124 | 0.7328 | 0.8822 |
| M5 | CrossAttn-Feat | raw | AUPRC | 0.3041 | 0.1832 | 0.4764 |
| M5 | CrossAttn-Feat | raw | Brier | 0.2588 | 0.2315 | 0.2839 |
| M5 | CrossAttn-Feat | raw | Accuracy | 0.6463 | 0.5852 | 0.7074 |
| M5 | CrossAttn-Feat | raw | F1 | 0.3520 | 0.2376 | 0.4627 |
| M5 | CrossAttn-Feat | norm | AUC-ROC | 0.8173 | 0.7368 | 0.8848 |
| M5 | CrossAttn-Feat | norm | AUPRC | 0.3304 | 0.1942 | 0.5179 |
| M5 | CrossAttn-Feat | norm | Brier | 0.2624 | 0.2362 | 0.2876 |
| M5 | CrossAttn-Feat | norm | Accuracy | 0.7074 | 0.6463 | 0.7643 |
| M5 | CrossAttn-Feat | norm | F1 | 0.3366 | 0.2157 | 0.4490 |
| M5 | CrossAttn-Feat | global_zscore | AUC-ROC | 0.8144 | 0.7382 | 0.8805 |
| M5 | CrossAttn-Feat | global_zscore | AUPRC | 0.3103 | 0.1868 | 0.4835 |
| M5 | CrossAttn-Feat | global_zscore | Brier | 0.2599 | 0.2341 | 0.2842 |
| M5 | CrossAttn-Feat | global_zscore | Accuracy | 0.6769 | 0.6157 | 0.7380 |
| M5 | CrossAttn-Feat | global_zscore | F1 | 0.3621 | 0.2400 | 0.4755 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M5 CrossAttn-Feat

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M5-raw | 0.8030 | 0.8124 | +0.0093 | -0.406 | 6.850e-01 | ns |
| M1-LR vs M5-norm | 0.8030 | 0.8173 | +0.0142 | -0.941 | 3.467e-01 | ns |
| M1-LR vs M5-global_zscore | 0.8030 | 0.8144 | +0.0114 | -0.621 | 5.345e-01 | ns |

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.7980 | -0.0051 | 0.223 | 8.239e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8118 | +0.0087 | -0.389 | 6.969e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7894 | -0.0136 | 0.529 | 5.971e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7864 | -0.0167 | 0.388 | 6.981e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8209 | +0.0179 | -0.696 | 4.862e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.7530 | -0.0500 | 1.470 | 1.417e-01 | ns |

## M1 LR vs M4 AECOnly

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M4-raw | 0.8030 | 0.5856 | -0.2175 | 2.737 | 6.201e-03 | ** |
| M1-LR vs M4-norm | 0.8030 | 0.5386 | -0.2644 | 3.623 | 2.909e-04 | *** |
| M1-LR vs M4-global_zscore | 0.8030 | 0.5785 | -0.2246 | 3.120 | 1.806e-03 | ** |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.7980 | 0.8022 | +0.0043 | -0.275 | 7.835e-01 | ns |
| M2-norm vs M2_2-norm | 0.8118 | 0.8059 | -0.0059 | 0.243 | 8.082e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.7894 | 0.8041 | +0.0146 | -0.726 | 4.679e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.7980 | 0.7864 | -0.0116 | 0.418 | 6.757e-01 | ns |
| M2-norm vs M3-norm | 0.8118 | 0.8209 | +0.0091 | -0.496 | 6.200e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.7894 | 0.7530 | -0.0364 | 1.629 | 1.034e-01 | ns |

## M4 AECOnly vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M4-raw vs M2-raw | 0.5856 | 0.7980 | +0.2124 | -3.139 | 1.693e-03 | ** |
| M4-norm vs M2-norm | 0.5386 | 0.8118 | +0.2732 | -3.899 | 9.661e-05 | *** |
| M4-global_zscore vs M2-global_zscore | 0.5785 | 0.7894 | +0.2110 | -3.535 | 4.085e-04 | *** |

