# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | raw | 0.8102 | 0.3062 | 0.2531 | 0.7205 | 0.3846 |
| M2_2 | CrossAttn | norm | 0.8291 | 0.3519 | 0.3388 | 0.6332 | 0.3538 |
| M3 | CrossAttn3 | norm | 0.8205 | 0.3206 | 0.2430 | 0.7773 | 0.3855 |

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
| **raw** | 0.8102 | 0.3062 | 0.2531 | 0.7205 | 0.3846 |
| std_scaled | 0.8022 | 0.2928 | 0.2257 | 0.7380 | 0.3617 |
| norm | 0.8053 | 0.3450 | 0.2933 | 0.6638 | 0.3636 |
| global_zscore | 0.8079 | 0.3075 | 0.2551 | 0.7380 | 0.3878 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.8246 | 0.3788 | 0.2118 | 0.7904 | 0.4000 |
| std_scaled | 0.7988 | 0.3126 | 0.2816 | 0.6812 | 0.3423 |
| **norm** | 0.8291 | 0.3519 | 0.3388 | 0.6332 | 0.3538 |
| global_zscore | 0.8130 | 0.3164 | 0.2665 | 0.6856 | 0.3684 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7874 | 0.2832 | 0.2443 | 0.7336 | 0.3441 |
| std_scaled | 0.8177 | 0.2911 | 0.2757 | 0.6987 | 0.3894 |
| **norm** | 0.8205 | 0.3206 | 0.2430 | 0.7773 | 0.3855 |
| global_zscore | 0.8049 | 0.2746 | 0.2707 | 0.7118 | 0.3889 |

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
| AUC-ROC  | 0.8061 | 0.8267 | +0.0206 | -1.132 | 3.21e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.4211 | +0.0118 | -0.343 | 7.49e-01 | 8.12e-01 |
| Brier ** | 0.1808 | 0.2542 | +0.0735 | -5.658 | 4.81e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7703 | +0.0142 | -0.383 | 7.21e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4298 | +0.0134 | -0.450 | 6.76e-01 | 1.00e+00 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8247 | +0.0186 | -1.283 | 2.69e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.4259 | +0.0166 | -0.424 | 6.94e-01 | 6.25e-01 |
| Brier * | 0.1808 | 0.2315 | +0.0507 | -4.557 | 1.04e-02 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7353 | -0.0208 | 1.187 | 3.01e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4109 | -0.0054 | 0.253 | 8.12e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8165 | +0.0103 | -0.981 | 3.82e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4140 | +0.0048 | -0.238 | 8.23e-01 | 6.25e-01 |
| Brier ** | 0.1808 | 0.2612 | +0.0804 | -7.610 | 1.60e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7626 | +0.0065 | -0.419 | 6.97e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4261 | +0.0097 | -0.576 | 5.95e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8210 | +0.0149 | -0.984 | 3.81e-01 | 3.12e-01 |
| AUPRC  | 0.4092 | 0.4203 | +0.0110 | -0.392 | 7.15e-01 | 8.12e-01 |
| Brier * | 0.1808 | 0.2368 | +0.0561 | -3.802 | 1.91e-02 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7539 | -0.0022 | 0.101 | 9.24e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4235 | +0.0072 | -0.377 | 7.25e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8267 | 0.8206 | -0.0061 | 0.654 | 5.49e-01 | 1.00e+00 |
| AUPRC  | 0.4211 | 0.3934 | -0.0276 | 1.450 | 2.21e-01 | 3.12e-01 |
| Brier  | 0.2542 | 0.2458 | -0.0084 | 1.193 | 2.99e-01 | 3.12e-01 |
| Accuracy  | 0.7703 | 0.7353 | -0.0350 | 0.666 | 5.42e-01 | 6.25e-01 |
| F1  | 0.4298 | 0.4103 | -0.0195 | 0.581 | 5.93e-01 | 8.12e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8247 | 0.8181 | -0.0067 | 1.015 | 3.67e-01 | 6.25e-01 |
| AUPRC  | 0.4259 | 0.3970 | -0.0289 | 1.812 | 1.44e-01 | 1.25e-01 |
| Brier  | 0.2315 | 0.2333 | +0.0018 | -0.174 | 8.71e-01 | 8.12e-01 |
| Accuracy  | 0.7353 | 0.7834 | +0.0481 | -1.116 | 3.27e-01 | 4.38e-01 |
| F1  | 0.4109 | 0.4426 | +0.0317 | -1.394 | 2.36e-01 | 3.12e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8165 | 0.8198 | +0.0033 | -0.602 | 5.80e-01 | 6.25e-01 |
| AUPRC  | 0.4140 | 0.4169 | +0.0029 | -0.169 | 8.74e-01 | 8.12e-01 |
| Brier  | 0.2612 | 0.2451 | -0.0161 | 0.765 | 4.87e-01 | 4.38e-01 |
| Accuracy  | 0.7626 | 0.7965 | +0.0339 | -1.142 | 3.17e-01 | 4.38e-01 |
| F1  | 0.4261 | 0.4454 | +0.0193 | -0.823 | 4.57e-01 | 4.38e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8210 | 0.8107 | -0.0103 | 1.050 | 3.53e-01 | 4.38e-01 |
| AUPRC  | 0.4203 | 0.3893 | -0.0310 | 2.000 | 1.16e-01 | 1.25e-01 |
| Brier  | 0.2368 | 0.2141 | -0.0227 | 2.046 | 1.10e-01 | 1.88e-01 |
| Accuracy  | 0.7539 | 0.7637 | +0.0099 | -0.185 | 8.62e-01 | 1.00e+00 |
| F1  | 0.4235 | 0.4205 | -0.0030 | 0.092 | 9.31e-01 | 8.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8206 | +0.0145 | -1.173 | 3.06e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.3934 | -0.0158 | 0.503 | 6.41e-01 | 8.12e-01 |
| Brier ** | 0.1808 | 0.2458 | +0.0651 | -8.576 | 1.02e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7353 | -0.0208 | 1.066 | 3.47e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4103 | -0.0060 | 0.405 | 7.07e-01 | 1.00e+00 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8181 | +0.0119 | -1.007 | 3.71e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3970 | -0.0122 | 0.340 | 7.51e-01 | 8.12e-01 |
| Brier ** | 0.1808 | 0.2333 | +0.0526 | -5.669 | 4.78e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7834 | +0.0274 | -0.566 | 6.02e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4426 | +0.0263 | -0.743 | 4.99e-01 | 4.38e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8198 | +0.0136 | -0.889 | 4.24e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4169 | +0.0077 | -0.254 | 8.12e-01 | 8.12e-01 |
| Brier * | 0.1808 | 0.2451 | +0.0643 | -4.120 | 1.46e-02 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7965 | +0.0405 | -1.091 | 3.37e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4454 | +0.0290 | -1.020 | 3.66e-01 | 6.25e-01 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8107 | +0.0045 | -0.359 | 7.38e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3893 | -0.0200 | 0.649 | 5.51e-01 | 8.12e-01 |
| Brier † | 0.1808 | 0.2141 | +0.0333 | -2.590 | 6.07e-02 | 1.25e-01 |
| Accuracy  | 0.7561 | 0.7637 | +0.0077 | -0.162 | 8.79e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4205 | +0.0042 | -0.125 | 9.07e-01 | 1.00e+00 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.8102 | 0.7259 | 0.8836 |
| M2 | CrossAttn | raw | AUPRC | 0.3062 | 0.1876 | 0.4724 |
| M2 | CrossAttn | raw | Brier | 0.2531 | 0.2247 | 0.2800 |
| M2 | CrossAttn | raw | Accuracy | 0.7205 | 0.6636 | 0.7817 |
| M2 | CrossAttn | raw | F1 | 0.3846 | 0.2580 | 0.5042 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.8022 | 0.7226 | 0.8729 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2928 | 0.1769 | 0.4603 |
| M2 | CrossAttn | std_scaled | Brier | 0.2257 | 0.2022 | 0.2483 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7380 | 0.6812 | 0.7948 |
| M2 | CrossAttn | std_scaled | F1 | 0.3617 | 0.2410 | 0.4828 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8053 | 0.7212 | 0.8764 |
| M2 | CrossAttn | norm | AUPRC | 0.3450 | 0.1993 | 0.5151 |
| M2 | CrossAttn | norm | Brier | 0.2933 | 0.2639 | 0.3209 |
| M2 | CrossAttn | norm | Accuracy | 0.6638 | 0.6026 | 0.7293 |
| M2 | CrossAttn | norm | F1 | 0.3636 | 0.2456 | 0.4762 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.8079 | 0.7210 | 0.8832 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.3075 | 0.1891 | 0.4792 |
| M2 | CrossAttn | global_zscore | Brier | 0.2551 | 0.2281 | 0.2811 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.7380 | 0.6812 | 0.7948 |
| M2 | CrossAttn | global_zscore | F1 | 0.3878 | 0.2637 | 0.5049 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.8246 | 0.7452 | 0.8910 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3788 | 0.2227 | 0.5695 |
| M2_2 | CrossAttn | raw | Brier | 0.2118 | 0.1898 | 0.2330 |
| M2_2 | CrossAttn | raw | Accuracy | 0.7904 | 0.7380 | 0.8428 |
| M2_2 | CrossAttn | raw | F1 | 0.4000 | 0.2580 | 0.5287 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7988 | 0.7071 | 0.8753 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3126 | 0.1861 | 0.4932 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2816 | 0.2538 | 0.3091 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6812 | 0.6201 | 0.7424 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3423 | 0.2264 | 0.4522 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8291 | 0.7495 | 0.8939 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3519 | 0.2086 | 0.5349 |
| M2_2 | CrossAttn | norm | Brier | 0.3388 | 0.3071 | 0.3691 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6332 | 0.5721 | 0.6987 |
| M2_2 | CrossAttn | norm | F1 | 0.3538 | 0.2435 | 0.4615 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.8130 | 0.7276 | 0.8837 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3164 | 0.1911 | 0.4929 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2665 | 0.2393 | 0.2924 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6856 | 0.6287 | 0.7467 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3684 | 0.2478 | 0.4808 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7874 | 0.6988 | 0.8674 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2832 | 0.1685 | 0.4462 |
| M3 | CrossAttn3 | raw | Brier | 0.2443 | 0.2186 | 0.2687 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7336 | 0.6769 | 0.7904 |
| M3 | CrossAttn3 | raw | F1 | 0.3441 | 0.2105 | 0.4655 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.8177 | 0.7332 | 0.8888 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2911 | 0.1893 | 0.4708 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2757 | 0.2506 | 0.3005 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.6987 | 0.6376 | 0.7598 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3894 | 0.2692 | 0.5038 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8205 | 0.7399 | 0.8897 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3206 | 0.1994 | 0.5274 |
| M3 | CrossAttn3 | norm | Brier | 0.2430 | 0.2186 | 0.2671 |
| M3 | CrossAttn3 | norm | Accuracy | 0.7773 | 0.7249 | 0.8297 |
| M3 | CrossAttn3 | norm | F1 | 0.3855 | 0.2469 | 0.5116 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.8049 | 0.7116 | 0.8844 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.2746 | 0.1823 | 0.4534 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2707 | 0.2425 | 0.2979 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7118 | 0.6550 | 0.7729 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3889 | 0.2655 | 0.5082 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.8102 | +0.0071 | -0.293 | 7.698e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.8022 | -0.0008 | 0.038 | 9.700e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8053 | +0.0022 | -0.150 | 8.807e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.8079 | +0.0049 | -0.211 | 8.331e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7874 | -0.0157 | 0.592 | 5.541e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.8177 | +0.0146 | -0.678 | 4.978e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8205 | +0.0175 | -0.793 | 4.275e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.8049 | +0.0018 | -0.075 | 9.403e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.8102 | 0.8246 | +0.0144 | -0.671 | 5.019e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.8022 | 0.7988 | -0.0035 | 0.156 | 8.763e-01 | ns |
| M2-norm vs M2_2-norm | 0.8053 | 0.8291 | +0.0238 | -1.803 | 7.139e-02 | † |
| M2-global_zscore vs M2_2-global_zscore | 0.8079 | 0.8130 | +0.0051 | -0.286 | 7.746e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.8102 | 0.7874 | -0.0228 | 1.516 | 1.295e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.8022 | 0.8177 | +0.0154 | -0.834 | 4.043e-01 | ns |
| M2-norm vs M3-norm | 0.8053 | 0.8205 | +0.0152 | -0.944 | 3.449e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.8079 | 0.8049 | -0.0030 | 0.179 | 8.576e-01 | ns |

