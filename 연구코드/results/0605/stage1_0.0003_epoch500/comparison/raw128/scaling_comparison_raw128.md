# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8081 | 0.3089 | 0.3346 | 0.6201 | 0.3459 |
| M2_2 | CrossAttn | norm | 0.8272 | 0.3313 | 0.3067 | 0.6376 | 0.3566 |
| M3 | CrossAttn3 | norm | 0.8154 | 0.3393 | 0.2887 | 0.6507 | 0.3651 |

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
| raw | 0.8002 | 0.2946 | 0.2602 | 0.7293 | 0.3800 |
| std_scaled | 0.8053 | 0.2943 | 0.2554 | 0.7074 | 0.3619 |
| **norm** | 0.8081 | 0.3089 | 0.3346 | 0.6201 | 0.3459 |
| global_zscore | 0.7815 | 0.2688 | 0.2654 | 0.6900 | 0.3486 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.8069 | 0.3412 | 0.2844 | 0.5721 | 0.3099 |
| std_scaled | 0.7937 | 0.3179 | 0.2663 | 0.6288 | 0.3200 |
| **norm** | 0.8272 | 0.3313 | 0.3067 | 0.6376 | 0.3566 |
| global_zscore | 0.7927 | 0.3023 | 0.2518 | 0.6769 | 0.3393 |

---

## Model 3 — Clinic + Scanner + AEC  (4 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| raw | 0.7894 | 0.2754 | 0.2614 | 0.7031 | 0.3704 |
| std_scaled | 0.7890 | 0.2928 | 0.2733 | 0.7074 | 0.3619 |
| **norm** | 0.8154 | 0.3393 | 0.2887 | 0.6507 | 0.3651 |
| global_zscore | 0.8108 | 0.3059 | 0.2584 | 0.7424 | 0.3789 |

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
| AUC-ROC  | 0.8061 | 0.8228 | +0.0166 | -1.153 | 3.13e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4073 | -0.0019 | 0.055 | 9.58e-01 | 6.25e-01 |
| Brier ** | 0.1808 | 0.2456 | +0.0648 | -7.424 | 1.76e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7451 | -0.0110 | 0.676 | 5.36e-01 | 9.38e-01 |
| F1  | 0.4163 | 0.4120 | -0.0044 | 0.191 | 8.58e-01 | 1.00e+00 |

### std_scaled  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8222 | +0.0160 | -1.257 | 2.77e-01 | 3.12e-01 |
| AUPRC  | 0.4092 | 0.4234 | +0.0141 | -0.502 | 6.42e-01 | 6.25e-01 |
| Brier ** | 0.1808 | 0.2337 | +0.0529 | -8.532 | 1.04e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7484 | -0.0077 | 0.174 | 8.71e-01 | 8.75e-01 |
| F1  | 0.4163 | 0.4202 | +0.0039 | -0.112 | 9.16e-01 | 8.75e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8118 | +0.0057 | -0.420 | 6.96e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4090 | -0.0002 | 0.009 | 9.93e-01 | 6.25e-01 |
| Brier *** | 0.1808 | 0.2473 | +0.0665 | -10.449 | 4.74e-04 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.8064 | +0.0503 | -1.414 | 2.30e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4480 | +0.0316 | -1.256 | 2.78e-01 | 4.38e-01 |

### global_zscore  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8189 | +0.0128 | -0.771 | 4.84e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4067 | -0.0025 | 0.070 | 9.47e-01 | 1.00e+00 |
| Brier *** | 0.1808 | 0.2443 | +0.0635 | -17.227 | 6.66e-05 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7418 | -0.0143 | 0.284 | 7.90e-01 | 8.75e-01 |
| F1  | 0.4163 | 0.4065 | -0.0099 | 0.255 | 8.12e-01 | 8.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: raw  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC † | 0.8228 | 0.8129 | -0.0099 | 2.228 | 8.99e-02 | 1.25e-01 |
| AUPRC  | 0.4073 | 0.3830 | -0.0243 | 1.662 | 1.72e-01 | 1.88e-01 |
| Brier  | 0.2456 | 0.2358 | -0.0098 | 0.586 | 5.89e-01 | 1.00e+00 |
| Accuracy  | 0.7451 | 0.7539 | +0.0089 | -0.256 | 8.10e-01 | 8.12e-01 |
| F1  | 0.4120 | 0.4217 | +0.0097 | -0.449 | 6.77e-01 | 8.12e-01 |

#### Case: std_scaled  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8222 | 0.8251 | +0.0029 | -0.260 | 8.07e-01 | 1.00e+00 |
| AUPRC  | 0.4234 | 0.4385 | +0.0152 | -0.487 | 6.51e-01 | 6.25e-01 |
| Brier  | 0.2337 | 0.2296 | -0.0040 | 0.345 | 7.47e-01 | 8.12e-01 |
| Accuracy  | 0.7484 | 0.8020 | +0.0535 | -1.102 | 3.32e-01 | 4.38e-01 |
| F1  | 0.4202 | 0.4516 | +0.0314 | -0.885 | 4.26e-01 | 4.38e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8118 | 0.8209 | +0.0090 | -0.956 | 3.93e-01 | 4.38e-01 |
| AUPRC  | 0.4090 | 0.4159 | +0.0069 | -0.330 | 7.58e-01 | 6.25e-01 |
| Brier  | 0.2473 | 0.2228 | -0.0245 | 1.381 | 2.39e-01 | 1.88e-01 |
| Accuracy † | 0.8064 | 0.7637 | -0.0427 | 2.150 | 9.80e-02 | 1.88e-01 |
| F1  | 0.4480 | 0.4158 | -0.0322 | 1.887 | 1.32e-01 | 1.88e-01 |

#### Case: global_zscore  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8189 | 0.8159 | -0.0031 | 1.098 | 3.34e-01 | 4.38e-01 |
| AUPRC  | 0.4067 | 0.3855 | -0.0212 | 1.258 | 2.77e-01 | 4.38e-01 |
| Brier  | 0.2443 | 0.2365 | -0.0077 | 0.390 | 7.17e-01 | 1.00e+00 |
| Accuracy  | 0.7418 | 0.7658 | +0.0240 | -1.531 | 2.01e-01 | 3.75e-01 |
| F1  | 0.4065 | 0.4236 | +0.0171 | -1.409 | 2.32e-01 | 3.75e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### raw  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8129 | +0.0067 | -0.367 | 7.32e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.3830 | -0.0262 | 0.656 | 5.47e-01 | 8.12e-01 |
| Brier ** | 0.1808 | 0.2358 | +0.0550 | -4.945 | 7.79e-03 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.7539 | -0.0021 | 0.063 | 9.53e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4217 | +0.0054 | -0.193 | 8.57e-01 | 8.12e-01 |

### std_scaled  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8251 | +0.0190 | -1.363 | 2.44e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.4385 | +0.0293 | -0.685 | 5.31e-01 | 6.25e-01 |
| Brier * | 0.1808 | 0.2296 | +0.0489 | -4.074 | 1.52e-02 | 6.25e-02 |
| Accuracy  | 0.7561 | 0.8020 | +0.0459 | -0.974 | 3.85e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4516 | +0.0353 | -0.894 | 4.22e-01 | 8.12e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8209 | +0.0147 | -0.812 | 4.62e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4159 | +0.0067 | -0.171 | 8.72e-01 | 8.12e-01 |
| Brier † | 0.1808 | 0.2228 | +0.0421 | -2.136 | 9.95e-02 | 1.25e-01 |
| Accuracy  | 0.7561 | 0.7637 | +0.0076 | -0.169 | 8.74e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4158 | -0.0005 | 0.016 | 9.88e-01 | 1.00e+00 |

### global_zscore  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8159 | +0.0097 | -0.532 | 6.23e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3855 | -0.0237 | 0.572 | 5.98e-01 | 8.12e-01 |
| Brier * | 0.1808 | 0.2365 | +0.0558 | -3.359 | 2.83e-02 | 1.25e-01 |
| Accuracy  | 0.7561 | 0.7658 | +0.0098 | -0.228 | 8.31e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4236 | +0.0072 | -0.199 | 8.52e-01 | 1.00e+00 |

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
| M2 | CrossAttn | raw | AUC-ROC | 0.8002 | 0.7079 | 0.8778 |
| M2 | CrossAttn | raw | AUPRC | 0.2946 | 0.1762 | 0.4578 |
| M2 | CrossAttn | raw | Brier | 0.2602 | 0.2325 | 0.2860 |
| M2 | CrossAttn | raw | Accuracy | 0.7293 | 0.6725 | 0.7860 |
| M2 | CrossAttn | raw | F1 | 0.3800 | 0.2553 | 0.4951 |
| M2 | CrossAttn | std_scaled | AUC-ROC | 0.8053 | 0.7191 | 0.8794 |
| M2 | CrossAttn | std_scaled | AUPRC | 0.2943 | 0.1789 | 0.4587 |
| M2 | CrossAttn | std_scaled | Brier | 0.2554 | 0.2283 | 0.2804 |
| M2 | CrossAttn | std_scaled | Accuracy | 0.7074 | 0.6505 | 0.7686 |
| M2 | CrossAttn | std_scaled | F1 | 0.3619 | 0.2391 | 0.4779 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8081 | 0.7264 | 0.8780 |
| M2 | CrossAttn | norm | AUPRC | 0.3089 | 0.1903 | 0.5061 |
| M2 | CrossAttn | norm | Brier | 0.3346 | 0.3047 | 0.3627 |
| M2 | CrossAttn | norm | Accuracy | 0.6201 | 0.5590 | 0.6857 |
| M2 | CrossAttn | norm | F1 | 0.3459 | 0.2362 | 0.4531 |
| M2 | CrossAttn | global_zscore | AUC-ROC | 0.7815 | 0.6981 | 0.8573 |
| M2 | CrossAttn | global_zscore | AUPRC | 0.2688 | 0.1567 | 0.4299 |
| M2 | CrossAttn | global_zscore | Brier | 0.2654 | 0.2365 | 0.2935 |
| M2 | CrossAttn | global_zscore | Accuracy | 0.6900 | 0.6288 | 0.7511 |
| M2 | CrossAttn | global_zscore | F1 | 0.3486 | 0.2281 | 0.4643 |
| M2_2 | CrossAttn | raw | AUC-ROC | 0.8069 | 0.7163 | 0.8814 |
| M2_2 | CrossAttn | raw | AUPRC | 0.3412 | 0.1969 | 0.5288 |
| M2_2 | CrossAttn | raw | Brier | 0.2844 | 0.2562 | 0.3113 |
| M2_2 | CrossAttn | raw | Accuracy | 0.5721 | 0.5066 | 0.6376 |
| M2_2 | CrossAttn | raw | F1 | 0.3099 | 0.2069 | 0.4156 |
| M2_2 | CrossAttn | std_scaled | AUC-ROC | 0.7937 | 0.7066 | 0.8698 |
| M2_2 | CrossAttn | std_scaled | AUPRC | 0.3179 | 0.1875 | 0.5084 |
| M2_2 | CrossAttn | std_scaled | Brier | 0.2663 | 0.2394 | 0.2919 |
| M2_2 | CrossAttn | std_scaled | Accuracy | 0.6288 | 0.5633 | 0.6900 |
| M2_2 | CrossAttn | std_scaled | F1 | 0.3200 | 0.2105 | 0.4320 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8272 | 0.7459 | 0.8943 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3313 | 0.2044 | 0.5316 |
| M2_2 | CrossAttn | norm | Brier | 0.3067 | 0.2768 | 0.3349 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6376 | 0.5764 | 0.7031 |
| M2_2 | CrossAttn | norm | F1 | 0.3566 | 0.2459 | 0.4651 |
| M2_2 | CrossAttn | global_zscore | AUC-ROC | 0.7927 | 0.7044 | 0.8695 |
| M2_2 | CrossAttn | global_zscore | AUPRC | 0.3023 | 0.1741 | 0.4787 |
| M2_2 | CrossAttn | global_zscore | Brier | 0.2518 | 0.2254 | 0.2770 |
| M2_2 | CrossAttn | global_zscore | Accuracy | 0.6769 | 0.6157 | 0.7380 |
| M2_2 | CrossAttn | global_zscore | F1 | 0.3393 | 0.2243 | 0.4538 |
| M3 | CrossAttn3 | raw | AUC-ROC | 0.7894 | 0.7035 | 0.8700 |
| M3 | CrossAttn3 | raw | AUPRC | 0.2754 | 0.1654 | 0.4365 |
| M3 | CrossAttn3 | raw | Brier | 0.2614 | 0.2343 | 0.2879 |
| M3 | CrossAttn3 | raw | Accuracy | 0.7031 | 0.6463 | 0.7642 |
| M3 | CrossAttn3 | raw | F1 | 0.3704 | 0.2500 | 0.4860 |
| M3 | CrossAttn3 | std_scaled | AUC-ROC | 0.7890 | 0.6915 | 0.8751 |
| M3 | CrossAttn3 | std_scaled | AUPRC | 0.2928 | 0.1794 | 0.4949 |
| M3 | CrossAttn3 | std_scaled | Brier | 0.2733 | 0.2444 | 0.3010 |
| M3 | CrossAttn3 | std_scaled | Accuracy | 0.7074 | 0.6507 | 0.7686 |
| M3 | CrossAttn3 | std_scaled | F1 | 0.3619 | 0.2418 | 0.4779 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8154 | 0.7346 | 0.8833 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3393 | 0.2007 | 0.5357 |
| M3 | CrossAttn3 | norm | Brier | 0.2887 | 0.2599 | 0.3154 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6507 | 0.5895 | 0.7162 |
| M3 | CrossAttn3 | norm | F1 | 0.3651 | 0.2481 | 0.4748 |
| M3 | CrossAttn3 | global_zscore | AUC-ROC | 0.8108 | 0.7260 | 0.8852 |
| M3 | CrossAttn3 | global_zscore | AUPRC | 0.3059 | 0.1940 | 0.5031 |
| M3 | CrossAttn3 | global_zscore | Brier | 0.2584 | 0.2318 | 0.2837 |
| M3 | CrossAttn3 | global_zscore | Accuracy | 0.7424 | 0.6856 | 0.7991 |
| M3 | CrossAttn3 | global_zscore | F1 | 0.3789 | 0.2526 | 0.5000 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-raw | 0.8030 | 0.8002 | -0.0028 | 0.117 | 9.067e-01 | ns |
| M1-LR vs M2-std_scaled | 0.8030 | 0.8053 | +0.0022 | -0.095 | 9.244e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8081 | +0.0051 | -0.262 | 7.930e-01 | ns |
| M1-LR vs M2-global_zscore | 0.8030 | 0.7815 | -0.0215 | 0.833 | 4.051e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-raw | 0.8030 | 0.7894 | -0.0136 | 0.490 | 6.238e-01 | ns |
| M1-LR vs M3-std_scaled | 0.8030 | 0.7890 | -0.0140 | 0.424 | 6.719e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8154 | +0.0124 | -0.452 | 6.515e-01 | ns |
| M1-LR vs M3-global_zscore | 0.8030 | 0.8108 | +0.0077 | -0.304 | 7.611e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M2_2-raw | 0.8002 | 0.8069 | +0.0067 | -0.432 | 6.655e-01 | ns |
| M2-std_scaled vs M2_2-std_scaled | 0.8053 | 0.7937 | -0.0116 | 0.630 | 5.288e-01 | ns |
| M2-norm vs M2_2-norm | 0.8081 | 0.8272 | +0.0191 | -1.086 | 2.773e-01 | ns |
| M2-global_zscore vs M2_2-global_zscore | 0.7815 | 0.7927 | +0.0112 | -0.453 | 6.507e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-raw vs M3-raw | 0.8002 | 0.7894 | -0.0108 | 0.602 | 5.469e-01 | ns |
| M2-std_scaled vs M3-std_scaled | 0.8053 | 0.7890 | -0.0163 | 0.890 | 3.735e-01 | ns |
| M2-norm vs M3-norm | 0.8081 | 0.8154 | +0.0073 | -0.419 | 6.749e-01 | ns |
| M2-global_zscore vs M3-global_zscore | 0.7815 | 0.8108 | +0.0293 | -1.765 | 7.758e-02 | † |

