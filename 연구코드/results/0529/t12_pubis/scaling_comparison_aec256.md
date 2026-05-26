# Scaling Comparison — Test Set Performance (AEC 256pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |
| M2 | CrossAttn | crop80 | 0.8657 | 0.4830 | 0.2362 | 0.6221 | 0.3810 |
| M2_2 | CrossAttn | excl_extreme | 0.8691 | 0.4767 | 0.1759 | 0.7727 | 0.4444 |
| M3 | CrossAttn3 | norm | 0.8600 | 0.4777 | 0.1507 | 0.6279 | 0.3725 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_all** | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm | 0.7988 | 0.2594 | 0.2193 | 0.7209 | 0.4146 |
| excl_extreme | 0.8396 | 0.4208 | 0.1830 | 0.7208 | 0.4557 |
| len128 | 0.8575 | 0.4799 | 0.1845 | 0.7558 | 0.4750 |
| **crop80** | 0.8657 | 0.4830 | 0.2362 | 0.6221 | 0.3810 |
| crop60 | 0.8575 | 0.4807 | 0.1947 | 0.7674 | 0.4872 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm | 0.8322 | 0.4236 | 0.1841 | 0.7267 | 0.4337 |
| **excl_extreme** | 0.8691 | 0.4767 | 0.1759 | 0.7727 | 0.4444 |
| len128 | 0.8051 | 0.3847 | 0.1762 | 0.6860 | 0.4000 |
| crop80 | 0.7528 | 0.2747 | 0.2069 | 0.6744 | 0.3488 |
| crop60 | 0.8392 | 0.3485 | 0.2038 | 0.6977 | 0.4222 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm** | 0.8600 | 0.4777 | 0.1507 | 0.6279 | 0.3725 |
| excl_extreme | 0.8464 | 0.5653 | 0.1946 | 0.6558 | 0.4045 |
| len128 | 0.8590 | 0.5586 | 0.2450 | 0.5640 | 0.3478 |
| crop80 | 0.8231 | 0.4210 | 0.2045 | 0.6744 | 0.3778 |
| crop60 | 0.8401 | 0.4136 | 0.1797 | 0.7209 | 0.4146 |

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

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8200 | +0.0087 | -0.987 | 3.79e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.4158 | +0.0311 | -0.624 | 5.66e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1745 | -0.0075 | 0.361 | 7.36e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7381 | -0.0173 | 0.309 | 7.73e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4520 | +0.0005 | -0.013 | 9.91e-01 | 8.12e-01 |

### excl_extreme  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8075 | -0.0038 | 0.333 | 7.56e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.3761 | -0.0085 | 0.138 | 8.97e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1761 | -0.0059 | 0.340 | 7.51e-01 | 8.12e-01 |
| Accuracy  | 0.7555 | 0.7372 | -0.0182 | 0.324 | 7.62e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4317 | -0.0197 | 0.585 | 5.90e-01 | 8.12e-01 |

### len128  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8193 | +0.0080 | -1.646 | 1.75e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3987 | +0.0140 | -0.840 | 4.48e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1789 | -0.0030 | 0.234 | 8.27e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7598 | +0.0044 | -0.102 | 9.24e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4558 | +0.0043 | -0.118 | 9.12e-01 | 8.12e-01 |

### crop80  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8138 | +0.0024 | -0.479 | 6.57e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3812 | -0.0035 | 0.460 | 6.69e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1825 | +0.0005 | -0.061 | 9.54e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7525 | -0.0030 | 0.054 | 9.59e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4626 | +0.0111 | -0.304 | 7.76e-01 | 8.12e-01 |

### crop60  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8162 | +0.0048 | -0.587 | 5.89e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3672 | -0.0175 | 1.138 | 3.19e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.2054 | +0.0234 | -1.228 | 2.87e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7817 | +0.0262 | -1.055 | 3.51e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4772 | +0.0257 | -1.096 | 3.35e-01 | 6.25e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8200 | 0.8081 | -0.0119 | 1.540 | 1.98e-01 | 1.88e-01 |
| AUPRC  | 0.4158 | 0.4079 | -0.0079 | 0.240 | 8.22e-01 | 8.12e-01 |
| Brier  | 0.1745 | 0.1615 | -0.0130 | 0.600 | 5.81e-01 | 8.12e-01 |
| Accuracy  | 0.7381 | 0.7236 | -0.0146 | 0.389 | 7.17e-01 | 8.12e-01 |
| F1  | 0.4520 | 0.4318 | -0.0201 | 0.692 | 5.27e-01 | 6.25e-01 |

#### Case: excl_extreme  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8075 | 0.8141 | +0.0065 | -0.616 | 5.71e-01 | 6.25e-01 |
| AUPRC  | 0.3761 | 0.4065 | +0.0304 | -1.049 | 3.53e-01 | 4.38e-01 |
| Brier  | 0.1761 | 0.1877 | +0.0117 | -0.563 | 6.03e-01 | 8.12e-01 |
| Accuracy  | 0.7372 | 0.7455 | +0.0083 | -0.182 | 8.64e-01 | 1.00e+00 |
| F1  | 0.4317 | 0.4333 | +0.0016 | -0.053 | 9.60e-01 | 1.00e+00 |

#### Case: len128  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.8193 | 0.8111 | -0.0082 | 6.189 | 3.46e-03 | 6.25e-02 |
| AUPRC  | 0.3987 | 0.3782 | -0.0204 | 1.285 | 2.68e-01 | 1.88e-01 |
| Brier  | 0.1789 | 0.1754 | -0.0035 | 0.133 | 9.01e-01 | 1.00e+00 |
| Accuracy  | 0.7598 | 0.7220 | -0.0378 | 1.645 | 1.75e-01 | 1.88e-01 |
| F1  | 0.4558 | 0.4340 | -0.0217 | 1.016 | 3.67e-01 | 4.38e-01 |

#### Case: crop80  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8138 | 0.8052 | -0.0086 | 1.457 | 2.19e-01 | 3.12e-01 |
| AUPRC  | 0.3812 | 0.3755 | -0.0056 | 0.144 | 8.93e-01 | 1.00e+00 |
| Brier  | 0.1825 | 0.1694 | -0.0131 | 1.166 | 3.08e-01 | 4.38e-01 |
| Accuracy  | 0.7525 | 0.7321 | -0.0203 | 0.429 | 6.90e-01 | 1.00e+00 |
| F1  | 0.4626 | 0.4459 | -0.0167 | 0.452 | 6.75e-01 | 8.12e-01 |

#### Case: crop60  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8162 | 0.8172 | +0.0010 | -0.191 | 8.57e-01 | 1.00e+00 |
| AUPRC  | 0.3672 | 0.3700 | +0.0029 | -0.202 | 8.49e-01 | 1.00e+00 |
| Brier  | 0.2054 | 0.1811 | -0.0242 | 0.882 | 4.28e-01 | 8.12e-01 |
| Accuracy  | 0.7817 | 0.7468 | -0.0349 | 1.374 | 2.41e-01 | 3.12e-01 |
| F1  | 0.4772 | 0.4567 | -0.0204 | 1.054 | 3.51e-01 | 3.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8081 | -0.0032 | 0.443 | 6.81e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.4079 | +0.0232 | -0.617 | 5.71e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1615 | -0.0204 | 1.741 | 1.57e-01 | 1.88e-01 |
| Accuracy  | 0.7555 | 0.7236 | -0.0319 | 0.676 | 5.36e-01 | 7.50e-01 |
| F1  | 0.4514 | 0.4318 | -0.0196 | 0.519 | 6.31e-01 | 1.00e+00 |

### excl_extreme  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8141 | +0.0027 | -0.154 | 8.85e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.4065 | +0.0219 | -0.353 | 7.42e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1877 | +0.0058 | -0.254 | 8.12e-01 | 8.12e-01 |
| Accuracy  | 0.7555 | 0.7455 | -0.0099 | 0.270 | 8.01e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4333 | -0.0181 | 0.576 | 5.95e-01 | 8.12e-01 |

### len128  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8111 | -0.0003 | 0.058 | 9.57e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.3782 | -0.0064 | 0.300 | 7.79e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1754 | -0.0065 | 0.464 | 6.67e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7220 | -0.0334 | 0.913 | 4.13e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4340 | -0.0174 | 0.657 | 5.47e-01 | 1.00e+00 |

### crop80  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8052 | -0.0062 | 1.013 | 3.68e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3755 | -0.0091 | 0.278 | 7.95e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.1694 | -0.0125 | 0.968 | 3.88e-01 | 4.38e-01 |
| Accuracy  | 0.7555 | 0.7321 | -0.0233 | 1.241 | 2.82e-01 | 4.38e-01 |
| F1  | 0.4514 | 0.4459 | -0.0056 | 0.617 | 5.71e-01 | 6.25e-01 |

### crop60  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8172 | +0.0059 | -0.556 | 6.08e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3700 | -0.0146 | 1.172 | 3.06e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1811 | -0.0008 | 0.054 | 9.60e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7468 | -0.0087 | 0.382 | 7.22e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4567 | +0.0053 | -0.469 | 6.64e-01 | 8.12e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_all | AUC-ROC | 0.8325 | 0.7386 | 0.9091 |
| M1 | LR | scale_all | AUPRC | 0.5008 | 0.2965 | 0.6937 |
| M1 | LR | scale_all | Brier | 0.1804 | 0.1527 | 0.2118 |
| M1 | LR | scale_all | Accuracy | 0.6977 | 0.6279 | 0.7674 |
| M1 | LR | scale_all | F1 | 0.3953 | 0.2535 | 0.5228 |
| M2 | CrossAttn | norm | AUC-ROC | 0.7988 | 0.7217 | 0.8664 |
| M2 | CrossAttn | norm | AUPRC | 0.2594 | 0.1625 | 0.4230 |
| M2 | CrossAttn | norm | Brier | 0.2193 | 0.1801 | 0.2584 |
| M2 | CrossAttn | norm | Accuracy | 0.7209 | 0.6512 | 0.7907 |
| M2 | CrossAttn | norm | F1 | 0.4146 | 0.2702 | 0.5455 |
| M2 | CrossAttn | excl_extreme | AUC-ROC | 0.8396 | 0.7641 | 0.9036 |
| M2 | CrossAttn | excl_extreme | AUPRC | 0.4208 | 0.2435 | 0.6203 |
| M2 | CrossAttn | excl_extreme | Brier | 0.1830 | 0.1458 | 0.2222 |
| M2 | CrossAttn | excl_extreme | Accuracy | 0.7208 | 0.6494 | 0.7922 |
| M2 | CrossAttn | excl_extreme | F1 | 0.4557 | 0.3188 | 0.5883 |
| M2 | CrossAttn | len128 | AUC-ROC | 0.8575 | 0.7682 | 0.9279 |
| M2 | CrossAttn | len128 | AUPRC | 0.4799 | 0.2950 | 0.7054 |
| M2 | CrossAttn | len128 | Brier | 0.1845 | 0.1488 | 0.2215 |
| M2 | CrossAttn | len128 | Accuracy | 0.7558 | 0.6919 | 0.8198 |
| M2 | CrossAttn | len128 | F1 | 0.4750 | 0.3333 | 0.6042 |
| M2 | CrossAttn | crop80 | AUC-ROC | 0.8657 | 0.7850 | 0.9317 |
| M2 | CrossAttn | crop80 | AUPRC | 0.4830 | 0.3038 | 0.7053 |
| M2 | CrossAttn | crop80 | Brier | 0.2362 | 0.2003 | 0.2725 |
| M2 | CrossAttn | crop80 | Accuracy | 0.6221 | 0.5465 | 0.6920 |
| M2 | CrossAttn | crop80 | F1 | 0.3810 | 0.2574 | 0.4957 |
| M2 | CrossAttn | crop60 | AUC-ROC | 0.8575 | 0.7826 | 0.9208 |
| M2 | CrossAttn | crop60 | AUPRC | 0.4807 | 0.2837 | 0.6703 |
| M2 | CrossAttn | crop60 | Brier | 0.1947 | 0.1639 | 0.2268 |
| M2 | CrossAttn | crop60 | Accuracy | 0.7674 | 0.7035 | 0.8314 |
| M2 | CrossAttn | crop60 | F1 | 0.4872 | 0.3380 | 0.6216 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8322 | 0.7193 | 0.9260 |
| M2_2 | CrossAttn | norm | AUPRC | 0.4236 | 0.2562 | 0.6455 |
| M2_2 | CrossAttn | norm | Brier | 0.1841 | 0.1455 | 0.2256 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7267 | 0.6568 | 0.7907 |
| M2_2 | CrossAttn | norm | F1 | 0.4337 | 0.2857 | 0.5618 |
| M2_2 | CrossAttn | excl_extreme | AUC-ROC | 0.8691 | 0.7430 | 0.9479 |
| M2_2 | CrossAttn | excl_extreme | AUPRC | 0.4767 | 0.2585 | 0.7161 |
| M2_2 | CrossAttn | excl_extreme | Brier | 0.1759 | 0.1425 | 0.2121 |
| M2_2 | CrossAttn | excl_extreme | Accuracy | 0.7727 | 0.7013 | 0.8377 |
| M2_2 | CrossAttn | excl_extreme | F1 | 0.4444 | 0.2711 | 0.5883 |
| M2_2 | CrossAttn | len128 | AUC-ROC | 0.8051 | 0.6853 | 0.9035 |
| M2_2 | CrossAttn | len128 | AUPRC | 0.3847 | 0.2178 | 0.5969 |
| M2_2 | CrossAttn | len128 | Brier | 0.1762 | 0.1353 | 0.2195 |
| M2_2 | CrossAttn | len128 | Accuracy | 0.6860 | 0.6163 | 0.7558 |
| M2_2 | CrossAttn | len128 | F1 | 0.4000 | 0.2647 | 0.5243 |
| M2_2 | CrossAttn | crop80 | AUC-ROC | 0.7528 | 0.6376 | 0.8576 |
| M2_2 | CrossAttn | crop80 | AUPRC | 0.2747 | 0.1631 | 0.4742 |
| M2_2 | CrossAttn | crop80 | Brier | 0.2069 | 0.1635 | 0.2527 |
| M2_2 | CrossAttn | crop80 | Accuracy | 0.6744 | 0.5988 | 0.7442 |
| M2_2 | CrossAttn | crop80 | F1 | 0.3488 | 0.2121 | 0.4706 |
| M2_2 | CrossAttn | crop60 | AUC-ROC | 0.8392 | 0.7523 | 0.9089 |
| M2_2 | CrossAttn | crop60 | AUPRC | 0.3485 | 0.2144 | 0.5600 |
| M2_2 | CrossAttn | crop60 | Brier | 0.2038 | 0.1712 | 0.2395 |
| M2_2 | CrossAttn | crop60 | Accuracy | 0.6977 | 0.6221 | 0.7618 |
| M2_2 | CrossAttn | crop60 | F1 | 0.4222 | 0.2826 | 0.5437 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8600 | 0.7749 | 0.9265 |
| M3 | CrossAttn3 | norm | AUPRC | 0.4777 | 0.2876 | 0.6829 |
| M3 | CrossAttn3 | norm | Brier | 0.1507 | 0.1246 | 0.1791 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6279 | 0.5523 | 0.7035 |
| M3 | CrossAttn3 | norm | F1 | 0.3725 | 0.2500 | 0.4870 |
| M3 | CrossAttn3 | excl_extreme | AUC-ROC | 0.8464 | 0.7434 | 0.9292 |
| M3 | CrossAttn3 | excl_extreme | AUPRC | 0.5653 | 0.3503 | 0.7591 |
| M3 | CrossAttn3 | excl_extreme | Brier | 0.1946 | 0.1565 | 0.2346 |
| M3 | CrossAttn3 | excl_extreme | Accuracy | 0.6558 | 0.5779 | 0.7273 |
| M3 | CrossAttn3 | excl_extreme | F1 | 0.4045 | 0.2683 | 0.5243 |
| M3 | CrossAttn3 | len128 | AUC-ROC | 0.8590 | 0.7697 | 0.9322 |
| M3 | CrossAttn3 | len128 | AUPRC | 0.5586 | 0.3472 | 0.7545 |
| M3 | CrossAttn3 | len128 | Brier | 0.2450 | 0.2060 | 0.2851 |
| M3 | CrossAttn3 | len128 | Accuracy | 0.5640 | 0.4884 | 0.6395 |
| M3 | CrossAttn3 | len128 | F1 | 0.3478 | 0.2316 | 0.4590 |
| M3 | CrossAttn3 | crop80 | AUC-ROC | 0.8231 | 0.7370 | 0.9020 |
| M3 | CrossAttn3 | crop80 | AUPRC | 0.4210 | 0.2333 | 0.6435 |
| M3 | CrossAttn3 | crop80 | Brier | 0.2045 | 0.1620 | 0.2493 |
| M3 | CrossAttn3 | crop80 | Accuracy | 0.6744 | 0.5988 | 0.7442 |
| M3 | CrossAttn3 | crop80 | F1 | 0.3778 | 0.2424 | 0.5047 |
| M3 | CrossAttn3 | crop60 | AUC-ROC | 0.8401 | 0.7435 | 0.9203 |
| M3 | CrossAttn3 | crop60 | AUPRC | 0.4136 | 0.2517 | 0.6529 |
| M3 | CrossAttn3 | crop60 | Brier | 0.1797 | 0.1419 | 0.2184 |
| M3 | CrossAttn3 | crop60 | Accuracy | 0.7209 | 0.6512 | 0.7907 |
| M3 | CrossAttn3 | crop60 | F1 | 0.4146 | 0.2703 | 0.5476 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-norm | 0.8325 | 0.7988 | -0.0337 | 0.633 | 5.265e-01 | ns |
| M1-LR vs M2-len128 | 0.8325 | 0.8575 | +0.0249 | -0.905 | 3.652e-01 | ns |
| M1-LR vs M2-crop80 | 0.8325 | 0.8657 | +0.0331 | -1.320 | 1.868e-01 | ns |
| M1-LR vs M2-crop60 | 0.8325 | 0.8575 | +0.0249 | -0.919 | 3.581e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-norm | 0.8325 | 0.8600 | +0.0274 | -1.057 | 2.904e-01 | ns |
| M1-LR vs M3-len128 | 0.8325 | 0.8590 | +0.0265 | -0.800 | 4.236e-01 | ns |
| M1-LR vs M3-crop80 | 0.8325 | 0.8231 | -0.0095 | 0.207 | 8.358e-01 | ns |
| M1-LR vs M3-crop60 | 0.8325 | 0.8401 | +0.0076 | -0.198 | 8.434e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M2_2-norm | 0.7988 | 0.8322 | +0.0334 | -0.596 | 5.509e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8396 | 0.5435 | -0.2961 | 3.786 | 1.529e-04 | *** |
| M2-len128 vs M2_2-len128 | 0.8575 | 0.8051 | -0.0523 | 1.418 | 1.563e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.8657 | 0.7528 | -0.1129 | 3.085 | 2.036e-03 | ** |
| M2-crop60 vs M2_2-crop60 | 0.8575 | 0.8392 | -0.0183 | 0.725 | 4.686e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M3-norm | 0.7988 | 0.8600 | +0.0612 | -1.289 | 1.973e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.8396 | 0.8464 | +0.0068 | -0.259 | 7.956e-01 | ns |
| M2-len128 vs M3-len128 | 0.8575 | 0.8590 | +0.0016 | -0.065 | 9.481e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8657 | 0.8231 | -0.0426 | 1.283 | 1.994e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.8575 | 0.8401 | -0.0173 | 0.674 | 5.004e-01 | ns |

