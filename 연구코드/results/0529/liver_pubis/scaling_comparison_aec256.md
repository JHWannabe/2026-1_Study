# Scaling Comparison — Test Set Performance (AEC 256pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8127 | 0.4105 | 0.1869 | 0.6481 | 0.3387 |
| M2 | CrossAttn | norm | 0.8269 | 0.4151 | 0.1966 | 0.5193 | 0.3000 |
| M2_2 | CrossAttn | norm | 0.8302 | 0.4266 | 0.1525 | 0.7511 | 0.3696 |
| M3 | CrossAttn3 | excl_extreme | 0.8274 | 0.4327 | 0.2231 | 0.6938 | 0.3469 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_all** | 0.8127 | 0.4105 | 0.1869 | 0.6481 | 0.3387 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm** | 0.8269 | 0.4151 | 0.1966 | 0.5193 | 0.3000 |
| excl_extreme | 0.8099 | 0.4257 | 0.2272 | 0.6172 | 0.3103 |
| len128 | 0.8075 | 0.4064 | 0.1825 | 0.7253 | 0.3846 |
| crop80 | 0.8133 | 0.3956 | 0.2106 | 0.6567 | 0.3443 |
| crop60 | 0.7919 | 0.3725 | 0.2007 | 0.7597 | 0.3778 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **norm** | 0.8302 | 0.4266 | 0.1525 | 0.7511 | 0.3696 |
| excl_extreme | 0.7931 | 0.3778 | 0.1453 | 0.7656 | 0.3467 |
| len128 | 0.8125 | 0.3178 | 0.1578 | 0.7639 | 0.3373 |
| crop80 | 0.7794 | 0.3347 | 0.2428 | 0.6223 | 0.3016 |
| crop60 | 0.7821 | 0.3821 | 0.1883 | 0.7425 | 0.3333 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm | 0.8063 | 0.3669 | 0.2394 | 0.6524 | 0.3306 |
| **excl_extreme** | 0.8274 | 0.4327 | 0.2231 | 0.6938 | 0.3469 |
| len128 | 0.8096 | 0.4306 | 0.1963 | 0.6867 | 0.3540 |
| crop80 | 0.8129 | 0.3738 | 0.2328 | 0.6180 | 0.3101 |
| crop60 | 0.8269 | 0.4038 | 0.2179 | 0.6695 | 0.3529 |

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
| AUC-ROC  | 0.8090 | 0.8183 | +0.0093 | -0.617 | 5.70e-01 | 6.25e-01 |
| AUPRC  | 0.3931 | 0.3640 | -0.0290 | 0.815 | 4.61e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.1602 | -0.0198 | 2.121 | 1.01e-01 | 1.25e-01 |
| Accuracy  | 0.7086 | 0.6946 | -0.0140 | 0.207 | 8.46e-01 | 8.75e-01 |
| F1  | 0.4075 | 0.3876 | -0.0199 | 0.398 | 7.11e-01 | 1.00e+00 |

### excl_extreme  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8287 | +0.0196 | -0.681 | 5.33e-01 | 1.00e+00 |
| AUPRC  | 0.3931 | 0.4194 | +0.0263 | -0.572 | 5.98e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.1762 | -0.0038 | 0.151 | 8.87e-01 | 8.12e-01 |
| Accuracy  | 0.7086 | 0.7346 | +0.0260 | -0.589 | 5.88e-01 | 6.25e-01 |
| F1  | 0.4075 | 0.4042 | -0.0033 | 0.096 | 9.28e-01 | 1.00e+00 |

### len128  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8296 | +0.0206 | -1.564 | 1.93e-01 | 1.88e-01 |
| AUPRC  | 0.3931 | 0.4363 | +0.0432 | -1.080 | 3.41e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1979 | +0.0179 | -1.166 | 3.08e-01 | 6.25e-01 |
| Accuracy  | 0.7086 | 0.7409 | +0.0323 | -0.669 | 5.40e-01 | 8.12e-01 |
| F1  | 0.4075 | 0.4200 | +0.0125 | -0.394 | 7.14e-01 | 1.00e+00 |

### crop80  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8249 | +0.0159 | -1.283 | 2.69e-01 | 4.38e-01 |
| AUPRC  | 0.3931 | 0.4029 | +0.0098 | -0.466 | 6.65e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1830 | +0.0029 | -0.183 | 8.63e-01 | 1.00e+00 |
| Accuracy  | 0.7086 | 0.7376 | +0.0290 | -0.776 | 4.81e-01 | 8.75e-01 |
| F1  | 0.4075 | 0.4216 | +0.0141 | -0.474 | 6.60e-01 | 1.00e+00 |

### crop60  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8259 | +0.0169 | -1.391 | 2.37e-01 | 4.38e-01 |
| AUPRC  | 0.3931 | 0.4225 | +0.0294 | -0.635 | 5.60e-01 | 6.25e-01 |
| Brier  | 0.1800 | 0.1778 | -0.0022 | 0.186 | 8.62e-01 | 8.12e-01 |
| Accuracy * | 0.7086 | 0.7978 | +0.0892 | -2.808 | 4.84e-02 | 1.25e-01 |
| F1 † | 0.4075 | 0.4575 | +0.0500 | -2.470 | 6.90e-02 | 1.25e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8183 | 0.8119 | -0.0064 | 0.635 | 5.60e-01 | 6.25e-01 |
| AUPRC  | 0.3640 | 0.3884 | +0.0243 | -0.765 | 4.87e-01 | 8.12e-01 |
| Brier  | 0.1602 | 0.1700 | +0.0097 | -0.772 | 4.83e-01 | 4.38e-01 |
| Accuracy  | 0.6946 | 0.7484 | +0.0538 | -1.093 | 3.36e-01 | 4.38e-01 |
| F1  | 0.3876 | 0.4104 | +0.0228 | -0.825 | 4.56e-01 | 6.25e-01 |

#### Case: excl_extreme  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8287 | 0.8296 | +0.0009 | -0.345 | 7.48e-01 | 1.00e+00 |
| AUPRC  | 0.4194 | 0.3908 | -0.0287 | 0.934 | 4.03e-01 | 6.25e-01 |
| Brier  | 0.1762 | 0.1686 | -0.0077 | 0.416 | 6.99e-01 | 6.25e-01 |
| Accuracy  | 0.7346 | 0.7883 | +0.0537 | -1.696 | 1.65e-01 | 2.50e-01 |
| F1 † | 0.4042 | 0.4487 | +0.0445 | -2.220 | 9.06e-02 | 1.25e-01 |

#### Case: len128  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8296 | 0.8167 | -0.0130 | 2.001 | 1.16e-01 | 6.25e-02 |
| AUPRC † | 0.4363 | 0.3849 | -0.0514 | 2.745 | 5.16e-02 | 6.25e-02 |
| Brier  | 0.1979 | 0.1644 | -0.0335 | 1.850 | 1.38e-01 | 1.25e-01 |
| Accuracy  | 0.7409 | 0.7591 | +0.0183 | -0.876 | 4.30e-01 | 6.25e-01 |
| F1  | 0.4200 | 0.4267 | +0.0067 | -0.333 | 7.56e-01 | 8.12e-01 |

#### Case: crop80  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8249 | 0.8207 | -0.0042 | 0.375 | 7.27e-01 | 1.00e+00 |
| AUPRC  | 0.4029 | 0.3958 | -0.0071 | 0.154 | 8.85e-01 | 8.12e-01 |
| Brier  | 0.1830 | 0.1739 | -0.0091 | 0.811 | 4.63e-01 | 4.38e-01 |
| Accuracy  | 0.7376 | 0.7312 | -0.0065 | 0.229 | 8.30e-01 | 1.00e+00 |
| F1  | 0.4216 | 0.4112 | -0.0104 | 0.449 | 6.77e-01 | 1.00e+00 |

#### Case: crop60  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8259 | 0.8137 | -0.0123 | 1.878 | 1.34e-01 | 1.25e-01 |
| AUPRC † | 0.4225 | 0.3428 | -0.0797 | 2.203 | 9.24e-02 | 1.25e-01 |
| Brier  | 0.1778 | 0.1978 | +0.0200 | -0.747 | 4.96e-01 | 6.25e-01 |
| Accuracy * | 0.7978 | 0.6978 | -0.1000 | 3.882 | 1.78e-02 | 6.25e-02 |
| F1 * | 0.4575 | 0.3995 | -0.0579 | 2.999 | 4.00e-02 | 6.25e-02 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8119 | +0.0029 | -0.129 | 9.04e-01 | 8.12e-01 |
| AUPRC  | 0.3931 | 0.3884 | -0.0047 | 0.073 | 9.45e-01 | 8.12e-01 |
| Brier  | 0.1800 | 0.1700 | -0.0101 | 1.095 | 3.35e-01 | 4.38e-01 |
| Accuracy  | 0.7086 | 0.7484 | +0.0398 | -0.960 | 3.92e-01 | 3.12e-01 |
| F1  | 0.4075 | 0.4104 | +0.0029 | -0.080 | 9.40e-01 | 6.25e-01 |

### excl_extreme  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8296 | +0.0205 | -0.722 | 5.10e-01 | 8.12e-01 |
| AUPRC  | 0.3931 | 0.3908 | -0.0023 | 0.039 | 9.71e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1686 | -0.0114 | 0.640 | 5.57e-01 | 8.12e-01 |
| Accuracy  | 0.7086 | 0.7883 | +0.0797 | -1.675 | 1.69e-01 | 3.12e-01 |
| F1  | 0.4075 | 0.4487 | +0.0412 | -1.257 | 2.77e-01 | 6.25e-01 |

### len128  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8167 | +0.0076 | -0.456 | 6.72e-01 | 6.25e-01 |
| AUPRC  | 0.3931 | 0.3849 | -0.0082 | 0.178 | 8.67e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1644 | -0.0156 | 1.102 | 3.32e-01 | 4.38e-01 |
| Accuracy  | 0.7086 | 0.7591 | +0.0505 | -0.976 | 3.84e-01 | 5.00e-01 |
| F1  | 0.4075 | 0.4267 | +0.0192 | -0.462 | 6.68e-01 | 8.12e-01 |

### crop80  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8207 | +0.0117 | -0.569 | 6.00e-01 | 6.25e-01 |
| AUPRC  | 0.3931 | 0.3958 | +0.0028 | -0.049 | 9.63e-01 | 1.00e+00 |
| Brier  | 0.1800 | 0.1739 | -0.0061 | 0.450 | 6.76e-01 | 1.00e+00 |
| Accuracy  | 0.7086 | 0.7312 | +0.0226 | -0.480 | 6.56e-01 | 6.25e-01 |
| F1  | 0.4075 | 0.4112 | +0.0037 | -0.095 | 9.29e-01 | 6.25e-01 |

### crop60  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8090 | 0.8137 | +0.0047 | -0.321 | 7.64e-01 | 6.25e-01 |
| AUPRC  | 0.3931 | 0.3428 | -0.0503 | 1.345 | 2.50e-01 | 3.12e-01 |
| Brier  | 0.1800 | 0.1978 | +0.0178 | -0.894 | 4.22e-01 | 8.12e-01 |
| Accuracy  | 0.7086 | 0.6978 | -0.0108 | 0.527 | 6.26e-01 | 8.75e-01 |
| F1  | 0.4075 | 0.3995 | -0.0079 | 0.481 | 6.55e-01 | 8.12e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_all | AUC-ROC | 0.8127 | 0.7287 | 0.8883 |
| M1 | LR | scale_all | AUPRC | 0.4105 | 0.2428 | 0.5883 |
| M1 | LR | scale_all | Brier | 0.1869 | 0.1618 | 0.2127 |
| M1 | LR | scale_all | Accuracy | 0.6481 | 0.5880 | 0.7082 |
| M1 | LR | scale_all | F1 | 0.3387 | 0.2376 | 0.4480 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8269 | 0.7446 | 0.9037 |
| M2 | CrossAttn | norm | AUPRC | 0.4151 | 0.2483 | 0.6093 |
| M2 | CrossAttn | norm | Brier | 0.1966 | 0.1711 | 0.2242 |
| M2 | CrossAttn | norm | Accuracy | 0.5193 | 0.4549 | 0.5837 |
| M2 | CrossAttn | norm | F1 | 0.3000 | 0.2105 | 0.3944 |
| M2 | CrossAttn | excl_extreme | AUC-ROC | 0.8099 | 0.7171 | 0.8890 |
| M2 | CrossAttn | excl_extreme | AUPRC | 0.4257 | 0.2405 | 0.6144 |
| M2 | CrossAttn | excl_extreme | Brier | 0.2272 | 0.1935 | 0.2638 |
| M2 | CrossAttn | excl_extreme | Accuracy | 0.6172 | 0.5502 | 0.6794 |
| M2 | CrossAttn | excl_extreme | F1 | 0.3103 | 0.1982 | 0.4122 |
| M2 | CrossAttn | len128 | AUC-ROC | 0.8075 | 0.7194 | 0.8901 |
| M2 | CrossAttn | len128 | AUPRC | 0.4064 | 0.2375 | 0.6043 |
| M2 | CrossAttn | len128 | Brier | 0.1825 | 0.1535 | 0.2118 |
| M2 | CrossAttn | len128 | Accuracy | 0.7253 | 0.6695 | 0.7854 |
| M2 | CrossAttn | len128 | F1 | 0.3846 | 0.2667 | 0.5082 |
| M2 | CrossAttn | crop80 | AUC-ROC | 0.8133 | 0.7307 | 0.8913 |
| M2 | CrossAttn | crop80 | AUPRC | 0.3956 | 0.2359 | 0.5893 |
| M2 | CrossAttn | crop80 | Brier | 0.2106 | 0.1800 | 0.2418 |
| M2 | CrossAttn | crop80 | Accuracy | 0.6567 | 0.5966 | 0.7167 |
| M2 | CrossAttn | crop80 | F1 | 0.3443 | 0.2385 | 0.4526 |
| M2 | CrossAttn | crop60 | AUC-ROC | 0.7919 | 0.6858 | 0.8808 |
| M2 | CrossAttn | crop60 | AUPRC | 0.3725 | 0.2194 | 0.5751 |
| M2 | CrossAttn | crop60 | Brier | 0.2007 | 0.1661 | 0.2342 |
| M2 | CrossAttn | crop60 | Accuracy | 0.7597 | 0.7039 | 0.8155 |
| M2 | CrossAttn | crop60 | F1 | 0.3778 | 0.2474 | 0.5053 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8302 | 0.7518 | 0.9007 |
| M2_2 | CrossAttn | norm | AUPRC | 0.4266 | 0.2569 | 0.6166 |
| M2_2 | CrossAttn | norm | Brier | 0.1525 | 0.1234 | 0.1841 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7511 | 0.6953 | 0.8026 |
| M2_2 | CrossAttn | norm | F1 | 0.3696 | 0.2368 | 0.4944 |
| M2_2 | CrossAttn | excl_extreme | AUC-ROC | 0.7931 | 0.6961 | 0.8790 |
| M2_2 | CrossAttn | excl_extreme | AUPRC | 0.3778 | 0.2017 | 0.5921 |
| M2_2 | CrossAttn | excl_extreme | Brier | 0.1453 | 0.1171 | 0.1740 |
| M2_2 | CrossAttn | excl_extreme | Accuracy | 0.7656 | 0.7081 | 0.8230 |
| M2_2 | CrossAttn | excl_extreme | F1 | 0.3467 | 0.2069 | 0.4918 |
| M2_2 | CrossAttn | len128 | AUC-ROC | 0.8125 | 0.7398 | 0.8785 |
| M2_2 | CrossAttn | len128 | AUPRC | 0.3178 | 0.1938 | 0.5029 |
| M2_2 | CrossAttn | len128 | Brier | 0.1578 | 0.1314 | 0.1862 |
| M2_2 | CrossAttn | len128 | Accuracy | 0.7639 | 0.7124 | 0.8155 |
| M2_2 | CrossAttn | len128 | F1 | 0.3373 | 0.1999 | 0.4658 |
| M2_2 | CrossAttn | crop80 | AUC-ROC | 0.7794 | 0.6950 | 0.8564 |
| M2_2 | CrossAttn | crop80 | AUPRC | 0.3347 | 0.1854 | 0.5130 |
| M2_2 | CrossAttn | crop80 | Brier | 0.2428 | 0.2091 | 0.2787 |
| M2_2 | CrossAttn | crop80 | Accuracy | 0.6223 | 0.5579 | 0.6867 |
| M2_2 | CrossAttn | crop80 | F1 | 0.3016 | 0.1964 | 0.4077 |
| M2_2 | CrossAttn | crop60 | AUC-ROC | 0.7821 | 0.6961 | 0.8631 |
| M2_2 | CrossAttn | crop60 | AUPRC | 0.3821 | 0.2196 | 0.5589 |
| M2_2 | CrossAttn | crop60 | Brier | 0.1883 | 0.1592 | 0.2186 |
| M2_2 | CrossAttn | crop60 | Accuracy | 0.7425 | 0.6867 | 0.7983 |
| M2_2 | CrossAttn | crop60 | F1 | 0.3333 | 0.2069 | 0.4555 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8063 | 0.7199 | 0.8891 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3669 | 0.2143 | 0.5583 |
| M3 | CrossAttn3 | norm | Brier | 0.2394 | 0.2082 | 0.2722 |
| M3 | CrossAttn3 | norm | Accuracy | 0.6524 | 0.5922 | 0.7124 |
| M3 | CrossAttn3 | norm | F1 | 0.3306 | 0.2243 | 0.4390 |
| M3 | CrossAttn3 | excl_extreme | AUC-ROC | 0.8274 | 0.7259 | 0.9109 |
| M3 | CrossAttn3 | excl_extreme | AUPRC | 0.4327 | 0.2489 | 0.6335 |
| M3 | CrossAttn3 | excl_extreme | Brier | 0.2231 | 0.1876 | 0.2611 |
| M3 | CrossAttn3 | excl_extreme | Accuracy | 0.6938 | 0.6316 | 0.7512 |
| M3 | CrossAttn3 | excl_extreme | F1 | 0.3469 | 0.2198 | 0.4572 |
| M3 | CrossAttn3 | len128 | AUC-ROC | 0.8096 | 0.7167 | 0.8937 |
| M3 | CrossAttn3 | len128 | AUPRC | 0.4306 | 0.2635 | 0.6177 |
| M3 | CrossAttn3 | len128 | Brier | 0.1963 | 0.1674 | 0.2252 |
| M3 | CrossAttn3 | len128 | Accuracy | 0.6867 | 0.6266 | 0.7425 |
| M3 | CrossAttn3 | len128 | F1 | 0.3540 | 0.2391 | 0.4697 |
| M3 | CrossAttn3 | crop80 | AUC-ROC | 0.8129 | 0.7294 | 0.8880 |
| M3 | CrossAttn3 | crop80 | AUPRC | 0.3738 | 0.2188 | 0.5604 |
| M3 | CrossAttn3 | crop80 | Brier | 0.2328 | 0.1989 | 0.2679 |
| M3 | CrossAttn3 | crop80 | Accuracy | 0.6180 | 0.5579 | 0.6781 |
| M3 | CrossAttn3 | crop80 | F1 | 0.3101 | 0.2087 | 0.4154 |
| M3 | CrossAttn3 | crop60 | AUC-ROC | 0.8269 | 0.7460 | 0.9002 |
| M3 | CrossAttn3 | crop60 | AUPRC | 0.4038 | 0.2435 | 0.5957 |
| M3 | CrossAttn3 | crop60 | Brier | 0.2179 | 0.1857 | 0.2526 |
| M3 | CrossAttn3 | crop60 | Accuracy | 0.6695 | 0.6094 | 0.7296 |
| M3 | CrossAttn3 | crop60 | F1 | 0.3529 | 0.2474 | 0.4667 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-norm | 0.8127 | 0.8269 | +0.0142 | -0.591 | 5.543e-01 | ns |
| M1-LR vs M2-len128 | 0.8127 | 0.8075 | -0.0052 | 0.254 | 7.999e-01 | ns |
| M1-LR vs M2-crop80 | 0.8127 | 0.8133 | +0.0006 | -0.021 | 9.835e-01 | ns |
| M1-LR vs M2-crop60 | 0.8127 | 0.7919 | -0.0208 | 0.801 | 4.230e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-norm | 0.8127 | 0.8063 | -0.0063 | 0.270 | 7.869e-01 | ns |
| M1-LR vs M3-len128 | 0.8127 | 0.8096 | -0.0031 | 0.137 | 8.910e-01 | ns |
| M1-LR vs M3-crop80 | 0.8127 | 0.8129 | +0.0002 | -0.009 | 9.931e-01 | ns |
| M1-LR vs M3-crop60 | 0.8127 | 0.8269 | +0.0142 | -0.676 | 4.990e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M2_2-norm | 0.8269 | 0.8302 | +0.0033 | -0.223 | 8.232e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8099 | 0.4932 | -0.3167 | 5.155 | 2.538e-07 | *** |
| M2-len128 vs M2_2-len128 | 0.8075 | 0.8125 | +0.0050 | -0.205 | 8.372e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.8133 | 0.7794 | -0.0338 | 0.990 | 3.222e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.7919 | 0.7821 | -0.0098 | 0.322 | 7.471e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M3-norm | 0.8269 | 0.8063 | -0.0206 | 1.247 | 2.122e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.8099 | 0.8274 | +0.0175 | -0.725 | 4.682e-01 | ns |
| M2-len128 vs M3-len128 | 0.8075 | 0.8096 | +0.0021 | -0.099 | 9.209e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8133 | 0.8129 | -0.0004 | 0.020 | 9.840e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.7919 | 0.8269 | +0.0350 | -1.699 | 8.940e-02 | † |

