# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | crop60 | 0.8132 | 0.3230 | 0.2308 | 0.6419 | 0.3492 |
| M2_2 | CrossAttn | norm | 0.8268 | 0.3300 | 0.2012 | 0.7555 | 0.3913 |
| M3 | CrossAttn3 | norm | 0.8492 | 0.3519 | 0.1796 | 0.7380 | 0.4118 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_all** | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.7974 | 0.3049 | 0.1501 | 0.7380 | 0.3478 |
| norm | 0.8118 | 0.2994 | 0.1985 | 0.6681 | 0.3770 |
| crop80 | 0.8041 | 0.3638 | 0.2134 | 0.7380 | 0.3750 |
| **crop60** | 0.8132 | 0.3230 | 0.2308 | 0.6419 | 0.3492 |
| excl_extreme | 0.7923 | 0.2954 | 0.1924 | 0.7073 | 0.3478 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.8118 | 0.3505 | 0.2188 | 0.6812 | 0.3423 |
| **norm** | 0.8268 | 0.3300 | 0.2012 | 0.7555 | 0.3913 |
| crop80 | 0.8087 | 0.3210 | 0.2298 | 0.6419 | 0.3387 |
| crop60 | 0.7992 | 0.3391 | 0.2053 | 0.6681 | 0.3214 |
| excl_extreme | 0.7229 | 0.3045 | 0.2256 | 0.6244 | 0.2936 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.8004 | 0.2862 | 0.1776 | 0.7074 | 0.3619 |
| **norm** | 0.8492 | 0.3519 | 0.1796 | 0.7380 | 0.4118 |
| crop80 | 0.8024 | 0.2900 | 0.2250 | 0.6070 | 0.3284 |
| crop60 | 0.7669 | 0.2538 | 0.1884 | 0.7118 | 0.3529 |
| excl_extreme | 0.7668 | 0.2446 | 0.2781 | 0.6390 | 0.3509 |

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

### len128  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8148 | +0.0086 | -0.693 | 5.26e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4073 | -0.0019 | 0.049 | 9.63e-01 | 8.12e-01 |
| Brier  | 0.1808 | 0.1824 | +0.0016 | -0.079 | 9.41e-01 | 1.00e+00 |
| Accuracy  | 0.7561 | 0.7669 | +0.0109 | -0.261 | 8.07e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4315 | +0.0151 | -0.346 | 7.47e-01 | 1.00e+00 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8124 | +0.0063 | -0.482 | 6.55e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4131 | +0.0039 | -0.118 | 9.12e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1528 | -0.0280 | 1.284 | 2.68e-01 | 3.12e-01 |
| Accuracy  | 0.7561 | 0.7757 | +0.0196 | -0.418 | 6.97e-01 | 7.50e-01 |
| F1  | 0.4163 | 0.4419 | +0.0255 | -0.546 | 6.14e-01 | 6.25e-01 |

### crop80  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8209 | +0.0148 | -0.996 | 3.76e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3672 | -0.0420 | 1.131 | 3.21e-01 | 6.25e-01 |
| Brier  | 0.1808 | 0.1757 | -0.0051 | 0.297 | 7.81e-01 | 6.25e-01 |
| Accuracy  | 0.7561 | 0.7813 | +0.0252 | -0.564 | 6.03e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4406 | +0.0242 | -0.650 | 5.51e-01 | 6.25e-01 |

### crop60  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8178 | +0.0116 | -0.981 | 3.82e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4032 | -0.0060 | 0.224 | 8.34e-01 | 8.12e-01 |
| Brier  | 0.1808 | 0.1915 | +0.0108 | -1.305 | 2.62e-01 | 3.12e-01 |
| Accuracy  | 0.7561 | 0.7309 | -0.0252 | 0.646 | 5.53e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4282 | +0.0118 | -0.265 | 8.04e-01 | 1.00e+00 |

### excl_extreme  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8237 | +0.0175 | -0.378 | 7.25e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4215 | +0.0122 | -0.157 | 8.83e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1847 | +0.0040 | -0.139 | 8.96e-01 | 8.12e-01 |
| Accuracy  | 0.7561 | 0.7286 | -0.0275 | 0.443 | 6.80e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4030 | -0.0134 | 0.207 | 8.46e-01 | 6.25e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: len128  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8148 | 0.8182 | +0.0035 | -0.687 | 5.30e-01 | 8.12e-01 |
| AUPRC  | 0.4073 | 0.4205 | +0.0132 | -0.511 | 6.36e-01 | 6.25e-01 |
| Brier  | 0.1824 | 0.1696 | -0.0127 | 0.762 | 4.89e-01 | 1.00e+00 |
| Accuracy  | 0.7669 | 0.7998 | +0.0329 | -0.586 | 5.89e-01 | 6.25e-01 |
| F1  | 0.4315 | 0.4534 | +0.0220 | -0.406 | 7.05e-01 | 6.25e-01 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8124 | 0.8187 | +0.0063 | -0.698 | 5.24e-01 | 6.25e-01 |
| AUPRC  | 0.4131 | 0.4088 | -0.0043 | 0.195 | 8.55e-01 | 1.00e+00 |
| Brier  | 0.1528 | 0.1689 | +0.0161 | -1.120 | 3.25e-01 | 4.38e-01 |
| Accuracy  | 0.7757 | 0.7779 | +0.0022 | -0.055 | 9.59e-01 | 6.25e-01 |
| F1  | 0.4419 | 0.4357 | -0.0062 | 0.126 | 9.06e-01 | 6.25e-01 |

#### Case: crop80  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8209 | 0.8121 | -0.0088 | 1.195 | 2.98e-01 | 3.12e-01 |
| AUPRC ** | 0.3672 | 0.4150 | +0.0478 | -4.645 | 9.70e-03 | 6.25e-02 |
| Brier  | 0.1757 | 0.1822 | +0.0065 | -0.224 | 8.34e-01 | 6.25e-01 |
| Accuracy  | 0.7813 | 0.7287 | -0.0526 | 1.660 | 1.72e-01 | 3.12e-01 |
| F1  | 0.4406 | 0.3992 | -0.0414 | 1.346 | 2.50e-01 | 3.12e-01 |

#### Case: crop60  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8178 | 0.8147 | -0.0031 | 0.361 | 7.36e-01 | 8.12e-01 |
| AUPRC  | 0.4032 | 0.4208 | +0.0176 | -0.350 | 7.44e-01 | 6.25e-01 |
| Brier  | 0.1915 | 0.1798 | -0.0118 | 0.477 | 6.58e-01 | 8.12e-01 |
| Accuracy  | 0.7309 | 0.7680 | +0.0371 | -0.896 | 4.21e-01 | 6.25e-01 |
| F1  | 0.4282 | 0.4253 | -0.0029 | 0.071 | 9.47e-01 | 1.00e+00 |

#### Case: excl_extreme  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8237 | 0.8221 | -0.0016 | 0.119 | 9.11e-01 | 1.00e+00 |
| AUPRC  | 0.4215 | 0.4201 | -0.0014 | 0.046 | 9.66e-01 | 1.00e+00 |
| Brier  | 0.1847 | 0.1919 | +0.0071 | -0.228 | 8.31e-01 | 8.12e-01 |
| Accuracy  | 0.7286 | 0.7945 | +0.0659 | -1.026 | 3.63e-01 | 4.38e-01 |
| F1  | 0.4030 | 0.4372 | +0.0342 | -0.962 | 3.90e-01 | 6.25e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8182 | +0.0121 | -1.053 | 3.52e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4205 | +0.0113 | -0.305 | 7.75e-01 | 8.12e-01 |
| Brier  | 0.1808 | 0.1696 | -0.0111 | 1.085 | 3.39e-01 | 4.38e-01 |
| Accuracy  | 0.7561 | 0.7998 | +0.0437 | -0.788 | 4.75e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4534 | +0.0371 | -0.844 | 4.46e-01 | 8.12e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8187 | +0.0125 | -0.859 | 4.39e-01 | 3.12e-01 |
| AUPRC  | 0.4092 | 0.4088 | -0.0004 | 0.009 | 9.93e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1689 | -0.0119 | 1.284 | 2.68e-01 | 3.12e-01 |
| Accuracy  | 0.7561 | 0.7779 | +0.0218 | -0.514 | 6.34e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4357 | +0.0193 | -0.503 | 6.41e-01 | 8.12e-01 |

### crop80  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8121 | +0.0059 | -0.454 | 6.73e-01 | 1.00e+00 |
| AUPRC  | 0.4092 | 0.4150 | +0.0058 | -0.150 | 8.88e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1822 | +0.0015 | -0.084 | 9.37e-01 | 8.12e-01 |
| Accuracy  | 0.7561 | 0.7287 | -0.0274 | 1.031 | 3.61e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.3992 | -0.0172 | 0.650 | 5.51e-01 | 6.25e-01 |

### crop60  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8147 | +0.0085 | -0.618 | 5.70e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4208 | +0.0116 | -0.211 | 8.43e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1798 | -0.0010 | 0.045 | 9.67e-01 | 1.00e+00 |
| Accuracy  | 0.7561 | 0.7680 | +0.0120 | -0.507 | 6.39e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4253 | +0.0090 | -0.358 | 7.38e-01 | 6.25e-01 |

### excl_extreme  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8221 | +0.0160 | -0.417 | 6.98e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4201 | +0.0109 | -0.138 | 8.97e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1919 | +0.0111 | -0.336 | 7.54e-01 | 8.12e-01 |
| Accuracy  | 0.7561 | 0.7945 | +0.0384 | -0.573 | 5.98e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4372 | +0.0209 | -0.329 | 7.58e-01 | 8.12e-01 |

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
| M2 | CrossAttn | len128 | AUC-ROC | 0.7974 | 0.7017 | 0.8772 |
| M2 | CrossAttn | len128 | AUPRC | 0.3049 | 0.1823 | 0.4791 |
| M2 | CrossAttn | len128 | Brier | 0.1501 | 0.1274 | 0.1726 |
| M2 | CrossAttn | len128 | Accuracy | 0.7380 | 0.6812 | 0.7905 |
| M2 | CrossAttn | len128 | F1 | 0.3478 | 0.2222 | 0.4731 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8118 | 0.7353 | 0.8783 |
| M2 | CrossAttn | norm | AUPRC | 0.2994 | 0.1767 | 0.4761 |
| M2 | CrossAttn | norm | Brier | 0.1985 | 0.1678 | 0.2287 |
| M2 | CrossAttn | norm | Accuracy | 0.6681 | 0.6070 | 0.7294 |
| M2 | CrossAttn | norm | F1 | 0.3770 | 0.2615 | 0.4898 |
| M2 | CrossAttn | crop80 | AUC-ROC | 0.8041 | 0.7097 | 0.8909 |
| M2 | CrossAttn | crop80 | AUPRC | 0.3638 | 0.2067 | 0.5550 |
| M2 | CrossAttn | crop80 | Brier | 0.2134 | 0.1761 | 0.2497 |
| M2 | CrossAttn | crop80 | Accuracy | 0.7380 | 0.6856 | 0.7948 |
| M2 | CrossAttn | crop80 | F1 | 0.3750 | 0.2500 | 0.4950 |
| M2 | CrossAttn | crop60 | AUC-ROC | 0.8132 | 0.7331 | 0.8855 |
| M2 | CrossAttn | crop60 | AUPRC | 0.3230 | 0.1941 | 0.5090 |
| M2 | CrossAttn | crop60 | Brier | 0.2308 | 0.1930 | 0.2666 |
| M2 | CrossAttn | crop60 | Accuracy | 0.6419 | 0.5807 | 0.7074 |
| M2 | CrossAttn | crop60 | F1 | 0.3492 | 0.2342 | 0.4606 |
| M2 | CrossAttn | excl_extreme | AUC-ROC | 0.7923 | 0.7034 | 0.8697 |
| M2 | CrossAttn | excl_extreme | AUPRC | 0.2954 | 0.1708 | 0.4765 |
| M2 | CrossAttn | excl_extreme | Brier | 0.1924 | 0.1573 | 0.2287 |
| M2 | CrossAttn | excl_extreme | Accuracy | 0.7073 | 0.6390 | 0.7707 |
| M2 | CrossAttn | excl_extreme | F1 | 0.3478 | 0.2222 | 0.4706 |
| M2_2 | CrossAttn | len128 | AUC-ROC | 0.8118 | 0.7266 | 0.8837 |
| M2_2 | CrossAttn | len128 | AUPRC | 0.3505 | 0.2087 | 0.5470 |
| M2_2 | CrossAttn | len128 | Brier | 0.2188 | 0.1852 | 0.2526 |
| M2_2 | CrossAttn | len128 | Accuracy | 0.6812 | 0.6201 | 0.7380 |
| M2_2 | CrossAttn | len128 | F1 | 0.3423 | 0.2273 | 0.4531 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8268 | 0.7514 | 0.8920 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3300 | 0.1946 | 0.5104 |
| M2_2 | CrossAttn | norm | Brier | 0.2012 | 0.1705 | 0.2311 |
| M2_2 | CrossAttn | norm | Accuracy | 0.7555 | 0.6987 | 0.8122 |
| M2_2 | CrossAttn | norm | F1 | 0.3913 | 0.2558 | 0.5111 |
| M2_2 | CrossAttn | crop80 | AUC-ROC | 0.8087 | 0.7157 | 0.8886 |
| M2_2 | CrossAttn | crop80 | AUPRC | 0.3210 | 0.1916 | 0.5032 |
| M2_2 | CrossAttn | crop80 | Brier | 0.2298 | 0.1999 | 0.2602 |
| M2_2 | CrossAttn | crop80 | Accuracy | 0.6419 | 0.5808 | 0.7074 |
| M2_2 | CrossAttn | crop80 | F1 | 0.3387 | 0.2255 | 0.4444 |
| M2_2 | CrossAttn | crop60 | AUC-ROC | 0.7992 | 0.7063 | 0.8782 |
| M2_2 | CrossAttn | crop60 | AUPRC | 0.3391 | 0.2037 | 0.5305 |
| M2_2 | CrossAttn | crop60 | Brier | 0.2053 | 0.1732 | 0.2383 |
| M2_2 | CrossAttn | crop60 | Accuracy | 0.6681 | 0.6070 | 0.7293 |
| M2_2 | CrossAttn | crop60 | F1 | 0.3214 | 0.2105 | 0.4324 |
| M2_2 | CrossAttn | excl_extreme | AUC-ROC | 0.7229 | 0.6088 | 0.8257 |
| M2_2 | CrossAttn | excl_extreme | AUPRC | 0.3045 | 0.1602 | 0.4811 |
| M2_2 | CrossAttn | excl_extreme | Brier | 0.2256 | 0.1886 | 0.2648 |
| M2_2 | CrossAttn | excl_extreme | Accuracy | 0.6244 | 0.5561 | 0.6927 |
| M2_2 | CrossAttn | excl_extreme | F1 | 0.2936 | 0.1800 | 0.4035 |
| M3 | CrossAttn3 | len128 | AUC-ROC | 0.8004 | 0.7130 | 0.8810 |
| M3 | CrossAttn3 | len128 | AUPRC | 0.2862 | 0.1846 | 0.4746 |
| M3 | CrossAttn3 | len128 | Brier | 0.1776 | 0.1502 | 0.2040 |
| M3 | CrossAttn3 | len128 | Accuracy | 0.7074 | 0.6507 | 0.7686 |
| M3 | CrossAttn3 | len128 | F1 | 0.3619 | 0.2400 | 0.4783 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8492 | 0.7667 | 0.9140 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3519 | 0.2244 | 0.5673 |
| M3 | CrossAttn3 | norm | Brier | 0.1796 | 0.1483 | 0.2097 |
| M3 | CrossAttn3 | norm | Accuracy | 0.7380 | 0.6812 | 0.7948 |
| M3 | CrossAttn3 | norm | F1 | 0.4118 | 0.2857 | 0.5345 |
| M3 | CrossAttn3 | crop80 | AUC-ROC | 0.8024 | 0.7089 | 0.8795 |
| M3 | CrossAttn3 | crop80 | AUPRC | 0.2900 | 0.1876 | 0.4867 |
| M3 | CrossAttn3 | crop80 | Brier | 0.2250 | 0.1957 | 0.2528 |
| M3 | CrossAttn3 | crop80 | Accuracy | 0.6070 | 0.5459 | 0.6725 |
| M3 | CrossAttn3 | crop80 | F1 | 0.3284 | 0.2237 | 0.4317 |
| M3 | CrossAttn3 | crop60 | AUC-ROC | 0.7669 | 0.6771 | 0.8469 |
| M3 | CrossAttn3 | crop60 | AUPRC | 0.2538 | 0.1564 | 0.4348 |
| M3 | CrossAttn3 | crop60 | Brier | 0.1884 | 0.1639 | 0.2122 |
| M3 | CrossAttn3 | crop60 | Accuracy | 0.7118 | 0.6550 | 0.7686 |
| M3 | CrossAttn3 | crop60 | F1 | 0.3529 | 0.2321 | 0.4762 |
| M3 | CrossAttn3 | excl_extreme | AUC-ROC | 0.7668 | 0.6746 | 0.8506 |
| M3 | CrossAttn3 | excl_extreme | AUPRC | 0.2446 | 0.1454 | 0.4235 |
| M3 | CrossAttn3 | excl_extreme | Brier | 0.2781 | 0.2367 | 0.3224 |
| M3 | CrossAttn3 | excl_extreme | Accuracy | 0.6390 | 0.5707 | 0.7024 |
| M3 | CrossAttn3 | excl_extreme | F1 | 0.3509 | 0.2299 | 0.4638 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-len128 | 0.8030 | 0.7974 | -0.0057 | 0.231 | 8.174e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8118 | +0.0087 | -0.469 | 6.394e-01 | ns |
| M1-LR vs M2-crop80 | 0.8030 | 0.8041 | +0.0010 | -0.032 | 9.747e-01 | ns |
| M1-LR vs M2-crop60 | 0.8030 | 0.8132 | +0.0102 | -0.423 | 6.726e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-len128 | 0.8030 | 0.8004 | -0.0026 | 0.078 | 9.376e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8492 | +0.0461 | -2.057 | 3.972e-02 | * |
| M1-LR vs M3-crop80 | 0.8030 | 0.8024 | -0.0006 | 0.021 | 9.833e-01 | ns |
| M1-LR vs M3-crop60 | 0.8030 | 0.7669 | -0.0362 | 1.092 | 2.747e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len128 vs M2_2-len128 | 0.7974 | 0.8118 | +0.0144 | -0.738 | 4.605e-01 | ns |
| M2-norm vs M2_2-norm | 0.8118 | 0.8268 | +0.0150 | -0.830 | 4.068e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.8041 | 0.8087 | +0.0047 | -0.155 | 8.765e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.8132 | 0.7992 | -0.0140 | 0.533 | 5.939e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.7923 | 0.4692 | -0.3231 | 4.030 | 5.571e-05 | *** |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len128 vs M3-len128 | 0.7974 | 0.8004 | +0.0030 | -0.123 | 9.022e-01 | ns |
| M2-norm vs M3-norm | 0.8118 | 0.8492 | +0.0374 | -1.876 | 6.062e-02 | † |
| M2-crop80 vs M3-crop80 | 0.8041 | 0.8024 | -0.0016 | 0.063 | 9.497e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.8132 | 0.7669 | -0.0463 | 2.292 | 2.188e-02 | * |
| M2-excl_extreme vs M3-excl_extreme | 0.7923 | 0.7668 | -0.0256 | 1.195 | 2.322e-01 | ns |

