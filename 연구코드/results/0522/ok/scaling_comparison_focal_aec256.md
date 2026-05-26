# Scaling Comparison — Test Set Performance (AEC 256pt, FocalLoss)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_clinic | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |
| M2 | CrossAttn | crop60/scale_clinic | 0.8874 | 0.5322 | 0.2339 | 0.5116 | 0.3333 |
| M2_2 | CrossAttn | len256/scale_clinic | 0.8464 | 0.4511 | 0.1785 | 0.6977 | 0.4222 |
| M3 | CrossAttn3 | norm/scale_clinic | 0.8641 | 0.4841 | 0.1805 | 0.7616 | 0.4384 |

---

## Model 1 — Clinic Only  (1 scaling case)

### Logistic Regression

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **scale_clinic** | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |

---

## Model 2 — Clinic + AEC (Matched)  (5 AEC variants)

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_clinic | 0.8710 | 0.5149 | 0.1953 | 0.6570 | 0.4040 |
| crop80/scale_clinic | 0.8616 | 0.5219 | 0.2090 | 0.6512 | 0.4000 |
| **crop60/scale_clinic** | 0.8874 | 0.5322 | 0.2339 | 0.5116 | 0.3333 |
| norm/scale_clinic | 0.8600 | 0.4793 | 0.1904 | 0.7151 | 0.4096 |
| excl_extreme/scale_clinic | 0.8854 | 0.6067 | 0.2196 | 0.6688 | 0.4270 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| **len256/scale_clinic** | 0.8464 | 0.4511 | 0.1785 | 0.6977 | 0.4222 |
| crop80/scale_clinic | 0.8448 | 0.4750 | 0.1619 | 0.7791 | 0.4865 |
| crop60/scale_clinic | 0.8455 | 0.4340 | 0.1774 | 0.7500 | 0.4557 |
| norm/scale_clinic | 0.8382 | 0.5059 | 0.1564 | 0.7267 | 0.4051 |
| excl_extreme/scale_clinic | 0.8329 | 0.4003 | 0.2165 | 0.6234 | 0.3409 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len256/scale_clinic | 0.8329 | 0.4106 | 0.1981 | 0.6163 | 0.3774 |
| crop80/scale_clinic | 0.8133 | 0.4270 | 0.2123 | 0.6105 | 0.3495 |
| crop60/scale_clinic | 0.8638 | 0.5098 | 0.2084 | 0.6744 | 0.4043 |
| **norm/scale_clinic** | 0.8641 | 0.4841 | 0.1805 | 0.7616 | 0.4384 |
| excl_extreme/scale_clinic | 0.8174 | 0.4456 | 0.1724 | 0.7013 | 0.4103 |

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

### len256/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8233 | +0.0120 | -0.933 | 4.03e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3935 | +0.0088 | -0.619 | 5.70e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1795 | -0.0024 | 0.156 | 8.84e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7249 | -0.0306 | 0.692 | 5.27e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4387 | -0.0127 | 0.429 | 6.90e-01 | 6.25e-01 |

### crop80/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8096 | -0.0018 | 0.178 | 8.67e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3765 | -0.0082 | 0.483 | 6.54e-01 | 6.25e-01 |
| Brier † | 0.1819 | 0.1939 | +0.0119 | -2.417 | 7.30e-02 | 1.25e-01 |
| Accuracy  | 0.7555 | 0.7393 | -0.0161 | 0.342 | 7.50e-01 | 6.25e-01 |
| F1  | 0.4514 | 0.4387 | -0.0127 | 0.348 | 7.45e-01 | 6.25e-01 |

### crop60/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8068 | -0.0045 | 0.420 | 6.96e-01 | 8.75e-01 |
| AUPRC  | 0.3847 | 0.3800 | -0.0046 | 0.176 | 8.69e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.1807 | -0.0012 | 0.153 | 8.86e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7306 | -0.0249 | 0.854 | 4.41e-01 | 4.38e-01 |
| F1  | 0.4514 | 0.4307 | -0.0207 | 0.963 | 3.90e-01 | 6.25e-01 |

### norm/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC ** | 0.8113 | 0.8183 | +0.0070 | -5.207 | 6.48e-03 | 6.25e-02 |
| AUPRC † | 0.3847 | 0.4151 | +0.0304 | -2.296 | 8.33e-02 | 6.25e-02 |
| Brier  | 0.1819 | 0.1869 | +0.0050 | -0.504 | 6.41e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7670 | +0.0115 | -0.310 | 7.72e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4568 | +0.0054 | -0.240 | 8.22e-01 | 1.00e+00 |

### excl_extreme/scale_clinic  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8065 | -0.0049 | 0.315 | 7.69e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3592 | -0.0255 | 0.704 | 5.20e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1863 | +0.0044 | -0.343 | 7.49e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7777 | +0.0222 | -0.422 | 6.95e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4655 | +0.0141 | -0.295 | 7.83e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. (aec_var, case) 키로 매칭.

#### Case: len256/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8233 | 0.8163 | -0.0071 | 1.096 | 3.34e-01 | 4.38e-01 |
| AUPRC  | 0.3935 | 0.4204 | +0.0269 | -0.855 | 4.41e-01 | 4.38e-01 |
| Brier  | 0.1795 | 0.1665 | -0.0130 | 0.661 | 5.45e-01 | 8.12e-01 |
| Accuracy  | 0.7249 | 0.7030 | -0.0219 | 0.878 | 4.30e-01 | 6.25e-01 |
| F1  | 0.4387 | 0.4211 | -0.0176 | 1.791 | 1.48e-01 | 6.25e-02 |

#### Case: crop80/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8096 | 0.8067 | -0.0028 | 0.476 | 6.59e-01 | 1.00e+00 |
| AUPRC  | 0.3765 | 0.3802 | +0.0037 | -0.233 | 8.27e-01 | 1.00e+00 |
| Brier  | 0.1939 | 0.1979 | +0.0041 | -0.355 | 7.40e-01 | 1.00e+00 |
| Accuracy  | 0.7393 | 0.7393 | -0.0000 | 0.000 | 1.00e+00 | 9.38e-01 |
| F1  | 0.4387 | 0.4426 | +0.0039 | -0.270 | 8.01e-01 | 1.00e+00 |

#### Case: crop60/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8068 | 0.8029 | -0.0039 | 0.754 | 4.93e-01 | 4.38e-01 |
| AUPRC  | 0.3800 | 0.3884 | +0.0083 | -0.346 | 7.47e-01 | 1.00e+00 |
| Brier  | 0.1807 | 0.2039 | +0.0232 | -1.633 | 1.78e-01 | 3.12e-01 |
| Accuracy  | 0.7306 | 0.7570 | +0.0264 | -0.928 | 4.06e-01 | 3.12e-01 |
| F1  | 0.4307 | 0.4558 | +0.0251 | -1.088 | 3.38e-01 | 4.38e-01 |

#### Case: norm/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8183 | 0.8130 | -0.0053 | 0.835 | 4.51e-01 | 6.25e-01 |
| AUPRC  | 0.4151 | 0.4233 | +0.0082 | -0.328 | 7.59e-01 | 8.12e-01 |
| Brier  | 0.1869 | 0.1840 | -0.0028 | 0.144 | 8.92e-01 | 6.25e-01 |
| Accuracy  | 0.7670 | 0.7699 | +0.0030 | -0.080 | 9.40e-01 | 1.00e+00 |
| F1  | 0.4568 | 0.4655 | +0.0087 | -0.248 | 8.16e-01 | 1.00e+00 |

#### Case: excl_extreme/scale_clinic  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8065 | 0.7939 | -0.0125 | 0.725 | 5.08e-01 | 4.38e-01 |
| AUPRC  | 0.3592 | 0.3706 | +0.0114 | -0.333 | 7.56e-01 | 6.25e-01 |
| Brier  | 0.1863 | 0.1733 | -0.0131 | 0.504 | 6.40e-01 | 6.25e-01 |
| Accuracy  | 0.7777 | 0.7391 | -0.0385 | 0.616 | 5.71e-01 | 8.12e-01 |
| F1  | 0.4655 | 0.4229 | -0.0426 | 0.652 | 5.50e-01 | 6.25e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len256/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8163 | +0.0049 | -0.331 | 7.57e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.4204 | +0.0357 | -1.325 | 2.56e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1665 | -0.0154 | 2.068 | 1.07e-01 | 1.25e-01 |
| Accuracy  | 0.7555 | 0.7030 | -0.0525 | 1.352 | 2.48e-01 | 3.12e-01 |
| F1  | 0.4514 | 0.4211 | -0.0303 | 1.012 | 3.69e-01 | 3.12e-01 |

### crop80/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8067 | -0.0046 | 0.320 | 7.65e-01 | 1.00e+00 |
| AUPRC  | 0.3847 | 0.3802 | -0.0045 | 0.254 | 8.12e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1979 | +0.0160 | -1.651 | 1.74e-01 | 1.88e-01 |
| Accuracy  | 0.7555 | 0.7393 | -0.0161 | 0.333 | 7.56e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4426 | -0.0088 | 0.273 | 7.98e-01 | 1.00e+00 |

### crop60/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8029 | -0.0084 | 0.652 | 5.50e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3884 | +0.0037 | -0.293 | 7.84e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.2039 | +0.0219 | -1.922 | 1.27e-01 | 1.25e-01 |
| Accuracy  | 0.7555 | 0.7570 | +0.0015 | -0.060 | 9.55e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4558 | +0.0044 | -0.181 | 8.65e-01 | 1.00e+00 |

### norm/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8130 | +0.0016 | -0.218 | 8.38e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.4233 | +0.0386 | -1.749 | 1.55e-01 | 6.25e-02 |
| Brier  | 0.1819 | 0.1840 | +0.0021 | -0.140 | 8.96e-01 | 8.12e-01 |
| Accuracy  | 0.7555 | 0.7699 | +0.0145 | -0.380 | 7.23e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4655 | +0.0141 | -0.446 | 6.78e-01 | 6.25e-01 |

### excl_extreme/scale_clinic  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.7939 | -0.0174 | 0.725 | 5.09e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.3706 | -0.0141 | 0.282 | 7.92e-01 | 8.12e-01 |
| Brier  | 0.1819 | 0.1733 | -0.0087 | 0.517 | 6.33e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7391 | -0.0163 | 0.307 | 7.74e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4229 | -0.0285 | 0.572 | 5.98e-01 | 6.25e-01 |

---

# Test Set — Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
|-------|-----|------|--------|-------:|---------:|---------:|
| M1 | LR | scale_clinic | AUC-ROC | 0.8325 | 0.7386 | 0.9091 |
| M1 | LR | scale_clinic | AUPRC | 0.5008 | 0.2965 | 0.6937 |
| M1 | LR | scale_clinic | Brier | 0.1804 | 0.1527 | 0.2118 |
| M1 | LR | scale_clinic | Accuracy | 0.6977 | 0.6279 | 0.7674 |
| M1 | LR | scale_clinic | F1 | 0.3953 | 0.2535 | 0.5228 |
| M2 | CrossAttn | len256/scale_clinic | AUC-ROC | 0.8710 | 0.8007 | 0.9319 |
| M2 | CrossAttn | len256/scale_clinic | AUPRC | 0.5149 | 0.3099 | 0.7214 |
| M2 | CrossAttn | len256/scale_clinic | Brier | 0.1953 | 0.1791 | 0.2114 |
| M2 | CrossAttn | len256/scale_clinic | Accuracy | 0.6570 | 0.5814 | 0.7267 |
| M2 | CrossAttn | len256/scale_clinic | F1 | 0.4040 | 0.2750 | 0.5234 |
| M2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8616 | 0.7775 | 0.9295 |
| M2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.5219 | 0.3114 | 0.7141 |
| M2 | CrossAttn | crop80/scale_clinic | Brier | 0.2090 | 0.1924 | 0.2253 |
| M2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.6512 | 0.5814 | 0.7209 |
| M2 | CrossAttn | crop80/scale_clinic | F1 | 0.4000 | 0.2727 | 0.5179 |
| M2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8874 | 0.8240 | 0.9400 |
| M2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.5322 | 0.3301 | 0.7254 |
| M2 | CrossAttn | crop60/scale_clinic | Brier | 0.2339 | 0.2149 | 0.2538 |
| M2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.5116 | 0.4360 | 0.5872 |
| M2 | CrossAttn | crop60/scale_clinic | F1 | 0.3333 | 0.2205 | 0.4394 |
| M2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8600 | 0.7717 | 0.9287 |
| M2 | CrossAttn | norm/scale_clinic | AUPRC | 0.4793 | 0.2948 | 0.6984 |
| M2 | CrossAttn | norm/scale_clinic | Brier | 0.1904 | 0.1747 | 0.2059 |
| M2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7151 | 0.6453 | 0.7849 |
| M2 | CrossAttn | norm/scale_clinic | F1 | 0.4096 | 0.2667 | 0.5361 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8854 | 0.8098 | 0.9472 |
| M2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.6067 | 0.3986 | 0.8051 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.2196 | 0.2017 | 0.2380 |
| M2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.6688 | 0.5909 | 0.7403 |
| M2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.4270 | 0.2921 | 0.5477 |
| M2_2 | CrossAttn | len256/scale_clinic | AUC-ROC | 0.8464 | 0.7469 | 0.9271 |
| M2_2 | CrossAttn | len256/scale_clinic | AUPRC | 0.4511 | 0.2644 | 0.6594 |
| M2_2 | CrossAttn | len256/scale_clinic | Brier | 0.1785 | 0.1607 | 0.1969 |
| M2_2 | CrossAttn | len256/scale_clinic | Accuracy | 0.6977 | 0.6279 | 0.7674 |
| M2_2 | CrossAttn | len256/scale_clinic | F1 | 0.4222 | 0.2857 | 0.5472 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUC-ROC | 0.8448 | 0.7331 | 0.9338 |
| M2_2 | CrossAttn | crop80/scale_clinic | AUPRC | 0.4750 | 0.2876 | 0.6903 |
| M2_2 | CrossAttn | crop80/scale_clinic | Brier | 0.1619 | 0.1397 | 0.1868 |
| M2_2 | CrossAttn | crop80/scale_clinic | Accuracy | 0.7791 | 0.7093 | 0.8372 |
| M2_2 | CrossAttn | crop80/scale_clinic | F1 | 0.4865 | 0.3332 | 0.6197 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUC-ROC | 0.8455 | 0.7556 | 0.9183 |
| M2_2 | CrossAttn | crop60/scale_clinic | AUPRC | 0.4340 | 0.2554 | 0.6474 |
| M2_2 | CrossAttn | crop60/scale_clinic | Brier | 0.1774 | 0.1607 | 0.1942 |
| M2_2 | CrossAttn | crop60/scale_clinic | Accuracy | 0.7500 | 0.6802 | 0.8140 |
| M2_2 | CrossAttn | crop60/scale_clinic | F1 | 0.4557 | 0.3077 | 0.5909 |
| M2_2 | CrossAttn | norm/scale_clinic | AUC-ROC | 0.8382 | 0.7461 | 0.9161 |
| M2_2 | CrossAttn | norm/scale_clinic | AUPRC | 0.5059 | 0.3032 | 0.6991 |
| M2_2 | CrossAttn | norm/scale_clinic | Brier | 0.1564 | 0.1381 | 0.1754 |
| M2_2 | CrossAttn | norm/scale_clinic | Accuracy | 0.7267 | 0.6570 | 0.7907 |
| M2_2 | CrossAttn | norm/scale_clinic | F1 | 0.4051 | 0.2609 | 0.5412 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUC-ROC | 0.8329 | 0.7132 | 0.9158 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | AUPRC | 0.4003 | 0.1951 | 0.6111 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Brier | 0.2165 | 0.1945 | 0.2403 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | Accuracy | 0.6234 | 0.5390 | 0.7013 |
| M2_2 | CrossAttn | excl_extreme/scale_clinic | F1 | 0.3409 | 0.2093 | 0.4598 |
| M3 | CrossAttn3 | len256/scale_clinic | AUC-ROC | 0.8329 | 0.7352 | 0.9135 |
| M3 | CrossAttn3 | len256/scale_clinic | AUPRC | 0.4106 | 0.2446 | 0.6380 |
| M3 | CrossAttn3 | len256/scale_clinic | Brier | 0.1981 | 0.1822 | 0.2130 |
| M3 | CrossAttn3 | len256/scale_clinic | Accuracy | 0.6163 | 0.5407 | 0.6919 |
| M3 | CrossAttn3 | len256/scale_clinic | F1 | 0.3774 | 0.2529 | 0.4906 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUC-ROC | 0.8133 | 0.7221 | 0.8960 |
| M3 | CrossAttn3 | crop80/scale_clinic | AUPRC | 0.4270 | 0.2426 | 0.6489 |
| M3 | CrossAttn3 | crop80/scale_clinic | Brier | 0.2123 | 0.1961 | 0.2290 |
| M3 | CrossAttn3 | crop80/scale_clinic | Accuracy | 0.6105 | 0.5347 | 0.6802 |
| M3 | CrossAttn3 | crop80/scale_clinic | F1 | 0.3495 | 0.2268 | 0.4655 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUC-ROC | 0.8638 | 0.7836 | 0.9334 |
| M3 | CrossAttn3 | crop60/scale_clinic | AUPRC | 0.5098 | 0.3164 | 0.7373 |
| M3 | CrossAttn3 | crop60/scale_clinic | Brier | 0.2084 | 0.1838 | 0.2317 |
| M3 | CrossAttn3 | crop60/scale_clinic | Accuracy | 0.6744 | 0.6047 | 0.7442 |
| M3 | CrossAttn3 | crop60/scale_clinic | F1 | 0.4043 | 0.2716 | 0.5283 |
| M3 | CrossAttn3 | norm/scale_clinic | AUC-ROC | 0.8641 | 0.7802 | 0.9299 |
| M3 | CrossAttn3 | norm/scale_clinic | AUPRC | 0.4841 | 0.3021 | 0.6923 |
| M3 | CrossAttn3 | norm/scale_clinic | Brier | 0.1805 | 0.1637 | 0.1970 |
| M3 | CrossAttn3 | norm/scale_clinic | Accuracy | 0.7616 | 0.6977 | 0.8256 |
| M3 | CrossAttn3 | norm/scale_clinic | F1 | 0.4384 | 0.2857 | 0.5750 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUC-ROC | 0.8174 | 0.7193 | 0.9012 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | AUPRC | 0.4456 | 0.2548 | 0.6518 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Brier | 0.1724 | 0.1452 | 0.2012 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | Accuracy | 0.7013 | 0.6234 | 0.7727 |
| M3 | CrossAttn3 | excl_extreme/scale_clinic | F1 | 0.4103 | 0.2580 | 0.5412 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-len256 | 0.8325 | 0.8710 | +0.0385 | -1.162 | 2.452e-01 | ns |
| M1-LR vs M2-crop80 | 0.8325 | 0.8616 | +0.0290 | -1.179 | 2.382e-01 | ns |
| M1-LR vs M2-crop60 | 0.8325 | 0.8874 | +0.0549 | -1.769 | 7.691e-02 | † |
| M1-LR vs M2-norm | 0.8325 | 0.8600 | +0.0274 | -0.973 | 3.305e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-len256 | 0.8325 | 0.8329 | +0.0003 | -0.010 | 9.922e-01 | ns |
| M1-LR vs M3-crop80 | 0.8325 | 0.8133 | -0.0192 | 0.413 | 6.797e-01 | ns |
| M1-LR vs M3-crop60 | 0.8325 | 0.8638 | +0.0312 | -0.951 | 3.417e-01 | ns |
| M1-LR vs M3-norm | 0.8325 | 0.8641 | +0.0315 | -1.142 | 2.534e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len256 vs M2_2-len256 | 0.8710 | 0.8464 | -0.0246 | 0.790 | 4.294e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.8616 | 0.8448 | -0.0167 | 0.535 | 5.927e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.8874 | 0.8455 | -0.0419 | 1.501 | 1.334e-01 | ns |
| M2-norm vs M2_2-norm | 0.8600 | 0.8382 | -0.0218 | 0.692 | 4.887e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.8854 | 0.5353 | -0.3502 | 4.215 | 2.498e-05 | *** |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len256 vs M3-len256 | 0.8710 | 0.8329 | -0.0382 | 1.512 | 1.306e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8616 | 0.8133 | -0.0482 | 1.358 | 1.744e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.8874 | 0.8638 | -0.0237 | 0.929 | 3.528e-01 | ns |
| M2-norm vs M3-norm | 0.8600 | 0.8641 | +0.0041 | -0.245 | 8.068e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.8854 | 0.8174 | -0.0680 | 2.491 | 1.274e-02 | * |

