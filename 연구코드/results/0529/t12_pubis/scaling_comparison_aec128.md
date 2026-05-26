# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8325 | 0.5008 | 0.1804 | 0.6977 | 0.3953 |
| M2 | CrossAttn | crop80 | 0.8761 | 0.5606 | 0.2396 | 0.6221 | 0.3810 |
| M2_2 | CrossAttn | crop60 | 0.8316 | 0.3781 | 0.1645 | 0.7674 | 0.4872 |
| M3 | CrossAttn3 | crop60 | 0.8780 | 0.5335 | 0.2005 | 0.6802 | 0.4086 |

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
| norm | 0.7777 | 0.2721 | 0.2370 | 0.6221 | 0.3689 |
| excl_extreme | 0.7848 | 0.3228 | 0.2244 | 0.7013 | 0.4250 |
| len128 | 0.8452 | 0.4449 | 0.2547 | 0.6105 | 0.3738 |
| **crop80** | 0.8761 | 0.5606 | 0.2396 | 0.6221 | 0.3810 |
| crop60 | 0.8565 | 0.4906 | 0.2009 | 0.6395 | 0.3922 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm | 0.8291 | 0.3908 | 0.1894 | 0.6337 | 0.3762 |
| excl_extreme | 0.7978 | 0.3298 | 0.1763 | 0.7208 | 0.3768 |
| len128 | 0.7988 | 0.3619 | 0.2100 | 0.5756 | 0.3303 |
| crop80 | 0.8272 | 0.3970 | 0.1321 | 0.7849 | 0.4478 |
| **crop60** | 0.8316 | 0.3781 | 0.1645 | 0.7674 | 0.4872 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| norm | 0.8691 | 0.4607 | 0.1487 | 0.7442 | 0.4634 |
| excl_extreme | 0.8729 | 0.6161 | 0.1852 | 0.7468 | 0.4658 |
| len128 | 0.8521 | 0.5305 | 0.1520 | 0.7558 | 0.4474 |
| crop80 | 0.8496 | 0.4948 | 0.1961 | 0.7384 | 0.4000 |
| **crop60** | 0.8780 | 0.5335 | 0.2005 | 0.6802 | 0.4086 |

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
| AUC-ROC † | 0.8113 | 0.8333 | +0.0220 | -2.306 | 8.24e-02 | 6.25e-02 |
| AUPRC  | 0.3847 | 0.3925 | +0.0078 | -0.274 | 7.98e-01 | 1.00e+00 |
| Brier † | 0.1819 | 0.1469 | -0.0350 | 2.251 | 8.75e-02 | 1.25e-01 |
| Accuracy  | 0.7555 | 0.7511 | -0.0044 | 0.201 | 8.50e-01 | 8.75e-01 |
| F1  | 0.4514 | 0.4510 | -0.0005 | 0.023 | 9.83e-01 | 1.00e+00 |

### excl_extreme  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8138 | +0.0025 | -0.224 | 8.34e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.3521 | -0.0325 | 1.619 | 1.81e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.2083 | +0.0264 | -0.850 | 4.43e-01 | 6.25e-01 |
| Accuracy  | 0.7555 | 0.7407 | -0.0148 | 0.598 | 5.82e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4392 | -0.0122 | 0.842 | 4.47e-01 | 4.38e-01 |

### len128  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8053 | -0.0060 | 1.387 | 2.38e-01 | 3.12e-01 |
| AUPRC * | 0.3847 | 0.3603 | -0.0244 | 2.857 | 4.61e-02 | 6.25e-02 |
| Brier  | 0.1819 | 0.1891 | +0.0072 | -0.673 | 5.38e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7146 | -0.0409 | 1.032 | 3.60e-01 | 5.00e-01 |
| F1  | 0.4514 | 0.4260 | -0.0254 | 1.081 | 3.40e-01 | 3.12e-01 |

### crop80  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8054 | -0.0059 | 1.089 | 3.37e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.4217 | +0.0370 | -1.624 | 1.80e-01 | 1.88e-01 |
| Brier  | 0.1819 | 0.1943 | +0.0124 | -0.790 | 4.74e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7336 | -0.0219 | 0.430 | 6.89e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4443 | -0.0071 | 0.264 | 8.05e-01 | 1.00e+00 |

### crop60  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8156 | +0.0043 | -0.890 | 4.24e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3677 | -0.0169 | 0.467 | 6.65e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1887 | +0.0068 | -0.413 | 7.00e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7277 | -0.0277 | 1.154 | 3.13e-01 | 3.12e-01 |
| F1  | 0.4514 | 0.4292 | -0.0223 | 1.337 | 2.52e-01 | 3.12e-01 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8333 | 0.8161 | -0.0172 | 1.281 | 2.69e-01 | 4.38e-01 |
| AUPRC  | 0.3925 | 0.3995 | +0.0070 | -0.453 | 6.74e-01 | 6.25e-01 |
| Brier † | 0.1469 | 0.1631 | +0.0161 | -2.265 | 8.62e-02 | 1.25e-01 |
| Accuracy  | 0.7511 | 0.7511 | +0.0000 | -0.001 | 1.00e+00 | 1.00e+00 |
| F1  | 0.4510 | 0.4489 | -0.0021 | 0.088 | 9.34e-01 | 1.00e+00 |

#### Case: excl_extreme  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8138 | 0.8050 | -0.0089 | 1.038 | 3.58e-01 | 4.38e-01 |
| AUPRC  | 0.3521 | 0.3486 | -0.0035 | 0.207 | 8.46e-01 | 6.25e-01 |
| Brier  | 0.2083 | 0.1846 | -0.0237 | 0.740 | 5.00e-01 | 8.12e-01 |
| Accuracy  | 0.7407 | 0.7811 | +0.0404 | -0.899 | 4.20e-01 | 6.25e-01 |
| F1  | 0.4392 | 0.4548 | +0.0156 | -0.558 | 6.06e-01 | 6.25e-01 |

#### Case: len128  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8053 | 0.8021 | -0.0032 | 0.327 | 7.60e-01 | 6.25e-01 |
| AUPRC  | 0.3603 | 0.3623 | +0.0020 | -0.102 | 9.24e-01 | 8.12e-01 |
| Brier  | 0.1891 | 0.1924 | +0.0032 | -0.207 | 8.46e-01 | 1.00e+00 |
| Accuracy  | 0.7146 | 0.6956 | -0.0190 | 0.537 | 6.20e-01 | 6.25e-01 |
| F1  | 0.4260 | 0.4246 | -0.0015 | 0.069 | 9.48e-01 | 1.00e+00 |

#### Case: crop80  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8054 | 0.7994 | -0.0060 | 1.023 | 3.64e-01 | 4.38e-01 |
| AUPRC  | 0.4217 | 0.3565 | -0.0652 | 1.968 | 1.20e-01 | 3.12e-01 |
| Brier  | 0.1943 | 0.1988 | +0.0046 | -0.294 | 7.83e-01 | 6.25e-01 |
| Accuracy  | 0.7336 | 0.7526 | +0.0190 | -0.558 | 6.06e-01 | 1.00e+00 |
| F1  | 0.4443 | 0.4491 | +0.0048 | -0.208 | 8.45e-01 | 1.00e+00 |

#### Case: crop60  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8156 | 0.8117 | -0.0039 | 0.609 | 5.75e-01 | 1.00e+00 |
| AUPRC  | 0.3677 | 0.3809 | +0.0132 | -0.276 | 7.96e-01 | 1.00e+00 |
| Brier  | 0.1887 | 0.2094 | +0.0207 | -1.920 | 1.27e-01 | 1.88e-01 |
| Accuracy  | 0.7277 | 0.7321 | +0.0044 | -0.243 | 8.20e-01 | 8.75e-01 |
| F1  | 0.4292 | 0.4426 | +0.0134 | -1.026 | 3.63e-01 | 4.38e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8161 | +0.0048 | -0.753 | 4.93e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3995 | +0.0148 | -0.516 | 6.33e-01 | 6.25e-01 |
| Brier  | 0.1819 | 0.1631 | -0.0188 | 1.550 | 1.96e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7511 | -0.0043 | 0.080 | 9.40e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4489 | -0.0025 | 0.073 | 9.45e-01 | 1.00e+00 |

### excl_extreme  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8050 | -0.0064 | 0.686 | 5.31e-01 | 4.38e-01 |
| AUPRC  | 0.3847 | 0.3486 | -0.0361 | 1.026 | 3.63e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1846 | +0.0026 | -0.338 | 7.52e-01 | 1.00e+00 |
| Accuracy  | 0.7555 | 0.7811 | +0.0257 | -0.641 | 5.57e-01 | 8.12e-01 |
| F1  | 0.4514 | 0.4548 | +0.0034 | -0.186 | 8.62e-01 | 1.00e+00 |

### len128  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8021 | -0.0092 | 0.764 | 4.87e-01 | 6.25e-01 |
| AUPRC  | 0.3847 | 0.3623 | -0.0223 | 1.072 | 3.44e-01 | 4.38e-01 |
| Brier  | 0.1819 | 0.1924 | +0.0104 | -1.287 | 2.67e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.6956 | -0.0598 | 1.275 | 2.71e-01 | 2.50e-01 |
| F1  | 0.4514 | 0.4246 | -0.0269 | 1.096 | 3.35e-01 | 6.25e-01 |

### crop80  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.7994 | -0.0120 | 1.690 | 1.66e-01 | 3.12e-01 |
| AUPRC  | 0.3847 | 0.3565 | -0.0282 | 1.336 | 2.52e-01 | 3.12e-01 |
| Brier  | 0.1819 | 0.1988 | +0.0169 | -1.368 | 2.43e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7526 | -0.0029 | 0.066 | 9.50e-01 | 1.00e+00 |
| F1  | 0.4514 | 0.4491 | -0.0023 | 0.084 | 9.37e-01 | 8.12e-01 |

### crop60  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8113 | 0.8117 | +0.0004 | -0.055 | 9.59e-01 | 8.12e-01 |
| AUPRC  | 0.3847 | 0.3809 | -0.0037 | 0.250 | 8.15e-01 | 1.00e+00 |
| Brier  | 0.1819 | 0.2094 | +0.0275 | -1.186 | 3.01e-01 | 3.12e-01 |
| Accuracy  | 0.7555 | 0.7321 | -0.0233 | 0.852 | 4.42e-01 | 3.12e-01 |
| F1  | 0.4514 | 0.4426 | -0.0089 | 0.381 | 7.23e-01 | 6.25e-01 |

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
| M2 | CrossAttn | norm | AUC-ROC | 0.7777 | 0.6998 | 0.8494 |
| M2 | CrossAttn | norm | AUPRC | 0.2721 | 0.1612 | 0.4590 |
| M2 | CrossAttn | norm | Brier | 0.2370 | 0.1884 | 0.2849 |
| M2 | CrossAttn | norm | Accuracy | 0.6221 | 0.5465 | 0.6919 |
| M2 | CrossAttn | norm | F1 | 0.3689 | 0.2418 | 0.4865 |
| M2 | CrossAttn | excl_extreme | AUC-ROC | 0.7848 | 0.6887 | 0.8682 |
| M2 | CrossAttn | excl_extreme | AUPRC | 0.3228 | 0.1897 | 0.5001 |
| M2 | CrossAttn | excl_extreme | Brier | 0.2244 | 0.1837 | 0.2670 |
| M2 | CrossAttn | excl_extreme | Accuracy | 0.7013 | 0.6234 | 0.7727 |
| M2 | CrossAttn | excl_extreme | F1 | 0.4250 | 0.2894 | 0.5556 |
| M2 | CrossAttn | len128 | AUC-ROC | 0.8452 | 0.7592 | 0.9155 |
| M2 | CrossAttn | len128 | AUPRC | 0.4449 | 0.2570 | 0.6511 |
| M2 | CrossAttn | len128 | Brier | 0.2547 | 0.2136 | 0.2964 |
| M2 | CrossAttn | len128 | Accuracy | 0.6105 | 0.5349 | 0.6860 |
| M2 | CrossAttn | len128 | F1 | 0.3738 | 0.2500 | 0.4906 |
| M2 | CrossAttn | crop80 | AUC-ROC | 0.8761 | 0.7934 | 0.9398 |
| M2 | CrossAttn | crop80 | AUPRC | 0.5606 | 0.3514 | 0.7671 |
| M2 | CrossAttn | crop80 | Brier | 0.2396 | 0.2029 | 0.2768 |
| M2 | CrossAttn | crop80 | Accuracy | 0.6221 | 0.5523 | 0.6919 |
| M2 | CrossAttn | crop80 | F1 | 0.3810 | 0.2574 | 0.5000 |
| M2 | CrossAttn | crop60 | AUC-ROC | 0.8565 | 0.7666 | 0.9281 |
| M2 | CrossAttn | crop60 | AUPRC | 0.4906 | 0.2983 | 0.7067 |
| M2 | CrossAttn | crop60 | Brier | 0.2009 | 0.1638 | 0.2391 |
| M2 | CrossAttn | crop60 | Accuracy | 0.6395 | 0.5698 | 0.7093 |
| M2 | CrossAttn | crop60 | F1 | 0.3922 | 0.2651 | 0.5094 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8291 | 0.7029 | 0.9259 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3908 | 0.2472 | 0.6310 |
| M2_2 | CrossAttn | norm | Brier | 0.1894 | 0.1499 | 0.2304 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6337 | 0.5581 | 0.7035 |
| M2_2 | CrossAttn | norm | F1 | 0.3762 | 0.2472 | 0.4954 |
| M2_2 | CrossAttn | excl_extreme | AUC-ROC | 0.7978 | 0.6890 | 0.8851 |
| M2_2 | CrossAttn | excl_extreme | AUPRC | 0.3298 | 0.1676 | 0.5242 |
| M2_2 | CrossAttn | excl_extreme | Brier | 0.1763 | 0.1413 | 0.2155 |
| M2_2 | CrossAttn | excl_extreme | Accuracy | 0.7208 | 0.6494 | 0.7857 |
| M2_2 | CrossAttn | excl_extreme | F1 | 0.3768 | 0.2264 | 0.5135 |
| M2_2 | CrossAttn | len128 | AUC-ROC | 0.7988 | 0.7038 | 0.8803 |
| M2_2 | CrossAttn | len128 | AUPRC | 0.3619 | 0.2023 | 0.5554 |
| M2_2 | CrossAttn | len128 | Brier | 0.2100 | 0.1771 | 0.2452 |
| M2_2 | CrossAttn | len128 | Accuracy | 0.5756 | 0.5000 | 0.6512 |
| M2_2 | CrossAttn | len128 | F1 | 0.3303 | 0.2105 | 0.4407 |
| M2_2 | CrossAttn | crop80 | AUC-ROC | 0.8272 | 0.7315 | 0.9051 |
| M2_2 | CrossAttn | crop80 | AUPRC | 0.3970 | 0.2195 | 0.5980 |
| M2_2 | CrossAttn | crop80 | Brier | 0.1321 | 0.1032 | 0.1650 |
| M2_2 | CrossAttn | crop80 | Accuracy | 0.7849 | 0.7151 | 0.8432 |
| M2_2 | CrossAttn | crop80 | F1 | 0.4478 | 0.2807 | 0.5975 |
| M2_2 | CrossAttn | crop60 | AUC-ROC | 0.8316 | 0.7412 | 0.9070 |
| M2_2 | CrossAttn | crop60 | AUPRC | 0.3781 | 0.2211 | 0.5910 |
| M2_2 | CrossAttn | crop60 | Brier | 0.1645 | 0.1341 | 0.1985 |
| M2_2 | CrossAttn | crop60 | Accuracy | 0.7674 | 0.6977 | 0.8314 |
| M2_2 | CrossAttn | crop60 | F1 | 0.4872 | 0.3448 | 0.6170 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8691 | 0.7963 | 0.9265 |
| M3 | CrossAttn3 | norm | AUPRC | 0.4607 | 0.2815 | 0.6638 |
| M3 | CrossAttn3 | norm | Brier | 0.1487 | 0.1220 | 0.1776 |
| M3 | CrossAttn3 | norm | Accuracy | 0.7442 | 0.6744 | 0.8081 |
| M3 | CrossAttn3 | norm | F1 | 0.4634 | 0.3210 | 0.5910 |
| M3 | CrossAttn3 | excl_extreme | AUC-ROC | 0.8729 | 0.7865 | 0.9413 |
| M3 | CrossAttn3 | excl_extreme | AUPRC | 0.6161 | 0.4012 | 0.7977 |
| M3 | CrossAttn3 | excl_extreme | Brier | 0.1852 | 0.1505 | 0.2232 |
| M3 | CrossAttn3 | excl_extreme | Accuracy | 0.7468 | 0.6753 | 0.8117 |
| M3 | CrossAttn3 | excl_extreme | F1 | 0.4658 | 0.3077 | 0.5974 |
| M3 | CrossAttn3 | len128 | AUC-ROC | 0.8521 | 0.7567 | 0.9343 |
| M3 | CrossAttn3 | len128 | AUPRC | 0.5305 | 0.3270 | 0.7404 |
| M3 | CrossAttn3 | len128 | Brier | 0.1520 | 0.1232 | 0.1832 |
| M3 | CrossAttn3 | len128 | Accuracy | 0.7558 | 0.6860 | 0.8198 |
| M3 | CrossAttn3 | len128 | F1 | 0.4474 | 0.2963 | 0.5783 |
| M3 | CrossAttn3 | crop80 | AUC-ROC | 0.8496 | 0.7639 | 0.9223 |
| M3 | CrossAttn3 | crop80 | AUPRC | 0.4948 | 0.2900 | 0.7094 |
| M3 | CrossAttn3 | crop80 | Brier | 0.1961 | 0.1623 | 0.2322 |
| M3 | CrossAttn3 | crop80 | Accuracy | 0.7384 | 0.6744 | 0.8023 |
| M3 | CrossAttn3 | crop80 | F1 | 0.4000 | 0.2499 | 0.5352 |
| M3 | CrossAttn3 | crop60 | AUC-ROC | 0.8780 | 0.7966 | 0.9423 |
| M3 | CrossAttn3 | crop60 | AUPRC | 0.5335 | 0.3345 | 0.7478 |
| M3 | CrossAttn3 | crop60 | Brier | 0.2005 | 0.1668 | 0.2338 |
| M3 | CrossAttn3 | crop60 | Accuracy | 0.6802 | 0.6047 | 0.7500 |
| M3 | CrossAttn3 | crop60 | F1 | 0.4086 | 0.2750 | 0.5294 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-norm | 0.8325 | 0.7777 | -0.0549 | 1.217 | 2.236e-01 | ns |
| M1-LR vs M2-len128 | 0.8325 | 0.8452 | +0.0126 | -0.427 | 6.690e-01 | ns |
| M1-LR vs M2-crop80 | 0.8325 | 0.8761 | +0.0435 | -1.342 | 1.796e-01 | ns |
| M1-LR vs M2-crop60 | 0.8325 | 0.8565 | +0.0240 | -1.038 | 2.994e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-norm | 0.8325 | 0.8691 | +0.0366 | -1.171 | 2.415e-01 | ns |
| M1-LR vs M3-len128 | 0.8325 | 0.8521 | +0.0196 | -0.536 | 5.922e-01 | ns |
| M1-LR vs M3-crop80 | 0.8325 | 0.8496 | +0.0170 | -0.451 | 6.523e-01 | ns |
| M1-LR vs M3-crop60 | 0.8325 | 0.8780 | +0.0454 | -1.452 | 1.466e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M2_2-norm | 0.7777 | 0.8291 | +0.0514 | -0.872 | 3.832e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.7848 | 0.5073 | -0.2775 | 4.406 | 1.054e-05 | *** |
| M2-len128 vs M2_2-len128 | 0.8452 | 0.7988 | -0.0464 | 1.836 | 6.637e-02 | † |
| M2-crop80 vs M2_2-crop80 | 0.8761 | 0.8272 | -0.0489 | 1.854 | 6.381e-02 | † |
| M2-crop60 vs M2_2-crop60 | 0.8565 | 0.8316 | -0.0249 | 0.743 | 4.573e-01 | ns |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-norm vs M3-norm | 0.7777 | 0.8691 | +0.0915 | -2.745 | 6.057e-03 | ** |
| M2-excl_extreme vs M3-excl_extreme | 0.7848 | 0.8729 | +0.0881 | -2.382 | 1.721e-02 | * |
| M2-len128 vs M3-len128 | 0.8452 | 0.8521 | +0.0069 | -0.339 | 7.346e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.8761 | 0.8496 | -0.0265 | 1.581 | 1.138e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.8565 | 0.8780 | +0.0214 | -1.050 | 2.937e-01 | ns |

