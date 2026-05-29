# Scaling Comparison — Test Set Performance (AEC 128pt)

## Best Cases Summary  (by Test overall AUC)

> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.

| Model | Sub-model | Best Case | AUC | AUPRC | Brier | Acc | F1 |
|-------|-----------|-----------|------: | ------: | ------: | ------: | ------:|
| M1 | LR | scale_all | 0.8030 | 0.3123 | 0.1913 | 0.7205 | 0.3725 |
| M2 | CrossAttn | norm | 0.8173 | 0.2855 | 0.1398 | 0.8210 | 0.3492 |
| M2_2 | CrossAttn | norm | 0.8175 | 0.3214 | 0.2174 | 0.6463 | 0.3415 |
| M3 | CrossAttn3 | norm | 0.8301 | 0.3806 | 0.2432 | 0.5022 | 0.2875 |

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
| len128 | 0.8008 | 0.3148 | 0.1452 | 0.7948 | 0.3380 |
| **norm** | 0.8173 | 0.2855 | 0.1398 | 0.8210 | 0.3492 |
| crop80 | 0.7994 | 0.2915 | 0.1677 | 0.6812 | 0.3540 |
| crop60 | 0.8150 | 0.2798 | 0.1644 | 0.7860 | 0.4368 |
| excl_extreme | 0.7737 | 0.2800 | 0.2381 | 0.5366 | 0.3066 |

---

## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (5 AEC variants)

> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.
> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.

### CrossAttn

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.7984 | 0.3187 | 0.2178 | 0.6157 | 0.3231 |
| **norm** | 0.8175 | 0.3214 | 0.2174 | 0.6463 | 0.3415 |
| crop80 | 0.8126 | 0.3176 | 0.2007 | 0.7162 | 0.3925 |
| crop60 | 0.7943 | 0.3247 | 0.2133 | 0.6943 | 0.3636 |
| excl_extreme | 0.7845 | 0.3114 | 0.1843 | 0.7854 | 0.3714 |

---

## Model 3 — Clinic + Scanner + AEC  (5 AEC variants)

### CrossAttn3

| Case | AUC | AUPRC | Brier | Acc | F1 |
|------|------: | ------: | ------: | ------: | ------:|
| len128 | 0.8028 | 0.2952 | 0.2237 | 0.7293 | 0.3673 |
| **norm** | 0.8301 | 0.3806 | 0.2432 | 0.5022 | 0.2875 |
| crop80 | 0.7998 | 0.2745 | 0.1850 | 0.7817 | 0.3590 |
| crop60 | 0.8004 | 0.2917 | 0.2035 | 0.7948 | 0.3733 |
| excl_extreme | 0.7851 | 0.2805 | 0.1682 | 0.7805 | 0.3478 |

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
| AUC-ROC  | 0.8061 | 0.8212 | +0.0151 | -1.034 | 3.60e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4052 | -0.0040 | 0.174 | 8.70e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1601 | -0.0206 | 1.344 | 2.50e-01 | 3.12e-01 |
| Accuracy  | 0.7561 | 0.7944 | +0.0383 | -1.041 | 3.57e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.4604 | +0.0441 | -1.286 | 2.68e-01 | 1.88e-01 |

### norm  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8129 | +0.0068 | -0.434 | 6.87e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3937 | -0.0156 | 0.544 | 6.15e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1751 | -0.0056 | 0.501 | 6.43e-01 | 1.00e+00 |
| Accuracy  | 0.7561 | 0.7911 | +0.0350 | -0.765 | 4.87e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4456 | +0.0293 | -0.725 | 5.09e-01 | 6.25e-01 |

### crop80  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8133 | +0.0071 | -0.709 | 5.18e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.3877 | -0.0215 | 0.657 | 5.47e-01 | 8.12e-01 |
| Brier  | 0.1808 | 0.1787 | -0.0021 | 0.211 | 8.43e-01 | 1.00e+00 |
| Accuracy  | 0.7561 | 0.7232 | -0.0329 | 0.949 | 3.96e-01 | 3.12e-01 |
| F1  | 0.4163 | 0.4017 | -0.0146 | 0.434 | 6.87e-01 | 8.12e-01 |

### crop60  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8193 | +0.0131 | -0.811 | 4.63e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3954 | -0.0139 | 0.220 | 8.36e-01 | 8.12e-01 |
| Brier  | 0.1808 | 0.1827 | +0.0020 | -0.105 | 9.21e-01 | 8.12e-01 |
| Accuracy  | 0.7561 | 0.7823 | +0.0262 | -0.803 | 4.67e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4404 | +0.0241 | -0.670 | 5.40e-01 | 6.25e-01 |

### excl_extreme  (M1-LR vs M2-CrossAttn)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8199 | +0.0137 | -0.312 | 7.70e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.3875 | -0.0217 | 0.263 | 8.05e-01 | 6.25e-01 |
| Brier  | 0.1808 | 0.1767 | -0.0040 | 0.148 | 8.89e-01 | 1.00e+00 |
| Accuracy  | 0.7561 | 0.7505 | -0.0056 | 0.133 | 9.01e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4103 | -0.0060 | 0.135 | 8.99e-01 | 1.00e+00 |

## M2 (CrossAttn) vs M3 (CrossAttn3)

> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.

#### Case: len128  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8212 | 0.8226 | +0.0014 | -0.204 | 8.49e-01 | 1.00e+00 |
| AUPRC  | 0.4052 | 0.4187 | +0.0135 | -0.314 | 7.69e-01 | 1.00e+00 |
| Brier  | 0.1601 | 0.1715 | +0.0114 | -0.680 | 5.34e-01 | 6.25e-01 |
| Accuracy  | 0.7944 | 0.8140 | +0.0197 | -0.667 | 5.41e-01 | 8.12e-01 |
| F1  | 0.4604 | 0.4573 | -0.0031 | 0.103 | 9.23e-01 | 1.00e+00 |

#### Case: norm  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8129 | 0.8211 | +0.0081 | -0.948 | 3.97e-01 | 3.75e-01 |
| AUPRC  | 0.3937 | 0.4202 | +0.0265 | -0.964 | 3.90e-01 | 4.38e-01 |
| Brier * | 0.1751 | 0.1503 | -0.0248 | 3.220 | 3.23e-02 | 6.25e-02 |
| Accuracy  | 0.7911 | 0.7724 | -0.0187 | 0.476 | 6.59e-01 | 8.12e-01 |
| F1  | 0.4456 | 0.4351 | -0.0105 | 0.297 | 7.82e-01 | 1.00e+00 |

#### Case: crop80  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8133 | 0.8201 | +0.0068 | -1.102 | 3.32e-01 | 3.12e-01 |
| AUPRC  | 0.3877 | 0.4139 | +0.0262 | -1.127 | 3.23e-01 | 3.12e-01 |
| Brier  | 0.1787 | 0.1902 | +0.0115 | -0.700 | 5.23e-01 | 8.12e-01 |
| Accuracy  | 0.7232 | 0.7943 | +0.0711 | -1.380 | 2.40e-01 | 4.38e-01 |
| F1  | 0.4017 | 0.4522 | +0.0504 | -1.287 | 2.68e-01 | 4.38e-01 |

#### Case: crop60  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8193 | 0.8210 | +0.0017 | -0.249 | 8.16e-01 | 1.00e+00 |
| AUPRC  | 0.3954 | 0.4053 | +0.0100 | -0.307 | 7.74e-01 | 1.00e+00 |
| Brier † | 0.1827 | 0.1721 | -0.0106 | 2.625 | 5.85e-02 | 1.25e-01 |
| Accuracy  | 0.7823 | 0.7889 | +0.0066 | -0.160 | 8.80e-01 | 1.00e+00 |
| F1  | 0.4404 | 0.4498 | +0.0094 | -0.295 | 7.83e-01 | 8.12e-01 |

#### Case: excl_extreme  (M2-CrossAttn vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8199 | 0.8131 | -0.0068 | 0.425 | 6.93e-01 | 6.25e-01 |
| AUPRC  | 0.3875 | 0.4281 | +0.0405 | -1.131 | 3.21e-01 | 3.12e-01 |
| Brier † | 0.1767 | 0.1570 | -0.0197 | 2.615 | 5.91e-02 | 6.25e-02 |
| Accuracy  | 0.7505 | 0.8151 | +0.0646 | -1.396 | 2.35e-01 | 3.75e-01 |
| F1  | 0.4103 | 0.4589 | +0.0486 | -0.941 | 4.00e-01 | 3.12e-01 |

## M1 (LR) vs M3 (CrossAttn3)

> A = M1 LR, B = M3 CrossAttn3.

### len128  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8226 | +0.0165 | -1.109 | 3.30e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4187 | +0.0095 | -0.185 | 8.62e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1715 | -0.0093 | 1.257 | 2.77e-01 | 3.12e-01 |
| Accuracy  | 0.7561 | 0.8140 | +0.0579 | -1.333 | 2.53e-01 | 4.38e-01 |
| F1  | 0.4163 | 0.4573 | +0.0409 | -1.129 | 3.22e-01 | 8.12e-01 |

### norm  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8211 | +0.0149 | -0.876 | 4.31e-01 | 4.38e-01 |
| AUPRC  | 0.4092 | 0.4202 | +0.0110 | -0.261 | 8.07e-01 | 1.00e+00 |
| Brier † | 0.1808 | 0.1503 | -0.0304 | 2.594 | 6.04e-02 | 1.25e-01 |
| Accuracy  | 0.7561 | 0.7724 | +0.0163 | -0.412 | 7.02e-01 | 1.00e+00 |
| F1  | 0.4163 | 0.4351 | +0.0187 | -0.491 | 6.49e-01 | 8.12e-01 |

### crop80  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8201 | +0.0139 | -0.947 | 3.97e-01 | 6.25e-01 |
| AUPRC  | 0.4092 | 0.4139 | +0.0047 | -0.115 | 9.14e-01 | 8.12e-01 |
| Brier  | 0.1808 | 0.1902 | +0.0094 | -0.709 | 5.17e-01 | 4.38e-01 |
| Accuracy  | 0.7561 | 0.7943 | +0.0382 | -0.938 | 4.01e-01 | 8.12e-01 |
| F1  | 0.4163 | 0.4522 | +0.0358 | -1.097 | 3.34e-01 | 6.25e-01 |

### crop60  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8210 | +0.0148 | -0.781 | 4.78e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4053 | -0.0039 | 0.111 | 9.17e-01 | 1.00e+00 |
| Brier  | 0.1808 | 0.1721 | -0.0087 | 0.398 | 7.11e-01 | 6.25e-01 |
| Accuracy  | 0.7561 | 0.7889 | +0.0328 | -0.671 | 5.39e-01 | 5.00e-01 |
| F1  | 0.4163 | 0.4498 | +0.0334 | -0.881 | 4.28e-01 | 4.38e-01 |

### excl_extreme  (M1-LR vs M3-CrossAttn3)

| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |
|--------|-------:|-------:|-------:|-------:|--------:|--------:|
| AUC-ROC  | 0.8061 | 0.8131 | +0.0069 | -0.158 | 8.82e-01 | 8.12e-01 |
| AUPRC  | 0.4092 | 0.4281 | +0.0188 | -0.234 | 8.27e-01 | 8.12e-01 |
| Brier  | 0.1808 | 0.1570 | -0.0238 | 0.872 | 4.32e-01 | 6.25e-01 |
| Accuracy  | 0.7561 | 0.8151 | +0.0591 | -1.036 | 3.59e-01 | 6.25e-01 |
| F1  | 0.4163 | 0.4589 | +0.0426 | -0.566 | 6.02e-01 | 1.00e+00 |

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
| M2 | CrossAttn | len128 | AUC-ROC | 0.8008 | 0.7069 | 0.8773 |
| M2 | CrossAttn | len128 | AUPRC | 0.3148 | 0.1862 | 0.4984 |
| M2 | CrossAttn | len128 | Brier | 0.1452 | 0.1199 | 0.1710 |
| M2 | CrossAttn | len128 | Accuracy | 0.7948 | 0.7424 | 0.8472 |
| M2 | CrossAttn | len128 | F1 | 0.3380 | 0.1972 | 0.4737 |
| M2 | CrossAttn | norm | AUC-ROC | 0.8173 | 0.7442 | 0.8805 |
| M2 | CrossAttn | norm | AUPRC | 0.2855 | 0.1794 | 0.4700 |
| M2 | CrossAttn | norm | Brier | 0.1398 | 0.1170 | 0.1633 |
| M2 | CrossAttn | norm | Accuracy | 0.8210 | 0.7686 | 0.8690 |
| M2 | CrossAttn | norm | F1 | 0.3492 | 0.1967 | 0.4912 |
| M2 | CrossAttn | crop80 | AUC-ROC | 0.7994 | 0.7100 | 0.8767 |
| M2 | CrossAttn | crop80 | AUPRC | 0.2915 | 0.1782 | 0.4892 |
| M2 | CrossAttn | crop80 | Brier | 0.1677 | 0.1425 | 0.1915 |
| M2 | CrossAttn | crop80 | Accuracy | 0.6812 | 0.6201 | 0.7424 |
| M2 | CrossAttn | crop80 | F1 | 0.3540 | 0.2393 | 0.4660 |
| M2 | CrossAttn | crop60 | AUC-ROC | 0.8150 | 0.7313 | 0.8849 |
| M2 | CrossAttn | crop60 | AUPRC | 0.2798 | 0.1814 | 0.4551 |
| M2 | CrossAttn | crop60 | Brier | 0.1644 | 0.1361 | 0.1918 |
| M2 | CrossAttn | crop60 | Accuracy | 0.7860 | 0.7336 | 0.8384 |
| M2 | CrossAttn | crop60 | F1 | 0.4368 | 0.3014 | 0.5664 |
| M2 | CrossAttn | excl_extreme | AUC-ROC | 0.7737 | 0.6777 | 0.8582 |
| M2 | CrossAttn | excl_extreme | AUPRC | 0.2800 | 0.1573 | 0.4611 |
| M2 | CrossAttn | excl_extreme | Brier | 0.2381 | 0.2065 | 0.2716 |
| M2 | CrossAttn | excl_extreme | Accuracy | 0.5366 | 0.4683 | 0.6049 |
| M2 | CrossAttn | excl_extreme | F1 | 0.3066 | 0.2016 | 0.4058 |
| M2_2 | CrossAttn | len128 | AUC-ROC | 0.7984 | 0.7067 | 0.8755 |
| M2_2 | CrossAttn | len128 | AUPRC | 0.3187 | 0.1973 | 0.5225 |
| M2_2 | CrossAttn | len128 | Brier | 0.2178 | 0.1855 | 0.2504 |
| M2_2 | CrossAttn | len128 | Accuracy | 0.6157 | 0.5502 | 0.6812 |
| M2_2 | CrossAttn | len128 | F1 | 0.3231 | 0.2137 | 0.4265 |
| M2_2 | CrossAttn | norm | AUC-ROC | 0.8175 | 0.7353 | 0.8879 |
| M2_2 | CrossAttn | norm | AUPRC | 0.3214 | 0.1921 | 0.4993 |
| M2_2 | CrossAttn | norm | Brier | 0.2174 | 0.1852 | 0.2474 |
| M2_2 | CrossAttn | norm | Accuracy | 0.6463 | 0.5852 | 0.7074 |
| M2_2 | CrossAttn | norm | F1 | 0.3415 | 0.2222 | 0.4496 |
| M2_2 | CrossAttn | crop80 | AUC-ROC | 0.8126 | 0.7318 | 0.8816 |
| M2_2 | CrossAttn | crop80 | AUPRC | 0.3176 | 0.1959 | 0.5198 |
| M2_2 | CrossAttn | crop80 | Brier | 0.2007 | 0.1745 | 0.2279 |
| M2_2 | CrossAttn | crop80 | Accuracy | 0.7162 | 0.6550 | 0.7773 |
| M2_2 | CrossAttn | crop80 | F1 | 0.3925 | 0.2653 | 0.5049 |
| M2_2 | CrossAttn | crop60 | AUC-ROC | 0.7943 | 0.6855 | 0.8859 |
| M2_2 | CrossAttn | crop60 | AUPRC | 0.3247 | 0.2015 | 0.5234 |
| M2_2 | CrossAttn | crop60 | Brier | 0.2133 | 0.1802 | 0.2460 |
| M2_2 | CrossAttn | crop60 | Accuracy | 0.6943 | 0.6332 | 0.7511 |
| M2_2 | CrossAttn | crop60 | F1 | 0.3636 | 0.2400 | 0.4787 |
| M2_2 | CrossAttn | excl_extreme | AUC-ROC | 0.7845 | 0.6865 | 0.8648 |
| M2_2 | CrossAttn | excl_extreme | AUPRC | 0.3114 | 0.1855 | 0.4834 |
| M2_2 | CrossAttn | excl_extreme | Brier | 0.1843 | 0.1538 | 0.2199 |
| M2_2 | CrossAttn | excl_extreme | Accuracy | 0.7854 | 0.7268 | 0.8390 |
| M2_2 | CrossAttn | excl_extreme | F1 | 0.3714 | 0.2143 | 0.5071 |
| M3 | CrossAttn3 | len128 | AUC-ROC | 0.8028 | 0.7148 | 0.8811 |
| M3 | CrossAttn3 | len128 | AUPRC | 0.2952 | 0.1890 | 0.4905 |
| M3 | CrossAttn3 | len128 | Brier | 0.2237 | 0.1923 | 0.2525 |
| M3 | CrossAttn3 | len128 | Accuracy | 0.7293 | 0.6725 | 0.7860 |
| M3 | CrossAttn3 | len128 | F1 | 0.3673 | 0.2444 | 0.4842 |
| M3 | CrossAttn3 | norm | AUC-ROC | 0.8301 | 0.7433 | 0.9029 |
| M3 | CrossAttn3 | norm | AUPRC | 0.3806 | 0.2244 | 0.5724 |
| M3 | CrossAttn3 | norm | Brier | 0.2432 | 0.2073 | 0.2762 |
| M3 | CrossAttn3 | norm | Accuracy | 0.5022 | 0.4367 | 0.5677 |
| M3 | CrossAttn3 | norm | F1 | 0.2875 | 0.1939 | 0.3787 |
| M3 | CrossAttn3 | crop80 | AUC-ROC | 0.7998 | 0.7059 | 0.8792 |
| M3 | CrossAttn3 | crop80 | AUPRC | 0.2745 | 0.1739 | 0.4562 |
| M3 | CrossAttn3 | crop80 | Brier | 0.1850 | 0.1522 | 0.2167 |
| M3 | CrossAttn3 | crop80 | Accuracy | 0.7817 | 0.7293 | 0.8341 |
| M3 | CrossAttn3 | crop80 | F1 | 0.3590 | 0.2222 | 0.4895 |
| M3 | CrossAttn3 | crop60 | AUC-ROC | 0.8004 | 0.7036 | 0.8807 |
| M3 | CrossAttn3 | crop60 | AUPRC | 0.2917 | 0.1827 | 0.4901 |
| M3 | CrossAttn3 | crop60 | Brier | 0.2035 | 0.1689 | 0.2352 |
| M3 | CrossAttn3 | crop60 | Accuracy | 0.7948 | 0.7424 | 0.8472 |
| M3 | CrossAttn3 | crop60 | F1 | 0.3733 | 0.2319 | 0.5107 |
| M3 | CrossAttn3 | excl_extreme | AUC-ROC | 0.7851 | 0.6907 | 0.8635 |
| M3 | CrossAttn3 | excl_extreme | AUPRC | 0.2805 | 0.1589 | 0.4617 |
| M3 | CrossAttn3 | excl_extreme | Brier | 0.1682 | 0.1434 | 0.1933 |
| M3 | CrossAttn3 | excl_extreme | Accuracy | 0.7805 | 0.7220 | 0.8390 |
| M3 | CrossAttn3 | excl_extreme | F1 | 0.3478 | 0.1923 | 0.4866 |

---

# Test Set — DeLong AUC Comparison

> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.
> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.
> \*\*\* p<0.001 · \*\* p<0.01 · \* p<0.05 · † p<0.10 · ns p≥0.10

## M1 LR vs M2 CrossAttn

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M2-len128 | 0.8030 | 0.8008 | -0.0022 | 0.097 | 9.229e-01 | ns |
| M1-LR vs M2-norm | 0.8030 | 0.8173 | +0.0142 | -0.829 | 4.073e-01 | ns |
| M1-LR vs M2-crop80 | 0.8030 | 0.7994 | -0.0037 | 0.144 | 8.853e-01 | ns |
| M1-LR vs M2-crop60 | 0.8030 | 0.8150 | +0.0120 | -0.537 | 5.914e-01 | ns |

## M1 LR vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M1-LR vs M3-len128 | 0.8030 | 0.8028 | -0.0002 | 0.008 | 9.937e-01 | ns |
| M1-LR vs M3-norm | 0.8030 | 0.8301 | +0.0270 | -1.035 | 3.007e-01 | ns |
| M1-LR vs M3-crop80 | 0.8030 | 0.7998 | -0.0033 | 0.123 | 9.022e-01 | ns |
| M1-LR vs M3-crop60 | 0.8030 | 0.8004 | -0.0026 | 0.091 | 9.275e-01 | ns |

## M2 Matched vs M2_2 Unmatched

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len128 vs M2_2-len128 | 0.8008 | 0.7984 | -0.0024 | 0.093 | 9.258e-01 | ns |
| M2-norm vs M2_2-norm | 0.8173 | 0.8175 | +0.0002 | -0.011 | 9.914e-01 | ns |
| M2-crop80 vs M2_2-crop80 | 0.7994 | 0.8126 | +0.0132 | -0.521 | 6.027e-01 | ns |
| M2-crop60 vs M2_2-crop60 | 0.8150 | 0.7943 | -0.0207 | 0.546 | 5.849e-01 | ns |
| M2-excl_extreme vs M2_2-excl_extreme | 0.7737 | 0.4868 | -0.2869 | 3.645 | 2.678e-04 | *** |

## M2 CrossAttn vs M3 CrossAttn3

| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
|-----------|------:|------:|------:|-------:|------:|-----|
| M2-len128 vs M3-len128 | 0.8008 | 0.8028 | +0.0020 | -0.083 | 9.341e-01 | ns |
| M2-norm vs M3-norm | 0.8173 | 0.8301 | +0.0128 | -0.630 | 5.287e-01 | ns |
| M2-crop80 vs M3-crop80 | 0.7994 | 0.7998 | +0.0004 | -0.019 | 9.846e-01 | ns |
| M2-crop60 vs M3-crop60 | 0.8150 | 0.8004 | -0.0146 | 0.733 | 4.636e-01 | ns |
| M2-excl_extreme vs M3-excl_extreme | 0.7737 | 0.7851 | +0.0114 | -0.491 | 6.231e-01 | ns |

