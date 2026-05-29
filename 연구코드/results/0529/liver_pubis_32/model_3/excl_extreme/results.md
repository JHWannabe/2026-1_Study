# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-28 22:10  |  5-Fold CV  |  Median best epoch: 7

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 277 | 239 | 86.3% | 38 | 13.7% |
| Train | F | 545 | 502 | 92.1% | 43 | 7.9% |
| Train | **All** | **822** | **741** | **90.1%** | **81** | **9.9%** |
| Test | M | 72 | 59 | 81.9% | 13 | 18.1% |
| Test | F | 133 | 124 | 93.2% | 9 | 6.8% |
| Test | **All** | **205** | **183** | **89.3%** | **22** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 277 | 60.19 ± 11.83 | 20.00 | 60.00 | 89.00 |
| Train | F | 545 | 55.20 ± 11.34 | 23.00 | 55.00 | 87.00 |
| Train | **All** | **822** | **56.88 ± 11.75** | **20.00** | **57.00** | **89.00** |
| Test | M | 72 | 59.21 ± 12.53 | 29.00 | 58.50 | 81.00 |
| Test | F | 133 | 54.96 ± 12.06 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **205** | **56.45 ± 12.40** | **23.00** | **57.00** | **83.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 277 | 24.10 ± 2.87 | 14.34 | 24.11 | 32.33 |
| Train | F | 545 | 23.04 ± 3.02 | 12.02 | 22.95 | 32.24 |
| Train | **All** | **822** | **23.40 ± 3.01** | **12.02** | **23.32** | **32.33** |
| Test | M | 72 | 24.20 ± 3.16 | 18.78 | 24.12 | 32.56 |
| Test | F | 133 | 22.93 ± 3.01 | 16.51 | 22.55 | 30.84 |
| Test | **All** | **205** | **23.38 ± 3.12** | **16.51** | **23.18** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8430 | 0.4268 | 0.2733 | 0.8424 | 0.5000 |
| 2 | 0.7970 | 0.3654 | 0.2326 | 0.8061 | 0.4074 |
| 3 | 0.9037 | 0.5027 | 0.1515 | 0.8049 | 0.4667 |
| 4 | 0.7073 | 0.2620 | 0.1799 | 0.7988 | 0.3529 |
| 5 | 0.8062 | 0.3056 | 0.1419 | 0.8049 | 0.4483 |
| **Mean** | **0.8114** | **0.3725** | **0.1958** | **0.8114** | **0.4351** |
| **±Std** | 0.0642 | 0.0856 | 0.0500 | 0.0157 | 0.0508 |

CrossAttn best val AUC per fold: Fold1=0.8430, Fold2=0.7970, Fold3=0.9037, Fold4=0.7073, Fold5=0.8062

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7688 | 0.2594 | 0.1770 | 0.7756 | 0.3235 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 72 | 0.7145 | 0.3112 | 0.2310 | 0.6944 | 0.3889 |
| F | 133 | 0.8145 | 0.2738 | 0.1477 | 0.8195 | 0.2500 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 148 | 35 |
| **True: Sarco**  | 11 | 11 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.7688 | 0.6702 | 0.8580 |
| AUPRC | 0.2594 | 0.1565 | 0.4520 |
| Brier | 0.1770 | 0.1475 | 0.2081 |
| Accuracy | 0.7756 | 0.7171 | 0.8294 |
| F1 | 0.3235 | 0.1791 | 0.4688 |

---

## 5. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (CrossAttn) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `training_curves.png` | Loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curve |
| `test_roc_by_sex.png` | Final test-set ROC curves by sex |
| `confusion_matrices.png` | Test-set confusion matrices |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
