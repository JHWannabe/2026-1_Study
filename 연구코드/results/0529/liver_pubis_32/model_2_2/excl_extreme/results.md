# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-28 22:10  |  5-Fold CV  |  Median best epoch: 14

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 290 | 247 | 85.2% | 43 | 14.8% |
| Train | F | 532 | 491 | 92.3% | 41 | 7.7% |
| Train | **All** | **822** | **738** | **89.8%** | **84** | **10.2%** |
| Test | M | 77 | 64 | 83.1% | 13 | 16.9% |
| Test | F | 128 | 118 | 92.2% | 10 | 7.8% |
| Test | **All** | **205** | **182** | **88.8%** | **23** | **11.2%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 290 | 59.75 ± 12.17 | 20.00 | 60.00 | 89.00 |
| Train | F | 532 | 55.09 ± 11.42 | 23.00 | 55.00 | 87.00 |
| Train | **All** | **822** | **56.73 ± 11.90** | **20.00** | **57.00** | **89.00** |
| Test | M | 77 | 60.06 ± 12.23 | 29.00 | 61.00 | 81.00 |
| Test | F | 128 | 55.41 ± 12.71 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **205** | **57.16 ± 12.73** | **23.00** | **58.00** | **83.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 290 | 24.36 ± 3.08 | 14.34 | 24.33 | 32.67 |
| Train | F | 532 | 23.17 ± 3.21 | 16.00 | 22.95 | 34.61 |
| Train | **All** | **822** | **23.59 ± 3.21** | **14.34** | **23.38** | **34.61** |
| Test | M | 77 | 24.46 ± 3.07 | 18.78 | 24.26 | 32.56 |
| Test | F | 128 | 23.14 ± 3.45 | 15.84 | 22.76 | 32.48 |
| Test | **All** | **205** | **23.63 ± 3.38** | **15.84** | **23.51** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8458 | 0.3086 | 0.1740 | 0.7091 | 0.4146 |
| 2 | 0.7432 | 0.2571 | 0.1304 | 0.8121 | 0.4151 |
| 3 | 0.8848 | 0.4793 | 0.1605 | 0.7317 | 0.4359 |
| 4 | 0.8235 | 0.2938 | 0.1824 | 0.6829 | 0.3810 |
| 5 | 0.7597 | 0.3221 | 0.2097 | 0.6463 | 0.3256 |
| **Mean** | **0.8114** | **0.3322** | **0.1714** | **0.7164** | **0.3944** |
| **±Std** | 0.0530 | 0.0767 | 0.0260 | 0.0556 | 0.0387 |

CrossAttn best val AUC per fold: Fold1=0.8458, Fold2=0.7432, Fold3=0.8848, Fold4=0.8235, Fold5=0.7597

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8060 | 0.3160 | 0.1929 | 0.6244 | 0.3419 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 77 | 0.7476 | 0.3543 | 0.2268 | 0.6234 | 0.4314 |
| F | 128 | 0.8339 | 0.2988 | 0.1725 | 0.6250 | 0.2727 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 108 | 74 |
| **True: Sarco**  | 3 | 20 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.8060 | 0.7069 | 0.8833 |
| AUPRC | 0.3160 | 0.1989 | 0.5112 |
| Brier | 0.1929 | 0.1641 | 0.2238 |
| Accuracy | 0.6244 | 0.5561 | 0.6878 |
| F1 | 0.3419 | 0.2330 | 0.4465 |

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
