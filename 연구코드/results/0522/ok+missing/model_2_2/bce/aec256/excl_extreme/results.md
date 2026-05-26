# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-22 01:21  |  5-Fold CV  |  Median best epoch: 4

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 302 | 256 | 84.8% | 46 | 15.2% |
| Train | F | 533 | 493 | 92.5% | 40 | 7.5% |
| Train | **All** | **835** | **749** | **89.7%** | **86** | **10.3%** |
| Test | M | 80 | 69 | 86.2% | 11 | 13.8% |
| Test | F | 129 | 118 | 91.5% | 11 | 8.5% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 302 | 59.41 ± 12.04 | 20.00 | 59.00 | 89.00 |
| Train | F | 533 | 55.34 ± 11.98 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **835** | **56.81 ± 12.16** | **14.00** | **57.00** | **91.00** |
| Test | M | 80 | 58.96 ± 12.47 | 29.00 | 60.00 | 84.00 |
| Test | F | 129 | 55.14 ± 12.04 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **209** | **56.60 ± 12.35** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 302 | 24.39 ± 2.93 | 17.33 | 24.26 | 32.67 |
| Train | F | 533 | 23.14 ± 3.20 | 16.00 | 22.95 | 34.61 |
| Train | **All** | **835** | **23.59 ± 3.16** | **16.00** | **23.46** | **34.61** |
| Test | M | 80 | 24.14 ± 3.35 | 14.34 | 24.03 | 32.56 |
| Test | F | 129 | 22.96 ± 3.62 | 12.02 | 22.51 | 32.48 |
| Test | **All** | **209** | **23.41 ± 3.56** | **12.02** | **23.26** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8980 | 0.4770 | 0.1534 | 0.7904 | 0.4776 |
| 2 | 0.7816 | 0.3061 | 0.2759 | 0.6826 | 0.3765 |
| 3 | 0.7769 | 0.3248 | 0.1730 | 0.7126 | 0.3846 |
| 4 | 0.7969 | 0.3405 | 0.1440 | 0.6407 | 0.3478 |
| 5 | 0.8568 | 0.5207 | 0.2014 | 0.8743 | 0.5714 |
| **Mean** | **0.8220** | **0.3938** | **0.1896** | **0.7401** | **0.4316** |
| **±Std** | 0.0475 | 0.0875 | 0.0474 | 0.0830 | 0.0824 |

CrossAttn best val AUC per fold: Fold1=0.8980, Fold2=0.7816, Fold3=0.7769, Fold4=0.7969, Fold5=0.8568

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8357 | 0.4007 | 0.2085 | 0.6986 | 0.3762 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 80 | 0.8551 | 0.4965 | 0.2549 | 0.6500 | 0.4400 |
| F | 129 | 0.7997 | 0.3580 | 0.1798 | 0.7287 | 0.3137 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 127 | 60 |
| **True: Sarco**  | 3 | 19 |

---

## 4. Figures

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
