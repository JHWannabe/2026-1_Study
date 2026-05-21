# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 18:51  |  5-Fold CV  |  Median best epoch: 6

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 242 | 204 | 84.3% | 38 | 15.7% |
| Train | F | 375 | 342 | 91.2% | 33 | 8.8% |
| Train | **All** | **617** | **546** | **88.5%** | **71** | **11.5%** |
| Test | M | 68 | 61 | 89.7% | 7 | 10.3% |
| Test | F | 86 | 74 | 86.0% | 12 | 14.0% |
| Test | **All** | **154** | **135** | **87.7%** | **19** | **12.3%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 242 | 59.31 ± 11.80 | 23.00 | 59.00 | 89.00 |
| Train | F | 375 | 55.79 ± 11.61 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **617** | **57.17 ± 11.81** | **14.00** | **58.00** | **91.00** |
| Test | M | 68 | 61.06 ± 12.10 | 29.00 | 61.00 | 83.00 |
| Test | F | 86 | 58.16 ± 13.91 | 18.00 | 59.00 | 87.00 |
| Test | **All** | **154** | **59.44 ± 13.22** | **18.00** | **59.50** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 242 | 24.11 ± 2.89 | 16.39 | 24.12 | 32.33 |
| Train | F | 375 | 22.72 ± 3.01 | 12.02 | 22.66 | 31.50 |
| Train | **All** | **617** | **23.27 ± 3.04** | **12.02** | **23.24** | **32.33** |
| Test | M | 68 | 24.54 ± 2.89 | 17.51 | 24.40 | 32.56 |
| Test | F | 86 | 23.39 ± 3.09 | 17.15 | 23.19 | 31.14 |
| Test | **All** | **154** | **23.90 ± 3.05** | **17.15** | **23.45** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8000 | 0.4955 | 0.2232 | 0.6048 | 0.3099 |
| 2 | 0.8312 | 0.3746 | 0.1740 | 0.7177 | 0.4068 |
| 3 | 0.7877 | 0.5269 | 0.1823 | 0.6992 | 0.3729 |
| 4 | 0.8342 | 0.3308 | 0.2113 | 0.6423 | 0.3714 |
| 5 | 0.8644 | 0.5486 | 0.1698 | 0.7724 | 0.4815 |
| **Mean** | **0.8235** | **0.4553** | **0.1921** | **0.6873** | **0.3885** |
| **±Std** | 0.0271 | 0.0865 | 0.0212 | 0.0585 | 0.0560 |

CrossAttn best val AUC per fold: Fold1=0.8000, Fold2=0.8312, Fold3=0.7877, Fold4=0.8342, Fold5=0.8644

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8055 | 0.2855 | 0.2373 | 0.6558 | 0.3908 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 68 | 0.7705 | 0.2211 | 0.2973 | 0.6029 | 0.3415 |
| F | 86 | 0.8502 | 0.5130 | 0.1898 | 0.6977 | 0.4348 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 84 | 51 |
| **True: Sarco**  | 2 | 17 |

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
