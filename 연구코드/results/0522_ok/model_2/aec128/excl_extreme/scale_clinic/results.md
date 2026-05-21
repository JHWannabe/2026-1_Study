# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 19:52  |  5-Fold CV  |  Median best epoch: 10

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 245 | 211 | 86.1% | 34 | 13.9% |
| Train | F | 372 | 337 | 90.6% | 35 | 9.4% |
| Train | **All** | **617** | **548** | **88.8%** | **69** | **11.2%** |
| Test | M | 64 | 53 | 82.8% | 11 | 17.2% |
| Test | F | 90 | 80 | 88.9% | 10 | 11.1% |
| Test | **All** | **154** | **133** | **86.4%** | **21** | **13.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 245 | 59.47 ± 11.67 | 23.00 | 59.00 | 85.00 |
| Train | F | 372 | 56.48 ± 11.97 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **617** | **57.67 ± 11.95** | **14.00** | **58.00** | **91.00** |
| Test | M | 64 | 60.75 ± 12.72 | 28.00 | 62.00 | 89.00 |
| Test | F | 90 | 54.97 ± 12.65 | 24.00 | 55.00 | 86.00 |
| Test | **All** | **154** | **57.37 ± 12.99** | **24.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 245 | 24.20 ± 2.76 | 16.39 | 24.17 | 32.56 |
| Train | F | 372 | 22.89 ± 3.12 | 12.02 | 22.81 | 31.50 |
| Train | **All** | **617** | **23.41 ± 3.05** | **12.02** | **23.31** | **32.56** |
| Test | M | 64 | 24.23 ± 3.36 | 17.33 | 24.12 | 32.33 |
| Test | F | 90 | 22.69 ± 2.83 | 16.51 | 22.61 | 30.63 |
| Test | **All** | **154** | **23.33 ± 3.15** | **16.51** | **23.25** | **32.33** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8019 | 0.3774 | 0.1908 | 0.6935 | 0.3667 |
| 2 | 0.7623 | 0.3153 | 0.1758 | 0.7742 | 0.3333 |
| 3 | 0.8742 | 0.4494 | 0.1287 | 0.7642 | 0.3556 |
| 4 | 0.7975 | 0.3866 | 0.2984 | 0.5203 | 0.3059 |
| 5 | 0.8014 | 0.3144 | 0.2138 | 0.6992 | 0.3934 |
| **Mean** | **0.8075** | **0.3686** | **0.2015** | **0.6903** | **0.3510** |
| **±Std** | 0.0365 | 0.0504 | 0.0559 | 0.0911 | 0.0297 |

CrossAttn best val AUC per fold: Fold1=0.8019, Fold2=0.7623, Fold3=0.8742, Fold4=0.7975, Fold5=0.8014

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8786 | 0.5451 | 0.1726 | 0.7013 | 0.4651 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 64 | 0.8559 | 0.5812 | 0.1900 | 0.6719 | 0.5116 |
| F | 90 | 0.8875 | 0.6231 | 0.1602 | 0.7222 | 0.4186 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 88 | 45 |
| **True: Sarco**  | 1 | 20 |

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
