# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 23:06  |  5-Fold CV  |  Median best epoch: 6

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
| 1 | 0.7942 | 0.5089 | 0.1725 | 0.8145 | 0.4651 |
| 2 | 0.7416 | 0.2515 | 0.1801 | 0.5968 | 0.3421 |
| 3 | 0.8670 | 0.4505 | 0.1402 | 0.8455 | 0.5366 |
| 4 | 0.7772 | 0.3008 | 0.1658 | 0.8130 | 0.4889 |
| 5 | 0.7811 | 0.3352 | 0.2259 | 0.6992 | 0.3729 |
| **Mean** | **0.7922** | **0.3694** | **0.1769** | **0.7538** | **0.4411** |
| **±Std** | 0.0412 | 0.0957 | 0.0279 | 0.0930 | 0.0727 |

CrossAttn best val AUC per fold: Fold1=0.7942, Fold2=0.7416, Fold3=0.8670, Fold4=0.7772, Fold5=0.7811

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8485 | 0.4976 | 0.2141 | 0.6883 | 0.4286 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 64 | 0.8388 | 0.5419 | 0.2181 | 0.6719 | 0.4878 |
| F | 90 | 0.8500 | 0.4930 | 0.2112 | 0.7000 | 0.3721 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 88 | 45 |
| **True: Sarco**  | 3 | 18 |

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
