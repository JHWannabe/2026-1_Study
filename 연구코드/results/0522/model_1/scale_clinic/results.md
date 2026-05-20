# SMI Binary Classification — Results

Generated: 2026-05-20 13:50  |  5-Fold CV  |  Model 1 (Clinic Only, LR)

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 334 | 280 | 83.8% | 54 | 16.2% |
| Train | F | 598 | 553 | 92.5% | 45 | 7.5% |
| Train | **All** | **932** | **833** | **89.4%** | **99** | **10.6%** |
| Test | M | 86 | 74 | 86.0% | 12 | 14.0% |
| Test | F | 148 | 135 | 91.2% | 13 | 8.8% |
| Test | **All** | **234** | **209** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 59.81 ± 12.21 | 20.00 | 60.00 | 89.00 |
| Train | F | 598 | 55.43 ± 11.87 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **932** | **57.00 ± 12.17** | **14.00** | **57.00** | **91.00** |
| Test | M | 86 | 58.88 ± 11.98 | 28.00 | 60.50 | 84.00 |
| Test | F | 148 | 54.66 ± 11.70 | 23.00 | 54.00 | 87.00 |
| Test | **All** | **234** | **56.21 ± 11.98** | **23.00** | **56.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 24.31 ± 3.02 | 14.34 | 24.27 | 32.67 |
| Train | F | 598 | 23.26 ± 3.27 | 12.02 | 23.06 | 34.61 |
| Train | **All** | **932** | **23.64 ± 3.22** | **12.02** | **23.56** | **34.61** |
| Test | M | 86 | 24.67 ± 3.13 | 17.43 | 24.15 | 32.56 |
| Test | F | 148 | 22.65 ± 3.12 | 16.44 | 22.14 | 34.20 |
| Test | **All** | **234** | **23.39 ± 3.27** | **16.44** | **23.30** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8751 | 0.4958 | 0.1892 | 0.7273 | 0.4138 |
| 2 | 0.8136 | 0.4156 | 0.1752 | 0.7540 | 0.3784 |
| 3 | 0.7980 | 0.4088 | 0.1989 | 0.7043 | 0.3529 |
| 4 | 0.7940 | 0.2960 | 0.1624 | 0.7527 | 0.3611 |
| 5 | 0.7804 | 0.4067 | 0.1720 | 0.7366 | 0.3288 |
| **Mean** | **0.8122** | **0.4046** | **0.1795** | **0.7350** | **0.3670** |
| **±Std** | 0.0332 | 0.0637 | 0.0129 | 0.0183 | 0.0283 |

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8149 | 0.3784 | 0.1934 | 0.6966 | 0.3604 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 86 | 0.7432 | 0.3389 | 0.2555 | 0.5930 | 0.3636 |
| F | 148 | 0.8610 | 0.4387 | 0.1573 | 0.7568 | 0.3571 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 143 | 66 |
| **True: Sarco**  | 5 | 20 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (LR) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `confusion_matrices.png` | Test-set confusion matrices (overall + by sex) |
| `test_roc_curves.png` | Final test-set ROC curve (overall) |
| `test_roc_by_sex.png` | Final test-set ROC curves split by sex |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
