# SMI Binary Classification — Results

Generated: 2026-05-25 17:47  |  5-Fold CV  |  Model 1 (Clinic Only, LR)

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 335 | 283 | 84.5% | 52 | 15.5% |
| Train | F | 595 | 548 | 92.1% | 47 | 7.9% |
| Train | **All** | **930** | **831** | **89.4%** | **99** | **10.6%** |
| Test | M | 85 | 71 | 83.5% | 14 | 16.5% |
| Test | F | 148 | 137 | 92.6% | 11 | 7.4% |
| Test | **All** | **233** | **208** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 335 | 59.78 ± 12.13 | 20.00 | 60.00 | 89.00 |
| Train | F | 595 | 55.26 ± 11.90 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **930** | **56.89 ± 12.18** | **14.00** | **57.00** | **91.00** |
| Test | M | 85 | 59.08 ± 12.34 | 29.00 | 60.00 | 84.00 |
| Test | F | 148 | 55.47 ± 11.64 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **233** | **56.79 ± 12.03** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 335 | 24.39 ± 2.93 | 17.33 | 24.24 | 32.67 |
| Train | F | 595 | 23.15 ± 3.17 | 16.00 | 22.96 | 34.61 |
| Train | **All** | **930** | **23.60 ± 3.15** | **16.00** | **23.45** | **34.61** |
| Test | M | 85 | 24.34 ± 3.40 | 14.34 | 24.26 | 32.56 |
| Test | F | 148 | 23.06 ± 3.53 | 12.02 | 22.64 | 32.48 |
| Test | **All** | **233** | **23.52 ± 3.53** | **12.02** | **23.41** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7852 | 0.3071 | 0.1524 | 0.6344 | 0.3462 |
| 2 | 0.8253 | 0.4085 | 0.1817 | 0.6022 | 0.3509 |
| 3 | 0.7913 | 0.3465 | 0.1884 | 0.6828 | 0.3789 |
| 4 | 0.8837 | 0.5483 | 0.1852 | 0.8656 | 0.5614 |
| 5 | 0.7595 | 0.3551 | 0.1926 | 0.7581 | 0.4000 |
| **Mean** | **0.8090** | **0.3931** | **0.1800** | **0.7086** | **0.4075** |
| **±Std** | 0.0428 | 0.0841 | 0.0143 | 0.0944 | 0.0794 |

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8127 | 0.4105 | 0.1869 | 0.6481 | 0.3387 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 85 | 0.7716 | 0.4681 | 0.2433 | 0.5529 | 0.4062 |
| F | 148 | 0.7963 | 0.3909 | 0.1546 | 0.7027 | 0.2667 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 130 | 78 |
| **True: Sarco**  | 4 | 21 |

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
