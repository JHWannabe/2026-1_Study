# SMI Binary Classification — Results

Generated: 2026-05-20 13:51  |  5-Fold CV  |  Model 1 (Clinic Only, LR)

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 288 | 245 | 85.1% | 43 | 14.9% |
| Train | F | 402 | 363 | 90.3% | 39 | 9.7% |
| Train | **All** | **690** | **608** | **88.1%** | **82** | **11.9%** |
| Test | M | 69 | 57 | 82.6% | 12 | 17.4% |
| Test | F | 104 | 95 | 91.3% | 9 | 8.7% |
| Test | **All** | **173** | **152** | **87.9%** | **21** | **12.1%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 288 | 59.74 ± 12.07 | 23.00 | 60.00 | 89.00 |
| Train | F | 402 | 56.08 ± 12.56 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **690** | **57.60 ± 12.49** | **14.00** | **58.00** | **91.00** |
| Test | M | 69 | 58.28 ± 10.48 | 32.00 | 58.00 | 85.00 |
| Test | F | 104 | 56.68 ± 10.86 | 32.00 | 57.00 | 86.00 |
| Test | **All** | **173** | **57.32 ± 10.74** | **32.00** | **58.00** | **86.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 288 | 24.41 ± 3.02 | 14.34 | 24.28 | 32.67 |
| Train | F | 402 | 23.06 ± 3.33 | 12.02 | 22.94 | 34.61 |
| Train | **All** | **690** | **23.62 ± 3.27** | **12.02** | **23.52** | **34.61** |
| Test | M | 69 | 24.45 ± 3.09 | 18.17 | 24.28 | 32.59 |
| Test | F | 104 | 22.66 ± 3.29 | 16.44 | 22.19 | 34.20 |
| Test | **All** | **173** | **23.37 ± 3.33** | **16.44** | **22.95** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7382 | 0.4100 | 0.1958 | 0.6812 | 0.2414 |
| 2 | 0.8463 | 0.3902 | 0.1702 | 0.7391 | 0.4375 |
| 3 | 0.8689 | 0.5325 | 0.1749 | 0.7391 | 0.4194 |
| 4 | 0.8405 | 0.3844 | 0.1973 | 0.6884 | 0.4416 |
| 5 | 0.7681 | 0.4366 | 0.1656 | 0.7536 | 0.3929 |
| **Mean** | **0.8124** | **0.4307** | **0.1808** | **0.7203** | **0.3865** |
| **±Std** | 0.0502 | 0.0541 | 0.0132 | 0.0296 | 0.0746 |

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8336 | 0.3913 | 0.1872 | 0.7052 | 0.4000 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 69 | 0.7763 | 0.3872 | 0.2096 | 0.6957 | 0.4878 |
| F | 104 | 0.8678 | 0.4365 | 0.1723 | 0.7115 | 0.3182 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 105 | 47 |
| **True: Sarco**  | 4 | 17 |

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
