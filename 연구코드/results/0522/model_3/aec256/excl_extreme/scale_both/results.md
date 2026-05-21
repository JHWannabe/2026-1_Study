# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 21:36  |  5-Fold CV  |  Median best epoch: 13

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 293 | 251 | 85.7% | 42 | 14.3% |
| Train | F | 542 | 500 | 92.3% | 42 | 7.7% |
| Train | **All** | **835** | **751** | **89.9%** | **84** | **10.1%** |
| Test | M | 67 | 56 | 83.6% | 11 | 16.4% |
| Test | F | 142 | 131 | 92.3% | 11 | 7.7% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 293 | 60.09 ± 11.89 | 20.00 | 60.00 | 89.00 |
| Train | F | 542 | 55.35 ± 11.64 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **835** | **57.01 ± 11.95** | **14.00** | **57.00** | **91.00** |
| Test | M | 67 | 58.01 ± 12.42 | 29.00 | 58.00 | 80.00 |
| Test | F | 142 | 54.63 ± 11.92 | 18.00 | 54.50 | 86.00 |
| Test | **All** | **209** | **55.72 ± 12.18** | **18.00** | **56.00** | **86.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 293 | 24.02 ± 2.96 | 14.34 | 23.94 | 32.33 |
| Train | F | 542 | 22.96 ± 3.06 | 12.02 | 22.91 | 32.24 |
| Train | **All** | **835** | **23.33 ± 3.07** | **12.02** | **23.31** | **32.33** |
| Test | M | 67 | 24.47 ± 2.65 | 19.23 | 24.26 | 32.56 |
| Test | F | 142 | 23.27 ± 2.90 | 16.44 | 22.94 | 31.50 |
| Test | **All** | **209** | **23.65 ± 2.87** | **16.44** | **23.59** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8642 | 0.3682 | 0.1647 | 0.7784 | 0.4308 |
| 2 | 0.8196 | 0.5375 | 0.1390 | 0.7665 | 0.3810 |
| 3 | 0.8765 | 0.3486 | 0.2031 | 0.6886 | 0.3810 |
| 4 | 0.8690 | 0.4472 | 0.3228 | 0.3772 | 0.2464 |
| 5 | 0.8267 | 0.3904 | 0.1331 | 0.8084 | 0.4074 |
| **Mean** | **0.8512** | **0.4184** | **0.1926** | **0.6838** | **0.3693** |
| **±Std** | 0.0233 | 0.0681 | 0.0697 | 0.1583 | 0.0642 |

CrossAttn best val AUC per fold: Fold1=0.8642, Fold2=0.8196, Fold3=0.8765, Fold4=0.8690, Fold5=0.8267

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7042 | 0.3752 | 0.1431 | 0.7799 | 0.3235 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 67 | 0.6916 | 0.4655 | 0.1998 | 0.6866 | 0.3636 |
| F | 142 | 0.7280 | 0.3398 | 0.1164 | 0.8239 | 0.2857 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 152 | 35 |
| **True: Sarco**  | 11 | 11 |

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
