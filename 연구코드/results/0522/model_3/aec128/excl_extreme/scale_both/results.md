# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 20:55  |  5-Fold CV  |  Median best epoch: 16

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 293 | 251 | 85.7% | 42 | 14.3% |
| Train | F | 542 | 500 | 92.3% | 42 | 7.7% |
| Train | **All** | **835** | **751** | **89.9%** | **84** | **10.1%** |
| Test | M | 68 | 57 | 83.8% | 11 | 16.2% |
| Test | F | 141 | 130 | 92.2% | 11 | 7.8% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 293 | 60.09 ± 11.89 | 20.00 | 60.00 | 89.00 |
| Train | F | 542 | 55.35 ± 11.64 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **835** | **57.01 ± 11.95** | **14.00** | **57.00** | **91.00** |
| Test | M | 68 | 58.19 ± 12.49 | 29.00 | 58.50 | 80.00 |
| Test | F | 141 | 54.57 ± 11.94 | 18.00 | 54.00 | 86.00 |
| Test | **All** | **209** | **55.75 ± 12.24** | **18.00** | **56.00** | **86.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 293 | 24.02 ± 2.96 | 14.34 | 23.94 | 32.33 |
| Train | F | 542 | 22.96 ± 3.06 | 12.02 | 22.91 | 32.24 |
| Train | **All** | **835** | **23.33 ± 3.07** | **12.02** | **23.31** | **32.33** |
| Test | M | 68 | 24.52 ± 2.66 | 19.23 | 24.27 | 32.56 |
| Test | F | 141 | 23.23 ± 2.85 | 16.44 | 22.93 | 31.50 |
| Test | **All** | **209** | **23.65 ± 2.85** | **16.44** | **23.59** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8415 | 0.3669 | 0.1276 | 0.8443 | 0.4583 |
| 2 | 0.8310 | 0.4454 | 0.2092 | 0.6407 | 0.3182 |
| 3 | 0.8549 | 0.4971 | 0.2042 | 0.6527 | 0.3256 |
| 4 | 0.8584 | 0.3771 | 0.1739 | 0.7186 | 0.3733 |
| 5 | 0.8475 | 0.4219 | 0.1591 | 0.7725 | 0.3871 |
| **Mean** | **0.8466** | **0.4217** | **0.1748** | **0.7257** | **0.3725** |
| **±Std** | 0.0098 | 0.0474 | 0.0300 | 0.0760 | 0.0505 |

CrossAttn best val AUC per fold: Fold1=0.8415, Fold2=0.8310, Fold3=0.8549, Fold4=0.8584, Fold5=0.8475

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7149 | 0.3065 | 0.1742 | 0.7368 | 0.3373 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 68 | 0.6699 | 0.3448 | 0.2118 | 0.6618 | 0.3784 |
| F | 141 | 0.7643 | 0.3254 | 0.1560 | 0.7730 | 0.3043 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 140 | 47 |
| **True: Sarco**  | 8 | 14 |

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
