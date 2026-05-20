# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:34  |  5-Fold CV  |  Median best epoch: 21

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 257 | 214 | 83.3% | 43 | 16.7% |
| Train | F | 363 | 330 | 90.9% | 33 | 9.1% |
| Train | **All** | **620** | **544** | **87.7%** | **76** | **12.3%** |
| Test | M | 57 | 51 | 89.5% | 6 | 10.5% |
| Test | F | 98 | 86 | 87.8% | 12 | 12.2% |
| Test | **All** | **155** | **137** | **88.4%** | **18** | **11.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 257 | 59.57 ± 11.72 | 28.00 | 60.00 | 89.00 |
| Train | F | 363 | 56.09 ± 12.40 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **620** | **57.53 ± 12.24** | **14.00** | **58.00** | **91.00** |
| Test | M | 57 | 58.04 ± 12.44 | 29.00 | 58.00 | 78.00 |
| Test | F | 98 | 56.66 ± 12.49 | 23.00 | 56.50 | 86.00 |
| Test | **All** | **155** | **57.17 ± 12.49** | **23.00** | **57.00** | **86.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 257 | 24.36 ± 3.09 | 14.34 | 24.28 | 32.67 |
| Train | F | 363 | 22.94 ± 3.14 | 16.00 | 22.77 | 34.61 |
| Train | **All** | **620** | **23.53 ± 3.19** | **14.34** | **23.37** | **34.61** |
| Test | M | 57 | 24.72 ± 2.87 | 19.22 | 24.11 | 32.56 |
| Test | F | 98 | 22.91 ± 3.63 | 12.02 | 22.58 | 34.20 |
| Test | **All** | **155** | **23.57 ± 3.48** | **12.02** | **23.24** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8031 | 0.3898 | 0.1745 | 0.7661 | 0.4528 |
| 2 | 0.9083 | 0.4689 | 0.1705 | 0.7339 | 0.4762 |
| 3 | 0.8734 | 0.5558 | 0.1899 | 0.6935 | 0.4242 |
| 4 | 0.8006 | 0.3871 | 0.2112 | 0.6694 | 0.3881 |
| 5 | 0.7587 | 0.4043 | 0.1946 | 0.7500 | 0.4364 |
| **Mean** | **0.8288** | **0.4412** | **0.1881** | **0.7226** | **0.4355** |
| **±Std** | 0.0542 | 0.0645 | 0.0147 | 0.0359 | 0.0294 |

CrossAttn best val AUC per fold: Fold1=0.8031, Fold2=0.9083, Fold3=0.8734, Fold4=0.8006, Fold5=0.7587

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8009 | 0.3988 | 0.2235 | 0.6516 | 0.3571 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 57 | 0.7320 | 0.3475 | 0.2829 | 0.5614 | 0.2857 |
| F | 98 | 0.8401 | 0.5410 | 0.1890 | 0.7041 | 0.4082 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 86 | 51 |
| **True: Sarco**  | 3 | 15 |

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
