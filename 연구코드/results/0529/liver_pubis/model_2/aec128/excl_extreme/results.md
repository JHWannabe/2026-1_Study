# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-25 17:57  |  5-Fold CV  |  Median best epoch: 10

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 291 | 248 | 85.2% | 43 | 14.8% |
| Train | F | 545 | 502 | 92.1% | 43 | 7.9% |
| Train | **All** | **836** | **750** | **89.7%** | **86** | **10.3%** |
| Test | M | 72 | 61 | 84.7% | 11 | 15.3% |
| Test | F | 137 | 126 | 92.0% | 11 | 8.0% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 291 | 59.79 ± 12.01 | 20.00 | 60.00 | 89.00 |
| Train | F | 545 | 55.03 ± 11.81 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **836** | **56.69 ± 12.10** | **14.00** | **57.00** | **91.00** |
| Test | M | 72 | 59.79 ± 12.11 | 29.00 | 61.50 | 84.00 |
| Test | F | 137 | 55.86 ± 11.35 | 23.00 | 56.00 | 83.00 |
| Test | **All** | **209** | **57.22 ± 11.76** | **23.00** | **58.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 291 | 24.12 ± 2.77 | 17.33 | 24.11 | 32.33 |
| Train | F | 545 | 23.04 ± 2.97 | 16.00 | 22.95 | 32.24 |
| Train | **All** | **836** | **23.42 ± 2.95** | **16.00** | **23.33** | **32.33** |
| Test | M | 72 | 24.22 ± 3.49 | 14.34 | 24.01 | 32.56 |
| Test | F | 137 | 22.90 ± 3.24 | 12.02 | 22.60 | 30.84 |
| Test | **All** | **209** | **23.36 ± 3.39** | **12.02** | **23.17** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7819 | 0.3325 | 0.1644 | 0.7381 | 0.4054 |
| 2 | 0.7922 | 0.4045 | 0.1032 | 0.7485 | 0.3824 |
| 3 | 0.8824 | 0.4752 | 0.1234 | 0.6647 | 0.3778 |
| 4 | 0.8624 | 0.3620 | 0.1545 | 0.7305 | 0.4156 |
| 5 | 0.8506 | 0.5062 | 0.1592 | 0.7665 | 0.4348 |
| **Mean** | **0.8339** | **0.4161** | **0.1410** | **0.7297** | **0.4032** |
| **±Std** | 0.0397 | 0.0658 | 0.0237 | 0.0347 | 0.0211 |

CrossAttn best val AUC per fold: Fold1=0.7819, Fold2=0.7922, Fold3=0.8824, Fold4=0.8624, Fold5=0.8506

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8211 | 0.4092 | 0.1584 | 0.5933 | 0.2975 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 72 | 0.8510 | 0.5381 | 0.1729 | 0.6528 | 0.4186 |
| F | 137 | 0.7807 | 0.2786 | 0.1508 | 0.5620 | 0.2308 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 106 | 81 |
| **True: Sarco**  | 4 | 18 |

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
