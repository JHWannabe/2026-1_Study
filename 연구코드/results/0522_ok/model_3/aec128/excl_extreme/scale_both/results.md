# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:34  |  5-Fold CV  |  Median best epoch: 9

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 247 | 215 | 87.0% | 32 | 13.0% |
| Train | F | 347 | 311 | 89.6% | 36 | 10.4% |
| Train | **All** | **594** | **526** | **88.6%** | **68** | **11.4%** |
| Test | M | 57 | 47 | 82.5% | 10 | 17.5% |
| Test | F | 91 | 84 | 92.3% | 7 | 7.7% |
| Test | **All** | **148** | **131** | **88.5%** | **17** | **11.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 247 | 59.72 ± 11.48 | 23.00 | 60.00 | 83.00 |
| Train | F | 347 | 56.12 ± 12.01 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **594** | **57.62 ± 11.93** | **14.00** | **58.00** | **91.00** |
| Test | M | 57 | 59.53 ± 13.22 | 28.00 | 59.00 | 89.00 |
| Test | F | 91 | 56.84 ± 12.94 | 24.00 | 56.00 | 87.00 |
| Test | **All** | **148** | **57.87 ± 13.11** | **24.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 247 | 24.24 ± 2.84 | 16.39 | 24.28 | 32.33 |
| Train | F | 347 | 22.72 ± 3.03 | 12.02 | 22.67 | 31.50 |
| Train | **All** | **594** | **23.35 ± 3.05** | **12.02** | **23.31** | **32.33** |
| Test | M | 57 | 24.19 ± 3.18 | 17.57 | 23.67 | 32.56 |
| Test | F | 91 | 23.13 ± 3.01 | 17.25 | 22.78 | 31.14 |
| Test | **All** | **148** | **23.54 ± 3.12** | **17.25** | **23.20** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7765 | 0.4594 | 0.2155 | 0.6723 | 0.3390 |
| 2 | 0.9014 | 0.5706 | 0.1264 | 0.8151 | 0.5217 |
| 3 | 0.8646 | 0.5254 | 0.1585 | 0.7227 | 0.4000 |
| 4 | 0.6898 | 0.2116 | 0.1797 | 0.7815 | 0.3158 |
| 5 | 0.8601 | 0.4202 | 0.2186 | 0.6695 | 0.4000 |
| **Mean** | **0.8185** | **0.4374** | **0.1798** | **0.7322** | **0.3953** |
| **±Std** | 0.0762 | 0.1243 | 0.0349 | 0.0582 | 0.0714 |

CrossAttn best val AUC per fold: Fold1=0.7765, Fold2=0.9014, Fold3=0.8646, Fold4=0.6898, Fold5=0.8601

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7881 | 0.3681 | 0.1826 | 0.6959 | 0.3662 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 57 | 0.8021 | 0.5263 | 0.1945 | 0.7193 | 0.5000 |
| F | 91 | 0.7466 | 0.1647 | 0.1751 | 0.6813 | 0.2564 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 90 | 41 |
| **True: Sarco**  | 4 | 13 |

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
