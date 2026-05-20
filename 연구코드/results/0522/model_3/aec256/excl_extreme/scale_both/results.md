# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 13:01  |  5-Fold CV  |  Median best epoch: 6

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
| 1 | 0.7961 | 0.4200 | 0.2118 | 0.6555 | 0.3492 |
| 2 | 0.9272 | 0.5935 | 0.1010 | 0.8824 | 0.6500 |
| 3 | 0.8524 | 0.5156 | 0.1632 | 0.7479 | 0.4643 |
| 4 | 0.6796 | 0.2797 | 0.2195 | 0.7143 | 0.2273 |
| 5 | 0.8689 | 0.3851 | 0.1496 | 0.7881 | 0.5098 |
| **Mean** | **0.8248** | **0.4388** | **0.1690** | **0.7576** | **0.4401** |
| **±Std** | 0.0838 | 0.1081 | 0.0434 | 0.0760 | 0.1436 |

CrossAttn best val AUC per fold: Fold1=0.7961, Fold2=0.9272, Fold3=0.8524, Fold4=0.6796, Fold5=0.8689

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7925 | 0.4475 | 0.2010 | 0.6554 | 0.3704 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 57 | 0.7809 | 0.5513 | 0.1906 | 0.7018 | 0.4848 |
| F | 91 | 0.7806 | 0.3167 | 0.2075 | 0.6264 | 0.2917 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 82 | 49 |
| **True: Sarco**  | 2 | 15 |

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
