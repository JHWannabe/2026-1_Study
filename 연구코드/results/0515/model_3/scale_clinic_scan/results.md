# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 17:11  |  5-Fold CV  |  Median best epoch: 8

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 373 | 309 | 82.8% | 64 | 17.2% |
| Train | F | 666 | 616 | 92.5% | 50 | 7.5% |
| Train | **All** | **1039** | **925** | **89.0%** | **114** | **11.0%** |
| Test | M | 98 | 82 | 83.7% | 16 | 16.3% |
| Test | F | 162 | 150 | 92.6% | 12 | 7.4% |
| Test | **All** | **260** | **232** | **89.2%** | **28** | **10.8%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 373 | 59.36 ± 12.75 | 18.00 | 59.00 | 89.00 |
| Train | F | 666 | 55.63 ± 12.02 | 14.00 | 55.00 | 87.00 |
| Train | **All** | **1039** | **56.97 ± 12.41** | **14.00** | **57.00** | **89.00** |
| Test | M | 98 | 61.47 ± 11.63 | 20.00 | 62.50 | 84.00 |
| Test | F | 162 | 55.33 ± 12.79 | 11.00 | 55.50 | 91.00 |
| Test | **All** | **260** | **57.64 ± 12.72** | **11.00** | **58.00** | **91.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 373 | 24.19 ± 3.35 | 14.48 | 24.22 | 36.76 |
| Train | F | 666 | 23.06 ± 3.39 | 14.40 | 22.75 | 39.49 |
| Train | **All** | **1039** | **23.46 ± 3.42** | **14.40** | **23.29** | **39.49** |
| Test | M | 98 | 24.06 ± 2.92 | 17.03 | 24.12 | 31.51 |
| Test | F | 162 | 23.12 ± 3.29 | 16.44 | 22.66 | 34.61 |
| Test | **All** | **260** | **23.47 ± 3.19** | **16.44** | **23.39** | **34.61** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.5920 | 0.2248 | 0.2001 | 0.7115 | 0.1892 |
| 2 | 0.6926 | 0.1910 | 0.1836 | 0.7740 | 0.3380 |
| 3 | 0.6219 | 0.1698 | 0.1967 | 0.7500 | 0.2121 |
| 4 | 0.4987 | 0.1905 | 0.2558 | 0.6731 | 0.1707 |
| 5 | 0.6671 | 0.2267 | 0.1795 | 0.7681 | 0.3514 |
| **Mean** | **0.6144** | **0.2006** | **0.2031** | **0.7354** | **0.2523** |
| **±Std** | 0.0676 | 0.0220 | 0.0274 | 0.0380 | 0.0767 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8118 | 0.3479 | 0.2142 | 0.6971 | 0.3762 |
| 2 | 0.8823 | 0.4864 | 0.1380 | 0.7837 | 0.4578 |
| 3 | 0.8188 | 0.4586 | 0.1606 | 0.7837 | 0.4156 |
| 4 | 0.7511 | 0.3668 | 0.1682 | 0.7692 | 0.3684 |
| 5 | 0.8278 | 0.4300 | 0.1783 | 0.7198 | 0.3830 |
| **Mean** | **0.8183** | **0.4179** | **0.1718** | **0.7507** | **0.4002** |
| **±Std** | 0.0418 | 0.0529 | 0.0250 | 0.0356 | 0.0330 |

CrossAttn best val AUC per fold: Fold1=0.8118, Fold2=0.8823, Fold3=0.8188, Fold4=0.7511, Fold5=0.8278

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5914 | 0.1471 | 0.1951 | 0.7500 | 0.2529 |
| CrossAttn | 0.7606 | 0.2505 | 0.1634 | 0.7885 | 0.3678 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.5831 | 0.1982 | 0.2341 | 0.6735 | 0.3333 |
| F | 162 | 0.5494 | 0.1105 | 0.1716 | 0.7963 | 0.1538 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.7081 | 0.2704 | 0.2427 | 0.6122 | 0.3871 |
| F | 162 | 0.7278 | 0.2633 | 0.1155 | 0.8951 | 0.3200 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 184 | 48 |
| **True: Sarco**  | 17 | 11 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 189 | 43 |
| **True: Sarco**  | 12 | 16 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (LR & CrossAttn) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `training_curves.png` | Loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curves |
| `test_roc_by_sex.png` | Final test-set ROC curves by sex |
| `confusion_matrices.png` | Test-set confusion matrices (LR & CrossAttn) |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
