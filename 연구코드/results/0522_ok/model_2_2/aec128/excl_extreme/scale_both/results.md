# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 12:35  |  5-Fold CV  |  Median best epoch: 29

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 256 | 217 | 84.8% | 39 | 15.2% |
| Train | F | 361 | 325 | 90.0% | 36 | 10.0% |
| Train | **All** | **617** | **542** | **87.8%** | **75** | **12.2%** |
| Test | M | 63 | 55 | 87.3% | 8 | 12.7% |
| Test | F | 91 | 82 | 90.1% | 9 | 9.9% |
| Test | **All** | **154** | **137** | **89.0%** | **17** | **11.0%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 256 | 59.57 ± 11.76 | 23.00 | 59.00 | 85.00 |
| Train | F | 361 | 56.66 ± 12.28 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **617** | **57.87 ± 12.15** | **14.00** | **58.00** | **91.00** |
| Test | M | 63 | 59.02 ± 12.66 | 28.00 | 61.00 | 89.00 |
| Test | F | 91 | 55.78 ± 12.86 | 24.00 | 56.00 | 86.00 |
| Test | **All** | **154** | **57.10 ± 12.87** | **24.00** | **57.50** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 256 | 24.53 ± 2.97 | 14.34 | 24.45 | 32.67 |
| Train | F | 361 | 22.94 ± 3.38 | 12.02 | 22.77 | 34.61 |
| Train | **All** | **617** | **23.60 ± 3.31** | **12.02** | **23.51** | **34.61** |
| Test | M | 63 | 24.50 ± 3.21 | 17.33 | 24.12 | 32.33 |
| Test | F | 91 | 22.99 ± 3.31 | 16.00 | 22.76 | 34.20 |
| Test | **All** | **154** | **23.61 ± 3.35** | **16.00** | **23.35** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8257 | 0.4684 | 0.2290 | 0.6694 | 0.4058 |
| 2 | 0.8606 | 0.4395 | 0.1857 | 0.7097 | 0.4194 |
| 3 | 0.8883 | 0.5211 | 0.1723 | 0.7317 | 0.4590 |
| 4 | 0.8340 | 0.5053 | 0.1536 | 0.7724 | 0.4400 |
| 5 | 0.6938 | 0.2721 | 0.2474 | 0.6016 | 0.3099 |
| **Mean** | **0.8205** | **0.4413** | **0.1976** | **0.6969** | **0.4068** |
| **±Std** | 0.0670 | 0.0893 | 0.0352 | 0.0581 | 0.0517 |

CrossAttn best val AUC per fold: Fold1=0.8257, Fold2=0.8606, Fold3=0.8883, Fold4=0.8340, Fold5=0.6938

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8652 | 0.4839 | 0.1417 | 0.8052 | 0.4828 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 63 | 0.8977 | 0.5995 | 0.1718 | 0.7619 | 0.4828 |
| F | 91 | 0.8401 | 0.4098 | 0.1209 | 0.8352 | 0.4828 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 110 | 27 |
| **True: Sarco**  | 3 | 14 |

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
