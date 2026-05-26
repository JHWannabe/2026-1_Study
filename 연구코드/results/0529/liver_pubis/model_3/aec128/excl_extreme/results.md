# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-25 17:57  |  5-Fold CV  |  Median best epoch: 13

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
| 1 | 0.7756 | 0.4094 | 0.2036 | 0.7440 | 0.3944 |
| 2 | 0.7933 | 0.4018 | 0.1807 | 0.8084 | 0.4483 |
| 3 | 0.8949 | 0.5036 | 0.1171 | 0.7665 | 0.4658 |
| 4 | 0.8427 | 0.3068 | 0.1459 | 0.7904 | 0.4615 |
| 5 | 0.8486 | 0.4774 | 0.1435 | 0.8144 | 0.4746 |
| **Mean** | **0.8310** | **0.4198** | **0.1582** | **0.7847** | **0.4489** |
| **±Std** | 0.0425 | 0.0686 | 0.0304 | 0.0263 | 0.0286 |

CrossAttn best val AUC per fold: Fold1=0.7756, Fold2=0.7933, Fold3=0.8949, Fold4=0.8427, Fold5=0.8486

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8119 | 0.4300 | 0.1805 | 0.6890 | 0.3434 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 72 | 0.8420 | 0.5744 | 0.1888 | 0.7083 | 0.4878 |
| F | 137 | 0.7641 | 0.2981 | 0.1761 | 0.6788 | 0.2414 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 127 | 60 |
| **True: Sarco**  | 5 | 17 |

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
