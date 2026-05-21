# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 21:35  |  5-Fold CV  |  Median best epoch: 8

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
| 1 | 0.8543 | 0.2973 | 0.1951 | 0.6766 | 0.3415 |
| 2 | 0.8596 | 0.5896 | 0.1665 | 0.7006 | 0.3421 |
| 3 | 0.8435 | 0.4498 | 0.1713 | 0.7066 | 0.3797 |
| 4 | 0.8541 | 0.3898 | 0.2144 | 0.6347 | 0.3297 |
| 5 | 0.8604 | 0.4643 | 0.1168 | 0.8024 | 0.3529 |
| **Mean** | **0.8544** | **0.4382** | **0.1728** | **0.7042** | **0.3492** |
| **±Std** | 0.0060 | 0.0959 | 0.0329 | 0.0552 | 0.0170 |

CrossAttn best val AUC per fold: Fold1=0.8543, Fold2=0.8596, Fold3=0.8435, Fold4=0.8541, Fold5=0.8604

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7151 | 0.3089 | 0.1729 | 0.7225 | 0.3095 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 67 | 0.6981 | 0.3704 | 0.2089 | 0.6716 | 0.3889 |
| F | 142 | 0.7224 | 0.2790 | 0.1559 | 0.7465 | 0.2500 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 138 | 49 |
| **True: Sarco**  | 9 | 13 |

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
