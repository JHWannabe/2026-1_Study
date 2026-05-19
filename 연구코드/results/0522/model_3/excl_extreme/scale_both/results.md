# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-19 13:09  |  5-Fold CV  |  Median best epoch: 8

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 214 | 179 | 83.6% | 35 | 16.4% |
| Train | F | 527 | 488 | 92.6% | 39 | 7.4% |
| Train | **All** | **741** | **667** | **90.0%** | **74** | **10.0%** |
| Test | M | 55 | 46 | 83.6% | 9 | 16.4% |
| Test | F | 130 | 121 | 93.1% | 9 | 6.9% |
| Test | **All** | **185** | **167** | **90.3%** | **18** | **9.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 214 | 60.95 ± 11.58 | 28.00 | 62.00 | 89.00 |
| Train | F | 527 | 55.22 ± 11.46 | 24.00 | 55.00 | 87.00 |
| Train | **All** | **741** | **56.88 ± 11.78** | **24.00** | **57.00** | **89.00** |
| Test | M | 55 | 63.82 ± 11.36 | 31.00 | 66.00 | 83.00 |
| Test | F | 130 | 57.32 ± 11.54 | 29.00 | 58.00 | 91.00 |
| Test | **All** | **185** | **59.25 ± 11.86** | **29.00** | **59.00** | **91.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 214 | 24.04 ± 2.74 | 15.22 | 24.33 | 31.94 |
| Train | F | 527 | 23.07 ± 2.94 | 15.63 | 22.92 | 31.93 |
| Train | **All** | **741** | **23.35 ± 2.92** | **15.22** | **23.31** | **31.94** |
| Test | M | 55 | 23.35 ± 2.07 | 19.29 | 23.28 | 27.62 |
| Test | F | 130 | 23.09 ± 2.70 | 16.80 | 22.98 | 30.05 |
| Test | **All** | **185** | **23.17 ± 2.53** | **16.80** | **23.02** | **30.05** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7930 | 0.4392 | 0.1742 | 0.8792 | 0.4706 |
| 2 | 0.7628 | 0.3610 | 0.1675 | 0.7297 | 0.2857 |
| 3 | 0.8065 | 0.3971 | 0.1190 | 0.8514 | 0.3889 |
| 4 | 0.8622 | 0.4097 | 0.2349 | 0.5811 | 0.3111 |
| 5 | 0.8426 | 0.3907 | 0.1862 | 0.7297 | 0.3548 |
| **Mean** | **0.8134** | **0.3996** | **0.1764** | **0.7542** | **0.3622** |
| **±Std** | 0.0354 | 0.0255 | 0.0371 | 0.1060 | 0.0648 |

CrossAttn best val AUC per fold: Fold1=0.7930, Fold2=0.7628, Fold3=0.8065, Fold4=0.8622, Fold5=0.8426

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7189 | 0.2721 | 0.2222 | 0.6595 | 0.2759 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 55 | 0.6667 | 0.3947 | 0.3443 | 0.4909 | 0.3333 |
| F | 130 | 0.7043 | 0.2065 | 0.1705 | 0.7308 | 0.2222 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 110 | 57 |
| **True: Sarco**  | 6 | 12 |

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
