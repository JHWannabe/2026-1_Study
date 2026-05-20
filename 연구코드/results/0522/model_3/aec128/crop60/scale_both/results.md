# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:30  |  5-Fold CV  |  Median best epoch: 23

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 332 | 276 | 83.1% | 56 | 16.9% |
| Train | F | 597 | 554 | 92.8% | 43 | 7.2% |
| Train | **All** | **929** | **830** | **89.3%** | **99** | **10.7%** |
| Test | M | 87 | 77 | 88.5% | 10 | 11.5% |
| Test | F | 146 | 131 | 89.7% | 15 | 10.3% |
| Test | **All** | **233** | **208** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 332 | 60.33 ± 11.99 | 20.00 | 60.00 | 89.00 |
| Train | F | 597 | 55.52 ± 11.67 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **929** | **57.24 ± 12.01** | **14.00** | **57.00** | **91.00** |
| Test | M | 87 | 56.93 ± 12.53 | 22.00 | 58.00 | 84.00 |
| Test | F | 146 | 54.41 ± 12.53 | 23.00 | 54.00 | 87.00 |
| Test | **All** | **233** | **55.35 ± 12.59** | **22.00** | **57.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 332 | 24.41 ± 3.03 | 16.39 | 24.39 | 32.67 |
| Train | F | 597 | 23.19 ± 3.21 | 12.02 | 22.96 | 34.61 |
| Train | **All** | **929** | **23.63 ± 3.20** | **12.02** | **23.56** | **34.61** |
| Test | M | 87 | 24.22 ± 3.01 | 14.34 | 23.94 | 32.56 |
| Test | F | 146 | 22.87 ± 3.38 | 15.84 | 22.44 | 34.20 |
| Test | **All** | **233** | **23.38 ± 3.31** | **14.34** | **23.31** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8726 | 0.4549 | 0.1715 | 0.7688 | 0.4267 |
| 2 | 0.8605 | 0.4581 | 0.1523 | 0.7581 | 0.4304 |
| 3 | 0.8494 | 0.3414 | 0.2019 | 0.7419 | 0.4146 |
| 4 | 0.8256 | 0.4810 | 0.1851 | 0.6828 | 0.3516 |
| 5 | 0.8285 | 0.3267 | 0.1321 | 0.8270 | 0.4483 |
| **Mean** | **0.8473** | **0.4124** | **0.1686** | **0.7557** | **0.4143** |
| **±Std** | 0.0181 | 0.0648 | 0.0245 | 0.0464 | 0.0331 |

CrossAttn best val AUC per fold: Fold1=0.8726, Fold2=0.8605, Fold3=0.8494, Fold4=0.8256, Fold5=0.8285

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.6663 | 0.2402 | 0.2143 | 0.6781 | 0.2424 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 87 | 0.5169 | 0.2197 | 0.2994 | 0.5747 | 0.1778 |
| F | 146 | 0.7796 | 0.4106 | 0.1636 | 0.7397 | 0.2963 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 146 | 62 |
| **True: Sarco**  | 13 | 12 |

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
