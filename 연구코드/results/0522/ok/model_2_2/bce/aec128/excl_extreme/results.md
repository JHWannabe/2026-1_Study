# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 23:05  |  5-Fold CV  |  Median best epoch: 20

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
| 1 | 0.7529 | 0.3136 | 0.1953 | 0.6613 | 0.3824 |
| 2 | 0.8183 | 0.4797 | 0.1639 | 0.7258 | 0.4333 |
| 3 | 0.8660 | 0.4783 | 0.1825 | 0.8455 | 0.5778 |
| 4 | 0.8216 | 0.4578 | 0.1904 | 0.8537 | 0.5263 |
| 5 | 0.7296 | 0.4001 | 0.2158 | 0.7805 | 0.4490 |
| **Mean** | **0.7977** | **0.4259** | **0.1896** | **0.7734** | **0.4738** |
| **±Std** | 0.0496 | 0.0631 | 0.0169 | 0.0728 | 0.0695 |

CrossAttn best val AUC per fold: Fold1=0.7529, Fold2=0.8183, Fold3=0.8660, Fold4=0.8216, Fold5=0.7296

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7965 | 0.2934 | 0.2056 | 0.7727 | 0.4262 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 63 | 0.8591 | 0.4710 | 0.2330 | 0.7302 | 0.4848 |
| F | 91 | 0.7507 | 0.2172 | 0.1867 | 0.8022 | 0.3571 |

---

## 3. Confusion Matrix (Test Set)

|   | Pred: Normal | Pred: Sarco |
|---|-------------:|------------:|
| **True: Normal** | 106 | 31 |
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
