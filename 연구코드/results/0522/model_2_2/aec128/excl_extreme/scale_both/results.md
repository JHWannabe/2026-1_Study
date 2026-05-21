# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 20:54  |  5-Fold CV  |  Median best epoch: 7

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 308 | 264 | 85.7% | 44 | 14.3% |
| Train | F | 527 | 487 | 92.4% | 40 | 7.6% |
| Train | **All** | **835** | **751** | **89.9%** | **84** | **10.1%** |
| Test | M | 72 | 61 | 84.7% | 11 | 15.3% |
| Test | F | 137 | 127 | 92.7% | 10 | 7.3% |
| Test | **All** | **209** | **188** | **90.0%** | **21** | **10.0%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 308 | 59.75 ± 11.85 | 20.00 | 60.00 | 89.00 |
| Train | F | 527 | 55.43 ± 11.97 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **835** | **57.02 ± 12.11** | **14.00** | **58.00** | **91.00** |
| Test | M | 72 | 57.39 ± 12.62 | 22.00 | 57.50 | 80.00 |
| Test | F | 137 | 55.06 ± 12.04 | 18.00 | 55.00 | 86.00 |
| Test | **All** | **209** | **55.86 ± 12.29** | **18.00** | **56.00** | **86.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 308 | 24.26 ± 3.01 | 14.34 | 24.14 | 32.59 |
| Train | F | 527 | 23.04 ± 3.29 | 12.02 | 22.91 | 34.20 |
| Train | **All** | **835** | **23.49 ± 3.24** | **12.02** | **23.43** | **34.20** |
| Test | M | 72 | 24.80 ± 3.03 | 18.44 | 24.44 | 32.67 |
| Test | F | 137 | 23.41 ± 3.19 | 16.44 | 22.93 | 34.61 |
| Test | **All** | **209** | **23.89 ± 3.21** | **16.44** | **23.66** | **34.61** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8675 | 0.4229 | 0.1324 | 0.8084 | 0.4667 |
| 2 | 0.8957 | 0.5593 | 0.1901 | 0.6766 | 0.3721 |
| 3 | 0.8698 | 0.3325 | 0.1132 | 0.8383 | 0.4000 |
| 4 | 0.8224 | 0.4297 | 0.1797 | 0.7186 | 0.3380 |
| 5 | 0.8741 | 0.5794 | 0.1263 | 0.7964 | 0.4516 |
| **Mean** | **0.8659** | **0.4648** | **0.1483** | **0.7677** | **0.4057** |
| **±Std** | 0.0240 | 0.0923 | 0.0307 | 0.0603 | 0.0481 |

CrossAttn best val AUC per fold: Fold1=0.8675, Fold2=0.8957, Fold3=0.8698, Fold4=0.8224, Fold5=0.8741

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.6925 | 0.2091 | 0.2056 | 0.6507 | 0.2474 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 72 | 0.6557 | 0.2590 | 0.2438 | 0.6250 | 0.3077 |
| F | 137 | 0.7252 | 0.2404 | 0.1856 | 0.6642 | 0.2069 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 124 | 64 |
| **True: Sarco**  | 9 | 12 |

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
