# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:34  |  5-Fold CV  |  Median best epoch: 8

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 252 | 215 | 85.3% | 37 | 14.7% |
| Train | F | 368 | 334 | 90.8% | 34 | 9.2% |
| Train | **All** | **620** | **549** | **88.5%** | **71** | **11.5%** |
| Test | M | 59 | 52 | 88.1% | 7 | 11.9% |
| Test | F | 96 | 84 | 87.5% | 12 | 12.5% |
| Test | **All** | **155** | **136** | **87.7%** | **19** | **12.3%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 252 | 60.13 ± 11.46 | 28.00 | 60.00 | 89.00 |
| Train | F | 368 | 56.15 ± 11.95 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **620** | **57.77 ± 11.91** | **14.00** | **58.00** | **91.00** |
| Test | M | 59 | 58.49 ± 12.91 | 23.00 | 59.00 | 83.00 |
| Test | F | 96 | 56.49 ± 12.82 | 23.00 | 56.00 | 86.00 |
| Test | **All** | **155** | **57.25 ± 12.90** | **23.00** | **58.00** | **86.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 252 | 24.22 ± 2.95 | 16.39 | 24.22 | 32.59 |
| Train | F | 368 | 22.81 ± 2.96 | 16.02 | 22.77 | 31.50 |
| Train | **All** | **620** | **23.39 ± 3.03** | **16.02** | **23.31** | **32.59** |
| Test | M | 59 | 24.49 ± 2.86 | 17.65 | 24.06 | 32.56 |
| Test | F | 96 | 22.85 ± 3.37 | 12.02 | 22.66 | 31.14 |
| Test | **All** | **155** | **23.47 ± 3.29** | **12.02** | **23.24** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7513 | 0.3859 | 0.2301 | 0.6210 | 0.2985 |
| 2 | 0.8091 | 0.3676 | 0.2357 | 0.6048 | 0.3467 |
| 3 | 0.8675 | 0.4669 | 0.1122 | 0.8306 | 0.5333 |
| 4 | 0.8078 | 0.3063 | 0.1614 | 0.7742 | 0.4400 |
| 5 | 0.8324 | 0.4410 | 0.1185 | 0.7984 | 0.4681 |
| **Mean** | **0.8136** | **0.3936** | **0.1716** | **0.7258** | **0.4173** |
| **±Std** | 0.0379 | 0.0565 | 0.0529 | 0.0940 | 0.0844 |

CrossAttn best val AUC per fold: Fold1=0.7513, Fold2=0.8091, Fold3=0.8675, Fold4=0.8078, Fold5=0.8324

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8603 | 0.4688 | 0.1614 | 0.7355 | 0.4384 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 59 | 0.8599 | 0.4153 | 0.1568 | 0.7797 | 0.4800 |
| F | 96 | 0.8661 | 0.5935 | 0.1643 | 0.7083 | 0.4167 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 98 | 38 |
| **True: Sarco**  | 3 | 16 |

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
