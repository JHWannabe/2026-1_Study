# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-22 00:54  |  5-Fold CV  |  Median best epoch: 4

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 289 | 246 | 85.1% | 43 | 14.9% |
| Train | F | 546 | 503 | 92.1% | 43 | 7.9% |
| Train | **All** | **835** | **749** | **89.7%** | **86** | **10.3%** |
| Test | M | 71 | 60 | 84.5% | 11 | 15.5% |
| Test | F | 138 | 127 | 92.0% | 11 | 8.0% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 289 | 59.77 ± 12.01 | 20.00 | 60.00 | 89.00 |
| Train | F | 546 | 54.97 ± 11.85 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **835** | **56.63 ± 12.12** | **14.00** | **57.00** | **91.00** |
| Test | M | 71 | 59.92 ± 12.32 | 29.00 | 61.00 | 84.00 |
| Test | F | 138 | 55.92 ± 11.36 | 23.00 | 55.50 | 83.00 |
| Test | **All** | **209** | **57.28 ± 11.85** | **23.00** | **58.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 289 | 24.10 ± 2.75 | 17.33 | 24.12 | 32.33 |
| Train | F | 546 | 23.05 ± 2.97 | 16.00 | 22.95 | 32.24 |
| Train | **All** | **835** | **23.41 ± 2.94** | **16.00** | **23.33** | **32.33** |
| Test | M | 71 | 24.15 ± 3.49 | 14.34 | 23.88 | 32.56 |
| Test | F | 138 | 22.96 ± 3.29 | 12.02 | 22.64 | 30.84 |
| Test | **All** | **209** | **23.36 ± 3.41** | **12.02** | **23.17** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8408 | 0.4123 | 0.1927 | 0.7605 | 0.4444 |
| 2 | 0.8863 | 0.4056 | 0.1984 | 0.7784 | 0.4638 |
| 3 | 0.8125 | 0.3808 | 0.1872 | 0.8263 | 0.4528 |
| 4 | 0.7694 | 0.3692 | 0.1757 | 0.6048 | 0.3125 |
| 5 | 0.7755 | 0.3183 | 0.1819 | 0.6168 | 0.3333 |
| **Mean** | **0.8169** | **0.3772** | **0.1872** | **0.7174** | **0.4014** |
| **±Std** | 0.0433 | 0.0334 | 0.0079 | 0.0897 | 0.0647 |

CrossAttn best val AUC per fold: Fold1=0.8408, Fold2=0.8863, Fold3=0.8125, Fold4=0.7694, Fold5=0.7755

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8194 | 0.4467 | 0.1996 | 0.6986 | 0.3368 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 71 | 0.8485 | 0.5928 | 0.1894 | 0.7324 | 0.4571 |
| F | 138 | 0.8003 | 0.3454 | 0.2048 | 0.6812 | 0.2667 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 130 | 57 |
| **True: Sarco**  | 6 | 16 |

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
