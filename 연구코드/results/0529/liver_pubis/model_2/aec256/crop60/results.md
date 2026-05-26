# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-25 18:39  |  5-Fold CV  |  Median best epoch: 14

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 335 | 283 | 84.5% | 52 | 15.5% |
| Train | F | 595 | 548 | 92.1% | 47 | 7.9% |
| Train | **All** | **930** | **831** | **89.4%** | **99** | **10.6%** |
| Test | M | 85 | 71 | 83.5% | 14 | 16.5% |
| Test | F | 148 | 137 | 92.6% | 11 | 7.4% |
| Test | **All** | **233** | **208** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 335 | 59.78 ± 12.13 | 20.00 | 60.00 | 89.00 |
| Train | F | 595 | 55.26 ± 11.90 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **930** | **56.89 ± 12.18** | **14.00** | **57.00** | **91.00** |
| Test | M | 85 | 59.08 ± 12.34 | 29.00 | 60.00 | 84.00 |
| Test | F | 148 | 55.47 ± 11.64 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **233** | **56.79 ± 12.03** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 335 | 24.39 ± 2.93 | 17.33 | 24.24 | 32.67 |
| Train | F | 595 | 23.15 ± 3.17 | 16.00 | 22.96 | 34.61 |
| Train | **All** | **930** | **23.60 ± 3.15** | **16.00** | **23.45** | **34.61** |
| Test | M | 85 | 24.34 ± 3.40 | 14.34 | 24.26 | 32.56 |
| Test | F | 148 | 23.06 ± 3.53 | 12.02 | 22.64 | 32.48 |
| Test | **All** | **233** | **23.52 ± 3.53** | **12.02** | **23.41** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8352 | 0.4404 | 0.1367 | 0.7688 | 0.4267 |
| 2 | 0.8127 | 0.4123 | 0.2137 | 0.7688 | 0.4267 |
| 3 | 0.8307 | 0.4884 | 0.1510 | 0.7957 | 0.4571 |
| 4 | 0.8792 | 0.4658 | 0.1803 | 0.8602 | 0.5357 |
| 5 | 0.7718 | 0.3056 | 0.2074 | 0.7957 | 0.4412 |
| **Mean** | **0.8259** | **0.4225** | **0.1778** | **0.7978** | **0.4575** |
| **±Std** | 0.0348 | 0.0637 | 0.0302 | 0.0334 | 0.0407 |

CrossAttn best val AUC per fold: Fold1=0.8352, Fold2=0.8127, Fold3=0.8307, Fold4=0.8792, Fold5=0.7718

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7919 | 0.3725 | 0.2007 | 0.7597 | 0.3778 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 85 | 0.7978 | 0.4673 | 0.2503 | 0.7059 | 0.4898 |
| F | 148 | 0.7525 | 0.3007 | 0.1722 | 0.7905 | 0.2439 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 160 | 48 |
| **True: Sarco**  | 8 | 17 |

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
