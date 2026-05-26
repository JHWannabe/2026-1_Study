# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-25 18:24  |  5-Fold CV  |  Median best epoch: 5

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 298 | 250 | 83.9% | 48 | 16.1% |
| Train | F | 538 | 496 | 92.2% | 42 | 7.8% |
| Train | **All** | **836** | **746** | **89.2%** | **90** | **10.8%** |
| Test | M | 77 | 66 | 85.7% | 11 | 14.3% |
| Test | F | 132 | 121 | 91.7% | 11 | 8.3% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 298 | 59.85 ± 12.04 | 20.00 | 60.00 | 86.00 |
| Train | F | 538 | 55.15 ± 11.89 | 18.00 | 55.00 | 91.00 |
| Train | **All** | **836** | **56.83 ± 12.15** | **18.00** | **57.00** | **91.00** |
| Test | M | 77 | 59.21 ± 11.70 | 32.00 | 60.00 | 84.00 |
| Test | F | 132 | 55.37 ± 11.84 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **209** | **56.78 ± 11.93** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 298 | 24.44 ± 2.89 | 17.51 | 24.42 | 32.67 |
| Train | F | 538 | 23.12 ± 3.16 | 16.00 | 23.01 | 34.61 |
| Train | **All** | **836** | **23.59 ± 3.13** | **16.00** | **23.51** | **34.61** |
| Test | M | 77 | 24.52 ± 3.39 | 14.34 | 24.28 | 32.56 |
| Test | F | 132 | 22.99 ± 3.54 | 12.02 | 22.64 | 32.48 |
| Test | **All** | **209** | **23.55 ± 3.57** | **12.02** | **23.52** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8196 | 0.4113 | 0.1434 | 0.8452 | 0.5185 |
| 2 | 0.7174 | 0.2591 | 0.1968 | 0.7305 | 0.3478 |
| 3 | 0.9109 | 0.5488 | 0.1181 | 0.7784 | 0.4932 |
| 4 | 0.7983 | 0.3127 | 0.1559 | 0.7485 | 0.4167 |
| 5 | 0.8195 | 0.3278 | 0.1841 | 0.7305 | 0.4156 |
| **Mean** | **0.8131** | **0.3720** | **0.1597** | **0.7667** | **0.4383** |
| **±Std** | 0.0617 | 0.1010 | 0.0282 | 0.0430 | 0.0610 |

CrossAttn best val AUC per fold: Fold1=0.8196, Fold2=0.7174, Fold3=0.9109, Fold4=0.7983, Fold5=0.8195

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7931 | 0.3778 | 0.1453 | 0.7656 | 0.3467 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 77 | 0.8154 | 0.4293 | 0.1669 | 0.7662 | 0.4706 |
| F | 132 | 0.7513 | 0.3444 | 0.1326 | 0.7652 | 0.2439 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 147 | 40 |
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
