# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-28 21:29  |  5-Fold CV  |  Median best epoch: 3

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 277 | 239 | 86.3% | 38 | 13.7% |
| Train | F | 545 | 502 | 92.1% | 43 | 7.9% |
| Train | **All** | **822** | **741** | **90.1%** | **81** | **9.9%** |
| Test | M | 72 | 59 | 81.9% | 13 | 18.1% |
| Test | F | 133 | 124 | 93.2% | 9 | 6.8% |
| Test | **All** | **205** | **183** | **89.3%** | **22** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 277 | 60.19 ± 11.83 | 20.00 | 60.00 | 89.00 |
| Train | F | 545 | 55.20 ± 11.34 | 23.00 | 55.00 | 87.00 |
| Train | **All** | **822** | **56.88 ± 11.75** | **20.00** | **57.00** | **89.00** |
| Test | M | 72 | 59.21 ± 12.53 | 29.00 | 58.50 | 81.00 |
| Test | F | 133 | 54.96 ± 12.06 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **205** | **56.45 ± 12.40** | **23.00** | **57.00** | **83.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 277 | 24.10 ± 2.87 | 14.34 | 24.11 | 32.33 |
| Train | F | 545 | 23.04 ± 3.02 | 12.02 | 22.95 | 32.24 |
| Train | **All** | **822** | **23.40 ± 3.01** | **12.02** | **23.32** | **32.33** |
| Test | M | 72 | 24.20 ± 3.16 | 18.78 | 24.12 | 32.56 |
| Test | F | 133 | 22.93 ± 3.01 | 16.51 | 22.55 | 30.84 |
| Test | **All** | **205** | **23.38 ± 3.12** | **16.51** | **23.18** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8080 | 0.3141 | 0.1501 | 0.7455 | 0.4000 |
| 2 | 0.8188 | 0.4645 | 0.2389 | 0.8364 | 0.4706 |
| 3 | 0.9160 | 0.5675 | 0.1367 | 0.7683 | 0.4571 |
| 4 | 0.7179 | 0.2849 | 0.1925 | 0.7012 | 0.3288 |
| 5 | 0.8387 | 0.3066 | 0.1654 | 0.7012 | 0.3951 |
| **Mean** | **0.8199** | **0.3875** | **0.1767** | **0.7505** | **0.4103** |
| **±Std** | 0.0634 | 0.1102 | 0.0362 | 0.0501 | 0.0506 |

CrossAttn best val AUC per fold: Fold1=0.8080, Fold2=0.8188, Fold3=0.9160, Fold4=0.7179, Fold5=0.8387

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7737 | 0.2800 | 0.2381 | 0.5366 | 0.3066 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 72 | 0.7106 | 0.3534 | 0.2517 | 0.5972 | 0.4528 |
| F | 133 | 0.8271 | 0.3072 | 0.2307 | 0.5038 | 0.2143 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 89 | 94 |
| **True: Sarco**  | 1 | 21 |

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
