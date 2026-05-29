# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-28 21:30  |  5-Fold CV  |  Median best epoch: 3

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
| 1 | 0.8466 | 0.4850 | 0.1469 | 0.8606 | 0.5490 |
| 2 | 0.8146 | 0.4436 | 0.2020 | 0.7212 | 0.3611 |
| 3 | 0.9075 | 0.5391 | 0.0991 | 0.9146 | 0.6316 |
| 4 | 0.7196 | 0.3344 | 0.1883 | 0.7988 | 0.3529 |
| 5 | 0.7770 | 0.3382 | 0.1487 | 0.7805 | 0.4000 |
| **Mean** | **0.8131** | **0.4281** | **0.1570** | **0.8151** | **0.4589** |
| **±Std** | 0.0634 | 0.0808 | 0.0361 | 0.0667 | 0.1115 |

CrossAttn best val AUC per fold: Fold1=0.8466, Fold2=0.8146, Fold3=0.9075, Fold4=0.7196, Fold5=0.7770

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7851 | 0.2805 | 0.1682 | 0.7805 | 0.3478 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 72 | 0.7132 | 0.3472 | 0.2156 | 0.6806 | 0.3429 |
| F | 133 | 0.8306 | 0.3398 | 0.1425 | 0.8346 | 0.3529 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 148 | 35 |
| **True: Sarco**  | 10 | 12 |

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
