# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-28 22:06  |  5-Fold CV  |  Median best epoch: 12

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 324 | 274 | 84.6% | 50 | 15.4% |
| Train | F | 590 | 544 | 92.2% | 46 | 7.8% |
| Train | **All** | **914** | **818** | **89.5%** | **96** | **10.5%** |
| Test | M | 82 | 69 | 84.1% | 13 | 15.9% |
| Test | F | 147 | 136 | 92.5% | 11 | 7.5% |
| Test | **All** | **229** | **205** | **89.5%** | **24** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 324 | 59.79 ± 12.13 | 20.00 | 60.00 | 89.00 |
| Train | F | 590 | 55.39 ± 11.41 | 23.00 | 55.00 | 87.00 |
| Train | **All** | **914** | **56.95 ± 11.86** | **20.00** | **57.00** | **89.00** |
| Test | M | 82 | 59.71 ± 12.20 | 29.00 | 60.00 | 81.00 |
| Test | F | 147 | 55.03 ± 12.43 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **229** | **56.71 ± 12.55** | **23.00** | **57.00** | **83.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 324 | 24.36 ± 3.00 | 14.34 | 24.26 | 32.67 |
| Train | F | 590 | 23.15 ± 3.19 | 12.02 | 22.95 | 34.61 |
| Train | **All** | **914** | **23.58 ± 3.17** | **12.02** | **23.44** | **34.61** |
| Test | M | 82 | 24.32 ± 3.06 | 18.78 | 24.17 | 32.56 |
| Test | F | 147 | 23.08 ± 3.44 | 15.84 | 22.60 | 32.48 |
| Test | **All** | **229** | **23.52 ± 3.36** | **15.84** | **23.41** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8377 | 0.3550 | 0.2054 | 0.7049 | 0.4000 |
| 2 | 0.8540 | 0.3920 | 0.1470 | 0.8743 | 0.5660 |
| 3 | 0.7908 | 0.2550 | 0.1729 | 0.7650 | 0.4267 |
| 4 | 0.7843 | 0.3394 | 0.2594 | 0.7322 | 0.3797 |
| 5 | 0.7933 | 0.3965 | 0.2542 | 0.8462 | 0.4815 |
| **Mean** | **0.8120** | **0.3476** | **0.2078** | **0.7845** | **0.4508** |
| **±Std** | 0.0282 | 0.0511 | 0.0441 | 0.0653 | 0.0670 |

CrossAttn best val AUC per fold: Fold1=0.8377, Fold2=0.8540, Fold3=0.7908, Fold4=0.7843, Fold5=0.7933

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8213 | 0.3396 | 0.2167 | 0.7162 | 0.3689 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 82 | 0.7380 | 0.3682 | 0.2851 | 0.6341 | 0.4231 |
| F | 147 | 0.8516 | 0.3581 | 0.1785 | 0.7619 | 0.3137 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 145 | 60 |
| **True: Sarco**  | 5 | 19 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.8213 | 0.7412 | 0.8902 |
| AUPRC | 0.3396 | 0.2048 | 0.5371 |
| Brier | 0.2167 | 0.1846 | 0.2493 |
| Accuracy | 0.7162 | 0.6594 | 0.7773 |
| F1 | 0.3689 | 0.2453 | 0.4894 |

---

## 5. Figures

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
