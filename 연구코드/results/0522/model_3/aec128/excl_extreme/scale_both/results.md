# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:49  |  5-Fold CV  |  Median best epoch: 16

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 286 | 240 | 83.9% | 46 | 16.1% |
| Train | F | 549 | 510 | 92.9% | 39 | 7.1% |
| Train | **All** | **835** | **750** | **89.8%** | **85** | **10.2%** |
| Test | M | 74 | 66 | 89.2% | 8 | 10.8% |
| Test | F | 135 | 121 | 89.6% | 14 | 10.4% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 286 | 60.38 ± 12.02 | 20.00 | 60.00 | 89.00 |
| Train | F | 549 | 55.34 ± 11.53 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **835** | **57.07 ± 11.94** | **14.00** | **57.00** | **91.00** |
| Test | M | 74 | 57.54 ± 11.97 | 31.00 | 58.00 | 84.00 |
| Test | F | 135 | 54.31 ± 12.57 | 23.00 | 54.00 | 87.00 |
| Test | **All** | **209** | **55.45 ± 12.46** | **23.00** | **57.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 286 | 24.14 ± 2.88 | 16.39 | 24.20 | 32.33 |
| Train | F | 549 | 23.09 ± 3.03 | 12.02 | 22.96 | 32.24 |
| Train | **All** | **835** | **23.45 ± 3.02** | **12.02** | **23.38** | **32.33** |
| Test | M | 74 | 24.02 ± 3.03 | 14.34 | 23.83 | 32.56 |
| Test | F | 135 | 22.80 ± 3.07 | 16.44 | 22.43 | 31.50 |
| Test | **All** | **209** | **23.23 ± 3.11** | **14.34** | **23.17** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8078 | 0.3591 | 0.1411 | 0.7844 | 0.3793 |
| 2 | 0.9251 | 0.6121 | 0.0980 | 0.8563 | 0.5385 |
| 3 | 0.8886 | 0.4840 | 0.1481 | 0.7904 | 0.4262 |
| 4 | 0.8141 | 0.3168 | 0.1711 | 0.7844 | 0.4194 |
| 5 | 0.8514 | 0.5060 | 0.2000 | 0.7006 | 0.3902 |
| **Mean** | **0.8574** | **0.4556** | **0.1517** | **0.7832** | **0.4307** |
| **±Std** | 0.0446 | 0.1062 | 0.0338 | 0.0494 | 0.0566 |

CrossAttn best val AUC per fold: Fold1=0.8078, Fold2=0.9251, Fold3=0.8886, Fold4=0.8141, Fold5=0.8514

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.6738 | 0.2630 | 0.2047 | 0.7129 | 0.2857 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 74 | 0.5871 | 0.2841 | 0.2673 | 0.6351 | 0.2286 |
| F | 135 | 0.7338 | 0.3301 | 0.1705 | 0.7556 | 0.3265 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 137 | 50 |
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
