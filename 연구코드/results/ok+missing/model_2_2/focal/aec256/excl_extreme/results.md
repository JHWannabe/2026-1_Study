# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-22 01:48  |  5-Fold CV  |  Median best epoch: 10

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 302 | 256 | 84.8% | 46 | 15.2% |
| Train | F | 533 | 493 | 92.5% | 40 | 7.5% |
| Train | **All** | **835** | **749** | **89.7%** | **86** | **10.3%** |
| Test | M | 80 | 69 | 86.2% | 11 | 13.8% |
| Test | F | 129 | 118 | 91.5% | 11 | 8.5% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 302 | 59.41 ± 12.04 | 20.00 | 59.00 | 89.00 |
| Train | F | 533 | 55.34 ± 11.98 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **835** | **56.81 ± 12.16** | **14.00** | **57.00** | **91.00** |
| Test | M | 80 | 58.96 ± 12.47 | 29.00 | 60.00 | 84.00 |
| Test | F | 129 | 55.14 ± 12.04 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **209** | **56.60 ± 12.35** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 302 | 24.39 ± 2.93 | 17.33 | 24.26 | 32.67 |
| Train | F | 533 | 23.14 ± 3.20 | 16.00 | 22.95 | 34.61 |
| Train | **All** | **835** | **23.59 ± 3.16** | **16.00** | **23.46** | **34.61** |
| Test | M | 80 | 24.14 ± 3.35 | 14.34 | 24.03 | 32.56 |
| Test | F | 129 | 22.96 ± 3.62 | 12.02 | 22.51 | 32.48 |
| Test | **All** | **209** | **23.41 ± 3.56** | **12.02** | **23.26** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8965 | 0.5480 | 0.1980 | 0.7605 | 0.4444 |
| 2 | 0.7851 | 0.2811 | 0.2535 | 0.7066 | 0.3797 |
| 3 | 0.7949 | 0.3359 | 0.2113 | 0.7425 | 0.4110 |
| 4 | 0.8161 | 0.4155 | 0.0985 | 0.8323 | 0.4615 |
| 5 | 0.8684 | 0.6747 | 0.1315 | 0.8443 | 0.5185 |
| **Mean** | **0.8322** | **0.4510** | **0.1786** | **0.7772** | **0.4430** |
| **±Std** | 0.0431 | 0.1434 | 0.0560 | 0.0529 | 0.0471 |

CrossAttn best val AUC per fold: Fold1=0.8965, Fold2=0.7851, Fold3=0.7949, Fold4=0.8161, Fold5=0.8684

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8140 | 0.3672 | 0.1711 | 0.7943 | 0.4416 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 80 | 0.8169 | 0.4314 | 0.2029 | 0.7250 | 0.4500 |
| F | 129 | 0.7935 | 0.3620 | 0.1513 | 0.8372 | 0.4324 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 149 | 38 |
| **True: Sarco**  | 5 | 17 |

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
