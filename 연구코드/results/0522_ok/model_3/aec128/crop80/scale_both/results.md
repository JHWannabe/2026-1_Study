# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:12  |  5-Fold CV  |  Median best epoch: 13

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 286 | 245 | 85.7% | 41 | 14.3% |
| Train | F | 374 | 335 | 89.6% | 39 | 10.4% |
| Train | **All** | **660** | **580** | **87.9%** | **80** | **12.1%** |
| Test | M | 66 | 53 | 80.3% | 13 | 19.7% |
| Test | F | 100 | 93 | 93.0% | 7 | 7.0% |
| Test | **All** | **166** | **146** | **88.0%** | **20** | **12.0%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 286 | 59.38 ± 11.37 | 23.00 | 60.00 | 83.00 |
| Train | F | 374 | 56.41 ± 12.15 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **660** | **57.70 ± 11.91** | **14.00** | **58.00** | **91.00** |
| Test | M | 66 | 59.38 ± 13.64 | 28.00 | 59.00 | 89.00 |
| Test | F | 100 | 56.60 ± 12.99 | 24.00 | 56.50 | 87.00 |
| Test | **All** | **166** | **57.70 ± 13.32** | **24.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 286 | 24.38 ± 3.00 | 14.34 | 24.35 | 32.67 |
| Train | F | 374 | 22.87 ± 3.33 | 12.02 | 22.74 | 34.61 |
| Train | **All** | **660** | **23.53 ± 3.27** | **12.02** | **23.45** | **34.61** |
| Test | M | 66 | 24.44 ± 3.12 | 17.57 | 24.00 | 32.56 |
| Test | F | 100 | 23.27 ± 3.32 | 16.44 | 22.81 | 34.20 |
| Test | **All** | **166** | **23.74 ± 3.29** | **16.44** | **23.35** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8384 | 0.3870 | 0.1681 | 0.7348 | 0.4262 |
| 2 | 0.7861 | 0.3174 | 0.1780 | 0.7273 | 0.3333 |
| 3 | 0.8314 | 0.4631 | 0.1927 | 0.7273 | 0.4375 |
| 4 | 0.8109 | 0.3406 | 0.1614 | 0.7424 | 0.3704 |
| 5 | 0.7786 | 0.4422 | 0.1519 | 0.7803 | 0.4082 |
| **Mean** | **0.8091** | **0.3901** | **0.1704** | **0.7424** | **0.3951** |
| **±Std** | 0.0237 | 0.0562 | 0.0140 | 0.0198 | 0.0384 |

CrossAttn best val AUC per fold: Fold1=0.8384, Fold2=0.7861, Fold3=0.8314, Fold4=0.8109, Fold5=0.7786

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8072 | 0.3906 | 0.2506 | 0.5843 | 0.3429 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 66 | 0.7968 | 0.5077 | 0.2295 | 0.6212 | 0.4681 |
| F | 100 | 0.7865 | 0.2883 | 0.2645 | 0.5600 | 0.2414 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 79 | 67 |
| **True: Sarco**  | 2 | 18 |

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
