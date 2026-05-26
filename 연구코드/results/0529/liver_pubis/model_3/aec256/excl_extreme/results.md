# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-25 18:24  |  5-Fold CV  |  Median best epoch: 13

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 291 | 248 | 85.2% | 43 | 14.8% |
| Train | F | 545 | 502 | 92.1% | 43 | 7.9% |
| Train | **All** | **836** | **750** | **89.7%** | **86** | **10.3%** |
| Test | M | 72 | 61 | 84.7% | 11 | 15.3% |
| Test | F | 137 | 126 | 92.0% | 11 | 8.0% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 291 | 59.79 ± 12.01 | 20.00 | 60.00 | 89.00 |
| Train | F | 545 | 55.05 ± 11.83 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **836** | **56.70 ± 12.11** | **14.00** | **57.00** | **91.00** |
| Test | M | 72 | 59.79 ± 12.11 | 29.00 | 61.50 | 84.00 |
| Test | F | 137 | 55.86 ± 11.35 | 23.00 | 56.00 | 83.00 |
| Test | **All** | **209** | **57.22 ± 11.76** | **23.00** | **58.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 291 | 24.12 ± 2.77 | 17.33 | 24.11 | 32.33 |
| Train | F | 545 | 23.03 ± 2.96 | 16.00 | 22.95 | 32.24 |
| Train | **All** | **836** | **23.41 ± 2.94** | **16.00** | **23.33** | **32.33** |
| Test | M | 72 | 24.22 ± 3.49 | 14.34 | 24.01 | 32.56 |
| Test | F | 137 | 22.90 ± 3.24 | 12.02 | 22.60 | 30.84 |
| Test | **All** | **209** | **23.36 ± 3.39** | **12.02** | **23.17** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7804 | 0.4495 | 0.1882 | 0.7917 | 0.4262 |
| 2 | 0.7659 | 0.2906 | 0.1945 | 0.7784 | 0.4127 |
| 3 | 0.8922 | 0.4340 | 0.1803 | 0.7665 | 0.4507 |
| 4 | 0.8824 | 0.3781 | 0.1168 | 0.7725 | 0.4722 |
| 5 | 0.8271 | 0.4016 | 0.1632 | 0.8323 | 0.4815 |
| **Mean** | **0.8296** | **0.3908** | **0.1686** | **0.7883** | **0.4487** |
| **±Std** | 0.0514 | 0.0559 | 0.0279 | 0.0236 | 0.0262 |

CrossAttn best val AUC per fold: Fold1=0.7804, Fold2=0.7659, Fold3=0.8922, Fold4=0.8824, Fold5=0.8271

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8274 | 0.4327 | 0.2231 | 0.6938 | 0.3469 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 72 | 0.8465 | 0.5512 | 0.2343 | 0.6806 | 0.4390 |
| F | 137 | 0.7951 | 0.3647 | 0.2173 | 0.7007 | 0.2807 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 128 | 59 |
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
