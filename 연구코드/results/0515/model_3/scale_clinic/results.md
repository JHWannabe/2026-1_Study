# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 16:51  |  5-Fold CV  |  Median best epoch: 28

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 373 | 309 | 82.8% | 64 | 17.2% |
| Train | F | 666 | 616 | 92.5% | 50 | 7.5% |
| Train | **All** | **1039** | **925** | **89.0%** | **114** | **11.0%** |
| Test | M | 98 | 82 | 83.7% | 16 | 16.3% |
| Test | F | 162 | 150 | 92.6% | 12 | 7.4% |
| Test | **All** | **260** | **232** | **89.2%** | **28** | **10.8%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 373 | 59.36 ± 12.75 | 18.00 | 59.00 | 89.00 |
| Train | F | 666 | 55.63 ± 12.02 | 14.00 | 55.00 | 87.00 |
| Train | **All** | **1039** | **56.97 ± 12.41** | **14.00** | **57.00** | **89.00** |
| Test | M | 98 | 61.47 ± 11.63 | 20.00 | 62.50 | 84.00 |
| Test | F | 162 | 55.33 ± 12.79 | 11.00 | 55.50 | 91.00 |
| Test | **All** | **260** | **57.64 ± 12.72** | **11.00** | **58.00** | **91.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 373 | 24.19 ± 3.35 | 14.48 | 24.22 | 36.76 |
| Train | F | 666 | 23.06 ± 3.39 | 14.40 | 22.75 | 39.49 |
| Train | **All** | **1039** | **23.46 ± 3.42** | **14.40** | **23.29** | **39.49** |
| Test | M | 98 | 24.06 ± 2.92 | 17.03 | 24.12 | 31.51 |
| Test | F | 162 | 23.12 ± 3.29 | 16.44 | 22.66 | 34.61 |
| Test | **All** | **260** | **23.47 ± 3.19** | **16.44** | **23.39** | **34.61** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.5805 | 0.2240 | 0.2033 | 0.7115 | 0.1892 |
| 2 | 0.6348 | 0.1538 | 0.2089 | 0.7308 | 0.2432 |
| 3 | 0.6470 | 0.1761 | 0.1900 | 0.7596 | 0.2647 |
| 4 | 0.5318 | 0.2144 | 0.2647 | 0.6394 | 0.2105 |
| 5 | 0.6826 | 0.2378 | 0.1818 | 0.7729 | 0.3733 |
| **Mean** | **0.6153** | **0.2012** | **0.2097** | **0.7229** | **0.2562** |
| **±Std** | 0.0531 | 0.0314 | 0.0291 | 0.0469 | 0.0641 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8270 | 0.4795 | 0.1275 | 0.8125 | 0.4507 |
| 2 | 0.8714 | 0.5464 | 0.1520 | 0.7644 | 0.4096 |
| 3 | 0.8270 | 0.4512 | 0.1478 | 0.7981 | 0.4324 |
| 4 | 0.7441 | 0.3702 | 0.1576 | 0.7596 | 0.3590 |
| 5 | 0.8639 | 0.4059 | 0.2201 | 0.6715 | 0.3704 |
| **Mean** | **0.8267** | **0.4506** | **0.1610** | **0.7612** | **0.4044** |
| **±Std** | 0.0452 | 0.0608 | 0.0313 | 0.0491 | 0.0352 |

CrossAttn best val AUC per fold: Fold1=0.8270, Fold2=0.8714, Fold3=0.8270, Fold4=0.7441, Fold5=0.8639

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5770 | 0.1384 | 0.2157 | 0.6769 | 0.1923 |
| CrossAttn | 0.7577 | 0.2326 | 0.2184 | 0.7346 | 0.3894 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.5648 | 0.1948 | 0.2582 | 0.6224 | 0.2449 |
| F | 162 | 0.5706 | 0.1175 | 0.1900 | 0.7099 | 0.1455 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.6898 | 0.2475 | 0.3322 | 0.5816 | 0.4225 |
| F | 162 | 0.7228 | 0.2131 | 0.1497 | 0.8272 | 0.3333 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 166 | 66 |
| **True: Sarco**  | 18 | 10 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 169 | 63 |
| **True: Sarco**  | 6 | 22 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (LR & CrossAttn) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `training_curves.png` | Loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curves |
| `test_roc_by_sex.png` | Final test-set ROC curves by sex |
| `confusion_matrices.png` | Test-set confusion matrices (LR & CrossAttn) |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
