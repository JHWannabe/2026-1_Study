# SMI Binary Classification — AECOnly Results

Generated: 2026-06-08 18:33  |  5-Fold CV  |  Median best epoch: 38

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

![Label Distribution](label_distribution.png)

---

## 1. Cross-Validation Summary

### AECOnly

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.5469 | 0.1802 | 0.3146 | 0.8579 | 0.2778 |
| 2 | 0.5372 | 0.1188 | 0.3010 | 0.4208 | 0.2319 |
| 3 | 0.5761 | 0.1580 | 0.3115 | 0.5301 | 0.2321 |
| 4 | 0.5761 | 0.1497 | 0.2850 | 0.5847 | 0.2549 |
| 5 | 0.5554 | 0.1701 | 0.3156 | 0.5275 | 0.2586 |
| **Mean** | **0.5583** | **0.1554** | **0.3056** | **0.5842** | **0.2511** |
| **±Std** | 0.0156 | 0.0210 | 0.0115 | 0.1468 | 0.0174 |

AECOnly best val AUC per fold: Fold1=0.5469, Fold2=0.5372, Fold3=0.5761, Fold4=0.5761, Fold5=0.5554

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| AECOnly | 0.5213 | 0.1126 | 0.3037 | 0.7293 | 0.1143 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 82 | 0.5195 | 0.1671 | 0.2989 | 0.5732 | 0.1860 |
| F | 147 | 0.3957 | 0.0663 | 0.3064 | 0.8163 | 0.0000 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 163 | 42 |
| **True: Sarco**  | 20 | 4 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.5213 | 0.3974 | 0.6417 |
| AUPRC | 0.1126 | 0.0724 | 0.1808 |
| Brier | 0.3037 | 0.2983 | 0.3084 |
| Accuracy | 0.7293 | 0.6725 | 0.7860 |
| F1 | 0.1143 | 0.0282 | 0.2222 |

---

## 5. Figures

| File | Description |
|------|-------------|
| `label_distribution.png` | Train/Test class·sex distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (AECOnly) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `training_curves.png` | Loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curve |
| `test_roc_by_sex.png` | Final test-set ROC curves by sex |
| `confusion_matrices.png` | Test-set confusion matrices |
| `calibration.png` | Calibration plot + Precision-Recall curve |
| `cam_aec_mean.png` | Grad-CAM mean ± std per class |
| `cam_aec_lines.png` | Grad-CAM individual samples per class |
| `cam_aec_heatmap.png` | Grad-CAM sample-level heatmap |
