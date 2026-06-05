# SMI Binary Classification — AECOnly Results

Generated: 2026-06-04 19:59  |  5-Fold CV  |  Median best epoch: 173

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
| 1 | 0.5920 | 0.1573 | 0.2867 | 0.3825 | 0.2517 |
| 2 | 0.6935 | 0.3005 | 0.2857 | 0.6175 | 0.2857 |
| 3 | 0.7327 | 0.2662 | 0.3112 | 0.8197 | 0.4000 |
| 4 | 0.6720 | 0.2366 | 0.2880 | 0.6831 | 0.3095 |
| 5 | 0.6561 | 0.1883 | 0.2996 | 0.7088 | 0.2933 |
| **Mean** | **0.6693** | **0.2298** | **0.2942** | **0.6423** | **0.3080** |
| **±Std** | 0.0464 | 0.0517 | 0.0098 | 0.1454 | 0.0497 |

AECOnly best val AUC per fold: Fold1=0.5920, Fold2=0.6935, Fold3=0.7327, Fold4=0.6720, Fold5=0.6561

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| AECOnly | 0.6030 | 0.2116 | 0.3163 | 0.5328 | 0.2190 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 82 | 0.6076 | 0.2674 | 0.2976 | 0.5976 | 0.2979 |
| F | 147 | 0.6049 | 0.1920 | 0.3267 | 0.4966 | 0.1778 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 107 | 98 |
| **True: Sarco**  | 9 | 15 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.6030 | 0.4728 | 0.7287 |
| AUPRC | 0.2116 | 0.1129 | 0.3971 |
| Brier | 0.3163 | 0.3029 | 0.3290 |
| Accuracy | 0.5328 | 0.4671 | 0.5983 |
| F1 | 0.2190 | 0.1304 | 0.3158 |

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
