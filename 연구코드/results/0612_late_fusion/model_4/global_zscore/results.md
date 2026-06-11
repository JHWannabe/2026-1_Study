# SMI Binary Classification — AECOnly Results

Generated: 2026-06-11 14:30  |  5-Fold CV  |  Median best epoch: 51

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
| 1 | 0.6077 | 0.1669 | 0.2616 | 0.3770 | 0.2500 |
| 2 | 0.6518 | 0.2005 | 0.3232 | 0.7541 | 0.3077 |
| 3 | 0.7404 | 0.3101 | 0.2912 | 0.8689 | 0.4545 |
| 4 | 0.6977 | 0.1915 | 0.3200 | 0.5628 | 0.2857 |
| 5 | 0.6448 | 0.1899 | 0.3014 | 0.6758 | 0.2892 |
| **Mean** | **0.6685** | **0.2118** | **0.2995** | **0.6477** | **0.3174** |
| **±Std** | 0.0460 | 0.0504 | 0.0223 | 0.1682 | 0.0711 |

AECOnly best val AUC per fold: Fold1=0.6077, Fold2=0.6518, Fold3=0.7404, Fold4=0.6977, Fold5=0.6448

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| AECOnly | 0.5785 | 0.1369 | 0.2795 | 0.8952 | 0.0000 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 82 | 0.6042 | 0.2404 | 0.2666 | 0.8415 | 0.0000 |
| F | 147 | 0.5561 | 0.0888 | 0.2867 | 0.9252 | 0.0000 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 205 | 0 |
| **True: Sarco**  | 24 | 0 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.5785 | 0.4567 | 0.6939 |
| AUPRC | 0.1369 | 0.0846 | 0.2462 |
| Brier | 0.2795 | 0.2720 | 0.2871 |
| Accuracy | 0.8952 | 0.8515 | 0.9301 |
| F1 | 0.0000 | 0.0000 | 0.0000 |

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
