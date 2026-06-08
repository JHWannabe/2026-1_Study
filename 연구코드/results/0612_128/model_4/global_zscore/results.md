# SMI Binary Classification — AECOnly Results

Generated: 2026-06-08 15:56  |  5-Fold CV  |  Median best epoch: 82

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
| 1 | 0.5917 | 0.1618 | 0.2265 | 0.3770 | 0.2500 |
| 2 | 0.6897 | 0.2774 | 0.2864 | 0.5355 | 0.2857 |
| 3 | 0.6772 | 0.1956 | 0.2657 | 0.7760 | 0.3692 |
| 4 | 0.6942 | 0.2232 | 0.2944 | 0.6885 | 0.3133 |
| 5 | 0.6371 | 0.1619 | 0.2959 | 0.7363 | 0.3143 |
| **Mean** | **0.6580** | **0.2040** | **0.2738** | **0.6227** | **0.3065** |
| **±Std** | 0.0388 | 0.0433 | 0.0260 | 0.1474 | 0.0392 |

AECOnly best val AUC per fold: Fold1=0.5917, Fold2=0.6897, Fold3=0.6772, Fold4=0.6942, Fold5=0.6371

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| AECOnly | 0.5852 | 0.1928 | 0.2956 | 0.4978 | 0.2069 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 82 | 0.6087 | 0.2326 | 0.2830 | 0.5488 | 0.3019 |
| F | 147 | 0.5922 | 0.2721 | 0.3026 | 0.4694 | 0.1522 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 99 | 106 |
| **True: Sarco**  | 9 | 15 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.5852 | 0.4565 | 0.7081 |
| AUPRC | 0.1928 | 0.0933 | 0.3357 |
| Brier | 0.2956 | 0.2868 | 0.3042 |
| Accuracy | 0.4978 | 0.4323 | 0.5633 |
| F1 | 0.2069 | 0.1231 | 0.2968 |

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
