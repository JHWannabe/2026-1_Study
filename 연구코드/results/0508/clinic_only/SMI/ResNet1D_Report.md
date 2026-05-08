# ResNet1D Report — SMI

**Model:** ResNet1D_ClinicOnly_noScale

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 4.5086 | 0.5192 | 0.7727 | 0.7395 | 3 |
| 2 | 4.4288 | 0.4968 | 0.7636 | 0.6682 | 3 |
| 3 | 4.0556 | 0.5908 | 0.7909 | 0.7289 | 3 |
| 4 | 4.2835 | 0.4893 | 0.7489 | 0.6664 | 3 |
| 5 | 4.5389 | 0.4787 | 0.7534 | 0.6470 | 3 |
| **Mean** | **4.3631** | **0.5150** | **0.7659** | **0.6900** | **3.0** |
| **Std** | **0.1774** | **0.0402** | **0.0150** | **0.0370** | |

## Test Set 성능 (Test 20%)

Test R² = **0.4953**
Test MAE = **4.6291**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.7061 (p=8.264e-43) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.5135 |
| 이진화 ACC (성별 기준) | 0.7527 |
| 이진화 AUC (성별 기준) | 0.6715 |
| 이진화 AUPRC (성별 기준) | 0.8572 |
| 이진화 Brier Score (성별 기준) | 0.3346 |

## 피처 선택 목록 (Fold별)

### Fold 1 (3개)

PatientSex, PatientAge, BMI

### Fold 2 (3개)

PatientSex, PatientAge, BMI

### Fold 3 (3개)

PatientSex, PatientAge, BMI

### Fold 4 (3개)

PatientSex, PatientAge, BMI

### Fold 5 (3개)

PatientSex, PatientAge, BMI

### 최종 모델 (Train 전체, 3개)

PatientSex, PatientAge, BMI

