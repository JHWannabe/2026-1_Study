# ResNet1D Report — TAMA

**Model:** ResNet1D_ClinicOnly_scaleXY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 12.4528 | 0.6765 | 0.7682 | 0.6693 | 3 |
| 2 | 12.0479 | 0.6781 | 0.7682 | 0.5537 | 3 |
| 3 | 11.1967 | 0.7260 | 0.8045 | 0.5910 | 3 |
| 4 | 13.4814 | 0.6081 | 0.7443 | 0.6713 | 3 |
| 5 | 11.7700 | 0.7001 | 0.7763 | 0.6777 | 3 |
| **Mean** | **12.1898** | **0.6778** | **0.7723** | **0.6326** | **3.0** |
| **Std** | **0.7642** | **0.0392** | **0.0194** | **0.0506** | |

## Test Set 성능 (Test 20%)

Test R² = **0.7125**
Test MAE = **12.7354**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.8446 (p=4.686e-76) |
| Shapiro-Wilk p | 0.0445 |
| Bias t-test p | 0.7921 |
| 이진화 ACC (성별 기준) | 0.7709 |
| 이진화 AUC (성별 기준) | 0.6383 |
| 이진화 AUPRC (성별 기준) | 0.8458 |
| 이진화 Brier Score (성별 기준) | 0.2419 |

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

