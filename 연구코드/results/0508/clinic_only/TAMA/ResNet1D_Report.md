# ResNet1D Report — TAMA

**Model:** ResNet1D_ClinicOnly

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 12.7311 | 0.6676 | 0.7636 | 0.6662 | 3 |
| 2 | 11.9838 | 0.6763 | 0.7864 | 0.5531 | 3 |
| 3 | 11.3504 | 0.7187 | 0.7864 | 0.5817 | 3 |
| 4 | 13.8453 | 0.5979 | 0.7215 | 0.6567 | 3 |
| 5 | 11.7548 | 0.7066 | 0.7443 | 0.6908 | 3 |
| **Mean** | **12.3331** | **0.6734** | **0.7604** | **0.6297** | **3.0** |
| **Std** | **0.8795** | **0.0422** | **0.0250** | **0.0529** | |

## Test Set 성능 (Test 20%)

Test R² = **0.7198**
Test MAE = **12.8148**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.8507 (p=3.229e-78) |
| Shapiro-Wilk p | 0.2423 |
| Bias t-test p | 0.1304 |
| 이진화 ACC (성별 기준) | 0.7636 |
| 이진화 AUC (성별 기준) | 0.6374 |

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

