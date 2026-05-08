# ResNet1D Report — SMI

**Model:** ResNet1D_ClinicOnly_scaleX

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 4.4618 | 0.5160 | 0.7818 | 0.7345 | 3 |
| 2 | 4.4606 | 0.4920 | 0.7727 | 0.6599 | 3 |
| 3 | 4.0353 | 0.6082 | 0.7773 | 0.7477 | 3 |
| 4 | 4.4469 | 0.4681 | 0.7671 | 0.6892 | 3 |
| 5 | 4.5351 | 0.4806 | 0.7671 | 0.6440 | 3 |
| **Mean** | **4.3879** | **0.5129** | **0.7732** | **0.6951** | **3.0** |
| **Std** | **0.1790** | **0.0502** | **0.0057** | **0.0405** | |

## Test Set 성능 (Test 20%)

Test R² = **0.4933**
Test MAE = **4.5566**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.7055 (p=1.021e-42) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.3457 |
| 이진화 ACC (성별 기준) | 0.7818 |
| 이진화 AUC (성별 기준) | 0.6937 |
| 이진화 AUPRC (성별 기준) | 0.8695 |
| 이진화 Brier Score (성별 기준) | 0.3315 |

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

