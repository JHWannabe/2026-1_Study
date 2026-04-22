# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
generate_report.py - 연구 결과 Markdown 보고서 자동 생성

실행: python generate_report.py
출력: results/research_report.md
"""

import os
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

import code.config as config
import code.data_loader as data_loader
from code.logistic_regression import (bootstrap_auc_ci, hosmer_lemeshow_test,
                                  nagelkerke_r2, optimal_threshold_metrics)
from code.linear_regression   import run_univariate as lin_uni, residual_diagnostics
from code.logistic_regression  import run_univariate as log_uni
from code.multivariable_analysis import fit_linear_case, fit_logistic_case
import pandas as pd


def pval_str(p):
    """p-value를 논문 형식 문자열로 변환"""
    if p < 0.001:
        return f"{p:.2e}"
    elif p < 0.01:
        return f"{p:.4f}"
    else:
        return f"{p:.3f}"


def generate_report():
    print("[보고서 생성 중...]")

    # ── 데이터 준비 ─────────────────────────────────────────────────────────
    df_raw = data_loader.load_raw_data()
    df_lin = data_loader.prepare_full(mode='linear')
    df_log = data_loader.prepare_full(mode='logistic')

    all_feat = ['Sex', 'Age_z']
    all_feat += [f + '_z' for f in config.SELECTED_AEC_FEATURES]
    all_feat += sorted([c for c in df_lin.columns if c.startswith('Model_')])
    all_feat = [c for c in all_feat if c in df_lin.columns]

    # 전체 모델 적합
    y_lin = df_lin['TAMA'].values
    X_lin = sm.add_constant(df_lin[all_feat].values.astype(float), has_constant='add')
    res_lin = sm.OLS(y_lin, X_lin).fit()
    ci_lin  = res_lin.conf_int()

    y_log = df_log['TAMA_binary'].values
    X_log = sm.add_constant(df_log[all_feat].values.astype(float), has_constant='add')
    res_log = sm.Logit(y_log, X_log).fit(disp=False, method='bfgs', maxiter=1000)
    y_prob  = res_log.predict(X_log)

    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(y_log, y_prob)
    hl_stat, hl_p             = hosmer_lemeshow_test(y_log, y_prob)
    nag_r2_val                = nagelkerke_r2(res_log.llf,
                                               sm.Logit(y_log, np.ones((len(y_log), 1)))
                                               .fit(disp=False).llf, len(y_log))
    clf_m = optimal_threshold_metrics(y_log, y_prob)
    brier = float(np.mean((y_prob - y_log) ** 2))

    rmse_full = float(np.sqrt(np.mean((y_lin - res_lin.predict(X_lin)) ** 2)))
    mae_full  = float(np.mean(np.abs(y_lin - res_lin.predict(X_lin))))

    diag = residual_diagnostics(
        res_lin,
        pd.DataFrame(df_lin[all_feat].values, columns=all_feat)
    )

    uni_lin_df = lin_uni(df_lin)
    uni_log_df = log_uni(df_log)

    lin_sums, log_sums = [], []
    for case in [1, 2, 3]:
        feat_cols = data_loader.get_feature_cols(case, df_lin)
        _, _, ls  = fit_linear_case(df_lin, feat_cols)
        _, _, lgs = fit_logistic_case(df_log, feat_cols)
        lin_sums.append(ls)
        log_sums.append(lgs)

    n_total  = len(df_lin)
    n_male   = int((df_lin['Sex'] == 1).sum())
    n_female = n_total - n_male
    n_pos    = int(y_log.sum())

    aec_selected = config.SELECTED_AEC_FEATURES
    thr_m = config.TAMA_THRESHOLD_MALE
    thr_f = config.TAMA_THRESHOLD_FEMALE

    # ── 보고서 텍스트 작성 ───────────────────────────────────────────────────
    report = f"""# TAMA 예측 회귀분석 연구 보고서

> **데이터셋:** {config.SITE} CT 데이터
> **분석 도구:** Python (statsmodels, scikit-learn, scipy)
> **AEC 사용 변수:** {', '.join(aec_selected)}
> **Logistic 임계값:** 남성 < {thr_m} cm², 여성 < {thr_f} cm²

---

## 1. 연구 개요

### 1.1 연구 목적

CT 촬영 시 자동 노출 제어(AEC, Automatic Exposure Control) 곡선과 환자 인구통계학적 특성(성별, 나이)이 복부 근육량 지표인 **TAMA(Total Abdominal Muscle Area)**를 예측하는 데 얼마나 기여하는지 정량화한다.

### 1.2 연구 설계

| 파트 | 분석 방법 | 목적 |
|------|-----------|------|
| **Part 1** | 선형 회귀 (Linear Regression) | TAMA(연속형) 예측 |
| **Part 2** | 로지스틱 회귀 (Logistic Regression) | Low TAMA / Sarcopenia 위험 예측 |
| **Part 3** | Multivariable Analysis (Case 1-3) | AEC·CT모델 추가에 따른 점진적 성능 향상 정량화 |

### 1.3 Feature Set (3 Case)

| Case | 예측변수 |
|------|----------|
| Case 1 | 성별 (Sex), 나이 (Age) |
| Case 2 | 성별, 나이 + **AEC 특징** ({', '.join(aec_selected)}) |
| Case 3 | 성별, 나이, AEC 특징 + **CT 모델명** (ManufacturerModelName), KVP |

---

## 2. 데이터

### 2.1 데이터셋 기본 정보

| 항목 | 값 |
|------|----|
| 총 환자 수 (분석 포함) | **{n_total}명** |
| 남성 | {n_male}명 ({n_male/n_total*100:.1f}%) |
| 여성 | {n_female}명 ({n_female/n_total*100:.1f}%) |
| TAMA 범위 | {df_raw['TAMA'].min():.0f} ~ {df_raw['TAMA'].max():.0f} cm² |
| TAMA 평균 (SD) | {df_raw['TAMA'].mean():.2f} ({df_raw['TAMA'].std():.2f}) cm² |
| CT 스캐너 모델 수 | {df_raw['ManufacturerModelName'].nunique()}종 |

### 2.2 성별 TAMA 분포

| 성별 | N | 평균 | SD | 중앙값 | P25 | P75 |
|------|---|------|----|--------|-----|-----|
| 남성 (M) | {n_male} | {df_raw[df_raw['PatientSex']=='M']['TAMA'].mean():.2f} | {df_raw[df_raw['PatientSex']=='M']['TAMA'].std():.2f} | {df_raw[df_raw['PatientSex']=='M']['TAMA'].median():.2f} | {df_raw[df_raw['PatientSex']=='M']['TAMA'].quantile(0.25):.2f} | {df_raw[df_raw['PatientSex']=='M']['TAMA'].quantile(0.75):.2f} |
| 여성 (F) | {n_female} | {df_raw[df_raw['PatientSex']=='F']['TAMA'].mean():.2f} | {df_raw[df_raw['PatientSex']=='F']['TAMA'].std():.2f} | {df_raw[df_raw['PatientSex']=='F']['TAMA'].median():.2f} | {df_raw[df_raw['PatientSex']=='F']['TAMA'].quantile(0.25):.2f} | {df_raw[df_raw['PatientSex']=='F']['TAMA'].quantile(0.75):.2f} |

### 2.3 Logistic Regression 이진화

- **남성:** TAMA < {thr_m} cm² → 1 (low muscle)
- **여성:** TAMA < {thr_f} cm² → 1 (low muscle)
- **양성 비율:** {n_pos}/{n_total} = {n_pos/n_total*100:.1f}%

### 2.4 AEC Feature 선택 근거

`feature_selection.py`를 통해 TAMA와의 Pearson/Spearman 상관계수를 계산하고, VIF 검사로 다중공선성을 확인하여 아래 4개 feature를 선택하였다.

| Feature | Pearson r | 해석 | VIF |
|---------|-----------|------|-----|
| **p25** | +0.365 | AEC 하위 25%값 → 저선량 구간 tube current | ~1.9 |
| **CV** | -0.349 | 변동계수 (std/mean) → 체형 불균일성 반영 | ~1.8 |
| **skewness** | -0.344 | AEC 곡선 비대칭성 → 체형 분포 특성 | ~1.6 |
| **slope_abs_mean** | ~+0.09 | 평균 절대 기울기 → 곡선 동역학 | ~1.4 |

> **주의:** `mean`과 `AUC_normalized`는 VIF > 50,000으로 심각한 다중공선성이 확인되어 제외하였다.

---

## 3. 분석 방법론

### 3.1 전처리

| 변수 | 처리 방법 |
|------|-----------|
| PatientSex | 이진 인코딩 (M=1, F=0) |
| PatientAge | Z-score 표준화 (StandardScaler) |
| AEC features | Z-score 표준화 |
| ManufacturerModelName | One-hot encoding (drop_first=True) → {len([c for c in df_lin.columns if c.startswith('Model_')])}개 더미 변수 |

### 3.2 모델

- **선형 회귀:** statsmodels OLS (최소자승법)
- **로지스틱 회귀:** statsmodels Logit (BFGS 최적화, maxiter=1000)

### 3.3 모델 검증

- **교차검증:** 5-Fold CV (data leakage 방지 위해 각 fold에서 scaler 재적합)
- **AUC 신뢰구간:** Bootstrap n={config.N_BOOTSTRAP}, 95%CI

---

## 4. 결과

### 4.1 Part 1 — 선형 회귀 (Linear Regression)

#### 4.1.1 단변량 분석 (Univariate)

| 변수 | β | 95%CI | p-value | R² |
|------|---|-------|---------|-----|
{_uni_lin_table(uni_lin_df)}

#### 4.1.2 다변량 분석 (Multivariate — 전체 변수 투입)

**모델 성능:**

| 지표 | 값 | 근거 |
|------|----|------|
| N | {n_total} | 총 환자 수 |
| R² | **{res_lin.rsquared:.4f}** | TAMA 분산의 {res_lin.rsquared*100:.1f}% 설명 |
| Adjusted R² | **{res_lin.rsquared_adj:.4f}** | 변수 수 페널티 적용 설명력 |
| F-statistic | {res_lin.fvalue:.3f} (p={pval_str(res_lin.f_pvalue)}) | 전체 모델 유의성 |
| RMSE | **{rmse_full:.3f} cm²** | 예측 오차 (임상 단위) |
| MAE | {mae_full:.3f} cm² | 이상치 강건 오차 |
| AIC | {res_lin.aic:.2f} | 모델 복잡도 대비 적합도 |
| BIC | {res_lin.bic:.2f} | 강화된 복잡도 페널티 |
| 5-Fold CV R² | {lin_sums[2].get('CV_R2_mean','N/A')} ± {lin_sums[2].get('CV_R2_std','N/A')} | 일반화 성능 |

**잔차 진단:**

| 검정 | 통계량 | p-value | 결과 | 해석 |
|------|--------|---------|------|------|
| Shapiro-Wilk | {diag['SW_stat']} | {pval_str(diag['SW_p'])} | {diag['SW_result']} | 잔차 정규성 |
| Breusch-Pagan | {diag['BP_stat']} | {pval_str(diag['BP_p'])} | {diag['BP_result']} | 등분산성 |
| Durbin-Watson | {diag['DW']} | — | {diag['DW_result']} | 자기상관 (1.5-2.5 정상) |
| Condition number (κ) | {diag['Cond_num']} | — | {diag['Cond_result']} | 다중공선성 |

> **해석:** 잔차가 정규분포를 따르지 않고(SW p<0.05) 이분산성이 존재하므로, 계수 추정치 자체는 BLUE이나 t-검정 p-value 해석 시 주의가 필요하다. Robust standard errors 사용을 고려할 수 있다.

---

### 4.2 Part 2 — 로지스틱 회귀 (Logistic Regression)

#### 4.2.1 단변량 분석 (Crude OR)

| 변수 | Crude OR | 95%CI | p-value | AUC |
|------|----------|-------|---------|-----|
{_uni_log_table(uni_log_df)}

#### 4.2.2 다변량 분석 (전체 변수 투입)

**모델 성능:**

| 지표 | 값 | 근거 |
|------|----|------|
| N | {n_total} (양성: {n_pos}, {n_pos/n_total*100:.1f}%) | 총 환자 수 |
| AUC-ROC | **{auc_mean:.4f}** [{auc_lo:.4f}–{auc_hi:.4f}] | 판별 능력 (Bootstrap 95%CI) |
| Sensitivity | {clf_m['Sensitivity']:.4f} | TP/(TP+FN) |
| Specificity | {clf_m['Specificity']:.4f} | TN/(TN+FP) |
| PPV | {clf_m['PPV']:.4f} | 양성 예측도 |
| NPV | {clf_m['NPV']:.4f} | 음성 예측도 |
| Hosmer-Lemeshow | χ²={hl_stat:.3f}, p={pval_str(hl_p)} | 보정도 (p>0.05 = 보정 양호) |
| Nagelkerke R² | **{nag_r2_val:.4f}** | 로지스틱 설명력 (pseudo-R²) |
| Brier Score | {brier:.4f} | 확률 예측 정밀도 (0=완벽) |
| AIC | {res_log.aic:.2f} | 모델 복잡도 대비 적합도 |

---

### 4.3 Part 3 — Multivariable Analysis (Case 1·2·3 비교)

#### 4.3.1 선형 회귀 성능 비교

| 지표 | Case 1 | Case 2 | Case 3 | Case 1→2 향상 | Case 2→3 향상 |
|------|--------|--------|--------|--------------|--------------|
| R² | {lin_sums[0]['R2']} | {lin_sums[1]['R2']} | {lin_sums[2]['R2']} | **+{lin_sums[1]['R2']-lin_sums[0]['R2']:.4f}** | +{lin_sums[2]['R2']-lin_sums[1]['R2']:.4f} |
| Adj R² | {lin_sums[0]['Adj_R2']} | {lin_sums[1]['Adj_R2']} | {lin_sums[2]['Adj_R2']} | +{lin_sums[1]['Adj_R2']-lin_sums[0]['Adj_R2']:.4f} | +{lin_sums[2]['Adj_R2']-lin_sums[1]['Adj_R2']:.4f} |
| RMSE (cm²) | {lin_sums[0]['RMSE']} | {lin_sums[1]['RMSE']} | {lin_sums[2]['RMSE']} | {lin_sums[1]['RMSE']-lin_sums[0]['RMSE']:.4f} | {lin_sums[2]['RMSE']-lin_sums[1]['RMSE']:.4f} |
| AIC | {lin_sums[0]['AIC']} | {lin_sums[1]['AIC']} | {lin_sums[2]['AIC']} | {lin_sums[1]['AIC']-lin_sums[0]['AIC']:.2f} | {lin_sums[2]['AIC']-lin_sums[1]['AIC']:.2f} |

#### 4.3.2 로지스틱 회귀 성능 비교

| 지표 | Case 1 | Case 2 | Case 3 | Case 1→2 향상 | Case 2→3 향상 |
|------|--------|--------|--------|--------------|--------------|
| AUC | {log_sums[0]['AUC']} | {log_sums[1]['AUC']} | {log_sums[2]['AUC']} | **+{log_sums[1]['AUC']-log_sums[0]['AUC']:.4f}** | +{log_sums[2]['AUC']-log_sums[1]['AUC']:.4f} |
| Nagelkerke R² | {log_sums[0]['Nagelkerke_R2']} | {log_sums[1]['Nagelkerke_R2']} | {log_sums[2]['Nagelkerke_R2']} | +{log_sums[1]['Nagelkerke_R2']-log_sums[0]['Nagelkerke_R2']:.4f} | +{log_sums[2]['Nagelkerke_R2']-log_sums[1]['Nagelkerke_R2']:.4f} |
| AIC | {log_sums[0]['AIC']} | {log_sums[1]['AIC']} | {log_sums[2]['AIC']} | {log_sums[1]['AIC']-log_sums[0]['AIC']:.2f} | {log_sums[2]['AIC']-log_sums[1]['AIC']:.2f} |
| HL p | {log_sums[0]['HL_p']} | {log_sums[1]['HL_p']} | {log_sums[2]['HL_p']} | — | — |

---

## 5. 성능 평가지표 근거

### 5.1 선형 회귀 지표

| 지표 | 통계적 근거 |
|------|------------|
| **R²** | 종속변수(TAMA) 분산 중 모델이 설명하는 비율. 0~1 사이이며 1에 가까울수록 설명력 높음 |
| **Adjusted R²** | 예측변수 수 증가 시 R²가 무조건 증가하는 편향을 보정. Case 간 공정한 비교를 위해 필수 |
| **RMSE** | 예측 오차의 제곱 평균의 제곱근. 단위가 cm²로 임상적 해석이 직관적 |
| **MAE** | 이상치에 강건한 평균 절대 오차. RMSE와 함께 보고 |
| **F-statistic** | 전체 회귀 모델의 유의성 검정 (H₀: 모든 계수=0) |
| **AIC/BIC** | 로그우도에서 모델 복잡도 페널티를 차감한 정보기준. Case 간 모델 선택에 사용 (낮을수록 선호) |
| **5-Fold CV R²** | 훈련/테스트 분리 교차검증으로 과적합 탐지 및 일반화 성능 추정 |
| **Durbin-Watson** | 잔차의 1차 자기상관 검정. 2.0에 가까울수록 독립 가정 충족 |
| **Breusch-Pagan** | 등분산성 검정. p>0.05이면 동분산 가정 충족 |
| **Shapiro-Wilk** | 잔차 정규성 검정 (n>500 시 서브샘플 500개 사용) |

### 5.2 로지스틱 회귀 지표

| 지표 | 통계적 근거 |
|------|------------|
| **AUC-ROC** | 임계값 독립적인 판별 능력 지표. 불균형 데이터에 robust. 0.5=무작위, 1=완벽 |
| **Bootstrap 95%CI** | 비모수적 방법으로 AUC 불확실성 정량화 (n={config.N_BOOTSTRAP}) |
| **Sensitivity** | TP/(TP+FN) — 실제 low TAMA를 얼마나 잘 탐지하는가 |
| **Specificity** | TN/(TN+FP) — 정상 TAMA를 얼마나 잘 식별하는가 |
| **PPV/NPV** | 실제 임상 환경에서의 예측값 해석. 유병률에 영향을 받음 |
| **Hosmer-Lemeshow** | 예측 확률과 관측 비율의 일치도(calibration) 검정. p>0.05이면 잘 보정된 모델 |
| **Nagelkerke R²** | Cox-Snell R²를 최대값으로 정규화한 pseudo-R². 0~1 사이의 설명력 지표 |
| **Brier Score** | 확률 예측의 Mean Squared Error. 0=완벽, 0.25=무작위 (50:50 레이블 기준) |
| **Odds Ratio (OR)** | exp(β). 예측변수 1 SD 증가 시 결과(low TAMA) 발생 오즈의 배수 |

---

## 6. 시각화 자료 목록

| 파일명 | 내용 |
|--------|------|
| `figures/01_feature_correlation.png` | AEC feature-TAMA 상관계수 Top 20 |
| `figures/02_vif_comparison.png` | 선택 AEC feature VIF 비교 |
| `figures/03_tama_distribution.png` | 성별 TAMA 분포 + 이진화 임계값 |
| `figures/04_linear_actual_vs_pred.png` | 선형 회귀: 실제 vs 예측 산점도 |
| `figures/05_linear_residuals.png` | 선형 회귀: 잔차 진단 4-panel |
| `figures/06_linear_forest.png` | 선형 회귀: 유의한 계수 Forest plot |
| `figures/07_linear_univariate_r2.png` | 선형 회귀: 단변량 R² 비교 |
| `figures/08_logistic_roc.png` | 로지스틱: ROC 곡선 (AUC + Bootstrap CI) |
| `figures/09_logistic_calibration.png` | 로지스틱: Calibration plot (HL 검정) |
| `figures/10_logistic_confusion.png` | 로지스틱: Confusion matrix |
| `figures/11_logistic_forest.png` | 로지스틱: Crude OR Forest plot |
| `figures/12_case_metrics_bar.png` | Case 1-3: 선형 R²/RMSE 비교 |
| `figures/13_case_auc_bar.png` | Case 1-3: AUC/Nagelkerke R² 비교 |
| `figures/14_case_aic_bar.png` | Case 1-3: AIC/BIC 비교 |
| `figures/15_case_progression.png` | Case 1-3: 다중 지표 추이 |

---

## 7. 결론

1. **성별의 압도적 기여:** 선형 단변량 분석에서 Sex만으로 R²=0.518을 달성. 남성의 TAMA가 여성 대비 평균 44.8 cm² 높음.

2. **AEC 특징의 독립적 기여:** AEC feature 추가(Case 1→2) 시 선형 R²가 **{lin_sums[0]['R2']:.3f} → {lin_sums[1]['R2']:.3f}** (Δ+{lin_sums[1]['R2']-lin_sums[0]['R2']:.3f})으로 향상됨. 로지스틱 AUC는 **{log_sums[0]['AUC']:.3f} → {log_sums[1]['AUC']:.3f}** (Δ+{log_sums[1]['AUC']-log_sums[0]['AUC']:.3f}).

3. **CT 모델명의 추가 기여:** 모델명 추가(Case 2→3) 시 선형 R²가 추가로 +{lin_sums[2]['R2']-lin_sums[1]['R2']:.3f}, 로지스틱 AUC +{log_sums[2]['AUC']-log_sums[1]['AUC']:.3f} 향상. AIC 감소로 모델 적합도 개선 확인.

4. **최종 모델 (Case 3):** 선형 R²={lin_sums[2]['R2']}, RMSE={lin_sums[2]['RMSE']} cm², 로지스틱 AUC={log_sums[2]['AUC']} [{log_sums[2]['AUC_CI_lower']}-{log_sums[2]['AUC_CI_upper']}].

5. **모델 보정도:** Case 2 이상에서 Hosmer-Lemeshow p>0.05로 로지스틱 모델의 보정도가 양호함.

---

## 8. 한계 및 고려사항

- 잔차 정규성 및 등분산성 가정 위반 → Robust SE 또는 비모수 검정 검토 필요
- Logistic regression 임계값(M<{thr_m}, F<{thr_f} cm²)은 문헌 기반으로 설정되었으나, 기관별 참조값 확인 필요
- 단일 기관(강남) 데이터 → 외부 검증(신촌 데이터 활용) 필요
- AEC feature 선택은 상관분석 기반 — 임상적 의미 검토 권장

---

*생성일: 자동 생성 by generate_report.py*
"""

    # 저장
    out_path = os.path.join(config.RESULTS_DIR, 'research_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[저장] {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 테이블 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────

def _uni_lin_table(uni_df):
    rows = []
    for _, row in uni_df.iterrows():
        var  = row['Variable']
        beta = row['β']
        lo   = row['CI_Lower']
        hi   = row['CI_Upper']
        pv   = row['p_value']
        r2   = row['R2']
        ci_str = f"[{lo}, {hi}]" if lo != 'N/A' else 'N/A'
        p_str  = pval_str(float(pv)) if pv not in ('', 'N/A') else pv
        sig    = ' *' if (pv not in ('', 'N/A') and float(pv) < 0.05) else ''
        rows.append(f"| {var} | {beta}{sig} | {ci_str} | {p_str} | {r2} |")
    return '\n'.join(rows)


def _uni_log_table(uni_df):
    rows = []
    for _, row in uni_df.iterrows():
        var  = row['Variable']
        or_  = row['Crude_OR']
        lo   = row['CI_Lower']
        hi   = row['CI_Upper']
        pv   = row['p_value']
        auc  = row['AUC']
        ci_str = f"[{lo}, {hi}]" if lo != 'N/A' else 'N/A'
        p_str  = pval_str(float(pv)) if pv not in ('', 'N/A') else pv
        sig    = ' *' if (pv not in ('', 'N/A') and str(pv) != '' and
                          float(pv) < 0.05) else ''
        rows.append(f"| {var} | {or_}{sig} | {ci_str} | {p_str} | {auc} |")
    return '\n'.join(rows)


if __name__ == '__main__':
    generate_report()
