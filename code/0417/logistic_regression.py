# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
logistic_regression.py - Part 2: 로지스틱 회귀 (단변량 + 다변량)

분석 대상: TAMA_binary (이진형, 성별 특이적 임계값 기반 sarcopenia 위험)
예측변수 (전체): Sex, Age, 선택 AEC features, ManufacturerModelName
방법:
  - 단변량 (Univariate): 각 변수 개별 Logit → Crude OR, 95%CI, p-value, AUC
  - 다변량 비교:
      Model A (AEC+CT모델): AEC features + ManufacturerModelName 더미 (성별·나이 제외)
      Model B (전체): AEC features + Sex + Age_z + ManufacturerModelName 더미
      → 성별·나이 추가 효과 비교
  - 모델 성능: AUC-ROC (Bootstrap 95%CI), Sensitivity, Specificity, PPV, NPV
  - 보정도: Hosmer-Lemeshow 검정
  - 설명력: Nagelkerke R²

결과 저장: results/logistic_results.xlsx

실행: python logistic_regression.py
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                              roc_curve)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

import config
import data_loader

os.makedirs(config.RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 통계 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_auc_ci(y_true: np.ndarray,
                     y_prob: np.ndarray,
                     n_bootstrap: int = None,
                     ci: float = 0.95) -> tuple:
    """
    Bootstrap으로 AUC-ROC 신뢰구간 계산.
    n_bootstrap = config.N_BOOTSTRAP (기본값)
    """
    if n_bootstrap is None:
        n_bootstrap = config.N_BOOTSTRAP

    aucs = []
    rng = np.random.default_rng(config.RANDOM_STATE)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(aucs, alpha * 100))
    upper = float(np.percentile(aucs, (1 - alpha) * 100))
    return float(np.mean(aucs)), lower, upper


def hosmer_lemeshow_test(y_true: np.ndarray,
                         y_prob: np.ndarray,
                         g: int = 10) -> tuple:
    """
    Hosmer-Lemeshow 검정 (보정도, calibration).
    g개 십분위수 그룹으로 예측 확률과 실제 발생률 비교.
    p > 0.05 → 모델이 데이터에 잘 보정됨.
    반환: (HL 카이제곱 통계량, p-value)
    """
    df_hl = pd.DataFrame({'y': y_true, 'p': y_prob})
    df_hl['decile'] = pd.qcut(df_hl['p'], q=g, duplicates='drop', labels=False)

    hl_stat = 0.0
    for _, grp in df_hl.groupby('decile'):
        n   = len(grp)
        obs = grp['y'].sum()
        exp = grp['p'].sum()
        if exp > 0 and (n - exp) > 0:
            hl_stat += (obs - exp) ** 2 / (exp * (1 - exp / n))

    p_value = 1 - chi2.cdf(hl_stat, df=g - 2)
    return round(float(hl_stat), 4), round(float(p_value), 6)


def nagelkerke_r2(llf: float, llnull: float, n: int) -> float:
    """
    Nagelkerke pseudo-R² (로지스틱 회귀 설명력).
    Cox-Snell R²를 최대 가능값으로 정규화.
    """
    cox_snell = 1 - np.exp(2 * (llnull - llf) / n)
    max_cs    = 1 - np.exp(2 * llnull / n)
    if max_cs == 0:
        return np.nan
    return round(float(cox_snell / max_cs), 4)


def optimal_threshold_metrics(y_true: np.ndarray,
                               y_prob: np.ndarray) -> dict:
    """
    Youden index (Sensitivity + Specificity - 1)를 최대화하는 임계값에서
    분류 성능 지표 계산.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thr = thresholds[best_idx]

    y_pred = (y_prob >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        'Threshold':   round(float(best_thr), 4),
        'Sensitivity': round(float(sensitivity), 4),
        'Specificity': round(float(specificity), 4),
        'PPV':         round(float(ppv), 4),
        'NPV':         round(float(npv), 4),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 단변량 분석
# ─────────────────────────────────────────────────────────────────────────────

def run_univariate(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 예측변수 개별 로지스틱 회귀 → Crude OR, 95%CI, p-value, AUC.
    ManufacturerModelName은 전체 더미를 투입하고 LR 카이제곱 통계 표시.
    """
    y = df['TAMA_binary'].values
    results = []

    indiv_vars = {
        'Sex (M=1, F=0)': 'Sex',
        'Age (표준화)':    'Age_z',
    }
    for feat in config.SELECTED_AEC_FEATURES:
        indiv_vars[f'AEC: {feat} (표준화)'] = feat + '_z'

    for label, col in indiv_vars.items():
        if col not in df.columns:
            continue
        X = sm.add_constant(df[[col]].values.astype(float), has_constant='add')
        try:
            res = sm.Logit(y, X).fit(disp=False, method='bfgs', maxiter=500)
            ci  = np.exp(res.conf_int())
            or_ = np.exp(res.params[1])
            p   = res.pvalues[1]
            y_prob = res.predict(X)
            auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else np.nan
            results.append({
                'Variable': label,
                'Crude_OR':  round(float(or_), 4),
                'CI_Lower':  round(float(ci[1, 0]), 4),
                'CI_Upper':  round(float(ci[1, 1]), 4),
                'p_value':   round(float(p), 6),
                'AUC':       round(float(auc), 4) if not np.isnan(auc) else 'N/A',
            })
        except Exception as e:
            results.append({
                'Variable': label, 'Crude_OR': 'Error',
                'CI_Lower': str(e)[:40], 'CI_Upper': '', 'p_value': '', 'AUC': '',
            })

    # ManufacturerModelName 범주형
    model_cols = sorted([c for c in df.columns if c.startswith('Model_')])
    if model_cols:
        X_cat = sm.add_constant(df[model_cols].values.astype(float), has_constant='add')
        try:
            res_cat = sm.Logit(y, X_cat).fit(disp=False, method='bfgs', maxiter=500)
            y_prob  = res_cat.predict(X_cat)
            auc_cat = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else np.nan
            # LR 카이제곱: 귀무모델 LL vs 현재 모델 LL
            res_null = sm.Logit(y, np.ones((len(y), 1))).fit(disp=False)
            lr_stat  = 2 * (res_cat.llf - res_null.llf)
            lr_p     = 1 - chi2.cdf(lr_stat, df=len(model_cols))
            results.append({
                'Variable': 'ManufacturerModelName (LR χ²-test)',
                'Crude_OR':  f'LR_χ²={round(lr_stat,3)}',
                'CI_Lower':  'N/A',
                'CI_Upper':  'N/A',
                'p_value':   round(float(lr_p), 6),
                'AUC':       round(float(auc_cat), 4) if not np.isnan(auc_cat) else 'N/A',
            })
        except Exception as e:
            results.append({
                'Variable': 'ManufacturerModelName', 'Crude_OR': 'Error',
                'CI_Lower': str(e)[:40], 'CI_Upper': '', 'p_value': '', 'AUC': '',
            })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 다변량 분석
# ─────────────────────────────────────────────────────────────────────────────

def run_multivariate(df: pd.DataFrame, feature_cols: list = None) -> tuple:
    """
    다중 로지스틱 회귀.
    feature_cols=None 이면 전체 변수(Sex+Age+AEC+Model) 사용.
    반환: (계수 DataFrame, 모델 요약 dict)
    """
    y = df['TAMA_binary'].values
    n = len(y)

    if feature_cols is None:
        all_feat = ['Sex', 'Age_z']
        all_feat += [f + '_z' for f in config.SELECTED_AEC_FEATURES]
        all_feat += sorted([c for c in df.columns if c.startswith('Model_')])
    else:
        all_feat = list(feature_cols)

    all_feat = [c for c in all_feat if c in df.columns]

    X   = df[all_feat].values.astype(float)
    X_c = sm.add_constant(X, has_constant='add')

    res = sm.Logit(y, X_c).fit(disp=False, method='bfgs', maxiter=1000)

    # 계수 테이블
    ci_log = res.conf_int()
    ci_exp = np.exp(ci_log)
    param_names = ['Intercept'] + all_feat
    coef_rows = []
    for i, name in enumerate(param_names):
        coef_rows.append({
            'Variable': name,
            'log_OR':   round(float(res.params[i]), 4),
            'Adj_OR':   round(float(np.exp(res.params[i])), 4),
            'CI_Lower': round(float(ci_exp[i, 0]), 4),
            'CI_Upper': round(float(ci_exp[i, 1]), 4),
            'SE':       round(float(res.bse[i]), 4),
            'z_stat':   round(float(res.tvalues[i]), 4),
            'p_value':  round(float(res.pvalues[i]), 6),
        })
    coef_df = pd.DataFrame(coef_rows)

    # 예측 확률
    y_prob = res.predict(X_c)

    # AUC with Bootstrap CI
    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(y, y_prob)

    # Youden 최적 임계값 기반 분류 성능
    clf_metrics = optimal_threshold_metrics(y, y_prob)

    # Hosmer-Lemeshow
    hl_stat, hl_p = hosmer_lemeshow_test(y, y_prob)

    # Nagelkerke R²
    res_null = sm.Logit(y, np.ones((n, 1))).fit(disp=False)
    nag_r2   = nagelkerke_r2(res.llf, res_null.llf, n)

    # Brier Score
    brier = float(np.mean((y_prob - y) ** 2))

    # 모델 요약
    summary = {
        'N':                n,
        'N_events':         int(y.sum()),
        'N_events_pct':     round(float(y.mean()) * 100, 1),
        'AUC_mean':         round(auc_mean, 4),
        'AUC_CI_lower':     round(auc_lo, 4),
        'AUC_CI_upper':     round(auc_hi, 4),
        'AUC_bootstrap_n':  config.N_BOOTSTRAP,
        **clf_metrics,
        'HL_stat':          hl_stat,
        'HL_p':             hl_p,
        'HL_result':        '보정 양호(p>0.05)' if hl_p > 0.05 else '보정 불량(p≤0.05)',
        'Nagelkerke_R2':    nag_r2,
        'Brier_score':      round(brier, 4),
        'AIC':              round(res.aic, 2),
        'BIC':              round(res.bic, 2),
        'n_predictors':     len(all_feat),
        'converged':        res.mle_retvals.get('converged', True),
    }

    return coef_df, summary


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Part 2 - 로지스틱 회귀 (Logistic Regression)")
    print(f"  종속변수: TAMA_binary")
    print(f"  임계값: M < {config.TAMA_THRESHOLD_MALE} cm², "
          f"F < {config.TAMA_THRESHOLD_FEMALE} cm²")
    print("=" * 60)

    print("\n[데이터 준비]")
    df = data_loader.prepare_full(mode='logistic')

    n_pos = int(df['TAMA_binary'].sum())
    n_tot = len(df)
    if n_pos < 20 or n_pos > n_tot - 20:
        print(f"  ⚠ 클래스 불균형 심각 (양성: {n_pos}/{n_tot}). "
              f"config.py의 임계값을 조정하세요.")

    # feature 목록 정의
    model_cols = sorted([c for c in df.columns if c.startswith('Model_')])
    aec_feat   = [f + '_z' for f in config.SELECTED_AEC_FEATURES]
    demo_feat  = ['Sex', 'Age_z']

    feat_A = aec_feat + model_cols            # AEC + CT모델 (성별·나이 제외)
    feat_B = demo_feat + aec_feat + model_cols  # 전체 (성별·나이 포함)

    # ── 단변량 ──────────────────────────────────────────────────────────────
    print("\n[단변량 분석]")
    uni_df = run_univariate(df)
    print(uni_df[['Variable', 'Crude_OR', 'CI_Lower', 'CI_Upper',
                  'p_value', 'AUC']].to_string(index=False))

    # ── 다변량 - Model A (성별·나이 제외) ───────────────────────────────────
    print("\n[다변량 분석 - Model A: AEC + CT모델 (성별·나이 제외)]")
    coef_A, summ_A = run_multivariate(df, feature_cols=feat_A)
    sig_A = coef_A[coef_A['p_value'] < 0.05]
    print(f"  ▶ 유의 계수 p<0.05 ({len(sig_A)}/{len(coef_A)}개):")
    print(sig_A[['Variable', 'Adj_OR', 'CI_Lower', 'CI_Upper', 'p_value']].to_string(index=False))
    print(f"\n  AUC={summ_A['AUC_mean']} [{summ_A['AUC_CI_lower']}-{summ_A['AUC_CI_upper']}]  "
          f"Sens={summ_A['Sensitivity']}  Spec={summ_A['Specificity']}  "
          f"Nag_R²={summ_A['Nagelkerke_R2']}  Brier={summ_A['Brier_score']}")
    if not summ_A.get('converged', True):
        print("  ⚠ 경고: Model A 수렴 실패.")

    # ── 다변량 - Model B (성별·나이 포함) ───────────────────────────────────
    print("\n[다변량 분석 - Model B: AEC + 성별 + 나이 + CT모델 (전체)]")
    coef_B, summ_B = run_multivariate(df, feature_cols=feat_B)
    sig_B = coef_B[coef_B['p_value'] < 0.05]
    print(f"  ▶ 유의 계수 p<0.05 ({len(sig_B)}/{len(coef_B)}개):")
    print(sig_B[['Variable', 'Adj_OR', 'CI_Lower', 'CI_Upper', 'p_value']].to_string(index=False))
    print(f"\n  AUC={summ_B['AUC_mean']} [{summ_B['AUC_CI_lower']}-{summ_B['AUC_CI_upper']}]  "
          f"Sens={summ_B['Sensitivity']}  Spec={summ_B['Specificity']}  "
          f"Nag_R²={summ_B['Nagelkerke_R2']}  Brier={summ_B['Brier_score']}")
    if not summ_B.get('converged', True):
        print("  ⚠ 경고: Model B 수렴 실패.")

    # ── 성능 비교 요약 ───────────────────────────────────────────────────────
    print("\n[성능 비교 요약: 성별·나이 추가 효과]")
    cmp_keys = ['AUC_mean', 'AUC_CI_lower', 'AUC_CI_upper',
                'Sensitivity', 'Specificity', 'PPV', 'NPV',
                'HL_p', 'HL_result', 'Nagelkerke_R2', 'Brier_score', 'AIC', 'BIC']
    print(f"  {'지표':<22} {'Model A (AEC만)':>18} {'Model B (AEC+성별나이)':>22}")
    print(f"  {'-'*64}")
    for k in cmp_keys:
        print(f"  {k:<22} {str(summ_A.get(k,'')):>18} {str(summ_B.get(k,'')):>22}")

    print(f"\n  ΔAUC (B-A) = {round(summ_B['AUC_mean'] - summ_A['AUC_mean'], 4)}")
    print(f"  ΔNagelkerke_R² (B-A) = "
          f"{round(summ_B['Nagelkerke_R2'] - summ_A['Nagelkerke_R2'], 4)}")

    # ── 저장 ────────────────────────────────────────────────────────────────
    out_path = os.path.join(config.RESULTS_DIR, 'logistic_results.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        uni_df.to_excel(writer, sheet_name='단변량_univariate', index=False)
        coef_A.to_excel(writer, sheet_name='ModelA_계수(성별나이제외)', index=False)
        coef_B.to_excel(writer, sheet_name='ModelB_계수(성별나이포함)', index=False)

        # 비교 요약 시트
        cmp_rows = []
        for k in cmp_keys + ['n_predictors']:
            cmp_rows.append({
                '항목': k,
                'Model_A (AEC+CT모델)':     summ_A.get(k, ''),
                'Model_B (AEC+성별+나이+CT)': summ_B.get(k, ''),
                '근거': _summary_note(k),
            })
        pd.DataFrame(cmp_rows).to_excel(writer, sheet_name='모델비교_summary', index=False)

        # 각 모델 전체 요약
        for label, summ in [('ModelA_summary', summ_A), ('ModelB_summary', summ_B)]:
            pd.DataFrame([
                {'항목': k, '값': v, '근거': _summary_note(k)}
                for k, v in summ.items()
            ]).to_excel(writer, sheet_name=label, index=False)

    print(f"\n[저장] {out_path}")
    print("=" * 60)


def _summary_note(key: str) -> str:
    notes = {
        'N':                '총 환자 수',
        'N_events':         '양성(low TAMA) 환자 수',
        'N_events_pct':     '양성 비율 (%)',
        'AUC_mean':         'AUC-ROC - 판별 능력 표준지표 (0.5=무작위, 1=완벽)',
        'AUC_CI_lower':     f'AUC 95%CI 하한 (Bootstrap n={config.N_BOOTSTRAP})',
        'AUC_CI_upper':     f'AUC 95%CI 상한 (Bootstrap n={config.N_BOOTSTRAP})',
        'Sensitivity':      '민감도: TP/(TP+FN) - low TAMA 감지율',
        'Specificity':      '특이도: TN/(TN+FP) - normal TAMA 구별률',
        'PPV':              '양성 예측도: TP/(TP+FP)',
        'NPV':              '음성 예측도: TN/(TN+FN)',
        'HL_stat':          'Hosmer-Lemeshow χ² - 보정도 검정 통계량',
        'HL_p':             'HL p-value (p>0.05 → 보정 양호)',
        'HL_result':        '',
        'Nagelkerke_R2':    'Nagelkerke pseudo-R² - 로지스틱 설명력 (0~1)',
        'Brier_score':      '확률 예측 정밀도 (0=완벽, 0.25=무작위)',
        'AIC':              '모델 복잡도 대비 적합도 (낮을수록 좋음)',
        'BIC':              'AIC보다 강한 복잡도 페널티',
    }
    return notes.get(key, '')


if __name__ == '__main__':
    main()
