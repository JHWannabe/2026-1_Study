"""
multivariable_analysis.py - Part 3: Multivariable Analysis (Case 비교)

3가지 feature set을 점진적으로 구축하여 AEC·모델명의 기여도 정량화.

Case 1: [Sex, Age]
Case 2: [Sex, Age, 선택 AEC features]
Case 3: [Sex, Age, 선택 AEC features, KVP, ManufacturerModelName]

각 Case마다:
  - Linear OLS: R², Adj R², RMSE, AIC, BIC, F-test
  - Logistic:   AUC (Bootstrap 95%CI), Nagelkerke R², HL-test, AIC, BIC

결과 저장: results/multivariable_results.xlsx
  - Sheet: 'Case_비교_요약'         ← 3 Case 핵심 지표 비교
  - Sheet: 'Case1_linear_coef'     ← Case 1 선형 회귀 계수
  - Sheet: 'Case1_logistic_coef'   ← Case 1 로지스틱 회귀 계수
  - … Case2, Case3 동일 구조

실행: python multivariable_analysis.py
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import config as config
import data_loader as data_loader
from logistic_regression import (bootstrap_auc_ci, hosmer_lemeshow_test,
                                  nagelkerke_r2, optimal_threshold_metrics)

os.makedirs(config.RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Case별 선형 회귀
# ─────────────────────────────────────────────────────────────────────────────

def fit_linear_case(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    지정 feature_cols로 OLS 실행.
    반환: (statsmodels result, 계수 DataFrame, 요약 dict)
    """
    y   = df['TAMA'].values
    X   = df[feature_cols].values.astype(float)
    X_c = sm.add_constant(X, has_constant='add')
    res = sm.OLS(y, X_c).fit()

    # 계수 테이블
    ci = res.conf_int()
    param_names = ['Intercept'] + feature_cols
    coef_rows = []
    for i, name in enumerate(param_names):
        coef_rows.append({
            'Variable': name,
            'β':        round(float(res.params[i]), 4),
            'CI_Lower': round(float(ci[i, 0]), 4),
            'CI_Upper': round(float(ci[i, 1]), 4),
            'SE':       round(float(res.bse[i]), 4),
            't_stat':   round(float(res.tvalues[i]), 4),
            'p_value':  f"{res.pvalues[i]:.4e}",
        })
    coef_df = pd.DataFrame(coef_rows)

    y_pred = res.predict(X_c)
    rmse   = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae    = float(mean_absolute_error(y, y_pred))

    summary = {
        'N':        len(y),
        'R2':       round(res.rsquared,     4),
        'Adj_R2':   round(res.rsquared_adj, 4),
        'F_stat':   round(res.fvalue,       4),
        'F_pvalue': f"{res.f_pvalue:.4e}",
        'RMSE':     round(rmse, 4),
        'MAE':      round(mae,  4),
        'AIC':      round(res.aic, 2),
        'BIC':      round(res.bic, 2),
    }
    return res, coef_df, summary


# ─────────────────────────────────────────────────────────────────────────────
# Case별 로지스틱 회귀
# ─────────────────────────────────────────────────────────────────────────────

def fit_logistic_case(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    지정 feature_cols로 Logit 실행.
    반환: (statsmodels result, 계수 DataFrame, 요약 dict)
    """
    y   = df['TAMA_binary'].values
    n   = len(y)
    X   = df[feature_cols].values.astype(float)
    X_c = sm.add_constant(X, has_constant='add')
    res = sm.Logit(y, X_c).fit(disp=False, method='bfgs', maxiter=1000)

    # 계수 테이블
    ci_log = res.conf_int()
    ci_exp = np.exp(ci_log)
    param_names = ['Intercept'] + feature_cols
    coef_rows = []
    for i, name in enumerate(param_names):
        coef_rows.append({
            'Variable': name,
            'log_OR':   round(float(res.params[i]), 4),
            'OR':       round(float(np.exp(res.params[i])), 4),
            'CI_Lower': round(float(ci_exp[i, 0]), 4),
            'CI_Upper': round(float(ci_exp[i, 1]), 4),
            'SE':       round(float(res.bse[i]), 4),
            'z_stat':   round(float(res.tvalues[i]), 4),
            'p_value':  f"{res.pvalues[i]:.4e}",
        })
    coef_df = pd.DataFrame(coef_rows)

    y_prob = res.predict(X_c)

    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(y, y_prob)
    hl_stat, hl_p            = hosmer_lemeshow_test(y, y_prob)

    res_null = sm.Logit(y, np.ones((n, 1))).fit(disp=False)
    nag_r2   = nagelkerke_r2(res.llf, res_null.llf, n)
    brier    = float(np.mean((y_prob - y) ** 2))

    summary = {
        'N':              n,
        'N_events':       int(y.sum()),
        'AUC':            round(auc_mean, 4),
        'AUC_CI_lower':   round(auc_lo,   4),
        'AUC_CI_upper':   round(auc_hi,   4),
        'HL_stat':        hl_stat,
        'HL_p':           f"{hl_p:.4e}",
        'HL_result':      '보정 양호' if hl_p > 0.05 else '보정 불량',
        'Nagelkerke_R2':  nag_r2,
        'Brier_score':    round(brier, 4),
        'AIC':            round(res.aic, 2),
        'BIC':            round(res.bic, 2),
        'converged':      res.mle_retvals.get('converged', True),
    }
    return res, coef_df, summary


# ─────────────────────────────────────────────────────────────────────────────
# 비교 요약 테이블 생성
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(lin_summaries: list, log_summaries: list) -> pd.DataFrame:
    """
    Case 1·2·3의 선형/로지스틱 요약을 하나의 비교 테이블로 통합.
    """
    rows = []

    linear_metrics = [
        ('선형 회귀',           None,         None),
        ('N',                  'N',           '총 환자 수'),
        ('R²',                 'R2',          'TAMA 분산 설명 비율'),
        ('Adjusted R²',        'Adj_R2',      '변수 수 페널티 적용 R²'),
        ('F p-value',          'F_pvalue',    '전체 모델 유의성'),
        ('RMSE (cm²)',         'RMSE',        '예측 오차'),
        ('MAE (cm²)',          'MAE',         '평균 절대 오차'),
        ('AIC',                'AIC',         '모델 복잡도 대비 적합도'),
        ('BIC',                'BIC',         '강화 복잡도 페널티'),
    ]
    logistic_metrics = [
        ('로지스틱 회귀',        None,          None),
        ('N',                  'N',           '총 환자 수'),
        ('양성 수 (events)',    'N_events',    'low TAMA 환자 수'),
        ('AUC',                'AUC',         '판별 능력 (0.5~1)'),
        ('AUC CI Lower',       'AUC_CI_lower','Bootstrap 95%CI 하한'),
        ('AUC CI Upper',       'AUC_CI_upper','Bootstrap 95%CI 상한'),
        ('HL p-value',         'HL_p',        '보정도 (p>0.05 양호)'),
        ('HL 결과',             'HL_result',   ''),
        ('Nagelkerke R²',      'Nagelkerke_R2','로지스틱 설명력'),
        ('Brier Score',        'Brier_score', '확률 예측 정밀도'),
        ('AIC',                'AIC',         '모델 복잡도 대비 적합도'),
        ('BIC',                'BIC',         '강화 복잡도 페널티'),
    ]

    for label, key, note in linear_metrics:
        if key is None:
            rows.append({'지표': f'── {label} ──', 'Case_1': '', 'Case_2': '', 'Case_3': '', '근거': ''})
        else:
            rows.append({
                '지표':   label,
                'Case_1': lin_summaries[0].get(key, ''),
                'Case_2': lin_summaries[1].get(key, ''),
                'Case_3': lin_summaries[2].get(key, ''),
                '근거':   note,
            })

    for label, key, note in logistic_metrics:
        if key is None:
            rows.append({'지표': f'── {label} ──', 'Case_1': '', 'Case_2': '', 'Case_3': '', '근거': ''})
        else:
            rows.append({
                '지표':   label,
                'Case_1': log_summaries[0].get(key, ''),
                'Case_2': log_summaries[1].get(key, ''),
                'Case_3': log_summaries[2].get(key, ''),
                '근거':   note,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Part 3 - Multivariable Analysis (Case 1·2·3 비교)")
    print("=" * 60)
    print(f"\n  Case 1: [Sex, Age]")
    print(f"  Case 2: [Sex, Age, AEC: {config.SELECTED_AEC_FEATURES}]")
    print(f"  Case 3: [Sex, Age, AEC, KVP, ManufacturerModelName]")
    print(f"\n  임계값: M < {config.TAMA_THRESHOLD_MALE} cm², "
          f"F < {config.TAMA_THRESHOLD_FEMALE} cm²")

    print("\n[데이터 준비]")
    # 로지스틱용 (TAMA_binary 포함)
    df_log = data_loader.prepare_full(mode='logistic')
    # 선형용 (동일 DataFrame, mode='linear' 결과와 동일)
    df_lin = df_log.copy()   # TAMA_binary 컬럼 존재해도 선형 회귀에서 사용 안 함

    lin_summaries = []
    log_summaries = []
    lin_coef_dfs  = {}
    log_coef_dfs  = {}

    for case in [1, 2, 3]:
        feat_cols = data_loader.get_feature_cols(case, df_lin)
        print(f"\n── Case {case} ({'  '.join(feat_cols[:4])}{'...' if len(feat_cols) > 4 else ''}) ──")

        # 선형 회귀
        _, lin_coef, lin_sum = fit_linear_case(df_lin, feat_cols)
        lin_summaries.append(lin_sum)
        lin_coef_dfs[case] = lin_coef
        print(f"  [Linear ] R²={lin_sum['R2']}, Adj R²={lin_sum['Adj_R2']}, "
              f"RMSE={lin_sum['RMSE']}, AIC={lin_sum['AIC']}")

        # 로지스틱 회귀
        _, log_coef, log_sum = fit_logistic_case(df_log, feat_cols)
        log_summaries.append(log_sum)
        log_coef_dfs[case] = log_coef
        print(f"  [Logistic] AUC={log_sum['AUC']} "
              f"[{log_sum['AUC_CI_lower']}–{log_sum['AUC_CI_upper']}], "
              f"Nagelkerke R²={log_sum['Nagelkerke_R2']}, "
              f"HL p={log_sum['HL_p']}, {log_sum['HL_result']}")
        if not log_sum.get('converged', True):
            print(f"  ⚠ Case {case} 로지스틱 수렴 실패 주의")

    # ── 비교 요약 ────────────────────────────────────────────────────────────
    print("\n[Case 비교 요약]")
    comp_df = build_comparison_table(lin_summaries, log_summaries)
    print(comp_df.to_string(index=False))

    # ── 저장 ────────────────────────────────────────────────────────────────
    out_path = os.path.join(config.RESULTS_DIR, 'multivariable_results.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        comp_df.to_excel(writer, sheet_name='Case_비교_요약', index=False)

        for case in [1, 2, 3]:
            lin_coef_dfs[case].to_excel(
                writer, sheet_name=f'Case{case}_linear_coef', index=False)
            log_coef_dfs[case].to_excel(
                writer, sheet_name=f'Case{case}_logistic_coef', index=False)

            # 각 Case 요약도 저장
            lin_sum_df = pd.DataFrame(
                list(lin_summaries[case - 1].items()), columns=['항목', '값'])
            log_sum_df = pd.DataFrame(
                list(log_summaries[case - 1].items()), columns=['항목', '값'])
            lin_sum_df.to_excel(writer, sheet_name=f'Case{case}_linear_summary',   index=False)
            log_sum_df.to_excel(writer, sheet_name=f'Case{case}_logistic_summary', index=False)

    print(f"\n[저장] {out_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
