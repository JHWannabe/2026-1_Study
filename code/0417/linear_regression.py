# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
linear_regression.py - Part 1: 선형 회귀 (단변량 + 다변량)

분석 대상: TAMA (연속형, cm²)
예측변수 (전체): Sex, Age, 선택 AEC features, ManufacturerModelName
방법:
  - 단변량 (Univariate): 각 변수를 개별 OLS → β, 95%CI, p-value, R²
  - 다변량 비교:
      Model A (AEC+CT모델): AEC features + ManufacturerModelName 더미 (성별·나이 제외)
      Model B (전체): AEC features + Sex + Age_z + ManufacturerModelName 더미
      → 성별·나이 추가 효과 비교
  - 잔차 진단: Shapiro-Wilk, Breusch-Pagan, Durbin-Watson, Condition number
  - 5-Fold Cross-Validation R²

결과 저장: results/linear_results.xlsx

실행: python linear_regression.py
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import config
import data_loader

os.makedirs(config.RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 잔차 진단
# ─────────────────────────────────────────────────────────────────────────────

def residual_diagnostics(model_result, X: pd.DataFrame) -> dict:
    """
    선형 회귀 잔차에 대한 4가지 진단 검정.

    - Shapiro-Wilk : 잔차 정규성 (n>500이면 서브샘플 500개 사용)
    - Breusch-Pagan : 등분산성 (p>0.05 = 동분산 가정 충족)
    - Durbin-Watson : 자기상관 (≈2.0 목표, <1.5 또는 >2.5 주의)
    - Condition number: 다중공선성 (κ<30 양호, >1000 심각)
    """
    resid = np.asarray(model_result.resid)

    # Shapiro-Wilk (최대 500개 서브샘플)
    if len(resid) > 500:
        rng = np.random.default_rng(config.RANDOM_STATE)
        resid_sample = rng.choice(resid, 500, replace=False)
    else:
        resid_sample = resid
    sw_stat, sw_p = stats.shapiro(resid_sample)

    # Breusch-Pagan (상수항 포함 exog 필요)
    X_arr   = X.values.astype(float)
    X_const = sm.add_constant(X_arr, has_constant='add')
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, X_const)

    # Durbin-Watson
    dw = durbin_watson(resid)

    # Condition number
    kappa = np.linalg.cond(np.column_stack([np.ones(len(X_arr)), X_arr]))

    return {
        'SW_stat':  round(sw_stat, 4),
        'SW_p':     round(sw_p, 6),
        'SW_result': '정규' if sw_p > 0.05 else '비정규',
        'BP_stat':  round(bp_lm, 4),
        'BP_p':     round(bp_p, 6),
        'BP_result': '등분산' if bp_p > 0.05 else '이분산',
        'DW':       round(dw, 4),
        'DW_result': '정상' if 1.5 < dw < 2.5 else '자기상관 주의',
        'Cond_num': round(kappa, 2),
        'Cond_result': '양호(κ<30)' if kappa < 30 else ('주의(κ<1000)' if kappa < 1000 else '심각(κ≥1000)'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold CV
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_linear(df_raw: pd.DataFrame, feature_cols: list) -> dict:
    """
    5-Fold CV로 선형 회귀 R² 추정.
    각 fold에서 scaler를 train에만 fit하여 data leakage 방지.
    """
    kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    cv_r2 = []

    for train_idx, test_idx in kf.split(df_raw):
        df_tr, df_te = data_loader.prepare_cv_fold(df_raw, train_idx, test_idx)

        # 이번 fold에 존재하는 Model_ 더미 컬럼만 사용
        fold_features = [c for c in feature_cols if c in df_tr.columns and c in df_te.columns]

        X_tr = df_tr[fold_features].values.astype(float)
        X_te = df_te[fold_features].values.astype(float)
        y_tr = df_tr['TAMA'].values
        y_te = df_te['TAMA'].values

        X_tr_c = sm.add_constant(X_tr, has_constant='add')
        X_te_c = sm.add_constant(X_te, has_constant='add')
        if X_te_c.shape[1] != X_tr_c.shape[1]:
            continue

        model = sm.OLS(y_tr, X_tr_c).fit()
        y_pred = model.predict(X_te_c)
        cv_r2.append(r2_score(y_te, y_pred))

    return {
        'CV_R2_mean': round(float(np.mean(cv_r2)), 4),
        'CV_R2_std':  round(float(np.std(cv_r2)),  4),
        'CV_folds':   len(cv_r2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 단변량 분석
# ─────────────────────────────────────────────────────────────────────────────

def run_univariate(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 예측변수 그룹별 단순 OLS 실행.
    ManufacturerModelName은 더미 전체를 투입하고 F-test 결과만 표시.
    """
    y = df['TAMA'].values
    results = []

    # ── 연속/이진 변수 ──────────────────────────────────────────────────────
    indiv_vars = {
        'Sex (M=1, F=0)': 'Sex',
        'Age (표준화)':    'Age_z',
    }
    # AEC features
    for feat in config.SELECTED_AEC_FEATURES:
        z_col = feat + '_z'
        indiv_vars[f'AEC: {feat} (표준화)'] = z_col

    for label, col in indiv_vars.items():
        if col not in df.columns:
            continue
        X = sm.add_constant(df[[col]].values.astype(float), has_constant='add')
        res = sm.OLS(y, X).fit()

        ci = res.conf_int()
        results.append({
            'Variable':   label,
            'β':          round(res.params[1], 4),
            'CI_Lower':   round(ci[1, 0], 4),
            'CI_Upper':   round(ci[1, 1], 4),
            'SE':         round(res.bse[1], 4),
            't_stat':     round(res.tvalues[1], 4),
            'p_value':    round(res.pvalues[1], 6),
            'R2':         round(res.rsquared, 4),
            'Adj_R2':     round(res.rsquared_adj, 4),
        })

    # ── ManufacturerModelName (범주형 → F-test) ─────────────────────────────
    model_cols = sorted([c for c in df.columns if c.startswith('Model_')])
    if model_cols:
        X_cat = sm.add_constant(
            df[model_cols].values.astype(float), has_constant='add'
        )
        res_cat = sm.OLS(y, X_cat).fit()
        # 전체 범주 F-test (절편 제외)
        f_stat   = res_cat.fvalue
        f_pvalue = res_cat.f_pvalue
        results.append({
            'Variable':  'ManufacturerModelName (F-test)',
            'β':         'N/A (범주형)',
            'CI_Lower':  'N/A',
            'CI_Upper':  'N/A',
            'SE':        'N/A',
            't_stat':    f'F={round(f_stat,3)}',
            'p_value':   round(f_pvalue, 6),
            'R2':        round(res_cat.rsquared, 4),
            'Adj_R2':    round(res_cat.rsquared_adj, 4),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 다변량 분석
# ─────────────────────────────────────────────────────────────────────────────

def run_multivariate(df: pd.DataFrame, df_raw: pd.DataFrame,
                     feature_cols: list = None) -> tuple:
    """
    다중 OLS 회귀.
    feature_cols=None 이면 전체 변수(Sex+Age+AEC+Model) 사용.
    반환: (계수 DataFrame, 모델 요약 dict, statsmodels 결과 객체)
    """
    y = df['TAMA'].values

    if feature_cols is None:
        # 기본: 전체 feature (Sex + Age_z + AEC_z + Model 더미)
        all_feat = ['Sex', 'Age_z']
        all_feat += [f + '_z' for f in config.SELECTED_AEC_FEATURES]
        all_feat += sorted([c for c in df.columns if c.startswith('Model_')])
    else:
        all_feat = list(feature_cols)

    all_feat = [c for c in all_feat if c in df.columns]

    X = df[all_feat].values.astype(float)
    X_c = sm.add_constant(X, has_constant='add')
    res = sm.OLS(y, X_c).fit()

    # 계수 테이블
    ci = res.conf_int()
    param_names = ['Intercept'] + all_feat
    coef_rows = []
    for i, name in enumerate(param_names):
        coef_rows.append({
            'Variable': name,
            'β':        round(res.params[i], 4),
            'CI_Lower': round(ci[i, 0], 4),
            'CI_Upper': round(ci[i, 1], 4),
            'SE':       round(res.bse[i], 4),
            't_stat':   round(res.tvalues[i], 4),
            'p_value':  round(res.pvalues[i], 6),
        })
    coef_df = pd.DataFrame(coef_rows)

    # 예측 성능
    y_pred = res.predict(X_c)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae  = float(mean_absolute_error(y, y_pred))

    # 5-Fold CV R²
    cv_res = cross_validate_linear(df_raw, all_feat)

    # 모델 요약
    summary = {
        'N':             len(y),
        'R2':            round(res.rsquared, 4),
        'Adj_R2':        round(res.rsquared_adj, 4),
        'F_stat':        round(res.fvalue, 4),
        'F_pvalue':      round(res.f_pvalue, 6),
        'AIC':           round(res.aic, 2),
        'BIC':           round(res.bic, 2),
        'RMSE':          round(rmse, 4),
        'MAE':           round(mae, 4),
        'CV_R2_mean':    cv_res['CV_R2_mean'],
        'CV_R2_std':     cv_res['CV_R2_std'],
        'CV_folds':      cv_res['CV_folds'],
        'n_predictors':  len(all_feat),
    }

    # 잔차 진단
    X_diag = pd.DataFrame(X, columns=all_feat)
    diag = residual_diagnostics(res, X_diag)
    summary.update(diag)

    return coef_df, summary, res


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Part 1 - 선형 회귀 (Linear Regression)")
    print("  종속변수: TAMA (cm², 연속형)")
    print("=" * 60)

    # 데이터 준비
    print("\n[데이터 준비]")
    df     = data_loader.prepare_full(mode='linear')
    df_raw = data_loader.load_raw_data()   # CV fold용 원시 데이터

    print(f"  예측변수 (AEC): {config.SELECTED_AEC_FEATURES}")
    model_cols = sorted([c for c in df.columns if c.startswith('Model_')])
    print(f"  모델 더미 변수: {len(model_cols)}개")

    # feature 목록 정의
    aec_feat   = [f + '_z' for f in config.SELECTED_AEC_FEATURES]
    demo_feat  = ['Sex', 'Age_z']

    feat_A = aec_feat + model_cols                    # AEC + CT모델 (성별·나이 제외)
    feat_B = demo_feat + aec_feat + model_cols        # 전체 (성별·나이 포함)

    # ── 단변량 분석 ─────────────────────────────────────────────────────────
    print("\n[단변량 분석]")
    uni_df = run_univariate(df)
    print(uni_df[['Variable', 'β', 'CI_Lower', 'CI_Upper', 'p_value', 'R2']].to_string(index=False))

    # ── 다변량 분석: Model A (성별·나이 제외) ───────────────────────────────
    print("\n[다변량 분석 - Model A: AEC + CT모델 (성별·나이 제외)]")
    coef_A, summ_A, _ = run_multivariate(df, df_raw, feature_cols=feat_A)
    sig_A = coef_A[coef_A['p_value'] < 0.05]
    print(f"  ▶ 유의 계수 p<0.05 ({len(sig_A)}/{len(coef_A)}개):")
    print(sig_A[['Variable', 'β', 'CI_Lower', 'CI_Upper', 'p_value']].to_string(index=False))
    print(f"\n  R²={summ_A['R2']}  Adj_R²={summ_A['Adj_R2']}  "
          f"RMSE={summ_A['RMSE']}  MAE={summ_A['MAE']}  "
          f"CV_R²={summ_A['CV_R2_mean']}±{summ_A['CV_R2_std']}")

    # ── 다변량 분석: Model B (성별·나이 포함) ───────────────────────────────
    print("\n[다변량 분석 - Model B: AEC + 성별 + 나이 + CT모델 (전체)]")
    coef_B, summ_B, _ = run_multivariate(df, df_raw, feature_cols=feat_B)
    sig_B = coef_B[coef_B['p_value'] < 0.05]
    print(f"  ▶ 유의 계수 p<0.05 ({len(sig_B)}/{len(coef_B)}개):")
    print(sig_B[['Variable', 'β', 'CI_Lower', 'CI_Upper', 'p_value']].to_string(index=False))
    print(f"\n  R²={summ_B['R2']}  Adj_R²={summ_B['Adj_R2']}  "
          f"RMSE={summ_B['RMSE']}  MAE={summ_B['MAE']}  "
          f"CV_R²={summ_B['CV_R2_mean']}±{summ_B['CV_R2_std']}")

    # ── 성능 비교 요약 ───────────────────────────────────────────────────────
    print("\n[성능 비교 요약: 성별·나이 추가 효과]")
    cmp_keys = ['R2', 'Adj_R2', 'RMSE', 'MAE', 'AIC', 'BIC', 'CV_R2_mean', 'CV_R2_std']
    print(f"  {'지표':<18} {'Model A (AEC만)':>18} {'Model B (AEC+성별나이)':>22}")
    print(f"  {'-'*60}")
    for k in cmp_keys:
        print(f"  {k:<18} {str(summ_A[k]):>18} {str(summ_B[k]):>22}")

    print(f"\n  ΔR²  (B-A) = {round(summ_B['R2'] - summ_A['R2'], 4)}")
    print(f"  ΔRMSE(A-B) = {round(summ_A['RMSE'] - summ_B['RMSE'], 4)}")

    # 잔차 진단 (Model B 기준)
    print("\n  ▶ 잔차 진단 (Model B):")
    for k in ['SW_stat', 'SW_p', 'SW_result', 'BP_p', 'BP_result',
              'DW', 'DW_result', 'Cond_num', 'Cond_result']:
        print(f"    {k:20s}: {summ_B[k]}")

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    out_path = os.path.join(config.RESULTS_DIR, 'linear_results.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        uni_df.to_excel(writer, sheet_name='단변량_univariate', index=False)
        coef_A.to_excel(writer, sheet_name='ModelA_계수(성별나이제외)', index=False)
        coef_B.to_excel(writer, sheet_name='ModelB_계수(성별나이포함)', index=False)

        # 비교 요약 시트
        cmp_rows = []
        for k in cmp_keys + ['n_predictors', 'SW_result', 'BP_result', 'DW', 'Cond_num']:
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
        'R2':          'TAMA 분산의 설명 비율 (0~1)',
        'Adj_R2':      '변수 수 증가 페널티 적용 R² - Case 간 공정 비교',
        'F_stat':      '전체 모델 F-통계량',
        'F_pvalue':    '전체 모델 유의성 (p<0.05 = 유의)',
        'AIC':         '모델 복잡도 대비 적합도 (낮을수록 좋음)',
        'BIC':         'AIC보다 강한 복잡도 페널티',
        'RMSE':        '예측 오차 (단위: cm², 낮을수록 정확)',
        'MAE':         '이상치에 강건한 예측 오차 (단위: cm²)',
        'CV_R2_mean':  f'{config.CV_FOLDS}-Fold CV R² 평균 - 일반화 성능',
        'CV_R2_std':   f'{config.CV_FOLDS}-Fold CV R² 표준편차',
        'SW_stat':     'Shapiro-Wilk 검정 통계량',
        'SW_p':        '정규성 검정 p-value (p>0.05 = 정규 분포)',
        'SW_result':   '',
        'BP_stat':     'Breusch-Pagan 검정 통계량',
        'BP_p':        '등분산성 검정 p-value (p>0.05 = 동분산)',
        'BP_result':   '',
        'DW':          'Durbin-Watson 자기상관 (1.5~2.5 = 정상)',
        'DW_result':   '',
        'Cond_num':    '조건수 κ - 다중공선성 (κ<30 양호)',
        'Cond_result': '',
        'n_predictors': '투입된 예측변수 수',
    }
    return notes.get(key, '')


if __name__ == '__main__':
    main()
