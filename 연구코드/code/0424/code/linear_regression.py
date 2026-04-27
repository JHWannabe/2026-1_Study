"""
linear_regression.py - Part 1: 선형 회귀 (단변량 + 다변량)

분석 대상: TAMA (연속형, cm²)
예측변수 (전체): Sex, Age, 선택 AEC features, ManufacturerModelName
방법:
  - 단변량 (Univariate): 각 변수를 개별 OLS → β, 95%CI, p-value, R²
  - 다변량 (Multivariate): 전체 변수 동시 투입 OLS → 계수 테이블 + 모델 요약
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

import config as config
import data_loader as data_loader

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

def run_multivariate(df: pd.DataFrame, df_raw: pd.DataFrame) -> tuple:
    """
    전체 예측변수 동시 투입 다중 OLS.
    반환: (계수 DataFrame, 모델 요약 dict, 잔차진단 dict)
    """
    y = df['TAMA'].values

    # 전체 feature (Sex + Age_z + AEC_z + Model 더미)
    all_feat = ['Sex', 'Age_z']
    all_feat += [f + '_z' for f in config.SELECTED_AEC_FEATURES]
    all_feat += sorted([c for c in df.columns if c.startswith('Model_')])
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
    df_raw, _ = data_loader.load_raw_data()   # CV fold용 원시 데이터
    df_raw = df_raw.dropna(subset=config.SELECTED_AEC_FEATURES).reset_index(drop=True)

    print(f"  예측변수 (AEC): {config.SELECTED_AEC_FEATURES}")
    print(f"  모델 더미 변수: {len([c for c in df.columns if c.startswith('Model_')])}개")

    # ── 단변량 분석 ─────────────────────────────────────────────────────────
    print("\n[단변량 분석]")
    uni_df = run_univariate(df)
    print(uni_df[['Variable', 'β', 'CI_Lower', 'CI_Upper', 'p_value', 'R2']].to_string(index=False))

    # ── 다변량 분석 ─────────────────────────────────────────────────────────
    print("\n[다변량 분석]")
    coef_df, summary, _ = run_multivariate(df, df_raw)

    # 유의한 변수만 터미널 출력 (전체 계수는 Excel 저장)
    sig = coef_df[coef_df['p_value'] < 0.05]
    print(f"\n  ▶ 유의한 계수 (p<0.05, {len(sig)}/{len(coef_df)}개 — 전체 테이블은 Excel 참조):")
    print(sig[['Variable', 'β', 'CI_Lower', 'CI_Upper', 'p_value']].to_string(index=False))

    print("\n  ▶ 모델 성능:")
    for k in ['N', 'R2', 'Adj_R2', 'F_stat', 'F_pvalue', 'RMSE', 'MAE',
              'AIC', 'BIC', 'CV_R2_mean', 'CV_R2_std']:
        print(f"    {k:20s}: {summary[k]}")

    print("\n  ▶ 잔차 진단:")
    for k in ['SW_stat', 'SW_p', 'SW_result', 'BP_p', 'BP_result',
              'DW', 'DW_result', 'Cond_num', 'Cond_result']:
        print(f"    {k:20s}: {summary[k]}")

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    out_path = os.path.join(config.RESULTS_DIR, 'linear_results.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        uni_df.to_excel(writer, sheet_name='단변량_univariate', index=False)
        coef_df.to_excel(writer, sheet_name='다변량_coefficients', index=False)

        # 모델 요약을 단일 컬럼 테이블로
        summary_df = pd.DataFrame([
            {'항목': k, '값': v, '근거': _summary_note(k)}
            for k, v in summary.items()
        ])
        summary_df.to_excel(writer, sheet_name='다변량_model_summary', index=False)

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
