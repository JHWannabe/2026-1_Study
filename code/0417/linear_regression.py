"""
Linear Regression Analysis
- p-value, confidence interval, R², adjusted R², RMSE, MAE 등 의학 연구 지표 포함
- 입력 구성이 확정되면 load_data() 함수만 수정하면 됨
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. 데이터 로드 (입력 확정 후 수정)
# ─────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, str, list[str]]:
    """
    Returns
    -------
    df       : 전처리 완료된 DataFrame
    outcome  : 종속변수(연속형) 컬럼명
    features : 독립변수 컬럼명 리스트
    """
    # 데이터 로드
    file_path = r"C:\Users\user\Desktop\Study\data\AEC\강남\강남_merged_features.xlsx"
    
    # metadata 시트에서 PatientID와 TAMA 추출
    df_meta = pd.read_excel(file_path, sheet_name="metadata", usecols=["PatientID", "TAMA"])
    
    # features 시트에서 모든 피처 추출
    df_feat = pd.read_excel(file_path, sheet_name="features")
    
    # PatientID 기준으로 merge
    df = pd.merge(df_feat, df_meta, on="PatientID", how="inner")
    
    # NaN 제거
    df = df.dropna()
    
    # PatientID는 제거 (모델 입력에 불필요)
    df = df.drop(columns=["PatientID"])
    
    outcome = "TAMA"
    features = [col for col in df.columns if col != outcome]
    
    return df, outcome, features


# ─────────────────────────────────────────────
# 2. 단변량 선형 회귀 (Univariable)
# ─────────────────────────────────────────────
def univariable_linear(df: pd.DataFrame, outcome: str, features: list[str]) -> pd.DataFrame:
    rows = []
    y = df[outcome]

    for feat in features:
        x = df[feat]
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        coef   = model.params[feat]
        se     = model.bse[feat]
        t_stat = model.tvalues[feat]
        p_val  = model.pvalues[feat]
        ci_lo, ci_hi = model.conf_int().loc[feat]

        rows.append({
            "Variable":    feat,
            "Coefficient": float(f"{coef:.4e}"),
            "SE":          float(f"{se:.4e}"),
            "t-statistic": float(f"{t_stat:.4e}"),
            "P-value":     float(f"{p_val:.4e}"),
            "95% CI Lower": float(f"{ci_lo:.4e}"),
            "95% CI Upper": float(f"{ci_hi:.4e}"),
            "R²":          float(f"{model.rsquared:.4e}"),
        })

    result = pd.DataFrame(rows).sort_values("P-value")
    return result


# ─────────────────────────────────────────────
# 3. 다변량 선형 회귀 (Multivariable)
# ─────────────────────────────────────────────
def multivariable_linear(df: pd.DataFrame, outcome: str, features: list[str]) -> dict:
    y = df[outcome]
    X = sm.add_constant(df[features])
    print(f"Running multivariable linear regression with features: {features}")
    model = sm.OLS(y, X).fit()

    # 계수 테이블
    coef_table = pd.DataFrame({
        "Coefficient":  model.params,
        "SE":           model.bse,
        "t-statistic":  model.tvalues,
        "P-value":      model.pvalues,
        "95% CI Lower": model.conf_int()[0],
        "95% CI Upper": model.conf_int()[1],
    }).round(4).drop(index="const", errors="ignore")

    # 모델 성능 지표
    residuals = model.resid
    n = len(y)
    k = len(features)

    rmse = np.sqrt(np.mean(residuals**2))
    mae  = np.mean(np.abs(residuals))

    # Breusch-Pagan 등분산성 검정
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
    # Durbin-Watson 자기상관 검정
    dw_stat = durbin_watson(residuals)

    model_metrics = {
        "N":                    n,
        "R²":                   round(model.rsquared, 4),
        "Adjusted R²":          round(model.rsquared_adj, 4),
        "F-statistic":          round(model.fvalue, 4),
        "F p-value":            round(model.f_pvalue, 4),
        "AIC":                  round(model.aic, 4),
        "BIC":                  round(model.bic, 4),
        "RMSE":                 round(rmse, 4),
        "MAE":                  round(mae, 4),
        "Durbin-Watson":        round(dw_stat, 4),
        "Breusch-Pagan p":      round(bp_p, 4),  # <0.05 → 이분산 의심
    }

    return {"model": model, "coef_table": coef_table, "metrics": model_metrics}


# ─────────────────────────────────────────────
# 4. 잔차 진단 플롯
# ─────────────────────────────────────────────
def plot_diagnostics(model, save_path: str = None):
    fitted   = model.fittedvalues
    residuals = model.resid
    std_resid = (residuals - residuals.mean()) / residuals.std()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Residuals vs Fitted
    axes[0].scatter(fitted, residuals, alpha=0.5, s=20)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    # Q-Q plot
    stats.probplot(residuals, plot=axes[1])
    axes[1].set_title("Normal Q-Q Plot")

    # Scale-Location
    axes[2].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.5, s=20)
    axes[2].set_xlabel("Fitted values")
    axes[2].set_ylabel("√|Standardized Residuals|")
    axes[2].set_title("Scale-Location")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 5. 결과 출력
# ─────────────────────────────────────────────
def print_results(uni_result: pd.DataFrame, multi_result: dict):
    print("=" * 60)
    print("[ Univariable Linear Regression ]")
    print("=" * 60)
    print(uni_result.to_string(index=False))

    print("\n" + "=" * 60)
    print("[ Multivariable Linear Regression — Model Metrics ]")
    print("=" * 60)
    for k, v in multi_result["metrics"].items():
        print(f"  {k:<22}: {v}")

    print("\n[ Multivariable Linear Regression — Coefficients ]")
    print(multi_result["coef_table"].to_string())


# ─────────────────────────────────────────────
# 6. 실행 진입점
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df, outcome, features = load_data()

    # 단변량 분석: 모든 피처 사용
    uni = univariable_linear(df, outcome, features)

    # 상위 3-5개 피처 추출 (p-value 기준, 낮을수록 유의미)
    top_features = uni.head(5)["Variable"].tolist()  # 상위 5개 (3-5개 범위)
    print(f"\n=> Top {len(top_features)} features for multivariable analysis: {top_features}\n")

    # 다변량 분석: 상위 피처만 사용
    multi = multivariable_linear(df, outcome, top_features)

    print_results(uni, multi)
    plot_diagnostics(multi["model"], save_path=r"C:\Users\user\Desktop\Study\result\0417\강남\linear_diagnostics.png")

    # 결과 저장 (p-value를 지수형으로)
    uni.to_csv(r"C:\Users\user\Desktop\Study\result\0417\강남\linear_univariable.csv", index=False, float_format='%.4e')
    multi["coef_table"].to_csv(r"C:\Users\user\Desktop\Study\result\0417\강남\linear_multivariable.csv", float_format='%.4e')
