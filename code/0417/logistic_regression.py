"""
Logistic Regression Analysis (Binary Outcome)
- p-value, Odds Ratio, 95% CI, AUC-ROC, Hosmer-Lemeshow 검정 등
- 입력 구성이 확정되면 load_data() 함수만 수정하면 됨
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    brier_score_loss
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
    outcome  : 종속변수(이진형) 컬럼명
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
    
    # TAMA를 중앙값 기준으로 이진화 (0/1)
    tama_median = df["TAMA"].median()
    df["TAMA_binary"] = (df["TAMA"] >= tama_median).astype(int)
    df = df.drop(columns=["PatientID", "TAMA"])
    
    outcome = "TAMA_binary"
    features = [col for col in df.columns if col != outcome]
    
    return df, outcome, features


# ─────────────────────────────────────────────
# 2. 단변량 로지스틱 회귀 (Univariable)
# ─────────────────────────────────────────────
def univariable_logistic(df: pd.DataFrame, outcome: str, features: list[str]) -> pd.DataFrame:
    rows = []
    y = df[outcome]

    for feat in features:
        X = sm.add_constant(df[feat])
        model = sm.Logit(y, X).fit(disp=False)

        coef   = model.params[feat]
        p_val  = model.pvalues[feat]
        ci_lo, ci_hi = model.conf_int().loc[feat]

        or_val  = np.exp(coef)
        or_lo   = np.exp(ci_lo)
        or_hi   = np.exp(ci_hi)

        rows.append({
            "Variable":     feat,
            "Coefficient":  float(f"{coef:.4e}"),
            "Odds Ratio":   float(f"{or_val:.4e}"),
            "95% CI Lower": float(f"{or_lo:.4e}"),
            "95% CI Upper": float(f"{or_hi:.4e}"),
            "P-value":      float(f"{p_val:.4e}"),
        })

    result = pd.DataFrame(rows).sort_values("P-value")
    return result


# ─────────────────────────────────────────────
# 3. 다변량 로지스틱 회귀 (Multivariable)
# ─────────────────────────────────────────────
def multivariable_logistic(df: pd.DataFrame, outcome: str, features: list[str]) -> dict:
    y = df[outcome]
    X = sm.add_constant(df[features])
    model = sm.Logit(y, X).fit(disp=False)

    coefs  = model.params.drop("const", errors="ignore")
    ci     = model.conf_int().drop("const", errors="ignore")
    pvals  = model.pvalues.drop("const", errors="ignore")

    coef_table = pd.DataFrame({
        "Coefficient":  coefs,
        "Odds Ratio":   np.exp(coefs),
        "95% CI Lower": np.exp(ci[0]),
        "95% CI Upper": np.exp(ci[1]),
        "P-value":      pvals,
    }).round(4)

    # ── 예측 확률 및 분류 지표 ──────────────────
    y_prob = model.predict(X)
    y_pred = (y_prob >= 0.5).astype(int)

    auc  = roc_auc_score(y, y_prob)
    brier = brier_score_loss(y, y_prob)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    # ── Hosmer-Lemeshow 검정 (10 그룹) ─────────
    hl_stat, hl_p = hosmer_lemeshow_test(y.values, y_prob.values)

    # ── VIF (다중공선성) ────────────────────────
    vif_df = calc_vif(df[features])

    # ── Nagelkerke R² ──────────────────────────
    nagelkerke_r2 = calc_nagelkerke_r2(model, len(y))

    model_metrics = {
        "N":                   len(y),
        "Events (outcome=1)":  int(y.sum()),
        "AUC-ROC":             round(auc, 4),
        "Brier Score":         round(brier, 4),  # 낮을수록 좋음
        "Sensitivity":         round(sensitivity, 4),
        "Specificity":         round(specificity, 4),
        "PPV":                 round(ppv, 4),
        "NPV":                 round(npv, 4),
        "Nagelkerke R²":       round(nagelkerke_r2, 4),
        "Log-Likelihood":      round(model.llf, 4),
        "AIC":                 round(model.aic, 4),
        "BIC":                 round(model.bic, 4),
        "Hosmer-Lemeshow χ²":  round(hl_stat, 4),
        "Hosmer-Lemeshow p":   round(hl_p, 4),  # >0.05 → 적합
    }

    return {
        "model":        model,
        "coef_table":   coef_table,
        "metrics":      model_metrics,
        "vif":          vif_df,
        "y_prob":       y_prob,
        "y_true":       y,
    }


# ─────────────────────────────────────────────
# 4. 보조 함수들
# ─────────────────────────────────────────────
def hosmer_lemeshow_test(y_true: np.ndarray, y_prob: np.ndarray, g: int = 10):
    """Hosmer-Lemeshow goodness-of-fit test"""
    df_hl = pd.DataFrame({"y": y_true, "prob": y_prob})
    df_hl["decile"] = pd.qcut(df_hl["prob"], q=g, duplicates="drop", labels=False)

    obs_list, exp_list = [], []
    for _, grp in df_hl.groupby("decile"):
        obs_list.append([grp["y"].sum(), len(grp) - grp["y"].sum()])
        exp_list.append([grp["prob"].sum(), len(grp) - grp["prob"].sum()])

    obs = np.array(obs_list)
    exp = np.array(exp_list)

    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((obs - exp) ** 2 / np.where(exp == 0, np.nan, exp))

    p_val = 1 - stats.chi2.cdf(chi2, df=g - 2)
    return chi2, p_val


def calc_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor — VIF > 10 이면 다중공선성 의심"""
    vif_data = []
    X_arr = X.assign(const=1).values
    for i, col in enumerate(X.columns):
        vif_data.append({"Variable": col, "VIF": round(variance_inflation_factor(X_arr, i), 4)})
    return pd.DataFrame(vif_data)


def calc_nagelkerke_r2(model, n: int) -> float:
    """Nagelkerke pseudo R²"""
    ll_null = model.llnull
    ll_full = model.llf
    r2_cs   = 1 - np.exp((2 / n) * (ll_null - ll_full))   # Cox-Snell
    r2_max  = 1 - np.exp((2 / n) * ll_null)
    return r2_cs / r2_max if r2_max != 0 else np.nan


# ─────────────────────────────────────────────
# 5. ROC 커브 플롯
# ─────────────────────────────────────────────
def plot_roc(y_true, y_prob, save_path: str = None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 6. 결과 출력
# ─────────────────────────────────────────────
def print_results(uni_result: pd.DataFrame, multi_result: dict):
    print("=" * 60)
    print("[ Univariable Logistic Regression ]")
    print("=" * 60)
    print(uni_result.to_string(index=False))

    print("\n" + "=" * 60)
    print("[ Multivariable Logistic Regression — Model Metrics ]")
    print("=" * 60)
    for k, v in multi_result["metrics"].items():
        print(f"  {k:<28}: {v}")

    print("\n[ Multivariable Logistic Regression — Coefficients & OR ]")
    print(multi_result["coef_table"].to_string())

    print("\n[ VIF (Multicollinearity Check) ]")
    print(multi_result["vif"].to_string(index=False))


# ─────────────────────────────────────────────
# 7. 실행 진입점
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df, outcome, features = load_data()

    # 단변량 분석: 모든 피처 사용
    uni = univariable_logistic(df, outcome, features)

    # 상위 3-5개 피처 추출 (p-value 기준, 낮을수록 유의미)
    top_features = uni.head(5)["Variable"].tolist()  # 상위 5개 (3-5개 범위)
    print(f"\n=> Top {len(top_features)} features for multivariable analysis: {top_features}\n")

    # 다변량 분석: 상위 피처만 사용
    multi = multivariable_logistic(df, outcome, top_features)

    print_results(uni, multi)
    plot_roc(multi["y_true"], multi["y_prob"], save_path=r"C:\Users\user\Desktop\Study\result\0417\강남\logistic_roc.png")

    # 결과 저장 (p-value를 지수형으로)
    uni.to_csv(r"C:\Users\user\Desktop\Study\result\0417\강남\logistic_univariable.csv", index=False, float_format='%.4e')
    multi["coef_table"].to_csv(r"C:\Users\user\Desktop\Study\result\0417\강남\logistic_multivariable.csv", float_format='%.4e')
