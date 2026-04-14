"""
Multivariable Analysis Pipeline
- 연속형 outcome  → 다변량 선형 회귀
- 이진형 outcome  → 다변량 로지스틱 회귀
- 변수 선택 전략  → Univariable screening (p < 0.05 또는 p < 0.1) → Multivariable
- 의학 연구 표준 보고 형식 (Table 2 스타일) 출력
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. 데이터 로드 (입력 확정 후 수정)
# ─────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, str, list[str], str]:
    """
    Returns
    -------
    df          : 전처리 완료된 DataFrame
    outcome     : 종속변수 컬럼명
    features    : 독립변수 컬럼명 리스트
    outcome_type: "continuous" 또는 "binary"
    """
    # TODO: 실제 데이터 경로 및 컬럼명으로 교체
    raise NotImplementedError("load_data()를 구현하세요.")


# ─────────────────────────────────────────────
# 2. 변수 선택: Univariable Screening
# ─────────────────────────────────────────────
def univariable_screening(
    df: pd.DataFrame,
    outcome: str,
    features: list[str],
    outcome_type: str,
    threshold: float = 0.05,
) -> tuple[pd.DataFrame, list[str]]:
    """
    단변량 분석 후 p < threshold 변수를 다변량 후보로 선택.
    의학 연구에서는 threshold=0.1 또는 0.2 를 쓰기도 함.
    """
    rows = []
    y = df[outcome]

    for feat in features:
        X = sm.add_constant(df[feat])

        if outcome_type == "binary":
            model = sm.Logit(y, X).fit(disp=False)
            coef  = model.params[feat]
            p_val = model.pvalues[feat]
            ci    = model.conf_int().loc[feat]
            stat  = {"effect": np.exp(coef), "effect_name": "OR",
                     "ci_lo": np.exp(ci[0]), "ci_hi": np.exp(ci[1])}
        else:
            model = sm.OLS(y, X).fit()
            coef  = model.params[feat]
            p_val = model.pvalues[feat]
            ci    = model.conf_int().loc[feat]
            stat  = {"effect": coef, "effect_name": "β",
                     "ci_lo": ci[0], "ci_hi": ci[1]}

        rows.append({
            "Variable":      feat,
            stat["effect_name"]: round(stat["effect"], 4),
            "95% CI":        f"{stat['ci_lo']:.3f} – {stat['ci_hi']:.3f}",
            "P-value":       round(p_val, 4),
            "Selected":      "✓" if p_val < threshold else "",
        })

    result_df = pd.DataFrame(rows).sort_values("P-value")
    selected  = result_df.loc[result_df["Selected"] == "✓", "Variable"].tolist()
    return result_df, selected


# ─────────────────────────────────────────────
# 3. 다변량 분석 (공통 인터페이스)
# ─────────────────────────────────────────────
def multivariable_analysis(
    df: pd.DataFrame,
    outcome: str,
    features: list[str],
    outcome_type: str,
) -> dict:
    y = df[outcome]
    X = sm.add_constant(df[features])

    if outcome_type == "binary":
        model = sm.Logit(y, X).fit(disp=False)
    else:
        model = sm.OLS(y, X).fit()

    coefs = model.params.drop("const", errors="ignore")
    ci    = model.conf_int().drop("const", errors="ignore")
    pvals = model.pvalues.drop("const", errors="ignore")

    if outcome_type == "binary":
        effect      = np.exp(coefs)
        effect_lo   = np.exp(ci[0])
        effect_hi   = np.exp(ci[1])
        effect_name = "Odds Ratio"
    else:
        effect      = coefs
        effect_lo   = ci[0]
        effect_hi   = ci[1]
        effect_name = "Coefficient (β)"

    coef_table = pd.DataFrame({
        effect_name:    effect,
        "95% CI Lower": effect_lo,
        "95% CI Upper": effect_hi,
        "P-value":      pvals,
    }).round(4)

    metrics = _compute_metrics(model, y, X, outcome_type)

    return {
        "model":        model,
        "coef_table":   coef_table,
        "metrics":      metrics,
        "outcome_type": outcome_type,
        "y_true":       y,
        "y_prob":       model.predict(X) if outcome_type == "binary" else None,
        "y_fitted":     model.fittedvalues,
    }


def _compute_metrics(model, y, X, outcome_type: str) -> dict:
    metrics: dict = {"N": len(y), "AIC": round(model.aic, 4), "BIC": round(model.bic, 4)}

    if outcome_type == "continuous":
        residuals = model.resid
        metrics.update({
            "R²":          round(model.rsquared, 4),
            "Adjusted R²": round(model.rsquared_adj, 4),
            "F-statistic": round(model.fvalue, 4),
            "F p-value":   round(model.f_pvalue, 4),
            "RMSE":        round(np.sqrt(np.mean(residuals**2)), 4),
            "MAE":         round(np.mean(np.abs(residuals)), 4),
        })
    else:
        y_prob = model.predict(X)
        y_pred = (y_prob >= 0.5).astype(int)
        auc    = roc_auc_score(y, y_prob)
        brier  = brier_score_loss(y, y_prob)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        hl_stat, hl_p = _hosmer_lemeshow(y.values, y_prob.values)
        nagelkerke    = _nagelkerke_r2(model, len(y))

        metrics.update({
            "Events (outcome=1)":  int(y.sum()),
            "AUC-ROC":             round(auc, 4),
            "Brier Score":         round(brier, 4),
            "Sensitivity":         round(sensitivity, 4),
            "Specificity":         round(specificity, 4),
            "PPV":                 round(ppv, 4),
            "NPV":                 round(npv, 4),
            "Nagelkerke R²":       round(nagelkerke, 4),
            "Hosmer-Lemeshow χ²":  round(hl_stat, 4),
            "Hosmer-Lemeshow p":   round(hl_p, 4),
        })

    return metrics


def _hosmer_lemeshow(y_true: np.ndarray, y_prob: np.ndarray, g: int = 10):
    df_hl = pd.DataFrame({"y": y_true, "prob": y_prob})
    df_hl["decile"] = pd.qcut(df_hl["prob"], q=g, duplicates="drop", labels=False)
    obs_list, exp_list = [], []
    for _, grp in df_hl.groupby("decile"):
        obs_list.append([grp["y"].sum(), len(grp) - grp["y"].sum()])
        exp_list.append([grp["prob"].sum(), len(grp) - grp["prob"].sum()])
    obs, exp = np.array(obs_list), np.array(exp_list)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((obs - exp) ** 2 / np.where(exp == 0, np.nan, exp))
    return chi2, 1 - stats.chi2.cdf(chi2, df=g - 2)


def _nagelkerke_r2(model, n: int) -> float:
    r2_cs = 1 - np.exp((2 / n) * (model.llnull - model.llf))
    r2_max = 1 - np.exp((2 / n) * model.llnull)
    return r2_cs / r2_max if r2_max != 0 else np.nan


# ─────────────────────────────────────────────
# 4. 의학 논문 Table 스타일 출력
# ─────────────────────────────────────────────
def build_paper_table(uni_df: pd.DataFrame, multi_result: dict, outcome_type: str) -> pd.DataFrame:
    """
    Univariable + Multivariable 결과를 하나의 테이블로 병합.
    의학 저널 Table 2 형식.
    """
    effect_col = "OR" if outcome_type == "binary" else "β"
    ct = multi_result["coef_table"].copy()
    col_effect = "Odds Ratio" if outcome_type == "binary" else "Coefficient (β)"

    # Multivariable 컬럼 이름 정리
    ct = ct.rename(columns={
        col_effect:      f"Multi {effect_col}",
        "95% CI Lower":  "Multi CI Lo",
        "95% CI Upper":  "Multi CI Hi",
        "P-value":       "Multi P",
    })
    ct["Multi 95% CI"] = ct.apply(
        lambda r: f"{r['Multi CI Lo']:.3f} – {r['Multi CI Hi']:.3f}", axis=1
    )

    # Univariable 컬럼 정리
    uni = uni_df[["Variable", effect_col, "95% CI", "P-value"]].rename(columns={
        effect_col:  f"Uni {effect_col}",
        "95% CI":    "Uni 95% CI",
        "P-value":   "Uni P",
    }).set_index("Variable")

    table = uni.join(
        ct[[f"Multi {effect_col}", "Multi 95% CI", "Multi P"]],
        how="left"
    ).reset_index()

    return table


# ─────────────────────────────────────────────
# 5. 시각화
# ─────────────────────────────────────────────
def plot_forest(coef_table: pd.DataFrame, outcome_type: str, save_path: str = None):
    """Forest plot (OR or β with 95% CI)"""
    col_effect = "Odds Ratio" if outcome_type == "binary" else "Coefficient (β)"
    ref_line   = 1.0 if outcome_type == "binary" else 0.0

    vars_  = coef_table.index.tolist()
    effects = coef_table[col_effect].values
    lo      = coef_table["95% CI Lower"].values
    hi      = coef_table["95% CI Upper"].values
    pvals   = coef_table["P-value"].values

    y_pos = np.arange(len(vars_))
    colors = ["#c0392b" if p < 0.05 else "#2980b9" for p in pvals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(vars_) * 0.55)))
    for i, (eff, l, h, c) in enumerate(zip(effects, lo, hi, colors)):
        ax.plot([l, h], [i, i], color=c, lw=1.5)
        ax.plot(eff, i, 'o', color=c, ms=7)

    ax.axvline(ref_line, color='gray', linestyle='--', lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(vars_)
    ax.set_xlabel(col_effect)
    ax.set_title("Forest Plot — Multivariable Analysis\n(red: p < 0.05, blue: p ≥ 0.05)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# 6. 결과 출력
# ─────────────────────────────────────────────
def print_results(uni_df: pd.DataFrame, multi_result: dict, paper_table: pd.DataFrame):
    print("=" * 70)
    print("[ Step 1: Univariable Screening ]")
    print("=" * 70)
    print(uni_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("[ Step 2: Multivariable Analysis — Model Metrics ]")
    print("=" * 70)
    for k, v in multi_result["metrics"].items():
        print(f"  {k:<30}: {v}")

    print("\n[ Step 2: Multivariable Analysis — Effect Estimates ]")
    print(multi_result["coef_table"].to_string())

    print("\n" + "=" * 70)
    print("[ Combined Table (Paper Format) ]")
    print("=" * 70)
    print(paper_table.to_string(index=False))


# ─────────────────────────────────────────────
# 7. 실행 진입점
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df, outcome, features, outcome_type = load_data()

    # Step 1: Univariable screening
    uni_df, selected_features = univariable_screening(
        df, outcome, features, outcome_type, threshold=0.05
    )

    if not selected_features:
        print("p < 0.05 변수가 없습니다. threshold를 0.1 또는 0.2로 조정하세요.")
        selected_features = features  # fallback: 전체 변수 투입

    # Step 2: Multivariable analysis
    multi_result = multivariable_analysis(df, outcome, selected_features, outcome_type)

    # 논문 형식 테이블
    paper_table = build_paper_table(uni_df, multi_result, outcome_type)

    # 출력
    print_results(uni_df, multi_result, paper_table)

    # 시각화
    plot_forest(multi_result["coef_table"], outcome_type, save_path="forest_plot.png")

    # 저장
    paper_table.to_csv("multivariable_paper_table.csv", index=False)
    multi_result["coef_table"].to_csv("multivariable_coef.csv")
