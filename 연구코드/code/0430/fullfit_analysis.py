# -*- coding: utf-8 -*-
"""
run_fullfit_analysis(): 전체 데이터 OLS/Logit full-fit 진단 플롯 (Figs 04-15).

CV 기반 run_one_analysis()가 일반화 성능(out-of-fold)을 측정하는 것과 달리,
이 모듈은 전체 데이터에 한 번 적합(full-fit)하여:
  - 잔차 진단 (정규성·등분산성): 모델 가정 충족 여부
  - 보정도 플롯 (Hosmer-Lemeshow): 로지스틱 확률 보정 수준
  - 케이스별 R²/AUC/AIC 추이: 변수군 추가에 따른 단조 증가 여부
대표 진단 모델은 Case4(임상+AEC_prev+스캐너)를 사용.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy import stats as scipy_stats
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from matplotlib.patches import Patch

from config import AEC_PREV, CASE_LABELS, COLORS, CV_RANDOM
from helpers import sig_stars


def run_fullfit_analysis(
    X_full: pd.DataFrame,
    y_cont: pd.Series,
    CASES: dict,
    OUT_DIR: Path,
    hosp_label: str,
) -> None:
    """Full-fit OLS/Logit 진단 플롯 생성 (Figs 04-15)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    female_mask = X_full["PatientSex_enc"] == 0
    male_mask   = X_full["PatientSex_enc"] == 1
    tama_female = y_cont[female_mask].quantile(0.25)
    tama_male   = y_cont[male_mask].quantile(0.25)

    y_bin = pd.Series(0, index=y_cont.index, dtype=int)
    y_bin[female_mask] = (y_cont[female_mask] < tama_female).astype(int)
    y_bin[male_mask]   = (y_cont[male_mask]   < tama_male).astype(int)

    def avail(feats):
        return [f for f in feats if f in X_full.columns]

    def safe_std(df: pd.DataFrame) -> pd.DataFrame:
        std  = df.std()
        good = std[std > 1e-10].index.tolist()
        if not good:
            return pd.DataFrame(index=df.index)
        Xs = (df[good] - df[good].mean()) / df[good].std()
        return Xs.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Case4/5는 수십 개의 CT 더미 변수로 인해 완전 다중공선성 발생 가능 → Case2 우선 사용
    diag_key   = next(
        (k for k in ["Case2_Clinical+AEC_prev", "Case3_Clinical+AEC_new",
                     "Case4_Clinical+AEC_prev+Scanner"]
         if k in CASES),
        list(CASES.keys())[-1],
    )
    diag_feats = avail(CASES[diag_key])
    X_diag     = X_full[diag_feats].copy()
    X_diag_std = safe_std(X_diag)

    # ── Fig 04: Linear actual vs predicted (full fit OLS) ──
    X_ols     = sm.add_constant(X_diag_std)
    ols_model = sm.OLS(y_cont, X_ols).fit()
    fitted    = ols_model.fittedvalues
    residuals = ols_model.resid
    r2_full   = ols_model.rsquared
    rmse_full = float(np.sqrt(np.mean(residuals ** 2)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_cont, fitted, alpha=0.3, s=8, color="#e07070", label="환자", rasterized=True)
    lo_v = min(float(y_cont.min()), float(fitted.min())) - 5
    hi_v = max(float(y_cont.max()), float(fitted.max())) + 5
    ax.plot([lo_v, hi_v], [lo_v, hi_v], "k--", lw=1.5, label="완벽 예측 (y=x)")
    z_v  = np.polyfit(y_cont, fitted, 1)
    xln  = np.linspace(lo_v, hi_v, 200)
    ax.plot(xln, np.poly1d(z_v)(xln), color="#3498db", lw=2, label="회귀 직선")
    ax.set_xlabel("실제 TAMA (cm²)"); ax.set_ylabel("예측 TAMA (cm²)")
    ax.set_title(f"선형 회귀: 실제 vs 예측 TAMA\nR²={r2_full:.3f}, RMSE={rmse_full:.2f} cm²")
    ax.legend(fontsize=9)
    plt.tight_layout(); fig.savefig(OUT_DIR / "04_linear_actual_vs_pred.png", dpi=150); plt.close()

    # ── Fig 05: Residual diagnostics (4-Panel) ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.2, s=6, color="#e07070", rasterized=True)
    ax.axhline(0, color="gray", lw=0.8)
    lf1 = sm_lowess(residuals.values, fitted.values, frac=0.3)
    ax.plot(lf1[:, 0], lf1[:, 1], color="#3498db", lw=2)
    ax.set_xlabel("Fitted Values"); ax.set_ylabel("Residuals"); ax.set_title("(a) Residuals vs Fitted")

    ax = axes[0, 1]
    (osm, osr), (sl_qq, ic_qq, r_qq) = scipy_stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, alpha=0.3, s=6, color="#e07070", rasterized=True)
    lo_qq = float(min(osm)); hi_qq = float(max(osm))
    ax.plot([lo_qq, hi_qq], [sl_qq * lo_qq + ic_qq, sl_qq * hi_qq + ic_qq],
            "r--", lw=1.5, label=f"r={r_qq:.3f}")
    ax.set_xlabel("이론적 분위수 (Normal Quantiles)"); ax.set_ylabel("실제 분위수 (Sample Quantiles)")
    ax.set_title("(b) Normal Q-Q Plot"); ax.legend(fontsize=9)

    ax = axes[1, 0]
    sqrt_res = np.sqrt(np.abs(residuals / residuals.std()))
    ax.scatter(fitted, sqrt_res, alpha=0.2, s=6, color="#e08040", rasterized=True)
    lf2 = sm_lowess(sqrt_res.values, fitted.values, frac=0.3)
    ax.plot(lf2[:, 0], lf2[:, 1], color="#3498db", lw=2)
    ax.set_xlabel("Fitted Values"); ax.set_ylabel("|Standardized Residuals|^0.5")
    ax.set_title("(c) Scale-Location (등분산성 확인)")

    ax = axes[1, 1]
    resid_arr = residuals.values
    sample    = resid_arr[:5000] if len(resid_arr) > 5000 else resid_arr
    W, p_sw   = shapiro(sample)
    ax.hist(resid_arr, bins=40, density=True, color="#7fb3d6", edgecolor="white", linewidth=0.3)
    xr = np.linspace(resid_arr.min(), resid_arr.max(), 200)
    ax.plot(xr, scipy_stats.norm.pdf(xr, resid_arr.mean(), resid_arr.std()),
            "r-", lw=2, label="정규분포 곡선")
    ax.set_xlabel("Residuals"); ax.set_ylabel("밀도")
    ax.set_title(f"(d) 잔차 분포\nShapiro-Wilk: W={W:.4f}, p={p_sw:.2e}")
    ax.legend(fontsize=8)

    plt.suptitle(f"[{hosp_label}] 선형 회귀 잔차 진단 (4-Panel Diagnostic)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(OUT_DIR / "05_linear_residuals.png", dpi=150); plt.close()

    # ── Fig 06: Linear forest plot (p < 0.05 only) ──
    ci_ols  = ols_model.conf_int()
    coef_df = pd.DataFrame({
        "feature": list(X_diag_std.columns),
        "coef":    ols_model.params.iloc[1:].values,
        "ci_lo":   ci_ols.iloc[1:, 0].values,
        "ci_hi":   ci_ols.iloc[1:, 1].values,
        "pval":    ols_model.pvalues.iloc[1:].values,
    })
    sig_df = coef_df[coef_df["pval"] < 0.05].sort_values("coef").reset_index(drop=True)
    if len(sig_df) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(sig_df) * 0.7 + 1.5)))
        fp_colors = ["#e07070" if c > 0 else "#7090d0" for c in sig_df["coef"]]
        ax.barh(range(len(sig_df)), sig_df["coef"].values,
                xerr=[sig_df["coef"].values - sig_df["ci_lo"].values,
                      sig_df["ci_hi"].values - sig_df["coef"].values],
                color=fp_colors, capsize=4, edgecolor="white", height=0.6)
        ax.set_yticks(range(len(sig_df))); ax.set_yticklabels(sig_df["feature"].values)
        ax.axvline(0, color="gray", lw=1.5, ls="--")
        span = float(sig_df["ci_hi"].max() - sig_df["ci_lo"].min())
        for i, row in sig_df.iterrows():
            ax.text(row["ci_hi"] + span * 0.02, i,
                    sig_stars(row["pval"]), va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("β 계수 (95% CI, 표준화 단위)")
        ax.set_title("선형 회귀 - 유의한 계수 Forest Plot\n(p<0.05, TAMA cm² 변화량 per 1 SD)")
        plt.tight_layout(); fig.savefig(OUT_DIR / "06_linear_forest.png", dpi=150); plt.close()

    # ── Fig 07: Linear univariate R² per feature ──
    uni_feats = avail(["PatientAge", "PatientSex_enc", "BMI"] + AEC_PREV)
    uni_rows  = []
    for f in uni_feats:
        xf = safe_std(X_full[[f]])
        if xf.empty:
            continue
        m_uni = sm.OLS(y_cont, sm.add_constant(xf)).fit()
        uni_rows.append({"feature": f, "r2": m_uni.rsquared, "pval": m_uni.pvalues.iloc[1]})
    uni_df = pd.DataFrame(uni_rows).sort_values("r2")
    lbl_map07 = {"PatientSex_enc": "Sex (M=1, F=0)", "PatientAge": "Age (표준화)", "BMI": "BMI (표준화)"}
    uni_df["label"] = uni_df["feature"].apply(lambda f: lbl_map07.get(f, f"AEC: {f} (표준화)"))

    fig, ax = plt.subplots(figsize=(9, max(4, len(uni_df) * 0.65 + 1)))
    cols_uni = ["#e07070" if p < 0.05 else "#aaaaaa" for p in uni_df["pval"]]
    ax.barh(uni_df["label"], uni_df["r2"], color=cols_uni, height=0.6)
    for i, (_, row) in enumerate(uni_df.iterrows()):
        ax.text(row["r2"] + 0.002, i, f"{row['r2']:.3f}", va="center", fontsize=9)
    ax.legend(handles=[Patch(color="#e07070", label="p<0.05 유의"),
                       Patch(color="#aaaaaa", label="p≥0.05 비유의")], fontsize=9)
    ax.set_xlabel("단변량 R² (TAMA 설명 분산 비율)")
    ax.set_title("선형 회귀 - 단변량 R² 비교\n(빨간색: p<0.05 유의)")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    plt.tight_layout(); fig.savefig(OUT_DIR / "07_linear_univariate_r2.png", dpi=150); plt.close()

    # ── Fig 08: Logistic ROC with bootstrap CI (n=1000) ──
    pipe_log_full = Pipeline([
        ("sc", StandardScaler()),
        ("m",  LogisticRegression(max_iter=2000, random_state=CV_RANDOM, solver="lbfgs")),
    ])
    pipe_log_full.fit(X_diag, y_bin)
    prob_full = pipe_log_full.predict_proba(X_diag)[:, 1]
    base_auc  = roc_auc_score(y_bin, prob_full)
    fpr_base, tpr_base, thr_roc = roc_curve(y_bin, prob_full)

    rng_bt   = np.random.RandomState(42)
    n_boot   = 1000
    boot_aucs, tprs_bt = [], []
    mfpr_bt  = np.linspace(0, 1, 200)
    for _ in range(n_boot):
        idx_bt = rng_bt.randint(0, len(y_bin), len(y_bin))
        if y_bin.iloc[idx_bt].nunique() < 2:
            continue
        boot_aucs.append(roc_auc_score(y_bin.iloc[idx_bt], prob_full[idx_bt]))
        fb, tb, _ = roc_curve(y_bin.iloc[idx_bt], prob_full[idx_bt])
        tprs_bt.append(np.interp(mfpr_bt, fb, tb))
    ci_lo_a     = float(np.percentile(boot_aucs, 2.5))
    ci_hi_a     = float(np.percentile(boot_aucs, 97.5))
    mean_tpr_bt = np.mean(tprs_bt, axis=0)
    std_tpr_bt  = np.std(tprs_bt, axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_base, tpr_base, color="#e07070", lw=2,
            label=f"AUC = {base_auc:.3f} (95%CI: {ci_lo_a:.3f}-{ci_hi_a:.3f})")
    ax.fill_between(mfpr_bt, mean_tpr_bt - std_tpr_bt, mean_tpr_bt + std_tpr_bt,
                    alpha=0.2, color="#e07070")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="무작위 (AUC=0.5)")
    ax.set_xlabel("1 - Specificity (FPR)"); ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(f"로지스틱 회귀 - ROC 곡선 (전체 모델)\nBootstrap 95%CI (n={n_boot})")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(OUT_DIR / "08_logistic_roc.png", dpi=150); plt.close()

    # ── Fig 09: Calibration plot (Hosmer-Lemeshow) ──
    n_bins_hl = 10
    df_cal = pd.DataFrame({"prob": prob_full, "y": y_bin.values})
    df_cal["decile"] = pd.qcut(df_cal["prob"], n_bins_hl, duplicates="drop")
    grps     = df_cal.groupby("decile", observed=True)
    mean_pd  = grps["prob"].mean()
    obs_prop = grps["y"].mean()
    n_g      = grps["y"].count()
    obs_sum  = grps["y"].sum()
    exp_sum  = grps["prob"].sum()
    chi2_hl  = ((obs_sum - exp_sum) ** 2 / (exp_sum * (1 - exp_sum / n_g))).sum()
    p_hl     = 1 - scipy_stats.chi2.cdf(chi2_hl, df=max(len(grps) - 2, 1))
    cal_status = "보정 양호" if p_hl >= 0.05 else "보정 불량"

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(mean_pd, obs_prop, s=n_g * 0.5, color="#e08040", zorder=5, label="관측 비율")
    z_cal = np.polyfit(mean_pd, obs_prop, 1)
    xc    = np.linspace(0, float(mean_pd.max()) + 0.05, 100)
    ax.plot(xc, np.poly1d(z_cal)(xc), color="#e07070", lw=2, label="회귀 직선")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="완벽 보정 (y=x)")
    ax.set_xlabel("예측 확률 (Predicted Probability)")
    ax.set_ylabel("관측 비율 (Observed Proportion)")
    ax.set_title(f"Calibration Plot (Hosmer-Lemeshow)\nHL χ²={chi2_hl:.2f}, p={p_hl:.4f} ({cal_status})")
    ax.text(0.05, 0.9, "원 크기 = 그룹 내 환자 수", transform=ax.transAxes, fontsize=9, color="gray")
    ax.legend(fontsize=9); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(OUT_DIR / "09_logistic_calibration.png", dpi=150); plt.close()

    # ── Fig 10: Confusion matrix (Youden optimal threshold) ──
    youden_j = tpr_base - fpr_base
    opt_idx  = int(np.argmax(youden_j))
    opt_thr  = float(thr_roc[opt_idx])
    pred_opt = (prob_full >= opt_thr).astype(int)
    cm_mat   = confusion_matrix(y_bin, pred_opt)
    n_tot    = len(y_bin)
    tn, fp_v, fn, tp_v = cm_mat.ravel()
    sens_v = tp_v / (tp_v + fn)   if tp_v + fn   else 0
    spec_v = tn   / (tn + fp_v)   if tn + fp_v   else 0
    ppv    = tp_v / (tp_v + fp_v) if tp_v + fp_v else 0
    npv    = tn   / (tn + fn)     if tn + fn     else 0
    cm_ann = np.array([
        [f"{tn}\n({tn/n_tot*100:.1f}%)",   f"{fp_v}\n({fp_v/n_tot*100:.1f}%)"],
        [f"{fn}\n({fn/n_tot*100:.1f}%)",   f"{tp_v}\n({tp_v/n_tot*100:.1f}%)"],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_mat, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm_ann[i, j], ha="center", va="center",
                    color="white" if cm_mat[i, j] > cm_mat.max() * 0.5 else "black",
                    fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["예측: Normal", "예측: Low TAMA"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["실제: Normal", "실제: Low TAMA"])
    ax.set_xlabel("예측 클래스"); ax.set_ylabel("실제 클래스")
    ax.set_title(f"Confusion Matrix (임계값={opt_thr:.3f})\n"
                 f"Sensitivity={sens_v:.3f}  Specificity={spec_v:.3f}\n"
                 f"PPV={ppv:.3f}  NPV={npv:.3f}")
    plt.tight_layout(); fig.savefig(OUT_DIR / "10_logistic_confusion.png", dpi=150); plt.close()

    # ── Fig 11: Logistic crude OR forest plot ──
    or_feats  = avail(["PatientAge", "PatientSex_enc", "BMI"] + AEC_PREV)
    lbl_map11 = {"PatientAge": "Age (표준화)", "PatientSex_enc": "Sex (M=1, F=0)", "BMI": "BMI (표준화)"}
    or_rows   = []
    for f in or_feats:
        xf_std = safe_std(X_full[[f]])
        if xf_std.empty:
            continue
        try:
            log_uni = sm.Logit(y_bin, sm.add_constant(xf_std)).fit(disp=False, maxiter=200)
            ci_uni  = log_uni.conf_int()
            or_rows.append({
                "label": lbl_map11.get(f, f"AEC: {f} (표준화)"),
                "OR":    float(np.exp(log_uni.params.iloc[1])),
                "ci_lo": float(np.exp(ci_uni.iloc[1, 0])),
                "ci_hi": float(np.exp(ci_uni.iloc[1, 1])),
                "pval":  float(log_uni.pvalues.iloc[1]),
            })
        except Exception:
            pass
    if or_rows:
        or_df = pd.DataFrame(or_rows).sort_values("OR", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9, max(4, len(or_df) * 0.75 + 1.5)))
        for i, row in or_df.iterrows():
            color = "#e07070" if row["OR"] >= 1 else "#7090d0"
            ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=color, lw=2)
            ax.scatter(row["OR"], i, color=color, s=80, zorder=5)
            ax.text(row["ci_hi"] * 1.08, i, sig_stars(row["pval"]), va="center", fontsize=9)
        ax.axvline(1, color="black", lw=1.5, ls="--", label="OR=1 (귀무)")
        ax.set_yticks(range(len(or_df))); ax.set_yticklabels(or_df["label"])
        ax.set_xscale("log"); ax.set_xlabel("Crude OR (95% CI, log scale)")
        ax.set_title("로지스틱 회귀 - 단변량 Crude OR Forest Plot\n(Low TAMA 위험 오즈비)")
        ax.legend(fontsize=9)
        plt.tight_layout(); fig.savefig(OUT_DIR / "11_logistic_forest.png", dpi=150); plt.close()

    # ── Figs 12-15: Case comparison (statsmodels full fit) ──
    lin_met, log_met = {}, {}
    for case_name, feats in CASES.items():
        fa = avail(feats)
        if not fa:
            continue
        X_c     = X_full[fa]
        X_c_std = safe_std(X_c)
        if X_c_std.empty:
            continue
        ols_c   = sm.OLS(y_cont, sm.add_constant(X_c_std)).fit()
        resid_c = ols_c.resid
        lin_met[case_name] = {
            "R2": ols_c.rsquared, "Adj_R2": ols_c.rsquared_adj,
            "RMSE": float(np.sqrt(np.mean(resid_c ** 2))),
            "AIC": ols_c.aic, "BIC": ols_c.bic,
        }
        try:
            log_c  = sm.Logit(y_bin, sm.add_constant(X_c_std)).fit(disp=False, maxiter=300)
            prob_c = log_c.predict(sm.add_constant(X_c_std))
            auc_c  = roc_auc_score(y_bin, prob_c)
            rng_c  = np.random.RandomState(42)
            boot_c = []
            for _ in range(500):
                idx_c = rng_c.randint(0, len(y_bin), len(y_bin))
                if y_bin.iloc[idx_c].nunique() > 1:
                    boot_c.append(roc_auc_score(y_bin.iloc[idx_c], prob_c.values[idx_c]))
            n_obs = len(y_bin)
            ll_f  = log_c.llf
            ll_n  = sm.Logit(y_bin, np.ones(n_obs)).fit(disp=False, maxiter=200).llf
            cs    = 1 - np.exp((2 / n_obs) * (ll_n - ll_f))
            nag   = cs / (1 - np.exp((2 / n_obs) * ll_n))
            log_met[case_name] = {
                "AUC": auc_c,
                "AUC_ci_lo": float(np.percentile(boot_c, 2.5))  if boot_c else auc_c,
                "AUC_ci_hi": float(np.percentile(boot_c, 97.5)) if boot_c else auc_c,
                "Nagelkerke": float(nag),
                "AIC": log_c.aic, "BIC": log_c.bic,
            }
        except Exception as e:
            print(f"      [{case_name}] logistic fit failed: {e}")

    case_keys  = list(lin_met.keys())
    case_short = [CASE_LABELS.get(k, k).replace("\n", " ") for k in case_keys]
    x_pos      = np.arange(len(case_keys))

    # Fig 12: Linear R² / RMSE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    r2_v  = [lin_met[c]["R2"]     for c in case_keys]
    adj_v = [lin_met[c]["Adj_R2"] for c in case_keys]
    w12   = 0.35
    b1 = ax.bar(x_pos - w12/2, r2_v,  w12, label="R²",         color="#5b9bd5", edgecolor="white")
    b2 = ax.bar(x_pos + w12/2, adj_v, w12, label="Adjusted R²", color="#ed7d31", edgecolor="white")
    for bar, v in list(zip(b1, r2_v)) + list(zip(b2, adj_v)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos); ax.set_xticklabels(case_short, fontsize=8)
    ax.set_ylabel("R² 값"); ax.set_title("선형 회귀 R² 비교\n(Case 1 → 5)")
    ax.legend(fontsize=9); ax.set_ylim(0, max(r2_v) * 1.2)
    ax = axes[1]
    rmse_v   = [lin_met[c]["RMSE"] for c in case_keys]
    min_rmse = min(rmse_v)
    bars12   = ax.bar(x_pos, rmse_v,
                      color=["#e74c3c" if v == min_rmse else "#7fbbdf" for v in rmse_v],
                      edgecolor="white", width=0.5)
    for bar, v in zip(bars12, rmse_v):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_pos); ax.set_xticklabels(case_short, fontsize=8)
    ax.set_ylabel("RMSE (cm²)"); ax.set_title("선형 회귀 RMSE 비교\n(낮을수록 예측 정확)")
    ax.set_ylim(0, max(rmse_v) * 1.2)
    plt.suptitle(f"[{hosp_label}] Multivariable Analysis - 선형 회귀 성능",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(OUT_DIR / "12_case_metrics_bar.png", dpi=150); plt.close()

    # Fig 13: Logistic AUC + Nagelkerke R²
    if log_met:
        log_keys  = list(log_met.keys())
        log_short = [CASE_LABELS.get(k, k).replace("\n", " ") for k in log_keys]
        x_log     = np.arange(len(log_keys))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        auc_v  = [log_met[c]["AUC"]       for c in log_keys]
        auc_lo = [log_met[c]["AUC_ci_lo"] for c in log_keys]
        auc_hi = [log_met[c]["AUC_ci_hi"] for c in log_keys]
        bars13 = ax.bar(x_log, auc_v, color=COLORS[:len(log_keys)], edgecolor="white", width=0.5)
        ax.errorbar(x_log, auc_v,
                    yerr=[[v - lo for v, lo in zip(auc_v, auc_lo)],
                          [hi - v for hi, v in zip(auc_hi, auc_v)]],
                    fmt="none", color="black", capsize=5, lw=1.5)
        for bar, v in zip(bars13, auc_v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.axhline(0.5, color="gray",    ls="--", lw=1, label="무작위 기준 (0.5)")
        ax.axhline(0.7, color="#f39c12", ls=":",  lw=1, label="양호 기준 (0.7)")
        ax.axhline(0.8, color="#27ae60", ls=":",  lw=1, label="우수 기준 (0.8)")
        ax.set_xticks(x_log); ax.set_xticklabels(log_short, fontsize=8)
        ax.set_ylabel("AUC-ROC"); ax.set_ylim(0, 1)
        ax.set_title("로지스틱 AUC 비교 (Bootstrap 95%CI)\nCase 1 → 5"); ax.legend(fontsize=8)
        ax = axes[1]
        nag_v   = [log_met[c]["Nagelkerke"] for c in log_keys]
        bars13b = ax.bar(x_log, nag_v, color=COLORS[:len(log_keys)], edgecolor="white", width=0.5)
        for bar, v in zip(bars13b, nag_v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x_log); ax.set_xticklabels(log_short, fontsize=8)
        ax.set_ylabel("Nagelkerke R²"); ax.set_title("로지스틱 Nagelkerke R²\n(설명력 비교)")
        plt.suptitle(f"[{hosp_label}] Multivariable Analysis - 로지스틱 회귀 성능",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(); fig.savefig(OUT_DIR / "13_case_auc_bar.png", dpi=150); plt.close()

    # Fig 14: AIC/BIC comparison
    if lin_met and log_met:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, met_d, title14 in zip(
            axes,
            [lin_met, log_met],
            ["선형 회귀 AIC/BIC\n(낮을수록 모델 적합도 우수)",
             "로지스틱 회귀 AIC/BIC\n(낮을수록 모델 적합도 우수)"],
        ):
            ck14   = list(met_d.keys())
            cs14   = [CASE_LABELS.get(c, c).replace("\n", " ") for c in ck14]
            x_l14  = np.arange(len(ck14))
            aic_14 = [met_d[c]["AIC"] for c in ck14]
            bic_14 = [met_d[c]["BIC"] for c in ck14]
            w14    = 0.35
            ax.bar(x_l14 - w14/2, aic_14, w14, label="AIC", color="#5b9bd5", edgecolor="white")
            ax.bar(x_l14 + w14/2, bic_14, w14, label="BIC", color="#e07070", edgecolor="white")
            top_y14 = max(max(aic_14), max(bic_14))
            for i in range(1, len(ck14)):
                delta14 = aic_14[i] - aic_14[0]
                ax.annotate(f"ΔAIC={delta14:+.0f}",
                            xy=(x_l14[i], max(aic_14[i], bic_14[i])),
                            xytext=(x_l14[i], top_y14 * 1.03),
                            fontsize=8, color="#555555", ha="center",
                            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8))
            ax.set_xticks(x_l14); ax.set_xticklabels(cs14, fontsize=8)
            ax.set_ylabel("AIC / BIC 값"); ax.set_title(title14)
            ax.legend(fontsize=9); ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
        plt.suptitle(f"[{hosp_label}] Multivariable Analysis - AIC/BIC 모델 비교",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(); fig.savefig(OUT_DIR / "14_case_aic_bar.png", dpi=150); plt.close()

    # Fig 15: Case progression line plot
    common15 = [c for c in lin_met if c in log_met]
    if len(common15) >= 2:
        c_lbl15   = [CASE_LABELS.get(c, c).replace("\n", " ") for c in common15]
        x15       = np.arange(len(common15))
        metrics15 = [
            ([lin_met[c]["R2"]         for c in common15], "Linear R² 추이",     "#2196F3"),
            ([lin_met[c]["Adj_R2"]     for c in common15], "Linear Adj R² 추이", "#FF9800"),
            ([log_met[c]["AUC"]        for c in common15], "Logistic AUC 추이",  "#F44336"),
            ([log_met[c]["Nagelkerke"] for c in common15], "Nagelkerke R² 추이", "#4CAF50"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, (vals, title15, color) in zip(axes.flat, metrics15):
            ax.plot(x15, vals, "o-", color=color, lw=2, ms=8)
            y_span = max(vals) - min(vals) if max(vals) != min(vals) else 0.05
            for i, v in enumerate(vals):
                ax.annotate(f"{v:.3f}", xy=(x15[i], v),
                            xytext=(x15[i], v + y_span * 0.15),
                            ha="center", fontsize=10, fontweight="bold", color=color)
                if i > 0:
                    delta15 = v - vals[i - 1]
                    ax.annotate(f"Δ= {delta15:+.3f}",
                                xy=((x15[i] + x15[i-1]) / 2, (v + vals[i-1]) / 2),
                                ha="center", fontsize=8, color="#777777")
            ax.set_xticks(x15)
            ax.set_xticklabels(c_lbl15, fontsize=8, rotation=15, ha="right")
            ax.set_title(title15); ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
            ax.set_ylim(min(vals) - y_span * 0.3, max(vals) + y_span * 0.4)
        plt.suptitle(f"[{hosp_label}] Case 1 → 5: 예측 성능 지표 추이\n"
                     f"(AEC와 Scanner 추가에 따른 성능 향상 정량화)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(); fig.savefig(OUT_DIR / "15_case_progression.png", dpi=150); plt.close()

    print(f"  Saved: figs 04-15 to {OUT_DIR}")
