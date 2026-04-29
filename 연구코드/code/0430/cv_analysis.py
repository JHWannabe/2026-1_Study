# -*- coding: utf-8 -*-
"""
run_one_analysis(): 단일 병원에 대해 5-Fold CV 선형+로지스틱 회귀 실행.

성별 특이적 P25 임계값 적용: 여성은 여성 P25, 남성은 남성 P25 기준으로 이진화.
결과 플롯(Figs 01-08)과 요약 Excel을 RESULT_DIR에 저장한다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from config import CASE_LABELS, COLORS, CV_SPLITS, CV_RANDOM
from helpers import linear_cv, logistic_cv


def run_one_analysis(
    X_full: pd.DataFrame,
    y_cont: pd.Series,
    CASES: dict,
    RESULT_DIR: Path,
    hosp_label: str,
    group_label: str,
) -> tuple[pd.DataFrame, tuple]:
    """5-Fold CV 회귀 수행 → 플롯 + Excel 저장 → (summary_df, clean_tuple) 반환."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    female_mask = X_full["PatientSex_enc"] == 0
    male_mask   = X_full["PatientSex_enc"] == 1
    tama_female = y_cont[female_mask].quantile(0.25)
    tama_male   = y_cont[male_mask].quantile(0.25)
    tama_threshold = {"female": tama_female, "male": tama_male}

    y_bin = pd.Series(0, index=y_cont.index, dtype=int)
    y_bin[female_mask] = (y_cont[female_mask] < tama_female).astype(int)
    y_bin[male_mask]   = (y_cont[male_mask]   < tama_male).astype(int)
    n = len(y_cont)
    print(f"\n    [{group_label}] n={n}  F_P25={tama_female:.1f}  M_P25={tama_male:.1f}"
          f"  low(1)={y_bin.sum()}  high(0)={(y_bin==0).sum()}")

    labels = [CASE_LABELS.get(k, k) for k in CASES]
    colors = COLORS[:len(CASES)]

    def avail(feats):
        return [f for f in feats if f in X_full.columns]

    # ── Run CV ──
    lin_res, log_res = {}, {}
    print(f"      [Linear]")
    for name, feats in CASES.items():
        r = linear_cv(X_full[avail(feats)], y_cont)
        lin_res[name] = r
        print(f"        {name}: R²={r['R2']:.4f}±{r['R2_std']:.4f}  "
              f"MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}")

    print(f"      [Logistic  F_P25={tama_female:.1f}  M_P25={tama_male:.1f}]")
    for name, feats in CASES.items():
        r = logistic_cv(X_full[avail(feats)], y_bin)
        log_res[name] = r
        print(f"        {name}: AUC={r['AUC']:.4f}±{r['AUC_std']:.4f}  "
              f"Acc={r['Accuracy']:.4f}  Sens={r['Sensitivity']:.4f}  Spec={r['Specificity']:.4f}")

    prefix = f"[{hosp_label}] [{group_label}]"

    # ── Fig 01: Linear OOF actual vs predicted ──
    fig, axes = plt.subplots(1, len(CASES), figsize=(len(CASES)*5, 5))
    if len(CASES) == 1: axes = [axes]
    for ax, (name, feats), color, lbl in zip(axes, CASES.items(), colors, labels):
        true = np.array(lin_res[name]["oof_true"])
        pred = np.array(lin_res[name]["oof_pred"])
        ax.scatter(true, pred, alpha=0.2, s=8, color=color, rasterized=True)
        lo = min(true.min(), pred.min()) - 5; hi = max(true.max(), pred.max()) + 5
        ax.plot([lo, hi], [lo, hi], "k--", lw=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        r2v = lin_res[name]['R2']; r2s = lin_res[name]['R2_std']
        ax.set_title(f"{lbl.replace(chr(10),' ')}\nR²={r2v:.3f}±{r2s:.3f}", fontsize=9)
        ax.set_xlabel("Actual TAMA"); ax.set_ylabel("Predicted TAMA")
    plt.suptitle(f"{prefix} Linear – Actual vs Predicted (5-fold OOF)", fontsize=12, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "01_linear_actual_vs_pred.png", dpi=150); plt.close()

    # ── Fig 02: Linear metrics bar ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["R2", "MAE", "RMSE"]):
        vals = [lin_res[n][metric] for n in CASES]
        errs = [lin_res[n][f"{metric}_std"] for n in CASES]
        bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, edgecolor="white", width=0.5)
        ax.set_title(metric); ax.set_ylabel(metric)
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(errs)*0.1,
                    f"{v:.3f}\n±{e:.3f}", ha="center", va="bottom", fontsize=7)
    plt.suptitle(f"{prefix} Linear Regression – CV Metrics", fontsize=12, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "02_linear_metrics_comparison.png", dpi=150); plt.close()

    # ── Fig 03: Linear coefficients ──
    fig, axes = plt.subplots(1, len(CASES), figsize=(len(CASES)*4, 5))
    if len(CASES) == 1: axes = [axes]
    for ax, (name, feats), color in zip(axes, CASES.items(), colors):
        fa = avail(feats)
        pipe = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
        pipe.fit(X_full[fa], y_cont)
        cdf = pd.DataFrame({"feature": fa, "coef": pipe.named_steps["m"].coef_}).sort_values("coef")
        ax.barh(cdf["feature"], cdf["coef"],
                color=["#e74c3c" if c < 0 else "#3498db" for c in cdf["coef"]])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"{name}\n(standardized coef)", fontsize=8); ax.tick_params(labelsize=7)
    plt.suptitle(f"{prefix} Linear – Standardized Coefficients", fontsize=11, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "03_linear_coefficients.png", dpi=150); plt.close()

    # ── Fig 04: Logistic ROC ──
    fig, ax = plt.subplots(figsize=(6, 6))
    for (name, _), color, lbl in zip(CASES.items(), colors, labels):
        mean_fpr = np.linspace(0, 1, 200)
        tprs_i   = [np.interp(mean_fpr, f, t)
                    for f, t in zip(log_res[name]["fprs"], log_res[name]["tprs"])]
        mean_tpr = np.mean(tprs_i, axis=0); std_tpr = np.std(tprs_i, axis=0)
        auc = log_res[name]["AUC"]; auc_std = log_res[name]["AUC_std"]
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                label=f"{lbl.replace(chr(10),' ')} (AUC={auc:.3f}±{auc_std:.3f})")
        ax.fill_between(mean_fpr, mean_tpr-std_tpr, mean_tpr+std_tpr, alpha=0.12, color=color)
    ax.plot([0,1],[0,1],"k--",lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{prefix} ROC (5-fold | F:{tama_female:.1f}/M:{tama_male:.1f})")
    ax.legend(fontsize=9); plt.tight_layout()
    fig.savefig(RESULT_DIR / "04_logistic_roc.png", dpi=150); plt.close()

    # ── Fig 05: Logistic metrics bar ──
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, ["AUC", "Accuracy", "Sensitivity", "Specificity"]):
        vals = [log_res[n][metric] for n in CASES]
        errs = [log_res[n][f"{metric}_std"] for n in CASES]
        bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, edgecolor="white", width=0.5)
        ax.set_ylim(0, 1.2); ax.set_title(metric); ax.set_ylabel(metric)
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(errs)*0.1,
                    f"{v:.3f}\n±{e:.3f}", ha="center", va="bottom", fontsize=7)
    plt.suptitle(f"{prefix} Logistic Regression – CV Metrics", fontsize=12, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "05_logistic_metrics_comparison.png", dpi=150); plt.close()

    # ── Fig 06: Logistic confusion matrices (OOF + Youden threshold) ──
    fig, axes = plt.subplots(1, len(CASES), figsize=(len(CASES)*4, 4))
    if len(CASES) == 1: axes = [axes]
    for ax, (name, feats), lbl in zip(axes, CASES.items(), labels):
        oof_prob = np.array(log_res[name]["oof_prob"])
        oof_true = np.array(log_res[name]["oof_true"])
        thr      = log_res[name]["youden_threshold"]
        oof_pred = (oof_prob >= thr).astype(int)
        cm = confusion_matrix(oof_true, oof_pred)
        tn_v, fp_v, fn_v, tp_v = cm.ravel()
        sens_v = tp_v / (tp_v + fn_v) if tp_v + fn_v else 0
        spec_v = tn_v / (tn_v + fp_v) if tn_v + fp_v else 0
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["High","Low"], yticklabels=["High","Low"], cbar=False)
        ax.set_title(f"{lbl.replace(chr(10), ' ')}\n"
                     f"Thr={thr:.2f}  Sens={sens_v:.3f}  Spec={spec_v:.3f}", fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle(f"{prefix} Confusion Matrix (5-fold OOF, Youden threshold, F:{tama_female:.1f}/M:{tama_male:.1f})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "06_logistic_confusion.png", dpi=150); plt.close()

    # ── Fig 07: Logistic coefficients ──
    fig, axes = plt.subplots(1, len(CASES), figsize=(len(CASES)*4, 5))
    if len(CASES) == 1: axes = [axes]
    for ax, (name, feats), color in zip(axes, CASES.items(), colors):
        fa   = avail(feats)
        pipe = Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000, random_state=CV_RANDOM))])
        pipe.fit(X_full[fa], y_bin)
        coefs = pipe.named_steps["m"].coef_[0]
        cdf   = pd.DataFrame({"feature": fa, "coef": coefs}).sort_values("coef")
        ax.barh(cdf["feature"], cdf["coef"],
                color=["#e74c3c" if c < 0 else "#3498db" for c in cdf["coef"]])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"{name}\n(AUC={log_res[name]['AUC']:.3f}±{log_res[name]['AUC_std']:.3f})",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    plt.suptitle(f"{prefix} Logistic – Standardized Coefficients", fontsize=11, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "07_logistic_coefficients.png", dpi=150); plt.close()

    # ── Fig 08: Overview R² & AUC ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(CASES))
    for ax, res_dict, metric, title in zip(
        axes, [lin_res, log_res], ["R2", "AUC"], ["Linear R²", "Logistic AUC"]
    ):
        vals = [res_dict[n][metric] for n in CASES]
        errs = [res_dict[n][f"{metric}_std"] for n in CASES]
        ax.bar(x, vals, yerr=errs, capsize=5, color=colors, edgecolor="white", width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title); ax.set_ylabel(title.split()[-1])
        if metric == "AUC": ax.set_ylim(0, 1)
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e + 0.005, f"{v:.3f}\n±{e:.3f}", ha="center", fontsize=7, fontweight="bold")
    plt.suptitle(f"{prefix} Case Comparison Overview", fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "08_case_comparison_overview.png", dpi=150); plt.close()

    # ── Summary DataFrame & Excel ──
    rows = []
    for name, lbl in zip(CASES.keys(), labels):
        lr, lo = lin_res[name], log_res[name]
        rows.append({
            "Hospital": hosp_label, "Sex": group_label, "Case": lbl.replace("\n", " "),
            "N_features": len(CASES[name]), "N_rows": n,
            "Lin_R2": round(lr["R2"], 4),   "Lin_R2_std": round(lr["R2_std"], 4),
            "Lin_MAE": round(lr["MAE"], 2), "Lin_RMSE": round(lr["RMSE"], 2),
            "TAMA_threshold": f"F:{tama_female:.1f}/M:{tama_male:.1f}",
            "Log_AUC": round(lo["AUC"], 4),   "Log_AUC_std": round(lo["AUC_std"], 4),
            "Log_Acc":  round(lo["Accuracy"],    4),
            "Log_Sens": round(lo["Sensitivity"], 4),
            "Log_Spec": round(lo["Specificity"], 4),
        })
    summary_df = pd.DataFrame(rows)

    fold_lin = pd.DataFrame(
        {n: lin_res[n]["fold_r2"]  for n in CASES},
        index=[f"Fold{i+1}" for i in range(CV_SPLITS)],
    )
    fold_log = pd.DataFrame(
        {n: log_res[n]["fold_auc"] for n in CASES},
        index=[f"Fold{i+1}" for i in range(CV_SPLITS)],
    )
    with pd.ExcelWriter(RESULT_DIR / "regression_results.xlsx") as writer:
        summary_df.to_excel(writer, sheet_name="summary",         index=False)
        fold_lin.to_excel(writer,   sheet_name="linear_fold_r2")
        fold_log.to_excel(writer,   sheet_name="logistic_fold_auc")
    print(f"      Saved to {RESULT_DIR}")

    return summary_df, (X_full, y_cont, y_bin, CASES, tama_threshold)
