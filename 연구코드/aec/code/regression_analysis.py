"""
Regression Analysis: Linear & Logistic Regression on TAMA
Target  : TAMA  (Linear = continuous | Logistic = binary below lower 25th percentile)
Cases   :
  Case 1 - Clinical baseline  : PatientAge, (PatientSex), BMI
  Case 2 - + AEC_prev         : Case 1 + [mean, CV, skewness, slope_abs_mean]
  Case 3 - + AEC_new          : Case 1 + pipeline-selected features
  Case 4 - + AEC_prev+Scanner : Case 2 + ManufacturerModelName, kVp
  Case 5 - + AEC_new+Scanner  : Case 3 + ManufacturerModelName, kVp
Hospitals : 강남, 신촌
Sex groups: 전체, 여성(F), 남성(M)  — sex-stratified groups drop PatientSex_enc
"""

import warnings
warnings.filterwarnings("ignore")

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score,
)
from sklearn.pipeline import Pipeline

# ────────────────────────────────────────────────
# 0. Global config
# ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
FS_RESULT  = SCRIPT_DIR.parent / "results" / "feature_selection"

CV_SPLITS = 5
CV_RANDOM = 42

AEC_PREV = ["mean", "CV", "skewness", "slope_abs_mean"]

fs_xlsx = FS_RESULT / "feature_selection_summary.xlsx"
if fs_xlsx.exists():
    AEC_NEW = pd.read_excel(str(fs_xlsx), sheet_name="final_features")["final_features"].tolist()
    print(f"[AEC-new] Loaded {len(AEC_NEW)} features from pipeline: {AEC_NEW}")
else:
    AEC_NEW = [
        "IQR", "band2_energy", "dominant_freq", "mean", "slope_max",
        "spectral_energy", "wavelet_cD1_energy", "wavelet_cD2_energy",
        "wavelet_energy_ratio_D1",
    ]
    print(f"[AEC-new] Fallback hardcoded {len(AEC_NEW)} features")

print(f"[AEC-prev] {len(AEC_PREV)} features: {AEC_PREV}")

HOSPITALS = {}
for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".xlsx") or "merged_features" not in fname:
        continue
    if "강남" in fname:
        HOSPITALS["gangnam"] = DATA_DIR / fname
    elif "신촌" in fname:
        HOSPITALS["sinchon"] = DATA_DIR / fname

print(f"[Data] Found hospitals: {list(HOSPITALS.keys())}")

CASE_LABELS = {
    "Case1_Clinical":                  "Case 1\nClinical",
    "Case2_Clinical+AEC_prev":         "Case 2\n+AEC_prev",
    "Case3_Clinical+AEC_new":          "Case 3\n+AEC_new",
    "Case4_Clinical+AEC_prev+Scanner": "Case 4\n+AEC_prev\n+Scanner",
    "Case5_Clinical+AEC_new+Scanner":  "Case 5\n+AEC_new\n+Scanner",
}
COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#1a5276", "#922b21"]

# ────────────────────────────────────────────────
# 1. Helpers
# ────────────────────────────────────────────────
def copy_to_temp(src: Path, temp_name: str) -> Path:
    dst = Path(os.environ["TEMP"]) / temp_name
    subprocess.run(
        ["powershell", "-Command",
         f'Copy-Item -Path "{src}" -Destination "{dst}" -Force'],
        capture_output=True,
    )
    return dst

def load_hospital(src: Path, temp_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = copy_to_temp(src, temp_name)
    feat = pd.read_excel(str(tmp), sheet_name="features")
    meta = pd.read_excel(str(tmp), sheet_name="metadata-bmi_add")
    return feat, meta

def linear_cv(X: pd.DataFrame, y: pd.Series) -> dict:
    kf   = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM)
    pipe = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    r2s, maes, rmses, all_true, all_pred = [], [], [], [], []
    for tr, te in kf.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], pred))
        maes.append(mean_absolute_error(y.iloc[te], pred))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
        all_true.extend(y.iloc[te].tolist())
        all_pred.extend(pred.tolist())
    return dict(R2=np.mean(r2s), R2_std=np.std(r2s),
                MAE=np.mean(maes), MAE_std=np.std(maes),
                RMSE=np.mean(rmses), RMSE_std=np.std(rmses),
                fold_r2=r2s, oof_true=all_true, oof_pred=all_pred)

def logistic_cv(X: pd.DataFrame, y: pd.Series) -> dict:
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM)
    aucs, accs, sens, specs, fprs_l, tprs_l = [], [], [], [], [], []
    for tr, te in skf.split(X, y):
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=2000, random_state=CV_RANDOM, solver="lbfgs")),
        ])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        pred = pipe.predict(X.iloc[te])
        aucs.append(roc_auc_score(y.iloc[te], prob))
        accs.append(accuracy_score(y.iloc[te], pred))
        tn, fp, fn, tp = confusion_matrix(y.iloc[te], pred).ravel()
        sens.append(tp / (tp + fn) if tp + fn else 0)
        specs.append(tn / (tn + fp) if tn + fp else 0)
        fpr, tpr, _ = roc_curve(y.iloc[te], prob)
        fprs_l.append(fpr); tprs_l.append(tpr)
    return dict(AUC=np.mean(aucs), AUC_std=np.std(aucs),
                Accuracy=np.mean(accs), Accuracy_std=np.std(accs),
                Sensitivity=np.mean(sens), Sensitivity_std=np.std(sens),
                Specificity=np.mean(specs), Specificity_std=np.std(specs),
                fold_auc=aucs, fprs=fprs_l, tprs=tprs_l)

def make_cases(clinical_feats: list, scanner_feats: list) -> dict:
    return {
        "Case1_Clinical":                  clinical_feats,
        "Case2_Clinical+AEC_prev":         clinical_feats + AEC_PREV,
        "Case3_Clinical+AEC_new":          clinical_feats + AEC_NEW,
        "Case4_Clinical+AEC_prev+Scanner": clinical_feats + AEC_PREV + scanner_feats,
        "Case5_Clinical+AEC_new+Scanner":  clinical_feats + AEC_NEW  + scanner_feats,
    }

def run_one_analysis(X_full, y_cont, CASES, RESULT_DIR, hosp_label, group_label):
    """Run full Linear+Logistic CV for one hospital/sex group. Returns (summary_df, clean_tuple)."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    tama_threshold = y_cont.quantile(0.25)
    y_bin = (y_cont < tama_threshold).astype(int)
    n = len(y_cont)
    print(f"\n    [{group_label}] n={n}  25th pct={tama_threshold:.1f}"
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

    print(f"      [Logistic  TAMA < {tama_threshold:.1f}]")
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
        auc = log_res[name]["AUC"]
        auc_std = log_res[name]["AUC_std"]
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                label=f"{lbl.replace(chr(10),' ')} (AUC={auc:.3f}±{auc_std:.3f})")
        ax.fill_between(mean_fpr, mean_tpr-std_tpr, mean_tpr+std_tpr, alpha=0.12, color=color)
    ax.plot([0,1],[0,1],"k--",lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{prefix} ROC (5-fold | 하위 25%={tama_threshold:.1f})")
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

    # ── Fig 06: Logistic confusion matrices ──
    fig, axes = plt.subplots(1, len(CASES), figsize=(len(CASES)*4, 4))
    if len(CASES) == 1: axes = [axes]
    for ax, (name, feats), lbl in zip(axes, CASES.items(), labels):
        fa = avail(feats)
        pipe = Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000, random_state=CV_RANDOM))])
        pipe.fit(X_full[fa], y_bin)
        cm = confusion_matrix(y_bin, pipe.predict(X_full[fa]))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["High","Low"], yticklabels=["High","Low"], cbar=False)
        ax.set_title(lbl.replace(chr(10), " "), fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle(f"{prefix} Confusion Matrix (full fit, 하위 25%={tama_threshold:.1f})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "06_logistic_confusion.png", dpi=150); plt.close()

    # ── Fig 07: Logistic coefficients ──
    fig, axes = plt.subplots(1, len(CASES), figsize=(len(CASES)*4, 5))
    if len(CASES) == 1: axes = [axes]
    for ax, (name, feats), color in zip(axes, CASES.items(), colors):
        fa = avail(feats)
        pipe = Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000, random_state=CV_RANDOM))])
        pipe.fit(X_full[fa], y_bin)
        coefs = pipe.named_steps["m"].coef_[0]
        cdf = pd.DataFrame({"feature": fa, "coef": coefs}).sort_values("coef")
        ax.barh(cdf["feature"], cdf["coef"],
                color=["#e74c3c" if c < 0 else "#3498db" for c in cdf["coef"]])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"{name}\n(AUC={log_res[name]['AUC']:.3f}±{log_res[name]['AUC_std']:.3f})", fontsize=8)
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
            "Lin_R2": round(lr["R2"], 4), "Lin_R2_std": round(lr["R2_std"], 4),
            "Lin_MAE": round(lr["MAE"], 2), "Lin_RMSE": round(lr["RMSE"], 2),
            "TAMA_threshold": tama_threshold,
            "Log_AUC": round(lo["AUC"], 4), "Log_AUC_std": round(lo["AUC_std"], 4),
            "Log_Acc": round(lo["Accuracy"], 4),
            "Log_Sens": round(lo["Sensitivity"], 4),
            "Log_Spec": round(lo["Specificity"], 4),
        })
    summary_df = pd.DataFrame(rows)

    fold_lin = pd.DataFrame(
        {n: lin_res[n]["fold_r2"]  for n in CASES},
        index=[f"Fold{i+1}" for i in range(CV_SPLITS)]
    )
    fold_log = pd.DataFrame(
        {n: log_res[n]["fold_auc"] for n in CASES},
        index=[f"Fold{i+1}" for i in range(CV_SPLITS)]
    )
    with pd.ExcelWriter(RESULT_DIR / "regression_results.xlsx") as writer:
        summary_df.to_excel(writer, sheet_name="summary",         index=False)
        fold_lin.to_excel(writer,   sheet_name="linear_fold_r2")
        fold_log.to_excel(writer,   sheet_name="logistic_fold_auc")
    print(f"      Saved to {RESULT_DIR}")

    return summary_df, (X_full, y_cont, y_bin, CASES, tama_threshold)


# ────────────────────────────────────────────────
# 2. Per-hospital analysis (all / female / male)
# ────────────────────────────────────────────────
all_summaries  = {}   # hosp_key -> summary_df (전체 only, for cross-hospital)
all_clean_data = {}   # hosp_key -> clean_tuple (전체 only, for external validation)

SEX_GROUPS = [
    ("all",    None, "전체"),
    ("female", 0,    "여성(F)"),
    ("male",   1,    "남성(M)"),
]

for hosp_key, data_path in HOSPITALS.items():
    hosp_label = "강남" if hosp_key == "gangnam" else "신촌"
    print(f"\n{'='*60}")
    print(f"HOSPITAL: {hosp_label} ({hosp_key})")
    print(f"{'='*60}")

    BASE_DIR = SCRIPT_DIR.parent / "results" / "regression" / hosp_key

    # ── Load & merge ──
    feat_df, meta_df = load_hospital(data_path, f"{hosp_key}_temp.xlsx")
    meta_df["TAMA"] = pd.to_numeric(meta_df["TAMA"], errors="coerce")

    df = feat_df.merge(
        meta_df[["PatientID", "PatientAge", "PatientSex",
                 "BMI", "ManufacturerModelName", "kVp", "TAMA"]],
        on="PatientID", how="inner",
    )
    df["PatientSex_enc"] = df["PatientSex"].map({"F": 0, "M": 1})
    model_dummies = pd.get_dummies(df["ManufacturerModelName"], prefix="model", drop_first=True)
    MODEL_COLS = model_dummies.columns.tolist()
    df = pd.concat([df, model_dummies], axis=1)

    SCANNER   = MODEL_COLS + ["kVp"]
    CLINICAL  = ["PatientAge", "PatientSex_enc", "BMI"]        # includes sex
    CLIN_NOSEX = ["PatientAge", "BMI"]                          # sex-stratified

    ALL_COLS = list(dict.fromkeys(
        CLINICAL + AEC_PREV + AEC_NEW + SCANNER + ["TAMA", "PatientSex_enc"]
    ))
    df_clean = df[ALL_COLS].dropna().reset_index(drop=True)
    print(f"  Rows after dropna: {len(df_clean)}"
          f"  (F={( df_clean['PatientSex_enc']==0).sum()}"
          f"  M={(df_clean['PatientSex_enc']==1).sum()})")

    group_summaries = {}

    for sex_key, sex_val, sex_label in SEX_GROUPS:
        if sex_val is None:
            df_sub = df_clean
            cases  = make_cases(CLINICAL, SCANNER)
        else:
            df_sub = df_clean[df_clean["PatientSex_enc"] == sex_val].reset_index(drop=True)
            cases  = make_cases(CLIN_NOSEX, SCANNER)

        if len(df_sub) < CV_SPLITS * 2:
            print(f"  [{sex_label}] too few rows ({len(df_sub)}), skipped")
            continue

        X_sub = df_sub.drop(columns=["TAMA"]).reset_index(drop=True)
        y_sub = df_sub["TAMA"].astype(float).reset_index(drop=True)

        result_dir = BASE_DIR / sex_key
        summary_df, clean_tuple = run_one_analysis(
            X_sub, y_sub, cases, result_dir, hosp_label, sex_label
        )
        group_summaries[sex_key] = summary_df

        if sex_key == "all":
            all_summaries[hosp_key]  = summary_df
            all_clean_data[hosp_key] = clean_tuple

    # ── Fig: Sex comparison – R² and AUC per case ──
    if len(group_summaries) == 3:
        sex_colors = {"all": "#555555", "female": "#e74c3c", "male": "#3498db"}
        sex_disp   = {"all": "전체", "female": "여성(F)", "male": "남성(M)"}
        case_short = ["C1\nClinical", "C2\n+AEC\nprev", "C3\n+AEC\nnew",
                      "C4\n+prev\n+Scan", "C5\n+new\n+Scan"]
        n_cases = 5
        x = np.arange(n_cases)
        w = 0.25

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, metric, std_key, title, ylim in zip(
            axes,
            ["Lin_R2", "Log_AUC"],
            ["Lin_R2_std", "Log_AUC_std"],
            ["Linear R²", "Logistic AUC"],
            [None, (0, 1)],
        ):
            for i, skey in enumerate(["all", "female", "male"]):
                df_g = group_summaries[skey]
                vals = [df_g.iloc[k][metric]   for k in range(n_cases)]
                errs = [df_g.iloc[k][std_key]  for k in range(n_cases)]
                offset = (i - 1) * w
                bars = ax.bar(x + offset, vals, w, yerr=errs, capsize=4,
                              color=sex_colors[skey], label=sex_disp[skey], edgecolor="white")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6)
            ax.set_xticks(x); ax.set_xticklabels(case_short, fontsize=8)
            ax.set_title(title); ax.set_ylabel(title.split()[-1])
            if ylim: ax.set_ylim(*ylim)
            ax.legend(fontsize=9)

        plt.suptitle(f"[{hosp_label}] 성별 비교: 전체 vs 여성 vs 남성",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(BASE_DIR / "sex_comparison.png", dpi=150); plt.close()
        print(f"\n  Saved: {BASE_DIR / 'sex_comparison.png'}")

        # Combined Excel
        combined_sex = pd.concat(group_summaries.values(), ignore_index=True)
        combined_sex.to_excel(BASE_DIR / "sex_comparison_summary.xlsx", index=False)
        print(f"  Saved: sex_comparison_summary.xlsx")

# ────────────────────────────────────────────────
# 3. Cross-hospital comparison (all group only)
# ────────────────────────────────────────────────
if len(all_summaries) == 2:
    print(f"\n{'='*60}")
    print("CROSS-HOSPITAL COMPARISON (전체)")
    print(f"{'='*60}")

    combined = pd.concat(all_summaries.values(), ignore_index=True)
    COMPARE_DIR = SCRIPT_DIR.parent / "results" / "regression"
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    print(combined[["Hospital","Case","N_rows","Lin_R2","Log_AUC",
                     "Log_Acc","Log_Sens","Log_Spec"]].to_string(index=False))

    case_short  = ["Case1\nClinical", "Case2\n+AEC\nprev", "Case3\n+AEC\nnew",
                   "Case4\n+prev\n+Scanner", "Case5\n+new\n+Scanner"]
    hosp_colors = {"gangnam": "#3498db", "sinchon": "#e74c3c"}
    hosp_labels = {"gangnam": "강남", "sinchon": "신촌"}
    n_cases = 5

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(n_cases); w = 0.35

    for ax, metric, std_key, title, ylim in zip(
        axes,
        ["Lin_R2", "Log_AUC"], ["Lin_R2_std", "Log_AUC_std"],
        ["Linear R²", "Logistic AUC"], [None, (0, 1)],
    ):
        for i, (hkey, hdf) in enumerate(all_summaries.items()):
            vals = [hdf.iloc[k][metric]   for k in range(n_cases)]
            errs = [hdf.iloc[k][std_key]  for k in range(n_cases)]
            offset = (i - 0.5) * w
            bars = ax.bar(x + offset, vals, w, yerr=errs, capsize=4,
                          color=hosp_colors[hkey], label=hosp_labels[hkey], edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(case_short, fontsize=8)
        ax.set_title(title); ax.set_ylabel(title.split()[-1])
        if ylim: ax.set_ylim(*ylim)
        ax.legend()

    plt.suptitle("병원별 케이스 비교: 강남 vs 신촌", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(COMPARE_DIR / "09_cross_hospital_comparison.png", dpi=150); plt.close()
    print(f"\n  Saved: 09_cross_hospital_comparison.png")

    combined.to_excel(COMPARE_DIR / "cross_hospital_summary.xlsx", index=False)
    print(f"  Saved: cross_hospital_summary.xlsx")

# ────────────────────────────────────────────────
# 4. External validation: train on A, test on B
# ────────────────────────────────────────────────
if len(all_clean_data) == 2:
    print(f"\n{'='*60}")
    print("EXTERNAL VALIDATION (train → test)")
    print(f"{'='*60}")

    COMPARE_DIR = SCRIPT_DIR.parent / "results" / "regression"
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    hosp_keys = list(all_clean_data.keys())
    hosp_labels_map = {"gangnam": "강남", "sinchon": "신촌"}

    directions = [(hosp_keys[0], hosp_keys[1]), (hosp_keys[1], hosp_keys[0])]
    ext_rows = []

    for train_key, test_key in directions:
        X_tr, y_tr_cont, y_tr_bin, CASES_tr, _ = all_clean_data[train_key]
        X_te, y_te_cont, y_te_bin, CASES_te, tama_te = all_clean_data[test_key]
        train_lbl = hosp_labels_map[train_key]
        test_lbl  = hosp_labels_map[test_key]
        print(f"\n  Train: {train_lbl} (n={len(X_tr)})  ->  Test: {test_lbl} (n={len(X_te)})")

        for case_name, feats_tr in CASES_tr.items():
            feats_common = [f for f in feats_tr if f in X_te.columns]
            missing = set(feats_tr) - set(feats_common)
            if missing:
                print(f"    [{case_name}] Warning: {len(missing)} features missing, skipped")
            if not feats_common:
                continue

            pipe_lin = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
            pipe_lin.fit(X_tr[feats_common], y_tr_cont)
            pred_cont = pipe_lin.predict(X_te[feats_common])
            lin_r2   = r2_score(y_te_cont, pred_cont)
            lin_mae  = mean_absolute_error(y_te_cont, pred_cont)
            lin_rmse = np.sqrt(mean_squared_error(y_te_cont, pred_cont))

            y_te_bin_ext = (y_te_cont < tama_te).astype(int)
            pipe_log = Pipeline([
                ("sc", StandardScaler()),
                ("m", LogisticRegression(max_iter=2000, random_state=CV_RANDOM, solver="lbfgs")),
            ])
            pipe_log.fit(X_tr[feats_common], y_tr_bin)
            prob    = pipe_log.predict_proba(X_te[feats_common])[:, 1]
            pred_bin = pipe_log.predict(X_te[feats_common])
            log_auc = roc_auc_score(y_te_bin_ext, prob)
            log_acc = accuracy_score(y_te_bin_ext, pred_bin)
            tn, fp, fn, tp = confusion_matrix(y_te_bin_ext, pred_bin).ravel()
            log_sens = tp / (tp + fn) if tp + fn else 0
            log_spec = tn / (tn + fp) if tn + fp else 0

            print(f"    {case_name}: LinR²={lin_r2:.4f}  AUC={log_auc:.4f}  "
                  f"Acc={log_acc:.4f}  Sens={log_sens:.4f}  Spec={log_spec:.4f}")

            ext_rows.append({
                "Train": train_lbl, "Test": test_lbl,
                "Case": case_name, "N_train": len(X_tr), "N_test": len(X_te),
                "N_features_used": len(feats_common),
                "Lin_R2": round(lin_r2, 4), "Lin_MAE": round(lin_mae, 2),
                "Lin_RMSE": round(lin_rmse, 2), "TAMA_threshold": tama_te,
                "Log_AUC": round(log_auc, 4), "Log_Acc": round(log_acc, 4),
                "Log_Sens": round(log_sens, 4), "Log_Spec": round(log_spec, 4),
            })

    ext_df = pd.DataFrame(ext_rows)

    directions_labels = [f"{hosp_labels_map[a]}->{hosp_labels_map[b]}" for a, b in directions]
    dir_colors = ["#8e44ad", "#27ae60"]
    case_names = ext_df["Case"].unique().tolist()
    n_c = len(case_names)
    xtick_labels = [c.replace("Case", "C").replace("_Clinical", "\nClinical")
                     .replace("+AEC_prev", "\n+AEC\nprev").replace("+AEC_new", "\n+AEC\nnew")
                     .replace("+Scanner", "\n+Scanner") for c in case_names]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(n_c); w = 0.35

    for ax, metric, title, ylim in zip(
        axes, ["Lin_R2", "Log_AUC"],
        ["External Linear R²", "External Logistic AUC"], [None, (0, 1)]
    ):
        for i, (dlabel, dcolor) in enumerate(zip(directions_labels, dir_colors)):
            sub = ext_df[ext_df["Train"].apply(lambda v: v in dlabel)
                         & ext_df["Test"].apply(lambda v: v in dlabel)]
            vals = [sub[sub["Case"] == c][metric].values[0] if len(sub[sub["Case"] == c]) else 0
                    for c in case_names]
            offset = (i - 0.5) * w
            bars = ax.bar(x + offset, vals, w, color=dcolor, label=dlabel, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(xtick_labels, fontsize=7)
        ax.set_title(title); ax.set_ylabel(metric.split("_")[-1])
        if ylim: ax.set_ylim(*ylim)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.legend(fontsize=8)

    plt.suptitle("외부 검증: 병원 간 교차 (Train -> Test)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(COMPARE_DIR / "10_external_validation.png", dpi=150); plt.close()
    print(f"\n  Saved: 10_external_validation.png")

    ext_df.to_excel(COMPARE_DIR / "external_validation_results.xlsx", index=False)
    print(f"  Saved: external_validation_results.xlsx")

print(f"\n{'='*60}")
print("ALL DONE")
print(f"{'='*60}")
