# -*- coding: utf-8 -*-
"""
메인 오케스트레이션 스크립트.

실행 순서:
  1. 병원별 루프 (강남 / 신촌)
     a. 데이터 로드 & 정제
     b. EDA (Figs 01-03, 16-20)
     c. 성별 그룹 루프 (전체 / 여성 / 남성) → 5-Fold CV
     d. 전체 그룹에 대해 full-fit 진단 플롯 (Figs 04-15)
  2. 교차 병원 비교
  3. 외부 검증 (train → test)
  4. BMI 기여도 분석

Usage:
    python 02_run_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    SCRIPT_DIR, AEC_PREV, AEC_CANDIDATES, HOSPITALS, CASE_LABELS, COLORS, CV_RANDOM,
)
from helpers import load_hospital, linear_cv, logistic_cv, make_cases
from fullfit_analysis import run_fullfit_analysis
from cv_analysis import run_one_analysis
from eda_plots import run_smi_stats, run_eda_figs

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score,
)
from sklearn.pipeline import Pipeline


# 전체 분석만 실행 — 성별 특이적 P25 임계값을 내부에서 적용
SEX_GROUPS = [
    ("all", None, "전체"),
]


def run_hospital(hosp_key: str, data_path: Path) -> tuple[dict, dict]:
    """
    단일 병원에 대해 데이터 로드 → EDA → 전체 그룹 CV → full-fit 진단 수행.
    Returns (group_summaries, clean_tuple_all)
      group_summaries: {sex_key: summary_df}
      clean_tuple_all: (X_sub, y_cont, y_bin, cases, smi_threshold_dict) for 전체 그룹
    """
    hosp_label = "강남" if hosp_key == "gangnam" else "신촌"
    print(f"\n{'='*60}")
    print(f"HOSPITAL: {hosp_label} ({hosp_key})")
    print(f"{'='*60}")

    BASE_DIR = SCRIPT_DIR.parent / "results" / "regression" / hosp_key
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 & 정제 ──
    feat_df, meta_df = load_hospital(data_path, f"{hosp_key}_temp.xlsx")
    meta_df["SMI"]   = pd.to_numeric(meta_df["SMI"], errors="coerce")

    df = feat_df.merge(
        meta_df[["PatientID", "PatientAge", "PatientSex",
                 "BMI", "ManufacturerModelName", "kVp", "SMI"]],
        on="PatientID", how="inner",
    )
    df["PatientSex_enc"] = df["PatientSex"].map({"F": 0, "M": 1})
    model_dummies = pd.get_dummies(df["ManufacturerModelName"], prefix="model", drop_first=True)
    MODEL_COLS    = model_dummies.columns.tolist()
    df = pd.concat([df, model_dummies], axis=1)

    SCANNER    = MODEL_COLS + ["kVp"]
    CLINICAL   = ["PatientAge", "PatientSex_enc", "BMI"]
    CLIN_NOSEX = ["PatientAge", "BMI"]

    ALL_COLS    = list(dict.fromkeys(CLINICAL + AEC_PREV + AEC_CANDIDATES + SCANNER + ["SMI", "PatientSex_enc"]))
    _clean_mask = df[ALL_COLS].notna().all(axis=1)
    df_clean    = df.loc[_clean_mask, ALL_COLS].reset_index(drop=True)
    df_eda      = df.loc[_clean_mask].reset_index(drop=True)
    total_patients = len(df_eda)
    n_dropped      = len(df) - total_patients

    print(f"  Rows after dropna: {total_patients}"
          f"  (F={(df_clean['PatientSex_enc']==0).sum()}"
          f"  M={(df_clean['PatientSex_enc']==1).sum()})"
          f"  - 제외: {n_dropped}행")

    # ── EDA ──
    run_smi_stats(df_clean, BASE_DIR, hosp_label)
    run_eda_figs(df_clean, df_eda, MODEL_COLS, BASE_DIR, total_patients, hosp_label)

    # ── 성별 그룹 루프 ──
    group_summaries  = {}
    clean_tuple_all  = None

    for sex_key, sex_val, sex_label in SEX_GROUPS:
        if sex_val is None:
            df_sub = df_clean
            cases  = make_cases(CLINICAL, SCANNER)
        else:
            df_sub = df_clean[df_clean["PatientSex_enc"] == sex_val].reset_index(drop=True)
            cases  = make_cases(CLIN_NOSEX, SCANNER)

        from config import CV_SPLITS
        if len(df_sub) < CV_SPLITS * 2:
            print(f"  [{sex_label}] too few rows ({len(df_sub)}), skipped")
            continue

        X_sub = df_sub.drop(columns=["SMI"]).reset_index(drop=True)
        y_sub = df_sub["SMI"].astype(float).reset_index(drop=True)

        result_dir = BASE_DIR / sex_key
        summary_df, clean_tuple = run_one_analysis(
            X_sub, y_sub, cases, result_dir, hosp_label, sex_label,
        )
        group_summaries[sex_key] = summary_df

        if sex_key == "all":
            clean_tuple_all = clean_tuple
            run_fullfit_analysis(X_sub, y_sub, cases, BASE_DIR, hosp_label)

    return group_summaries, clean_tuple_all



def run_cross_hospital(all_summaries: dict, COMPARE_DIR: Path) -> None:
    """강남 vs 신촌 성능 비교 그래프 + Excel 저장."""
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_summaries.values(), ignore_index=True)
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
            vals = [hdf.iloc[k][metric]  for k in range(n_cases)]
            errs = [hdf.iloc[k][std_key] for k in range(n_cases)]
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


def run_external_validation(all_clean_data: dict, COMPARE_DIR: Path) -> None:
    """Train on A, Test on B 외부 검증."""
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    hosp_labels_map = {"gangnam": "강남", "sinchon": "신촌"}
    hosp_keys       = list(all_clean_data.keys())
    directions      = [(hosp_keys[0], hosp_keys[1]), (hosp_keys[1], hosp_keys[0])]
    ext_rows        = []

    for train_key, test_key in directions:
        X_tr, y_tr_cont, y_tr_bin, CASES_tr, _ = all_clean_data[train_key]
        X_te, y_te_cont, y_te_bin, CASES_te, smi_te = all_clean_data[test_key]
        train_lbl = hosp_labels_map[train_key]
        test_lbl  = hosp_labels_map[test_key]
        print(f"\n  Train: {train_lbl} (n={len(X_tr)})  ->  Test: {test_lbl} (n={len(X_te)})")

        for case_name, feats_tr in CASES_tr.items():
            feats_common = [f for f in feats_tr if f in X_te.columns]
            missing      = set(feats_tr) - set(feats_common)
            if missing:
                print(f"    [{case_name}] Warning: {len(missing)} features missing, skipped")
            if not feats_common:
                continue

            pipe_lin = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
            pipe_lin.fit(X_tr[feats_common], y_tr_cont)
            pred_cont = pipe_lin.predict(X_te[feats_common])
            lin_r2    = r2_score(y_te_cont, pred_cont)
            lin_mae   = mean_absolute_error(y_te_cont, pred_cont)
            lin_rmse  = np.sqrt(mean_squared_error(y_te_cont, pred_cont))

            female_mask_te = X_te["PatientSex_enc"] == 0
            male_mask_te   = X_te["PatientSex_enc"] == 1
            y_te_bin_ext = pd.Series(0, index=y_te_cont.index, dtype=int)
            y_te_bin_ext[female_mask_te] = (y_te_cont[female_mask_te] < smi_te["female"]).astype(int)
            y_te_bin_ext[male_mask_te]   = (y_te_cont[male_mask_te]   < smi_te["male"]).astype(int)
            pipe_log = Pipeline([
                ("sc", StandardScaler()),
                ("m",  LogisticRegression(max_iter=2000, random_state=CV_RANDOM, solver="lbfgs")),
            ])
            pipe_log.fit(X_tr[feats_common], y_tr_bin)
            prob      = pipe_log.predict_proba(X_te[feats_common])[:, 1]
            pred_bin  = pipe_log.predict(X_te[feats_common])
            log_auc   = roc_auc_score(y_te_bin_ext, prob)
            log_acc   = accuracy_score(y_te_bin_ext, pred_bin)
            tn, fp, fn, tp = confusion_matrix(y_te_bin_ext, pred_bin).ravel()
            log_sens  = tp / (tp + fn) if tp + fn else 0
            log_spec  = tn / (tn + fp) if tn + fp else 0

            print(f"    {case_name}: LinR²={lin_r2:.4f}  AUC={log_auc:.4f}  "
                  f"Acc={log_acc:.4f}  Sens={log_sens:.4f}  Spec={log_spec:.4f}")

            ext_rows.append({
                "Train": train_lbl, "Test": test_lbl,
                "Case": case_name, "N_train": len(X_tr), "N_test": len(X_te),
                "N_features_used": len(feats_common),
                "Lin_R2": round(lin_r2, 4), "Lin_MAE": round(lin_mae, 2),
                "Lin_RMSE": round(lin_rmse, 2), "SMI_threshold": smi_te,
                "Log_AUC": round(log_auc, 4), "Log_Acc": round(log_acc, 4),
                "Log_Sens": round(log_sens, 4), "Log_Spec": round(log_spec, 4),
            })

    ext_df = pd.DataFrame(ext_rows)
    directions_labels = [f"{hosp_labels_map[a]}->{hosp_labels_map[b]}" for a, b in directions]
    dir_colors  = ["#8e44ad", "#27ae60"]
    case_names  = ext_df["Case"].unique().tolist()
    n_c         = len(case_names)
    xtick_labels = [c.replace("Case", "C").replace("_Clinical", "\nClinical")
                    .replace("+AEC_prev", "\n+AEC\nprev").replace("+AEC_new", "\n+AEC\nnew")
                    .replace("+Scanner", "\n+Scanner") for c in case_names]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(n_c); w = 0.35
    for ax, metric, title, ylim in zip(
        axes, ["Lin_R2", "Log_AUC"],
        ["External Linear R²", "External Logistic AUC"], [None, (0, 1)],
    ):
        for i, (dlabel, dcolor) in enumerate(zip(directions_labels, dir_colors)):
            sub  = ext_df[ext_df["Train"].apply(lambda v: v in dlabel)
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


def run_bmi_analysis(all_clean_data: dict, BMI_COMPARE_DIR: Path) -> None:
    """BMI 유무 비교 (Case 1 / 2 / 4 × no-BMI vs +BMI)."""
    BMI_COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    bmi_all_rows = []

    for hosp_key, (X_full, y_cont, y_bin, CASES_main, smi_thr) in all_clean_data.items():
        hosp_label  = "강남" if hosp_key == "gangnam" else "신촌"
        print(f"\n  [{hosp_label}] n={len(y_cont)}")

        scanner_cols = [c for c in X_full.columns if c.startswith("model_") or c == "kVp"]
        cases_bmi = {
            "C1_noBMI": [f for f in ["PatientAge", "PatientSex_enc"]                     if f in X_full.columns],
            "C1_BMI":   [f for f in ["PatientAge", "PatientSex_enc", "BMI"]              if f in X_full.columns],
            "C2_noBMI": [f for f in ["PatientAge", "PatientSex_enc"] + AEC_PREV          if f in X_full.columns],
            "C2_BMI":   [f for f in ["PatientAge", "PatientSex_enc", "BMI"] + AEC_PREV   if f in X_full.columns],
            "C4_noBMI": [f for f in ["PatientAge", "PatientSex_enc"] + AEC_PREV + scanner_cols  if f in X_full.columns],
            "C4_BMI":   [f for f in ["PatientAge", "PatientSex_enc", "BMI"] + AEC_PREV + scanner_cols if f in X_full.columns],
        }
        LABELS_BMI = {
            "C1_noBMI": "Case 1\n(no BMI)", "C1_BMI": "Case 1\n(+BMI)",
            "C2_noBMI": "Case 2\n(no BMI)", "C2_BMI": "Case 2\n(+BMI)",
            "C4_noBMI": "Case 4\n(no BMI)", "C4_BMI": "Case 4\n(+BMI)",
        }

        lin_bmi, log_bmi = {}, {}
        for cname, feats in cases_bmi.items():
            if not feats: continue
            lr = linear_cv(X_full[feats], y_cont)
            lo = logistic_cv(X_full[feats], y_bin)
            lin_bmi[cname] = lr; log_bmi[cname] = lo
            print(f"    {cname:12s}: Lin R²={lr['R2']:.4f}±{lr['R2_std']:.4f}"
                  f"  AUC={lo['AUC']:.4f}±{lo['AUC_std']:.4f}")
            bmi_all_rows.append({
                "Hospital": hosp_label, "Case": cname,
                "Label": LABELS_BMI[cname].replace("\n", " "),
                "Has_BMI": "BMI" in cname,
                "N_features": len(feats),
                "Lin_R2":    round(lr["R2"], 4),  "Lin_R2_std":  round(lr["R2_std"], 4),
                "Lin_MAE":   round(lr["MAE"], 2),  "Lin_RMSE":    round(lr["RMSE"], 2),
                "Log_AUC":   round(lo["AUC"], 4),  "Log_AUC_std": round(lo["AUC_std"], 4),
                "Log_Acc":   round(lo["Accuracy"], 4),
                "Log_Sens":  round(lo["Sensitivity"], 4),
                "Log_Spec":  round(lo["Specificity"], 4),
                "SMI_threshold": f"F:{smi_thr['female']:.2f}/M:{smi_thr['male']:.2f}",
            })

        if len(lin_bmi) < 6:
            print(f"    일부 케이스 누락 - BMI 비교 플롯 생략"); continue

        case_pairs  = [("C1_noBMI","C1_BMI"), ("C2_noBMI","C2_BMI"), ("C4_noBMI","C4_BMI")]
        pair_labels = ["Case 1\n(임상 기준선)", "Case 2\n(+AEC_prev)", "Case 4\n(+AEC_prev\n+Scanner)"]
        x_pos = np.arange(len(case_pairs)); w_bar = 0.35

        # Fig A: R² & AUC
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, res_d, metric, std_key, ylabel, ylim in [
            (axes[0], lin_bmi, "R2",  "R2_std",  "5-Fold CV R²",  None),
            (axes[1], log_bmi, "AUC", "AUC_std", "5-Fold CV AUC", (0, 1)),
        ]:
            for i, (no_k, bmi_k) in enumerate(case_pairs):
                v_no  = res_d[no_k][metric];  e_no  = res_d[no_k][std_key]
                v_bmi = res_d[bmi_k][metric]; e_bmi = res_d[bmi_k][std_key]
                b1 = ax.bar(x_pos[i] - w_bar/2, v_no,  w_bar,
                            color="#aaaaaa", edgecolor="white",
                            yerr=e_no,  capsize=4, label="no BMI" if i == 0 else "")
                b2 = ax.bar(x_pos[i] + w_bar/2, v_bmi, w_bar,
                            color="#2ecc71", edgecolor="white",
                            yerr=e_bmi, capsize=4, label="+BMI"   if i == 0 else "")
                for bar, v in [(b1, v_no), (b2, v_bmi)]:
                    ax.text(bar[0].get_x()+bar[0].get_width()/2,
                            bar[0].get_height()+max(e_no,e_bmi)+0.003,
                            f"{v:.4f}", ha="center", va="bottom", fontsize=8)
                delta = v_bmi - v_no
                mid_x = x_pos[i]; mid_y = max(v_no + e_no, v_bmi + e_bmi) + 0.025
                ax.annotate(f"Δ={delta:+.4f}", xy=(mid_x, mid_y), ha="center", fontsize=9,
                            fontweight="bold", color="#27ae60" if delta > 0 else "#e74c3c")
            ax.set_xticks(x_pos); ax.set_xticklabels(pair_labels, fontsize=9)
            ax.set_ylabel(ylabel)
            if ylim: ax.set_ylim(*ylim)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
            ax.legend(fontsize=10)
        plt.suptitle(f"[{hosp_label}] BMI 유무 비교 — Case 1 / 2 / 4\n"
                     f"(회색 = no BMI / 초록 = +BMI  |  Δ = BMI 추가 효과)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_a = BMI_COMPARE_DIR / hosp_key / "bmi_comparison_r2_auc.png"
        out_a.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_a, dpi=150); plt.close()
        print(f"    Saved: {out_a.name}")

        # Fig B: Delta 막대
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        deltas_r2  = [lin_bmi[bk]["R2"]  - lin_bmi[nk]["R2"]  for nk, bk in case_pairs]
        deltas_auc = [log_bmi[bk]["AUC"] - log_bmi[nk]["AUC"] for nk, bk in case_pairs]
        for ax, deltas, ylabel, title_d in [
            (axes[0], deltas_r2,  "ΔR² (BMI 추가 효과)",  "선형 R² 변화량"),
            (axes[1], deltas_auc, "ΔAUC (BMI 추가 효과)", "로지스틱 AUC 변화량"),
        ]:
            colors_d = ["#27ae60" if d >= 0 else "#e74c3c" for d in deltas]
            bars_d   = ax.bar(pair_labels, deltas, color=colors_d, edgecolor="white", width=0.5)
            ax.axhline(0, color="black", lw=1.2)
            for bar, d in zip(bars_d, deltas):
                y_pos = d + 0.0005 if d >= 0 else d - 0.0015
                ax.text(bar.get_x()+bar.get_width()/2, y_pos,
                        f"{d:+.4f}", ha="center", va="bottom" if d >= 0 else "top",
                        fontsize=10, fontweight="bold",
                        color="#27ae60" if d >= 0 else "#e74c3c")
            ax.set_ylabel(ylabel); ax.set_title(title_d)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
        plt.suptitle(f"[{hosp_label}] BMI 추가 효과 (Δ = +BMI − no BMI)\n"
                     f"양수=BMI 포함 시 향상 / 음수=BMI 포함 시 저하",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_b = BMI_COMPARE_DIR / hosp_key / "bmi_delta_effect.png"
        fig.savefig(out_b, dpi=150); plt.close()
        print(f"    Saved: {out_b.name}")

    # Excel 저장
    if bmi_all_rows:
        bmi_df  = pd.DataFrame(bmi_all_rows)
        bmi_out = BMI_COMPARE_DIR / "bmi_comparison_summary.xlsx"
        delta_rows = []
        for hosp in bmi_df["Hospital"].unique():
            sub = bmi_df[bmi_df["Hospital"] == hosp]
            for base in ["C1", "C2", "C4"]:
                row_no  = sub[sub["Case"] == f"{base}_noBMI"]
                row_bmi = sub[sub["Case"] == f"{base}_BMI"]
                if row_no.empty or row_bmi.empty: continue
                delta_rows.append({
                    "Hospital":       hosp, "Case_base": base,
                    "Delta_Lin_R2":   round(float(row_bmi["Lin_R2"].values[0])  - float(row_no["Lin_R2"].values[0]),  4),
                    "Delta_Lin_RMSE": round(float(row_bmi["Lin_RMSE"].values[0]) - float(row_no["Lin_RMSE"].values[0]), 2),
                    "Delta_Log_AUC":  round(float(row_bmi["Log_AUC"].values[0]) - float(row_no["Log_AUC"].values[0]), 4),
                    "Delta_Log_Acc":  round(float(row_bmi["Log_Acc"].values[0]) - float(row_no["Log_Acc"].values[0]), 4),
                    "noBMI_Lin_R2":   float(row_no["Lin_R2"].values[0]),
                    "BMI_Lin_R2":     float(row_bmi["Lin_R2"].values[0]),
                    "noBMI_Log_AUC":  float(row_no["Log_AUC"].values[0]),
                    "BMI_Log_AUC":    float(row_bmi["Log_AUC"].values[0]),
                })
        delta_df = pd.DataFrame(delta_rows)
        print(f"\n  BMI Delta 요약:")
        print(delta_df[["Hospital","Case_base","Delta_Lin_R2","Delta_Log_AUC"]].to_string(index=False))
        with pd.ExcelWriter(str(bmi_out)) as writer:
            bmi_df.to_excel(writer,   sheet_name="all_results", index=False)
            delta_df.to_excel(writer, sheet_name="delta_bmi",   index=False)
        print(f"\n  Saved: {bmi_out}")


def main():
    # ── 1. 병원별 분석 ──
    all_summaries  = {}
    all_clean_data = {}

    for hosp_key, data_path in HOSPITALS.items():
        group_summaries, clean_tuple_all = run_hospital(hosp_key, data_path)
        if clean_tuple_all is not None:
            all_summaries[hosp_key]  = group_summaries["all"]
            all_clean_data[hosp_key] = clean_tuple_all

    COMPARE_DIR = SCRIPT_DIR.parent / "results" / "regression"

    # ── 2. 교차 병원 비교 ──
    if len(all_summaries) == 2:
        print(f"\n{'='*60}\nCROSS-HOSPITAL COMPARISON (전체)\n{'='*60}")
        run_cross_hospital(all_summaries, COMPARE_DIR)

    # ── 3. 외부 검증 ──
    if len(all_clean_data) == 2:
        print(f"\n{'='*60}\nEXTERNAL VALIDATION (train → test)\n{'='*60}")
        run_external_validation(all_clean_data, COMPARE_DIR)

    # ── 4. BMI 기여도 분석 ──
    if all_clean_data:
        print(f"\n{'='*60}\nBMI CONTRIBUTION ANALYSIS (Case 1 / 2 / 4 × BMI 유무)\n{'='*60}")
        run_bmi_analysis(all_clean_data, COMPARE_DIR)

    print(f"\n{'='*60}\nALL DONE\n{'='*60}")


if __name__ == "__main__":
    main()
