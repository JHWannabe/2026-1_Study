# -*- coding: utf-8 -*-
"""
EDA 플롯 생성 모듈.

run_tama_stats(): 성별 TAMA 기술통계 + Figs 19, 20
run_eda_figs():   Figs 01 (AEC 상관), 02 (VIF), 03 (TAMA 분포), 16 (스캐너), 17 (kVp), 18 (상관행렬)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as sm_vif
from matplotlib.patches import Patch

from config import AEC_PREV


def run_tama_stats(df_clean: pd.DataFrame, BASE_DIR: Path, hosp_label: str) -> pd.DataFrame:
    """
    성별 TAMA 기술통계 계산 → Excel + Fig 19 (히스토그램 패널) + Fig 20 (박스플롯) 저장.
    Returns: tama_stats_df
    """
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 기술통계 ──
    tama_stats_rows = []
    for sex_enc, sex_label in [(None, "전체"), (0, "여성(F)"), (1, "남성(M)")]:
        sub  = df_clean if sex_enc is None else df_clean[df_clean["PatientSex_enc"] == sex_enc]
        tama = sub["TAMA"].dropna()
        tama_stats_rows.append({
            "그룹":   sex_label,
            "N":      len(tama),
            "Mean":   round(tama.mean(), 2),
            "SD":     round(tama.std(), 2),
            "Median": round(tama.median(), 2),
            "P25":    round(tama.quantile(0.25), 2),
            "P75":    round(tama.quantile(0.75), 2),
            "Min":    round(tama.min(), 2),
            "Max":    round(tama.max(), 2),
        })

    tama_stats_df = pd.DataFrame(tama_stats_rows)
    print(f"\n  [TAMA 기술통계 - {hosp_label}]")
    print(tama_stats_df.to_string(index=False))
    tama_stats_df.to_excel(BASE_DIR / "tama_sex_stats.xlsx", index=False)
    print(f"  Saved: tama_sex_stats.xlsx")

    # ── Fig 19: TAMA 히스토그램 패널 (전체 / 여성 / 남성) ──
    palette = {"전체": "#7f8c8d", "여성(F)": "#e67e22", "남성(M)": "#2980b9"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (sex_enc, sex_label) in zip(axes, [(None, "전체"), (0, "여성(F)"), (1, "남성(M)")]):
        sub      = df_clean if sex_enc is None else df_clean[df_clean["PatientSex_enc"] == sex_enc]
        tama     = sub["TAMA"].dropna()
        color    = palette[sex_label]
        p25_v    = tama.quantile(0.25)
        median_v = tama.median()
        mean_v   = tama.mean()

        ax.hist(tama, bins=35, color=color, alpha=0.7, edgecolor="white", linewidth=0.4)
        ax.axvline(p25_v,    color="#e74c3c", linestyle="--", linewidth=1.8, label=f"P25: {p25_v:.1f}")
        ax.axvline(median_v, color="#2c3e50", linestyle="-",  linewidth=1.8, label=f"Median: {median_v:.1f}")
        ax.axvline(mean_v,   color="#27ae60", linestyle=":",  linewidth=1.8, label=f"Mean: {mean_v:.1f}")
        ax.set_xlabel("TAMA (cm²)", fontsize=10)
        ax.set_ylabel("환자 수", fontsize=10)
        ax.set_title(
            f"{sex_label}  (N={len(tama)})\n"
            f"Mean={mean_v:.1f}, SD={tama.std():.1f}, Median={median_v:.1f}\n"
            f"P25={p25_v:.1f}, P75={tama.quantile(0.75):.1f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"[{hosp_label}] 성별 TAMA 분포 (정제 후 데이터 기준)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(BASE_DIR / "19_tama_sex_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: 19_tama_sex_distribution.png")

    # ── Fig 20: 성별 TAMA 박스플롯 ──
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = {
        "여성(F)": df_clean[df_clean["PatientSex_enc"] == 0]["TAMA"].dropna(),
        "남성(M)": df_clean[df_clean["PatientSex_enc"] == 1]["TAMA"].dropna(),
    }
    bp = ax.boxplot(
        list(groups.values()),
        labels=list(groups.keys()),
        patch_artist=True,
        medianprops=dict(color="#e74c3c", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    box_colors = ["#e67e22", "#2980b9"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for i, (sex_label, tama) in enumerate(groups.items(), start=1):
        ax.scatter([i + 0.25] * len(tama), tama,
                   alpha=0.15, s=6, color=box_colors[i-1], zorder=3)
    ax.set_ylabel("TAMA (cm²)", fontsize=11)
    ax.set_title(f"[{hosp_label}] 성별 TAMA 분포 박스플롯\n(점: 개별 환자)",
                 fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(BASE_DIR / "20_tama_sex_boxplot.png", dpi=150)
    plt.close()
    print(f"  Saved: 20_tama_sex_boxplot.png")

    return tama_stats_df


def run_eda_figs(
    df_clean: pd.DataFrame,
    df_eda: pd.DataFrame,
    MODEL_COLS: list,
    BASE_DIR: Path,
    total_patients: int,
    hosp_label: str,
) -> None:
    """
    EDA 그래프 생성.
    - Fig 01: AEC Feature-TAMA Pearson 상관 Top 20
    - Fig 02: VIF (다중공선성 검사)
    - Fig 03: 성별 TAMA 분포 및 이진화 임계값
    - Fig 16: CT 스캐너 모델 분포
    - Fig 17: kVp 분포
    - Fig 18: Pearson 상관행렬 (Clinical + AEC_prev)
    """
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Fig 01: AEC Feature - TAMA 상관 Top 20 ──
    non_aec_set = ({"PatientID", "PatientAge", "PatientSex", "PatientSex_enc",
                    "BMI", "ManufacturerModelName", "kVp", "TAMA"} | set(MODEL_COLS))
    aec_all_cols = [c for c in df_eda.columns if c not in non_aec_set]
    corr_rows01  = []
    for f in aec_all_cols:
        sub = df_eda[[f, "TAMA"]].dropna()
        if len(sub) < 10:
            continue
        r01, _ = scipy_stats.pearsonr(sub[f], sub["TAMA"])
        corr_rows01.append({"feature": f, "r": r01})
    if corr_rows01:
        corr_df01 = (
            pd.DataFrame(corr_rows01)
            .assign(abs_r=lambda d: d["r"].abs())
            .sort_values("abs_r", ascending=False)
            .head(20)
            .sort_values("r")
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        cols01  = ["#e07070" if r > 0 else "#7090d0" for r in corr_df01["r"]]
        ax.barh(corr_df01["feature"], corr_df01["r"], color=cols01, height=0.7)
        ax.axvline(0, color="black", lw=0.8)
        ax.legend(handles=[Patch(color="#e07070", label="양의 상관 (+)"),
                           Patch(color="#7090d0", label="음의 상관 (-)")], fontsize=9)
        ax.set_xlabel("Pearson r with TAMA")
        ax.set_title(f"AEC Feature - TAMA 상관계수 Top 20\n(붉은색=양, 파란색=음)")
        ax.xaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
        plt.tight_layout()
        fig.savefig(BASE_DIR / "01_feature_correlation.png", dpi=150); plt.close()
        print(f"  Saved: 01_feature_correlation.png")

    # ── Fig 02: VIF (Clinical + AEC_prev) ──
    vif_feats02 = [f for f in ["PatientAge", "PatientSex_enc", "BMI"] + AEC_PREV
                   if f in df_clean.columns]
    X_vif02     = df_clean[vif_feats02].copy()
    X_vif02_std = (X_vif02 - X_vif02.mean()) / X_vif02.std()
    lbl_vif     = {"PatientSex_enc": "Sex", "PatientAge": "PatientAge", "BMI": "BMI"}
    vif_names02 = [lbl_vif.get(f, f) for f in vif_feats02]
    vif_vals02  = [sm_vif(X_vif02_std.values, i) for i in range(X_vif02_std.shape[1])]
    vif_data02  = sorted(zip(vif_names02, vif_vals02), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(9, max(4, len(vif_data02) * 0.7 + 1.5)))
    vif_ns = [n for n, _ in vif_data02]; vif_vs = [v for _, v in vif_data02]
    ax.barh(vif_ns, vif_vs, color="#27ae60", height=0.6)
    for i, (n, v) in enumerate(vif_data02):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(5,  color="#f39c12", ls="--", lw=1.5, label="VIF=5 (주의)")
    ax.axvline(10, color="#e07070", ls="--", lw=1.5, label="VIF=10 (위험)")
    ax.set_xlabel("VIF (분산팽창인수)")
    ax.set_title("선택 변수의 VIF (다중공선성 검사)\nVIF<5: 낮음, 5-10: 중간, >10: 높음")
    ax.legend(fontsize=9); ax.xaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout(); fig.savefig(BASE_DIR / "02_vif_comparison.png", dpi=150); plt.close()
    print(f"  Saved: 02_vif_comparison.png")

    # ── Fig 03: 성별 TAMA 분포 및 이진화 임계값 ──
    df_m03  = df_clean[df_clean["PatientSex_enc"] == 1]["TAMA"].astype(float)
    df_f03  = df_clean[df_clean["PatientSex_enc"] == 0]["TAMA"].astype(float)
    thr_m03 = df_m03.quantile(0.25)
    thr_f03 = df_f03.quantile(0.25)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, lbl_sex, color, thr in [
        (axes[0], df_m03, "남성 (M)", "#6baed6", thr_m03),
        (axes[1], df_f03, "여성 (F)", "#fd8d3c", thr_f03),
    ]:
        ax.hist(data, bins=30, color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(thr, color="#e74c3c", ls="--", lw=2, label=f"임계값 {thr:.0f} cm²")
        ax.axvline(data.mean(), color="black", ls="-", lw=1.5, label=f"평균 {data.mean():.1f} cm²")
        low_n = int((data < thr).sum())
        ax.set_xlabel("TAMA (cm²)"); ax.set_ylabel("환자 수")
        ax.set_title(f"{lbl_sex} TAMA 분포\n(N={len(data)}, 중앙값={data.median():.0f})")
        ax.text(0.62, 0.92, f"임계값 이하: {low_n}명 ({low_n/len(data)*100:.1f}%)",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.legend(fontsize=9)
    plt.suptitle(f"[{hosp_label}] 성별 TAMA 분포 및 이진화 임계값",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(BASE_DIR / "03_tama_distribution.png", dpi=150); plt.close()
    print(f"  Saved: 03_tama_distribution.png")

    # ── Fig 16: CT 스캐너 모델 분포 ──
    scanner_counts  = df_eda["ManufacturerModelName"].value_counts()
    n_total_models  = scanner_counts.shape[0]
    top10           = scanner_counts.head(10)
    others_n        = scanner_counts.iloc[10:].sum()
    plot_scan = pd.concat([top10, pd.Series({"기타": others_n})]) if others_n > 0 else top10
    plot_scan = plot_scan[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ["#3498db" if name != "기타" and i == len(plot_scan) - 1 else "#a0aab0"
                  for i, name in enumerate(plot_scan.index)]
    ax.barh(plot_scan.index, plot_scan.values, color=bar_colors)
    for name, val in zip(plot_scan.index, plot_scan.values):
        pct = val / total_patients * 100
        ax.text(val + total_patients * 0.004, list(plot_scan.index).index(name),
                f"{val} ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel("환자 수")
    ax.set_title(f"CT 스캐너 모델 분포 (총 {n_total_models}종)")
    ax.set_xlim(0, plot_scan.max() * 1.25)
    plt.tight_layout()
    fig.savefig(BASE_DIR / "16_scanner_distribution.png", dpi=150); plt.close()
    print(f"  Saved: 16_scanner_distribution.png")

    # ── Fig 17: kVp 분포 ──
    kvp_counts   = df_eda["kVp"].dropna().astype(int).value_counts().sort_index()
    kvp_missing  = total_patients - kvp_counts.sum()
    dominant_kvp = kvp_counts.idxmax()
    fig, ax = plt.subplots(figsize=(9, 5))
    kvp_colors = ["#3498db" if v == dominant_kvp else "#a0aab0" for v in kvp_counts.index]
    bars = ax.bar(kvp_counts.index.astype(str), kvp_counts.values, color=kvp_colors, width=0.6)
    for bar, val in zip(bars, kvp_counts.values):
        pct = val / total_patients * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + kvp_counts.max() * 0.01,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("kVp"); ax.set_ylabel("환자 수")
    missing_note = f"  (kVp 결측: {kvp_missing}명)" if kvp_missing > 0 else ""
    ax.set_title(f"kVp 분포 (주요값: {dominant_kvp} kVp)\n전체 {total_patients}명 기준{missing_note}")
    ax.set_ylim(0, kvp_counts.max() * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(BASE_DIR / "17_kvp_distribution.png", dpi=150); plt.close()
    print(f"  Saved: 17_kvp_distribution.png"
          + (f"  (kVp 결측 {kvp_missing}명 제외됨)" if kvp_missing > 0 else ""))

    # ── Fig 18: Pearson 상관행렬 (Clinical + AEC_prev) ──
    corr_feat_order = [f for f in ["PatientAge", "PatientSex_enc", "BMI"] + AEC_PREV
                       if f in df_clean.columns]
    X_corr   = df_clean[corr_feat_order].rename(columns={"PatientSex_enc": "Sex"})
    corr_mat = X_corr.corr(method="pearson")
    n_cf     = len(corr_mat)
    mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(max(7, n_cf * 0.9), max(6, n_cf * 0.85)))
    sns.heatmap(
        corr_mat, mask=mask_upper, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, square=True,
        annot_kws={"size": 10},
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("선택 Feature 간 Pearson 상관행렬\n(|r| > 0.8: 다중공선성 주의)", fontsize=12)
    plt.tight_layout()
    fig.savefig(BASE_DIR / "18_correlation_matrix.png", dpi=150); plt.close()
    print(f"  Saved: 18_correlation_matrix.png")
