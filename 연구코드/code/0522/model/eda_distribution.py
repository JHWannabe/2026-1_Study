"""
전체 데이터셋 분포 분석 — 모델 실행 전 EDA.

저장 위치: 연구코드/results/0522/eda/
  이미지: eda_01_overview.png ~ eda_10_aec_per_patient_stats.png
  엑셀:   eda_statistics.xlsx  (4개 시트)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats as scipy_stats

# ── 경로 설정 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)
sys.path.insert(0, os.path.dirname(__file__))

DATA_PATH   = os.path.join(PROJECT_ROOT, "연구코드/data/강남/강남_merged_features.xlsx")
OUT_DIR     = os.path.join(PROJECT_ROOT, "연구코드/results/0522/eda")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 상수 (model/config.py와 동일) ──────────────────────────────────────────
SMI_THRESH_M  = 40.96
SMI_THRESH_F  = 30.6
MIN_MFR_RATIO = 0.05
KVP_FILTER    = 100
AEC_LEN       = 256

# ── 스타일 ─────────────────────────────────────────────────────────────────
PALETTE      = {"overall": "#4C72B0", "M": "#2196F3", "F": "#E91E63",
                "sarc1": "#E53935",  "sarc0": "#43A047"}
LABEL_MAP    = {0: "Normal", 1: "Sarcopenia"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150,
})


# ══════════════════════════════════════════════════════════════════════════
# 1.  데이터 로드
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading metadata ...")
    meta = pd.read_excel(DATA_PATH, sheet_name="metadata")
    base_cols = ["PatientID", "PatientAge", "PatientSex", "BMI", "SMI",
                 "kVp", "ManufacturerModelName"]
    opt_cols  = [c for c in ["TAMA"] if c in meta.columns]
    meta = meta[base_cols + opt_cols].dropna().reset_index(drop=True)
    meta["PatientID"] = meta["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    # kVp 필터
    before = len(meta)
    meta = meta[meta["kVp"] == KVP_FILTER].reset_index(drop=True)
    print(f"  kVp=={KVP_FILTER} filter: {before} -> {len(meta)}")

    # 소수 제조사 제거
    mfr_counts = meta["ManufacturerModelName"].value_counts()
    major = mfr_counts[mfr_counts / len(meta) >= MIN_MFR_RATIO].index
    removed = (~meta["ManufacturerModelName"].isin(major)).sum()
    if removed:
        print(f"  Minor MFR removed: {removed} samples")
    meta = meta[meta["ManufacturerModelName"].isin(major)].reset_index(drop=True)

    # 18세 미만 제거
    before_age = len(meta)
    meta = meta[meta["PatientAge"] >= 18].reset_index(drop=True)
    removed_age = before_age - len(meta)
    if removed_age:
        print(f"  Age < 18 removed: {removed_age} samples ({before_age} -> {len(meta)})")

    # 레이블
    meta["label"] = meta.apply(
        lambda r: 1 if r["SMI"] <= (SMI_THRESH_M if r["PatientSex"] == "M" else SMI_THRESH_F) else 0,
        axis=1,
    )

    print("Loading AEC (256-pt) ...")
    aec = pd.read_excel(DATA_PATH, sheet_name=f"aec_{AEC_LEN}")
    aec["PatientID"] = aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    aec_cols = [c for c in aec.columns if c != "PatientID"][:AEC_LEN]

    df = pd.merge(meta, aec[["PatientID"] + aec_cols], on="PatientID", how="inner").reset_index(drop=True)

    # flat AEC 제거
    flat = df[aec_cols].std(axis=1) == 0
    if flat.any():
        print(f"  Flat AEC removed: {flat.sum()}")
    df = df[~flat].reset_index(drop=True)

    print(f"  Final samples: {len(df)}")
    return df, aec_cols


# ══════════════════════════════════════════════════════════════════════════
# 2.  헬퍼
# ══════════════════════════════════════════════════════════════════════════

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def desc_stats(series: pd.Series) -> dict:
    return {
        "N": len(series), "Mean": series.mean(), "SD": series.std(),
        "Min": series.min(), "P25": series.quantile(0.25),
        "Median": series.median(), "P75": series.quantile(0.75),
        "Max": series.max(),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3.  그림 함수들
# ══════════════════════════════════════════════════════════════════════════

# ── Fig 01: 전체 개요 ───────────────────────────────────────────────────
def fig_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Dataset Overview", fontweight="bold", fontsize=13)

    # (a) 전체 N, 성비
    ax = axes[0]
    n_m = (df["PatientSex"] == "M").sum()
    n_f = (df["PatientSex"] == "F").sum()
    bars = ax.bar(["Male", "Female"], [n_m, n_f],
                  color=[PALETTE["M"], PALETTE["F"]], width=0.5)
    ax.bar_label(bars, labels=[f"{n_m}\n({n_m/len(df)*100:.1f}%)",
                                f"{n_f}\n({n_f/len(df)*100:.1f}%)"],
                 padding=3, fontsize=9)
    ax.set_title(f"Sex Distribution  (N={len(df)})")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(n_m, n_f) * 1.25)

    # (b) Sarcopenia 유병률 by sex
    ax = axes[1]
    groups = {"Overall": df, "Male": df[df["PatientSex"] == "M"],
              "Female": df[df["PatientSex"] == "F"]}
    prev = {g: sub["label"].mean() * 100 for g, sub in groups.items()}
    colors = [PALETTE["overall"], PALETTE["M"], PALETTE["F"]]
    bars = ax.bar(list(prev.keys()), list(prev.values()), color=colors, width=0.5)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in prev.values()],
                 padding=3, fontsize=9)
    ax.set_title("Sarcopenia Prevalence")
    ax.set_ylabel("Prevalence (%)")
    ax.set_ylim(0, max(prev.values()) * 1.3)

    # (c) Scanner 분포
    ax = axes[2]
    mfr = df["ManufacturerModelName"].value_counts()
    colors_mfr = sns.color_palette("Set2", len(mfr))
    bars = ax.barh(mfr.index, mfr.values, color=colors_mfr)
    ax.bar_label(bars, labels=[f"{v} ({v/len(df)*100:.1f}%)" for v in mfr.values],
                 padding=3, fontsize=8)
    ax.set_title("Scanner Model")
    ax.set_xlabel("Count")
    ax.set_xlim(0, mfr.max() * 1.35)

    plt.tight_layout()
    savefig("eda_01_overview.png")


# ── Fig 02~05: 임상 연속 변수 ──────────────────────────────────────────
def fig_continuous(df, col, label_str, unit, thresh=None, fname=None):
    has_tama = col in df.columns
    if not has_tama:
        return

    fig = plt.figure(figsize=(16, 5))
    gs  = GridSpec(1, 4, figure=fig, wspace=0.35)
    fig.suptitle(f"{label_str} Distribution", fontweight="bold", fontsize=13)

    # (a) 전체 히스토그램 + KDE
    ax = fig.add_subplot(gs[0])
    series = df[col].dropna()
    ax.hist(series, bins=30, color=PALETTE["overall"], alpha=0.7, edgecolor="white")
    ax2 = ax.twinx()
    kde_x = np.linspace(series.min(), series.max(), 300)
    kde   = scipy_stats.gaussian_kde(series)
    ax2.plot(kde_x, kde(kde_x), color="navy", lw=1.5)
    ax2.set_yticks([])
    if thresh is not None:
        ax.axvline(SMI_THRESH_M, color=PALETTE["M"], ls="--", lw=1.2, label=f"M thr={SMI_THRESH_M}")
        ax.axvline(SMI_THRESH_F, color=PALETTE["F"], ls="--", lw=1.2, label=f"F thr={SMI_THRESH_F}")
        ax.legend(fontsize=7)
    ax.set_title("Overall")
    ax.set_xlabel(unit)
    ax.set_ylabel("Count")

    # (b) Box by sex
    ax = fig.add_subplot(gs[1])
    data_sex = [df.loc[df["PatientSex"] == s, col].dropna() for s in ["M", "F"]]
    bp = ax.boxplot(data_sex, patch_artist=True, widths=0.45, notch=False)
    for patch, color in zip(bp["boxes"], [PALETTE["M"], PALETTE["F"]]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    if thresh is not None:
        ax.axhline(SMI_THRESH_M, color=PALETTE["M"], ls="--", lw=1)
        ax.axhline(SMI_THRESH_F, color=PALETTE["F"], ls="--", lw=1)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Male", "Female"])
    ax.set_title("By Sex"); ax.set_ylabel(unit)

    # (c) Violin by sex
    ax = fig.add_subplot(gs[2])
    plot_df = df[["PatientSex", col]].dropna()
    sns.violinplot(ax=ax, data=plot_df, x="PatientSex", y=col,
                   palette={"M": PALETTE["M"], "F": PALETTE["F"]},
                   inner="quartile", order=["M", "F"])
    ax.set_title("Violin by Sex"); ax.set_ylabel(unit); ax.set_xlabel("")

    # (d) Box by sarcopenia label
    ax = fig.add_subplot(gs[3])
    data_lbl = [df.loc[df["label"] == l, col].dropna() for l in [0, 1]]
    bp = ax.boxplot(data_lbl, patch_artist=True, widths=0.45)
    for patch, color in zip(bp["boxes"], [PALETTE["sarc0"], PALETTE["sarc1"]]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Normal", "Sarcopenia"])
    ax.set_title("By Label"); ax.set_ylabel(unit)

    # 통계 텍스트
    for i, (grp, sub) in enumerate(
        [("M", df[df["PatientSex"]=="M"][col].dropna()),
         ("F", df[df["PatientSex"]=="F"][col].dropna())]
    ):
        t, p = scipy_stats.ttest_ind(
            df.loc[(df["PatientSex"]==grp) & (df["label"]==0), col].dropna(),
            df.loc[(df["PatientSex"]==grp) & (df["label"]==1), col].dropna(),
        )
        fig.text(0.25 + i * 0.2, -0.02,
                 f"{grp}: Normal vs Sarcopenia  t={t:.2f}  p={p:.3f}",
                 ha="center", fontsize=8, color=PALETTE[grp])

    plt.tight_layout()
    savefig(fname or f"eda_{col}.png")


# ── Fig 06: Scanner 상세 ───────────────────────────────────────────────
def fig_scanner(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Scanner Distribution", fontweight="bold", fontsize=13)

    # (a) 모델별 sex 분포 stacked bar
    ax = axes[0]
    mfr_sex = df.groupby(["ManufacturerModelName", "PatientSex"]).size().unstack(fill_value=0)
    mfr_sex.plot(kind="bar", stacked=True, ax=ax,
                 color=[PALETTE["M"], PALETTE["F"]], width=0.6)
    ax.set_title("Scanner Model × Sex"); ax.set_xlabel(""); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Sex")

    # (b) 모델별 sarcopenia 유병률
    ax = axes[1]
    prev = df.groupby("ManufacturerModelName")["label"].mean() * 100
    bars = ax.bar(prev.index, prev.values,
                  color=sns.color_palette("Set2", len(prev)), width=0.6)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in prev.values], padding=3, fontsize=9)
    ax.set_title("Sarcopenia Prevalence by Scanner"); ax.set_ylabel("Prevalence (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, prev.max() * 1.3)

    plt.tight_layout()
    savefig("eda_06_scanner.png")


# ── Fig 07: Correlation matrix ────────────────────────────────────────
def fig_correlation(df):
    cont_cols = [c for c in ["PatientAge", "BMI", "SMI", "TAMA"] if c in df.columns]
    corr = df[cont_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix — Clinical Variables", fontweight="bold")
    plt.tight_layout()
    savefig("eda_07_correlation.png")


# ── Fig 08: AEC 평균 곡선 ─────────────────────────────────────────────
def fig_aec_mean_curves(df, aec_cols):
    x = np.arange(1, len(aec_cols) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
    fig.suptitle("AEC Mean Curves (256-pt interpolation)", fontweight="bold", fontsize=13)

    def plot_curve(ax, groups, title):
        for lbl, sub, color in groups:
            vals = sub[aec_cols].values.astype(float)
            m    = np.nanmean(vals, axis=0)
            s    = np.nanstd(vals,  axis=0)
            ax.plot(x, m, color=color, lw=1.8, label=lbl)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)
        ax.set_title(title); ax.set_xlabel("Slice position (caudal → cranial)")
        ax.set_ylabel("mA"); ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(32))

    plot_curve(axes[0],
               [("Overall", df, PALETTE["overall"])],
               "Overall Mean ± 1 SD")
    plot_curve(axes[1],
               [("Male",   df[df["PatientSex"]=="M"], PALETTE["M"]),
                ("Female", df[df["PatientSex"]=="F"], PALETTE["F"])],
               "By Sex")
    plot_curve(axes[2],
               [("Normal",     df[df["label"]==0], PALETTE["sarc0"]),
                ("Sarcopenia", df[df["label"]==1], PALETTE["sarc1"])],
               "By Sarcopenia Label")

    plt.tight_layout()
    savefig("eda_08_aec_mean_curves.png")


# ── Fig 09: AEC Heatmap ───────────────────────────────────────────────
def fig_aec_heatmap(df, aec_cols):
    # label, sex 기준 정렬
    sort_df = df.sort_values(["label", "PatientSex"]).reset_index(drop=True)
    mat = sort_df[aec_cols].values.astype(float)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis",
                   interpolation="nearest", origin="upper")
    plt.colorbar(im, ax=ax, label="mA", shrink=0.8)

    # 경계선 (label 변경 지점)
    split_idx = sort_df["label"].searchsorted(1)
    ax.axhline(split_idx - 0.5, color="red", lw=1.5, ls="--",
               label=f"Sarcopenia boundary (row {split_idx})")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("AEC Heatmap  (sorted by label, then sex — caudal→cranial)",
                 fontweight="bold")
    ax.set_xlabel("Slice position"); ax.set_ylabel("Patient (sorted)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(32))
    plt.tight_layout()
    savefig("eda_09_aec_heatmap.png")


# ── Fig 10: AEC per-patient statistics ────────────────────────────────
def fig_aec_per_patient_stats(df, aec_cols):
    vals   = df[aec_cols].values.astype(float)
    aec_mean  = vals.mean(axis=1)
    aec_std   = vals.std(axis=1)
    aec_range = vals.max(axis=1) - vals.min(axis=1)
    aec_cv    = aec_std / (aec_mean + 1e-8)

    stat_df = pd.DataFrame({
        "mean_mA": aec_mean, "std_mA": aec_std,
        "range_mA": aec_range, "CV": aec_cv,
        "label": df["label"].values,
        "PatientSex": df["PatientSex"].values,
    })

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("AEC Per-Patient Statistics", fontweight="bold", fontsize=13)

    stats_info = [
        ("mean_mA",  "Mean mA"),
        ("std_mA",   "SD mA"),
        ("range_mA", "Range mA"),
        ("CV",       "CV (std/mean)"),
    ]

    for col_idx, (scol, slabel) in enumerate(stats_info):
        # 상단: 히스토그램 by label
        ax = axes[0][col_idx]
        for lbl, color in [(0, PALETTE["sarc0"]), (1, PALETTE["sarc1"])]:
            sub = stat_df.loc[stat_df["label"] == lbl, scol]
            ax.hist(sub, bins=25, alpha=0.6, color=color, label=LABEL_MAP[lbl],
                    density=True, edgecolor="white")
        ax.set_title(slabel); ax.set_xlabel(slabel); ax.set_ylabel("Density")
        ax.legend(fontsize=7)

        # 하단: 박스 by sex × label
        ax = axes[1][col_idx]
        groups, positions, colors_box = [], [], []
        tick_labels = []
        for j, (sex, lbl) in enumerate([("M", 0), ("M", 1), ("F", 0), ("F", 1)]):
            sub = stat_df.loc[(stat_df["PatientSex"] == sex) & (stat_df["label"] == lbl), scol]
            groups.append(sub.values); positions.append(j + 1)
            colors_box.append(PALETTE["M"] if sex == "M" else PALETTE["F"])
            tick_labels.append(f"{sex}\n{LABEL_MAP[lbl][:3]}")
        bp = ax.boxplot(groups, positions=positions, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color); patch.set_alpha(0.65)
        ax.set_xticks(positions); ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel(slabel)

    plt.tight_layout()
    savefig("eda_10_aec_per_patient_stats.png")


# ══════════════════════════════════════════════════════════════════════════
# 4.  Excel 통계 저장
# ══════════════════════════════════════════════════════════════════════════

def save_excel(df, aec_cols):
    out_path = os.path.join(OUT_DIR, "eda_statistics.xlsx")
    print(f"  Saving Excel → {out_path}")

    # ── Sheet 1: 임상 변수 기술통계 ───────────────────────────────────
    cont_cols = [c for c in ["PatientAge", "BMI", "SMI", "TAMA"] if c in df.columns]
    rows = []
    for col in cont_cols:
        for grp, mask in [
            ("Overall", df.index),
            ("Male",    df[df["PatientSex"] == "M"].index),
            ("Female",  df[df["PatientSex"] == "F"].index),
            ("Normal",  df[df["label"] == 0].index),
            ("Sarc",    df[df["label"] == 1].index),
        ]:
            s = df.loc[mask, col].dropna()
            d = desc_stats(s)
            rows.append({"Variable": col, "Group": grp, **d})
    sheet1 = pd.DataFrame(rows)

    # ── Sheet 2: Sarcopenia 유병률 ────────────────────────────────────
    rows2 = []
    for grp, sub in [
        ("Overall", df),
        ("Male",    df[df["PatientSex"] == "M"]),
        ("Female",  df[df["PatientSex"] == "F"]),
    ]:
        pos = sub["label"].sum()
        rows2.append({
            "Group": grp, "N": len(sub),
            "Sarcopenia_N": int(pos),
            "Prevalence_%": round(pos / len(sub) * 100, 2),
        })
    # 성별 × 스캐너
    for mfr in df["ManufacturerModelName"].unique():
        sub = df[df["ManufacturerModelName"] == mfr]
        pos = sub["label"].sum()
        rows2.append({
            "Group": f"Scanner: {mfr}", "N": len(sub),
            "Sarcopenia_N": int(pos),
            "Prevalence_%": round(pos / len(sub) * 100, 2),
        })
    sheet2 = pd.DataFrame(rows2)

    # ── Sheet 3: 스캐너 분포 ─────────────────────────────────────────
    mfr_cnt = df["ManufacturerModelName"].value_counts().rename("Count").reset_index()
    mfr_cnt.columns = ["ManufacturerModelName", "Count"]
    mfr_cnt["Ratio_%"] = (mfr_cnt["Count"] / len(df) * 100).round(2)
    for sex in ["M", "F"]:
        mfr_cnt[f"N_{sex}"] = mfr_cnt["ManufacturerModelName"].map(
            df[df["PatientSex"] == sex]["ManufacturerModelName"].value_counts()
        ).fillna(0).astype(int)
    mfr_cnt["Sarcopenia_%"] = mfr_cnt["ManufacturerModelName"].map(
        df.groupby("ManufacturerModelName")["label"].mean() * 100
    ).round(2)
    sheet3 = mfr_cnt

    # ── Sheet 4: AEC per-patient 기술통계 ─────────────────────────────
    vals = df[aec_cols].values.astype(float)
    aec_stats = pd.DataFrame({
        "PatientID":  df["PatientID"].values,
        "PatientSex": df["PatientSex"].values,
        "label":      df["label"].values,
        "mean_mA":    vals.mean(axis=1).round(2),
        "std_mA":     vals.std(axis=1).round(2),
        "min_mA":     vals.min(axis=1).round(2),
        "max_mA":     vals.max(axis=1).round(2),
        "range_mA":   (vals.max(axis=1) - vals.min(axis=1)).round(2),
        "CV":         (vals.std(axis=1) / (vals.mean(axis=1) + 1e-8)).round(4),
    })

    # AEC 통계 요약 (by group)
    aec_sum_rows = []
    for col in ["mean_mA", "std_mA", "range_mA", "CV"]:
        for grp, mask in [
            ("Overall", aec_stats.index),
            ("Male",    aec_stats[aec_stats["PatientSex"] == "M"].index),
            ("Female",  aec_stats[aec_stats["PatientSex"] == "F"].index),
            ("Normal",  aec_stats[aec_stats["label"] == 0].index),
            ("Sarc",    aec_stats[aec_stats["label"] == 1].index),
        ]:
            s = aec_stats.loc[mask, col]
            d = desc_stats(s)
            aec_sum_rows.append({"AEC_stat": col, "Group": grp, **d})
    sheet4 = pd.DataFrame(aec_sum_rows)

    # ── 저장 ─────────────────────────────────────────────────────────
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        sheet1.round(3).to_excel(writer, sheet_name="clinical_stats",      index=False)
        sheet2.to_excel(writer,          sheet_name="sarcopenia_prev",      index=False)
        sheet3.to_excel(writer,          sheet_name="scanner_distribution", index=False)
        sheet4.round(4).to_excel(writer, sheet_name="aec_stats",           index=False)

    print(f"  Excel saved  (4 sheets)  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# 5.  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print("  EDA - Full Dataset Distribution Analysis")
    print(f"{'='*60}\n")

    df, aec_cols = load_data()

    print("\n[Fig 01] Overview ...")
    fig_overview(df)

    print("[Fig 02] Age distribution ...")
    fig_continuous(df, "PatientAge", "Age (years)", "Age (years)", fname="eda_02_age.png")

    print("[Fig 03] BMI distribution ...")
    fig_continuous(df, "BMI", "BMI (kg/m²)", "BMI (kg/m²)", fname="eda_03_bmi.png")

    print("[Fig 04] SMI distribution ...")
    fig_continuous(df, "SMI", "SMI (cm²/m²)", "SMI (cm²/m²)",
                   thresh=True, fname="eda_04_smi.png")

    if "TAMA" in df.columns:
        print("[Fig 05] TAMA distribution ...")
        fig_continuous(df, "TAMA", "TAMA (cm²)", "TAMA (cm²)", fname="eda_05_tama.png")
    else:
        print("[Fig 05] TAMA column not found — skipped")

    print("[Fig 06] Scanner distribution ...")
    fig_scanner(df)

    print("[Fig 07] Correlation matrix ...")
    fig_correlation(df)

    print("[Fig 08] AEC mean curves ...")
    fig_aec_mean_curves(df, aec_cols)

    print("[Fig 09] AEC heatmap ...")
    fig_aec_heatmap(df, aec_cols)

    print("[Fig 10] AEC per-patient stats ...")
    fig_aec_per_patient_stats(df, aec_cols)

    print("\n[Excel] Saving statistics ...")
    save_excel(df, aec_cols)

    print(f"\n{'='*60}")
    print(f"  Done.  Output → {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
