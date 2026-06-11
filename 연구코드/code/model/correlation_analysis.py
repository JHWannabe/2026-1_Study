"""
AEC (128 points) vs Clinical features 상관관계 분석.

AEC를 128포인트 그대로 사용:
  - raw AEC       : 원시 HU 값
  - norm AEC      : 환자별 row-wise z-score (곡선 형태만 보존)

분석 항목:
  1. 임상 변수 간 상관계수 히트맵 (Pearson + Spearman)
  2. 포인트별 AEC-Clinical 상관계수 프로파일 (raw / norm 각각)
     → 각 임상 변수와 AEC[t] 간의 r값을 t=0..127 선으로 표시
  3. (Clinical vars) x (128 AEC points) 상관계수 히트맵
  4. 그룹별 평균 AEC 곡선 (raw + norm, Sarcopenia / Sex)
  5. 포인트별 상관계수 수치 CSV 저장

저장 경로: RESULTS_DIR/correlation/
"""
import sys, os, warnings
from typing import Literal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

from config import DATA_PATH, AEC_SHEET, AEC_LEN, RESULTS_DIR, SMI_THRESH_M, SMI_THRESH_F

OUT_DIR = os.path.join(RESULTS_DIR, "correlation")
os.makedirs(OUT_DIR, exist_ok=True)

# 분석할 임상 변수 (kVp 제외)
CLIN_VARS: dict[str, str] = {
    "Age":        "PatientAge",
    "BMI":        "BMI",
    "SMI":        "SMI",
    "Sex(M=1)":   "sex_enc",
    "Sarcopenia": "sarcopenia",
}
CLIN_COLORS = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4"]


# ── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_full_df() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """metadata + aec_128 inner join.

    Returns
    -------
    df       : 임상 변수 DataFrame
    X_raw    : (N, 128) raw AEC
    X_norm   : (N, 128) row-wise z-score AEC
    aec_cols : aec 컬럼 이름 리스트
    """
    df_meta = pd.read_excel(DATA_PATH, sheet_name="metadata")
    df_meta["PatientID"] = (
        df_meta["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    base_cols = ["PatientID", "PatientAge", "PatientSex", "BMI", "SMI"]
    optional  = [c for c in ["TAMA"] if c in df_meta.columns]
    df_meta   = df_meta[base_cols + optional].dropna().reset_index(drop=True)

    df_aec = pd.read_excel(DATA_PATH, sheet_name=AEC_SHEET)
    df_aec["PatientID"] = (
        df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    aec_cols = sorted(
        [c for c in df_aec.columns if str(c).startswith("aec_")],
        key=lambda x: int(str(x).split("_")[1]),
    )[:AEC_LEN]

    df = pd.merge(
        df_meta, df_aec[["PatientID"] + aec_cols], on="PatientID", how="inner"
    ).reset_index(drop=True)

    def _label(row):
        t = SMI_THRESH_M if row["PatientSex"] == "M" else SMI_THRESH_F
        return 1 if row["SMI"] <= t else 0

    df["sarcopenia"] = df.apply(_label, axis=1)
    df["sex_enc"]    = (df["PatientSex"] == "M").astype(int)

    if "TAMA" in df.columns:
        CLIN_VARS["TAMA"] = "TAMA"
        CLIN_COLORS.append("#42D4F4")

    X_raw: np.ndarray = df[aec_cols].to_numpy(dtype=np.float32)
    mu    = X_raw.mean(axis=1, keepdims=True)
    sd    = X_raw.std(axis=1,  keepdims=True) + 1e-8
    X_norm: np.ndarray = ((X_raw - mu) / sd).astype(np.float32)

    return df, X_raw, X_norm, aec_cols


# ── 포인트별 상관계수 계산 ───────────────────────────────────────────────────

def pointwise_corr(
    X_aec: np.ndarray, df: pd.DataFrame, method: Literal["pearson", "spearman"] = "pearson"
) -> pd.DataFrame:
    """AEC[t] vs 각 임상 변수의 상관계수를 (T, n_vars) DataFrame으로 반환.

    method : "pearson" | "spearman"
    """
    T = X_aec.shape[1]
    result = {}
    for label, col in CLIN_VARS.items():
        y = df[col].to_numpy(dtype=np.float64)
        rs = np.empty(T)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in range(T):
                x = X_aec[:, t].astype(np.float64)
                if np.std(x) == 0:
                    rs[t] = np.nan
                elif method == "pearson":
                    rs[t] = np.corrcoef(x, y)[0, 1]
                else:
                    rs[t] = pd.Series(x).corr(pd.Series(y), method="spearman")
        result[label] = rs
    return pd.DataFrame(result, index=np.arange(T))


# ── Plot 1 : 임상 변수 간 상관계수 히트맵 ────────────────────────────────────

def plot_clinical_corr_heatmap(df: pd.DataFrame):
    sub = df[[v for v in CLIN_VARS.values()]].rename(
        columns={v: k for k, v in CLIN_VARS.items()}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Clinical Variables Correlation Matrix", fontsize=13, fontweight="bold")

    corr_pairs = [
        (sub.corr(method="pearson"),  "Pearson r"),
        (sub.corr(method="spearman"), "Spearman rho"),
    ]
    for ax, (corr, title) in zip(axes, corr_pairs):
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 10}, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "clinical_corr_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [저장] {path}")


# ── Plot 2 : 포인트별 AEC-Clinical 상관계수 프로파일 ─────────────────────────

def plot_pointwise_corr_profile(
    corr_raw: pd.DataFrame, corr_norm: pd.DataFrame
):
    """raw / norm AEC 각각의 포인트별 상관계수 프로파일 라인 차트."""
    n_vars = len(CLIN_VARS)
    fig, axes = plt.subplots(n_vars, 2, figsize=(14, 3 * n_vars), sharex=True)
    fig.suptitle("Per-point AEC Correlation with Clinical Variables", fontsize=13, fontweight="bold")

    t = np.arange(AEC_LEN)
    for row, (label, color) in enumerate(zip(CLIN_VARS.keys(), CLIN_COLORS)):
        for col, (corr_df, aec_label) in enumerate(
            [(corr_raw, "Raw AEC"), (corr_norm, "Norm AEC (row z-score)")]
        ):
            ax = axes[row][col]
            r_vals = corr_df[label].values
            ax.plot(t, r_vals, color=color, lw=1.5)
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.fill_between(t, 0, r_vals, alpha=0.15, color=color)
            ax.set_ylabel(f"r  ({label})", fontsize=9)
            ax.set_ylim(-1, 1)
            ax.set_xlim(0, AEC_LEN - 1)
            if row == 0:
                ax.set_title(aec_label, fontsize=10, fontweight="bold")
            if row == n_vars - 1:
                ax.set_xlabel("AEC point index", fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pointwise_corr_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [저장] {path}")


# ── Plot 3 : (Clinical vars) x (128 points) 상관계수 히트맵 ─────────────────

def plot_corr_heatmap_2d(corr_raw: pd.DataFrame, corr_norm: pd.DataFrame):
    """임상 변수 × AEC 포인트 상관계수를 2D 히트맵으로 표시."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 6))
    fig.suptitle("Clinical x AEC Point Correlation Heatmap", fontsize=13, fontweight="bold")

    for ax, corr_df, title in zip(
        axes,
        [corr_raw, corr_norm],
        ["Raw AEC", "Norm AEC (row z-score)"],
    ):
        data = corr_df[list(CLIN_VARS.keys())].T  # (n_vars, 128)
        sns.heatmap(
            data, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            cbar_kws={"shrink": 0.6, "label": "r"},
            xticklabels=16, yticklabels=True,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("AEC point index", fontsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "corr_heatmap_2d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [저장] {path}")


# ── Plot 4 : 그룹별 평균 AEC 곡선 ───────────────────────────────────────────

def plot_mean_aec_curve(
    df: pd.DataFrame, X_raw: np.ndarray, X_norm: np.ndarray
):
    groups = [
        ("sarcopenia", [(0, "Normal", "#4C72B0"), (1, "Sarcopenia", "#DD8452")], "Sarcopenia"),
        ("PatientSex", [("M", "Male", "#2196F3"), ("F", "Female", "#E91E63")],    "Sex"),
    ]
    t = np.arange(AEC_LEN)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Mean AEC Curve by Group", fontsize=13, fontweight="bold")

    for col, (grp_col, items, grp_title) in enumerate(groups):
        for row, (X_aec, aec_label) in enumerate([(X_raw, "Raw AEC (HU)"), (X_norm, "Norm AEC (z-score)")]):
            ax = axes[row][col]
            for val, label, color in items:
                idx  = (df[grp_col] == val).to_numpy()
                mean = X_aec[idx].mean(axis=0)
                sem  = X_aec[idx].std(axis=0) / np.sqrt(idx.sum())
                ax.plot(t, mean, color=color, lw=2, label=f"{label} (n={idx.sum()})")
                ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.18)
            ax.axhline(0, color="gray", lw=0.6, ls="--")
            ax.set_xlabel("AEC point index", fontsize=9)
            ax.set_ylabel(aec_label, fontsize=9)
            ax.set_title(f"{grp_title} — {aec_label}", fontsize=9)
            ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "mean_aec_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [저장] {path}")


# ── CSV 저장 ──────────────────────────────────────────────────────────────────

def save_corr_csv(corr_raw: pd.DataFrame, corr_norm: pd.DataFrame):
    corr_raw.index.name  = "aec_point"
    corr_norm.index.name = "aec_point"
    corr_raw.to_csv(os.path.join(OUT_DIR,  "pointwise_corr_raw.csv"))
    corr_norm.to_csv(os.path.join(OUT_DIR, "pointwise_corr_norm.csv"))
    print(f"  [저장] pointwise_corr_raw.csv / pointwise_corr_norm.csv")


# ── 콘솔 요약 출력 ────────────────────────────────────────────────────────────

def print_summary(corr_raw: pd.DataFrame, corr_norm: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  Peak |r| per clinical variable  (Raw AEC, Pearson)")
    print("=" * 60)
    print(f"  {'Variable':<14}  {'peak |r|':>8}  {'at point':>8}")
    print("  " + "-" * 38)
    for label in CLIN_VARS:
        r = corr_raw[label].values
        peak_idx = int(np.nanargmax(np.abs(r)))
        print(f"  {label:<14}  {r[peak_idx]:>8.3f}  {peak_idx:>8d}")

    print("\n" + "=" * 60)
    print("  Peak |r| per clinical variable  (Norm AEC, Pearson)")
    print("=" * 60)
    print(f"  {'Variable':<14}  {'peak |r|':>8}  {'at point':>8}")
    print("  " + "-" * 38)
    for label in CLIN_VARS:
        r = corr_norm[label].values
        peak_idx = int(np.nanargmax(np.abs(r)))
        print(f"  {label:<14}  {r[peak_idx]:>8.3f}  {peak_idx:>8d}")
    print()


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n데이터 로드 중: {DATA_PATH}")
    df, X_raw, X_norm, aec_cols = load_full_df()
    print(f"  N = {len(df)}  (Sarcopenia: {df['sarcopenia'].sum()}, Normal: {(df['sarcopenia']==0).sum()})")
    print(f"  AEC shape: {X_raw.shape}  /  Norm shape: {X_norm.shape}")
    print(f"  저장 경로: {OUT_DIR}\n")

    print("[1/5] 임상 변수 간 상관계수 히트맵...")
    plot_clinical_corr_heatmap(df)

    print("[2/5] 포인트별 AEC-Clinical 상관계수 계산 (Pearson)...")
    corr_raw  = pointwise_corr(X_raw,  df, method="pearson")
    corr_norm = pointwise_corr(X_norm, df, method="pearson")

    print("[3/5] 포인트별 상관계수 프로파일 라인 차트...")
    plot_pointwise_corr_profile(corr_raw, corr_norm)

    print("[4/5] Clinical x AEC point 2D 상관계수 히트맵...")
    plot_corr_heatmap_2d(corr_raw, corr_norm)

    print("[5/5] 그룹별 평균 AEC 곡선...")
    plot_mean_aec_curve(df, X_raw, X_norm)

    save_corr_csv(corr_raw, corr_norm)
    print_summary(corr_raw, corr_norm)
    print("분석 완료.")


if __name__ == "__main__":
    main()
