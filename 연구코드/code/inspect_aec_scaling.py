"""
AEC 스케일링 비교 — config.AEC_VARIANTS 4종(raw / std_scaled / norm / global_zscore)을
xlsx 파일의 각 시트에 저장한다.

출력 파일: 연구코드/results/aec_scaling_compare_aec{N}.xlsx
  시트 raw          : 원본 AEC 값 (전처리 없음)
  시트 std_scaled   : StandardScaler 열 방향 표준화 (position별 독립 μ/σ)
  시트 norm         : 행 방향 z-score 정규화 (환자별 독립 μ/σ)
  시트 global_zscore: 전체 행렬 단일 μ/σ로 정규화

첫 3열(PatientID, label, sex)은 공통 식별자.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from model.config import DATA_PATH, RESULTS_DIR, SMI_THRESH_M, SMI_THRESH_F
from model.data import load_data_with_aec


def _row_normalize(X: np.ndarray) -> np.ndarray:
    """행 방향 z-score 정규화."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


def _aec_cols(n: int) -> list[str]:
    return [f"pos_{i+1}" for i in range(n)]


def _load_meta_ids(aec_sheet: str) -> pd.DataFrame:
    """merge 기준 PatientID·label·sex를 반환 (AEC 컬럼 제외)."""
    df_meta = pd.read_excel(DATA_PATH, sheet_name="metadata")
    df_meta["PatientID"] = df_meta["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    df_aec = pd.read_excel(DATA_PATH, sheet_name=aec_sheet)
    df_aec["PatientID"] = df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    base_cols = ["PatientID", "PatientAge", "PatientSex", "BMI", "SMI"]
    df_meta = df_meta[base_cols].dropna().reset_index(drop=True)

    df_merged = pd.merge(df_meta, df_aec[["PatientID"]], on="PatientID", how="inner").reset_index(drop=True)
    df_merged["label"] = df_merged.apply(
        lambda r: 1 if r["SMI"] <= (SMI_THRESH_M if r["PatientSex"] == "M" else SMI_THRESH_F) else 0,
        axis=1,
    )
    return df_merged[["PatientID", "label", "PatientSex"]]


def save_aec_comparison(aec_size: int) -> None:
    aec_sheet = f"aec_{aec_size}"
    print(f"[aec{aec_size}] Loading data ...")

    X_clin, X_aec_raw, y, sex = load_data_with_aec(aec_len=aec_size, aec_sheet=aec_sheet)
    meta_df = _load_meta_ids(aec_sheet)
    assert len(meta_df) == len(X_aec_raw), "PatientID 행 수 불일치"

    col_names = _aec_cols(X_aec_raw.shape[1])

    # ── std_scaled: 열 방향 표준화 (position별 독립 μ/σ) ─────────
    sc = StandardScaler()
    X_aec_std = sc.fit_transform(X_aec_raw).astype(np.float32)

    # ── norm: 행 방향 z-score (환자별 독립 μ/σ) ──────────────────
    X_aec_norm = _row_normalize(X_aec_raw)

    # ── global_zscore: 전체 행렬 단일 μ/σ ────────────────────────
    g_mean = float(X_aec_raw.mean())
    g_std  = max(float(X_aec_raw.std()), 1e-8)
    X_aec_global = ((X_aec_raw - g_mean) / g_std).astype(np.float32)

    # ── xlsx 저장 ──────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(RESULTS_DIR), "aec_inspection")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"aec_scaling_compare_aec{aec_size}.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, X in [
            ("raw",           X_aec_raw),
            ("std_scaled",    X_aec_std),
            ("norm",          X_aec_norm),
            ("global_zscore", X_aec_global),
        ]:
            df_sheet = pd.concat(
                [meta_df.reset_index(drop=True),
                 pd.DataFrame(X, columns=col_names)],
                axis=1,
            )
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  sheet '{sheet_name}' saved  ({df_sheet.shape[0]} rows × {df_sheet.shape[1]} cols)")

    print(f"[aec{aec_size}] Saved → {out_path}\n")

    _plot_scaling_comparison(aec_size, out_dir, X_aec_raw, X_aec_std, X_aec_norm, X_aec_global, np.asarray(y))


def _plot_scaling_comparison(
    aec_size: int,
    out_dir: str,
    X_raw: np.ndarray,
    X_std: np.ndarray,
    X_norm: np.ndarray,
    X_global: np.ndarray,
    y: np.ndarray,
) -> None:
    cases = [
        ("raw",           X_raw),
        ("std_scaled",    X_std),
        ("norm",          X_norm),
        ("global_zscore", X_global),
    ]
    labels_arr = np.asarray(y)
    pos_mask = labels_arr == 1
    neg_mask = labels_arr == 0

    # ── 1) 전체 값 분포 (히스토그램 2x2) ────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"AEC{aec_size} — Value Distribution by Scaling", fontsize=14)
    for ax, (name, X) in zip(axes.flat, cases):
        vals = X.flatten()
        ax.hist(vals, bins=80, color="steelblue", alpha=0.7, density=True)
        ax.set_title(name)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.set_xlim(np.percentile(vals, 0.5), np.percentile(vals, 99.5))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"aec{aec_size}_dist_hist.png"), dpi=150)
    plt.close(fig)
    print(f"  plot saved: aec{aec_size}_dist_hist.png")

    # ── 2) 평균 프로파일 (pos vs neg) 2x2 ───────────────────────
    positions = np.arange(X_raw.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"AEC{aec_size} — Mean Profile (pos vs neg)", fontsize=14)
    for ax, (name, X) in zip(axes.flat, cases):
        mu_pos = X[pos_mask].mean(axis=0) if pos_mask.any() else np.zeros(X.shape[1])
        mu_neg = X[neg_mask].mean(axis=0) if neg_mask.any() else np.zeros(X.shape[1])
        ax.plot(positions, mu_neg, label="neg (0)", color="royalblue", linewidth=1.2)
        ax.plot(positions, mu_pos, label="pos (1)", color="tomato",    linewidth=1.2)
        ax.set_title(name)
        ax.set_xlabel("position index")
        ax.set_ylabel("mean value")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"aec{aec_size}_mean_profile.png"), dpi=150)
    plt.close(fig)
    print(f"  plot saved: aec{aec_size}_mean_profile.png")

    # ── 3) 박스플롯 비교 (케이스 × 분위수 요약) ─────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    data_for_box = [X.flatten() for _, X in cases]
    bp = ax.boxplot(data_for_box, patch_artist=True, notch=False,
                    labels=[name for name, _ in cases])
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(f"AEC{aec_size} — Box Plot by Scaling")
    ax.set_ylabel("value")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"aec{aec_size}_boxplot.png"), dpi=150)
    plt.close(fig)
    print(f"  plot saved: aec{aec_size}_boxplot.png")


if __name__ == "__main__":
    save_aec_comparison(aec_size=128)
