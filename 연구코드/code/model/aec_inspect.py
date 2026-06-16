"""
AEC 스케일링 검사 — 3종 변환(raw / norm / global_zscore)을
xlsx로 저장하고 분포·평균 곡선 시각화.
(std_scaled는 SCALING_CASES에서 주석 처리 — 열 방향 표준화는 실험 제외)

실행:
  python 연구코드/code/aec_inspect.py

출력: RESULTS_DIR/aec_inspection/
  aec_scaling_compare_aec{N}.xlsx  — 3종 스케일링 데이터 (PatientID·label·sex 포함)
  aec{N}_dist_hist.png             — 값 분포 히스토그램 (1×3)
  aec{N}_boxplot.png               — 박스플롯 비교 (3종)
  aec{N}_mean_curves.png           — label × sex 4그룹 평균 ± std 곡선 (1×3)
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_PATH, RESULTS_DIR, SMI_THRESH_M, SMI_THRESH_F
from data import load_data_with_aec


# ── 한글 폰트 ─────────────────────────────────────────────────

def _set_korean_font() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    for font in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_set_korean_font()


# ── 상수 ──────────────────────────────────────────────────────

SCALING_CASES = [
    ("raw",           "Raw AEC"),
    # ("std_scaled",    "Std Scaled (열 방향)"),
    ("norm",          "Norm (행 z-score)"),
    ("global_zscore", "Global Z-score"),
]

GROUPS = [
    dict(label=0, sex="M", color="#1565C0", ls="-",  name="정상 남성"),
    dict(label=0, sex="F", color="#C62828", ls="-",  name="정상 여성"),
    dict(label=1, sex="M", color="#1565C0", ls="--", name="근감소증 남성"),
    dict(label=1, sex="F", color="#C62828", ls="--", name="근감소증 여성"),
]


# ── 데이터 로드·스케일링 ──────────────────────────────────────

def _row_normalize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


def _load_meta_ids(aec_sheet: str) -> pd.DataFrame:
    """merge 기준 PatientID·label·PatientSex 반환."""
    df_meta = pd.read_excel(DATA_PATH, sheet_name="metadata")
    df_meta["PatientID"] = df_meta["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df_aec = pd.read_excel(DATA_PATH, sheet_name=aec_sheet)
    df_aec["PatientID"] = df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    df = pd.merge(
        df_meta[["PatientID", "PatientSex", "SMI"]].dropna(),
        df_aec[["PatientID"]], on="PatientID", how="inner",
    ).reset_index(drop=True)
    df["label"] = df.apply(
        lambda r: 1 if r["SMI"] <= (SMI_THRESH_M if r["PatientSex"] == "M" else SMI_THRESH_F) else 0,
        axis=1,
    )
    return df[["PatientID", "label", "PatientSex"]]


def _build_scaled(X_raw: np.ndarray) -> tuple:
    """SCALING_CASES 순서에 맞춰 스케일링된 배열들을 반환."""
    # X_std = StandardScaler().fit_transform(X_raw).astype(np.float32)
    X_norm = _row_normalize(X_raw)
    g_mean = float(X_raw.mean())
    g_std  = max(float(X_raw.std()), 1e-8)
    X_global = ((X_raw - g_mean) / g_std).astype(np.float32)
    return X_raw.copy(), X_norm, X_global


# ── xlsx 저장 ──────────────────────────────────────────────────

def _save_xlsx(aec_size: int, out_dir: str, meta_df: pd.DataFrame, scaled: tuple) -> None:
    col_names = [f"pos_{i+1}" for i in range(scaled[0].shape[1])]
    out_path = os.path.join(out_dir, f"aec_scaling_compare_aec{aec_size}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for (key, _), X in zip(SCALING_CASES, scaled):
            df = pd.concat(
                [meta_df.reset_index(drop=True), pd.DataFrame(X, columns=col_names)], axis=1,
            )
            df.to_excel(writer, sheet_name=key, index=False)
            print(f"  sheet '{key}'  ({len(df)} rows × {len(df.columns)} cols)")
    print(f"  xlsx → {out_path}")


# ── 시각화 ────────────────────────────────────────────────────

def _plot_dist_hist(aec_size: int, out_dir: str, scaled: tuple) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f"AEC{aec_size} — Value Distribution by Scaling", fontsize=14)
    for ax, ((_, title), X) in zip(axes.flat, zip(SCALING_CASES, scaled)):
        vals = X.flatten()
        ax.hist(vals, bins=80, color="steelblue", alpha=0.7, density=True)
        ax.set_title(title)
        ax.set_xlabel("value"); ax.set_ylabel("density")
        ax.set_xlim(np.percentile(vals, 0.5), np.percentile(vals, 99.5))
    fig.tight_layout()
    path = os.path.join(out_dir, f"aec{aec_size}_dist_hist.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  plot  → {os.path.basename(path)}")


def _plot_boxplot(aec_size: int, out_dir: str, scaled: tuple) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        [X.flatten() for X in scaled],
        patch_artist=True, notch=False,
        tick_labels=[key for key, _ in SCALING_CASES],
    )
    for patch, color in zip(bp["boxes"], ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_title(f"AEC{aec_size} — Box Plot by Scaling"); ax.set_ylabel("value")
    fig.tight_layout()
    path = os.path.join(out_dir, f"aec{aec_size}_boxplot.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  plot  → {os.path.basename(path)}")


def _plot_mean_curves(aec_size: int, out_dir: str, meta_df: pd.DataFrame, scaled: tuple) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(24, 5.75))
    fig.suptitle(
        f"AEC 스케일링 방법별 그룹 평균 곡선 (pubis → liver, {aec_size} 위치)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    for ax, ((_, title), X) in zip(axes, zip(SCALING_CASES, scaled)):
        x = np.arange(1, X.shape[1] + 1)
        for g in GROUPS:
            mask = (meta_df["label"] == g["label"]) & (meta_df["PatientSex"] == g["sex"])
            if not mask.any():
                continue
            subset = X[mask.values]
            mean, std = subset.mean(axis=0), subset.std(axis=0)
            ax.plot(x, mean, color=g["color"], lw=1.8, ls=g["ls"],
                    label=f"{g['name']} (n={mask.sum()})")
            ax.fill_between(x, mean - std, mean + std, color=g["color"], alpha=0.12)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"슬라이스 위치 (pubis=1 → liver={aec_size})", fontsize=9)
        ax.set_ylabel("AEC 값", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(1, X.shape[1])

        sex_handles = [
            mlines.Line2D([], [], color=c, lw=2, label=n)
            for c, n in [("#1565C0", "남성"), ("#C62828", "여성")]
        ]
        label_handles = [
            mlines.Line2D([], [], color="gray", lw=2, ls=ls, label=n)
            for ls, n in [("-", "정상"), ("--", "근감소증")]
        ]
        leg1 = ax.legend(handles=sex_handles,   fontsize=8, loc="upper left",  title="성별",  title_fontsize=8)
        ax.add_artist(leg1)
        ax.legend(          handles=label_handles, fontsize=8, loc="upper right", title="상태",  title_fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, f"aec{aec_size}_mean_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  plot  → {os.path.basename(path)}")


# ── 진입점 ────────────────────────────────────────────────────

def run(aec_size: int = 128) -> None:
    aec_sheet = f"aec_{aec_size}"
    print(f"\n[aec{aec_size}] Loading data ...")
    _, X_aec_raw, _, _ = load_data_with_aec(aec_len=aec_size, aec_sheet=aec_sheet)
    meta_df = _load_meta_ids(aec_sheet)
    assert len(meta_df) == len(X_aec_raw), "PatientID 행 수 불일치"

    scaled = _build_scaled(X_aec_raw)

    out_dir = os.path.join(RESULTS_DIR, "aec_inspection")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[aec{aec_size}] Saving to {out_dir} ...")

    _save_xlsx(aec_size, out_dir, meta_df, scaled)
    _plot_dist_hist(aec_size, out_dir, scaled)
    _plot_boxplot(aec_size, out_dir, scaled)
    _plot_mean_curves(aec_size, out_dir, meta_df, scaled)
    print(f"\n[aec{aec_size}] Done.\n")


if __name__ == "__main__":
    run(aec_size=128)
