"""
AEC 스케일링별 평균 곡선 시각화 (label × sex 4개 그룹)

입력: RESULTS_DIR/aec_inspection/aec_scaling_compare_aec128.xlsx
출력: 동일 폴더에
  aec_scaling_mean_curves.png         — 전체 128 위치
  aec_scaling_mean_curves_crop80.png  — 중앙 80% 구간 (양끝 10% 제거)

각 스케일링(raw / std_scaled / norm / global_zscore)에 대해
정상M / 정상F / 근감소증M / 근감소증F 4개 그룹의 mean ± std 곡선을 그린다.
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model.config import RESULTS_DIR

# ── 한글 폰트 설정 (Windows 기본 맑은 고딕) ──────────────────────
def _set_korean_font():
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_set_korean_font()

# ── 경로 설정 ─────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_THIS_DIR, "..")
XLSX_PATH = os.path.join(RESULTS_DIR, "aec_inspection", "aec_scaling_compare_aec128.xlsx")
OUT_DIR = os.path.dirname(XLSX_PATH)

SHEET_LABELS = [
    ("raw",           "Raw AEC"),
    ("std_scaled",    "Std Scaled (열 방향)"),
    ("norm",          "Norm (행 z-score)"),
    ("global_zscore", "Global Z-score"),
]

# label × sex 4개 그룹 정의
GROUPS = [
    dict(label=0, sex="M", color="#1565C0", ls="-",  name="정상 남성"),
    dict(label=0, sex="F", color="#42A5F5", ls="--", name="정상 여성"),
    dict(label=1, sex="M", color="#B71C1C", ls="-",  name="근감소증 남성"),
    dict(label=1, sex="F", color="#EF9A9A", ls="--", name="근감소증 여성"),
]


def _pos_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pos_")]


def _row_normalize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1,  keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


def _build_crop80_sheets() -> dict[str, pd.DataFrame]:
    """raw 시트에서 crop80 슬라이싱 후 AEC_VARIANTS 4종을 반환."""
    df_raw = pd.read_excel(XLSX_PATH, sheet_name="raw")
    pos_cols = _pos_cols(df_raw)
    P = len(pos_cols)
    s, e = int(0.1 * P), int(0.9 * P)          # 12, 115  →  103 위치
    crop_cols = pos_cols[s:e]
    new_cols  = [f"pos_{i+1}" for i in range(len(crop_cols))]

    meta = df_raw[["PatientID", "label", "PatientSex"]].reset_index(drop=True)
    X_raw = df_raw[crop_cols].values.astype(np.float32)

    X_std = StandardScaler().fit_transform(X_raw).astype(np.float32)
    X_norm = _row_normalize(X_raw)

    g_mean = float(X_raw.mean())
    g_std  = max(float(X_raw.std()), 1e-8)
    X_global = ((X_raw - g_mean) / g_std).astype(np.float32)

    sheets = {}
    for name, X in [
        ("raw",           X_raw),
        ("std_scaled",    X_std),
        ("norm",          X_norm),
        ("global_zscore", X_global),
    ]:
        sheets[name] = pd.concat(
            [meta, pd.DataFrame(X, columns=new_cols)], axis=1
        )
    return sheets


def _draw_curves(axes, sheet_data: dict[str, pd.DataFrame], x_label_suffix: str) -> None:
    for ax, (sheet_key, title) in zip(axes, SHEET_LABELS):
        df = sheet_data[sheet_key]
        pos_cols = _pos_cols(df)
        x = np.arange(1, len(pos_cols) + 1)

        for g in GROUPS:
            mask = (df["label"] == g["label"]) & (df["PatientSex"] == g["sex"])
            subset = df[mask][pos_cols].values.astype(float)
            n = len(subset)
            if n == 0:
                continue
            mean = subset.mean(axis=0)
            std  = subset.std(axis=0)
            ax.plot(x, mean, color=g["color"], lw=1.8, ls=g["ls"],
                    label=f"{g['name']} (n={n})")
            ax.fill_between(x, mean - std, mean + std,
                            color=g["color"], alpha=0.12)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(x_label_suffix, fontsize=9)
        ax.set_ylabel("AEC 값", fontsize=9)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(1, len(pos_cols))


def plot_mean_curves() -> None:
    # ── 전체 128 위치 ─────────────────────────────────────────
    sheet_data = {key: pd.read_excel(XLSX_PATH, sheet_name=key)
                  for key, _ in SHEET_LABELS}

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.75))
    fig.suptitle("AEC 스케일링 방법별 그룹 평균 곡선 (pubis → liver, 128 위치)",
                 fontsize=14, fontweight="bold", y=1.02)
    _draw_curves(axes, sheet_data, "슬라이스 위치 (pubis=1 → liver=128)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "aec_scaling_mean_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료 → {out}")

    # ── crop80 (중앙 80%, 103 위치) ───────────────────────────
    crop80_data = _build_crop80_sheets()
    P_crop = len(_pos_cols(crop80_data["raw"]))   # 103

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.75))
    fig.suptitle(f"AEC 스케일링 방법별 그룹 평균 곡선 — crop80 (pubis → liver, {P_crop} 위치)",
                 fontsize=14, fontweight="bold", y=1.02)
    _draw_curves(axes, crop80_data, f"슬라이스 위치 (pubis=1 → liver={P_crop}, 양끝 10% 제거)")
    fig.tight_layout()
    out_crop = os.path.join(OUT_DIR, "aec_scaling_mean_curves_crop80.png")
    fig.savefig(out_crop, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료 → {out_crop}")


if __name__ == "__main__":
    plot_mean_curves()
