"""
AEC 스케일링별 평균 곡선 시각화 (label × sex 4개 그룹)

입력: RESULTS_DIR/aec_inspection/aec_scaling_compare_aec128.xlsx
출력: 동일 폴더에
  aec_scaling_mean_curves.png  — 전체 128 위치

각 스케일링(raw / std_scaled / norm / global_zscore)에 대해
정상M / 정상F / 근감소증M / 근감소증F 4개 그룹의 mean ± std 곡선을 그린다.
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
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

# 색상: 성별, 선 종류: 정상/근감소증
SEX_COLOR  = {"M": "#1565C0", "F": "#C62828"}   # 남성=파랑, 여성=빨강
LABEL_LS   = {0: "-", 1: "--"}                  # 정상=실선, 근감소증=점선
SEX_NAME   = {"M": "남성", "F": "여성"}
LABEL_NAME = {0: "정상", 1: "근감소증"}

GROUPS = [
    dict(label=0, sex="M", color=SEX_COLOR["M"], ls=LABEL_LS[0], name="정상 남성"),
    dict(label=0, sex="F", color=SEX_COLOR["F"], ls=LABEL_LS[0], name="정상 여성"),
    dict(label=1, sex="M", color=SEX_COLOR["M"], ls=LABEL_LS[1], name="근감소증 남성"),
    dict(label=1, sex="F", color=SEX_COLOR["F"], ls=LABEL_LS[1], name="근감소증 여성"),
]


def _pos_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pos_")]


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
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(1, len(pos_cols))

        # 범례: 색상(성별) + 선 종류(상태) 두 블록
        sex_handles = [
            mlines.Line2D([], [], color=SEX_COLOR[s], lw=2, label=SEX_NAME[s])
            for s in ["M", "F"]
        ]
        ls_handles = [
            mlines.Line2D([], [], color="gray", lw=2, ls=LABEL_LS[lb], label=LABEL_NAME[lb])
            for lb in [0, 1]
        ]
        leg1 = ax.legend(handles=sex_handles, fontsize=8, loc="upper left",
                         title="성별", title_fontsize=8)
        ax.add_artist(leg1)
        ax.legend(handles=ls_handles, fontsize=8, loc="upper right",
                  title="상태", title_fontsize=8)


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


if __name__ == "__main__":
    plot_mean_curves()
