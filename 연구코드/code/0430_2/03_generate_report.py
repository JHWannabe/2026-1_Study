# -*- coding: utf-8 -*-
"""
03_generate_plots.py
────────────────────────────────────────────────────────────────
cross_hospital_summary.xlsx 와 external_validation_results.xlsx를
읽어 아래 두 PNG를 생성한다.

  results/regression/09_cross_hospital_comparison.png
  results/regression/10_external_validation.png

실행: python 03_generate_plots.py
────────────────────────────────────────────────────────────────
"""

import sys, io, os, subprocess, tempfile
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── 경로 ─────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
RESULT_ROOT = SCRIPT_DIR.parent.parent / "results"
REG_DIR     = RESULT_ROOT / "regression"

OUT_CROSS = REG_DIR / "09_cross_hospital_comparison.png"
OUT_EXT   = REG_DIR / "10_external_validation.png"

# ── 한글 폰트 설정 ────────────────────────────────────────────
import matplotlib.font_manager as fm

def _set_korean_font():
    candidates = [
        "Malgun Gothic", "NanumGothic", "NanumBarunGothic",
        "AppleGothic", "Gulim", "Dotum",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False

_set_korean_font()

# ── 스타일 상수 ───────────────────────────────────────────────
COLORS = {
    "gangnam": "#2196F3",   # 파랑
    "sinchon": "#FF5722",   # 주황
    "gangnam_light": "#BBDEFB",
    "sinchon_light": "#FFCCBC",
    "grid":    "#E0E0E0",
    "text":    "#212121",
    "bg":      "#FAFAFA",
}

HOSP_LABELS = {"gangnam": "강남", "sinchon": "신촌"}


# ── Excel 안전 로드 (OneDrive 잠금 우회) ──────────────────────
def _load_excel(path: Path, sheet_name=0) -> pd.DataFrame:
    tmp = Path(tempfile.gettempdir()) / f"plot_tmp_{path.name}"
    subprocess.run(
        ["powershell", "-Command",
         f'Copy-Item -Path "{path}" -Destination "{tmp}" -Force'],
        capture_output=True,
    )
    src = tmp if tmp.exists() else path
    if not src.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(str(src), sheet_name=sheet_name)
    except Exception as e:
        print(f"   [WARN] {path.name} 로드 실패: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────
print("[1] 데이터 로드 중...")

cross_df = _load_excel(REG_DIR / "cross_hospital_summary.xlsx")
ext_df   = _load_excel(REG_DIR / "external_validation_results.xlsx") \
           if (REG_DIR / "external_validation_results.xlsx").exists() \
           else pd.DataFrame()

print(f"    cross_df : {cross_df.shape}")
print(f"    ext_df   : {ext_df.shape}")


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def _f(v, d=4):
    try:    return round(float(v), d)
    except: return np.nan

def _add_value_labels(ax, bars, fmt="{:.4f}", fontsize=8, color="white", offset=0.005):
    """막대 위/안쪽에 수치 레이블 추가"""
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h - offset if h > 0.05 else h + offset,
            fmt.format(h),
            ha='center', va='top' if h > 0.05 else 'bottom',
            fontsize=fontsize, color=color, fontweight='bold'
        )

def _style_ax(ax, title, ylabel, ylim=None, legend=True):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8, color=COLORS['text'])
    ax.set_ylabel(ylabel, fontsize=10, color=COLORS['text'])
    ax.set_facecolor(COLORS['bg'])
    ax.yaxis.grid(True, color=COLORS['grid'], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left','bottom']:
        ax.spines[spine].set_color('#BDBDBD')
    if ylim:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(fontsize=9, framealpha=0.9, edgecolor='#BDBDBD')


# ═════════════════════════════════════════════════════════════
# Figure 1 : cross_hospital_comparison.png
#   - 4-Panel : Linear R² / RMSE / Logistic AUC / Sensitivity+Specificity
#   - x축: Case, 색상: 병원(강남/신촌), 오차막대: std
# ═════════════════════════════════════════════════════════════
def plot_cross_hospital(cross_df: pd.DataFrame, out_path: Path):
    if cross_df.empty:
        print("   [SKIP] cross_df 없음 → cross_hospital_comparison.png 건너뜀")
        return

    # ── 전체 그룹만 필터 ─────────────────────────────────────
    if "Sex" in cross_df.columns:
        mask = cross_df["Sex"].astype(str).str.contains("전체|all", case=False, na=False)
        df = cross_df[mask].copy() if mask.any() else cross_df.copy()
    else:
        df = cross_df.copy()

    # ── Case 정렬 ─────────────────────────────────────────────
    case_col  = "Case"
    hosp_col  = "Hospital"
    hospitals = sorted(df[hosp_col].unique()) if hosp_col in df.columns else []
    cases     = df[case_col].unique().tolist() if case_col in df.columns else []

    # Case 표시명 단축
    def _shorten(c):
        c = str(c)
        for k, v in [("Clinical","C1"),("AEC_prev+Scanner","C4"),
                     ("AEC_new+Scanner","C5"),("AEC_prev","C2"),("AEC_new","C3")]:
            if k in c:
                return v
        return c[:6]

    case_labels = [_shorten(c) for c in cases]
    x = np.arange(len(cases))
    n_h = len(hospitals)
    bar_w = 0.35 if n_h == 2 else 0.5

    fig = plt.figure(figsize=(16, 12), facecolor='white')
    fig.suptitle("교차 병원 비교: 강남 vs 신촌\n(전체 그룹, 5-Fold CV)",
                 fontsize=15, fontweight='bold', y=0.98, color=COLORS['text'])

    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                  left=0.07, right=0.97, top=0.91, bottom=0.08)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    ax_r2, ax_rmse, ax_auc, ax_sens_spec = axes

    hosp_colors = [COLORS["gangnam"], COLORS["sinchon"]]
    hosp_colors_light = [COLORS["gangnam_light"], COLORS["sinchon_light"]]

    def _get(df, hosp, col, default=np.nan):
        sub = df[df[hosp_col] == hosp]
        if sub.empty or col not in sub.columns:
            return [default] * len(cases)
        vals = []
        for c in cases:
            row = sub[sub[case_col] == c]
            vals.append(_f(row[col].values[0]) if not row.empty else default)
        return vals

    # ── Panel 1: Linear R² ───────────────────────────────────
    for i, (hosp, color) in enumerate(zip(hospitals, hosp_colors)):
        vals = _get(df, hosp, "Lin_R2")
        stds = _get(df, hosp, "Lin_R2_std")
        offset = (i - (n_h-1)/2) * bar_w
        bars = ax_r2.bar(
            x + offset, vals, bar_w,
            color=color, alpha=0.85, zorder=3,
            label=HOSP_LABELS.get(str(hosp).lower(), hosp),
            yerr=[s if not np.isnan(s) else 0 for s in stds],
            capsize=4, error_kw=dict(ecolor='#555', elinewidth=1.2, capthick=1.2)
        )
        _add_value_labels(ax_r2, bars, fmt="{:.4f}", fontsize=7.5)

    ax_r2.set_xticks(x); ax_r2.set_xticklabels(case_labels, fontsize=10)
    _style_ax(ax_r2, "선형 회귀 — R² (5-Fold CV)", "R²  (높을수록 우수)", ylim=(0, 1.05))

    # ── Panel 2: Linear RMSE ─────────────────────────────────
    for i, (hosp, color) in enumerate(zip(hospitals, hosp_colors)):
        vals = _get(df, hosp, "Lin_RMSE")
        offset = (i - (n_h-1)/2) * bar_w
        bars = ax_rmse.bar(
            x + offset, vals, bar_w,
            color=color, alpha=0.85, zorder=3,
            label=HOSP_LABELS.get(str(hosp).lower(), hosp)
        )
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax_rmse.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                             f"{h:.2f}", ha='center', va='bottom',
                             fontsize=7.5, color=COLORS['text'], fontweight='bold')

    ax_rmse.set_xticks(x); ax_rmse.set_xticklabels(case_labels, fontsize=10)
    _style_ax(ax_rmse, "선형 회귀 — RMSE  (cm²)", "RMSE  (낮을수록 우수)")

    # ── Panel 3: Logistic AUC ────────────────────────────────
    for i, (hosp, color) in enumerate(zip(hospitals, hosp_colors)):
        vals = _get(df, hosp, "Log_AUC")
        stds = _get(df, hosp, "Log_AUC_std")
        offset = (i - (n_h-1)/2) * bar_w
        bars = ax_auc.bar(
            x + offset, vals, bar_w,
            color=color, alpha=0.85, zorder=3,
            label=HOSP_LABELS.get(str(hosp).lower(), hosp),
            yerr=[s if not np.isnan(s) else 0 for s in stds],
            capsize=4, error_kw=dict(ecolor='#555', elinewidth=1.2, capthick=1.2)
        )
        _add_value_labels(ax_auc, bars, fmt="{:.4f}", fontsize=7.5)

    # AUC 0.5 기준선
    ax_auc.axhline(0.5, color='#9E9E9E', ls='--', lw=1.2, label='AUC=0.5 (기준)')
    ax_auc.set_xticks(x); ax_auc.set_xticklabels(case_labels, fontsize=10)
    _style_ax(ax_auc, "로지스틱 회귀 — AUC (5-Fold CV)", "AUC  (높을수록 우수)", ylim=(0.4, 1.05))

    # ── Panel 4: Sensitivity & Specificity (선 그래프) ───────
    markers = ['o', 's']
    lstyles = ['-', '--']
    for i, (hosp, color) in enumerate(zip(hospitals, hosp_colors)):
        sens = _get(df, hosp, "Log_Sens")
        spec = _get(df, hosp, "Log_Spec")
        label_h = HOSP_LABELS.get(str(hosp).lower(), hosp)
        ax_sens_spec.plot(x, sens, color=color, marker=markers[i],
                          ls=lstyles[0], lw=1.8, ms=7,
                          label=f"{label_h} Sensitivity")
        ax_sens_spec.plot(x, spec, color=color, marker=markers[i],
                          ls=lstyles[1], lw=1.8, ms=7, alpha=0.7,
                          label=f"{label_h} Specificity")

    ax_sens_spec.set_xticks(x); ax_sens_spec.set_xticklabels(case_labels, fontsize=10)
    _style_ax(ax_sens_spec, "민감도 & 특이도  (선: Sens, 점선: Spec)",
              "비율", ylim=(0, 1.08))

    # 공통 x축 레이블 설명
    fig.text(0.5, 0.01,
             "C1=Clinical, C2=+AEC_prev, C3=+AEC_new, C4=+AEC_prev+Scanner, C5=+AEC_new+Scanner",
             ha='center', fontsize=9, color='#616161')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   ✔ 저장: {out_path}")


# ═════════════════════════════════════════════════════════════
# Figure 2 : external_validation.png
#   - 2행 × 3열 레이아웃
#   - 행 0: 강남(학습) → 신촌(검증)
#   - 행 1: 신촌(학습) → 강남(검증)
#   - 열 0: R²  (내부 CV vs 외부 검증)
#   - 열 1: AUC (내부 CV vs 외부 검증)
#   - 열 2: Acc / Sens / Spec (외부 검증)
# ═════════════════════════════════════════════════════════════
def plot_external_validation(ext_df: pd.DataFrame, cross_df: pd.DataFrame, out_path: Path):
    if ext_df.empty:
        print("   [SKIP] ext_df 없음 → external_validation.png 건너뜀")
        return

    case_col = "Case"

    import re

    def _case_num(c):
        """Case 이름에서 번호(1~5) 추출 → 정규화 키."""
        m = re.search(r'[Cc]ase\s*(\d)', str(c))
        return int(m.group(1)) if m else -1

    def _shorten(c):
        n = _case_num(c)
        return f"C{n}" if n > 0 else str(c)[:6]

    def _get_cv(hosp_val):
        """cross_df에서 특정 병원의 전체 CV 결과 반환."""
        if cross_df.empty or "Hospital" not in cross_df.columns:
            return pd.DataFrame()
        sub = cross_df[cross_df["Hospital"] == hosp_val].copy()
        if "Sex" in sub.columns:
            sex_mask = sub["Sex"].astype(str).str.contains("전체|all", case=False, na=False)
            if sex_mask.any():
                sub = sub[sex_mask]
        return sub

    # ── 방향별로 ext_df 분리 ─────────────────────────────────
    train_vals = ext_df["Train"].unique().tolist()
    if len(train_vals) < 2:
        train_vals = train_vals * 2

    # N_train 기준으로 강남/신촌 판별 (강남 > 신촌)
    n_trains = {tv: ext_df[ext_df["Train"] == tv]["N_train"].iloc[0]
                for tv in train_vals}
    sorted_trains = sorted(train_vals, key=lambda tv: -n_trains[tv])
    hosp_display = {sorted_trains[0]: "강남", sorted_trains[1]: "신촌"}
    hosp_color   = {sorted_trains[0]: COLORS["gangnam"],
                    sorted_trains[1]: COLORS["sinchon"]}
    hosp_color_cv= {sorted_trains[0]: COLORS["gangnam_light"],
                    sorted_trains[1]: COLORS["sinchon_light"]}

    directions = [(sorted_trains[0], sorted_trains[1]),
                  (sorted_trains[1], sorted_trains[0])]

    fig = plt.figure(figsize=(21, 14), facecolor='white')
    fig.suptitle("외부 검증: 양방향 교차 검증\n(Internal 5-Fold CV  vs  External Validation)",
                 fontsize=15, fontweight='bold', y=0.99, color=COLORS['text'])

    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.50, wspace=0.30,
                  left=0.06, right=0.97, top=0.91, bottom=0.09)

    cls_cols   = ["Log_Acc", "Log_Sens", "Log_Spec"]
    cls_label  = {"Log_Acc": "Accuracy", "Log_Sens": "Sensitivity", "Log_Spec": "Specificity"}
    colors_cls = ["#4CAF50", "#2196F3", "#FF9800"]

    for row, (train_val, test_val) in enumerate(directions):
        train_lbl  = hosp_display[train_val]
        test_lbl   = hosp_display[test_val]
        t_color    = hosp_color[train_val]
        e_color    = hosp_color[test_val]
        t_color_cv = hosp_color_cv[train_val]

        ext_sub = ext_df[ext_df["Train"] == train_val].copy()
        cv_sub  = _get_cv(train_val)

        # ── Case 번호 기준으로 매칭 테이블 구성 ───────────────
        # ext_sub의 Case 이름을 번호로 인덱싱
        cases_ext = ext_sub[case_col].tolist() if case_col in ext_sub.columns else []
        ext_by_num = {_case_num(c): c for c in cases_ext}  # {1: 'Case1_Clinical', ...}

        # cv_sub의 Case 이름을 번호로 인덱싱
        cases_cv_list = cv_sub[case_col].tolist() if (not cv_sub.empty and case_col in cv_sub.columns) else []
        cv_by_num = {_case_num(c): c for c in cases_cv_list}  # {1: 'Case 1 Clinical', ...}

        # 공통 번호 기준 정렬
        common_nums = sorted(set(ext_by_num.keys()) & set(cv_by_num.keys()))

        x = np.arange(len(cases_ext))
        bw = 0.35
        x_c = np.arange(len(common_nums))

        row_title_prefix = f"[{train_lbl} → {test_lbl}]  "

        # ── Col 0: R² ────────────────────────────────────────
        ax_r2 = fig.add_subplot(gs[row, 0])

        if common_nums:
            r2_cv  = [_f(cv_sub[cv_sub[case_col]==cv_by_num[n]]["Lin_R2"].values[0])  for n in common_nums]
            r2_std = [_f(cv_sub[cv_sub[case_col]==cv_by_num[n]]["Lin_R2_std"].values[0]) if "Lin_R2_std" in cv_sub.columns else 0
                      for n in common_nums]
            r2_ext = [_f(ext_sub[ext_sub[case_col]==ext_by_num[n]]["Lin_R2"].values[0]) for n in common_nums]

            bars1 = ax_r2.bar(x_c - bw/2, r2_cv,  bw,
                              color=t_color_cv, edgecolor=t_color, linewidth=1.2,
                              alpha=0.9, zorder=3,
                              label=f"{train_lbl} (5-Fold CV)",
                              yerr=[s if not np.isnan(s) else 0 for s in r2_std],
                              capsize=4, error_kw=dict(ecolor='#555', elinewidth=1.2, capthick=1.2))
            bars2 = ax_r2.bar(x_c + bw/2, r2_ext, bw,
                              color=e_color, alpha=0.85, zorder=3,
                              label=f"{test_lbl} (외부 검증)")

            _add_value_labels(ax_r2, bars1, fmt="{:.4f}", fontsize=7.5, color=COLORS['text'], offset=-0.005)
            _add_value_labels(ax_r2, bars2, fmt="{:.4f}", fontsize=7.5, color='white')

            ax_r2.set_xticks(x_c)
            ax_r2.set_xticklabels([f"C{n}" for n in common_nums], fontsize=10)
        _style_ax(ax_r2, row_title_prefix + "선형 R²  (CV vs 외부검증)", "R²", ylim=(0, 1.05))

        # ── Col 1: AUC ───────────────────────────────────────
        ax_auc = fig.add_subplot(gs[row, 1])

        if common_nums:
            auc_cv  = [_f(cv_sub[cv_sub[case_col]==cv_by_num[n]]["Log_AUC"].values[0])  for n in common_nums]
            auc_std = [_f(cv_sub[cv_sub[case_col]==cv_by_num[n]]["Log_AUC_std"].values[0]) if "Log_AUC_std" in cv_sub.columns else 0
                       for n in common_nums]
            auc_ext = [_f(ext_sub[ext_sub[case_col]==ext_by_num[n]]["Log_AUC"].values[0]) for n in common_nums]

            bars1 = ax_auc.bar(x_c - bw/2, auc_cv,  bw,
                               color=t_color_cv, edgecolor=t_color, linewidth=1.2,
                               alpha=0.9, zorder=3,
                               label=f"{train_lbl} (5-Fold CV)",
                               yerr=[s if not np.isnan(s) else 0 for s in auc_std],
                               capsize=4, error_kw=dict(ecolor='#555', elinewidth=1.2, capthick=1.2))
            bars2 = ax_auc.bar(x_c + bw/2, auc_ext, bw,
                               color=e_color, alpha=0.85, zorder=3,
                               label=f"{test_lbl} (외부 검증)")

            _add_value_labels(ax_auc, bars1, fmt="{:.4f}", fontsize=7.5, color=COLORS['text'], offset=-0.005)
            _add_value_labels(ax_auc, bars2, fmt="{:.4f}", fontsize=7.5, color='white')

            ax_auc.set_xticks(x_c)
            ax_auc.set_xticklabels([f"C{n}" for n in common_nums], fontsize=10)
            ax_auc.axhline(0.5, color='#9E9E9E', ls='--', lw=1.2, label='AUC=0.5 (기준)')
        _style_ax(ax_auc, row_title_prefix + "로지스틱 AUC  (CV vs 외부검증)", "AUC", ylim=(0.4, 1.05))

        # ── Col 2: Acc / Sens / Spec (외부 검증) ─────────────
        ax_cls = fig.add_subplot(gs[row, 2])

        avail_cls = [c for c in cls_cols if c in ext_sub.columns]
        if avail_cls and cases_ext:
            bw_c = 0.22
            for j, (col, color) in enumerate(zip(avail_cls, colors_cls)):
                vals = [_f(ext_sub[ext_sub[case_col]==c][col].values[0]) for c in cases_ext]
                offset = (j - (len(avail_cls)-1)/2) * bw_c
                bars = ax_cls.bar(x + offset, vals, bw_c, color=color, alpha=0.85,
                                  label=cls_label[col], zorder=3)
                for bar in bars:
                    h = bar.get_height()
                    if not np.isnan(h):
                        ax_cls.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                                    f"{h:.3f}", ha='center', va='bottom',
                                    fontsize=7, color=COLORS['text'])
            ax_cls.set_xticks(x)
            ax_cls.set_xticklabels([_shorten(c) for c in cases_ext], fontsize=10)
        else:
            ax_cls.text(0.5, 0.5, "외부 검증 데이터 없음",
                        ha='center', va='center', fontsize=11, color='#9E9E9E',
                        transform=ax_cls.transAxes)

        _style_ax(ax_cls, row_title_prefix + f"분류 성능  ({test_lbl} 외부검증)", "비율", ylim=(0, 1.12))

    fig.text(0.5, 0.02,
             "C1=Clinical, C2=+AEC_prev, C3=+AEC_new, C4=+AEC_prev+Scanner, C5=+AEC_new+Scanner",
             ha='center', fontsize=9, color='#616161')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   ✔ 저장: {out_path}")


# ─────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────
print("[2] 그래프 생성 중...")
plot_cross_hospital(cross_df, OUT_CROSS)
plot_external_validation(ext_df, cross_df, OUT_EXT)
print("[3] 완료.")