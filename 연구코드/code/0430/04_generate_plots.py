# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
generate_plots.py  (0430 버전)
────────────────────────────────────────────────────────────────
regression_analysis.py 와 feature_selection_pipeline.py 가 이미
개별 그래프를 생성하므로, 이 스크립트는 결과 Excel을 읽어
고수준 요약·비교 그래프 12개를 추가로 생성한다.

생성 그래프 (results/figures_0430/):
  01_aec_pipeline_cv_r2.png      피처 선택 CV R² — AEC_prev vs AEC_new (3개 데이터셋)
  02_aec_feature_matrix.png      데이터셋별 최종 피처 선택 히트맵
  03_cross_hospital_linear.png   강남 vs 신촌 선형 R²·RMSE (Case 1~5)
  04_cross_hospital_logistic.png 강남 vs 신촌 AUC (Case 1~5)
  05_sex_strat_linear.png        성별 층화 선형 R² (강남 기준)
  06_sex_strat_logistic.png      성별 층화 AUC (강남 기준)
  07_aec_prev_vs_new.png         AEC_prev vs AEC_new 직접 비교 (두 병원)
  08_bmi_contribution_gangnam.png BMI 추가 효과 — 강남
  09_bmi_contribution_sinchon.png BMI 추가 효과 — 신촌
  10_case_progression_gangnam.png Case 1→5 성능 추이 — 강남
  11_case_progression_sinchon.png Case 1→5 성능 추이 — 신촌
  12_sex_n_threshold.png         병원×성별 그룹별 N수 & 임계값

실행: python generate_plots.py
"""

import os
import warnings
import subprocess
import tempfile
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── 한글 폰트 설정 ──────────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

DPI = 300

SCRIPT_DIR  = Path(__file__).parent
RESULT_ROOT = SCRIPT_DIR.parent.parent / "results"
FS_DIR      = RESULT_ROOT / "feature_selection"
REG_DIR     = RESULT_ROOT / "regression"
FIG_DIR     = RESULT_ROOT / "figures_0430"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'case1': '#2ecc71',
    'case2': '#3498db',
    'case3': '#e74c3c',
    'case4': '#1a5276',
    'case5': '#922b21',
    'gangnam': '#4E79A7',
    'sinchon': '#F28E2B',
    'prev':    '#95a5a6',
    'new':     '#e74c3c',
    'all':     '#2ecc71',
    'female':  '#F28E2B',
    'male':    '#4E79A7',
    'gray':    '#BAB0AC',
}
CASE_COLORS = [COLORS['case1'], COLORS['case2'], COLORS['case3'],
               COLORS['case4'], COLORS['case5']]


# ─────────────────────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────────────────────

def savefig(fname: str):
    path = FIG_DIR / fname
    plt.savefig(str(path), dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [저장] {fname}")


def _load_excel(path: Path, sheet_name=0) -> pd.DataFrame:
    """OneDrive 잠금 우회: temp 폴더에 복사 후 읽기."""
    tmp = Path(tempfile.gettempdir()) / f"gp0430_{path.name}"
    subprocess.run(
        ["powershell", "-Command",
         f'Copy-Item -Path "{path}" -Destination "{tmp}" -Force'],
        capture_output=True,
    )
    if not tmp.exists():
        print(f"  [경고] 파일 없음: {path}")
        return pd.DataFrame()
    try:
        return pd.read_excel(str(tmp), sheet_name=sheet_name)
    except Exception as e:
        print(f"  [경고] Excel 읽기 실패 {path.name}: {e}")
        return pd.DataFrame()


def load_reg(hosp_key: str, sex_key: str) -> pd.DataFrame:
    p = REG_DIR / hosp_key / sex_key / "regression_results.xlsx"
    if p.exists():
        return _load_excel(p, sheet_name="summary")
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 01 - 피처 선택 파이프라인 CV R²
# ─────────────────────────────────────────────────────────────

def plot_aec_pipeline_cv_r2():
    fs_path = FS_DIR / "cross_dataset_comparison.xlsx"
    df = _load_excel(fs_path, sheet_name="summary")
    if df.empty:
        print("  [스킵] 01 — cross_dataset_comparison.xlsx 없음")
        return

    datasets   = df['dataset'].tolist()
    prev_r2s   = df['prev_r2'].tolist()
    pipe_r2s   = df['pipeline_r2'].tolist()
    delta_r2s  = df['delta_r2'].tolist()

    x = np.arange(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 3), 5))
    b1 = ax.bar(x - w/2, prev_r2s,  w, label="AEC_prev (수동 4개 피처)",
                color=COLORS['prev'], edgecolor='white')
    b2 = ax.bar(x + w/2, pipe_r2s,  w, label="AEC_new (파이프라인 자동 선택)",
                color=COLORS['new'],  edgecolor='white')

    for bar, v in list(zip(b1, prev_r2s)) + list(zip(b2, pipe_r2s)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for i, delta in enumerate(delta_r2s):
        ax.annotate(f"Δ={delta:+.4f}",
                    xy=(x[i] + w/2, pipe_r2s[i]),
                    xytext=(x[i] + w/2, pipe_r2s[i] + 0.035),
                    ha='center', fontsize=8, color='#555',
                    arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("5-fold CV R²", fontsize=11)
    ax.set_title("피처 선택: AEC_prev vs AEC_new 파이프라인\n(강남 / 신촌 / 병합 데이터셋)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig("01_aec_pipeline_cv_r2.png")


# ─────────────────────────────────────────────────────────────
# 02 - 데이터셋별 최종 피처 선택 히트맵
# ─────────────────────────────────────────────────────────────

def plot_aec_feature_matrix():
    fs_path = FS_DIR / "cross_dataset_comparison.xlsx"
    df = _load_excel(fs_path, sheet_name="feature_matrix")
    if df.empty:
        print("  [스킵] 02 — feature_matrix 시트 없음")
        return

    feat_col  = "feature" if "feature" in df.columns else df.columns[0]
    data_cols = [c for c in df.columns if c != feat_col]

    heat_data = df[data_cols].copy()
    heat_data.index = df[feat_col]
    heat_data = heat_data.astype(float)

    fig, ax = plt.subplots(figsize=(max(6, len(data_cols) * 2.5),
                                    max(5, len(heat_data) * 0.55 + 1.5)))
    sns.heatmap(heat_data, annot=True, fmt='.0f', cmap='Blues',
                linewidths=0.5, cbar=False, ax=ax, vmin=0, vmax=1,
                annot_kws={"size": 12, "fontweight": "bold"})
    ax.set_title("데이터셋별 최종 선택 AEC 피처 (1=선택, 0=미선택)", fontsize=13, fontweight='bold')
    ax.set_xlabel("데이터셋", fontsize=11)
    ax.set_ylabel("AEC Feature", fontsize=11)
    plt.tight_layout()
    savefig("02_aec_feature_matrix.png")


# ─────────────────────────────────────────────────────────────
# 03 - 교차 병원 선형 회귀 R²·RMSE
# ─────────────────────────────────────────────────────────────

def plot_cross_hospital_linear():
    cross_path = REG_DIR / "cross_hospital_summary.xlsx"
    df = _load_excel(cross_path)
    if df.empty:
        print("  [스킵] 03 — cross_hospital_summary.xlsx 없음")
        return

    # 전체(all) 그룹만 필터
    df_all = df[df.apply(lambda r: "전체" in str(r.get("Sex", "")) or
                         r.get("Sex", "") in ("all", "전체"), axis=1)].copy()
    if df_all.empty:
        df_all = df.copy()

    hospitals = df_all['Hospital'].unique() if 'Hospital' in df_all.columns else []
    if len(hospitals) < 1:
        print("  [스킵] 03 — Hospital 컬럼 없음")
        return

    cases = df_all['Case'].unique().tolist()
    x = np.arange(len(cases))
    w = 0.35 / max(len(hospitals), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # R² 비교
    ax = axes[0]
    for i, hosp in enumerate(hospitals):
        sub = df_all[df_all['Hospital'] == hosp]
        r2s = [sub[sub['Case'] == c]['Lin_R2'].values[0]
               if not sub[sub['Case'] == c].empty else 0 for c in cases]
        color = COLORS.get(hosp.lower()[:7], CASE_COLORS[i % len(CASE_COLORS)])
        bars = ax.bar(x + (i - len(hospitals)/2 + 0.5) * w, r2s, w,
                      label=hosp, color=color, edgecolor='white')
        for bar, v in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in cases], fontsize=8)
    ax.set_ylabel("Linear R²", fontsize=11)
    ax.set_title("교차 병원 선형 R² 비교\n(전체 그룹)", fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # RMSE 비교
    ax = axes[1]
    for i, hosp in enumerate(hospitals):
        sub = df_all[df_all['Hospital'] == hosp]
        rmses = [sub[sub['Case'] == c]['Lin_RMSE'].values[0]
                 if not sub[sub['Case'] == c].empty else 0 for c in cases]
        color = COLORS.get(hosp.lower()[:7], CASE_COLORS[i % len(CASE_COLORS)])
        bars = ax.bar(x + (i - len(hospitals)/2 + 0.5) * w, rmses, w,
                      label=hosp, color=color, edgecolor='white')
        for bar, v in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{v:.2f}", ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in cases], fontsize=8)
    ax.set_ylabel("RMSE (cm²)", fontsize=11)
    ax.set_title("교차 병원 선형 RMSE 비교\n(낮을수록 예측 정확)", fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.suptitle("교차 병원 비교 — 선형 회귀 성능 (Case 1~5)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig("03_cross_hospital_linear.png")


# ─────────────────────────────────────────────────────────────
# 04 - 교차 병원 로지스틱 AUC
# ─────────────────────────────────────────────────────────────

def plot_cross_hospital_logistic():
    cross_path = REG_DIR / "cross_hospital_summary.xlsx"
    df = _load_excel(cross_path)
    if df.empty:
        print("  [스킵] 04 — cross_hospital_summary.xlsx 없음")
        return

    df_all = df[df.apply(lambda r: "전체" in str(r.get("Sex", "")) or
                         r.get("Sex", "") in ("all", "전체"), axis=1)].copy()
    if df_all.empty:
        df_all = df.copy()

    hospitals = df_all['Hospital'].unique() if 'Hospital' in df_all.columns else []
    cases = df_all['Case'].unique().tolist()
    x = np.arange(len(cases))
    w = 0.35 / max(len(hospitals), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # AUC 비교
    ax = axes[0]
    for i, hosp in enumerate(hospitals):
        sub = df_all[df_all['Hospital'] == hosp]
        aucs = [sub[sub['Case'] == c]['Log_AUC'].values[0]
                if not sub[sub['Case'] == c].empty else 0 for c in cases]
        color = COLORS.get(hosp.lower()[:7], CASE_COLORS[i % len(CASE_COLORS)])
        bars = ax.bar(x + (i - len(hospitals)/2 + 0.5) * w, aucs, w,
                      label=hosp, color=color, edgecolor='white')
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.axhline(0.5, color='gray', ls='--', lw=1, label='무작위 기준 (0.5)')
    ax.axhline(0.7, color='#f39c12', ls=':', lw=1, label='양호 기준 (0.7)')
    ax.axhline(0.8, color='#27ae60', ls=':', lw=1, label='우수 기준 (0.8)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in cases], fontsize=8)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.set_title("교차 병원 AUC 비교\n(전체 그룹)", fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # Sensitivity / Specificity (전체 평균)
    ax = axes[1]
    metric_data = {}
    for hosp in hospitals:
        sub = df_all[df_all['Hospital'] == hosp]
        sens = [sub[sub['Case'] == c]['Log_Sens'].values[0]
                if not sub[sub['Case'] == c].empty else 0 for c in cases]
        spec = [sub[sub['Case'] == c]['Log_Spec'].values[0]
                if not sub[sub['Case'] == c].empty else 0 for c in cases]
        metric_data[f"{hosp}_Sens"] = sens
        metric_data[f"{hosp}_Spec"] = spec

    palette = ['#3498db', '#e74c3c', '#2ecc71', '#e67e22']
    for idx, (key, vals) in enumerate(metric_data.items()):
        ax.plot(x, vals, 'o-', color=palette[idx % len(palette)],
                lw=1.8, ms=6, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in cases], fontsize=8)
    ax.set_ylabel("Sensitivity / Specificity", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("교차 병원 Sensitivity·Specificity\n(Case 1~5)", fontsize=12)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.suptitle("교차 병원 비교 — 로지스틱 회귀 성능 (Case 1~5)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig("04_cross_hospital_logistic.png")


# ─────────────────────────────────────────────────────────────
# 05 - 성별 층화 선형 R²
# ─────────────────────────────────────────────────────────────

def _collect_sex_data(hosp_key: str):
    """병원별 전체/여/남 summary 수집 → {sex_label: df} dict."""
    result = {}
    for sex_key, sex_label in [('all', '전체'), ('female', '여성(F)'), ('male', '남성(M)')]:
        df = load_reg(hosp_key, sex_key)
        if not df.empty:
            result[sex_label] = df
    return result


def plot_sex_stratification_linear(hosp_key='gangnam', hosp_label='강남'):
    sex_data = _collect_sex_data(hosp_key)
    if not sex_data:
        print(f"  [스킵] 05 — {hosp_key} regression_results 없음")
        return

    # 모든 케이스 수집
    all_cases = []
    for df in sex_data.values():
        for c in df['Case'].tolist():
            if c not in all_cases:
                all_cases.append(c)

    sex_labels = list(sex_data.keys())
    x = np.arange(len(all_cases))
    w = 0.25

    sex_colors = [COLORS['all'], COLORS['female'], COLORS['male']]
    fig, ax = plt.subplots(figsize=(max(12, len(all_cases) * 2.5), 6))

    for i, (sex_label, df) in enumerate(sex_data.items()):
        r2s = []
        errs = []
        for c in all_cases:
            row = df[df['Case'] == c]
            if not row.empty:
                r2s.append(float(row['Lin_R2'].values[0]))
                errs.append(float(row['Lin_R2_std'].values[0]) if 'Lin_R2_std' in row.columns else 0)
            else:
                r2s.append(0); errs.append(0)
        offset = (i - len(sex_labels)/2 + 0.5) * w
        bars = ax.bar(x + offset, r2s, w, yerr=errs, capsize=4,
                      label=sex_label, color=sex_colors[i], edgecolor='white', alpha=0.85)
        for bar, v in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in all_cases], fontsize=8)
    ax.set_ylabel("Linear R² (5-fold CV)", fontsize=11)
    ax.set_title(f"[{hosp_label}] 성별 층화 선형 R² 비교\n(전체 / 여성 / 남성 독립 모델, Case 1~5)",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig("05_sex_strat_linear.png")


# ─────────────────────────────────────────────────────────────
# 06 - 성별 층화 로지스틱 AUC
# ─────────────────────────────────────────────────────────────

def plot_sex_stratification_logistic(hosp_key='gangnam', hosp_label='강남'):
    sex_data = _collect_sex_data(hosp_key)
    if not sex_data:
        print(f"  [스킵] 06 — {hosp_key} regression_results 없음")
        return

    all_cases = []
    for df in sex_data.values():
        for c in df['Case'].tolist():
            if c not in all_cases:
                all_cases.append(c)

    sex_labels = list(sex_data.keys())
    x = np.arange(len(all_cases))
    w = 0.25
    sex_colors = [COLORS['all'], COLORS['female'], COLORS['male']]

    fig, ax = plt.subplots(figsize=(max(12, len(all_cases) * 2.5), 6))

    for i, (sex_label, df) in enumerate(sex_data.items()):
        aucs = []
        errs = []
        for c in all_cases:
            row = df[df['Case'] == c]
            if not row.empty:
                aucs.append(float(row['Log_AUC'].values[0]))
                errs.append(float(row['Log_AUC_std'].values[0]) if 'Log_AUC_std' in row.columns else 0)
            else:
                aucs.append(0); errs.append(0)
        offset = (i - len(sex_labels)/2 + 0.5) * w
        bars = ax.bar(x + offset, aucs, w, yerr=errs, capsize=4,
                      label=sex_label, color=sex_colors[i], edgecolor='white', alpha=0.85)
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=7)

    ax.axhline(0.5, color='gray', ls='--', lw=1, label='무작위 (0.5)')
    ax.axhline(0.7, color='#f39c12', ls=':', lw=1, label='양호 (0.7)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in all_cases], fontsize=8)
    ax.set_ylabel("AUC-ROC (5-fold CV)", fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.set_title(f"[{hosp_label}] 성별 층화 AUC 비교\n(전체 / 여성 / 남성 독립 모델, Case 1~5)",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig("06_sex_strat_logistic.png")


# ─────────────────────────────────────────────────────────────
# 07 - AEC_prev vs AEC_new 직접 비교
# ─────────────────────────────────────────────────────────────

def plot_aec_prev_vs_new():
    hosp_configs = [('gangnam', '강남'), ('sinchon', '신촌')]
    rows = []
    for hosp_key, hosp_label in hosp_configs:
        df = load_reg(hosp_key, 'all')
        if df.empty:
            continue
        for _, row in df.iterrows():
            rows.append({
                'Hospital': hosp_label,
                'Case': row['Case'],
                'Lin_R2': row.get('Lin_R2', 0),
                'Log_AUC': row.get('Log_AUC', 0),
            })

    if not rows:
        print("  [스킵] 07 — 회귀 결과 없음")
        return

    df_all = pd.DataFrame(rows)
    hospitals = df_all['Hospital'].unique().tolist()

    # Case 레이블에서 AEC_prev/AEC_new 키워드 감지
    prev_cases = [c for c in df_all['Case'].unique() if 'AEC_prev' in c and 'Scanner' not in c]
    new_cases  = [c for c in df_all['Case'].unique() if 'AEC_new'  in c and 'Scanner' not in c]

    if not prev_cases or not new_cases:
        print("  [스킵] 07 — AEC_prev/AEC_new 케이스 식별 불가")
        return

    prev_label = prev_cases[0]
    new_label  = new_cases[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(axes, ['Lin_R2', 'Log_AUC'], ['선형 R²', 'AUC']):
        x = np.arange(len(hospitals))
        w = 0.35
        prev_vals = [df_all[(df_all['Hospital'] == h) & (df_all['Case'] == prev_label)][metric].values[0]
                     if not df_all[(df_all['Hospital'] == h) & (df_all['Case'] == prev_label)].empty else 0
                     for h in hospitals]
        new_vals  = [df_all[(df_all['Hospital'] == h) & (df_all['Case'] == new_label)][metric].values[0]
                     if not df_all[(df_all['Hospital'] == h) & (df_all['Case'] == new_label)].empty else 0
                     for h in hospitals]

        b1 = ax.bar(x - w/2, prev_vals, w, label='AEC_prev', color=COLORS['prev'], edgecolor='white')
        b2 = ax.bar(x + w/2, new_vals,  w, label='AEC_new',  color=COLORS['new'],  edgecolor='white')
        for bar, v in list(zip(b1, prev_vals)) + list(zip(b2, new_vals)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(hospitals, fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"AEC_prev vs AEC_new — {title}\n(두 병원 전체 그룹)", fontsize=12)
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    plt.suptitle("AEC_prev(수동 4개) vs AEC_new(파이프라인 자동) 성능 비교",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig("07_aec_prev_vs_new.png")


# ─────────────────────────────────────────────────────────────
# 08-09 - BMI 기여도 분석
# ─────────────────────────────────────────────────────────────

def plot_bmi_contribution(hosp_key: str, hosp_label: str, fig_name: str):
    bmi_path = REG_DIR / hosp_key / "bmi_comparison_summary.xlsx"
    if not bmi_path.exists():
        print(f"  [스킵] {fig_name} — bmi_comparison_summary.xlsx 없음 ({hosp_key})")
        return

    all_df   = _load_excel(bmi_path, sheet_name="all_results")
    delta_df = _load_excel(bmi_path, sheet_name="delta_bmi")

    if all_df.empty:
        print(f"  [스킵] {fig_name} — all_results 시트 없음")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) R² 비교 — no BMI vs +BMI
    ax = axes[0]
    labels_a = all_df['Label'].tolist() if 'Label' in all_df.columns else all_df.index.tolist()
    r2_vals  = all_df['Lin_R2'].tolist()
    bmi_flags = ['BMI' in str(l) for l in labels_a]
    colors_a = [COLORS['new'] if f else COLORS['prev'] for f in bmi_flags]
    bars = ax.bar(range(len(labels_a)), r2_vals, color=colors_a, edgecolor='white', width=0.5)
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.4f}", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(labels_a)))
    ax.set_xticklabels([str(l) for l in labels_a], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Linear R² (CV)", fontsize=11)
    ax.set_title("no BMI vs +BMI — 선형 R²", fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(handles=[mpatches.Patch(color=COLORS['new'], label='+BMI'),
                       mpatches.Patch(color=COLORS['prev'], label='no BMI')], fontsize=9)

    # (b) Delta 효과
    if not delta_df.empty:
        ax = axes[1]
        cases_d = delta_df['Case_base'].tolist() if 'Case_base' in delta_df.columns else []
        dr2  = [float(v) for v in delta_df['Delta_Lin_R2'].tolist()]
        dauc = [float(v) for v in delta_df['Delta_Log_AUC'].tolist()]
        x_d  = np.arange(len(cases_d))
        w_d  = 0.35
        ax.bar(x_d - w_d/2, dr2,  w_d, label='ΔR²',  color='#3498db', edgecolor='white')
        ax.bar(x_d + w_d/2, dauc, w_d, label='ΔAUC', color='#e74c3c', edgecolor='white')
        ax.axhline(0, color='black', lw=0.8)
        for i, (r, a) in enumerate(zip(dr2, dauc)):
            ax.text(i - w_d/2, r + (0.003 if r >= 0 else -0.008), f"{r:+.4f}",
                    ha='center', va='bottom', fontsize=9)
            ax.text(i + w_d/2, a + (0.003 if a >= 0 else -0.008), f"{a:+.4f}",
                    ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x_d)
        ax.set_xticklabels(cases_d, fontsize=10)
        ax.set_ylabel("Δ = +BMI − no BMI", fontsize=11)
        ax.set_title("BMI 추가 효과 (Δ)", fontsize=12)
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    plt.suptitle(f"[{hosp_label}] BMI 기여도 분석 — no BMI vs +BMI (Case 1/2/4)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig(fig_name)


# ─────────────────────────────────────────────────────────────
# 10-11 - Case 1→5 성능 추이
# ─────────────────────────────────────────────────────────────

def plot_case_progression(hosp_key: str, hosp_label: str, fig_name: str):
    df = load_reg(hosp_key, 'all')
    if df.empty:
        print(f"  [스킵] {fig_name} — {hosp_key}/all regression_results 없음")
        return

    cases = df['Case'].tolist()
    x = np.arange(len(cases))

    metrics = [
        ('Lin_R2',    'Linear R² 추이',     '#2196F3'),
        ('Lin_RMSE',  'Linear RMSE 추이',   '#FF9800'),
        ('Log_AUC',   'Logistic AUC 추이',  '#F44336'),
        ('Log_Sens',  'Sensitivity 추이',   '#4CAF50'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, title, color) in zip(axes.flat, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].tolist()
        ax.plot(x, vals, 'o-', color=color, lw=2.5, ms=10,
                markerfacecolor='white', markeredgewidth=2.5)
        y_span = max(vals) - min(vals) if max(vals) != min(vals) else 0.05
        for xi, v in zip(x, vals):
            ax.text(xi, v + y_span * 0.1, f"{v:.3f}", ha='center', fontsize=10,
                    fontweight='bold', color=color)
        for i in range(1, len(vals)):
            delta = vals[i] - vals[i-1]
            ax.annotate(f"Δ={delta:+.3f}",
                        xy=((x[i] + x[i-1]) / 2, (vals[i] + vals[i-1]) / 2),
                        ha='center', fontsize=8, color='gray')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(' ', '\n') for c in cases], fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(min(vals) - y_span * 0.3, max(vals) + y_span * 0.5)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    plt.suptitle(f"[{hosp_label}] Case 1→5 예측 성능 지표 추이\n"
                 f"(임상 기준선 → AEC_prev/new → 스캐너 추가에 따른 성능 정량화)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig(fig_name)


# ─────────────────────────────────────────────────────────────
# 12 - 병원×성별 그룹별 N수 & 임계값
# ─────────────────────────────────────────────────────────────

def plot_sex_n_threshold():
    rows = []
    for hosp_key, hosp_label in [('gangnam', '강남'), ('sinchon', '신촌')]:
        df = load_reg(hosp_key, 'all')
        if df.empty:
            continue
        n_rows = df['N_rows'].iloc[0] if 'N_rows' in df.columns else 0
        thr_str = str(df['TAMA_threshold'].iloc[0]) if 'TAMA_threshold' in df.columns else "F:0/M:0"
        try:
            parts = dict(p.split(':') for p in thr_str.split('/'))
            thr_f = float(parts.get('F', 0))
            thr_m = float(parts.get('M', 0))
        except Exception:
            thr_f = thr_m = 0.0
        rows.append({'병원': hosp_label, 'N': n_rows, 'F_thr': thr_f, 'M_thr': thr_m})

    if not rows:
        print("  [스킵] 12 — 회귀 결과 없음")
        return

    df = pd.DataFrame(rows)
    hosp_labels = df['병원'].tolist()
    n_vals = df['N'].tolist()
    x = np.arange(len(hosp_labels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # N수
    ax = axes[0]
    colors_n = [COLORS['gangnam'] if '강남' in l else COLORS['sinchon'] for l in hosp_labels]
    bars = ax.bar(x, n_vals, color=colors_n, edgecolor='white', width=0.5)
    for bar, v in zip(bars, n_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(int(v)), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(hosp_labels, fontsize=11)
    ax.set_ylabel("N (환자 수)", fontsize=11)
    ax.set_title("병원별 분석 대상 환자 수", fontsize=12)
    ax.legend(handles=[mpatches.Patch(color=COLORS['gangnam'], label='강남'),
                       mpatches.Patch(color=COLORS['sinchon'], label='신촌')], fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # 성별 특이적 임계값
    ax = axes[1]
    f_vals = df['F_thr'].tolist()
    m_vals = df['M_thr'].tolist()
    bars_f = ax.bar(x - w/2, f_vals, w, color='#e74c3c', edgecolor='white', label='여성(F)')
    bars_m = ax.bar(x + w/2, m_vals, w, color='#3498db', edgecolor='white', label='남성(M)')
    for bar, v in zip(bars_f, f_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e74c3c')
    for bar, v in zip(bars_m, m_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(hosp_labels, fontsize=11)
    ax.set_ylabel("이진화 임계값 (cm²)", fontsize=11)
    ax.set_title("병원별 성별 특이적 Low-TAMA 임계값\n(성별 내 하위 25%)", fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.suptitle("병원별 분석 요약 — N수 & 성별 특이적 이진화 임계값",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig("12_sex_n_threshold.png")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  generate_plots.py (0430) — 고수준 요약 그래프 생성")
    print(f"  저장 위치: {FIG_DIR}")
    print("=" * 60)

    print("\n[피처 선택 그래프]")
    plot_aec_pipeline_cv_r2()                                      # 01
    plot_aec_feature_matrix()                                       # 02

    print("\n[교차 병원 비교 그래프]")
    plot_cross_hospital_linear()                                    # 03
    plot_cross_hospital_logistic()                                  # 04

    print("\n[성별 층화 분석 그래프 (강남)]")
    plot_sex_stratification_linear(hosp_key='gangnam', hosp_label='강남')   # 05
    plot_sex_stratification_logistic(hosp_key='gangnam', hosp_label='강남') # 06

    print("\n[AEC_prev vs AEC_new 비교]")
    plot_aec_prev_vs_new()                                         # 07

    print("\n[BMI 기여도 분석]")
    plot_bmi_contribution('gangnam', '강남', '08_bmi_contribution_gangnam.png')  # 08
    plot_bmi_contribution('sinchon', '신촌', '09_bmi_contribution_sinchon.png')  # 09

    print("\n[Case 1→5 성능 추이]")
    plot_case_progression('gangnam', '강남', '10_case_progression_gangnam.png')  # 10
    plot_case_progression('sinchon', '신촌', '11_case_progression_sinchon.png')  # 11

    print("\n[N수 & 임계값 요약]")
    plot_sex_n_threshold()                                         # 12

    print(f"\n[완료] 12개 그래프 저장 → {FIG_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
