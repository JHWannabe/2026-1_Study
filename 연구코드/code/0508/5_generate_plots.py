# -*- coding: utf-8 -*-
"""
5_generate_plots.py
────────────────────────────────────────────────────────────────
clinic_only / aec_only / aec_feature_only 분석 결과를 비교하는
고수준 요약 그래프를 생성한다.

생성 그래프 (results/0508/figures/):
  01_linear_comparison.png   Linear R² / RMSE 비교 (3 방법 × TAMA/SMI)
  02_logistic_comparison.png Logistic AUC 비교 (3 방법 × TAMA/SMI)
  03_overview.png            전체 지표 한 눈에 보기 (2×3 패널)

실행: python 5_generate_plots.py
────────────────────────────────────────────────────────────────
"""

import sys, io, re
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR  = Path(__file__).parent
RESULT_ROOT = SCRIPT_DIR.parent.parent / "results"
BASE_DIR    = RESULT_ROOT / "0508"
FIG_DIR     = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    ("clinic_only",          "임상 변수"),
    ("aec_only",             "AEC 원시"),
    ("aec_feature_only",     "AEC 피처(전체)"),
    ("aec_feature_selected", "AEC 피처(선택)"),
    ("clinic_aec_feature",   "임상+AEC(선택)"),
]
METHODS_RESNET1D = [
    ("clinic_only",        "임상 변수"),
    ("aec_only",           "AEC 원시"),
    ("aec_feature_only",   "AEC 피처(fold 선택)"),
    ("clinic_aec_feature", "임상+AEC(fold 선택)"),
]
TARGETS = ["TAMA", "SMI"]

COLORS = {
    'clinic_only':          '#4E79A7',
    'aec_only':             '#F28E2B',
    'aec_feature_only':     '#E15759',
    'aec_feature_selected': '#59A14F',
    'clinic_aec_feature':   '#76B7B2',
    'TAMA': '#59A14F',
    'SMI':  '#B07AA1',
}
METHOD_COLORS = [COLORS[k] for k, _ in METHODS]


# ── MD 파싱 (4_generate_report.py 와 동일) ──────────────────
def _parse_linear(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding='utf-8')
    result = {}
    m = re.search(
        r'\|\s*\*\*Mean\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|',
        text)
    if m:
        result['mse_mean'], result['rmse_mean'], result['r2_mean'] = \
            float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = re.search(
        r'\|\s*\*\*Std\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|',
        text)
    if m:
        result['mse_std'], result['rmse_std'], result['r2_std'] = \
            float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = re.search(
        r'Test Set 성능.*?\n\|[^\n]+\n\|---[^\n]+\n\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*',
        text, re.DOTALL)
    if m:
        result['test_mse'], result['test_rmse'], result['test_r2'] = \
            float(m.group(1)), float(m.group(2)), float(m.group(3))
    return result


def _parse_resnet1d(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding='utf-8')
    result = {}
    m = re.search(
        r'\|\s*\*\*Mean\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|'
        r'\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*',
        text)
    if m:
        result['mae_mean'], result['r2_mean'], result['acc_mean'], result['auc_mean'] = \
            float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
    m = re.search(
        r'\|\s*\*\*Std\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*\s*\|'
        r'\s*\*\*([-0-9.]+)\*\*\s*\|\s*\*\*([-0-9.]+)\*\*',
        text)
    if m:
        result['mae_std'], result['r2_std'], result['acc_std'], result['auc_std'] = \
            float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
    m = re.search(r'Test R² = \*\*([-0-9.]+)\*\*', text)
    if m:
        result['test_r2'] = float(m.group(1))
    m = re.search(r'Test MAE = \*\*([-0-9.]+)\*\*', text)
    if m:
        result['test_mae'] = float(m.group(1))
    m = re.search(r'이진화 ACC[^\|]*\|\s*([-0-9.]+)', text)
    if m:
        result['test_acc'] = float(m.group(1))
    m = re.search(r'이진화 AUC[^\|]*\|\s*([-0-9.]+)', text)
    if m:
        result['test_auc'] = float(m.group(1))
    return result


def _parse_logistic(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding='utf-8')
    result = {}
    m = re.search(
        r'\|\s*\*\*Mean\*\*\s*\|\s*\*\*([0-9.]+)\*\*\s*\|\s*\*\*([0-9.]+)\*\*\s*\|',
        text)
    if m:
        result['acc_mean'], result['auc_mean'] = float(m.group(1)), float(m.group(2))
    m = re.search(
        r'\|\s*\*\*Std\*\*\s*\|\s*\*\*([0-9.]+)\*\*\s*\|\s*\*\*([0-9.]+)\*\*\s*\|',
        text)
    if m:
        result['acc_std'], result['auc_std'] = float(m.group(1)), float(m.group(2))
    m = re.search(
        r'Test Set 성능.*?\n\|[^\n]+\n\|---[^\n]+\n\|\s*\*\*([0-9.]+)\*\*\s*\|\s*\*\*([0-9.]+)\*\*',
        text, re.DOTALL)
    if m:
        result['test_acc'], result['test_auc'] = float(m.group(1)), float(m.group(2))
    return result


def collect_metrics() -> dict:
    data = {}
    for method_key, _ in METHODS:
        data[method_key] = {}
        for target in TARGETS:
            data[method_key][target] = {
                'linear':   _parse_linear(BASE_DIR / method_key / target / "Linear_Report.md"),
                'logistic': _parse_logistic(BASE_DIR / method_key / target / "Logistic_Report.md"),
                'resnet1d': _parse_resnet1d(BASE_DIR / method_key / target / "ResNet1D_Report.md"),
            }
    return data


# ── 그래프 공통 스타일 ────────────────────────────────────────
def _style_ax(ax, title, ylabel, ylim=None):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)


def _bar_labels(ax, bars, fmt='{:.4f}', offset_ratio=0.02, fontsize=8):
    ymax = max((b.get_height() for b in bars if not np.isnan(b.get_height())), default=1)
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + ymax * offset_ratio,
                fmt.format(h),
                ha='center', va='bottom', fontsize=fontsize, fontweight='bold')


# ── 01: Linear R² / RMSE 비교 ────────────────────────────────
def plot_linear_comparison(data: dict):
    method_labels = [label for _, label in METHODS]
    x = np.arange(len(METHODS))
    bw = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.suptitle('선형 회귀 성능 비교 — 분석 방법 × 타겟',
                 fontsize=14, fontweight='bold')

    # R² 비교
    ax = axes[0]
    for ti, target in enumerate(TARGETS):
        r2s  = [data[mk][target]['linear'].get('r2_mean',  np.nan) for mk, _ in METHODS]
        stds = [data[mk][target]['linear'].get('r2_std',   0)      for mk, _ in METHODS]
        offset = (ti - 0.5) * bw
        bars = ax.bar(x + offset, r2s, bw,
                      color=COLORS[target], alpha=0.8,
                      label=target,
                      yerr=stds, capsize=4,
                      error_kw=dict(ecolor='#555', elinewidth=1.2))
        _bar_labels(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.legend(fontsize=10)
    _style_ax(ax, 'Linear R² (5-Fold CV Mean±Std)', 'R²  (높을수록 우수)')

    # RMSE 비교
    ax = axes[1]
    for ti, target in enumerate(TARGETS):
        rmses = [data[mk][target]['linear'].get('rmse_mean', np.nan) for mk, _ in METHODS]
        stds  = [data[mk][target]['linear'].get('rmse_std',  0)      for mk, _ in METHODS]
        offset = (ti - 0.5) * bw
        bars = ax.bar(x + offset, rmses, bw,
                      color=COLORS[target], alpha=0.8,
                      label=target,
                      yerr=stds, capsize=4,
                      error_kw=dict(ecolor='#555', elinewidth=1.2))
        _bar_labels(ax, bars, fmt='{:.3f}')

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.legend(fontsize=10)
    _style_ax(ax, 'Linear RMSE (5-Fold CV Mean±Std)', 'RMSE  (낮을수록 우수)')

    plt.tight_layout()
    out = FIG_DIR / '01_linear_comparison.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [저장] {out.name}')


# ── 02: Logistic AUC 비교 ────────────────────────────────────
def plot_logistic_comparison(data: dict):
    method_labels = [label for _, label in METHODS]
    x = np.arange(len(METHODS))
    bw = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.suptitle('로지스틱 회귀 성능 비교 — 분석 방법 × 타겟',
                 fontsize=14, fontweight='bold')

    # AUC 비교
    ax = axes[0]
    for ti, target in enumerate(TARGETS):
        aucs = [data[mk][target]['logistic'].get('auc_mean', np.nan) for mk, _ in METHODS]
        stds = [data[mk][target]['logistic'].get('auc_std',  0)      for mk, _ in METHODS]
        offset = (ti - 0.5) * bw
        bars = ax.bar(x + offset, aucs, bw,
                      color=COLORS[target], alpha=0.8,
                      label=target,
                      yerr=stds, capsize=4,
                      error_kw=dict(ecolor='#555', elinewidth=1.2))
        _bar_labels(ax, bars)

    ax.axhline(0.5, color='gray', ls='--', lw=1, label='무작위 기준 (0.5)')
    ax.axhline(0.7, color='#f39c12', ls=':', lw=1, label='양호 기준 (0.7)')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    _style_ax(ax, 'Logistic AUC-ROC (5-Fold CV)', 'AUC  (높을수록 우수)', ylim=(0.4, 1.05))

    # Accuracy 비교
    ax = axes[1]
    for ti, target in enumerate(TARGETS):
        accs = [data[mk][target]['logistic'].get('acc_mean', np.nan) for mk, _ in METHODS]
        stds = [data[mk][target]['logistic'].get('acc_std',  0)      for mk, _ in METHODS]
        offset = (ti - 0.5) * bw
        bars = ax.bar(x + offset, accs, bw,
                      color=COLORS[target], alpha=0.8,
                      label=target,
                      yerr=stds, capsize=4,
                      error_kw=dict(ecolor='#555', elinewidth=1.2))
        _bar_labels(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.legend(fontsize=10)
    _style_ax(ax, 'Logistic Accuracy (5-Fold CV)', 'Accuracy  (높을수록 우수)', ylim=(0, 1.05))

    plt.tight_layout()
    out = FIG_DIR / '02_logistic_comparison.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [저장] {out.name}')


# ── 03: 전체 개요 (2×3 패널) ─────────────────────────────────
def plot_overview(data: dict):
    method_labels = [label for _, label in METHODS]
    x = np.arange(len(METHODS))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
    fig.suptitle('0508 분석 전체 성능 지표 개요\n(clinic_only / aec_only / aec_feature_only / aec_feature_selected / clinic_aec_feature)',
                 fontsize=13, fontweight='bold')

    metrics = [
        ('linear',   'r2_mean',  'r2_std',   'Linear R²',       None,       '{:.4f}'),
        ('linear',   'rmse_mean','rmse_std',  'Linear RMSE',     None,       '{:.3f}'),
        ('linear',   'test_r2',  None,        'Linear Test R²',  (0, 1.05),  '{:.4f}'),
        ('logistic', 'auc_mean', 'auc_std',   'Logistic AUC',    (0.4, 1.05),'{:.4f}'),
        ('logistic', 'acc_mean', 'acc_std',   'Logistic Acc',    (0, 1.05),  '{:.4f}'),
        ('logistic', 'test_auc', None,        'Logistic Test AUC',(0.4, 1.05),'{:.4f}'),
    ]

    bw = 0.35
    for ax, (reg_type, val_key, std_key, title, ylim, fmt) in zip(axes.flat, metrics):
        for ti, target in enumerate(TARGETS):
            vals = [data[mk][target][reg_type].get(val_key, np.nan) for mk, _ in METHODS]
            stds = ([data[mk][target][reg_type].get(std_key, 0) for mk, _ in METHODS]
                    if std_key else None)
            offset = (ti - 0.5) * bw
            kw = dict(yerr=stds, capsize=3, error_kw=dict(ecolor='#555', elinewidth=1)) if stds else {}
            bars = ax.bar(x + offset, vals, bw,
                          color=COLORS[target], alpha=0.8, label=target, **kw)
            _bar_labels(ax, bars, fmt=fmt, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=8, rotation=10)
        ax.legend(fontsize=8)
        _style_ax(ax, title, '', ylim=ylim)

        if reg_type == 'logistic' and 'auc' in val_key.lower():
            ax.axhline(0.5, color='gray', ls='--', lw=0.8)
            ax.axhline(0.7, color='#f39c12', ls=':', lw=0.8)

    plt.tight_layout()
    out = FIG_DIR / '03_overview.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [저장] {out.name}')


# ── 04: ResNet1D R² / MAE 비교 ───────────────────────────────
def plot_resnet1d_comparison(data: dict):
    method_labels = [label for _, label in METHODS_RESNET1D]
    x = np.arange(len(METHODS_RESNET1D))
    bw = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    fig.suptitle('ResNet1D (1D CNN) 성능 비교 — 분석 방법 × 타겟',
                 fontsize=14, fontweight='bold')

    rn_colors = {
        'clinic_only':        '#4E79A7',
        'aec_only':           '#F28E2B',
        'aec_feature_only':   '#E15759',
        'clinic_aec_feature': '#76B7B2',
    }

    # R² 비교
    ax = axes[0]
    for ti, target in enumerate(TARGETS):
        r2s  = [data[mk][target]['resnet1d'].get('r2_mean',  np.nan) for mk, _ in METHODS_RESNET1D]
        stds = [data[mk][target]['resnet1d'].get('r2_std',   0)      for mk, _ in METHODS_RESNET1D]
        offset = (ti - 0.5) * bw
        bars = ax.bar(x + offset, r2s, bw,
                      color=COLORS[target], alpha=0.8,
                      label=target,
                      yerr=stds, capsize=4,
                      error_kw=dict(ecolor='#555', elinewidth=1.2))
        _bar_labels(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.legend(fontsize=10)
    _style_ax(ax, 'ResNet1D R² (5-Fold CV Mean±Std)', 'R²  (높을수록 우수)')

    # MAE 비교
    ax = axes[1]
    for ti, target in enumerate(TARGETS):
        maes = [data[mk][target]['resnet1d'].get('mae_mean', np.nan) for mk, _ in METHODS_RESNET1D]
        stds = [data[mk][target]['resnet1d'].get('mae_std',  0)      for mk, _ in METHODS_RESNET1D]
        offset = (ti - 0.5) * bw
        bars = ax.bar(x + offset, maes, bw,
                      color=COLORS[target], alpha=0.8,
                      label=target,
                      yerr=stds, capsize=4,
                      error_kw=dict(ecolor='#555', elinewidth=1.2))
        _bar_labels(ax, bars, fmt='{:.3f}')

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.legend(fontsize=10)
    _style_ax(ax, 'ResNet1D MAE (5-Fold CV Mean±Std)', 'MAE  (낮을수록 우수)')

    plt.tight_layout()
    out = FIG_DIR / '04_resnet1d_comparison.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [저장] {out.name}')


# ── 05: ResNet1D AUC / Accuracy 비교 ─────────────────────────
def plot_resnet1d_auc_acc(data: dict):
    method_labels = [label for _, label in METHODS_RESNET1D]
    x = np.arange(len(METHODS_RESNET1D))
    bw = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    fig.suptitle('ResNet1D (1D CNN) AUC / Accuracy 비교 — 분석 방법 × 타겟',
                 fontsize=14, fontweight='bold')

    # AUC 비교 (5-Fold CV)
    ax = axes[0]
    for ti, target in enumerate(TARGETS):
        aucs = [data[mk][target]['resnet1d'].get('auc_mean', np.nan) for mk, _ in METHODS_RESNET1D]
        stds = [data[mk][target]['resnet1d'].get('auc_std',  0)      for mk, _ in METHODS_RESNET1D]
        offset = (ti - 0.5) * bw
        bars = ax.bar(x + offset, aucs, bw,
                      color=COLORS[target], alpha=0.8,
                      label=target,
                      yerr=stds, capsize=4,
                      error_kw=dict(ecolor='#555', elinewidth=1.2))
        _bar_labels(ax, bars)

    ax.axhline(0.5, color='gray', ls='--', lw=1, label='무작위 기준 (0.5)')
    ax.axhline(0.7, color='#f39c12', ls=':', lw=1, label='양호 기준 (0.7)')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    _style_ax(ax, 'ResNet1D AUC (5-Fold CV Mean±Std)', 'AUC  (높을수록 우수)', ylim=(0.4, 1.05))

    # Accuracy 비교 (5-Fold CV)
    ax = axes[1]
    for ti, target in enumerate(TARGETS):
        accs = [data[mk][target]['resnet1d'].get('acc_mean', np.nan) for mk, _ in METHODS_RESNET1D]
        stds = [data[mk][target]['resnet1d'].get('acc_std',  0)      for mk, _ in METHODS_RESNET1D]
        offset = (ti - 0.5) * bw
        bars = ax.bar(x + offset, accs, bw,
                      color=COLORS[target], alpha=0.8,
                      label=target,
                      yerr=stds, capsize=4,
                      error_kw=dict(ecolor='#555', elinewidth=1.2))
        _bar_labels(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.legend(fontsize=10)
    _style_ax(ax, 'ResNet1D Accuracy (5-Fold CV Mean±Std)', 'Accuracy  (높을수록 우수)', ylim=(0, 1.05))

    plt.tight_layout()
    out = FIG_DIR / '05_resnet1d_auc_acc.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [저장] {out.name}')


# ── 메인 ─────────────────────────────────────────────────────
def main():
    print('=' * 55)
    print('  5_generate_plots.py — 비교 그래프 생성')
    print(f'  저장 위치: {FIG_DIR}')
    print('=' * 55)

    print('\n[1] 지표 수집 중...')
    data = collect_metrics()

    print('\n[2] 그래프 생성 중...')
    plot_linear_comparison(data)
    plot_logistic_comparison(data)
    plot_overview(data)
    plot_resnet1d_comparison(data)
    plot_resnet1d_auc_acc(data)

    print(f'\n[완료] 5개 그래프 → {FIG_DIR}')
    print('=' * 55)


if __name__ == '__main__':
    main()
