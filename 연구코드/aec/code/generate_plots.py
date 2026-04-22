# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
generate_plots.py - 연구 결과 시각화 전체 생성 (15개 그래프)

생성 그래프:
  01_feature_correlation.png     AEC feature-TAMA 상관계수 Top 20
  02_vif_comparison.png          선택 AEC feature VIF
  03_tama_distribution.png       성별 TAMA 분포 + 임계값
  04_linear_actual_vs_pred.png   선형: 실제 vs 예측 산점도
  05_linear_residuals.png        선형: 잔차 진단 4-panel
  06_linear_forest.png           선형: 계수 forest plot (유의 변수)
  07_linear_univariate_r2.png    선형: 단변량 R² 비교
  08_logistic_roc.png            로지스틱: ROC 곡선
  09_logistic_calibration.png    로지스틱: Calibration plot
  10_logistic_confusion.png      로지스틱: Confusion matrix
  11_logistic_forest.png         로지스틱: Crude OR forest plot (단변량)
  12_case_metrics_bar.png        Case 1-3: 선형 성능 비교 (R², RMSE)
  13_case_auc_bar.png            Case 1-3: 로지스틱 AUC 비교
  14_case_aic_bar.png            Case 1-3: AIC/BIC 비교
  15_case_progression.png        Case 1-3: 다중 지표 추이

실행: python generate_plots.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import scipy.stats as spstats
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import statsmodels.api as sm

import config as config
import data_loader as data_loader
from logistic_regression import (bootstrap_auc_ci, hosmer_lemeshow_test,
                                  nagelkerke_r2, optimal_threshold_metrics)
from feature_selection import (compute_correlations, AEC_FEATURE_COLS,
                               scanner_distribution, kvp_distribution)

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.dpi'] = 150

DPI      = 300
FIG_DIR  = os.path.join(config.RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    'case1': '#4E79A7',
    'case2': '#F28E2B',
    'case3': '#E15759',
    'pos':   '#E15759',
    'neg':   '#4E79A7',
    'male':  '#4E79A7',
    'female':'#F28E2B',
    'gray':  '#BAB0AC',
}


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def savefig(fname: str):
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [저장] {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 01 - AEC Feature 상관계수 Top 20
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_correlation(df_raw):
    corr_df = compute_correlations(df_raw)
    top20   = corr_df.head(20).copy()
    top20['r_float'] = top20['Pearson_r'].apply(lambda x: float(x))

    colors = [COLORS['pos'] if v > 0 else COLORS['neg'] for v in top20['r_float']]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top20['Feature'][::-1], top20['r_float'][::-1],
                   color=colors[::-1], edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Pearson r with TAMA", fontsize=11)
    ax.set_title("AEC Feature - TAMA 상관계수 Top 20\n(붉은색=양, 파란색=음)", fontsize=12)
    ax.set_xlim(-0.45, 0.45)
    ax.grid(axis='x', alpha=0.3)

    pos_patch = mpatches.Patch(color=COLORS['pos'], label='양의 상관 (+)')
    neg_patch = mpatches.Patch(color=COLORS['neg'], label='음의 상관 (-)')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc='lower right')
    fig.tight_layout()
    savefig("01_feature_correlation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 02 - VIF 비교
# ─────────────────────────────────────────────────────────────────────────────

def plot_vif(df_raw):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    df_enc = data_loader.encode_sex(df_raw.copy())
    vif_cols = ['PatientAge', 'Sex'] + config.SELECTED_AEC_FEATURES
    available = [c for c in vif_cols if c in df_enc.columns]
    df_v = df_enc[available].dropna()
    X = np.column_stack([np.ones(len(df_v)), df_v.values])
    vif_vals, feat_names = [], []
    for i, name in enumerate(available):
        vif_vals.append(variance_inflation_factor(X, i + 1))
        feat_names.append(name)

    colors = ['#2ecc71' if v < 5 else ('#f39c12' if v < 10 else '#e74c3c')
              for v in vif_vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(feat_names[::-1], vif_vals[::-1], color=colors[::-1],
                   edgecolor='white')
    ax.axvline(5,  color='orange', linestyle='--', linewidth=1.2, label='VIF=5 (주의)')
    ax.axvline(10, color='red',    linestyle='--', linewidth=1.2, label='VIF=10 (위험)')
    ax.set_xlabel("VIF (분산팽창인수)", fontsize=11)
    ax.set_title("선택 변수의 VIF (다중공선성 검사)\nVIF<5: 낮음, 5-10: 중간, >10: 높음", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, vif_vals[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va='center', fontsize=9)
    fig.tight_layout()
    savefig("02_vif_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 03 - TAMA 성별 분포 + 임계값
# ─────────────────────────────────────────────────────────────────────────────

def plot_tama_distribution(df_raw):
    male   = df_raw[df_raw['PatientSex'] == 'M']['TAMA']
    female = df_raw[df_raw['PatientSex'] == 'F']['TAMA']

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    for ax, data, sex_label, thresh, color in [
        (axes[0], male,   '남성 (M)', config.TAMA_THRESHOLD_MALE,   COLORS['male']),
        (axes[1], female, '여성 (F)', config.TAMA_THRESHOLD_FEMALE, COLORS['female']),
    ]:
        ax.hist(data, bins=35, color=color, alpha=0.75, edgecolor='white')
        ax.axvline(thresh, color='red', linestyle='--', linewidth=2,
                   label=f'임계값: {thresh} cm²')
        ax.axvline(data.mean(), color='black', linestyle='-', linewidth=1.5,
                   label=f'평균: {data.mean():.1f} cm²')
        ax.set_title(f"{sex_label} TAMA 분포\n(N={len(data)}, 중앙값={data.median():.1f})", fontsize=11)
        ax.set_xlabel("TAMA (cm²)", fontsize=10)
        ax.set_ylabel("환자 수", fontsize=10)
        n_pos = (data < thresh).sum()
        ax.legend(fontsize=9, title=f"임계값 이하: {n_pos}명 ({n_pos/len(data)*100:.1f}%)")
        ax.grid(alpha=0.3)

    fig.suptitle("성별 TAMA 분포 및 이진화 임계값", fontsize=13, fontweight='bold')
    fig.tight_layout()
    savefig("03_tama_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 04 - 선형 회귀: 실제 vs 예측
# ─────────────────────────────────────────────────────────────────────────────

def plot_linear_actual_vs_pred(df, res, feat_cols):
    X_c    = sm.add_constant(df[feat_cols].values.astype(float), has_constant='add')
    y      = df['TAMA'].values
    y_pred = res.predict(X_c)
    r2     = res.rsquared
    rmse   = np.sqrt(np.mean((y - y_pred) ** 2))

    lims = [min(y.min(), y_pred.min()) - 5, max(y.max(), y_pred.max()) + 5]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y, y_pred, alpha=0.3, s=20, color=COLORS['case3'], label='환자')
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='완벽 예측 (y=x)')
    z = np.polyfit(y, y_pred, 1)
    p = np.poly1d(z)
    xline = np.linspace(lims[0], lims[1], 200)
    ax.plot(xline, p(xline), color=COLORS['case1'], linewidth=2, label='회귀 직선')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("실제 TAMA (cm²)", fontsize=11)
    ax.set_ylabel("예측 TAMA (cm²)", fontsize=11)
    ax.set_title(f"선형 회귀: 실제 vs 예측 TAMA\nR²={r2:.3f}, RMSE={rmse:.2f} cm²", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig("04_linear_actual_vs_pred.png")


# ─────────────────────────────────────────────────────────────────────────────
# 05 - 선형 회귀: 잔차 진단 4-panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_linear_residuals(df, res, feat_cols):
    X_c    = sm.add_constant(df[feat_cols].values.astype(float), has_constant='add')
    y_pred = res.predict(X_c)
    resid  = np.asarray(res.resid)
    std_resid = resid / resid.std()

    fig = plt.figure(figsize=(12, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

    # (a) Residuals vs Fitted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, resid, alpha=0.3, s=15, color=COLORS['case3'])
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
    lowess_fit = sm.nonparametric.lowess(resid, y_pred, frac=0.2)
    ax1.plot(lowess_fit[:, 0], lowess_fit[:, 1], color='blue', linewidth=1.5)
    ax1.set_xlabel("Fitted Values", fontsize=10); ax1.set_ylabel("Residuals", fontsize=10)
    ax1.set_title("(a) Residuals vs Fitted", fontsize=11); ax1.grid(alpha=0.3)

    # (b) Q-Q Plot
    ax2 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, r) = spstats.probplot(resid, dist='norm')
    ax2.scatter(osm, osr, alpha=0.4, s=15, color=COLORS['case3'])
    qq_line = np.array([osm[0], osm[-1]])
    ax2.plot(qq_line, slope * qq_line + intercept, 'r--', linewidth=2, label=f'r={r:.3f}')
    ax2.set_xlabel("이론적 분위수 (Normal Quantiles)", fontsize=10)
    ax2.set_ylabel("실제 분위수 (Sample Quantiles)", fontsize=10)
    ax2.set_title("(b) Normal Q-Q Plot", fontsize=11); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # (c) Scale-Location
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_pred, np.sqrt(np.abs(std_resid)), alpha=0.3, s=15, color=COLORS['case2'])
    lowess2 = sm.nonparametric.lowess(np.sqrt(np.abs(std_resid)), y_pred, frac=0.2)
    ax3.plot(lowess2[:, 0], lowess2[:, 1], color='blue', linewidth=1.5)
    ax3.set_xlabel("Fitted Values", fontsize=10); ax3.set_ylabel("|Standardized Residuals|^0.5", fontsize=9)
    ax3.set_title("(c) Scale-Location (등분산성 확인)", fontsize=11); ax3.grid(alpha=0.3)

    # (d) Residuals Histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(resid, bins=40, color=COLORS['case1'], alpha=0.75, edgecolor='white', density=True)
    x_range = np.linspace(resid.min(), resid.max(), 200)
    norm_pdf = spstats.norm.pdf(x_range, resid.mean(), resid.std())
    ax4.plot(x_range, norm_pdf, 'r-', linewidth=2, label='정규분포 곡선')
    sw_stat, sw_p = spstats.shapiro(resid[:500] if len(resid) > 500 else resid)
    ax4.set_xlabel("Residuals", fontsize=10); ax4.set_ylabel("밀도", fontsize=10)
    ax4.set_title(f"(d) 잔차 분포\nShapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4e}", fontsize=11)
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    fig.suptitle("선형 회귀 잔차 진단 (4-Panel Diagnostic)", fontsize=13, fontweight='bold')
    savefig("05_linear_residuals.png")


# ─────────────────────────────────────────────────────────────────────────────
# 06 - 선형 회귀: 계수 Forest Plot (유의한 변수)
# ─────────────────────────────────────────────────────────────────────────────

def plot_linear_forest(coef_df: pd.DataFrame):
    df_sig = coef_df[(coef_df['Variable'] != 'Intercept') &
                     (coef_df['p_value'] < 0.05)].copy()
    if df_sig.empty:
        print("  [스킵] 유의한 계수 없음 - forest plot 생략")
        return

    df_sig = df_sig.sort_values('β')
    betas    = df_sig['β'].values
    ci_lower = df_sig['CI_Lower'].values
    ci_upper = df_sig['CI_Upper'].values
    labels   = df_sig['Variable'].values

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_sig) * 0.5)))
    y_pos = np.arange(len(df_sig))
    colors = [COLORS['pos'] if b > 0 else COLORS['neg'] for b in betas]

    ax.barh(y_pos, betas, xerr=[betas - ci_lower, ci_upper - betas],
            color=colors, alpha=0.8, error_kw={'ecolor': 'black', 'capsize': 4})
    ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([l.replace('_z', ' (표준화)') for l in labels], fontsize=9)
    ax.set_xlabel("β 계수 (95% CI, 표준화 단위)", fontsize=11)
    ax.set_title("선형 회귀 - 유의한 계수 Forest Plot\n(p<0.05, TAMA cm² 변화량 per 1 SD)", fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    for i, (b, p) in enumerate(zip(betas, df_sig['p_value'].values)):
        stars = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
        ax.text(ci_upper[i] + 0.2, i, stars, va='center', fontsize=9, color='black')

    fig.tight_layout()
    savefig("06_linear_forest.png")


# ─────────────────────────────────────────────────────────────────────────────
# 07 - 선형 회귀: 단변량 R² 비교
# ─────────────────────────────────────────────────────────────────────────────

def plot_linear_univariate_r2(uni_df: pd.DataFrame):
    df_plot = uni_df[uni_df['β'] != 'N/A (범주형)'].copy()
    df_plot['R2_float'] = df_plot['R2'].astype(float)
    df_plot['p_float']  = df_plot['p_value'].astype(float)
    df_plot = df_plot.sort_values('R2_float', ascending=True)

    colors = ['#e74c3c' if p < 0.05 else COLORS['gray']
              for p in df_plot['p_float']]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df_plot['Variable'], df_plot['R2_float'],
                   color=colors, edgecolor='white')
    ax.set_xlabel("단변량 R² (TAMA 설명 분산 비율)", fontsize=11)
    ax.set_title("선형 회귀 - 단변량 R² 비교\n(빨간색: p<0.05 유의)", fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, df_plot['R2_float']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', fontsize=9)
    sig_patch = mpatches.Patch(color='#e74c3c', label='p<0.05 유의')
    ns_patch  = mpatches.Patch(color=COLORS['gray'],  label='p≥0.05 비유의')
    ax.legend(handles=[sig_patch, ns_patch], fontsize=9)
    fig.tight_layout()
    savefig("07_linear_univariate_r2.png")


# ─────────────────────────────────────────────────────────────────────────────
# 08 - 로지스틱: ROC 곡선
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve_fig(df, log_feat_cols):
    y = df['TAMA_binary'].values
    X_c = sm.add_constant(df[log_feat_cols].values.astype(float), has_constant='add')
    res = sm.Logit(y, X_c).fit(disp=False, method='bfgs', maxiter=1000)
    y_prob = res.predict(X_c)

    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(y, y_prob)
    fpr, tpr, _ = roc_curve(y, y_prob)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(fpr, tpr, color=COLORS['case3'], linewidth=2.5,
            label=f'AUC = {auc_mean:.3f} (95%CI: {auc_lo:.3f}-{auc_hi:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='무작위 (AUC=0.5)')
    ax.fill_between(fpr, tpr, alpha=0.08, color=COLORS['case3'])
    ax.set_xlabel("1 - Specificity (FPR)", fontsize=12)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax.set_title("로지스틱 회귀 - ROC 곡선 (전체 모델)\n"
                 f"Bootstrap 95%CI (n={config.N_BOOTSTRAP})", fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig("08_logistic_roc.png")

    return y, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# 09 - 로지스틱: Calibration Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(y, y_prob):
    df_cal = pd.DataFrame({'y': y, 'p': y_prob})
    try:
        df_cal['decile'] = pd.qcut(df_cal['p'], q=10, duplicates='drop', labels=False)
    except Exception:
        df_cal['decile'] = pd.cut(df_cal['p'], bins=10, labels=False)

    cal_df = df_cal.groupby('decile').agg(
        obs=('y', 'mean'),
        pred=('p', 'mean'),
        n=('y', 'count')
    ).reset_index()

    hl_stat, hl_p = hosmer_lemeshow_test(y, y_prob)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(cal_df['pred'], cal_df['obs'], s=cal_df['n'] / 5,
               color=COLORS['case2'], alpha=0.85, zorder=3, label='관측 비율')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='완벽 보정 (y=x)')

    z = np.polyfit(cal_df['pred'], cal_df['obs'], 1)
    xline = np.linspace(cal_df['pred'].min(), cal_df['pred'].max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), color=COLORS['case3'], linewidth=2,
            linestyle='-', label='회귀 직선')

    ax.set_xlabel("예측 확률 (Predicted Probability)", fontsize=11)
    ax.set_ylabel("관측 비율 (Observed Proportion)", fontsize=11)
    ax.set_title(f"Calibration Plot (Hosmer-Lemeshow)\n"
                 f"HL χ²={hl_stat:.2f}, p={hl_p:.4f} "
                 f"({'보정 양호' if hl_p > 0.05 else '보정 불량'})", fontsize=12)
    ax.legend(fontsize=9); ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.92, "원 크기 = 그룹 내 환자 수", transform=ax.transAxes,
            fontsize=8, color='gray')
    fig.tight_layout()
    savefig("09_logistic_calibration.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10 - 로지스틱: Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix_fig(y, y_prob):
    metrics = optimal_threshold_metrics(y, y_prob)
    thr = metrics['Threshold']
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['예측: Normal', '예측: Low TAMA'], fontsize=11)
    ax.set_yticklabels(['실제: Normal', '실제: Low TAMA'], fontsize=11)
    ax.set_xlabel("예측 클래스", fontsize=11); ax.set_ylabel("실제 클래스", fontsize=11)

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f"{val}\n({val/total*100:.1f}%)",
                    ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.colorbar(im, ax=ax)
    ax.set_title(
        f"Confusion Matrix (임계값={thr:.3f})\n"
        f"Sensitivity={metrics['Sensitivity']:.3f}  "
        f"Specificity={metrics['Specificity']:.3f}\n"
        f"PPV={metrics['PPV']:.3f}  NPV={metrics['NPV']:.3f}", fontsize=11)
    fig.tight_layout()
    savefig("10_logistic_confusion.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11 - 로지스틱: Crude OR Forest Plot (단변량)
# ─────────────────────────────────────────────────────────────────────────────

def plot_logistic_forest(uni_df: pd.DataFrame):
    df_plot = uni_df[~uni_df['Crude_OR'].astype(str).str.contains('Error|N/A|χ')].copy()
    df_plot['OR_f']  = pd.to_numeric(df_plot['Crude_OR'], errors='coerce')
    df_plot['lo_f']  = pd.to_numeric(df_plot['CI_Lower'], errors='coerce')
    df_plot['hi_f']  = pd.to_numeric(df_plot['CI_Upper'], errors='coerce')
    df_plot['p_f']   = pd.to_numeric(df_plot['p_value'],  errors='coerce')
    df_plot = df_plot.dropna(subset=['OR_f']).sort_values('OR_f')

    fig, ax = plt.subplots(figsize=(9, max(4, len(df_plot) * 0.7)))
    y_pos = np.arange(len(df_plot))

    for i, row in enumerate(df_plot.itertuples()):
        color = COLORS['pos'] if row.OR_f > 1 else COLORS['neg']
        ax.scatter(row.OR_f, i, color=color, s=80, zorder=5)
        ax.plot([row.lo_f, row.hi_f], [i, i], color=color, linewidth=2)

    ax.axvline(1.0, color='black', linestyle='--', linewidth=1.5, label='OR=1 (귀무)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['Variable'].tolist(), fontsize=10)
    ax.set_xscale('log')
    ax.set_xlabel("Crude OR (95% CI, log scale)", fontsize=11)
    ax.set_title("로지스틱 회귀 - 단변량 Crude OR Forest Plot\n(Low TAMA 위험 오즈비)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.3)

    for i, row in enumerate(df_plot.itertuples()):
        stars = '***' if row.p_f < 0.001 else ('**' if row.p_f < 0.01 else ('*' if row.p_f < 0.05 else 'ns'))
        ax.text(row.hi_f * 1.05, i, f"  {stars}", va='center', fontsize=10)

    fig.tight_layout()
    savefig("11_logistic_forest.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12 - Case 비교: 선형 성능 (R², RMSE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_case_metrics_bar(lin_sums):
    cases  = ['Case 1\n[Sex, Age]',
              'Case 2\n[+AEC]',
              'Case 3\n[+Model]']
    r2     = [s['R2']   for s in lin_sums]
    adj_r2 = [s['Adj_R2'] for s in lin_sums]
    rmse   = [s['RMSE'] for s in lin_sums]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(3)
    w = 0.35

    # R² vs Adj R²
    axes[0].bar(x - w/2, r2,     w, label='R²',         color=COLORS['case1'], alpha=0.85)
    axes[0].bar(x + w/2, adj_r2, w, label='Adjusted R²', color=COLORS['case2'], alpha=0.85)
    for i, (a, b) in enumerate(zip(r2, adj_r2)):
        axes[0].text(i - w/2, a + 0.003, f"{a:.3f}", ha='center', fontsize=9)
        axes[0].text(i + w/2, b + 0.003, f"{b:.3f}", ha='center', fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(cases, fontsize=10)
    axes[0].set_ylabel("R² 값", fontsize=11)
    axes[0].set_title("선형 회귀 R² 비교\n(Case 1 → 3)", fontsize=12)
    axes[0].legend(fontsize=10); axes[0].set_ylim(0, 0.85); axes[0].grid(axis='y', alpha=0.3)

    # RMSE
    axes[1].bar(x, rmse, color=[COLORS['case1'], COLORS['case2'], COLORS['case3']], alpha=0.85)
    for i, v in enumerate(rmse):
        axes[1].text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=10)
    axes[1].set_xticks(x); axes[1].set_xticklabels(cases, fontsize=10)
    axes[1].set_ylabel("RMSE (cm²)", fontsize=11)
    axes[1].set_title("선형 회귀 RMSE 비교\n(낮을수록 예측 정확)", fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle("Multivariable Analysis - 선형 회귀 성능 (Case 1 vs 2 vs 3)", fontsize=13, fontweight='bold')
    fig.tight_layout()
    savefig("12_case_metrics_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13 - Case 비교: 로지스틱 AUC
# ─────────────────────────────────────────────────────────────────────────────

def plot_case_auc(log_sums):
    cases  = ['Case 1\n[Sex, Age]', 'Case 2\n[+AEC]', 'Case 3\n[+Model]']
    aucs   = [s['AUC']          for s in log_sums]
    auc_lo = [s['AUC_CI_lower'] for s in log_sums]
    auc_hi = [s['AUC_CI_upper'] for s in log_sums]
    nag_r2 = [s['Nagelkerke_R2'] for s in log_sums]
    colors = [COLORS['case1'], COLORS['case2'], COLORS['case3']]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # AUC with CI
    x = np.arange(3)
    axes[0].bar(x, aucs, color=colors, alpha=0.8, zorder=2)
    for i, (a, lo, hi) in enumerate(zip(aucs, auc_lo, auc_hi)):
        axes[0].errorbar(i, a, yerr=[[a - lo], [hi - a]],
                         fmt='none', color='black', capsize=6, linewidth=2, zorder=3)
        axes[0].text(i, hi + 0.005, f"{a:.3f}", ha='center', fontsize=10, fontweight='bold')
    axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=1.2, label='무작위 기준 (0.5)')
    axes[0].axhline(0.7, color='orange', linestyle=':', linewidth=1.2, label='양호 기준 (0.7)')
    axes[0].axhline(0.8, color='green', linestyle=':', linewidth=1.2, label='우수 기준 (0.8)')
    axes[0].set_xticks(x); axes[0].set_xticklabels(cases, fontsize=10)
    axes[0].set_ylabel("AUC-ROC", fontsize=11)
    axes[0].set_title("로지스틱 AUC 비교 (Bootstrap 95%CI)\nCase 1 → 3", fontsize=12)
    axes[0].set_ylim(0.45, 0.95)
    axes[0].legend(fontsize=8, loc='upper left'); axes[0].grid(axis='y', alpha=0.3)

    # Nagelkerke R²
    axes[1].bar(x, nag_r2, color=colors, alpha=0.8)
    for i, v in enumerate(nag_r2):
        axes[1].text(i, v + 0.003, f"{v:.3f}", ha='center', fontsize=10)
    axes[1].set_xticks(x); axes[1].set_xticklabels(cases, fontsize=10)
    axes[1].set_ylabel("Nagelkerke R²", fontsize=11)
    axes[1].set_title("로지스틱 Nagelkerke R²\n(설명력 비교)", fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle("Multivariable Analysis - 로지스틱 회귀 성능 (Case 1 vs 2 vs 3)", fontsize=13, fontweight='bold')
    fig.tight_layout()
    savefig("13_case_auc_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 14 - Case 비교: AIC/BIC
# ─────────────────────────────────────────────────────────────────────────────

def plot_case_aic(lin_sums, log_sums):
    cases  = ['Case 1', 'Case 2', 'Case 3']
    lin_aic = [s['AIC'] for s in lin_sums]
    lin_bic = [s['BIC'] for s in lin_sums]
    log_aic = [s['AIC'] for s in log_sums]
    log_bic = [s['BIC'] for s in log_sums]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(3)
    w = 0.35

    for ax, aic_vals, bic_vals, title in [
        (axes[0], lin_aic, lin_bic, '선형 회귀 AIC/BIC'),
        (axes[1], log_aic, log_bic, '로지스틱 회귀 AIC/BIC'),
    ]:
        ax.bar(x - w/2, aic_vals, w, label='AIC', color=COLORS['case1'], alpha=0.85)
        ax.bar(x + w/2, bic_vals, w, label='BIC', color=COLORS['case3'], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(cases, fontsize=10)
        ax.set_ylabel("AIC / BIC 값", fontsize=11)
        ax.set_title(f"{title}\n(낮을수록 모델 적합도 우수)", fontsize=12)
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
        # 변화량 표시
        for i in range(1, 3):
            delta_aic = aic_vals[i] - aic_vals[i - 1]
            ax.annotate(f'ΔAIC={delta_aic:+.0f}',
                        xy=(i, aic_vals[i]), xytext=(i + 0.15, aic_vals[i] + (max(aic_vals) - min(aic_vals)) * 0.03),
                        fontsize=8, color=COLORS['case1'],
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    fig.suptitle("Multivariable Analysis - AIC/BIC 모델 비교", fontsize=13, fontweight='bold')
    fig.tight_layout()
    savefig("14_case_aic_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 15 - Case 추이: 다중 지표 종합
# ─────────────────────────────────────────────────────────────────────────────

def plot_case_progression(lin_sums, log_sums):
    case_labels = ['Case 1\n[Sex, Age]', 'Case 2\n[+AEC]', 'Case 3\n[+Model]']
    x = np.arange(3)

    metrics = {
        'Linear R²':          [s['R2']           for s in lin_sums],
        'Linear Adj R²':      [s['Adj_R2']       for s in lin_sums],
        'Logistic AUC':       [s['AUC']          for s in log_sums],
        'Nagelkerke R²':      [s['Nagelkerke_R2'] for s in log_sums],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    palette = [COLORS['case1'], COLORS['case2'], COLORS['case3'], '#59A14F']

    for idx, (name, vals) in enumerate(metrics.items()):
        ax = axes[idx]
        ax.plot(x, vals, 'o-', color=palette[idx], linewidth=2.5, markersize=10,
                markerfacecolor='white', markeredgewidth=2.5)
        for xi, val in zip(x, vals):
            ax.text(xi, val + 0.003, f"{val:.3f}", ha='center', fontsize=11, fontweight='bold')
        # ΔDELTA 주석
        for i in range(1, 3):
            delta = vals[i] - vals[i - 1]
            ax.annotate(f'Δ={delta:+.3f}',
                        xy=(i, vals[i]),
                        xytext=(i - 0.3, vals[i] + (max(vals) - min(vals)) * 0.15 + 0.01),
                        fontsize=9, color='gray',
                        arrowprops=dict(arrowstyle='->', color='lightgray', lw=1))
        ax.set_xticks(x); ax.set_xticklabels(case_labels, fontsize=9)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f"{name} 추이", fontsize=12)
        ax.set_ylim(min(vals) * 0.9, max(vals) * 1.12)
        ax.grid(alpha=0.3)

    fig.suptitle("Case 1 → 2 → 3: 예측 성능 지표 추이\n(AEC와 CT모델 추가에 따른 성능 향상 정량화)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    savefig("15_case_progression.png")


# ─────────────────────────────────────────────────────────────────────────────
# 16 - CT 스캐너 분포
# ─────────────────────────────────────────────────────────────────────────────

def plot_scanner_distribution(df_raw):
    df_sc = scanner_distribution(df_raw)
    top_n = 10
    top   = df_sc.head(top_n).copy()
    other_n = df_sc['N'][top_n:].sum() if len(df_sc) > top_n else 0
    if other_n > 0:
        other_pct = round(other_n / df_sc['N'].sum() * 100, 1)
        top = pd.concat([top, pd.DataFrame([{'Model': '기타', 'N': other_n, 'Pct': other_pct}])],
                        ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.55)))
    colors = [COLORS['case1'] if i == 0 else COLORS['gray'] for i in range(len(top))]
    bars = ax.barh(top['Model'][::-1], top['N'][::-1], color=colors[::-1],
                   edgecolor='white', linewidth=0.5)
    for bar, n, pct in zip(bars, top['N'][::-1], top['Pct'][::-1]):
        ax.text(bar.get_width() + df_sc['N'].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{n} ({pct}%)", va='center', fontsize=9)
    ax.set_xlabel('환자 수', fontsize=11)
    ax.set_title(f'CT 스캐너 모델 분포 (총 {len(df_sc)}종)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, df_sc['N'].max() * 1.25)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    savefig("16_scanner_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 17 - kVp 분포
# ─────────────────────────────────────────────────────────────────────────────

def plot_kvp_distribution(df_raw):
    kvp_df = kvp_distribution(df_raw)
    if kvp_df.empty:
        print("  [스킵] kVp 컬럼 없음")
        return

    dominant_kvp = kvp_df.loc[kvp_df['N'].idxmax(), 'kVp']
    colors = [COLORS['case1'] if v == dominant_kvp else COLORS['gray']
              for v in kvp_df['kVp']]

    fig, ax = plt.subplots(figsize=(max(6, len(kvp_df) * 0.9), 5))
    bars = ax.bar(kvp_df['kVp'].astype(str), kvp_df['N'], color=colors,
                  edgecolor='white', linewidth=0.5)
    for bar, n, pct in zip(bars, kvp_df['N'], kvp_df['Pct']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + kvp_df['N'].max() * 0.01,
                f"{n}\n({pct}%)", ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('kVp', fontsize=11)
    ax.set_ylabel('환자 수', fontsize=11)
    ax.set_title(f'kVp 분포 (주요값: {dominant_kvp} kVp)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, kvp_df['N'].max() * 1.2)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    savefig("17_kvp_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  시각화 생성 - TAMA 회귀분석 연구")
    print(f"  저장 위치: {FIG_DIR}")
    print("=" * 60)

    # ── 데이터 준비 ─────────────────────────────────────────────────────────
    print("\n[데이터 로드]")
    df_raw, _ = data_loader.load_raw_data()
    df_lin  = data_loader.prepare_full(mode='linear')
    df_log  = data_loader.prepare_full(mode='logistic')

    all_feat = ['Sex', 'Age_z']
    all_feat += [f + '_z' for f in config.SELECTED_AEC_FEATURES]
    all_feat += sorted([c for c in df_lin.columns if c.startswith('Model_')])
    all_feat = [c for c in all_feat if c in df_lin.columns]

    # ── 전체 모델 적합 (시각화용) ─────────────────────────────────────────
    print("\n[모델 적합 - 시각화용]")
    y_lin  = df_lin['TAMA'].values
    X_lin  = sm.add_constant(df_lin[all_feat].values.astype(float), has_constant='add')
    res_lin = sm.OLS(y_lin, X_lin).fit()

    # 계수 DataFrame (forest plot용)
    ci = res_lin.conf_int()
    coef_df = pd.DataFrame({
        'Variable': ['Intercept'] + all_feat,
        'β':        np.round(res_lin.params, 4),
        'CI_Lower': np.round(ci[:, 0], 4),
        'CI_Upper': np.round(ci[:, 1], 4),
        'p_value':  np.round(res_lin.pvalues, 6),
    })

    # 로지스틱 단변량 (forest plot용)
    from logistic_regression import run_univariate as log_uni
    uni_log_df = log_uni(df_log)

    # 선형 단변량 (R² bar용)
    from linear_regression import run_univariate as lin_uni
    uni_lin_df = lin_uni(df_lin)

    # Multivariable Analysis 재계산 (summary용)
    print("\n[Multivariable Case 결과 재계산]")
    from multivariable_analysis import fit_linear_case, fit_logistic_case
    lin_sums, log_sums = [], []
    for case in [1, 2, 3]:
        feat_cols = data_loader.get_feature_cols(case, df_lin)
        _, _, ls = fit_linear_case(df_lin, feat_cols)
        _, _, lgs = fit_logistic_case(df_log, feat_cols)
        lin_sums.append(ls)
        log_sums.append(lgs)

    # ── 그래프 생성 ─────────────────────────────────────────────────────────
    print("\n[그래프 생성 시작]")

    plot_feature_correlation(df_raw)             # 01
    plot_vif(df_raw)                             # 02
    plot_tama_distribution(df_raw)               # 03
    plot_linear_actual_vs_pred(df_lin, res_lin, all_feat)  # 04
    plot_linear_residuals(df_lin, res_lin, all_feat)       # 05
    plot_linear_forest(coef_df)                  # 06
    plot_linear_univariate_r2(uni_lin_df)        # 07
    y_log, y_prob = plot_roc_curve_fig(df_log, all_feat)   # 08
    plot_calibration(y_log, y_prob)              # 09
    plot_confusion_matrix_fig(y_log, y_prob)     # 10
    plot_logistic_forest(uni_log_df)             # 11
    plot_case_metrics_bar(lin_sums)              # 12
    plot_case_auc(log_sums)                      # 13
    plot_case_aic(lin_sums, log_sums)            # 14
    plot_case_progression(lin_sums, log_sums)    # 15
    plot_scanner_distribution(df_raw)           # 16
    plot_kvp_distribution(df_raw)               # 17

    print(f"\n[완료] 17개 그래프 저장 → {FIG_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
