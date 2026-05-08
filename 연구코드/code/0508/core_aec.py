"""
공통 분석 모듈 — Linear/Logistic Regression + 시각화
Input: aec1~aec256 (256개 AEC 피처)
Output: TAMA, SMI
"""
from core_resnet1d import run_resnet1d as _resnet1d_run
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    brier_score_loss,
    confusion_matrix, classification_report,
)
from sklearn.calibration import calibration_curve

# ── 경로 설정 ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'data', '강남_최종_정리본.xlsx'))
SHEET_NAME = 'aec_interpolation_final'
RESULTS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'results', '0508', 'aec_only'))
FEATURES = [f'aec{i}' for i in range(1, 257)]
TOP_N_COEF = 20  # 계수 시각화 시 상위 N개만 표시


# ── 데이터 / 저장 유틸 ────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    df['PatientSex'] = df['PatientSex'].map({'M': 0, 'F': 1})
    return df


def save_fig(target: str, filename: str) -> None:
    os.makedirs(os.path.join(RESULTS_DIR, target), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, target, filename), dpi=150, bbox_inches='tight')
    plt.close()


# ── Linear Regression ─────────────────────────────────────
def run_linear_regression(target: str) -> None:
    df = load_data()
    x = df[FEATURES]
    y = df[target]

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_mse, fold_rmse, fold_r2 = [], [], []

    for train_idx, val_idx in kf.split(x_tr):
        xf_tr, xf_val = x_tr.iloc[train_idx], x_tr.iloc[val_idx]
        yf_tr, yf_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
        result = sm.OLS(yf_tr, sm.add_constant(xf_tr)).fit()
        y_pred = result.predict(sm.add_constant(xf_val)).to_numpy()
        mse_fold = mean_squared_error(yf_val, y_pred)
        fold_mse.append(mse_fold)
        fold_rmse.append(np.sqrt(mse_fold))
        fold_r2.append(r2_score(yf_val, y_pred))

    print(f'  5-Fold CV (Train 80%)  MSE={np.mean(fold_mse):.4f}+/-{np.std(fold_mse):.4f}'
          f'  RMSE={np.mean(fold_rmse):.4f}+/-{np.std(fold_rmse):.4f}'
          f'  R2={np.mean(fold_r2):.4f}+/-{np.std(fold_r2):.4f}')

    full_result = sm.OLS(y_tr, sm.add_constant(x_tr)).fit()
    coef_arr = full_result.params[FEATURES].to_numpy()
    pval_arr = full_result.pvalues[FEATURES].to_numpy()
    intercept = float(full_result.params['const'])

    y_pred_te = full_result.predict(sm.add_constant(x_te)).to_numpy()
    test_mse  = mean_squared_error(y_te, y_pred_te)
    test_rmse = np.sqrt(test_mse)
    test_r2   = r2_score(y_te, y_pred_te)
    residuals = y_te.values - y_pred_te

    coef_s = pd.Series(coef_arr, index=FEATURES)
    pval_s = pd.Series(pval_arr, index=FEATURES)
    print(f'  Test (20%)  RMSE={test_rmse:.4f}  R2={test_r2:.4f}')
    print(f'  상위 {TOP_N_COEF} 계수 (Train):')
    for feat in coef_s.abs().nlargest(TOP_N_COEF).index:
        print(f'    {feat}: {coef_s[feat]:.6f}  p={pval_s[feat]:.4f}')
    print(f'  Intercept: {intercept:.4f}')

    _plot_actual_vs_predicted(y_te, y_pred_te, test_r2, target)
    _plot_residuals(y_pred_te, residuals, target)
    _plot_linear_coefficients(coef_arr, target)

    df['Sex_Label'] = df['PatientSex'].map({0: 'Male', 1: 'Female'})
    _plot_distribution_by_sex(df, target)
    _plot_scatter_by_sex(df, 'BMI', target)
    _plot_scatter_by_sex(df, 'PatientAge', target)

    _save_linear_md(target, fold_mse, fold_rmse, fold_r2,
                    test_mse, test_rmse, test_r2,
                    coef_arr, pval_arr, intercept, df)


def _plot_actual_vs_predicted(y_test, y_pred, r2, target):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidths=0.3)
    lims = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, 'r--', label='Perfect Prediction')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'Actual {target}')
    ax.set_ylabel(f'Predicted {target}')
    ax.set_title(f'Actual vs Predicted {target}  (R²={r2:.3f})')
    ax.legend()
    save_fig(target, 'Linear_Actual_vs_Predicted.png')


def _plot_residuals(y_pred, residuals, target):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.3)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel(f'Predicted {target}')
    ax.set_ylabel('Residuals')
    ax.set_title(f'Residual Plot — {target} Linear Regression')
    save_fig(target, 'Linear_Residuals.png')


def _plot_linear_coefficients(coef_array, target):
    coef_s = pd.Series(coef_array, index=FEATURES)
    top = coef_s.abs().nlargest(TOP_N_COEF)
    top_coef = coef_s[top.index].sort_values()
    colors = ['#d73027' if c > 0 else '#4575b4' for c in top_coef]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top_coef.index, top_coef.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Top {TOP_N_COEF} Feature Coefficients — {target} Linear Regression')
    save_fig(target, 'Linear_Coefficients.png')


def _plot_distribution_by_sex(df, target):
    fig, ax = plt.subplots(figsize=(7, 5))
    for sex_label, grp in df.groupby('Sex_Label'):
        sns.kdeplot(grp[target], ax=ax, label=sex_label, fill=True, alpha=0.3)
    ax.set_xlabel(target)
    ax.set_title(f'{target} Distribution by Sex')
    ax.legend()
    save_fig(target, 'Distribution_by_Sex.png')


def _plot_scatter_by_sex(df, x_col, target):
    fig, ax = plt.subplots(figsize=(6, 5))
    for sex_label, grp in df.groupby('Sex_Label'):
        ax.scatter(grp[x_col], grp[target], alpha=0.5, label=sex_label,
                   edgecolors='k', linewidths=0.2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(target)
    ax.set_title(f'{x_col} vs {target} by Sex')
    ax.legend()
    save_fig(target, f'{x_col}_vs_{target}_by_Sex.png')


# ── Logistic Regression ───────────────────────────────────
def run_logistic_regression(target: str) -> None:
    df = load_data()
    df[f'{target}_Binary'] = df.groupby('PatientSex')[target].transform(
        lambda x: (x <= x.quantile(0.25)).astype(int)
    )

    print(f'\n  {target} 하위 25% Thresholds by Sex:')
    for sex_code, label in [(0, 'Male'), (1, 'Female')]:
        threshold = df[df['PatientSex'] == sex_code][target].quantile(0.25)
        print(f'    {label}: {threshold:.4f}')

    x = df[FEATURES]
    y = np.array(df[f'{target}_Binary'], dtype=int)

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_acc, fold_auc, fold_auprc, fold_brier = [], [], [], []

    for train_idx, val_idx in skf.split(x_tr, y_tr):
        xf_tr, xf_val = x_tr.iloc[train_idx], x_tr.iloc[val_idx]
        yf_tr, yf_val = y_tr[train_idx], y_tr[val_idx]
        result = sm.Logit(yf_tr, sm.add_constant(xf_tr)).fit(
            method='bfgs', maxiter=5000, disp=0
        )
        y_prob = result.predict(sm.add_constant(xf_val)).to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        fold_acc.append((y_pred == yf_val).mean())
        fold_auc.append(roc_auc_score(yf_val, y_prob))
        fold_auprc.append(average_precision_score(yf_val, y_prob))
        fold_brier.append(brier_score_loss(yf_val, y_prob))

    print(f'  5-Fold CV (Train 80%)  Accuracy={np.mean(fold_acc):.4f}+/-{np.std(fold_acc):.4f}'
          f'  AUC-ROC={np.mean(fold_auc):.4f}+/-{np.std(fold_auc):.4f}'
          f'  AUPRC={np.mean(fold_auprc):.4f}+/-{np.std(fold_auprc):.4f}'
          f'  Brier={np.mean(fold_brier):.4f}+/-{np.std(fold_brier):.4f}')

    full_result = sm.Logit(y_tr, sm.add_constant(x_tr)).fit(
        method='bfgs', maxiter=5000, disp=0
    )
    coef_arr = full_result.params[FEATURES].to_numpy()
    pval_arr = full_result.pvalues[FEATURES].to_numpy()
    or_arr = np.exp(coef_arr)

    y_prob_te = full_result.predict(sm.add_constant(x_te)).to_numpy()
    y_pred_te = (y_prob_te >= 0.5).astype(int)
    test_acc   = (y_pred_te == y_te).mean()
    test_auc   = roc_auc_score(y_te, y_prob_te)
    test_auprc = average_precision_score(y_te, y_prob_te)
    test_brier = brier_score_loss(y_te, y_prob_te)
    test_cm    = confusion_matrix(y_te, y_pred_te)

    coef_s = pd.Series(coef_arr, index=FEATURES)
    pval_s = pd.Series(pval_arr, index=FEATURES)
    or_s = pd.Series(or_arr, index=FEATURES)
    print(f'  Test (20%)  Accuracy={test_acc:.4f}  AUC-ROC={test_auc:.4f}'
          f'  AUPRC={test_auprc:.4f}  Brier={test_brier:.4f}')
    print(classification_report(y_te, y_pred_te, target_names=['Normal', f'Low {target}']))
    print(f'\n  Train Logit 결과 (상위 {TOP_N_COEF} 계수):')
    for feat in coef_s.abs().nlargest(TOP_N_COEF).index:
        print(f'    {feat}: coef={coef_s[feat]:.6f}  OR={or_s[feat]:.4f}  p={pval_s[feat]:.4f}')

    _plot_confusion_matrix(test_cm, target)
    _plot_roc_curve(pd.Series(y_te), y_prob_te, test_auc, target)
    _plot_prob_distribution(pd.Series(y_te), y_prob_te, target)
    _plot_logistic_coefficients(coef_arr, target)
    _plot_pr_curve(pd.Series(y_te), y_prob_te, test_auprc, target)
    _plot_calibration(pd.Series(y_te), y_prob_te, target)

    thresholds = {
        label: df[df['PatientSex'] == code][target].quantile(0.25)
        for code, label in [(0, 'Male'), (1, 'Female')]
    }
    df['Sex_Label'] = df['PatientSex'].map({0: 'Male', 1: 'Female'})
    _save_logistic_md(target, fold_acc, fold_auc, fold_auprc, fold_brier,
                      test_acc, test_auc, test_auprc, test_brier, test_cm,
                      y_te, y_pred_te, thresholds, coef_arr, pval_arr, df)


def _plot_confusion_matrix(conf_matrix, target):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', f'Low {target}'],
                yticklabels=['Normal', f'Low {target}'], ax=ax)
    ax.set_title(f'Confusion Matrix — {target} Logistic Regression')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    save_fig(target, 'Logistic_Confusion_Matrix.png')


def _plot_roc_curve(y_test, y_prob, auc, target):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {target} Logistic Regression')
    ax.legend(loc='lower right')
    save_fig(target, 'Logistic_ROC_Curve.png')


def _plot_prob_distribution(y_test, y_prob, target):
    prob_df = pd.DataFrame({'Probability': y_prob, 'Actual': y_test.values})
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, grp in prob_df.groupby('Actual'):
        name = f'Low {target}' if label == 1 else 'Normal'
        sns.kdeplot(grp['Probability'], ax=ax, label=name, fill=True, alpha=0.3, clip=(0, 1))
    ax.set_xlabel(f'Predicted Probability (Low {target})')
    ax.set_title(f'Predicted Probability Distribution — {target}')
    ax.legend()
    save_fig(target, 'Logistic_Prob_Distribution.png')


def _plot_logistic_coefficients(coef_array, target):
    coef_s = pd.Series(coef_array, index=FEATURES)
    top = coef_s.abs().nlargest(TOP_N_COEF)
    top_coef = coef_s[top.index].sort_values()
    colors = ['#d73027' if c > 0 else '#4575b4' for c in top_coef]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top_coef.index, top_coef.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient (log-odds)')
    ax.set_title(f'Top {TOP_N_COEF} Feature Coefficients — {target} Logistic Regression')
    save_fig(target, 'Logistic_Coefficients.png')


def _plot_pr_curve(y_test, y_prob, auprc, target):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f'AUPRC = {auprc:.3f}')
    baseline = float(y_test.mean())
    ax.axhline(baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'PR Curve — {target} Logistic Regression')
    ax.legend(loc='upper right')
    save_fig(target, 'Logistic_PR_Curve.png')


def _plot_calibration(y_test, y_prob, target):
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, 's-', label='Model')
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Plot — {target} Logistic Regression')
    ax.legend()
    save_fig(target, 'Logistic_Calibration.png')


# ── MD 보고서 ──────────────────────────────────────────────
def _sex_dist_table(df: pd.DataFrame, col: str) -> str:
    rows = ['| 통계 | Male | Female |', '|---|---|---|']
    stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    labels = ['Count', 'Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max']
    desc = df.groupby('Sex_Label')[col].describe()
    for stat, label in zip(stats, labels):
        male = desc.loc['Male', stat] if 'Male' in desc.index else float('nan')
        female = desc.loc['Female', stat] if 'Female' in desc.index else float('nan')
        fmt = '.0f' if stat == 'count' else '.4f'
        rows.append(f'| {label} | {male:{fmt}} | {female:{fmt}} |')
    return '\n'.join(rows)


def _top_coef_table(coef_array, pvalues, n=TOP_N_COEF) -> str:
    coef_s = pd.Series(coef_array, index=FEATURES)
    pval_s = pd.Series(pvalues, index=FEATURES)
    top = coef_s.abs().nlargest(n)
    rows = ['| Feature | Coefficient | P-value |', '|---|---|---|']
    for feat in top.index:
        rows.append(f'| {feat} | {coef_s[feat]:.6f} | {pval_s[feat]:.4f} |')
    return '\n'.join(rows)


def _top_logit_table(coef_array, pvalues, n=TOP_N_COEF) -> str:
    coef_s = pd.Series(coef_array, index=FEATURES)
    pval_s = pd.Series(pvalues, index=FEATURES)
    or_s = np.exp(coef_s)
    top = coef_s.abs().nlargest(n)
    rows = ['| Feature | Coefficient | Odds Ratio | P-value |', '|---|---|---|---|']
    for feat in top.index:
        rows.append(f'| {feat} | {coef_s[feat]:.6f} | {or_s[feat]:.4f} | {pval_s[feat]:.4f} |')
    return '\n'.join(rows)


def _save_linear_md(target, fold_mse, fold_rmse, fold_r2,
                    test_mse, test_rmse, test_r2,
                    coef, pvalues, intercept, df):
    lines = [
        f'# Linear Regression Report — {target}',
        f'',
        f'**Input features:** aec1 ~ aec256 ({len(FEATURES)}개)',
        '',
        '## 성별 데이터 분포',
        '',
        f'### {target}',
        '',
        _sex_dist_table(df, target),
        '',
        '### BMI',
        '',
        _sex_dist_table(df, 'BMI'),
        '',
        '### PatientAge',
        '',
        _sex_dist_table(df, 'PatientAge'),
        '',
        '## 5-Fold CV 성능 (Train 80%)',
        '',
        '| Fold | MSE | RMSE | R² |',
        '|---|---|---|---|',
    ]
    for i, (m, r, r2) in enumerate(zip(fold_mse, fold_rmse, fold_r2), 1):
        lines.append(f'| {i} | {m:.4f} | {r:.4f} | {r2:.4f} |')
    lines += [
        f'| **Mean** | **{np.mean(fold_mse):.4f}** | **{np.mean(fold_rmse):.4f}** | **{np.mean(fold_r2):.4f}** |',
        f'| **Std** | **{np.std(fold_mse):.4f}** | **{np.std(fold_rmse):.4f}** | **{np.std(fold_r2):.4f}** |',
        '',
        '## Test Set 성능 (Test 20%)',
        '',
        '| MSE | RMSE | R² |',
        '|---|---|---|',
        f'| **{test_mse:.4f}** | **{test_rmse:.4f}** | **{test_r2:.4f}** |',
        '',
        f'## 상위 {TOP_N_COEF} 계수 (Train 학습)',
        '',
        _top_coef_table(coef, pvalues),
        f'| Intercept | {intercept:.4f} |',
    ]

    os.makedirs(os.path.join(RESULTS_DIR, target), exist_ok=True)
    path = os.path.join(RESULTS_DIR, target, 'Linear_Report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  → Linear MD 저장: {path}')


def _save_logistic_md(target, fold_acc, fold_auc, fold_auprc, fold_brier,
                      test_acc, test_auc, test_auprc, test_brier, test_cm,
                      test_y_true, test_y_pred, thresholds, coef, pvalues, df):
    df = df.copy()
    tn, fp, fn, tp = test_cm.ravel()
    cr_text = classification_report(test_y_true, test_y_pred, target_names=['Normal', f'Low {target}'])

    lines = [
        f'# Logistic Regression Report — {target}',
        '',
        f'**Input features:** aec1 ~ aec256 ({len(FEATURES)}개)',
        '',
        '## 성별 데이터 분포',
        '',
        f'### {target}',
        '',
        _sex_dist_table(df, target),
        '',
        '### BMI',
        '',
        _sex_dist_table(df, 'BMI'),
        '',
        '### PatientAge',
        '',
        _sex_dist_table(df, 'PatientAge'),
        '',
        '## 성별 하위 25% 임계값',
        '',
        '| Sex | Threshold |',
        '|---|---|',
        f'| Male | {thresholds["Male"]:.4f} |',
        f'| Female | {thresholds["Female"]:.4f} |',
        '',
        '## 5-Fold CV 성능 (Train 80%)',
        '',
        '| Fold | Accuracy | AUC-ROC | AUPRC | Brier |',
        '|---|---|---|---|---|',
    ]
    for i, (acc, a, ap, br) in enumerate(zip(fold_acc, fold_auc, fold_auprc, fold_brier), 1):
        lines.append(f'| {i} | {acc:.4f} | {a:.4f} | {ap:.4f} | {br:.4f} |')
    lines += [
        f'| **Mean** | **{np.mean(fold_acc):.4f}** | **{np.mean(fold_auc):.4f}** | **{np.mean(fold_auprc):.4f}** | **{np.mean(fold_brier):.4f}** |',
        f'| **Std** | **{np.std(fold_acc):.4f}** | **{np.std(fold_auc):.4f}** | **{np.std(fold_auprc):.4f}** | **{np.std(fold_brier):.4f}** |',
        '',
        '## Test Set 성능 (Test 20%)',
        '',
        '| Accuracy | AUC-ROC | AUPRC | Brier |',
        '|---|---|---|---|',
        f'| **{test_acc:.4f}** | **{test_auc:.4f}** | **{test_auprc:.4f}** | **{test_brier:.4f}** |',
        '',
        '## Confusion Matrix (Test)',
        '',
        f'|  | Pred Normal | Pred Low {target} |',
        '|---|---|---|',
        f'| Actual Normal | {tn} | {fp} |',
        f'| Actual Low {target} | {fn} | {tp} |',
        '',
        '## Classification Report (Test)',
        '',
        '```',
        cr_text.strip(),
        '```',
        '',
        f'## 상위 {TOP_N_COEF} 계수 (Train 학습)',
        '',
        _top_logit_table(coef, pvalues),
    ]

    os.makedirs(os.path.join(RESULTS_DIR, target), exist_ok=True)
    path = os.path.join(RESULTS_DIR, target, 'Logistic_Report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  → Logistic MD 저장: {path}')


# ── ResNet1D ───────────────────────────────────────────────
def run_resnet1d(target: str) -> None:
    df = load_data()
    y  = np.array(df[target].values, dtype=float)
    _resnet1d_run(df, y, target, RESULTS_DIR,
                  label='ResNet1D_AECOnly',
                  feature_selector=lambda df_tr: FEATURES)
