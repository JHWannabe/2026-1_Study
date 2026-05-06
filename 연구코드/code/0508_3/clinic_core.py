"""
공통 분석 모듈 — Linear/Logistic Regression + 시각화
Input: aec1~aec256 (256개 AEC 피처)
Output: TAMA, SMI
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score,
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)

# ── 경로 설정 ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'data', '강남_최종_정리본.xlsx'))
SHEET_NAME = 'aec_interpolation_final'
RESULTS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'results', '0508_3'))
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

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_mse, fold_rmse, fold_r2 = [], [], []
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        all_y_true.extend(y_test.values)
        all_y_pred.extend(y_pred)
        mse_fold = mean_squared_error(y_test, y_pred)
        fold_mse.append(mse_fold)
        fold_rmse.append(np.sqrt(mse_fold))
        fold_r2.append(r2_score(y_test, y_pred))

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    overall_r2 = r2_score(all_y_true, all_y_pred)
    residuals = all_y_true - all_y_pred

    print(f'  5-Fold CV  MSE={np.mean(fold_mse):.4f}+/-{np.std(fold_mse):.4f}'
          f'  RMSE={np.mean(fold_rmse):.4f}+/-{np.std(fold_rmse):.4f}'
          f'  R2={np.mean(fold_r2):.4f}+/-{np.std(fold_r2):.4f}')

    full_model = LinearRegression()
    full_model.fit(x, y)

    coef_s = pd.Series(full_model.coef_, index=FEATURES)
    top_coef = coef_s.abs().nlargest(TOP_N_COEF)
    print(f'  상위 {TOP_N_COEF} 계수 (전체 데이터):')
    for feat in top_coef.index:
        print(f'    {feat}: {coef_s[feat]:.6f}')
    print(f'  Intercept: {full_model.intercept_:.4f}')

    _plot_actual_vs_predicted(pd.Series(all_y_true), all_y_pred, overall_r2, target)
    _plot_residuals(all_y_pred, residuals, target)
    _plot_linear_coefficients(full_model.coef_, target)

    df['Sex_Label'] = df['PatientSex'].map({0: 'Male', 1: 'Female'})
    _plot_distribution_by_sex(df, target)
    _plot_scatter_by_sex(df, 'BMI', target)
    _plot_scatter_by_sex(df, 'PatientAge', target)

    _save_linear_md(target, fold_mse, fold_rmse, fold_r2, overall_r2,
                    full_model.coef_, full_model.intercept_, df)


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

    x = df[FEATURES].values
    y = df[f'{target}_Binary'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_acc, fold_auc = [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for train_idx, test_idx in skf.split(x, y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)
        model = LogisticRegression(max_iter=2000, C=0.1)
        model.fit(x_train_s, y_train)
        y_pred = model.predict(x_test_s)
        y_prob = model.predict_proba(x_test_s)[:, 1]
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        fold_acc.append((y_pred == y_test).mean())
        fold_auc.append(roc_auc_score(y_test, y_prob))

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    auc = roc_auc_score(all_y_true, all_y_prob)
    conf_matrix = confusion_matrix(all_y_true, all_y_pred)

    print(f'  5-Fold CV  Accuracy={np.mean(fold_acc):.4f}+/-{np.std(fold_acc):.4f}'
          f'  AUC-ROC={np.mean(fold_auc):.4f}+/-{np.std(fold_auc):.4f}')
    print(classification_report(all_y_true, all_y_pred, target_names=['Normal', f'Low {target}']))

    full_scaler = StandardScaler()
    x_scaled = full_scaler.fit_transform(x)
    full_model = LogisticRegression(max_iter=2000, C=0.1)
    full_model.fit(x_scaled, y)

    _plot_confusion_matrix(conf_matrix, target)
    _plot_roc_curve(pd.Series(all_y_true), all_y_prob, auc, target)
    _plot_prob_distribution(pd.Series(all_y_true), all_y_prob, target)
    _plot_logistic_coefficients(full_model.coef_[0], target)

    thresholds = {
        label: df[df['PatientSex'] == code][target].quantile(0.25)
        for code, label in [(0, 'Male'), (1, 'Female')]
    }
    df['Sex_Label'] = df['PatientSex'].map({0: 'Male', 1: 'Female'})
    _save_logistic_md(target, fold_acc, fold_auc, auc, conf_matrix,
                      all_y_true, all_y_pred, thresholds, full_model.coef_[0], df)


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


def _top_coef_table(coef_array, n=TOP_N_COEF) -> str:
    coef_s = pd.Series(coef_array, index=FEATURES)
    top = coef_s.abs().nlargest(n)
    rows = [f'| Feature | Coefficient |', '|---|---|']
    for feat in top.index:
        rows.append(f'| {feat} | {coef_s[feat]:.6f} |')
    return '\n'.join(rows)


def _save_linear_md(target, fold_mse, fold_rmse, fold_r2, overall_r2, coef, intercept, df):
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
        '## 5-Fold CV 성능',
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
        f'OOF 전체 R² = **{overall_r2:.4f}**',
        '',
        f'## 상위 {TOP_N_COEF} 계수 (전체 데이터 학습)',
        '',
        _top_coef_table(coef),
        f'| Intercept | {intercept:.4f} |',
    ]

    os.makedirs(os.path.join(RESULTS_DIR, target), exist_ok=True)
    path = os.path.join(RESULTS_DIR, target, 'Linear_Report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  → Linear MD 저장: {path}')


def _save_logistic_md(target, fold_acc, fold_auc, auc, conf_matrix,
                      all_y_true, all_y_pred, thresholds, coef, df):
    df = df.copy()
    tn, fp, fn, tp = conf_matrix.ravel()
    cr_text = classification_report(all_y_true, all_y_pred, target_names=['Normal', f'Low {target}'])

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
        '## 5-Fold CV 성능',
        '',
        '| Fold | Accuracy | AUC-ROC |',
        '|---|---|---|',
    ]
    for i, (acc, a) in enumerate(zip(fold_acc, fold_auc), 1):
        lines.append(f'| {i} | {acc:.4f} | {a:.4f} |')
    lines += [
        f'| **Mean** | **{np.mean(fold_acc):.4f}** | **{np.mean(fold_auc):.4f}** |',
        f'| **Std** | **{np.std(fold_acc):.4f}** | **{np.std(fold_auc):.4f}** |',
        '',
        f'OOF 전체 AUC-ROC = **{auc:.4f}**',
        '',
        '## Confusion Matrix (OOF)',
        '',
        f'|  | Pred Normal | Pred Low {target} |',
        '|---|---|---|',
        f'| Actual Normal | {tn} | {fp} |',
        f'| Actual Low {target} | {fn} | {tp} |',
        '',
        '## Classification Report (OOF)',
        '',
        '```',
        cr_text.strip(),
        '```',
        '',
        f'## 상위 {TOP_N_COEF} 계수 (전체 데이터 학습)',
        '',
        _top_coef_table(coef),
    ]

    os.makedirs(os.path.join(RESULTS_DIR, target), exist_ok=True)
    path = os.path.join(RESULTS_DIR, target, 'Logistic_Report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  → Logistic MD 저장: {path}')
