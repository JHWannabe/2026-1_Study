"""
공통 분석 모듈 — Linear/Logistic Regression + 시각화
TARGET 인자만 바꿔서 TAMA, SMI 등 어떤 연속형 변수에도 재사용 가능.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)

# ── 경로 설정 ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'data', '강남_최종_정리본.xlsx'))
SHEET_NAME = 'aec_feature_filtered'
RESULTS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'results', '0508'))
FEATURES = ['PatientSex', 'PatientAge', 'BMI']


# ── 데이터 / 저장 유틸 ────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    df['PatientSex'] = df['PatientSex'].map({'M': 0, 'F': 1})
    return df


def save_fig(filename: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


# ── Linear Regression ─────────────────────────────────────
def run_linear_regression(target: str) -> None:
    df = load_data()
    x = df[FEATURES]
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    residuals = y_test.values - y_pred

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'  MSE={mse:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}')
    print('  Coefficients:')
    for feat, coef in zip(FEATURES, model.coef_):
        print(f'    {feat}: {coef:.4f}')
    print(f'  Intercept: {model.intercept_:.4f}')

    _plot_actual_vs_predicted(y_test, y_pred, r2, target)
    _plot_residuals(y_pred, residuals, target)
    _plot_linear_coefficients(model.coef_, target)

    df['Sex_Label'] = df['PatientSex'].map({0: 'Male', 1: 'Female'})
    _plot_distribution_by_sex(df, target)
    _plot_scatter_by_sex(df, 'BMI', target)
    _plot_scatter_by_sex(df, 'PatientAge', target)


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
    save_fig(f'{target}_Linear_Actual_vs_Predicted.png')


def _plot_residuals(y_pred, residuals, target):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.3)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel(f'Predicted {target}')
    ax.set_ylabel('Residuals')
    ax.set_title(f'Residual Plot — {target} Linear Regression')
    save_fig(f'{target}_Linear_Residuals.png')


def _plot_linear_coefficients(coef_array, target):
    coef_s = pd.Series(coef_array, index=FEATURES).sort_values()
    colors = ['#d73027' if c > 0 else '#4575b4' for c in coef_s]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(coef_s.index, coef_s.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Feature Coefficients — {target} Linear Regression')
    save_fig(f'{target}_Linear_Coefficients.png')


def _plot_distribution_by_sex(df, target):
    fig, ax = plt.subplots(figsize=(7, 5))
    for sex_label, grp in df.groupby('Sex_Label'):
        sns.kdeplot(grp[target], ax=ax, label=sex_label, fill=True, alpha=0.3)
    ax.set_xlabel(target)
    ax.set_title(f'{target} Distribution by Sex')
    ax.legend()
    save_fig(f'{target}_Distribution_by_Sex.png')


def _plot_scatter_by_sex(df, x_col, target):
    fig, ax = plt.subplots(figsize=(6, 5))
    for sex_label, grp in df.groupby('Sex_Label'):
        ax.scatter(grp[x_col], grp[target], alpha=0.5, label=sex_label,
                   edgecolors='k', linewidths=0.2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(target)
    ax.set_title(f'{x_col} vs {target} by Sex')
    ax.legend()
    save_fig(f'{target}_{x_col}_vs_TARGET_by_Sex.png')


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
    y = df[f'{target}_Binary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    accuracy = (y_pred == y_test).mean()
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'  Accuracy={accuracy:.4f}  AUC-ROC={auc:.4f}')
    print(classification_report(y_test, y_pred, target_names=['Normal', f'Low {target}']))

    _plot_confusion_matrix(conf_matrix, target)
    _plot_roc_curve(y_test, y_prob, auc, target)
    _plot_prob_distribution(y_test, y_prob, target)
    _plot_logistic_coefficients(model.coef_[0], target)


def _plot_confusion_matrix(conf_matrix, target):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', f'Low {target}'],
                yticklabels=['Normal', f'Low {target}'], ax=ax)
    ax.set_title(f'Confusion Matrix — {target} Logistic Regression')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    save_fig(f'{target}_Logistic_Confusion_Matrix.png')


def _plot_roc_curve(y_test, y_prob, auc, target):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {target} Logistic Regression')
    ax.legend(loc='lower right')
    save_fig(f'{target}_Logistic_ROC_Curve.png')


def _plot_prob_distribution(y_test, y_prob, target):
    prob_df = pd.DataFrame({'Probability': y_prob, 'Actual': y_test.values})
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, grp in prob_df.groupby('Actual'):
        name = f'Low {target}' if label == 1 else 'Normal'
        sns.kdeplot(grp['Probability'], ax=ax, label=name, fill=True, alpha=0.3, clip=(0, 1))
    ax.set_xlabel(f'Predicted Probability (Low {target})')
    ax.set_title(f'Predicted Probability Distribution — {target}')
    ax.legend()
    save_fig(f'{target}_Logistic_Prob_Distribution.png')


def _plot_logistic_coefficients(coef_array, target):
    coef_s = pd.Series(coef_array, index=FEATURES).sort_values()
    colors = ['#d73027' if c > 0 else '#4575b4' for c in coef_s]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(coef_s.index, coef_s.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient (log-odds)')
    ax.set_title(f'Feature Coefficients — {target} Logistic Regression')
    save_fig(f'{target}_Logistic_Coefficients.png')
