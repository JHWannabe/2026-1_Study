"""
Model 4: Age + Sex + BMI + handcrafted AEC features  (Logistic Regression)
Feature selection 전략:
  Step 1) 상관계수 > 0.9 인 AEC feature 중복 제거
  Step 2) Age/Sex/BMI 고정 후 incremental AEC feature 선택
          (L2 penalty 포함 LR로 train AUC 기준 forward selection 없이,
           전체 비중복 feature를 L2 LR에 넣어 자동 regularize)
  Step 3) VIF는 경고 출력만 (최종 판단은 external validation)
ref: 이홍선교수님_260506.docx §10 item 4
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_gangnam_handcrafted, train_test_idx, CLIN_COLS, HC_COLS
from utils import compute_metrics, save_eval_plots, save_report

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, '..', '..', 'results', '0508_2', 'model4_clinic_hc'))
SEED = 42
CORR_THRESH = 0.90   # 상관계수 임계값 (중복 제거)


def _dedup_features(X_df: pd.DataFrame, cols: list) -> list:
    """상관계수 > CORR_THRESH 인 feature 중 하나만 남김 (첫 번째 유지)"""
    corr = X_df[cols].corr().abs()
    to_drop = set()
    for i, c in enumerate(cols):
        if c in to_drop:
            continue
        for c2 in cols[i+1:]:
            if c2 not in to_drop and corr.loc[c, c2] > CORR_THRESH:
                to_drop.add(c2)
    remaining = [c for c in cols if c not in to_drop]
    print(f'  Feature dedup: {len(cols)} -> {len(remaining)} '
          f'(removed {len(to_drop)} corr>{CORR_THRESH})')
    return remaining


def _vif_warning(X_df: pd.DataFrame, cols: list, bmi_corr_thresh: float = 0.7):
    """BMI와 상관계수 높은 feature 경고 출력 (VIF 경고등)"""
    corr_with_bmi = X_df[cols].corrwith(X_df['BMI']).abs()
    high = corr_with_bmi[corr_with_bmi > bmi_corr_thresh]
    if len(high) > 0:
        print(f'  [VIF 경고] BMI 상관계수 > {bmi_corr_thresh} feature:')
        for feat, val in high.sort_values(ascending=False).items():
            print(f'    {feat}: r={val:.3f}')


def run_model4(target: str = 'SMI') -> dict:
    print(f'\n{"="*60}')
    print(f'[Model 4] Age + Sex + BMI + handcrafted AEC  |  target={target}')
    print(f'{"="*60}')

    df = load_gangnam_handcrafted(target)
    tr_idx, te_idx = train_test_idx(df, target)
    df_train = df.iloc[tr_idx].reset_index(drop=True)
    df_test  = df.iloc[te_idx].reset_index(drop=True)
    y_train, y_test = df_train[f'{target}_bin'].values, df_test[f'{target}_bin'].values

    # Feature selection: train set 기준으로만 수행 (leakage 방지)
    hc_selected = _dedup_features(df_train, HC_COLS)
    _vif_warning(df_train, hc_selected)
    all_feats = CLIN_COLS + hc_selected
    print(f'  Total features: {len(all_feats)} ({len(CLIN_COLS)} clinical + {len(hc_selected)} AEC HC)')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_metrics = []

    for fold, (tr, vl) in enumerate(skf.split(df_train, y_train), 1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(df_train.iloc[tr][all_feats].values)
        X_vl = sc.transform(df_train.iloc[vl][all_feats].values)
        y_tr, y_vl = y_train[tr], y_train[vl]
        lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED, solver='lbfgs')
        lr.fit(X_tr, y_tr)
        prob = lr.predict_proba(X_vl)[:, 1]
        m = compute_metrics(y_vl, prob)
        cv_metrics.append(m)
        print(f'  Fold {fold}: AUC={m["AUC"]:.4f}  AUPRC={m["AUPRC"]:.4f}'
              f'  Brier={m["Brier"]:.4f}  Acc={m["Acc"]:.4f}')

    means = {k: np.mean([m[k] for m in cv_metrics]) for k in ['AUC','AUPRC','Brier','Acc']}
    print(f'\n  CV Mean  AUC={means["AUC"]:.4f}  AUPRC={means["AUPRC"]:.4f}'
          f'  Brier={means["Brier"]:.4f}  Acc={means["Acc"]:.4f}')

    # 최종 모델
    sc_f = StandardScaler()
    X_tr_f = sc_f.fit_transform(df_train[all_feats].values)
    X_te_f = sc_f.transform(df_test[all_feats].values)
    lr_f   = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED, solver='lbfgs')
    lr_f.fit(X_tr_f, y_train)
    prob_te = lr_f.predict_proba(X_te_f)[:, 1]

    test_m = compute_metrics(y_test, prob_te)
    print(f'  Test  AUC={test_m["AUC"]:.4f}  AUPRC={test_m["AUPRC"]:.4f}'
          f'  Brier={test_m["Brier"]:.4f}  Acc={test_m["Acc"]:.4f}')

    cr_text = classification_report(
        y_test, (prob_te >= 0.5).astype(int),
        target_names=['Normal', f'Low {target}'])
    print(cr_text)

    # Top 10 계수 출력
    coef_s = pd.Series(lr_f.coef_[0], index=all_feats)
    print('  Top 10 |coef|:')
    for feat in coef_s.abs().nlargest(10).index:
        print(f'    {feat}: {coef_s[feat]:.4f}')

    subdir = os.path.join(RESULTS_DIR, target)
    save_eval_plots(y_test, prob_te, 'Model4_ClinicHC', subdir)
    save_report(cv_metrics, test_m, cr_text,
                'Model4_ClinicHC',
                f'Age + Sex + BMI + handcrafted AEC ({len(hc_selected)} features, corr-dedup)',
                target, subdir)
    return test_m


if __name__ == '__main__':
    run_model4('SMI')
