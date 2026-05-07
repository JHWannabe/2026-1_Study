"""
Model 1: Age + Sex + BMI  (Logistic Regression baseline)
ref: 이홍선교수님_260506.docx §10 item 4
"""
import os, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_gangnam, train_test_idx, CLIN_COLS
from utils import compute_metrics, save_eval_plots, save_report

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, '..', '..', 'results', '0508_2', 'model1_clinic'))
SEED = 42


def run_model1(target: str = 'SMI') -> dict:
    print(f'\n{"="*60}')
    print(f'[Model 1] Age + Sex + BMI  |  target={target}')
    print(f'{"="*60}')

    df = load_gangnam(target)
    tr_idx, te_idx = train_test_idx(df, target)
    df_train, df_test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    y_train, y_test  = df_train[f'{target}_bin'].values, df_test[f'{target}_bin'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_metrics = []

    for fold, (tr, vl) in enumerate(skf.split(df_train, y_train), 1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(df_train.iloc[tr][CLIN_COLS].values)
        X_vl = sc.transform(df_train.iloc[vl][CLIN_COLS].values)
        y_tr, y_vl = y_train[tr], y_train[vl]

        lr = LogisticRegression(max_iter=1000, random_state=SEED)
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
    X_tr_f = sc_f.fit_transform(df_train[CLIN_COLS].values)
    X_te_f = sc_f.transform(df_test[CLIN_COLS].values)
    lr_f   = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_f.fit(X_tr_f, y_train)
    prob_te = lr_f.predict_proba(X_te_f)[:, 1]

    test_m = compute_metrics(y_test, prob_te)
    print(f'  Test  AUC={test_m["AUC"]:.4f}  AUPRC={test_m["AUPRC"]:.4f}'
          f'  Brier={test_m["Brier"]:.4f}  Acc={test_m["Acc"]:.4f}')

    cr_text = classification_report(
        y_test, (prob_te >= 0.5).astype(int),
        target_names=['Normal', f'Low {target}'])
    print(cr_text)

    print('  Coefficients:')
    for feat, coef in zip(CLIN_COLS, lr_f.coef_[0]):
        print(f'    {feat}: {coef:.4f}')

    subdir = os.path.join(RESULTS_DIR, target)
    save_eval_plots(y_test, prob_te, 'Model1_ClinicOnly', subdir)
    save_report(cv_metrics, test_m, cr_text,
                'Model1_ClinicOnly', 'Age + Sex + BMI (Logistic Regression)',
                target, subdir)
    return test_m


if __name__ == '__main__':
    run_model1('SMI')
