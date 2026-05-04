# -*- coding: utf-8 -*-
"""
Utility functions shared across analysis modules.
"""

import os
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score,
)
from sklearn.pipeline import Pipeline

from config import AEC_PREV, AEC_CANDIDATES, AEC_SELECT_K, CV_SPLITS, CV_RANDOM


def copy_to_temp(src: Path, temp_name: str) -> Path:
    """OneDrive 잠금 우회: PowerShell로 임시 폴더에 복사 후 경로 반환."""
    dst = Path(os.environ["TEMP"]) / temp_name
    subprocess.run(
        ["powershell", "-Command",
         f'Copy-Item -Path "{src}" -Destination "{dst}" -Force'],
        capture_output=True,
    )
    return dst


def load_hospital(src: Path, temp_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Excel(features + metadata-bmi_add 시트)을 읽어 (feat_df, meta_df) 반환."""
    tmp  = copy_to_temp(src, temp_name)
    feat = pd.read_excel(str(tmp), sheet_name="features")
    meta = pd.read_excel(str(tmp), sheet_name="metadata-bmi_add")
    return feat, meta


def _select_aec_features(X_aec_train: pd.DataFrame,
                          y_train: pd.Series,
                          k: int) -> list[str]:
    """
    Train fold 데이터만 사용해 Pearson |r| 상위 k개 AEC 피처 선택.
    data leakage 방지: test fold 정보를 일절 사용하지 않음.
    """
    abs_r = {c: abs(pearsonr(X_aec_train[c], y_train)[0]) for c in X_aec_train.columns}
    top_k = sorted(abs_r, key=abs_r.get, reverse=True)[:min(k, len(abs_r))]
    return top_k


def linear_cv(X: pd.DataFrame, y: pd.Series,
              aec_candidate_cols: list = None) -> dict:
    """
    5-Fold CV 선형 회귀. fold별 StandardScaler 재적합으로 data leakage 방지.

    aec_candidate_cols가 제공될 경우: 각 fold의 train set에서
    SelectKBest(f_regression)로 AEC 피처를 선택하고 해당 피처만 사용.
    None이면 X 전체 컬럼을 그대로 사용.
    """
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM)
    r2s, maes, rmses, all_true, all_pred = [], [], [], [], []

    fixed_cols = ([c for c in X.columns if c not in aec_candidate_cols]
                  if aec_candidate_cols else list(X.columns))

    if aec_candidate_cols:
        print(f"  [linear_cv] AEC fold-wise selection (top-{AEC_SELECT_K} by |r|)")

    for fold, (tr, te) in enumerate(kf.split(X), 1):
        if aec_candidate_cols:
            sel_aec = _select_aec_features(
                X[aec_candidate_cols].iloc[tr], y.iloc[tr], AEC_SELECT_K
            )
            fold_cols = fixed_cols + sel_aec
            print(f"    Fold {fold}: {sel_aec}")
        else:
            fold_cols = fixed_cols

        pipe = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
        pipe.fit(X[fold_cols].iloc[tr], y.iloc[tr])
        pred = pipe.predict(X[fold_cols].iloc[te])

        r2s.append(r2_score(y.iloc[te], pred))
        maes.append(mean_absolute_error(y.iloc[te], pred))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
        all_true.extend(y.iloc[te].tolist())
        all_pred.extend(pred.tolist())

    return dict(
        R2=np.mean(r2s),   R2_std=np.std(r2s),
        MAE=np.mean(maes), MAE_std=np.std(maes),
        RMSE=np.mean(rmses), RMSE_std=np.std(rmses),
        fold_r2=r2s, oof_true=all_true, oof_pred=all_pred,
    )


def logistic_cv(X: pd.DataFrame, y: pd.Series,
                aec_candidate_cols: list = None) -> dict:
    """
    5-Fold StratifiedKFold 로지스틱 회귀.
    Sensitivity/Specificity는 각 fold별 Youden Index 최적 threshold로 산출.

    aec_candidate_cols가 제공될 경우: 각 fold의 train set에서
    SelectKBest(f_regression)로 AEC 피처를 선택.
    """
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM)
    aucs, accs, sens, specs, fprs_l, tprs_l = [], [], [], [], [], []
    oof_prob_all, oof_true_all = [], []

    fixed_cols = ([c for c in X.columns if c not in aec_candidate_cols]
                  if aec_candidate_cols else list(X.columns))

    if aec_candidate_cols:
        print(f"  [logistic_cv] AEC fold-wise selection (top-{AEC_SELECT_K} by |r|)")

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        if aec_candidate_cols:
            sel_aec = _select_aec_features(
                X[aec_candidate_cols].iloc[tr], y.iloc[tr], AEC_SELECT_K
            )
            fold_cols = fixed_cols + sel_aec
            print(f"    Fold {fold}: {sel_aec}")
        else:
            fold_cols = fixed_cols

        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("m",  LogisticRegression(max_iter=2000, random_state=CV_RANDOM, solver="lbfgs")),
        ])
        pipe.fit(X[fold_cols].iloc[tr], y.iloc[tr])
        prob = pipe.predict_proba(X[fold_cols].iloc[te])[:, 1]
        fpr, tpr, thresholds = roc_curve(y.iloc[te], prob)

        # Youden Index 최적 threshold (Sensitivity + Specificity - 1 최대화)
        youden   = tpr - fpr
        best_idx = np.argmax(youden)
        best_thr = thresholds[best_idx]
        pred_opt = (prob >= best_thr).astype(int)

        aucs.append(roc_auc_score(y.iloc[te], prob))
        accs.append(accuracy_score(y.iloc[te], pred_opt))
        tn, fp, fn, tp = confusion_matrix(y.iloc[te], pred_opt).ravel()
        sens.append(tp / (tp + fn) if tp + fn else 0)
        specs.append(tn / (tn + fp) if tn + fp else 0)
        fprs_l.append(fpr); tprs_l.append(tpr)
        oof_prob_all.extend(prob.tolist())
        oof_true_all.extend(y.iloc[te].tolist())

    # 전체 OOF 기준 최종 Youden threshold (confusion matrix 시각화용)
    fpr_all, tpr_all, thr_all = roc_curve(oof_true_all, oof_prob_all)
    best_global = thr_all[np.argmax(tpr_all - fpr_all)]

    return dict(
        AUC=np.mean(aucs),      AUC_std=np.std(aucs),
        Accuracy=np.mean(accs), Accuracy_std=np.std(accs),
        Sensitivity=np.mean(sens), Sensitivity_std=np.std(sens),
        Specificity=np.mean(specs), Specificity_std=np.std(specs),
        fold_auc=aucs, fprs=fprs_l, tprs=tprs_l,
        oof_prob=oof_prob_all, oof_true=oof_true_all,
        youden_threshold=float(best_global),
    )


def make_cases(clinical_feats: list, scanner_feats: list) -> dict:
    """5단계 케이스 딕셔너리 생성. 층화 분석 시 clinical_feats에서 Sex 제거.
    Case3/5는 AEC_CANDIDATES 전체를 포함 — CV fold 내 SelectKBest로 최종 선택."""
    return {
        "Case1_Clinical":                  clinical_feats,
        "Case2_Clinical+AEC_prev":         clinical_feats + AEC_PREV,
        "Case3_Clinical+AEC_new":          clinical_feats + AEC_CANDIDATES,
        "Case4_Clinical+AEC_prev+Scanner": clinical_feats + AEC_PREV + scanner_feats,
        "Case5_Clinical+AEC_new+Scanner":  clinical_feats + AEC_CANDIDATES + scanner_feats,
    }


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"
