"""
5-Fold Stratified Cross-Validation 루프.

각 함수는 fold별 성능 지표·ROC 데이터·학습 이력을 수집하고,
최적 epoch(best val AUC)를 기록해 최종 test 평가(evaluate.py)에 전달한다.

  run_cross_validation        — M1: LR
  run_cross_validation_cross  — M2/M2_2: CrossAttn
  run_cross_validation_cross3 — M3: CrossAttn3

스케일링 정책:
  Clinical (Age·BMI): _maybe_scale_clin — sex_enc(이진값)은 제외하고 col 0, 2만 표준화
  AEC 전 컬럼       : _maybe_scale(StandardScaler)
  Fold 누수 방지    : scaler를 train fold에서 fit 후 val fold에 transform만 적용
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from config import N_FOLDS, SEED, EPOCHS
from models import (build_cross_attn, make_dual_loaders, train_cross_epoch, eval_cross_loader,
                    build_cross_attn3, make_quad_loaders, train_cross3_epoch, eval_cross3_loader)
from metrics import group_metrics


def _maybe_scale(sc, X_tr, X_val):
    return sc.fit_transform(X_tr), sc.transform(X_val)


def _maybe_scale_clin(X_tr, X_val):
    """Clinical 스케일링: do_scale=True이면 Age(col 0)·BMI(col 2)만 표준화. sex_enc(col 1)은 그대로."""
    sc = StandardScaler()
    X_tr_s  = X_tr.copy()
    X_val_s = X_val.copy()
    X_tr_s[:,  [0, 2]] = sc.fit_transform(X_tr[:,  [0, 2]])
    X_val_s[:, [0, 2]] = sc.transform(X_val[:, [0, 2]])
    return X_tr_s, X_val_s


def _youden_threshold(y_true, y_prob):
    """ROC 기반 Youden's J (sensitivity + specificity - 1) 최대화 임계값 반환."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def run_cross_validation(X_cv, y_cv, scale_X=True):
    """
    LR에 대해 N_FOLDS 교차검증을 수행하고 fold별 지표·ROC·최적 임계값을 반환.

    Returns: lr_cv, lr_roc_folds, lr_best_thresholds
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    lr_cv             = []
    lr_roc_folds      = []
    lr_best_thresholds = []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold Cross-Validation  [scale_X={scale_X}]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")

        X_ftr,  y_ftr  = X_cv[tr_i],  y_cv[tr_i]
        X_fval, y_fval = X_cv[val_i], y_cv[val_i]

        Xtr_s, Xval_s = _maybe_scale_clin(X_ftr, X_fval)

        lr_f     = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr_f.fit(Xtr_s, y_ftr)
        lr_fprob = lr_f.predict_proba(Xval_s)[:, 1]

        best_thresh = _youden_threshold(y_fval, lr_fprob)
        lr_fp = (lr_fprob >= best_thresh).astype(int)
        lr_best_thresholds.append(best_thresh)

        m_lr = group_metrics(y_fval, lr_fp, lr_fprob)
        lr_cv.append({"fold": fold, **m_lr})
        fpr, tpr, _ = roc_curve(y_fval, lr_fprob)
        lr_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_lr["auc"]})

        print(f"  LR — AUC: {m_lr['auc']:.4f}  AUPRC: {m_lr['auprc']:.4f}"
              f"  Brier: {m_lr['brier']:.4f}  Acc: {m_lr['acc']:.4f}  F1: {m_lr['f1']:.4f}"
              f"  (thresh={best_thresh:.3f})")

    return lr_cv, lr_roc_folds, lr_best_thresholds


def run_cross_validation_cross(X_clin_cv, X_aec_cv, y_cv,
                               scale_clin=True):
    """
    ClinAECCrossAttn에 대해 N_FOLDS 교차검증을 수행하고 fold별 지표·ROC·학습 이력·최적 임계값을 반환.
    AEC는 항상 StandardScaler로 표준화한다.

    Returns: ca_cv, ca_roc_folds, ca_histories, ca_best_epochs, ca_best_thresholds
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    ca_cv, ca_roc_folds, ca_histories, ca_best_epochs, ca_best_thresholds = [], [], [], [], []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold CV  [CrossAttn | scale_clin={scale_clin}, scale_aec=True]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")

        X_clin_tr,  X_clin_val  = X_clin_cv[tr_i],  X_clin_cv[val_i]
        X_aec_tr,   X_aec_val   = X_aec_cv[tr_i],   X_aec_cv[val_i]
        y_tr,       y_val        = y_cv[tr_i],        y_cv[val_i]

        X_clin_tr, X_clin_val = _maybe_scale_clin(X_clin_tr, X_clin_val)
        X_aec_tr,  X_aec_val  = _maybe_scale(StandardScaler(), X_aec_tr,  X_aec_val)

        # ── CrossAttn ─────────────────────────────────────────
        tr_dl, val_dl = make_dual_loaders(X_clin_tr, X_aec_tr, y_tr, X_clin_val, X_aec_val, y_val)
        model, crit, opt, sched = build_cross_attn(y_tr)

        best_auc, best_epoch = 0.0, 0
        best_state: dict = {}
        hist = {"train_loss": [], "val_loss": [], "val_auc": []}

        for ep in range(1, EPOCHS + 1):
            t_loss = train_cross_epoch(model, tr_dl, crit, opt)
            sched.step()
            v_loss, vp, vt = eval_cross_loader(model, val_dl, crit)
            v_auc = roc_auc_score(vt, vp)

            hist["train_loss"].append(t_loss)
            hist["val_loss"].append(v_loss)
            hist["val_auc"].append(v_auc)

            if v_auc > best_auc:
                best_auc, best_epoch = v_auc, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        _, ca_fprob, _ = eval_cross_loader(model, val_dl, crit)

        best_thresh = _youden_threshold(y_val, ca_fprob)
        ca_fp = (ca_fprob >= best_thresh).astype(int)
        ca_best_thresholds.append(best_thresh)

        m_ca = group_metrics(y_val, ca_fp, ca_fprob)
        ca_cv.append({"fold": fold, **m_ca})
        fpr, tpr, _ = roc_curve(y_val, ca_fprob)
        ca_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_ca["auc"]})
        ca_histories.append(hist)
        ca_best_epochs.append(best_epoch)

        print(f"  CA   — AUC: {m_ca['auc']:.4f}  AUPRC: {m_ca['auprc']:.4f}"
              f"  Brier: {m_ca['brier']:.4f}  Acc: {m_ca['acc']:.4f}  F1: {m_ca['f1']:.4f}"
              f"  (best ep={best_epoch}, thresh={best_thresh:.3f})")

    return (ca_cv, ca_roc_folds, ca_histories, ca_best_epochs, ca_best_thresholds)


def run_cross_validation_cross3(X_clin_cv, X_aec_cv, X_scan_mfr_cv,
                                 y_cv, n_manufacturers,
                                 scale_clin=True):
    """
    ClinAECScanCrossAttn에 대해 N_FOLDS 교차검증을 수행하고 fold별 지표·ROC·학습 이력·최적 임계값을 반환.
    AEC는 항상 StandardScaler로 표준화한다. ManufacturerModelName은 Embedding으로 처리하므로 스케일링하지 않는다.

    Returns: ca3_cv, ca3_roc_folds, ca3_histories, ca3_best_epochs, ca3_best_thresholds
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    ca3_cv, ca3_roc_folds, ca3_histories, ca3_best_epochs, ca3_best_thresholds = [], [], [], [], []

    print("=" * 65)
    print(f"{N_FOLDS}-Fold CV  [CrossAttn3 | scale_clin={scale_clin}, scale_aec=True]")
    print("=" * 65)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")

        X_clin_tr,  X_clin_val  = X_clin_cv[tr_i],     X_clin_cv[val_i]
        X_aec_tr,   X_aec_val   = X_aec_cv[tr_i],      X_aec_cv[val_i]
        X_mfr_tr,   X_mfr_val   = X_scan_mfr_cv[tr_i], X_scan_mfr_cv[val_i]
        y_tr,       y_val        = y_cv[tr_i],           y_cv[val_i]

        X_clin_tr, X_clin_val = _maybe_scale_clin(X_clin_tr, X_clin_val)
        X_aec_tr,  X_aec_val  = _maybe_scale(StandardScaler(), X_aec_tr,  X_aec_val)

        # ── CrossAttn3 ────────────────────────────────────────
        tr_dl, val_dl = make_quad_loaders(
            X_clin_tr, X_aec_tr, X_mfr_tr, y_tr,
            X_clin_val, X_aec_val, X_mfr_val, y_val,
        )
        model, crit, opt, sched = build_cross_attn3(y_tr, n_manufacturers)

        best_auc, best_epoch = 0.0, 0
        best_state: dict = {}
        hist = {"train_loss": [], "val_loss": [], "val_auc": []}

        for ep in range(1, EPOCHS + 1):
            t_loss = train_cross3_epoch(model, tr_dl, crit, opt)
            sched.step()
            v_loss, vp, vt = eval_cross3_loader(model, val_dl, crit)
            v_auc = roc_auc_score(vt, vp)

            hist["train_loss"].append(t_loss)
            hist["val_loss"].append(v_loss)
            hist["val_auc"].append(v_auc)

            if v_auc > best_auc:
                best_auc, best_epoch = v_auc, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        _, ca3_fprob, _ = eval_cross3_loader(model, val_dl, crit)

        best_thresh = _youden_threshold(y_val, ca3_fprob)
        ca3_fp = (ca3_fprob >= best_thresh).astype(int)
        ca3_best_thresholds.append(best_thresh)

        m_ca3 = group_metrics(y_val, ca3_fp, ca3_fprob)
        ca3_cv.append({"fold": fold, **m_ca3})
        fpr, tpr, _ = roc_curve(y_val, ca3_fprob)
        ca3_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_ca3["auc"]})
        ca3_histories.append(hist)
        ca3_best_epochs.append(best_epoch)

        print(f"  CA3  — AUC: {m_ca3['auc']:.4f}  AUPRC: {m_ca3['auprc']:.4f}"
              f"  Brier: {m_ca3['brier']:.4f}  Acc: {m_ca3['acc']:.4f}  F1: {m_ca3['f1']:.4f}"
              f"  (best ep={best_epoch}, thresh={best_thresh:.3f})")

    return (ca3_cv, ca3_roc_folds, ca3_histories, ca3_best_epochs, ca3_best_thresholds)
