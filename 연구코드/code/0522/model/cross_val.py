"""
5-Fold Stratified Cross-Validation 루프.

각 함수는 fold별 성능 지표·ROC 데이터·학습 이력을 수집하고,
최적 epoch(best val AUC)를 기록해 최종 test 평가(evaluate.py)에 전달한다.

  run_cross_validation        — M1: LR + ResNet1D
  run_cross_validation_cross  — M2/M2_2: LR + CrossAttn + ResNet1D
  run_cross_validation_cross3 — M3: LR + CrossAttn3 + ResNet1D

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
from models import (build_resnet, make_loaders, train_one_epoch, eval_loader,
                    build_cross_attn, make_dual_loaders, train_cross_epoch, eval_cross_loader,
                    build_cross_attn3, make_quad_loaders, train_cross3_epoch, eval_cross3_loader)
from metrics import group_metrics


def _maybe_scale(sc, X_tr, X_val, do_scale):
    """do_scale=True이면 sc로 fit_transform/transform, False이면 복사본 반환."""
    if do_scale:
        return sc.fit_transform(X_tr), sc.transform(X_val)
    return X_tr.copy(), X_val.copy()


def _maybe_scale_clin(X_tr, X_val, do_scale):
    """Clinical 스케일링: do_scale=True이면 Age(col 0)·BMI(col 2)만 표준화. sex_enc(col 1)은 그대로."""
    if not do_scale:
        return X_tr.copy(), X_val.copy()
    sc = StandardScaler()
    X_tr_s  = X_tr.copy()
    X_val_s = X_val.copy()
    X_tr_s[:,  [0, 2]] = sc.fit_transform(X_tr[:,  [0, 2]])
    X_val_s[:, [0, 2]] = sc.transform(X_val[:, [0, 2]])
    return X_tr_s, X_val_s


def run_cross_validation(X_cv, y_cv, scale_X=True):
    """
    LR + ResNet1D에 대해 N_FOLDS 교차검증을 수행하고 fold별 지표·ROC·학습 이력을 반환.

    ResNet1D는 각 fold에서 best AUC epoch의 모델 상태를 복원해 평가한다.
    Returns: lr_cv, rn_cv, lr_roc_folds, rn_roc_folds, rn_histories, best_epochs
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    lr_cv, rn_cv = [], []
    lr_roc_folds, rn_roc_folds = [], []
    rn_histories = []
    best_epochs  = []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold Cross-Validation  [scale_X={scale_X}]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")

        X_ftr,  y_ftr  = X_cv[tr_i],  y_cv[tr_i]
        X_fval, y_fval = X_cv[val_i], y_cv[val_i]

        # Logistic Regression
        Xtr_s, Xval_s = _maybe_scale_clin(X_ftr, X_fval, scale_X)

        lr_f     = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr_f.fit(Xtr_s, y_ftr)
        lr_fp    = lr_f.predict(Xval_s)
        lr_fprob = lr_f.predict_proba(Xval_s)[:, 1]

        m_lr = group_metrics(y_fval, lr_fp, lr_fprob)
        lr_cv.append({"fold": fold, **m_lr})
        fpr, tpr, _ = roc_curve(y_fval, lr_fprob)
        lr_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_lr["auc"]})

        print(f"  LR   — AUC: {m_lr['auc']:.4f}  AUPRC: {m_lr['auprc']:.4f}"
              f"  Brier: {m_lr['brier']:.4f}  Acc: {m_lr['acc']:.4f}  F1: {m_lr['f1']:.4f}")

        # ResNet1D
        Xtr_rn, Xval_rn = _maybe_scale_clin(X_ftr, X_fval, scale_X)

        tr_dl, val_dl = make_loaders(Xtr_rn, y_ftr, Xval_rn, y_fval)
        model, crit, opt, sched = build_resnet(y_ftr)

        best_auc, best_epoch = 0.0, 0
        best_state: dict = {}
        hist = {"train_loss": [], "val_loss": [], "val_auc": []}

        for ep in range(1, EPOCHS + 1):
            t_loss = train_one_epoch(model, tr_dl, crit, opt)
            sched.step()
            v_loss, vp, vt = eval_loader(model, val_dl, crit)
            v_auc = roc_auc_score(vt, vp)

            hist["train_loss"].append(t_loss)
            hist["val_loss"].append(v_loss)
            hist["val_auc"].append(v_auc)

            # val AUC 가 개선될 때마다 가중치 스냅샷 저장 (early stopping 대신 best 복원)
            if v_auc > best_auc:
                best_auc, best_epoch = v_auc, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # best epoch 시점의 가중치로 복원 후 최종 fold 평가
        model.load_state_dict(best_state)
        _, rn_fprob, _ = eval_loader(model, val_dl, crit)
        rn_fp = (rn_fprob >= 0.5).astype(int)

        m_rn = group_metrics(y_fval, rn_fp, rn_fprob)
        rn_cv.append({"fold": fold, **m_rn})
        fpr, tpr, _ = roc_curve(y_fval, rn_fprob)
        rn_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_rn["auc"]})
        rn_histories.append(hist)
        best_epochs.append(best_epoch)

        print(f"  RN1D — AUC: {m_rn['auc']:.4f}  AUPRC: {m_rn['auprc']:.4f}"
              f"  Brier: {m_rn['brier']:.4f}  Acc: {m_rn['acc']:.4f}  F1: {m_rn['f1']:.4f}"
              f"  (best ep={best_epoch})")

    return lr_cv, rn_cv, lr_roc_folds, rn_roc_folds, rn_histories, best_epochs


def run_cross_validation_cross(X_clin_cv, X_aec_cv, y_cv,
                               scale_clin=True, scale_aec=True):
    """
    LR + ClinAECCrossAttn + ResNet1D에 대해 N_FOLDS 교차검증을 수행하고 fold별 지표·ROC·학습 이력을 반환.

    LR/ResNet1D는 Clinic+AEC hstack 입력, CrossAttn은 별도 인코더를 사용한다.
    Returns: lr_cv, ca_cv, rn_cv, lr_roc_folds, ca_roc_folds, rn_roc_folds,
             ca_histories, rn_histories, ca_best_epochs, rn_best_epochs
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    lr_cv,  lr_roc_folds  = [], []
    ca_cv, ca_roc_folds, ca_histories, ca_best_epochs = [], [], [], []
    rn_cv, rn_roc_folds, rn_histories, rn_best_epochs = [], [], [], []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold CV  [CrossAttn | scale_clin={scale_clin}, scale_aec={scale_aec}]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")

        X_clin_tr,  X_clin_val  = X_clin_cv[tr_i],  X_clin_cv[val_i]
        X_aec_tr,   X_aec_val   = X_aec_cv[tr_i],   X_aec_cv[val_i]
        y_tr,       y_val        = y_cv[tr_i],        y_cv[val_i]

        X_clin_tr, X_clin_val = _maybe_scale_clin(X_clin_tr, X_clin_val, scale_clin)
        X_aec_tr,  X_aec_val  = _maybe_scale(StandardScaler(), X_aec_tr,  X_aec_val,  scale_aec)

        # ── Logistic Regression (clinic + AEC 연결) ───────────
        X_lr_tr  = np.hstack([X_clin_tr,  X_aec_tr])
        X_lr_val = np.hstack([X_clin_val, X_aec_val])

        lr_f     = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr_f.fit(X_lr_tr, y_tr)
        lr_fp    = lr_f.predict(X_lr_val)
        lr_fprob = lr_f.predict_proba(X_lr_val)[:, 1]

        m_lr = group_metrics(y_val, lr_fp, lr_fprob)
        lr_cv.append({"fold": fold, **m_lr})
        fpr, tpr, _ = roc_curve(y_val, lr_fprob)
        lr_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_lr["auc"]})

        print(f"  LR   — AUC: {m_lr['auc']:.4f}  AUPRC: {m_lr['auprc']:.4f}"
              f"  Brier: {m_lr['brier']:.4f}  Acc: {m_lr['acc']:.4f}  F1: {m_lr['f1']:.4f}")

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
        ca_fp = (ca_fprob >= 0.5).astype(int)

        m_ca = group_metrics(y_val, ca_fp, ca_fprob)
        ca_cv.append({"fold": fold, **m_ca})
        fpr, tpr, _ = roc_curve(y_val, ca_fprob)
        ca_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_ca["auc"]})
        ca_histories.append(hist)
        ca_best_epochs.append(best_epoch)

        print(f"  CA   — AUC: {m_ca['auc']:.4f}  AUPRC: {m_ca['auprc']:.4f}"
              f"  Brier: {m_ca['brier']:.4f}  Acc: {m_ca['acc']:.4f}  F1: {m_ca['f1']:.4f}"
              f"  (best ep={best_epoch})")

        # ── ResNet1D (clinic + AEC hstack, 이미 스케일됨) ─────────
        tr_dl_rn, val_dl_rn = make_loaders(X_lr_tr, y_tr, X_lr_val, y_val)
        model_rn, crit_rn, opt_rn, sched_rn = build_resnet(y_tr)

        best_auc_rn, best_epoch_rn = 0.0, 0
        best_state_rn: dict = {}
        hist_rn = {"train_loss": [], "val_loss": [], "val_auc": []}

        for ep in range(1, EPOCHS + 1):
            t_loss_rn = train_one_epoch(model_rn, tr_dl_rn, crit_rn, opt_rn)
            sched_rn.step()
            v_loss_rn, vp_rn, vt_rn = eval_loader(model_rn, val_dl_rn, crit_rn)
            v_auc_rn = roc_auc_score(vt_rn, vp_rn)

            hist_rn["train_loss"].append(t_loss_rn)
            hist_rn["val_loss"].append(v_loss_rn)
            hist_rn["val_auc"].append(v_auc_rn)

            if v_auc_rn > best_auc_rn:
                best_auc_rn, best_epoch_rn = v_auc_rn, ep
                best_state_rn = {k: v.cpu().clone() for k, v in model_rn.state_dict().items()}

        model_rn.load_state_dict(best_state_rn)
        _, rn_fprob, _ = eval_loader(model_rn, val_dl_rn, crit_rn)
        rn_fp_fold = (rn_fprob >= 0.5).astype(int)

        m_rn = group_metrics(y_val, rn_fp_fold, rn_fprob)
        rn_cv.append({"fold": fold, **m_rn})
        fpr, tpr, _ = roc_curve(y_val, rn_fprob)
        rn_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_rn["auc"]})
        rn_histories.append(hist_rn)
        rn_best_epochs.append(best_epoch_rn)

        print(f"  RN1D — AUC: {m_rn['auc']:.4f}  AUPRC: {m_rn['auprc']:.4f}"
              f"  Brier: {m_rn['brier']:.4f}  Acc: {m_rn['acc']:.4f}  F1: {m_rn['f1']:.4f}"
              f"  (best ep={best_epoch_rn})")

    return (lr_cv, ca_cv, rn_cv,
            lr_roc_folds, ca_roc_folds, rn_roc_folds,
            ca_histories, rn_histories,
            ca_best_epochs, rn_best_epochs)


def run_cross_validation_cross3(X_clin_cv, X_aec_cv, X_scan_mfr_cv,
                                 y_cv, n_manufacturers,
                                 scale_clin=True, scale_aec=True):
    """
    LR + ClinAECScanCrossAttn + ResNet1D에 대해 N_FOLDS 교차검증을 수행하고 fold별 지표·ROC·학습 이력을 반환.

    ManufacturerModelName은 Embedding으로 처리하므로 스케일링하지 않는다.
    ResNet1D는 LR과 동일한 hstack 입력을 사용한다.
    Returns: lr_cv, ca3_cv, rn_cv, lr_roc_folds, ca3_roc_folds, rn_roc_folds,
             ca3_histories, rn_histories, ca3_best_epochs, rn_best_epochs
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lr_cv,  lr_roc_folds  = [], []
    ca3_cv, ca3_roc_folds, ca3_histories, ca3_best_epochs = [], [], [], []
    rn_cv,  rn_roc_folds,  rn_histories,  rn_best_epochs  = [], [], [], []

    print("=" * 65)
    print(f"{N_FOLDS}-Fold CV  [CrossAttn3 | scale_clin={scale_clin}, scale_aec={scale_aec}]")
    print("=" * 65)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")

        X_clin_tr,  X_clin_val  = X_clin_cv[tr_i],     X_clin_cv[val_i]
        X_aec_tr,   X_aec_val   = X_aec_cv[tr_i],      X_aec_cv[val_i]
        X_mfr_tr,   X_mfr_val   = X_scan_mfr_cv[tr_i], X_scan_mfr_cv[val_i]
        y_tr,       y_val        = y_cv[tr_i],           y_cv[val_i]

        X_clin_tr, X_clin_val = _maybe_scale_clin(X_clin_tr, X_clin_val, scale_clin)
        X_aec_tr,  X_aec_val  = _maybe_scale(StandardScaler(), X_aec_tr,  X_aec_val,  scale_aec)

        # ── Logistic Regression (clinic + mfr + AEC 연결) ─────
        X_lr_tr  = np.hstack([X_clin_tr,  X_mfr_tr.reshape(-1,1).astype(float),  X_aec_tr])
        X_lr_val = np.hstack([X_clin_val, X_mfr_val.reshape(-1,1).astype(float), X_aec_val])

        lr_f     = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr_f.fit(X_lr_tr, y_tr)
        lr_fp    = lr_f.predict(X_lr_val)
        lr_fprob = lr_f.predict_proba(X_lr_val)[:, 1]

        m_lr = group_metrics(y_val, lr_fp, lr_fprob)
        lr_cv.append({"fold": fold, **m_lr})
        fpr, tpr, _ = roc_curve(y_val, lr_fprob)
        lr_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_lr["auc"]})

        print(f"  LR   — AUC: {m_lr['auc']:.4f}  AUPRC: {m_lr['auprc']:.4f}"
              f"  Brier: {m_lr['brier']:.4f}  Acc: {m_lr['acc']:.4f}  F1: {m_lr['f1']:.4f}")

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
        ca3_fp = (ca3_fprob >= 0.5).astype(int)

        m_ca3 = group_metrics(y_val, ca3_fp, ca3_fprob)
        ca3_cv.append({"fold": fold, **m_ca3})
        fpr, tpr, _ = roc_curve(y_val, ca3_fprob)
        ca3_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_ca3["auc"]})
        ca3_histories.append(hist)
        ca3_best_epochs.append(best_epoch)

        print(f"  CA3  — AUC: {m_ca3['auc']:.4f}  AUPRC: {m_ca3['auprc']:.4f}"
              f"  Brier: {m_ca3['brier']:.4f}  Acc: {m_ca3['acc']:.4f}  F1: {m_ca3['f1']:.4f}"
              f"  (best ep={best_epoch})")

        # ── ResNet1D (clinic + scan + AEC hstack, 이미 스케일됨) ──
        tr_dl_rn, val_dl_rn = make_loaders(X_lr_tr, y_tr, X_lr_val, y_val)
        model_rn, crit_rn, opt_rn, sched_rn = build_resnet(y_tr)

        best_auc_rn, best_epoch_rn = 0.0, 0
        best_state_rn: dict = {}
        hist_rn = {"train_loss": [], "val_loss": [], "val_auc": []}

        for ep in range(1, EPOCHS + 1):
            t_loss_rn = train_one_epoch(model_rn, tr_dl_rn, crit_rn, opt_rn)
            sched_rn.step()
            v_loss_rn, vp_rn, vt_rn = eval_loader(model_rn, val_dl_rn, crit_rn)
            v_auc_rn = roc_auc_score(vt_rn, vp_rn)

            hist_rn["train_loss"].append(t_loss_rn)
            hist_rn["val_loss"].append(v_loss_rn)
            hist_rn["val_auc"].append(v_auc_rn)

            if v_auc_rn > best_auc_rn:
                best_auc_rn, best_epoch_rn = v_auc_rn, ep
                best_state_rn = {k: v.cpu().clone() for k, v in model_rn.state_dict().items()}

        model_rn.load_state_dict(best_state_rn)
        _, rn_fprob, _ = eval_loader(model_rn, val_dl_rn, crit_rn)
        rn_fp_fold = (rn_fprob >= 0.5).astype(int)

        m_rn = group_metrics(y_val, rn_fp_fold, rn_fprob)
        rn_cv.append({"fold": fold, **m_rn})
        fpr, tpr, _ = roc_curve(y_val, rn_fprob)
        rn_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_rn["auc"]})
        rn_histories.append(hist_rn)
        rn_best_epochs.append(best_epoch_rn)

        print(f"  RN1D — AUC: {m_rn['auc']:.4f}  AUPRC: {m_rn['auprc']:.4f}"
              f"  Brier: {m_rn['brier']:.4f}  Acc: {m_rn['acc']:.4f}  F1: {m_rn['f1']:.4f}"
              f"  (best ep={best_epoch_rn})")

    return (lr_cv, ca3_cv, rn_cv,
            lr_roc_folds, ca3_roc_folds, rn_roc_folds,
            ca3_histories, rn_histories,
            ca3_best_epochs, rn_best_epochs)
