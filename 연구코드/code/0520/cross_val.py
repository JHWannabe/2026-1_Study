"""
0520 Cross-Validation 루프.

0515 cross_val.py의 모든 함수를 포함하며, 아래를 추가한다:
  run_cross_validation_regression — M6: SMI 회귀 CV
      (OOF 예측값 + fold별 MAE/RMSE/R² 반환)
  run_cross_validation_cross_calib — M2 + OOF logit 수집 (캘리브레이션용)

스케일링 정책 (0515 동일):
  Clinical: Age·BMI만 표준화, sex_enc 제외
  AEC     : 전 컬럼 표준화
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_error, r2_score

from config import N_FOLDS, SEED, EPOCHS, SMI_THRESH_M, SMI_THRESH_F
from models import (
    build_resnet, make_loaders, train_one_epoch, eval_loader,
    build_cross_attn, make_dual_loaders, train_cross_epoch,
    eval_cross_loader, eval_cross_loader_logits,
    build_regression, make_regression_loaders,
    train_regression_epoch, eval_regression_loader,
)
from metrics import group_metrics


def _maybe_scale(sc, X_tr, X_val, do_scale):
    if do_scale:
        return sc.fit_transform(X_tr), sc.transform(X_val)
    return X_tr.copy(), X_val.copy()


def _maybe_scale_clin(X_tr, X_val, do_scale):
    if not do_scale:
        return X_tr.copy(), X_val.copy()
    sc = StandardScaler()
    X_tr_s, X_val_s = X_tr.copy(), X_val.copy()
    X_tr_s[:,  [0, 2]] = sc.fit_transform(X_tr[:,  [0, 2]])
    X_val_s[:, [0, 2]] = sc.transform(X_val[:, [0, 2]])
    return X_tr_s, X_val_s


# ── 0515 원본 함수 ─────────────────────────────────────────────

def run_cross_validation(X_cv, y_cv, scale_X=True):
    """M1: LR + ResNet1D 5-Fold CV."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lr_cv, rn_cv = [], []
    lr_roc_folds, rn_roc_folds = [], []
    rn_histories, best_epochs = [], []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold Cross-Validation  [scale_X={scale_X}]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")
        X_ftr, y_ftr   = X_cv[tr_i],  y_cv[tr_i]
        X_fval, y_fval = X_cv[val_i], y_cv[val_i]

        Xtr_s, Xval_s = _maybe_scale_clin(X_ftr, X_fval, scale_X)
        lr_f = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr_f.fit(Xtr_s, y_ftr)
        lr_fp    = lr_f.predict(Xval_s)
        lr_fprob = lr_f.predict_proba(Xval_s)[:, 1]
        m_lr = group_metrics(y_fval, lr_fp, lr_fprob)
        lr_cv.append({"fold": fold, **m_lr})
        fpr, tpr, _ = roc_curve(y_fval, lr_fprob)
        lr_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_lr["auc"]})
        print(f"  LR   — AUC: {m_lr['auc']:.4f}  Acc: {m_lr['acc']:.4f}  F1: {m_lr['f1']:.4f}")

        Xtr_rn, Xval_rn = _maybe_scale_clin(X_ftr, X_fval, scale_X)
        tr_dl, val_dl = make_loaders(Xtr_rn, y_ftr, Xval_rn, y_fval)
        model, crit, opt, sched = build_resnet(y_ftr)
        best_auc, best_epoch, best_state = 0.0, 0, {}
        hist = {"train_loss": [], "val_loss": [], "val_auc": []}
        for ep in range(1, EPOCHS + 1):
            t_loss = train_one_epoch(model, tr_dl, crit, opt); sched.step()
            v_loss, vp, vt = eval_loader(model, val_dl, crit)
            v_auc = roc_auc_score(vt, vp)
            hist["train_loss"].append(t_loss)
            hist["val_loss"].append(v_loss)
            hist["val_auc"].append(v_auc)
            if v_auc > best_auc:
                best_auc, best_epoch = v_auc, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        _, rn_fprob, _ = eval_loader(model, val_dl, crit)
        rn_fp = (rn_fprob >= 0.5).astype(int)
        m_rn = group_metrics(y_fval, rn_fp, rn_fprob)
        rn_cv.append({"fold": fold, **m_rn})
        fpr, tpr, _ = roc_curve(y_fval, rn_fprob)
        rn_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_rn["auc"]})
        rn_histories.append(hist)
        best_epochs.append(best_epoch)
        print(f"  RN1D — AUC: {m_rn['auc']:.4f}  Acc: {m_rn['acc']:.4f}  F1: {m_rn['f1']:.4f}"
              f"  (best ep={best_epoch})")

    return lr_cv, rn_cv, lr_roc_folds, rn_roc_folds, rn_histories, best_epochs


def run_cross_validation_cross(X_clin_cv, X_aec_cv, y_cv,
                               scale_clin=True, scale_aec=True):
    """M2/M2_2: LR + CrossAttn + ResNet1D 5-Fold CV."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lr_cv, lr_roc_folds = [], []
    ca_cv, ca_roc_folds, ca_histories, ca_best_epochs = [], [], [], []
    rn_cv, rn_roc_folds, rn_histories, rn_best_epochs = [], [], [], []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold CV  [CrossAttn | scale_clin={scale_clin}, scale_aec={scale_aec}]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")
        X_clin_tr, X_clin_val = X_clin_cv[tr_i], X_clin_cv[val_i]
        X_aec_tr,  X_aec_val  = X_aec_cv[tr_i],  X_aec_cv[val_i]
        y_tr,      y_val       = y_cv[tr_i],       y_cv[val_i]

        X_clin_tr, X_clin_val = _maybe_scale_clin(X_clin_tr, X_clin_val, scale_clin)
        X_aec_tr,  X_aec_val  = _maybe_scale(StandardScaler(), X_aec_tr,  X_aec_val,  scale_aec)

        X_lr_tr  = np.hstack([X_clin_tr,  X_aec_tr])
        X_lr_val = np.hstack([X_clin_val, X_aec_val])
        lr_f = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr_f.fit(X_lr_tr, y_tr)
        lr_fp, lr_fprob = lr_f.predict(X_lr_val), lr_f.predict_proba(X_lr_val)[:, 1]
        m_lr = group_metrics(y_val, lr_fp, lr_fprob)
        lr_cv.append({"fold": fold, **m_lr})
        fpr, tpr, _ = roc_curve(y_val, lr_fprob)
        lr_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_lr["auc"]})
        print(f"  LR   — AUC: {m_lr['auc']:.4f}  Acc: {m_lr['acc']:.4f}  F1: {m_lr['f1']:.4f}")

        tr_dl, val_dl = make_dual_loaders(X_clin_tr, X_aec_tr, y_tr, X_clin_val, X_aec_val, y_val)
        model, crit, opt, sched = build_cross_attn(y_tr)
        best_auc, best_epoch, best_state = 0.0, 0, {}
        hist = {"train_loss": [], "val_loss": [], "val_auc": []}
        for ep in range(1, EPOCHS + 1):
            t_loss = train_cross_epoch(model, tr_dl, crit, opt); sched.step()
            v_loss, vp, vt = eval_cross_loader(model, val_dl, crit)
            v_auc = roc_auc_score(vt, vp)
            hist["train_loss"].append(t_loss); hist["val_loss"].append(v_loss); hist["val_auc"].append(v_auc)
            if v_auc > best_auc:
                best_auc, best_epoch = v_auc, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        _, ca_fprob, _ = eval_cross_loader(model, val_dl, crit)
        ca_fp = (ca_fprob >= 0.5).astype(int)
        m_ca = group_metrics(y_val, ca_fp, ca_fprob)
        ca_cv.append({"fold": fold, **m_ca}); ca_histories.append(hist); ca_best_epochs.append(best_epoch)
        fpr, tpr, _ = roc_curve(y_val, ca_fprob)
        ca_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_ca["auc"]})
        print(f"  CA   — AUC: {m_ca['auc']:.4f}  Acc: {m_ca['acc']:.4f}  (best ep={best_epoch})")

        tr_dl_rn, val_dl_rn = make_loaders(X_lr_tr, y_tr, X_lr_val, y_val)
        model_rn, crit_rn, opt_rn, sched_rn = build_resnet(y_tr)
        best_auc_rn, best_epoch_rn, best_state_rn = 0.0, 0, {}
        hist_rn = {"train_loss": [], "val_loss": [], "val_auc": []}
        for ep in range(1, EPOCHS + 1):
            t_loss_rn = train_one_epoch(model_rn, tr_dl_rn, crit_rn, opt_rn); sched_rn.step()
            v_loss_rn, vp_rn, vt_rn = eval_loader(model_rn, val_dl_rn, crit_rn)
            v_auc_rn = roc_auc_score(vt_rn, vp_rn)
            hist_rn["train_loss"].append(t_loss_rn); hist_rn["val_loss"].append(v_loss_rn); hist_rn["val_auc"].append(v_auc_rn)
            if v_auc_rn > best_auc_rn:
                best_auc_rn, best_epoch_rn = v_auc_rn, ep
                best_state_rn = {k: v.cpu().clone() for k, v in model_rn.state_dict().items()}
        model_rn.load_state_dict(best_state_rn)
        _, rn_fprob, _ = eval_loader(model_rn, val_dl_rn, crit_rn)
        m_rn = group_metrics(y_val, (rn_fprob >= 0.5).astype(int), rn_fprob)
        rn_cv.append({"fold": fold, **m_rn}); rn_histories.append(hist_rn); rn_best_epochs.append(best_epoch_rn)
        fpr, tpr, _ = roc_curve(y_val, rn_fprob)
        rn_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_rn["auc"]})

    return (lr_cv, ca_cv, rn_cv,
            lr_roc_folds, ca_roc_folds, rn_roc_folds,
            ca_histories, rn_histories,
            ca_best_epochs, rn_best_epochs)


# ── 0520 새 함수 ──────────────────────────────────────────────

def run_cross_validation_cross_calib(X_clin_cv, X_aec_cv, y_cv,
                                     scale_clin=True, scale_aec=True):
    """
    M2 CrossAttn CV + OOF logit 수집 (캘리브레이션 T 탐색용).

    run_cross_validation_cross와 동일하나, 각 fold의 validation set에서
    CrossAttn raw logit을 수집해 반환한다.

    Returns: (동일 반환값) + oof_logits: (N_cv,) array, oof_labels: (N_cv,) array
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lr_cv, lr_roc_folds = [], []
    ca_cv, ca_roc_folds, ca_histories, ca_best_epochs = [], [], [], []
    rn_cv, rn_roc_folds, rn_histories, rn_best_epochs = [], [], [], []
    oof_logits_list, oof_labels_list = [], []

    print("=" * 60)
    print(f"{N_FOLDS}-Fold CV [CrossAttn + OOF logit | calib mode]")
    print("=" * 60)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")
        X_clin_tr, X_clin_val = X_clin_cv[tr_i], X_clin_cv[val_i]
        X_aec_tr,  X_aec_val  = X_aec_cv[tr_i],  X_aec_cv[val_i]
        y_tr,      y_val       = y_cv[tr_i],       y_cv[val_i]

        X_clin_tr, X_clin_val = _maybe_scale_clin(X_clin_tr, X_clin_val, scale_clin)
        X_aec_tr,  X_aec_val  = _maybe_scale(StandardScaler(), X_aec_tr, X_aec_val, scale_aec)

        X_lr_tr  = np.hstack([X_clin_tr, X_aec_tr])
        X_lr_val = np.hstack([X_clin_val, X_aec_val])
        lr_f = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr_f.fit(X_lr_tr, y_tr)
        lr_fp, lr_fprob = lr_f.predict(X_lr_val), lr_f.predict_proba(X_lr_val)[:, 1]
        m_lr = group_metrics(y_val, lr_fp, lr_fprob)
        lr_cv.append({"fold": fold, **m_lr})
        fpr, tpr, _ = roc_curve(y_val, lr_fprob)
        lr_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_lr["auc"]})

        tr_dl, val_dl = make_dual_loaders(X_clin_tr, X_aec_tr, y_tr, X_clin_val, X_aec_val, y_val)
        model, crit, opt, sched = build_cross_attn(y_tr)
        best_auc, best_epoch, best_state = 0.0, 0, {}
        hist = {"train_loss": [], "val_loss": [], "val_auc": []}
        for ep in range(1, EPOCHS + 1):
            t_loss = train_cross_epoch(model, tr_dl, crit, opt); sched.step()
            v_loss, vp, vt = eval_cross_loader(model, val_dl, crit)
            v_auc = roc_auc_score(vt, vp)
            hist["train_loss"].append(t_loss); hist["val_loss"].append(v_loss); hist["val_auc"].append(v_auc)
            if v_auc > best_auc:
                best_auc, best_epoch = v_auc, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

        # OOF logit 수집 (캘리브레이션 T 탐색에 사용)
        _, val_logits, ca_fprob, _ = eval_cross_loader_logits(model, val_dl, crit)
        oof_logits_list.append(val_logits)
        oof_labels_list.append(y_val)

        ca_fp = (ca_fprob >= 0.5).astype(int)
        m_ca = group_metrics(y_val, ca_fp, ca_fprob)
        ca_cv.append({"fold": fold, **m_ca}); ca_histories.append(hist); ca_best_epochs.append(best_epoch)
        fpr, tpr, _ = roc_curve(y_val, ca_fprob)
        ca_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_ca["auc"]})
        print(f"  CA   — AUC: {m_ca['auc']:.4f}  (best ep={best_epoch})")

        tr_dl_rn, val_dl_rn = make_loaders(X_lr_tr, y_tr, X_lr_val, y_val)
        model_rn, crit_rn, opt_rn, sched_rn = build_resnet(y_tr)
        best_auc_rn, best_epoch_rn, best_state_rn = 0.0, 0, {}
        hist_rn = {"train_loss": [], "val_loss": [], "val_auc": []}
        for ep in range(1, EPOCHS + 1):
            train_one_epoch(model_rn, tr_dl_rn, crit_rn, opt_rn); sched_rn.step()
            v_loss_rn, vp_rn, vt_rn = eval_loader(model_rn, val_dl_rn, crit_rn)
            v_auc_rn = roc_auc_score(vt_rn, vp_rn)
            hist_rn["train_loss"].append(0.0); hist_rn["val_loss"].append(v_loss_rn); hist_rn["val_auc"].append(v_auc_rn)
            if v_auc_rn > best_auc_rn:
                best_auc_rn, best_epoch_rn = v_auc_rn, ep
                best_state_rn = {k: v.cpu().clone() for k, v in model_rn.state_dict().items()}
        model_rn.load_state_dict(best_state_rn)
        _, rn_fprob, _ = eval_loader(model_rn, val_dl_rn, crit_rn)
        m_rn = group_metrics(y_val, (rn_fprob >= 0.5).astype(int), rn_fprob)
        rn_cv.append({"fold": fold, **m_rn}); rn_histories.append(hist_rn); rn_best_epochs.append(best_epoch_rn)
        fpr, tpr, _ = roc_curve(y_val, rn_fprob)
        rn_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_rn["auc"]})

    oof_logits = np.concatenate(oof_logits_list)
    oof_labels = np.concatenate(oof_labels_list)

    return (lr_cv, ca_cv, rn_cv,
            lr_roc_folds, ca_roc_folds, rn_roc_folds,
            ca_histories, rn_histories,
            ca_best_epochs, rn_best_epochs,
            oof_logits, oof_labels)


def run_cross_validation_regression(X_clin_cv, X_aec_cv, y_smi_cv, sex_cv,
                                    scale_clin=True, scale_aec=True):
    """
    M6: SMIRegressionNet 5-Fold CV.

    각 fold에서:
      - val 예측 SMI → sex-specific threshold → 이진 분류 지표 계산
      - 회귀 지표: MAE, RMSE, R²

    Returns
    -------
    reg_cv        : list of fold dict (mae, rmse, r2, auc, f1, ...)
    reg_histories : list of fold training history
    best_epochs   : list of fold best epoch (min val MAE)
    oof_pred_smi  : (N_cv,) float — OOF SMI 예측값
    oof_true_smi  : (N_cv,) float — OOF 실제 SMI
    oof_true_bin  : (N_cv,) int   — OOF 이진 레이블
    oof_sex       : (N_cv,) str   — OOF 성별
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # stratify by binary label
    from sklearn.metrics import roc_auc_score as _auc
    y_bin_cv = np.array([1 if (sex_cv[i] == "M" and y_smi_cv[i] <= 40.96)
                             or (sex_cv[i] == "F" and y_smi_cv[i] <= 30.6)
                         else 0
                         for i in range(len(y_smi_cv))], dtype=np.int64)

    reg_cv, reg_histories, best_epochs = [], [], []
    oof_pred_list, oof_true_list, oof_bin_list, oof_sex_list = [], [], [], []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold CV  [SMI Regression | scale_clin={scale_clin}]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_bin_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")
        X_clin_tr, X_clin_val = X_clin_cv[tr_i], X_clin_cv[val_i]
        X_aec_tr,  X_aec_val  = X_aec_cv[tr_i],  X_aec_cv[val_i]
        y_smi_tr,  y_smi_val  = y_smi_cv[tr_i],  y_smi_cv[val_i]
        y_bin_val = y_bin_cv[val_i]
        sex_val   = sex_cv[val_i]

        X_clin_tr, X_clin_val = _maybe_scale_clin(X_clin_tr, X_clin_val, scale_clin)
        X_aec_tr,  X_aec_val  = _maybe_scale(StandardScaler(), X_aec_tr, X_aec_val, scale_aec)

        tr_dl, val_dl = make_regression_loaders(X_clin_tr, X_aec_tr, y_smi_tr,
                                                X_clin_val, X_aec_val, y_smi_val)
        model, crit, opt, sched = build_regression(y_smi_tr)

        best_mae, best_epoch, best_state = float("inf"), 0, {}
        hist = {"train_loss": [], "val_mae": []}

        for ep in range(1, EPOCHS + 1):
            t_loss = train_regression_epoch(model, tr_dl, crit, opt); sched.step()
            v_loss, v_preds, v_trues = eval_regression_loader(model, val_dl, crit)
            v_mae = mean_absolute_error(v_trues, v_preds)
            hist["train_loss"].append(t_loss); hist["val_mae"].append(v_mae)
            if v_mae < best_mae:
                best_mae, best_epoch = v_mae, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        _, final_preds, final_trues = eval_regression_loader(model, val_dl, crit)

        # SMI → 이진 분류 (sex-specific threshold)
        smi_thresh = np.array([SMI_THRESH_M if s == "M" else SMI_THRESH_F for s in sex_val])
        y_bin_pred  = (final_preds <= smi_thresh).astype(int)

        mae  = mean_absolute_error(final_trues, final_preds)
        rmse = float(np.sqrt(np.mean((final_trues - final_preds) ** 2)))
        r2   = r2_score(final_trues, final_preds)

        # 음성 확률 = (threshold - pred_smi) 으로 간단히 근사
        # threshold 이하면 sarcopenia=1이므로, 예측 SMI가 threshold보다 낮을수록 sarcopenia 확률↑
        # prob_sarco = sigmoid(-(pred_smi - threshold) / scale)
        from scipy.special import expit
        smi_prob = expit(-(final_preds - smi_thresh) / 5.0).astype(np.float32)

        from sklearn.metrics import roc_auc_score as rauc, f1_score, accuracy_score
        auc = rauc(y_bin_val, smi_prob) if len(np.unique(y_bin_val)) > 1 else float("nan")
        f1  = f1_score(y_bin_val, y_bin_pred, zero_division=0)
        acc = accuracy_score(y_bin_val, y_bin_pred)

        reg_cv.append({"fold": fold, "mae": mae, "rmse": rmse, "r2": r2,
                       "auc": auc, "f1": f1, "acc": acc})
        reg_histories.append(hist)
        best_epochs.append(best_epoch)

        oof_pred_list.append(final_preds)
        oof_true_list.append(final_trues)
        oof_bin_list.append(y_bin_val)
        oof_sex_list.append(sex_val)

        print(f"  REG  — MAE: {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.3f}"
              f"  | AUC: {auc:.4f}  F1: {f1:.4f}  (best ep={best_epoch})")

    return (reg_cv, reg_histories, best_epochs,
            np.concatenate(oof_pred_list), np.concatenate(oof_true_list),
            np.concatenate(oof_bin_list), np.concatenate(oof_sex_list))
