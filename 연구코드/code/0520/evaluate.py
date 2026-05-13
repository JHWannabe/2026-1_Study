"""
0520 최종 Test Set 평가.

0515 evaluate.py의 모든 함수를 포함하며, 아래를 추가한다:
  evaluate_test_regression   — M6: SMI 회귀 최종 테스트
  evaluate_test_cross_calib  — M2 + Temperature Scaling 평가
  evaluate_ensemble          — M7: 앙상블 (avg/weighted/stacking) 평가
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    average_precision_score, brier_score_loss,
    mean_absolute_error, r2_score,
)

from config import SEED, SMI_THRESH_M, SMI_THRESH_F
from models import (
    build_resnet, make_loaders, train_one_epoch, eval_loader,
    build_cross_attn, make_dual_loaders, train_cross_epoch,
    eval_cross_loader, eval_cross_loader_logits,
    build_regression, make_regression_loaders,
    train_regression_epoch, eval_regression_loader,
)
from calibration import (
    find_optimal_temperature, calibrate_probs, calibration_summary,
)


def _print_by_sex(y_true, y_pred, y_prob, sex):
    from sklearn.metrics import classification_report
    for s in ["M", "F"]:
        mask = sex == s
        if mask.sum() == 0: continue
        yt, yp, ypr = y_true[mask], y_pred[mask], y_prob[mask]
        has_both = len(np.unique(yt)) > 1
        auc   = roc_auc_score(yt, ypr) if has_both else float("nan")
        auprc = average_precision_score(yt, ypr) if has_both else float("nan")
        brier = brier_score_loss(yt, ypr) if has_both else float("nan")
        print(f"  [{s}] n={mask.sum()}  AUC: {auc:.4f}  AUPRC: {auprc:.4f}"
              f"  Brier: {brier:.4f}  Acc: {accuracy_score(yt, yp):.4f}"
              f"  F1: {f1_score(yt, yp, zero_division=0):.4f}")


def _scale_or_copy(X_tr, X_te, do_scale):
    if do_scale:
        sc = StandardScaler()
        return sc.fit_transform(X_tr), sc.transform(X_te)
    return X_tr.copy(), X_te.copy()


def _scale_clin_te(X_cv, X_te, do_scale):
    if not do_scale:
        return X_cv.copy(), X_te.copy()
    sc = StandardScaler()
    X_cv_s, X_te_s = X_cv.copy(), X_te.copy()
    X_cv_s[:, [0, 2]] = sc.fit_transform(X_cv[:, [0, 2]])
    X_te_s[:, [0, 2]] = sc.transform(X_te[:, [0, 2]])
    return X_cv_s, X_te_s


# ── 0515 원본 함수 ─────────────────────────────────────────────

def evaluate_test(X_cv, y_cv, X_te, y_te, sex_te, med_epoch, scale_X=True):
    """M1: LR + ResNet1D 최종 테스트."""
    print(f"\n{'='*55}\nFinal Test Evaluation (scale_X={scale_X})\n{'='*55}")
    X_cv_s, X_te_s = _scale_clin_te(X_cv, X_te, scale_X)
    lr_final = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr_final.fit(X_cv_s, y_cv)
    lr_pred = lr_final.predict(X_te_s)
    lr_prob = lr_final.predict_proba(X_te_s)[:, 1]
    print(f"LR  AUC: {roc_auc_score(y_te, lr_prob):.4f}")

    X_cv_rn, X_te_rn = _scale_clin_te(X_cv, X_te, scale_X)
    tr_dl, te_dl = make_loaders(X_cv_rn, y_cv, X_te_rn, y_te)
    model_f, crit_f, opt_f, sched_f = build_resnet(y_cv)
    for _ in range(1, med_epoch + 1):
        train_one_epoch(model_f, tr_dl, crit_f, opt_f); sched_f.step()
    _, rn_prob_te, rn_true_te = eval_loader(model_f, te_dl, crit_f)
    rn_pred_te = (rn_prob_te >= 0.5).astype(int)
    print(f"RN  AUC: {roc_auc_score(rn_true_te, rn_prob_te):.4f}")
    return lr_pred, lr_prob, rn_pred_te, rn_prob_te, rn_true_te


def evaluate_test_cross(X_clin_cv, X_aec_cv, y_cv,
                        X_clin_te, X_aec_te, y_te, sex_te,
                        med_epoch, rn_med_epoch=None,
                        scale_clin=True, scale_aec=True):
    """M2/M2_2: LR + CrossAttn + ResNet1D 최종 테스트."""
    rn_med_epoch = rn_med_epoch or med_epoch
    print(f"\n{'='*55}\nCrossAttn Test Evaluation\n{'='*55}")

    X_clin_cv_s, X_clin_te_s = _scale_clin_te(X_clin_cv, X_clin_te, scale_clin)
    X_aec_cv_s,  X_aec_te_s  = _scale_or_copy(X_aec_cv, X_aec_te, scale_aec)

    X_lr_cv = np.hstack([X_clin_cv_s, X_aec_cv_s])
    X_lr_te = np.hstack([X_clin_te_s, X_aec_te_s])
    lr_final = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr_final.fit(X_lr_cv, y_cv)
    lr_pred = lr_final.predict(X_lr_te)
    lr_prob = lr_final.predict_proba(X_lr_te)[:, 1]
    print(f"LR  AUC: {roc_auc_score(y_te, lr_prob):.4f}")

    tr_dl, te_dl = make_dual_loaders(X_clin_cv_s, X_aec_cv_s, y_cv,
                                     X_clin_te_s,  X_aec_te_s,  y_te)
    model_f, crit_f, opt_f, sched_f = build_cross_attn(y_cv)
    for _ in range(1, med_epoch + 1):
        train_cross_epoch(model_f, tr_dl, crit_f, opt_f); sched_f.step()
    _, ca_prob_te, ca_true_te = eval_cross_loader(model_f, te_dl, crit_f)
    ca_pred_te = (ca_prob_te >= 0.5).astype(int)
    print(f"CA  AUC: {roc_auc_score(ca_true_te, ca_prob_te):.4f}")

    X_rn_cv = np.hstack([X_clin_cv_s, X_aec_cv_s])
    X_rn_te = np.hstack([X_clin_te_s, X_aec_te_s])
    tr_dl_rn, te_dl_rn = make_loaders(X_rn_cv, y_cv, X_rn_te, y_te)
    model_rn_f, crit_rn_f, opt_rn_f, sched_rn_f = build_resnet(y_cv)
    for _ in range(1, rn_med_epoch + 1):
        train_one_epoch(model_rn_f, tr_dl_rn, crit_rn_f, opt_rn_f); sched_rn_f.step()
    _, rn_prob_te, rn_true_te = eval_loader(model_rn_f, te_dl_rn, crit_rn_f)
    rn_pred_te = (rn_prob_te >= 0.5).astype(int)
    print(f"RN  AUC: {roc_auc_score(rn_true_te, rn_prob_te):.4f}")

    return lr_pred, lr_prob, ca_pred_te, ca_prob_te, ca_true_te, rn_pred_te, rn_prob_te


# ── 0520 새 함수 ──────────────────────────────────────────────

def evaluate_test_cross_calib(X_clin_cv, X_aec_cv, y_cv,
                              X_clin_te, X_aec_te, y_te, sex_te,
                              med_epoch, oof_logits, oof_labels,
                              scale_clin=True, scale_aec=True):
    """
    M2 CrossAttn + Temperature Scaling 최종 테스트.

    1. 전체 CV 세트로 CrossAttn 재학습
    2. oof_logits/oof_labels로 최적 T 탐색
    3. test set에서 raw logit → calibrated prob 반환

    Returns
    -------
    ca_prob_raw    : (N_te,) — 캘리브레이션 전 확률
    ca_prob_calib  : (N_te,) — 캘리브레이션 후 확률
    ca_true_te     : (N_te,) — 실제 레이블
    T_opt          : float   — 최적 온도
    calib_stats    : dict
    """
    print(f"\n{'='*60}\nCrossAttn + Temperature Scaling — Test Evaluation\n{'='*60}")

    X_clin_cv_s, X_clin_te_s = _scale_clin_te(X_clin_cv, X_clin_te, scale_clin)
    X_aec_cv_s,  X_aec_te_s  = _scale_or_copy(X_aec_cv, X_aec_te, scale_aec)

    tr_dl, te_dl = make_dual_loaders(X_clin_cv_s, X_aec_cv_s, y_cv,
                                     X_clin_te_s,  X_aec_te_s,  y_te)
    model_f, crit_f, opt_f, sched_f = build_cross_attn(y_cv)
    for _ in range(1, med_epoch + 1):
        train_cross_epoch(model_f, tr_dl, crit_f, opt_f); sched_f.step()

    # test set raw logit + prob
    _, te_logits, ca_prob_raw, ca_true_te = eval_cross_loader_logits(model_f, te_dl, crit_f)

    # OOF logit으로 최적 T 탐색 (test 누수 없음)
    T_opt = find_optimal_temperature(oof_logits, oof_labels)
    ca_prob_calib = calibrate_probs(te_logits, T_opt)

    calib_stats = calibration_summary(
        "M2 CrossAttn", ca_true_te, ca_prob_raw, ca_prob_calib, T_opt
    )

    print(f"\n  Before calibration — AUC: {roc_auc_score(ca_true_te, ca_prob_raw):.4f}"
          f"  Brier: {brier_score_loss(ca_true_te, ca_prob_raw):.4f}")
    print(f"  After  calibration — AUC: {roc_auc_score(ca_true_te, ca_prob_calib):.4f}"
          f"  Brier: {brier_score_loss(ca_true_te, ca_prob_calib):.4f}")

    return ca_prob_raw, ca_prob_calib, ca_true_te, T_opt, calib_stats


def evaluate_test_regression(X_clin_cv, X_aec_cv, y_smi_cv, y_bin_cv, sex_cv,
                             X_clin_te, X_aec_te, y_smi_te, y_bin_te, sex_te,
                             med_epoch, scale_clin=True, scale_aec=True):
    """
    M6: SMIRegressionNet 최종 테스트.

    Returns
    -------
    pred_smi   : (N_te,) 예측 SMI
    true_smi   : (N_te,) 실제 SMI
    bin_pred   : (N_te,) 이진 예측 (threshold 적용)
    bin_prob   : (N_te,) sarcopenia 확률 (sigmoid 근사)
    true_bin   : (N_te,) 실제 이진 레이블
    reg_stats  : dict — MAE, RMSE, R², AUC
    """
    from scipy.special import expit
    print(f"\n{'='*55}\nSMI Regression — Test Evaluation (ep={med_epoch})\n{'='*55}")

    X_clin_cv_s, X_clin_te_s = _scale_clin_te(X_clin_cv, X_clin_te, scale_clin)
    X_aec_cv_s,  X_aec_te_s  = _scale_or_copy(X_aec_cv, X_aec_te, scale_aec)

    tr_dl, te_dl = make_regression_loaders(X_clin_cv_s, X_aec_cv_s, y_smi_cv,
                                           X_clin_te_s,  X_aec_te_s,  y_smi_te)
    model_f, crit_f, opt_f, sched_f = build_regression(y_smi_cv)
    for _ in range(1, med_epoch + 1):
        train_regression_epoch(model_f, tr_dl, crit_f, opt_f); sched_f.step()

    _, pred_smi, true_smi = eval_regression_loader(model_f, te_dl, crit_f)

    smi_thresh = np.array([SMI_THRESH_M if s == "M" else SMI_THRESH_F for s in sex_te])
    bin_pred   = (pred_smi <= smi_thresh).astype(int)
    bin_prob   = expit(-(pred_smi - smi_thresh) / 5.0).astype(np.float32)

    mae  = mean_absolute_error(true_smi, pred_smi)
    rmse = float(np.sqrt(np.mean((true_smi - pred_smi) ** 2)))
    r2   = r2_score(true_smi, pred_smi)
    auc  = roc_auc_score(y_bin_te, bin_prob) if len(np.unique(y_bin_te)) > 1 else float("nan")
    f1   = f1_score(y_bin_te, bin_pred, zero_division=0)
    acc  = accuracy_score(y_bin_te, bin_pred)

    reg_stats = {"mae": mae, "rmse": rmse, "r2": r2, "auc": auc, "f1": f1, "acc": acc}

    print(f"  Regression — MAE: {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.4f}")
    print(f"  Binary     — AUC: {auc:.4f}  F1: {f1:.4f}  Acc: {acc:.4f}")
    print("  By Sex:")
    _print_by_sex(y_bin_te, bin_pred, bin_prob, sex_te)

    return pred_smi, true_smi, bin_pred, bin_prob, y_bin_te, reg_stats


def evaluate_ensemble(
    y_te,
    probs_m1_lr: np.ndarray,
    probs_m2_ca: np.ndarray,
    oof_probs_m1_lr: np.ndarray,
    oof_probs_m2_ca: np.ndarray,
    oof_labels: np.ndarray,
    sex_te: np.ndarray,
    weights: tuple = (0.4, 0.6),
):
    """
    M7: 앙상블 평가 (M1 LR + M2 CrossAttn).

    방법 3가지:
      avg      : (prob_m1 + prob_m2) / 2
      weighted : w1 * prob_m1 + w2 * prob_m2  (weights 인자)
      stacking : OOF prob으로 학습한 LR meta-classifier

    Returns
    -------
    ensemble_results : dict — avg/weighted/stacking 각각의 지표
    """
    print(f"\n{'='*55}\nEnsemble Evaluation\n{'='*55}")

    results = {}
    for name, probs in [
        ("avg",      (probs_m1_lr + probs_m2_ca) / 2),
        ("weighted", weights[0] * probs_m1_lr + weights[1] * probs_m2_ca),
    ]:
        preds = (probs >= 0.5).astype(int)
        auc   = roc_auc_score(y_te, probs)
        auprc = average_precision_score(y_te, probs)
        brier = brier_score_loss(y_te, probs)
        acc   = accuracy_score(y_te, preds)
        f1    = f1_score(y_te, preds, zero_division=0)
        results[name] = {"auc": auc, "auprc": auprc, "brier": brier, "acc": acc, "f1": f1,
                         "probs": probs, "preds": preds}
        print(f"  [{name:8s}]  AUC: {auc:.4f}  AUPRC: {auprc:.4f}  "
              f"Brier: {brier:.4f}  Acc: {acc:.4f}  F1: {f1:.4f}")

    # Stacking: OOF 확률로 메타 LR 학습
    X_oof = np.stack([oof_probs_m1_lr, oof_probs_m2_ca], axis=1)
    meta_lr = LogisticRegression(max_iter=500, random_state=SEED)
    meta_lr.fit(X_oof, oof_labels)
    X_te_stack = np.stack([probs_m1_lr, probs_m2_ca], axis=1)
    stack_pred  = meta_lr.predict(X_te_stack)
    stack_prob  = meta_lr.predict_proba(X_te_stack)[:, 1]
    auc   = roc_auc_score(y_te, stack_prob)
    auprc = average_precision_score(y_te, stack_prob)
    brier = brier_score_loss(y_te, stack_prob)
    acc   = accuracy_score(y_te, stack_pred)
    f1    = f1_score(y_te, stack_pred, zero_division=0)
    results["stacking"] = {"auc": auc, "auprc": auprc, "brier": brier, "acc": acc, "f1": f1,
                           "probs": stack_prob, "preds": stack_pred,
                           "meta_coef": meta_lr.coef_[0].tolist()}
    print(f"  [stacking]  AUC: {auc:.4f}  AUPRC: {auprc:.4f}  "
          f"Brier: {brier:.4f}  Acc: {acc:.4f}  F1: {f1:.4f}"
          f"  (coef: M1={meta_lr.coef_[0][0]:.3f}, M2={meta_lr.coef_[0][1]:.3f})")

    return results
