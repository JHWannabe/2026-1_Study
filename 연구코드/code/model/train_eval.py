"""
모델 학습·평가 파이프라인 (교차검증 + 최종 Test Set 평가).

스케일링:
  _scale_aec(X_a, X_b, mode) — AEC 모드별 정규화 (fold tr/val·cv/test 공용)
  _scale_clin(X_a, X_b)      — Clinical Age·BMI 표준화 (sex_enc 제외)

5-Fold Stratified CV:
  run_cross_validation          — M1: LR
  run_cross_validation_cross    — M2/M2_2: CrossAttn
  run_cross_validation_cross3   — M3: CrossAttn3
  run_cross_validation_aec_only — M4: AECOnlyNet

Test Set 최종 평가:
  evaluate_test          — M1: LR
  evaluate_test_cross    — M2/M2_2: CrossAttn
  evaluate_test_cross3   — M3: CrossAttn3
  evaluate_test_aec_only — M4: AECOnlyNet

스케일링 정책:
  Clinical (Age·BMI): _scale_clin — sex_enc(col 1)은 StandardScaler 제외
  AEC              : _scale_aec  — mode에 따라 column/global/none
  MFR index        : 스케일링 없음 (nn.Embedding index)
  Fold/CV 누수 방지: X_a(train/CV)에서 fit, X_b(val/test)에 transform만 적용
"""
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, f1_score,
    classification_report, average_precision_score, brier_score_loss,
)
import torch

from config import N_FOLDS, SEED, EPOCHS
from models import (
    build_resnet, make_loaders, train_epoch, eval_epoch,
    build_aec_only,
<<<<<<< HEAD
    build_cross_attn, build_cross_attn_feat, make_dual_loaders,
    build_cross_attn3, make_quad_loaders,
=======
    build_cross_attn, make_dual_loaders, train_cross_epoch, eval_cross_loader,
    build_cross_attn3, make_quad_loaders, train_cross3_epoch, eval_cross3_loader,
>>>>>>> 85e98357048347f31c593090de905f10997a7cde
    build_late_fusion, build_late_fusion3,
)

from metrics import group_metrics, print_bootstrap_ci
from data import augment_aec


# ── Scaling helpers ───────────────────────────────────────────

def _scale_aec(X_a: np.ndarray, X_b: np.ndarray, mode: str):
    """AEC 정규화. X_a(train/CV)로 fit 후 X_b(val/test)에 transform. mode: "column"|"global"|"none"."""
    if mode == "column":
        sc = StandardScaler()
        return sc.fit_transform(X_a), sc.transform(X_b)
    if mode == "global":
        g_mean = float(X_a.mean())
        g_std  = max(float(X_a.std()), 1e-8)
        return ((X_a - g_mean) / g_std).astype(np.float32), \
               ((X_b - g_mean) / g_std).astype(np.float32)
    return X_a.copy(), X_b.copy()


def _scale_clin(X_a: np.ndarray, X_b: np.ndarray):
    """Clinical Age(col 0)·BMI(col 2) 표준화. sex_enc(col 1)은 그대로. X_a로 fit."""
    sc = StandardScaler()
    Xa, Xb = X_a.copy(), X_b.copy()
    Xa[:, [0, 2]] = sc.fit_transform(X_a[:, [0, 2]])
    Xb[:, [0, 2]] = sc.transform(X_b[:, [0, 2]])
    return Xa, Xb


def _scale_combined(X_a: np.ndarray, X_b: np.ndarray):
    """Clinic+AEC 결합 피처 스케일링. col 1(sex_enc 이진값) 제외, 나머지 전체 StandardScaler."""
    cols = [i for i in range(X_a.shape[1]) if i != 1]
    sc = StandardScaler()
    Xa, Xb = X_a.copy(), X_b.copy()
    Xa[:, cols] = sc.fit_transform(X_a[:, cols])
    Xb[:, cols] = sc.transform(X_b[:, cols])
    return Xa, Xb


def _youden_threshold(y_true, y_prob):
    """ROC 기반 Youden's J (sensitivity + specificity - 1) 최대화 임계값 반환."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


# ── Training loop helpers ─────────────────────────────────────

def _train_loop(model, tr_dl, val_dl, crit, opt, sched, train_fn, eval_fn):
    """Best-val-AUC 기준으로 EPOCHS 동안 학습. (best_epoch, best_state, hist) 반환."""
    best_auc, best_epoch = 0.0, 0
    best_state: dict = {}
    hist: dict = {"train_loss": [], "val_loss": [], "val_auc": []}
    for ep in range(1, EPOCHS + 1):
        t_loss = train_fn(model, tr_dl, crit, opt)
        sched.step()
        v_loss, vp, vt = eval_fn(model, val_dl, crit)
        v_auc = roc_auc_score(vt, vp)
        hist["train_loss"].append(t_loss)
        hist["val_loss"].append(v_loss)
        hist["val_auc"].append(v_auc)
        if v_auc > best_auc:
            best_auc, best_epoch = v_auc, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_epoch, best_state, hist


def _final_train(model, loader, crit, opt, sched, n_epochs, train_fn=None):
    """최종 모델을 n_epochs 동안 학습."""
    if train_fn is None:
        train_fn = train_epoch
    for _ in range(1, n_epochs + 1):
        train_fn(model, loader, crit, opt)
        sched.step()


# ── Shared CV / test-eval internals ──────────────────────────

def _print_test_stats(label, y_true, y_pred, y_prob, sex_te):
    """test set 전체·성별 지표를 콘솔 출력."""
    print(f"\n{label} — Test Set (Overall):"
          f"  AUC: {roc_auc_score(y_true, y_prob):.4f}"
          f"  AUPRC: {average_precision_score(y_true, y_prob):.4f}"
          f"  Brier: {brier_score_loss(y_true, y_prob):.4f}"
          f"  Acc: {accuracy_score(y_true, y_pred):.4f}"
          f"  F1: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"\n{label} — Test Set (By Sex):")
    _print_by_sex(y_true, y_pred, y_prob, sex_te)
    print(f"\n{'─'*55}\nTest Set — Bootstrap CI\n{'─'*55}")


def _cv_dual(build_fn, X_clin_cv, X_aec_cv, y_cv, scale_aec, label, augment=True):
    """Dual-input (Clinic+AEC) 모델의 공통 N_FOLDS 교차검증."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv, roc_folds, histories, best_epochs, best_thresholds = [], [], [], [], []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold CV  [{label} | scale_aec={scale_aec}]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")
        X_clin_tr, X_clin_val = _scale_clin(X_clin_cv[tr_i], X_clin_cv[val_i])
        X_aec_tr,  X_aec_val  = _scale_aec(X_aec_cv[tr_i],  X_aec_cv[val_i], scale_aec)
        if augment:
            X_aec_tr = augment_aec(X_aec_tr, rng=np.random.default_rng(SEED + fold))
        y_tr, y_val = y_cv[tr_i], y_cv[val_i]

        tr_dl, val_dl = make_dual_loaders(X_clin_tr, X_aec_tr, y_tr, X_clin_val, X_aec_val, y_val)
        model, crit, opt, sched = build_fn(y_tr)
        best_epoch, best_state, hist = _train_loop(
            model, tr_dl, val_dl, crit, opt, sched, train_epoch, eval_epoch
        )
        model.load_state_dict(best_state)
        _, fprob, _ = eval_epoch(model, val_dl, crit)

        best_thresh = _youden_threshold(y_val, fprob)
        best_thresholds.append(best_thresh)
        m = group_metrics(y_val, (fprob >= best_thresh).astype(int), fprob)
        cv.append({"fold": fold, **m})
        fpr, tpr, _ = roc_curve(y_val, fprob)
        roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m["auc"]})
        histories.append(hist)
        best_epochs.append(best_epoch)
        print(f"  {label[:4]:4} — AUC: {m['auc']:.4f}  AUPRC: {m['auprc']:.4f}"
              f"  Brier: {m['brier']:.4f}  Acc: {m['acc']:.4f}  F1: {m['f1']:.4f}"
              f"  (best ep={best_epoch}, thresh={best_thresh:.3f})")

    return cv, roc_folds, histories, best_epochs, best_thresholds


def _cv_quad(build_fn, X_clin_cv, X_aec_cv, X_mfr_cv, y_cv, n_mfr, scale_aec, label):
    """Quad-input (Clinic+AEC+Scanner) 모델의 공통 N_FOLDS 교차검증."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv, roc_folds, histories, best_epochs, best_thresholds = [], [], [], [], []

    print("=" * 65)
    print(f"{N_FOLDS}-Fold CV  [{label} | scale_aec={scale_aec}]")
    print("=" * 65)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")
        X_clin_tr, X_clin_val = _scale_clin(X_clin_cv[tr_i], X_clin_cv[val_i])
        X_aec_tr,  X_aec_val  = _scale_aec(X_aec_cv[tr_i],  X_aec_cv[val_i], scale_aec)
        X_aec_tr = augment_aec(X_aec_tr, rng=np.random.default_rng(SEED + fold))
        X_mfr_tr, X_mfr_val   = X_mfr_cv[tr_i], X_mfr_cv[val_i]
        y_tr, y_val = y_cv[tr_i], y_cv[val_i]

        tr_dl, val_dl = make_quad_loaders(
            X_clin_tr, X_aec_tr, X_mfr_tr, y_tr,
            X_clin_val, X_aec_val, X_mfr_val, y_val,
        )
        model, crit, opt, sched = build_fn(y_tr, n_mfr)
        best_epoch, best_state, hist = _train_loop(
            model, tr_dl, val_dl, crit, opt, sched, train_epoch, eval_epoch
        )
        model.load_state_dict(best_state)
        _, fprob, _ = eval_epoch(model, val_dl, crit)

        best_thresh = _youden_threshold(y_val, fprob)
        best_thresholds.append(best_thresh)
        m = group_metrics(y_val, (fprob >= best_thresh).astype(int), fprob)
        cv.append({"fold": fold, **m})
        fpr, tpr, _ = roc_curve(y_val, fprob)
        roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m["auc"]})
        histories.append(hist)
        best_epochs.append(best_epoch)
        print(f"  {label[:4]:4} — AUC: {m['auc']:.4f}  AUPRC: {m['auprc']:.4f}"
              f"  Brier: {m['brier']:.4f}  Acc: {m['acc']:.4f}  F1: {m['f1']:.4f}"
              f"  (best ep={best_epoch}, thresh={best_thresh:.3f})")

    return cv, roc_folds, histories, best_epochs, best_thresholds


def _eval_dual(build_fn, label, ci_key,
               X_clin_cv, X_aec_cv, y_cv,
               X_clin_te, X_aec_te, y_te, sex_te,
               med_epoch, scale_aec, threshold, weight_path):
    """Dual-input 모델의 공통 최종 test set 평가."""
    X_clin_cv_s, X_clin_te_s = _scale_clin(X_clin_cv, X_clin_te)
    X_aec_cv_s,  X_aec_te_s  = _scale_aec(X_aec_cv, X_aec_te, scale_aec)

    tr_dl, te_dl = make_dual_loaders(X_clin_cv_s, X_aec_cv_s, y_cv,
                                     X_clin_te_s,  X_aec_te_s,  y_te)
    model_f, crit_f, opt_f, sched_f = build_fn(y_cv)
    print(f"{label} — training final model for {med_epoch} epochs on full CV set …")
    _final_train(model_f, tr_dl, crit_f, opt_f, sched_f, med_epoch)

    if weight_path:
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        torch.save(model_f.state_dict(), weight_path)
        print(f"  [{label}] weights → {weight_path}")

    _, prob_te, true_te = eval_epoch(model_f, te_dl, crit_f)
    pred_te = (prob_te >= threshold).astype(int)
    _print_test_stats(label, true_te, pred_te, prob_te, sex_te)
    ci = print_bootstrap_ci(label, true_te, pred_te, prob_te)
    return pred_te, prob_te, true_te, {ci_key: ci}, model_f, X_clin_te_s, X_aec_te_s


def _eval_quad(build_fn, label, ci_key,
               X_clin_cv, X_aec_cv, X_mfr_cv, y_cv,
               X_clin_te, X_aec_te, X_mfr_te, y_te,
               sex_te, med_epoch, n_mfr, scale_aec, threshold, weight_path):
    """Quad-input 모델의 공통 최종 test set 평가."""
    X_clin_cv_s, X_clin_te_s = _scale_clin(X_clin_cv, X_clin_te)
    X_aec_cv_s,  X_aec_te_s  = _scale_aec(X_aec_cv, X_aec_te, scale_aec)

    tr_dl, te_dl = make_quad_loaders(
        X_clin_cv_s, X_aec_cv_s, X_mfr_cv, y_cv,
        X_clin_te_s, X_aec_te_s, X_mfr_te, y_te,
    )
    model_f, crit_f, opt_f, sched_f = build_fn(y_cv, n_mfr)
    print(f"{label} — training final model for {med_epoch} epochs on full CV set …")
    _final_train(model_f, tr_dl, crit_f, opt_f, sched_f, med_epoch)

    if weight_path:
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        torch.save(model_f.state_dict(), weight_path)
        print(f"  [{label}] weights → {weight_path}")

    _, prob_te, true_te = eval_epoch(model_f, te_dl, crit_f)
    pred_te = (prob_te >= threshold).astype(int)
    _print_test_stats(label, true_te, pred_te, prob_te, sex_te)
    ci = print_bootstrap_ci(label, true_te, pred_te, prob_te)
    return pred_te, prob_te, true_te, {ci_key: ci}, model_f, X_clin_te_s, X_aec_te_s


# ── Print helper ──────────────────────────────────────────────

def _print_by_sex(y_true, y_pred, y_prob, sex):
    """성별(M/F)별 분류 지표(AUC·AUPRC·Brier·Acc·F1)와 classification_report 출력."""
    for s in ["M", "F"]:
        mask = sex == s
        if not mask.any():
            continue
        yt, yp, ypr = y_true[mask], y_pred[mask], y_prob[mask]
        has_both = len(np.unique(yt)) > 1
        auc   = roc_auc_score(yt, ypr)           if has_both else float("nan")
        auprc = average_precision_score(yt, ypr) if has_both else float("nan")
        brier = brier_score_loss(yt, ypr)        if has_both else float("nan")
        print(f"  [{s}] n={mask.sum()}  AUC: {auc:.4f}  AUPRC: {auprc:.4f}"
              f"  Brier: {brier:.4f}  Acc: {accuracy_score(yt, yp):.4f}"
              f"  F1: {f1_score(yt, yp, zero_division=0):.4f}")
        print(classification_report(yt, yp, target_names=["Normal", "Sarcopenia"],
                                    zero_division=0))


# ══════════════════════════════════════════════════════════════
#  5-Fold Stratified Cross-Validation
# ══════════════════════════════════════════════════════════════

def run_cross_validation(X_cv, y_cv):
    """LR에 대해 N_FOLDS 교차검증. Returns: (lr_cv, lr_roc_folds, lr_best_thresholds)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lr_cv, lr_roc_folds, lr_best_thresholds = [], [], []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold Cross-Validation  [LR]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")
        Xtr_s, Xval_s = _scale_clin(X_cv[tr_i], X_cv[val_i])
        y_ftr, y_fval = y_cv[tr_i], y_cv[val_i]

        lr_f = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
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


def run_cross_validation_combined(X_cv, y_cv):
    """Clinic+AEC 결합 피처(14) LR에 대해 N_FOLDS 교차검증.
    Returns: (lr_cv, lr_roc_folds, lr_best_thresholds)"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lr_cv, lr_roc_folds, lr_best_thresholds = [], [], []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold Cross-Validation  [LR + AEC Hand-crafted Features]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")
        Xtr_s, Xval_s = _scale_combined(X_cv[tr_i], X_cv[val_i])
        y_ftr, y_fval = y_cv[tr_i], y_cv[val_i]

        lr_f = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr_f.fit(Xtr_s, y_ftr)
        lr_fprob = lr_f.predict_proba(Xval_s)[:, 1]

        best_thresh = _youden_threshold(y_fval, lr_fprob)
        lr_fp = (lr_fprob >= best_thresh).astype(int)
        lr_best_thresholds.append(best_thresh)

        m_lr = group_metrics(y_fval, lr_fp, lr_fprob)
        lr_cv.append({"fold": fold, **m_lr})
        fpr, tpr, _ = roc_curve(y_fval, lr_fprob)
        lr_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_lr["auc"]})

        print(f"  LR+AEC — AUC: {m_lr['auc']:.4f}  AUPRC: {m_lr['auprc']:.4f}"
              f"  Brier: {m_lr['brier']:.4f}  Acc: {m_lr['acc']:.4f}  F1: {m_lr['f1']:.4f}"
              f"  (thresh={best_thresh:.3f})")

    return lr_cv, lr_roc_folds, lr_best_thresholds


def run_cross_validation_cross(X_clin_cv, X_aec_cv, y_cv, scale_aec="column"):
    """ClinAECCrossAttn에 대해 N_FOLDS 교차검증."""
    return _cv_dual(build_cross_attn, X_clin_cv, X_aec_cv, y_cv, scale_aec, "CrossAttn")


def run_cross_validation_cross3(X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv, n_manufacturers,
                                 scale_aec="column"):
    """ClinAECScanCrossAttn에 대해 N_FOLDS 교차검증."""
    return _cv_quad(build_cross_attn3, X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv,
                    n_manufacturers, scale_aec, "CrossAttn3")


def run_cross_validation_cross_feat(X_clin_cv, X_feat_cv, y_cv):
    """Clinic + AEC 11 hand-crafted features CrossAttn에 대해 N_FOLDS 교차검증."""
    return _cv_dual(build_cross_attn_feat, X_clin_cv, X_feat_cv, y_cv, "column",
                    "CrossAttn-Feat", augment=False)


def run_cross_validation_aec_only(X_aec_cv, y_cv, scale_aec="column"):
    """AECOnlyNet에 대해 N_FOLDS 교차검증.
    Returns: (cv, roc_folds, histories, best_epochs, best_thresholds)"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv, roc_folds, histories, best_epochs, best_thresholds = [], [], [], [], []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold CV  [AECOnly | scale_aec={scale_aec}]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_aec_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")
        X_aec_tr, X_aec_val = _scale_aec(X_aec_cv[tr_i], X_aec_cv[val_i], scale_aec)
        X_aec_tr = augment_aec(X_aec_tr, rng=np.random.default_rng(SEED + fold))
        y_tr, y_val = y_cv[tr_i], y_cv[val_i]

        tr_dl, val_dl = make_loaders(X_aec_tr, y_tr, X_aec_val, y_val)
        model, crit, opt, sched = build_aec_only(y_tr)
        best_epoch, best_state, hist = _train_loop(
            model, tr_dl, val_dl, crit, opt, sched, train_epoch, eval_epoch
        )
        model.load_state_dict(best_state)
        _, fprob, _ = eval_epoch(model, val_dl, crit)

        best_thresh = _youden_threshold(y_val, fprob)
        fp = (fprob >= best_thresh).astype(int)
        best_thresholds.append(best_thresh)

        m = group_metrics(y_val, fp, fprob)
        cv.append({"fold": fold, **m})
        fpr, tpr, _ = roc_curve(y_val, fprob)
        roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m["auc"]})
        histories.append(hist)
        best_epochs.append(best_epoch)

        print(f"  AEC  — AUC: {m['auc']:.4f}  AUPRC: {m['auprc']:.4f}"
              f"  Brier: {m['brier']:.4f}  Acc: {m['acc']:.4f}  F1: {m['f1']:.4f}"
              f"  (best ep={best_epoch}, thresh={best_thresh:.3f})")

    return cv, roc_folds, histories, best_epochs, best_thresholds


# ══════════════════════════════════════════════════════════════
#  Test Set 최종 평가
# ══════════════════════════════════════════════════════════════

def evaluate_test(X_cv, y_cv, X_te, y_te, sex_te, threshold=0.5):
    """전체 CV 세트로 LR 최종 모델 학습 후 test set 예측.
    Returns: (lr_pred, lr_prob, stats_te)"""
    print(f"\n{'='*55}\nFinal Test Evaluation  [LR | threshold={threshold:.3f}]\n{'='*55}")
    X_cv_s, X_te_s = _scale_clin(X_cv, X_te)

    lr_final = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr_final.fit(X_cv_s, y_cv)
    lr_prob = lr_final.predict_proba(X_te_s)[:, 1]
    lr_pred = (lr_prob >= threshold).astype(int)

    print(f"\nLR — Test Set (Overall):"
          f"  AUC: {roc_auc_score(y_te, lr_prob):.4f}"
          f"  AUPRC: {average_precision_score(y_te, lr_prob):.4f}"
          f"  Brier: {brier_score_loss(y_te, lr_prob):.4f}"
          f"  Acc: {accuracy_score(y_te, lr_pred):.4f}"
          f"  F1: {f1_score(y_te, lr_pred, zero_division=0):.4f}")
    print("\nLR — Test Set (By Sex):")
    _print_by_sex(y_te, lr_pred, lr_prob, sex_te)

    print(f"\n{'─'*55}\nTest Set — Bootstrap CI\n{'─'*55}")
    ci_lr = print_bootstrap_ci("LR", y_te, lr_pred, lr_prob)
    return lr_pred, lr_prob, {"bootstrap_lr": ci_lr}


def evaluate_test_combined(X_cv, y_cv, X_te, y_te, sex_te, threshold=0.5):
    """Clinic+AEC 결합 피처 LR 최종 모델 학습 후 test set 예측.
    Returns: (lr_pred, lr_prob, stats_te, lr_final, X_te_s)"""
    print(f"\n{'='*55}\nFinal Test Evaluation  [LR+AEC | threshold={threshold:.3f}]\n{'='*55}")
    X_cv_s, X_te_s = _scale_combined(X_cv, X_te)

    lr_final = LogisticRegression(max_iter=10000, random_state=SEED, class_weight="balanced")
    lr_final.fit(X_cv_s, y_cv)
    lr_prob = lr_final.predict_proba(X_te_s)[:, 1]
    lr_pred = (lr_prob >= threshold).astype(int)

    print(f"\nLR+AEC — Test Set (Overall):"
          f"  AUC: {roc_auc_score(y_te, lr_prob):.4f}"
          f"  AUPRC: {average_precision_score(y_te, lr_prob):.4f}"
          f"  Brier: {brier_score_loss(y_te, lr_prob):.4f}"
          f"  Acc: {accuracy_score(y_te, lr_pred):.4f}"
          f"  F1: {f1_score(y_te, lr_pred, zero_division=0):.4f}")
    print("\nLR+AEC — Test Set (By Sex):")
    _print_by_sex(y_te, lr_pred, lr_prob, sex_te)

    print(f"\n{'─'*55}\nTest Set — Bootstrap CI\n{'─'*55}")
    ci_lr = print_bootstrap_ci("LR+AEC", y_te, lr_pred, lr_prob)
    return lr_pred, lr_prob, {"bootstrap_lr_aec": ci_lr}, lr_final, X_te_s


def evaluate_test_cross_feat(X_clin_cv, X_feat_cv, y_cv,
                             X_clin_te, X_feat_te, y_te, sex_te,
                             med_epoch, threshold=0.5, weight_path=None):
    """Clinic + AEC 11 hand-crafted features CrossAttn 최종 모델 학습 후 test set 예측."""
    return _eval_dual(build_cross_attn_feat, "CrossAttn-Feat", "bootstrap_ca_feat",
                      X_clin_cv, X_feat_cv, y_cv, X_clin_te, X_feat_te, y_te, sex_te,
                      med_epoch, "column", threshold, weight_path)


def evaluate_test_cross(X_clin_cv, X_aec_cv, y_cv,
                        X_clin_te, X_aec_te, y_te, sex_te,
                        med_epoch, scale_aec="column", threshold=0.5, weight_path=None):
    """전체 CV 세트로 CrossAttn 최종 모델 학습 후 test set 예측."""
    return _eval_dual(build_cross_attn, "CrossAttn", "bootstrap_ca",
                      X_clin_cv, X_aec_cv, y_cv, X_clin_te, X_aec_te, y_te, sex_te,
                      med_epoch, scale_aec, threshold, weight_path)


def evaluate_test_cross3(X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv,
                          X_clin_te, X_aec_te, X_scan_mfr_te, y_te,
                          sex_te, med_epoch, n_manufacturers,
                          scale_aec="column", threshold=0.5, weight_path=None):
    """전체 CV 세트로 CrossAttn3 최종 모델 학습 후 test set 예측."""
    return _eval_quad(build_cross_attn3, "CrossAttn3", "bootstrap_ca3",
                      X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv,
                      X_clin_te, X_aec_te, X_scan_mfr_te, y_te,
                      sex_te, med_epoch, n_manufacturers, scale_aec, threshold, weight_path)


def run_cross_validation_late_fusion(X_clin_cv, X_aec_cv, y_cv, scale_aec="column"):
    """ClinAECLateFusion에 대해 N_FOLDS 교차검증.
    Returns: (lf_cv, lf_roc_folds, lf_histories, lf_best_epochs, lf_best_thresholds)"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lf_cv, lf_roc_folds, lf_histories, lf_best_epochs, lf_best_thresholds = [], [], [], [], []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold CV  [LateFusion | scale_aec={scale_aec}]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")
        X_clin_tr, X_clin_val = _scale_clin(X_clin_cv[tr_i], X_clin_cv[val_i])
        X_aec_tr,  X_aec_val  = _scale_aec(X_aec_cv[tr_i],  X_aec_cv[val_i], scale_aec)
        X_aec_tr = augment_aec(X_aec_tr, rng=np.random.default_rng(SEED + fold))
        y_tr, y_val = y_cv[tr_i], y_cv[val_i]

        tr_dl, val_dl = make_dual_loaders(X_clin_tr, X_aec_tr, y_tr, X_clin_val, X_aec_val, y_val)
        model, crit, opt, sched = build_late_fusion(y_tr)
        best_epoch, best_state, hist = _train_loop(
            model, tr_dl, val_dl, crit, opt, sched, train_cross_epoch, eval_cross_loader
        )
        model.load_state_dict(best_state)
        _, lf_fprob, _ = eval_cross_loader(model, val_dl, crit)

        best_thresh = _youden_threshold(y_val, lf_fprob)
        lf_fp = (lf_fprob >= best_thresh).astype(int)
        lf_best_thresholds.append(best_thresh)

        m_lf = group_metrics(y_val, lf_fp, lf_fprob)
        lf_cv.append({"fold": fold, **m_lf})
        fpr, tpr, _ = roc_curve(y_val, lf_fprob)
        lf_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_lf["auc"]})
        lf_histories.append(hist)
        lf_best_epochs.append(best_epoch)

        print(f"  LF   — AUC: {m_lf['auc']:.4f}  AUPRC: {m_lf['auprc']:.4f}"
              f"  Brier: {m_lf['brier']:.4f}  Acc: {m_lf['acc']:.4f}  F1: {m_lf['f1']:.4f}"
              f"  (best ep={best_epoch}, thresh={best_thresh:.3f})")

    return lf_cv, lf_roc_folds, lf_histories, lf_best_epochs, lf_best_thresholds


def run_cross_validation_late_fusion3(X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv, n_manufacturers,
                                      scale_aec="column"):
    """ClinAECScanLateFusion에 대해 N_FOLDS 교차검증.
    Returns: (lf3_cv, lf3_roc_folds, lf3_histories, lf3_best_epochs, lf3_best_thresholds)"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lf3_cv, lf3_roc_folds, lf3_histories, lf3_best_epochs, lf3_best_thresholds = [], [], [], [], []

    print("=" * 65)
    print(f"{N_FOLDS}-Fold CV  [LateFusion3 | scale_aec={scale_aec}]")
    print("=" * 65)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")
        X_clin_tr, X_clin_val = _scale_clin(X_clin_cv[tr_i], X_clin_cv[val_i])
        X_aec_tr,  X_aec_val  = _scale_aec(X_aec_cv[tr_i],  X_aec_cv[val_i], scale_aec)
        X_aec_tr = augment_aec(X_aec_tr, rng=np.random.default_rng(SEED + fold))
        X_mfr_tr, X_mfr_val   = X_scan_mfr_cv[tr_i], X_scan_mfr_cv[val_i]
        y_tr, y_val = y_cv[tr_i], y_cv[val_i]

        tr_dl, val_dl = make_quad_loaders(
            X_clin_tr, X_aec_tr, X_mfr_tr, y_tr,
            X_clin_val, X_aec_val, X_mfr_val, y_val,
        )
        model, crit, opt, sched = build_late_fusion3(y_tr, n_manufacturers)
        best_epoch, best_state, hist = _train_loop(
            model, tr_dl, val_dl, crit, opt, sched, train_cross3_epoch, eval_cross3_loader
        )
        model.load_state_dict(best_state)
        _, lf3_fprob, _ = eval_cross3_loader(model, val_dl, crit)

        best_thresh = _youden_threshold(y_val, lf3_fprob)
        lf3_fp = (lf3_fprob >= best_thresh).astype(int)
        lf3_best_thresholds.append(best_thresh)

        m_lf3 = group_metrics(y_val, lf3_fp, lf3_fprob)
        lf3_cv.append({"fold": fold, **m_lf3})
        fpr, tpr, _ = roc_curve(y_val, lf3_fprob)
        lf3_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m_lf3["auc"]})
        lf3_histories.append(hist)
        lf3_best_epochs.append(best_epoch)

        print(f"  LF3  — AUC: {m_lf3['auc']:.4f}  AUPRC: {m_lf3['auprc']:.4f}"
              f"  Brier: {m_lf3['brier']:.4f}  Acc: {m_lf3['acc']:.4f}  F1: {m_lf3['f1']:.4f}"
              f"  (best ep={best_epoch}, thresh={best_thresh:.3f})")

    return lf3_cv, lf3_roc_folds, lf3_histories, lf3_best_epochs, lf3_best_thresholds


def evaluate_test_late_fusion(X_clin_cv, X_aec_cv, y_cv,
                              X_clin_te, X_aec_te, y_te, sex_te,
                              med_epoch, scale_aec="column",
                              threshold=0.5, weight_path=None):
    """전체 CV 세트로 LateFusion 최종 모델 학습 후 test set 예측.
    Returns: (lf_pred_te, lf_prob_te, lf_true_te, stats_te, model, X_clin_te_s, X_aec_te_s)"""
    print(f"\n{'='*55}\nLateFusion Test Evaluation"
          f"  [scale_aec={scale_aec} | threshold={threshold:.3f}]\n{'='*55}")
    X_clin_cv_s, X_clin_te_s = _scale_clin(X_clin_cv, X_clin_te)
    X_aec_cv_s,  X_aec_te_s  = _scale_aec(X_aec_cv,  X_aec_te, scale_aec)

    tr_dl, te_dl = make_dual_loaders(X_clin_cv_s, X_aec_cv_s, y_cv,
                                     X_clin_te_s,  X_aec_te_s,  y_te)
    model_f, crit_f, opt_f, sched_f = build_late_fusion(y_cv)
    print(f"LateFusion — training final model for {med_epoch} epochs on full CV set …")
    _final_train(model_f, tr_dl, crit_f, opt_f, sched_f, med_epoch, train_cross_epoch)

    if weight_path is not None:
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        torch.save(model_f.state_dict(), weight_path)
        print(f"  [LateFusion] weights → {weight_path}")

    _, lf_prob_te, lf_true_te = eval_cross_loader(model_f, te_dl, crit_f)
    lf_pred_te = (lf_prob_te >= threshold).astype(int)

    print(f"\nLateFusion — Test Set (Overall):"
          f"  AUC: {roc_auc_score(lf_true_te, lf_prob_te):.4f}"
          f"  AUPRC: {average_precision_score(lf_true_te, lf_prob_te):.4f}"
          f"  Brier: {brier_score_loss(lf_true_te, lf_prob_te):.4f}"
          f"  Acc: {accuracy_score(lf_true_te, lf_pred_te):.4f}"
          f"  F1: {f1_score(lf_true_te, lf_pred_te, zero_division=0):.4f}")
    print("\nLateFusion — Test Set (By Sex):")
    _print_by_sex(lf_true_te, lf_pred_te, lf_prob_te, sex_te)

    print(f"\n{'─'*55}\nTest Set — Bootstrap CI\n{'─'*55}")
    ci_lf = print_bootstrap_ci("LateFusion", lf_true_te, lf_pred_te, lf_prob_te)
    return lf_pred_te, lf_prob_te, lf_true_te, {"bootstrap_lf": ci_lf}, model_f, X_clin_te_s, X_aec_te_s


def evaluate_test_late_fusion3(X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv,
                               X_clin_te, X_aec_te, X_scan_mfr_te, y_te,
                               sex_te, med_epoch, n_manufacturers,
                               scale_aec="column", threshold=0.5, weight_path=None):
    """전체 CV 세트로 LateFusion3 최종 모델 학습 후 test set 예측.
    Returns: (lf3_pred_te, lf3_prob_te, lf3_true_te, stats_te, model, X_clin_te_s, X_aec_te_s)"""
    print(f"\n{'='*65}\nLateFusion3 Test Evaluation"
          f"  [scale_aec={scale_aec} | threshold={threshold:.3f}]\n{'='*65}")
    X_clin_cv_s, X_clin_te_s = _scale_clin(X_clin_cv, X_clin_te)
    X_aec_cv_s,  X_aec_te_s  = _scale_aec(X_aec_cv,  X_aec_te, scale_aec)

    tr_dl, te_dl = make_quad_loaders(
        X_clin_cv_s, X_aec_cv_s, X_scan_mfr_cv, y_cv,
        X_clin_te_s, X_aec_te_s, X_scan_mfr_te, y_te,
    )
    model_f, crit_f, opt_f, sched_f = build_late_fusion3(y_cv, n_manufacturers)
    print(f"LateFusion3 — training final model for {med_epoch} epochs on full CV set …")
    _final_train(model_f, tr_dl, crit_f, opt_f, sched_f, med_epoch, train_cross3_epoch)

    if weight_path is not None:
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        torch.save(model_f.state_dict(), weight_path)
        print(f"  [LateFusion3] weights → {weight_path}")

    _, lf3_prob_te, lf3_true_te = eval_cross3_loader(model_f, te_dl, crit_f)
    lf3_pred_te = (lf3_prob_te >= threshold).astype(int)

    print(f"\nLateFusion3 — Test Set (Overall):"
          f"  AUC: {roc_auc_score(lf3_true_te, lf3_prob_te):.4f}"
          f"  AUPRC: {average_precision_score(lf3_true_te, lf3_prob_te):.4f}"
          f"  Brier: {brier_score_loss(lf3_true_te, lf3_prob_te):.4f}"
          f"  Acc: {accuracy_score(lf3_true_te, lf3_pred_te):.4f}"
          f"  F1: {f1_score(lf3_true_te, lf3_pred_te, zero_division=0):.4f}")
    print("\nLateFusion3 — Test Set (By Sex):")
    _print_by_sex(lf3_true_te, lf3_pred_te, lf3_prob_te, sex_te)

    print(f"\n{'─'*55}\nTest Set — Bootstrap CI\n{'─'*55}")
    ci_lf3 = print_bootstrap_ci("LateFusion3", lf3_true_te, lf3_pred_te, lf3_prob_te)
    return lf3_pred_te, lf3_prob_te, lf3_true_te, {"bootstrap_lf3": ci_lf3}, model_f, X_clin_te_s, X_aec_te_s


def evaluate_test_aec_only(X_aec_cv, y_cv, X_aec_te, y_te, sex_te,
                           med_epoch, scale_aec="column", threshold=0.5, weight_path=None):
    """전체 CV 세트로 AECOnlyNet 최종 모델 학습 후 test set 예측."""
    print(f"\n{'='*55}\nAECOnly Test Evaluation  [scale_aec={scale_aec} | threshold={threshold:.3f}]\n{'='*55}")
    X_aec_cv_s, X_aec_te_s = _scale_aec(X_aec_cv, X_aec_te, scale_aec)
    tr_dl, te_dl = make_loaders(X_aec_cv_s, y_cv, X_aec_te_s, y_te)
    model_f, crit_f, opt_f, sched_f = build_aec_only(y_cv)
    print(f"AECOnly — training final model for {med_epoch} epochs on full CV set …")
    _final_train(model_f, tr_dl, crit_f, opt_f, sched_f, med_epoch)
    if weight_path:
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        torch.save(model_f.state_dict(), weight_path)
        print(f"  [AECOnly] weights → {weight_path}")
    _, prob_te, true_te = eval_epoch(model_f, te_dl, crit_f)
    pred_te = (prob_te >= threshold).astype(int)
    _print_test_stats("AECOnly", true_te, pred_te, prob_te, sex_te)
    ci = print_bootstrap_ci("AECOnly", true_te, pred_te, prob_te)
    return pred_te, prob_te, true_te, {"bootstrap_aec_only": ci}, model_f, X_aec_te_s


def run_cross_validation_late_fusion(X_clin_cv, X_aec_cv, y_cv, scale_aec="column"):
    """ClinAECLateFusion에 대해 N_FOLDS 교차검증."""
    return _cv_dual(build_late_fusion, X_clin_cv, X_aec_cv, y_cv, scale_aec, "LateFusion")


def run_cross_validation_late_fusion3(X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv, n_manufacturers,
                                      scale_aec="column"):
    """ClinAECScanLateFusion에 대해 N_FOLDS 교차검증."""
    return _cv_quad(build_late_fusion3, X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv,
                    n_manufacturers, scale_aec, "LateFusion3")


def evaluate_test_late_fusion(X_clin_cv, X_aec_cv, y_cv,
                              X_clin_te, X_aec_te, y_te, sex_te,
                              med_epoch, scale_aec="column", threshold=0.5, weight_path=None):
    """전체 CV 세트로 LateFusion 최종 모델 학습 후 test set 예측."""
    return _eval_dual(build_late_fusion, "LateFusion", "bootstrap_lf",
                      X_clin_cv, X_aec_cv, y_cv, X_clin_te, X_aec_te, y_te, sex_te,
                      med_epoch, scale_aec, threshold, weight_path)


def evaluate_test_late_fusion3(X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv,
                               X_clin_te, X_aec_te, X_scan_mfr_te, y_te,
                               sex_te, med_epoch, n_manufacturers,
                               scale_aec="column", threshold=0.5, weight_path=None):
    """전체 CV 세트로 LateFusion3 최종 모델 학습 후 test set 예측."""
    return _eval_quad(build_late_fusion3, "LateFusion3", "bootstrap_lf3",
                      X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv,
                      X_clin_te, X_aec_te, X_scan_mfr_te, y_te,
                      sex_te, med_epoch, n_manufacturers, scale_aec, threshold, weight_path)
