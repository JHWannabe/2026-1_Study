"""
최종 Test Set 평가.

CV 전체 세트로 모델을 재학습한 뒤 test set에서 예측한다.

  evaluate_test        — M1: LR + ResNet1D
  evaluate_test_cross  — M2/M2_2: LR + CrossAttn + ResNet1D
  evaluate_test_cross3 — M3: LR + CrossAttn3 + ResNet1D

Deep model 학습 epoch:
  CV fold별 best val AUC epoch의 중앙값(median)을 사용한다.
  최댓값 대신 중앙값을 쓰는 이유: 특정 fold의 이상치 epoch에 과적합되는 것을 방지.

스케일링 정책:
  Clinical: _scale_clin_te — Age(col 0)·BMI(col 2)만 표준화, sex_enc(col 1) 제외
  AEC     : _scale_or_copy — 전 컬럼 표준화
  CV 세트 전체로 scaler를 fit 후 test set에 transform만 적용 (누수 방지).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, classification_report,
    average_precision_score, brier_score_loss,
)

from config import SEED
from metrics import print_bootstrap_ci, print_delong_comparison
from models import (build_resnet, make_loaders, train_one_epoch, eval_loader,
                    build_cross_attn, make_dual_loaders, train_cross_epoch, eval_cross_loader,
                    build_cross_attn3, make_quad_loaders, train_cross3_epoch, eval_cross3_loader)


def _print_by_sex(y_true, y_pred, y_prob, sex):
    """성별(M/F)별 분류 지표(AUC·AUPRC·Brier·Acc·F1)와 classification_report를 콘솔 출력."""
    for s in ["M", "F"]:
        mask = sex == s
        if mask.sum() == 0:
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


def _scale_or_copy(X_tr, X_te, do_scale):
    """do_scale=True이면 StandardScaler로 fit_transform/transform, False이면 복사본 반환."""
    if do_scale:
        sc = StandardScaler()
        return sc.fit_transform(X_tr), sc.transform(X_te)
    return X_tr.copy(), X_te.copy()


def _scale_clin_te(X_cv, X_te, do_scale):
    """Clinical 스케일링: do_scale=True이면 Age(col 0)·BMI(col 2)만 표준화. sex_enc(col 1)은 그대로."""
    if not do_scale:
        return X_cv.copy(), X_te.copy()
    sc = StandardScaler()
    X_cv_s = X_cv.copy()
    X_te_s = X_te.copy()
    X_cv_s[:, [0, 2]] = sc.fit_transform(X_cv[:, [0, 2]])
    X_te_s[:, [0, 2]] = sc.transform(X_te[:, [0, 2]])
    return X_cv_s, X_te_s


def evaluate_test(X_cv, y_cv, X_te, y_te, sex_te, med_epoch, scale_X=True):
    """
    전체 CV 세트로 LR·ResNet1D 최종 모델을 학습하고 test set 예측 결과를 반환.

    ResNet1D는 CV에서 구한 median best epoch 수만큼 훈련 후 마지막 상태로 평가한다.
    Returns: lr_pred, lr_prob, rn_pred_te, rn_prob_te, rn_true_te
    """
    print(f"\n{'='*55}")
    print(f"Final Test Evaluation  (scale_X={scale_X})")
    print(f"{'='*55}")

    # Logistic Regression
    X_cv_s, X_te_s = _scale_clin_te(X_cv, X_te, scale_X)

    lr_final = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr_final.fit(X_cv_s, y_cv)
    lr_pred = lr_final.predict(X_te_s)
    lr_prob = lr_final.predict_proba(X_te_s)[:, 1]

    print(f"\nLogistic Regression — Test Set (Overall):")
    print(f"  AUC: {roc_auc_score(y_te, lr_prob):.4f}"
          f"  AUPRC: {average_precision_score(y_te, lr_prob):.4f}"
          f"  Brier: {brier_score_loss(y_te, lr_prob):.4f}"
          f"  Acc: {accuracy_score(y_te, lr_pred):.4f}"
          f"  F1: {f1_score(y_te, lr_pred):.4f}")
    print("\nLogistic Regression — Test Set (By Sex):")
    _print_by_sex(y_te, lr_pred, lr_prob, sex_te)

    # ResNet1D
    print(f"ResNet1D — training final model for {med_epoch} epochs on full CV set …")
    X_cv_rn, X_te_rn = _scale_clin_te(X_cv, X_te, scale_X)

    tr_dl, te_dl = make_loaders(X_cv_rn, y_cv, X_te_rn, y_te)
    model_f, crit_f, opt_f, sched_f = build_resnet(y_cv)

    # CV median best epoch 수만큼 학습 후 마지막 상태로 평가
    # (검증 세트 없으므로 early stopping 불가 → 고정 epoch 사용)
    for _ in range(1, med_epoch + 1):
        train_one_epoch(model_f, tr_dl, crit_f, opt_f)
        sched_f.step()

    _, rn_prob_te, rn_true_te = eval_loader(model_f, te_dl, crit_f)
    rn_pred_te = (rn_prob_te >= 0.5).astype(int)

    print(f"\nResNet1D — Test Set (Overall):")
    print(f"  AUC: {roc_auc_score(rn_true_te, rn_prob_te):.4f}"
          f"  AUPRC: {average_precision_score(rn_true_te, rn_prob_te):.4f}"
          f"  Brier: {brier_score_loss(rn_true_te, rn_prob_te):.4f}"
          f"  Acc: {accuracy_score(rn_true_te, rn_pred_te):.4f}"
          f"  F1: {f1_score(rn_true_te, rn_pred_te):.4f}")
    print("\nResNet1D — Test Set (By Sex):")
    _print_by_sex(rn_true_te, rn_pred_te, rn_prob_te, sex_te)

    # ── Bootstrap CI + DeLong ────────────────────────────────
    print(f"\n{'─'*55}")
    print("Test Set — Bootstrap CI & DeLong Comparison")
    print(f"{'─'*55}")
    ci_lr = print_bootstrap_ci("LR",       y_te, lr_pred,    lr_prob)
    ci_rn = print_bootstrap_ci("ResNet1D", rn_true_te, rn_pred_te, rn_prob_te)
    dl_lr_rn = print_delong_comparison("LR", "ResNet1D", y_te, lr_prob, rn_prob_te)

    stats_te = {
        "bootstrap_lr": ci_lr, "bootstrap_rn": ci_rn,
        "delong_lr_rn": dl_lr_rn,
    }
    return lr_pred, lr_prob, rn_pred_te, rn_prob_te, rn_true_te, stats_te


def evaluate_test_cross(X_clin_cv, X_aec_cv, y_cv,
                        X_clin_te, X_aec_te, y_te, sex_te,
                        med_epoch, rn_med_epoch=None,
                        scale_clin=True, scale_aec=True):
    """
    전체 CV 세트로 LR·ClinAECCrossAttn·ResNet1D 최종 모델을 학습하고 test set 예측 결과를 반환.

    LR/ResNet1D는 Clinic+AEC hstack 입력, CrossAttn은 별도 인코더를 사용한다.
    rn_med_epoch: ResNet1D용 median best epoch (None이면 med_epoch 사용).
    Returns: lr_pred, lr_prob, ca_pred_te, ca_prob_te, ca_true_te, rn_pred_te, rn_prob_te
    """
    rn_med_epoch = rn_med_epoch if rn_med_epoch is not None else med_epoch
    print(f"\n{'='*55}")
    print(f"CrossAttn Test Evaluation  (scale_clin={scale_clin}, scale_aec={scale_aec})")
    print(f"{'='*55}")

    X_clin_cv_s, X_clin_te_s = _scale_clin_te(X_clin_cv, X_clin_te, scale_clin)
    X_aec_cv_s,  X_aec_te_s  = _scale_or_copy(X_aec_cv,  X_aec_te,  scale_aec)

    # Logistic Regression (clinic + AEC 연결)
    X_lr_cv = np.hstack([X_clin_cv_s, X_aec_cv_s])
    X_lr_te = np.hstack([X_clin_te_s, X_aec_te_s])
    lr_final = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr_final.fit(X_lr_cv, y_cv)
    lr_pred = lr_final.predict(X_lr_te)
    lr_prob = lr_final.predict_proba(X_lr_te)[:, 1]

    print(f"\nLogistic Regression — Test Set (Overall):")
    print(f"  AUC: {roc_auc_score(y_te, lr_prob):.4f}"
          f"  AUPRC: {average_precision_score(y_te, lr_prob):.4f}"
          f"  Brier: {brier_score_loss(y_te, lr_prob):.4f}"
          f"  Acc: {accuracy_score(y_te, lr_pred):.4f}"
          f"  F1: {f1_score(y_te, lr_pred, zero_division=0):.4f}")
    print("\nLogistic Regression — Test Set (By Sex):")
    _print_by_sex(y_te, lr_pred, lr_prob, sex_te)

    tr_dl, te_dl = make_dual_loaders(X_clin_cv_s, X_aec_cv_s, y_cv,
                                     X_clin_te_s,  X_aec_te_s,  y_te)
    model_f, crit_f, opt_f, sched_f = build_cross_attn(y_cv)

    print(f"CrossAttn — training final model for {med_epoch} epochs on full CV set …")
    for _ in range(1, med_epoch + 1):
        train_cross_epoch(model_f, tr_dl, crit_f, opt_f)
        sched_f.step()

    _, ca_prob_te, ca_true_te = eval_cross_loader(model_f, te_dl, crit_f)
    ca_pred_te = (ca_prob_te >= 0.5).astype(int)

    print(f"\nCrossAttn — Test Set (Overall):")
    print(f"  AUC: {roc_auc_score(ca_true_te, ca_prob_te):.4f}"
          f"  AUPRC: {average_precision_score(ca_true_te, ca_prob_te):.4f}"
          f"  Brier: {brier_score_loss(ca_true_te, ca_prob_te):.4f}"
          f"  Acc: {accuracy_score(ca_true_te, ca_pred_te):.4f}"
          f"  F1: {f1_score(ca_true_te, ca_pred_te, zero_division=0):.4f}")
    print("\nCrossAttn — Test Set (By Sex):")
    _print_by_sex(ca_true_te, ca_pred_te, ca_prob_te, sex_te)

    # ResNet1D (clinic + AEC hstack, 이미 스케일됨)
    X_rn_cv = np.hstack([X_clin_cv_s, X_aec_cv_s])
    X_rn_te = np.hstack([X_clin_te_s, X_aec_te_s])

    tr_dl_rn, te_dl_rn = make_loaders(X_rn_cv, y_cv, X_rn_te, y_te)
    model_rn_f, crit_rn_f, opt_rn_f, sched_rn_f = build_resnet(y_cv)

    print(f"ResNet1D — training final model for {rn_med_epoch} epochs on full CV set …")
    for _ in range(1, rn_med_epoch + 1):
        train_one_epoch(model_rn_f, tr_dl_rn, crit_rn_f, opt_rn_f)
        sched_rn_f.step()

    _, rn_prob_te, rn_true_te = eval_loader(model_rn_f, te_dl_rn, crit_rn_f)
    rn_pred_te = (rn_prob_te >= 0.5).astype(int)

    print(f"\nResNet1D — Test Set (Overall):")
    print(f"  AUC: {roc_auc_score(rn_true_te, rn_prob_te):.4f}"
          f"  AUPRC: {average_precision_score(rn_true_te, rn_prob_te):.4f}"
          f"  Brier: {brier_score_loss(rn_true_te, rn_prob_te):.4f}"
          f"  Acc: {accuracy_score(rn_true_te, rn_pred_te):.4f}"
          f"  F1: {f1_score(rn_true_te, rn_pred_te, zero_division=0):.4f}")
    print("\nResNet1D — Test Set (By Sex):")
    _print_by_sex(rn_true_te, rn_pred_te, rn_prob_te, sex_te)

    # ── Bootstrap CI + DeLong ────────────────────────────────
    print(f"\n{'─'*55}")
    print("Test Set — Bootstrap CI & DeLong Comparison")
    print(f"{'─'*55}")
    ci_lr = print_bootstrap_ci("LR",        ca_true_te, lr_pred,    lr_prob)
    ci_ca = print_bootstrap_ci("CrossAttn", ca_true_te, ca_pred_te, ca_prob_te)
    ci_rn = print_bootstrap_ci("ResNet1D",  rn_true_te, rn_pred_te, rn_prob_te)
    dl_lr_ca  = print_delong_comparison("LR", "CrossAttn", ca_true_te, lr_prob,    ca_prob_te)
    dl_lr_rn  = print_delong_comparison("LR", "ResNet1D",  rn_true_te, lr_prob,    rn_prob_te)
    dl_ca_rn  = print_delong_comparison("CrossAttn", "ResNet1D", ca_true_te, ca_prob_te, rn_prob_te)

    stats_te = {
        "bootstrap_lr": ci_lr, "bootstrap_ca": ci_ca, "bootstrap_rn": ci_rn,
        "delong_lr_ca": dl_lr_ca, "delong_lr_rn": dl_lr_rn, "delong_ca_rn": dl_ca_rn,
    }
    return lr_pred, lr_prob, ca_pred_te, ca_prob_te, ca_true_te, rn_pred_te, rn_prob_te, stats_te


def evaluate_test_cross3(X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv,
                          X_clin_te, X_aec_te, X_scan_mfr_te, y_te,
                          sex_te, med_epoch, n_manufacturers,
                          rn_med_epoch=None,
                          scale_clin=True, scale_aec=True):
    """
    전체 CV 세트로 LR·ClinAECScanCrossAttn·ResNet1D 최종 모델을 학습하고 test set 예측 결과를 반환.

    ManufacturerModelName은 Embedding으로 처리하므로 스케일링하지 않는다.
    ResNet1D는 LR과 동일한 hstack 입력을 사용한다.
    rn_med_epoch: ResNet1D용 median best epoch (None이면 med_epoch 사용).
    Returns: lr_pred, lr_prob, ca3_pred_te, ca3_prob_te, ca3_true_te, rn_pred_te, rn_prob_te
    """
    rn_med_epoch = rn_med_epoch if rn_med_epoch is not None else med_epoch
    print(f"\n{'='*65}")
    print(f"CrossAttn3 Test Evaluation  (scale_clin={scale_clin}, scale_aec={scale_aec})")
    print(f"{'='*65}")

    X_clin_cv_s, X_clin_te_s = _scale_clin_te(X_clin_cv, X_clin_te, scale_clin)
    X_aec_cv_s,  X_aec_te_s  = _scale_or_copy(X_aec_cv,  X_aec_te,  scale_aec)

    # Logistic Regression (clinic + mfr + AEC 연결)
    X_lr_cv = np.hstack([X_clin_cv_s, X_scan_mfr_cv.reshape(-1, 1).astype(float), X_aec_cv_s])
    X_lr_te = np.hstack([X_clin_te_s, X_scan_mfr_te.reshape(-1, 1).astype(float), X_aec_te_s])
    lr_final = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr_final.fit(X_lr_cv, y_cv)
    lr_pred = lr_final.predict(X_lr_te)
    lr_prob = lr_final.predict_proba(X_lr_te)[:, 1]

    print(f"\nLogistic Regression — Test Set (Overall):")
    print(f"  AUC: {roc_auc_score(y_te, lr_prob):.4f}"
          f"  AUPRC: {average_precision_score(y_te, lr_prob):.4f}"
          f"  Brier: {brier_score_loss(y_te, lr_prob):.4f}"
          f"  Acc: {accuracy_score(y_te, lr_pred):.4f}"
          f"  F1: {f1_score(y_te, lr_pred, zero_division=0):.4f}")
    print("\nLogistic Regression — Test Set (By Sex):")
    _print_by_sex(y_te, lr_pred, lr_prob, sex_te)

    tr_dl, te_dl = make_quad_loaders(
        X_clin_cv_s, X_aec_cv_s, X_scan_mfr_cv, y_cv,
        X_clin_te_s, X_aec_te_s, X_scan_mfr_te, y_te,
    )
    model_f, crit_f, opt_f, sched_f = build_cross_attn3(y_cv, n_manufacturers)

    print(f"CrossAttn3 — training final model for {med_epoch} epochs on full CV set …")
    for _ in range(1, med_epoch + 1):
        train_cross3_epoch(model_f, tr_dl, crit_f, opt_f)
        sched_f.step()

    _, ca3_prob_te, ca3_true_te = eval_cross3_loader(model_f, te_dl, crit_f)
    ca3_pred_te = (ca3_prob_te >= 0.5).astype(int)

    print(f"\nCrossAttn3 — Test Set (Overall):")
    print(f"  AUC: {roc_auc_score(ca3_true_te, ca3_prob_te):.4f}"
          f"  AUPRC: {average_precision_score(ca3_true_te, ca3_prob_te):.4f}"
          f"  Brier: {brier_score_loss(ca3_true_te, ca3_prob_te):.4f}"
          f"  Acc: {accuracy_score(ca3_true_te, ca3_pred_te):.4f}"
          f"  F1: {f1_score(ca3_true_te, ca3_pred_te, zero_division=0):.4f}")
    print("\nCrossAttn3 — Test Set (By Sex):")
    _print_by_sex(ca3_true_te, ca3_pred_te, ca3_prob_te, sex_te)

    # ResNet1D (clinic + mfr + AEC hstack, 이미 스케일됨)
    X_rn_cv = np.hstack([X_clin_cv_s, X_scan_mfr_cv.reshape(-1, 1).astype(float), X_aec_cv_s])
    X_rn_te = np.hstack([X_clin_te_s, X_scan_mfr_te.reshape(-1, 1).astype(float), X_aec_te_s])

    tr_dl_rn, te_dl_rn = make_loaders(X_rn_cv, y_cv, X_rn_te, y_te)
    model_rn_f, crit_rn_f, opt_rn_f, sched_rn_f = build_resnet(y_cv)

    print(f"ResNet1D — training final model for {rn_med_epoch} epochs on full CV set …")
    for _ in range(1, rn_med_epoch + 1):
        train_one_epoch(model_rn_f, tr_dl_rn, crit_rn_f, opt_rn_f)
        sched_rn_f.step()

    _, rn_prob_te, rn_true_te = eval_loader(model_rn_f, te_dl_rn, crit_rn_f)
    rn_pred_te = (rn_prob_te >= 0.5).astype(int)

    print(f"\nResNet1D — Test Set (Overall):")
    print(f"  AUC: {roc_auc_score(rn_true_te, rn_prob_te):.4f}"
          f"  AUPRC: {average_precision_score(rn_true_te, rn_prob_te):.4f}"
          f"  Brier: {brier_score_loss(rn_true_te, rn_prob_te):.4f}"
          f"  Acc: {accuracy_score(rn_true_te, rn_pred_te):.4f}"
          f"  F1: {f1_score(rn_true_te, rn_pred_te, zero_division=0):.4f}")
    print("\nResNet1D — Test Set (By Sex):")
    _print_by_sex(rn_true_te, rn_pred_te, rn_prob_te, sex_te)

    # ── Bootstrap CI + DeLong ────────────────────────────────
    print(f"\n{'─'*55}")
    print("Test Set — Bootstrap CI & DeLong Comparison")
    print(f"{'─'*55}")
    ci_lr  = print_bootstrap_ci("LR",          ca3_true_te, lr_pred,     lr_prob)
    ci_ca3 = print_bootstrap_ci("CrossAttn3",  ca3_true_te, ca3_pred_te, ca3_prob_te)
    ci_rn  = print_bootstrap_ci("ResNet1D",    rn_true_te,  rn_pred_te,  rn_prob_te)
    dl_lr_ca3 = print_delong_comparison("LR", "CrossAttn3", ca3_true_te, lr_prob,     ca3_prob_te)
    dl_lr_rn  = print_delong_comparison("LR", "ResNet1D",   rn_true_te,  lr_prob,     rn_prob_te)
    dl_ca3_rn = print_delong_comparison("CrossAttn3", "ResNet1D", ca3_true_te, ca3_prob_te, rn_prob_te)

    stats_te = {
        "bootstrap_lr": ci_lr, "bootstrap_ca3": ci_ca3, "bootstrap_rn": ci_rn,
        "delong_lr_ca3": dl_lr_ca3, "delong_lr_rn": dl_lr_rn, "delong_ca3_rn": dl_ca3_rn,
    }
    return lr_pred, lr_prob, ca3_pred_te, ca3_prob_te, ca3_true_te, rn_pred_te, rn_prob_te, stats_te
