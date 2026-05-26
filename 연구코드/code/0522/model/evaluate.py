"""
최종 Test Set 평가.

CV 전체 세트로 모델을 재학습한 뒤 test set에서 예측한다.

  evaluate_test        — M1: LR
  evaluate_test_cross  — M2/M2_2: CrossAttn
  evaluate_test_cross3 — M3: CrossAttn3

Deep model 학습 epoch:
  CV fold별 best val AUC epoch의 중앙값(median)을 사용한다.
  최댓값 대신 중앙값을 쓰는 이유: 특정 fold의 이상치 epoch에 과적합되는 것을 방지.

스케일링 정책:
  Clinical: _scale_clin_te — Age(col 0)·BMI(col 2)만 표준화, sex_enc(col 1) 제외
  AEC     : _scale_or_copy — 전 컬럼 표준화
  MFR     : 스케일링 없음 — nn.Embedding index(1-indexed 정수)이므로 표준화 불필요
  CV 세트 전체로 scaler를 fit 후 test set에 transform만 적용 (누수 방지).

return_model 옵션:
  evaluate_test_cross / evaluate_test_cross3 에서 return_model=True 를 전달하면
  (pred, prob, true, stats, model, X_clin_te_s, X_aec_te_s) 7-tuple 을 반환한다.
  스케일된 test 배열을 함께 반환해 호출 측에서 attention map 시각화에 바로 사용할 수 있다.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, classification_report,
    average_precision_score, brier_score_loss,
)

from config import SEED
from metrics import print_bootstrap_ci
from models import (build_cross_attn, make_dual_loaders, train_cross_epoch, eval_cross_loader,
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


def evaluate_test(X_cv, y_cv, X_te, y_te, sex_te, scale_X=True, threshold=0.5):
    """
    전체 CV 세트로 LR 최종 모델을 학습하고 test set 예측 결과를 반환.

    threshold: CV fold별 Youden's J 최적값의 중앙값 (기본 0.5)
    Returns: lr_pred, lr_prob, stats_te
    """
    print(f"\n{'='*55}")
    print(f"Final Test Evaluation  (scale_X={scale_X}, threshold={threshold:.3f})")
    print(f"{'='*55}")

    X_cv_s, X_te_s = _scale_clin_te(X_cv, X_te, scale_X)

    lr_final = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr_final.fit(X_cv_s, y_cv)
    lr_prob = lr_final.predict_proba(X_te_s)[:, 1]
    lr_pred = (lr_prob >= threshold).astype(int)

    print(f"\nLogistic Regression — Test Set (Overall):")
    print(f"  AUC: {roc_auc_score(y_te, lr_prob):.4f}"
          f"  AUPRC: {average_precision_score(y_te, lr_prob):.4f}"
          f"  Brier: {brier_score_loss(y_te, lr_prob):.4f}"
          f"  Acc: {accuracy_score(y_te, lr_pred):.4f}"
          f"  F1: {f1_score(y_te, lr_pred):.4f}")
    print("\nLogistic Regression — Test Set (By Sex):")
    _print_by_sex(y_te, lr_pred, lr_prob, sex_te)

    # ── Bootstrap CI ────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Test Set — Bootstrap CI")
    print(f"{'─'*55}")
    ci_lr = print_bootstrap_ci("LR", y_te, lr_pred, lr_prob)

    stats_te = {"bootstrap_lr": ci_lr}
    return lr_pred, lr_prob, stats_te


def evaluate_test_cross(X_clin_cv, X_aec_cv, y_cv,
                        X_clin_te, X_aec_te, y_te, sex_te,
                        med_epoch,
                        scale_clin=True,
                        return_model=False, threshold=0.5):
    """
    전체 CV 세트로 ClinAECCrossAttn 최종 모델을 학습하고 test set 예측 결과를 반환.
    AEC는 항상 StandardScaler로 표준화한다.

    threshold: CV fold별 Youden's J 최적값의 중앙값 (기본 0.5)
    return_model=False (기본): ca_pred_te, ca_prob_te, ca_true_te, stats_te
    return_model=True        : 위 4개 + model, X_clin_te_s, X_aec_te_s (attention map용)
    """
    print(f"\n{'='*55}")
    print(f"CrossAttn Test Evaluation  (scale_clin={scale_clin}, scale_aec=True, threshold={threshold:.3f})")
    print(f"{'='*55}")

    X_clin_cv_s, X_clin_te_s = _scale_clin_te(X_clin_cv, X_clin_te, scale_clin)
    X_aec_cv_s,  X_aec_te_s  = _scale_or_copy(X_aec_cv,  X_aec_te,  True)

    tr_dl, te_dl = make_dual_loaders(X_clin_cv_s, X_aec_cv_s, y_cv,
                                     X_clin_te_s,  X_aec_te_s,  y_te)
    model_f, crit_f, opt_f, sched_f = build_cross_attn(y_cv)

    print(f"CrossAttn — training final model for {med_epoch} epochs on full CV set …")
    for _ in range(1, med_epoch + 1):
        train_cross_epoch(model_f, tr_dl, crit_f, opt_f)
        sched_f.step()

    _, ca_prob_te, ca_true_te = eval_cross_loader(model_f, te_dl, crit_f)
    ca_pred_te = (ca_prob_te >= threshold).astype(int)

    print(f"\nCrossAttn — Test Set (Overall):")
    print(f"  AUC: {roc_auc_score(ca_true_te, ca_prob_te):.4f}"
          f"  AUPRC: {average_precision_score(ca_true_te, ca_prob_te):.4f}"
          f"  Brier: {brier_score_loss(ca_true_te, ca_prob_te):.4f}"
          f"  Acc: {accuracy_score(ca_true_te, ca_pred_te):.4f}"
          f"  F1: {f1_score(ca_true_te, ca_pred_te, zero_division=0):.4f}")
    print("\nCrossAttn — Test Set (By Sex):")
    _print_by_sex(ca_true_te, ca_pred_te, ca_prob_te, sex_te)

    # ── Bootstrap CI ────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Test Set — Bootstrap CI")
    print(f"{'─'*55}")
    ci_ca = print_bootstrap_ci("CrossAttn", ca_true_te, ca_pred_te, ca_prob_te)

    stats_te = {"bootstrap_ca": ci_ca}
    if return_model:
        return ca_pred_te, ca_prob_te, ca_true_te, stats_te, model_f, X_clin_te_s, X_aec_te_s
    return ca_pred_te, ca_prob_te, ca_true_te, stats_te


def evaluate_test_cross3(X_clin_cv, X_aec_cv, X_scan_mfr_cv, y_cv,
                          X_clin_te, X_aec_te, X_scan_mfr_te, y_te,
                          sex_te, med_epoch, n_manufacturers,
                          scale_clin=True,
                          return_model=False, threshold=0.5):
    """
    전체 CV 세트로 ClinAECScanCrossAttn 최종 모델을 학습하고 test set 예측 결과를 반환.
    AEC는 항상 StandardScaler로 표준화한다.

    threshold: CV fold별 Youden's J 최적값의 중앙값 (기본 0.5)
    ManufacturerModelName은 nn.Embedding index(1-indexed 정수)이므로 스케일링하지 않는다.
    return_model=False (기본): ca3_pred_te, ca3_prob_te, ca3_true_te, stats_te
    return_model=True        : 위 4개 + model, X_clin_te_s, X_aec_te_s (attention map용)
    """
    print(f"\n{'='*65}")
    print(f"CrossAttn3 Test Evaluation  (scale_clin={scale_clin}, scale_aec=True, threshold={threshold:.3f})")
    print(f"{'='*65}")

    X_clin_cv_s, X_clin_te_s = _scale_clin_te(X_clin_cv, X_clin_te, scale_clin)
    X_aec_cv_s,  X_aec_te_s  = _scale_or_copy(X_aec_cv,  X_aec_te,  True)

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
    ca3_pred_te = (ca3_prob_te >= threshold).astype(int)

    print(f"\nCrossAttn3 — Test Set (Overall):")
    print(f"  AUC: {roc_auc_score(ca3_true_te, ca3_prob_te):.4f}"
          f"  AUPRC: {average_precision_score(ca3_true_te, ca3_prob_te):.4f}"
          f"  Brier: {brier_score_loss(ca3_true_te, ca3_prob_te):.4f}"
          f"  Acc: {accuracy_score(ca3_true_te, ca3_pred_te):.4f}"
          f"  F1: {f1_score(ca3_true_te, ca3_pred_te, zero_division=0):.4f}")
    print("\nCrossAttn3 — Test Set (By Sex):")
    _print_by_sex(ca3_true_te, ca3_pred_te, ca3_prob_te, sex_te)

    # ── Bootstrap CI ────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Test Set — Bootstrap CI")
    print(f"{'─'*55}")
    ci_ca3 = print_bootstrap_ci("CrossAttn3", ca3_true_te, ca3_pred_te, ca3_prob_te)

    stats_te = {"bootstrap_ca3": ci_ca3}
    if return_model:
        return ca3_pred_te, ca3_prob_te, ca3_true_te, stats_te, model_f, X_clin_te_s, X_aec_te_s
    return ca3_pred_te, ca3_prob_te, ca3_true_te, stats_te
