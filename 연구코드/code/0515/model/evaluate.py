import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, classification_report,
    average_precision_score, brier_score_loss,
)

from config import SEED
from models import (build_resnet, make_loaders, train_one_epoch, eval_loader,
                    build_cross_attn, make_dual_loaders, train_cross_epoch, eval_cross_loader,
                    build_cross_attn3, make_quad_loaders, train_cross3_epoch, eval_cross3_loader)


def _print_by_sex(y_true, y_pred, y_prob, sex):
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
    if do_scale:
        sc = StandardScaler()
        return sc.fit_transform(X_tr), sc.transform(X_te)
    return X_tr.copy(), X_te.copy()


def evaluate_test(X_cv, y_cv, X_te, y_te, sex_te, med_epoch, scale_X=True):
    print(f"\n{'='*55}")
    print(f"Final Test Evaluation  (scale_X={scale_X})")
    print(f"{'='*55}")

    # Logistic Regression
    X_cv_s, X_te_s = _scale_or_copy(X_cv, X_te, scale_X)

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
    X_cv_rn, X_te_rn = _scale_or_copy(X_cv, X_te, scale_X)

    tr_dl, te_dl = make_loaders(X_cv_rn, y_cv, X_te_rn, y_te)
    model_f, crit_f, opt_f, sched_f = build_resnet(y_cv)

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

    return lr_pred, lr_prob, rn_pred_te, rn_prob_te, rn_true_te


def evaluate_test_cross(X_clin_cv, X_aec_cv, y_cv,
                        X_clin_te, X_aec_te, y_te, sex_te, med_epoch,
                        scale_clin=True, scale_aec=True):
    print(f"\n{'='*55}")
    print(f"CrossAttn Test Evaluation  (scale_clin={scale_clin}, scale_aec={scale_aec})")
    print(f"{'='*55}")

    X_clin_cv_s, X_clin_te_s = _scale_or_copy(X_clin_cv, X_clin_te, scale_clin)
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

    return lr_pred, lr_prob, ca_pred_te, ca_prob_te, ca_true_te


def evaluate_test_cross3(X_clin_cv, X_aec_cv, X_scan_kvp_cv, X_scan_mfr_cv, y_cv,
                          X_clin_te, X_aec_te, X_scan_kvp_te, X_scan_mfr_te, y_te,
                          sex_te, med_epoch, n_manufacturers,
                          scale_clin=True, scale_aec=True, scale_scan=True):
    print(f"\n{'='*65}")
    print(f"CrossAttn3 Test Evaluation  "
          f"(scale_clin={scale_clin}, scale_aec={scale_aec}, scale_scan={scale_scan})")
    print(f"{'='*65}")

    X_clin_cv_s, X_clin_te_s = _scale_or_copy(X_clin_cv, X_clin_te, scale_clin)
    X_aec_cv_s,  X_aec_te_s  = _scale_or_copy(X_aec_cv,  X_aec_te,  scale_aec)

    # kVp: scale_scan 플래그로 독립 제어
    kvp_cv_s, kvp_te_s = _scale_or_copy(
        X_scan_kvp_cv.reshape(-1, 1), X_scan_kvp_te.reshape(-1, 1), scale_scan
    )
    kvp_cv_s, kvp_te_s = kvp_cv_s.ravel(), kvp_te_s.ravel()

    # Logistic Regression (clinic + kVp + mfr + AEC 연결)
    X_lr_cv = np.hstack([X_clin_cv_s, kvp_cv_s.reshape(-1, 1),
                          X_scan_mfr_cv.reshape(-1, 1).astype(float), X_aec_cv_s])
    X_lr_te = np.hstack([X_clin_te_s, kvp_te_s.reshape(-1, 1),
                          X_scan_mfr_te.reshape(-1, 1).astype(float), X_aec_te_s])
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
        X_clin_cv_s, X_aec_cv_s, kvp_cv_s, X_scan_mfr_cv, y_cv,
        X_clin_te_s, X_aec_te_s, kvp_te_s, X_scan_mfr_te, y_te,
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

    return lr_pred, lr_prob, ca3_pred_te, ca3_prob_te, ca3_true_te
