import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from config import N_FOLDS, SEED, EPOCHS
from sklearn.linear_model import LogisticRegression
from models import (build_resnet, make_loaders, train_one_epoch, eval_loader,
                    build_cross_attn, make_dual_loaders, train_cross_epoch, eval_cross_loader,
                    build_cross_attn3, make_quad_loaders, train_cross3_epoch, eval_cross3_loader)
from metrics import group_metrics


def _maybe_scale(sc, X_tr, X_val, do_scale):
    if do_scale:
        return sc.fit_transform(X_tr), sc.transform(X_val)
    return X_tr.copy(), X_val.copy()


def run_cross_validation(X_cv, y_cv, scale_X=True):
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
        Xtr_s, Xval_s = _maybe_scale(StandardScaler(), X_ftr, X_fval, scale_X)

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
        Xtr_rn, Xval_rn = _maybe_scale(StandardScaler(), X_ftr, X_fval, scale_X)

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

        print(f"  RN1D — AUC: {m_rn['auc']:.4f}  AUPRC: {m_rn['auprc']:.4f}"
              f"  Brier: {m_rn['brier']:.4f}  Acc: {m_rn['acc']:.4f}  F1: {m_rn['f1']:.4f}"
              f"  (best ep={best_epoch})")

    return lr_cv, rn_cv, lr_roc_folds, rn_roc_folds, rn_histories, best_epochs


def run_cross_validation_cross(X_clin_cv, X_aec_cv, y_cv,
                               scale_clin=True, scale_aec=True):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    lr_cv,  lr_roc_folds  = [], []
    ca_cv, ca_roc_folds, ca_histories, best_epochs = [], [], [], []

    print("=" * 55)
    print(f"{N_FOLDS}-Fold CV  [CrossAttn | scale_clin={scale_clin}, scale_aec={scale_aec}]")
    print("=" * 55)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")

        X_clin_tr,  X_clin_val  = X_clin_cv[tr_i],  X_clin_cv[val_i]
        X_aec_tr,   X_aec_val   = X_aec_cv[tr_i],   X_aec_cv[val_i]
        y_tr,       y_val        = y_cv[tr_i],        y_cv[val_i]

        X_clin_tr, X_clin_val = _maybe_scale(StandardScaler(), X_clin_tr, X_clin_val, scale_clin)
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
        best_epochs.append(best_epoch)

        print(f"  CA   — AUC: {m_ca['auc']:.4f}  AUPRC: {m_ca['auprc']:.4f}"
              f"  Brier: {m_ca['brier']:.4f}  Acc: {m_ca['acc']:.4f}  F1: {m_ca['f1']:.4f}"
              f"  (best ep={best_epoch})")

    return lr_cv, ca_cv, lr_roc_folds, ca_roc_folds, ca_histories, best_epochs


def run_cross_validation_cross3(X_clin_cv, X_aec_cv, X_scan_kvp_cv, X_scan_mfr_cv,
                                 y_cv, n_manufacturers,
                                 scale_clin=True, scale_aec=True, scale_scan=True):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lr_cv,  lr_roc_folds  = [], []
    ca3_cv, ca3_roc_folds, ca3_histories, best_epochs = [], [], [], []

    print("=" * 65)
    print(f"{N_FOLDS}-Fold CV  [CrossAttn3 | "
          f"scale_clin={scale_clin}, scale_aec={scale_aec}, scale_scan={scale_scan}]")
    print("=" * 65)

    for fold, (tr_i, val_i) in enumerate(skf.split(X_clin_cv, y_cv), 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────────")

        X_clin_tr,  X_clin_val  = X_clin_cv[tr_i],     X_clin_cv[val_i]
        X_aec_tr,   X_aec_val   = X_aec_cv[tr_i],      X_aec_cv[val_i]
        X_kvp_tr,   X_kvp_val   = X_scan_kvp_cv[tr_i], X_scan_kvp_cv[val_i]
        X_mfr_tr,   X_mfr_val   = X_scan_mfr_cv[tr_i], X_scan_mfr_cv[val_i]
        y_tr,       y_val        = y_cv[tr_i],           y_cv[val_i]

        X_clin_tr, X_clin_val = _maybe_scale(StandardScaler(), X_clin_tr, X_clin_val, scale_clin)
        X_aec_tr,  X_aec_val  = _maybe_scale(StandardScaler(), X_aec_tr,  X_aec_val,  scale_aec)

        # kVp: 연속형 스캐너 파라미터 — scale_scan 플래그로 독립 제어
        kvp_tr_2d, kvp_val_2d = _maybe_scale(
            StandardScaler(),
            X_kvp_tr.reshape(-1, 1), X_kvp_val.reshape(-1, 1),
            scale_scan,
        )
        X_kvp_tr, X_kvp_val = kvp_tr_2d.ravel(), kvp_val_2d.ravel()

        # ── Logistic Regression (clinic + scan + AEC 연결) ────
        X_lr_tr  = np.hstack([X_clin_tr,  X_kvp_tr.reshape(-1,1),
                               X_mfr_tr.reshape(-1,1).astype(float),  X_aec_tr])
        X_lr_val = np.hstack([X_clin_val, X_kvp_val.reshape(-1,1),
                               X_mfr_val.reshape(-1,1).astype(float), X_aec_val])

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
            X_clin_tr, X_aec_tr, X_kvp_tr, X_mfr_tr, y_tr,
            X_clin_val, X_aec_val, X_kvp_val, X_mfr_val, y_val,
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
        best_epochs.append(best_epoch)

        print(f"  CA3  — AUC: {m_ca3['auc']:.4f}  AUPRC: {m_ca3['auprc']:.4f}"
              f"  Brier: {m_ca3['brier']:.4f}  Acc: {m_ca3['acc']:.4f}  F1: {m_ca3['f1']:.4f}"
              f"  (best ep={best_epoch})")

    return lr_cv, ca3_cv, lr_roc_folds, ca3_roc_folds, ca3_histories, best_epochs
