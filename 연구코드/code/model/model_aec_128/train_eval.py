import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, f1_score,
    classification_report, average_precision_score, brier_score_loss,
)

from config import N_FOLDS, SEED, DEVICE, EPOCHS, LR, BATCH_SIZE, PATIENCE, AEC_NORM, AEC_OUT
from models import AECFusionModel
from metrics import group_metrics, bootstrap_ci_md


# ── Normalization ─────────────────────────────────────────────

def _row_norm(X):
    mu  = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mu) / std


def _apply_norm_aec(X_tr, X_val, norm):
    """AEC 피처 정규화 (3-mode)."""
    if norm == "global":
        sc = StandardScaler()
        return sc.fit_transform(X_tr), sc.transform(X_val)
    elif norm == "rowwise":
        return _row_norm(X_tr.copy()), _row_norm(X_val.copy())
    else:  # raw
        return X_tr.copy(), X_val.copy()


def _apply_norm_clinic(X_tr, X_val):
    """Clinic 피처는 항상 global StandardScaler."""
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_val)


# ── DataLoader & Training ─────────────────────────────────────

def _make_loader(X_aec, X_clinic, y, shuffle):
    X_t = torch.tensor(X_aec[:, None, :], dtype=torch.float32)   # (N, 1, 128)
    C_t = torch.tensor(X_clinic,           dtype=torch.float32)   # (N, 3)
    y_t = torch.tensor(y,                  dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, C_t, y_t), batch_size=BATCH_SIZE, shuffle=shuffle)


def _train_model(X_aec_tr, X_clinic_tr, y_tr, X_aec_val, X_clinic_val, y_val):
    pos_weight = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], device=DEVICE
    )
    model = AECFusionModel(n_clinic=X_clinic_tr.shape[1], aec_out=AEC_OUT).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)

    tr_loader  = _make_loader(X_aec_tr,  X_clinic_tr,  y_tr,  shuffle=True)
    val_loader = _make_loader(X_aec_val, X_clinic_val, y_val, shuffle=False)

    best_auc, no_improve, best_state = 0.0, 0, None
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for _ in range(EPOCHS):
        model.train()
        batch_losses = []
        for Xb, Cb, yb in tr_loader:
            Xb, Cb, yb = Xb.to(DEVICE), Cb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb, Cb), yb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        history["train_loss"].append(float(np.mean(batch_losses)))

        model.eval()
        probs, val_batch_losses = [], []
        with torch.no_grad():
            for Xb, Cb, yb in val_loader:
                Xb, Cb, yb = Xb.to(DEVICE), Cb.to(DEVICE), yb.to(DEVICE)
                out = model(Xb, Cb)
                val_batch_losses.append(crit(out, yb).item())
                probs.append(torch.sigmoid(out).cpu().numpy())
        probs = np.concatenate(probs)
        history["val_loss"].append(float(np.mean(val_batch_losses)))

        auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.0
        history["val_auc"].append(auc)
        if auc > best_auc:
            best_auc   = auc
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

        sched.step()

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, history


def _predict(model, X_aec, X_clinic):
    dummy = np.zeros(len(X_aec), dtype=np.float32)
    loader = _make_loader(X_aec, X_clinic, dummy, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for Xb, Cb, _ in loader:
            probs.append(
                torch.sigmoid(model(Xb.to(DEVICE), Cb.to(DEVICE))).cpu().numpy()
            )
    return np.concatenate(probs)


# ── CV & Evaluation ───────────────────────────────────────────

def run_cross_validation(X_aec_cv, X_clinic_cv, y_cv, sex_cv, norm):
    """AECFusionModel 5-Fold CV. (cv_results, roc_folds, best_thresholds, md_text) 반환."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_results, roc_folds, best_thresholds, fold_histories = [], [], [], []
    fold_lines = [
        f"## AECFusionModel [{norm}] — {N_FOLDS}-Fold CV Summary\n",
        "| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 | Threshold |",
        "|------|---------|-------|-------|----------|----|-----------|",
    ]

    for fold, (tr_i, val_i) in enumerate(skf.split(X_aec_cv, y_cv), 1):
        X_aec_tr,  X_aec_val  = _apply_norm_aec(X_aec_cv[tr_i], X_aec_cv[val_i], norm)
        X_clinic_tr, X_clinic_val = _apply_norm_clinic(X_clinic_cv[tr_i], X_clinic_cv[val_i])
        y_tr, y_val = y_cv[tr_i], y_cv[val_i]

        model, history = _train_model(X_aec_tr, X_clinic_tr, y_tr, X_aec_val, X_clinic_val, y_val)
        prob  = _predict(model, X_aec_val, X_clinic_val)

        fpr, tpr, thresholds = roc_curve(y_val, prob)
        best_thresh = float(thresholds[np.argmax(tpr - fpr)])
        pred = (prob >= best_thresh).astype(int)
        best_thresholds.append(best_thresh)

        m = group_metrics(y_val, pred, prob)
        cv_results.append({"fold": fold, **m})
        roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m["auc"]})

        fold_lines.append(
            f"| {fold} | {m['auc']:.4f} | {m['auprc']:.4f} |"
            f" {m['brier']:.4f} | {m['acc']:.4f} | {m['f1']:.4f} | {best_thresh:.3f} |"
        )
        fold_histories.append(history)

    keys = ["auc", "auprc", "brier", "acc", "f1"]
    vals = {k: [m[k] for m in cv_results] for k in keys}
    fold_lines.append(
        f"| **Mean** | {np.mean(vals['auc']):.4f} | {np.mean(vals['auprc']):.4f} |"
        f" {np.mean(vals['brier']):.4f} | {np.mean(vals['acc']):.4f} | {np.mean(vals['f1']):.4f} | — |"
    )
    fold_lines.append(
        f"| **±Std** | {np.std(vals['auc']):.4f} | {np.std(vals['auprc']):.4f} |"
        f" {np.std(vals['brier']):.4f} | {np.std(vals['acc']):.4f} | {np.std(vals['f1']):.4f} | — |"
    )

    return cv_results, roc_folds, best_thresholds, "\n".join(fold_lines), fold_histories


def evaluate_test(X_aec_cv, X_clinic_cv, y_cv, sex_cv,
                  X_aec_te, X_clinic_te, y_te, sex_te, norm, threshold=0.5):
    """전체 CV 세트로 AECFusionModel 최종 학습 후 test set 평가."""
    X_aec_cv_s, X_aec_te_s     = _apply_norm_aec(X_aec_cv, X_aec_te, norm)
    X_clinic_cv_s, X_clinic_te_s = _apply_norm_clinic(X_clinic_cv, X_clinic_te)

    # early stopping용 internal val split (test set 미사용)
    idx = np.arange(len(y_cv))
    tr_i, val_i = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=y_cv)

    model, final_history = _train_model(
        X_aec_cv_s[tr_i], X_clinic_cv_s[tr_i], y_cv[tr_i],
        X_aec_cv_s[val_i], X_clinic_cv_s[val_i], y_cv[val_i],
    )
    prob = _predict(model, X_aec_te_s, X_clinic_te_s)
    pred = (prob >= threshold).astype(int)

    lines = [f"## Test Set Evaluation\n\n**Threshold:** {threshold:.3f}\n"]
    lines += [
        "### Overall\n",
        "| AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
        "|---------|-------|-------|----------|----|",
        f"| {roc_auc_score(y_te, prob):.4f}"
        f" | {average_precision_score(y_te, prob):.4f}"
        f" | {brier_score_loss(y_te, prob):.4f}"
        f" | {accuracy_score(y_te, pred):.4f}"
        f" | {f1_score(y_te, pred, zero_division=0):.4f} |",
    ]

    for s in ["M", "F"]:
        mask = sex_te == s
        if not mask.any():
            continue
        yt, yp, ypr = y_te[mask], pred[mask], prob[mask]
        has_both = len(np.unique(yt)) > 1
        auc   = roc_auc_score(yt, ypr)           if has_both else float("nan")
        auprc = average_precision_score(yt, ypr) if has_both else float("nan")
        brier = brier_score_loss(yt, ypr)        if has_both else float("nan")
        sex_label = "Male" if s == "M" else "Female"
        lines += [
            f"\n### {sex_label} (n={mask.sum()})\n",
            "| AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
            "|---------|-------|-------|----------|----|",
            f"| {auc:.4f} | {auprc:.4f} | {brier:.4f}"
            f" | {accuracy_score(yt, yp):.4f} | {f1_score(yt, yp, zero_division=0):.4f} |",
            f"\n```\n{classification_report(yt, yp, target_names=['Normal', 'Sarcopenia'], zero_division=0)}```",
        ]

    ci, ci_text = bootstrap_ci_md("AECFusion", y_te, pred, prob)
    lines.append("\n" + ci_text)

    return pred, prob, ci, "\n".join(lines), model, X_aec_te_s, X_clinic_te_s, final_history


def compute_grad_cam(model, X_aec_s, X_clinic_s, batch_size=32):
    """Grad-CAM over the last ResBlock for each test sample.

    Batch-wise computation (eval mode: BatchNorm uses running stats → samples independent).
    Returns cam_maps: (N, 128) np.ndarray, ReLU'd and max-normalized per sample.
    """
    model.eval()
    all_cams = []

    for start in range(0, len(X_aec_s), batch_size):
        X_b = torch.tensor(
            X_aec_s[start:start + batch_size, None, :], dtype=torch.float32
        ).to(DEVICE)
        C_b = torch.tensor(
            X_clinic_s[start:start + batch_size], dtype=torch.float32
        ).to(DEVICE)

        saved_act = [None]

        def fwd_hook(m, inp, out):
            saved_act[0] = out

        h = model.blocks[-1].register_forward_hook(fwd_hook)
        with torch.enable_grad():
            logits = model(X_b, C_b)
        h.remove()

        act = saved_act[0]
        assert act is not None, "Forward hook did not capture activation"
        grads = torch.autograd.grad(logits.sum(), act)[0]        # (B, base_ch, 128)

        weights = grads.mean(dim=2, keepdim=True)                # (B, base_ch, 1)
        cam = torch.relu((weights * act.detach()).sum(dim=1))    # (B, 128)
        cam = cam.cpu().numpy()

        mx = cam.max(axis=1, keepdims=True)
        mx[mx == 0] = 1.0
        all_cams.append(cam / mx)

    return np.concatenate(all_cams, axis=0)  # (N, 128)
