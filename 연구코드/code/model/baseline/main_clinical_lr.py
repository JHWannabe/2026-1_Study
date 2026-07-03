import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from config import DATA_PATH, SEED, RESULTS_MODEL_1_DIR, SPLIT_TRAIN_ID_PATH, SPLIT_TEST_ID_PATH
from data import load_data, split_data

CLINICAL_THRESH_M = 40.96
CLINICAL_THRESH_F = 30.6

# Youden threshold 최적화에 사용할 CV fold 수
CV_FOLDS = 5

OUTPUT_PATH = os.path.join(RESULTS_MODEL_1_DIR, "clinical_lr_predictions.xlsx")


def _make_labels(smi, sex):
    thresh = np.where(sex == "M", CLINICAL_THRESH_M, CLINICAL_THRESH_F)
    return (smi <= thresh).astype(int), thresh


def _youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def _joudens_j(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens + spec - 1


def _oof_youden_threshold(X_s, y, n_splits=CV_FOLDS, seed=SEED):
    """K-fold OOF 확률로 Youden threshold 최적화.

    훈련 데이터에서 직접 threshold를 구하면 같은 데이터로 학습·평가하는
    낙관적 편향이 생긴다. OOF 확률은 각 샘플이 held-out 상태일 때의
    확률이므로 테스트 분포와 유사한 조건에서 threshold를 선택할 수 있다.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(y))
    fold_thresholds = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_s, y)):
        lr_fold = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
        lr_fold.fit(X_s[tr_idx], y[tr_idx])
        fold_prob = lr_fold.predict_proba(X_s[val_idx])[:, 1]
        oof_prob[val_idx] = fold_prob
        fold_thresh = _youden_threshold(y[val_idx], fold_prob)
        fold_thresholds.append(fold_thresh)
        print(f"    Fold {fold+1}: threshold={fold_thresh:.4f}  "
              f"J={_joudens_j(y[val_idx], fold_prob, fold_thresh):.4f}")

    oof_threshold = _youden_threshold(y, oof_prob)
    print(f"  Fold별 mean={np.mean(fold_thresholds):.4f} ± std={np.std(fold_thresholds):.4f}")
    print(f"  OOF 전체 threshold (채택): {oof_threshold:.4f}")
    return oof_threshold, oof_prob, fold_thresholds


def _save_youden_plot(y_true, oof_prob, oof_threshold, fold_thresholds):
    """OOF ROC 곡선 + Youden's J 곡선 + fold별 threshold 분포를 저장."""
    fpr, tpr, thresholds = roc_curve(y_true, oof_prob)
    roc_auc = auc(fpr, tpr)
    j_scores = tpr - fpr

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    opt_idx = int(np.argmax(j_scores))
    axes[0].plot(fpr, tpr, lw=2, label=f"OOF ROC (AUC={roc_auc:.3f})")
    axes[0].scatter(fpr[opt_idx], tpr[opt_idx], color="red", zorder=5,
                    label=f"Optimal (thresh={oof_threshold:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("FPR (1-Specificity)")
    axes[0].set_ylabel("TPR (Sensitivity)")
    axes[0].set_title(f"OOF ROC ({CV_FOLDS}-Fold CV)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    valid = thresholds < 2
    axes[1].plot(thresholds[valid], j_scores[valid], lw=2, label="Youden's J (OOF)")
    axes[1].axvline(oof_threshold, color="red", lw=1.5, linestyle="--",
                    label=f"OOF optimal ({oof_threshold:.3f})")
    for i, ft in enumerate(fold_thresholds):
        axes[1].axvline(ft, color="gray", lw=0.8, linestyle=":",
                        label=f"Fold {i+1} ({ft:.3f})")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Youden's J  (Sensitivity + Specificity - 1)")
    axes[1].set_title("Youden's J vs Threshold")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(RESULTS_MODEL_1_DIR, "youden_threshold_optimization.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Clinical LR] Youden 최적화 플롯 저장 → {save_path}")


def _save_cm(y_true, y_pred, tag):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Sarcopenia"],
                yticklabels=["Normal", "Sarcopenia"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({tag})")
    fig.tight_layout()
    save_path = os.path.join(RESULTS_MODEL_1_DIR, f"confusion_matrix_{tag.lower()}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Clinical LR] Confusion Matrix 저장 → {save_path}")


def _load_ids(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def run():
    X, smi, imata, sex, patient_ids = load_data()
    X_cv, smi_cv, _, sex_cv, X_te, smi_te, _, sex_te = split_data(
        X, smi, imata, sex, patient_ids
    )

    pid_set = set(patient_ids)
    cv_ids = [p for p in _load_ids(SPLIT_TRAIN_ID_PATH) if p in pid_set]
    te_ids = [p for p in _load_ids(SPLIT_TEST_ID_PATH)  if p in pid_set]

    y_cv, thresh_cv = _make_labels(smi_cv, sex_cv)
    y_te, thresh_te = _make_labels(smi_te, sex_te)

    scaler = StandardScaler()
    X_cv_s = scaler.fit_transform(X_cv)
    X_te_s = scaler.transform(X_te)

    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr.fit(X_cv_s, y_cv)

    prob_cv = lr.predict_proba(X_cv_s)[:, 1]
    prob_te = lr.predict_proba(X_te_s)[:, 1]

    # ── Youden threshold 최적화: OOF CV ──────────────────────────────────
    print(f"[Clinical LR] {CV_FOLDS}-Fold OOF Youden threshold 최적화...")
    binary_threshold, oof_prob_cv, fold_thresholds = _oof_youden_threshold(X_cv_s, y_cv)

    pred_cv = (prob_cv >= binary_threshold).astype(int)
    pred_te = (prob_te >= binary_threshold).astype(int)

    print(f"[Clinical LR] Train sarcopenia: {y_cv.mean():.3f}  Test: {y_te.mean():.3f}")

    os.makedirs(RESULTS_MODEL_1_DIR, exist_ok=True)
    _save_youden_plot(y_cv, oof_prob_cv, binary_threshold, fold_thresholds)
    _save_cm(y_cv, pred_cv, "Train")
    _save_cm(y_te, pred_te, "Test")

    def _build_df(ids, X_arr, smi_arr, sex_arr, thresh_arr, y_arr, prob_arr, oof_arr, pred_arr, split_tag):
        return pd.DataFrame({
            "PatientID":        ids,
            "PatientAge":       X_arr[:, 0],
            "PatientSex":       sex_arr,
            "Height":           X_arr[:, 2],
            "Weight":           X_arr[:, 3],
            "SMI":              smi_arr,
            "SMI_Threshold":    thresh_arr.round(4),
            "Label":            y_arr,
            "LR_Probability":   prob_arr.round(6),
            "OOF_Probability":  oof_arr,
            "LR_Prediction":    pred_arr,
            "Binary_Threshold": round(binary_threshold, 4),
            "Split":            split_tag,
        })

    oof_te = np.full(len(te_ids), np.nan)
    df_cv = _build_df(cv_ids, X_cv, smi_cv, sex_cv, thresh_cv, y_cv, prob_cv, oof_prob_cv.round(6), pred_cv, "Train")
    df_te = _build_df(te_ids, X_te, smi_te, sex_te, thresh_te, y_te, prob_te, oof_te, pred_te, "Test")

    df_all = pd.concat([df_cv, df_te], ignore_index=True)

    df_meta = pd.DataFrame([
        {"항목": "SMI Threshold (Male)",              "값": CLINICAL_THRESH_M},
        {"항목": "SMI Threshold (Female)",             "값": CLINICAL_THRESH_F},
        {"항목": "LR Binary Threshold (OOF Youden's J)", "값": round(binary_threshold, 4)},
        {"항목": "Threshold 최적화 방법",              "값": f"StratifiedKFold {CV_FOLDS}-Fold OOF"},
        {"항목": "Fold별 threshold",                   "값": " / ".join(f"{t:.4f}" for t in fold_thresholds)},
        {"항목": "Fold threshold mean ± std",          "값": f"{np.mean(fold_thresholds):.4f} ± {np.std(fold_thresholds):.4f}"},
        {"항목": "Label 기준",   "값": "SMI <= threshold -> 1(sarcopenia), > threshold -> 0(normal)"},
        {"항목": "입력 변수",    "값": "PatientAge, PatientSex(M=1/F=0), Height, Weight"},
        {"항목": "Scaler",       "값": "StandardScaler (Train fit)"},
        {"항목": "class_weight", "값": "balanced"},
        {"항목": "SEED",         "값": SEED},
        {"항목": "Train N",      "값": len(df_cv)},
        {"항목": "Test N",       "값": len(df_te)},
        {"항목": "Train Sarcopenia Rate", "값": round(float(y_cv.mean()), 4)},
        {"항목": "Test Sarcopenia Rate",  "값": round(float(y_te.mean()), 4)},
    ])

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="Predictions", index=False)
        df_meta.to_excel(writer, sheet_name="Model_Info", index=False)

    print(f"[Clinical LR] 저장 완료 → {OUTPUT_PATH}")
    print(f"  총 {len(df_all)}명  (Train {len(df_cv)} / Test {len(df_te)})")
    return df_all, df_meta


if __name__ == "__main__":
    np.random.seed(SEED)
    run()
