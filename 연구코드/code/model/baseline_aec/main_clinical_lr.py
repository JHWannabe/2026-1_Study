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

# Step2 재분류 구간: Youden threshold ± MARGIN 사이 확률을 가진 케이스
AMBIGUOUS_MARGIN = 0.15

# Youden threshold 최적화에 사용할 CV fold 수
CV_FOLDS = 5

AEC_COLS = [f"aec_{i}" for i in range(1, 129)]

OUTPUT_PATH = os.path.join(RESULTS_MODEL_1_DIR, "clinical_lr_predictions.xlsx")


def _make_labels(smi, sex):
    thresh = np.where(sex == "M", CLINICAL_THRESH_M, CLINICAL_THRESH_F)
    return (smi <= thresh).astype(int), thresh


def _youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[int(np.argmax(tpr - fpr))])


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


def _joudens_j(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens + spec - 1


def _save_youden_plot(y_true, oof_prob, oof_threshold, fold_thresholds):
    """OOF ROC 곡선 + Youden's J 곡선 + fold별 threshold 분포를 저장."""
    fpr, tpr, thresholds = roc_curve(y_true, oof_prob)
    roc_auc = auc(fpr, tpr)
    j_scores = tpr - fpr

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 왼쪽: OOF ROC 곡선 + 최적 threshold 포인트
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

    # 오른쪽: Youden's J vs threshold + fold별 threshold 분산
    valid = thresholds < 2  # roc_curve 첫 더미값 제외
    axes[1].plot(thresholds[valid], j_scores[valid], lw=2, label="Youden's J (OOF)")
    axes[1].axvline(oof_threshold, color="red", lw=1.5, linestyle="--",
                    label=f"OOF optimal ({oof_threshold:.3f})")
    for i, ft in enumerate(fold_thresholds):
        axes[1].axvline(ft, color="gray", lw=0.8, linestyle=":",
                        label=f"Fold {i+1} ({ft:.3f})" if i == 0 else f"Fold {i+1} ({ft:.3f})")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Youden's J  (Sensitivity + Specificity - 1)")
    axes[1].set_title("Youden's J vs Threshold")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(RESULTS_MODEL_1_DIR, "youden_threshold_optimization.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Step1] Youden 최적화 플롯 저장 → {save_path}")


def _load_ids(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _load_aec_data():
    """aec_128 시트에서 PatientID + 128차원 AEC 특성 로드."""
    df = pd.read_excel(DATA_PATH, sheet_name="aec_128")[["PatientID"] + AEC_COLS]
    df["PatientID"] = df["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return df.set_index("PatientID")


def _get_aec_matrix(ids, aec_df):
    """환자 ID 리스트에 대해 AEC 특성 행렬 반환. AEC 데이터 없는 환자는 NaN 행."""
    return aec_df.reindex(ids)[AEC_COLS].values.astype(np.float32)


def _psm_match(X_clinical, y, seed=SEED):
    """Age/Sex/Height/Weight 기반 propensity score 1:1 nearest-neighbor 매칭.

    사르코페니아(1) 케이스를 정상(0) 컨트롤에 1:1 매칭한다.
    반환값은 매칭된 훈련 인덱스 배열 (케이스+컨트롤 순서 혼합).
    """
    scaler_ps = StandardScaler()
    X_s = scaler_ps.fit_transform(X_clinical)

    ps_model = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    ps_model.fit(X_s, y)
    ps = ps_model.predict_proba(X_s)[:, 1]

    case_idx = np.where(y == 1)[0]
    ctrl_idx = np.where(y == 0)[0]

    matched = []
    used_ctrl = set()
    for ci in case_idx:
        dists = np.abs(ps[ctrl_idx] - ps[ci])
        for cj in ctrl_idx[np.argsort(dists)]:
            if cj not in used_ctrl:
                matched.append((ci, int(cj)))
                used_ctrl.add(int(cj))
                break

    matched_idx = np.array([idx for pair in matched for idx in pair])
    print(f"  [PSM] 케이스 {len(case_idx)}명 → 매칭 성공 {len(matched)}쌍 "
          f"(총 {len(matched_idx)}명 / 원본 {len(y)}명)")
    return matched_idx


def _save_cm(y_true, y_pred, tag, title_suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Sarcopenia"],
                yticklabels=["Normal", "Sarcopenia"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({tag}){title_suffix}")
    fig.tight_layout()
    save_path = os.path.join(RESULTS_MODEL_1_DIR, f"confusion_matrix_{tag.lower()}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Clinical LR] Confusion Matrix 저장 → {save_path}")
    return cm


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

    # ── Step 1: 임상 변수 Logistic Regression ──────────────────────────────
    scaler1 = StandardScaler()
    X_cv_s = scaler1.fit_transform(X_cv)
    X_te_s = scaler1.transform(X_te)

    lr1 = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr1.fit(X_cv_s, y_cv)

    prob_cv = lr1.predict_proba(X_cv_s)[:, 1]
    prob_te = lr1.predict_proba(X_te_s)[:, 1]

    # ── Youden threshold 최적화: OOF CV ──────────────────────────────────
    print(f"[Step1] {CV_FOLDS}-Fold OOF Youden threshold 최적화...")
    binary_threshold, oof_prob_cv, fold_thresholds = _oof_youden_threshold(X_cv_s, y_cv)

    pred1_cv = (prob_cv >= binary_threshold).astype(int)
    pred1_te = (prob_te >= binary_threshold).astype(int)

    print(f"[Step1] Train sarcopenia: {y_cv.mean():.3f}  Test: {y_te.mean():.3f}")

    os.makedirs(RESULTS_MODEL_1_DIR, exist_ok=True)
    _save_youden_plot(y_cv, oof_prob_cv, binary_threshold, fold_thresholds)

    # Step1 confusion matrix (최종 비교용 저장)
    _save_cm(y_cv, pred1_cv, "Step1_Train", " -Clinical LR only")
    _save_cm(y_te, pred1_te, "Step1_Test",  " -Clinical LR only")

    # ── Step 2: 애매한 케이스를 AEC 특성으로 재분류 ──────────────────────
    lo = binary_threshold - AMBIGUOUS_MARGIN
    hi = binary_threshold + AMBIGUOUS_MARGIN

    ambig_cv = (prob_cv >= lo) & (prob_cv <= hi)
    ambig_te = (prob_te >= lo) & (prob_te <= hi)

    print(f"[Step2] 애매 구간: [{lo:.4f}, {hi:.4f}]  "
          f"Train 해당: {ambig_cv.sum()}명  Test 해당: {ambig_te.sum()}명")

    aec_df = _load_aec_data()

    # 임상 + AEC 통합 특성 행렬 구성
    aec_cv = _get_aec_matrix(cv_ids, aec_df)
    aec_te = _get_aec_matrix(te_ids, aec_df)

    # AEC: 환자 내 흐름 패턴 보존을 위해 row-wise 정규화
    def _rowwise_normalize(arr):
        mean = np.nanmean(arr, axis=1, keepdims=True)
        std  = np.nanstd(arr, axis=1, keepdims=True)
        return (arr - mean) / (std + 1e-8)

    aec_cv_scaled = _rowwise_normalize(aec_cv)
    aec_te_scaled = _rowwise_normalize(aec_te)

    X_cv_combined = np.concatenate([X_cv, aec_cv_scaled], axis=1)
    X_te_combined = np.concatenate([X_te, aec_te_scaled], axis=1)

    # AEC 값 없는 환자(NaN)가 있는 행은 step2 적용 제외 처리
    has_aec_cv = ~np.isnan(aec_cv).any(axis=1)
    has_aec_te = ~np.isnan(aec_te).any(axis=1)

    # Step2 학습: PSM 매칭 후 임상+AEC 통합 모델
    # PSM: AEC 데이터가 있는 훈련 환자 내에서 1:1 nearest-neighbor 매칭
    aec_avail_idx = np.where(has_aec_cv)[0]
    print(f"[PSM] AEC 보유 훈련 환자 {len(aec_avail_idx)}명에 PSM 적용...")
    psm_rel_idx = _psm_match(X_cv[aec_avail_idx], y_cv[aec_avail_idx])
    train2_idx = aec_avail_idx[psm_rel_idx]

    X_tr2 = X_cv_combined[train2_idx].copy()
    y_tr2 = y_cv[train2_idx]

    psm_n_pairs = len(psm_rel_idx) // 2

    # 임상 4개만 column-wise 정규화 (AEC 128개는 row-wise 완료 상태)
    scaler2 = StandardScaler()
    X_tr2[:, :4] = scaler2.fit_transform(X_tr2[:, :4])

    lr2 = LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced", C=0.1)
    lr2.fit(X_tr2, y_tr2)

    # Step2 예측: Step1 오분류(FN/FP) 케이스에만 적용
    prob2_cv = np.full(len(cv_ids), np.nan)
    prob2_te = np.full(len(te_ids), np.nan)

    apply2_cv = ambig_cv & has_aec_cv
    apply2_te = ambig_te & has_aec_te

    if apply2_cv.any():
        X_cv2 = X_cv_combined[apply2_cv].copy()
        X_cv2[:, :4] = scaler2.transform(X_cv2[:, :4])
        prob2_cv[apply2_cv] = lr2.predict_proba(X_cv2)[:, 1]

    if apply2_te.any():
        X_te2 = X_te_combined[apply2_te].copy()
        X_te2[:, :4] = scaler2.transform(X_te2[:, :4])
        prob2_te[apply2_te] = lr2.predict_proba(X_te2)[:, 1]

    # Step2 Youden threshold (CV 전체 재산출 후 Step1 threshold 재사용)
    # 동일 threshold 기준으로 step2 확률도 분류
    pred2_cv = np.where(~np.isnan(prob2_cv), (prob2_cv >= binary_threshold).astype(int), -1)
    pred2_te = np.where(~np.isnan(prob2_te), (prob2_te >= binary_threshold).astype(int), -1)

    # ── 최종 예측: Step1 base + Step2 override ────────────────────────────
    final_pred_cv = pred1_cv.copy()
    final_pred_te = pred1_te.copy()

    final_pred_cv[apply2_cv] = pred2_cv[apply2_cv]
    final_pred_te[apply2_te] = pred2_te[apply2_te]

    step_cv = np.where(apply2_cv, 2, 1)
    step_te = np.where(apply2_te, 2, 1)

    print(f"[Final] Train  Step1: {(step_cv==1).sum()}명  Step2: {(step_cv==2).sum()}명")
    print(f"[Final] Test   Step1: {(step_te==1).sum()}명  Step2: {(step_te==2).sum()}명")

    # Step1→Final 변경 케이스
    changed_cv = (pred1_cv != final_pred_cv).sum()
    changed_te = (pred1_te != final_pred_te).sum()
    print(f"[Final] 예측 변경 - Train: {changed_cv}명  Test: {changed_te}명")

    # 최종 Confusion Matrix
    _save_cm(y_cv, final_pred_cv, "Final_Train", " -2-Step")
    _save_cm(y_te, final_pred_te, "Final_Test",  " -2-Step")

    # ── Excel 출력 ─────────────────────────────────────────────────────────
    def _build_df(ids, X_arr, smi_arr, sex_arr, thresh_arr, y_arr,
                  prob1_arr, pred1_arr, prob2_arr, pred2_arr, final_arr, step_arr, split_tag):
        return pd.DataFrame({
            "PatientID":           ids,
            "PatientAge":          X_arr[:, 0],
            "PatientSex":          sex_arr,
            "Height":              X_arr[:, 2],
            "Weight":              X_arr[:, 3],
            "SMI":                 smi_arr,
            "SMI_Threshold":       thresh_arr.round(4),
            "Label":               y_arr,
            "Step1_Prob":          prob1_arr.round(6),
            "Step1_Pred":          pred1_arr,
            "Step2_Prob":          np.where(~np.isnan(prob2_arr), prob2_arr.round(6), np.nan),
            "Step2_Pred":          np.where(pred2_arr >= 0, pred2_arr.astype(float), np.nan),
            "Final_Pred":          final_arr,
            "Decision_Step":       step_arr,
            "Binary_Threshold":    round(binary_threshold, 4),
            "Ambiguous_Low":       round(lo, 4),
            "Ambiguous_High":      round(hi, 4),
            "Split":               split_tag,
        })

    df_cv  = _build_df(cv_ids, X_cv, smi_cv, sex_cv, thresh_cv, y_cv,
                       prob_cv, pred1_cv, prob2_cv, pred2_cv, final_pred_cv, step_cv, "Train")
    df_te  = _build_df(te_ids, X_te, smi_te, sex_te, thresh_te, y_te,
                       prob_te, pred1_te, prob2_te, pred2_te, final_pred_te, step_te, "Test")
    df_all = pd.concat([df_cv, df_te], ignore_index=True)

    # Confusion Matrix 수치 정리
    cm1_tr = confusion_matrix(y_cv, pred1_cv)
    cm1_te = confusion_matrix(y_te, pred1_te)
    cmF_tr = confusion_matrix(y_cv, final_pred_cv)
    cmF_te = confusion_matrix(y_te, final_pred_te)

    df_meta = pd.DataFrame([
        {"항목": "SMI Threshold (Male)",              "값": CLINICAL_THRESH_M},
        {"항목": "SMI Threshold (Female)",             "값": CLINICAL_THRESH_F},
        {"항목": "LR Binary Threshold (OOF Youden's J)", "값": round(binary_threshold, 4)},
        {"항목": "Threshold 최적화 방법",                "값": f"StratifiedKFold {CV_FOLDS}-Fold OOF"},
        {"항목": "Fold별 threshold",                    "값": " / ".join(f"{t:.4f}" for t in fold_thresholds)},
        {"항목": "Fold threshold mean ± std",           "값": f"{np.mean(fold_thresholds):.4f} ± {np.std(fold_thresholds):.4f}"},
        {"항목": "Ambiguous Margin",                   "값": AMBIGUOUS_MARGIN},
        {"항목": "Ambiguous 구간 Low",                  "값": round(lo, 4)},
        {"항목": "Ambiguous 구간 High",                 "값": round(hi, 4)},
        {"항목": "Step1 입력 변수",                     "값": "PatientAge, PatientSex(M=1/F=0), Height, Weight"},
        {"항목": "Step2 입력 변수",                     "값": "Step1 변수 + AEC 1~128"},
        {"항목": "Step2 PSM 방법",                     "값": "1:1 Nearest-Neighbor PSM (임상 변수 기반 propensity score)"},
        {"항목": "Step2 PSM 매칭 변수",                 "값": "PatientAge, PatientSex, Height, Weight"},
        {"항목": "Step2 PSM 매칭 쌍 수",               "값": psm_n_pairs},
        {"항목": "Step2 PSM 학습 데이터 수",            "값": len(train2_idx)},
        {"항목": "Step2 C (L2 정규화)",                 "값": 0.1},
        {"항목": "Scaler",                             "값": "StandardScaler (Train fit)"},
        {"항목": "class_weight",                       "값": "balanced"},
        {"항목": "SEED",                               "값": SEED},
        {"항목": "Train N",                            "값": len(df_cv)},
        {"항목": "Test N",                             "값": len(df_te)},
        {"항목": "Train Ambiguous N",                  "값": int(ambig_cv.sum())},
        {"항목": "Test Ambiguous N",                   "값": int(ambig_te.sum())},
        {"항목": "Train Step2 적용 N (AEC 있음)",       "값": int(apply2_cv.sum())},
        {"항목": "Test Step2 적용 N (AEC 있음)",        "값": int(apply2_te.sum())},
        {"항목": "Train 예측 변경 N",                   "값": int(changed_cv)},
        {"항목": "Test 예측 변경 N",                    "값": int(changed_te)},
        {"항목": "Step1 Train -TN/FP/FN/TP",         "값": f"{cm1_tr[0,0]}/{cm1_tr[0,1]}/{cm1_tr[1,0]}/{cm1_tr[1,1]}"},
        {"항목": "Step1 Test  -TN/FP/FN/TP",         "값": f"{cm1_te[0,0]}/{cm1_te[0,1]}/{cm1_te[1,0]}/{cm1_te[1,1]}"},
        {"항목": "Final Train -TN/FP/FN/TP",         "값": f"{cmF_tr[0,0]}/{cmF_tr[0,1]}/{cmF_tr[1,0]}/{cmF_tr[1,1]}"},
        {"항목": "Final Test  -TN/FP/FN/TP",         "값": f"{cmF_te[0,0]}/{cmF_te[0,1]}/{cmF_te[1,0]}/{cmF_te[1,1]}"},
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
