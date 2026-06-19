import os
import numpy as np

import matplotlib
matplotlib.use('Agg')

from config import RESULTS_MODEL_1_DIR, THRESH_M1, SEED
from data import load_data, split_data, compute_thresholds, make_labels
from train_eval import run_cross_validation, evaluate_test
from visualize import save_all


def run_model():
    """Model (Clinic Only LR) 전체 파이프라인 실행."""
    X, smi, imata, sex, patient_ids = load_data()
    X_cv, smi_cv, imata_cv, sex_cv, X_te, smi_te, imata_te, sex_te = split_data(X, smi, imata, sex, patient_ids)

    thresh_m, thresh_f = compute_thresholds(smi_cv, sex_cv)
    y_cv = make_labels(smi_cv, sex_cv, thresh_m, thresh_f)
    y_te = make_labels(smi_te, sex_te, thresh_m, thresh_f)

    os.makedirs(RESULTS_MODEL_1_DIR, exist_ok=True)

    lr_cv, lr_roc_folds, lr_thresholds, cv_text = run_cross_validation(X_cv, y_cv)

    med_thresh = THRESH_M1 if THRESH_M1 is not None else float(np.median(lr_thresholds))
    lr_pred, lr_prob, ci, test_text = evaluate_test(X_cv, y_cv, X_te, y_te, sex_te, threshold=med_thresh)

    save_all(
        lr_roc_folds, lr_cv,
        X_cv, y_cv, sex_cv,
        X_te, y_te, lr_pred, lr_prob,
        sex_te, out_dir=RESULTS_MODEL_1_DIR, ci_dict=ci,
    )

    # IMATA 상위 25% 서브그룹 평가
    imata_q75 = float(np.percentile(imata_te, 75))
    top25_mask = imata_te >= imata_q75
    print(f"[IMATA top25%] cutoff={imata_q75:.4f}, n={top25_mask.sum()}/{len(imata_te)}")
    _, _, ci_top25, test_text_top25 = evaluate_test(
        X_cv, y_cv,
        X_te[top25_mask], y_te[top25_mask], sex_te[top25_mask],
        threshold=med_thresh,
    )
    test_text_top25 = f"## Test Set — IMATA 상위 25% (≥{imata_q75:.4f}, n={top25_mask.sum()})\n\n" + test_text_top25

    md_content = (
        "# Model — Clinic Only LR\n\n"
        + cv_text + "\n\n"
        + test_text + "\n\n"
        + test_text_top25 + "\n"
    )
    results_md_path = os.path.join(RESULTS_MODEL_1_DIR, "results.md")
    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return {
        "lr_cv_folds":   lr_cv,
        "lr_prob":        lr_prob,
        "y_te":           y_te,
        "ci":             ci,
        "ci_top25":       ci_top25,
    }


if __name__ == "__main__":
    np.random.seed(SEED)
    run_model()
