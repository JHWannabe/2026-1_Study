import os
import numpy as np

import matplotlib
matplotlib.use('Agg')

from config import RESULTS_MODEL_1_DIR, THRESH_M1, SEED
from data import load_data, split_data
from train_eval import run_cross_validation, evaluate_test
from visualize import save_all


def run_model():
    """Model (Clinic Only LR) 전체 파이프라인 실행."""
    X, y, sex, patient_ids = load_data()
    X_cv, y_cv, sex_cv, X_te, y_te, sex_te = split_data(X, y, sex, patient_ids)

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

    md_content = (
        "# Model 1 — Clinic Only LR\n\n"
        + cv_text + "\n\n"
        + test_text + "\n"
    )
    results_md_path = os.path.join(RESULTS_MODEL_1_DIR, "results.md")
    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return {
        "lr_cv_folds": lr_cv,
        "lr_prob":     lr_prob,
        "y_te":        y_te,
        "ci":          ci,
    }


if __name__ == "__main__":
    np.random.seed(SEED)
    run_model()
