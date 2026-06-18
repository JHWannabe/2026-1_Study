import os
import numpy as np

import matplotlib
matplotlib.use('Agg')

from config import RESULTS_MODEL_1_DIR, THRESH_M1, SEED, AEC_NORM
from data import load_data, split_data
from train_eval import run_cross_validation, evaluate_test, compute_gradient_saliency
from visualize import (save_all, plot_norm_comparison, plot_aec_heatmap,
                       plot_cam_segment8, plot_cam_by_sex,
                       plot_aec_crossing_pattern,
                       plot_training_curves, plot_final_training_curve)


def run_model():
    """AEC 8-segment 피처 (60-dim) + Clinic, SegmentFusionModel — AEC_NORM 목록 순회 실행."""
    X_aec, X_clinic, y, sex, patient_ids, X_aec_raw = load_data()
    X_aec_cv, X_clinic_cv, y_cv, sex_cv, X_aec_te, X_clinic_te, y_te, sex_te, X_aec_raw_cv, X_aec_raw_te = split_data(X_aec, X_clinic, y, sex, patient_ids, X_aec_raw)

    results = {}
    for norm in AEC_NORM:
        out_dir = os.path.join(RESULTS_MODEL_1_DIR, norm)
        os.makedirs(out_dir, exist_ok=True)

        cv_folds, roc_folds, thresholds, cv_text, fold_histories = run_cross_validation(
            X_aec_cv, X_clinic_cv, y_cv, sex_cv, norm
        )

        med_thresh = THRESH_M1 if THRESH_M1 is not None else float(np.median(thresholds))
        pred, prob, ci, test_text, final_model, X_aec_te_s, X_clinic_te_s, final_history = evaluate_test(
            X_aec_cv, X_clinic_cv, y_cv, sex_cv,
            X_aec_te, X_clinic_te, y_te, sex_te,
            norm, threshold=med_thresh,
        )

        save_all(
            roc_folds, cv_folds,
            X_clinic_cv, y_cv, sex_cv,
            X_clinic_te, y_te, pred, prob,
            sex_te, out_dir=out_dir, ci_dict=ci, norm=norm,
        )

        plot_training_curves(fold_histories, out_dir, model_name="SegmentFusion", norm=norm)
        plot_final_training_curve(final_history, out_dir, model_name="SegmentFusion", norm=norm)

        saliency = compute_gradient_saliency(final_model, X_aec_te_s, X_clinic_te_s)
        plot_aec_heatmap(X_aec_raw_te, y_te, prob, out_dir)
        plot_cam_segment8(saliency, y_te, prob, out_dir)
        plot_cam_by_sex(saliency, y_te, prob, sex_te, out_dir)   # Direction 3

        md_content = (
            f"# Model — SegmentFusion [{norm}]\n\n"
            + cv_text + "\n\n"
            + test_text + "\n"
        )
        with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
            f.write(md_content)

        results[norm] = {"cv_folds": cv_folds, "prob": prob, "y_te": y_te,
                         "ci": ci, "threshold": med_thresh}

    comp_path = os.path.join(RESULTS_MODEL_1_DIR, "aec_individual_normalization_compare.png")
    plot_norm_comparison(results, comp_path)

    # Direction 5: 80번째 슬라이스 교차 패턴 분석 (여성, 모델 무관한 raw 데이터 분석)
    plot_aec_crossing_pattern(X_aec_raw_te, y_te, sex_te, RESULTS_MODEL_1_DIR)

    return results


if __name__ == "__main__":
    np.random.seed(SEED)
    run_model()
