import os
import numpy as np

import matplotlib
matplotlib.use('Agg')

from config import RESULTS_MODEL_1_DIR, THRESH_M1, SEED, AEC_NORM
from data import load_data, split_data, crop_to_liver_region
from train_eval import run_cross_validation, evaluate_test, compute_grad_cam
from visualize import save_all, plot_norm_comparison, plot_aec_heatmap, plot_cam_128, plot_training_curves, plot_final_training_curve


def run_model():
    """Model (AEC→scalar + Clinic, AECFusionModel) — AEC_NORM 목록 순회 실행."""
    X_aec, X_clinic, y, sex, patient_ids = load_data()
    X_aec_cv, X_clinic_cv, y_cv, sex_cv, X_aec_te, X_clinic_te, y_te, sex_te = \
        split_data(X_aec, X_clinic, y, sex, patient_ids)

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

        plot_training_curves(fold_histories, out_dir, model_name="AECFusion", norm=norm)
        plot_final_training_curve(final_history, out_dir, model_name="AECFusion", norm=norm)

        cam_maps = compute_grad_cam(final_model, X_aec_te_s, X_clinic_te_s)
        plot_aec_heatmap(X_aec_te, y_te, prob, out_dir)
        plot_cam_128(cam_maps, y_te, prob, out_dir)

        md_content = (
            f"# Model — AECFusion [{norm}]\n\n"
            + cv_text + "\n\n"
            + test_text + "\n"
        )
        with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
            f.write(md_content)

        results[norm] = {"cv_folds": cv_folds, "prob": prob, "y_te": y_te,
                         "ci": ci, "threshold": med_thresh}

    comp_path = os.path.join(RESULTS_MODEL_1_DIR, "aec_individual_normalization_compare.png")
    plot_norm_comparison(results, comp_path)

    # Direction 4: Liver 후반부 집중 크롭 실험
    run_liver_crop_experiment(X_aec_cv, X_clinic_cv, y_cv, sex_cv,
                              X_aec_te, X_clinic_te, y_te, sex_te)

    return results


def run_liver_crop_experiment(X_aec_cv, X_clinic_cv, y_cv, sex_cv,
                               X_aec_te, X_clinic_te, y_te, sex_te,
                               start_dims=(40, 51, 64)):
    """Direction 4: Liver 후반부 집중 크롭 AEC로 AECFusionModel 학습·비교.

    start_dims: 크롭 시작 인덱스 목록 — 40%/51%/64% 지점 비교.
    결과는 results/model_aec_128/liver_crop{start_dim}/{norm}/ 에 저장.
    """
    for start_dim in start_dims:
        tag = f"crop{start_dim}"
        print(f"\n[Direction 4] Liver crop - start_dim={start_dim} ({start_dim/128*100:.0f}%)")

        X_aec_cv_c = crop_to_liver_region(X_aec_cv, start_dim=start_dim)
        X_aec_te_c = crop_to_liver_region(X_aec_te, start_dim=start_dim)

        crop_norm_results = {}
        for norm in AEC_NORM:
            out_dir = os.path.join(RESULTS_MODEL_1_DIR, f"liver_{tag}", norm)
            os.makedirs(out_dir, exist_ok=True)

            cv_folds, roc_folds, thresholds, cv_text, fold_histories = run_cross_validation(
                X_aec_cv_c, X_clinic_cv, y_cv, sex_cv, norm
            )
            med_thresh = float(np.median(thresholds))
            pred, prob, ci, test_text, final_model, X_aec_te_s, X_clinic_te_s, final_history = evaluate_test(
                X_aec_cv_c, X_clinic_cv, y_cv, sex_cv,
                X_aec_te_c, X_clinic_te, y_te, sex_te,
                norm, threshold=med_thresh,
            )

            save_all(
                roc_folds, cv_folds,
                X_clinic_cv, y_cv, sex_cv,
                X_clinic_te, y_te, pred, prob,
                sex_te, out_dir=out_dir, ci_dict=ci, norm=norm,
            )

            plot_training_curves(fold_histories, out_dir, model_name=f"AECFusion-Crop{start_dim}", norm=norm)
            plot_final_training_curve(final_history, out_dir, model_name=f"AECFusion-Crop{start_dim}", norm=norm)

            cam_maps = compute_grad_cam(final_model, X_aec_te_s, X_clinic_te_s)
            plot_aec_heatmap(X_aec_te_c, y_te, prob, out_dir)
            plot_cam_128(cam_maps, y_te, prob, out_dir)

            md_content = (
                f"# Model — AECFusion Liver Crop start={start_dim} [{norm}]\n\n"
                + cv_text + "\n\n"
                + test_text + "\n"
            )
            with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
                f.write(md_content)

            crop_norm_results[norm] = {"cv_folds": cv_folds, "prob": prob,
                                        "y_te": y_te, "ci": ci}
            print(f"  [{norm}] test AUC = {ci.get('auc', (0,))[0]:.4f}")

        comp_path = os.path.join(RESULTS_MODEL_1_DIR, f"liver_{tag}", "aec_normalization_compare.png")
        plot_norm_comparison(crop_norm_results, comp_path)


if __name__ == "__main__":
    np.random.seed(SEED)
    run_model()
