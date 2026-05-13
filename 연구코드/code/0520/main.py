"""
0520 실험 진입점.

0515 실험(M1-M3)을 기반으로 아래 6가지 신규 실험을 순차 실행한다:
  M4   : AEC 핸드크래프트 피처(11개) LR / ResNet1D
  M5   : 성별 분리 CrossAttn (M-only, F-only) — 통합 모델 대비 성능 비교
  M6   : SMI 연속값 회귀 → 이진 분류
  Calib: M2 CrossAttn Temperature Scaling (OOF 기반 캘리브레이션)
  M7   : 앙상블 (M1 LR + M2 CrossAttn)
  Attn : CrossAttn Attention 가중치 시각화

실행 순서:
  M1 baseline → M2 + Calib → M4 → M5 → M6 → M7 → Attention

주의:
  - M1/M2/M3 원본 실험은 0515 main.py 참조
  - 본 스크립트에서의 M1/M2 실행은 앙상블·캘리브레이션·비교용에만 사용
"""
import os
import sys
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from config import (
    DEVICE, SEED, N_FOLDS,
    RESULTS_BASE, RESULTS_DIR_M4,
    RESULTS_DIR_M5_M, RESULTS_DIR_M5_F,
    RESULTS_DIR_M6, RESULTS_DIR_M7,
    RESULTS_DIR_CALIB, RESULTS_DIR_ATTN,
    SMI_THRESH_M, SMI_THRESH_F,
)
from data import (
    load_data, split_data,
    load_data_with_aec, split_data_dual,
    load_data_with_smi, split_data_regression,
    load_data_sexsplit_aec,
    print_stats,
)
from aec_features import extract_aec_features_fast
from cross_val import (
    run_cross_validation,
    run_cross_validation_cross,
    run_cross_validation_cross_calib,
    run_cross_validation_regression,
)
from evaluate import (
    evaluate_test,
    evaluate_test_cross,
    evaluate_test_cross_calib,
    evaluate_test_regression,
    evaluate_ensemble,
)
from metrics import print_cv_summary, compare_fold_metrics
from visualize import (
    plot_data_distribution,
    plot_calibration,
    plot_regression_scatter,
    plot_calibration_compare,
    plot_ensemble_roc,
    plot_sex_roc_compare,
    save_regression_report_md,
    save_calibration_report_md,
    save_ensemble_report_md,
)
from attention_vis import run_attention_analysis

from scipy.special import expit


# ── 공통 헬퍼 ─────────────────────────────────────────────────

def _metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (accuracy_score, f1_score,
                                  average_precision_score, brier_score_loss)
    return {
        "auc":   roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "acc":   accuracy_score(y_true, y_pred),
        "f1":    f1_score(y_true, y_pred, zero_division=0),
    }


def _oof_probs_lr(X_cv, y_cv, scale_clin=True):
    """M1 LR의 OOF 확률 — 앙상블 stacking용."""
    Xs = X_cv.copy()
    if scale_clin:
        sc = StandardScaler()
        Xs[:, [0, 2]] = sc.fit_transform(Xs[:, [0, 2]])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lr  = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    return cross_val_predict(lr, Xs, y_cv, cv=skf, method="predict_proba")[:, 1]


def _oof_probs_lr_aec(X_clin_cv, X_aec_cv, y_cv, scale_clin=True, scale_aec=False):
    """M2 LR(Clin+AEC concat)의 OOF 확률 — 앙상블 stacking용."""
    Xc = X_clin_cv.copy()
    Xa = X_aec_cv.copy()
    if scale_clin:
        sc = StandardScaler()
        Xc[:, [0, 2]] = sc.fit_transform(Xc[:, [0, 2]])
    if scale_aec:
        sa = StandardScaler()
        Xa = sa.fit_transform(Xa)
    X  = np.hstack([Xc, Xa])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lr  = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    return cross_val_predict(lr, X, y_cv, cv=skf, method="predict_proba")[:, 1]


# ── 실험 워커 함수 ────────────────────────────────────────────

def _run_m1_baseline(X_cv, y_cv, sex_cv, X_te, y_te, sex_te):
    """M1 LR + ResNet1D (앙상블·비교용 baseline). run.log → RESULTS_DIR_M4 parent."""
    out_dir = os.path.join(RESULTS_BASE, "model_1_ref", "scale_clinic")
    os.makedirs(out_dir, exist_ok=True)
    buf = io.StringIO(); sys.stdout = buf
    try:
        lr_cv, rn_cv, lr_roc_folds, rn_roc_folds, rn_histories, best_epochs = \
            run_cross_validation(X_cv, y_cv, scale_X=True)
        print_cv_summary("LR",       lr_cv)
        print_cv_summary("ResNet1D", rn_cv)

        med_epoch = int(np.median(best_epochs))
        lr_pred, lr_prob, rn_pred, rn_prob, rn_true = evaluate_test(
            X_cv, y_cv, X_te, y_te, sex_te, med_epoch, scale_X=True,
        )
        plot_data_distribution(X_cv, y_cv, sex_cv, X_te, y_te, sex_te, out_dir=out_dir)
        plot_calibration(
            [("LR", y_te, lr_prob, "steelblue"), ("ResNet1D", rn_true, rn_prob, "tomato")],
            out_path=f"{out_dir}/calibration.png",
        )
        result = {
            "lr_cv": lr_cv, "rn_cv": rn_cv,
            "lr_prob_te": lr_prob, "lr_pred_te": lr_pred,
            "rn_prob_te": rn_prob, "rn_pred_te": rn_pred,
            "y_te": y_te, "sex_te": sex_te,
            "med_epoch": med_epoch,
        }
    finally:
        sys.stdout = sys.__stdout__
    with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    return result


def _run_m2_calib(X_clin_cv, X_aec_cv, y_cv, sex_cv,
                  X_clin_te, X_aec_te, y_te, sex_te):
    """
    M2 CrossAttn CV (OOF logit 수집) + Temperature Scaling.
    앙상블 · 캘리브레이션 · 비교용 기준 모델.
    """
    out_dir = os.path.join(RESULTS_BASE, "model_2_ref", "scale_clinic")
    os.makedirs(out_dir, exist_ok=True)
    buf = io.StringIO(); sys.stdout = buf
    try:
        (lr_cv, ca_cv, rn_cv,
         lr_roc_folds, ca_roc_folds, rn_roc_folds,
         ca_histories, rn_histories,
         ca_best_epochs, rn_best_epochs,
         oof_logits, oof_labels) = run_cross_validation_cross_calib(
            X_clin_cv, X_aec_cv, y_cv, scale_clin=True, scale_aec=False,
        )
        print_cv_summary("LR",        lr_cv)
        print_cv_summary("CrossAttn", ca_cv)
        print_cv_summary("ResNet1D",  rn_cv)

        med_epoch = int(np.median(ca_best_epochs))

        # 캘리브레이션 포함 test 평가
        ca_prob_raw, ca_prob_calib, ca_true, T_opt, calib_stats = \
            evaluate_test_cross_calib(
                X_clin_cv, X_aec_cv, y_cv,
                X_clin_te, X_aec_te, y_te, sex_te,
                med_epoch, oof_logits, oof_labels,
                scale_clin=True, scale_aec=False,
            )

        # 캘리브레이션 전 모델의 일반 test 평가 (LR + ResNet1D도 필요)
        rn_med_epoch = int(np.median(rn_best_epochs))
        lr_pred, lr_prob, ca_pred_raw, ca_prob_raw2, ca_true2, rn_pred, rn_prob = \
            evaluate_test_cross(
                X_clin_cv, X_aec_cv, y_cv,
                X_clin_te, X_aec_te, y_te, sex_te,
                med_epoch, rn_med_epoch=rn_med_epoch,
                scale_clin=True, scale_aec=False,
            )

        plot_data_distribution(X_clin_cv, y_cv, sex_cv, X_clin_te, y_te, sex_te,
                               out_dir=out_dir)
        plot_calibration(
            [("LR", y_te, lr_prob, "steelblue"),
             ("CrossAttn", ca_true, ca_prob_raw, "tomato")],
            out_path=f"{out_dir}/calibration.png",
        )

        # Calibration 시각화 저장
        os.makedirs(RESULTS_DIR_CALIB, exist_ok=True)
        plot_calibration_compare(
            ca_true, ca_prob_raw, ca_prob_calib,
            "M2 CrossAttn (Before)", "M2 CrossAttn (After, T={:.2f})".format(T_opt),
            out_dir=RESULTS_DIR_CALIB,
        )
        save_calibration_report_md(calib_stats, RESULTS_DIR_CALIB)

        result = {
            "ca_cv": ca_cv, "lr_cv": lr_cv,
            "ca_prob_te_raw":   ca_prob_raw,
            "ca_prob_te_calib": ca_prob_calib,
            "ca_pred_te": ca_pred_raw,
            "ca_true_te": ca_true,
            "lr_prob_te": lr_prob,
            "lr_pred_te": lr_pred,
            "oof_logits": oof_logits,
            "oof_labels": oof_labels,
            "oof_ca_probs": expit(oof_logits),
            "y_te": y_te, "sex_te": sex_te,
            "med_epoch": med_epoch,
            "T_opt": T_opt,
            "calib_stats": calib_stats,
        }
    finally:
        sys.stdout = sys.__stdout__
    with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    return result


def _run_m4(X_clin_cv, X_aec_cv, y_cv, sex_cv,
            X_clin_te, X_aec_te, y_te, sex_te):
    """Model 4: AEC 핸드크래프트 피처(11개) + Clinic → LR / ResNet1D."""
    out_dir = os.path.join(RESULTS_DIR_M4, "scale_clinic")
    os.makedirs(out_dir, exist_ok=True)
    buf = io.StringIO(); sys.stdout = buf
    try:
        print("=" * 55)
        print("  MODEL 4 — AEC Handcrafted Features (11)")
        print("=" * 55)

        feats_cv = extract_aec_features_fast(X_aec_cv)
        feats_te = extract_aec_features_fast(X_aec_te)

        # Clinic + AEC 핸드크래프트 피처 concat
        X4_cv = np.hstack([X_clin_cv, feats_cv]).astype(np.float32)
        X4_te = np.hstack([X_clin_te, feats_te]).astype(np.float32)

        # Age·BMI + 11 AEC features 표준화, sex_enc(col 1) 제외
        sc = StandardScaler()
        feat_cols = [0, 2] + list(range(3, X4_cv.shape[1]))  # exclude sex_enc
        X4_cv_s, X4_te_s = X4_cv.copy(), X4_te.copy()
        X4_cv_s[:, feat_cols] = sc.fit_transform(X4_cv[:, feat_cols])
        X4_te_s[:, feat_cols] = sc.transform(X4_te[:, feat_cols])

        lr_cv, rn_cv, lr_roc_folds, rn_roc_folds, rn_histories, best_epochs = \
            run_cross_validation(X4_cv_s, y_cv, scale_X=False)  # already scaled

        print_cv_summary("LR",       lr_cv)
        print_cv_summary("ResNet1D", rn_cv)
        compare_fold_metrics("LR", lr_cv, "ResNet1D", rn_cv)

        med_epoch = int(np.median(best_epochs))
        lr_pred, lr_prob, rn_pred, rn_prob, rn_true = evaluate_test(
            X4_cv_s, y_cv, X4_te_s, y_te, sex_te, med_epoch, scale_X=False,
        )

        plot_data_distribution(X_clin_cv, y_cv, sex_cv, X_clin_te, y_te, sex_te,
                               out_dir=out_dir)
        plot_calibration(
            [("LR", y_te, lr_prob, "steelblue"), ("ResNet1D", rn_true, rn_prob, "tomato")],
            out_path=f"{out_dir}/calibration.png",
        )

        _save_m4_report_md(lr_cv, rn_cv,
                           _metrics(y_te,    lr_pred, lr_prob),
                           _metrics(rn_true, rn_pred, rn_prob),
                           out_dir)
        result = {
            "lr_cv": lr_cv, "rn_cv": rn_cv,
            "lr_prob_te": lr_prob, "rn_prob_te": rn_prob,
            "y_te": y_te,
        }
    finally:
        sys.stdout = sys.__stdout__
    with open(os.path.join(RESULTS_DIR_M4, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  [M4] Done  (log → {RESULTS_DIR_M4}/run.log)")
    return result


def _save_m4_report_md(lr_cv, rn_cv, lr_te, rn_te, out_dir):
    from metrics import group_metrics
    lines = [
        "# Model 4 — AEC Handcrafted Features",
        "",
        "11 피처: mean, std, max, min, range, AUC, peak_pos, skewness, kurtosis, slope_1h, slope_2h",
        "",
        f"## CV Results (5-Fold)",
        "",
        "| Model | AUC | AUPRC | Brier | Acc | F1 |",
        "|-------|----:|------:|------:|----:|---:|",
    ]
    for label, folds in [("LR", lr_cv), ("ResNet1D", rn_cv)]:
        auc_m   = np.mean([f["auc"]   for f in folds])
        auprc_m = np.mean([f["auprc"] for f in folds])
        brier_m = np.mean([f["brier"] for f in folds])
        acc_m   = np.mean([f["acc"]   for f in folds])
        f1_m    = np.mean([f["f1"]    for f in folds])
        lines.append(f"| {label} | {auc_m:.4f} | {auprc_m:.4f} | {brier_m:.4f} | {acc_m:.4f} | {f1_m:.4f} |")

    lines += [
        "",
        "## Test Results",
        "",
        "| Model | AUC | AUPRC | Brier | Acc | F1 |",
        "|-------|----:|------:|------:|----:|---:|",
        f"| LR       | {lr_te['auc']:.4f} | {lr_te['auprc']:.4f} | {lr_te['brier']:.4f} | {lr_te['acc']:.4f} | {lr_te['f1']:.4f} |",
        f"| ResNet1D | {rn_te['auc']:.4f} | {rn_te['auprc']:.4f} | {rn_te['brier']:.4f} | {rn_te['acc']:.4f} | {rn_te['f1']:.4f} |",
    ]
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _run_m5(X_clin_cv, X_aec_cv, y_cv, sex_cv,
            X_clin_te, X_aec_te, y_te, sex_te,
            ca_prob_combined_te):
    """
    Model 5: 성별 분리 CrossAttn.

    남성/여성 데이터로 각각 별도 CrossAttn 학습 후 성능 비교.
    """
    buf = io.StringIO(); sys.stdout = buf
    try:
        print("=" * 55)
        print("  MODEL 5 — Sex-Stratified CrossAttn")
        print("=" * 55)

        results = {}
        for sx, out_dir in [("M", RESULTS_DIR_M5_M), ("F", RESULTS_DIR_M5_F)]:
            os.makedirs(os.path.join(out_dir, "scale_clinic"), exist_ok=True)
            sub_out = os.path.join(out_dir, "scale_clinic")

            mask_cv = sex_cv == sx
            mask_te = sex_te == sx
            if mask_cv.sum() < N_FOLDS or mask_te.sum() == 0:
                print(f"  [{sx}] 샘플 수 부족 → 건너뜀")
                continue

            Xc_cv_s = X_clin_cv[mask_cv]; Xa_cv_s = X_aec_cv[mask_cv]
            y_cv_s  = y_cv[mask_cv];      sex_cv_s = sex_cv[mask_cv]
            Xc_te_s = X_clin_te[mask_te]; Xa_te_s = X_aec_te[mask_te]
            y_te_s  = y_te[mask_te];      sex_te_s = sex_te[mask_te]

            if len(np.unique(y_cv_s)) < 2 or len(np.unique(y_te_s)) < 2:
                print(f"  [{sx}] 단일 클래스 → 건너뜀")
                continue

            print(f"\n  [{sx}] n_cv={mask_cv.sum()}, n_te={mask_te.sum()}"
                  f"  (Sarco={y_cv_s.mean()*100:.1f}%)")

            (lr_cv, ca_cv, rn_cv,
             lr_roc_folds, ca_roc_folds, rn_roc_folds,
             ca_histories, rn_histories,
             ca_best_epochs, rn_best_epochs) = run_cross_validation_cross(
                Xc_cv_s, Xa_cv_s, y_cv_s, scale_clin=True, scale_aec=False,
            )
            print_cv_summary(f"CA [{sx}]", ca_cv)

            med_epoch = int(np.median(ca_best_epochs))
            rn_med_epoch = int(np.median(rn_best_epochs))
            lr_pred, lr_prob, ca_pred, ca_prob, ca_true, rn_pred, rn_prob = \
                evaluate_test_cross(
                    Xc_cv_s, Xa_cv_s, y_cv_s,
                    Xc_te_s, Xa_te_s, y_te_s, sex_te_s,
                    med_epoch, rn_med_epoch=rn_med_epoch,
                    scale_clin=True, scale_aec=False,
                )

            plot_data_distribution(Xc_cv_s, y_cv_s, sex_cv_s, Xc_te_s, y_te_s,
                                   sex_te_s, out_dir=sub_out)
            plot_calibration(
                [("LR", y_te_s, lr_prob, "steelblue"),
                 (f"CA [{sx}]", ca_true, ca_prob, "tomato")],
                out_path=f"{sub_out}/calibration.png",
            )
            results[sx] = {"ca_cv": ca_cv, "ca_prob_te": ca_prob, "ca_true_te": ca_true,
                           "y_te": y_te_s, "sex_te": sex_te_s, "n_te": mask_te.sum()}

        # Sex-specific vs Combined 비교 ROC
        if "M" in results and "F" in results:
            mask_te_m = sex_te == "M"; mask_te_f = sex_te == "F"
            prob_comb_m = ca_prob_combined_te[mask_te_m]
            prob_comb_f = ca_prob_combined_te[mask_te_f]
            y_te_m = y_te[mask_te_m]; y_te_f = y_te[mask_te_f]

            if (len(np.unique(y_te_m)) > 1 and len(np.unique(y_te_f)) > 1):
                plot_sex_roc_compare(
                    y_te_m, prob_comb_m, results["M"]["ca_prob_te"],
                    y_te_f, prob_comb_f, results["F"]["ca_prob_te"],
                    out_dir=RESULTS_BASE + "/model_5",
                )

        _save_m5_summary_md(results, out_dir=RESULTS_BASE + "/model_5")

    finally:
        sys.stdout = sys.__stdout__

    log_dir = RESULTS_BASE + "/model_5"
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  [M5] Done  (log → {log_dir}/run.log)")
    return results


def _save_m5_summary_md(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lines = [
        "# Model 5 — Sex-Stratified CrossAttn",
        "",
        "## CV Results (mean ± std, 5-Fold)",
        "",
        "| Sex | n_te | AUC (CV mean) | AUC (Test) |",
        "|-----|-----:|-------------:|-----------:|",
    ]
    for sx in ["M", "F"]:
        if sx not in results:
            lines.append(f"| {sx} | — | — | — |")
            continue
        r = results[sx]
        cv_aucs = [f["auc"] for f in r["ca_cv"]]
        te_auc  = roc_auc_score(r["y_te"], r["ca_prob_te"]) if len(np.unique(r["y_te"])) > 1 else float("nan")
        lines.append(f"| {sx} | {r['n_te']} "
                     f"| {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f} "
                     f"| {te_auc:.4f} |")
    with open(os.path.join(out_dir, "results_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _run_m6(X_clin_cv, X_aec_cv, y_smi_cv, y_bin_cv, sex_cv,
            X_clin_te, X_aec_te, y_smi_te, y_bin_te, sex_te):
    """Model 6: SMI 연속값 회귀 → 이진 분류."""
    out_dir = RESULTS_DIR_M6
    os.makedirs(out_dir, exist_ok=True)
    buf = io.StringIO(); sys.stdout = buf
    try:
        print("=" * 55)
        print("  MODEL 6 — SMI Regression (HuberLoss)")
        print("=" * 55)

        reg_cv, reg_histories, best_epochs, \
        oof_pred, oof_true, oof_bin, oof_sex = run_cross_validation_regression(
            X_clin_cv, X_aec_cv, y_smi_cv, sex_cv,
            scale_clin=True, scale_aec=False,
        )

        cv_mae   = np.mean([f["mae"] for f in reg_cv])
        cv_rmse  = np.mean([f["rmse"] for f in reg_cv])
        cv_r2    = np.mean([f["r2"]  for f in reg_cv])
        cv_auc   = np.mean([f["auc"] for f in reg_cv if not np.isnan(f["auc"])])
        print(f"\nCV mean — MAE: {cv_mae:.3f}  RMSE: {cv_rmse:.3f}  R²: {cv_r2:.3f}  AUC: {cv_auc:.4f}")

        med_epoch = int(np.median(best_epochs))
        pred_smi, true_smi, bin_pred, bin_prob, true_bin, reg_stats_test = \
            evaluate_test_regression(
                X_clin_cv, X_aec_cv, y_smi_cv, y_bin_cv, sex_cv,
                X_clin_te, X_aec_te, y_smi_te, y_bin_te, sex_te,
                med_epoch, scale_clin=True, scale_aec=False,
            )

        plot_regression_scatter(pred_smi, true_smi, sex_te, out_dir,
                                model_label="M6 SMI Regression")
        save_regression_report_md(reg_cv, reg_stats_test, pred_smi, true_smi, out_dir)

        result = {
            "reg_cv": reg_cv, "reg_stats_test": reg_stats_test,
            "pred_smi": pred_smi, "true_smi": true_smi,
            "bin_prob": bin_prob, "true_bin": true_bin,
        }
    finally:
        sys.stdout = sys.__stdout__
    with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  [M6] Done  (log → {out_dir}/run.log)")
    return result


def _run_m7(m1_result, m2_result, X_clin_cv, X_aec_cv):
    """Model 7: 앙상블 (M1 LR + M2 CrossAttn avg/weighted/stacking)."""
    out_dir = RESULTS_DIR_M7
    os.makedirs(out_dir, exist_ok=True)
    buf = io.StringIO(); sys.stdout = buf
    try:
        print("=" * 55)
        print("  MODEL 7 — Ensemble (M1 LR + M2 CrossAttn)")
        print("=" * 55)

        y_te      = m2_result["y_te"]
        sex_te    = m2_result["sex_te"]
        probs_m1  = m1_result["lr_prob_te"]
        probs_m2  = m2_result["ca_prob_te_raw"]

        oof_labels    = m2_result["oof_labels"]
        oof_probs_m1  = _oof_probs_lr(X_clin_cv, oof_labels, scale_clin=True)
        oof_probs_m2  = m2_result["oof_ca_probs"]

        # M1 y_te 와 M2 y_te가 일치해야 함 (동일 split)
        assert np.array_equal(m1_result["y_te"], m2_result["y_te"]), \
            "M1 / M2 test label 불일치 — 동일 split 여부 확인 필요"

        ensemble_results = evaluate_ensemble(
            y_te,
            probs_m1_lr=probs_m1,
            probs_m2_ca=probs_m2,
            oof_probs_m1_lr=oof_probs_m1,
            oof_probs_m2_ca=oof_probs_m2,
            oof_labels=oof_labels,
            sex_te=sex_te,
            weights=(0.4, 0.6),
        )

        individual_probs = {"M1 LR": probs_m1, "M2 CrossAttn": probs_m2}
        individual_aucs  = {
            "M1 LR":      roc_auc_score(y_te, probs_m1),
            "M2 CrossAttn": roc_auc_score(y_te, probs_m2),
        }
        plot_ensemble_roc(y_te, ensemble_results, individual_probs, individual_aucs,
                          out_dir=out_dir)
        save_ensemble_report_md(ensemble_results, individual_aucs, y_te, out_dir)

    finally:
        sys.stdout = sys.__stdout__
    with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  [M7] Done  (log → {out_dir}/run.log)")
    return ensemble_results


# ── 요약 비교 md ──────────────────────────────────────────────

def _save_final_summary_md(m1_r, m2_r, m4_r, m5_r, m6_r, m7_r):
    """전체 실험 요약 markdown 저장."""
    lines = [
        "# 0520 Experiment Summary",
        "",
        "| Model | Sub-model | AUC | AUPRC | Brier | Note |",
        "|-------|-----------|----:|------:|------:|------|",
    ]

    def _row(model, sub, y_te, prob, note=""):
        if len(np.unique(y_te)) < 2: return
        from sklearn.metrics import average_precision_score, brier_score_loss
        auc   = roc_auc_score(y_te, prob)
        auprc = average_precision_score(y_te, prob)
        brier = brier_score_loss(y_te, prob)
        lines.append(f"| {model} | {sub} | {auc:.4f} | {auprc:.4f} | {brier:.4f} | {note} |")

    _row("M1 (ref)", "LR",           m1_r["y_te"], m1_r["lr_prob_te"])
    _row("M1 (ref)", "ResNet1D",     m1_r["y_te"], m1_r["rn_prob_te"])
    _row("M2 (ref)", "CrossAttn",    m2_r["y_te"], m2_r["ca_prob_te_raw"])
    _row("M2 (ref)", "CA+Calib",     m2_r["y_te"], m2_r["ca_prob_te_calib"],
         f"T={m2_r['T_opt']:.3f}")

    _row("M4",       "LR",           m4_r["y_te"], m4_r["lr_prob_te"], "AEC handcraft 11-feat")
    _row("M4",       "ResNet1D",     m4_r["y_te"], m4_r["rn_prob_te"])

    if m5_r:
        for sx in ["M", "F"]:
            if sx in m5_r:
                r = m5_r[sx]
                _row(f"M5 ({sx})", "CrossAttn", r["y_te"], r["ca_prob_te"], "sex-split")

    _row("M6",       "Reg→Binary",   m6_r["true_bin"], m6_r["bin_prob"],
         f"MAE={m6_r['reg_stats_test']['mae']:.2f}")

    if m7_r:
        for method in ["avg", "weighted", "stacking"]:
            if method in m7_r:
                _row("M7", method, m2_r["y_te"], m7_r[method]["probs"])

    md_path = os.path.join(RESULTS_BASE, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Summary saved → {md_path}")


# ── 메인 ─────────────────────────────────────────────────────

def run_all():
    print(f"Device: {DEVICE}\n")

    # ── 데이터 로드 ─────────────────────────────────────────────
    print("Loading data ...")
    X,      y,      sex     = load_data()
    X_clin, X_aec,  y2, sex2 = load_data_with_aec()
    X_clin_r, X_aec_r, y_smi, y_bin, sex_r = load_data_with_smi()

    print("\n=== M1 dataset ==="); print_stats(y,   sex)
    print("=== M2 dataset ==="); print_stats(y2,  sex2)
    print("=== M6 dataset ==="); print_stats(y_bin, sex_r)

    # ── Data split ───────────────────────────────────────────────
    X_cv,      y_cv,    sex_cv,    X_te,      y_te,    sex_te    = split_data(X, y, sex)
    X_clin_cv, X_aec_cv, y2_cv, sex2_cv, \
    X_clin_te, X_aec_te, y2_te, sex2_te                          = split_data_dual(X_clin, X_aec, y2, sex2)
    (X_clin_rcv, X_aec_rcv, y_smi_cv, y_bin_cv, sex_rcv,
     X_clin_rte, X_aec_rte, y_smi_te, y_bin_te, sex_rte)        = split_data_regression(
        X_clin_r, X_aec_r, y_smi, y_bin, sex_r)

    # ── M1 baseline (앙상블용) ────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 1/6] M1 Baseline")
    print("=" * 60)
    m1_result = _run_m1_baseline(X_cv, y_cv, sex_cv, X_te, y_te, sex_te)

    # ── M2 + Calibration ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 2/6] M2 CrossAttn + Temperature Scaling")
    print("=" * 60)
    m2_result = _run_m2_calib(
        X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
        X_clin_te, X_aec_te, y2_te, sex2_te,
    )

    # ── M4: AEC Handcrafted Features ─────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 3/6] M4 — AEC Handcrafted Features")
    print("=" * 60)
    m4_result = _run_m4(
        X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
        X_clin_te, X_aec_te, y2_te, sex2_te,
    )

    # ── M5: Sex-Stratified ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 4/6] M5 — Sex-Stratified CrossAttn")
    print("=" * 60)
    m5_result = _run_m5(
        X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
        X_clin_te, X_aec_te, y2_te, sex2_te,
        ca_prob_combined_te=m2_result["ca_prob_te_raw"],
    )

    # ── M6: SMI Regression ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 5/6] M6 — SMI Regression")
    print("=" * 60)
    m6_result = _run_m6(
        X_clin_rcv, X_aec_rcv, y_smi_cv, y_bin_cv, sex_rcv,
        X_clin_rte, X_aec_rte, y_smi_te, y_bin_te, sex_rte,
    )

    # ── M7: Ensemble ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 6/6] M7 — Ensemble + Attention")
    print("=" * 60)
    m7_result = _run_m7(m1_result, m2_result, X_clin_cv, X_aec_cv)

    # ── Attention Visualization ───────────────────────────────
    print("\n  Running Attention Visualization ...")
    try:
        run_attention_analysis(
            X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
            X_clin_te, X_aec_te, y2_te, sex2_te,
            med_epoch=m2_result["med_epoch"],
            scale_clin=True, scale_aec=False,
            out_dir=RESULTS_DIR_ATTN,
        )
    except Exception as e:
        print(f"  [Attention] Warning: {e}")

    # ── 최종 요약 ─────────────────────────────────────────────
    print("\n  Saving final summary ...")
    _save_final_summary_md(m1_result, m2_result, m4_result, m5_result, m6_result, m7_result)
    print("\n  All experiments done.")


if __name__ == "__main__":
    run_all()
