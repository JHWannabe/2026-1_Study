"""
SMI Binary Classification — 모델별 스케일링 케이스 비교 실험 진입점.

모델 구성:
  Model 1   : Clinic Only (Age, Sex, BMI)                — LR / ResNet1D
  Model 2   : Clinic + AEC Matched                       — LR / CrossAttn / ResNet1D
  Model 2_2 : Clinic + AEC Unmatched (음성 대조군)         — LR / CrossAttn / ResNet1D
  Model 3   : Clinic + Scanner(MFR Embedding) + AEC      — LR / CrossAttn3 / ResNet1D

각 모델은 AEC_VARIANTS(7종) × CASES_M*(스케일링 조건) 조합을 순차 실행한다.
Model 1/2/2_2/3 는 ProcessPoolExecutor 로 병렬 실행되며,
결과는 scaling_comparison.md 와 각 모델 디렉토리 run.log 에 저장된다.

주의: label(y, 0/1)에는 StandardScaler를 적용하지 않는다.
      Clinical 스케일링 시 sex_enc(이진값)는 제외하고 Age·BMI만 표준화한다.
"""

import os
import sys
import io
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    average_precision_score, brier_score_loss,
)

import matplotlib
matplotlib.use('Agg')

from config import DEVICE, RESULTS_DIR, RESULTS_DIR_CROSS, RESULTS_DIR_CROSS_2_2, RESULTS_DIR_CROSS3, AEC_VARIANTS
from data import (load_data, split_data,
                  load_data_with_aec, load_data_with_aec_unmatched, split_data_dual,
                  load_data_with_aec_meta, split_data_quad,
                  aec_variant, print_stats)
from cross_val import (run_cross_validation, run_cross_validation_cross,
                       run_cross_validation_cross3)
from evaluate import evaluate_test, evaluate_test_cross, evaluate_test_cross3
from metrics import print_cv_summary, compare_fold_metrics
from visualize import save_all, save_all_cross

# ── 모델별 스케일링 케이스 정의 ──────────────────────────────
# 각 튜플: (결과 디렉토리 이름, scale_clinic, [scale_aec])
# scale_clinic: Age·BMI에 StandardScaler 적용 여부 (sex_enc 제외)
# scale_aec   : AEC 전 컬럼에 StandardScaler 적용 여부

# Model 1: (case_name, scale_clinic)
CASES_M1 = [
    ("scale_clinic", True),   # Age·BMI 표준화
]

# Model 2/2_2: (case_name, scale_clinic, scale_aec)
CASES_M2 = [
    ("scale_clinic", True,  False),  # Age·BMI만 표준화
    ("scale_both",   True,  True),   # Age·BMI + AEC 모두 표준화
]

CASES_M2_2 = CASES_M2[:]   # Unmatched 음성 대조군은 M2와 동일 조건

# Model 3: kVp 제거로 scale_scan 없음 — (case_name, scale_clinic, scale_aec)
CASES_M3 = [
    ("scale_clinic", True,  False),
    ("scale_both",   True,  True),
]


def _metrics(y_true, y_pred, y_prob):
    """AUC·AUPRC·Brier·Accuracy·F1 다섯 지표를 dict로 반환."""
    return {
        "auc":   roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "acc":   accuracy_score(y_true, y_pred),
        "f1":    f1_score(y_true, y_pred, zero_division=0),
    }


def _case_label(r):
    """aec_var 가 있으면 'aec_var/case', 없으면 'case' 만 반환."""
    if "aec_var" in r:
        return f"{r['aec_var']}/{r['case']}"
    return r["case"]


# ── 모델별 워커 함수 (subprocess 에서 실행) ───────────────────
# 각 모델의 모든 케이스를 순차 실행하고 결과 리스트를 반환.
# 상세 출력은 <model_results_dir>/run.log 에 저장.

def _run_model1(X_cv, y_cv, sex_cv, X_te, y_te, sex_te):
    """Model 1(Clinic Only)의 모든 scaling case를 순차 실행하고 결과 리스트를 반환. 로그는 run.log에 저장."""
    buf = io.StringIO()
    sys.stdout = buf
    results = []
    try:
        print(f"{'='*60}")
        print("  MODEL 1 — Clinic Only  (2 cases)")
        print(f"{'='*60}")

        for case_name, sc in CASES_M1:
            print(f"\n{'#'*60}")
            print(f"  [M1] CASE : {case_name}  (scale_clinic={sc})")
            print(f"{'#'*60}")

            out1 = os.path.join(RESULTS_DIR, case_name)
            os.makedirs(out1, exist_ok=True)

            (lr_cv, rn_cv,
             lr_roc_folds, rn_roc_folds,
             rn_histories, best_epochs) = run_cross_validation(X_cv, y_cv, scale_X=sc)

            print_cv_summary("LR",       lr_cv)
            print_cv_summary("ResNet1D", rn_cv)

            stat_lr_rn = compare_fold_metrics("LR", lr_cv, "ResNet1D", rn_cv)

            med_epoch = int(np.median(best_epochs))
            lr_pred, lr_prob, rn_pred_te, rn_prob_te, rn_true_te = evaluate_test(
                X_cv, y_cv, X_te, y_te, sex_te, med_epoch, scale_X=sc
            )

            save_all(
                lr_roc_folds, rn_roc_folds, lr_cv, rn_cv,
                X_cv, y_cv, sex_cv,
                X_te, y_te, lr_pred, rn_true_te, rn_pred_te,
                rn_histories, med_epoch, lr_prob, rn_prob_te,
                sex_te, out_dir=out1,
            )

            results.append({
                "case":         case_name,
                "scale_clinic": sc,
                "m1_lr":        _metrics(y_te,       lr_pred,    lr_prob),
                "m1_rn":        _metrics(rn_true_te, rn_pred_te, rn_prob_te),
                "stat_lr_rn":   stat_lr_rn,
                "lr_cv_folds":  lr_cv,
                "rn_cv_folds":  rn_cv,
            })
    finally:
        sys.stdout = sys.__stdout__

    with open(os.path.join(RESULTS_DIR, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return results


def _run_model2(X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                X_clin_te, X_aec_te, y2_te, sex2_te):
    """Model 2(Clinic+AEC Matched)의 모든 AEC len × scaling case를 순차 실행하고 결과 리스트를 반환. 로그는 run.log에 저장."""
    buf = io.StringIO()
    sys.stdout = buf
    results = []
    try:
        print(f"{'='*60}")
        print(f"  MODEL 2 — Clinic + AEC  ({len(AEC_VARIANTS)} AEC variants × {len(CASES_M2)} cases)")
        print(f"{'='*60}")

        for aec_var in AEC_VARIANTS:
            X_aec_cv_v, mask_cv = aec_variant(X_aec_cv, aec_var)
            X_aec_te_v, mask_te = aec_variant(X_aec_te, aec_var)
            # excl_extreme 변형은 샘플 필터링 포함
            X_clin_cv_v = X_clin_cv[mask_cv] if mask_cv is not None else X_clin_cv
            y2_cv_v     = y2_cv[mask_cv]     if mask_cv is not None else y2_cv
            sex2_cv_v   = sex2_cv[mask_cv]   if mask_cv is not None else sex2_cv
            X_clin_te_v = X_clin_te[mask_te] if mask_te is not None else X_clin_te
            y2_te_v     = y2_te[mask_te]     if mask_te is not None else y2_te
            sex2_te_v   = sex2_te[mask_te]   if mask_te is not None else sex2_te

            for case_name, sc, sa in CASES_M2:
                print(f"\n{'#'*60}")
                print(f"  [M2 {aec_var}] CASE : {case_name}  (scale_clinic={sc}, scale_aec={sa})")
                print(f"{'#'*60}")

                out2 = os.path.join(RESULTS_DIR_CROSS, aec_var, case_name)
                os.makedirs(out2, exist_ok=True)

                (lr2_cv, ca_cv, rn2_cv,
                 lr2_roc_folds, ca_roc_folds, rn2_roc_folds,
                 ca_histories, rn2_histories,
                 ca_best_epochs2, rn2_best_epochs) = run_cross_validation_cross(
                    X_clin_cv_v, X_aec_cv_v, y2_cv_v, scale_clin=sc, scale_aec=sa,
                )

                print_cv_summary("LR",        lr2_cv)
                print_cv_summary("CrossAttn", ca_cv)
                print_cv_summary("ResNet1D",  rn2_cv)

                stat_lr_ca = compare_fold_metrics("LR", lr2_cv, "CrossAttn", ca_cv)

                med_epoch2    = int(np.median(ca_best_epochs2))
                rn_med_epoch2 = int(np.median(rn2_best_epochs))
                (lr2_pred_te, lr2_prob_te, ca_pred_te, ca_prob_te, ca_true_te,
                 rn2_pred_te, rn2_prob_te) = evaluate_test_cross(
                    X_clin_cv_v, X_aec_cv_v, y2_cv_v,
                    X_clin_te_v, X_aec_te_v, y2_te_v, sex2_te_v,
                    med_epoch2, rn_med_epoch=rn_med_epoch2,
                    scale_clin=sc, scale_aec=sa,
                )

                save_all_cross(
                    lr2_cv, ca_cv, lr2_roc_folds, ca_roc_folds, ca_histories, med_epoch2,
                    X_clin_cv_v, y2_cv_v, sex2_cv_v,
                    X_clin_te_v, y2_te_v,
                    lr2_pred_te, lr2_prob_te,
                    ca_pred_te, ca_true_te, sex2_te_v, ca_prob_te,
                    model_label=f"model 2 ({aec_var})", out_dir=out2,
                )

                results.append({
                    "aec_var":      aec_var,
                    "case":         case_name,
                    "scale_clinic": sc,
                    "scale_aec":    sa,
                    "m2_lr":        _metrics(y2_te_v,    lr2_pred_te,  lr2_prob_te),
                    "m2_ca":        _metrics(ca_true_te, ca_pred_te,   ca_prob_te),
                    "m2_rn":        _metrics(ca_true_te, rn2_pred_te,  rn2_prob_te),
                    "stat_lr_ca":   stat_lr_ca,
                    "lr_cv_folds":  lr2_cv,
                    "ca_cv_folds":  ca_cv,
                    "rn_cv_folds":  rn2_cv,
                })
    finally:
        sys.stdout = sys.__stdout__

    with open(os.path.join(RESULTS_DIR_CROSS, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return results


def _run_model2_2(X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                  X_clin_te, X_aec_te, y2_te, sex2_te):
    """
    Model 2_2: Model 2와 동일한 ClinAECCrossAttn 구조를 사용하되
    Clinic-AEC가 서로 다른 환자 데이터로 섞인 상태(Unmatching)로 학습/평가.
    Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 가진다는 증거.
    """
    buf = io.StringIO()
    sys.stdout = buf
    results = []
    try:
        print(f"{'='*60}")
        print(f"  MODEL 2_2 — Clinic + AEC (Unmatched)  ({len(AEC_VARIANTS)} AEC variants × {len(CASES_M2_2)} cases)")
        print(f"{'='*60}")

        for aec_var in AEC_VARIANTS:
            X_aec_cv_v, mask_cv = aec_variant(X_aec_cv, aec_var)
            X_aec_te_v, mask_te = aec_variant(X_aec_te, aec_var)
            X_clin_cv_v = X_clin_cv[mask_cv] if mask_cv is not None else X_clin_cv
            y2_cv_v     = y2_cv[mask_cv]     if mask_cv is not None else y2_cv
            sex2_cv_v   = sex2_cv[mask_cv]   if mask_cv is not None else sex2_cv
            X_clin_te_v = X_clin_te[mask_te] if mask_te is not None else X_clin_te
            y2_te_v     = y2_te[mask_te]     if mask_te is not None else y2_te
            sex2_te_v   = sex2_te[mask_te]   if mask_te is not None else sex2_te

            for case_name, sc, sa in CASES_M2_2:
                print(f"\n{'#'*60}")
                print(f"  [M2_2 {aec_var}] CASE : {case_name}  (scale_clinic={sc}, scale_aec={sa})")
                print(f"{'#'*60}")

                out2_2 = os.path.join(RESULTS_DIR_CROSS_2_2, aec_var, case_name)
                os.makedirs(out2_2, exist_ok=True)

                (lr2_cv, ca_cv, rn2_2_cv,
                 lr2_roc_folds, ca_roc_folds, rn2_2_roc_folds,
                 ca_histories, rn2_2_histories,
                 ca_best_epochs2_2, rn2_2_best_epochs) = run_cross_validation_cross(
                    X_clin_cv_v, X_aec_cv_v, y2_cv_v, scale_clin=sc, scale_aec=sa,
                )

                print_cv_summary("LR",        lr2_cv)
                print_cv_summary("CrossAttn", ca_cv)
                print_cv_summary("ResNet1D",  rn2_2_cv)

                stat_lr_ca_u = compare_fold_metrics("LR", lr2_cv, "CrossAttn", ca_cv)

                med_epoch2_2    = int(np.median(ca_best_epochs2_2))
                rn_med_epoch2_2 = int(np.median(rn2_2_best_epochs))
                (lr2_pred_te, lr2_prob_te, ca_pred_te, ca_prob_te, ca_true_te,
                 rn2_2_pred_te, rn2_2_prob_te) = evaluate_test_cross(
                    X_clin_cv_v, X_aec_cv_v, y2_cv_v,
                    X_clin_te_v, X_aec_te_v, y2_te_v, sex2_te_v,
                    med_epoch2_2, rn_med_epoch=rn_med_epoch2_2,
                    scale_clin=sc, scale_aec=sa,
                )

                save_all_cross(
                    lr2_cv, ca_cv, lr2_roc_folds, ca_roc_folds, ca_histories, med_epoch2_2,
                    X_clin_cv_v, y2_cv_v, sex2_cv_v,
                    X_clin_te_v, y2_te_v,
                    lr2_pred_te, lr2_prob_te,
                    ca_pred_te, ca_true_te, sex2_te_v, ca_prob_te,
                    model_label=f"model 2_2 ({aec_var}, unmatched)", out_dir=out2_2,
                )

                results.append({
                    "aec_var":      aec_var,
                    "case":         case_name,
                    "scale_clinic": sc,
                    "scale_aec":    sa,
                    "m2_2_lr":      _metrics(y2_te_v,    lr2_pred_te,    lr2_prob_te),
                    "m2_2_ca":      _metrics(ca_true_te, ca_pred_te,     ca_prob_te),
                    "m2_2_rn":      _metrics(ca_true_te, rn2_2_pred_te,  rn2_2_prob_te),
                    "stat_lr_ca":   stat_lr_ca_u,
                    "lr_cv_folds":  lr2_cv,
                    "ca_cv_folds":  ca_cv,
                    "rn_cv_folds":  rn2_2_cv,
                })
    finally:
        sys.stdout = sys.__stdout__

    with open(os.path.join(RESULTS_DIR_CROSS_2_2, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return results


def _run_model3(X_clin3_cv, X_aec3_cv, X_mfr_cv, y3_cv, sex3_cv,
                X_clin3_te, X_aec3_te, X_mfr_te, y3_te, sex3_te, n_mfr):
    """Model 3(Clinic+Scanner+AEC)의 모든 AEC variant × scaling case를 순차 실행하고 결과 리스트를 반환. 로그는 run.log에 저장."""
    buf = io.StringIO()
    sys.stdout = buf
    results = []
    try:
        print(f"{'='*60}")
        print(f"  MODEL 3 — Clinic + Scanner + AEC  ({len(AEC_VARIANTS)} AEC variants × {len(CASES_M3)} cases)")
        print(f"{'='*60}")

        for aec_var in AEC_VARIANTS:
            X_aec3_cv_v, mask_cv = aec_variant(X_aec3_cv, aec_var)
            X_aec3_te_v, mask_te = aec_variant(X_aec3_te, aec_var)
            X_clin3_cv_v = X_clin3_cv[mask_cv] if mask_cv is not None else X_clin3_cv
            X_mfr3_cv_v  = X_mfr_cv[mask_cv]   if mask_cv is not None else X_mfr_cv
            y3_cv_v      = y3_cv[mask_cv]       if mask_cv is not None else y3_cv
            sex3_cv_v    = sex3_cv[mask_cv]     if mask_cv is not None else sex3_cv
            X_clin3_te_v = X_clin3_te[mask_te] if mask_te is not None else X_clin3_te
            X_mfr3_te_v  = X_mfr_te[mask_te]   if mask_te is not None else X_mfr_te
            y3_te_v      = y3_te[mask_te]       if mask_te is not None else y3_te
            sex3_te_v    = sex3_te[mask_te]     if mask_te is not None else sex3_te

            for case_name, sc, sa in CASES_M3:
                print(f"\n{'#'*60}")
                print(f"  [M3 {aec_var}] CASE : {case_name}"
                      f"  (scale_clinic={sc}, scale_aec={sa})")
                print(f"{'#'*60}")

                out3 = os.path.join(RESULTS_DIR_CROSS3, aec_var, case_name)
                os.makedirs(out3, exist_ok=True)

                (lr3_cv, ca3_cv, rn3_cv,
                 lr3_roc_folds, ca3_roc_folds, rn3_roc_folds,
                 ca3_histories, rn3_histories,
                 ca3_best_epochs3, rn3_best_epochs) = run_cross_validation_cross3(
                    X_clin3_cv_v, X_aec3_cv_v, X_mfr3_cv_v, y3_cv_v, n_mfr,
                    scale_clin=sc, scale_aec=sa,
                )

                print_cv_summary("LR",         lr3_cv)
                print_cv_summary("CrossAttn3", ca3_cv)
                print_cv_summary("ResNet1D",   rn3_cv)

                stat_lr_ca3 = compare_fold_metrics("LR", lr3_cv, "CrossAttn3", ca3_cv)

                med_epoch3    = int(np.median(ca3_best_epochs3))
                rn_med_epoch3 = int(np.median(rn3_best_epochs))
                (lr3_pred_te, lr3_prob_te, ca3_pred_te, ca3_prob_te, ca3_true_te,
                 rn3_pred_te, rn3_prob_te) = evaluate_test_cross3(
                    X_clin3_cv_v, X_aec3_cv_v, X_mfr3_cv_v, y3_cv_v,
                    X_clin3_te_v, X_aec3_te_v, X_mfr3_te_v, y3_te_v,
                    sex3_te_v, med_epoch3, n_mfr,
                    rn_med_epoch=rn_med_epoch3,
                    scale_clin=sc, scale_aec=sa,
                )

                save_all_cross(
                    lr3_cv, ca3_cv, lr3_roc_folds, ca3_roc_folds, ca3_histories, med_epoch3,
                    X_clin3_cv_v, y3_cv_v, sex3_cv_v,
                    X_clin3_te_v, y3_te_v,
                    lr3_pred_te, lr3_prob_te,
                    ca3_pred_te, ca3_true_te, sex3_te_v, ca3_prob_te,
                    model_label=f"model 3 ({aec_var})", out_dir=out3,
                )

                results.append({
                    "aec_var":      aec_var,
                    "case":         case_name,
                    "scale_clinic": sc,
                    "scale_aec":    sa,
                    "m3_lr":        _metrics(y3_te_v,      lr3_pred_te,  lr3_prob_te),
                    "m3_ca3":       _metrics(ca3_true_te,  ca3_pred_te,  ca3_prob_te),
                    "m3_rn":        _metrics(ca3_true_te,  rn3_pred_te,  rn3_prob_te),
                    "stat_lr_ca3":  stat_lr_ca3,
                    "lr_cv_folds":  lr3_cv,
                    "ca3_cv_folds": ca3_cv,
                    "rn_cv_folds":  rn3_cv,
                })
    finally:
        sys.stdout = sys.__stdout__

    with open(os.path.join(RESULTS_DIR_CROSS3, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return results


def run_all_cases():
    """데이터 로드 후 Model 1/2/2_2/3을 병렬 실행하고 결과 비교 테이블·md 파일을 저장."""
    print(f"Device  : {DEVICE}\n")

    # ── 공통 데이터 로드 ──────────────────────────────────────
    X,           y,       sex       = load_data()
    X_clin,      X_aec,   y2, sex2  = load_data_with_aec()
    X_clin_u,    X_aec_u, y2u, sex2u = load_data_with_aec_unmatched()
    X_clin3, X_aec3, X_scan_mfr, y3, sex3, n_mfr = load_data_with_aec_meta()

    print("=== Model 1   dataset ==="); print_stats(y,   sex)
    print("=== Model 2   dataset ==="); print_stats(y2,  sex2)
    print("=== Model 2_2 dataset ==="); print_stats(y2u, sex2u)
    print("=== Model 3   dataset ==="); print_stats(y3,  sex3)

    X_cv,       y_cv,   sex_cv,   X_te,       y_te,   sex_te   = split_data(X, y, sex)
    X_clin_cv,  X_aec_cv,  y2_cv,  sex2_cv, \
    X_clin_te,  X_aec_te,  y2_te,  sex2_te                     = split_data_dual(X_clin,   X_aec,   y2,  sex2)
    X_clin_ucv, X_aec_ucv, y2u_cv, sex2u_cv, \
    X_clin_ute, X_aec_ute, y2u_te, sex2u_te                    = split_data_dual(X_clin_u, X_aec_u, y2u, sex2u)
    (X_clin3_cv, X_aec3_cv, X_mfr_cv, y3_cv, sex3_cv,
     X_clin3_te, X_aec3_te, X_mfr_te, y3_te, sex3_te) = split_data_quad(
        X_clin3, X_aec3, X_scan_mfr, y3, sex3,
    )

    # ── Model 1/2/2_2/3 병렬 실행 ────────────────────────────
    print("\n  Launching Model 1, 2, 2_2, 3 in parallel ...\n")

    with ProcessPoolExecutor(max_workers=4) as executor:
        fut1 = executor.submit(
            _run_model1,
            X_cv, y_cv, sex_cv, X_te, y_te, sex_te,
        )
        fut2 = executor.submit(
            _run_model2,
            X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
            X_clin_te, X_aec_te, y2_te, sex2_te,
        )
        fut2_2 = executor.submit(
            _run_model2_2,
            X_clin_ucv, X_aec_ucv, y2u_cv, sex2u_cv,
            X_clin_ute, X_aec_ute, y2u_te, sex2u_te,
        )
        fut3 = executor.submit(
            _run_model3,
            X_clin3_cv, X_aec3_cv, X_mfr_cv, y3_cv, sex3_cv,
            X_clin3_te, X_aec3_te, X_mfr_te, y3_te, sex3_te, n_mfr,
        )
        results_m1   = fut1.result()
        results_m2   = fut2.result()
        results_m2_2 = fut2_2.result()
        results_m3   = fut3.result()

    print("  All models done.\n")

    # ── 비교 테이블 출력 & 저장 ──────────────────────────────
    _print_comparison(results_m1, results_m2, results_m2_2, results_m3)
    _save_comparison_md(results_m1, results_m2, results_m2_2, results_m3)


# ── 출력 헬퍼 ────────────────────────────────────────────────

_METRICS_DEF = [("AUC", "auc"), ("AUPRC", "auprc"), ("Brier", "brier"),
                ("Acc", "acc"), ("F1", "f1")]


def _best_case(results, metric_key):
    """Test 전체 AUC 기준으로 best case 반환."""
    return max(results, key=lambda r: r[metric_key]["auc"])


def _model_table_str(results, model_key, col=8):
    """단일 모델 결과를 콘솔 테이블 문자열로 반환. best case에 <- BEST 표시."""
    best = _best_case(results, model_key)
    hdr_parts = [f"{'Case':<32}"]
    for mname, _ in _METRICS_DEF:
        hdr_parts.append(f"{mname:>{col}}")
    hdr = " ".join(hdr_parts)
    rows = [hdr, "-" * len(hdr)]
    for r in results:
        m = r[model_key]
        row = f"{_case_label(r):<32}"
        for _, mk in _METRICS_DEF:
            row += f" {m[mk]:>{col}.4f}"
        if r is best:
            row += "  <- BEST"
        rows.append(row)
    return "\n".join(rows)


def _print_best_summary(results_m1, results_m2, results_m2_2, results_m3):
    """각 모델/서브모델의 best case(Test AUC 기준)를 요약 출력."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  BEST CASES SUMMARY  (by Test overall AUC)")
    print(sep)
    entries = [
        ("M1",   "LR",         results_m1,   "m1_lr"),
        ("M1",   "ResNet1D",   results_m1,   "m1_rn"),
        ("M2",   "LR",         results_m2,   "m2_lr"),
        ("M2",   "CrossAttn",  results_m2,   "m2_ca"),
        ("M2",   "ResNet1D",   results_m2,   "m2_rn"),
        ("M2_2", "LR",         results_m2_2, "m2_2_lr"),
        ("M2_2", "CrossAttn",  results_m2_2, "m2_2_ca"),
        ("M2_2", "ResNet1D",   results_m2_2, "m2_2_rn"),
        ("M3",   "LR",         results_m3,   "m3_lr"),
        ("M3",   "CrossAttn3", results_m3,   "m3_ca3"),
        ("M3",   "ResNet1D",   results_m3,   "m3_rn"),
    ]
    col = 8
    hdr = f"  {'Model':<6} {'Sub-model':<12} {'Best Case':<32}"
    for mname, _ in _METRICS_DEF:
        hdr += f" {mname:>{col}}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for model_lbl, sub_lbl, results, key in entries:
        best = _best_case(results, key)
        m = best[key]
        row = f"  {model_lbl:<6} {sub_lbl:<12} {_case_label(best):<32}"
        for _, mk in _METRICS_DEF:
            row += f" {m[mk]:>{col}.4f}"
        print(row)


def _print_comparison(results_m1, results_m2, results_m2_2, results_m3):
    """Model 1~3의 모든 case 결과를 서브모델별 콘솔 테이블로 출력하고, 마지막에 best case 요약을 출력."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  MODEL 1 — Test Set Performance  (2 scaling cases)")
    print(sep)
    for label, key in [("LR", "m1_lr"), ("ResNet1D", "m1_rn")]:
        print(f"\n  [{label}]")
        print(_model_table_str(results_m1, key))

    print(f"\n{sep}")
    print("  MODEL 2 — Clinic + AEC (Matched)  (4 scaling cases)")
    print(sep)
    for label, key in [("LR", "m2_lr"), ("CrossAttn", "m2_ca"), ("ResNet1D", "m2_rn")]:
        print(f"\n  [{label}]")
        print(_model_table_str(results_m2, key))

    print(f"\n{sep}")
    print("  MODEL 2_2 — Clinic + AEC (Unmatched)  (4 scaling cases)")
    print(sep)
    for label, key in [("LR", "m2_2_lr"), ("CrossAttn", "m2_2_ca"), ("ResNet1D", "m2_2_rn")]:
        print(f"\n  [{label}]")
        print(_model_table_str(results_m2_2, key))

    print(f"\n{sep}")
    print(f"  MODEL 3 — Clinic + Scanner + AEC  ({len(AEC_VARIANTS)} AEC variants × {len(CASES_M3)} scaling cases)")
    print(sep)
    for label, key in [("LR", "m3_lr"), ("CrossAttn3", "m3_ca3"), ("ResNet1D", "m3_rn")]:
        print(f"\n  [{label}]")
        print(_model_table_str(results_m3, key))

    _print_best_summary(results_m1, results_m2, results_m2_2, results_m3)


def _md_table(results, model_key):
    """Test AUC 기준 best case 행을 **굵게** 표시."""
    best = _best_case(results, model_key)
    col_hdr = " | ".join(mn for mn, _ in _METRICS_DEF)
    col_sep = " | ".join("------:" for _ in _METRICS_DEF)
    lines = [
        f"| Case | {col_hdr} |",
        f"|------|{col_sep}|",
    ]
    for r in results:
        m = r[model_key]
        lbl = _case_label(r)
        cells = " | ".join(f"{m[mk]:.4f}" for _, mk in _METRICS_DEF)
        if r is best:
            lines.append(f"| **{lbl}** | {cells} |")
        else:
            lines.append(f"| {lbl} | {cells} |")
    return "\n".join(lines)


def _fold_stats(folds1, folds2):
    """compare_fold_metrics와 동일한 통계 계산, 출력 없이 dict만 반환."""
    from scipy import stats as scipy_stats
    result = {}
    for key in ["auc", "auprc", "brier", "acc", "f1"]:
        v1 = np.array([m[key] for m in folds1], dtype=float)
        v2 = np.array([m[key] for m in folds2], dtype=float)
        diff = v2 - v1
        t_stat, t_pval = scipy_stats.ttest_rel(v1, v2)
        try:
            _, w_pval = scipy_stats.wilcoxon(diff)
        except ValueError:
            w_pval = float("nan")
        result[key] = {
            "mean1": float(v1.mean()), "mean2": float(v2.mean()),
            "delta_mean": float(diff.mean()),
            "t_stat": float(t_stat), "t_pval": float(t_pval),
            "w_pval": float(w_pval),
        }
    return result


def _cross_model_md_block(results_a, fold_key_a, label_a,
                           results_b, fold_key_b, label_b,
                           stat_rows_fn, hdr, sep,
                           key_fn=lambda r: r["case"]):
    """key_fn으로 추출한 키가 일치하는 케이스끼리 cross-model 비교 md 블록 생성."""
    dict_a = {key_fn(r): r for r in results_a}
    dict_b = {key_fn(r): r for r in results_b}
    common = [c for c in dict_a if c in dict_b]
    if not common:
        return ["> 매칭되는 case 없음.\n"]
    lines = []
    for key in common:
        stat = _fold_stats(dict_a[key][fold_key_a], dict_b[key][fold_key_b])
        case_lbl = f"{key[0]}/{key[1]}" if isinstance(key, tuple) else key
        lines.append(f"#### Case: {case_lbl}  ({label_a} vs {label_b})")
        lines.append("")
        lines += [hdr, sep]
        lines += stat_rows_fn(stat)
        lines.append("")
    return lines


def _best_cases_summary_md(results_m1, results_m2, results_m2_2, results_m3):
    """각 모델/서브모델별 best case(Test AUC 기준) 요약 markdown 테이블 반환."""
    entries = [
        ("M1",   "LR",         results_m1,   "m1_lr"),
        ("M1",   "ResNet1D",   results_m1,   "m1_rn"),
        ("M2",   "LR",         results_m2,   "m2_lr"),
        ("M2",   "CrossAttn",  results_m2,   "m2_ca"),
        ("M2",   "ResNet1D",   results_m2,   "m2_rn"),
        ("M2_2", "LR",         results_m2_2, "m2_2_lr"),
        ("M2_2", "CrossAttn",  results_m2_2, "m2_2_ca"),
        ("M2_2", "ResNet1D",   results_m2_2, "m2_2_rn"),
        ("M3",   "LR",         results_m3,   "m3_lr"),
        ("M3",   "CrossAttn3", results_m3,   "m3_ca3"),
        ("M3",   "ResNet1D",   results_m3,   "m3_rn"),
    ]
    col_hdr = " | ".join(mn for mn, _ in _METRICS_DEF)
    col_sep = " | ".join("------:" for _ in _METRICS_DEF)
    lines = [
        f"| Model | Sub-model | Best Case | {col_hdr} |",
        f"|-------|-----------|-----------|{col_sep}|",
    ]
    for model_lbl, sub_lbl, results, key in entries:
        best = _best_case(results, key)
        m = best[key]
        cells = " | ".join(f"{m[mk]:.4f}" for _, mk in _METRICS_DEF)
        lines.append(f"| {model_lbl} | {sub_lbl} | {_case_label(best)} | {cells} |")
    return "\n".join(lines)


def _save_comparison_md(results_m1, results_m2, results_m2_2, results_m3):
    """모든 모델·케이스의 비교 테이블과 통계 검정 결과를 scaling_comparison.md로 저장."""
    lines = [
        "# Scaling Comparison — Test Set Performance",
        "",
        "## Best Cases Summary  (by Test overall AUC)",
        "",
        "> 각 모델/서브모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.",
        "",
        _best_cases_summary_md(results_m1, results_m2, results_m2_2, results_m3),
        "",
        "---",
        "",
        "## Model 1 — Clinic Only  (2 scaling cases)",
        "",
        "### Logistic Regression",
        "",
        _md_table(results_m1, "m1_lr"),
        "",
        "### ResNet1D",
        "",
        _md_table(results_m1, "m1_rn"),
        "",
        "---",
        "",
        "## Model 2 — Clinic + AEC (Matched)  (4 scaling cases)",
        "",
        "### Logistic Regression",
        "",
        _md_table(results_m2, "m2_lr"),
        "",
        "### CrossAttn",
        "",
        _md_table(results_m2, "m2_ca"),
        "",
        "### ResNet1D",
        "",
        _md_table(results_m2, "m2_rn"),
        "",
        "---",
        "",
        "## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  (4 scaling cases)",
        "",
        "> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.",
        "> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.",
        "",
        "### Logistic Regression",
        "",
        _md_table(results_m2_2, "m2_2_lr"),
        "",
        "### CrossAttn",
        "",
        _md_table(results_m2_2, "m2_2_ca"),
        "",
        "### ResNet1D",
        "",
        _md_table(results_m2_2, "m2_2_rn"),
        "",
        "---",
        "",
        "## Model 3 — Clinic + Scanner + AEC  (8 scaling cases)",
        "",
        "### Logistic Regression",
        "",
        _md_table(results_m3, "m3_lr"),
        "",
        "### CrossAttn3",
        "",
        _md_table(results_m3, "m3_ca3"),
        "",
        "### ResNet1D",
        "",
        _md_table(results_m3, "m3_rn"),
        "",
        "---",
        "",
        "# Fold-level Statistical Tests",
        "",
        "> Paired t-test + Wilcoxon signed-rank (n=5 folds)",
        "> p-value는 지수표현. Δ Mean = Model2 − Model1 (양수 → Model2 우세)",
        "> \\*\\*\\* p<0.001 · \\*\\* p<0.01 · \\* p<0.05 · † p<0.10",
        "",
        "## Model 1: LR vs ResNet1D",
        "",
    ]

    _STAT_METRICS = [("auc","AUC-ROC"),("auprc","AUPRC"),
                     ("brier","Brier"),("acc","Accuracy"),("f1","F1")]
    _STAT_HDR  = "| Metric | Mean M1 | Mean M2 | Δ Mean | t-stat | t p-val | W p-val |"
    _STAT_SEP  = "|--------|--------:|--------:|-------:|-------:|--------:|--------:|"

    def _stat_rows(stat_dict):
        rows = []
        for mk, mlabel in _STAT_METRICS:
            s = stat_dict[mk]
            sig = ("***" if s["t_pval"] < 0.001 else
                   "**"  if s["t_pval"] < 0.01  else
                   "*"   if s["t_pval"] < 0.05  else
                   "†"   if s["t_pval"] < 0.10  else "")
            rows.append(
                f"| {mlabel} {sig} "
                f"| {s['mean1']:.4f} | {s['mean2']:.4f} | {s['delta_mean']:+.4f} "
                f"| {s['t_stat']:.3f} | {s['t_pval']:.2e} | {s['w_pval']:.2e} |"
            )
        return rows

    for r in results_m1:
        lines.append(f"### [M1] Case: {r['case']}  (LR vs ResNet1D)")
        lines.append("")
        lines += [_STAT_HDR, _STAT_SEP]
        lines += _stat_rows(r["stat_lr_rn"])
        lines.append("")

    lines += [
        "## Model 2: LR vs CrossAttn",
        "",
    ]
    for r in results_m2:
        lines.append(f"### [M2] Case: {_case_label(r)}  (LR vs CrossAttn)")
        lines.append("")
        lines += [_STAT_HDR, _STAT_SEP]
        lines += _stat_rows(r["stat_lr_ca"])
        lines.append("")

    lines += [
        "## Model 2_2 (Unmatched): LR vs CrossAttn",
        "",
    ]
    for r in results_m2_2:
        lines.append(f"### [M2_2] Case: {_case_label(r)}  (LR vs CrossAttn, Unmatched)")
        lines.append("")
        lines += [_STAT_HDR, _STAT_SEP]
        lines += _stat_rows(r["stat_lr_ca"])
        lines.append("")

    lines += [
        "## Model 3: LR vs CrossAttn3",
        "",
    ]
    for r in results_m3:
        lines.append(f"### [M3] Case: {_case_label(r)}  (LR vs CrossAttn3)")
        lines.append("")
        lines += [_STAT_HDR, _STAT_SEP]
        lines += _stat_rows(r["stat_lr_ca3"])
        lines.append("")

    lines += [
        "## Model 2: LR vs ResNet1D",
        "",
    ]
    for r in results_m2:
        stat = _fold_stats(r["lr_cv_folds"], r["rn_cv_folds"])
        lines.append(f"### [M2] Case: {_case_label(r)}  (LR vs ResNet1D)")
        lines.append("")
        lines += [_STAT_HDR, _STAT_SEP]
        lines += _stat_rows(stat)
        lines.append("")

    lines += [
        "## Model 2_2 (Unmatched): LR vs ResNet1D",
        "",
    ]
    for r in results_m2_2:
        stat = _fold_stats(r["lr_cv_folds"], r["rn_cv_folds"])
        lines.append(f"### [M2_2] Case: {_case_label(r)}  (LR vs ResNet1D, Unmatched)")
        lines.append("")
        lines += [_STAT_HDR, _STAT_SEP]
        lines += _stat_rows(stat)
        lines.append("")

    lines += [
        "## Model 3: LR vs ResNet1D",
        "",
    ]
    for r in results_m3:
        stat = _fold_stats(r["lr_cv_folds"], r["rn_cv_folds"])
        lines.append(f"### [M3] Case: {_case_label(r)}  (LR vs ResNet1D)")
        lines.append("")
        lines += [_STAT_HDR, _STAT_SEP]
        lines += _stat_rows(stat)
        lines.append("")

    # ── Cross-model 비교 (같은 모델 계열, input 확장에 따른 유의성) ──────
    _CSTAT_HDR = "| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |"
    _CSTAT_SEP = "|--------|-------:|-------:|-------:|-------:|--------:|--------:|"

    def _cstat_rows(stat_dict):
        rows = []
        for mk, mlabel in _STAT_METRICS:
            s = stat_dict[mk]
            sig = ("***" if s["t_pval"] < 0.001 else
                   "**"  if s["t_pval"] < 0.01  else
                   "*"   if s["t_pval"] < 0.05  else
                   "†"   if s["t_pval"] < 0.10  else "")
            rows.append(
                f"| {mlabel} {sig} "
                f"| {s['mean1']:.4f} | {s['mean2']:.4f} | {s['delta_mean']:+.4f} "
                f"| {s['t_stat']:.3f} | {s['t_pval']:.2e} | {s['w_pval']:.2e} |"
            )
        return rows

    # M1과 비교 시 M2/M3는 full-curve baseline(len256)만 사용
    _m2_base = [r for r in results_m2   if r["aec_var"] == "len256"]
    _m3_base = [r for r in results_m3   if r["aec_var"] == "len256"]
    _duo_key = lambda r: (r["aec_var"], r["case"])  # noqa: E731

    lines += [
        "---",
        "",
        "# Cross-Model Comparison — 동일 모델 계열, Input 확장에 따른 유의성",
        "",
        "> LR / Deep Model 각각에서 M1/M2/M2_2/M3 간 pairwise 비교.",
        "> M1 비교 시 M2/M3는 len256(full AEC baseline)만 사용.",
        "> M2↔M2_2, M2↔M3 비교 시 (aec_var, case) 키로 매칭. Δ Mean = B − A (양수 → B 우세).",
        "> M2_2는 Clinic-AEC Unmatching 음성 대조군.",
        "> \\*\\*\\* p<0.001 · \\*\\* p<0.01 · \\* p<0.05 · † p<0.10",
        "",
        "## LR: M1 (Clinic) vs M2 (Clinic+AEC, Matched, len256)",
        "",
    ]
    lines += _cross_model_md_block(
        results_m1, "lr_cv_folds", "M1-LR",
        _m2_base,   "lr_cv_folds", "M2-LR(len256)",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
    )

    lines += [
        "## LR: M1 (Clinic) vs M3 (Clinic+Scanner+AEC, len256)",
        "",
    ]
    lines += _cross_model_md_block(
        results_m1, "lr_cv_folds", "M1-LR",
        _m3_base,   "lr_cv_folds", "M3-LR(len256)",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
    )

    lines += [
        "## LR: M2 (Matched) vs M2_2 (Unmatched) — 음성 대조",
        "",
    ]
    lines += _cross_model_md_block(
        results_m2,   "lr_cv_folds", "M2-LR",
        results_m2_2, "lr_cv_folds", "M2_2-LR",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
        key_fn=_duo_key,
    )

    lines += [
        "## LR: M2 (Clinic+AEC) vs M3 (Clinic+Scanner+AEC)",
        "",
    ]
    lines += _cross_model_md_block(
        results_m2, "lr_cv_folds", "M2-LR",
        results_m3, "lr_cv_folds", "M3-LR",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
        key_fn=_duo_key,
    )

    lines += [
        "---",
        "",
        "## Deep (CrossAttn): M1 ResNet1D (Clinic) vs M2 CrossAttn (Clinic+AEC, len256)",
        "",
    ]
    lines += _cross_model_md_block(
        results_m1, "rn_cv_folds", "M1-ResNet1D",
        _m2_base,   "ca_cv_folds", "M2-CrossAttn(len256)",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
    )

    lines += [
        "## Deep (CrossAttn): M1 ResNet1D (Clinic) vs M3 CrossAttn3 (Clinic+Scanner+AEC, len256)",
        "",
    ]
    lines += _cross_model_md_block(
        results_m1, "rn_cv_folds",  "M1-ResNet1D",
        _m3_base,   "ca3_cv_folds", "M3-CrossAttn3(len256)",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
    )

    lines += [
        "## Deep (CrossAttn): M2 CrossAttn (Matched) vs M2_2 CrossAttn (Unmatched) — 음성 대조",
        "",
    ]
    lines += _cross_model_md_block(
        results_m2,   "ca_cv_folds", "M2-CrossAttn",
        results_m2_2, "ca_cv_folds", "M2_2-CrossAttn",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
        key_fn=_duo_key,
    )

    lines += [
        "## Deep (CrossAttn): M2 CrossAttn (Clinic+AEC) vs M3 CrossAttn3 (Clinic+Scanner+AEC)",
        "",
    ]
    lines += _cross_model_md_block(
        results_m2, "ca_cv_folds",  "M2-CrossAttn",
        results_m3, "ca3_cv_folds", "M3-CrossAttn3",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
        key_fn=_duo_key,
    )

    lines += [
        "---",
        "",
        "## Deep (ResNet1D): M1 ResNet1D (Clinic) vs M2 ResNet1D (Clinic+AEC, len256)",
        "",
    ]
    lines += _cross_model_md_block(
        results_m1, "rn_cv_folds", "M1-ResNet1D",
        _m2_base,   "rn_cv_folds", "M2-ResNet1D(len256)",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
    )

    lines += [
        "## Deep (ResNet1D): M1 ResNet1D (Clinic) vs M3 ResNet1D (Clinic+Scanner+AEC, len256)",
        "",
    ]
    lines += _cross_model_md_block(
        results_m1, "rn_cv_folds", "M1-ResNet1D",
        _m3_base,   "rn_cv_folds", "M3-ResNet1D(len256)",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
    )

    lines += [
        "## Deep (ResNet1D): M2 ResNet1D (Matched) vs M2_2 ResNet1D (Unmatched) — 음성 대조",
        "",
    ]
    lines += _cross_model_md_block(
        results_m2,   "rn_cv_folds", "M2-ResNet1D",
        results_m2_2, "rn_cv_folds", "M2_2-ResNet1D",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
        key_fn=_duo_key,
    )

    lines += [
        "## Deep (ResNet1D): M2 ResNet1D (Clinic+AEC) vs M3 ResNet1D (Clinic+Scanner+AEC)",
        "",
    ]
    lines += _cross_model_md_block(
        results_m2, "rn_cv_folds", "M2-ResNet1D",
        results_m3, "rn_cv_folds", "M3-ResNet1D",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
        key_fn=_duo_key,
    )

    md_path = os.path.join(os.path.dirname(RESULTS_DIR), "scaling_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Comparison saved → {md_path}")


if __name__ == "__main__":
    run_all_cases()
