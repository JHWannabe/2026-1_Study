"""
SMI Binary Classification — 모델별 AEC variant 비교 실험 진입점.

모델 구성:
  Model 1   : Clinic Only (Age, Sex, BMI)                — LR
  Model 2   : Clinic + AEC Matched                       — CrossAttn
  Model 2_2 : Clinic + AEC Unmatched (음성 대조군)         — CrossAttn
  Model 3   : Clinic + Scanner(MFR Embedding) + AEC      — CrossAttn3

각 모델은 AEC_VARIANTS(5종) × AEC_SIZES(2종) 조합을 순차 실행한다.
Model 1/2/2_2/3 는 ProcessPoolExecutor 로 병렬 실행되며,
결과는 scaling_comparison.md 와 각 모델 디렉토리 run.log 에 저장된다.

스케일링 원칙:
  - Clinic(Age·BMI): StandardScaler 항상 적용
  - AEC: StandardScaler 항상 적용 (열 방향 표준화)
  - sex_enc(이진값), label(y, 0/1), MFR index에는 StandardScaler를 적용하지 않는다
  - AEC 행 방향 정규화는 "norm" variant sensitivity로만 비교

Attention Map 시각화:
  Model 2/2_2/3의 CrossAttn 최종 모델로 test set에서 attention weight를 추출한다.
  Clinical↔AEC 양방향 attention을 클래스 분리 bar chart와 샘플별 heatmap으로
  각 디렉토리에 저장한다.
"""

import os
import sys
import io
import numpy as np
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    average_precision_score, brier_score_loss,
)

import matplotlib
matplotlib.use('Agg')

from config import DEVICE, RESULTS_DIR, RESULTS_MODEL_1_DIR, RESULTS_MODEL_2_DIR, RESULTS_MODEL_2_2_DIR, RESULTS_MODEL_3_DIR, AEC_VARIANTS, AEC_SIZES
from data import (load_data, split_data,
                  load_data_with_aec, load_data_with_aec_unmatched, split_data_dual,
                  load_data_with_aec_meta, split_data_quad,
                  aec_variant, print_stats, describe_dataset)
from cross_val import (run_cross_validation, run_cross_validation_cross,
                       run_cross_validation_cross3)
from evaluate import evaluate_test, evaluate_test_cross, evaluate_test_cross3
from metrics import print_cv_summary, print_delong_comparison, delong_test
from visualize import (save_all, save_all_cross, plot_test_roc_with_baseline,
                       plot_roc_all_models, plot_attention_maps)


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
    """aec_var 가 있으면 aec_var, 없으면 case 반환."""
    return r.get("aec_var", r["case"])


# ── 모델별 워커 함수 (subprocess 에서 실행) ───────────────────
# 각 모델의 모든 케이스를 순차 실행하고 결과 리스트를 반환.
# 상세 출력은 <model_results_dir>/run.log 에 저장.

def _run_model1(X_cv, y_cv, sex_cv, X_te, y_te, sex_te):
    """Model 1(Clinic Only)을 실행하고 결과 리스트를 반환. 로그는 run.log에 저장."""
    buf = io.StringIO()
    sys.stdout = buf
    results = []
    try:
        print(f"{'='*60}")
        print("  MODEL 1 — Clinic Only")
        print(f"{'='*60}")

        out1 = RESULTS_MODEL_1_DIR
        os.makedirs(out1, exist_ok=True)

        print(f"  [M1] Cross-validating LR ...", flush=True)
        (lr_cv, lr_roc_folds, lr_best_thresholds) = run_cross_validation(X_cv, y_cv, scale_X=True)

        print_cv_summary("LR", lr_cv)

        med_thresh_lr = float(np.median(lr_best_thresholds))
        print(f"  [M1] Evaluating on test set (thresh={med_thresh_lr:.3f}) ...", flush=True)
        (lr_pred, lr_prob, stats_te) = evaluate_test(
            X_cv, y_cv, X_te, y_te, sex_te, scale_X=True, threshold=med_thresh_lr
        )

        print(f"  [M1] Saving figures ...", flush=True)
        save_all(
            lr_roc_folds, lr_cv,
            X_cv, y_cv, sex_cv,
            X_te, y_te, lr_pred, lr_prob,
            sex_te, out_dir=out1,
        )
        print(f"  [M1] Done.", flush=True)

        results.append({
            "case":        "scale_all",
            "m1_lr":       _metrics(y_te, lr_pred, lr_prob),
            "lr_cv_folds": lr_cv,
            "test_stats":  stats_te,
            "y_te":        y_te,
            "lr_prob":     lr_prob,
        })
    finally:
        sys.stdout = sys.__stdout__

    with open(os.path.join(RESULTS_MODEL_1_DIR, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return results


def _run_model2(X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                X_clin_te, X_aec_te, y2_te, sex2_te,
                aec_size: int = 128, aec_variants: list | None = None,
                progress_queue=None):
    """Model 2(Clinic+AEC Matched): AEC 변형별로 CrossAttn을 실행하고 결과 리스트를 반환.
    평가 후 attention map 4종을 케이스 디렉토리에 저장. 로그는 run.log에 저장."""
    if aec_variants is None:
        aec_variants = AEC_VARIANTS
    buf = io.StringIO()
    sys.stdout = buf
    results = []
    try:
        print(f"{'='*60}")
        print(f"  MODEL 2 — Clinic + AEC (aec{aec_size})  ({len(aec_variants)} AEC variants)")
        print(f"{'='*60}")

        for aec_var in aec_variants:
            X_aec_cv_v, mask_cv = aec_variant(X_aec_cv, aec_var)
            X_aec_te_v, mask_te = aec_variant(X_aec_te, aec_var)
            # excl_extreme 변형은 샘플 필터링 포함
            X_clin_cv_v = X_clin_cv[mask_cv] if mask_cv is not None else X_clin_cv
            y2_cv_v     = y2_cv[mask_cv]     if mask_cv is not None else y2_cv
            sex2_cv_v   = sex2_cv[mask_cv]   if mask_cv is not None else sex2_cv
            X_clin_te_v = X_clin_te[mask_te] if mask_te is not None else X_clin_te
            y2_te_v     = y2_te[mask_te]     if mask_te is not None else y2_te
            sex2_te_v   = sex2_te[mask_te]   if mask_te is not None else sex2_te

            print(f"\n{'#'*60}")
            print(f"  [M2 {aec_var}]  (scale_clinic=True, scale_aec=True)")
            print(f"{'#'*60}")

            out2 = os.path.join(RESULTS_MODEL_2_DIR, f"aec{aec_size}", aec_var)
            os.makedirs(out2, exist_ok=True)

            print(f"  [M2/{aec_var}] Cross-validating CrossAttn ...", flush=True)
            (ca_cv, ca_roc_folds,
             ca_histories, ca_best_epochs2,
             ca_best_thresholds2) = run_cross_validation_cross(
                X_clin_cv_v, X_aec_cv_v, y2_cv_v, scale_clin=True,
            )

            print_cv_summary("CrossAttn", ca_cv)

            med_epoch2 = int(np.median(ca_best_epochs2))
            med_thresh2 = float(np.median(ca_best_thresholds2))
            print(f"  [M2/{aec_var}] Evaluating on test set (med_epoch={med_epoch2}, thresh={med_thresh2:.3f}) ...", flush=True)
            (ca_pred_te, ca_prob_te, ca_true_te, stats_te2,
             model_te2, X_clin_te_s2, X_aec_te_s2) = evaluate_test_cross(  # type: ignore[misc]
                X_clin_cv_v, X_aec_cv_v, y2_cv_v,
                X_clin_te_v, X_aec_te_v, y2_te_v, sex2_te_v,
                med_epoch2, scale_clin=True, return_model=True,
                threshold=med_thresh2,
            )

            print(f"  [M2/{aec_var}] Saving figures ...", flush=True)
            save_all_cross(
                ca_cv, ca_roc_folds, ca_histories, med_epoch2,
                X_clin_cv_v, y2_cv_v, sex2_cv_v,
                X_clin_te_v, y2_te_v,
                ca_pred_te, ca_true_te, sex2_te_v, ca_prob_te,
                model_label=f"model 2 ({aec_var})", out_dir=out2,
            )

            print(f"  [M2/{aec_var}] Plotting attention maps ...", flush=True)
            plot_attention_maps(
                model_te2, X_clin_te_s2, X_aec_te_s2, ca_true_te,
                out_dir=out2, aec_var=aec_var,
                model_label=f"Model 2 ({aec_var})",
            )
            print(f"  [M2/{aec_var}] Done.", flush=True)

            results.append({
                "aec_var":     aec_var,
                "case":        aec_var,
                "out_dir":     out2,
                "m2_ca":       _metrics(ca_true_te, ca_pred_te, ca_prob_te),
                "ca_cv_folds": ca_cv,
                "test_stats":  stats_te2,
                "y_true_te":   ca_true_te,
                "ca_prob_te":  ca_prob_te,
            })
            if progress_queue is not None:
                progress_queue.put(("M2", aec_var))
    finally:
        sys.stdout = sys.__stdout__

    log_dir = os.path.join(RESULTS_MODEL_2_DIR, f"aec{aec_size}")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return results


def _run_model2_2(X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                  X_clin_te, X_aec_te, y2_te, sex2_te,
                  aec_size: int = 128, aec_variants: list | None = None,
                  progress_queue=None):
    """
    Model 2_2: Model 2와 동일한 ClinAECCrossAttn 구조를 사용하되
    Clinic-AEC가 서로 다른 환자 데이터로 섞인 상태(Unmatching)로 학습/평가.
    Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 가진다는 증거.
    평가 후 attention map 4종을 케이스 디렉토리에 저장.
    """
    if aec_variants is None:
        aec_variants = AEC_VARIANTS
    buf = io.StringIO()
    sys.stdout = buf
    results = []
    try:
        print(f"{'='*60}")
        print(f"  MODEL 2_2 — Clinic + AEC Unmatched (aec{aec_size})  ({len(aec_variants)} AEC variants)")
        print(f"{'='*60}")

        for aec_var in aec_variants:
            X_aec_cv_v, mask_cv = aec_variant(X_aec_cv, aec_var)
            X_aec_te_v, mask_te = aec_variant(X_aec_te, aec_var)
            X_clin_cv_v = X_clin_cv[mask_cv] if mask_cv is not None else X_clin_cv
            y2_cv_v     = y2_cv[mask_cv]     if mask_cv is not None else y2_cv
            sex2_cv_v   = sex2_cv[mask_cv]   if mask_cv is not None else sex2_cv
            X_clin_te_v = X_clin_te[mask_te] if mask_te is not None else X_clin_te
            y2_te_v     = y2_te[mask_te]     if mask_te is not None else y2_te
            sex2_te_v   = sex2_te[mask_te]   if mask_te is not None else sex2_te

            print(f"\n{'#'*60}")
            print(f"  [M2_2 {aec_var}]  (scale_clinic=True, scale_aec=True)")
            print(f"{'#'*60}")

            out2_2 = os.path.join(RESULTS_MODEL_2_2_DIR, f"aec{aec_size}", aec_var)
            os.makedirs(out2_2, exist_ok=True)

            print(f"  [M2_2/{aec_var}] Cross-validating CrossAttn ...", flush=True)
            (ca_cv, ca_roc_folds,
             ca_histories, ca_best_epochs2_2,
             ca_best_thresholds2_2) = run_cross_validation_cross(
                X_clin_cv_v, X_aec_cv_v, y2_cv_v, scale_clin=True,
            )

            print_cv_summary("CrossAttn", ca_cv)

            med_epoch2_2 = int(np.median(ca_best_epochs2_2))
            med_thresh2_2 = float(np.median(ca_best_thresholds2_2))
            print(f"  [M2_2/{aec_var}] Evaluating on test set (med_epoch={med_epoch2_2}, thresh={med_thresh2_2:.3f}) ...", flush=True)
            (ca_pred_te, ca_prob_te, ca_true_te, stats_te2_2,
             model_te2_2, X_clin_te_s2_2, X_aec_te_s2_2) = evaluate_test_cross(  # type: ignore[misc]
                X_clin_cv_v, X_aec_cv_v, y2_cv_v,
                X_clin_te_v, X_aec_te_v, y2_te_v, sex2_te_v,
                med_epoch2_2, scale_clin=True, return_model=True,
                threshold=med_thresh2_2,
            )

            print(f"  [M2_2/{aec_var}] Saving figures ...", flush=True)
            save_all_cross(
                ca_cv, ca_roc_folds, ca_histories, med_epoch2_2,
                X_clin_cv_v, y2_cv_v, sex2_cv_v,
                X_clin_te_v, y2_te_v,
                ca_pred_te, ca_true_te, sex2_te_v, ca_prob_te,
                model_label=f"model 2_2 ({aec_var}, unmatched)", out_dir=out2_2,
            )

            print(f"  [M2_2/{aec_var}] Plotting attention maps ...", flush=True)
            plot_attention_maps(
                model_te2_2, X_clin_te_s2_2, X_aec_te_s2_2, ca_true_te,
                out_dir=out2_2, aec_var=aec_var,
                model_label=f"Model 2_2 Unmatched ({aec_var})",
            )
            print(f"  [M2_2/{aec_var}] Done.", flush=True)

            results.append({
                "aec_var":     aec_var,
                "case":        aec_var,
                "out_dir":     out2_2,
                "m2_2_ca":     _metrics(ca_true_te, ca_pred_te, ca_prob_te),
                "ca_cv_folds": ca_cv,
                "test_stats":  stats_te2_2,
                "y_true_te":   ca_true_te,
                "ca_prob_te":  ca_prob_te,
            })
            if progress_queue is not None:
                progress_queue.put(("M2_2", aec_var))
    finally:
        sys.stdout = sys.__stdout__

    log_dir = os.path.join(RESULTS_MODEL_2_2_DIR, f"aec{aec_size}")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return results


def _run_model3(X_clin3_cv, X_aec3_cv, X_mfr_cv, y3_cv, sex3_cv,
                X_clin3_te, X_aec3_te, X_mfr_te, y3_te, sex3_te, n_mfr,
                aec_size: int = 128, aec_variants: list | None = None,
                progress_queue=None):
    """Model 3(Clinic+Scanner+AEC): AEC 변형별로 CrossAttn3를 실행하고 결과 리스트를 반환.
    평가 후 attention map 4종을 케이스 디렉토리에 저장. 로그는 run.log에 저장."""
    if aec_variants is None:
        aec_variants = AEC_VARIANTS
    buf = io.StringIO()
    sys.stdout = buf
    results = []
    try:
        print(f"{'='*60}")
        print(f"  MODEL 3 — Clinic + Scanner + AEC (aec{aec_size})  ({len(aec_variants)} AEC variants)")
        print(f"{'='*60}")

        for aec_var in aec_variants:
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

            print(f"\n{'#'*60}")
            print(f"  [M3 {aec_var}]  (scale_clinic=True, scale_aec=True)")
            print(f"{'#'*60}")

            out3 = os.path.join(RESULTS_MODEL_3_DIR, f"aec{aec_size}", aec_var)
            os.makedirs(out3, exist_ok=True)

            print(f"  [M3/{aec_var}] Cross-validating CrossAttn3 ...", flush=True)
            (ca3_cv, ca3_roc_folds,
             ca3_histories, ca3_best_epochs3,
             ca3_best_thresholds3) = run_cross_validation_cross3(
                X_clin3_cv_v, X_aec3_cv_v, X_mfr3_cv_v, y3_cv_v, n_mfr,
                scale_clin=True,
            )

            print_cv_summary("CrossAttn3", ca3_cv)

            med_epoch3 = int(np.median(ca3_best_epochs3))
            med_thresh3 = float(np.median(ca3_best_thresholds3))
            print(f"  [M3/{aec_var}] Evaluating on test set (med_epoch={med_epoch3}, thresh={med_thresh3:.3f}) ...", flush=True)
            (ca3_pred_te, ca3_prob_te, ca3_true_te, stats_te3,
             model_te3, X_clin3_te_s, X_aec3_te_s) = evaluate_test_cross3(  # type: ignore[misc]
                X_clin3_cv_v, X_aec3_cv_v, X_mfr3_cv_v, y3_cv_v,
                X_clin3_te_v, X_aec3_te_v, X_mfr3_te_v, y3_te_v,
                sex3_te_v, med_epoch3, n_mfr,
                scale_clin=True, return_model=True,
                threshold=med_thresh3,
            )

            print(f"  [M3/{aec_var}] Saving figures ...", flush=True)
            save_all_cross(
                ca3_cv, ca3_roc_folds, ca3_histories, med_epoch3,
                X_clin3_cv_v, y3_cv_v, sex3_cv_v,
                X_clin3_te_v, y3_te_v,
                ca3_pred_te, ca3_true_te, sex3_te_v, ca3_prob_te,
                model_label=f"model 3 ({aec_var})", out_dir=out3,
            )

            print(f"  [M3/{aec_var}] Plotting attention maps ...", flush=True)
            plot_attention_maps(
                model_te3, X_clin3_te_s, X_aec3_te_s, ca3_true_te,
                out_dir=out3, aec_var=aec_var,
                model_label=f"Model 3 ({aec_var})",
                X_mfr_te=X_mfr3_te_v,
            )
            print(f"  [M3/{aec_var}] Done.", flush=True)

            results.append({
                "aec_var":      aec_var,
                "case":         aec_var,
                "out_dir":      out3,
                "m3_ca3":       _metrics(ca3_true_te, ca3_pred_te, ca3_prob_te),
                "ca3_cv_folds": ca3_cv,
                "test_stats":   stats_te3,
                "y_true_te":    ca3_true_te,
                "ca3_prob_te":  ca3_prob_te,
            })
            if progress_queue is not None:
                progress_queue.put(("M3", aec_var))
    finally:
        sys.stdout = sys.__stdout__

    log_dir = os.path.join(RESULTS_MODEL_3_DIR, f"aec{aec_size}")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return results


def _print_delong_comparisons(results_m1, results_m2, results_m2_2, results_m3,
                              aec_size: int = 128):
    """모델 간 Test-set AUC를 DeLong (1988) 검정으로 쌍별 비교해 콘솔 출력.

    비교 쌍:
      M1 LR  vs  M2 CrossAttn     (aec_var별, 동일 test set인 경우만)
      M1 LR  vs  M3 CrossAttn3    (aec_var별, 동일 test set인 경우만)
      M2 Matched  vs  M2_2 Unmatched  (aec_var별, 동일 크기인 경우만)
      M2 CrossAttn  vs  M3 CrossAttn3 (aec_var별, 동일 test set)

    excl_extreme 변형은 M1 test set과 크기가 달라 M1 vs M2/M3 비교에서 제외.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  AEC {aec_size}pt — DeLong Test  (Test-set ROC AUC 쌍별 비교)")
    print(sep)

    r1      = results_m1[0]
    m1_y    = r1["y_te"]
    m1_prob = r1["lr_prob"]

    r2_2_dict = {r["aec_var"]: r for r in results_m2_2}
    r3_dict   = {r["aec_var"]: r for r in results_m3}

    # ── M1 vs M2 ──────────────────────────────────────────────
    print(f"\n  [M1 LR  vs  M2 CrossAttn]")
    for r2 in results_m2:
        y2 = r2["y_true_te"]
        if len(y2) != len(m1_y):
            print(f"    {r2['aec_var']} — skip (sample size mismatch: M1={len(m1_y)}, M2={len(y2)})")
            continue
        print_delong_comparison(
            f"M1-LR", f"M2-{r2['aec_var']}",
            m1_y, m1_prob, r2["ca_prob_te"],
        )

    # ── M1 vs M3 ──────────────────────────────────────────────
    print(f"\n  [M1 LR  vs  M3 CrossAttn3]")
    for r3 in results_m3:
        y3 = r3["y_true_te"]
        if len(y3) != len(m1_y):
            print(f"    {r3['aec_var']} — skip (sample size mismatch: M1={len(m1_y)}, M3={len(y3)})")
            continue
        print_delong_comparison(
            f"M1-LR", f"M3-{r3['aec_var']}",
            m1_y, m1_prob, r3["ca3_prob_te"],
        )

    # ── M2 Matched vs M2_2 Unmatched ──────────────────────────
    print(f"\n  [M2 Matched  vs  M2_2 Unmatched]")
    for r2 in results_m2:
        key = r2["aec_var"]
        r22 = r2_2_dict.get(key)
        if r22 is None:
            continue
        if len(r2["y_true_te"]) != len(r22["y_true_te"]):
            print(f"    {key} — skip (sample size mismatch)")
            continue
        print_delong_comparison(
            f"M2-{r2['aec_var']}", f"M2_2-{r22['aec_var']}",
            r2["y_true_te"], r2["ca_prob_te"], r22["ca_prob_te"],
        )

    # ── M2 vs M3 ──────────────────────────────────────────────
    print(f"\n  [M2 CrossAttn  vs  M3 CrossAttn3]")
    for r2 in results_m2:
        key = r2["aec_var"]
        r3 = r3_dict.get(key)
        if r3 is None:
            continue
        if len(r2["y_true_te"]) != len(r3["y_true_te"]):
            print(f"    {key} — skip (sample size mismatch)")
            continue
        print_delong_comparison(
            f"M2-{r2['aec_var']}", f"M3-{r3['aec_var']}",
            r2["y_true_te"], r2["ca_prob_te"], r3["ca3_prob_te"],
        )


def _plot_comparison_roc_curves(results_m1, results_m2, results_m2_2, results_m3,
                                aec_size: int = 128):
    """병렬 실행 완료 후, baseline을 포함한 test_roc_curves.png를 각 디렉토리에 덮어씀.
    - M2/M3: Model 1 LR을 baseline으로 비교
    - M2_2: 동일 aec_var의 Model 2 Matched를 baseline으로 비교
    결과 경로는 각 result dict의 out_dir 필드를 사용.
    """
    r1 = results_m1[0]
    m1_y_te    = r1["y_te"]
    m1_lr_prob = r1["lr_prob"]

    size_tag = f"aec{aec_size}"

    for r in results_m2:
        plot_test_roc_with_baseline(
            primary_true=r["y_true_te"],
            primary_prob=r["ca_prob_te"],
            primary_label=f"Model 2 CrossAttn ({r['aec_var']})",
            baseline_true=m1_y_te,
            baseline_prob=m1_lr_prob,
            baseline_label="Model 1 LR (baseline)",
            out_path=os.path.join(r["out_dir"], "test_roc_curves.png"),
        )

    for r in results_m3:
        plot_test_roc_with_baseline(
            primary_true=r["y_true_te"],
            primary_prob=r["ca3_prob_te"],
            primary_label=f"Model 3 CrossAttn3 ({r['aec_var']})",
            baseline_true=m1_y_te,
            baseline_prob=m1_lr_prob,
            baseline_label="Model 1 LR (baseline)",
            out_path=os.path.join(r["out_dir"], "test_roc_curves.png"),
        )

    m2_dict = {r["aec_var"]: r for r in results_m2}
    for r in results_m2_2:
        key = r["aec_var"]
        r2 = m2_dict.get(key)
        if r2 is None:
            continue
        plot_test_roc_with_baseline(
            primary_true=r["y_true_te"],
            primary_prob=r["ca_prob_te"],
            primary_label=f"Model 2_2 Unmatched ({r['aec_var']})",
            baseline_true=r2["y_true_te"],
            baseline_prob=r2["ca_prob_te"],
            baseline_label=f"Model 2 Matched ({r['aec_var']}, baseline)",
            out_path=os.path.join(r["out_dir"], "test_roc_curves.png"),
        )

    # ── aec_var별 전체 모델 비교 (하나의 이미지) ─────────────────
    comparison_dir = os.path.join(os.path.dirname(RESULTS_MODEL_3_DIR), "comparison", size_tag)
    os.makedirs(comparison_dir, exist_ok=True)

    aec_variants_used = list({r["aec_var"] for r in results_m2})
    r2_dict   = {r["aec_var"]: r for r in results_m2}
    r2_2_dict = {r["aec_var"]: r for r in results_m2_2}
    r3_dict   = {r["aec_var"]: r for r in results_m3}

    for aec_var in aec_variants_used:
        if aec_var not in r2_dict or aec_var not in r2_2_dict or aec_var not in r3_dict:
            continue
        r2  = r2_dict[aec_var]
        r22 = r2_2_dict[aec_var]
        r3  = r3_dict[aec_var]
        plot_roc_all_models(
            aec_var=aec_var,
            r1_y=r1["y_te"],         r1_prob=r1["lr_prob"],
            r2_y=r2["y_true_te"],    r2_prob=r2["ca_prob_te"],
            r2_2_y=r22["y_true_te"], r2_2_prob=r22["ca_prob_te"],
            r3_y=r3["y_true_te"],    r3_prob=r3["ca3_prob_te"],
            out_path=os.path.join(comparison_dir, f"roc_all_models_{aec_var}.png"),
        )

    print(f"  [{size_tag}] Comparison ROC curves saved.")


def run_all_cases():
    """Model 1은 1회, Model 2/2_2/3은 AEC_SIZES × AEC_VARIANTS 조합으로 실행해 결과를 비교·저장."""
    print(f"Device  : {DEVICE}\n")

    describe_dataset()

    # ── Model 1: AEC 미사용, 1회만 실행 ──────────────────────
    print("\n[Data] Loading Model 1 data ...")
    X, y, sex = load_data()
    X_cv, y_cv, sex_cv, X_te, y_te, sex_te = split_data(X, y, sex)
    print("[Data] Model 1 data ready.")
    print("=== Model 1 dataset ==="); print_stats(y, sex)

    print("\n[Model 1] Starting ...")
    with ProcessPoolExecutor(max_workers=1) as executor:
        results_m1 = executor.submit(
            _run_model1, X_cv, y_cv, sex_cv, X_te, y_te, sex_te,
        ).result()
    print("[Model 1] Finished.\n")

    # ── Model 2/2_2/3: AEC 크기별 반복 실행 ──────────────────
    for aec_size in AEC_SIZES:
        aec_sheet    = f"aec_{aec_size}"
        aec_variants = AEC_VARIANTS

        print(f"\n{'='*60}")
        print(f"  AEC SIZE : {aec_size} points  (sheet={aec_sheet})")
        print(f"{'='*60}\n")

        print(f"[Data] Loading aec{aec_size} datasets ...")
        X_clin,  X_aec,  y2,  sex2  = load_data_with_aec(aec_len=aec_size, aec_sheet=aec_sheet)
        X_clin_u, X_aec_u, y2u, sex2u = load_data_with_aec_unmatched(aec_len=aec_size, aec_sheet=aec_sheet)
        X_clin3, X_aec3, X_scan_mfr, y3, sex3, n_mfr = load_data_with_aec_meta(aec_len=aec_size, aec_sheet=aec_sheet)
        print(f"[Data] aec{aec_size} datasets loaded.")

        print(f"=== Model 2   dataset (aec{aec_size}) ==="); print_stats(y2,  sex2)
        print(f"=== Model 2_2 dataset (aec{aec_size}) ==="); print_stats(y2u, sex2u)
        print(f"=== Model 3   dataset (aec{aec_size}) ==="); print_stats(y3,  sex3)

        print(f"[Data] Splitting aec{aec_size} datasets ...")
        X_clin_cv,  X_aec_cv,  y2_cv,  sex2_cv, \
        X_clin_te,  X_aec_te,  y2_te,  sex2_te  = split_data_dual(X_clin,   X_aec,   y2,  sex2)
        X_clin_ucv, X_aec_ucv, y2u_cv, sex2u_cv, \
        X_clin_ute, X_aec_ute, y2u_te, sex2u_te = split_data_dual(X_clin_u, X_aec_u, y2u, sex2u)
        (X_clin3_cv, X_aec3_cv, X_mfr_cv, y3_cv, sex3_cv,
         X_clin3_te, X_aec3_te, X_mfr_te, y3_te, sex3_te) = split_data_quad(
            X_clin3, X_aec3, X_scan_mfr, y3, sex3,
        )
        print(f"[Data] aec{aec_size} splits ready.")

        print(f"\n  Launching Model 2, 2_2, 3 in parallel (aec{aec_size}) ...\n")

        n_variants = len(aec_variants)
        with Manager() as mp_manager:
            q = mp_manager.Queue()
            bars = {
                "M2":   tqdm(total=n_variants, desc=f"M2    aec{aec_size}", position=0, leave=True),
                "M2_2": tqdm(total=n_variants, desc=f"M2_2  aec{aec_size}", position=1, leave=True),
                "M3":   tqdm(total=n_variants, desc=f"M3    aec{aec_size}", position=2, leave=True),
            }
            with ProcessPoolExecutor(max_workers=3) as executor:
                fut2 = executor.submit(
                    _run_model2,
                    X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                    X_clin_te, X_aec_te, y2_te, sex2_te,
                    aec_size, aec_variants, q,
                )
                fut2_2 = executor.submit(
                    _run_model2_2,
                    X_clin_ucv, X_aec_ucv, y2u_cv, sex2u_cv,
                    X_clin_ute, X_aec_ute, y2u_te, sex2u_te,
                    aec_size, aec_variants, q,
                )
                fut3 = executor.submit(
                    _run_model3,
                    X_clin3_cv, X_aec3_cv, X_mfr_cv, y3_cv, sex3_cv,
                    X_clin3_te, X_aec3_te, X_mfr_te, y3_te, sex3_te, n_mfr,
                    aec_size, aec_variants, q,
                )
                total_updates = n_variants * 3
                received = 0
                while received < total_updates:
                    try:
                        model, done_var = q.get(timeout=0.5)
                        bars[model].update(1)
                        bars[model].set_postfix(variant=done_var)
                        received += 1
                    except Exception:
                        if all(f.done() for f in [fut2, fut2_2, fut3]):
                            break
                results_m2   = fut2.result()
                results_m2_2 = fut2_2.result()
                results_m3   = fut3.result()
            for bar in bars.values():
                bar.close()

        print(f"  [aec{aec_size}] All models done.\n")

        print(f"[aec{aec_size}] Plotting comparison ROC curves ...")
        _plot_comparison_roc_curves(results_m1, results_m2, results_m2_2, results_m3, aec_size)
        print(f"[aec{aec_size}] Printing comparison table ...")
        _print_comparison(results_m1, results_m2, results_m2_2, results_m3, aec_size)
        _print_delong_comparisons(results_m1, results_m2, results_m2_2, results_m3, aec_size)
        print(f"[aec{aec_size}] Saving comparison markdown ...")
        _save_comparison_md(results_m1, results_m2, results_m2_2, results_m3, aec_size)
        print(f"[aec{aec_size}] All done.\n")


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
    """각 모델의 best case(Test AUC 기준)를 요약 출력."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  BEST CASES SUMMARY  (by Test overall AUC)")
    print(sep)
    entries = [
        ("M1",   "LR",         results_m1,   "m1_lr"),
        ("M2",   "CrossAttn",  results_m2,   "m2_ca"),
        ("M2_2", "CrossAttn",  results_m2_2, "m2_2_ca"),
        ("M3",   "CrossAttn3", results_m3,   "m3_ca3"),
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


def _print_comparison(results_m1, results_m2, results_m2_2, results_m3,
                      aec_size: int = 128):
    """Model 1~3의 모든 case 결과를 콘솔 테이블로 출력하고, 마지막에 best case 요약을 출력."""
    n_var = len({r["aec_var"] for r in results_m2})
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  AEC {aec_size}pt — MODEL 1 — Test Set Performance  (1 scaling case)")
    print(sep)
    print(f"\n  [LR]")
    print(_model_table_str(results_m1, "m1_lr"))

    print(f"\n{sep}")
    print(f"  AEC {aec_size}pt — MODEL 2 — Clinic + AEC (Matched)  ({n_var} AEC variants)")
    print(sep)
    print(f"\n  [CrossAttn]")
    print(_model_table_str(results_m2, "m2_ca"))

    print(f"\n{sep}")
    print(f"  AEC {aec_size}pt — MODEL 2_2 — Clinic + AEC (Unmatched)  ({n_var} AEC variants)")
    print(sep)
    print(f"\n  [CrossAttn]")
    print(_model_table_str(results_m2_2, "m2_2_ca"))

    print(f"\n{sep}")
    print(f"  AEC {aec_size}pt — MODEL 3 — Clinic + Scanner + AEC  ({n_var} AEC variants)")
    print(sep)
    print(f"\n  [CrossAttn3]")
    print(_model_table_str(results_m3, "m3_ca3"))

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
            w_pval = float(scipy_stats.wilcoxon(diff)[1])  # type: ignore[arg-type]
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
    """각 모델별 best case(Test AUC 기준) 요약 markdown 테이블 반환."""
    entries = [
        ("M1",   "LR",         results_m1,   "m1_lr"),
        ("M2",   "CrossAttn",  results_m2,   "m2_ca"),
        ("M2_2", "CrossAttn",  results_m2_2, "m2_2_ca"),
        ("M3",   "CrossAttn3", results_m3,   "m3_ca3"),
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


def _save_comparison_md(results_m1, results_m2, results_m2_2, results_m3,
                        aec_size: int = 128):
    """모든 모델·케이스의 비교 테이블과 통계 검정 결과를 scaling_comparison_aec{N}.md로 저장."""
    lines = [
        f"# Scaling Comparison — Test Set Performance (AEC {aec_size}pt)",
        "",
        "## Best Cases Summary  (by Test overall AUC)",
        "",
        "> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.",
        "",
        _best_cases_summary_md(results_m1, results_m2, results_m2_2, results_m3),
        "",
        "---",
        "",
        "## Model 1 — Clinic Only  (1 scaling case)",
        "",
        "### Logistic Regression",
        "",
        _md_table(results_m1, "m1_lr"),
        "",
        "---",
        "",
        f"## Model 2 — Clinic + AEC (Matched)  ({len(AEC_VARIANTS)} AEC variants)",
        "",
        "### CrossAttn",
        "",
        _md_table(results_m2, "m2_ca"),
        "",
        "---",
        "",
        f"## Model 2_2 — Clinic + AEC (Unmatched, Negative Control)  ({len(AEC_VARIANTS)} AEC variants)",
        "",
        "> Clinic과 AEC가 서로 다른 환자 데이터로 섞인 상태.",
        "> Model 2 > Model 2_2 이면 Clinic-AEC 대응이 실질적인 예측력을 제공함을 의미.",
        "",
        "### CrossAttn",
        "",
        _md_table(results_m2_2, "m2_2_ca"),
        "",
        "---",
        "",
        f"## Model 3 — Clinic + Scanner + AEC  ({len(AEC_VARIANTS)} AEC variants)",
        "",
        "### CrossAttn3",
        "",
        _md_table(results_m3, "m3_ca3"),
        "",
        "---",
        "",
        "# Cross-Model Comparison — Fold-level Statistical Tests",
        "",
        "> Paired t-test + Wilcoxon signed-rank (n=5 folds).",
        "> p-value는 지수표현. Δ Mean = B − A (양수 → B 우세).",
        "> M1·M2·M3 간 pairwise 비교 (M2_2 음성 대조군 제외).",
        "> M1은 단일 case로 M2/M3 각 AEC variant와 개별 비교.",
        "> M1↔M2/M3는 데이터셋이 다를 수 있으므로 해석 시 주의.",
        "> \\*\\*\\* p<0.001 · \\*\\* p<0.01 · \\* p<0.05 · † p<0.10",
        "",
    ]

    _STAT_METRICS = [("auc", "AUC-ROC"), ("auprc", "AUPRC"),
                     ("brier", "Brier"), ("acc", "Accuracy"), ("f1", "F1")]
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

    _duo_key = lambda r: r["aec_var"]  # noqa: E731
    m1_r      = results_m1[0]  # M1은 단일 case
    m1_y_te   = m1_r["y_te"]
    m1_lr_prob = m1_r["lr_prob"]

    # ── M1 LR vs M2 CrossAttn ──────────────────────────────────
    lines += [
        "## M1 (LR) vs M2 (CrossAttn)",
        "",
        "> A = M1 LR, B = M2 CrossAttn.",
        "",
    ]
    for r2 in results_m2:
        stat = _fold_stats(m1_r["lr_cv_folds"], r2["ca_cv_folds"])
        lines.append(f"### {_case_label(r2)}  (M1-LR vs M2-CrossAttn)")
        lines.append("")
        lines += [_CSTAT_HDR, _CSTAT_SEP]
        lines += _cstat_rows(stat)
        lines.append("")

    # ── M2 CrossAttn vs M3 CrossAttn3 ─────────────────────────
    lines += [
        "## M2 (CrossAttn) vs M3 (CrossAttn3)",
        "",
        "> A = M2 CrossAttn, B = M3 CrossAttn3. aec_var 키로 매칭.",
        "",
    ]
    lines += _cross_model_md_block(
        results_m2, "ca_cv_folds",  "M2-CrossAttn",
        results_m3, "ca3_cv_folds", "M3-CrossAttn3",
        _cstat_rows, _CSTAT_HDR, _CSTAT_SEP,
        key_fn=_duo_key,
    )

    # ── M1 LR vs M3 CrossAttn3 ────────────────────────────────
    lines += [
        "## M1 (LR) vs M3 (CrossAttn3)",
        "",
        "> A = M1 LR, B = M3 CrossAttn3.",
        "",
    ]
    for r3 in results_m3:
        stat = _fold_stats(m1_r["lr_cv_folds"], r3["ca3_cv_folds"])
        lines.append(f"### {_case_label(r3)}  (M1-LR vs M3-CrossAttn3)")
        lines.append("")
        lines += [_CSTAT_HDR, _CSTAT_SEP]
        lines += _cstat_rows(stat)
        lines.append("")

    # ── Test Set: Bootstrap 95% CI ──────────────────────────────────────────
    lines += [
        "---",
        "",
        "# Test Set — Bootstrap 95% CI  (n_boot=2000)",
        "",
        "> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.",
        "",
    ]
    _CI_HDR = "| Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |"
    _CI_SEP = "|-------|-----|------|--------|-------:|---------:|---------:|"
    _CI_METS = [("AUC-ROC", "auc"), ("AUPRC", "auprc"),
                ("Brier",   "brier"), ("Accuracy", "acc"), ("F1", "f1")]

    def _ci_rows(model_lbl: str, sub_lbl: str, case_lbl: str,
                 ci_dict: dict[str, tuple[float, float, float]]) -> list[str]:
        rows: list[str] = []
        for mname, mkey in _CI_METS:
            if mkey in ci_dict:
                est, lo, hi = ci_dict[mkey]
                rows.append(f"| {model_lbl} | {sub_lbl} | {case_lbl} "
                             f"| {mname} | {est:.4f} | {lo:.4f} | {hi:.4f} |")
        return rows

    lines += [_CI_HDR, _CI_SEP]
    for r in results_m1:
        ts = r.get("test_stats", {})
        lbl = r["case"]
        lines += _ci_rows("M1", "LR", lbl, ts.get("bootstrap_lr", {}))
    for r in results_m2:
        ts = r.get("test_stats", {})
        lbl = _case_label(r)
        lines += _ci_rows("M2", "CrossAttn", lbl, ts.get("bootstrap_ca", {}))
    for r in results_m2_2:
        ts = r.get("test_stats", {})
        lbl = _case_label(r)
        lines += _ci_rows("M2_2", "CrossAttn", lbl, ts.get("bootstrap_ca", {}))
    for r in results_m3:
        ts = r.get("test_stats", {})
        lbl = _case_label(r)
        lines += _ci_rows("M3", "CrossAttn3", lbl, ts.get("bootstrap_ca3", {}))
    lines.append("")

    # ── Test Set: DeLong AUC 비교 ──────────────────────────────────────────────
    lines += [
        "---",
        "",
        "# Test Set — DeLong AUC Comparison",
        "",
        "> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.",
        "> excl_extreme 변형은 M1 test set과 샘플 크기가 달라 M1 vs M2/M3 비교에서 제외.",
        "> \\*\\*\\* p<0.001 · \\*\\* p<0.01 · \\* p<0.05 · † p<0.10 · ns p≥0.10",
        "",
    ]
    _DL_HDR = "| Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |"
    _DL_SEP = "|-----------|------:|------:|------:|-------:|------:|-----|"

    def _dl_row(name_a, name_b, y_true, prob_a, prob_b):
        if len(y_true) == 0:
            return None
        auc_a, auc_b, z, p = delong_test(y_true, prob_a, prob_b)
        delta = auc_b - auc_a
        sig = ("***" if p < 0.001 else "**" if p < 0.01 else
               "*"   if p < 0.05  else "†"  if p < 0.10 else "ns")
        return (f"| {name_a} vs {name_b} "
                f"| {auc_a:.4f} | {auc_b:.4f} | {delta:+.4f} "
                f"| {z:.3f} | {p:.3e} | {sig} |")

    r2_2_dict_dl = {r["aec_var"]: r for r in results_m2_2}
    r3_dict_dl   = {r["aec_var"]: r for r in results_m3}

    for section_title, pairs in [
        ("## M1 LR vs M2 CrossAttn", [
            (f"M1-LR", f"M2-{r['aec_var']}", m1_y_te, m1_lr_prob, r["ca_prob_te"], r["y_true_te"])
            for r in results_m2
        ]),
        ("## M1 LR vs M3 CrossAttn3", [
            (f"M1-LR", f"M3-{r['aec_var']}", m1_y_te, m1_lr_prob, r["ca3_prob_te"], r["y_true_te"])
            for r in results_m3
        ]),
        ("## M2 Matched vs M2_2 Unmatched", [
            (f"M2-{r['aec_var']}", f"M2_2-{r['aec_var']}",
             r["y_true_te"], r["ca_prob_te"],
             r2_2_dict_dl[r["aec_var"]]["ca_prob_te"],
             r["y_true_te"])
            for r in results_m2
            if r["aec_var"] in r2_2_dict_dl
            and len(r["y_true_te"]) == len(r2_2_dict_dl[r["aec_var"]]["y_true_te"])
        ]),
        ("## M2 CrossAttn vs M3 CrossAttn3", [
            (f"M2-{r['aec_var']}", f"M3-{r['aec_var']}",
             r["y_true_te"], r["ca_prob_te"],
             r3_dict_dl[r["aec_var"]]["ca3_prob_te"],
             r["y_true_te"])
            for r in results_m2
            if r["aec_var"] in r3_dict_dl
            and len(r["y_true_te"]) == len(r3_dict_dl[r["aec_var"]]["y_true_te"])
        ]),
    ]:
        lines += [section_title, "", _DL_HDR, _DL_SEP]
        for name_a, name_b, y_cmp, prob_a, prob_b, y_check in pairs:
            if len(y_cmp) != len(y_check):
                continue
            row = _dl_row(name_a, name_b, y_cmp, prob_a, prob_b)
            if row:
                lines.append(row)
        lines.append("")

    md_path = os.path.join(os.path.dirname(RESULTS_DIR), f"scaling_comparison_aec{aec_size}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Comparison saved → {md_path}")


if __name__ == "__main__":
    run_all_cases()
