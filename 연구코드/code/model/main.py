"""
SMI Binary Classification — 모델별 AEC variant 비교 실험 진입점.

현재 실행 대상: Model 1 (Clinic Only) + Model 5 (Clinic + AEC Hand-crafted Features)

모델 구성:
  Model 1    : Clinic Only (Age, Sex, BMI)                        — LR            [RUN]
  Model 2    : Clinic + AEC Matched                               — CrossAttn     [SKIP]
  Model 5    : Clinic + AEC Hand-crafted Features (60개)          — CrossAttn     [RUN]
                 통계(mean·std·max·min·skew·kurt·auc)
                 시점(peak_pos·valley_pos·first_val·last_val)
                 구간평균(early/mid/late, q1~q4), 구간std
                 백분위(p5·p10·p25·p50·p75·p90·p95·iqr)
                 형태(range·cv·rms·energy)
                 기울기(slope_rise·slope_fall·rise_auc·fall_auc)
                 1차차분(abs_mean·std·abs_max), 2차차분(abs_mean·std·abs_max)
                 자기상관(lag1·lag2), FFT(low·mid·high·centroid)
                 비율(auc_ratio·symmetry·mean_to_max·late_to_early·start_to_end)
                 임계(above_mean·above_p75·below_p25·peak_half·valley_depth·tail_mean)
M1은 1회 단독 실행, M5는 ProcessPoolExecutor 로 AEC_VARIANTS 병렬 실행.
결과는 scaling_comparison.md 와 각 모델 디렉토리 run.log 에 저장된다.
실행 여부는 상단 RUN_M* 플래그로 제어한다.

스케일링 원칙:
  - Clinic(Age·BMI): StandardScaler 항상 적용
  - AEC: variant에 따라 scale_mode가 결정됨 (aec_variant() 반환값 참고)
    · raw          → scale_mode="none"   (전처리 없음)
    · std_scaled   → scale_mode="column" (열 방향 StandardScaler)
    · norm         → scale_mode="none"   (행 방향 z-score만, 사전 적용)
    · global_zscore→ scale_mode="global" (Train set 전체 단일 μ/σ)
  - sex_enc(이진값), label(y, 0/1), MFR index에는 StandardScaler를 적용하지 않는다

Threshold:
  각 모델의 이진 분류 기준 확률값. 기본값은 CV fold별 Youden index 최적값의 중앙값.
  config.py의 THRESH_M* 를 float으로 설정하면 해당 값으로 고정된다 (None → 자동).

Attention Map 시각화:
  Model 2/2_2/3/5의 CrossAttn 최종 모델로 test set에서 attention weight를 추출한다.
  Clinical↔AEC 양방향 attention을 클래스 분리 bar chart와 샘플별 heatmap으로
  각 디렉토리에 저장한다.
"""

import os
import sys
import io
from contextlib import contextmanager
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

from config import (DEVICE, RESULTS_DIR, RESULTS_MODEL_1_DIR, RESULTS_MODEL_2_DIR,
                    RESULTS_MODEL_5_DIR,
                    AEC_VARIANTS, AEC_LEN, AEC_SHEET,
                    LR_RATE, HIDDEN, N_HEADS, N_BLOCKS, GRAD_CLIP, N_CA_LAYERS,
                    THRESH_M1, THRESH_M2, THRESH_M5)
from data import (load_data, split_data,
                  load_data_with_aec, split_data_dual,                   
                  extract_aec_features_batch, aec_variant, print_stats)
from train_eval import (run_cross_validation, run_cross_validation_cross,
                        run_cross_validation_cross_feat,
                        evaluate_test, evaluate_test_cross,
                        evaluate_test_cross_feat)
from metrics import print_cv_summary, print_delong_comparison, delong_test
from visualize import (save_all, save_all_cross, plot_test_roc_with_baseline,
                       plot_roc_all_models, plot_attention_maps, plot_cam_aec,
                       plot_individual_aec_normalization)


# ── 모델 실행 toggle ─────────────────────────────────────────
# True: 해당 모델 실행 / False: 건너뜀 (결과 = [])
RUN_M1    = True
RUN_M2    = True
RUN_M5    = True


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


def _apply_mask(arrays, mask):
    """mask가 None이면 원본 반환, 아니면 각 배열에 mask 인덱싱 적용."""
    if mask is None:
        return arrays
    return [a[mask] for a in arrays]


@contextmanager
def _capture_log(log_path: str):
    """stdout을 캡처해 log_path에 저장하는 컨텍스트 매니저."""
    buf = io.StringIO()
    sys.stdout = buf
    try:
        yield
    finally:
        sys.stdout = sys.__stdout__
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(buf.getvalue())


# ── 모델별 워커 함수 (subprocess 에서 실행) ───────────────────
# 각 모델의 모든 케이스를 순차 실행하고 결과 리스트를 반환.
# 상세 출력은 <model_results_dir>/run.log 에 저장.

def _run_model1(X_cv, y_cv, sex_cv, X_te, y_te, sex_te, out_dir: str | None = None):
    """Model 1(Clinic Only)을 실행하고 결과 리스트를 반환. 로그는 run.log에 저장."""
    out1 = out_dir or RESULTS_MODEL_1_DIR
    os.makedirs(out1, exist_ok=True)
    results = []
    with _capture_log(os.path.join(out1, "run.log")):
        print(f"{'='*60}")
        print("  MODEL 1 — Clinic Only")
        print(f"{'='*60}")

        print(f"  [M1] Cross-validating LR ...", flush=True)
        (lr_cv, lr_roc_folds, lr_best_thresholds) = run_cross_validation(X_cv, y_cv)

        print_cv_summary("LR", lr_cv)

        med_thresh_lr = THRESH_M1 if THRESH_M1 is not None else float(np.median(lr_best_thresholds))
        print(f"  [M1] Evaluating on test set (thresh={med_thresh_lr:.3f}{' [override]' if THRESH_M1 is not None else ''}) ...", flush=True)
        (lr_pred, lr_prob, stats_te) = evaluate_test(
            X_cv, y_cv, X_te, y_te, sex_te, threshold=med_thresh_lr
        )

        print(f"  [M1] Saving figures ...", flush=True)
        save_all(
            lr_roc_folds, lr_cv,
            X_cv, y_cv, sex_cv,
            X_te, y_te, lr_pred, lr_prob,
            sex_te, out_dir=out1,
            ci_dict=stats_te.get("bootstrap_lr", {}),
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
    return results

def _run_model2(X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                X_clin_te, X_aec_te, y2_te, sex2_te,
                aec_size: int = 128, aec_variants: list | None = None,
                progress_queue=None, out_base_dir: str | None = None):
    """Model 2(Clinic+AEC Matched): AEC 변형별로 CrossAttn을 실행하고 결과 리스트를 반환.
    attention map 4종 + Grad-CAM 3종을 케이스 디렉토리에 저장. 로그는 run.log에 저장."""
    if aec_variants is None:
        aec_variants = AEC_VARIANTS
    results = []
    _base = out_base_dir or RESULTS_MODEL_2_DIR
    with _capture_log(os.path.join(_base, "run.log")):
        half = X_aec_cv.shape[1] // 2
        X_aec_cv = X_aec_cv[:, half:]
        X_aec_te = X_aec_te[:, half:]
        aec_size  = X_aec_cv.shape[1]
        print(f"{'='*60}")
        print(f"  MODEL 2 — Clinic + AEC (aec{aec_size}, 후반50%)  ({len(aec_variants)} AEC variants)")
        print(f"{'='*60}")

        for aec_var in aec_variants:
            X_aec_cv_v, mask_cv, scale_aec_v = aec_variant(X_aec_cv, aec_var)
            X_aec_te_v, mask_te, _            = aec_variant(X_aec_te, aec_var)
            X_clin_cv_v, y2_cv_v, sex2_cv_v  = _apply_mask([X_clin_cv, y2_cv, sex2_cv], mask_cv)
            X_clin_te_v, y2_te_v, sex2_te_v  = _apply_mask([X_clin_te, y2_te, sex2_te], mask_te)

            print(f"\n{'#'*60}")
            print(f"  [M2 {aec_var}]  (scale_clinic=True, scale_aec={scale_aec_v})")
            print(f"{'#'*60}")

            out2 = os.path.join(_base, aec_var)

            print(f"  [M2/{aec_var}] Cross-validating CrossAttn ...", flush=True)
            (ca_cv, ca_roc_folds,
             ca_histories, ca_best_epochs2,
             ca_best_thresholds2) = run_cross_validation_cross(
                X_clin_cv_v, X_aec_cv_v, y2_cv_v, scale_aec=scale_aec_v,
            )

            print_cv_summary("CrossAttn", ca_cv)

            med_epoch2 = int(np.median(ca_best_epochs2))
            med_thresh2 = THRESH_M2 if THRESH_M2 is not None else float(np.median(ca_best_thresholds2))
            print(f"  [M2/{aec_var}] Evaluating on test set (med_epoch={med_epoch2}, thresh={med_thresh2:.3f}{' [override]' if THRESH_M2 is not None else ''}) ...", flush=True)
            (ca_pred_te, ca_prob_te, ca_true_te, stats_te2,
             model_te2, X_clin_te_s2, X_aec_te_s2) = evaluate_test_cross(
                X_clin_cv_v, X_aec_cv_v, y2_cv_v,
                X_clin_te_v, X_aec_te_v, y2_te_v, sex2_te_v,
                med_epoch2, scale_aec=scale_aec_v,
                threshold=med_thresh2,
                weight_path=os.path.join(out2, f"M2_{aec_var}_weights.pt"),
            )

            os.makedirs(out2, exist_ok=True)
            print(f"  [M2/{aec_var}] Saving figures ...", flush=True)
            save_all_cross(
                ca_cv, ca_roc_folds, ca_histories, med_epoch2,
                X_clin_cv_v, y2_cv_v, sex2_cv_v,
                X_clin_te_v, y2_te_v,
                ca_pred_te, ca_true_te, sex2_te_v, ca_prob_te,
                model_label=f"model 2 ({aec_var})", out_dir=out2,
                ci_dict=stats_te2.get("bootstrap_ca", {}),
            )
            print(f"  [M2/{aec_var}] Plotting attention maps ...", flush=True)
            plot_attention_maps(
                model_te2, X_clin_te_s2, X_aec_te_s2, ca_true_te,
                out_dir=out2, aec_var=aec_var,
                model_label=f"Model 2 ({aec_var})",
            )
            print(f"  [M2/{aec_var}] Plotting Grad-CAM ...", flush=True)
            plot_cam_aec(
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
    return results

def _run_model5(X_clin_cv, X_aec_cv, y_cv, sex_cv,
                X_clin_te, X_aec_te, y_te, sex_te,
                aec_size: int = 128, aec_variants: list | None = None,
                progress_queue=None, out_base_dir: str | None = None):
    """Model 5(Clinic + AEC Hand-crafted Features CrossAttn): AEC 변형별로 실행하고 결과 리스트를 반환.
    각 variant의 AEC 신호를 변환한 뒤 hand-crafted 피처(60개)를 추출해 CrossAttn 모델을 학습한다.
    Attention map을 variant 디렉토리에 저장. 로그는 run.log에 저장."""
    if aec_variants is None:
        aec_variants = AEC_VARIANTS
    results = []
    _base = out_base_dir or RESULTS_MODEL_5_DIR
    with _capture_log(os.path.join(_base, "run.log")):
        print(f"{'='*60}")
        print(f"  MODEL 5 — Clinic + AEC Hand-crafted Features (CrossAttn)  ({len(aec_variants)} AEC variants)")
        print(f"{'='*60}")

        for aec_var in aec_variants:
            X_aec_cv_v, _, scale_mode = aec_variant(X_aec_cv, aec_var)
            X_aec_te_v, _, _          = aec_variant(X_aec_te, aec_var)

            # global_zscore: CV set 전체 통계로 AEC 신호를 정규화한 뒤 피처 추출
            if scale_mode == "global":
                g_mean = float(X_aec_cv_v.mean())
                g_std  = max(float(X_aec_cv_v.std()), 1e-8)
                X_aec_cv_v = ((X_aec_cv_v - g_mean) / g_std).astype(np.float32)
                X_aec_te_v = ((X_aec_te_v - g_mean) / g_std).astype(np.float32)

            # variant-transformed AEC → hand-crafted 피처 (60개)
            X_feat_cv_v = extract_aec_features_batch(X_aec_cv_v)
            X_feat_te_v = extract_aec_features_batch(X_aec_te_v)

            print(f"\n{'#'*60}")
            print(f"  [M5 {aec_var}]  (scale_mode={scale_mode})")
            print(f"{'#'*60}")

            out5 = os.path.join(_base, aec_var)
            os.makedirs(out5, exist_ok=True)

            print(f"  [M5/{aec_var}] Cross-validating CrossAttn-Feat ...", flush=True)
            (ca_cv, ca_roc_folds,
             ca_histories, ca_best_epochs,
             ca_best_thresholds) = run_cross_validation_cross_feat(
                X_clin_cv, X_feat_cv_v, y_cv,
            )

            print_cv_summary("CrossAttn-Feat", ca_cv)

            med_epoch  = int(np.median(ca_best_epochs))
            med_thresh = THRESH_M5 if THRESH_M5 is not None else float(np.median(ca_best_thresholds))
            print(f"  [M5/{aec_var}] Evaluating on test set (med_epoch={med_epoch}, thresh={med_thresh:.3f}{' [override]' if THRESH_M5 is not None else ''}) ...", flush=True)
            (ca_pred_te, ca_prob_te, ca_true_te, stats_te,
             model_te5, X_clin_te_s, X_feat_te_s) = evaluate_test_cross_feat(
                X_clin_cv, X_feat_cv_v, y_cv,
                X_clin_te, X_feat_te_v, y_te, sex_te,
                med_epoch, threshold=med_thresh,
                weight_path=os.path.join(out5, f"M5_{aec_var}_weights.pt"),
            )

            print(f"  [M5/{aec_var}] Saving figures ...", flush=True)
            save_all_cross(
                ca_cv, ca_roc_folds, ca_histories, med_epoch,
                X_clin_cv, y_cv, sex_cv,
                X_clin_te, y_te,
                ca_pred_te, ca_true_te, sex_te, ca_prob_te,
                model_label=f"model 5 ({aec_var})", out_dir=out5,
                ci_dict=stats_te.get("bootstrap_ca_feat", {}),
            )
            print(f"  [M5/{aec_var}] Plotting attention maps ...", flush=True)
            plot_attention_maps(
                model_te5, X_clin_te_s, X_feat_te_s, ca_true_te,
                out_dir=out5, aec_var=aec_var,
                model_label=f"Model 5 ({aec_var})",
            )
            print(f"  [M5/{aec_var}] Done.", flush=True)

            results.append({
                "aec_var":     aec_var,
                "case":        aec_var,
                "out_dir":     out5,
                "m5_ca":       _metrics(ca_true_te, ca_pred_te, ca_prob_te),
                "ca_cv_folds": ca_cv,
                "test_stats":  stats_te,
                "y_true_te":   ca_true_te,
                "ca_prob_te":  ca_prob_te,
            })
            if progress_queue is not None:
                progress_queue.put(("M5", aec_var))
    return results

def _print_delong_comparisons(results_m1, results_m2, results_m5,
                              aec_size: int = 128):
    """모델 간 Test-set AUC를 DeLong (1988) 검정으로 쌍별 비교해 콘솔 출력."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  AEC {aec_size}pt — DeLong Test  (Test-set ROC AUC 쌍별 비교)")
    print(sep)

    r1      = results_m1[0] if results_m1 else None
    m1_y    = r1["y_te"]    if r1 else None
    m1_prob = r1["lr_prob"] if r1 else None

    r2_dict   = {r["aec_var"]: r for r in results_m2}
    r5_dict   = {r["aec_var"]: r for r in results_m5}

    # ── M1 vs M2 ──────────────────────────────────────────────
    if results_m1 and results_m2 and m1_y is not None:
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

    # ── M1 vs M5 ──────────────────────────────────────────────
    if results_m1 and results_m5 and m1_y is not None:
        print(f"\n  [M1 LR  vs  M5 CrossAttn-Feat]")
        for r5 in results_m5:
            y5 = r5["y_true_te"]
            if len(y5) != len(m1_y):
                print(f"    {_case_label(r5)} — skip (sample size mismatch: M1={len(m1_y)}, M5={len(y5)})")
                continue
            print_delong_comparison(
                "M1-LR", f"M5-{_case_label(r5)}",
                m1_y, m1_prob, r5["ca_prob_te"],
            )

def _plot_comparison_roc_curves(results_m1, results_m2, results_m5,
                                aec_size: int = 128, results_dir: str | None = None):
    """병렬 실행 완료 후, baseline을 포함한 test_roc_curves.png를 각 디렉토리에 덮어씀.
    M2/M5: Model 1 LR을 baseline으로 비교."""
    r1 = results_m1[0] if results_m1 else None
    m1_y_te    = r1["y_te"]    if r1 else None
    m1_lr_prob = r1["lr_prob"] if r1 else None

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


    m2_dict = {r["aec_var"]: r for r in results_m2}


    # ── aec_var별 전체 모델 비교 (하나의 이미지) ─────────────────
    comparison_dir = results_dir if results_dir is not None else os.path.dirname(RESULTS_MODEL_2_DIR)

    aec_variants_used = list({r["aec_var"] for r in results_m2})
    r2_dict   = {r["aec_var"]: r for r in results_m2}


    for aec_var in aec_variants_used:
        if aec_var not in r2_dict:
            continue
        r2  = r2_dict[aec_var]
        if r1:
            plot_roc_all_models(
                aec_var=aec_var,
                r1_y=r1["y_te"],         r1_prob=r1["lr_prob"],
                r2_y=r2["y_true_te"],    r2_prob=r2["ca_prob_te"],
                out_path=os.path.join(comparison_dir, f"roc_all_models_{aec_var}.png"),
            )

    if results_m5 and r1:
        for r5 in results_m5:
            plot_test_roc_with_baseline(
                primary_true=r5["y_true_te"],
                primary_prob=r5["ca_prob_te"],
                primary_label=f"Model 5 CrossAttn-Feat ({_case_label(r5)})",
                baseline_true=m1_y_te,
                baseline_prob=m1_lr_prob,
                baseline_label="Model 1 LR (baseline)",
                out_path=os.path.join(r5["out_dir"], "test_roc_curves.png"),
            )

    print(f"  Comparison ROC curves saved.")

def run_all_cases():
    """Model 1은 1회, Model 2/5는 AEC_VARIANTS 병렬 실행."""
    print(f"{'='*60}")
    print(f"  LR={LR_RATE}  HIDDEN={HIDDEN}  N_HEADS={N_HEADS}  "
          f"N_BLOCKS={N_BLOCKS}  GRAD_CLIP={GRAD_CLIP}  N_CA_LAYERS={N_CA_LAYERS}")
    print(f"{'='*60}")
    print(f"Device  : {DEVICE}\n")

    # describe_dataset()

    # ── Model 1: AEC 미사용, 1회만 실행 ──────────────────────
    results_m1 = []
    if RUN_M1:
        print("\n[Data] Loading Model 1 data ...")
        X, y, sex = load_data()
        X_cv, y_cv, sex_cv, X_te, y_te, sex_te = split_data(X, y, sex)
        print("[Data] Model 1 data ready.")
        print("=== Model 1 dataset ==="); print_stats(y, sex)

        print("\n[Model 1] Starting ...")
        with ProcessPoolExecutor(max_workers=1) as executor:
            results_m1 = executor.submit(
                _run_model1, X_cv, y_cv, sex_cv, X_te, y_te, sex_te,
                RESULTS_MODEL_1_DIR,
            ).result()
        print("[Model 1] Finished.\n")

    aec_variants = AEC_VARIANTS
    actual_size  = AEC_LEN

    results_dir = os.path.join(RESULTS_DIR, "comparison")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  AEC: raw{actual_size}  sheet={AEC_SHEET}")
    print(f"{'='*60}\n")

    print("[Data] Loading AEC datasets ...")
    X_clin,  X_aec,  y2,  sex2  = load_data_with_aec(
        aec_len=AEC_LEN, aec_sheet=AEC_SHEET, crop_points=None)
    print("[Data] AEC datasets loaded.")

    print("=== Model 2   dataset ===")
    print_stats(y2,  sex2)

    print("[Data] Splitting datasets ...")
    X_clin_cv,  X_aec_cv,  y2_cv,  sex2_cv, \
    X_clin_te,  X_aec_te,  y2_te,  sex2_te  = split_data_dual(X_clin,   X_aec,   y2,  sex2)
    print("[Data] All splits ready.")

    print("[AEC] Saving individual AEC normalization comparison ...")
    plot_individual_aec_normalization(
        X_aec_cv, X_aec_te, y2_te, sex2_te, out_dir=results_dir,
    )

    results_m2 = results_m5 = []

    _active_cfg = [
        ("M2",   RUN_M2,   "M2    ", 0),
        ("M5",   RUN_M5,   "M5    ", 1),
    ]
    _active_keys = [k for k, flag, _, _ in _active_cfg if flag]
    active_label = "/".join(_active_keys) or "(none)"
    print(f"\n  Launching {active_label} in parallel ...\n")

    n_var = len(aec_variants)
    if _active_keys:
        with Manager() as mp_manager:
            q = mp_manager.Queue()
            bars = {k: tqdm(total=n_var, desc=desc, position=pos, leave=True)
                    for k, flag, desc, pos in _active_cfg if flag}
            with ProcessPoolExecutor(max_workers=len(_active_keys)) as executor:
                futures = {}
                if RUN_M2:
                    futures["M2"] = executor.submit(
                        _run_model2,
                        X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                        X_clin_te, X_aec_te, y2_te, sex2_te,
                        actual_size, aec_variants, q, RESULTS_MODEL_2_DIR,
                    )
                if RUN_M5:
                    futures["M5"] = executor.submit(
                        _run_model5,
                        X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                        X_clin_te, X_aec_te, y2_te, sex2_te,
                        actual_size, aec_variants, q, RESULTS_MODEL_5_DIR,
                    )
                total_updates = n_var * len(futures)
                received = 0
                while received < total_updates:
                    try:
                        model_tag, done_var = q.get(timeout=0.5)
                        bars[model_tag].update(1)
                        bars[model_tag].set_postfix(variant=done_var)
                        received += 1
                    except Exception:
                        if all(fut.done() for fut in futures.values()):
                            break
                if RUN_M2:    results_m2    = futures["M2"].result()
                if RUN_M5:    results_m5    = futures["M5"].result()
            for bar in bars.values():
                bar.close()
    print("  All models done.\n")

    print("[Results] Plotting comparison ROC curves ...")
    _plot_comparison_roc_curves(results_m1, results_m2, results_m5,
                                actual_size, results_dir=results_dir)
    print("[Results] Printing comparison table ...")
    _print_comparison(results_m1, results_m2, actual_size,
                      results_m5=results_m5)
    _print_delong_comparisons(results_m1, results_m2, results_m5,
                              actual_size, )
    print("[Results] Saving comparison markdown ...")
    _save_comparison_md(results_m1, results_m2, actual_size, results_dir=results_dir, results_m5=results_m5)
    print("[Results] All done.\n")

# ── 출력 헬퍼 ────────────────────────────────────────────────
_METRICS_DEF = [("AUC", "auc"), ("AUPRC", "auprc"), ("Brier", "brier"),
                ("Acc", "acc"), ("F1", "f1")]

def _best_case(results, metric_key):
    """Test 전체 AUC 기준으로 best case 반환. 결과가 없으면 None."""
    if not results:
        return None
    return max(results, key=lambda r: r[metric_key]["auc"])

def _model_table_str(results, model_key, col=8):
    """단일 모델 결과를 콘솔 테이블 문자열로 반환. best case에 <- BEST 표시."""
    if not results:
        return "(skip — 모델 미실행)"
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

def _print_best_summary(results_m1, results_m2, results_m5):
    """각 모델의 best case(Test AUC 기준)를 요약 출력."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  BEST CASES SUMMARY  (by Test overall AUC)")
    print(sep)
    entries = [
        ("M1",    "LR",             results_m1,              "m1_lr"),
        ("M2",    "CrossAttn",      results_m2,              "m2_ca"),
        ("M5",    "CrossAttn-Feat", results_m5 or [],        "m5_ca"),
    ]
    col = 8
    hdr = f"  {'Model':<6} {'Sub-model':<12} {'Best Case':<32}"
    for mname, _ in _METRICS_DEF:
        hdr += f" {mname:>{col}}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for model_lbl, sub_lbl, results, key in entries:
        best = _best_case(results, key)
        if best is None:
            continue
        m = best[key]
        row = f"  {model_lbl:<6} {sub_lbl:<12} {_case_label(best):<32}"
        for _, mk in _METRICS_DEF:
            row += f" {m[mk]:>{col}.4f}"
        print(row)

def _print_comparison(results_m1, results_m2,
                      aec_size: int = 128, results_m5=None):
    """Model 1~5/LF의 모든 case 결과를 콘솔 테이블로 출력하고, 마지막에 best case 요약을 출력."""
    n_var = len({r["aec_var"] for r in results_m2}) if results_m2 else 0
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

    n_var5 = len({r["aec_var"] for r in results_m5}) if results_m5 else 0
    print(f"\n{sep}")
    print(f"  AEC {aec_size}pt — MODEL 5 — Clinic + AEC Hand-crafted Features CrossAttn  ({n_var5} AEC variants)")
    print(sep)
    print(f"\n  [CrossAttn-Feat]")
    print(_model_table_str(results_m5, "m5_ca"))

    _print_best_summary(results_m1, results_m2, results_m5)

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

def _best_cases_summary_md(results_m1, results_m2, results_m5):
    """각 모델별 best case(Test AUC 기준) 요약 markdown 테이블 반환."""
    entries = [
        ("M1",    "LR",             results_m1,              "m1_lr"),
        ("M2",    "CrossAttn",      results_m2,              "m2_ca"),
        ("M5",    "CrossAttn-Feat", results_m5 or [],        "m5_ca"),
    ]
    col_hdr = " | ".join(mn for mn, _ in _METRICS_DEF)
    col_sep = " | ".join("------:" for _ in _METRICS_DEF)
    lines = [
        f"| Model | Sub-model | Best Case | {col_hdr} |",
        f"|-------|-----------|-----------|{col_sep}|",
    ]
    for model_lbl, sub_lbl, results, key in entries:
        best = _best_case(results, key)
        if best is None:
            continue
        m = best[key]
        cells = " | ".join(f"{m[mk]:.4f}" for _, mk in _METRICS_DEF)
        lines.append(f"| {model_lbl} | {sub_lbl} | {_case_label(best)} | {cells} |")
    return "\n".join(lines)

def _save_comparison_md(results_m1, results_m2,
                        aec_size: int = 128, results_dir: str | None = None,
                        results_m5=None):
    """모든 모델·케이스의 비교 테이블과 통계 검정 결과를 scaling_comparison.md로 저장."""
    lines = [
        f"# Scaling Comparison — Test Set Performance (AEC {aec_size}pt)",
        "",
        "## Best Cases Summary  (by Test overall AUC)",
        "",
        "> 각 모델에서 Test 전체 AUC가 가장 높은 case. 세부 테이블에서 **굵게** 표시.",
        "",
        _best_cases_summary_md(results_m1, results_m2, results_m5),
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
        f"## Model 5 — Clinic + AEC Hand-crafted Features CrossAttn  ({len(AEC_VARIANTS)} AEC variants)",
        "",
        "> Age/Sex/BMI + AEC hand-crafted 피처 60개(통계·시점·구간평균·백분위·형태·기울기·차분·자기상관·FFT·비율) → Cross Attention.",
        "",
        "### CrossAttn-Feat",
        "",
        _md_table(results_m5 or [], "m5_ca"),
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
    m1_r       = results_m1[0] if results_m1 else None
    m1_y_te    = m1_r["y_te"]    if m1_r else None
    m1_lr_prob = m1_r["lr_prob"] if m1_r else None

    # ── M1 LR vs M2 CrossAttn ──────────────────────────────────
    lines += [
        "## M1 (LR) vs M2 (CrossAttn)",
        "",
        "> A = M1 LR, B = M2 CrossAttn.",
        "",
    ]
    if m1_r:
        for r2 in results_m2:
            stat = _fold_stats(m1_r["lr_cv_folds"], r2["ca_cv_folds"])
            lines.append(f"### {_case_label(r2)}  (M1-LR vs M2-CrossAttn)")
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
    for r in (results_m5 or []):
        ts = r.get("test_stats", {})
        lbl = _case_label(r)
        lines += _ci_rows("M5", "CrossAttn-Feat", lbl, ts.get("bootstrap_ca_feat", {}))
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


    for section_title, pairs in [
        ("## M1 LR vs M5 CrossAttn-Feat", [
            ("M1-LR", f"M5-{_case_label(r)}", m1_y_te, m1_lr_prob, r["ca_prob_te"], r["y_true_te"])
            for r in (results_m5 or [])
            if m1_y_te is not None and len(r["y_true_te"]) == len(m1_y_te)
        ]),
        ("## M1 LR vs M2 CrossAttn", [
            (f"M1-LR", f"M2-{r['aec_var']}", m1_y_te, m1_lr_prob, r["ca_prob_te"], r["y_true_te"])
            for r in results_m2
            if m1_y_te is not None
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

    if results_dir is not None:
        md_path = os.path.join(results_dir, "scaling_comparison.md")
    else:
        md_path = os.path.join(os.path.dirname(RESULTS_DIR), "scaling_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Comparison saved → {md_path}")

if __name__ == "__main__":
    run_all_cases()