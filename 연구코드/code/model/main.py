"""
SMI Binary Classification — 모델별 AEC variant 비교 실험 진입점.

모델 구성:
  Model 1    : Clinic Only (Age, Sex, BMI)                        — LR            [RUN]
  Model 2    : Clinic + AEC 128pt Matched                         — CrossAttn     [RUN]
  Model 3    : Clinic + Global(26) + 4등분 통계(16) + pairwise(12)  — CrossAttn     [RUN]
                 피처 총 54개: 전체 글로벌 통계 + 4구간 mean/std/max/min
                               + C(4,2)=6 pairs × (ratio+diff) 전체 관계
  Model 4    : Clinic + Global(26) + 8등분 통계(16) + pairwise(56)  — CrossAttn     [RUN]
                 피처 총 98개: 전체 글로벌 통계 + 8구간 mean/std
                               + C(8,2)=28 pairs × (ratio+diff) 전체 관계
  Model 5    : Clinic + Global(26) + 16등분 통계(16) + pairwise(240) — CrossAttn    [RUN]
                 피처 총 282개: 전체 글로벌 통계 + 16구간 mean
                                + C(16,2)=120 pairs × (ratio+diff) 전체 관계

M1은 1회 단독 실행, M2~M5는 ProcessPoolExecutor로 AEC_VARIANTS 병렬 실행.
결과는 scaling_comparison.md와 각 모델 디렉토리 run.log에 저장된다.
실행 여부는 상단 RUN_M* 플래그로 제어한다.

스케일링 원칙:
  - Clinic(Age·BMI): StandardScaler 항상 적용
  - AEC: variant에 따라 scale_mode가 결정됨 (aec_variant() 반환값 참고)
    · raw          → scale_mode="none"
    · norm         → scale_mode="none"   (행 방향 z-score 사전 적용)
    · global_zscore→ scale_mode="global" (Train set 전체 단일 μ/σ)
  - sex_enc, label, MFR index에는 StandardScaler 미적용

Threshold:
  각 모델의 이진 분류 기준 확률값. 기본값은 CV fold별 Youden index 최적값의 중앙값.
  config.py의 THRESH_M* 를 float으로 설정하면 해당 값으로 고정된다 (None → 자동).
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

from config import (DEVICE, RESULTS_DIR,
                    RESULTS_MODEL_1_DIR, RESULTS_MODEL_2_DIR,
                    RESULTS_MODEL_3_DIR, RESULTS_MODEL_4_DIR, RESULTS_MODEL_5_DIR,
                    AEC_VARIANTS, AEC_LEN, AEC_SHEET,
                    LR_RATE, HIDDEN, N_HEADS, N_BLOCKS, GRAD_CLIP, N_CA_LAYERS,
                    THRESH_M1, THRESH_M2, THRESH_M3, THRESH_M4, THRESH_M5)
from data import (load_data, split_data,
                  load_data_with_aec, split_data_dual,
                  extract_aec_features_m3, extract_aec_features_m4, extract_aec_features_m5,
                  aec_variant, print_stats)
from train_eval import (run_cross_validation, run_cross_validation_cross,
                        run_cross_validation_cross_feat,
                        evaluate_test, evaluate_test_cross,
                        evaluate_test_cross_feat)
from metrics import print_cv_summary, print_delong_comparison, delong_test
from visualize import (save_all, save_all_cross, plot_test_roc_with_baseline,
                       plot_roc_all_models, plot_attention_maps, plot_cam_aec,
                       plot_individual_aec_normalization)


# ── 모델 실행 toggle ─────────────────────────────────────────
RUN_M1 = True
RUN_M2 = True
RUN_M3 = True
RUN_M4 = True
RUN_M5 = True


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
    return r.get("aec_var", r["case"])


def _apply_mask(arrays, mask):
    if mask is None:
        return arrays
    return [a[mask] for a in arrays]


@contextmanager
def _capture_log(log_path: str):
    buf = io.StringIO()
    sys.stdout = buf
    try:
        yield
    finally:
        sys.stdout = sys.__stdout__
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(buf.getvalue())


# ── 모델별 워커 함수 ──────────────────────────────────────────

def _run_model1(X_cv, y_cv, sex_cv, X_te, y_te, sex_te, out_dir: str | None = None):
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
        print(f"  [M1] Evaluating on test set (thresh={med_thresh_lr:.3f}) ...", flush=True)
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
    """Model 2(Clinic+AEC 128pt Matched): AEC 전체(128pt)를 CrossAttn으로 학습."""
    if aec_variants is None:
        aec_variants = AEC_VARIANTS
    results = []
    _base = out_base_dir or RESULTS_MODEL_2_DIR
    with _capture_log(os.path.join(_base, "run.log")):
        print(f"{'='*60}")
        print(f"  MODEL 2 — Clinic + AEC (aec{aec_size}, 전체 128pt)  ({len(aec_variants)} AEC variants)")
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

            med_epoch2  = int(np.median(ca_best_epochs2))
            med_thresh2 = THRESH_M2 if THRESH_M2 is not None else float(np.median(ca_best_thresholds2))
            print(f"  [M2/{aec_var}] Evaluating on test set (med_epoch={med_epoch2}, thresh={med_thresh2:.3f}) ...", flush=True)
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


def _run_feat_model(model_num: int, metric_key: str, extract_fn,
                    out_base_dir: str, thresh_val,
                    X_clin_cv, X_aec_cv, y_cv, sex_cv,
                    X_clin_te, X_aec_te, y_te, sex_te,
                    n_feats: int,
                    aec_variants: list | None = None,
                    progress_queue=None):
    """M3/M4/M5 공통 워커: AEC를 extract_fn으로 feature로 변환해 CrossAttn(scalar) 학습."""
    if aec_variants is None:
        aec_variants = AEC_VARIANTS
    results = []
    tag = f"M{model_num}"
    with _capture_log(os.path.join(out_base_dir, "run.log")):
        print(f"{'='*60}")
        print(f"  MODEL {model_num} — Clinic + AEC Features ({n_feats}개)  ({len(aec_variants)} AEC variants)")
        print(f"{'='*60}")

        for aec_var in aec_variants:
            X_aec_cv_v, _, scale_mode = aec_variant(X_aec_cv, aec_var)
            X_aec_te_v, _, _          = aec_variant(X_aec_te, aec_var)

            if scale_mode == "global":
                g_mean     = float(X_aec_cv_v.mean())
                g_std      = max(float(X_aec_cv_v.std()), 1e-8)
                X_aec_cv_v = ((X_aec_cv_v - g_mean) / g_std).astype(np.float32)
                X_aec_te_v = ((X_aec_te_v - g_mean) / g_std).astype(np.float32)

            X_feat_cv = extract_fn(X_aec_cv_v)
            X_feat_te = extract_fn(X_aec_te_v)

            print(f"\n{'#'*60}")
            print(f"  [{tag} {aec_var}]  (scale_mode={scale_mode}, n_feat={X_feat_cv.shape[1]})")
            print(f"{'#'*60}")

            out_dir = os.path.join(out_base_dir, aec_var)
            os.makedirs(out_dir, exist_ok=True)

            print(f"  [{tag}/{aec_var}] Cross-validating CrossAttn-Feat ...", flush=True)
            (ca_cv, ca_roc_folds,
             ca_histories, ca_best_epochs,
             ca_best_thresholds) = run_cross_validation_cross_feat(
                X_clin_cv, X_feat_cv, y_cv,
            )

            print_cv_summary(f"CrossAttn-{tag}", ca_cv)

            med_epoch  = int(np.median(ca_best_epochs))
            med_thresh = thresh_val if thresh_val is not None else float(np.median(ca_best_thresholds))
            print(f"  [{tag}/{aec_var}] Evaluating on test set (med_epoch={med_epoch}, thresh={med_thresh:.3f}) ...", flush=True)
            (ca_pred_te, ca_prob_te, ca_true_te, stats_te,
             model_te, X_clin_te_s, X_feat_te_s) = evaluate_test_cross_feat(
                X_clin_cv, X_feat_cv, y_cv,
                X_clin_te, X_feat_te, y_te, sex_te,
                med_epoch, threshold=med_thresh,
                weight_path=os.path.join(out_dir, f"{tag}_{aec_var}_weights.pt"),
            )

            print(f"  [{tag}/{aec_var}] Saving figures ...", flush=True)
            save_all_cross(
                ca_cv, ca_roc_folds, ca_histories, med_epoch,
                X_clin_cv, y_cv, sex_cv,
                X_clin_te, y_te,
                ca_pred_te, ca_true_te, sex_te, ca_prob_te,
                model_label=f"model {model_num} ({aec_var})", out_dir=out_dir,
                ci_dict=stats_te.get("bootstrap_ca_feat", {}),
            )
            print(f"  [{tag}/{aec_var}] Plotting attention maps ...", flush=True)
            plot_attention_maps(
                model_te, X_clin_te_s, X_feat_te_s, ca_true_te,
                out_dir=out_dir, aec_var=aec_var,
                model_label=f"Model {model_num} ({aec_var})",
            )
            print(f"  [{tag}/{aec_var}] Done.", flush=True)

            results.append({
                "aec_var":     aec_var,
                "case":        aec_var,
                "out_dir":     out_dir,
                metric_key:    _metrics(ca_true_te, ca_pred_te, ca_prob_te),
                "ca_cv_folds": ca_cv,
                "test_stats":  stats_te,
                "y_true_te":   ca_true_te,
                "ca_prob_te":  ca_prob_te,
            })
            if progress_queue is not None:
                progress_queue.put((tag, aec_var))
    return results


# ── 비교 출력 헬퍼 ─────────────────────────────────────────────
_METRICS_DEF = [("AUC", "auc"), ("AUPRC", "auprc"), ("Brier", "brier"),
                ("Acc", "acc"), ("F1", "f1")]


def _best_case(results, metric_key):
    if not results:
        return None
    return max(results, key=lambda r: r[metric_key]["auc"])


def _model_table_str(results, model_key, col=8):
    if not results:
        return "(skip — 모델 미실행)"
    best = _best_case(results, model_key)
    hdr = f"{'Case':<32}" + "".join(f" {mn:>{col}}" for mn, _ in _METRICS_DEF)
    rows = [hdr, "-" * len(hdr)]
    for r in results:
        m   = r[model_key]
        row = f"{_case_label(r):<32}" + "".join(f" {m[mk]:>{col}.4f}" for _, mk in _METRICS_DEF)
        if r is best:
            row += "  <- BEST"
        rows.append(row)
    return "\n".join(rows)


def _print_best_summary(results_m1, results_m2, feat_models):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  BEST CASES SUMMARY  (by Test overall AUC)")
    print(sep)
    entries = [
        ("M1", "LR",          results_m1, "m1_lr"),
        ("M2", "CrossAttn",   results_m2, "m2_ca"),
    ] + [(tag, f"CrossAttn-{tag}", res, mkey) for tag, mkey, res, _ in feat_models]
    col = 8
    hdr = f"  {'Model':<6} {'Sub-model':<14} {'Best Case':<32}"
    for mn, _ in _METRICS_DEF:
        hdr += f" {mn:>{col}}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for model_lbl, sub_lbl, results, key in entries:
        best = _best_case(results, key)
        if best is None:
            continue
        m   = best[key]
        row = f"  {model_lbl:<6} {sub_lbl:<14} {_case_label(best):<32}"
        row += "".join(f" {m[mk]:>{col}.4f}" for _, mk in _METRICS_DEF)
        print(row)


def _print_comparison(results_m1, results_m2, aec_size: int, feat_models):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  AEC {aec_size}pt — MODEL 1 — Test Set Performance  (1 scaling case)")
    print(sep)
    print(f"\n  [LR]")
    print(_model_table_str(results_m1, "m1_lr"))

    n_var = len({r["aec_var"] for r in results_m2}) if results_m2 else 0
    print(f"\n{sep}")
    print(f"  AEC {aec_size}pt — MODEL 2 — Clinic + AEC 128pt  ({n_var} AEC variants)")
    print(sep)
    print(f"\n  [CrossAttn]")
    print(_model_table_str(results_m2, "m2_ca"))

    for tag, mkey, res, desc in feat_models:
        n = len({r["aec_var"] for r in res}) if res else 0
        print(f"\n{sep}")
        print(f"  AEC {aec_size}pt — MODEL {tag[-1]} — {desc}  ({n} AEC variants)")
        print(sep)
        print(f"\n  [CrossAttn-{tag}]")
        print(_model_table_str(res, mkey))

    _print_best_summary(results_m1, results_m2, feat_models)


def _print_delong_comparisons(results_m1, results_m2, feat_models, aec_size: int = 128):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  AEC {aec_size}pt — DeLong Test  (Test-set ROC AUC 쌍별 비교)")
    print(sep)

    r1      = results_m1[0] if results_m1 else None
    m1_y    = r1["y_te"]    if r1 else None
    m1_prob = r1["lr_prob"] if r1 else None

    if results_m1 and results_m2 and m1_y is not None:
        print(f"\n  [M1 LR  vs  M2 CrossAttn]")
        for r2 in results_m2:
            y2 = r2["y_true_te"]
            if len(y2) != len(m1_y):
                print(f"    {r2['aec_var']} — skip (sample size mismatch)")
                continue
            print_delong_comparison("M1-LR", f"M2-{r2['aec_var']}", m1_y, m1_prob, r2["ca_prob_te"])

    for tag, mkey, res, _ in feat_models:
        if results_m1 and res and m1_y is not None:
            print(f"\n  [M1 LR  vs  {tag} CrossAttn-Feat]")
            for r in res:
                y = r["y_true_te"]
                if len(y) != len(m1_y):
                    print(f"    {_case_label(r)} — skip (sample size mismatch)")
                    continue
                print_delong_comparison("M1-LR", f"{tag}-{_case_label(r)}", m1_y, m1_prob, r["ca_prob_te"])


def _plot_comparison_roc_curves(results_m1, results_m2, feat_models,
                                aec_size: int = 128, results_dir: str | None = None):
    r1          = results_m1[0] if results_m1 else None
    m1_y_te     = r1["y_te"]    if r1 else None
    m1_lr_prob  = r1["lr_prob"] if r1 else None

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

    comparison_dir = results_dir or os.path.dirname(RESULTS_MODEL_2_DIR)
    aec_variants_used = list({r["aec_var"] for r in results_m2})
    r2_dict = {r["aec_var"]: r for r in results_m2}
    for aec_var in aec_variants_used:
        if aec_var not in r2_dict or r1 is None:
            continue
        r2 = r2_dict[aec_var]
        plot_roc_all_models(
            aec_var=aec_var,
            r1_y=r1["y_te"],      r1_prob=r1["lr_prob"],
            r2_y=r2["y_true_te"], r2_prob=r2["ca_prob_te"],
            out_path=os.path.join(comparison_dir, f"roc_all_models_{aec_var}.png"),
        )

    for tag, mkey, res, _ in feat_models:
        if res and r1:
            for r in res:
                plot_test_roc_with_baseline(
                    primary_true=r["y_true_te"],
                    primary_prob=r["ca_prob_te"],
                    primary_label=f"Model {tag[-1]} CrossAttn-Feat ({_case_label(r)})",
                    baseline_true=m1_y_te,
                    baseline_prob=m1_lr_prob,
                    baseline_label="Model 1 LR (baseline)",
                    out_path=os.path.join(r["out_dir"], "test_roc_curves.png"),
                )

    print(f"  Comparison ROC curves saved.")


def run_all_cases():
    """Model 1은 1회, Model 2~5는 AEC_VARIANTS 병렬 실행."""
    print(f"{'='*60}")
    print(f"  LR={LR_RATE}  HIDDEN={HIDDEN}  N_HEADS={N_HEADS}  "
          f"N_BLOCKS={N_BLOCKS}  GRAD_CLIP={GRAD_CLIP}  N_CA_LAYERS={N_CA_LAYERS}")
    print(f"{'='*60}")
    print(f"Device  : {DEVICE}\n")

    # ── Model 1 ──────────────────────────────────────────────────
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
    X_clin, X_aec, y2, sex2 = load_data_with_aec(
        aec_len=AEC_LEN, aec_sheet=AEC_SHEET, crop_points=None)
    print("[Data] AEC datasets loaded.")
    print("=== AEC dataset ==="); print_stats(y2, sex2)

    print("[Data] Splitting datasets ...")
    X_clin_cv, X_aec_cv, y2_cv, sex2_cv, \
    X_clin_te, X_aec_te, y2_te, sex2_te = split_data_dual(X_clin, X_aec, y2, sex2)
    print("[Data] All splits ready.")

    print("[AEC] Saving individual AEC normalization comparison ...")
    plot_individual_aec_normalization(
        X_aec_cv, X_aec_te, y2_te, sex2_te, out_dir=results_dir,
    )

    results_m2 = results_m3 = results_m4 = results_m5 = []

    _active_cfg = [
        ("M2", RUN_M2, "M2    ", 0),
        ("M3", RUN_M3, "M3    ", 1),
        ("M4", RUN_M4, "M4    ", 2),
        ("M5", RUN_M5, "M5    ", 3),
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
                if RUN_M3:
                    futures["M3"] = executor.submit(
                        _run_feat_model,
                        3, "m3_ca", extract_aec_features_m3,
                        RESULTS_MODEL_3_DIR, THRESH_M3,
                        X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                        X_clin_te, X_aec_te, y2_te, sex2_te,
                        54, aec_variants, q,
                    )
                if RUN_M4:
                    futures["M4"] = executor.submit(
                        _run_feat_model,
                        4, "m4_ca", extract_aec_features_m4,
                        RESULTS_MODEL_4_DIR, THRESH_M4,
                        X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                        X_clin_te, X_aec_te, y2_te, sex2_te,
                        98, aec_variants, q,
                    )
                if RUN_M5:
                    futures["M5"] = executor.submit(
                        _run_feat_model,
                        5, "m5_ca", extract_aec_features_m5,
                        RESULTS_MODEL_5_DIR, THRESH_M5,
                        X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                        X_clin_te, X_aec_te, y2_te, sex2_te,
                        282, aec_variants, q,
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
                if RUN_M2: results_m2 = futures["M2"].result()
                if RUN_M3: results_m3 = futures["M3"].result()
                if RUN_M4: results_m4 = futures["M4"].result()
                if RUN_M5: results_m5 = futures["M5"].result()
            for bar in bars.values():
                bar.close()
    print("  All models done.\n")

    # feat_models: (tag, metric_key, results, description)
    feat_models = []
    if RUN_M3 and results_m3:
        feat_models.append(("M3", "m3_ca", results_m3, "Global(26)+4등분통계(16)+pairwise(12)=54개"))
    if RUN_M4 and results_m4:
        feat_models.append(("M4", "m4_ca", results_m4, "Global(26)+8등분통계(16)+pairwise(56)=98개"))
    if RUN_M5 and results_m5:
        feat_models.append(("M5", "m5_ca", results_m5, "Global(26)+16등분통계(16)+pairwise(240)=282개"))

    print("[Results] Plotting comparison ROC curves ...")
    _plot_comparison_roc_curves(results_m1, results_m2, feat_models,
                                actual_size, results_dir=results_dir)
    print("[Results] Printing comparison table ...")
    _print_comparison(results_m1, results_m2, actual_size, feat_models)
    _print_delong_comparisons(results_m1, results_m2, feat_models, actual_size)
    print("[Results] Saving comparison markdown ...")
    _save_comparison_md(results_m1, results_m2, actual_size, feat_models,
                        results_dir=results_dir)
    print("[Results] All done.\n")


# ── Markdown 저장 ─────────────────────────────────────────────

def _md_table(results, model_key):
    if not results:
        return "_결과 없음_"
    best = _best_case(results, model_key)
    col_hdr = " | ".join(mn for mn, _ in _METRICS_DEF)
    col_sep = " | ".join("------:" for _ in _METRICS_DEF)
    lines = [f"| Case | {col_hdr} |", f"|------|{col_sep}|"]
    for r in results:
        m   = r[model_key]
        lbl = _case_label(r)
        cells = " | ".join(f"{m[mk]:.4f}" for _, mk in _METRICS_DEF)
        if r is best:
            lines.append(f"| **{lbl}** | {cells} |")
        else:
            lines.append(f"| {lbl} | {cells} |")
    return "\n".join(lines)


def _fold_stats(folds1, folds2):
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


def _best_cases_summary_md(results_m1, results_m2, feat_models):
    entries = [
        ("M1", "LR",          results_m1, "m1_lr"),
        ("M2", "CrossAttn",   results_m2, "m2_ca"),
    ] + [(tag, f"CrossAttn-{tag}", res, mkey) for tag, mkey, res, _ in feat_models]
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
        m     = best[key]
        cells = " | ".join(f"{m[mk]:.4f}" for _, mk in _METRICS_DEF)
        lines.append(f"| {model_lbl} | {sub_lbl} | {_case_label(best)} | {cells} |")
    return "\n".join(lines)


def _save_comparison_md(results_m1, results_m2, aec_size: int, feat_models,
                        results_dir: str | None = None):
    lines = [
        f"# Scaling Comparison — Test Set Performance (AEC {aec_size}pt)",
        "",
        "## Best Cases Summary  (by Test overall AUC)",
        "",
        "> 각 모델에서 Test 전체 AUC가 가장 높은 case.",
        "",
        _best_cases_summary_md(results_m1, results_m2, feat_models),
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
        f"## Model 2 — Clinic + AEC 128pt  ({len(AEC_VARIANTS)} AEC variants)",
        "",
        "### CrossAttn",
        "",
        _md_table(results_m2, "m2_ca"),
        "",
        "---",
        "",
    ]

    for tag, mkey, res, desc in feat_models:
        lines += [
            f"## Model {tag[-1]} — {desc}  ({len(AEC_VARIANTS)} AEC variants)",
            "",
            f"### CrossAttn-{tag}",
            "",
            _md_table(res, mkey),
            "",
            "---",
            "",
        ]

    # Cross-model CV 통계 (M1 vs M2)
    _STAT_METRICS = [("auc", "AUC-ROC"), ("auprc", "AUPRC"),
                     ("brier", "Brier"), ("acc", "Accuracy"), ("f1", "F1")]
    _CSTAT_HDR = "| Metric | Mean A | Mean B | Δ Mean | t-stat | t p-val | W p-val |"
    _CSTAT_SEP = "|--------|-------:|-------:|-------:|-------:|--------:|--------:|"

    def _cstat_rows(stat_dict):
        rows = []
        for mk, mlabel in _STAT_METRICS:
            s   = stat_dict[mk]
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

    lines += [
        "# Cross-Model Comparison — Fold-level Statistical Tests",
        "",
        "> Paired t-test + Wilcoxon signed-rank (n=5 folds).",
        "> p-value는 지수표현. Δ Mean = B − A (양수 → B 우세).",
        "> \\*\\*\\* p<0.001 · \\*\\* p<0.01 · \\* p<0.05 · † p<0.10",
        "",
        "## M1 (LR) vs M2 (CrossAttn)",
        "",
        "> A = M1 LR, B = M2 CrossAttn.",
        "",
    ]
    m1_r = results_m1[0] if results_m1 else None
    if m1_r:
        for r2 in results_m2:
            stat = _fold_stats(m1_r["lr_cv_folds"], r2["ca_cv_folds"])
            lines.append(f"### {_case_label(r2)}  (M1-LR vs M2-CrossAttn)")
            lines.append("")
            lines += [_CSTAT_HDR, _CSTAT_SEP]
            lines += _cstat_rows(stat)
            lines.append("")

    # Bootstrap CI
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

    def _ci_rows(model_lbl, sub_lbl, case_lbl, ci_dict):
        rows = []
        for mname, mkey in _CI_METS:
            if mkey in ci_dict:
                est, lo, hi = ci_dict[mkey]
                rows.append(f"| {model_lbl} | {sub_lbl} | {case_lbl} "
                             f"| {mname} | {est:.4f} | {lo:.4f} | {hi:.4f} |")
        return rows

    lines += [_CI_HDR, _CI_SEP]
    for r in results_m1:
        ts = r.get("test_stats", {})
        lines += _ci_rows("M1", "LR", r["case"], ts.get("bootstrap_lr", {}))
    for r in results_m2:
        ts = r.get("test_stats", {})
        lines += _ci_rows("M2", "CrossAttn", _case_label(r), ts.get("bootstrap_ca", {}))
    for tag, mkey, res, _ in feat_models:
        for r in res:
            ts = r.get("test_stats", {})
            lines += _ci_rows(tag, f"CrossAttn-{tag}", _case_label(r), ts.get("bootstrap_ca_feat", {}))
    lines.append("")

    # DeLong 비교
    m1_r       = results_m1[0] if results_m1 else None
    m1_y_te    = m1_r["y_te"]    if m1_r else None
    m1_lr_prob = m1_r["lr_prob"] if m1_r else None

    lines += [
        "---",
        "",
        "# Test Set — DeLong AUC Comparison",
        "",
        "> DeLong (1988) 검정 — 동일 test set에서 두 모델의 ROC AUC를 쌍별 비교.",
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

    lines += ["## M1 LR vs M2 CrossAttn", "", _DL_HDR, _DL_SEP]
    for r2 in results_m2:
        if m1_y_te is not None:
            row = _dl_row("M1-LR", f"M2-{r2['aec_var']}", m1_y_te, m1_lr_prob, r2["ca_prob_te"])
            if row:
                lines.append(row)
    lines.append("")

    for tag, mkey, res, _ in feat_models:
        lines += [f"## M1 LR vs {tag} CrossAttn-Feat", "", _DL_HDR, _DL_SEP]
        for r in res:
            if m1_y_te is not None and len(r["y_true_te"]) == len(m1_y_te):
                row = _dl_row("M1-LR", f"{tag}-{_case_label(r)}", m1_y_te, m1_lr_prob, r["ca_prob_te"])
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
