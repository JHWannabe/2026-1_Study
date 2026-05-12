"""
SMI Binary Classification — Per-Model Scaling Case Comparison
  Model 1 : Clinic Only         (LR + ResNet1D)          — 2 cases (scale_clinic)
  Model 2 : Clinic + AEC        (Cross-Attention)         — 4 cases (scale_clinic × scale_aec)
  Model 3 : Clinic + Scanner + AEC (Cross-Attention)      — 8 cases (scale_clinic × scale_aec × scale_scan)

Note: output label(y, 0/1)은 StandardScaler 미적용. scale 대상은 입력 feature만.
병렬 실행: Model 1/2/3 를 동시에 실행. 각 모델 내 케이스는 순차 처리.
각 모델 상세 로그 → <model_results_dir>/run.log
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

from config import DEVICE, RESULTS_DIR, RESULTS_DIR_CROSS, RESULTS_DIR_CROSS_2_2, RESULTS_DIR_CROSS3
from data import (load_data, split_data,
                  load_data_with_aec, load_data_with_aec_unmatched, split_data_dual,
                  load_data_with_aec_meta, split_data_quad,
                  print_stats)
from cross_val import (run_cross_validation, run_cross_validation_cross,
                       run_cross_validation_cross3)
from evaluate import evaluate_test, evaluate_test_cross, evaluate_test_cross3
from metrics import print_cv_summary, compare_fold_metrics
from visualize import save_all, save_all_cross

# ── 모델별 스케일링 케이스 정의 ──────────────────────────────
# Model 1: (case_name, scale_clinic)
CASES_M1 = [
    ("no_scale",     False),
    ("scale_clinic", True),
]

# Model 2: (case_name, scale_clinic, scale_aec)
CASES_M2 = [
    ("no_scale",     False, False),
    ("scale_clinic", True,  False),
    ("scale_aec",    False, True),
    ("scale_both",   True,  True),
]

# Model 2_2: Model 2와 동일한 케이스 — Clinic/AEC Unmatching 실험
CASES_M2_2 = [
    ("no_scale",     False, False),
    ("scale_clinic", True,  False),
    ("scale_aec",    False, True),
    ("scale_both",   True,  True),
]

# Model 3: (case_name, scale_clinic, scale_aec, scale_scan)
CASES_M3 = [
    ("no_scale",          False, False, False),
    ("scale_clinic",      True,  False, False),
    ("scale_aec",         False, True,  False),
    ("scale_scan",        False, False, True),
    ("scale_clinic_aec",  True,  True,  False),
    ("scale_clinic_scan", True,  False, True),
    ("scale_aec_scan",    False, True,  True),
    ("scale_all",         True,  True,  True),
]


def _metrics(y_true, y_pred, y_prob):
    return {
        "auc":   roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "acc":   accuracy_score(y_true, y_pred),
        "f1":    f1_score(y_true, y_pred, zero_division=0),
    }


# ── 모델별 워커 함수 (subprocess 에서 실행) ───────────────────
# 각 모델의 모든 케이스를 순차 실행하고 결과 리스트를 반환.
# 상세 출력은 <model_results_dir>/run.log 에 저장.

def _run_model1(X_cv, y_cv, sex_cv, X_te, y_te, sex_te):
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
            })
    finally:
        sys.stdout = sys.__stdout__

    with open(os.path.join(RESULTS_DIR, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return results


def _run_model2(X_clin_cv, X_aec_cv, y2_cv, sex2_cv,
                X_clin_te, X_aec_te, y2_te, sex2_te):
    buf = io.StringIO()
    sys.stdout = buf
    results = []
    try:
        print(f"{'='*60}")
        print("  MODEL 2 — Clinic + AEC  (4 cases)")
        print(f"{'='*60}")

        for case_name, sc, sa in CASES_M2:
            print(f"\n{'#'*60}")
            print(f"  [M2] CASE : {case_name}  (scale_clinic={sc}, scale_aec={sa})")
            print(f"{'#'*60}")

            out2 = os.path.join(RESULTS_DIR_CROSS, case_name)
            os.makedirs(out2, exist_ok=True)

            lr2_cv, ca_cv, lr2_roc_folds, ca_roc_folds, ca_histories, best_epochs2 = \
                run_cross_validation_cross(
                    X_clin_cv, X_aec_cv, y2_cv, scale_clin=sc, scale_aec=sa,
                )

            print_cv_summary("LR",        lr2_cv)
            print_cv_summary("CrossAttn", ca_cv)

            stat_lr_ca = compare_fold_metrics("LR", lr2_cv, "CrossAttn", ca_cv)

            med_epoch2 = int(np.median(best_epochs2))
            lr2_pred_te, lr2_prob_te, ca_pred_te, ca_prob_te, ca_true_te = evaluate_test_cross(
                X_clin_cv, X_aec_cv, y2_cv,
                X_clin_te, X_aec_te, y2_te, sex2_te, med_epoch2,
                scale_clin=sc, scale_aec=sa,
            )

            save_all_cross(
                lr2_cv, ca_cv, lr2_roc_folds, ca_roc_folds, ca_histories, med_epoch2,
                X_clin_cv, y2_cv, sex2_cv,
                X_clin_te, y2_te,
                lr2_pred_te, lr2_prob_te,
                ca_pred_te, ca_true_te, sex2_te, ca_prob_te,
                model_label="model 2", out_dir=out2,
            )

            results.append({
                "case":         case_name,
                "scale_clinic": sc,
                "scale_aec":    sa,
                "m2_lr":        _metrics(y2_te,      lr2_pred_te, lr2_prob_te),
                "m2_ca":        _metrics(ca_true_te, ca_pred_te,  ca_prob_te),
                "stat_lr_ca":   stat_lr_ca,
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
        print("  MODEL 2_2 — Clinic + AEC (Unmatched)  (4 cases)")
        print(f"{'='*60}")

        for case_name, sc, sa in CASES_M2_2:
            print(f"\n{'#'*60}")
            print(f"  [M2_2] CASE : {case_name}  (scale_clinic={sc}, scale_aec={sa})")
            print(f"{'#'*60}")

            out2_2 = os.path.join(RESULTS_DIR_CROSS_2_2, case_name)
            os.makedirs(out2_2, exist_ok=True)

            lr2_cv, ca_cv, lr2_roc_folds, ca_roc_folds, ca_histories, best_epochs2 = \
                run_cross_validation_cross(
                    X_clin_cv, X_aec_cv, y2_cv, scale_clin=sc, scale_aec=sa,
                )

            print_cv_summary("LR",        lr2_cv)
            print_cv_summary("CrossAttn", ca_cv)

            stat_lr_ca_u = compare_fold_metrics("LR", lr2_cv, "CrossAttn", ca_cv)

            med_epoch2 = int(np.median(best_epochs2))
            lr2_pred_te, lr2_prob_te, ca_pred_te, ca_prob_te, ca_true_te = evaluate_test_cross(
                X_clin_cv, X_aec_cv, y2_cv,
                X_clin_te, X_aec_te, y2_te, sex2_te, med_epoch2,
                scale_clin=sc, scale_aec=sa,
            )

            save_all_cross(
                lr2_cv, ca_cv, lr2_roc_folds, ca_roc_folds, ca_histories, med_epoch2,
                X_clin_cv, y2_cv, sex2_cv,
                X_clin_te, y2_te,
                lr2_pred_te, lr2_prob_te,
                ca_pred_te, ca_true_te, sex2_te, ca_prob_te,
                model_label="model 2_2 (unmatched)", out_dir=out2_2,
            )

            results.append({
                "case":         case_name,
                "scale_clinic": sc,
                "scale_aec":    sa,
                "m2_2_lr":      _metrics(y2_te,      lr2_pred_te, lr2_prob_te),
                "m2_2_ca":      _metrics(ca_true_te, ca_pred_te,  ca_prob_te),
                "stat_lr_ca":   stat_lr_ca_u,
            })
    finally:
        sys.stdout = sys.__stdout__

    with open(os.path.join(RESULTS_DIR_CROSS_2_2, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return results


def _run_model3(X_clin3_cv, X_aec3_cv, X_kvp_cv, X_mfr_cv, y3_cv, sex3_cv,
                X_clin3_te, X_aec3_te, X_kvp_te, X_mfr_te, y3_te, sex3_te, n_mfr):
    buf = io.StringIO()
    sys.stdout = buf
    results = []
    try:
        print(f"{'='*60}")
        print("  MODEL 3 — Clinic + Scanner + AEC  (8 cases)")
        print(f"{'='*60}")

        for case_name, sc, sa, ss in CASES_M3:
            print(f"\n{'#'*60}")
            print(f"  [M3] CASE : {case_name}"
                  f"  (scale_clinic={sc}, scale_aec={sa}, scale_scan={ss})")
            print(f"{'#'*60}")

            out3 = os.path.join(RESULTS_DIR_CROSS3, case_name)
            os.makedirs(out3, exist_ok=True)

            lr3_cv, ca3_cv, lr3_roc_folds, ca3_roc_folds, ca3_histories, best_epochs3 = \
                run_cross_validation_cross3(
                    X_clin3_cv, X_aec3_cv, X_kvp_cv, X_mfr_cv, y3_cv, n_mfr,
                    scale_clin=sc, scale_aec=sa, scale_scan=ss,
                )

            print_cv_summary("LR",         lr3_cv)
            print_cv_summary("CrossAttn3", ca3_cv)

            stat_lr_ca3 = compare_fold_metrics("LR", lr3_cv, "CrossAttn3", ca3_cv)

            med_epoch3 = int(np.median(best_epochs3))
            lr3_pred_te, lr3_prob_te, ca3_pred_te, ca3_prob_te, ca3_true_te = evaluate_test_cross3(
                X_clin3_cv, X_aec3_cv, X_kvp_cv, X_mfr_cv, y3_cv,
                X_clin3_te, X_aec3_te, X_kvp_te, X_mfr_te, y3_te,
                sex3_te, med_epoch3, n_mfr,
                scale_clin=sc, scale_aec=sa, scale_scan=ss,
            )

            save_all_cross(
                lr3_cv, ca3_cv, lr3_roc_folds, ca3_roc_folds, ca3_histories, med_epoch3,
                X_clin3_cv, y3_cv, sex3_cv,
                X_clin3_te, y3_te,
                lr3_pred_te, lr3_prob_te,
                ca3_pred_te, ca3_true_te, sex3_te, ca3_prob_te,
                model_label="model 3", out_dir=out3,
            )

            results.append({
                "case":         case_name,
                "scale_clinic": sc,
                "scale_aec":    sa,
                "scale_scan":   ss,
                "m3_lr":        _metrics(y3_te,       lr3_pred_te, lr3_prob_te),
                "m3_ca3":       _metrics(ca3_true_te, ca3_pred_te, ca3_prob_te),
                "stat_lr_ca3":  stat_lr_ca3,
            })
    finally:
        sys.stdout = sys.__stdout__

    with open(os.path.join(RESULTS_DIR_CROSS3, "run.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return results


def run_all_cases():
    print(f"Device  : {DEVICE}\n")

    # ── 공통 데이터 로드 ──────────────────────────────────────
    X,           y,       sex       = load_data()
    X_clin,      X_aec,   y2, sex2  = load_data_with_aec()
    X_clin_u,    X_aec_u, y2u, sex2u = load_data_with_aec_unmatched()
    X_clin3, X_aec3, X_scan_kvp, X_scan_mfr, y3, sex3, n_mfr = load_data_with_aec_meta()

    print("=== Model 1   dataset ==="); print_stats(y,   sex)
    print("=== Model 2   dataset ==="); print_stats(y2,  sex2)
    print("=== Model 2_2 dataset ==="); print_stats(y2u, sex2u)
    print("=== Model 3   dataset ==="); print_stats(y3,  sex3)

    X_cv,       y_cv,   sex_cv,   X_te,       y_te,   sex_te   = split_data(X, y, sex)
    X_clin_cv,  X_aec_cv,  y2_cv,  sex2_cv, \
    X_clin_te,  X_aec_te,  y2_te,  sex2_te                     = split_data_dual(X_clin,   X_aec,   y2,  sex2)
    X_clin_ucv, X_aec_ucv, y2u_cv, sex2u_cv, \
    X_clin_ute, X_aec_ute, y2u_te, sex2u_te                    = split_data_dual(X_clin_u, X_aec_u, y2u, sex2u)
    (X_clin3_cv, X_aec3_cv, X_kvp_cv, X_mfr_cv, y3_cv, sex3_cv,
     X_clin3_te, X_aec3_te, X_kvp_te, X_mfr_te, y3_te, sex3_te) = split_data_quad(
        X_clin3, X_aec3, X_scan_kvp, X_scan_mfr, y3, sex3,
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
            X_clin3_cv, X_aec3_cv, X_kvp_cv, X_mfr_cv, y3_cv, sex3_cv,
            X_clin3_te, X_aec3_te, X_kvp_te, X_mfr_te, y3_te, sex3_te, n_mfr,
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


def _model_table_str(results, model_key, col=8):
    """단일 모델 결과를 콘솔 테이블 문자열로 반환."""
    hdr_parts = [f"{'Case':<20}"]
    for mname, _ in _METRICS_DEF:
        hdr_parts.append(f"{mname:>{col}}")
    hdr = " ".join(hdr_parts)
    rows = [hdr, "-" * len(hdr)]
    for r in results:
        m = r[model_key]
        row = f"{r['case']:<20}"
        for _, mk in _METRICS_DEF:
            row += f" {m[mk]:>{col}.4f}"
        rows.append(row)
    return "\n".join(rows)


def _print_comparison(results_m1, results_m2, results_m2_2, results_m3):
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
    for label, key in [("LR", "m2_lr"), ("CrossAttn", "m2_ca")]:
        print(f"\n  [{label}]")
        print(_model_table_str(results_m2, key))

    print(f"\n{sep}")
    print("  MODEL 2_2 — Clinic + AEC (Unmatched)  (4 scaling cases)")
    print(sep)
    for label, key in [("LR", "m2_2_lr"), ("CrossAttn", "m2_2_ca")]:
        print(f"\n  [{label}]")
        print(_model_table_str(results_m2_2, key))

    print(f"\n{sep}")
    print("  MODEL 3 — Test Set Performance  (8 scaling cases)")
    print(sep)
    for label, key in [("LR", "m3_lr"), ("CrossAttn3", "m3_ca3")]:
        print(f"\n  [{label}]")
        print(_model_table_str(results_m3, key))


def _md_table(results, model_key):
    col_hdr = " | ".join(mn for mn, _ in _METRICS_DEF)
    col_sep = " | ".join("------:" for _ in _METRICS_DEF)
    lines = [
        f"| Case | {col_hdr} |",
        f"|------|{col_sep}|",
    ]
    for r in results:
        m = r[model_key]
        cells = " | ".join(f"{m[mk]:.4f}" for _, mk in _METRICS_DEF)
        lines.append(f"| {r['case']} | {cells} |")
    return "\n".join(lines)


def _save_comparison_md(results_m1, results_m2, results_m2_2, results_m3):
    lines = [
        "# Scaling Comparison — Test Set Performance",
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
        lines.append(f"### [M2] Case: {r['case']}  (LR vs CrossAttn)")
        lines.append("")
        lines += [_STAT_HDR, _STAT_SEP]
        lines += _stat_rows(r["stat_lr_ca"])
        lines.append("")

    lines += [
        "## Model 2_2 (Unmatched): LR vs CrossAttn",
        "",
    ]
    for r in results_m2_2:
        lines.append(f"### [M2_2] Case: {r['case']}  (LR vs CrossAttn, Unmatched)")
        lines.append("")
        lines += [_STAT_HDR, _STAT_SEP]
        lines += _stat_rows(r["stat_lr_ca"])
        lines.append("")

    lines += [
        "## Model 3: LR vs CrossAttn3",
        "",
    ]
    for r in results_m3:
        lines.append(f"### [M3] Case: {r['case']}  (LR vs CrossAttn3)")
        lines.append("")
        lines += [_STAT_HDR, _STAT_SEP]
        lines += _stat_rows(r["stat_lr_ca3"])
        lines.append("")

    md_path = os.path.join(os.path.dirname(RESULTS_DIR), "scaling_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Comparison saved → {md_path}")


if __name__ == "__main__":
    run_all_cases()
