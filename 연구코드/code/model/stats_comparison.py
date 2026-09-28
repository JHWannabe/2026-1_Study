"""
baseline vs baseline_aec 통계 비교
  - DeLong test (vs chance, vs 서로)
  - Bootstrap 95% CI
  - McNemar's test
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from metrics import delong_test_pair, bootstrap_ci, mcnemar_test

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
BASELINE_PATH     = os.path.join(_ROOT, "baseline",     "clinical_lr_predictions.xlsx")
BASELINE_AEC_PATH = os.path.join(_ROOT, "baseline_aec", "clinical_lr_predictions.xlsx")
OUTPUT_PATH       = os.path.join(_ROOT, "stats_comparison.xlsx")


def _sig(p):
    if np.isnan(p):   return "n/a"
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    return "ns"


def _cm_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return dict(TP=tp, FN=fn, TN=tn, FP=fp,
                Sensitivity=sens, Specificity=spec, PPV=ppv, NPV=npv)


def run():
    df_base = pd.read_excel(BASELINE_PATH,     sheet_name="Predictions")
    df_aec  = pd.read_excel(BASELINE_AEC_PATH, sheet_name="Predictions")

    summary_rows = []
    delong_rows  = []
    ci_rows      = []

    for split in ["Train", "Test"]:
        b = df_base[df_base["Split"] == split].reset_index(drop=True)
        a = df_aec [df_aec ["Split"] == split].reset_index(drop=True)

        y      = b["Label"].values.astype(int)
        prob1  = b["LR_Probability"].values.astype(float)
        pred1  = b["LR_Prediction"].values.astype(int)

        # baseline_aec: 애매 구간은 Step2_Prob, 나머지는 Step1_Prob
        prob2_raw = a["Step2_Prob"].values
        prob2 = np.where(pd.notna(prob2_raw),
                         prob2_raw.astype(float),
                         a["Step1_Prob"].values.astype(float))
        pred2 = a["Final_Pred"].values.astype(int)

        # ── 포인트 추정 ──────────────────────────────────────────────────
        auc1 = roc_auc_score(y, prob1)
        auc2 = roc_auc_score(y, prob2)
        m1   = _cm_metrics(y, pred1)
        m2   = _cm_metrics(y, pred2)

        for tag, auc, m in [("M1_baseline", auc1, m1),
                             ("M2_baseline_aec", auc2, m2)]:
            summary_rows.append({
                "Split": split, "Model": tag,
                "AUC": round(auc, 4),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in m.items()},
            })

        # ── DeLong test: 두 모델 AUC 쌍 비교 ───────────────────────────
        _, _, z12, p12 = delong_test_pair(y, prob1, prob2)

        delong_rows.append({
            "Split": split,
            "비교": "M1 vs M2 (DeLong paired)",
            "AUC1": round(auc1, 4), "AUC2": round(auc2, 4),
            "z": round(z12, 4), "p-value": f"{p12:.3e}",
            "유의성": _sig(p12),
        })

        # ── Bootstrap 95% CI ─────────────────────────────────────────────
        ci1 = bootstrap_ci(y, prob1, pred1)
        ci2 = bootstrap_ci(y, prob2, pred2)

        for tag, auc, ci in [("M1_baseline", auc1, ci1),
                              ("M2_baseline_aec", auc2, ci2)]:
            for metric, (lo, hi) in ci.items():
                point = (auc if metric == "AUC"
                         else _cm_metrics(y, pred1 if "M1" in tag else pred2).get(metric, np.nan))
                ci_rows.append({
                    "Split": split, "Model": tag, "Metric": metric,
                    "Point": round(float(point), 4) if not np.isnan(float(point)) else np.nan,
                    "CI_Lower": round(lo, 4),
                    "CI_Upper": round(hi, 4),
                    "95% CI": f"[{lo:.4f}, {hi:.4f}]",
                })

        # ── McNemar's test ───────────────────────────────────────────────
        chi2_mc, p_mc = mcnemar_test(y, pred1, pred2)
        b_mc = int(np.sum((pred1 == y) & (pred2 != y)))
        c_mc = int(np.sum((pred1 != y) & (pred2 == y)))
        delong_rows.append({
            "Split": split,
            "비교": "McNemar (M1 vs M2 예측 쌍 비교)",
            "AUC1": f"b={b_mc}", "AUC2": f"c={c_mc}",
            "z": round(chi2_mc, 4) if not np.isnan(chi2_mc) else np.nan,
            "p-value": f"{p_mc:.3e}" if not np.isnan(p_mc) else np.nan,
            "유의성": _sig(p_mc),
        })

        # 콘솔 출력
        print(f"\n{'='*60}")
        print(f"  {split} Set")
        print(f"{'='*60}")
        print(f"  {'Model':<20} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6}")
        for tag, auc, m in [("M1_baseline", auc1, m1), ("M2_baseline_aec", auc2, m2)]:
            print(f"  {tag:<20} {auc:.4f} {m['Sensitivity']:.4f} "
                  f"{m['Specificity']:.4f} {m['PPV']:.4f} {m['NPV']:.4f}")

        print(f"\n  [DeLong]  M1 vs M2:  z={z12:.3f}  p={p12:.3e}  {_sig(p12)}")
        print(f"\n  [McNemar]  b={b_mc}  c={c_mc}  chi2={chi2_mc:.3f}  p={p_mc:.3e}  {_sig(p_mc)}")

        print(f"\n  [Bootstrap 95% CI]")
        for tag, ci in [("M1", ci1), ("M2", ci2)]:
            print(f"  {tag}:  " + "  ".join(
                f"{k}=[{lo:.3f},{hi:.3f}]" for k, (lo, hi) in ci.items()))

    # ── Excel 저장 ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary",    index=False)
        pd.DataFrame(delong_rows ).to_excel(writer, sheet_name="DeLong_McNemar", index=False)
        pd.DataFrame(ci_rows     ).to_excel(writer, sheet_name="Bootstrap_CI",   index=False)

    print(f"\n[stats] 저장 완료 → {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
