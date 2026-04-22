"""
TAMA Prediction Case Study
===========================
Data   : 강남_patient_info.xlsx
Output : TAMA (continuous, cm²)
Model  : Ridge Regression (SAG solver) + 5-Fold CV

Input Case:
  Case 1 : [PatientSex, PatientAge]
  Case 2 : [PatientSex, PatientAge, aec_feature]
  Case 3 : [PatientSex, PatientAge, aec_feature, Manufacturer]
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

np.random.seed(42)

# ─── 경로 ────────────────────────────────────────────────────────────────────
SITE         = "신촌"
EXCEL_PATH = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\raw\{SITE}_patient_info.xlsx")
OUTPUT_DIR = Path(rf"C:\Users\user\Desktop\Study\result\0410\{SITE}\case_study")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
RIDGE_ALPHA  = 1.0

# ─── 1. 데이터 로드 ───────────────────────────────────────────────────────────
print("=" * 60)
print(f"  TAMA Case Study – {SITE}세브란스병원")
print("=" * 60)

df = pd.read_excel(EXCEL_PATH)
print(f"\n  원본: {df.shape[0]}행 × {df.shape[1]}열")
print(f"  컬럼: {df.columns.tolist()}")

# TAMA 비수치 값 처리
df["TAMA"] = (
    df["TAMA"].astype(str).str.strip()
    .replace({"NO_LINK": np.nan, "N/A": np.nan, "na": np.nan,
              "n/a": np.nan, "": np.nan, "nan": np.nan})
)
df["TAMA"] = pd.to_numeric(df["TAMA"], errors="coerce")

# TAMA 기준 결측 제거
df_clean = df.dropna(subset=["TAMA"]).reset_index(drop=True)
print(f"  TAMA 유효 행: {len(df_clean)}행")
print(f"  TAMA 범위: {df_clean['TAMA'].min():.1f} ~ {df_clean['TAMA'].max():.1f}"
      f"  (mean={df_clean['TAMA'].mean():.1f}, std={df_clean['TAMA'].std():.1f})")

# ─── 2. 인코딩 ────────────────────────────────────────────────────────────────
df_enc = df_clean.copy()

# PatientSex, Manufacturer: 전체 인코딩
for col in ["PatientSex", "Manufacturer"]:
    if col in df_enc.columns and df_enc[col].dtype == object:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        print(f"  [{col}] 인코딩: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# aec_feature: NaN을 유지하면서 non-NaN 값만 인코딩
if "aec_feature" in df_enc.columns:
    mask = df_enc["aec_feature"].notna()
    if mask.any():
        le = LabelEncoder()
        df_enc.loc[mask, "aec_feature"] = le.fit_transform(
            df_enc.loc[mask, "aec_feature"].astype(str)
        )
        df_enc["aec_feature"] = pd.to_numeric(df_enc["aec_feature"], errors="coerce")
        print(f"  [aec_feature] 인코딩: {mask.sum()}개 non-NaN 값")

# ─── 3. Case 정의 ─────────────────────────────────────────────────────────────
CASES = {
    "Case 1: Sex + Age":                      ["PatientSex", "PatientAge"],
    "Case 2: Sex + Age + AEC":                ["PatientSex", "PatientAge", "aec_feature"],
    "Case 3: Sex + Age + AEC + Manufacturer": ["PatientSex", "PatientAge", "aec_feature", "Manufacturer"],
}

# 실제 존재하는 컬럼만 유지
CASES = {
    name: [c for c in cols if c in df_enc.columns]
    for name, cols in CASES.items()
}

print(f"\n  Case 구성:")
for name, cols in CASES.items():
    print(f"    {name}  ({len(cols)}개: {cols})")

# ─── 4. 학습 및 평가 ──────────────────────────────────────────────────────────
KF = KFold(5, shuffle=True, random_state=RANDOM_STATE)
results: list[dict] = []

print()
for case_name, feat_cols in CASES.items():
    print(f"─── {case_name} {'─' * max(0, 52 - len(case_name))}")

    # Case별 필요 컬럼 기준으로 결측 제거
    df_case = df_enc.dropna(subset=feat_cols + ["TAMA"]).reset_index(drop=True)
    print(f"  유효 샘플: {len(df_case)}행")
    if len(df_case) < 10:
        print(f"  [SKIP] 샘플 수 부족\n")
        continue

    y = df_case["TAMA"].to_numpy(dtype=float)
    X = df_case[feat_cols].to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    X_all_s = scaler.transform(X)

    model = Ridge(alpha=RIDGE_ALPHA, solver="sag",
                  max_iter=5000, random_state=RANDOM_STATE)
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)

    cv_r2 = cross_val_score(
        Ridge(alpha=RIDGE_ALPHA, solver="sag", max_iter=5000, random_state=RANDOM_STATE),
        X_all_s, y, cv=KF, scoring="r2", n_jobs=1
    )
    cv_rmse = cross_val_score(
        Ridge(alpha=RIDGE_ALPHA, solver="sag", max_iter=5000, random_state=RANDOM_STATE),
        X_all_s, y, cv=KF, scoring="neg_root_mean_squared_error", n_jobs=1
    )

    residuals     = y_te - y_pred
    sw_stat, sw_p = stats.shapiro(residuals[:min(500, len(residuals))])

    print(f"  Train/Test : {len(y_tr)} / {len(y_te)}")
    print(f"  RMSE       : {rmse:.3f} cm²")
    print(f"  MAE        : {mae:.3f} cm²")
    print(f"  R²         : {r2:.4f}")
    print(f"  5-CV R²    : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  5-CV RMSE  : {-cv_rmse.mean():.3f} ± {cv_rmse.std():.3f} cm²")
    print(f"  Shapiro-Wilk p={sw_p:.4f}  "
          f"{'(정규성 만족)' if sw_p > 0.05 else '(정규성 위반)'}\n")

    results.append({
        "Case":         case_name,
        "N_features":   len(feat_cols),
        "Features":     " | ".join(feat_cols),
        "N_train":      len(y_tr),
        "N_test":       len(y_te),
        "RMSE":         round(rmse, 4),
        "MAE":          round(mae, 4),
        "R2":           round(r2, 4),
        "CV_R2_mean":   round(cv_r2.mean(), 4),
        "CV_R2_std":    round(cv_r2.std(), 4),
        "CV_RMSE_mean": round(-cv_rmse.mean(), 4),
        "CV_RMSE_std":  round(cv_rmse.std(), 4),
        "SW_p":         round(sw_p, 4),
        "_y_te":        y_te,
        "_y_pred":      y_pred,
        "_residuals":   residuals,
    })

# ─── 5. 결과 저장 ─────────────────────────────────────────────────────────────
summary_df = pd.DataFrame([
    {k: v for k, v in r.items() if not k.startswith("_")}
    for r in results
])
summary_df.to_excel(OUTPUT_DIR / "case_study_summary.xlsx", index=False)

print("=" * 60)
print("  Case 비교 요약")
print("=" * 60)
print(summary_df[["Case", "N_features", "RMSE", "MAE",
                   "R2", "CV_R2_mean", "CV_R2_std"]].to_string(index=False))
best_r2   = summary_df.loc[summary_df["R2"].idxmax()]
best_rmse = summary_df.loc[summary_df["RMSE"].idxmin()]
print(f"\n  최고 R²  : {best_r2['Case']}  (R²={best_r2['R2']:.4f})")
print(f"  최저 RMSE: {best_rmse['Case']}  (RMSE={best_rmse['RMSE']:.3f} cm²)")

# ─── 6. 시각화 ────────────────────────────────────────────────────────────────
COLORS = ["#1565C0", "#2E7D32", "#C62828"]
n      = len(results)

# 6-1. Actual vs Predicted + Residual (case별)
fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
fig.suptitle(f"TAMA Prediction Case Comparison", fontsize=13, fontweight="bold")

for i, r in enumerate(results):
    lbl   = r["Case"].split(":")[0]
    col   = COLORS[i]
    y_te  = r["_y_te"]
    y_pr  = r["_y_pred"]
    resid = r["_residuals"]
    lim   = [min(y_te.min(), y_pr.min()) - 5,
             max(y_te.max(), y_pr.max()) + 5]

    axes[0, i].scatter(y_te, y_pr, alpha=0.4, s=12, color=col)
    axes[0, i].plot(lim, lim, "r--", lw=1.2)
    axes[0, i].set_xlabel("Actual TAMA (cm²)")
    axes[0, i].set_ylabel("Predicted TAMA (cm²)")
    axes[0, i].set_title(f"{lbl}\nR²={r['R2']:.3f}  RMSE={r['RMSE']:.2f}")
    axes[0, i].grid(True, alpha=0.3)

    axes[1, i].scatter(y_pr, resid, alpha=0.4, s=12, color=col)
    axes[1, i].axhline(0, color="red", lw=1.2, ls="--")
    axes[1, i].set_xlabel("Predicted TAMA (cm²)")
    axes[1, i].set_ylabel("Residuals")
    axes[1, i].set_title(f"{lbl} – Residuals")
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "case_scatter_residual.png", dpi=150, bbox_inches="tight")
plt.close()

# 6-2. 메트릭 비교 bar chart
short = [r["Case"].split(":")[0] for r in results]
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
fig2.suptitle(f"Case Metric Comparison", fontsize=12, fontweight="bold")

for ax, (metric, vals) in zip(axes2, [
    ("R²",    [r["R2"]         for r in results]),
    ("RMSE",  [r["RMSE"]       for r in results]),
    ("CV R²", [r["CV_R2_mean"] for r in results]),
]):
    bars = ax.bar(short, vals, color=COLORS[:n], alpha=0.85, edgecolor="white")
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{v:.3f}", ha="center", fontsize=9)

plt.tight_layout()
fig2.savefig(OUTPUT_DIR / "case_metric_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  [저장] {OUTPUT_DIR / 'case_study_summary.xlsx'}")
print(f"  [저장] {OUTPUT_DIR / 'case_scatter_residual.png'}")
print(f"  [저장] {OUTPUT_DIR / 'case_metric_comparison.png'}")
print("=" * 60)
