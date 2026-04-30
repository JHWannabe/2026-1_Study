# -*- coding: utf-8 -*-
"""
AEC Feature Candidate Extraction
Target: SMI (Skeletal Muscle Index)

Pipeline (simplified):
  Step 1. Pre-filter  : near-zero variance (VarianceThreshold threshold=0.01)
  Step 2. Redundancy  : Pearson |r| >= 0.80 high-correlation removal

이후 AEC 최종 선택은 CV fold 내 LassoCV로 수행 (data leakage 방지).
저장: results/feature_selection/feature_selection_summary.xlsx
      sheet "candidate_features"
"""

import warnings
warnings.filterwarnings("ignore")

import os
import subprocess
import tempfile
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# ────────────────────────────────────────────────
# 0. Paths & constants
# ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
RESULT_DIR = SCRIPT_DIR.parent / "results" / "feature_selection"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_FEATURES = ["mean"]
CORR_THRESHOLD = 0.80


# ────────────────────────────────────────────────
# 1. Data loading helpers
# ────────────────────────────────────────────────

def _load_xlsx(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = Path(tempfile.gettempdir()) / f"aec_temp_{xlsx_path.stem}.xlsx"
    subprocess.run(
        ["powershell", "-Command",
         f'Copy-Item -Path "{xlsx_path}" -Destination "{tmp}" -Force'],
        capture_output=True,
    )
    if not tmp.exists():
        raise RuntimeError(f"Failed to copy {xlsx_path} to temp.")
    feat_df = pd.read_excel(str(tmp), sheet_name="features")
    meta_df = pd.read_excel(str(tmp), sheet_name="metadata-bmi_add")
    return feat_df, meta_df


def _prepare_dataset(feat_df: pd.DataFrame,
                     meta_df: pd.DataFrame,
                     label: str) -> tuple[pd.DataFrame, pd.Series]:
    meta_df = meta_df.copy()
    meta_df["SMI"] = pd.to_numeric(meta_df["SMI"], errors="coerce")
    meta_df = meta_df.dropna(subset=["SMI"])

    df = feat_df.merge(meta_df[["PatientID", "SMI"]], on="PatientID", how="inner")
    print(f"  [{label}] {df.shape[0]}명 | SMI: {df['SMI'].min():.2f}~{df['SMI'].max():.2f}"
          f" | mean={df['SMI'].mean():.2f}")

    X_raw = df.drop(columns=["PatientID", "SMI"])
    y_raw = df["SMI"].astype(float)
    df_complete = pd.concat([X_raw, y_raw.rename("SMI")], axis=1).dropna().reset_index(drop=True)
    n_dropped = len(X_raw) - len(df_complete)
    print(f"  [{label}] dropna: {n_dropped}행 제거 → {len(df_complete)}행 사용")

    X = df_complete.drop(columns=["SMI"])
    y = df_complete["SMI"].astype(float)
    return X, y


# ────────────────────────────────────────────────
# 2. Pipeline helper
# ────────────────────────────────────────────────

def _drop_high_corr(corr_mat: pd.DataFrame,
                    threshold: float = CORR_THRESHOLD,
                    protected=None) -> list:
    protected_set = set(protected or [])
    to_drop = set()
    cols    = corr_mat.columns.tolist()
    mat     = corr_mat.values  # numpy array — avoids pandas Scalar type issues
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(mat[i, j]) >= threshold and cols[j] not in to_drop:
                if cols[j] not in protected_set:
                    to_drop.add(cols[j])
                elif cols[i] not in protected_set:
                    to_drop.add(cols[i])
    return list(to_drop)


# ────────────────────────────────────────────────
# 3. Main pipeline
# ────────────────────────────────────────────────

def run_pipeline(X: pd.DataFrame, label: str, out_dir: Path) -> list:
    """Variance + correlation pre-filtering → AEC candidate list."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PIPELINE: {label}  (n_features={X.shape[1]})")
    print(f"{'='*60}")

    # Step 1. Near-zero variance
    X_scaled = StandardScaler().fit_transform(X)
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X_scaled)
    low_var  = [c for c, keep in zip(X.columns, vt.get_support()) if not keep]
    X_step1  = X.drop(columns=low_var)
    print(f"  Step 1 (variance): removed {len(low_var)} → remaining {X_step1.shape[1]}")
    if low_var:
        print(f"    removed: {low_var}")

    # Step 2. High Pearson correlation (|r| >= CORR_THRESHOLD)
    corr_mat  = X_step1.corr(method="pearson")
    high_corr = _drop_high_corr(corr_mat, threshold=CORR_THRESHOLD, protected=FIXED_FEATURES)
    X_step2   = X_step1.drop(columns=high_corr)
    candidates = X_step2.columns.tolist()
    print(f"  Step 2 (corr>={CORR_THRESHOLD}): removed {len(high_corr)} → remaining {len(candidates)}")
    if high_corr:
        print(f"    removed: {sorted(high_corr)}")
    print(f"  AEC Candidates ({len(candidates)}): {sorted(candidates)}")

    # Correlation heatmap
    n = len(X_step1.columns)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(6, n * 0.42)))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=False, ax=ax, square=True, linewidths=0.3)
    ax.set_title(f"Pearson Correlation – {label} (before Step 2 removal)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "01_correlation_heatmap.png", dpi=150)
    plt.close()

    # Save Excel summary
    with pd.ExcelWriter(out_dir / "feature_selection_summary.xlsx") as writer:
        pd.DataFrame({"removed_low_variance": low_var}).to_excel(
            writer, sheet_name="removed_low_var", index=False)
        pd.DataFrame({"removed_high_corr": sorted(high_corr)}).to_excel(
            writer, sheet_name="removed_high_corr", index=False)
        pd.DataFrame({"candidate_features": sorted(candidates)}).to_excel(
            writer, sheet_name="candidate_features", index=False)

    print(f"  Saved: {out_dir / 'feature_selection_summary.xlsx'}")
    return candidates


# ────────────────────────────────────────────────
# 4. Load data & run
# ────────────────────────────────────────────────
all_xlsx = {f: DATA_DIR / f for f in os.listdir(DATA_DIR)
            if f.endswith(".xlsx") and "merged_features" in f}
gangnam_path = next((v for k, v in all_xlsx.items() if "강남" in k), None)
sinchon_path = next((v for k, v in all_xlsx.items() if "신촌" in k), None)

if not gangnam_path and not sinchon_path:
    raise FileNotFoundError(f"merged_features xlsx 파일을 {DATA_DIR}에서 찾을 수 없습니다.")

print(f"[Data] 강남: {gangnam_path.name if gangnam_path else '없음'}")
print(f"[Data] 신촌: {sinchon_path.name if sinchon_path else '없음'}")

results = {}
X_gn, X_sc = None, None

if gangnam_path:
    feat_gn, meta_gn = _load_xlsx(gangnam_path)
    X_gn, _ = _prepare_dataset(feat_gn, meta_gn, "강남")
    results["gangnam"] = run_pipeline(X_gn, "강남", RESULT_DIR / "gangnam")

if sinchon_path:
    feat_sc, meta_sc = _load_xlsx(sinchon_path)
    X_sc, _ = _prepare_dataset(feat_sc, meta_sc, "신촌")
    results["sinchon"] = run_pipeline(X_sc, "신촌", RESULT_DIR / "sinchon")

if X_gn is not None and X_sc is not None:
    common_cols = X_gn.columns.intersection(X_sc.columns).tolist()
    X_merged    = pd.concat([X_gn[common_cols], X_sc[common_cols]], ignore_index=True)
    results["merged"] = run_pipeline(X_merged, "병합(강남+신촌)", RESULT_DIR / "merged")

# Copy primary result to top-level for config.py
primary_key = "merged" if "merged" in results else ("gangnam" if "gangnam" in results else "sinchon")
src_xlsx = RESULT_DIR / primary_key / "feature_selection_summary.xlsx"
dst_xlsx = RESULT_DIR / "feature_selection_summary.xlsx"
shutil.copy2(src_xlsx, dst_xlsx)
print(f"\n  AEC_CANDIDATES source: '{primary_key}' → {dst_xlsx}")

print(f"\n{'='*60}")
print("ALL DONE")
for key, cands in results.items():
    print(f"  [{key}] {len(cands)} candidates: {sorted(cands)}")
print(f"{'='*60}")
