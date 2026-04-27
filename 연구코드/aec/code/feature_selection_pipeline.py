"""
Feature Selection Pipeline for AEC Signal Features
Target: TAMA (continuous, regression)

Pipeline:
  Step 1. Pre-filter   : missing rate > 10%, near-zero variance
  Step 2. Redundancy   : Pearson |r| >= 0.95 (VIF reported only, not used to remove)
  Step 3. Univariate   : MI > 0 OR Spearman p < 0.05  (no top-N limit)
  Step 4. Model-based  : LASSO / RFECV / RF Permutation  (each selects its own count)
  Final  : 7 candidate sets evaluated by 5-fold CV R²;
           pool <= 20 -> exhaustive 2^N search,
           pool > 20  -> Sequential Feature Selector (forward + backward)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold, mutual_info_regression, RFECV,
    SequentialFeatureSelector,
)
from sklearn.linear_model import LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ────────────────────────────────────────────────
# 0. Paths & data loading
# ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
RESULT_DIR = SCRIPT_DIR.parent / "results" / "feature_selection"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_candidates = [
    DATA_DIR / f for f in os.listdir(DATA_DIR)
    if f.endswith(".xlsx") and "merged_features" in f and "\xc2\xb4\xc2\xb4\xc2\xb4\xc2\xb4" not in f
    and "신촌" not in f
]
# fallback: pick any merged_features that is NOT 신촌
if not _candidates:
    _candidates = [
        DATA_DIR / f for f in os.listdir(DATA_DIR)
        if f.endswith(".xlsx") and "merged_features" in f
    ]
    _candidates = [p for p in _candidates if "신촌" not in str(p)]
if not _candidates:
    raise FileNotFoundError(f"Cannot find merged_features xlsx in {DATA_DIR}")
data_file = _candidates[0]
print(f"[Data] Using: {data_file.name}")

_TEMP_XLSX = Path(os.environ["TEMP"]) / "aec_data_temp.xlsx"

def _ensure_temp_copy(src: Path) -> None:
    import subprocess
    subprocess.run(
        ["powershell", "-Command",
         f'Copy-Item -Path "{src}" -Destination "{_TEMP_XLSX}" -Force'],
        capture_output=True,
    )
    if not _TEMP_XLSX.exists():
        raise RuntimeError("Failed to copy data file to temp path.")

def read_sheet(sheet_name: str) -> pd.DataFrame:
    if not _TEMP_XLSX.exists():
        _ensure_temp_copy(data_file)
    return pd.read_excel(str(_TEMP_XLSX), sheet_name=sheet_name)

feat_df = read_sheet("features")
meta_df = read_sheet("metadata-bmi_add")

meta_df["TAMA"] = pd.to_numeric(meta_df["TAMA"], errors="coerce")
meta_df = meta_df.dropna(subset=["TAMA"])

df = feat_df.merge(meta_df[["PatientID", "TAMA"]], on="PatientID", how="inner")
print(f"[Data] {df.shape[0]} rows | TAMA: {df['TAMA'].min():.0f} - {df['TAMA'].max():.0f}"
      f" | mean={df['TAMA'].mean():.1f}")

X_raw = df.drop(columns=["PatientID", "TAMA"])
y_raw = df["TAMA"].astype(float)

# ────────────────────────────────────────────────
# Dataset preparation: drop rows with any missing value
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("DATASET PREPARATION (dropna)")
print("="*60)

null_ratio = X_raw.isnull().sum() / len(X_raw)
null_feats = null_ratio[null_ratio > 0].sort_values(ascending=False)
print(f"  Features with missing values ({len(null_feats)}):")
for f, r in null_feats.items():
    print(f"    - {f}: {r*100:.1f}%  ({int(r * len(X_raw))} rows)")

df_complete = pd.concat([X_raw, y_raw.rename("TAMA")], axis=1).dropna().reset_index(drop=True)
n_dropped   = len(X_raw) - len(df_complete)
print(f"\n  Original rows : {len(X_raw)}")
print(f"  Dropped rows  : {n_dropped}  ({n_dropped/len(X_raw)*100:.1f}%)")
print(f"  Remaining rows: {len(df_complete)}")

X_filled = df_complete.drop(columns=["TAMA"])   # complete, no NaN
y        = df_complete["TAMA"].astype(float)

# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

# Features that are always included regardless of any filter
FIXED_FEATURES = ["mean"]

def cv_r2(X_sub: pd.DataFrame) -> float:
    """5-fold CV R² with Ridge regression on standardized features."""
    if X_sub.shape[1] == 0:
        return -np.inf
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_sub)
    scores = cross_val_score(Ridge(), Xs, y, cv=cv5, scoring="r2")
    return float(scores.mean())

# ────────────────────────────────────────────────
# Step 1. Pre-filter: near-zero variance only
#         (missing rate filter removed — rows already dropped above)
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1. Pre-filter (near-zero variance)")
print("="*60)

scaler_var = StandardScaler()
X_scaled   = scaler_var.fit_transform(X_filled)
vt         = VarianceThreshold(threshold=0.01)
vt.fit(X_scaled)
low_var_feats = [c for c, keep in zip(X_filled.columns, vt.get_support()) if not keep]
print(f"  Near-zero variance: {len(low_var_feats)} features")
for f in low_var_feats:
    print(f"    - {f}")

high_null_feats = []   # kept for pipeline_log compatibility (no features removed by missing rate)
step1_remove    = low_var_feats
X_step1 = X_filled.drop(columns=step1_remove)
print(f"  [Step 1] Removed {len(step1_remove)} -> Remaining: {X_step1.shape[1]}")

# ────────────────────────────────────────────────
# Step 2. Redundancy: Pearson correlation only
#         (VIF calculated but NOT used to remove features)
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2. Redundancy (correlation >= 0.95 removed; VIF informational)")
print("="*60)

corr_matrix = X_step1.corr(method="pearson")

def drop_high_corr(corr_mat, threshold=0.95, protected=None):
    """Remove the latter of each highly-correlated pair; never remove protected features."""
    protected = set(protected or [])
    to_drop   = set()
    cols      = corr_mat.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_mat.iloc[i, j]) >= threshold and cols[j] not in to_drop:
                if cols[j] not in protected:
                    to_drop.add(cols[j])
                elif cols[i] not in protected:
                    to_drop.add(cols[i])  # protect cols[j], remove cols[i] instead
    return list(to_drop)

high_corr_feats = drop_high_corr(corr_matrix, threshold=0.95, protected=FIXED_FEATURES)
X_step2 = X_step1.drop(columns=high_corr_feats)
print(f"  Correlation removed: {len(high_corr_feats)} -> Remaining: {X_step2.shape[1]}")

# VIF (informational only)
vif_df = pd.DataFrame({
    "feature": X_step2.columns,
    "VIF": [variance_inflation_factor(X_step2.values, i) for i in range(X_step2.shape[1])],
}).sort_values("VIF", ascending=False)
print(f"  VIF > 10 count (informational): {(vif_df['VIF'] > 10).sum()}")
vif_df.to_excel(RESULT_DIR / "step2_vif.xlsx", index=False)

# Correlation heatmap
fig, ax = plt.subplots(figsize=(18, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            annot=False, ax=ax, square=True, linewidths=0.3)
ax.set_title("Pearson Correlation Matrix (Step 1 features)", fontsize=12)
plt.tight_layout()
fig.savefig(RESULT_DIR / "01_correlation_heatmap.png", dpi=150)
plt.close()
print(f"  Saved: 01_correlation_heatmap.png")

# ────────────────────────────────────────────────
# Step 3. Univariate: MI > 0 OR Spearman p < 0.05
#         No top-N limit — all qualifying features pass
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3. Univariate association (MI > 0 OR Spearman p < 0.05, no limit)")
print("="*60)

mi_scores = mutual_info_regression(X_step2, y, random_state=42)
mi_df = pd.DataFrame({"feature": X_step2.columns, "MI": mi_scores}).sort_values("MI", ascending=False)

spearman_rows = []
for col in X_step2.columns:
    rho, pval = stats.spearmanr(X_step2[col], y)
    spearman_rows.append({"feature": col, "spearman_rho": rho, "p_value": pval})
spearman_df = pd.DataFrame(spearman_rows).sort_values("p_value")

univariate_df = mi_df.merge(spearman_df, on="feature").sort_values("MI", ascending=False)
univariate_df.to_excel(RESULT_DIR / "step3_univariate.xlsx", index=False)

mi_pos     = set(mi_df[mi_df["MI"] > 0]["feature"])
sig_spear  = set(spearman_df[spearman_df["p_value"] < 0.05]["feature"])
# FIXED_FEATURES are always included (even if they fail the univariate filter)
fixed_in_step2 = [f for f in FIXED_FEATURES if f in X_step2.columns]
step3_feats = list((mi_pos | sig_spear) | set(fixed_in_step2))

print(f"  MI > 0:              {len(mi_pos)} features")
print(f"  Spearman p < 0.05:   {len(sig_spear)} features")
print(f"  Fixed (always in):   {fixed_in_step2}")
print(f"  Union (Step 3 pass): {len(step3_feats)} features")

# MI bar chart
fig, ax = plt.subplots(figsize=(14, max(6, len(step3_feats) * 0.32 + 1)))
plot_df = mi_df[mi_df["feature"].isin(step3_feats)]
colors  = ["#e74c3c" if f in sig_spear else "#f39c12" for f in plot_df["feature"]]
ax.barh(plot_df["feature"][::-1], plot_df["MI"][::-1], color=colors[::-1])
ax.set_xlabel("Mutual Information Score")
ax.set_title("Step 3 candidate features\n(red=MI>0 AND p<0.05 | orange=MI>0 only / p<0.05 only)")
plt.tight_layout()
fig.savefig(RESULT_DIR / "02_mutual_information.png", dpi=150)
plt.close()
print(f"  Saved: 02_mutual_information.png")

# ────────────────────────────────────────────────
# Step 4. Model-based: LASSO / RFECV / RF Permutation
#         Operate on ALL step3 candidates, no fixed count
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4. Model-based selection")
print("="*60)

X_s4 = X_step2[step3_feats].copy()
scaler4 = StandardScaler()
X_s4_sc = scaler4.fit_transform(X_s4)

# 4-1. LASSO (coef > 0)
print(f"  [4-1] LassoCV on {X_s4.shape[1]} features...")
lasso = LassoCV(cv=cv5, random_state=42, max_iter=10000).fit(X_s4_sc, y)
lasso_coef_df = pd.DataFrame({"feature": X_s4.columns, "coef": np.abs(lasso.coef_)}).sort_values("coef", ascending=False)
lasso_selected = set(lasso_coef_df[lasso_coef_df["coef"] > 0]["feature"])
print(f"    alpha={lasso.alpha_:.4f} | selected: {len(lasso_selected)}")

# 4-2. RFECV (Ridge, optimal n by CV R²)
print(f"  [4-2] RFECV (Ridge) on {X_s4.shape[1]} features...")
rfecv = RFECV(estimator=Ridge(), step=1, cv=cv5, scoring="r2", min_features_to_select=1)
rfecv.fit(X_s4_sc, y)
rfe_selected = set(X_s4.columns[rfecv.support_])
print(f"    optimal n={rfecv.n_features_} | selected: {len(rfe_selected)}")

# 4-3. RF Permutation Importance (perm_mean > 0)
print(f"  [4-3] RF Permutation Importance on {X_s4.shape[1]} features...")
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_s4, y)
perm = permutation_importance(rf, X_s4, y, n_repeats=15, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    "feature":   X_s4.columns,
    "perm_mean": perm.importances_mean,
    "perm_std":  perm.importances_std,
}).sort_values("perm_mean", ascending=False)
perm_selected = set(perm_df[perm_df["perm_mean"] > 0]["feature"])
print(f"    selected: {len(perm_selected)}")

# Permutation importance plot
fig, ax = plt.subplots(figsize=(10, max(5, len(step3_feats) * 0.35 + 1)))
ax.barh(perm_df["feature"][::-1], perm_df["perm_mean"][::-1],
        xerr=perm_df["perm_std"][::-1], capsize=2, color="#3498db")
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Mean decrease in R² (permutation)")
ax.set_title("RF Permutation Importance – Step 3 candidates")
plt.tight_layout()
fig.savefig(RESULT_DIR / "03_permutation_importance.png", dpi=150)
plt.close()
print(f"  Saved: 03_permutation_importance.png")

# ────────────────────────────────────────────────
# Build voting table
# ────────────────────────────────────────────────
all_method_feats = lasso_selected | rfe_selected | perm_selected
votes = Counter()
for s in (lasso_selected, rfe_selected, perm_selected):
    votes.update(s)

mi_lookup  = mi_df.set_index("feature")["MI"]
rho_lookup = spearman_df.set_index("feature")["spearman_rho"]
pv_lookup  = spearman_df.set_index("feature")["p_value"]

vote_df = pd.DataFrame([
    {
        "feature":      f,
        "votes":        votes.get(f, 0),
        "LASSO":        int(f in lasso_selected),
        "RFE":          int(f in rfe_selected),
        "RF_Perm":      int(f in perm_selected),
        "MI":           mi_lookup.get(f, np.nan),
        "spearman_rho": rho_lookup.get(f, np.nan),
        "p_value":      pv_lookup.get(f, np.nan),
    }
    for f in sorted(all_method_feats, key=lambda x: -votes.get(x, 0))
])

# ────────────────────────────────────────────────
# Final. Evaluate ALL candidate sets, then exhaustive / SFS search
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL. Evaluate candidate sets + exhaustive / SFS search")
print("="*60)

# --- 7 pre-defined candidate sets ---
candidate_sets = {
    "LASSO_only":  list(lasso_selected),
    "RFE_only":    list(rfe_selected),
    "RF_Perm_only":list(perm_selected),
    "vote_ge1":    [f for f, v in votes.items() if v >= 1],
    "vote_ge2":    [f for f, v in votes.items() if v >= 2],
    "vote_ge3":    [f for f, v in votes.items() if v >= 3],
    "union_all":   list(all_method_feats),
}

set_results = []
for name, feats in candidate_sets.items():
    if not feats:
        set_results.append({"set_name": name, "n_features": 0, "cv_r2": np.nan,
                             "features": ""})
        continue
    r2 = cv_r2(X_s4[feats])
    set_results.append({"set_name": name, "n_features": len(feats),
                        "cv_r2": r2, "features": ", ".join(sorted(feats))})
    print(f"  {name:16s} | n={len(feats):2d} | CV R²={r2:.4f}")

# --- Exhaustive / Sequential search over the union pool ---
pool = list(all_method_feats)
n_pool = len(pool)
print(f"\n  Search pool: {n_pool} features")

best_feats_search = []
best_r2_search    = -np.inf

if n_pool == 0:
    print("  Pool is empty — skipping search.")
elif n_pool <= 20:
    print(f"  Pool <= 20: exhaustive search (2^{n_pool} = {2**n_pool:,} subsets)...")
    for r in range(1, n_pool + 1):
        for combo in itertools.combinations(pool, r):
            r2 = cv_r2(X_s4[list(combo)])
            if r2 > best_r2_search:
                best_r2_search = r2
                best_feats_search = list(combo)
    print(f"  Best by exhaustive: n={len(best_feats_search)} | CV R²={best_r2_search:.4f}")
    print(f"  Features: {sorted(best_feats_search)}")
else:
    print(f"  Pool > 20: Sequential Feature Selector (forward then backward)...")
    scaler_sfs = StandardScaler()
    X_pool_sc  = scaler_sfs.fit_transform(X_s4[pool])

    # Forward SFS — find optimal count
    sfs_fwd = SequentialFeatureSelector(
        Ridge(), direction="forward", scoring="r2", cv=cv5, n_features_to_select="auto"
    )
    sfs_fwd.fit(X_pool_sc, y)
    fwd_feats = [pool[i] for i, s in enumerate(sfs_fwd.get_support()) if s]
    fwd_r2    = cv_r2(X_s4[fwd_feats])
    print(f"  Forward  SFS: n={len(fwd_feats)} | CV R²={fwd_r2:.4f} | {sorted(fwd_feats)}")

    # Backward SFS
    sfs_bwd = SequentialFeatureSelector(
        Ridge(), direction="backward", scoring="r2", cv=cv5, n_features_to_select="auto"
    )
    sfs_bwd.fit(X_pool_sc, y)
    bwd_feats = [pool[i] for i, s in enumerate(sfs_bwd.get_support()) if s]
    bwd_r2    = cv_r2(X_s4[bwd_feats])
    print(f"  Backward SFS: n={len(bwd_feats)} | CV R²={bwd_r2:.4f} | {sorted(bwd_feats)}")

    if fwd_r2 >= bwd_r2:
        best_feats_search, best_r2_search = fwd_feats, fwd_r2
    else:
        best_feats_search, best_r2_search = bwd_feats, bwd_r2

if best_feats_search:
    set_results.append({
        "set_name":    "exhaustive_or_SFS_best",
        "n_features":  len(best_feats_search),
        "cv_r2":       best_r2_search,
        "features":    ", ".join(sorted(best_feats_search)),
    })

# --- Overall best ---
results_df = pd.DataFrame(set_results).sort_values("cv_r2", ascending=False).reset_index(drop=True)
print("\n  === All candidate sets ranked by CV R² ===")
print(results_df[["set_name", "n_features", "cv_r2"]].to_string(index=False))

best_row   = results_df.iloc[0]
best_name  = best_row["set_name"]
best_r2    = best_row["cv_r2"]
final_feats = [f.strip() for f in best_row["features"].split(",")]

# Force FIXED_FEATURES into the final set if SFS dropped them
forced_in = [f for f in FIXED_FEATURES if f in X_s4.columns and f not in final_feats]
if forced_in:
    final_feats = final_feats + forced_in
    best_r2     = cv_r2(X_s4[final_feats])
    print(f"\n  Forced into final set: {forced_in}  | updated CV R²={best_r2:.4f}")

print(f"\n  *** BEST SET: '{best_name}' | n={len(final_feats)} | CV R²={best_r2:.4f} ***")
print(f"  Features: {sorted(final_feats)}")

# ────────────────────────────────────────────────
# VIF Pruning: iteratively remove highest-VIF feature until all VIF < 10
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("VIF PRUNING (iterative, threshold=10)")
print("="*60)

VIF_THRESHOLD = 10

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

vif_pruned_feats = final_feats.copy()
vif_log = []

while True:
    X_vif_check = X_s4[vif_pruned_feats].copy()
    vif_round   = compute_vif(X_vif_check)
    vif_log.append(vif_round.assign(n_remaining=len(vif_pruned_feats)))

    # Exclude fixed features from removal candidates
    removable = vif_round[~vif_round["feature"].isin(FIXED_FEATURES)]
    if removable.empty:
        print(f"  n={len(vif_pruned_feats):2d} | only fixed features remain -> stop")
        break

    max_vif  = removable.iloc[0]["VIF"]
    max_feat = removable.iloc[0]["feature"]

    print(f"  n={len(vif_pruned_feats):2d} | max removable VIF={max_vif:.2f} ({max_feat})", end="")

    if max_vif <= VIF_THRESHOLD or len(vif_pruned_feats) <= 2:
        print("  -> all pass")
        break

    print(f"  -> remove '{max_feat}'")
    vif_pruned_feats.remove(max_feat)

vif_pruned_r2 = cv_r2(X_s4[vif_pruned_feats])
print(f"\n  After VIF pruning: n={len(vif_pruned_feats)} | CV R²={vif_pruned_r2:.4f}")
print(f"  Features: {sorted(vif_pruned_feats)}")

# Compare before / after
r2_change = vif_pruned_r2 - best_r2
print(f"  CV R² change: {r2_change:+.4f} vs pre-pruning ({best_r2:.4f})")

# Save VIF pruning log
vif_log_df = pd.concat(vif_log, ignore_index=True)

# ────────────────────────────────────────────────
# Plots
# ────────────────────────────────────────────────

# Plot 1: CV R² comparison across all sets
fig, ax = plt.subplots(figsize=(10, 5))
plot_res = results_df.dropna(subset=["cv_r2"]).sort_values("cv_r2")
colors_bar = ["#e74c3c" if n == best_name else "#3498db" for n in plot_res["set_name"]]
ax.barh(plot_res["set_name"], plot_res["cv_r2"], color=colors_bar)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("5-fold CV R²")
ax.set_title("CV R² by Candidate Feature Set\n(red = best)")
for i, (r2, n) in enumerate(zip(plot_res["cv_r2"], plot_res["n_features"])):
    ax.text(max(r2, 0) + 0.001, i, f" n={n}", va="center", fontsize=8)
plt.tight_layout()
fig.savefig(RESULT_DIR / "04_cv_r2_comparison.png", dpi=150)
plt.close()
print(f"\n  Saved: 04_cv_r2_comparison.png")

# use VIF-pruned features for all subsequent plots
final_feats    = vif_pruned_feats
best_r2        = vif_pruned_r2

# Plot 2: Voting + Spearman of best features
final_vote_df   = vote_df[vote_df["feature"].isin(final_feats)].sort_values("votes", ascending=True)
final_spear_df  = spearman_df[spearman_df["feature"].isin(final_feats)].sort_values("spearman_rho")

fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(final_feats) * 0.5 + 1)))

v_colors = {3: "#e74c3c", 2: "#e67e22", 1: "#bdc3c7", 0: "#ecf0f1"}
axes[0].barh(final_vote_df["feature"],
             final_vote_df["votes"],
             color=[v_colors.get(v, "#bdc3c7") for v in final_vote_df["votes"]])
axes[0].set_xlabel("Votes (LASSO + RFE + RF, max 3)")
axes[0].set_title("Ensemble Votes – Best Feature Set")

rho_colors = ["#e74c3c" if r < 0 else "#3498db" for r in final_spear_df["spearman_rho"]]
axes[1].barh(final_spear_df["feature"], final_spear_df["spearman_rho"], color=rho_colors)
axes[1].axvline(0, color="gray", linewidth=0.8)
axes[1].set_xlabel("Spearman rho with TAMA")
axes[1].set_title("Spearman Correlation – Best Feature Set")

plt.suptitle(f"Best set: '{best_name}' | n={len(final_feats)} | CV R²={best_r2:.4f}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(RESULT_DIR / "05_final_features_summary.png", dpi=150)
plt.close()
print(f"  Saved: 05_final_features_summary.png")

# ────────────────────────────────────────────────
# Final feature visualizations
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL FEATURE VISUALIZATIONS")
print("="*60)

X_final = X_s4[final_feats].copy()
df_final = X_final.copy()
df_final["TAMA"] = y.values

n_final = len(final_feats)

# ── Fig 06: Correlation heatmap (final features only, annotated) ──
corr_final = X_final.corr(method="pearson")
fig, ax = plt.subplots(figsize=(max(8, n_final * 0.7), max(6, n_final * 0.65)))
mask_upper = np.triu(np.ones_like(corr_final, dtype=bool), k=1)
sns.heatmap(
    corr_final, mask=mask_upper, cmap="coolwarm", center=0, vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 8},
    ax=ax, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
)
ax.set_title(f"Pearson Correlation – Final {n_final} Features", fontsize=13, pad=12)
plt.tight_layout()
fig.savefig(RESULT_DIR / "06_final_correlation_heatmap.png", dpi=150)
plt.close()
print(f"  Saved: 06_final_correlation_heatmap.png")

# ── Fig 07: Scatter plots — each feature vs TAMA ──
ncols = 4
nrows = (n_final + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
axes_flat = axes.flatten() if nrows > 1 else np.array(axes).flatten()

for i, feat in enumerate(sorted(final_feats)):
    ax = axes_flat[i]
    ax.scatter(X_final[feat], y, alpha=0.25, s=8, color="#2980b9", rasterized=True)

    # linear trend line
    z = np.polyfit(X_final[feat], y, 1)
    xline = np.linspace(X_final[feat].min(), X_final[feat].max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), color="#e74c3c", linewidth=1.5)

    rho, pval = stats.spearmanr(X_final[feat], y)
    pval_str  = f"{pval:.3f}" if pval >= 0.001 else "<0.001"
    ax.set_title(f"{feat}\n(rho={rho:.3f}, p={pval_str})", fontsize=8)
    ax.set_xlabel(feat, fontsize=7)
    ax.set_ylabel("TAMA", fontsize=7)
    ax.tick_params(labelsize=7)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle(f"Final Features vs TAMA (n={n_final})", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(RESULT_DIR / "07_final_scatter_vs_tama.png", dpi=150)
plt.close()
print(f"  Saved: 07_final_scatter_vs_tama.png")

# ── Fig 08: Distribution (KDE + rug) of each final feature ──
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
axes_flat = axes.flatten() if nrows > 1 else np.array(axes).flatten()

for i, feat in enumerate(sorted(final_feats)):
    ax = axes_flat[i]
    sns.histplot(X_final[feat], kde=True, ax=ax, color="#27ae60", bins=30,
                 edgecolor="white", linewidth=0.3)
    ax.axvline(X_final[feat].median(), color="#e74c3c", linestyle="--",
               linewidth=1, label=f"median={X_final[feat].median():.2f}")
    ax.set_title(feat, fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle(f"Distribution of Final Features (n={n_final})", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(RESULT_DIR / "08_final_distributions.png", dpi=150)
plt.close()
print(f"  Saved: 08_final_distributions.png")

# ── Fig 09: Clustermap (hierarchical clustering of final features + TAMA) ──
df_cluster = X_final.copy()
df_cluster["TAMA"] = y.values
df_cluster_std = (df_cluster - df_cluster.mean()) / df_cluster.std()
cg = sns.clustermap(
    df_cluster_std.T,
    cmap="vlag", figsize=(max(10, n_final * 0.5), n_final * 0.65 + 2),
    col_cluster=True, row_cluster=True,
    yticklabels=True, xticklabels=False,
    cbar_pos=(0.02, 0.85, 0.03, 0.12),
)
cg.figure.suptitle(f"Clustermap – Final Features + TAMA (z-scored, n={n_final})",
                   fontsize=11, y=1.01)
cg.figure.savefig(RESULT_DIR / "09_final_clustermap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 09_final_clustermap.png")

# ── Fig 10: TAMA quantile groups — boxplot per feature ──
df_box = X_final.copy()
df_box["TAMA_group"] = pd.qcut(y, q=3, labels=["Low", "Mid", "High"])
df_box_std = df_box.copy()
for feat in final_feats:
    s = df_box_std[feat].std()
    if s > 0:
        df_box_std[feat] = (df_box_std[feat] - df_box_std[feat].mean()) / s

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))
axes_flat = axes.flatten() if nrows > 1 else np.array(axes).flatten()
palette   = {"Low": "#3498db", "Mid": "#f39c12", "High": "#e74c3c"}

for i, feat in enumerate(sorted(final_feats)):
    ax = axes_flat[i]
    sns.boxplot(data=df_box_std, x="TAMA_group", y=feat, order=["Low", "Mid", "High"],
                palette=palette, ax=ax, width=0.5, fliersize=2)
    ax.set_title(feat, fontsize=8)
    ax.set_xlabel("TAMA tertile", fontsize=7)
    ax.set_ylabel("z-score", fontsize=7)
    ax.tick_params(labelsize=7)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle(f"Feature Values by TAMA Tertile (z-scored, n={n_final})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(RESULT_DIR / "10_final_boxplot_by_tama_group.png", dpi=150)
plt.close()
print(f"  Saved: 10_final_boxplot_by_tama_group.png")

# ────────────────────────────────────────────────
# Save Excel summary
# ────────────────────────────────────────────────
n_pre_vif = len([f.strip() for f in best_row["features"].split(",")])
pipeline_log = pd.DataFrame([
    {"Step": "Dataset prep - dropna",     "Removed": n_dropped,                        "Features": f"{n_dropped} rows dropped"},
    {"Step": "Step 1 - Near-zero var",    "Removed": len(low_var_feats),               "Features": ", ".join(low_var_feats)},
    {"Step": "Step 2 - High correlation", "Removed": len(high_corr_feats),             "Features": ", ".join(high_corr_feats)},
    {"Step": "Step 3 - Union pass",       "Removed": X_step2.shape[1]-len(step3_feats),"Features": ""},
    {"Step": "Step 4 - SFS best",         "Removed": 0,                                "Features": ", ".join(sorted([f.strip() for f in best_row["features"].split(",")]))},
    {"Step": "VIF pruning (VIF>10)",      "Removed": n_pre_vif - len(final_feats),     "Features": ", ".join(sorted(final_feats))},
])

# Final VIF table for the pruned set
final_vif_df = compute_vif(X_s4[final_feats])

with pd.ExcelWriter(RESULT_DIR / "feature_selection_summary.xlsx") as writer:
    pipeline_log.to_excel(writer,      sheet_name="pipeline_log",    index=False)
    results_df.to_excel(writer,        sheet_name="set_cv_r2",       index=False)
    vote_df.to_excel(writer,           sheet_name="vote_details",     index=False)
    univariate_df.to_excel(writer,     sheet_name="univariate_stats", index=False)
    vif_df.to_excel(writer,            sheet_name="vif_step2",        index=False)
    vif_log_df.to_excel(writer,        sheet_name="vif_pruning_log",  index=False)
    final_vif_df.to_excel(writer,      sheet_name="vif_final",        index=False)
    pd.DataFrame({"final_features": sorted(final_feats)}).to_excel(
        writer, sheet_name="final_features", index=False)

print(f"  Saved: feature_selection_summary.xlsx")

# ────────────────────────────────────────────────
# Comparison: previous selection vs pipeline result
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON: Previous selection vs Pipeline result")
print("="*60)

PREVIOUS_FEATS = ["mean", "CV", "skewness", "slope_abs_mean"]

# Use X_filled so skewness (removed in Step 1) can still be evaluated fairly
prev_available = [f for f in PREVIOUS_FEATS if f in X_filled.columns]
prev_missing   = [f for f in PREVIOUS_FEATS if f not in X_filled.columns]
if prev_missing:
    print(f"  WARNING: {prev_missing} not found in data, excluded from evaluation")

X_prev    = X_filled[prev_available].copy()
prev_r2   = cv_r2(X_prev)
new_r2    = best_r2
new_feats = sorted(final_feats)

print(f"\n  Previous set ({len(prev_available)}): {prev_available}")
print(f"    CV R² = {prev_r2:.4f}")
print(f"\n  Pipeline set ({len(new_feats)}): {new_feats}")
print(f"    CV R² = {new_r2:.4f}")
print(f"\n  Difference (pipeline - previous): {new_r2 - prev_r2:+.4f}")

# Per-feature univariate stats for both sets
def feat_stats(feats, X_source):
    rows = []
    for f in feats:
        if f not in X_source.columns:
            continue
        rho, pval = stats.spearmanr(X_source[f], y)
        mi_val = mutual_info_regression(X_source[[f]], y, random_state=42)[0]
        rows.append({"feature": f, "spearman_rho": rho, "p_value": pval, "MI": mi_val})
    return pd.DataFrame(rows)

prev_stats = feat_stats(prev_available, X_filled)
new_stats  = feat_stats(new_feats, X_filled)

# ── Fig 11: CV R² bar comparison ──
compare_sets = {
    f"Previous\n({len(prev_available)} feats)": prev_r2,
    f"Pipeline\n({len(new_feats)} feats)":      new_r2,
}
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(compare_sets.keys(), compare_sets.values(),
              color=["#95a5a6", "#e74c3c"], width=0.4, edgecolor="white")
ax.set_ylabel("5-fold CV R²")
ax.set_title("Previous vs Pipeline Feature Set\n(CV R² comparison)")
ax.set_ylim(0, max(compare_sets.values()) * 1.25)
for bar, val in zip(bars, compare_sets.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(RESULT_DIR / "11_comparison_cv_r2.png", dpi=150)
plt.close()
print(f"\n  Saved: 11_comparison_cv_r2.png")

# ── Fig 12: Spearman rho side-by-side ──
all_compared = list(dict.fromkeys(prev_available + new_feats))  # preserve order, deduplicate
rho_prev = {r["feature"]: r["spearman_rho"] for _, r in prev_stats.iterrows()}
rho_new  = {r["feature"]: r["spearman_rho"] for _, r in new_stats.iterrows()}

x      = np.arange(len(all_compared))
width  = 0.38
fig, ax = plt.subplots(figsize=(max(10, len(all_compared) * 0.7), 5))
rho_p_vals = [rho_prev.get(f, 0) for f in all_compared]
rho_n_vals = [rho_new.get(f, 0)  for f in all_compared]
b1 = ax.bar(x - width/2, rho_p_vals, width, label="Previous",  color="#95a5a6", edgecolor="white")
b2 = ax.bar(x + width/2, rho_n_vals, width, label="Pipeline",  color="#e74c3c", edgecolor="white")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_xticks(x)
ax.set_xticklabels(all_compared, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Spearman rho with TAMA")
ax.set_title("Spearman Correlation – Previous vs Pipeline Features")
ax.legend()

# mark features unique to each set
for i, f in enumerate(all_compared):
    in_prev = f in prev_available
    in_new  = f in new_feats
    tag = ""
    if in_prev and not in_new:
        tag = "●"   # previous only
    elif in_new and not in_prev:
        tag = "★"   # pipeline only
    if tag:
        ax.text(x[i], max(abs(rho_prev.get(f, 0)), abs(rho_new.get(f, 0))) + 0.01,
                tag, ha="center", fontsize=9)

ax.text(0.01, 0.97, "● previous only  ★ pipeline only", transform=ax.transAxes,
        fontsize=8, va="top", color="gray")
plt.tight_layout()
fig.savefig(RESULT_DIR / "12_comparison_spearman.png", dpi=150)
plt.close()
print(f"  Saved: 12_comparison_spearman.png")

# ── Fig 13: Scatter plots — previous features vs TAMA ──
n_prev = len(prev_available)
ncols_p = min(n_prev, 4)
nrows_p = (n_prev + ncols_p - 1) // ncols_p
fig, axes = plt.subplots(nrows_p, ncols_p, figsize=(ncols_p * 4, nrows_p * 3.5))
axes_flat = np.array(axes).flatten()
for i, feat in enumerate(prev_available):
    ax = axes_flat[i]
    ax.scatter(X_filled[feat], y, alpha=0.25, s=8, color="#95a5a6", rasterized=True)
    z = np.polyfit(X_filled[feat], y, 1)
    xline = np.linspace(X_filled[feat].min(), X_filled[feat].max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), color="#e74c3c", linewidth=1.5)
    rho, pval = stats.spearmanr(X_filled[feat], y)
    pval_str = f"{pval:.3f}" if pval >= 0.001 else "<0.001"
    ax.set_title(f"{feat}\n(rho={rho:.3f}, p={pval_str})", fontsize=9)
    ax.set_xlabel(feat, fontsize=8)
    ax.set_ylabel("TAMA", fontsize=8)
    ax.tick_params(labelsize=7)
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.suptitle(f"Previous Features vs TAMA  |  CV R²={prev_r2:.4f}", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(RESULT_DIR / "13_comparison_prev_scatter.png", dpi=150)
plt.close()
print(f"  Saved: 13_comparison_prev_scatter.png")

# Save comparison table to Excel
comp_prev = prev_stats.assign(set="previous")
comp_new  = new_stats.assign(set="pipeline")
comp_df   = pd.concat([comp_prev, comp_new], ignore_index=True)
comp_summary = pd.DataFrame([
    {"set": "previous", "features": ", ".join(prev_available), "n": len(prev_available), "cv_r2": prev_r2},
    {"set": "pipeline", "features": ", ".join(new_feats),      "n": len(new_feats),      "cv_r2": new_r2},
])
with pd.ExcelWriter(RESULT_DIR / "comparison_prev_vs_pipeline.xlsx") as writer:
    comp_summary.to_excel(writer, sheet_name="summary",    index=False)
    comp_df.to_excel(writer,      sheet_name="feat_stats", index=False)
print(f"  Saved: comparison_prev_vs_pipeline.xlsx")

print("\n" + "="*60)
print(f"DONE  ->  {RESULT_DIR}")
print(f"BEST SET : '{best_name}'")
print(f"N        : {len(final_feats)}")
print(f"CV R2    : {best_r2:.4f}")
print(f"FEATURES : {sorted(final_feats)}")
print("="*60)
