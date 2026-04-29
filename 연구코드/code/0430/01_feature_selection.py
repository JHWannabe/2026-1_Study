"""
Feature Selection Pipeline for AEC Signal Features
Target: TAMA (continuous, regression)

Pipeline:
  Step 1. Pre-filter   : near-zero variance
  Step 2. Redundancy   : Pearson |r| >= 0.95 (VIF reported only, not used to remove)
  Step 3. Univariate   : MI > 0 OR Spearman p < 0.05  (no top-N limit)
  Step 4. Model-based  : LASSO / RFECV / RF Permutation  (each selects its own count)
  Final  : 7 candidate sets evaluated by 5-fold CV R²;
           pool <= 20 -> exhaustive 2^N search,
           pool > 20  -> Sequential Feature Selector (forward + backward)

실행 대상 (3개):
  1. 강남 단독
  2. 신촌 단독
  3. 병합 (강남 + 신촌, 공통 피처만 사용)

결과 저장:
  results/feature_selection/gangnam/   ← 강남 단독
  results/feature_selection/sinchon/   ← 신촌 단독
  results/feature_selection/merged/    ← 병합
  results/feature_selection/cross_dataset_comparison.*  ← 3개 비교

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[0424 feature_selection.py → 0430 feature_selection_pipeline.py 설계 변경]

0424 방식 (수동 가이드):
  - 전체 AEC 피처와 TAMA 간 Pearson/Spearman 상관계수 + VIF만 계산
  - 결과 보고서(feature_selection_report.xlsx)를 보고 연구자가 수동으로
    config.py의 SELECTED_AEC_FEATURES를 결정
  - 단일 병원(강남)만 분석
  - 결과: ['mean', 'CV', 'skewness', 'slope_abs_mean'] (AEC_prev)

0430 방식 (자동 파이프라인):
  - 4단계 자동 필터링 + 앙상블 모델 기반 선택으로 최적 피처 세트를 도출
  - 강남·신촌 각각과 병합 데이터 3가지 실행 후 재현성 비교
  - 선택 결과가 각 하위 폴더의 feature_selection_summary.xlsx "final_features" 시트에 저장
  - regression_analysis.py는 merged 결과를 AEC_new 기본값으로 사용

핵심 도입 이유:
  AEC 피처 수가 60개 이상일 때 단순 상관계수만으로 최적 세트를 선택하면
  과적합·다중공선성 위험이 있음. 앙상블 투표 + CV R² 기준으로 객관적 선택.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
warnings.filterwarnings("ignore")

import os
import itertools
import subprocess
import tempfile
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
# 0. Paths & constants
# ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
RESULT_DIR = SCRIPT_DIR.parent / "results" / "feature_selection"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# mean은 AEC 신호의 평균 진폭 수준을 나타내는 가장 해석 가능한 피처.
# 어떤 필터링 단계를 거치더라도 최종 세트에 강제 포함하여 임상 해석 가능성을 유지.
FIXED_FEATURES = ["mean"]

# 0424 수동 선택 피처 (비교 기준)
PREVIOUS_FEATS = ["mean", "CV", "skewness", "slope_abs_mean"]


# ────────────────────────────────────────────────
# 1. Data loading helpers
# ────────────────────────────────────────────────

def _load_xlsx(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """xlsx 파일을 temp에 복사한 뒤 features·metadata 시트를 읽어 반환."""
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
    """features + metadata(TAMA)를 병합하고 결측 제거 후 (X, y) 반환."""
    meta_df = meta_df.copy()
    meta_df["TAMA"] = pd.to_numeric(meta_df["TAMA"], errors="coerce")
    meta_df = meta_df.dropna(subset=["TAMA"])

    df = feat_df.merge(meta_df[["PatientID", "TAMA"]], on="PatientID", how="inner")
    print(f"  [{label}] {df.shape[0]}명 | TAMA: {df['TAMA'].min():.0f}~{df['TAMA'].max():.0f}"
          f" | mean={df['TAMA'].mean():.1f}")

    X_raw = df.drop(columns=["PatientID", "TAMA"])
    y_raw = df["TAMA"].astype(float)

    df_complete = pd.concat([X_raw, y_raw.rename("TAMA")], axis=1).dropna().reset_index(drop=True)
    n_dropped = len(X_raw) - len(df_complete)
    print(f"  [{label}] dropna: {n_dropped}행 제거 → {len(df_complete)}행 사용")

    X = df_complete.drop(columns=["TAMA"])
    y = df_complete["TAMA"].astype(float)
    return X, y


# ────────────────────────────────────────────────
# 2. Pipeline helpers (stateless, no global y)
# ────────────────────────────────────────────────

def drop_high_corr(corr_mat, threshold=0.95, protected=None):
    """
    |r| ≥ threshold인 쌍에서 후순위(j번째) 피처를 제거.
    protected 피처가 제거 대상이면 대신 선순위(i번째)를 제거하여 보호.
    threshold=0.95: AEC 피처 간 r=0.95 이상이면 사실상 동일 정보 → 한 쪽만 유지.
    """
    protected = set(protected or [])
    to_drop   = set()
    cols      = corr_mat.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_mat.iloc[i, j]) >= threshold and cols[j] not in to_drop:
                if cols[j] not in protected:
                    to_drop.add(cols[j])
                elif cols[i] not in protected:
                    to_drop.add(cols[i])
    return list(to_drop)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)


# ────────────────────────────────────────────────
# 3. Main pipeline function
# ────────────────────────────────────────────────

def run_pipeline(X_filled: pd.DataFrame, y: pd.Series,
                 label: str, out_dir: Path) -> dict:
    """
    단일 데이터셋(X_filled, y)에 대해 전체 피처 선택 파이프라인을 실행.

    강남·신촌·병합 세 가지 데이터셋에 각각 독립적으로 호출하여
    데이터셋별 최적 AEC 피처 세트와 CV R²를 비교할 수 있도록 함수로 캡슐화.

    반환값: {"label", "final_feats", "best_r2", "prev_r2", "n"}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # cv_r2는 y를 클로저로 캡처 — 각 호출마다 해당 데이터셋의 y를 사용
    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

    def cv_r2(X_sub: pd.DataFrame) -> float:
        """
        Ridge 회귀를 사용한 5-fold CV R² 추정.
        Ridge를 사용하는 이유: 다중공선성에 강건하며 OLS와 동일한 선형 구조 가정.
        """
        if X_sub.shape[1] == 0:
            return -np.inf
        Xs = StandardScaler().fit_transform(X_sub)
        return float(cross_val_score(Ridge(), Xs, y, cv=cv5, scoring="r2").mean())

    n_total = len(y)
    print(f"\n{'='*60}")
    print(f"  PIPELINE: {label}  (n={n_total})")
    print(f"  출력 경로: {out_dir}")
    print(f"{'='*60}")

    # ── Step 1. Pre-filter: near-zero variance ──────────────────
    # 분산이 거의 0인 피처는 TAMA와 어떤 관계도 맺을 수 없으므로 조기 제거.
    # 표준화 후 threshold=0.01 적용하여 스케일 차이에 무관하게 판별.
    print("\n" + "="*60)
    print(f"STEP 1. Pre-filter (near-zero variance)  [{label}]")
    print("="*60)

    X_scaled  = StandardScaler().fit_transform(X_filled)
    vt        = VarianceThreshold(threshold=0.01)
    vt.fit(X_scaled)
    low_var_feats = [c for c, keep in zip(X_filled.columns, vt.get_support()) if not keep]
    print(f"  Near-zero variance: {len(low_var_feats)} features")

    X_step1 = X_filled.drop(columns=low_var_feats)
    print(f"  [Step 1] Removed {len(low_var_feats)} -> Remaining: {X_step1.shape[1]}")

    # ── Step 2. Redundancy: Pearson correlation ─────────────────
    # |r| ≥ 0.95인 쌍에서 후순위 피처 제거. VIF는 참고용만 계산.
    # VIF 기반 제거는 Step 4 이후 별도 VIF pruning 단계에서 수행.
    print("\n" + "="*60)
    print(f"STEP 2. Redundancy (correlation >= 0.95)  [{label}]")
    print("="*60)

    corr_matrix    = X_step1.corr(method="pearson")
    high_corr_feats = drop_high_corr(corr_matrix, threshold=0.95, protected=FIXED_FEATURES)
    X_step2 = X_step1.drop(columns=high_corr_feats)
    print(f"  Correlation removed: {len(high_corr_feats)} -> Remaining: {X_step2.shape[1]}")

    vif_df = pd.DataFrame({
        "feature": X_step2.columns,
        "VIF": [variance_inflation_factor(X_step2.values, i) for i in range(X_step2.shape[1])],
    }).sort_values("VIF", ascending=False)
    print(f"  VIF > 10 count (informational): {(vif_df['VIF'] > 10).sum()}")
    vif_df.to_excel(out_dir / "step2_vif.xlsx", index=False)

    fig, ax = plt.subplots(figsize=(18, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=False, ax=ax, square=True, linewidths=0.3)
    ax.set_title(f"Pearson Correlation Matrix – {label} (Step 1 features)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "01_correlation_heatmap.png", dpi=150)
    plt.close()

    # ── Step 3. Univariate: MI > 0 OR Spearman p < 0.05 ────────
    # MI와 Spearman을 OR로 결합: 비선형 탐지 + 통계적 유의성을 상보적으로 활용.
    # 상위 N 제한 없음 — Step 4 모델 기반 단계에 충분한 후보를 넘김.
    print("\n" + "="*60)
    print(f"STEP 3. Univariate association  [{label}]")
    print("="*60)

    mi_scores  = mutual_info_regression(X_step2, y, random_state=42)
    mi_df      = pd.DataFrame({"feature": X_step2.columns, "MI": mi_scores}).sort_values("MI", ascending=False)

    spearman_rows = []
    for col in X_step2.columns:
        rho, pval = stats.spearmanr(X_step2[col], y)
        spearman_rows.append({"feature": col, "spearman_rho": rho, "p_value": pval})
    spearman_df = pd.DataFrame(spearman_rows).sort_values("p_value")

    univariate_df = mi_df.merge(spearman_df, on="feature").sort_values("MI", ascending=False)
    univariate_df.to_excel(out_dir / "step3_univariate.xlsx", index=False)

    mi_pos        = set(mi_df[mi_df["MI"] > 0]["feature"])
    sig_spear     = set(spearman_df[spearman_df["p_value"] < 0.05]["feature"])
    fixed_in_step2 = [f for f in FIXED_FEATURES if f in X_step2.columns]
    step3_feats   = list((mi_pos | sig_spear) | set(fixed_in_step2))

    print(f"  MI > 0:              {len(mi_pos)} features")
    print(f"  Spearman p < 0.05:   {len(sig_spear)} features")
    print(f"  Fixed (always in):   {fixed_in_step2}")
    print(f"  Union (Step 3 pass): {len(step3_feats)} features")

    plot_df = mi_df[mi_df["feature"].isin(step3_feats)]
    fig, ax = plt.subplots(figsize=(14, max(6, len(step3_feats) * 0.32 + 1)))
    colors  = ["#e74c3c" if f in sig_spear else "#f39c12" for f in plot_df["feature"]]
    ax.barh(plot_df["feature"][::-1], plot_df["MI"][::-1], color=colors[::-1])
    ax.set_xlabel("Mutual Information Score")
    ax.set_title(f"Step 3 candidates [{label}]\n(red=MI>0 AND p<0.05 | orange=MI>0 only / p<0.05 only)")
    plt.tight_layout()
    fig.savefig(out_dir / "02_mutual_information.png", dpi=150)
    plt.close()

    # ── Step 4. Model-based: LASSO / RFECV / RF Permutation ─────
    # 세 모델의 앙상블 투표로 pool 구성.
    #   - LASSO: 선형 희소성 기반 선택
    #   - RFECV: 기여도 순서 기반 재귀 제거
    #   - RF Permutation: 비선형·교호작용 포착
    print("\n" + "="*60)
    print(f"STEP 4. Model-based selection  [{label}]")
    print("="*60)

    X_s4    = X_step2[step3_feats].copy()
    X_s4_sc = StandardScaler().fit_transform(X_s4)

    # 4-1. LASSO
    print(f"  [4-1] LassoCV on {X_s4.shape[1]} features...")
    lasso = LassoCV(cv=cv5, random_state=42, max_iter=10000).fit(X_s4_sc, y)
    lasso_coef_df  = pd.DataFrame({"feature": X_s4.columns, "coef": np.abs(lasso.coef_)}).sort_values("coef", ascending=False)
    lasso_selected = set(lasso_coef_df[lasso_coef_df["coef"] > 0]["feature"])
    print(f"    alpha={lasso.alpha_:.4f} | selected: {len(lasso_selected)}")

    # 4-2. RFECV (Ridge)
    print(f"  [4-2] RFECV (Ridge) on {X_s4.shape[1]} features...")
    rfecv = RFECV(estimator=Ridge(), step=1, cv=cv5, scoring="r2", min_features_to_select=1)
    rfecv.fit(X_s4_sc, y)
    rfe_selected = set(X_s4.columns[rfecv.support_])
    print(f"    optimal n={rfecv.n_features_} | selected: {len(rfe_selected)}")

    # 4-3. RF Permutation Importance
    print(f"  [4-3] RF Permutation Importance on {X_s4.shape[1]} features...")
    rf   = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_s4, y)
    perm = permutation_importance(rf, X_s4, y, n_repeats=15, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature":   X_s4.columns,
        "perm_mean": perm.importances_mean,
        "perm_std":  perm.importances_std,
    }).sort_values("perm_mean", ascending=False)
    perm_selected = set(perm_df[perm_df["perm_mean"] > 0]["feature"])
    print(f"    selected: {len(perm_selected)}")

    fig, ax = plt.subplots(figsize=(10, max(5, len(step3_feats) * 0.35 + 1)))
    ax.barh(perm_df["feature"][::-1], perm_df["perm_mean"][::-1],
            xerr=perm_df["perm_std"][::-1], capsize=2, color="#3498db")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean decrease in R² (permutation)")
    ax.set_title(f"RF Permutation Importance – {label}")
    plt.tight_layout()
    fig.savefig(out_dir / "03_permutation_importance.png", dpi=150)
    plt.close()

    # 앙상블 투표 집계
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

    # ── Final. Evaluate candidate sets + exhaustive / SFS search ─
    # 7개 사전 정의 후보 세트를 CV R²로 평가 후 pool 전체 탐색.
    # pool ≤ 20: 완전 탐색(2^N) | pool > 20: forward+backward SFS
    print("\n" + "="*60)
    print(f"FINAL. Evaluate candidate sets + exhaustive/SFS search  [{label}]")
    print("="*60)

    candidate_sets = {
        "LASSO_only":   list(lasso_selected),
        "RFE_only":     list(rfe_selected),
        "RF_Perm_only": list(perm_selected),
        "vote_ge1":     [f for f, v in votes.items() if v >= 1],
        "vote_ge2":     [f for f, v in votes.items() if v >= 2],
        "vote_ge3":     [f for f, v in votes.items() if v >= 3],
        "union_all":    list(all_method_feats),
    }

    set_results = []
    for name, feats in candidate_sets.items():
        if not feats:
            set_results.append({"set_name": name, "n_features": 0, "cv_r2": np.nan, "features": ""})
            continue
        r2 = cv_r2(X_s4[feats])
        set_results.append({"set_name": name, "n_features": len(feats),
                            "cv_r2": r2, "features": ", ".join(sorted(feats))})
        print(f"  {name:16s} | n={len(feats):2d} | CV R²={r2:.4f}")

    pool   = list(all_method_feats)
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
        X_pool_sc = StandardScaler().fit_transform(X_s4[pool])

        sfs_fwd = SequentialFeatureSelector(
            Ridge(), direction="forward", scoring="r2", cv=cv5, n_features_to_select="auto"
        )
        sfs_fwd.fit(X_pool_sc, y)
        fwd_feats = [pool[i] for i, s in enumerate(sfs_fwd.get_support()) if s]
        fwd_r2    = cv_r2(X_s4[fwd_feats])
        print(f"  Forward  SFS: n={len(fwd_feats)} | CV R²={fwd_r2:.4f} | {sorted(fwd_feats)}")

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
            "set_name":   "exhaustive_or_SFS_best",
            "n_features": len(best_feats_search),
            "cv_r2":      best_r2_search,
            "features":   ", ".join(sorted(best_feats_search)),
        })

    results_df = pd.DataFrame(set_results).sort_values("cv_r2", ascending=False).reset_index(drop=True)
    print("\n  === All candidate sets ranked by CV R² ===")
    print(results_df[["set_name", "n_features", "cv_r2"]].to_string(index=False))

    best_row    = results_df.iloc[0]
    best_name   = best_row["set_name"]
    best_r2     = best_row["cv_r2"]
    final_feats = [f.strip() for f in best_row["features"].split(",")]

    # FIXED_FEATURES가 탐색 과정에서 누락된 경우 강제 포함
    forced_in = [f for f in FIXED_FEATURES if f in X_s4.columns and f not in final_feats]
    if forced_in:
        final_feats = final_feats + forced_in
        best_r2     = cv_r2(X_s4[final_feats])
        print(f"\n  Forced into final set: {forced_in}  | updated CV R²={best_r2:.4f}")

    print(f"\n  *** BEST SET: '{best_name}' | n={len(final_feats)} | CV R²={best_r2:.4f} ***")
    print(f"  Features: {sorted(final_feats)}")

    # ── VIF Pruning ──────────────────────────────────────────────
    # VIF > 10인 피처를 반복 제거하여 다중공선성 해소.
    # FIXED_FEATURES는 VIF가 높아도 보호.
    print("\n" + "="*60)
    print(f"VIF PRUNING (iterative, threshold=10)  [{label}]")
    print("="*60)

    VIF_THRESHOLD    = 10
    vif_pruned_feats = final_feats.copy()
    vif_log          = []

    while True:
        X_vif_check = X_s4[vif_pruned_feats].copy()
        vif_round   = compute_vif(X_vif_check)
        vif_log.append(vif_round.assign(n_remaining=len(vif_pruned_feats)))

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
    print(f"  CV R² change: {vif_pruned_r2 - best_r2:+.4f} vs pre-pruning ({best_r2:.4f})")

    vif_log_df  = pd.concat(vif_log, ignore_index=True)
    final_feats = vif_pruned_feats
    best_r2     = vif_pruned_r2

    # ── Plots ────────────────────────────────────────────────────
    # Plot 1: CV R² comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_res   = results_df.dropna(subset=["cv_r2"]).sort_values("cv_r2")
    colors_bar = ["#e74c3c" if n == best_name else "#3498db" for n in plot_res["set_name"]]
    ax.barh(plot_res["set_name"], plot_res["cv_r2"], color=colors_bar)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("5-fold CV R²")
    ax.set_title(f"CV R² by Candidate Feature Set [{label}]\n(red = best)")
    for i, (r2, n) in enumerate(zip(plot_res["cv_r2"], plot_res["n_features"])):
        ax.text(max(r2, 0) + 0.001, i, f" n={n}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "04_cv_r2_comparison.png", dpi=150)
    plt.close()

    # Plot 2: Voting + Spearman
    final_vote_df  = vote_df[vote_df["feature"].isin(final_feats)].sort_values("votes", ascending=True)
    final_spear_df = spearman_df[spearman_df["feature"].isin(final_feats)].sort_values("spearman_rho")
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(final_feats) * 0.5 + 1)))
    v_colors = {3: "#e74c3c", 2: "#e67e22", 1: "#bdc3c7", 0: "#ecf0f1"}
    axes[0].barh(final_vote_df["feature"], final_vote_df["votes"],
                 color=[v_colors.get(v, "#bdc3c7") for v in final_vote_df["votes"]])
    axes[0].set_xlabel("Votes (LASSO + RFE + RF, max 3)")
    axes[0].set_title("Ensemble Votes – Best Feature Set")
    rho_colors = ["#e74c3c" if r < 0 else "#3498db" for r in final_spear_df["spearman_rho"]]
    axes[1].barh(final_spear_df["feature"], final_spear_df["spearman_rho"], color=rho_colors)
    axes[1].axvline(0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("Spearman rho with TAMA")
    axes[1].set_title("Spearman Correlation – Best Feature Set")
    plt.suptitle(f"Best set: '{best_name}' | n={len(final_feats)} | CV R²={best_r2:.4f}  [{label}]",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "05_final_features_summary.png", dpi=150)
    plt.close()

    # Figs 06-10: final feature visualizations
    X_final = X_s4[final_feats].copy()
    n_final = len(final_feats)
    ncols   = 4
    nrows   = max(1, (n_final + ncols - 1) // ncols)

    corr_final = X_final.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(8, n_final * 0.7), max(6, n_final * 0.65)))
    mask_upper = np.triu(np.ones_like(corr_final, dtype=bool), k=1)
    sns.heatmap(corr_final, mask=mask_upper, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                ax=ax, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Pearson Correlation – Final {n_final} Features [{label}]", fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / "06_final_correlation_heatmap.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes_flat = np.array(axes).flatten()
    for i, feat in enumerate(sorted(final_feats)):
        ax = axes_flat[i]
        ax.scatter(X_final[feat], y, alpha=0.25, s=8, color="#2980b9", rasterized=True)
        z = np.polyfit(X_final[feat], y, 1)
        xline = np.linspace(X_final[feat].min(), X_final[feat].max(), 100)
        ax.plot(xline, np.poly1d(z)(xline), color="#e74c3c", linewidth=1.5)
        rho, pval = stats.spearmanr(X_final[feat], y)
        pval_str  = f"{pval:.3f}" if pval >= 0.001 else "<0.001"
        ax.set_title(f"{feat}\n(rho={rho:.3f}, p={pval_str})", fontsize=8)
        ax.set_xlabel(feat, fontsize=7); ax.set_ylabel("TAMA", fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f"Final Features vs TAMA [{label}]  n={n_final}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "07_final_scatter_vs_tama.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes_flat = np.array(axes).flatten()
    for i, feat in enumerate(sorted(final_feats)):
        ax = axes_flat[i]
        sns.histplot(X_final[feat], kde=True, ax=ax, color="#27ae60", bins=30,
                     edgecolor="white", linewidth=0.3)
        ax.axvline(X_final[feat].median(), color="#e74c3c", linestyle="--",
                   linewidth=1, label=f"median={X_final[feat].median():.2f}")
        ax.set_title(feat, fontsize=8); ax.tick_params(labelsize=7); ax.legend(fontsize=6)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f"Distribution of Final Features [{label}]", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "08_final_distributions.png", dpi=150)
    plt.close()

    df_cluster     = X_final.copy(); df_cluster["TAMA"] = y.values
    df_cluster_std = (df_cluster - df_cluster.mean()) / df_cluster.std()
    cg = sns.clustermap(df_cluster_std.T, cmap="vlag",
                        figsize=(max(10, n_final * 0.5), n_final * 0.65 + 2),
                        col_cluster=True, row_cluster=True,
                        yticklabels=True, xticklabels=False,
                        cbar_pos=(0.02, 0.85, 0.03, 0.12))
    cg.figure.suptitle(f"Clustermap – Final Features + TAMA [{label}]", fontsize=11, y=1.01)
    cg.figure.savefig(out_dir / "09_final_clustermap.png", dpi=150, bbox_inches="tight")
    plt.close()

    df_box = X_final.copy()
    df_box["TAMA_group"] = pd.qcut(y, q=3, labels=["Low", "Mid", "High"])
    df_box_std = df_box.copy()
    for feat in final_feats:
        s = df_box_std[feat].std()
        if s > 0:
            df_box_std[feat] = (df_box_std[feat] - df_box_std[feat].mean()) / s
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))
    axes_flat = np.array(axes).flatten()
    palette   = {"Low": "#3498db", "Mid": "#f39c12", "High": "#e74c3c"}
    for i, feat in enumerate(sorted(final_feats)):
        ax = axes_flat[i]
        sns.boxplot(data=df_box_std, x="TAMA_group", y=feat, order=["Low", "Mid", "High"],
                    palette=palette, ax=ax, width=0.5, fliersize=2)
        ax.set_title(feat, fontsize=8); ax.set_xlabel("TAMA tertile", fontsize=7)
        ax.set_ylabel("z-score", fontsize=7); ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f"Feature Values by TAMA Tertile [{label}]", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "10_final_boxplot_by_tama_group.png", dpi=150)
    plt.close()

    # ── Save Excel summary ───────────────────────────────────────
    n_pre_vif    = len([f.strip() for f in best_row["features"].split(",")])
    pipeline_log = pd.DataFrame([
        {"Step": "Step 1 - Near-zero var",    "Removed": len(low_var_feats),                "Features": ", ".join(low_var_feats)},
        {"Step": "Step 2 - High correlation", "Removed": len(high_corr_feats),              "Features": ", ".join(high_corr_feats)},
        {"Step": "Step 3 - Union pass",       "Removed": X_step2.shape[1]-len(step3_feats), "Features": ""},
        {"Step": "Step 4 - Best search",      "Removed": 0,                                 "Features": ", ".join(sorted([f.strip() for f in best_row["features"].split(",")]))},
        {"Step": "VIF pruning (VIF>10)",      "Removed": n_pre_vif - len(final_feats),      "Features": ", ".join(sorted(final_feats))},
    ])
    final_vif_df = compute_vif(X_s4[final_feats])

    with pd.ExcelWriter(out_dir / "feature_selection_summary.xlsx") as writer:
        pipeline_log.to_excel(writer,  sheet_name="pipeline_log",    index=False)
        results_df.to_excel(writer,    sheet_name="set_cv_r2",       index=False)
        vote_df.to_excel(writer,       sheet_name="vote_details",    index=False)
        univariate_df.to_excel(writer, sheet_name="univariate_stats",index=False)
        vif_df.to_excel(writer,        sheet_name="vif_step2",       index=False)
        vif_log_df.to_excel(writer,    sheet_name="vif_pruning_log", index=False)
        final_vif_df.to_excel(writer,  sheet_name="vif_final",       index=False)
        pd.DataFrame({"final_features": sorted(final_feats)}).to_excel(
            writer, sheet_name="final_features", index=False)
    print(f"  Saved: {out_dir / 'feature_selection_summary.xlsx'}")

    # ── Comparison: previous selection vs pipeline result ────────
    # 0424 수동 선택(AEC_prev)과 파이프라인 선택(AEC_new)을 CV R²로 직접 비교.
    prev_available = [f for f in PREVIOUS_FEATS if f in X_filled.columns]
    prev_r2 = cv_r2(X_filled[prev_available]) if prev_available else np.nan
    new_r2  = best_r2
    new_feats = sorted(final_feats)

    print(f"\n  Previous set ({len(prev_available)}): {prev_available}  CV R²={prev_r2:.4f}")
    print(f"  Pipeline set ({len(new_feats)}): {new_feats}  CV R²={new_r2:.4f}")
    print(f"  Difference (pipeline - previous): {new_r2 - prev_r2:+.4f}")

    def feat_stats(feats, X_source):
        rows = []
        for f in feats:
            if f not in X_source.columns:
                continue
            rho, pval = stats.spearmanr(X_source[f], y)
            mi_val = mutual_info_regression(X_source[[f]], y, random_state=42)[0]
            rows.append({"feature": f, "spearman_rho": rho, "p_value": pval, "MI": mi_val})
        return pd.DataFrame(rows)

    prev_stats_df = feat_stats(prev_available, X_filled)
    new_stats_df  = feat_stats(new_feats, X_filled)

    # Fig 11: CV R² bar comparison
    compare_sets = {
        f"Previous\n({len(prev_available)} feats)": prev_r2,
        f"Pipeline\n({len(new_feats)} feats)":      new_r2,
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(compare_sets.keys(), compare_sets.values(),
                  color=["#95a5a6", "#e74c3c"], width=0.4, edgecolor="white")
    ax.set_ylabel("5-fold CV R²")
    ax.set_title(f"Previous vs Pipeline Feature Set [{label}]")
    ax.set_ylim(0, max(v for v in compare_sets.values() if not np.isnan(v)) * 1.25)
    for bar, val in zip(bars, compare_sets.values()):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "11_comparison_cv_r2.png", dpi=150)
    plt.close()

    # Fig 12: Spearman side-by-side
    all_compared = list(dict.fromkeys(prev_available + new_feats))
    rho_prev = {r["feature"]: r["spearman_rho"] for _, r in prev_stats_df.iterrows()}
    rho_new  = {r["feature"]: r["spearman_rho"] for _, r in new_stats_df.iterrows()}
    x     = np.arange(len(all_compared))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(all_compared) * 0.7), 5))
    ax.bar(x - width/2, [rho_prev.get(f, 0) for f in all_compared], width, label="Previous",  color="#95a5a6", edgecolor="white")
    ax.bar(x + width/2, [rho_new.get(f, 0)  for f in all_compared], width, label="Pipeline",  color="#e74c3c", edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(all_compared, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Spearman rho with TAMA")
    ax.set_title(f"Spearman Correlation – Previous vs Pipeline [{label}]")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "12_comparison_spearman.png", dpi=150)
    plt.close()

    print(f"\n  [DONE] {label} -> {out_dir}")
    print(f"  BEST SET: '{best_name}' | n={len(final_feats)} | CV R²={best_r2:.4f}")
    print(f"  FEATURES: {sorted(final_feats)}")

    return {
        "label":       label,
        "n":           n_total,
        "final_feats": final_feats,
        "best_r2":     best_r2,
        "prev_r2":     prev_r2,
        "best_name":   best_name,
    }


# ════════════════════════════════════════════════
# 4. 데이터 로드 & 3개 실행
# ════════════════════════════════════════════════

# 강남·신촌 파일 탐색
all_xlsx = {f: DATA_DIR / f for f in os.listdir(DATA_DIR)
            if f.endswith(".xlsx") and "merged_features" in f}
gangnam_path = next((v for k, v in all_xlsx.items() if "강남" in k), None)
sinchon_path = next((v for k, v in all_xlsx.items() if "신촌" in k), None)

if not gangnam_path and not sinchon_path:
    raise FileNotFoundError(f"merged_features xlsx 파일을 {DATA_DIR}에서 찾을 수 없습니다.")

print(f"[Data] 강남: {gangnam_path.name if gangnam_path else '없음'}")
print(f"[Data] 신촌: {sinchon_path.name if sinchon_path else '없음'}")

# 각 데이터셋 로드
datasets = {}   # key -> (X_filled, y, label, out_dir)

if gangnam_path:
    feat_gn, meta_gn = _load_xlsx(gangnam_path)
    X_gn, y_gn = _prepare_dataset(feat_gn, meta_gn, "강남")
    datasets["gangnam"] = (X_gn, y_gn, "강남", RESULT_DIR / "gangnam")

if sinchon_path:
    feat_sc, meta_sc = _load_xlsx(sinchon_path)
    X_sc, y_sc = _prepare_dataset(feat_sc, meta_sc, "신촌")
    datasets["sinchon"] = (X_sc, y_sc, "신촌", RESULT_DIR / "sinchon")

# 병합: 두 병원의 공통 피처 컬럼만 사용
if gangnam_path and sinchon_path:
    common_cols = X_gn.columns.intersection(X_sc.columns).tolist()
    X_merged    = pd.concat([X_gn[common_cols], X_sc[common_cols]], ignore_index=True)
    y_merged    = pd.concat([y_gn, y_sc], ignore_index=True)
    n_diff      = len(X_gn.columns) + len(X_sc.columns) - 2 * len(common_cols)
    print(f"\n[병합] 공통 피처: {len(common_cols)}개 (비공통 제외: {n_diff}개)")
    datasets["merged"] = (X_merged, y_merged, "병합(강남+신촌)", RESULT_DIR / "merged")

# 3개 데이터셋에 대해 순차 실행
pipeline_results = {}
for key, (X, y_data, lbl, out_dir) in datasets.items():
    pipeline_results[key] = run_pipeline(X, y_data, lbl, out_dir)

# ════════════════════════════════════════════════
# 5. Cross-dataset comparison
# ════════════════════════════════════════════════
print(f"\n{'='*60}")
print("CROSS-DATASET COMPARISON")
print(f"{'='*60}")

# 요약 테이블
summary_rows = []
for key, res in pipeline_results.items():
    summary_rows.append({
        "dataset":     res["label"],
        "n":           res["n"],
        "n_feats":     len(res["final_feats"]),
        "best_set":    res["best_name"],
        "pipeline_r2": round(res["best_r2"], 4),
        "prev_r2":     round(res["prev_r2"], 4),
        "delta_r2":    round(res["best_r2"] - res["prev_r2"], 4),
        "features":    ", ".join(sorted(res["final_feats"])),
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df[["dataset", "n", "n_feats", "pipeline_r2", "prev_r2", "delta_r2"]].to_string(index=False))

# Fig: CV R² bar — 3개 데이터셋 비교
keys_plot   = list(pipeline_results.keys())
labels_plot = [pipeline_results[k]["label"] for k in keys_plot]
pipe_r2s    = [pipeline_results[k]["best_r2"] for k in keys_plot]
prev_r2s    = [pipeline_results[k]["prev_r2"] for k in keys_plot]

x     = np.arange(len(keys_plot))
width = 0.35
fig, ax = plt.subplots(figsize=(max(8, len(keys_plot) * 3), 5))
b1 = ax.bar(x - width/2, prev_r2s,  width, label="Previous (AEC_prev, 4 feats)", color="#95a5a6", edgecolor="white")
b2 = ax.bar(x + width/2, pipe_r2s,  width, label="Pipeline (AEC_new)",            color="#e74c3c", edgecolor="white")
for bar, v in list(zip(b1, prev_r2s)) + list(zip(b2, pipe_r2s)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels_plot, fontsize=11)
ax.set_ylabel("5-fold CV R²")
ax.set_title("Cross-dataset comparison: Previous vs Pipeline Feature Set\n(강남 / 신촌 / 병합)")
ax.legend(fontsize=10); ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(RESULT_DIR / "cross_dataset_comparison_r2.png", dpi=150)
plt.close()
print(f"\n  Saved: cross_dataset_comparison_r2.png")

# Fig: 피처 집합 Venn-style — 데이터셋별 최종 피처 유무 히트맵
all_final_feats = sorted(set(f for res in pipeline_results.values() for f in res["final_feats"]))
heatmap_data = pd.DataFrame(
    {res["label"]: [int(f in res["final_feats"]) for f in all_final_feats]
     for res in pipeline_results.values()},
    index=all_final_feats,
)
fig, ax = plt.subplots(figsize=(max(6, len(pipeline_results) * 2.5), max(5, len(all_final_feats) * 0.5 + 1)))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
            cbar=False, ax=ax, vmin=0, vmax=1,
            annot_kws={"size": 11, "fontweight": "bold"})
ax.set_title("데이터셋별 최종 선택 피처 (1=선택, 0=미선택)", fontsize=12)
ax.set_xlabel("데이터셋"); ax.set_ylabel("AEC Feature")
plt.tight_layout()
fig.savefig(RESULT_DIR / "cross_dataset_feature_heatmap.png", dpi=150)
plt.close()
print(f"  Saved: cross_dataset_feature_heatmap.png")

# Excel 저장
feat_list_rows = []
for res in pipeline_results.values():
    for f in sorted(res["final_feats"]):
        feat_list_rows.append({"dataset": res["label"], "feature": f})

with pd.ExcelWriter(RESULT_DIR / "cross_dataset_comparison.xlsx") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    heatmap_data.reset_index().rename(columns={"index": "feature"}).to_excel(
        writer, sheet_name="feature_matrix", index=False)
    pd.DataFrame(feat_list_rows).to_excel(writer, sheet_name="feature_list", index=False)

print(f"  Saved: cross_dataset_comparison.xlsx")

# regression_analysis.py 호환: merged 결과를 최상위 feature_selection_summary.xlsx에 복사
# (merged가 없으면 gangnam → sinchon 순으로 대체)
primary_key = "merged" if "merged" in pipeline_results else (
    "gangnam" if "gangnam" in pipeline_results else "sinchon")
primary_src = datasets[primary_key][2]   # label
import shutil
src_xlsx = RESULT_DIR / primary_key / "feature_selection_summary.xlsx"
dst_xlsx = RESULT_DIR / "feature_selection_summary.xlsx"
shutil.copy2(src_xlsx, dst_xlsx)
print(f"\n  regression_analysis.py용 feature_selection_summary.xlsx")
print(f"  → '{primary_src}' 결과 복사 완료: {dst_xlsx}")

print(f"\n{'='*60}")
print("ALL DONE")
for res in pipeline_results.values():
    print(f"  [{res['label']:12s}] n={res['n']:4d} | n_feats={len(res['final_feats']):2d} "
          f"| pipeline CV R²={res['best_r2']:.4f} | prev CV R²={res['prev_r2']:.4f} "
          f"| Δ={res['best_r2']-res['prev_r2']:+.4f}")
    print(f"              features: {sorted(res['final_feats'])}")
print(f"{'='*60}")
