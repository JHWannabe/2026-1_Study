"""
Regression Analysis: Linear & Logistic Regression on TAMA
Target  : TAMA  (Linear = continuous | Logistic = binary below lower 25th percentile)
Cases   :
  Case 1 - Clinical baseline  : PatientAge, (PatientSex), BMI
  Case 2 - + AEC_prev         : Case 1 + [mean, CV, skewness, slope_abs_mean]
  Case 3 - + AEC_new          : Case 1 + pipeline-selected features
  Case 4 - + AEC_prev+Scanner : Case 2 + ManufacturerModelName, kVp
  Case 5 - + AEC_new+Scanner  : Case 3 + ManufacturerModelName, kVp
Hospitals : 강남, 신촌
Sex groups: 전체, 여성(F), 남성(M)  — sex-stratified groups drop PatientSex_enc

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[0424 → 0430 연구 설계 변경 요약]

1. BMI 공변량 추가 (핵심 변경)
   - 0424: 임상 기준선 = PatientAge + PatientSex
   - 0430: 임상 기준선 = PatientAge + PatientSex + BMI
   → BMI는 체지방량·제지방량과 직접 연관된 교란변수로, 보정 없이는 AEC
     피처의 독립적 예측력이 과대 추정될 수 있음. 이를 통제하여 AEC의
     순수 기여도를 분리.

2. Case 3→5단계 체계로 확장 (AEC_prev vs AEC_new 이중 비교)
   - 0424: Case 0(AEC만)/1(임상)/2(임상+AEC)/3(임상+AEC+스캐너) — 단일 AEC 세트
   - 0430: Case 1~5 — AEC_prev(4개 고정 피처) vs AEC_new(파이프라인 자동 선택)
     를 스캐너 유무와 조합하여 교차 비교
   → "어떤 AEC 피처 세트가 더 유용한가"를 직접 검정하는 비교 설계.

3. 성별 층화 분석 신설 (전체 / 여성 / 남성)
   - 0424: 성별을 공변량(Sex=M/F 더미)으로만 처리, 층화 없음
   - 0430: 전체·여성·남성 3개 서브그룹 각각에 독립 모델 적합
   → 근감소 위험도와 AEC 신호 특성이 성별에 따라 다를 수 있으므로,
     교호작용(interaction) 대신 층화 전략으로 이질성 탐색.
     층화 그룹에서는 PatientSex를 예측변수에서 제거(내재적 보정).

4. 다병원 자동 순회 + 교차 병원 비교 추가
   - 0424: config.py의 SITE 변수로 한 번에 한 병원만 분석
   - 0430: data/ 폴더를 자동 스캔하여 강남·신촌 모두 순회,
     Cross-hospital comparison 섹션에서 재현성(generalizability) 비교.

5. 이진화 임계값 단순화
   - 0424: 성별 특이적 임계값 (남성 P25, 여성 P25 별도 산출)
   - 0430: 분석 그룹(전체·여성·남성) 내 하위 25%를 동적으로 산출
   → 층화 분석과 일관성을 유지하기 위해 그룹-내 P25로 단순화.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
warnings.filterwarnings("ignore")

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score,
)
from sklearn.pipeline import Pipeline
from scipy import stats as scipy_stats
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as sm_vif
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from matplotlib.patches import Patch

# ────────────────────────────────────────────────
# 0. Global config
# ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
FS_RESULT  = SCRIPT_DIR.parent / "results" / "feature_selection"

CV_SPLITS = 5
CV_RANDOM = 42

# AEC_PREV: 0424부터 사용해 온 4개의 고정 피처 세트.
# 진폭(mean), 변동성(CV), 비대칭도(skewness), 평균 절대 기울기(slope_abs_mean)로
# AEC 신호의 수준·변동·형태·추세를 요약.
AEC_PREV = ["mean", "CV", "skewness", "slope_abs_mean"]

# AEC_NEW: feature_selection 파이프라인(상관계수·MI·순열 중요도·CV-R²)이
# 자동 선택한 피처 세트. 0430에서 신규 도입.
# AEC_PREV와의 성능 차이를 Case 2 vs Case 3 (Case 4 vs Case 5)로 비교하여
# 데이터 기반 피처 선택의 효과를 정량화하는 것이 목적.
fs_xlsx = FS_RESULT / "feature_selection_summary.xlsx"
if fs_xlsx.exists():
    AEC_NEW = pd.read_excel(str(fs_xlsx), sheet_name="final_features")["final_features"].tolist()
    print(f"[AEC-new] Loaded {len(AEC_NEW)} features from pipeline: {AEC_NEW}")
else:
    # 파이프라인 결과 파일이 없을 때 사용하는 대체값 (주파수·웨이블릿 기반 피처 포함)
    AEC_NEW = [
        "IQR", "band2_energy", "dominant_freq", "mean", "slope_max",
        "spectral_energy", "wavelet_cD1_energy", "wavelet_cD2_energy",
        "wavelet_energy_ratio_D1",
    ]
    print(f"[AEC-new] Fallback hardcoded {len(AEC_NEW)} features")

print(f"[AEC-prev] {len(AEC_PREV)} features: {AEC_PREV}")

HOSPITALS = {}
for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".xlsx") or "merged_features" not in fname:
        continue
    if "강남" in fname:
        HOSPITALS["gangnam"] = DATA_DIR / fname
    elif "신촌" in fname:
        HOSPITALS["sinchon"] = DATA_DIR / fname

print(f"[Data] Found hospitals: {list(HOSPITALS.keys())}")

CASE_LABELS = {
    "Case1_Clinical":                  "Case 1\nClinical",
    "Case2_Clinical+AEC_prev":         "Case 2\n+AEC_prev",
    "Case3_Clinical+AEC_new":          "Case 3\n+AEC_new",
    "Case4_Clinical+AEC_prev+Scanner": "Case 4\n+AEC_prev\n+Scanner",
    "Case5_Clinical+AEC_new+Scanner":  "Case 5\n+AEC_new\n+Scanner",
}
COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#1a5276", "#922b21"]

# ────────────────────────────────────────────────
# 1. Helpers
# ────────────────────────────────────────────────
def copy_to_temp(src: Path, temp_name: str) -> Path:
    dst = Path(os.environ["TEMP"]) / temp_name
    subprocess.run(
        ["powershell", "-Command",
         f'Copy-Item -Path "{src}" -Destination "{dst}" -Force'],
        capture_output=True,
    )
    return dst

def load_hospital(src: Path, temp_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = copy_to_temp(src, temp_name)
    feat = pd.read_excel(str(tmp), sheet_name="features")
    meta = pd.read_excel(str(tmp), sheet_name="metadata-bmi_add")
    return feat, meta

def linear_cv(X: pd.DataFrame, y: pd.Series) -> dict:
    # 5-Fold CV로 선형 회귀 일반화 성능 추정.
    # Pipeline 내부에서 fold별로 StandardScaler를 재적합하여 data leakage 방지.
    # out-of-fold 예측값(oof_true/oof_pred)을 수집해 실제 vs 예측 scatter plot에 사용.
    kf   = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM)
    pipe = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    r2s, maes, rmses, all_true, all_pred = [], [], [], [], []
    for tr, te in kf.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], pred))
        maes.append(mean_absolute_error(y.iloc[te], pred))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
        all_true.extend(y.iloc[te].tolist())
        all_pred.extend(pred.tolist())
    return dict(R2=np.mean(r2s), R2_std=np.std(r2s),
                MAE=np.mean(maes), MAE_std=np.std(maes),
                RMSE=np.mean(rmses), RMSE_std=np.std(rmses),
                fold_r2=r2s, oof_true=all_true, oof_pred=all_pred)

def logistic_cv(X: pd.DataFrame, y: pd.Series) -> dict:
    # StratifiedKFold: 이진 결과(low TAMA = 1)의 클래스 불균형이 fold마다
    # 같은 비율로 유지되도록 층화 샘플링. 소표본 성별 그룹에서 특히 중요.
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM)
    aucs, accs, sens, specs, fprs_l, tprs_l = [], [], [], [], [], []
    for tr, te in skf.split(X, y):
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=2000, random_state=CV_RANDOM, solver="lbfgs")),
        ])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        pred = pipe.predict(X.iloc[te])
        aucs.append(roc_auc_score(y.iloc[te], prob))
        accs.append(accuracy_score(y.iloc[te], pred))
        tn, fp, fn, tp = confusion_matrix(y.iloc[te], pred).ravel()
        sens.append(tp / (tp + fn) if tp + fn else 0)
        specs.append(tn / (tn + fp) if tn + fp else 0)
        fpr, tpr, _ = roc_curve(y.iloc[te], prob)
        fprs_l.append(fpr); tprs_l.append(tpr)
    return dict(AUC=np.mean(aucs), AUC_std=np.std(aucs),
                Accuracy=np.mean(accs), Accuracy_std=np.std(accs),
                Sensitivity=np.mean(sens), Sensitivity_std=np.std(sens),
                Specificity=np.mean(specs), Specificity_std=np.std(specs),
                fold_auc=aucs, fprs=fprs_l, tprs=tprs_l)

def make_cases(clinical_feats: list, scanner_feats: list) -> dict:
    # 0424의 3단계(임상 / 임상+AEC / 임상+AEC+스캐너)에서
    # 0430의 5단계 구조로 확장.
    # AEC_prev(전통적 4피처)와 AEC_new(파이프라인 선택)를 나란히 비교하여
    # AEC 피처 선택 전략의 우열을 Case 2 vs 3, Case 4 vs 5 쌍으로 검정.
    # clinical_feats: 전체 분석 시 [Age, Sex, BMI], 층화 분석 시 [Age, BMI] (Sex 제거)
    return {
        "Case1_Clinical":                  clinical_feats,
        "Case2_Clinical+AEC_prev":         clinical_feats + AEC_PREV,
        "Case3_Clinical+AEC_new":          clinical_feats + AEC_NEW,
        "Case4_Clinical+AEC_prev+Scanner": clinical_feats + AEC_PREV + scanner_feats,
        "Case5_Clinical+AEC_new+Scanner":  clinical_feats + AEC_NEW  + scanner_feats,
    }

def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def run_fullfit_analysis(X_full: pd.DataFrame, y_cont: pd.Series,
                         CASES: dict, OUT_DIR: Path, hosp_label: str) -> None:
    """
    전체 데이터로 OLS/Logit 적합 후 진단 플롯·케이스 비교 생성 (Figs 04-15).

    CV 기반 run_one_analysis()가 일반화 성능(out-of-fold)을 측정하는 것과 달리,
    이 함수는 전체 데이터에 한 번 적합(full-fit)한 결과를 사용하여:
      - 잔차 진단 (정규성·등분산성·자기상관): 모델 가정 충족 여부 확인
      - 보정도 플롯 (Hosmer-Lemeshow): 로지스틱 모델의 확률 보정 수준 시각화
      - 케이스별 R²/AUC/AIC 추이: 변수군 추가에 따른 단조 증가 여부 검증
    대표 진단 모델은 Case4(임상+AEC_prev+스캐너)를 사용 — 변수 구성이 가장 풍부함.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 이진화: 그룹 내 하위 25%를 low-TAMA(=1)로 정의.
    # 0424는 성별 특이적 임계값(남성·여성 각각 P25)을 사용했으나,
    # 0430 전체(all) 분석에서는 그룹 통합 P25를 사용하여 이진화 일관성 확보.
    tama_threshold = y_cont.quantile(0.25)
    y_bin = (y_cont < tama_threshold).astype(int)

    def avail(feats):
        return [f for f in feats if f in X_full.columns]

    def safe_std(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize, dropping zero-variance columns to avoid inf/NaN."""
        std = df.std()
        good = std[std > 1e-10].index.tolist()
        if not good:
            return pd.DataFrame(index=df.index)
        Xs = (df[good] - df[good].mean()) / df[good].std()
        return Xs.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Use Case4 (Clinical+AEC_prev+Scanner) for single-model diagnostics
    diag_key = next((k for k in ["Case4_Clinical+AEC_prev+Scanner", "Case2_Clinical+AEC_prev"]
                     if k in CASES), list(CASES.keys())[-1])
    diag_feats = avail(CASES[diag_key])
    X_diag     = X_full[diag_feats].copy()
    X_diag_std = safe_std(X_diag)

    # ── Fig 04: Linear actual vs predicted (full fit OLS) ──
    X_ols      = sm.add_constant(X_diag_std)
    ols_model  = sm.OLS(y_cont, X_ols).fit()
    fitted     = ols_model.fittedvalues
    residuals  = ols_model.resid
    r2_full    = ols_model.rsquared
    rmse_full  = float(np.sqrt(np.mean(residuals ** 2)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_cont, fitted, alpha=0.3, s=8, color="#e07070", label="환자", rasterized=True)
    lo_v = min(float(y_cont.min()), float(fitted.min())) - 5
    hi_v = max(float(y_cont.max()), float(fitted.max())) + 5
    ax.plot([lo_v, hi_v], [lo_v, hi_v], "k--", lw=1.5, label="완벽 예측 (y=x)")
    z_v  = np.polyfit(y_cont, fitted, 1)
    xln  = np.linspace(lo_v, hi_v, 200)
    ax.plot(xln, np.poly1d(z_v)(xln), color="#3498db", lw=2, label="회귀 직선")
    ax.set_xlabel("실제 TAMA (cm²)"); ax.set_ylabel("예측 TAMA (cm²)")
    ax.set_title(f"선형 회귀: 실제 vs 예측 TAMA\nR²={r2_full:.3f}, RMSE={rmse_full:.2f} cm²")
    ax.legend(fontsize=9)
    plt.tight_layout(); fig.savefig(OUT_DIR / "04_linear_actual_vs_pred.png", dpi=150); plt.close()

    # ── Fig 05: Residual diagnostics (4-Panel) ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.2, s=6, color="#e07070", rasterized=True)
    ax.axhline(0, color="gray", lw=0.8)
    lf1 = sm_lowess(residuals.values, fitted.values, frac=0.3)
    ax.plot(lf1[:, 0], lf1[:, 1], color="#3498db", lw=2)
    ax.set_xlabel("Fitted Values"); ax.set_ylabel("Residuals"); ax.set_title("(a) Residuals vs Fitted")

    ax = axes[0, 1]
    (osm, osr), (sl_qq, ic_qq, r_qq) = scipy_stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, alpha=0.3, s=6, color="#e07070", rasterized=True)
    lo_qq = float(min(osm)); hi_qq = float(max(osm))
    ax.plot([lo_qq, hi_qq], [sl_qq * lo_qq + ic_qq, sl_qq * hi_qq + ic_qq],
            "r--", lw=1.5, label=f"r={r_qq:.3f}")
    ax.set_xlabel("이론적 분위수 (Normal Quantiles)"); ax.set_ylabel("실제 분위수 (Sample Quantiles)")
    ax.set_title("(b) Normal Q-Q Plot"); ax.legend(fontsize=9)

    ax = axes[1, 0]
    sqrt_res = np.sqrt(np.abs(residuals / residuals.std()))
    ax.scatter(fitted, sqrt_res, alpha=0.2, s=6, color="#e08040", rasterized=True)
    lf2 = sm_lowess(sqrt_res.values, fitted.values, frac=0.3)
    ax.plot(lf2[:, 0], lf2[:, 1], color="#3498db", lw=2)
    ax.set_xlabel("Fitted Values"); ax.set_ylabel("|Standardized Residuals|^0.5")
    ax.set_title("(c) Scale-Location (등분산성 확인)")

    ax = axes[1, 1]
    resid_arr = residuals.values
    sample    = resid_arr[:5000] if len(resid_arr) > 5000 else resid_arr
    W, p_sw   = shapiro(sample)
    ax.hist(resid_arr, bins=40, density=True, color="#7fb3d6", edgecolor="white", linewidth=0.3)
    xr = np.linspace(resid_arr.min(), resid_arr.max(), 200)
    ax.plot(xr, scipy_stats.norm.pdf(xr, resid_arr.mean(), resid_arr.std()),
            "r-", lw=2, label="정규분포 곡선")
    ax.set_xlabel("Residuals"); ax.set_ylabel("밀도")
    ax.set_title(f"(d) 잔차 분포\nShapiro-Wilk: W={W:.4f}, p={p_sw:.2e}")
    ax.legend(fontsize=8)

    plt.suptitle(f"[{hosp_label}] 선형 회귀 잔차 진단 (4-Panel Diagnostic)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(OUT_DIR / "05_linear_residuals.png", dpi=150); plt.close()

    # ── Fig 06: Linear forest plot (p < 0.05 only) ──
    ci_ols = ols_model.conf_int()
    coef_df = pd.DataFrame({
        "feature": list(X_diag_std.columns),
        "coef":    ols_model.params.iloc[1:].values,
        "ci_lo":   ci_ols.iloc[1:, 0].values,
        "ci_hi":   ci_ols.iloc[1:, 1].values,
        "pval":    ols_model.pvalues.iloc[1:].values,
    })
    sig_df = coef_df[coef_df["pval"] < 0.05].sort_values("coef").reset_index(drop=True)
    if len(sig_df) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(sig_df) * 0.7 + 1.5)))
        fp_colors = ["#e07070" if c > 0 else "#7090d0" for c in sig_df["coef"]]
        ax.barh(range(len(sig_df)), sig_df["coef"].values,
                xerr=[sig_df["coef"].values - sig_df["ci_lo"].values,
                      sig_df["ci_hi"].values - sig_df["coef"].values],
                color=fp_colors, capsize=4, edgecolor="white", height=0.6)
        ax.set_yticks(range(len(sig_df))); ax.set_yticklabels(sig_df["feature"].values)
        ax.axvline(0, color="gray", lw=1.5, ls="--")
        span = float(sig_df["ci_hi"].max() - sig_df["ci_lo"].min())
        for i, row in sig_df.iterrows():
            ax.text(row["ci_hi"] + span * 0.02, i,
                    sig_stars(row["pval"]), va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("β 계수 (95% CI, 표준화 단위)")
        ax.set_title("선형 회귀 - 유의한 계수 Forest Plot\n(p<0.05, TAMA cm² 변화량 per 1 SD)")
        plt.tight_layout(); fig.savefig(OUT_DIR / "06_linear_forest.png", dpi=150); plt.close()

    # ── Fig 07: Linear univariate R² per feature ──
    uni_feats = avail(["PatientAge", "PatientSex_enc", "BMI"] + AEC_PREV)
    uni_rows  = []
    for f in uni_feats:
        xf = safe_std(X_full[[f]])
        if xf.empty:
            continue
        m_uni = sm.OLS(y_cont, sm.add_constant(xf)).fit()
        uni_rows.append({"feature": f, "r2": m_uni.rsquared, "pval": m_uni.pvalues.iloc[1]})
    uni_df = pd.DataFrame(uni_rows).sort_values("r2")
    lbl_map07 = {"PatientSex_enc": "Sex (M=1, F=0)", "PatientAge": "Age (표준화)", "BMI": "BMI (표준화)"}
    uni_df["label"] = uni_df["feature"].apply(lambda f: lbl_map07.get(f, f"AEC: {f} (표준화)"))

    fig, ax = plt.subplots(figsize=(9, max(4, len(uni_df) * 0.65 + 1)))
    cols_uni = ["#e07070" if p < 0.05 else "#aaaaaa" for p in uni_df["pval"]]
    ax.barh(uni_df["label"], uni_df["r2"], color=cols_uni, height=0.6)
    for i, (_, row) in enumerate(uni_df.iterrows()):
        ax.text(row["r2"] + 0.002, i, f"{row['r2']:.3f}", va="center", fontsize=9)
    ax.legend(handles=[Patch(color="#e07070", label="p<0.05 유의"),
                       Patch(color="#aaaaaa", label="p≥0.05 비유의")], fontsize=9)
    ax.set_xlabel("단변량 R² (TAMA 설명 분산 비율)")
    ax.set_title("선형 회귀 - 단변량 R² 비교\n(빨간색: p<0.05 유의)")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    plt.tight_layout(); fig.savefig(OUT_DIR / "07_linear_univariate_r2.png", dpi=150); plt.close()

    # ── Fig 08: Logistic ROC with bootstrap CI (n=1000) ──
    pipe_log_full = Pipeline([
        ("sc", StandardScaler()),
        ("m",  LogisticRegression(max_iter=2000, random_state=CV_RANDOM, solver="lbfgs")),
    ])
    pipe_log_full.fit(X_diag, y_bin)
    prob_full = pipe_log_full.predict_proba(X_diag)[:, 1]
    base_auc  = roc_auc_score(y_bin, prob_full)
    fpr_base, tpr_base, thr_roc = roc_curve(y_bin, prob_full)

    # Bootstrap AUC 95%CI (n=1000): 단일 전체-fit AUC의 불확실성을 정량화.
    # 복원 추출로 ROC 곡선을 1000회 반복하여 2.5th/97.5th percentile을 신뢰구간으로 사용.
    rng_bt   = np.random.RandomState(42)
    n_boot   = 1000
    boot_aucs, tprs_bt = [], []
    mfpr_bt  = np.linspace(0, 1, 200)
    for _ in range(n_boot):
        idx_bt = rng_bt.randint(0, len(y_bin), len(y_bin))
        if y_bin.iloc[idx_bt].nunique() < 2:
            continue
        boot_aucs.append(roc_auc_score(y_bin.iloc[idx_bt], prob_full[idx_bt]))
        fb, tb, _ = roc_curve(y_bin.iloc[idx_bt], prob_full[idx_bt])
        tprs_bt.append(np.interp(mfpr_bt, fb, tb))
    ci_lo_a = float(np.percentile(boot_aucs, 2.5))
    ci_hi_a = float(np.percentile(boot_aucs, 97.5))
    mean_tpr_bt = np.mean(tprs_bt, axis=0)
    std_tpr_bt  = np.std(tprs_bt, axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_base, tpr_base, color="#e07070", lw=2,
            label=f"AUC = {base_auc:.3f} (95%CI: {ci_lo_a:.3f}-{ci_hi_a:.3f})")
    ax.fill_between(mfpr_bt, mean_tpr_bt - std_tpr_bt, mean_tpr_bt + std_tpr_bt,
                    alpha=0.2, color="#e07070")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="무작위 (AUC=0.5)")
    ax.set_xlabel("1 - Specificity (FPR)"); ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(f"로지스틱 회귀 - ROC 곡선 (전체 모델)\nBootstrap 95%CI (n={n_boot})")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(OUT_DIR / "08_logistic_roc.png", dpi=150); plt.close()

    # ── Fig 09: Calibration plot (Hosmer-Lemeshow) ──
    # 예측 확률을 10 십분위수로 분할하여 관측 비율과 비교.
    # HL χ² p>0.05 → 모델이 관측값에 잘 보정됨(calibrated).
    # 판별력(AUC)이 높아도 보정도가 낮으면 개별 확률 예측이 부정확하므로
    # 임상 적용 가능성 평가 시 AUC와 함께 반드시 제시해야 함.
    n_bins_hl = 10
    df_cal = pd.DataFrame({"prob": prob_full, "y": y_bin.values})
    df_cal["decile"] = pd.qcut(df_cal["prob"], n_bins_hl, duplicates="drop")
    grps     = df_cal.groupby("decile", observed=True)
    mean_pd  = grps["prob"].mean()
    obs_prop = grps["y"].mean()
    n_g      = grps["y"].count()
    obs_sum  = grps["y"].sum()
    exp_sum  = grps["prob"].sum()
    chi2_hl  = ((obs_sum - exp_sum) ** 2 / (exp_sum * (1 - exp_sum / n_g))).sum()
    p_hl     = 1 - scipy_stats.chi2.cdf(chi2_hl, df=max(len(grps) - 2, 1))
    cal_status = "보정 양호" if p_hl >= 0.05 else "보정 불량"

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(mean_pd, obs_prop, s=n_g * 0.5, color="#e08040", zorder=5, label="관측 비율")
    z_cal = np.polyfit(mean_pd, obs_prop, 1)
    xc    = np.linspace(0, float(mean_pd.max()) + 0.05, 100)
    ax.plot(xc, np.poly1d(z_cal)(xc), color="#e07070", lw=2, label="회귀 직선")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="완벽 보정 (y=x)")
    ax.set_xlabel("예측 확률 (Predicted Probability)")
    ax.set_ylabel("관측 비율 (Observed Proportion)")
    ax.set_title(f"Calibration Plot (Hosmer-Lemeshow)\nHL χ²={chi2_hl:.2f}, p={p_hl:.4f} ({cal_status})")
    ax.text(0.05, 0.9, "원 크기 = 그룹 내 환자 수", transform=ax.transAxes, fontsize=9, color="gray")
    ax.legend(fontsize=9); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(OUT_DIR / "09_logistic_calibration.png", dpi=150); plt.close()

    # ── Fig 10: Confusion matrix (Youden optimal threshold) ──
    youden_j = tpr_base - (1 - fpr_base)
    opt_idx  = int(np.argmax(youden_j))
    opt_thr  = float(thr_roc[opt_idx])
    pred_opt = (prob_full >= opt_thr).astype(int)
    cm_mat   = confusion_matrix(y_bin, pred_opt)
    n_tot    = len(y_bin)
    tn, fp_v, fn, tp_v = cm_mat.ravel()
    sens_v = tp_v / (tp_v + fn)   if tp_v + fn   else 0
    spec_v = tn   / (tn + fp_v)   if tn + fp_v   else 0
    ppv    = tp_v / (tp_v + fp_v) if tp_v + fp_v else 0
    npv    = tn   / (tn + fn)     if tn + fn     else 0
    cm_ann = np.array([[f"{tn}\n({tn/n_tot*100:.1f}%)",   f"{fp_v}\n({fp_v/n_tot*100:.1f}%)"],
                        [f"{fn}\n({fn/n_tot*100:.1f}%)",   f"{tp_v}\n({tp_v/n_tot*100:.1f}%)"]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_mat, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm_ann[i, j], ha="center", va="center",
                    color="white" if cm_mat[i, j] > cm_mat.max() * 0.5 else "black",
                    fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["예측: Normal", "예측: Low TAMA"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["실제: Normal", "실제: Low TAMA"])
    ax.set_xlabel("예측 클래스"); ax.set_ylabel("실제 클래스")
    ax.set_title(f"Confusion Matrix (임계값={opt_thr:.3f})\n"
                 f"Sensitivity={sens_v:.3f}  Specificity={spec_v:.3f}\n"
                 f"PPV={ppv:.3f}  NPV={npv:.3f}")
    plt.tight_layout(); fig.savefig(OUT_DIR / "10_logistic_confusion.png", dpi=150); plt.close()

    # ── Fig 11: Logistic crude OR forest plot ──
    or_feats   = avail(["PatientAge", "PatientSex_enc", "BMI"] + AEC_PREV)
    lbl_map11  = {"PatientAge": "Age (표준화)", "PatientSex_enc": "Sex (M=1, F=0)", "BMI": "BMI (표준화)"}
    or_rows    = []
    for f in or_feats:
        xf_std = safe_std(X_full[[f]])
        if xf_std.empty:
            continue
        try:
            log_uni = sm.Logit(y_bin, sm.add_constant(xf_std)).fit(disp=False, maxiter=200)
            ci_uni  = log_uni.conf_int()
            or_rows.append({
                "label": lbl_map11.get(f, f"AEC: {f} (표준화)"),
                "OR":    float(np.exp(log_uni.params.iloc[1])),
                "ci_lo": float(np.exp(ci_uni.iloc[1, 0])),
                "ci_hi": float(np.exp(ci_uni.iloc[1, 1])),
                "pval":  float(log_uni.pvalues.iloc[1]),
            })
        except Exception:
            pass
    if or_rows:
        or_df = pd.DataFrame(or_rows).sort_values("OR", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9, max(4, len(or_df) * 0.75 + 1.5)))
        for i, row in or_df.iterrows():
            color = "#e07070" if row["OR"] >= 1 else "#7090d0"
            ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=color, lw=2)
            ax.scatter(row["OR"], i, color=color, s=80, zorder=5)
            ax.text(row["ci_hi"] * 1.08, i, sig_stars(row["pval"]), va="center", fontsize=9)
        ax.axvline(1, color="black", lw=1.5, ls="--", label="OR=1 (귀무)")
        ax.set_yticks(range(len(or_df))); ax.set_yticklabels(or_df["label"])
        ax.set_xscale("log"); ax.set_xlabel("Crude OR (95% CI, log scale)")
        ax.set_title("로지스틱 회귀 - 단변량 Crude OR Forest Plot\n(Low TAMA 위험 오즈비)")
        ax.legend(fontsize=9)
        plt.tight_layout(); fig.savefig(OUT_DIR / "11_logistic_forest.png", dpi=150); plt.close()

    # ── Figs 12-15: Case comparison (statsmodels full fit) ──
    lin_met, log_met = {}, {}
    for case_name, feats in CASES.items():
        fa = avail(feats)
        if not fa:
            continue
        X_c     = X_full[fa]
        X_c_std = safe_std(X_c)
        if X_c_std.empty:
            continue
        ols_c   = sm.OLS(y_cont, sm.add_constant(X_c_std)).fit()
        resid_c = ols_c.resid
        lin_met[case_name] = {
            "R2": ols_c.rsquared, "Adj_R2": ols_c.rsquared_adj,
            "RMSE": float(np.sqrt(np.mean(resid_c ** 2))),
            "AIC": ols_c.aic, "BIC": ols_c.bic,
        }
        try:
            log_c  = sm.Logit(y_bin, sm.add_constant(X_c_std)).fit(disp=False, maxiter=300)
            prob_c = log_c.predict(sm.add_constant(X_c_std))
            auc_c  = roc_auc_score(y_bin, prob_c)
            rng_c  = np.random.RandomState(42)
            boot_c = []
            for _ in range(500):
                idx_c = rng_c.randint(0, len(y_bin), len(y_bin))
                if y_bin.iloc[idx_c].nunique() > 1:
                    boot_c.append(roc_auc_score(y_bin.iloc[idx_c], prob_c.values[idx_c]))
            n_obs  = len(y_bin)
            ll_f   = log_c.llf
            ll_n   = sm.Logit(y_bin, np.ones(n_obs)).fit(disp=False, maxiter=200).llf
            cs     = 1 - np.exp((2 / n_obs) * (ll_n - ll_f))
            nag    = cs / (1 - np.exp((2 / n_obs) * ll_n))
            log_met[case_name] = {
                "AUC": auc_c,
                "AUC_ci_lo": float(np.percentile(boot_c, 2.5))  if boot_c else auc_c,
                "AUC_ci_hi": float(np.percentile(boot_c, 97.5)) if boot_c else auc_c,
                "Nagelkerke": float(nag),
                "AIC": log_c.aic, "BIC": log_c.bic,
            }
        except Exception as e:
            print(f"      [{case_name}] logistic fit failed: {e}")

    case_keys  = list(lin_met.keys())
    case_short = [CASE_LABELS.get(k, k).replace("\n", " ") for k in case_keys]
    x_pos      = np.arange(len(case_keys))

    # Fig 12: Linear R² / RMSE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    r2_v   = [lin_met[c]["R2"]     for c in case_keys]
    adj_v  = [lin_met[c]["Adj_R2"] for c in case_keys]
    w12    = 0.35
    b1 = ax.bar(x_pos - w12/2, r2_v,  w12, label="R²",         color="#5b9bd5", edgecolor="white")
    b2 = ax.bar(x_pos + w12/2, adj_v, w12, label="Adjusted R²", color="#ed7d31", edgecolor="white")
    for bar, v in list(zip(b1, r2_v)) + list(zip(b2, adj_v)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos); ax.set_xticklabels(case_short, fontsize=8)
    ax.set_ylabel("R² 값"); ax.set_title("선형 회귀 R² 비교\n(Case 1 → 5)")
    ax.legend(fontsize=9); ax.set_ylim(0, max(r2_v) * 1.2)
    ax = axes[1]
    rmse_v   = [lin_met[c]["RMSE"] for c in case_keys]
    min_rmse = min(rmse_v)
    bars12 = ax.bar(x_pos, rmse_v,
                    color=["#e74c3c" if v == min_rmse else "#7fbbdf" for v in rmse_v],
                    edgecolor="white", width=0.5)
    for bar, v in zip(bars12, rmse_v):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_pos); ax.set_xticklabels(case_short, fontsize=8)
    ax.set_ylabel("RMSE (cm²)"); ax.set_title("선형 회귀 RMSE 비교\n(낮을수록 예측 정확)")
    ax.set_ylim(0, max(rmse_v) * 1.2)
    plt.suptitle(f"[{hosp_label}] Multivariable Analysis - 선형 회귀 성능",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(OUT_DIR / "12_case_metrics_bar.png", dpi=150); plt.close()

    # Fig 13: Logistic AUC + Nagelkerke R²
    if log_met:
        log_keys  = list(log_met.keys())
        log_short = [CASE_LABELS.get(k, k).replace("\n", " ") for k in log_keys]
        x_log     = np.arange(len(log_keys))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        auc_v  = [log_met[c]["AUC"]       for c in log_keys]
        auc_lo = [log_met[c]["AUC_ci_lo"] for c in log_keys]
        auc_hi = [log_met[c]["AUC_ci_hi"] for c in log_keys]
        bars13 = ax.bar(x_log, auc_v, color=COLORS[:len(log_keys)], edgecolor="white", width=0.5)
        ax.errorbar(x_log, auc_v,
                    yerr=[[v - lo for v, lo in zip(auc_v, auc_lo)],
                          [hi - v for hi, v in zip(auc_hi, auc_v)]],
                    fmt="none", color="black", capsize=5, lw=1.5)
        for bar, v in zip(bars13, auc_v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.axhline(0.5, color="gray",    ls="--", lw=1, label="무작위 기준 (0.5)")
        ax.axhline(0.7, color="#f39c12", ls=":",  lw=1, label="양호 기준 (0.7)")
        ax.axhline(0.8, color="#27ae60", ls=":",  lw=1, label="우수 기준 (0.8)")
        ax.set_xticks(x_log); ax.set_xticklabels(log_short, fontsize=8)
        ax.set_ylabel("AUC-ROC"); ax.set_ylim(0, 1)
        ax.set_title("로지스틱 AUC 비교 (Bootstrap 95%CI)\nCase 1 → 5"); ax.legend(fontsize=8)
        ax = axes[1]
        nag_v  = [log_met[c]["Nagelkerke"] for c in log_keys]
        bars13b = ax.bar(x_log, nag_v, color=COLORS[:len(log_keys)], edgecolor="white", width=0.5)
        for bar, v in zip(bars13b, nag_v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x_log); ax.set_xticklabels(log_short, fontsize=8)
        ax.set_ylabel("Nagelkerke R²"); ax.set_title("로지스틱 Nagelkerke R²\n(설명력 비교)")
        plt.suptitle(f"[{hosp_label}] Multivariable Analysis - 로지스틱 회귀 성능",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(); fig.savefig(OUT_DIR / "13_case_auc_bar.png", dpi=150); plt.close()

    # Fig 14: AIC/BIC comparison
    if lin_met and log_met:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, met_d, title14 in zip(
            axes,
            [lin_met, log_met],
            ["선형 회귀 AIC/BIC\n(낮을수록 모델 적합도 우수)",
             "로지스틱 회귀 AIC/BIC\n(낮을수록 모델 적합도 우수)"],
        ):
            ck14   = list(met_d.keys())
            cs14   = [CASE_LABELS.get(c, c).replace("\n", " ") for c in ck14]
            x_l14  = np.arange(len(ck14))
            aic_14 = [met_d[c]["AIC"] for c in ck14]
            bic_14 = [met_d[c]["BIC"] for c in ck14]
            w14    = 0.35
            ax.bar(x_l14 - w14/2, aic_14, w14, label="AIC", color="#5b9bd5", edgecolor="white")
            ax.bar(x_l14 + w14/2, bic_14, w14, label="BIC", color="#e07070", edgecolor="white")
            top_y14 = max(max(aic_14), max(bic_14))
            for i in range(1, len(ck14)):
                delta14 = aic_14[i] - aic_14[0]
                ax.annotate(f"ΔAIC={delta14:+.0f}",
                            xy=(x_l14[i], max(aic_14[i], bic_14[i])),
                            xytext=(x_l14[i], top_y14 * 1.03),
                            fontsize=8, color="#555555", ha="center",
                            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8))
            ax.set_xticks(x_l14); ax.set_xticklabels(cs14, fontsize=8)
            ax.set_ylabel("AIC / BIC 값"); ax.set_title(title14)
            ax.legend(fontsize=9); ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
        plt.suptitle(f"[{hosp_label}] Multivariable Analysis - AIC/BIC 모델 비교",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(); fig.savefig(OUT_DIR / "14_case_aic_bar.png", dpi=150); plt.close()

    # Fig 15: Case progression line plot
    common15 = [c for c in lin_met if c in log_met]
    if len(common15) >= 2:
        c_lbl15 = [CASE_LABELS.get(c, c).replace("\n", " ") for c in common15]
        x15     = np.arange(len(common15))
        metrics15 = [
            ([lin_met[c]["R2"]         for c in common15], "Linear R² 추이",     "#2196F3"),
            ([lin_met[c]["Adj_R2"]     for c in common15], "Linear Adj R² 추이", "#FF9800"),
            ([log_met[c]["AUC"]        for c in common15], "Logistic AUC 추이",  "#F44336"),
            ([log_met[c]["Nagelkerke"] for c in common15], "Nagelkerke R² 추이", "#4CAF50"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, (vals, title15, color) in zip(axes.flat, metrics15):
            ax.plot(x15, vals, "o-", color=color, lw=2, ms=8)
            y_span = max(vals) - min(vals) if max(vals) != min(vals) else 0.05
            for i, v in enumerate(vals):
                ax.annotate(f"{v:.3f}", xy=(x15[i], v),
                            xytext=(x15[i], v + y_span * 0.15),
                            ha="center", fontsize=10, fontweight="bold", color=color)
                if i > 0:
                    delta15 = v - vals[i - 1]
                    ax.annotate(f"Δ= {delta15:+.3f}",
                                xy=((x15[i] + x15[i-1]) / 2, (v + vals[i-1]) / 2),
                                ha="center", fontsize=8, color="#777777")
            ax.set_xticks(x15)
            ax.set_xticklabels(c_lbl15, fontsize=8, rotation=15, ha="right")
            ax.set_title(title15); ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
            ax.set_ylim(min(vals) - y_span * 0.3, max(vals) + y_span * 0.4)
        plt.suptitle(f"[{hosp_label}] Case 1 → 5: 예측 성능 지표 추이\n"
                     f"(AEC와 Scanner 추가에 따른 성능 향상 정량화)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(); fig.savefig(OUT_DIR / "15_case_progression.png", dpi=150); plt.close()

    print(f"  Saved: figs 04-15 to {OUT_DIR}")


def run_one_analysis(X_full, y_cont, CASES, RESULT_DIR, hosp_label, group_label):
    """
    단일 병원·성별 그룹에 대해 5-Fold CV 선형+로지스틱 회귀를 실행하고
    결과 플롯과 요약 Excel을 저장한다.

    0430에서 신설된 성별 층화 구조(전체·여성·남성)에 대응하는 핵심 함수.
    각 그룹별로 독립적으로 호출되어 그룹-내 P25를 이진화 임계값으로 사용함으로써
    비교 기준이 그룹의 분포를 반영하도록 한다.
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 그룹-내 하위 25%를 low-TAMA 기준으로 설정.
    # 층화 분석(여성-only, 남성-only)에서는 해당 성별의 분포를 기준으로 임계값을
    # 재산출하므로, 성별 특이적 위험 집단 정의와 동일한 효과를 가짐.
    tama_threshold = y_cont.quantile(0.25)
    y_bin = (y_cont < tama_threshold).astype(int)
    n = len(y_cont)
    print(f"\n    [{group_label}] n={n}  25th pct={tama_threshold:.1f}"
          f"  low(1)={y_bin.sum()}  high(0)={(y_bin==0).sum()}")

    labels = [CASE_LABELS.get(k, k) for k in CASES]
    colors = COLORS[:len(CASES)]

    def avail(feats):
        return [f for f in feats if f in X_full.columns]

    # ── Run CV ──
    lin_res, log_res = {}, {}
    print(f"      [Linear]")
    for name, feats in CASES.items():
        r = linear_cv(X_full[avail(feats)], y_cont)
        lin_res[name] = r
        print(f"        {name}: R²={r['R2']:.4f}±{r['R2_std']:.4f}  "
              f"MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}")

    print(f"      [Logistic  TAMA < {tama_threshold:.1f}]")
    for name, feats in CASES.items():
        r = logistic_cv(X_full[avail(feats)], y_bin)
        log_res[name] = r
        print(f"        {name}: AUC={r['AUC']:.4f}±{r['AUC_std']:.4f}  "
              f"Acc={r['Accuracy']:.4f}  Sens={r['Sensitivity']:.4f}  Spec={r['Specificity']:.4f}")

    prefix = f"[{hosp_label}] [{group_label}]"

    # ── Fig 01: Linear OOF actual vs predicted ──
    fig, axes = plt.subplots(1, len(CASES), figsize=(len(CASES)*5, 5))
    if len(CASES) == 1: axes = [axes]
    for ax, (name, feats), color, lbl in zip(axes, CASES.items(), colors, labels):
        true = np.array(lin_res[name]["oof_true"])
        pred = np.array(lin_res[name]["oof_pred"])
        ax.scatter(true, pred, alpha=0.2, s=8, color=color, rasterized=True)
        lo = min(true.min(), pred.min()) - 5; hi = max(true.max(), pred.max()) + 5
        ax.plot([lo, hi], [lo, hi], "k--", lw=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        r2v = lin_res[name]['R2']; r2s = lin_res[name]['R2_std']
        ax.set_title(f"{lbl.replace(chr(10),' ')}\nR²={r2v:.3f}±{r2s:.3f}", fontsize=9)
        ax.set_xlabel("Actual TAMA"); ax.set_ylabel("Predicted TAMA")
    plt.suptitle(f"{prefix} Linear – Actual vs Predicted (5-fold OOF)", fontsize=12, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "01_linear_actual_vs_pred.png", dpi=150); plt.close()

    # ── Fig 02: Linear metrics bar ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["R2", "MAE", "RMSE"]):
        vals = [lin_res[n][metric] for n in CASES]
        errs = [lin_res[n][f"{metric}_std"] for n in CASES]
        bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, edgecolor="white", width=0.5)
        ax.set_title(metric); ax.set_ylabel(metric)
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(errs)*0.1,
                    f"{v:.3f}\n±{e:.3f}", ha="center", va="bottom", fontsize=7)
    plt.suptitle(f"{prefix} Linear Regression – CV Metrics", fontsize=12, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "02_linear_metrics_comparison.png", dpi=150); plt.close()

    # ── Fig 03: Linear coefficients ──
    fig, axes = plt.subplots(1, len(CASES), figsize=(len(CASES)*4, 5))
    if len(CASES) == 1: axes = [axes]
    for ax, (name, feats), color in zip(axes, CASES.items(), colors):
        fa = avail(feats)
        pipe = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
        pipe.fit(X_full[fa], y_cont)
        cdf = pd.DataFrame({"feature": fa, "coef": pipe.named_steps["m"].coef_}).sort_values("coef")
        ax.barh(cdf["feature"], cdf["coef"],
                color=["#e74c3c" if c < 0 else "#3498db" for c in cdf["coef"]])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"{name}\n(standardized coef)", fontsize=8); ax.tick_params(labelsize=7)
    plt.suptitle(f"{prefix} Linear – Standardized Coefficients", fontsize=11, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "03_linear_coefficients.png", dpi=150); plt.close()

    # ── Fig 04: Logistic ROC ──
    fig, ax = plt.subplots(figsize=(6, 6))
    for (name, _), color, lbl in zip(CASES.items(), colors, labels):
        mean_fpr = np.linspace(0, 1, 200)
        tprs_i   = [np.interp(mean_fpr, f, t)
                    for f, t in zip(log_res[name]["fprs"], log_res[name]["tprs"])]
        mean_tpr = np.mean(tprs_i, axis=0); std_tpr = np.std(tprs_i, axis=0)
        auc = log_res[name]["AUC"]
        auc_std = log_res[name]["AUC_std"]
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                label=f"{lbl.replace(chr(10),' ')} (AUC={auc:.3f}±{auc_std:.3f})")
        ax.fill_between(mean_fpr, mean_tpr-std_tpr, mean_tpr+std_tpr, alpha=0.12, color=color)
    ax.plot([0,1],[0,1],"k--",lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{prefix} ROC (5-fold | 하위 25%={tama_threshold:.1f})")
    ax.legend(fontsize=9); plt.tight_layout()
    fig.savefig(RESULT_DIR / "04_logistic_roc.png", dpi=150); plt.close()

    # ── Fig 05: Logistic metrics bar ──
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, ["AUC", "Accuracy", "Sensitivity", "Specificity"]):
        vals = [log_res[n][metric] for n in CASES]
        errs = [log_res[n][f"{metric}_std"] for n in CASES]
        bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, edgecolor="white", width=0.5)
        ax.set_ylim(0, 1.2); ax.set_title(metric); ax.set_ylabel(metric)
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(errs)*0.1,
                    f"{v:.3f}\n±{e:.3f}", ha="center", va="bottom", fontsize=7)
    plt.suptitle(f"{prefix} Logistic Regression – CV Metrics", fontsize=12, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "05_logistic_metrics_comparison.png", dpi=150); plt.close()

    # ── Fig 06: Logistic confusion matrices ──
    fig, axes = plt.subplots(1, len(CASES), figsize=(len(CASES)*4, 4))
    if len(CASES) == 1: axes = [axes]
    for ax, (name, feats), lbl in zip(axes, CASES.items(), labels):
        fa = avail(feats)
        pipe = Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000, random_state=CV_RANDOM))])
        pipe.fit(X_full[fa], y_bin)
        cm = confusion_matrix(y_bin, pipe.predict(X_full[fa]))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["High","Low"], yticklabels=["High","Low"], cbar=False)
        ax.set_title(lbl.replace(chr(10), " "), fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle(f"{prefix} Confusion Matrix (full fit, 하위 25%={tama_threshold:.1f})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "06_logistic_confusion.png", dpi=150); plt.close()

    # ── Fig 07: Logistic coefficients ──
    fig, axes = plt.subplots(1, len(CASES), figsize=(len(CASES)*4, 5))
    if len(CASES) == 1: axes = [axes]
    for ax, (name, feats), color in zip(axes, CASES.items(), colors):
        fa = avail(feats)
        pipe = Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000, random_state=CV_RANDOM))])
        pipe.fit(X_full[fa], y_bin)
        coefs = pipe.named_steps["m"].coef_[0]
        cdf = pd.DataFrame({"feature": fa, "coef": coefs}).sort_values("coef")
        ax.barh(cdf["feature"], cdf["coef"],
                color=["#e74c3c" if c < 0 else "#3498db" for c in cdf["coef"]])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"{name}\n(AUC={log_res[name]['AUC']:.3f}±{log_res[name]['AUC_std']:.3f})", fontsize=8)
        ax.tick_params(labelsize=7)
    plt.suptitle(f"{prefix} Logistic – Standardized Coefficients", fontsize=11, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "07_logistic_coefficients.png", dpi=150); plt.close()

    # ── Fig 08: Overview R² & AUC ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(CASES))
    for ax, res_dict, metric, title in zip(
        axes, [lin_res, log_res], ["R2", "AUC"], ["Linear R²", "Logistic AUC"]
    ):
        vals = [res_dict[n][metric] for n in CASES]
        errs = [res_dict[n][f"{metric}_std"] for n in CASES]
        ax.bar(x, vals, yerr=errs, capsize=5, color=colors, edgecolor="white", width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title); ax.set_ylabel(title.split()[-1])
        if metric == "AUC": ax.set_ylim(0, 1)
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e + 0.005, f"{v:.3f}\n±{e:.3f}", ha="center", fontsize=7, fontweight="bold")
    plt.suptitle(f"{prefix} Case Comparison Overview", fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULT_DIR / "08_case_comparison_overview.png", dpi=150); plt.close()

    # ── Summary DataFrame & Excel ──
    rows = []
    for name, lbl in zip(CASES.keys(), labels):
        lr, lo = lin_res[name], log_res[name]
        rows.append({
            "Hospital": hosp_label, "Sex": group_label, "Case": lbl.replace("\n", " "),
            "N_features": len(CASES[name]), "N_rows": n,
            "Lin_R2": round(lr["R2"], 4), "Lin_R2_std": round(lr["R2_std"], 4),
            "Lin_MAE": round(lr["MAE"], 2), "Lin_RMSE": round(lr["RMSE"], 2),
            "TAMA_threshold": tama_threshold,
            "Log_AUC": round(lo["AUC"], 4), "Log_AUC_std": round(lo["AUC_std"], 4),
            "Log_Acc": round(lo["Accuracy"], 4),
            "Log_Sens": round(lo["Sensitivity"], 4),
            "Log_Spec": round(lo["Specificity"], 4),
        })
    summary_df = pd.DataFrame(rows)

    fold_lin = pd.DataFrame(
        {n: lin_res[n]["fold_r2"]  for n in CASES},
        index=[f"Fold{i+1}" for i in range(CV_SPLITS)]
    )
    fold_log = pd.DataFrame(
        {n: log_res[n]["fold_auc"] for n in CASES},
        index=[f"Fold{i+1}" for i in range(CV_SPLITS)]
    )
    with pd.ExcelWriter(RESULT_DIR / "regression_results.xlsx") as writer:
        summary_df.to_excel(writer, sheet_name="summary",         index=False)
        fold_lin.to_excel(writer,   sheet_name="linear_fold_r2")
        fold_log.to_excel(writer,   sheet_name="logistic_fold_auc")
    print(f"      Saved to {RESULT_DIR}")

    return summary_df, (X_full, y_cont, y_bin, CASES, tama_threshold)


# ────────────────────────────────────────────────
# 2. Per-hospital analysis (all / female / male)
# ────────────────────────────────────────────────
# [0424 대비 변경점]
# 0424: config.py에서 SITE를 수동으로 지정하여 한 병원씩 실행.
# 0430: data/ 폴더를 자동 탐색하여 강남·신촌을 루프로 처리.
#       각 병원 내에서 SEX_GROUPS 3개 그룹을 추가로 순회함으로써
#       6개 서브분석(2병원 × 3성별)이 단일 실행으로 완료됨.
all_summaries  = {}   # hosp_key -> summary_df (전체 only, for cross-hospital)
all_clean_data = {}   # hosp_key -> clean_tuple (전체 only, for external validation)

# 성별 층화 분석 그룹 정의. (sex_key, PatientSex_enc 값, 출력 레이블)
# sex_val=None이면 전체(all) 분석으로, CLINICAL(Sex 포함)을 사용.
# sex_val=0(여성) 또는 1(남성)이면 해당 그룹만 필터링하고 Sex를 예측변수에서 제거(CLIN_NOSEX).
# Sex가 내재적으로 고정되므로 공변량으로 포함하면 완전 공선(perfect collinearity)이 발생함.
SEX_GROUPS = [
    ("all",    None, "전체"),
    ("female", 0,    "여성(F)"),
    ("male",   1,    "남성(M)"),
]

for hosp_key, data_path in HOSPITALS.items():
    hosp_label = "강남" if hosp_key == "gangnam" else "신촌"
    print(f"\n{'='*60}")
    print(f"HOSPITAL: {hosp_label} ({hosp_key})")
    print(f"{'='*60}")

    BASE_DIR = SCRIPT_DIR.parent / "results" / "regression" / hosp_key

    # ── Load & merge ──
    feat_df, meta_df = load_hospital(data_path, f"{hosp_key}_temp.xlsx")
    meta_df["TAMA"] = pd.to_numeric(meta_df["TAMA"], errors="coerce")

    df = feat_df.merge(
        # metadata-bmi_add 시트 사용: 0424의 metadata-value 시트에 BMI 컬럼이 추가된 버전.
        # BMI 추가가 0430 핵심 변경사항이므로, 데이터 소스 시트명도 함께 변경됨.
        meta_df[["PatientID", "PatientAge", "PatientSex",
                 "BMI", "ManufacturerModelName", "kVp", "TAMA"]],
        on="PatientID", how="inner",
    )
    df["PatientSex_enc"] = df["PatientSex"].map({"F": 0, "M": 1})
    # ManufacturerModelName → one-hot 더미 (drop_first=True: 기준 스캐너 대비 상대적 효과 추정)
    model_dummies = pd.get_dummies(df["ManufacturerModelName"], prefix="model", drop_first=True)
    MODEL_COLS = model_dummies.columns.tolist()
    df = pd.concat([df, model_dummies], axis=1)

    SCANNER   = MODEL_COLS + ["kVp"]
    # CLINICAL: 전체(all) 분석의 기준선 변수. BMI가 0430에서 새로 추가됨.
    # → BMI를 보정함으로써 AEC 피처의 독립적 기여도를 더 정확히 추정 가능.
    CLINICAL  = ["PatientAge", "PatientSex_enc", "BMI"]        # includes sex
    # CLIN_NOSEX: 성별 층화 분석(여성 only, 남성 only)에서 사용.
    # 그룹 내에서 Sex는 상수이므로 제거하지 않으면 행렬이 singular해짐.
    CLIN_NOSEX = ["PatientAge", "BMI"]                          # sex-stratified

    ALL_COLS = list(dict.fromkeys(
        CLINICAL + AEC_PREV + AEC_NEW + SCANNER + ["TAMA", "PatientSex_enc"]
    ))
    df_clean = df[ALL_COLS].dropna().reset_index(drop=True)
    print(f"  Rows after dropna: {len(df_clean)}"
          f"  (F={( df_clean['PatientSex_enc']==0).sum()}"
          f"  M={(df_clean['PatientSex_enc']==1).sum()})")

    # ── EDA Figs 16–18 ──
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Fig 16: Scanner distribution
    scanner_counts = df["ManufacturerModelName"].value_counts()
    n_total_models = scanner_counts.shape[0]
    total_patients = len(df)
    top10 = scanner_counts.head(10)
    others_n = scanner_counts.iloc[10:].sum()
    plot_scan = pd.concat([top10, pd.Series({"기타": others_n})]) if others_n > 0 else top10
    plot_scan = plot_scan[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ["#3498db" if name != "기타" and i == len(plot_scan) - 1 else "#a0aab0"
                  for i, name in enumerate(plot_scan.index)]
    ax.barh(plot_scan.index, plot_scan.values, color=bar_colors)
    for name, val in zip(plot_scan.index, plot_scan.values):
        pct = val / total_patients * 100
        ax.text(val + total_patients * 0.004, list(plot_scan.index).index(name),
                f"{val} ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel("환자 수")
    ax.set_title(f"CT 스캐너 모델 분포 (총 {n_total_models}종)")
    ax.set_xlim(0, plot_scan.max() * 1.25)
    plt.tight_layout()
    fig.savefig(BASE_DIR / "16_scanner_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: 16_scanner_distribution.png")

    # Fig 17: kVp distribution
    kvp_counts = df["kVp"].dropna().astype(int).value_counts().sort_index()
    kvp_total = kvp_counts.sum()
    dominant_kvp = kvp_counts.idxmax()

    fig, ax = plt.subplots(figsize=(9, 5))
    kvp_colors = ["#3498db" if v == dominant_kvp else "#a0aab0" for v in kvp_counts.index]
    bars = ax.bar(kvp_counts.index.astype(str), kvp_counts.values, color=kvp_colors, width=0.6)
    for bar, val in zip(bars, kvp_counts.values):
        pct = val / kvp_total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + kvp_counts.max() * 0.01,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("kVp")
    ax.set_ylabel("환자 수")
    ax.set_title(f"kVp 분포 (주요값: {dominant_kvp} kVp)")
    ax.set_ylim(0, kvp_counts.max() * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(BASE_DIR / "17_kvp_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: 17_kvp_distribution.png")

    # Fig 18: Pearson correlation matrix (Clinical + AEC_prev)
    corr_feat_order = [f for f in ["PatientAge", "PatientSex_enc", "BMI"] + AEC_PREV
                       if f in df_clean.columns]
    rename_map = {"PatientSex_enc": "Sex"}
    X_corr = df_clean[corr_feat_order].rename(columns=rename_map)
    corr_mat = X_corr.corr(method="pearson")
    n_cf = len(corr_mat)
    mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(max(7, n_cf * 0.9), max(6, n_cf * 0.85)))
    sns.heatmap(
        corr_mat, mask=mask_upper, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, square=True,
        annot_kws={"size": 10},
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("선택 Feature 간 Pearson 상관행렬\n(|r| > 0.8: 다중공선성 주의)", fontsize=12)
    plt.tight_layout()
    fig.savefig(BASE_DIR / "18_correlation_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved: 18_correlation_matrix.png")

    # ── Fig 01: AEC Feature - TAMA Pearson 상관계수 Top 20 ──
    non_aec_set = ({"PatientID", "PatientAge", "PatientSex", "PatientSex_enc",
                    "BMI", "ManufacturerModelName", "kVp", "TAMA"} | set(MODEL_COLS))
    aec_all_cols = [c for c in df.columns if c not in non_aec_set]
    df_tama01   = df.dropna(subset=["TAMA"])
    corr_rows01 = []
    for f in aec_all_cols:
        sub = df_tama01[[f, "TAMA"]].dropna()
        if len(sub) < 10:
            continue
        r01, _ = scipy_stats.pearsonr(sub[f], sub["TAMA"])
        corr_rows01.append({"feature": f, "r": r01})
    if corr_rows01:
        corr_df01 = (pd.DataFrame(corr_rows01)
                     .assign(abs_r=lambda d: d["r"].abs())
                     .sort_values("abs_r", ascending=False)
                     .head(20)
                     .sort_values("r"))
        fig, ax = plt.subplots(figsize=(10, 7))
        cols01  = ["#e07070" if r > 0 else "#7090d0" for r in corr_df01["r"]]
        ax.barh(corr_df01["feature"], corr_df01["r"], color=cols01, height=0.7)
        ax.axvline(0, color="black", lw=0.8)
        ax.legend(handles=[Patch(color="#e07070", label="양의 상관 (+)"),
                           Patch(color="#7090d0", label="음의 상관 (-)")], fontsize=9)
        ax.set_xlabel("Pearson r with TAMA")
        ax.set_title(f"AEC Feature - TAMA 상관계수 Top 20\n(붉은색=양, 파란색=음)")
        ax.xaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
        plt.tight_layout(); fig.savefig(BASE_DIR / "01_feature_correlation.png", dpi=150); plt.close()
        print(f"  Saved: 01_feature_correlation.png")

    # ── Fig 02: VIF 다중공선성 검사 (Clinical + AEC_prev) ──
    vif_feats02 = [f for f in ["PatientAge", "PatientSex_enc", "BMI"] + AEC_PREV
                   if f in df_clean.columns]
    X_vif02     = df_clean[vif_feats02].copy()
    X_vif02_std = (X_vif02 - X_vif02.mean()) / X_vif02.std()
    lbl_vif     = {"PatientSex_enc": "Sex", "PatientAge": "PatientAge", "BMI": "BMI"}
    vif_names02 = [lbl_vif.get(f, f) for f in vif_feats02]
    vif_vals02  = [sm_vif(X_vif02_std.values, i) for i in range(X_vif02_std.shape[1])]
    vif_data02  = sorted(zip(vif_names02, vif_vals02), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(9, max(4, len(vif_data02) * 0.7 + 1.5)))
    vif_ns = [n for n, _ in vif_data02]; vif_vs = [v for _, v in vif_data02]
    ax.barh(vif_ns, vif_vs, color="#27ae60", height=0.6)
    for i, (n, v) in enumerate(vif_data02):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(5,  color="#f39c12", ls="--", lw=1.5, label="VIF=5 (주의)")
    ax.axvline(10, color="#e07070", ls="--", lw=1.5, label="VIF=10 (위험)")
    ax.set_xlabel("VIF (분산팽창인수)")
    ax.set_title("선택 변수의 VIF (다중공선성 검사)\nVIF<5: 낮음, 5-10: 중간, >10: 높음")
    ax.legend(fontsize=9); ax.xaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout(); fig.savefig(BASE_DIR / "02_vif_comparison.png", dpi=150); plt.close()
    print(f"  Saved: 02_vif_comparison.png")

    # ── Fig 03: 성별 TAMA 분포 및 이진화 임계값 ──
    df_m03  = df_clean[df_clean["PatientSex_enc"] == 1]["TAMA"].astype(float)
    df_f03  = df_clean[df_clean["PatientSex_enc"] == 0]["TAMA"].astype(float)
    thr_m03 = df_m03.quantile(0.25)
    thr_f03 = df_f03.quantile(0.25)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, lbl_sex, color, thr in [
        (axes[0], df_m03, "남성 (M)", "#6baed6", thr_m03),
        (axes[1], df_f03, "여성 (F)", "#fd8d3c", thr_f03),
    ]:
        ax.hist(data, bins=30, color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(thr, color="#e74c3c", ls="--", lw=2, label=f"임계값 {thr:.0f} cm²")
        ax.axvline(data.mean(), color="black", ls="-", lw=1.5, label=f"평균 {data.mean():.1f} cm²")
        low_n = int((data < thr).sum())
        ax.set_xlabel("TAMA (cm²)"); ax.set_ylabel("환자 수")
        ax.set_title(f"{lbl_sex} TAMA 분포\n(N={len(data)}, 중앙값={data.median():.0f})")
        ax.text(0.62, 0.92, f"임계값 이하: {low_n}명 ({low_n/len(data)*100:.1f}%)",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.legend(fontsize=9)
    plt.suptitle(f"[{hosp_label}] 성별 TAMA 분포 및 이진화 임계값",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(BASE_DIR / "03_tama_distribution.png", dpi=150); plt.close()
    print(f"  Saved: 03_tama_distribution.png")

    group_summaries = {}

    for sex_key, sex_val, sex_label in SEX_GROUPS:
        if sex_val is None:
            # 전체 분석: 성별을 공변량(PatientSex_enc)으로 포함
            df_sub = df_clean
            cases  = make_cases(CLINICAL, SCANNER)
        else:
            # 층화 분석: 해당 성별만 필터링 후 Sex 공변량 제거
            # 0424에는 없던 분석 축 — 성별 내 AEC 예측력의 이질성을 탐색하는 목적
            df_sub = df_clean[df_clean["PatientSex_enc"] == sex_val].reset_index(drop=True)
            cases  = make_cases(CLIN_NOSEX, SCANNER)

        if len(df_sub) < CV_SPLITS * 2:
            print(f"  [{sex_label}] too few rows ({len(df_sub)}), skipped")
            continue

        X_sub = df_sub.drop(columns=["TAMA"]).reset_index(drop=True)
        y_sub = df_sub["TAMA"].astype(float).reset_index(drop=True)

        result_dir = BASE_DIR / sex_key
        summary_df, clean_tuple = run_one_analysis(
            X_sub, y_sub, cases, result_dir, hosp_label, sex_label
        )
        group_summaries[sex_key] = summary_df

        if sex_key == "all":
            # 전체 그룹에 대해서만 full-fit 진단 플롯(잔차·보정·케이스 비교)을 추가 생성.
            # 층화 그룹(여성·남성)은 CV 결과 플롯만 저장 — 표본 크기 제한을 고려한 설계.
            all_summaries[hosp_key]  = summary_df
            all_clean_data[hosp_key] = clean_tuple
            run_fullfit_analysis(X_sub, y_sub, cases, BASE_DIR, hosp_label)

    # ── Fig: Sex comparison – R² and AUC per case ──
    if len(group_summaries) == 3:
        sex_colors = {"all": "#555555", "female": "#e74c3c", "male": "#3498db"}
        sex_disp   = {"all": "전체", "female": "여성(F)", "male": "남성(M)"}
        case_short = ["C1\nClinical", "C2\n+AEC\nprev", "C3\n+AEC\nnew",
                      "C4\n+prev\n+Scan", "C5\n+new\n+Scan"]
        n_cases = 5
        x = np.arange(n_cases)
        w = 0.25

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, metric, std_key, title, ylim in zip(
            axes,
            ["Lin_R2", "Log_AUC"],
            ["Lin_R2_std", "Log_AUC_std"],
            ["Linear R²", "Logistic AUC"],
            [None, (0, 1)],
        ):
            for i, skey in enumerate(["all", "female", "male"]):
                df_g = group_summaries[skey]
                vals = [df_g.iloc[k][metric]   for k in range(n_cases)]
                errs = [df_g.iloc[k][std_key]  for k in range(n_cases)]
                offset = (i - 1) * w
                bars = ax.bar(x + offset, vals, w, yerr=errs, capsize=4,
                              color=sex_colors[skey], label=sex_disp[skey], edgecolor="white")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6)
            ax.set_xticks(x); ax.set_xticklabels(case_short, fontsize=8)
            ax.set_title(title); ax.set_ylabel(title.split()[-1])
            if ylim: ax.set_ylim(*ylim)
            ax.legend(fontsize=9)

        plt.suptitle(f"[{hosp_label}] 성별 비교: 전체 vs 여성 vs 남성",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(BASE_DIR / "sex_comparison.png", dpi=150); plt.close()
        print(f"\n  Saved: {BASE_DIR / 'sex_comparison.png'}")

        # Combined Excel
        combined_sex = pd.concat(group_summaries.values(), ignore_index=True)
        combined_sex.to_excel(BASE_DIR / "sex_comparison_summary.xlsx", index=False)
        print(f"  Saved: sex_comparison_summary.xlsx")

# ────────────────────────────────────────────────
# 3. Cross-hospital comparison (all group only)
# ────────────────────────────────────────────────
# [0430 신설 섹션]
# 0424는 한 병원씩 독립 실행하여 병원 간 비교가 불가능했음.
# 0430에서는 강남·신촌 결과를 하나의 스크립트에서 수집하여
# 각 Case별 R²·AUC의 병원 간 일치성(재현성)을 정량적으로 제시.
# → AEC 기반 예측 모델의 외적 타당도(external validity) 근거 제공.
if len(all_summaries) == 2:
    print(f"\n{'='*60}")
    print("CROSS-HOSPITAL COMPARISON (전체)")
    print(f"{'='*60}")

    combined = pd.concat(all_summaries.values(), ignore_index=True)
    COMPARE_DIR = SCRIPT_DIR.parent / "results" / "regression"
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    print(combined[["Hospital","Case","N_rows","Lin_R2","Log_AUC",
                     "Log_Acc","Log_Sens","Log_Spec"]].to_string(index=False))

    case_short  = ["Case1\nClinical", "Case2\n+AEC\nprev", "Case3\n+AEC\nnew",
                   "Case4\n+prev\n+Scanner", "Case5\n+new\n+Scanner"]
    hosp_colors = {"gangnam": "#3498db", "sinchon": "#e74c3c"}
    hosp_labels = {"gangnam": "강남", "sinchon": "신촌"}
    n_cases = 5

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(n_cases); w = 0.35

    for ax, metric, std_key, title, ylim in zip(
        axes,
        ["Lin_R2", "Log_AUC"], ["Lin_R2_std", "Log_AUC_std"],
        ["Linear R²", "Logistic AUC"], [None, (0, 1)],
    ):
        for i, (hkey, hdf) in enumerate(all_summaries.items()):
            vals = [hdf.iloc[k][metric]   for k in range(n_cases)]
            errs = [hdf.iloc[k][std_key]  for k in range(n_cases)]
            offset = (i - 0.5) * w
            bars = ax.bar(x + offset, vals, w, yerr=errs, capsize=4,
                          color=hosp_colors[hkey], label=hosp_labels[hkey], edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(case_short, fontsize=8)
        ax.set_title(title); ax.set_ylabel(title.split()[-1])
        if ylim: ax.set_ylim(*ylim)
        ax.legend()

    plt.suptitle("병원별 케이스 비교: 강남 vs 신촌", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(COMPARE_DIR / "09_cross_hospital_comparison.png", dpi=150); plt.close()
    print(f"\n  Saved: 09_cross_hospital_comparison.png")

    combined.to_excel(COMPARE_DIR / "cross_hospital_summary.xlsx", index=False)
    print(f"  Saved: cross_hospital_summary.xlsx")

# ────────────────────────────────────────────────
# 4. External validation: train on A, test on B
# ────────────────────────────────────────────────
if len(all_clean_data) == 2:
    print(f"\n{'='*60}")
    print("EXTERNAL VALIDATION (train → test)")
    print(f"{'='*60}")

    COMPARE_DIR = SCRIPT_DIR.parent / "results" / "regression"
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    hosp_keys = list(all_clean_data.keys())
    hosp_labels_map = {"gangnam": "강남", "sinchon": "신촌"}

    directions = [(hosp_keys[0], hosp_keys[1]), (hosp_keys[1], hosp_keys[0])]
    ext_rows = []

    for train_key, test_key in directions:
        X_tr, y_tr_cont, y_tr_bin, CASES_tr, _ = all_clean_data[train_key]
        X_te, y_te_cont, y_te_bin, CASES_te, tama_te = all_clean_data[test_key]
        train_lbl = hosp_labels_map[train_key]
        test_lbl  = hosp_labels_map[test_key]
        print(f"\n  Train: {train_lbl} (n={len(X_tr)})  ->  Test: {test_lbl} (n={len(X_te)})")

        for case_name, feats_tr in CASES_tr.items():
            feats_common = [f for f in feats_tr if f in X_te.columns]
            missing = set(feats_tr) - set(feats_common)
            if missing:
                print(f"    [{case_name}] Warning: {len(missing)} features missing, skipped")
            if not feats_common:
                continue

            pipe_lin = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
            pipe_lin.fit(X_tr[feats_common], y_tr_cont)
            pred_cont = pipe_lin.predict(X_te[feats_common])
            lin_r2   = r2_score(y_te_cont, pred_cont)
            lin_mae  = mean_absolute_error(y_te_cont, pred_cont)
            lin_rmse = np.sqrt(mean_squared_error(y_te_cont, pred_cont))

            y_te_bin_ext = (y_te_cont < tama_te).astype(int)
            pipe_log = Pipeline([
                ("sc", StandardScaler()),
                ("m", LogisticRegression(max_iter=2000, random_state=CV_RANDOM, solver="lbfgs")),
            ])
            pipe_log.fit(X_tr[feats_common], y_tr_bin)
            prob    = pipe_log.predict_proba(X_te[feats_common])[:, 1]
            pred_bin = pipe_log.predict(X_te[feats_common])
            log_auc = roc_auc_score(y_te_bin_ext, prob)
            log_acc = accuracy_score(y_te_bin_ext, pred_bin)
            tn, fp, fn, tp = confusion_matrix(y_te_bin_ext, pred_bin).ravel()
            log_sens = tp / (tp + fn) if tp + fn else 0
            log_spec = tn / (tn + fp) if tn + fp else 0

            print(f"    {case_name}: LinR²={lin_r2:.4f}  AUC={log_auc:.4f}  "
                  f"Acc={log_acc:.4f}  Sens={log_sens:.4f}  Spec={log_spec:.4f}")

            ext_rows.append({
                "Train": train_lbl, "Test": test_lbl,
                "Case": case_name, "N_train": len(X_tr), "N_test": len(X_te),
                "N_features_used": len(feats_common),
                "Lin_R2": round(lin_r2, 4), "Lin_MAE": round(lin_mae, 2),
                "Lin_RMSE": round(lin_rmse, 2), "TAMA_threshold": tama_te,
                "Log_AUC": round(log_auc, 4), "Log_Acc": round(log_acc, 4),
                "Log_Sens": round(log_sens, 4), "Log_Spec": round(log_spec, 4),
            })

    ext_df = pd.DataFrame(ext_rows)

    directions_labels = [f"{hosp_labels_map[a]}->{hosp_labels_map[b]}" for a, b in directions]
    dir_colors = ["#8e44ad", "#27ae60"]
    case_names = ext_df["Case"].unique().tolist()
    n_c = len(case_names)
    xtick_labels = [c.replace("Case", "C").replace("_Clinical", "\nClinical")
                     .replace("+AEC_prev", "\n+AEC\nprev").replace("+AEC_new", "\n+AEC\nnew")
                     .replace("+Scanner", "\n+Scanner") for c in case_names]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(n_c); w = 0.35

    for ax, metric, title, ylim in zip(
        axes, ["Lin_R2", "Log_AUC"],
        ["External Linear R²", "External Logistic AUC"], [None, (0, 1)]
    ):
        for i, (dlabel, dcolor) in enumerate(zip(directions_labels, dir_colors)):
            sub = ext_df[ext_df["Train"].apply(lambda v: v in dlabel)
                         & ext_df["Test"].apply(lambda v: v in dlabel)]
            vals = [sub[sub["Case"] == c][metric].values[0] if len(sub[sub["Case"] == c]) else 0
                    for c in case_names]
            offset = (i - 0.5) * w
            bars = ax.bar(x + offset, vals, w, color=dcolor, label=dlabel, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(xtick_labels, fontsize=7)
        ax.set_title(title); ax.set_ylabel(metric.split("_")[-1])
        if ylim: ax.set_ylim(*ylim)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.legend(fontsize=8)

    plt.suptitle("외부 검증: 병원 간 교차 (Train -> Test)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(COMPARE_DIR / "10_external_validation.png", dpi=150); plt.close()
    print(f"\n  Saved: 10_external_validation.png")

    ext_df.to_excel(COMPARE_DIR / "external_validation_results.xlsx", index=False)
    print(f"  Saved: external_validation_results.xlsx")

print(f"\n{'='*60}")
print("ALL DONE")
print(f"{'='*60}")
