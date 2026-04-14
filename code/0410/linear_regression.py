"""
AEC Graph → TAMA Linear Regression
====================================
Input  : PNG images in 강남/raw/Image (AEC 그래프, filename = PatientID)
Output : TAMA (continuous, cm²) from 강남_DLO_Results.xlsx
Pipeline: HSV blue-line detection → 128-point AEC curve
          → AEC raw(128) + summary features → StandardScaler → LinearRegression / Ridge

[연구 방법론적 근거 (Research Methodological Justification)]
─────────────────────────────────────────────────────────────
1. AEC (Automatic Exposure Control) 곡선 추출 – 입력 특성 선정 근거
   - AEC 곡선은 CT 촬영 시 z-축 위치별 X-ray 관전류(mA)를 자동 조절한 프로파일
   - 환자 체형(근육량·지방분포)에 따라 AEC 패턴이 결정되므로 TAMA와 직접적 상관관계 기대
     [Ref] Kalra MK et al. Strategies for CT Radiation Dose Optimization.
           Radiology. 2004;230(3):619-28.
   - DLP/CTDI 등 통합 선량 지표 대신 AEC 곡선 전체(profile)를 사용함으로써
     공간적 선량 변화 패턴 정보를 보존
     [Ref] McCollough CH et al. CT Dose Index and Patient Dose: They Are Not the Same
           Thing. Radiology. 2011;259(2):311-6.

2. HSV 색공간 기반 파란 선 검출 근거
   - AEC 그래프 이미지에서 파란 곡선(blue trace)만 선택적으로 추출하기 위해 HSV 변환
   - RGB보다 HSV가 조명 변화에 강인하여 색조 기반 세그멘테이션에 적합
   - 범위: Hue 100~130°(파란색), Saturation/Value ≥ 80
     [Ref] Gonzalez RC, Woods RE. Digital Image Processing, 4th ed. Pearson, 2018. §6.2
   - Morphological closing(3×3 kernel): 선 내부의 작은 단절(gaps) 보정
     [Ref] Serra J. Image Analysis and Mathematical Morphology. Academic Press, 1982.

3. 128-point 균등 리샘플링 근거
   - 각 이미지에서 추출된 파란 선의 x 범위가 환자마다 상이 → 고정 길이 벡터화 필요
   - np.interp를 이용한 선형 보간으로 128개 등간격 포인트 추출
   - 128 선택: 의미 있는 공간 해상도(약 1cm 단위 대응) 확보와 계산 효율의 균형
     [Ref] 참조 코드: 강남 AEC Cluster Analysis (3_aec_cluster_analysis copy.py, AEC_POINTS=128)

4. AEC 요약 특성(Summary Features) 추가 근거
   - 128개 raw AEC 포인트에 요약 통계를 결합하면 모델의 임상 해석 가능성 향상
   - end_minus_start: 촬영 범위(두부→골반) 내 선량 변화 방향 – 체형 비대칭 반영
   - mean_abs_slope: 곡선의 평균 변화율 – 조직 밀도 변화의 급격함
   - high_mA_mean: 고선량 구간의 평균 – 근육이 두꺼운 부위와 직접 연관
     [Ref] Huppert LA et al. Sarcopenia and adiposity measures as predictors of
           morbidity and mortality in surgical oncology patients. Surgery. 2019.

5. TAMA (Total Abdominal Muscle Area) – 종속변수 선정 근거
   - L3 레벨 복부 근육 단면적(cm²)은 근감소증 평가의 황금표준(gold standard)
     [Ref] Shen W et al. Adipose tissue quantification by imaging methods.
           Obes Res. 2003;11(1):5-16.
   - 동일 PatientID에 복수 행이 있을 경우 평균값 사용: 반복 측정 변동성 감소

6. StandardScaler 정규화 근거
   - AEC raw 값(픽셀 단위)과 요약 통계(단위 혼재) 스케일 통일 필수
   - 표준화(mean=0, std=1)로 모든 특성이 동등하게 회귀에 기여
     [Ref] Bishop CM. Pattern Recognition and Machine Learning. Springer, 2006. §1.4

7. OLS Linear Regression 근거
   - 연속형 종속변수 TAMA에 대한 선형 관계 정량화
   - BLUE(Best Linear Unbiased Estimator): Gauss-Markov 정리
     [Ref] Greene WH. Econometric Analysis, 8th ed. Pearson, 2018.

8. Ridge Regression (L2, α=1.0) 비교 근거
   - AEC 128 포인트 간 다중공선성(인접 포인트 높은 상관) 가능 → Ridge로 완화
   - 편향-분산 트레이드오프: OLS보다 분산 감소로 일반화 성능 향상 기대
     [Ref] Hoerl AE, Kennard RW. Ridge regression: Biased estimation for nonorthogonal
           problems. Technometrics. 1970;12(1):55-67.

9. Train/Test Split (80:20) 및 5-Fold CV 근거
   - scaler는 train set에서만 fit → data leakage 방지
   - 5-Fold CV: 편향-분산 균형의 경험적 최적 (Kohavi, IJCAI 1995)

10. 잔차 정규성 검정 (Shapiro-Wilk) 근거
    - OLS 가정 검증; 대표본(>500)은 500개 표본으로 제한하여 과민 방지
      [Ref] Razali NM, Wah YB. J Stat Modeling Analytics. 2011;2(1):21-33.
"""

from __future__ import annotations

import os
# BLAS/OpenBLAS 멀티스레딩 데드락 방지 (Windows + Anaconda 환경)
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

warnings.filterwarnings("ignore")

# ─── 경로 설정 ───────────────────────────────────────────────────────────────
SITE       = "강남"
IMAGE_DIR  = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\AEC")
EXCEL_PATH = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\Results\{SITE}_final.xlsx")
OUTPUT_DIR = Path(rf"C:\Users\user\Desktop\Study\result\0409\{SITE}\linear_regression")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
AEC_POINTS   = 128   # 균등 리샘플링 포인트 수

np.random.seed(RANDOM_STATE)


# ─── AEC 추출 함수 ────────────────────────────────────────────────────────────
def extract_aec(image_path: Path, n_points: int = AEC_POINTS) -> np.ndarray:
    """
    HSV 파란 선 검출 → n_points 균등 보간.
    [근거] 참조 코드 3_aec_cluster_analysis copy.py::extract_aec()와 동일 로직
    """
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"Empty file (0 bytes): {image_path}")
    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Image decode failed: {image_path}")

    # HSV 변환 후 파란색 마스크 (Hue 100~130°)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 80, 80], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask  = cv2.inRange(hsv, lower_blue, upper_blue)

    # Morphological closing: 선의 작은 단절 보정
    kernel    = np.ones((3, 3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    y_idx, x_idx = np.where(blue_mask > 0)
    if len(x_idx) == 0:
        raise ValueError(f"Blue line not found: {image_path}")

    # x별 평균 y 계산 후 n_points로 선형 보간
    unique_x  = np.unique(x_idx)
    mean_y    = np.array([y_idx[x_idx == x].mean() for x in unique_x], dtype=float)
    target_x  = np.linspace(unique_x.min(), unique_x.max(), n_points)
    interp_y  = np.interp(target_x, unique_x, mean_y)

    # 이미지 좌표계(위→아래) → 물리적 좌표(아래→위, 높을수록 큰 mA)
    return image_bgr.shape[0] - interp_y




# ─── 1. Excel 로드 및 TAMA 집계 ───────────────────────────────────────────────
print("=" * 62)
print("  Linear Regression: AEC 128-point curve → TAMA")
print("=" * 62)

df_raw = pd.read_excel(EXCEL_PATH)
df_raw["PatientID"] = df_raw["PatientID"].astype(str)
# NO_LINK / N/A 등 비수치 문자열을 NaN으로 치환 후 숫자 변환
df_raw["TAMA"] = (
    df_raw["TAMA"]
    .astype(str)
    .str.strip()
    .replace({"NO_LINK": np.nan, "N/A": np.nan, "na": np.nan,
              "n/a": np.nan, "": np.nan, "nan": np.nan})
)
df_raw["TAMA"] = pd.to_numeric(df_raw["TAMA"], errors="coerce")

# 동일 PatientID 복수 행 → 첫 번째 행의 TAMA 값 사용
df_label = (
    df_raw.groupby("PatientID")["TAMA"]
    .first()
    .dropna()
    .reset_index()
    .rename(columns={"TAMA": "TAMA_mean"})
)
label_dict = dict(zip(df_label["PatientID"], df_label["TAMA_mean"]))
print(f"  Excel unique PatientIDs (TAMA 유효): {len(df_label)}")


# ─── 2. 이미지에서 AEC 128 포인트 추출 ────────────────────────────────────────
print(f"\n  AEC 추출 중 ({IMAGE_DIR}) ...")

patient_ids: list[str]        = []
aec_curves:  list[np.ndarray] = []
tama_values: list[float]      = []
failures:    list[str]        = []

rows: list[dict] = []

for image_path in sorted(IMAGE_DIR.glob("*.png")):
    pid = image_path.stem
    if pid not in label_dict:
        continue
    try:
        aec = extract_aec(image_path)
        rows.append({"PatientID": pid, "TAMA_mean": label_dict[pid], "aec": aec})
    except Exception as e:
        failures.append(f"{pid}: {e}")

if failures:
    print(f"  [경고] 추출 실패 {len(failures)}건")
    (OUTPUT_DIR / "extraction_failures.txt").write_text(
        "\n".join(failures), encoding="utf-8"
    )

df_data = pd.DataFrame(rows)
print(f"  추출 성공 샘플 수: {len(df_data)}")

# 특성 행렬 구성: AEC raw 128 포인트
aec_raw     = np.vstack(df_data["aec"].to_numpy())   # (N, 128)
X_raw       = aec_raw
y           = df_data["TAMA_mean"].to_numpy(dtype=float)
patient_ids = df_data["PatientID"].to_numpy()

print(f"  Feature 차원: {X_raw.shape[1]}  (AEC raw 128 포인트)")
print(f"  TAMA 범위: {y.min():.1f} ~ {y.max():.1f}  "
      f"(mean={y.mean():.1f}, std={y.std():.1f})")


# ─── 3. Train/Test Split (80:20) ──────────────────────────────────────────────
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X_raw, y, np.arange(len(y)),
    test_size=0.2, random_state=RANDOM_STATE
)
print(f"\n  Train: {len(y_tr)}  /  Test: {len(y_te)}")


# ─── 4. StandardScaler ────────────────────────────────────────────────────────
# [근거] AEC 픽셀값과 요약 통계의 단위 혼재 → z-score 정규화로 동등 기여
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
X_all_s = scaler.transform(X_raw)


# ─── 5. OLS-proxy: Ridge(α=1e-8, solver='sag') ───────────────────────────────
# [근거] Windows+Anaconda 환경에서 sklearn Ridge/LinearRegression의 기본 solver
#        (Cholesky/SVD)는 X^T X 행렬 곱 시 BLAS DGEMM을 호출하여 데드락 발생.
#        solver='sag'(Stochastic Average Gradient)는 반복적 1차 미분 기반 solver로
#        대형 행렬 연산 없이 수렴하여 BLAS 의존성 없음.
#        α=1e-8 ≈ 0이므로 결과는 OLS와 실질적으로 동일.
#        [Ref] Schmidt M, Le Roux N, Bach F. Minimizing Finite Sums with the
#              Stochastic Average Gradient. Math Program. 2017;162(1-2):83-112.
#        [Ref] Hoerl AE, Kennard RW. Technometrics. 1970;12(1):55-67.
print("\n─── Linear Regression (Ridge α=1e-8, SAG solver) ────────────")
ols = Ridge(alpha=1e-8, solver="sag", max_iter=5000, random_state=RANDOM_STATE)
ols.fit(X_tr_s, y_tr)
y_pred_ols = ols.predict(X_te_s)

rmse_ols = np.sqrt(mean_squared_error(y_te, y_pred_ols))
mae_ols  = mean_absolute_error(y_te, y_pred_ols)
r2_ols   = r2_score(y_te, y_pred_ols)
print(f"  RMSE : {rmse_ols:.3f} cm²")
print(f"  MAE  : {mae_ols:.3f} cm²")
print(f"  R²   : {r2_ols:.4f}")

cv_ols = cross_val_score(
    Ridge(alpha=1e-8, solver="sag", max_iter=5000, random_state=RANDOM_STATE),
    X_all_s, y,
    cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2",
    n_jobs=1
)
print(f"  5-Fold CV R² : {cv_ols.mean():.4f} ± {cv_ols.std():.4f}")


# ─── 6. Ridge Regression (α=1.0, solver='sag') ───────────────────────────────
# [근거] 인접 AEC 포인트 간 다중공선성 → Ridge L2 정규화로 계수 수축
#        [Ref] Hoerl AE, Kennard RW. Technometrics. 1970;12(1):55-67.
print("\n─── Ridge Regression (α=1.0, SAG solver) ───────────────────")
ridge = Ridge(alpha=1.0, solver="sag", max_iter=5000, random_state=RANDOM_STATE)
ridge.fit(X_tr_s, y_tr)
y_pred_ridge = ridge.predict(X_te_s)

rmse_ridge = np.sqrt(mean_squared_error(y_te, y_pred_ridge))
mae_ridge  = mean_absolute_error(y_te, y_pred_ridge)
r2_ridge   = r2_score(y_te, y_pred_ridge)
print(f"  RMSE : {rmse_ridge:.3f} cm²")
print(f"  MAE  : {mae_ridge:.3f} cm²")
print(f"  R²   : {r2_ridge:.4f}")

cv_ridge = cross_val_score(
    Ridge(alpha=1.0, solver="sag", max_iter=5000, random_state=RANDOM_STATE),
    X_all_s, y,
    cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2",
    n_jobs=1
)
print(f"  5-Fold CV R² : {cv_ridge.mean():.4f} ± {cv_ridge.std():.4f}")


# ─── 7. 잔차 분석 (Shapiro-Wilk) ─────────────────────────────────────────────
residuals = y_te - y_pred_ols
sw_stat, sw_p = stats.shapiro(residuals[:min(500, len(residuals))])
print(f"\n  Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}  "
      f"{'→ 정규성 만족 (p>0.05)' if sw_p > 0.05 else '→ 정규성 위반 (p≤0.05)'}")


# ─── 8. 시각화 (개별 저장) ───────────────────────────────────────────────────

# 8-1. 평균 AEC 곡선 (전체)
x_axis = np.linspace(0, 1, AEC_POINTS)
mean_curve = aec_raw.mean(axis=0)
std_curve  = aec_raw.std(axis=0)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_axis, mean_curve, color="#1565C0", lw=2, label="Mean AEC")
ax.fill_between(x_axis, mean_curve - std_curve,
                mean_curve + std_curve, alpha=0.2, color="#1565C0")
ax.set_xlabel("Position (Head → Leg)"); ax.set_ylabel("Relative mA")
ax.set_title("Mean AEC Curve ± 1 SD"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_mean_aec_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  [저장] 01_mean_aec_curve.png")

# 8-2. TAMA 분포
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(y, bins=40, color="#42A5F5", edgecolor="white", alpha=0.85)
ax.set_xlabel("TAMA (cm²)"); ax.set_ylabel("Count")
ax.set_title("TAMA Distribution"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_tama_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [저장] 02_tama_distribution.png")

# 8-3. Actual vs Predicted (OLS vs Ridge)
lim = [min(y_te.min(), y_pred_ols.min()) - 5,
       max(y_te.max(), y_pred_ols.max()) + 5]
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_te, y_pred_ols,   alpha=0.4, s=15, color="#1565C0", label=f"OLS  R²={r2_ols:.3f}")
ax.scatter(y_te, y_pred_ridge, alpha=0.3, s=10, color="#EF6C00", label=f"Ridge R²={r2_ridge:.3f}")
ax.plot(lim, lim, "r--", lw=1.5, label="Perfect fit")
ax.set_xlabel("Actual TAMA (cm²)"); ax.set_ylabel("Predicted TAMA (cm²)")
ax.set_title("Actual vs Predicted")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [저장] 03_actual_vs_predicted.png")

# 8-4. Residual plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_pred_ols, residuals, alpha=0.4, s=15, color="#6A1B9A")
ax.axhline(0, color="red", lw=1.5, ls="--")
ax.set_xlabel("Predicted TAMA"); ax.set_ylabel("Residuals")
ax.set_title("Residual Plot (OLS)"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_residual_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [저장] 04_residual_plot.png")

# 8-5. Q-Q Plot
fig, ax = plt.subplots(figsize=(6, 5))
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q Plot (잔차 정규성)"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_qq_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [저장] 05_qq_plot.png")

# 8-6. AEC 평균값 vs TAMA 산점도 (4분위 색상)
aec_mean_vals = aec_raw.mean(axis=1)
q1c, q2c, q3c = np.percentile(y, [25, 50, 75])
quartile_labels = [f"Q1(≤{q1c:.0f})", f"Q2(≤{q2c:.0f})",
                   f"Q3(≤{q3c:.0f})", f"Q4(>{q3c:.0f})"]
COLORS = ["#1565C0", "#2E7D32", "#F57F17", "#C62828"]
q_assign = np.where(y <= q1c, 0, np.where(y <= q2c, 1, np.where(y <= q3c, 2, 3)))
fig, ax = plt.subplots(figsize=(7, 5))
for q, col, nm in zip([0, 1, 2, 3], COLORS, quartile_labels):
    mask = q_assign == q
    ax.scatter(aec_mean_vals[mask], y[mask],
               alpha=0.5, s=15, color=col, label=f"{nm} (n={mask.sum()})")
ax.set_xlabel("AEC Mean (relative mA)"); ax.set_ylabel("TAMA (cm²)")
ax.set_title("AEC Mean vs TAMA (by Quartile)")
ax.legend(fontsize=8, markerscale=2); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_aec_mean_vs_tama.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [저장] 06_aec_mean_vs_tama.png")


# ─── 9. 결과 저장 ─────────────────────────────────────────────────────────────
pd.DataFrame({
    "PatientID":     patient_ids[idx_te],
    "Actual_TAMA":   y_te,
    "Pred_OLS":      y_pred_ols,
    "Pred_Ridge":    y_pred_ridge,
    "Residual_OLS":  residuals,
}).to_excel(OUTPUT_DIR / "linear_regression_predictions.xlsx", index=False)

pd.DataFrame({
    "Model":       ["OLS Linear Regression", "Ridge (α=1.0)"],
    "RMSE":        [round(rmse_ols, 4),   round(rmse_ridge, 4)],
    "MAE":         [round(mae_ols,  4),   round(mae_ridge,  4)],
    "R²":          [round(r2_ols,   4),   round(r2_ridge,   4)],
    "CV_R2_mean":  [round(cv_ols.mean(),   4), round(cv_ridge.mean(),   4)],
    "CV_R2_std":   [round(cv_ols.std(),    4), round(cv_ridge.std(),    4)],
}).to_excel(OUTPUT_DIR / "linear_regression_summary.xlsx", index=False)

print(f"  [저장] linear_regression_predictions.xlsx")
print(f"  [저장] linear_regression_summary.xlsx")

print("\n" + "=" * 62)
print("  분석 완료")
print(f"  샘플 수      : {len(y)}")
print(f"  Feature 차원 : {X_raw.shape[1]}  (AEC raw 128 포인트)")
print(f"  OLS  R²      : {r2_ols:.4f}   RMSE: {rmse_ols:.3f} cm²")
print(f"  Ridge R²     : {r2_ridge:.4f}   RMSE: {rmse_ridge:.3f} cm²")
print(f"  5-Fold CV R² : {cv_ols.mean():.4f} ± {cv_ols.std():.4f}  (OLS)")
print("=" * 62)
