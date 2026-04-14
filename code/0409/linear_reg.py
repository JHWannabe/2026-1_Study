import ast
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, max_error, explained_variance_score,
)
from scipy import stats
from scipy.stats import shapiro, pearsonr, spearmanr, jarque_bera

# ── 한글 폰트 설정 ─────────────────────────────────────────────
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ──────────────────────────────────────────────────
SITE = "신촌"
excel_path = rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\Results\{SITE}_final.xlsx"
out_dir = rf"C:\Users\user\Desktop\Study\result\0409\{SITE}"
os.makedirs(out_dir, exist_ok=True)

# ── 데이터 로드 ────────────────────────────────────────────────
df = pd.read_excel(excel_path)
X = np.array(df["AEC"].apply(ast.literal_eval).tolist())   # (N, 128)
y = df["TAMA"].values                                        # (N,)

n_samples, n_features = X.shape
print(f"샘플 수: {n_samples},  AEC 차원: {n_features}")

# ── Train/Test 분할 ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)},  Test: {len(X_test)}")

# ── 모델 학습 ──────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)
residuals     = y_test - y_pred_test


# ══════════════════════════════════════════════════════════════
# 1. 기본 회귀 지표 (Test set)
# ══════════════════════════════════════════════════════════════
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

r2_test   = r2_score(y_test, y_pred_test)
r2_train  = r2_score(y_train, y_pred_train)
adj_r2    = adjusted_r2(r2_test, len(y_test), n_features)
mae_val   = mean_absolute_error(y_test, y_pred_test)
mse_val   = mean_squared_error(y_test, y_pred_test)
rmse_val  = np.sqrt(mse_val)
medae_val = median_absolute_error(y_test, y_pred_test)
maxe_val  = max_error(y_test, y_pred_test)
evs_val   = explained_variance_score(y_test, y_pred_test)
mape_val  = mape(y_test, y_pred_test)
smape_val = smape(y_test, y_pred_test)
bias_val  = np.mean(residuals)           # 평균 잔차 (편향)
std_res   = np.std(residuals, ddof=1)


# ══════════════════════════════════════════════════════════════
# 2. 교차검증 (5-Fold CV on full dataset)
# ══════════════════════════════════════════════════════════════
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    LinearRegression(), X, y, cv=kf,
    scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
    return_train_score=True,
)
cv_r2_mean   = cv_results["test_r2"].mean()
cv_r2_std    = cv_results["test_r2"].std()
cv_mae_mean  = -cv_results["test_neg_mean_absolute_error"].mean()
cv_rmse_mean = -cv_results["test_neg_root_mean_squared_error"].mean()


# ══════════════════════════════════════════════════════════════
# 3. 상관 분석
# ══════════════════════════════════════════════════════════════
pearson_r, pearson_p   = pearsonr(y_test, y_pred_test)
spearman_r, spearman_p = spearmanr(y_test, y_pred_test)


# ══════════════════════════════════════════════════════════════
# 4. 잔차 정규성 검정
# ══════════════════════════════════════════════════════════════
sw_stat, sw_p = shapiro(residuals)               # Shapiro-Wilk (N<5000 권장)
jb_stat, jb_p = jarque_bera(residuals)           # Jarque-Bera


# ══════════════════════════════════════════════════════════════
# 5. Durbin-Watson (잔차 자기상관)
# ══════════════════════════════════════════════════════════════
def durbin_watson(resid):
    diff = np.diff(resid)
    return np.sum(diff ** 2) / np.sum(resid ** 2)

dw_stat = durbin_watson(residuals)


# ══════════════════════════════════════════════════════════════
# 6. 다중공선성 지표 (설계 행렬의 조건 수)
# ══════════════════════════════════════════════════════════════
X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
cond_number = np.linalg.cond(X_train_aug)


# ══════════════════════════════════════════════════════════════
# 7. 결과 파일 저장
# ══════════════════════════════════════════════════════════════
sep = "=" * 55
report_path = os.path.join(out_dir, "regression_metrics.txt")

lines = []
lines.append(f"\n{sep}")
lines.append(f"  선형 회귀 종합 평가  ({SITE},  AEC→TAMA)")
lines.append(sep)

lines.append("\n[기본 회귀 지표 — Test set]")
lines.append(f"  R²              : {r2_test:.4f}")
lines.append(f"  Adjusted R²     : {adj_r2:.4f}  (p={n_features} 패널티 반영)")
lines.append(f"  R² (Train)      : {r2_train:.4f}  (과적합 확인용)")
lines.append(f"  MAE             : {mae_val:.4f}")
lines.append(f"  MSE             : {mse_val:.4f}")
lines.append(f"  RMSE            : {rmse_val:.4f}")
lines.append(f"  Median AE       : {medae_val:.4f}")
lines.append(f"  Max Error       : {maxe_val:.4f}")
lines.append(f"  MAPE            : {mape_val:.2f} %")
lines.append(f"  sMAPE           : {smape_val:.2f} %")
lines.append(f"  Explained Var.  : {evs_val:.4f}")
lines.append(f"  Bias (mean res.): {bias_val:.4f}  (0에 가까울수록 비편향)")
lines.append(f"  Residual Std    : {std_res:.4f}")

lines.append("\n[5-Fold 교차검증]")
lines.append(f"  CV R²           : {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
lines.append(f"  CV MAE          : {cv_mae_mean:.4f}")
lines.append(f"  CV RMSE         : {cv_rmse_mean:.4f}")

lines.append("\n[상관 분석]")
lines.append(f"  Pearson  r      : {pearson_r:.4f}   p={pearson_p:.4e}")
lines.append(f"  Spearman ρ      : {spearman_r:.4f}   p={spearman_p:.4e}")

lines.append("\n[잔차 정규성 검정]")
lines.append(f"  Shapiro-Wilk    : W={sw_stat:.4f},  p={sw_p:.4e}  "
             + ("→ 정규" if sw_p > 0.05 else "→ 비정규 (주의)"))
lines.append(f"  Jarque-Bera     : stat={jb_stat:.4f}, p={jb_p:.4e}  "
             + ("→ 정규" if jb_p > 0.05 else "→ 비정규 (주의)"))

lines.append("\n[잔차 자기상관]")
lines.append(f"  Durbin-Watson   : {dw_stat:.4f}  (2.0 근방이면 자기상관 없음)")

lines.append("\n[다중공선성]")
lines.append(f"  조건 수 (κ)     : {cond_number:.2e}  "
             "(30 이상 주의 / 1000 이상 심각)")
lines.append(sep + "\n")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[저장] {report_path}")


# ══════════════════════════════════════════════════════════════
# 8. 시각화 (4-panel figure)
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 11))
gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── 패널 1: 실제 vs 예측 ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
vmin, vmax = min(y.min(), y_pred_test.min()), max(y.max(), y_pred_test.max())
ax1.scatter(y_test, y_pred_test, alpha=0.7, edgecolors="k", linewidths=0.4, s=50)
ax1.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.5, label="완벽한 예측")
ax1.set_xlabel("실제 TAMA")
ax1.set_ylabel("예측 TAMA")
ax1.set_title(f"실제 vs 예측\nR²={r2_test:.3f}, RMSE={rmse_val:.3f}")
ax1.legend(fontsize=8)

# ── 패널 2: 잔차 vs 예측값 ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_pred_test, residuals, alpha=0.7, edgecolors="k", linewidths=0.4, s=50)
ax2.axhline(0, color="r", linestyle="--", lw=1.5)
ax2.axhline( 2 * std_res, color="orange", linestyle=":", lw=1.2, label="±2σ")
ax2.axhline(-2 * std_res, color="orange", linestyle=":", lw=1.2)
ax2.set_xlabel("예측 TAMA")
ax2.set_ylabel("잔차 (실제 − 예측)")
ax2.set_title(f"잔차 vs 예측값\nBias={bias_val:.3f}, DW={dw_stat:.3f}")
ax2.legend(fontsize=8)

# ── 패널 3: Q-Q 플롯 ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
(osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
ax3.plot(osm, osr, "o", alpha=0.7, markersize=4, markeredgewidth=0.4,
         markeredgecolor="k")
ax3.plot(osm, slope * np.array(osm) + intercept, "r--", lw=1.5)
ax3.set_xlabel("이론적 분위수 (Normal)")
ax3.set_ylabel("표본 분위수")
ax3.set_title(f"잔차 Q-Q 플롯\nShapiro-Wilk p={sw_p:.3e}")

# ── 패널 4: 잔차 히스토그램 ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(residuals, bins="auto", edgecolor="k", alpha=0.75, density=True)
xr = np.linspace(residuals.min(), residuals.max(), 200)
ax4.plot(xr, stats.norm.pdf(xr, bias_val, std_res),
         "r-", lw=1.8, label="정규분포 피팅")
ax4.set_xlabel("잔차")
ax4.set_ylabel("밀도")
ax4.set_title(f"잔차 분포\nJarque-Bera p={jb_p:.3e}")
ax4.legend(fontsize=8)

fig.suptitle(
    f"선형 회귀 종합 평가  [{SITE}]  AEC(128-dim) → TAMA\n"
    f"CV R²={cv_r2_mean:.3f}±{cv_r2_std:.3f}  |  "
    f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e})  |  "
    f"Spearman ρ={spearman_r:.3f}",
    fontsize=11, y=1.01,
)

panel_path = os.path.join(out_dir, "regression_full_eval.png")
fig.savefig(panel_path, dpi=150, bbox_inches="tight")
print(f"[저장] {panel_path}")

# ── 패널 5: 상위 20개 계수 크기 ───────────────────────────────
coef = model.coef_.ravel()
top_idx = np.argsort(np.abs(coef))[-20:][::-1]

fig2, ax5 = plt.subplots(figsize=(10, 5))
colors = ["steelblue" if c > 0 else "tomato" for c in coef[top_idx]]
ax5.bar(range(20), coef[top_idx], color=colors, edgecolor="k", linewidth=0.5)
ax5.set_xticks(range(20))
ax5.set_xticklabels([f"AEC[{i}]" for i in top_idx], rotation=45, ha="right", fontsize=8)
ax5.axhline(0, color="k", lw=0.8)
ax5.set_ylabel("회귀 계수")
ax5.set_title("상위 20개 AEC 계수 (|β| 기준)\n(파랑=양수 기여, 빨강=음수 기여)")
fig2.tight_layout()

coef_path = os.path.join(out_dir, "regression_coef_top20.png")
fig2.savefig(coef_path, dpi=150, bbox_inches="tight")
print(f"[저장] {coef_path}")

plt.close("all")
