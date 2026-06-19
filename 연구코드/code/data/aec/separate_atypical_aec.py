import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

EXCEL_PATH = r"c:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\강남\강남_liver_merged_features_ok.xlsx"
OUT_CSV = r"c:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\강남\aec_128_atypical_flag.csv"

# ── 데이터 로드 ──────────────────────────────────────────────
df = pd.read_excel(EXCEL_PATH, sheet_name="aec_128")
aec_cols = [f"aec_{i}" for i in range(1, 129)]
x = np.arange(1, 129)

X = df[aec_cols].values.astype(float)

# ── 형상 비교: z-score 정규화 후 평균 곡선과의 피어슨 상관계수 ──
X_norm = zscore(X, axis=1)
mean_curve = X_norm.mean(axis=0)
corrs = np.array([np.corrcoef(X_norm[i], mean_curve)[0, 1] for i in range(len(X_norm))])

Q1 = np.percentile(corrs, 25)
IQR = np.percentile(corrs, 75) - Q1
threshold = Q1 - 1.5 * IQR  # ≈ 0.14

df["corr_with_mean"] = corrs
df["atypical"] = corrs < threshold

typical_mask = ~df["atypical"].values
atypical_mask = df["atypical"].values

print(f"Threshold (Q1 - 1.5×IQR): {threshold:.4f}")
print(f"Typical  : {typical_mask.sum()}명")
print(f"Atypical : {atypical_mask.sum()}명")

# ── 결과 저장 ────────────────────────────────────────────────
df[["PatientID", "n_slices_cropped", "z_range", "SMI", "corr_with_mean", "atypical"]].to_csv(
    OUT_CSV, index=False, encoding="utf-8-sig"
)
print(f"저장 완료: {OUT_CSV}")

# ── 개요 그래프: 전체 형상 분포 ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
for i in range(len(X_norm)):
    if typical_mask[i]:
        ax.plot(x, X_norm[i], color="steelblue", alpha=0.05, linewidth=0.5)
for i in range(len(X_norm)):
    if atypical_mask[i]:
        ax.plot(x, X_norm[i], color="tomato", alpha=0.3, linewidth=0.7)
ax.plot(x, mean_curve, color="navy", linewidth=2.0, label="Mean (typical)")
ax.set_title(f"All Curves  |  typical={typical_mask.sum()}  atypical={atypical_mask.sum()}\n"
             f"Blue=typical  Red=atypical  threshold={threshold:.4f}")
ax.set_xlabel("AEC Index")
ax.set_ylabel("Z-score")
ax.set_xlim(1, 128)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend()

ax = axes[1]
ax.hist(corrs, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(threshold, color="tomato", linewidth=2, linestyle="--", label=f"threshold={threshold:.4f}")
ax.set_title("Correlation with Mean Curve Distribution")
ax.set_xlabel("Pearson Correlation")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.show()
plt.close(fig)

# ── 비정형 환자별 그래프 ─────────────────────────────────────
atypical_df = df[atypical_mask].reset_index(drop=True)

for idx, row in atypical_df.iterrows():
    patient_id = row["PatientID"]
    smi = row["SMI"]
    n_slices = row["n_slices_cropped"]
    z_range = row["z_range"]
    corr = row["corr_with_mean"]
    aec_values = row[aec_cols].values
    aec_norm = zscore(aec_values)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    # 원본 값
    ax = axes[0]
    ax.plot(x, aec_values, color="tomato", linewidth=1.4, label="Patient (raw)")
    ax.set_title(f"[{idx+1}/{len(atypical_df)}] PatientID: {patient_id}  |  SMI: {smi:.2f}  "
                 f"|  slices: {n_slices}  |  z_range: {z_range}")
    ax.set_xlabel("AEC Index")
    ax.set_ylabel("AEC Value")
    ax.set_xlim(1, 128)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # z-score 정규화 후 평균과 비교
    ax = axes[1]
    for i in np.where(typical_mask)[0][:200]:  # 전형 환자 200명 배경
        ax.plot(x, X_norm[i], color="steelblue", alpha=0.04, linewidth=0.5)
    ax.plot(x, mean_curve, color="navy", linewidth=2.0, label="Mean curve")
    ax.plot(x, aec_norm, color="tomato", linewidth=1.6, label=f"Patient (corr={corr:.3f})")
    ax.set_title("Shape Comparison (Z-score normalized)")
    ax.set_xlabel("AEC Index")
    ax.set_ylabel("Z-score")
    ax.set_xlim(1, 128)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close(fig)
