import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.signal import find_peaks

EXCEL_PATH = r"c:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\강남\강남_liver_merged_features_ok.xlsx"
OUT_CSV    = r"c:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\강남\aec_128_peak_group.csv"

PEAK_PROMINENCE = 0.5
PEAK_DISTANCE   = 5

GROUP_LABELS = {0: "0 peaks (단조)", 1: "1 peak", 2: "2 peaks", 3: "3+ peaks"}
COLORS       = {0: "steelblue", 1: "tomato", 2: "seagreen", 3: "darkorchid"}

# ── 데이터 로드 ───────────────────────────────────────────────────────
df = pd.read_excel(EXCEL_PATH, sheet_name="aec_128")
aec_cols = [f"aec_{i}" for i in range(1, 129)]
x = np.arange(1, 129)

X = df[aec_cols].values.astype(float)
X_norm: np.ndarray = np.array(zscore(X, axis=1))

# ── 피크 수 기반 그룹 할당 ────────────────────────────────────────────
def count_peaks(curve: np.ndarray) -> int:
    peaks, _ = find_peaks(curve, prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE)
    return len(peaks)

peak_counts = np.array([count_peaks(X_norm[i]) for i in range(len(X_norm))])
# 3개 이상은 동일 그룹
groups = np.where(peak_counts >= 3, 3, peak_counts)

df["n_peaks"] = peak_counts
df["group"]   = groups

group_ids = sorted(GROUP_LABELS.keys())
counts    = {g: int((groups == g).sum()) for g in group_ids}

print(f"Peak 검출 기준: prominence={PEAK_PROMINENCE}, distance={PEAK_DISTANCE}")
for g in group_ids:
    print(f"  Group {g} [{GROUP_LABELS[g]}]: {counts[g]}명")

df[["PatientID", "n_slices_cropped", "z_range", "SMI", "n_peaks", "group"]].to_csv(
    OUT_CSV, index=False, encoding="utf-8-sig"
)
print(f"CSV 저장: {OUT_CSV}\n")

# ── [1] 개요 그래프 ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle(
    f"AEC-128  Peak-based Shape Groups  (prominence={PEAK_PROMINENCE})",
    fontsize=13, fontweight="bold"
)

# 전체 곡선 + 그룹 평균
ax = axes[0]
for g in group_ids:
    idx = np.where(groups == g)[0]
    for i in idx:
        ax.plot(x, X_norm[i], color=COLORS[g], alpha=0.05, linewidth=0.5)
for g in group_ids:
    idx = np.where(groups == g)[0]
    mean_g = X_norm[idx].mean(axis=0)
    ax.plot(x, mean_g, color=COLORS[g], linewidth=2.5,
            label=f"Group {g}: {GROUP_LABELS[g]}  (n={counts[g]})")
ax.set_title("All Curves  —  bold = group mean")
ax.set_xlabel("AEC Index"); ax.set_ylabel("Z-score"); ax.set_xlim(1, 128)
ax.grid(True, linestyle="--", alpha=0.3); ax.legend(fontsize=9)

# 그룹 평균 곡선만
ax = axes[1]
for g in group_ids:
    idx = np.where(groups == g)[0]
    mean_g = X_norm[idx].mean(axis=0)
    ax.plot(x, mean_g, color=COLORS[g], linewidth=2.5,
            label=f"Group {g}: {GROUP_LABELS[g]}  (n={counts[g]})")
ax.set_title("Group Mean Curves")
ax.set_xlabel("AEC Index"); ax.set_ylabel("Z-score"); ax.set_xlim(1, 128)
ax.grid(True, linestyle="--", alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.show()
plt.close(fig)


# ── [2] 그룹별 샘플 10개 Raw 값 그리드 출력 ─────────────────────────
N_SAMPLES = 10
NCOLS     = 5
NROWS     = N_SAMPLES // NCOLS   # = 2

rng = np.random.default_rng(seed=42)

for g in group_ids:
    group_idx = np.where(groups == g)[0]
    sample_idx = rng.choice(group_idx, size=min(N_SAMPLES, len(group_idx)), replace=False)

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(20, 7))
    fig.suptitle(
        f"Group {g}: {GROUP_LABELS[g]}  (n={counts[g]})  —  Random {len(sample_idx)} Samples  [Raw AEC Value]",
        fontsize=12, fontweight="bold"
    )

    for ax_idx, row_idx in enumerate(sample_idx):
        row     = df.iloc[row_idx]
        pid     = row["PatientID"]
        n_peaks = int(row["n_peaks"])
        raw     = X[row_idx]
        normed  = X_norm[row_idx]

        peaks, _ = find_peaks(normed, prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE)

        ax = axes[ax_idx // NCOLS, ax_idx % NCOLS]
        ax.plot(x, raw, color=COLORS[g], linewidth=1.2)
        if len(peaks):
            ax.scatter(x[peaks], raw[peaks], color="black", zorder=5, s=25)
        ax.set_title(f"ID:{pid}  peaks:{n_peaks}", fontsize=8)
        ax.set_xlim(1, 128)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    # 빈 subplot 숨기기
    for ax_idx in range(len(sample_idx), NROWS * NCOLS):
        axes[ax_idx // NCOLS, ax_idx % NCOLS].set_visible(False)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
