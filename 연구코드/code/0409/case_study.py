import ast
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

"""
Input Case:
  Case 1 : [PatientSex, PatientAge]
  Case 2 : [PatientSex, PatientAge, AEC(128-dim)]
  Case 3 : [PatientSex, PatientAge, AEC(128-dim), ManufacturerModelName]
"""

# ── 한글 폰트 설정 ─────────────────────────────────────────────
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ──────────────────────────────────────────────────
SITE = "신촌"
excel_path = rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\Results\{SITE}_final.xlsx"
out_dir    = rf"C:\Users\user\Desktop\Study\result\0409\{SITE}"
os.makedirs(out_dir, exist_ok=True)

# ── 데이터 로드 ────────────────────────────────────────────────
df   = pd.read_excel(excel_path)
aec  = np.array(df["AEC"].apply(ast.literal_eval).tolist())   # (N, 128)
tama = df["TAMA"].values                                        # (N,)

# ── 범주형 인코딩 ──────────────────────────────────────────────
le_sex = LabelEncoder()
le_mfr = LabelEncoder()
sex_enc = le_sex.fit_transform(df["PatientSex"]).reshape(-1, 1)   # M/F → 0/1
mfr_enc = le_mfr.fit_transform(df["ManufacturerModelName"]).reshape(-1, 1)
age     = df["PatientAge"].values.reshape(-1, 1)

# ── 케이스별 입력 특징 구성 ────────────────────────────────────
cases = {
    "Case 1\n[Sex, Age]":
        np.hstack([sex_enc, age]),
    "Case 2\n[Sex, Age, AEC]":
        np.hstack([sex_enc, age, aec]),
    "Case 3\n[Sex, Age, AEC, Manufacturer]":
        np.hstack([sex_enc, age, aec, mfr_enc]),
}
case_labels = list(cases.keys())
case_labels_short = ["Case 1 [Sex, Age]", "Case 2 [Sex, Age, AEC]", "Case 3 [Sex, Age, AEC, Manufacturer]"]

# ── TAMA → 4분위 레이블 (분류용) ──────────────────────────────
q1, q2, q3 = np.percentile(tama, [25, 50, 75])
y_cls = np.digitize(tama, bins=[q1, q2, q3])   # 0,1,2,3
n_classes = 4


# ══════════════════════════════════════════════════════════════
# 1. Ridge 회귀 — 5-Fold CV
# ══════════════════════════════════════════════════════════════
kf = KFold(n_splits=5, shuffle=True, random_state=42)

reg_metrics = {k: {} for k in case_labels}

for label, X_raw in cases.items():
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    cv = cross_validate(
        Ridge(alpha=1.0), X, tama, cv=kf,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
    )
    reg_metrics[label]["R²"]          = cv["test_r2"]
    reg_metrics[label]["MAE"]         = -cv["test_neg_mean_absolute_error"]
    reg_metrics[label]["RMSE"]        = -cv["test_neg_root_mean_squared_error"]


# ══════════════════════════════════════════════════════════════
# 2. 로지스틱 회귀 (다항) — 5-Fold Stratified CV
# ══════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf_metrics = {k: {} for k in case_labels}

for label, X_raw in cases.items():
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    cv = cross_validate(
        LogisticRegression(multi_class="multinomial", solver="lbfgs",
                           max_iter=1000, random_state=42),
        X, y_cls, cv=skf,
        scoring=["accuracy", "balanced_accuracy", "f1_macro",
                 "roc_auc_ovr_weighted"],
    )
    clf_metrics[label]["Accuracy"]      = cv["test_accuracy"]
    clf_metrics[label]["Bal. Accuracy"] = cv["test_balanced_accuracy"]
    clf_metrics[label]["F1 (macro)"]    = cv["test_f1_macro"]
    clf_metrics[label]["AUC (weighted)"]= cv["test_roc_auc_ovr_weighted"]


# ══════════════════════════════════════════════════════════════
# 3. 결과 파일 저장
# ══════════════════════════════════════════════════════════════
sep  = "=" * 65
sep2 = "-" * 65
report_path = os.path.join(out_dir, "case_study_metrics.txt")

lines = []
lines.append(f"\n{sep}")
lines.append(f"  케이스 스터디  [{SITE}]  입력 특징 조합 비교")
lines.append(f"  4분위 경계: Q1={q1:.1f}, Q2={q2:.1f}, Q3={q3:.1f}")
lines.append(f"  샘플 수: {len(tama)}")
lines.append(sep)

# ── 회귀 테이블 ───────────────────────────────────────────────
lines.append("\n[Ridge 회귀 — 5-Fold CV]  (mean ± std)")
lines.append(f"  {'케이스':<38} {'R²':>12} {'MAE':>12} {'RMSE':>12}")
lines.append("  " + sep2)
for label, m in reg_metrics.items():
    name = label.replace("\n", " ")
    r2   = m["R²"]
    mae  = m["MAE"]
    rmse = m["RMSE"]
    lines.append(
        f"  {name:<38} "
        f"{r2.mean():>6.4f}±{r2.std():.4f}  "
        f"{mae.mean():>6.2f}±{mae.std():.2f}  "
        f"{rmse.mean():>6.2f}±{rmse.std():.2f}"
    )

# ── 분류 테이블 ───────────────────────────────────────────────
lines.append("\n[로지스틱 회귀 (TAMA 4분위) — 5-Fold Stratified CV]  (mean ± std)")
lines.append(f"  {'케이스':<38} {'Acc':>10} {'BalAcc':>10} {'F1':>10} {'AUC':>10}")
lines.append("  " + sep2)
for label, m in clf_metrics.items():
    name = label.replace("\n", " ")
    acc  = m["Accuracy"]
    bacc = m["Bal. Accuracy"]
    f1   = m["F1 (macro)"]
    auc  = m["AUC (weighted)"]
    lines.append(
        f"  {name:<38} "
        f"{acc.mean():.4f}±{acc.std():.4f}  "
        f"{bacc.mean():.4f}±{bacc.std():.4f}  "
        f"{f1.mean():.4f}±{f1.std():.4f}  "
        f"{auc.mean():.4f}±{auc.std():.4f}"
    )

lines.append(f"\n{sep}\n")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[저장] {report_path}")


# ══════════════════════════════════════════════════════════════
# 4. 시각화
# ══════════════════════════════════════════════════════════════
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

def _bar_group(ax, means, stds, ylabel, title, ylim_top=None):
    x = np.arange(len(case_labels_short))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=COLORS, edgecolor="k", linewidth=0.6, alpha=0.85,
                  error_kw=dict(elinewidth=1.2, ecolor="black"))
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (stds.max() * 0.05),
                f"{m:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels_short, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    if ylim_top:
        ax.set_ylim(0, ylim_top)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

# ── Figure 1: 회귀 지표 비교 (3×1) ───────────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(13, 4))
reg_items = [
    ("R²",   "R²",   1.05),
    ("MAE",  "MAE",  None),
    ("RMSE", "RMSE", None),
]
for ax, (key, ylabel, ylim_top) in zip(axes1, reg_items):
    means = np.array([reg_metrics[l][key].mean() for l in case_labels])
    stds  = np.array([reg_metrics[l][key].std()  for l in case_labels])
    _bar_group(ax, means, stds, ylabel,
               f"Ridge 회귀 — {ylabel} (5-Fold CV)", ylim_top)

fig1.suptitle(f"케이스별 Ridge 회귀 성능 비교  [{SITE}]", fontsize=12, y=1.02)
fig1.tight_layout()
reg_path = os.path.join(out_dir, "case_study_regression.png")
fig1.savefig(reg_path, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"[저장] {reg_path}")

# ── Figure 2: 분류 지표 비교 (4×1) ───────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
clf_items = [
    ("Accuracy",       "Accuracy"),
    ("Bal. Accuracy",  "Balanced Accuracy"),
    ("F1 (macro)",     "F1 (macro)"),
    ("AUC (weighted)", "AUC (weighted)"),
]
for ax, (key, ylabel) in zip(axes2, clf_items):
    means = np.array([clf_metrics[l][key].mean() for l in case_labels])
    stds  = np.array([clf_metrics[l][key].std()  for l in case_labels])
    _bar_group(ax, means, stds, ylabel,
               f"로지스틱 회귀 — {ylabel}\n(5-Fold Stratified CV)", ylim_top=1.05)

fig2.suptitle(f"케이스별 로지스틱 회귀 성능 비교  [{SITE}]  (TAMA 4분위)", fontsize=12, y=1.02)
fig2.tight_layout()
clf_path = os.path.join(out_dir, "case_study_logistic.png")
fig2.savefig(clf_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"[저장] {clf_path}")

# ── Figure 3: Fold별 R² / Accuracy 추이 비교 ─────────────────
fig3, (ax_r, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
folds = np.arange(1, 6)

for label, color, short in zip(case_labels, COLORS, case_labels_short):
    ax_r.plot(folds, reg_metrics[label]["R²"], marker="o", color=color,
              label=short, lw=1.6)
    ax_a.plot(folds, clf_metrics[label]["Accuracy"], marker="o", color=color,
              label=short, lw=1.6)

for ax, title, ylabel in [
    (ax_r, "Ridge 회귀 — Fold별 R²",          "R²"),
    (ax_a, "로지스틱 — Fold별 Accuracy",       "Accuracy"),
]:
    ax.set_xticks(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.5)

fig3.suptitle(f"케이스별 Fold 안정성 비교  [{SITE}]", fontsize=12)
fig3.tight_layout()
fold_path = os.path.join(out_dir, "case_study_fold_stability.png")
fig3.savefig(fold_path, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"[저장] {fold_path}")
