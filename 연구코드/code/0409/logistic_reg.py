import ast
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc,
    precision_score, recall_score, f1_score,
)
from scipy.stats import chi2_contingency

# ── 한글 폰트 설정 ─────────────────────────────────────────────
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ──────────────────────────────────────────────────
SITE = "강남"
excel_path = rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\Results\{SITE}_final.xlsx"
out_dir = rf"C:\Users\user\Desktop\Study\result\0409\{SITE}"
os.makedirs(out_dir, exist_ok=True)

# ── 데이터 로드 ────────────────────────────────────────────────
df = pd.read_excel(excel_path)
X_raw = np.array(df["AEC"].apply(ast.literal_eval).tolist())   # (N, 128)
tama  = df["TAMA"].values                                        # (N,)

n_samples, n_features = X_raw.shape

# ── TAMA → 4분위 레이블 ────────────────────────────────────────
# Q1=0(하위 25%), Q2=1, Q3=2, Q4=3(상위 25%)
q1, q2, q3 = np.percentile(tama, [25, 50, 75])
y = np.digitize(tama, bins=[q1, q2, q3])           # 0,1,2,3
class_names = ["Q1", "Q2", "Q3", "Q4"]
n_classes = 4

# ── Train/Test 분할 (Stratified) ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

# ── 모델 학습 (다항 로지스틱 회귀) ───────────────────────────
model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)
model.fit(X_train, y_train)
y_pred       = model.predict(X_test)
y_prob       = model.predict_proba(X_test)          # (N_test, 4)
y_train_pred = model.predict(X_train)


# ══════════════════════════════════════════════════════════════
# 1. 기본 분류 지표 (Test set)
# ══════════════════════════════════════════════════════════════
acc          = accuracy_score(y_test, y_pred)
bal_acc      = balanced_accuracy_score(y_test, y_pred)
kappa        = cohen_kappa_score(y_test, y_pred)
mcc          = matthews_corrcoef(y_test, y_pred)
train_acc    = accuracy_score(y_train, y_train_pred)

prec_macro   = precision_score(y_test, y_pred, average="macro",    zero_division=0)
rec_macro    = recall_score(y_test, y_pred,    average="macro",    zero_division=0)
f1_macro     = f1_score(y_test, y_pred,        average="macro",    zero_division=0)
prec_weight  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec_weight   = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
f1_weight    = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

# OvR ROC-AUC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
roc_auc_macro  = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
roc_auc_weight = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="weighted")


# ══════════════════════════════════════════════════════════════
# 2. 혼동 행렬 χ² 검정 (예측과 실제의 독립성 검정)
# ══════════════════════════════════════════════════════════════
cm = confusion_matrix(y_test, y_pred)
chi2_stat, chi2_p, chi2_dof, _ = chi2_contingency(cm)


# ══════════════════════════════════════════════════════════════
# 3. 5-Fold Stratified 교차검증
# ══════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    LogisticRegression(multi_class="multinomial", solver="lbfgs",
                       max_iter=1000, random_state=42),
    X_raw, y, cv=skf,
    scoring=["accuracy", "balanced_accuracy", "f1_macro",
             "roc_auc_ovr_weighted"],
    return_train_score=True,
)
cv_acc_mean  = cv_results["test_accuracy"].mean()
cv_acc_std   = cv_results["test_accuracy"].std()
cv_bacc_mean = cv_results["test_balanced_accuracy"].mean()
cv_f1_mean   = cv_results["test_f1_macro"].mean()
cv_auc_mean  = cv_results["test_roc_auc_ovr_weighted"].mean()


# ══════════════════════════════════════════════════════════════
# 4. 다중공선성 지표 (조건 수)
# ══════════════════════════════════════════════════════════════
cond_number = np.linalg.cond(X_train)


# ══════════════════════════════════════════════════════════════
# 5. 결과 파일 저장
# ══════════════════════════════════════════════════════════════
sep = "=" * 58
report_path = os.path.join(out_dir, "logistic_metrics.txt")

lines = []
lines.append(f"\n{sep}")
lines.append(f"  로지스틱 회귀 종합 평가  ({SITE},  AEC→TAMA 4분위)")
lines.append(sep)
lines.append(f"  샘플 수: {n_samples},  AEC 차원: {n_features}")
lines.append(f"  4분위 경계: Q1={q1:.3f}, Q2(중앙)={q2:.3f}, Q3={q3:.3f}")
lines.append("  클래스 분포: " +
             ", ".join(f"{class_names[i]}={int((y == i).sum())}" for i in range(n_classes)))
lines.append(f"  Train: {len(X_train)},  Test: {len(X_test)}")

lines.append("\n[기본 분류 지표 — Test set]")
lines.append(f"  Accuracy           : {acc:.4f}")
lines.append(f"  Balanced Accuracy  : {bal_acc:.4f}  (클래스 불균형 보정)")
lines.append(f"  Accuracy (Train)   : {train_acc:.4f}  (과적합 확인용)")
lines.append(f"  Cohen's Kappa      : {kappa:.4f}  (우연 일치 보정, >0.6이면 양호)")
lines.append(f"  MCC                : {mcc:.4f}  (다중클래스 균형 지표, [-1,1])")

lines.append("\n[Precision / Recall / F1]")
lines.append(f"  Macro  Precision   : {prec_macro:.4f}")
lines.append(f"  Macro  Recall      : {rec_macro:.4f}")
lines.append(f"  Macro  F1          : {f1_macro:.4f}")
lines.append(f"  Weight Precision   : {prec_weight:.4f}")
lines.append(f"  Weight Recall      : {rec_weight:.4f}")
lines.append(f"  Weight F1          : {f1_weight:.4f}")

lines.append("\n[ROC-AUC (One-vs-Rest)]")
lines.append(f"  Macro  AUC         : {roc_auc_macro:.4f}")
lines.append(f"  Weighted AUC       : {roc_auc_weight:.4f}")

lines.append("\n[Confusion Matrix χ² 검정]")
lines.append(f"  χ²={chi2_stat:.3f},  df={chi2_dof},  p={chi2_p:.4e}  "
             + ("→ 유의함 (예측이 실제와 연관)" if chi2_p < 0.05 else "→ 유의하지 않음"))

lines.append("\n[5-Fold Stratified 교차검증]")
lines.append(f"  CV Accuracy        : {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
lines.append(f"  CV Balanced Acc.   : {cv_bacc_mean:.4f}")
lines.append(f"  CV Macro F1        : {cv_f1_mean:.4f}")
lines.append(f"  CV Weighted AUC    : {cv_auc_mean:.4f}")

lines.append("\n[클래스별 상세 리포트]")
lines.append(classification_report(y_test, y_pred, target_names=class_names, digits=4))

lines.append("\n[다중공선성]")
lines.append(f"  조건 수 (κ)        : {cond_number:.2e}  "
             "(30 이상 주의 / 1000 이상 심각)")
lines.append(sep + "\n")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[저장] {report_path}")


# ══════════════════════════════════════════════════════════════
# 6. 시각화 (4-panel)
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 11))
gs  = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

# ── 패널 1: 혼동 행렬 ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax1, colorbar=False, cmap="Blues")
ax1.set_title(f"Confusion Matrix\nAcc={acc:.3f}, κ={kappa:.3f}")
ax1.set_xticklabels(class_names, rotation=20, ha="right", fontsize=8)
ax1.set_yticklabels(class_names, fontsize=8)

# ── 패널 2: ROC 곡선 (OvR, 클래스별) ─────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
colors_roc = ["steelblue", "darkorange", "green", "crimson"]
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc_i   = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color=colors_roc[i], lw=1.8,
             label=f"{class_names[i]} (AUC={roc_auc_i:.3f})")
ax2.plot([0, 1], [0, 1], "k--", lw=1)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title(f"ROC Curve (One-vs-Rest)\nmacro AUC={roc_auc_macro:.3f}")
ax2.legend(fontsize=7.5, loc="lower right")

# ── 패널 3: 예측 확률 분포 (박스플롯, 클래스별) ──────────────
ax3 = fig.add_subplot(gs[1, 0])
prob_data = [y_prob[y_test == i, i] for i in range(n_classes)]
bp = ax3.boxplot(prob_data, patch_artist=True, notch=False,
                 medianprops=dict(color="black", lw=2))
for patch, color in zip(bp["boxes"], colors_roc):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax3.set_xticklabels(class_names, fontsize=9)
ax3.set_ylabel("예측 확률 (정답 클래스)")
ax3.set_title("정답 클래스별 예측 확률 분포\n(높을수록 클래스 구별력 우수)")

# ── 패널 4: 상위 20개 계수 크기 (Q4 vs Q1 대비) ──────────────
ax4 = fig.add_subplot(gs[1, 1])
# 클래스 3(Q4) 대비 클래스 0(Q1) 계수 차이 → 어떤 AEC 밴드가 고분위 구분에 기여
coef_diff = model.coef_[3] - model.coef_[0]
top_idx   = np.argsort(np.abs(coef_diff))[-20:][::-1]
bar_colors = ["steelblue" if v > 0 else "tomato" for v in coef_diff[top_idx]]
ax4.bar(range(20), coef_diff[top_idx], color=bar_colors,
        edgecolor="k", linewidth=0.5)
ax4.set_xticks(range(20))
ax4.set_xticklabels([f"AEC[{i}]" for i in top_idx],
                    rotation=45, ha="right", fontsize=8)
ax4.axhline(0, color="k", lw=0.8)
ax4.set_ylabel("계수 차이 (Q4 − Q1)")
ax4.set_title("상위 20개 AEC 계수 (Q4 vs Q1)\n(파랑=Q4 기여, 빨강=Q1 기여)")

fig.suptitle(
    f"로지스틱 회귀 종합 평가  [{SITE}]  AEC(128-dim) → TAMA 4분위\n"
    f"CV Acc={cv_acc_mean:.3f}±{cv_acc_std:.3f}  |  "
    f"CV F1(macro)={cv_f1_mean:.3f}  |  "
    f"CV AUC(weighted)={cv_auc_mean:.3f}",
    fontsize=11, y=1.01,
)

panel_path = os.path.join(out_dir, "logistic_full_eval.png")
fig.savefig(panel_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[저장] {panel_path}")

# ── 패널 5: 5-Fold CV fold별 성능 추이 ───────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(11, 4))

folds = np.arange(1, 6)
for ax, key, label in zip(
    axes,
    ["test_accuracy", "test_f1_macro"],
    ["Accuracy", "Macro F1"],
):
    vals = cv_results[key]
    ax.bar(folds, vals, color="steelblue", alpha=0.75, edgecolor="k")
    ax.axhline(vals.mean(), color="r", linestyle="--", lw=1.5,
               label=f"평균={vals.mean():.3f}")
    ax.set_ylim(0, 1)
    ax.set_xticks(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel(label)
    ax.set_title(f"5-Fold CV {label}")
    ax.legend(fontsize=9)

fig2.suptitle(f"교차검증 Fold별 성능  [{SITE}]", fontsize=11)
fig2.tight_layout()

cv_path = os.path.join(out_dir, "logistic_cv_folds.png")
fig2.savefig(cv_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"[저장] {cv_path}")
