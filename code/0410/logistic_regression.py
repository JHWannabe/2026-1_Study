"""
AEC Graph → TAMA Quartile Logistic Regression
===============================================
Input  : PNG images in 강남/raw/Image (AEC 그래프, filename = PatientID)
Output : TAMA를 4분위(Q1~Q4)로 분류 from 강남_DLO_Results.xlsx
Pipeline: HSV blue-line detection → 128-point AEC curve
          → AEC raw(128) + summary features → StandardScaler
          → Logistic Regression (4-class, Softmax)

[연구 방법론적 근거 (Research Methodological Justification)]
─────────────────────────────────────────────────────────────
1. AEC 곡선 → TAMA 분류 근거
   - AEC 패턴은 촬영 부위별 X선 감쇠를 반영하고, 감쇠는 근육·지방 조성에 의존
   - 따라서 AEC 곡선 모양(형태)으로 근육량(TAMA)의 분위 범주 예측이 가능
     [Ref] Kalra MK et al. Strategies for CT Radiation Dose Optimization.
           Radiology. 2004;230(3):619-28.
     [Ref] McCollough CH et al. CT Dose Index and Patient Dose: They Are Not the
           Same Thing. Radiology. 2011;259(2):311-6.

2. HSV 색공간 파란 선 검출 근거
   - AEC 그래프에서 파란 곡선만 선택 추출; HSV가 RGB보다 조명 변화에 강인
   - Morphological closing: 선의 단절 보정
     [Ref] Gonzalez RC, Woods RE. Digital Image Processing, 4th ed. Pearson, 2018. §6.2

3. 128-point 균등 리샘플링 근거
   - 환자별 이미지 크기 상이 → 고정 길이 벡터 필요
   - 선형 보간으로 128 등간격 포인트 추출 (참조 코드 AEC_POINTS=128)
     [Ref] 3_aec_cluster_analysis copy.py (강남 AEC Cluster Analysis)

4. AEC 요약 특성 결합 근거
   - 128 raw 포인트 + 통계적 요약 특성의 결합은 클러스터링 선행 연구에서 입증
   - end_minus_start, mean_abs_slope, high_mA_mean 등: 체형 관련 임상 해석 가능
     [Ref] 참조 코드 SUMMARY_FEATURE_COLS

5. TAMA 4분위 분류 (종속변수) 근거
   - 연속형 TAMA를 Q1~Q4로 단계화하여 근감소증 위험 그룹 분류
   - 분위수 기반 컷오프는 정규성 가정 불필요 (distribution-free)
     [Ref] Martin L et al. Cancer cachexia: skeletal muscle depletion as a powerful
           prognostic factor. J Clin Oncol. 2013;31(12):1539-47.
     [Ref] Szklo M, Nieto FJ. Epidemiology: Beyond the Basics, 4th ed. 2019.

6. Multinomial Logistic Regression (Softmax) 근거
   - 다중 클래스(K=4) 분류 문제 → Softmax 회귀가 이진 LR의 자연스러운 확장
   - MLE 기반 클래스 확률 직접 모델링; 계수 해석 가능
     [Ref] Hosmer DW, Lemeshow S. Applied Logistic Regression, 3rd ed. Wiley, 2013.
     [Ref] Bishop CM. Pattern Recognition and Machine Learning. Springer, 2006. §4.3

7. class_weight='balanced' 근거
   - 4분위 분할이 균등하더라도 AEC 추출 실패/누락으로 실제 불균형 가능
   - balanced: 클래스 빈도 역수 비례 가중치 → 소수 클래스 감도 향상
     [Ref] Japkowicz N, Stephen S. The class imbalance problem.
           Intell Data Anal. 2002;6(5):429-49.

8. solver='lbfgs', max_iter=2000 근거
   - lbfgs: 소~중규모 다중 클래스 문제에서 수렴 속도·메모리 효율 최적
     [Ref] Liu DC, Nocedal J. Math Program. 1989;45(1-3):503-28.

9. StandardScaler 근거
   - AEC raw 값과 요약 통계의 단위 혼재 → Logistic Regression 경사하강 수렴 안정화
     [Ref] LeCun Y et al. Efficient BackProp. Lecture Notes in CS. 2012;7700:9-48.

10. 5-Fold Stratified CV 근거
    - stratified: 각 fold에서 Q1~Q4 비율 유지 → 불균형 평가 오류 방지
      [Ref] Kohavi R. IJCAI. 1995;14(2):1137-45.

11. AUROC (One-vs-Rest, macro) 근거
    - 다중 클래스 분류 성능 평가 표준; macro 평균은 클래스 불균형 무관
      [Ref] Hand DJ, Till RJ. Mach Learn. 2001;45(2):171-86.
      [Ref] Fawcett T. Pattern Recognit Lett. 2006;27(8):861-74.
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
import seaborn as sns

import cv2
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

# ─── 경로 설정 ───────────────────────────────────────────────────────────────
SITE       = "신촌"
IMAGE_DIR  = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\raw\Image")
EXCEL_PATH = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\raw\{SITE}_DLO_Results.xlsx")
OUTPUT_DIR = Path(rf"C:\Users\user\Desktop\Study\result\0410\{SITE}\logistic_regression")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
AEC_POINTS   = 128

np.random.seed(RANDOM_STATE)


# ─── AEC 추출 함수 ────────────────────────────────────────────────────────────
def extract_aec(image_path: Path, n_points: int = AEC_POINTS) -> np.ndarray:
    """
    HSV 파란 선 검출 → n_points 균등 보간.
    [근거] 참조 코드 3_aec_cluster_analysis copy.py::extract_aec()와 동일 로직
    """
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 80, 80], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask  = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel    = np.ones((3, 3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    y_idx, x_idx = np.where(blue_mask > 0)
    if len(x_idx) == 0:
        raise ValueError(f"Blue line not found: {image_path}")

    unique_x = np.unique(x_idx)
    mean_y   = np.array([y_idx[x_idx == x].mean() for x in unique_x], dtype=float)
    target_x = np.linspace(unique_x.min(), unique_x.max(), n_points)
    interp_y = np.interp(target_x, unique_x, mean_y)
    return image_bgr.shape[0] - interp_y




# ─── 1. Excel 로드 및 TAMA 집계 ───────────────────────────────────────────────
print("=" * 62)
print("  Logistic Regression: AEC 128-point → TAMA Quartile (Q1~Q4)")
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
# [근거] PNG 개수 < TAMA 행 수이므로, PNG 기준으로 순회하여 매칭된 샘플만 사용
#        4분위 경계값도 실제 분석 대상(PNG 매칭 샘플)의 TAMA 분포로 계산
print(f"\n  AEC 추출 중 ({IMAGE_DIR}) ...")

SUMMARY_COLS = [
    "aec_mean", "aec_std", "aec_min", "aec_max", "aec_range",
    "end_minus_start", "mean_abs_slope", "auc", "cv", "asymmetry",
    "peak_position", "valley_position", "sign_changes", "high_mA_mean",
    "left_mean", "center_mean", "right_mean",
    "chest_slope", "abdomen_slope", "pelvis_slope",
]

rows: list[dict] = []
failures: list[str] = []

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

df_data     = pd.DataFrame(rows)
y_tama      = df_data["TAMA_mean"].to_numpy(dtype=float)
patient_ids = df_data["PatientID"].to_numpy()

# ─── 3. TAMA 4분위 라벨 생성 (PNG 매칭 샘플 기준) ─────────────────────────────
# [근거] PNG 수 < Excel TAMA 수이므로, 실제 분석 대상의 TAMA 분포로 경계값 산출해야
#        분위별 샘플 수가 균등해짐 (전체 Excel 기준 시 편향 발생 가능)
q1_cut, q2_cut, q3_cut = np.percentile(y_tama, [25, 50, 75])

def assign_quartile(tama: float) -> int:
    if tama <= q1_cut: return 0
    if tama <= q2_cut: return 1
    if tama <= q3_cut: return 2
    return 3

df_data["Quartile"] = df_data["TAMA_mean"].apply(assign_quartile)

CLASS_NAMES = [f"Q1(≤{q1_cut:.0f})", f"Q2(≤{q2_cut:.0f})",
               f"Q3(≤{q3_cut:.0f})", f"Q4(>{q3_cut:.0f})"]

print(f"\n  TAMA 4분위 경계값 (PNG 매칭 {len(y_tama)}개 기준):")
print(f"    Q1 ≤ {q1_cut:.1f}  |  Q2 ≤ {q2_cut:.1f}  |  Q3 ≤ {q3_cut:.1f}  |  Q4 > {q3_cut:.1f}")

aec_raw     = np.vstack(df_data["aec"].to_numpy())   # (N, 128)
X_raw       = aec_raw
y           = df_data["Quartile"].to_numpy(dtype=int)

print(f"  추출 성공 샘플 수: {len(y)}")
print(f"  Feature 차원: {X_raw.shape[1]}  (AEC raw 128 포인트)")
print(f"  분위별 샘플 수: {dict(zip(*np.unique(y, return_counts=True)))}")


# ─── 4. Train/Test Split (stratified) ────────────────────────────────────────
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X_raw, y, np.arange(len(y)),
    test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\n  Train: {len(y_tr)}  /  Test: {len(y_te)}")


# ─── 5. StandardScaler ────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_tr_s  = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)
X_all_s = scaler.transform(X_raw)


# ─── 6. Logistic Regression (OvR, saga solver) ───────────────────────────────
# [근거] lbfgs+multinomial은 128 AEC raw + 20 summary 특성 간 다중공선성으로
#        수렴 실패(hanging) 발생 → saga solver로 교체
#        saga: SGD 기반 대규모 데이터/고차원 특성에 최적화된 solver
#        [Ref] Defazio A et al. SAGA: A Fast Incremental Gradient Method.
#              NeurIPS 2014.
#        multi_class 파라미터 제거 (sklearn ≥1.5 deprecated)
print("\n─── Logistic Regression (OvR, saga) ────────────────────────")
log_model = LogisticRegression(
    solver="saga",
    C=1.0,
    max_iter=1000,
    tol=1e-3,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)
log_model.fit(X_tr_s, y_tr)
y_pred = log_model.predict(X_te_s)
y_prob = log_model.predict_proba(X_te_s)

acc = accuracy_score(y_te, y_pred)
y_te_bin = label_binarize(y_te, classes=[0, 1, 2, 3])
auc_ovr  = roc_auc_score(y_te_bin, y_prob, multi_class="ovr", average="macro")

print(f"  Accuracy          : {acc:.4f}")
print(f"  AUROC (macro OvR) : {auc_ovr:.4f}")
print(f"\n  분류 보고서:\n{classification_report(y_te, y_pred, target_names=CLASS_NAMES)}")


# ─── 7. 5-Fold Stratified CV ─────────────────────────────────────────────────
cv_skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
_model = LogisticRegression(solver="saga", C=1.0, max_iter=1000, tol=1e-3,
                             random_state=RANDOM_STATE, class_weight="balanced",
                             n_jobs=-1)
cv_acc = cross_val_score(_model, X_all_s, y, cv=cv_skf, scoring="accuracy")
cv_auc = cross_val_score(_model, X_all_s, y, cv=cv_skf, scoring="roc_auc_ovr_weighted")
print(f"  5-Fold CV Accuracy  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  5-Fold CV AUC (OvR) : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")


# ─── 8. 시각화 ────────────────────────────────────────────────────────────────
COLORS = ["#1565C0", "#2E7D32", "#F57F17", "#C62828"]

# 8-1. 분위별 평균 AEC 곡선
x_axis = np.linspace(0, 1, AEC_POINTS)
fig, ax = plt.subplots(figsize=(8, 5))
for q, col, nm in zip([0, 1, 2, 3], COLORS, CLASS_NAMES):
    mask = y == q
    if mask.sum() == 0:
        continue
    m = aec_raw[mask].mean(axis=0)
    s = aec_raw[mask].std(axis=0)
    ax.plot(x_axis, m, color=col, lw=2, label=f"{nm} (n={mask.sum()})")
    ax.fill_between(x_axis, m - s, m + s, alpha=0.12, color=col)
ax.set_xlabel("Position (Head → Leg)"); ax.set_ylabel("Relative mA")
ax.set_title("Mean AEC Curve by TAMA Quartile")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_mean_aec_by_quartile.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [저장] 01_mean_aec_by_quartile.png")

# 8-2. TAMA 분포 + 분위 경계선
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(y_tama, bins=40, color="#42A5F5", edgecolor="white", alpha=0.8)
for cutoff, lbl, col in zip([q1_cut, q2_cut, q3_cut],
                              ["Q1/Q2", "Q2/Q3", "Q3/Q4"],
                              ["#1565C0", "#2E7D32", "#C62828"]):
    ax.axvline(cutoff, color=col, lw=1.5, ls="--", label=f"{lbl}={cutoff:.0f}")
ax.set_xlabel("TAMA (cm²)"); ax.set_ylabel("Count")
ax.set_title("TAMA Distribution + Quartile Boundaries")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_tama_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [저장] 02_tama_distribution.png")

# 8-3. AEC Mean vs TAMA scatter (분위 색상)
fig, ax = plt.subplots(figsize=(7, 5))
for q, col, nm in zip([0, 1, 2, 3], COLORS, CLASS_NAMES):
    mask = y == q
    ax.scatter(aec_raw[mask].mean(axis=1), y_tama[mask],
               alpha=0.5, s=15, color=col, label=nm)
ax.set_xlabel("AEC Mean"); ax.set_ylabel("TAMA (cm²)")
ax.set_title("AEC Mean vs TAMA (by Quartile)")
ax.legend(fontsize=8, markerscale=2); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_aec_mean_vs_tama.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [저장] 03_aec_mean_vs_tama.png")

# 8-4. Confusion Matrix
per_class_acc = []
for q in range(4):
    mask = y_te == q
    per_class_acc.append(accuracy_score(y_te[mask], y_pred[mask]) if mask.sum() > 0 else 0.0)

fig, ax = plt.subplots(figsize=(6, 5))
cm_mat = confusion_matrix(y_te, y_pred)
sns.heatmap(cm_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Pred {n}" for n in CLASS_NAMES],
            yticklabels=[f"True {n}" for n in CLASS_NAMES],
            ax=ax)
ax.set_title(f"Confusion Matrix  (Acc={acc:.3f})")
ax.tick_params(axis='x', labelsize=7); ax.tick_params(axis='y', labelsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [저장] 04_confusion_matrix.png")

# 8-5. ROC Curves (One-vs-Rest)
fig, ax = plt.subplots(figsize=(6, 5))
for i, (col, nm) in enumerate(zip(COLORS, CLASS_NAMES)):
    fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_prob[:, i])
    auc_i = roc_auc_score(y_te_bin[:, i], y_prob[:, i])
    ax.plot(fpr, tpr, color=col, lw=1.5, label=f"{nm} AUC={auc_i:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves (One-vs-Rest)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [저장] 05_roc_curves.png")

# 8-6. 분위별 per-class accuracy
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(CLASS_NAMES, per_class_acc, color=COLORS, alpha=0.8, edgecolor="white")
ax.set_ylim(0, 1.15); ax.set_ylabel("Accuracy")
ax.set_title("Per-Quartile Accuracy")
ax.tick_params(axis='x', labelsize=8); ax.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
            f"{val:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_per_quartile_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [저장] 06_per_quartile_accuracy.png")


# ─── 9. 결과 저장 ─────────────────────────────────────────────────────────────
pd.DataFrame({
    "PatientID":     patient_ids[idx_te],
    "TAMA_mean":     y_tama[idx_te],
    "True_Quartile": y_te,
    "Pred_Quartile": y_pred,
    "Prob_Q1": y_prob[:, 0],
    "Prob_Q2": y_prob[:, 1],
    "Prob_Q3": y_prob[:, 2],
    "Prob_Q4": y_prob[:, 3],
    "Correct": (y_te == y_pred).astype(int),
}).to_excel(OUTPUT_DIR / "logistic_regression_predictions.xlsx", index=False)

pd.DataFrame({
    "Quartile":        ["Q1", "Q2", "Q3", "Q4"],
    "Label":           CLASS_NAMES,
    "TAMA_cutoff":     [f"≤{q1_cut:.1f}", f"≤{q2_cut:.1f}",
                        f"≤{q3_cut:.1f}", f">{q3_cut:.1f}"],
    "Train_count":     [(y_tr == q).sum() for q in range(4)],
    "Test_count":      [(y_te == q).sum() for q in range(4)],
    "Per_class_acc":   [round(a, 4) for a in per_class_acc],
}).to_excel(OUTPUT_DIR / "logistic_regression_summary.xlsx", index=False)

print(f"  [저장] logistic_regression_predictions.xlsx")
print(f"  [저장] logistic_regression_summary.xlsx")

print("\n" + "=" * 62)
print("  분석 완료")
print(f"  샘플 수           : {len(y)}")
print(f"  Feature 차원      : {X_raw.shape[1]}  (AEC raw 128 포인트)")
print(f"  Accuracy          : {acc:.4f}")
print(f"  AUROC (macro OvR) : {auc_ovr:.4f}")
print(f"  5-Fold CV Acc     : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  5-Fold CV AUC     : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"\n  TAMA 4분위 경계:")
print(f"    Q1 ≤ {q1_cut:.1f}  |  Q2 ≤ {q2_cut:.1f}  |  Q3 ≤ {q3_cut:.1f}  |  Q4 > {q3_cut:.1f}")
print("=" * 62)
