"""
Ablation Study: StandardScaler 유무 비교
=========================================
대상 파일:
  - linear_regression.py  (AEC 128pt → TAMA, Ridge)
  - logistic_regression.py (AEC 128pt → TAMA Quartile, Logistic)
  - case_study.py          (Tabular → TAMA, Ridge, 3 Cases)

각 모델을 WITH / WITHOUT StandardScaler로 동시 실행하여 성능 비교.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, roc_auc_score,
)

np.random.seed(42)

SITE         = "신촌"
RANDOM_STATE = 42
AEC_POINTS   = 128
RIDGE_ALPHA  = 1.0

IMAGE_DIR    = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\raw\Image")
DLO_PATH     = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\raw\{SITE}_DLO_Results.xlsx")
INFO_PATH    = Path(rf"C:\Users\user\Desktop\Study\data\AEC\{SITE}\raw\{SITE}_patient_info.xlsx")
OUTPUT_DIR   = Path(rf"C:\Users\user\Desktop\Study\result\0410\{SITE}\ablation_scaler")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 공통 유틸
# ═══════════════════════════════════════════════════════════════════════════════

def run_with_scaler(X: np.ndarray, use_scaler: bool,
                    X_tr: np.ndarray, X_te: np.ndarray) -> tuple:
    """use_scaler 플래그에 따라 스케일 적용 여부 결정."""
    if use_scaler:
        sc = StandardScaler()
        X_tr_s  = sc.fit_transform(X_tr)
        X_te_s  = sc.transform(X_te)
        X_all_s = sc.transform(X)
    else:
        X_tr_s  = X_tr.copy()
        X_te_s  = X_te.copy()
        X_all_s = X.copy()
    return X_tr_s, X_te_s, X_all_s


def extract_aec(image_path: Path, n_points: int = AEC_POINTS) -> np.ndarray:
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"Empty file: {image_path}")
    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Decode failed: {image_path}")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([100, 80, 80], dtype=np.uint8),
                       np.array([130, 255, 255], dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    y_idx, x_idx = np.where(mask > 0)
    if len(x_idx) == 0:
        raise ValueError(f"Blue line not found: {image_path}")
    unique_x = np.unique(x_idx)
    mean_y   = np.array([y_idx[x_idx == x].mean() for x in unique_x], dtype=float)
    return image_bgr.shape[0] - np.interp(
        np.linspace(unique_x.min(), unique_x.max(), n_points), unique_x, mean_y
    )




# ═══════════════════════════════════════════════════════════════════════════════
# AEC 데이터 로드 (linear / logistic 공통)
# ═══════════════════════════════════════════════════════════════════════════════

def load_aec_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X_raw, y_tama, y_quartile, patient_ids) 반환."""
    df_raw = pd.read_excel(DLO_PATH)
    df_raw["PatientID"] = df_raw["PatientID"].astype(str)
    df_raw["TAMA"] = pd.to_numeric(
        df_raw["TAMA"].astype(str).str.strip()
        .replace({"NO_LINK": np.nan, "N/A": np.nan, "na": np.nan,
                  "n/a": np.nan, "": np.nan, "nan": np.nan}),
        errors="coerce"
    )
    label_dict = (
        df_raw.groupby("PatientID")["TAMA"]
        .first().dropna()
        .to_dict()
    )

    rows = []
    for img_path in sorted(IMAGE_DIR.glob("*.png")):
        pid = img_path.stem
        if pid not in label_dict:
            continue
        try:
            aec = extract_aec(img_path)
            rows.append({"PatientID": pid, "TAMA_mean": label_dict[pid], "aec": aec})
        except Exception:
            pass

    df = pd.DataFrame(rows)
    aec_raw = np.vstack(df["aec"].to_numpy())   # (N, 128)
    X_raw   = aec_raw
    y_tama  = df["TAMA_mean"].to_numpy(dtype=float)

    q1, q2, q3 = np.percentile(y_tama, [25, 50, 75])
    y_q = np.array([0 if t <= q1 else 1 if t <= q2 else 2 if t <= q3 else 3
                    for t in y_tama], dtype=int)

    print(f"  [AEC 데이터] 샘플: {len(y_tama)}, Feature: {X_raw.shape[1]}")
    return X_raw, y_tama, y_q, df["PatientID"].to_numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Linear Regression Ablation
# ═══════════════════════════════════════════════════════════════════════════════

def ablation_linear(X_raw, y_tama):
    print("\n" + "=" * 60)
    print("  [1] Linear Regression – StandardScaler Ablation")
    print("=" * 60)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y_tama, test_size=0.2, random_state=RANDOM_STATE
    )
    KF = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for use_sc in [True, False]:
        label = "With Scaler" if use_sc else "Without Scaler"
        X_tr_s, X_te_s, X_all_s = run_with_scaler(X_raw, use_sc, X_tr, X_te)

        for alpha, mname in [(1e-8, "OLS-proxy"), (1.0, "Ridge(α=1.0)")]:
            model = Ridge(alpha=alpha, solver="sag",
                          max_iter=5000, random_state=RANDOM_STATE)
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)

            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2   = r2_score(y_te, y_pred)
            cv_r2 = cross_val_score(
                Ridge(alpha=alpha, solver="sag", max_iter=5000, random_state=RANDOM_STATE),
                X_all_s, y_tama, cv=KF, scoring="r2", n_jobs=1
            )
            records.append({
                "Scaler": label, "Model": mname,
                "RMSE": round(rmse, 4), "R2": round(r2, 4),
                "CV_R2_mean": round(cv_r2.mean(), 4),
                "CV_R2_std":  round(cv_r2.std(), 4),
            })
            print(f"  [{label}] {mname:16s}  RMSE={rmse:.3f}  R²={r2:.4f}"
                  f"  CV R²={cv_r2.mean():.4f}±{cv_r2.std():.4f}")

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Logistic Regression Ablation
# ═══════════════════════════════════════════════════════════════════════════════

def ablation_logistic(X_raw, y_q):
    print("\n" + "=" * 60)
    print("  [2] Logistic Regression – StandardScaler Ablation")
    print("=" * 60)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y_q, test_size=0.2, random_state=RANDOM_STATE, stratify=y_q
    )
    SKF = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for use_sc in [True, False]:
        label = "With Scaler" if use_sc else "Without Scaler"
        X_tr_s, X_te_s, X_all_s = run_with_scaler(X_raw, use_sc, X_tr, X_te)

        model = LogisticRegression(
            solver="saga", C=1.0, max_iter=1000, tol=1e-3,
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        )
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        y_prob = model.predict_proba(X_te_s)

        acc = accuracy_score(y_te, y_pred)
        y_te_bin = label_binarize(y_te, classes=[0, 1, 2, 3])
        auc = roc_auc_score(y_te_bin, y_prob, multi_class="ovr", average="macro")

        cv_acc = cross_val_score(
            LogisticRegression(solver="saga", C=1.0, max_iter=1000, tol=1e-3,
                               random_state=RANDOM_STATE, class_weight="balanced",
                               n_jobs=-1),
            X_all_s, y_q, cv=SKF, scoring="accuracy", n_jobs=1
        )
        records.append({
            "Scaler": label,
            "Accuracy": round(acc, 4), "AUROC": round(auc, 4),
            "CV_Acc_mean": round(cv_acc.mean(), 4),
            "CV_Acc_std":  round(cv_acc.std(), 4),
        })
        print(f"  [{label}]  Acc={acc:.4f}  AUC={auc:.4f}"
              f"  CV Acc={cv_acc.mean():.4f}±{cv_acc.std():.4f}")

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Case Study Ablation (Tabular)
# ═══════════════════════════════════════════════════════════════════════════════

def ablation_case_study():
    print("\n" + "=" * 60)
    print("  [3] Case Study (Tabular) – StandardScaler Ablation")
    print("=" * 60)

    df = pd.read_excel(INFO_PATH)
    df["TAMA"] = pd.to_numeric(
        df["TAMA"].astype(str).str.strip()
        .replace({"NO_LINK": np.nan, "N/A": np.nan, "na": np.nan,
                  "n/a": np.nan, "": np.nan, "nan": np.nan}),
        errors="coerce"
    )
    df_clean = df.dropna(subset=["TAMA"]).reset_index(drop=True)

    df_enc = df_clean.copy()
    for col in ["PatientSex", "Manufacturer"]:
        if col in df_enc.columns and df_enc[col].dtype == object:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    # aec_feature: NaN 보존하면서 non-NaN 값만 인코딩
    if "aec_feature" in df_enc.columns:
        mask = df_enc["aec_feature"].notna()
        if mask.any():
            df_enc.loc[mask, "aec_feature"] = LabelEncoder().fit_transform(
                df_enc.loc[mask, "aec_feature"].astype(str)
            )
            df_enc["aec_feature"] = pd.to_numeric(df_enc["aec_feature"], errors="coerce")

    CASES = {
        "Case 1: Sex+Age":           ["PatientSex", "PatientAge"],
        "Case 2: Sex+Age+AEC":       ["PatientSex", "PatientAge", "aec_feature"],
        "Case 3: Sex+Age+AEC+Mfr":  ["PatientSex", "PatientAge", "aec_feature", "Manufacturer"],
    }
    CASES = {n: [c for c in cols if c in df_enc.columns]
             for n, cols in CASES.items()}

    KF = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for case_name, feat_cols in CASES.items():
        df_case = df_enc.dropna(subset=feat_cols + ["TAMA"]).reset_index(drop=True)
        if len(df_case) < 10:
            print(f"  [SKIP] {case_name}: 샘플 수 부족 ({len(df_case)})")
            continue

        y = df_case["TAMA"].to_numpy(dtype=float)
        X = df_case[feat_cols].to_numpy(dtype=float)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        for use_sc in [True, False]:
            label = "With Scaler" if use_sc else "Without Scaler"
            X_tr_s, X_te_s, X_all_s = run_with_scaler(X, use_sc, X_tr, X_te)

            model = Ridge(alpha=RIDGE_ALPHA, solver="sag",
                          max_iter=5000, random_state=RANDOM_STATE)
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)

            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2   = r2_score(y_te, y_pred)
            cv_r2 = cross_val_score(
                Ridge(alpha=RIDGE_ALPHA, solver="sag",
                      max_iter=5000, random_state=RANDOM_STATE),
                X_all_s, y, cv=KF, scoring="r2", n_jobs=1
            )
            records.append({
                "Case": case_name, "Scaler": label,
                "RMSE": round(rmse, 4), "R2": round(r2, 4),
                "CV_R2_mean": round(cv_r2.mean(), 4),
                "CV_R2_std":  round(cv_r2.std(), 4),
            })
            print(f"  [{label}] {case_name}  RMSE={rmse:.3f}  R²={r2:.4f}"
                  f"  CV R²={cv_r2.mean():.4f}±{cv_r2.std():.4f}")

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 실행
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print(f"  Ablation Study: StandardScaler – {SITE}")
print("=" * 60)

X_raw, y_tama, y_q, _ = load_aec_dataset()

df_linear   = ablation_linear(X_raw, y_tama)
df_logistic = ablation_logistic(X_raw, y_q)
df_case     = ablation_case_study()


# ─── 결과 저장 ────────────────────────────────────────────────────────────────
with pd.ExcelWriter(OUTPUT_DIR / "ablation_scaler_results.xlsx") as writer:
    df_linear.to_excel(writer,   sheet_name="Linear_Regression", index=False)
    df_logistic.to_excel(writer, sheet_name="Logistic_Regression", index=False)
    df_case.to_excel(writer,     sheet_name="Case_Study", index=False)
print(f"\n  [저장] ablation_scaler_results.xlsx")


# ─── 시각화 ───────────────────────────────────────────────────────────────────
COLORS = {"With Scaler": "#1565C0", "Without Scaler": "#E53935"}
W = 0.35

# 1. Linear Regression 비교
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Linear Regression – Scaler Ablation", fontsize=12, fontweight="bold")

for ax, metric, ylabel in zip(
    axes,
    ["RMSE", "R2", "CV_R2_mean"],
    ["RMSE (cm²)", "R²", "CV R²"]
):
    for i, model_name in enumerate(df_linear["Model"].unique()):
        sub = df_linear[df_linear["Model"] == model_name]
        x_base = np.array([i])
        for j, scaler_label in enumerate(["With Scaler", "Without Scaler"]):
            row = sub[sub["Scaler"] == scaler_label]
            val = row[metric].values[0]
            ax.bar(x_base + (j - 0.5) * W, val, W * 0.9,
                   color=COLORS[scaler_label],
                   label=scaler_label if i == 0 else "",
                   alpha=0.85, edgecolor="white")
            ax.text(x_base[0] + (j - 0.5) * W, val + abs(val) * 0.01,
                    f"{val:.3f}", ha="center", fontsize=7)

    ax.set_xticks(range(df_linear["Model"].nunique()))
    ax.set_xticklabels(df_linear["Model"].unique(), fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(ylabel)
    ax.grid(True, alpha=0.3, axis="y")
    if metric == "RMSE":
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_linear_ablation.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Logistic Regression 비교
fig, axes = plt.subplots(1, 3, figsize=(11, 5))
fig.suptitle("Logistic Regression – Scaler Ablation", fontsize=12, fontweight="bold")

for ax, metric, ylabel in zip(
    axes,
    ["Accuracy", "AUROC", "CV_Acc_mean"],
    ["Accuracy", "AUROC", "CV Accuracy"]
):
    vals  = [df_logistic[df_logistic["Scaler"] == s][metric].values[0]
             for s in ["With Scaler", "Without Scaler"]]
    bars  = ax.bar(["With Scaler", "Without Scaler"], vals,
                   color=[COLORS["With Scaler"], COLORS["Without Scaler"]],
                   alpha=0.85, edgecolor="white")
    ax.set_ylabel(ylabel); ax.set_title(ylabel)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + max(vals) * 0.02, f"{v:.4f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_logistic_ablation.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Case Study 비교
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Case Study (Tabular) – Scaler Ablation", fontsize=12, fontweight="bold")

cases     = df_case["Case"].unique()
x_pos     = np.arange(len(cases))

for ax, metric, ylabel in zip(axes, ["R2", "RMSE"], ["R²", "RMSE (cm²)"]):
    for j, scaler_label in enumerate(["With Scaler", "Without Scaler"]):
        vals = [df_case[(df_case["Case"] == c) & (df_case["Scaler"] == scaler_label)][metric].values[0]
                for c in cases]
        bars = ax.bar(x_pos + (j - 0.5) * W, vals, W * 0.9,
                      color=COLORS[scaler_label], label=scaler_label,
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + abs(v) * 0.01, f"{v:.3f}", ha="center", fontsize=7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.split(":")[0] for c in cases], fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(ylabel)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_case_study_ablation.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  [저장] 01_linear_ablation.png")
print(f"  [저장] 02_logistic_ablation.png")
print(f"  [저장] 03_case_study_ablation.png")

# ─── 최종 요약 출력 ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  최종 요약")
print("=" * 60)
print("\n[Linear Regression]")
print(df_linear.to_string(index=False))
print("\n[Logistic Regression]")
print(df_logistic.to_string(index=False))
print("\n[Case Study]")
print(df_case[["Case","Scaler","RMSE","R2","CV_R2_mean","CV_R2_std"]].to_string(index=False))
print("=" * 60)
