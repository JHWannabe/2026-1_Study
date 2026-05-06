"""
clinic.py — PatientAge / PatientSex / BMI → SMI
  Train 70% → 5-Fold CV  |  Valid 15%  |  Test 15%
  Threshold : fold train SMI 하위 25th percentile
"""
import sys
import warnings
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

DATA_PATH     = Path(__file__).parents[2] / "data" / "강남_merged_features.xlsx"
SHEET_META    = "metadata-bmi_add"
BASELINE_COLS = ["PatientAge", "PatientSex", "BMI"]
SEED          = 42
N_SPLITS      = 5

np.random.seed(SEED)


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_META)
    if df["PatientSex"].dtype == object:
        gmap = {v: i for i, v in enumerate(sorted(df["PatientSex"].unique()))}
        print(f"  성별 인코딩: {gmap}")
        df["PatientSex"] = df["PatientSex"].map(gmap)
    df = df[BASELINE_COLS + ["SMI"]].dropna()
    print(f"[데이터] rows={len(df)}  SMI mean={df['SMI'].mean():.3f}  std={df['SMI'].std():.3f}")
    return df.reset_index(drop=True)


def compute_auc(y_true_bin: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true_bin)) < 2:
        return float("nan")
    return roc_auc_score(y_true_bin, scores)


def binarize(y: np.ndarray, threshold: float) -> np.ndarray:
    """SMI < threshold → 1 (저근육 위험군)"""
    return (y < threshold).astype(int)


def split_data(n: int):
    tr_vl_idx, te_idx = train_test_split(np.arange(n), test_size=0.15, random_state=SEED)
    tr_idx, vl_idx    = train_test_split(tr_vl_idx, test_size=0.15 / 0.85, random_state=SEED)
    return tr_idx, vl_idx, te_idx


def run_cv(X: np.ndarray, y: np.ndarray, tr_idx: np.ndarray):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    lr_aucs, log_aucs = [], []
    lr_rocs, log_rocs = [], []

    print(f"\n{'='*60}")
    print(f" {N_SPLITS}-Fold CV  |  features: {BASELINE_COLS}  |  train n={len(tr_idx)}")
    print(f"{'='*60}")

    for fold, (tr_rel, vl_rel) in enumerate(kf.split(tr_idx), 1):
        tr_i = tr_idx[tr_rel]
        vl_i = tr_idx[vl_rel]

        X_tr, y_tr = X[tr_i], y[tr_i]
        X_vl, y_vl = X[vl_i], y[vl_i]

        thr      = np.percentile(y_tr, 25)
        y_tr_bin = binarize(y_tr, thr)
        y_vl_bin = binarize(y_vl, thr)

        imputer = SimpleImputer(strategy="mean")
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_vl_sc = scaler.transform(imputer.transform(X_vl))

        lin      = LinearRegression()
        lin.fit(X_tr_sc, y_tr)
        lr_score = lin.predict(X_vl_sc)
        lr_auc   = compute_auc(y_vl_bin, -lr_score)
        fpr, tpr, _ = roc_curve(y_vl_bin, -lr_score)
        lr_aucs.append(lr_auc); lr_rocs.append((fpr, tpr))

        log       = LogisticRegression(max_iter=1000, random_state=SEED)
        log.fit(X_tr_sc, y_tr_bin)
        log_score = log.predict_proba(X_vl_sc)[:, 1]
        log_auc   = compute_auc(y_vl_bin, log_score)
        fpr2, tpr2, _ = roc_curve(y_vl_bin, log_score)
        log_aucs.append(log_auc); log_rocs.append((fpr2, tpr2))

        print(f"  Fold {fold}  thr={thr:.3f}  "
              f"LinearReg AUC={lr_auc:.4f}  "
              f"LogisticReg AUC={log_auc:.4f}  "
              f"(위험군 비율: valid {y_vl_bin.mean():.2%})")

    print(f"\n{'─'*60}")
    print(f"  LinearReg   CV AUC  mean={np.mean(lr_aucs):.4f} ± {np.std(lr_aucs):.4f}")
    print(f"  LogisticReg CV AUC  mean={np.mean(log_aucs):.4f} ± {np.std(log_aucs):.4f}")

    return lr_aucs, log_aucs, lr_rocs, log_rocs


def _fit_and_eval(X: np.ndarray, y: np.ndarray,
                  tr_i: np.ndarray, ev_i: np.ndarray, label: str):
    X_tr, y_tr = X[tr_i], y[tr_i]
    X_ev, y_ev = X[ev_i], y[ev_i]

    thr      = np.percentile(y_tr, 25)
    y_tr_bin = binarize(y_tr, thr)
    y_ev_bin = binarize(y_ev, thr)

    imputer = SimpleImputer(strategy="mean")
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(imputer.fit_transform(X_tr))
    X_ev_sc = scaler.transform(imputer.transform(X_ev))

    lin      = LinearRegression()
    lin.fit(X_tr_sc, y_tr)
    lr_score = lin.predict(X_ev_sc)
    lr_auc   = compute_auc(y_ev_bin, -lr_score)
    lr_fpr, lr_tpr, _ = roc_curve(y_ev_bin, -lr_score)

    log       = LogisticRegression(max_iter=1000, random_state=SEED)
    log.fit(X_tr_sc, y_tr_bin)
    log_score = log.predict_proba(X_ev_sc)[:, 1]
    log_auc   = compute_auc(y_ev_bin, log_score)
    log_fpr, log_tpr, _ = roc_curve(y_ev_bin, log_score)

    print(f"\n{'='*60}")
    print(f" [{label}]  n={len(ev_i)}  threshold={thr:.3f}  위험군 비율={y_ev_bin.mean():.2%}")
    print(f"  LinearReg   AUC = {lr_auc:.4f}")
    print(f"  LogisticReg AUC = {log_auc:.4f}")

    if label == "Test":
        print(f"\n  [LinearReg 계수]")
        for feat, coef in zip(BASELINE_COLS, lin.coef_):
            print(f"    {feat:15s}: {coef:+.4f}")
        print(f"    {'intercept':15s}: {lin.intercept_:+.4f}")
        print(f"\n  [LogisticReg 계수]")
        for feat, coef in zip(BASELINE_COLS, log.coef_[0]):
            print(f"    {feat:15s}: {coef:+.4f}")
        print(f"    {'intercept':15s}: {log.intercept_[0]:+.4f}")

    return (lr_fpr, lr_tpr, lr_auc), (log_fpr, log_tpr, log_auc), thr


def plot_results(cv_lr_rocs, cv_log_rocs, lr_aucs, log_aucs,
                 valid_lr, valid_log, test_lr, test_log,
                 thr_valid: float, thr_test: float, save_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"SMI 예측 ROC (Threshold=25th pct 위험군 기준)\n"
        f"Features: {BASELINE_COLS}",
        fontsize=12, fontweight="bold"
    )

    for ax, rocs, aucs_list, vl, te, title, color in zip(
        axes,
        [cv_lr_rocs, cv_log_rocs],
        [lr_aucs, log_aucs],
        [valid_lr, valid_log],
        [test_lr, test_log],
        ["Linear Regression", "Logistic Regression"],
        ["steelblue", "darkorange"],
    ):
        for i, (fpr, tpr) in enumerate(rocs):
            ax.plot(fpr, tpr, alpha=0.35, lw=1.2, color=color,
                    label=f"CV Fold {i+1} AUC={aucs_list[i]:.3f}")

        vl_fpr, vl_tpr, vl_auc = vl
        ax.plot(vl_fpr, vl_tpr, color="forestgreen", lw=2.0, linestyle="--",
                label=f"Valid AUC={vl_auc:.3f}  (thr={thr_valid:.2f})", zorder=4)

        te_fpr, te_tpr, te_auc = te
        ax.plot(te_fpr, te_tpr, color="crimson", lw=2.5, linestyle="-.",
                label=f"Test  AUC={te_auc:.3f}  (thr={thr_test:.2f})", zorder=5)

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"{title}\nCV mean AUC={np.mean(aucs_list):.3f}±{np.std(aucs_list):.3f}")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    out = save_dir / "clinic_baseline_roc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n시각화 저장: {out}")


if __name__ == "__main__":
    df = load_data()
    X  = df[BASELINE_COLS].values.astype(float)
    y  = df["SMI"].values.astype(float)

    tr_idx, vl_idx, te_idx = split_data(len(df))
    print(f"\n[데이터 분할]  Train={len(tr_idx)}  Valid={len(vl_idx)}  Test={len(te_idx)}")

    lr_aucs, log_aucs, lr_rocs, log_rocs = run_cv(X, y, tr_idx)

    print("\n── Valid 평가 (train → valid) ──")
    valid_lr, valid_log, thr_valid = _fit_and_eval(X, y, tr_idx, vl_idx, "Valid")

    print("\n── Test 평가 (train+valid → test) ──")
    tr_vl_idx = np.concatenate([tr_idx, vl_idx])
    test_lr, test_log, thr_test = _fit_and_eval(X, y, tr_vl_idx, te_idx, "Test")

    save_dir = Path(__file__).parents[2] / "results" / "0508_2"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_results(lr_rocs, log_rocs, lr_aucs, log_aucs,
                 valid_lr, valid_log, test_lr, test_log,
                 thr_valid, thr_test, save_dir)
