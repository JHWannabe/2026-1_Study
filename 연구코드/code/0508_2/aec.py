"""
aec.py — AEC feature → SMI (두 시트 비교)
  aec_feature_filtered   : 통계 feature 65개
  aec_interpolation_final: 보간 curve 256점
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

DATA_PATH  = Path(__file__).parents[2] / "data" / "강남_merged_features.xlsx"
SEED       = 42
N_SPLITS   = 5

COMPARE_SHEETS = [
    "aec_feature_filtered",
    "aec_interpolation_final",
]

_META_COLS = {"PatientID", "PatientAge", "PatientSex", "BMI", "SMI"}

AEC_COLS: list[str] = []  # run() 호출 시 시트별로 갱신됨

np.random.seed(SEED)


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
    print(f" {N_SPLITS}-Fold CV  |  features: {len(AEC_COLS)}개  |  train n={len(tr_idx)}")
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
        top_n   = 10
        lr_coef = lin.coef_
        top_lr  = np.argsort(np.abs(lr_coef))[::-1][:top_n]
        print(f"\n  [LinearReg 계수 상위 {top_n}]")
        for i in top_lr:
            print(f"    {AEC_COLS[i]:30s}: {lr_coef[i]:+.4f}")
        print(f"    {'intercept':30s}: {lin.intercept_:+.4f}")

        log_coef = log.coef_[0]
        top_log  = np.argsort(np.abs(log_coef))[::-1][:top_n]
        print(f"\n  [LogisticReg 계수 상위 {top_n}]")
        for i in top_log:
            print(f"    {AEC_COLS[i]:30s}: {log_coef[i]:+.4f}")
        print(f"    {'intercept':30s}: {log.intercept_[0]:+.4f}")

    return (lr_fpr, lr_tpr, lr_auc), (log_fpr, log_tpr, log_auc), thr


def plot_results(cv_lr_rocs, cv_log_rocs, lr_aucs, log_aucs,
                 valid_lr, valid_log, test_lr, test_log,
                 thr_valid: float, thr_test: float,
                 save_dir: Path, sheet_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"SMI 예측 ROC (Threshold=25th pct 위험군 기준)\n"
        f"Features: {sheet_name} ({len(AEC_COLS)}개)",
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
    out = save_dir / f"aec_roc_{sheet_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n시각화 저장: {out}")


def run(sheet_name: str, save_dir: Path) -> tuple[float, float, float, float]:
    """한 시트에 대해 전체 파이프라인 실행. (lr_valid_auc, log_valid_auc, lr_test_auc, log_test_auc) 반환."""
    global AEC_COLS

    print(f"\n{'#'*60}")
    print(f"  Sheet: {sheet_name}")
    print(f"{'#'*60}")

    df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
    AEC_COLS = [c for c in df.columns if c not in _META_COLS]
    df = df[["SMI"] + AEC_COLS].dropna().reset_index(drop=True)
    print(f"[데이터] rows={len(df)}  features={len(AEC_COLS)}  "
          f"SMI mean={df['SMI'].mean():.3f}  std={df['SMI'].std():.3f}")

    X = df[AEC_COLS].values.astype(float)
    y = df["SMI"].values.astype(float)

    tr_idx, vl_idx, te_idx = split_data(len(df))
    print(f"[데이터 분할]  Train={len(tr_idx)}  Valid={len(vl_idx)}  Test={len(te_idx)}")

    lr_aucs, log_aucs, lr_rocs, log_rocs = run_cv(X, y, tr_idx)

    print("\n── Valid 평가 (train → valid) ──")
    valid_lr, valid_log, thr_valid = _fit_and_eval(X, y, tr_idx, vl_idx, "Valid")

    print("\n── Test 평가 (train+valid → test) ──")
    tr_vl_idx = np.concatenate([tr_idx, vl_idx])
    test_lr, test_log, thr_test = _fit_and_eval(X, y, tr_vl_idx, te_idx, "Test")

    plot_results(lr_rocs, log_rocs, lr_aucs, log_aucs,
                 valid_lr, valid_log, test_lr, test_log,
                 thr_valid, thr_test, save_dir, sheet_name)

    _, _, vl_lr_auc  = valid_lr
    _, _, vl_log_auc = valid_log
    _, _, te_lr_auc  = test_lr
    _, _, te_log_auc = test_log
    return vl_lr_auc, vl_log_auc, te_lr_auc, te_log_auc


if __name__ == "__main__":
    save_dir = Path(__file__).parents[2] / "results" / "0508_2"
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sheet in COMPARE_SHEETS:
        vl_lr, vl_log, te_lr, te_log = run(sheet, save_dir)
        results.append((sheet, vl_lr, vl_log, te_lr, te_log))

    print(f"\n\n{'='*70}")
    print("  [비교 결과]  AUC (위험군 분류, threshold=25th pct)")
    print(f"{'='*70}")
    print(f"  {'Sheet':<30} {'Valid_LR':>9} {'Valid_Log':>10} {'Test_LR':>8} {'Test_Log':>9}")
    print(f"  {'─'*66}")
    for sheet, vl_lr, vl_log, te_lr, te_log in results:
        print(f"  {sheet:<30} {vl_lr:>9.4f} {vl_log:>10.4f} {te_lr:>8.4f} {te_log:>9.4f}")
    print(f"{'='*70}")
