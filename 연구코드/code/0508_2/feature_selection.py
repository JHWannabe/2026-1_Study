"""
feature_selection.py  —  Age / Sex / BMI + selected AEC features → SMI regression
  Step 1. AEC 내부 중복 제거 (고상관 feature 제거)
  Step 2. Age/Sex/BMI 통제 후 partial correlation으로 AEC feature 선택
  Step 3. VIF 경고 출력 (제거 기준 아님)
"""
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

DATA_PATH     = Path(__file__).parents[2] / "data" / "강남_merged_features.xlsx"
SHEET_META    = "metadata-bmi_add"
SHEET_AEC     = "aec_feature_filtered"
BASELINE_COLS = ["PatientAge", "PatientSex", "BMI"]

CORR_THR     = 0.90
PARTIAL_PVAL = 0.10
VIF_WARN     = 10.0

BATCH_SIZE = 32
EPOCHS     = 5000
LR         = 1e-3
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_and_merge(data_path: Path | str) -> pd.DataFrame:
    meta = pd.read_excel(data_path, sheet_name=SHEET_META)
    aec  = pd.read_excel(data_path, sheet_name=SHEET_AEC)

    overlap = [c for c in aec.columns if c in meta.columns and c != "PatientID"]
    aec = aec.drop(columns=overlap)

    df = meta.merge(aec, on="PatientID", how="inner")
    print(f"[데이터] meta={len(meta)}, aec={len(aec)}, merged={len(df)}")

    if df["PatientSex"].dtype == object:
        gmap = {v: i for i, v in enumerate(sorted(df["PatientSex"].unique()))}
        print(f"  성별 인코딩: {gmap}")
        df["PatientSex"] = df["PatientSex"].map(gmap)

    return df


def get_aec_cols(df: pd.DataFrame) -> list[str]:
    meta_cols = set(pd.read_excel(DATA_PATH, sheet_name=SHEET_META, nrows=0).columns)
    return [c for c in df.columns if c not in meta_cols and c != "PatientID"]


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 보정 → adjusted p-value 반환"""
    n     = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, int)
    ranks[order] = np.arange(1, n + 1)
    adj = pvals * n / ranks
    for i in range(n - 2, -1, -1):  # monotonicity 보장
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return np.minimum(adj, 1.0)


def step1_remove_redundant(df: pd.DataFrame, aec_cols: list[str]) -> list[str]:
    data = df[aec_cols].copy()
    data = data.dropna(axis=1, how="all")
    data = data.loc[:, data.std() > 0]

    cols      = list(data.columns)
    variances = data.var()
    cols.sort(key=lambda c: -variances[c])

    corr = data[cols].corr(method="pearson").abs()
    kept = []
    for c in cols:
        if all(corr.loc[c, k] < CORR_THR for k in kept):
            kept.append(c)

    removed = [c for c in cols if c not in kept]
    print(f"\n[Step 1] 중복 제거: {len(cols)} → {len(kept)} "
          f"(제거 {len(removed)}개, 상관 기준 r>{CORR_THR})")
    if removed:
        print(f"  제거된 feature: {removed}")
    return kept


def step2_partial_corr_selection(df: pd.DataFrame,
                                  aec_cols: list[str],
                                  outcome: str = "SMI") -> list[str]:
    sub = df[BASELINE_COLS + aec_cols + [outcome]].dropna()
    B   = sub[BASELINE_COLS].values.astype(float)
    y   = sub[outcome].values.astype(float)

    lr      = LinearRegression()
    y_resid = y - lr.fit(B, y).predict(B)

    partial_rs, pvals = [], []
    for c in aec_cols:
        x = sub[c].values.astype(float)
        if np.std(x) == 0:
            partial_rs.append(0.0); pvals.append(1.0)
            continue
        x_resid = x - lr.fit(B, x).predict(B)
        r, p    = stats.pearsonr(x_resid, y_resid)
        partial_rs.append(r); pvals.append(p)

    adj_pvals = _fdr_bh(np.array(pvals))
    result_df = pd.DataFrame({
        "feature":   aec_cols,
        "partial_r": partial_rs,
        "p_raw":     pvals,
        "p_adj_BH":  adj_pvals,
    }).sort_values("p_adj_BH")

    print(f"\n[Step 2] partial correlation (controlling Age/Sex/BMI)  FDR threshold={PARTIAL_PVAL}")
    print(result_df.to_string(index=False, float_format="{:.4f}".format))

    selected = result_df.loc[result_df["p_adj_BH"] < PARTIAL_PVAL, "feature"].tolist()
    print(f"\n  선택된 AEC feature ({len(selected)}개): {selected}")
    return selected


def step3_vif_report(df: pd.DataFrame, final_cols: list[str], outcome: str = "SMI"):
    cols = BASELINE_COLS + final_cols
    sub  = df[cols + [outcome]].dropna()
    X    = sub[cols].values.astype(float)
    X_c  = np.column_stack([np.ones(len(X)), X])
    vif_vals = [variance_inflation_factor(X_c, i + 1) for i in range(X.shape[1])]

    vif_df = pd.DataFrame({"feature": cols, "VIF": vif_vals}).sort_values("VIF", ascending=False)
    print(f"\n[Step 3] VIF (경고등: VIF > {VIF_WARN})")
    print(vif_df.to_string(index=False, float_format="{:.2f}".format))

    warn = vif_df[vif_df["VIF"] > VIF_WARN]["feature"].tolist()
    if warn:
        print(f"  ⚠ 높은 VIF feature: {warn}  ← external validation에서 최종 판정")
    else:
        print("  VIF 이상 없음")


def run_feature_selection(df: pd.DataFrame, aec_cols: list[str] | None = None) -> list[str]:
    if aec_cols is None:
        aec_cols = get_aec_cols(df)
        print(f"\n전체 AEC feature 수: {len(aec_cols)}")

    step1_cols = step1_remove_redundant(df, aec_cols)
    step2_cols = step2_partial_corr_selection(df, step1_cols)

    if step2_cols:
        step3_vif_report(df, step2_cols)
    else:
        print("\n[Step 3] 선택된 AEC feature가 없어 VIF 계산 생략")

    return step2_cols


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def _loader(X, y, shuffle):
    return DataLoader(TabularDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)


class ClinicalMLP(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),  nn.BatchNorm1d(64),  nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(64, 128),          nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, 64),          nn.BatchNorm1d(64),  nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x): return self.net(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y_b)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, preds, trues = 0.0, [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        p = model(X_b)
        total += criterion(p, y_b).item() * len(y_b)
        preds.append(p.cpu().numpy())
        trues.append(y_b.cpu().numpy())
    return total / len(loader.dataset), np.concatenate(preds).ravel(), np.concatenate(trues).ravel()


def _fit(train_loader, val_loader, model, device, log_interval=500):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_loss, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % log_interval == 0:
            print(f"  {epoch:5d} | train {tr:.4f} | val {vl:.4f}")
    return best_state, criterion


def _evaluate(model, loader, criterion, y_scaler, device):
    _, ps, ts = eval_epoch(model, loader, criterion, device)
    preds = y_scaler.inverse_transform(ps.reshape(-1, 1)).ravel()
    trues = y_scaler.inverse_transform(ts.reshape(-1, 1)).ravel()
    return trues, preds


def plot_results(trues, preds, save_dir: Path, title="Clinic+AEC"):
    resid = preds - trues
    r,  p_r  = stats.pearsonr(trues, preds)
    _,  p_sw = stats.shapiro(resid)
    _,  p_bt = stats.ttest_1samp(resid, 0)

    thr        = np.percentile(trues, 25)
    y_bin      = (trues >= thr).astype(int)
    score_norm = (preds - preds.min()) / (np.ptp(preds) + 1e-8)
    fpr, tpr, _ = roc_curve(y_bin, score_norm)
    roc_auc    = auc(fpr, tpr)
    cm         = confusion_matrix(y_bin, (preds >= thr).astype(int))

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)
    ax  = [fig.add_subplot(gs[r_i, c_i]) for r_i in range(2) for c_i in range(3)]

    lo, hi = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    ax[0].scatter(trues, preds, alpha=0.6, edgecolors="k", linewidths=0.4)
    ax[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
    ax[0].set(xlabel="True SMI", ylabel="Predicted SMI",
              title=f"Pred vs True\nr={r:.3f}  p={p_r:.2e}  R²={r2_score(trues, preds):.3f}")
    ax[0].legend()

    ax[1].hist(resid, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax[1].axvline(0, color="r", linestyle="--", lw=1.5)
    ax[1].set(xlabel="Residual (Pred−True)", ylabel="Count",
              title=f"Residual Distribution\nSW p={p_sw:.3f}  bias p={p_bt:.3f}")

    mean_v = (trues + preds) / 2
    diff_v = preds - trues
    md, sd = diff_v.mean(), diff_v.std()
    ax[2].scatter(mean_v, diff_v, alpha=0.6, edgecolors="k", linewidths=0.4)
    ax[2].axhline(md,           color="r",    lw=1.5, label=f"Mean {md:+.3f}")
    ax[2].axhline(md + 1.96*sd, color="gray", lw=1.2, ls="--", label=f"+1.96SD {md+1.96*sd:+.3f}")
    ax[2].axhline(md - 1.96*sd, color="gray", lw=1.2, ls="--", label=f"−1.96SD {md-1.96*sd:+.3f}")
    ax[2].set(xlabel="Mean(True, Pred)", ylabel="Pred−True", title="Bland-Altman Plot")
    ax[2].legend(fontsize=8)

    ax[3].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC={roc_auc:.3f}")
    ax[3].plot([0, 1], [0, 1], "k--", lw=1)
    ax[3].set(xlabel="FPR", ylabel="TPR", title=f"ROC (threshold=25th pct {thr:.2f})")
    ax[3].legend()

    im = ax[4].imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax[4])
    cls = [f"<{thr:.1f}", f"≥{thr:.1f}"]
    ax[4].set_xticks([0, 1]); ax[4].set_xticklabels(cls, fontsize=9)
    ax[4].set_yticks([0, 1]); ax[4].set_yticklabels(cls, fontsize=9)
    ax[4].set(xlabel="Predicted", ylabel="True", title="Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax[4].text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)

    (osm, osr), (slope, intercept, _) = stats.probplot(resid)
    ax[5].scatter(osm, osr, alpha=0.6, edgecolors="k", linewidths=0.4)
    qq = np.array([osm[0], osm[-1]])
    ax[5].plot(qq, slope*qq + intercept, "r--", lw=1.5)
    ax[5].set(xlabel="Theoretical Quantiles", ylabel="Sample Quantiles",
              title=f"Q-Q Plot (SW p={p_sw:.3f})")

    fig.suptitle(f"{title} — Test Set Evaluation", fontsize=14, fontweight="bold", y=1.01)
    out = save_dir / f"{title.lower().replace(' ', '_')}_test_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"시각화 저장: {out}")
    print(f"[통계] r={r:.4f} (p={p_r:.3e})  SW-p={p_sw:.4f}  bias-p={p_bt:.4f}  AUC={roc_auc:.4f}")


def prepare_XY(df: pd.DataFrame, selected_aec: list[str]):
    all_feat_cols = BASELINE_COLS + selected_aec
    sub = df[all_feat_cols + ["SMI"]].dropna()
    X = sub[all_feat_cols].values.astype(float)
    y = sub["SMI"].values.astype(float)
    print(f"학습 데이터: {X.shape}  (baseline {len(BASELINE_COLS)}개 + AEC {len(selected_aec)}개)")
    print(f"SMI  mean={y.mean():.3f}  std={y.std():.3f}")
    return X, y


def run_cv(n_splits=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df      = load_and_merge(DATA_PATH)
    aec_all = get_aec_cols(df)
    print(f"전체 AEC feature 수: {len(aec_all)}")

    needed = BASELINE_COLS + aec_all + ["SMI"]
    sub    = df[needed].dropna(subset=BASELINE_COLS + ["SMI"]).reset_index(drop=True)
    y_all  = sub["SMI"].values.astype(float)

    tv_idx, te_idx = train_test_split(np.arange(len(sub)), test_size=0.15, random_state=SEED)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_maes, fold_r2s, fold_features = [], [], []
    best_mae_cv = float("inf")
    best_state_cv, best_x_scaler, best_y_scaler, best_imputer, best_selected_aec = None, None, None, None, None

    for fold, (tr_rel, vl_rel) in enumerate(kf.split(tv_idx), 1):
        tr_idx   = tv_idx[tr_rel]
        vl_idx   = tv_idx[vl_rel]
        df_train = sub.iloc[tr_idx].copy()

        print(f"\n{'─'*55}")
        print(f" Fold {fold}/{n_splits}  (train {len(tr_idx)}, val {len(vl_idx)})")
        print(f"{'─'*55}")

        selected_aec = run_feature_selection(df_train, aec_all)
        fold_features.append(selected_aec)
        feat_cols = BASELINE_COLS + selected_aec

        X_tr = sub.iloc[tr_idx][feat_cols].values.astype(float)
        y_tr = y_all[tr_idx]
        X_vl = sub.iloc[vl_idx][feat_cols].values.astype(float)
        y_vl = y_all[vl_idx]

        imputer  = SimpleImputer(strategy="mean")
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_tr     = imputer.fit_transform(X_tr)
        X_vl     = imputer.transform(X_vl)
        X_tr_sc  = x_scaler.fit_transform(X_tr)
        y_tr_sc  = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
        X_vl_sc  = x_scaler.transform(X_vl)
        y_vl_sc  = y_scaler.transform(y_vl.reshape(-1, 1)).ravel()

        model = ClinicalMLP(in_features=len(feat_cols)).to(device)
        best_state, criterion = _fit(
            _loader(X_tr_sc, y_tr_sc, True),
            _loader(X_vl_sc, y_vl_sc, False),
            model, device)

        model.load_state_dict(best_state)
        trues, preds = _evaluate(model, _loader(X_vl_sc, y_vl_sc, False),
                                 criterion, y_scaler, device)
        mae, r2 = mean_absolute_error(trues, preds), r2_score(trues, preds)
        fold_maes.append(mae); fold_r2s.append(r2)
        print(f"  → Fold {fold}  MAE={mae:.4f}  R²={r2:.4f}  features({len(feat_cols)}): {feat_cols}")

        if mae < best_mae_cv:
            best_mae_cv       = mae
            best_state_cv     = best_state
            best_imputer      = imputer
            best_x_scaler     = x_scaler
            best_y_scaler     = y_scaler
            best_selected_aec = selected_aec

    print(f"\n{'='*55}")
    print(f" {n_splits}-Fold CV Summary (Clinic + AEC, selection per fold)")
    for i, (m, r, feats) in enumerate(zip(fold_maes, fold_r2s, fold_features), 1):
        print(f"  Fold {i}: MAE {m:.4f}  R² {r:.4f}  AEC선택={feats}")
    print(f"  Mean : MAE {np.mean(fold_maes):.4f}±{np.std(fold_maes):.4f}"
          f"  R² {np.mean(fold_r2s):.4f}±{np.std(fold_r2s):.4f}")

    assert best_imputer is not None
    best_feat_cols = BASELINE_COLS + best_selected_aec
    X_te    = sub.iloc[te_idx][best_feat_cols].values.astype(float)
    y_te    = y_all[te_idx]
    X_te    = best_imputer.transform(X_te)
    X_te_sc = best_x_scaler.transform(X_te)
    y_te_sc = best_y_scaler.transform(y_te.reshape(-1, 1)).ravel()

    model_final = ClinicalMLP(in_features=len(best_feat_cols)).to(device)
    model_final.load_state_dict(best_state_cv)
    criterion = nn.HuberLoss(delta=1.0)
    trues, preds = _evaluate(model_final, _loader(X_te_sc, y_te_sc, False),
                             criterion, best_y_scaler, device)
    print(f"\n[Best fold → Test] MAE={mean_absolute_error(trues, preds):.4f}"
          f"  R²={r2_score(trues, preds):.4f}")
    print(f"  사용된 feature: {best_feat_cols}")

    save_dir = Path(__file__).parents[2] / "results" / "0508_2"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_results(trues, preds, save_dir)

    torch.save({"model_state": best_state_cv, "in_features": len(best_feat_cols),
                "selected_aec": best_selected_aec}, save_dir / "clinic_aec_best.pt")
    with open(save_dir / "clinic_aec_x_scaler.pkl", "wb") as f: pickle.dump(best_x_scaler, f)
    with open(save_dir / "clinic_aec_y_scaler.pkl", "wb") as f: pickle.dump(best_y_scaler, f)
    print(f"모델 저장: {save_dir / 'clinic_aec_best.pt'}")


def test_only(ckpt_path: str):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_dir = Path(ckpt_path).parent

    with open(ckpt_dir / "clinic_aec_x_scaler.pkl", "rb") as f: x_scaler = pickle.load(f)
    with open(ckpt_dir / "clinic_aec_y_scaler.pkl", "rb") as f: y_scaler = pickle.load(f)

    selected_aec = ckpt["selected_aec"]
    df = load_and_merge(DATA_PATH)
    X, y = prepare_XY(df, selected_aec)
    X_sc = x_scaler.transform(X)
    y_sc = y_scaler.transform(y.reshape(-1, 1)).ravel()
    _, X_te, _, y_te = train_test_split(X_sc, y_sc, test_size=0.15, random_state=SEED)

    model = ClinicalMLP(in_features=ckpt["in_features"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    criterion = nn.HuberLoss(delta=1.0)
    trues, preds = _evaluate(model, _loader(X_te, y_te, False), criterion, y_scaler, device)
    print(f"[Test] MAE={mean_absolute_error(trues, preds):.4f}  R²={r2_score(trues, preds):.4f}")
    save_dir = Path(__file__).parents[2] / "results" / "0508_2"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_results(trues, preds, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", metavar="CKPT", default=None)
    args = parser.parse_args()

    if args.test_only:  test_only(args.test_only)
    else:               run_cv()
