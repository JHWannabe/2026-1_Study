"""
0702(06~09)의 cross-attention 라인은 전부 binary Output(정상/저SMI)을 타깃으로 삼았고,
14개 변형 전부 clinic-only(ROC-AUC 0.776)를 못 이겼다. 10번(propensity matching)은
그 이유를 "원래 효과크기의 대부분(47~92%)이 체격(Age/Height/Weight) confounding"이라고
설명했다.

이 스크립트는 아키텍처(09의 global/patient-wise 정규화 + clinic-skip)는 그대로 두고
타깃만 바꾼다: binary Output 대신 continuous SMI를 직접 회귀한다. binary화는 cutoff
근방의 정보를 버리므로, "cross-attention이 이기지 못한 이유가 이분화로 인한 정보손실
때문인지, 아니면 10번처럼 정보량 자체의 한계인지"를 이 피벗으로 가른다.

설계 결정:
  - SMI는 남녀 평균/분산이 크게 다르다(F 38.2+-5.5, M 48.6+-8.1). 그대로 회귀하면
    모델이 "성별을 맞히는 것"만으로 상당한 설명력을 얻어버리므로, 타깃을 성별 내
    z-score로 표준화한다(mu/sigma는 매 fold의 train 데이터에서만 계산 -> leakage 없음).
  - fold 분할은 회귀라 StratifiedKFold를 y로 직접 못 쓰므로, (Sex_M x SMI 성별-내
    quartile) 조합을 층화 라벨로 써서 RepeatedStratifiedKFold(5-fold x 4 repeat = 20
    fold)를 처음부터 사용한다(07의 교훈: 5-fold만 쓰면 검정력 부족).
  - 07/09에서 이미 효과 없음이 확인된 스무딩/early-stopping/warm-start/FDA-prior는
    배제하고, 09에서 검증된 4개 변형(global, global_skip, patient_wise,
    patient_wise_skip)만 이식한다.
  - 손실함수는 SMI 표준화 후에도 남는 이상치(z가 -4 근방까지 나옴)에 덜 민감하도록
    MSE 대신 SmoothL1(Huber)을 쓴다.
  - 핵심 차별화 지표(4단계): clinic-only 회귀의 test-fold 잔차(clinic이 설명 못하는
    부분)를 aec 스칼라 피처가 유의하게 예측하는지 별도로 검정한다. 이건 "AEC가 clinic
    위에 진짜 추가 정보를 주는가"를 가장 직접적으로 보여주는 지표다.

출력: 콘솔 비교표, figures/11/smi_r2_comparison.png,
      figures/11/cross_attention_smi_weights_{tag}.png,
      excel/model_comparison_results.xlsx에 SMI_* 시트 추가
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_DIR = os.path.join(BASE_DIR, "..", "excel")
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures", "11")
os.makedirs(FIGURES_DIR, exist_ok=True)

MERGED_FILE = os.path.join(EXCEL_DIR, "강남_liver_merged_features.xlsx")
RESULTS_FILE = os.path.join(EXCEL_DIR, "model_comparison_results.xlsx")

CLINIC_FEATURES = ["PatientAge", "Sex_M", "Height", "Weight"]
SIGNIFICANT_SEGMENTS = [(62, 71), (103, 115), (120, 124)]  # 0701 pointwise FDR 유의구간(1-based, inclusive)
AEC_SEGMENTS = {  # 04_build_model_features.py와 동일 정의
    "slope_48_55": (62, 71),
    "slope_80_90": (103, 115),
    "slope_94_97": (120, 124),
}
AEC_SCALAR_FEATURES = ["mean_mA"] + list(AEC_SEGMENTS.keys())

SEQ_LEN = 128
N_SPLITS = 5
N_REPEATS = 4  # 07의 교훈: 처음부터 20-fold
RANDOM_STATE = 42
D_MODEL = 16
N_HEADS = 2
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
RIDGE_ALPHA = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- 데이터 로드 ----------

def load_data():
    xls = pd.ExcelFile(MERGED_FILE)
    meta = pd.read_excel(xls, "metadata")
    aec = pd.read_excel(xls, "aec_128")
    aec_cols = [c for c in aec.columns if c.startswith("aec_")]

    df = meta.merge(aec[["PatientID"] + aec_cols], on="PatientID", how="inner")
    df["Sex_M"] = (df["PatientSex"] == "M").astype(int)
    df = df.dropna(subset=["SMI"] + CLINIC_FEATURES + aec_cols).reset_index(drop=True)

    X = df[aec_cols].to_numpy(dtype=float)
    df["mean_mA"] = X.mean(axis=1)
    for name, (start, end) in AEC_SEGMENTS.items():
        window = X[:, start - 1:end]
        df[name] = np.diff(window, axis=1).mean(axis=1)

    return df, aec_cols


def make_strata(df):
    """RepeatedStratifiedKFold용 층화 라벨: (성별 x 성별-내 SMI quartile)."""
    bins = df.groupby("Sex_M")["SMI"].transform(lambda s: pd.qcut(s, q=4, labels=False, duplicates="drop"))
    return (df["Sex_M"].to_numpy() * 4 + bins.to_numpy()).astype(int)


def zscore_target_by_sex(df, tr):
    """train fold의 성별별 mu/sigma로 전체 SMI를 z-score. leakage 없음(train만으로 적합)."""
    y = np.zeros(len(df))
    train_df = df.iloc[tr]
    for sex_val in (0, 1):
        mu = train_df.loc[train_df["Sex_M"] == sex_val, "SMI"].mean()
        sigma = train_df.loc[train_df["Sex_M"] == sex_val, "SMI"].std()
        mask = (df["Sex_M"] == sex_val).to_numpy()
        y[mask] = (df.loc[mask, "SMI"].to_numpy(float) - mu) / sigma
    return y


# ---------- Cross-Attention 회귀 모델 (09의 CrossAttentionNet에서 head만 회귀용으로 변경) ----------

class CrossAttentionRegressor(nn.Module):
    def __init__(self, n_clinic, seq_len, d_model=D_MODEL, n_heads=N_HEADS, dropout=0.2, clinic_skip=False):
        super().__init__()
        self.clinic_value_proj = nn.Linear(1, d_model)
        self.clinic_id_embed = nn.Embedding(n_clinic, d_model)
        self.aec_value_proj = nn.Linear(1, d_model)
        self.aec_pos_embed = nn.Embedding(seq_len, d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.register_buffer("clinic_ids", torch.arange(n_clinic))
        self.register_buffer("aec_ids", torch.arange(seq_len))

        self.clinic_skip = clinic_skip
        if clinic_skip:
            self.clinic_skip_linear = nn.Linear(n_clinic, 1)

    def forward(self, clinic, aec):
        clinic_tok = self.clinic_value_proj(clinic.unsqueeze(-1)) + self.clinic_id_embed(self.clinic_ids)
        aec_tok = self.aec_value_proj(aec.unsqueeze(-1)) + self.aec_pos_embed(self.aec_ids)

        attn_out, attn_weights = self.cross_attn(query=clinic_tok, key=aec_tok, value=aec_tok,
                                                  need_weights=True, average_attn_weights=True)
        h = self.norm(clinic_tok + attn_out)
        pooled = h.mean(dim=1)
        pred = self.head(pooled).squeeze(-1)
        if self.clinic_skip:
            pred = pred + self.clinic_skip_linear(clinic).squeeze(-1)
        return pred, attn_weights  # attn_weights: (B, n_clinic, seq_len)


def patient_wise_normalize(aec):
    mu = aec.mean(axis=1, keepdims=True)
    sigma = aec.std(axis=1, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (aec - mu) / sigma


def scale_aec(aec_tr, aec_te, mode):
    if mode == "global":
        mu = aec_tr.mean()
        sigma = aec_tr.std()
        return (aec_tr - mu) / sigma, (aec_te - mu) / sigma
    if mode == "patient_wise":
        return patient_wise_normalize(aec_tr), patient_wise_normalize(aec_te)
    raise ValueError(f"unknown aec_mode: {mode}")


def train_one_fold_ca(clinic_tr, aec_tr, y_tr, clinic_te, aec_te, n_clinic, clinic_skip):
    torch.manual_seed(RANDOM_STATE)
    model = CrossAttentionRegressor(n_clinic=n_clinic, seq_len=SEQ_LEN, clinic_skip=clinic_skip).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()

    clinic_tr_t = torch.tensor(clinic_tr, dtype=torch.float32, device=DEVICE)
    aec_tr_t = torch.tensor(aec_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)

    n = len(y_tr)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            pred, _ = model(clinic_tr_t[idx], aec_tr_t[idx])
            loss = loss_fn(pred, y_tr_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        clinic_te_t = torch.tensor(clinic_te, dtype=torch.float32, device=DEVICE)
        aec_te_t = torch.tensor(aec_te, dtype=torch.float32, device=DEVICE)
        pred, attn_w = model(clinic_te_t, aec_te_t)
        pred = pred.cpu().numpy()
        attn_w = attn_w.cpu().numpy()
    return pred, attn_w


# ---------- 지표 ----------

def eval_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "spearman": stats.spearmanr(y_true, y_pred).correlation,
    }


# ---------- 변형별 실행 ----------

def run_ridge_variant(df, folds, cols, name):
    rows = []
    for fold_id, (tr, te) in enumerate(folds):
        y = zscore_target_by_sex(df, tr)
        Xtr = df.iloc[tr][cols].to_numpy(float)
        Xte = df.iloc[te][cols].to_numpy(float)
        scaler = StandardScaler().fit(Xtr)
        reg = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(Xtr), y[tr])
        pred = reg.predict(scaler.transform(Xte))
        row = {"variant": name, "fold": fold_id}
        row.update(eval_metrics(y[te], pred))
        rows.append(row)
    return pd.DataFrame(rows)


def run_cross_attention_variant(df, folds, aec_cols, aec_mode, clinic_skip, tag):
    rows = []
    attn_accum = np.zeros((len(CLINIC_FEATURES), SEQ_LEN))
    for fold_id, (tr, te) in enumerate(folds):
        y = zscore_target_by_sex(df, tr)

        clinic_scaler = StandardScaler().fit(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_tr = clinic_scaler.transform(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_te = clinic_scaler.transform(df.iloc[te][CLINIC_FEATURES].to_numpy(float))

        aec_tr_raw = df.iloc[tr][aec_cols].to_numpy(float)
        aec_te_raw = df.iloc[te][aec_cols].to_numpy(float)
        aec_tr, aec_te = scale_aec(aec_tr_raw, aec_te_raw, aec_mode)

        pred, attn_w = train_one_fold_ca(clinic_tr, aec_tr, y[tr], clinic_te, aec_te,
                                          n_clinic=len(CLINIC_FEATURES), clinic_skip=clinic_skip)
        row = {"variant": f"CrossAttn({tag})", "fold": fold_id}
        row.update(eval_metrics(y[te], pred))
        rows.append(row)
        attn_accum += attn_w.mean(axis=0)
        print(f"  [SMI/{tag}] fold {fold_id + 1}/{len(folds)} 완료 (r2={row['r2']:.3f})")
    return pd.DataFrame(rows), attn_accum / len(folds)


def run_residual_predictivity_test(df, folds):
    """clinic-only Ridge의 test-fold 잔차를, aec 스칼라 피처(train만으로 적합)가
    얼마나 예측하는지 fold별로 검정한다. 이게 되면 AEC가 clinic 위에 주는 진짜 추가
    정보의 가장 직접적인 증거."""
    rows = []
    for fold_id, (tr, te) in enumerate(folds):
        y = zscore_target_by_sex(df, tr)

        clinic_scaler = StandardScaler().fit(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_tr = clinic_scaler.transform(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_te = clinic_scaler.transform(df.iloc[te][CLINIC_FEATURES].to_numpy(float))
        clinic_reg = Ridge(alpha=RIDGE_ALPHA).fit(clinic_tr, y[tr])
        resid_tr = y[tr] - clinic_reg.predict(clinic_tr)
        resid_te = y[te] - clinic_reg.predict(clinic_te)

        aec_scaler = StandardScaler().fit(df.iloc[tr][AEC_SCALAR_FEATURES].to_numpy(float))
        aec_tr = aec_scaler.transform(df.iloc[tr][AEC_SCALAR_FEATURES].to_numpy(float))
        aec_te = aec_scaler.transform(df.iloc[te][AEC_SCALAR_FEATURES].to_numpy(float))
        resid_reg = Ridge(alpha=RIDGE_ALPHA).fit(aec_tr, resid_tr)
        resid_pred_te = resid_reg.predict(aec_te)

        rows.append({
            "fold": fold_id,
            "r2": r2_score(resid_te, resid_pred_te),
            "spearman": stats.spearmanr(resid_te, resid_pred_te).correlation,
        })
    return pd.DataFrame(rows)


# ---------- 시각화 ----------

def plot_r2_comparison(all_scores):
    summary = all_scores.groupby("variant")["r2"].agg(["mean", "std"]).reindex(
        all_scores["variant"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary.index, summary["mean"], yerr=summary["std"], capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("R² (20-fold 평균 ± SD)")
    ax.set_title("Continuous SMI(성별-내 z-score) 예측: clinic-only vs aec 추가/cross-attention")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "smi_r2_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def plot_attention(attn_mean, tag):
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(attn_mean, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(CLINIC_FEATURES)))
    ax.set_yticklabels(CLINIC_FEATURES)
    ax.set_xlabel("aec index (1=치골 pubis, 128=간 상부 liver upper)")
    ax.set_title(f"Cross-Attention 가중치 (SMI 회귀, {tag}, clinic 토큰별 query, fold 평균) "
                  "- 빨간 음영은 pointwise FDR 유의구간")
    fig.colorbar(im, ax=ax, label="attention weight")

    for start, end in SIGNIFICANT_SEGMENTS:
        ax.axvspan(start - 1, end - 1, color="red", alpha=0.15)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"cross_attention_smi_weights_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def summarize(name, scores):
    print(f"[{name}] R2={scores['r2'].mean():.4f}+/-{scores['r2'].std():.4f}  "
          f"RMSE={scores['rmse'].mean():.4f}  Spearman={scores['spearman'].mean():.4f}  "
          f"(n_fold={len(scores)})")


CA_VARIANTS = [
    {"tag": "global", "aec_mode": "global", "clinic_skip": False},
    {"tag": "global_skip", "aec_mode": "global", "clinic_skip": True},
    {"tag": "patient_wise", "aec_mode": "patient_wise", "clinic_skip": False},
    {"tag": "patient_wise_skip", "aec_mode": "patient_wise", "clinic_skip": True},
]


def main():
    df, aec_cols = load_data()
    strata = make_strata(df)

    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    folds = list(rkf.split(df, strata))
    print(f"=== Continuous SMI(성별-내 z-score) cross-attention 비교: "
          f"{N_SPLITS}-fold x {N_REPEATS} repeat = {len(folds)} folds "
          f"(n={len(df)}, device={DEVICE}) ===\n")

    clinic_scores = run_ridge_variant(df, folds, CLINIC_FEATURES, "1) clinic-only")
    print("clinic-only(Ridge, 20-fold) 완료")
    aec_scalar_scores = run_ridge_variant(df, folds, CLINIC_FEATURES + AEC_SCALAR_FEATURES,
                                           "2) clinic+aec(scalar)")
    print("clinic+aec(scalar, Ridge, 20-fold) 완료\n")

    ca_results = {}
    for variant in CA_VARIANTS:
        ca_scores, attn_mean = run_cross_attention_variant(
            df, folds, aec_cols, variant["aec_mode"], variant["clinic_skip"], tag=variant["tag"])
        ca_results[variant["tag"]] = ca_scores
        plot_attention(attn_mean, variant["tag"])
        print()

    all_scores = pd.concat(
        [clinic_scores, aec_scalar_scores] + list(ca_results.values()), ignore_index=True)

    print("=== 요약 ===")
    summarize("clinic-only (Ridge)", clinic_scores)
    summarize("clinic+aec(scalar, Ridge)", aec_scalar_scores)
    for variant in CA_VARIANTS:
        summarize(f"cross-attention ({variant['tag']})", ca_results[variant["tag"]])

    print("\n=== clinic-only 대비 Wilcoxon paired test (같은 fold) ===")
    baseline = clinic_scores.sort_values("fold")
    comparison_rows = []
    for name, scores in [("clinic+aec(scalar)", aec_scalar_scores)] + \
                        [(f"CrossAttn({v['tag']})", ca_results[v["tag"]]) for v in CA_VARIANTS]:
        sub = scores.sort_values("fold")
        row = {"variant": name}
        for metric in ["r2", "rmse", "spearman"]:
            diff = sub[metric].to_numpy() - baseline[metric].to_numpy()
            _, p = stats.wilcoxon(diff)
            row[f"{metric}_diff_vs_clinic"] = diff.mean()
            row[f"{metric}_wilcoxon_p"] = p
            print(f"[{metric}] {name} - clinic-only: mean_diff={diff.mean():+.4f}, p={p:.4g}")
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)

    print("\n=== 4단계: clinic 잔차를 aec 스칼라 피처가 예측하는가 (residual predictivity test) ===")
    resid_scores = run_residual_predictivity_test(df, folds)
    print(f"[residual R2] mean={resid_scores['r2'].mean():.4f}+/-{resid_scores['r2'].std():.4f}")
    print(f"[residual Spearman] mean={resid_scores['spearman'].mean():.4f}+/-{resid_scores['spearman'].std():.4f}")
    _, p_r2 = stats.wilcoxon(resid_scores["r2"].to_numpy())
    _, p_rho = stats.wilcoxon(resid_scores["spearman"].to_numpy())
    print(f"[one-sample Wilcoxon vs 0] r2 p={p_r2:.4g}, spearman p={p_rho:.4g}")

    plot_r2_comparison(all_scores)

    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        all_scores.to_excel(writer, sheet_name="SMI_CV_scores", index=False)
        comparison_df.to_excel(writer, sheet_name="SMI_Summary_vs_clinic", index=False)
        resid_scores.to_excel(writer, sheet_name="SMI_residual_test", index=False)
    print(f"\n저장: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
