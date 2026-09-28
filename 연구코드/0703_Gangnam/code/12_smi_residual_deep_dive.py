"""
11번(cross_attention_continuous_smi)의 residual predictivity test에서 처음으로
"유의미한 양의 방향" 신호가 나왔다: clinic-only Ridge 잔차를 aec 스칼라 피처(4개)가
예측하는 Spearman rho=+0.055 (Wilcoxon one-sample p=0.0001). 이 스크립트는 이 결과를
세 방향으로 심화 검증한다.

1. 통계적 엄밀화 (permutation test + FDR):
   11번의 Wilcoxon 검정은 20-fold의 fold별 통계량을 마치 20개의 독립표본인 것처럼
   다뤘는데, 이는 fold가 서로 겹치는(5-fold x 4 repeat) repeated-CV 구조를 무시하는
   근사다. 여기서는 aec 피처와 잔차의 실제 짝을 깨뜨리는 permutation test(train 쪽만
   셔플, test는 실제 값 유지 -> fold 구조와 clinic 잔차 자체는 그대로 보존한 채 "우연히
   이 정도 상관이 나올 확률"을 직접 시뮬레이션)로 재검증한다. 또한 11번+이 스크립트가
   보고하는 모든 p-value(중복인 RMSE는 R2와 동일 순위라 제외, 총 18개)에 Benjamini-
  Hochberg FDR을 적용해서 다중비교를 통제한다.

2. 신호 분해:
   mean_mA, slope_48_55, slope_80_90, slope_94_97 4개를 각각 단독으로 residual test에
   돌려서 어느 피처가 신호를 주도하는지 확인한다. 10번(propensity matching)은 레벨
   신호(mean_mA)가 confounding에 가장 취약(matched 후 방향까지 역전되는 경우 있음)했고
   기울기 3개 구간은 방향을 유지했다고 보고했다 -> 이 잔차 신호도 slope 쪽이 더 강할
   것이라는 가설을 직접 검증.

3. Cross-Attention 내부 수렴 검증:
   11번의 global_skip 모델(logit = attention_head(pooled) + clinic_skip_linear(clinic))을
   동일하게 재학습하되, skip을 더하기 전의 순수 attention_head(pooled) 출력만 따로
   뽑아서 clinic-only Ridge 잔차와 상관되는지 본다. 선형 Ridge(2번)와 신경망의 attention
   경로가 서로 다른 방법론임에도 같은 방향의 잔차 신호에 수렴하면, 이는 신호가 아키텍처에
   따른 아티팩트가 아니라는 강한 교차검증 증거가 된다. (주의: 이 부분은 신경망 재학습
   비용 때문에 permutation이 아닌 11번과 같은 Wilcoxon one-sample 검정만 적용 -> 1번의
   permutation 검정보다 rigor가 낮은 확인용 체크로 한정.)

출력: 콘솔 비교표, figures/12/residual_permutation_null_{all4,mean_mA,slope_48_55,
      slope_80_90,slope_94_97}.png, figures/12/attn_convergence_scatter.png,
      excel/model_comparison_results.xlsx에 SMI_Permutation_*, SMI_FDR_summary,
      SMI_AttnConvergence 시트 추가
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
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_DIR = os.path.join(BASE_DIR, "..", "excel")
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures", "12")
os.makedirs(FIGURES_DIR, exist_ok=True)

MERGED_FILE = os.path.join(EXCEL_DIR, "강남_liver_merged_features.xlsx")
RESULTS_FILE = os.path.join(EXCEL_DIR, "model_comparison_results.xlsx")

CLINIC_FEATURES = ["PatientAge", "Sex_M", "Height", "Weight"]
AEC_SEGMENTS = {
    "slope_48_55": (62, 71),
    "slope_80_90": (103, 115),
    "slope_94_97": (120, 124),
}
AEC_SCALAR_FEATURES = ["mean_mA"] + list(AEC_SEGMENTS.keys())
SIGNIFICANT_SEGMENTS = [(62, 71), (103, 115), (120, 124)]

SEQ_LEN = 128
N_SPLITS = 5
N_REPEATS = 4  # 11번과 동일 -> 같은 random_state로 동일한 fold 재현
RANDOM_STATE = 42
RIDGE_ALPHA = 1.0
N_PERM = 2000

D_MODEL = 16
N_HEADS = 2
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 11번 콘솔 출력에서 그대로 가져온 p-value(RMSE는 R2와 순위가 동일해 중복이라 제외)
PRIOR_PVALUES_FROM_SCRIPT11 = {
    "11:clinic+aec(scalar) r2": 0.2774,
    "11:clinic+aec(scalar) spearman": 0.1054,
    "11:CrossAttn(global) r2": 1.907e-05,
    "11:CrossAttn(global) spearman": 0.2611,
    "11:CrossAttn(global_skip) r2": 0.8695,
    "11:CrossAttn(global_skip) spearman": 0.00486,
    "11:CrossAttn(patient_wise) r2": 1.907e-06,
    "11:CrossAttn(patient_wise) spearman": 0.002325,
    "11:CrossAttn(patient_wise_skip) r2": 0.0001049,
    "11:CrossAttn(patient_wise_skip) spearman": 0.3683,
    "11:residual r2 (vs 0)": 0.01208,
    "11:residual spearman (vs 0)": 0.0001049,
}


# ---------- 데이터 로드 (11번과 동일) ----------

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
    bins = df.groupby("Sex_M")["SMI"].transform(lambda s: pd.qcut(s, q=4, labels=False, duplicates="drop"))
    return (df["Sex_M"].to_numpy() * 4 + bins.to_numpy()).astype(int)


def zscore_target_by_sex(df, tr):
    y = np.zeros(len(df))
    train_df = df.iloc[tr]
    for sex_val in (0, 1):
        mu = train_df.loc[train_df["Sex_M"] == sex_val, "SMI"].mean()
        sigma = train_df.loc[train_df["Sex_M"] == sex_val, "SMI"].std()
        mask = (df["Sex_M"] == sex_val).to_numpy()
        y[mask] = (df.loc[mask, "SMI"].to_numpy(float) - mu) / sigma
    return y


# ---------- 1+2. Residual predictivity: 관측치 + permutation null (fold별 clinic 잔차는 캐싱) ----------

def precompute_fold_residuals(df, folds):
    cache = []
    for tr, te in folds:
        y = zscore_target_by_sex(df, tr)
        clinic_scaler = StandardScaler().fit(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_tr = clinic_scaler.transform(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_te = clinic_scaler.transform(df.iloc[te][CLINIC_FEATURES].to_numpy(float))
        clinic_reg = Ridge(alpha=RIDGE_ALPHA).fit(clinic_tr, y[tr])
        resid_tr = y[tr] - clinic_reg.predict(clinic_tr)
        resid_te = y[te] - clinic_reg.predict(clinic_te)
        cache.append({"tr": tr, "te": te, "resid_tr": resid_tr, "resid_te": resid_te})
    return cache


def residual_predictivity_once(df, cache, feature_cols, permute, rng):
    rhos, r2s = [], []
    for entry in cache:
        tr, te, resid_tr, resid_te = entry["tr"], entry["te"], entry["resid_tr"], entry["resid_te"]
        aec_scaler = StandardScaler().fit(df.iloc[tr][feature_cols].to_numpy(float))
        aec_tr = aec_scaler.transform(df.iloc[tr][feature_cols].to_numpy(float))
        aec_te = aec_scaler.transform(df.iloc[te][feature_cols].to_numpy(float))

        target_tr = rng.permutation(resid_tr) if permute else resid_tr
        reg = Ridge(alpha=RIDGE_ALPHA).fit(aec_tr, target_tr)
        pred_te = reg.predict(aec_te)

        rhos.append(stats.spearmanr(resid_te, pred_te).correlation)
        r2s.append(r2_score(resid_te, pred_te))
    return float(np.mean(rhos)), float(np.mean(r2s)), rhos, r2s


def permutation_test(df, cache, tag, feature_cols, n_perm=N_PERM, seed=RANDOM_STATE):
    obs_rho, obs_r2, obs_rhos, obs_r2s = residual_predictivity_once(
        df, cache, feature_cols, permute=False, rng=np.random.default_rng(seed))

    rng_master = np.random.default_rng(seed)
    null_rhos = np.empty(n_perm)
    for i in range(n_perm):
        rng = np.random.default_rng(rng_master.integers(2**32 - 1))
        null_rhos[i], _, _, _ = residual_predictivity_once(df, cache, feature_cols, permute=True, rng=rng)

    p_one_sided = (np.sum(null_rhos >= obs_rho) + 1) / (n_perm + 1)
    return {
        "tag": tag, "feature_set": ",".join(feature_cols), "obs_rho": obs_rho, "obs_r2": obs_r2,
        "null_mean": null_rhos.mean(), "null_sd": null_rhos.std(),
        "perm_p": p_one_sided, "n_perm": n_perm,
    }, null_rhos, obs_rhos


def plot_null_distribution(null_rhos, obs_rho, tag):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_rhos, bins=40, color="lightgray", edgecolor="white")
    ax.axvline(obs_rho, color="crimson", linewidth=2, label=f"관측값 rho={obs_rho:.4f}")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("permutation null: fold-평균 Spearman rho (train쪽 잔차-피처 짝 셔플)")
    ax.set_title(f"Residual predictivity permutation test - {tag}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"residual_permutation_null_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


# ---------- 3. Cross-Attention 내부 수렴 검증 ----------

class CrossAttentionRegressor(nn.Module):
    """11번과 동일 구조. forward가 skip 이전의 순수 attention_head 출력(head_only)도 반환."""

    def __init__(self, n_clinic, seq_len, d_model=D_MODEL, n_heads=N_HEADS, dropout=0.2, clinic_skip=True):
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
        head_only = self.head(pooled).squeeze(-1)
        pred = head_only
        if self.clinic_skip:
            pred = pred + self.clinic_skip_linear(clinic).squeeze(-1)
        return pred, attn_weights, head_only


def patient_wise_normalize(aec):
    mu = aec.mean(axis=1, keepdims=True)
    sigma = aec.std(axis=1, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (aec - mu) / sigma


def train_and_extract_head_only(clinic_tr, aec_tr, y_tr, clinic_te, aec_te):
    torch.manual_seed(RANDOM_STATE)
    model = CrossAttentionRegressor(n_clinic=len(CLINIC_FEATURES), seq_len=SEQ_LEN, clinic_skip=True).to(DEVICE)
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
            pred, _, _ = model(clinic_tr_t[idx], aec_tr_t[idx])
            loss = loss_fn(pred, y_tr_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        clinic_te_t = torch.tensor(clinic_te, dtype=torch.float32, device=DEVICE)
        aec_te_t = torch.tensor(aec_te, dtype=torch.float32, device=DEVICE)
        pred, _, head_only = model(clinic_te_t, aec_te_t)
        pred = pred.cpu().numpy()
        head_only = head_only.cpu().numpy()
    return pred, head_only


def run_attention_convergence_check(df, folds, aec_cols, cache):
    rows = []
    for fold_id, ((tr, te), entry) in enumerate(zip(folds, cache)):
        y = zscore_target_by_sex(df, tr)

        clinic_scaler = StandardScaler().fit(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_tr = clinic_scaler.transform(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_te = clinic_scaler.transform(df.iloc[te][CLINIC_FEATURES].to_numpy(float))

        aec_tr_raw = df.iloc[tr][aec_cols].to_numpy(float)
        aec_te_raw = df.iloc[te][aec_cols].to_numpy(float)
        mu, sigma = aec_tr_raw.mean(), aec_tr_raw.std()
        aec_tr = (aec_tr_raw - mu) / sigma
        aec_te = (aec_te_raw - mu) / sigma

        pred, head_only = train_and_extract_head_only(clinic_tr, aec_tr, y[tr], clinic_te, aec_te)

        resid_te = entry["resid_te"]  # 같은 fold의 clinic-only Ridge 잔차(1번/2번과 동일 정의)
        rho_pred = stats.spearmanr(y[te], pred).correlation
        rho_head_vs_resid = stats.spearmanr(resid_te, head_only).correlation
        rows.append({"fold": fold_id, "r2_total": r2_score(y[te], pred), "spearman_total": rho_pred,
                      "spearman_headonly_vs_clinicresid": rho_head_vs_resid})
        print(f"  [AttnConverge] fold {fold_id + 1}/{len(folds)} 완료 "
              f"(total r2={rows[-1]['r2_total']:.3f}, head_only vs resid rho={rho_head_vs_resid:.3f})")
    return pd.DataFrame(rows)


def plot_attention_convergence(conv_df, linear_obs_rho):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(np.zeros(len(conv_df)) + np.random.default_rng(0).normal(0, 0.02, len(conv_df)),
               conv_df["spearman_headonly_vs_clinicresid"], alpha=0.6, label="fold별 값(신경망 attention 경로)")
    ax.axhline(conv_df["spearman_headonly_vs_clinicresid"].mean(), color="crimson",
               label=f"신경망 평균={conv_df['spearman_headonly_vs_clinicresid'].mean():.4f}")
    ax.axhline(linear_obs_rho, color="steelblue", linestyle="--",
               label=f"선형 Ridge(스칼라 피처) 관측값={linear_obs_rho:.4f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([])
    ax.set_ylabel("Spearman(clinic-only 잔차, aec 유래 신호)")
    ax.set_title("잔차 신호 수렴: 선형 Ridge vs Cross-Attention head_only")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "attn_convergence_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def main():
    df, aec_cols = load_data()
    strata = make_strata(df)
    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    folds = list(rkf.split(df, strata))
    print(f"=== SMI residual 심화 검증: {N_SPLITS}-fold x {N_REPEATS} repeat = {len(folds)} folds "
          f"(n={len(df)}, device={DEVICE}) ===\n")

    cache = precompute_fold_residuals(df, folds)

    # ---- 1+2. permutation test: all4 + 개별 피처 4개 ----
    print("=== 1+2. Residual predictivity permutation test (all4 + 개별 피처) ===")
    feature_sets = {"all4": AEC_SCALAR_FEATURES}
    feature_sets.update({name: [name] for name in AEC_SCALAR_FEATURES})

    perm_summary_rows = []
    perm_pvalues = {}
    for tag, cols in feature_sets.items():
        result, null_rhos, obs_rhos = permutation_test(df, cache, tag, cols)
        perm_summary_rows.append(result)
        perm_pvalues[f"12:permutation residual rho ({tag})"] = result["perm_p"]
        plot_null_distribution(null_rhos, result["obs_rho"], tag)
        print(f"[{tag}] 관측 rho={result['obs_rho']:+.4f}  null 평균={result['null_mean']:+.4f}"
              f"+/-{result['null_sd']:.4f}  permutation p={result['perm_p']:.4g}")
    perm_summary_df = pd.DataFrame(perm_summary_rows)

    # ---- 3. Cross-Attention 내부 수렴 검증 ----
    print("\n=== 3. Cross-Attention head_only vs clinic-residual 수렴 검증 ===")
    conv_df = run_attention_convergence_check(df, folds, aec_cols, cache)
    _, p_conv = stats.wilcoxon(conv_df["spearman_headonly_vs_clinicresid"].to_numpy())
    print(f"\n[AttnConverge] head_only vs clinic잔차 Spearman 평균="
          f"{conv_df['spearman_headonly_vs_clinicresid'].mean():+.4f}+/-"
          f"{conv_df['spearman_headonly_vs_clinicresid'].std():.4f}, Wilcoxon(vs 0) p={p_conv:.4g}")
    all4_obs_rho = perm_summary_df.loc[perm_summary_df["tag"] == "all4", "obs_rho"].iloc[0]
    plot_attention_convergence(conv_df, all4_obs_rho)

    # ---- FDR 보정: 11번 + 12번의 모든 p-value ----
    print("\n=== FDR 보정 (11번+12번 전체 p-value, Benjamini-Hochberg) ===")
    all_pvalues = dict(PRIOR_PVALUES_FROM_SCRIPT11)
    all_pvalues.update(perm_pvalues)
    all_pvalues["12:AttnConverge head_only vs resid (vs 0)"] = p_conv

    labels = list(all_pvalues.keys())
    raw_p = np.array([all_pvalues[k] for k in labels])
    reject, adj_p, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")
    fdr_df = pd.DataFrame({"test": labels, "raw_p": raw_p, "fdr_adj_p": adj_p, "significant_fdr05": reject})
    fdr_df = fdr_df.sort_values("raw_p").reset_index(drop=True)
    pd.set_option("display.width", 160)
    print(fdr_df.to_string(index=False, float_format=lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)))

    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        perm_summary_df.to_excel(writer, sheet_name="SMI_Permutation_test", index=False)
        conv_df.to_excel(writer, sheet_name="SMI_AttnConvergence", index=False)
        fdr_df.to_excel(writer, sheet_name="SMI_FDR_summary", index=False)
    print(f"\n저장: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
