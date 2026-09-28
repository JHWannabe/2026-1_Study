"""
지금까지(04~07)의 모델은 남녀를 풀링한 하나의 모델이었고, Sex_M은 다른 clinic 변수와
똑같은 취급을 받는 입력 하나였을 뿐이다. 그런데 0701의 pointwise FDR 결과는 남/여의
유의구간이 서로 다르다:
  - 남: aec_62~71(48~55%), aec_103~113(80~88%), aec_120~124(94~97%)  (3개 구간)
  - 여: aec_104~115(81~90%)                                          (1개 구간만, 위치도 다름)
그런데 04_build_model_features.py의 slope_48_55/slope_94_97은 "남성 유의구간"을 남녀
구분 없이 모든 환자에게 그대로 적용했고, 06/07의 attention-plot 음영/FDA-prior도 남녀를
합친 근사 구간 하나만 썼다. 즉 여성 환자가 남성에서 발견된 신호로 채점되고 있었다.

이 스크립트는 그 미스매치를 교정한 두 가지를 같은 20-fold 위에서 비교한다:

1) 로지스틱 스칼라 피처: "성별-정합(sex-aware)" 피처
   - slope_48_55_sa, slope_94_97_sa: 여성 행은 0으로 마스킹(여성에게 유의하지 않은 구간이므로)
   - slope_80_90_sa: 남/여 각자의 실제 유의구간 윈도우(103~113 / 104~115)를 각자 사용
   vs 기존 04의 "풀링(pooled)" 피처(남성 구간을 남녀 모두에게 그대로 적용, 103~115로 뭉뚱그림)
   vs clinic-only

2) Cross-Attention의 FDA-prior(07의 lambda_prior)를 성별 조건부로 확장:
   같은 배치 안에서도 남성 행은 prior_M(3구간), 여성 행은 prior_F(1구간, 104~115)로
   각각 다른 정규화 타깃을 적용 -> "여성에게 남성 구간을 보라고 유도하던" 07의 pooled prior
   대비 개선이 있는지 확인.
3) aec 정규화 축(09/07과 동일 아이디어): 위 no_prior/pooled_prior/sex_prior는 전부 "global"
   정규화만 썼다. 09에서 확인한 "patient-wise 정규화는 skip이 있으면 global과 차이가 없다"는
   결론이 성별 조건부 prior와 결합해도 유지되는지, 세 변형을 patient-wise(pw_*) 버전으로도
   재실행해서 대조한다.

모든 비교는 동일한 RepeatedStratifiedKFold(5-fold x 4 repeat = 20 fold, random_state=42)
위에서 Wilcoxon paired test.

출력: 콘솔 비교표, figures/cross_attention_sexprior_weights_{tag}.png,
      excel/model_comparison_results.xlsx에 SexStrat_* 시트 추가
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_DIR = os.path.join(BASE_DIR, "..", "excel")
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures", "08")
os.makedirs(FIGURES_DIR, exist_ok=True)

MERGED_FILE = os.path.join(EXCEL_DIR, "강남_liver_merged_features.xlsx")
LABEL_FILE = os.path.join(EXCEL_DIR, "강남_aec_128.xlsx")
RESULTS_FILE = os.path.join(EXCEL_DIR, "model_comparison_results.xlsx")

CLINIC_FEATURES = ["PatientAge", "Sex_M", "Height", "Weight"]

# 0701 pointwise FDR 유의구간(1-based, inclusive) - 남녀가 다르다는 게 핵심 관찰
MALE_SEGMENTS = [(62, 71), (103, 113), (120, 124)]
FEMALE_SEGMENTS = [(104, 115)]
POOLED_SEGMENTS = [(62, 71), (103, 115), (120, 124)]  # 04/06/07이 지금까지 쓰던 남녀 통합 근사 구간

SEQ_LEN = 128
N_SPLITS = 5
N_REPEATS = 4  # 07과 동일 20-fold(검정력 확보된 설정)를 유지
RANDOM_STATE = 42
D_MODEL = 16
N_HEADS = 2
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_PRIOR = 0.05
PRIOR_BOOST = 5.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    xls = pd.ExcelFile(MERGED_FILE)
    meta = pd.read_excel(xls, "metadata")
    aec = pd.read_excel(xls, "aec_128")
    aec_cols = [c for c in aec.columns if c.startswith("aec_")]
    df = meta.merge(aec[["PatientID"] + aec_cols], on="PatientID", how="inner")

    label_xls = pd.ExcelFile(LABEL_FILE)
    sheets = ["M_normal", "M_low_SMI", "F_normal", "F_low_SMI"]
    labels = pd.concat(
        [pd.read_excel(label_xls, s, usecols=["PatientID", "Output"]) for s in sheets],
        ignore_index=True,
    )
    df = df.merge(labels, on="PatientID", how="left")
    assert df["Output"].isna().sum() == 0, "Output 라벨 조인 실패한 환자 존재"

    df["Sex_M"] = (df["PatientSex"] == "M").astype(int)
    return df.reset_index(drop=True), aec_cols


def segment_slope(X, start, end):
    window = X[:, start - 1:end]
    return np.diff(window, axis=1).mean(axis=1)


def build_scalar_features(df, aec_cols):
    X = df[aec_cols].to_numpy(dtype=float)
    is_male = df["Sex_M"].to_numpy().astype(bool)

    out = pd.DataFrame(index=df.index)
    out["mean_mA"] = X.mean(axis=1)  # 남녀 공통 유의(0701) - 그대로 유지

    # 기존 04 방식: 남성 구간을 남녀 구분 없이 통합 근사 구간으로 그대로 적용
    out["slope_48_55_pooled"] = segment_slope(X, *POOLED_SEGMENTS[0])
    out["slope_80_90_pooled"] = segment_slope(X, *POOLED_SEGMENTS[1])
    out["slope_94_97_pooled"] = segment_slope(X, *POOLED_SEGMENTS[2])

    # 성별-정합 버전: 여성에게 유의하지 않은 구간은 0으로 마스킹, 80~90%대는 각자의 실제 구간 사용
    slope_4855_male = segment_slope(X, *MALE_SEGMENTS[0])
    slope_8090_male = segment_slope(X, *MALE_SEGMENTS[1])
    slope_9497_male = segment_slope(X, *MALE_SEGMENTS[2])
    slope_8090_female = segment_slope(X, *FEMALE_SEGMENTS[0])

    out["slope_48_55_sa"] = np.where(is_male, slope_4855_male, 0.0)
    out["slope_94_97_sa"] = np.where(is_male, slope_9497_male, 0.0)
    out["slope_80_90_sa"] = np.where(is_male, slope_8090_male, slope_8090_female)

    return out


def make_logreg():
    return LogisticRegression(class_weight="balanced", max_iter=1000)


def run_logreg_variant(df, y, folds, cols):
    rows = []
    for fold_id, (tr, te) in enumerate(folds):
        Xtr = df.iloc[tr][cols].to_numpy(float)
        Xte = df.iloc[te][cols].to_numpy(float)
        scaler = StandardScaler().fit(Xtr)
        clf = make_logreg().fit(scaler.transform(Xtr), y[tr])
        proba = clf.predict_proba(scaler.transform(Xte))[:, 1]
        rows.append({"fold": fold_id, "roc_auc": roc_auc_score(y[te], proba),
                      "pr_auc": average_precision_score(y[te], proba)})
    return pd.DataFrame(rows)


def build_prior_tensor(segments, boost):
    prior = np.ones(SEQ_LEN, dtype=np.float64)
    for start, end in segments:
        prior[start - 1:end] *= boost
    prior /= prior.sum()
    return torch.tensor(prior, dtype=torch.float32, device=DEVICE)


class CrossAttentionNet(nn.Module):
    def __init__(self, n_clinic, seq_len, d_model=D_MODEL, n_heads=N_HEADS, dropout=0.2):
        super().__init__()
        self.clinic_value_proj = nn.Linear(1, d_model)
        self.clinic_id_embed = nn.Embedding(n_clinic, d_model)
        self.aec_value_proj = nn.Linear(1, d_model)
        self.aec_pos_embed = nn.Embedding(seq_len, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )
        self.register_buffer("clinic_ids", torch.arange(n_clinic))
        self.register_buffer("aec_ids", torch.arange(seq_len))
        self.clinic_skip_linear = nn.Linear(n_clinic, 1)

    def forward(self, clinic, aec):
        clinic_tok = self.clinic_value_proj(clinic.unsqueeze(-1)) + self.clinic_id_embed(self.clinic_ids)
        aec_tok = self.aec_value_proj(aec.unsqueeze(-1)) + self.aec_pos_embed(self.aec_ids)
        attn_out, attn_weights = self.cross_attn(query=clinic_tok, key=aec_tok, value=aec_tok,
                                                  need_weights=True, average_attn_weights=True)
        h = self.norm(clinic_tok + attn_out)
        pooled = h.mean(dim=1)
        logit = self.head(pooled).squeeze(-1) + self.clinic_skip_linear(clinic).squeeze(-1)
        return logit, attn_weights  # attn_weights: (B, n_clinic, seq_len)


def prior_loss_pooled(attn_weights, prior_dist, eps=1e-8):
    attn_mean = attn_weights.mean(dim=(0, 1))
    attn_mean = attn_mean / attn_mean.sum()
    return -(prior_dist * torch.log(attn_mean + eps)).sum()


def prior_loss_sex(attn_weights, sex_raw, prior_M, prior_F, eps=1e-8):
    """attn_weights: (B, n_clinic, seq_len). 배치 안에서 남/여 행을 나눠 각자의 prior와 비교."""
    per_sample = attn_weights.mean(dim=1)  # (B, seq_len), clinic 4토큰 평균 -> 여전히 합=1인 분포
    male_mask = sex_raw == 1
    female_mask = ~male_mask
    loss = per_sample.new_zeros(())
    if male_mask.any():
        m_dist = per_sample[male_mask].mean(dim=0)
        m_dist = m_dist / m_dist.sum()
        loss = loss + (-(prior_M * torch.log(m_dist + eps)).sum())
    if female_mask.any():
        f_dist = per_sample[female_mask].mean(dim=0)
        f_dist = f_dist / f_dist.sum()
        loss = loss + (-(prior_F * torch.log(f_dist + eps)).sum())
    return loss


def train_one_fold(clinic_tr, aec_tr, y_tr, sex_tr, clinic_te, aec_te, n_clinic,
                    prior_mode="none", prior_pooled=None, prior_M=None, prior_F=None):
    torch.manual_seed(RANDOM_STATE)
    model = CrossAttentionNet(n_clinic=n_clinic, seq_len=SEQ_LEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                               dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    clinic_tr_t = torch.tensor(clinic_tr, dtype=torch.float32, device=DEVICE)
    aec_tr_t = torch.tensor(aec_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    sex_tr_t = torch.tensor(sex_tr, dtype=torch.long, device=DEVICE)

    n = len(y_tr)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            logit, attn_w = model(clinic_tr_t[idx], aec_tr_t[idx])
            loss = loss_fn(logit, y_tr_t[idx])
            if prior_mode == "pooled":
                loss = loss + LAMBDA_PRIOR * prior_loss_pooled(attn_w, prior_pooled)
            elif prior_mode == "sex":
                loss = loss + LAMBDA_PRIOR * prior_loss_sex(attn_w, sex_tr_t[idx], prior_M, prior_F)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        clinic_te_t = torch.tensor(clinic_te, dtype=torch.float32, device=DEVICE)
        aec_te_t = torch.tensor(aec_te, dtype=torch.float32, device=DEVICE)
        logit, attn_w = model(clinic_te_t, aec_te_t)
        proba = torch.sigmoid(logit).cpu().numpy()
        attn_w = attn_w.cpu().numpy()
    return proba, attn_w


def patient_wise_normalize(aec):
    """09_patient_wise_normalization.py와 동일: 각 환자를 자기 자신의 128개 값의
    평균/표준편차로 정규화(레벨 제거, 모양만 남김)."""
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


def run_cross_attention(df, y, folds, aec_cols, aec_mode, prior_mode, prior_pooled, prior_M, prior_F, tag):
    rows = []
    attn_accum_m = np.zeros((len(CLINIC_FEATURES), SEQ_LEN))
    attn_accum_f = np.zeros((len(CLINIC_FEATURES), SEQ_LEN))
    n_m_folds = 0
    n_f_folds = 0
    for fold_id, (tr, te) in enumerate(folds):
        clinic_scaler = StandardScaler().fit(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_tr = clinic_scaler.transform(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_te = clinic_scaler.transform(df.iloc[te][CLINIC_FEATURES].to_numpy(float))

        aec_tr_raw = df.iloc[tr][aec_cols].to_numpy(float)
        aec_te_raw = df.iloc[te][aec_cols].to_numpy(float)
        aec_tr, aec_te = scale_aec(aec_tr_raw, aec_te_raw, aec_mode)
        sex_tr = df.iloc[tr]["Sex_M"].to_numpy()
        sex_te = df.iloc[te]["Sex_M"].to_numpy()

        proba, attn_w = train_one_fold(clinic_tr, aec_tr, y[tr], sex_tr, clinic_te, aec_te,
                                        n_clinic=len(CLINIC_FEATURES), prior_mode=prior_mode,
                                        prior_pooled=prior_pooled, prior_M=prior_M, prior_F=prior_F)
        rows.append({"fold": fold_id, "roc_auc": roc_auc_score(y[te], proba),
                      "pr_auc": average_precision_score(y[te], proba)})

        male_te = sex_te == 1
        if male_te.any():
            attn_accum_m += attn_w[male_te].mean(axis=0)
            n_m_folds += 1
        if (~male_te).any():
            attn_accum_f += attn_w[~male_te].mean(axis=0)
            n_f_folds += 1
        print(f"  [SexStrat/{tag}] fold {fold_id + 1}/{len(folds)} 완료 (roc_auc={rows[-1]['roc_auc']:.3f})")
    return (pd.DataFrame(rows), attn_accum_m / max(n_m_folds, 1), attn_accum_f / max(n_f_folds, 1))


def plot_attention_by_sex(attn_m, attn_f, tag):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, attn_mean, sex_label, segments in [
        (axes[0], attn_m, "남성 테스트 케이스 평균", MALE_SEGMENTS),
        (axes[1], attn_f, "여성 테스트 케이스 평균", FEMALE_SEGMENTS),
    ]:
        im = ax.imshow(attn_mean, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(CLINIC_FEATURES)))
        ax.set_yticklabels(CLINIC_FEATURES)
        ax.set_title(f"{tag}: {sex_label} (빨간 음영 = 해당 성별 FDR 유의구간)")
        fig.colorbar(im, ax=ax, label="attention weight")
        for start, end in segments:
            ax.axvspan(start - 1, end - 1, color="red", alpha=0.15)
    axes[1].set_xlabel("aec index (1=치골 pubis, 128=간 상부 liver upper)")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"cross_attention_sexprior_weights_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def summarize(name, scores):
    print(f"[{name}] ROC-AUC={scores['roc_auc'].mean():.4f}+/-{scores['roc_auc'].std():.4f}  "
          f"PR-AUC={scores['pr_auc'].mean():.4f}+/-{scores['pr_auc'].std():.4f}  (n_fold={len(scores)})")


def wilcoxon_report(label, a, b, metric):
    diff = a[metric].to_numpy() - b[metric].to_numpy()
    _, p = stats.wilcoxon(diff)
    print(f"[{metric}] {label}: mean={diff.mean():+.4f}, Wilcoxon p={p:.4g}")


def main():
    df, aec_cols = load_data()
    y = df["Output"].to_numpy()
    n_male, n_female = int(df["Sex_M"].sum()), int((1 - df["Sex_M"]).sum())
    print(f"환자 수: 남={n_male}, 여={n_female}, 전체={len(df)}\n")

    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    folds = list(rkf.split(df, y))
    print(f"=== 성별-정합 모델 비교: {N_SPLITS}-fold x {N_REPEATS} repeat = {len(folds)} folds "
          f"(device={DEVICE}) ===\n")

    # --- 1) 로지스틱: pooled 스칼라 피처 vs 성별-정합(sex-aware) 스칼라 피처 ---
    feat_df = build_scalar_features(df, aec_cols)
    df_ext = pd.concat([df, feat_df], axis=1)

    clinic_scores = run_logreg_variant(df_ext, y, folds, CLINIC_FEATURES)
    pooled_cols = CLINIC_FEATURES + ["mean_mA", "slope_48_55_pooled", "slope_80_90_pooled", "slope_94_97_pooled"]
    sa_cols = CLINIC_FEATURES + ["mean_mA", "slope_48_55_sa", "slope_80_90_sa", "slope_94_97_sa"]
    pooled_scores = run_logreg_variant(df_ext, y, folds, pooled_cols)
    sa_scores = run_logreg_variant(df_ext, y, folds, sa_cols)

    print("--- 로지스틱 회귀 (20-fold) ---")
    summarize("clinic-only", clinic_scores)
    summarize("clinic+aec(pooled, 04 방식)", pooled_scores)
    summarize("clinic+aec(sex-aware)", sa_scores)
    wilcoxon_report("pooled - clinic-only", pooled_scores, clinic_scores, "roc_auc")
    wilcoxon_report("pooled - clinic-only", pooled_scores, clinic_scores, "pr_auc")
    wilcoxon_report("sex-aware - clinic-only", sa_scores, clinic_scores, "roc_auc")
    wilcoxon_report("sex-aware - clinic-only", sa_scores, clinic_scores, "pr_auc")
    wilcoxon_report("sex-aware - pooled", sa_scores, pooled_scores, "roc_auc")
    wilcoxon_report("sex-aware - pooled", sa_scores, pooled_scores, "pr_auc")
    print()

    # --- 2) Cross-Attention: pooled FDA-prior vs 성별 조건부 FDA-prior ---
    prior_pooled = build_prior_tensor(POOLED_SEGMENTS, PRIOR_BOOST)
    prior_M = build_prior_tensor(MALE_SEGMENTS, PRIOR_BOOST)
    prior_F = build_prior_tensor(FEMALE_SEGMENTS, PRIOR_BOOST)

    print("--- Cross-Attention (20-fold, global 정규화) ---")
    ca_none, m_none, f_none = run_cross_attention(df, y, folds, aec_cols, "global", "none", None, None, None, "no_prior")
    ca_pooled, m_pooled, f_pooled = run_cross_attention(df, y, folds, aec_cols, "global", "pooled", prior_pooled, None, None, "pooled_prior")
    ca_sex, m_sex, f_sex = run_cross_attention(df, y, folds, aec_cols, "global", "sex", None, prior_M, prior_F, "sex_prior")

    print("--- Cross-Attention (20-fold, patient-wise 정규화 - 09/07의 후속 대조) ---")
    ca_pw_none, m_pw_none, f_pw_none = run_cross_attention(df, y, folds, aec_cols, "patient_wise", "none", None, None, None, "pw_no_prior")
    ca_pw_pooled, m_pw_pooled, f_pw_pooled = run_cross_attention(df, y, folds, aec_cols, "patient_wise", "pooled", prior_pooled, None, None, "pw_pooled_prior")
    ca_pw_sex, m_pw_sex, f_pw_sex = run_cross_attention(df, y, folds, aec_cols, "patient_wise", "sex", None, prior_M, prior_F, "pw_sex_prior")

    plot_attention_by_sex(m_none, f_none, "no_prior")
    plot_attention_by_sex(m_pooled, f_pooled, "pooled_prior")
    plot_attention_by_sex(m_sex, f_sex, "sex_prior")
    plot_attention_by_sex(m_pw_none, f_pw_none, "pw_no_prior")
    plot_attention_by_sex(m_pw_pooled, f_pw_pooled, "pw_pooled_prior")
    plot_attention_by_sex(m_pw_sex, f_pw_sex, "pw_sex_prior")

    summarize("clinic-only", clinic_scores)
    summarize("cross-attn (no_prior = global_skip)", ca_none)
    summarize("cross-attn (pooled_prior)", ca_pooled)
    summarize("cross-attn (sex_prior)", ca_sex)
    summarize("cross-attn (pw_no_prior = patient_wise_skip)", ca_pw_none)
    summarize("cross-attn (pw_pooled_prior)", ca_pw_pooled)
    summarize("cross-attn (pw_sex_prior)", ca_pw_sex)
    wilcoxon_report("no_prior - clinic-only", ca_none, clinic_scores, "roc_auc")
    wilcoxon_report("no_prior - clinic-only", ca_none, clinic_scores, "pr_auc")
    wilcoxon_report("sex_prior - clinic-only", ca_sex, clinic_scores, "roc_auc")
    wilcoxon_report("sex_prior - clinic-only", ca_sex, clinic_scores, "pr_auc")
    wilcoxon_report("sex_prior - no_prior", ca_sex, ca_none, "roc_auc")
    wilcoxon_report("sex_prior - no_prior", ca_sex, ca_none, "pr_auc")
    wilcoxon_report("sex_prior - pooled_prior", ca_sex, ca_pooled, "roc_auc")
    wilcoxon_report("sex_prior - pooled_prior", ca_sex, ca_pooled, "pr_auc")

    print()
    wilcoxon_report("pw_no_prior - clinic-only", ca_pw_none, clinic_scores, "roc_auc")
    wilcoxon_report("pw_no_prior - clinic-only", ca_pw_none, clinic_scores, "pr_auc")
    wilcoxon_report("pw_no_prior - no_prior (global vs patient-wise)", ca_pw_none, ca_none, "roc_auc")
    wilcoxon_report("pw_no_prior - no_prior (global vs patient-wise)", ca_pw_none, ca_none, "pr_auc")
    wilcoxon_report("pw_sex_prior - clinic-only", ca_pw_sex, clinic_scores, "roc_auc")
    wilcoxon_report("pw_sex_prior - clinic-only", ca_pw_sex, clinic_scores, "pr_auc")
    wilcoxon_report("pw_sex_prior - pw_no_prior", ca_pw_sex, ca_pw_none, "roc_auc")
    wilcoxon_report("pw_sex_prior - pw_no_prior", ca_pw_sex, ca_pw_none, "pr_auc")
    wilcoxon_report("pw_sex_prior - pw_pooled_prior", ca_pw_sex, ca_pw_pooled, "roc_auc")
    wilcoxon_report("pw_sex_prior - pw_pooled_prior", ca_pw_sex, ca_pw_pooled, "pr_auc")
    wilcoxon_report("pw_sex_prior - sex_prior (global vs patient-wise)", ca_pw_sex, ca_sex, "roc_auc")
    wilcoxon_report("pw_sex_prior - sex_prior (global vs patient-wise)", ca_pw_sex, ca_sex, "pr_auc")

    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        clinic_scores.to_excel(writer, sheet_name="SexStrat_clinic_baseline", index=False)
        pooled_scores.to_excel(writer, sheet_name="SexStrat_logreg_pooled", index=False)
        sa_scores.to_excel(writer, sheet_name="SexStrat_logreg_sexaware", index=False)
        ca_none.to_excel(writer, sheet_name="SexStrat_CA_no_prior", index=False)
        ca_pooled.to_excel(writer, sheet_name="SexStrat_CA_pooled_prior", index=False)
        ca_sex.to_excel(writer, sheet_name="SexStrat_CA_sex_prior", index=False)
        ca_pw_none.to_excel(writer, sheet_name="SexStrat_CA_pw_no_prior", index=False)
        ca_pw_pooled.to_excel(writer, sheet_name="SexStrat_CA_pw_pooled_prior", index=False)
        ca_pw_sex.to_excel(writer, sheet_name="SexStrat_CA_pw_sex_prior", index=False)
    print(f"\n저장: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
