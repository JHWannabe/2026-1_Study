"""
06_cross_attention_model.py의 후속. global_skip이 지금까지 최선(0.7525)이지만 여전히
clinic-only(0.7764)에 못 미쳤고, 06은 N_REPEATS=1(5-fold만) 이라 통계 검정력이 약했다.
이번 스크립트는 네 가지를 검증한다:

1) N_REPEATS=4로 실제 5x4=20-fold 재실행 -> global_skip vs clinic-only 갭이
   노이즈인지 실제인지 검정력을 확보해서 재확인.
2) skip warm-start(two-stage): clinic_skip_linear가 이미 clinic-only와 동등한 선형항을
   책임지는데, attention branch와 처음부터 같이 학습하면 attention의 초기 노이즈가
   공유 로짓(logit = head(pooled) + skip_linear(clinic))을 통해 skip 학습 신호까지
   흔들 수 있다는 가설. EPOCHS=60 예산은 그대로 두고, 앞 WARMUP_EPOCHS 동안은
   attn_scale=0(순수 skip+clinic_tok만 학습), 이후 RAMP_EPOCHS 동안 0->1로 선형 램프,
   나머지는 attn_scale=1로 정상 학습.
3) FDA 유의구간 attention prior: 06에서 attention이 유의구간과 "부분적으로만" 겹친다는
   관찰(index~114만 일치)을 이용해, attention이 유의구간(48~55%, 80~90%, 94~97%) 쪽에
   더 많은 확률질량을 두도록 유도하는 cross-entropy 정규화 항(lambda_prior)을 손실에 추가.
4) aec 정규화 축(09_patient_wise_normalization.py와 동일 아이디어): 위 1~3은 전부
   "global"(population 단일 스칼라, 레벨 보존) 정규화만 썼다. 09에서 "patient-wise"
   (환자 자신의 평균/표준편차, 레벨 제거) 정규화가 skip 없이는 global보다 유의하게
   나았지만 skip이 있으면 차이가 사라진다는 결과가 나왔는데, 07의 모델은 항상 skip이
   켜져 있으므로(clinic_skip_linear가 무조건 forward에 포함) warmstart/prior와 결합해도
   같은 결론(차이 없음)이 유지되는지 각 변형을 patient-wise 버전(pw_*)으로도 재실행해서
   직접 대조한다.

세 가지(warmstart/prior/조합) x 두 정규화(global/patient-wise) = 8개 변형을 같은 20-fold
위에서 비교하고, 전부 clinic-only(logreg, 같은 fold)와 Wilcoxon paired test.

출력: 콘솔 비교표, figures/cross_attention_v2_weights_{tag}.png,
      excel/model_comparison_results.xlsx에 CrossAttnV2_* 시트 추가
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
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures", "07")
os.makedirs(FIGURES_DIR, exist_ok=True)

MERGED_FILE = os.path.join(EXCEL_DIR, "강남_liver_merged_features.xlsx")
LABEL_FILE = os.path.join(EXCEL_DIR, "강남_aec_128.xlsx")
RESULTS_FILE = os.path.join(EXCEL_DIR, "model_comparison_results.xlsx")

CLINIC_FEATURES = ["PatientAge", "Sex_M", "Height", "Weight"]
SIGNIFICANT_SEGMENTS = [(62, 71), (103, 115), (120, 124)]  # pointwise FDR 유의구간(1-based, inclusive)

SEQ_LEN = 128
N_SPLITS = 5
N_REPEATS = 4  # 06에서는 1이었음(검정력 부족) -> 실제 5x4=20-fold로 재실행
RANDOM_STATE = 42
D_MODEL = 16
N_HEADS = 2
EPOCHS = 60  # 06의 global_skip과 동일 예산(공정 비교) - warmstart는 이 60 안에서 일정만 재배치
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

WARMUP_EPOCHS = 15   # 이 동안 attn_scale=0 (skip+clinic_tok만 학습)
RAMP_EPOCHS = 10     # 이 동안 attn_scale 0->1 선형 램프
LAMBDA_PRIOR = 0.05  # FDA 유의구간 prior 정규화 강도
PRIOR_BOOST = 5.0    # 유의구간 위치에 부여하는 상대 가중치(1.0=uniform과 동일)

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


def build_prior_distribution(seq_len, segments, boost):
    """유의구간(1-based inclusive)에 boost배 가중치를 준 뒤 정규화한 target 분포."""
    prior = np.ones(seq_len, dtype=np.float64)
    for start, end in segments:
        prior[start - 1:end] *= boost
    prior /= prior.sum()
    return torch.tensor(prior, dtype=torch.float32, device=DEVICE)


class CrossAttentionNet(nn.Module):
    """clinic_skip=True면 clinic-only 로지스틱회귀와 동등한 선형항을 최종 로짓에 residual로
    더해준다(DAFT류 tabular-skip). forward에 attn_scale을 받아 attention 기여도를
    0(=skip만)에서 1(=정상)까지 조절할 수 있게 해서 warm-start 스케줄을 구현한다."""

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

    def forward(self, clinic, aec, attn_scale=1.0):
        clinic_tok = self.clinic_value_proj(clinic.unsqueeze(-1)) + self.clinic_id_embed(self.clinic_ids)
        aec_tok = self.aec_value_proj(aec.unsqueeze(-1)) + self.aec_pos_embed(self.aec_ids)

        attn_out, attn_weights = self.cross_attn(query=clinic_tok, key=aec_tok, value=aec_tok,
                                                  need_weights=True, average_attn_weights=True)
        h = self.norm(clinic_tok + attn_scale * attn_out)
        pooled = h.mean(dim=1)
        logit = self.head(pooled).squeeze(-1)
        if self.clinic_skip:
            logit = logit + self.clinic_skip_linear(clinic).squeeze(-1)
        return logit, attn_weights  # attn_weights: (B, n_clinic, seq_len), attn_scale과 무관하게 계산됨


def attn_scale_schedule(epoch, warmup_epochs, ramp_epochs):
    if warmup_epochs == 0 and ramp_epochs == 0:
        return 1.0
    if epoch < warmup_epochs:
        return 0.0
    if epoch < warmup_epochs + ramp_epochs:
        return (epoch - warmup_epochs + 1) / ramp_epochs
    return 1.0


def prior_reg_loss(attn_weights, prior_dist, eps=1e-8):
    """attn_weights: (B, n_clinic, seq_len). 배치+clinic 토큰 평균 분포와 prior_dist 사이
    cross-entropy -> prior가 높은 위치에 attention이 더 몰리도록 유도(부드러운 유도, 강제 아님)."""
    attn_mean = attn_weights.mean(dim=(0, 1))
    attn_mean = attn_mean / attn_mean.sum()
    return -(prior_dist * torch.log(attn_mean + eps)).sum()


def train_one_fold(clinic_tr, aec_tr, y_tr, clinic_te, aec_te, n_clinic,
                    warmup_epochs=0, ramp_epochs=0, lambda_prior=0.0, prior_dist=None):
    torch.manual_seed(RANDOM_STATE)
    model = CrossAttentionNet(n_clinic=n_clinic, seq_len=SEQ_LEN, clinic_skip=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                               dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    clinic_tr_t = torch.tensor(clinic_tr, dtype=torch.float32, device=DEVICE)
    aec_tr_t = torch.tensor(aec_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)

    n = len(y_tr)
    model.train()
    for epoch in range(EPOCHS):
        scale = attn_scale_schedule(epoch, warmup_epochs, ramp_epochs)
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            logit, attn_w = model(clinic_tr_t[idx], aec_tr_t[idx], attn_scale=scale)
            loss = loss_fn(logit, y_tr_t[idx])
            if lambda_prior > 0:
                loss = loss + lambda_prior * prior_reg_loss(attn_w, prior_dist)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        clinic_te_t = torch.tensor(clinic_te, dtype=torch.float32, device=DEVICE)
        aec_te_t = torch.tensor(aec_te, dtype=torch.float32, device=DEVICE)
        logit, attn_w = model(clinic_te_t, aec_te_t, attn_scale=1.0)
        proba = torch.sigmoid(logit).cpu().numpy()
        attn_w = attn_w.cpu().numpy()
    return proba, attn_w


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


def patient_wise_normalize(aec):
    """09_patient_wise_normalization.py와 동일: 각 환자를 자기 자신의 128개 값의
    평균/표준편차로 정규화(레벨 제거, 모양만 남김). fold의 train/test 구분과 무관하게
    환자 자신의 값만 쓰므로 leakage가 없다."""
    mu = aec.mean(axis=1, keepdims=True)
    sigma = aec.std(axis=1, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (aec - mu) / sigma


def scale_aec(aec_tr, aec_te, mode):
    """'global'은 train fold 전체를 합친 단일 평균/표준편차(레벨 보존).
    'patient_wise'는 각 환자 자신의 평균/표준편차(레벨 제거, 모양만 남김)."""
    if mode == "global":
        mu = aec_tr.mean()
        sigma = aec_tr.std()
        return (aec_tr - mu) / sigma, (aec_te - mu) / sigma
    if mode == "patient_wise":
        return patient_wise_normalize(aec_tr), patient_wise_normalize(aec_te)
    raise ValueError(f"unknown aec_mode: {mode}")


def run_cross_attention(df, y, folds, aec_cols, aec_mode, warmup_epochs, ramp_epochs, lambda_prior, prior_dist, tag):
    rows = []
    attn_accum = np.zeros((len(CLINIC_FEATURES), SEQ_LEN))
    for fold_id, (tr, te) in enumerate(folds):
        clinic_scaler = StandardScaler().fit(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_tr = clinic_scaler.transform(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_te = clinic_scaler.transform(df.iloc[te][CLINIC_FEATURES].to_numpy(float))

        aec_tr_raw = df.iloc[tr][aec_cols].to_numpy(float)
        aec_te_raw = df.iloc[te][aec_cols].to_numpy(float)
        aec_tr, aec_te = scale_aec(aec_tr_raw, aec_te_raw, aec_mode)

        proba, attn_w = train_one_fold(clinic_tr, aec_tr, y[tr], clinic_te, aec_te,
                                        n_clinic=len(CLINIC_FEATURES),
                                        warmup_epochs=warmup_epochs, ramp_epochs=ramp_epochs,
                                        lambda_prior=lambda_prior, prior_dist=prior_dist)
        rows.append({"fold": fold_id, "roc_auc": roc_auc_score(y[te], proba),
                      "pr_auc": average_precision_score(y[te], proba)})
        attn_accum += attn_w.mean(axis=0)
        print(f"  [CrossAttentionV2/{tag}] fold {fold_id + 1}/{len(folds)} 완료 "
              f"(roc_auc={rows[-1]['roc_auc']:.3f})")
    return pd.DataFrame(rows), attn_accum / len(folds)


def plot_attention(attn_mean, tag):
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(attn_mean, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(CLINIC_FEATURES)))
    ax.set_yticklabels(CLINIC_FEATURES)
    ax.set_xlabel("aec index (1=치골 pubis, 128=간 상부 liver upper)")
    ax.set_title(f"Cross-Attention v2 가중치 ({tag}, clinic 토큰별 query, fold 평균) "
                  "- 빨간 음영은 pointwise FDR 유의구간")
    fig.colorbar(im, ax=ax, label="attention weight")

    for start, end in SIGNIFICANT_SEGMENTS:
        ax.axvspan(start - 1, end - 1, color="red", alpha=0.15)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"cross_attention_v2_weights_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def summarize(name, scores):
    print(f"[{name}] ROC-AUC={scores['roc_auc'].mean():.4f}+/-{scores['roc_auc'].std():.4f}  "
          f"PR-AUC={scores['pr_auc'].mean():.4f}+/-{scores['pr_auc'].std():.4f}  "
          f"(n_fold={len(scores)})")


# global_skip(06 재현, 이 20-fold 위에서 재확인) 대비 warmstart/prior/둘다의 개별+결합 기여도를 본다.
# pw_*는 동일한 조합을 patient-wise 정규화로 재실행한 짝(09의 결론이 warmstart/prior와 결합해도
# 유지되는지 확인하기 위함).
MODEL_VARIANTS = [
    {"tag": "global_skip", "aec_mode": "global", "warmup_epochs": 0, "ramp_epochs": 0, "lambda_prior": 0.0},
    {"tag": "warmstart", "aec_mode": "global", "warmup_epochs": WARMUP_EPOCHS, "ramp_epochs": RAMP_EPOCHS, "lambda_prior": 0.0},
    {"tag": "prior", "aec_mode": "global", "warmup_epochs": 0, "ramp_epochs": 0, "lambda_prior": LAMBDA_PRIOR},
    {"tag": "warmstart_prior", "aec_mode": "global", "warmup_epochs": WARMUP_EPOCHS, "ramp_epochs": RAMP_EPOCHS, "lambda_prior": LAMBDA_PRIOR},
    {"tag": "pw_skip", "aec_mode": "patient_wise", "warmup_epochs": 0, "ramp_epochs": 0, "lambda_prior": 0.0},
    {"tag": "pw_warmstart", "aec_mode": "patient_wise", "warmup_epochs": WARMUP_EPOCHS, "ramp_epochs": RAMP_EPOCHS, "lambda_prior": 0.0},
    {"tag": "pw_prior", "aec_mode": "patient_wise", "warmup_epochs": 0, "ramp_epochs": 0, "lambda_prior": LAMBDA_PRIOR},
    {"tag": "pw_warmstart_prior", "aec_mode": "patient_wise", "warmup_epochs": WARMUP_EPOCHS, "ramp_epochs": RAMP_EPOCHS, "lambda_prior": LAMBDA_PRIOR},
]

PAIRWISE_COMPARISONS = [
    ("warmstart", "global_skip"),
    ("prior", "global_skip"),
    ("warmstart_prior", "global_skip"),
    ("pw_skip", "global_skip"),  # 09의 patient_wise_skip vs global_skip을 이 아키텍처에서 재확인
    ("pw_warmstart", "pw_skip"),
    ("pw_prior", "pw_skip"),
    ("pw_warmstart_prior", "pw_skip"),
]


def main():
    df, aec_cols = load_data()
    y = df["Output"].to_numpy()
    prior_dist = build_prior_distribution(SEQ_LEN, SIGNIFICANT_SEGMENTS, PRIOR_BOOST)

    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    folds = list(rkf.split(df, y))
    print(f"=== Cross-Attention v2 비교: {N_SPLITS}-fold x {N_REPEATS} repeat "
          f"= {len(folds)} folds (n={len(df)}, device={DEVICE}) ===\n")

    clinic_scores = run_logreg_variant(df, y, folds, CLINIC_FEATURES)
    print("clinic-only(logreg, 20-fold) 완료\n")

    ca_results = {}
    for variant in MODEL_VARIANTS:
        ca_scores, attn_mean = run_cross_attention(
            df, y, folds, aec_cols, aec_mode=variant["aec_mode"],
            warmup_epochs=variant["warmup_epochs"], ramp_epochs=variant["ramp_epochs"],
            lambda_prior=variant["lambda_prior"], prior_dist=prior_dist, tag=variant["tag"])
        ca_results[variant["tag"]] = ca_scores
        plot_attention(attn_mean, variant["tag"])
        print()

    summarize("clinic-only (logreg)", clinic_scores)
    for variant in MODEL_VARIANTS:
        summarize(f"cross-attention ({variant['tag']})", ca_results[variant["tag"]])

    print()
    for variant in MODEL_VARIANTS:
        tag = variant["tag"]
        for metric in ["roc_auc", "pr_auc"]:
            diff = ca_results[tag][metric].to_numpy() - clinic_scores[metric].to_numpy()
            _, p = stats.wilcoxon(diff)
            print(f"[{metric}] cross-attention({tag}) - clinic-only 차이: "
                  f"mean={diff.mean():+.4f}, Wilcoxon p={p:.4g}")

    print()
    for tag_a, tag_b in PAIRWISE_COMPARISONS:
        for metric in ["roc_auc", "pr_auc"]:
            diff = ca_results[tag_a][metric].to_numpy() - ca_results[tag_b][metric].to_numpy()
            _, p = stats.wilcoxon(diff)
            print(f"[{metric}] cross-attention({tag_a}) - cross-attention({tag_b}) 차이: "
                  f"mean={diff.mean():+.4f}, Wilcoxon p={p:.4g}")

    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        clinic_scores.to_excel(writer, sheet_name="CrossAttnV2_clinic_baseline", index=False)
        for variant in MODEL_VARIANTS:
            tag = variant["tag"]
            ca_results[tag].to_excel(writer, sheet_name=f"CrossAttnV2_{tag}", index=False)
    print(f"저장: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
