"""
06/07의 "global" aec 정규화(train fold 전체를 합친 단일 평균/표준편차로 스케일만 통일,
환자 개인의 절대 mA 레벨 차이는 그대로 보존)를 "patient-wise" 정규화(각 환자를 자기 자신의
128개 값의 평균/표준편차로 정규화 - 환자 개인의 레벨을 제거하고 곡선 "모양"만 남김)와
직접 비교한다.

동기: 0702의 05번 결론은 "mean_mA(곡선 레벨)가 Weight와 상관 0.54라서 clinic 변수와
이미 중복된다"였다. global 정규화는 이 레벨 정보를 그대로 유지한 채 모델에 넣지만,
patient-wise 정규화는 레벨을 명시적으로 제거하므로 "clinic이 이미 아는 체격 신호"와
겹치지 않는 순수한 형태(shape) 정보만 남는다. 0701의 Derivative FPCA가 찾아낸
"레벨로 설명 안 되는 국소적 기울기 패턴"을 예측 모델에 반영하는 첫 시도.

06은 N_REPEATS=1(5-fold)이라 통계 검정력이 부족했다는 게 07에서 확인된 교훈이므로,
이 스크립트는 처음부터 5-fold x 4 repeat = 20-fold로 실행한다.

비교 대상 (모두 clinic_skip 유무를 교차):
  - global            : 기존 06/07의 population 단일 스칼라 정규화, skip 없음
  - global_skip        : 위 + clinic-only와 동등한 선형항을 skip으로 추가 (07의 최선 재현)
  - patient_wise       : 환자별 자기 자신의 평균/표준편차로 정규화, skip 없음
  - patient_wise_skip  : 위 + skip

출력: 콘솔 비교표, figures/cross_attention_patient_wise_weights_{tag}.png,
      excel/model_comparison_results.xlsx에 CrossAttnPW_* 시트 추가
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
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures", "09")
os.makedirs(FIGURES_DIR, exist_ok=True)

MERGED_FILE = os.path.join(EXCEL_DIR, "강남_liver_merged_features.xlsx")
LABEL_FILE = os.path.join(EXCEL_DIR, "강남_aec_128.xlsx")
RESULTS_FILE = os.path.join(EXCEL_DIR, "model_comparison_results.xlsx")

CLINIC_FEATURES = ["PatientAge", "Sex_M", "Height", "Weight"]
SIGNIFICANT_SEGMENTS = [(62, 71), (103, 115), (120, 124)]  # pointwise FDR 유의구간(1-based, inclusive)

SEQ_LEN = 128
N_SPLITS = 5
N_REPEATS = 4  # 07의 교훈: 06처럼 5-fold만 쓰면 검정력 부족 -> 처음부터 20-fold
RANDOM_STATE = 42
D_MODEL = 16
N_HEADS = 2
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

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


class CrossAttentionNet(nn.Module):
    """06과 동일한 구조. clinic_skip=True면 clinic-only 로지스틱과 동등한 선형항을
    최종 로짓에 residual로 더해준다(DAFT류 tabular-skip)."""

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
        logit = self.head(pooled).squeeze(-1)
        if self.clinic_skip:
            logit = logit + self.clinic_skip_linear(clinic).squeeze(-1)
        return logit, attn_weights  # attn_weights: (B, n_clinic, seq_len)


def train_one_fold(clinic_tr, aec_tr, y_tr, clinic_te, aec_te, n_clinic, clinic_skip=False):
    torch.manual_seed(RANDOM_STATE)
    model = CrossAttentionNet(n_clinic=n_clinic, seq_len=SEQ_LEN, clinic_skip=clinic_skip).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                               dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
            logit, _ = model(clinic_tr_t[idx], aec_tr_t[idx])
            loss = loss_fn(logit, y_tr_t[idx])
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
    """각 환자(row)를 자기 자신의 128개 값의 평균/표준편차로 정규화.
    fold의 train/test 구분과 무관하게 환자 자신의 값만 쓰므로 leakage가 없다."""
    mu = aec.mean(axis=1, keepdims=True)
    sigma = aec.std(axis=1, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (aec - mu) / sigma


def scale_aec(aec_tr, aec_te, mode):
    """'global'은 train fold 전체를 합친 단일 평균/표준편차(레벨 보존, 스케일만 통일).
    'patient_wise'는 각 환자 자신의 평균/표준편차(레벨 제거, 모양만 남김)."""
    if mode == "global":
        mu = aec_tr.mean()
        sigma = aec_tr.std()
        return (aec_tr - mu) / sigma, (aec_te - mu) / sigma
    if mode == "patient_wise":
        return patient_wise_normalize(aec_tr), patient_wise_normalize(aec_te)
    raise ValueError(f"unknown aec_mode: {mode}")


def run_cross_attention(df, y, folds, aec_cols, aec_mode, clinic_skip, tag):
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
                                        n_clinic=len(CLINIC_FEATURES), clinic_skip=clinic_skip)
        rows.append({"fold": fold_id, "roc_auc": roc_auc_score(y[te], proba),
                      "pr_auc": average_precision_score(y[te], proba)})
        attn_accum += attn_w.mean(axis=0)
        print(f"  [PatientWise/{tag}] fold {fold_id + 1}/{len(folds)} 완료 "
              f"(roc_auc={rows[-1]['roc_auc']:.3f})")
    return pd.DataFrame(rows), attn_accum / len(folds)


def plot_attention(attn_mean, tag):
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(attn_mean, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(CLINIC_FEATURES)))
    ax.set_yticklabels(CLINIC_FEATURES)
    ax.set_xlabel("aec index (1=치골 pubis, 128=간 상부 liver upper)")
    ax.set_title(f"Cross-Attention 가중치 ({tag}, clinic 토큰별 query, fold 평균) "
                  "- 빨간 음영은 pointwise FDR 유의구간")
    fig.colorbar(im, ax=ax, label="attention weight")

    for start, end in SIGNIFICANT_SEGMENTS:
        ax.axvspan(start - 1, end - 1, color="red", alpha=0.15)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"cross_attention_patient_wise_weights_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def summarize(name, scores):
    print(f"[{name}] ROC-AUC={scores['roc_auc'].mean():.4f}+/-{scores['roc_auc'].std():.4f}  "
          f"PR-AUC={scores['pr_auc'].mean():.4f}+/-{scores['pr_auc'].std():.4f}  "
          f"(n_fold={len(scores)})")


MODEL_VARIANTS = [
    {"tag": "global", "aec_mode": "global", "clinic_skip": False},
    {"tag": "global_skip", "aec_mode": "global", "clinic_skip": True},
    {"tag": "patient_wise", "aec_mode": "patient_wise", "clinic_skip": False},
    {"tag": "patient_wise_skip", "aec_mode": "patient_wise", "clinic_skip": True},
]

# patient_wise를 같은 skip 유무 조건의 global과 짝지어 "정규화 방식만 바꾼" 단독 기여도를 본다.
PAIRWISE_COMPARISONS = [
    ("patient_wise", "global"),
    ("patient_wise_skip", "global_skip"),
    ("patient_wise_skip", "patient_wise"),
]


def main():
    df, aec_cols = load_data()
    y = df["Output"].to_numpy()

    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    folds = list(rkf.split(df, y))
    print(f"=== Patient-wise vs Global 정규화 비교: {N_SPLITS}-fold x {N_REPEATS} repeat "
          f"= {len(folds)} folds (n={len(df)}, device={DEVICE}) ===\n")

    clinic_scores = run_logreg_variant(df, y, folds, CLINIC_FEATURES)
    print("clinic-only(logreg, 20-fold) 완료\n")

    ca_results = {}
    for variant in MODEL_VARIANTS:
        ca_scores, attn_mean = run_cross_attention(
            df, y, folds, aec_cols, variant["aec_mode"], variant["clinic_skip"], tag=variant["tag"])
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
        clinic_scores.to_excel(writer, sheet_name="CrossAttnPW_clinic_baseline", index=False)
        for variant in MODEL_VARIANTS:
            tag = variant["tag"]
            ca_results[tag].to_excel(writer, sheet_name=f"CrossAttnPW_{tag}", index=False)
    print(f"저장: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
