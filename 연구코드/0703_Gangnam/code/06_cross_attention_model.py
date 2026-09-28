"""
Cross-Attention 기반 신경망으로 clinic + aec 원곡선을 결합해서 Output(저SMI) 분류.

1~5번 모델은 사람이 미리 골라낸 스칼라 피처(mean_mA, 유의구간 slope 3개)를 썼지만,
이 모델은 aec_1~aec_128 원곡선 전체를 입력으로 주고, "clinic 4개를 query로,
곡선 전체(128개 위치)를 key/value로 삼는 cross-attention"이 학습을 통해 스스로
어느 위치에 주목할지 찾도록 한다.
-> pointwise FDR 검정으로 수작업으로 찾은 유의구간(48~55%, 80~90%, 94~97%, 빨간 음영)과
   모델이 학습으로 찾은 attention 가중치가 겹치는지 figures/cross_attention_weights_*.png에서 비교.

구조:
  clinic 4개 스칼라 -> Linear(1,d_model) 값 임베딩 + feature-id 임베딩 (query, 4 토큰)
  aec 128개 스칼라  -> Linear(1,d_model) 값 임베딩 + position 임베딩 (key/value, 128 토큰)
  nn.MultiheadAttention(d_model, n_heads) -> residual + LayerNorm -> mean pool -> MLP head

학습: BCEWithLogitsLoss(pos_weight) + Adam. clinic은 train fold 기준 StandardScaler(위치별).
      aec는 두 가지 전처리로 비교: "raw"(원본 mA 그대로) vs "global"(train fold의 128개 위치를
      전부 합친 단일 평균/표준편차로 정규화 - 위치 간 상대적 곡선 모양은 보존, 스케일만 통일).
      추가로 "global_skip": clinic-only 로지스틱과 동등한 선형항을 최종 로짓에 residual로
      더해주는 DAFT류 skip connection(참고: Polsterl et al., MICCAI 2021, arXiv:2107.05990)
      -> attention은 aec가 주는 추가 정보만 책임지면 되도록 부담을 던다.
      이 위에 두 가지를 각각 추가로 얹어 개별 기여도를 비교:
      - "global_skip_smooth": aec를 02_aec_fda_pipeline.py와 같은 B-spline 기저(20개)로
        환자별 개별 스무딩한 뒤 같은 128 그리드에 재평가해서 사용(포지션 정렬은 유지, 고주파 노이즈만 감소).
        스무딩은 환자 각자의 곡선만으로 결정되므로(그룹/fold 정보 미사용) fold와 무관하게 1회만 계산.
      - "global_skip_earlystop": 고정 60 epoch 대신, train fold를 다시 85/15로 쪼개 validation
        ROC-AUC가 patience(10 epoch) 동안 개선 안 되면 멈추고 best epoch 가중치로 복원.
평가: 5-fold x 4 repeat = 20 fold (신경망 학습비용 때문에 로지스틱 20-repeat보다 적게 잡음).
      같은 20-fold 위에서 clinic-only 로지스틱회귀도 다시 계산해서 Wilcoxon paired test로 비교.
      raw vs global, global vs global_skip, (global_skip 대비) smooth/earlystop 각각 단독 기여도를
      같은 fold로 Wilcoxon paired test.

출력: 콘솔 비교표, figures/cross_attention_weights_{tag}.png,
      excel/model_comparison_results.xlsx에 CrossAttn_* 시트 추가
"""

import copy
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
from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_DIR = os.path.join(BASE_DIR, "..", "excel")
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures", "06")
os.makedirs(FIGURES_DIR, exist_ok=True)

MERGED_FILE = os.path.join(EXCEL_DIR, "강남_liver_merged_features.xlsx")
LABEL_FILE = os.path.join(EXCEL_DIR, "강남_aec_128.xlsx")
RESULTS_FILE = os.path.join(EXCEL_DIR, "model_comparison_results.xlsx")

CLINIC_FEATURES = ["PatientAge", "Sex_M", "Height", "Weight"]
SIGNIFICANT_SEGMENTS = [(62, 71), (103, 115), (120, 124)]  # pointwise FDR 검정에서 찾은 구간(1-based)

SEQ_LEN = 128
N_SPLITS = 5
N_REPEATS = 1
RANDOM_STATE = 42
D_MODEL = 16
N_HEADS = 2
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

N_BASIS_SMOOTH = 20  # 02_aec_fda_pipeline.py와 동일한 B-spline 기저 개수

EARLYSTOP_VAL_FRAC = 0.15
EARLYSTOP_PATIENCE = 10
EARLYSTOP_MAX_EPOCHS = 200

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


def smooth_aec_cols(df, aec_cols, n_basis=N_BASIS_SMOOTH):
    """환자별 aec 곡선을 B-spline 기저로 개별 스무딩하고 같은 128 그리드에 재평가.
    각 곡선은 자기 자신의 값만으로 스무딩되므로(그룹/fold 라벨 미사용) 전체 데이터에 한 번만 적용해도
    fold 간 정보 누출이 없다. 위치(포지션) 정렬은 그대로 유지되어 attention 해석이 raw와 호환된다."""
    grid = np.arange(1, len(aec_cols) + 1)
    fd = FDataGrid(data_matrix=df[aec_cols].to_numpy(float), grid_points=grid)
    basis = BSplineBasis(n_basis=n_basis, domain_range=fd.domain_range)
    smoothed = fd.to_basis(basis).to_grid(grid).data_matrix[:, :, 0]
    smooth_cols = [f"{c}_smooth" for c in aec_cols]
    smooth_df = pd.DataFrame(smoothed, columns=smooth_cols, index=df.index)
    return pd.concat([df, smooth_df], axis=1), smooth_cols


class CrossAttentionNet(nn.Module):
    """clinic_skip=True면 clinic-only 로지스틱회귀와 동등한 선형항을 최종 로짓에 residual로
    더해준다(DAFT류 tabular-skip 아이디어). attention은 그 위에 aec가 주는 추가 정보만 책임지면 됨."""

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
        # clinic: (B, n_clinic), aec: (B, seq_len)
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


def train_one_fold_earlystop(clinic_tr, aec_tr, y_tr, clinic_te, aec_te, n_clinic, clinic_skip=False):
    """고정 EPOCHS 대신 train fold를 다시 나눠 validation ROC-AUC 기준 early stopping."""
    torch.manual_seed(RANDOM_STATE)
    model = CrossAttentionNet(n_clinic=n_clinic, seq_len=SEQ_LEN, clinic_skip=clinic_skip).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    idx_all = np.arange(len(y_tr))
    idx_sub, idx_val = train_test_split(idx_all, test_size=EARLYSTOP_VAL_FRAC, stratify=y_tr,
                                          random_state=RANDOM_STATE)
    clinic_sub, clinic_val = clinic_tr[idx_sub], clinic_tr[idx_val]
    aec_sub, aec_val = aec_tr[idx_sub], aec_tr[idx_val]
    y_sub, y_val = y_tr[idx_sub], y_tr[idx_val]

    pos_weight = torch.tensor([(y_sub == 0).sum() / max((y_sub == 1).sum(), 1)],
                               dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    clinic_sub_t = torch.tensor(clinic_sub, dtype=torch.float32, device=DEVICE)
    aec_sub_t = torch.tensor(aec_sub, dtype=torch.float32, device=DEVICE)
    y_sub_t = torch.tensor(y_sub, dtype=torch.float32, device=DEVICE)
    clinic_val_t = torch.tensor(clinic_val, dtype=torch.float32, device=DEVICE)
    aec_val_t = torch.tensor(aec_val, dtype=torch.float32, device=DEVICE)

    n = len(y_sub)
    best_val_auc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    n_epochs_used = EARLYSTOP_MAX_EPOCHS
    for epoch in range(EARLYSTOP_MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            logit, _ = model(clinic_sub_t[idx], aec_sub_t[idx])
            loss = loss_fn(logit, y_sub_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logit, _ = model(clinic_val_t, aec_val_t)
            val_proba = torch.sigmoid(val_logit).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_proba)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= EARLYSTOP_PATIENCE:
            n_epochs_used = epoch + 1
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        clinic_te_t = torch.tensor(clinic_te, dtype=torch.float32, device=DEVICE)
        aec_te_t = torch.tensor(aec_te, dtype=torch.float32, device=DEVICE)
        logit, attn_w = model(clinic_te_t, aec_te_t)
        proba = torch.sigmoid(logit).cpu().numpy()
        attn_w = attn_w.cpu().numpy()
    return proba, attn_w, n_epochs_used


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


def scale_aec(aec_tr, aec_te, mode):
    """train fold 기준으로 aec를 전처리. 'raw'는 원곡선 그대로,
    'global'은 128개 위치를 다 합친 단일 평균/표준편차로 정규화(곡선 모양은 보존, 스케일만 통일)."""
    if mode == "raw":
        return aec_tr.copy(), aec_te.copy()
    if mode == "global":
        mu = aec_tr.mean()
        sigma = aec_tr.std()
        return (aec_tr - mu) / sigma, (aec_te - mu) / sigma
    raise ValueError(f"unknown aec_mode: {mode}")


def run_cross_attention(df, y, folds, aec_cols, aec_mode, clinic_skip=False, early_stop=False, tag=""):
    rows = []
    attn_accum = np.zeros((len(CLINIC_FEATURES), SEQ_LEN))
    for fold_id, (tr, te) in enumerate(folds):
        clinic_scaler = StandardScaler().fit(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_tr = clinic_scaler.transform(df.iloc[tr][CLINIC_FEATURES].to_numpy(float))
        clinic_te = clinic_scaler.transform(df.iloc[te][CLINIC_FEATURES].to_numpy(float))

        aec_tr_raw = df.iloc[tr][aec_cols].to_numpy(float)
        aec_te_raw = df.iloc[te][aec_cols].to_numpy(float)
        aec_tr, aec_te = scale_aec(aec_tr_raw, aec_te_raw, aec_mode)

        if early_stop:
            proba, attn_w, n_epochs = train_one_fold_earlystop(
                clinic_tr, aec_tr, y[tr], clinic_te, aec_te,
                n_clinic=len(CLINIC_FEATURES), clinic_skip=clinic_skip)
            epoch_info = f", stopped@{n_epochs}ep"
        else:
            proba, attn_w = train_one_fold(clinic_tr, aec_tr, y[tr], clinic_te, aec_te,
                                            n_clinic=len(CLINIC_FEATURES), clinic_skip=clinic_skip)
            epoch_info = ""
        rows.append({"fold": fold_id, "roc_auc": roc_auc_score(y[te], proba),
                      "pr_auc": average_precision_score(y[te], proba)})
        attn_accum += attn_w.mean(axis=0)
        print(f"  [CrossAttention/{tag}] fold {fold_id + 1}/{len(folds)} 완료 "
              f"(roc_auc={rows[-1]['roc_auc']:.3f}{epoch_info})")
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
    path = os.path.join(FIGURES_DIR, f"cross_attention_weights_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def summarize(name, scores):
    print(f"[{name}] ROC-AUC={scores['roc_auc'].mean():.4f}+/-{scores['roc_auc'].std():.4f}  "
          f"PR-AUC={scores['pr_auc'].mean():.4f}+/-{scores['pr_auc'].std():.4f}  "
          f"(n_fold={len(scores)})")


# tag: 콘솔 라벨/파일명/시트명에 쓰는 짧은 키(엑셀 시트명 31자 제한 때문에 "CA_scores_" 접두어 사용).
# aec_source: "raw"(aec_cols 그대로) 또는 "smooth"(B-spline 스무딩 버전, smooth_aec_cols가 채워줌).
# global_skip이 지금까지의 최선(clinic-only와 거의 근접, 이전 대화 결과) -> 그 위에 smooth/early-stop을
# 각각 하나씩만 더 얹어서 개별 기여도를 독립적으로 비교(둘을 동시에 넣지 않음).
MODEL_VARIANTS = [
    {"tag": "raw", "aec_source": "raw", "aec_mode": "raw", "clinic_skip": False, "early_stop": False},
    {"tag": "global", "aec_source": "raw", "aec_mode": "global", "clinic_skip": False, "early_stop": False},
    {"tag": "global_skip", "aec_source": "raw", "aec_mode": "global", "clinic_skip": True, "early_stop": False},
    {"tag": "global_skip_smooth", "aec_source": "smooth", "aec_mode": "global", "clinic_skip": True, "early_stop": False},
    {"tag": "global_skip_estop", "aec_source": "raw", "aec_mode": "global", "clinic_skip": True, "early_stop": True},
]

# (baseline과 비교할 대상, 무엇과 비교하는지): 각 아이디어의 "global_skip 대비 단독 기여도"를 보기 위함.
PAIRWISE_COMPARISONS = [
    ("global", "raw"),
    ("global_skip", "global"),
    ("global_skip_smooth", "global_skip"),
    ("global_skip_estop", "global_skip"),
]


def main():
    df, aec_cols = load_data()
    y = df["Output"].to_numpy()
    df, aec_smooth_cols = smooth_aec_cols(df, aec_cols)

    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    folds = list(rkf.split(df, y))
    print(f"=== Cross-Attention 비교: {N_SPLITS}-fold x {N_REPEATS} repeat (n={len(df)}, "
          f"device={DEVICE}) ===\n")

    clinic_scores = run_logreg_variant(df, y, folds, CLINIC_FEATURES)
    print("clinic-only(logreg, 이 fold 세트로 재계산) 완료\n")

    ca_results = {}
    for variant in MODEL_VARIANTS:
        cols = aec_smooth_cols if variant["aec_source"] == "smooth" else aec_cols
        ca_scores, attn_mean = run_cross_attention(
            df, y, folds, cols, variant["aec_mode"],
            clinic_skip=variant["clinic_skip"], early_stop=variant["early_stop"], tag=variant["tag"])
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
        clinic_scores.to_excel(writer, sheet_name="CrossAttn_clinic_baseline", index=False)
        for variant in MODEL_VARIANTS:
            tag = variant["tag"]
            ca_results[tag].to_excel(writer, sheet_name=f"CrossAttn_{tag}", index=False)  # 시트명 31자 제한
    print(f"저장: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
