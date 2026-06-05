"""
PyTorch 모델 정의 및 학습·평가 유틸리티.

모델 계층:
  ResNet1D              — tabular 특징을 1D Conv로 처리 (M1 베이스라인)
  AECOnlyNet            — AEC 시퀀스만 입력하는 분류 모델 (M4)
  ClinAECCrossAttn      — Clinic + AEC Bidirectional Cross-Attention (M2)
  ClinAECScanCrossAttn  — Clinic + Scanner(MFR Embedding) + AEC Cross-Attention (M3)

서브모듈:
  ResBlock1D            — Conv1d×2 + BN + ReLU 잔차 블록
  ResNet1DEncoder       — AEC 시퀀스 → (B, n_tokens, d_model) 인코더
  ScalarFeatureTokenizer — 각 scalar feature → d_model 독립 토큰
  CrossAttentionBlock   — Pre-norm Cross-Attention + FFN + Dropout
  MfrTokenizer          — ManufacturerModelName 정수 → Embedding 토큰

손실함수: 모든 모델에 FocalLossWithLogits(gamma=FOCAL_GAMMA, pos_weight) 사용.
build_* 함수는 모델·손실함수·옵티마이저·스케줄러를 묶어 반환한다.
"""
import numpy as np
from typing import cast, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from config import DEVICE, LR_RATE, EPOCHS, HIDDEN, N_BLOCKS, N_HEADS, BATCH_SIZE, FOCAL_GAMMA, GRAD_CLIP, N_CA_LAYERS


class FocalLossWithLogits(nn.Module):
    """
    Binary Focal Loss (logit 입력).

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    pos_weight: BCEWithLogitsLoss와 동일하게 양성 클래스 가중치 적용 (클래스 불균형 보정).
    gamma=0이면 가중 BCE와 동일.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=cast(Optional[torch.Tensor], self.pos_weight), reduction="none"
        )
        p_t = torch.exp(-bce)
        return ((1 - p_t) ** self.gamma * bce).mean()


class TabularDataset(Dataset):
    """단일 입력(X, y) 테이블 데이터를 float32/float32 텐서로 래핑하는 Dataset."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResBlock1D(nn.Module):
    """Conv1d(3) × 2 + BatchNorm + ReLU 잔차 블록. 입출력 채널 수 동일(ch)."""

    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


class ResNet1D(nn.Module):
    """
    1D 테이블 피처를 채널 차원으로 처리하는 ResNet 분류 모델.

    stem → N_BLOCKS개 ResBlock → AdaptiveAvgPool → FC head → 스칼라 logit 출력.
    """

    def __init__(self, hidden=HIDDEN, n_blocks=N_BLOCKS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock1D(hidden) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x.unsqueeze(1)))).squeeze(-1)


def build_resnet(y_tr_arr):
    """ResNet1D 모델·FocalLossWithLogits(pos_weight)·AdamW·CosineAnnealingLR을 생성해 반환."""
    # sarcopenia(양성) 비율이 낮으므로 pos_weight = n_negative/n_positive 로 클래스 불균형 보정
    pos_w = torch.tensor(
        [(y_tr_arr == 0).sum() / y_tr_arr.sum()], dtype=torch.float32
    ).to(DEVICE)
    model = ResNet1D().to(DEVICE)
    crit  = FocalLossWithLogits(gamma=FOCAL_GAMMA, pos_weight=pos_w)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, crit, opt, sched


def make_loaders(X_tr, y_tr, X_val, y_val):
    """TabularDataset으로 train DataLoader(shuffle=True)와 val DataLoader를 생성해 반환."""
    tr_dl  = DataLoader(TabularDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TabularDataset(X_val, y_val), batch_size=BATCH_SIZE)
    return tr_dl, val_dl


def train_one_epoch(model, loader, crit, opt):
    """ResNet1D 모델을 한 epoch 학습하고 샘플 평균 loss를 반환."""
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        opt.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_loader(model, loader, crit):
    """ResNet1D 모델을 평가 모드로 실행해 (avg_loss, probs, trues)를 반환."""
    model.eval()
    total, probs, trues = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        total += crit(logits, yb).item() * len(yb)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        trues.extend(yb.cpu().numpy())
    return total / len(loader.dataset), np.array(probs), np.array(trues)


# ── AEC Only Model (M4) ──────────────────────────────────────

class AECOnlyNet(nn.Module):
    """
    AEC 시퀀스만을 입력으로 하는 ResNet1D 기반 분류 모델 (Model 4).

    임상 특징 없이 AEC 128 포인트에서 직접 sarcopenia를 분류한다.
    """

    def __init__(self, hidden=HIDDEN, n_blocks=N_BLOCKS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock1D(hidden) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x.unsqueeze(1)))).squeeze(-1)


def build_aec_only(y_tr_arr):
    """AECOnlyNet 모델·FocalLossWithLogits(pos_weight)·AdamW·CosineAnnealingLR을 생성해 반환."""
    pos_w = torch.tensor(
        [(y_tr_arr == 0).sum() / y_tr_arr.sum()], dtype=torch.float32
    ).to(DEVICE)
    model = AECOnlyNet().to(DEVICE)
    crit  = FocalLossWithLogits(gamma=FOCAL_GAMMA, pos_weight=pos_w)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, crit, opt, sched


# ── Clinic + AEC Cross-Attention Model ───────────────────────

class ResNet1DEncoder(nn.Module):
    """
    AEC 시퀀스 (B, n_aec) → ResNet1D 인코딩 → (B, n_tokens, d_model) 토큰 반환.
    ScalarFeatureTokenizer 대신 사용하면 AEC의 1D 국소 패턴을 학습할 수 있다.
    """
    def __init__(self, d_model: int = HIDDEN, n_blocks: int = N_BLOCKS, n_tokens: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock1D(d_model) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n_aec) → (B, d_model, n_tokens) → (B, n_tokens, d_model)
        return self.pool(self.blocks(self.stem(x.unsqueeze(1)))).transpose(1, 2)


class DualDataset(Dataset):
    """Clinic(X_clin)·AEC(X_aec)·레이블(y) 쌍을 float32 텐서로 래핑하는 Dataset."""

    def __init__(self, X_clin, X_aec, y):
        self.X_clin = torch.tensor(X_clin, dtype=torch.float32)
        self.X_aec  = torch.tensor(X_aec,  dtype=torch.float32)
        self.y      = torch.tensor(y,       dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_clin[idx], self.X_aec[idx], self.y[idx]


def make_dual_loaders(X_clin_tr, X_aec_tr, y_tr, X_clin_val, X_aec_val, y_val):
    """DualDataset으로 train/val DataLoader를 생성해 반환."""
    tr_dl  = DataLoader(DualDataset(X_clin_tr, X_aec_tr, y_tr),  batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(DualDataset(X_clin_val, X_aec_val, y_val), batch_size=BATCH_SIZE)
    return tr_dl, val_dl


class ScalarFeatureTokenizer(nn.Module):
    """
    각 scalar feature를 독립 Linear로 d_model 차원 토큰으로 변환.
    입력: (B, F)  →  출력: (B, F, d_model)
    """
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        self.feature_embedding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat(
            [self.projections[i](x[:, i:i+1]).unsqueeze(1) for i in range(len(self.projections))],
            dim=1,
        )                                   # (B, F, d_model)
        return tokens + self.feature_embedding


class CrossAttentionBlock(nn.Module):
    """
    query_tokens가 context_tokens를 참고하는 Cross-Attention 블록.
    Pre-norm + FFN + dropout 포함.
    """
    def __init__(self, d_model: int, num_heads: int,
                 ff_hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens: torch.Tensor,
                context_tokens: torch.Tensor,
                return_attention: bool = False):
        attn_out, attn_weights = self.cross_attn(
            query=query_tokens, key=context_tokens, value=context_tokens,
            need_weights=True, average_attn_weights=True,
        )
        x = self.norm1(query_tokens + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        if return_attention:
            return x, attn_weights
        return x


class ClinAECCrossAttn(nn.Module):
    """
    Clinic (Age, Sex, BMI) + AEC (256 scalar)를
    Bidirectional Cross-Attention으로 결합하는 분류 모델.

    방향 1: clinical → Query,  AEC → Key/Value
    방향 2: AEC      → Query,  clinical → Key/Value

    aec_encoder:
      'scalar' — ScalarFeatureTokenizer (각 AEC 값을 독립 토큰)
      'resnet'  — ResNet1DEncoder (AEC 시퀀스를 1D Conv로 인코딩, 국소 패턴 학습)
    """
    def __init__(
        self,
        num_clinical_features: int = 3,
        num_aec_features: int = 256,
        d_model: int = HIDDEN,
        num_heads: int = N_HEADS,
        ff_hidden_dim: int = 128,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.1,
        aec_encoder: str = 'resnet',
        n_aec_tokens: int = 32,
    ):
        super().__init__()
        self.clinical_tokenizer = ScalarFeatureTokenizer(num_clinical_features, d_model)

        if aec_encoder == 'resnet':
            self.aec_encoder = ResNet1DEncoder(d_model=d_model, n_tokens=n_aec_tokens)
        else:
            self.aec_encoder = ScalarFeatureTokenizer(num_aec_features, d_model)

        self.clin_to_aec_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(N_CA_LAYERS)
        ])
        self.aec_to_clin_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(N_CA_LAYERS)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, classifier_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(self, clinical_x: torch.Tensor, aec_x: torch.Tensor,
                return_attention: bool = False):
        c_tokens = self.clinical_tokenizer(clinical_x)
        a_tokens = self.aec_encoder(aec_x)

        attn_c2a = attn_a2c = None
        for i, (c2a, a2c) in enumerate(zip(self.clin_to_aec_layers, self.aec_to_clin_layers)):
            last = (i == len(self.clin_to_aec_layers) - 1)
            if return_attention and last:
                c_tokens, attn_c2a = c2a(c_tokens, a_tokens, return_attention=True)
                a_tokens, attn_a2c = a2c(a_tokens, c_tokens, return_attention=True)
            else:
                c_tokens = c2a(c_tokens, a_tokens)
                a_tokens = a2c(a_tokens, c_tokens)

        fused = torch.cat([c_tokens.mean(1), a_tokens.mean(1)], dim=-1)  # (B, d_model*2)
        logits = self.classifier(fused).squeeze(-1)

        if return_attention:
            return logits, {"clinical_to_aec": attn_c2a, "aec_to_clinical": attn_a2c}
        return logits


def build_cross_attn(y_tr_arr, num_clinical_features=3, num_aec_features=256,
                     aec_encoder='resnet'):
    """ClinAECCrossAttn 모델·FocalLossWithLogits(pos_weight)·AdamW·CosineAnnealingLR을 생성해 반환."""
    pos_w = torch.tensor(
        [(y_tr_arr == 0).sum() / y_tr_arr.sum()], dtype=torch.float32
    ).to(DEVICE)
    model = ClinAECCrossAttn(
        num_clinical_features=num_clinical_features,
        num_aec_features=num_aec_features,
        aec_encoder=aec_encoder,
    ).to(DEVICE)
    crit  = FocalLossWithLogits(gamma=FOCAL_GAMMA, pos_weight=pos_w)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, crit, opt, sched


# ── Clinic + Scanner + AEC Model (Model 3) ───────────────────

class QuadDataset(Dataset):
    """Clinic·AEC·ManufacturerModelName·레이블을 텐서로 래핑하는 Dataset."""

    def __init__(self, X_clin, X_aec, X_scan_mfr, y):
        self.X_clin     = torch.tensor(X_clin,     dtype=torch.float32)
        self.X_aec      = torch.tensor(X_aec,      dtype=torch.float32)
        self.X_scan_mfr = torch.tensor(X_scan_mfr, dtype=torch.long)
        self.y          = torch.tensor(y,           dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_clin[idx], self.X_aec[idx], self.X_scan_mfr[idx], self.y[idx])


def make_quad_loaders(X_clin_tr, X_aec_tr, X_scan_mfr_tr, y_tr,
                      X_clin_val, X_aec_val, X_scan_mfr_val, y_val):
    """QuadDataset으로 train/val DataLoader를 생성해 반환."""
    tr_dl  = DataLoader(
        QuadDataset(X_clin_tr, X_aec_tr, X_scan_mfr_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_dl = DataLoader(
        QuadDataset(X_clin_val, X_aec_val, X_scan_mfr_val, y_val),
        batch_size=BATCH_SIZE,
    )
    return tr_dl, val_dl


class MfrTokenizer(nn.Module):
    """ManufacturerModelName (1-indexed 정수) → (B, 1, d_model) 토큰."""
    def __init__(self, n_manufacturers: int, d_model: int):
        super().__init__()
        self.mfr_emb = nn.Embedding(n_manufacturers + 1, d_model, padding_idx=0)
        self.feature_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, mfr: torch.Tensor) -> torch.Tensor:
        return self.mfr_emb(mfr).unsqueeze(1) + self.feature_embedding  # (B, 1, d_model)


class ClinAECScanCrossAttn(nn.Module):
    """
    Clinic (Age, Sex, BMI) + Scanner (ManufacturerModelName) + AEC 결합 모델.

    임상 데이터와 스캐너 파라미터를 별도 토크나이저로 처리한 뒤
    Bidirectional Cross-Attention으로 AEC와 결합한다.

      clinical_tokenizer : ScalarFeatureTokenizer(3)  → (B, 3, d_model)
      mfr_tokenizer      : MfrTokenizer(n_mfr)        → (B, 1, d_model)
      [clinical | mfr] concat                         → (B, 4, d_model)
      aec_encoder        : ResNet1DEncoder            → (B, n_aec_tokens, d_model)

    Cross-Attention (bidirectional):
      cs_to_aec : [clin|mfr] → Query,  AEC → Key/Value
      aec_to_cs : AEC        → Query,  [clin|mfr] → Key/Value
    """
    def __init__(
        self,
        n_manufacturers:       int,
        num_clinical_features: int   = 3,
        d_model:               int   = HIDDEN,
        num_heads:             int   = N_HEADS,
        ff_hidden_dim:         int   = 128,
        classifier_hidden_dim: int   = 128,
        dropout:               float = 0.1,
        n_aec_tokens:          int   = 32,
    ):
        super().__init__()
        self.clinical_tokenizer = ScalarFeatureTokenizer(num_clinical_features, d_model)
        self.mfr_tokenizer      = MfrTokenizer(n_manufacturers, d_model)
        self.aec_encoder        = ResNet1DEncoder(d_model=d_model, n_tokens=n_aec_tokens)

        self.cs_to_aec_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(N_CA_LAYERS)
        ])
        self.aec_to_cs_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(N_CA_LAYERS)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, classifier_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(self, clinical_x: torch.Tensor, aec_x: torch.Tensor,
                mfr_x: torch.Tensor, return_attention: bool = False):
        c_tokens  = self.clinical_tokenizer(clinical_x)    # (B, 3, d_model)
        s_tokens  = self.mfr_tokenizer(mfr_x)              # (B, 1, d_model)
        a_tokens  = self.aec_encoder(aec_x)                # (B, n_aec_tokens, d_model)
        cs_tokens = torch.cat([c_tokens, s_tokens], dim=1) # (B, 4, d_model)

        attn_cs2a = attn_a2cs = None
        for i, (cs2a, a2cs) in enumerate(zip(self.cs_to_aec_layers, self.aec_to_cs_layers)):
            last = (i == len(self.cs_to_aec_layers) - 1)
            if return_attention and last:
                cs_tokens, attn_cs2a = cs2a(cs_tokens, a_tokens, return_attention=True)
                a_tokens,  attn_a2cs = a2cs(a_tokens, cs_tokens, return_attention=True)
            else:
                cs_tokens = cs2a(cs_tokens, a_tokens)
                a_tokens  = a2cs(a_tokens, cs_tokens)

        fused  = torch.cat([cs_tokens.mean(1), a_tokens.mean(1)], dim=-1)  # (B, d_model*2)
        logits = self.classifier(fused).squeeze(-1)

        if return_attention:
            return logits, {"cs_to_aec": attn_cs2a, "aec_to_cs": attn_a2cs}
        return logits


def build_cross_attn3(y_tr_arr, n_manufacturers):
    """ClinAECScanCrossAttn 모델·FocalLossWithLogits(pos_weight)·AdamW·CosineAnnealingLR을 생성해 반환."""
    pos_w = torch.tensor(
        [(y_tr_arr == 0).sum() / y_tr_arr.sum()], dtype=torch.float32
    ).to(DEVICE)
    model = ClinAECScanCrossAttn(
        n_manufacturers=n_manufacturers,
    ).to(DEVICE)
    crit  = FocalLossWithLogits(gamma=FOCAL_GAMMA, pos_weight=pos_w)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, crit, opt, sched


def train_cross3_epoch(model, loader, crit, opt):
    """ClinAECScanCrossAttn 모델을 한 epoch 학습하고 샘플 평균 loss를 반환."""
    model.train()
    total = 0.0
    for xc, xa, xmfr, yb in loader:
        xc, xa, xmfr, yb = xc.to(DEVICE), xa.to(DEVICE), xmfr.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xc, xa, xmfr), yb)
        loss.backward()
        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        opt.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_cross3_loader(model, loader, crit):
    """ClinAECScanCrossAttn 모델을 평가 모드로 실행해 (avg_loss, probs, trues)를 반환."""
    model.eval()
    total, probs, trues = 0.0, [], []
    for xc, xa, xmfr, yb in loader:
        xc, xa, xmfr, yb = xc.to(DEVICE), xa.to(DEVICE), xmfr.to(DEVICE), yb.to(DEVICE)
        logits = model(xc, xa, xmfr)
        total += crit(logits, yb).item() * len(yb)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        trues.extend(yb.cpu().numpy())
    return total / len(loader.dataset), np.array(probs), np.array(trues)


def train_cross_epoch(model, loader, crit, opt):
    """ClinAECCrossAttn 모델을 한 epoch 학습하고 샘플 평균 loss를 반환."""
    model.train()
    total = 0.0
    for xc, xa, yb in loader:
        xc, xa, yb = xc.to(DEVICE), xa.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xc, xa), yb)
        loss.backward()
        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        opt.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_cross_loader(model, loader, crit):
    """ClinAECCrossAttn 모델을 평가 모드로 실행해 (avg_loss, probs, trues)를 반환."""
    model.eval()
    total, probs, trues = 0.0, [], []
    for xc, xa, yb in loader:
        xc, xa, yb = xc.to(DEVICE), xa.to(DEVICE), yb.to(DEVICE)
        logits = model(xc, xa)
        total += crit(logits, yb).item() * len(yb)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        trues.extend(yb.cpu().numpy())
    return total / len(loader.dataset), np.array(probs), np.array(trues)
