"""
0520 모델 정의.

0515 models.py의 모든 클래스/함수를 포함하며, 아래를 추가한다:
  SMIRegressionNet    — ResNet1D 기반 SMI 연속값 예측 (M6)
  build_regression    — SMIRegressionNet + HuberLoss + AdamW + CosineAnnealingLR
  RegressionDataset   — (X_clin, X_aec, y_smi) 쌍 Dataset
  train_regression_epoch — SMI 회귀 1 epoch 학습
  eval_regression_loader — SMI 회귀 평가 (loss, preds, trues)
  logit_collect_hook  — CrossAttn OOF logit 수집용 래퍼

0515 원본:
  ResBlock1D, ResNet1D, build_resnet, make_loaders,
  train_one_epoch, eval_loader,
  ResNet1DEncoder, DualDataset, make_dual_loaders,
  ScalarFeatureTokenizer, CrossAttentionBlock,
  ClinAECCrossAttn, build_cross_attn,
  QuadDataset, make_quad_loaders, MfrTokenizer,
  ClinAECScanCrossAttn, build_cross_attn3,
  train_cross_epoch, eval_cross_loader,
  train_cross3_epoch, eval_cross3_loader
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import DEVICE, LR_RATE, EPOCHS, HIDDEN, N_BLOCKS, N_HEADS, BATCH_SIZE


# ═══════════════════════════════════════════════════════════════
# 0515 원본 모델 (변경 없음)
# ═══════════════════════════════════════════════════════════════

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class ResBlock1D(nn.Module):
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
    pos_w = torch.tensor(
        [(y_tr_arr == 0).sum() / y_tr_arr.sum()], dtype=torch.float32
    ).to(DEVICE)
    model = ResNet1D().to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, crit, opt, sched


def make_loaders(X_tr, y_tr, X_val, y_val):
    tr_dl  = DataLoader(TabularDataset(X_tr, y_tr),  batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TabularDataset(X_val, y_val), batch_size=BATCH_SIZE)
    return tr_dl, val_dl


def train_one_epoch(model, loader, crit, opt):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward(); opt.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_loader(model, loader, crit):
    model.eval()
    total, probs, trues = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        total += crit(logits, yb).item() * len(yb)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        trues.extend(yb.cpu().numpy())
    return total / len(loader.dataset), np.array(probs), np.array(trues)


@torch.no_grad()
def eval_loader_logits(model, loader, crit):
    """eval_loader와 동일하나 sigmoid 전 raw logit도 반환 (캘리브레이션용)."""
    model.eval()
    total, logits_all, probs, trues = 0.0, [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logit = model(xb)
        total += crit(logit, yb).item() * len(yb)
        logits_all.extend(logit.cpu().numpy())
        probs.extend(torch.sigmoid(logit).cpu().numpy())
        trues.extend(yb.cpu().numpy())
    return (total / len(loader.dataset),
            np.array(logits_all), np.array(probs), np.array(trues))


class ResNet1DEncoder(nn.Module):
    def __init__(self, d_model=HIDDEN, n_blocks=N_BLOCKS, n_tokens=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock1D(d_model) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
    def forward(self, x):
        return self.pool(self.blocks(self.stem(x.unsqueeze(1)))).transpose(1, 2)


class DualDataset(Dataset):
    def __init__(self, X_clin, X_aec, y):
        self.X_clin = torch.tensor(X_clin, dtype=torch.float32)
        self.X_aec  = torch.tensor(X_aec,  dtype=torch.float32)
        self.y      = torch.tensor(y,       dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_clin[idx], self.X_aec[idx], self.y[idx]


def make_dual_loaders(X_clin_tr, X_aec_tr, y_tr, X_clin_val, X_aec_val, y_val):
    tr_dl  = DataLoader(DualDataset(X_clin_tr, X_aec_tr, y_tr),  batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(DualDataset(X_clin_val, X_aec_val, y_val), batch_size=BATCH_SIZE)
    return tr_dl, val_dl


class ScalarFeatureTokenizer(nn.Module):
    def __init__(self, num_features, d_model):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_features)])
        self.feature_embedding = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)
    def forward(self, x):
        tokens = torch.cat(
            [self.projections[i](x[:, i:i+1]).unsqueeze(1) for i in range(len(self.projections))],
            dim=1,
        )
        return tokens + self.feature_embedding


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim=128, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(ff_hidden_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, query_tokens, context_tokens, return_attention=False):
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
    """Clinic + AEC Bidirectional Cross-Attention (M2)."""
    def __init__(self, num_clinical_features=3, num_aec_features=256,
                 d_model=HIDDEN, num_heads=N_HEADS, ff_hidden_dim=128,
                 classifier_hidden_dim=128, dropout=0.1,
                 aec_encoder='resnet', n_aec_tokens=32):
        super().__init__()
        self.clinical_tokenizer = ScalarFeatureTokenizer(num_clinical_features, d_model)
        if aec_encoder == 'resnet':
            self.aec_encoder = ResNet1DEncoder(d_model=d_model, n_tokens=n_aec_tokens)
        else:
            self.aec_encoder = ScalarFeatureTokenizer(num_aec_features, d_model)
        self.clinical_to_aec = CrossAttentionBlock(d_model, num_heads, ff_hidden_dim, dropout)
        self.aec_to_clinical = CrossAttentionBlock(d_model, num_heads, ff_hidden_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, classifier_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(classifier_hidden_dim, 1),
        )
    def forward(self, clinical_x, aec_x, return_attention=False):
        c_tokens = self.clinical_tokenizer(clinical_x)
        a_tokens = self.aec_encoder(aec_x)
        if return_attention:
            c_fused, attn_c2a = self.clinical_to_aec(c_tokens, a_tokens, return_attention=True)
            a_fused, attn_a2c = self.aec_to_clinical(a_tokens, c_tokens, return_attention=True)
        else:
            c_fused = self.clinical_to_aec(c_tokens, a_tokens)
            a_fused = self.aec_to_clinical(a_tokens, c_tokens)
        fused  = torch.cat([c_fused.mean(1), a_fused.mean(1)], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        if return_attention:
            return logits, {"clinical_to_aec": attn_c2a, "aec_to_clinical": attn_a2c}
        return logits


def build_cross_attn(y_tr_arr, num_clinical_features=3, num_aec_features=256, aec_encoder='resnet'):
    pos_w = torch.tensor(
        [(y_tr_arr == 0).sum() / y_tr_arr.sum()], dtype=torch.float32
    ).to(DEVICE)
    model = ClinAECCrossAttn(
        num_clinical_features=num_clinical_features,
        num_aec_features=num_aec_features,
        aec_encoder=aec_encoder,
    ).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, crit, opt, sched


def train_cross_epoch(model, loader, crit, opt):
    model.train()
    total = 0.0
    for xc, xa, yb in loader:
        xc, xa, yb = xc.to(DEVICE), xa.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xc, xa), yb)
        loss.backward(); opt.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_cross_loader(model, loader, crit):
    model.eval()
    total, probs, trues = 0.0, [], []
    for xc, xa, yb in loader:
        xc, xa, yb = xc.to(DEVICE), xa.to(DEVICE), yb.to(DEVICE)
        logits = model(xc, xa)
        total += crit(logits, yb).item() * len(yb)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        trues.extend(yb.cpu().numpy())
    return total / len(loader.dataset), np.array(probs), np.array(trues)


@torch.no_grad()
def eval_cross_loader_logits(model, loader, crit):
    """캘리브레이션용: raw logit도 반환."""
    model.eval()
    total, logits_all, probs, trues = 0.0, [], [], []
    for xc, xa, yb in loader:
        xc, xa, yb = xc.to(DEVICE), xa.to(DEVICE), yb.to(DEVICE)
        logit = model(xc, xa)
        total += crit(logit, yb).item() * len(yb)
        logits_all.extend(logit.cpu().numpy())
        probs.extend(torch.sigmoid(logit).cpu().numpy())
        trues.extend(yb.cpu().numpy())
    return (total / len(loader.dataset),
            np.array(logits_all), np.array(probs), np.array(trues))


class QuadDataset(Dataset):
    def __init__(self, X_clin, X_aec, X_scan_mfr, y):
        self.X_clin     = torch.tensor(X_clin,     dtype=torch.float32)
        self.X_aec      = torch.tensor(X_aec,      dtype=torch.float32)
        self.X_scan_mfr = torch.tensor(X_scan_mfr, dtype=torch.long)
        self.y          = torch.tensor(y,           dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.X_clin[idx], self.X_aec[idx], self.X_scan_mfr[idx], self.y[idx]


def make_quad_loaders(X_clin_tr, X_aec_tr, X_scan_mfr_tr, y_tr,
                      X_clin_val, X_aec_val, X_scan_mfr_val, y_val):
    tr_dl  = DataLoader(QuadDataset(X_clin_tr, X_aec_tr, X_scan_mfr_tr, y_tr),
                        batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(QuadDataset(X_clin_val, X_aec_val, X_scan_mfr_val, y_val),
                        batch_size=BATCH_SIZE)
    return tr_dl, val_dl


class MfrTokenizer(nn.Module):
    def __init__(self, n_manufacturers, d_model):
        super().__init__()
        self.mfr_emb = nn.Embedding(n_manufacturers + 1, d_model, padding_idx=0)
        self.feature_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    def forward(self, mfr):
        return self.mfr_emb(mfr).unsqueeze(1) + self.feature_embedding


class ClinAECScanCrossAttn(nn.Module):
    """Clinic + Scanner(MFR Embedding) + AEC Cross-Attention (M3)."""
    def __init__(self, n_manufacturers, num_clinical_features=3,
                 d_model=HIDDEN, num_heads=N_HEADS, ff_hidden_dim=128,
                 classifier_hidden_dim=128, dropout=0.1, n_aec_tokens=32):
        super().__init__()
        self.clinical_tokenizer = ScalarFeatureTokenizer(num_clinical_features, d_model)
        self.mfr_tokenizer      = MfrTokenizer(n_manufacturers, d_model)
        self.aec_encoder        = ResNet1DEncoder(d_model=d_model, n_tokens=n_aec_tokens)
        self.cs_to_aec = CrossAttentionBlock(d_model, num_heads, ff_hidden_dim, dropout)
        self.aec_to_cs = CrossAttentionBlock(d_model, num_heads, ff_hidden_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, classifier_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(classifier_hidden_dim, 1),
        )
    def forward(self, clinical_x, aec_x, mfr_x, return_attention=False):
        c_tokens  = self.clinical_tokenizer(clinical_x)
        s_tokens  = self.mfr_tokenizer(mfr_x)
        a_tokens  = self.aec_encoder(aec_x)
        cs_tokens = torch.cat([c_tokens, s_tokens], dim=1)
        if return_attention:
            cs_fused, attn_cs2a = self.cs_to_aec(cs_tokens, a_tokens, return_attention=True)
            a_fused,  attn_a2cs = self.aec_to_cs(a_tokens, cs_tokens, return_attention=True)
        else:
            cs_fused = self.cs_to_aec(cs_tokens, a_tokens)
            a_fused  = self.aec_to_cs(a_tokens, cs_tokens)
        fused  = torch.cat([cs_fused.mean(1), a_fused.mean(1)], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        if return_attention:
            return logits, {"cs_to_aec": attn_cs2a, "aec_to_cs": attn_a2cs}
        return logits


def build_cross_attn3(y_tr_arr, n_manufacturers):
    pos_w = torch.tensor(
        [(y_tr_arr == 0).sum() / y_tr_arr.sum()], dtype=torch.float32
    ).to(DEVICE)
    model = ClinAECScanCrossAttn(n_manufacturers=n_manufacturers).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, crit, opt, sched


def train_cross3_epoch(model, loader, crit, opt):
    model.train()
    total = 0.0
    for xc, xa, xmfr, yb in loader:
        xc, xa, xmfr, yb = xc.to(DEVICE), xa.to(DEVICE), xmfr.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xc, xa, xmfr), yb)
        loss.backward(); opt.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_cross3_loader(model, loader, crit):
    model.eval()
    total, probs, trues = 0.0, [], []
    for xc, xa, xmfr, yb in loader:
        xc, xa, xmfr, yb = xc.to(DEVICE), xa.to(DEVICE), xmfr.to(DEVICE), yb.to(DEVICE)
        logits = model(xc, xa, xmfr)
        total += crit(logits, yb).item() * len(yb)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        trues.extend(yb.cpu().numpy())
    return total / len(loader.dataset), np.array(probs), np.array(trues)


# ═══════════════════════════════════════════════════════════════
# 0520 새 모델 — SMI 회귀 (M6)
# ═══════════════════════════════════════════════════════════════

class RegressionDataset(Dataset):
    """(X_clin, X_aec, y_smi) float32 텐서 Dataset."""
    def __init__(self, X_clin, X_aec, y_smi):
        self.X_clin = torch.tensor(X_clin, dtype=torch.float32)
        self.X_aec  = torch.tensor(X_aec,  dtype=torch.float32)
        self.y_smi  = torch.tensor(y_smi,  dtype=torch.float32)
    def __len__(self): return len(self.y_smi)
    def __getitem__(self, idx): return self.X_clin[idx], self.X_aec[idx], self.y_smi[idx]


def make_regression_loaders(X_clin_tr, X_aec_tr, y_smi_tr,
                             X_clin_val, X_aec_val, y_smi_val):
    tr_dl  = DataLoader(RegressionDataset(X_clin_tr, X_aec_tr, y_smi_tr),
                        batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(RegressionDataset(X_clin_val, X_aec_val, y_smi_val),
                        batch_size=BATCH_SIZE)
    return tr_dl, val_dl


class SMIRegressionNet(nn.Module):
    """
    Clinic + AEC → 연속 SMI 값 예측 네트워크.

    구조:
      AEC  : ResNet1DEncoder → (B, n_tokens, d_model)
      Clin : ScalarFeatureTokenizer → (B, 3, d_model)
      Bidirectional CrossAttn
      → concatenate mean-pooled → FC → scalar SMI output
    """
    def __init__(self, num_clinical_features=3,
                 d_model=HIDDEN, num_heads=N_HEADS,
                 ff_hidden_dim=128, classifier_hidden_dim=128,
                 dropout=0.1, n_aec_tokens=32):
        super().__init__()
        self.clinical_tokenizer = ScalarFeatureTokenizer(num_clinical_features, d_model)
        self.aec_encoder        = ResNet1DEncoder(d_model=d_model, n_tokens=n_aec_tokens)
        self.clin_to_aec = CrossAttentionBlock(d_model, num_heads, ff_hidden_dim, dropout)
        self.aec_to_clin = CrossAttentionBlock(d_model, num_heads, ff_hidden_dim, dropout)
        self.regressor = nn.Sequential(
            nn.Linear(d_model * 2, classifier_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(self, clinical_x, aec_x):
        c_tokens = self.clinical_tokenizer(clinical_x)
        a_tokens = self.aec_encoder(aec_x)
        c_fused  = self.clin_to_aec(c_tokens, a_tokens)
        a_fused  = self.aec_to_clin(a_tokens, c_tokens)
        fused    = torch.cat([c_fused.mean(1), a_fused.mean(1)], dim=-1)
        return self.regressor(fused).squeeze(-1)   # (B,) raw SMI prediction


def build_regression(smi_values: np.ndarray):
    """SMIRegressionNet + HuberLoss(delta=5) + AdamW + CosineAnnealingLR."""
    model = SMIRegressionNet().to(DEVICE)
    crit  = nn.HuberLoss(delta=5.0)   # SMI 스케일 (~20–60 cm²/m²)에 맞는 delta
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, crit, opt, sched


def train_regression_epoch(model, loader, crit, opt):
    """SMIRegressionNet 1 epoch 학습 → 평균 loss 반환."""
    model.train()
    total = 0.0
    for xc, xa, y_smi in loader:
        xc, xa, y_smi = xc.to(DEVICE), xa.to(DEVICE), y_smi.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xc, xa), y_smi)
        loss.backward(); opt.step()
        total += loss.item() * len(y_smi)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_regression_loader(model, loader, crit):
    """SMIRegressionNet 평가 → (avg_loss, preds, trues) 반환."""
    model.eval()
    total, preds, trues = 0.0, [], []
    for xc, xa, y_smi in loader:
        xc, xa, y_smi = xc.to(DEVICE), xa.to(DEVICE), y_smi.to(DEVICE)
        pred = model(xc, xa)
        total += crit(pred, y_smi).item() * len(y_smi)
        preds.extend(pred.cpu().numpy())
        trues.extend(y_smi.cpu().numpy())
    return total / len(loader.dataset), np.array(preds), np.array(trues)
