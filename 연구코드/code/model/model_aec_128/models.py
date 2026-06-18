import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class AECFusionModel(nn.Module):
    """AEC 128-dim → ResNet → aec_out-dim
    + Clinic (Age, sex, BMI) n_clinic-dim → concat → MLP → logit."""

    def __init__(self, base_ch=64, num_blocks=3, dropout=0.2, n_clinic=3, aec_out=16):
        super().__init__()
        # AEC branch
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[
            ResBlock1D(base_ch, dropout=dropout) for _ in range(num_blocks)
        ])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # dual-pool(base_ch*2) → aec_out 차원 projection
        self.aec_fc = nn.Sequential(
            nn.Linear(base_ch * 2, base_ch),
            nn.LayerNorm(base_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_ch, aec_out),
        )

        # Clinic embedding (small MLP, concat 전 표현 강화)
        clinic_hidden = max(n_clinic * 2, 8)
        self.clinic_emb = nn.Sequential(
            nn.Linear(n_clinic, clinic_hidden),
            nn.ReLU(inplace=True),
        )

        # Fusion head
        self.classifier = nn.Sequential(
            nn.Linear(aec_out + clinic_hidden, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_aec, x_clinic):      # (B,1,128), (B, n_clinic)
        x = self.stem(x_aec)
        x = self.blocks(x)                                         # (B, base_ch, 128)
        avg = self.avg_pool(x).squeeze(-1)                         # (B, base_ch)
        mx  = self.max_pool(x).squeeze(-1)                         # (B, base_ch)
        aec_vec    = self.aec_fc(torch.cat([avg, mx], dim=1))      # (B, aec_out)
        clinic_vec = self.clinic_emb(x_clinic)                     # (B, clinic_hidden)
        fused = torch.cat([aec_vec, clinic_vec], dim=1)            # (B, aec_out+clinic_hidden)
        return self.classifier(fused).squeeze(-1)                   # (B,)  logit


# ── Direction 2: Gating Mechanism ─────────────────────────────────────────────

class GatedAECFusionModel(nn.Module):
    """AEC-128 ResNet1D + Clinic 융합 시 학습된 스칼라 게이트(α) 적용.

    α = σ(W · [aec_vec ; clinic_vec]) ∈ (0, 1)
    fused = α · proj(aec_vec) + (1 − α) · proj(clinic_vec)

    forward(x_aec, x_clinic, return_gate=True) → (logit, α)
    α → 1 : AEC 우세,  α → 0 : Clinic 우세
    환자별 AEC 기여도를 수치로 해석 가능.
    """

    def __init__(self, base_ch=64, num_blocks=3, dropout=0.2,
                 n_clinic=3, aec_out=16, proj_dim=16):
        super().__init__()
        # AEC branch (ResNet1D — AECFusionModel과 동일 구조)
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[
            ResBlock1D(base_ch, dropout=dropout) for _ in range(num_blocks)
        ])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.aec_fc = nn.Sequential(
            nn.Linear(base_ch * 2, base_ch),
            nn.LayerNorm(base_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_ch, aec_out),
        )
        clinic_hidden = max(n_clinic * 2, 8)
        self.clinic_emb = nn.Sequential(
            nn.Linear(n_clinic, clinic_hidden),
            nn.ReLU(inplace=True),
        )
        # 공통 차원으로 projection
        self.aec_proj    = nn.Linear(aec_out,       proj_dim)
        self.clinic_proj = nn.Linear(clinic_hidden, proj_dim)
        # 게이트: [aec_vec ; clinic_vec] → α ∈ (0,1)
        self.gate = nn.Sequential(
            nn.Linear(aec_out + clinic_hidden, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_aec, x_clinic, return_gate=False):
        x = self.stem(x_aec)
        x = self.blocks(x)
        avg = self.avg_pool(x).squeeze(-1)
        mx  = self.max_pool(x).squeeze(-1)
        aec_vec    = self.aec_fc(torch.cat([avg, mx], dim=1))   # (B, aec_out)
        clinic_vec = self.clinic_emb(x_clinic)                  # (B, clinic_hidden)

        alpha  = self.gate(torch.cat([aec_vec, clinic_vec], dim=1))   # (B, 1)
        fused  = (alpha * self.aec_proj(aec_vec)
                  + (1 - alpha) * self.clinic_proj(clinic_vec))       # (B, proj_dim)
        logit  = self.classifier(fused).squeeze(-1)                   # (B,)

        if return_gate:
            return logit, alpha.squeeze(-1)   # α: (B,) ∈ (0,1)
        return logit


# ── Direction 6: 128×128 Correlation Matrix Input ─────────────────────────────

class AECCorrModel(nn.Module):
    """AEC 128-dim 자기상관행렬(128×128) → 2D CNN → classifier.

    x (B,1,128) 정규화 후 outer product → (B,1,128,128)
    슬라이스 간 전역 공동분산 패턴을 2D CNN으로 학습.
    현재 AECFusionModel(1D ResNet)의 보완 실험 모델.
    """

    def __init__(self, dropout=0.2, n_clinic=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(4),                            # → 16 × 32 × 32
            nn.Conv2d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(4),                            # → 32 × 8 × 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                    # → 64 × 1 × 1
        )
        clinic_hidden = max(n_clinic * 2, 8)
        self.clinic_emb = nn.Sequential(
            nn.Linear(n_clinic, clinic_hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 + clinic_hidden, 32),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_aec, x_clinic):          # x_aec: (B, 1, 128)
        x   = x_aec.squeeze(1)                    # (B, 128)
        x_n = F.normalize(x, dim=1)
        # 자기상관행렬: x_nᵀ · x_n → (B, 128, 128)
        corr = torch.bmm(x_n.unsqueeze(2), x_n.unsqueeze(1)).unsqueeze(1)  # (B,1,128,128)
        cnn_out    = self.cnn(corr).squeeze(-1).squeeze(-1)                  # (B, 64)
        clinic_vec = self.clinic_emb(x_clinic)
        return self.classifier(torch.cat([cnn_out, clinic_vec], dim=1)).squeeze(-1)
