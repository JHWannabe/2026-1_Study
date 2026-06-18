import torch
import torch.nn as nn


class SegmentFusionModel(nn.Module):
    """AEC segment 피처 (intra 32 + inter 28 = 60-dim) + Clinic → MLP → logit.

    aec_feat_dim: extract_segment_features 출력 차원 (기본 60)
    n_clinic:     clinic 피처 수 (기본 3: Age, sex, BMI)
    aec_out:      AEC branch 출력 차원
    clinic_out:   Clinic branch 출력 차원 (clinic 피처 독립 임베딩)
    """

    def __init__(self, aec_feat_dim=60, n_clinic=3, aec_out=16, clinic_out=8, dropout=0.2):
        super().__init__()
        self.aec_branch = nn.Sequential(
            nn.Linear(aec_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, aec_out),
            nn.BatchNorm1d(aec_out),   # fusion 전 스케일 정규화
            nn.ReLU(inplace=True),     # 활성화 없던 마지막 레이어 보완
        )
        # clinic 피처를 raw concat 대신 별도 임베딩 후 fusion
        self.clinic_branch = nn.Sequential(
            nn.Linear(n_clinic, clinic_out),
            nn.BatchNorm1d(clinic_out),
            nn.ReLU(inplace=True),
        )
        fusion_dim = aec_out + clinic_out
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.BatchNorm1d(32),        # fusion 직후 정규화
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_aec, x_clinic):   # (B, 60), (B, n_clinic)
        aec_vec    = self.aec_branch(x_aec)                        # (B, aec_out)
        clinic_vec = self.clinic_branch(x_clinic)                  # (B, clinic_out)
        fused      = torch.cat([aec_vec, clinic_vec], dim=1)       # (B, aec_out+clinic_out)
        return self.classifier(fused).squeeze(-1)                   # (B,)  logit
