import torch
import torch.nn as nn
import torch.nn.functional as F

class ScalarFeatureTokenizer(nn.Module):
    """
    Tabular scalar feature를 feature token으로 변환하는 모듈.
    입력:
        x: (B, F)
            B = batch size
            F = number of features
    출력:
        tokens: (B, F, d_model)
    """
    def __init__(self, num_features: int, d_model: int):
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        self.feature_embedding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_list = []

        for i in range(self.num_features):
            xi = x[:, i:i + 1] # (B, 1)
            token_i = self.feature_projections[i](xi) # (B, d_model)
            token_i = token_i.unsqueeze(1) # (B, 1, d_model)
            token_list.append(token_i)

        tokens = torch.cat(token_list, dim=1) # (B, F, d_model)
        tokens = tokens + self.feature_embedding
        return tokens
    
class CrossAttentionBlock(nn.Module):
    """
        query_tokens가 context_tokens를 참고.
        Ex]
        clinical_tokens가 query
        aec_tokens가 key/value
        -> clinical feature가 AEC feature 중 무엇을 참고할지 학습
    """
    def __init__(
    self,
    d_model: int = 64,
    num_heads: int = 4,
    dropout: float = 0.1,
    ff_hidden_dim: int = 128,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
        nn.Linear(d_model, ff_hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(ff_hidden_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(
    self,
    query_tokens: torch.Tensor,
    context_tokens: torch.Tensor,
    return_attention: bool = False,
    ):
        attn_output, attn_weights = self.cross_attn(
        query=query_tokens,
        key=context_tokens,
        value=context_tokens,
        need_weights=True,
        average_attn_weights=True,
        )
        x = self.norm1(query_tokens + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        if return_attention:
            return x, attn_weights
    

class BidirectionalCrossAttention(nn.Module):
    """
    Clinical feature와 AEC feature를 bidirectional cross-attention으로 결합하는 모
    델.
    방향 1:
    Clinical → Query
    AEC → Key/Value
    방향 2:
    AEC → Query
    Clinical → Key/Value
    """
    def __init__(
    self,
    num_clinical_features: int,
    num_aec_features: int,
    d_model: int = 64,
    num_heads: int = 4,
    ff_hidden_dim: int = 128,
    classifier_hidden_dim: int = 128,
    output_dim: int = 1,
    dropout: float = 0.1,
    ):
        super().__init__()
        self.clinical_tokenizer = ScalarFeatureTokenizer(
            num_features=num_clinical_features,
            d_model=d_model,
        )
        self.aec_tokenizer = ScalarFeatureTokenizer(
            num_features=num_aec_features,
            d_model=d_model,
        )
        self.clinical_to_aec = CrossAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            ff_hidden_dim=ff_hidden_dim,
        )
        self.aec_to_clinical = CrossAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            ff_hidden_dim=ff_hidden_dim,
        )
        self.classifier = nn.Sequential(
        nn.Linear(d_model * 2, classifier_hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(classifier_hidden_dim, output_dim),
        )
    def forward(
        self,
        clinical_x: torch.Tensor,
        aec_x: torch.Tensor,
        return_attention: bool = False,
    ):
        # 1. scalar feature → feature tokens
        clinical_tokens = self.clinical_tokenizer(clinical_x)
        aec_tokens = self.aec_tokenizer(aec_x)

        # 2. Bidirectional cross-attention
        if return_attention:
            clinical_fused, attn_clinical_to_aec = self.clinical_to_aec(
                query_tokens=clinical_tokens,
                context_tokens=aec_tokens,
                return_attention=True,
            )
            aec_fused, attn_aec_to_clinical = self.aec_to_clinical(
                query_tokens=aec_tokens,
                context_tokens=clinical_tokens,
                return_attention=True,
            )
        else:
            clinical_fused = self.clinical_to_aec(
                query_tokens=clinical_tokens,
                context_tokens=aec_tokens,
                return_attention=False,
            )
            aec_fused = self.aec_to_clinical(
                query_tokens=aec_tokens,
                context_tokens=clinical_tokens,
                return_attention=False,
            )
        # 3. Token pooling
        clinical_repr = clinical_fused.mean(dim=1)
        aec_repr = aec_fused.mean(dim=1)
        # 4. Final fusion
        fused_repr = torch.cat([clinical_repr, aec_repr], dim=1)
        # 5. Prediction
        logits = self.classifier(fused_repr)
        if return_attention:
            attention_dict = {
                "clinical_to_aec": attn_clinical_to_aec,
                "aec_to_clinical": attn_aec_to_clinical,
            }
            return logits, attention_dict
        return logits

# Model 사용 예시
num_clinical_features = 6
num_aec_features = 10
model = BidirectionalCrossAttention(
    num_clinical_features=num_clinical_features,
    num_aec_features=num_aec_features,
    d_model=64,
    num_heads=4,
    ff_hidden_dim=128,
    classifier_hidden_dim=128,
    output_dim=1,
    dropout=0.1,
)